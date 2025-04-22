#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
import pickle
import sys
import re



def extract_lang_from_path(filepath):
    """Extracts a 2-letter language code (e.g., 'en', 'fr', 'zh') from the filepath."""
    # Try extracting from filename first
    basename = os.path.basename(filepath)
    match = re.search(r'[._-]([a-z]{2})[._-]', basename, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # If not in filename, try from directory structure
    norm_path = os.path.normpath(filepath)
    path_parts = norm_path.split(os.sep)
    for part in reversed(path_parts[:-1]):
        if len(part) == 2 and part.isalpha():
            return part.lower()

    print(f"Warning: Could not determine language code from path: {filepath}. Returning 'unknown'.")
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Word-level contextual embedding brain encoding analysis.')
    parser.add_argument('--hrf_csv', type=str, required=True,
                        help='Path to the aligned word embedding CSV file (design matrix X).')
    parser.add_argument('--pickle_data_dir', type=str, required=True,
                        help='Base directory containing subject subdirectories with fmri_atlas_data.pkl files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the results CSV file.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Ridge regression regularization coefficient (informational, as RidgeCV selects best alpha).')
    parser.add_argument('--model_name', type=str, default="word_context",
                        help='Name identifier for the model to include in output filename.')
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Load Design Matrix (X) and Determine Language ---
    print(f"Loading design matrix: {args.hrf_csv}")
    try:
        X = pd.read_csv(args.hrf_csv, header=None).values
        n_timepoints, n_features = X.shape
        print(f"  Design matrix shape: {X.shape}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Extract language code
    lang = extract_lang_from_path(args.hrf_csv)
    if lang == 'unknown':
        print("Warning: Could not determine language from path. Please ensure path contains language code.")
        lang_input = input("Enter language code manually (en/fr/zh): ").strip().lower()
        if lang_input in ['en', 'fr', 'zh']:
            lang = lang_input
        else:
            print("Error: Invalid language code. Must be 'en', 'fr', or 'zh'.")
            sys.exit(1)
    
    print(f"  Detected Language: {lang.upper()}")
    
    # --- 2. Find Subjects and Pickle Files for the target language ---
    print(f"\nSearching for subjects matching language '{lang.upper()}' in: {args.pickle_data_dir}")
    subject_pattern = os.path.join(args.pickle_data_dir, f'sub-{lang.upper()}*')
    print(f"  Subject pattern: {subject_pattern}")
    subject_dirs = sorted(glob.glob(subject_pattern))
    
    subjects = []
    subject_pickle_paths = {}
    for d in subject_dirs:
        # Double check the directory name starts correctly
        sub_id = os.path.basename(d)
        if not sub_id.lower().startswith(f'sub-{lang.lower()}'):
            continue
        
        pickle_file = os.path.join(d, 'fmri_atlas_data.pkl')
        if os.path.isfile(pickle_file):
            subjects.append(sub_id)
            subject_pickle_paths[sub_id] = pickle_file
        else:
            print(f"  Pickle file not found in {d}, skipping directory.")
    
    if not subjects:
        print(f"\nError: No subjects found for language '{lang.upper()}' with proper pickle files")
        sys.exit(1)
    
    print(f"\nFound {len(subjects)} subjects for language '{lang.upper()}'.")
    
    # --- 3. Cross-Validation Setup ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_scaler = StandardScaler()
    
    results = []
    skipped_subjects = []
    
    # --- 4. Process Each Subject ---
    for sub_id in tqdm(subjects, desc=f'Processing {lang.upper()} subjects'):
        pickle_path = subject_pickle_paths[sub_id]
        
        try:
            # --- Load BOLD Data (Y) ---
            with open(pickle_path, 'rb') as handle:
                masked_bold_data = pickle.load(handle)
            
            # Cast to float64 for precision
            masked_bold_data = masked_bold_data.astype(np.float64)
            
            # Validate dimensions
            if masked_bold_data.shape[0] != n_timepoints:
                print(f"\nError: Timepoints mismatch for {sub_id}. "
                      f"Pickle data has {masked_bold_data.shape[0]} timepoints, "
                      f"HRF CSV has {n_timepoints}. Skipping.")
                skipped_subjects.append(sub_id)
                continue
            
            n_masked_voxels = masked_bold_data.shape[1]
            if n_masked_voxels == 0:
                print(f"\nWarning: No masked voxels found for {sub_id}. Skipping.")
                skipped_subjects.append(sub_id)
                continue
            
            # --- Cross-Validation Loop ---
            testY_all = []
            predY_all = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = masked_bold_data[train_idx, :], masked_bold_data[test_idx, :]
                
                # Standardize X
                X_train_scaled = X_scaler.fit_transform(X_train)
                X_test_scaled = X_scaler.transform(X_test)
                
                # Standardize Y per fold
                fold_y_scaler = StandardScaler()
                y_train_scaled = fold_y_scaler.fit_transform(y_train)
                
                # Handle constant features
                if np.all(fold_y_scaler.scale_ > 1e-8):
                    y_test_scaled = fold_y_scaler.transform(y_test)
                else:
                    y_test_scaled = y_test.copy()
                    non_constant_mask = fold_y_scaler.scale_ > 1e-8
                    if np.any(non_constant_mask):
                        y_test_scaled[:, non_constant_mask] = fold_y_scaler.transform(y_test[:, non_constant_mask])
                
                # Train RidgeCV model
                model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                model.fit(X_train_scaled, y_train_scaled)
                
                # Predict
                y_pred_scaled = model.predict(X_test_scaled)
                
                # Collect test ground truth and predictions
                testY_all.append(y_test_scaled)
                predY_all.append(y_pred_scaled)
            
            # --- Aggregate Results and Calculate Metrics ---
            testY_all = np.concatenate(testY_all, axis=0)
            predY_all = np.concatenate(predY_all, axis=0)
            
            # Calculate Pearson correlation per masked voxel
            r_per_voxel_masked = np.full(n_masked_voxels, np.nan, dtype=np.float32)
            p_per_voxel_masked = np.full(n_masked_voxels, np.nan, dtype=np.float32)
            constant_voxels = 0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                for i in range(n_masked_voxels):
                    test_data = testY_all[:, i]
                    pred_data = predY_all[:, i]
                    
                    # Check for constant data
                    if np.all(test_data == test_data[0]) or np.all(pred_data == pred_data[0]):
                        constant_voxels += 1
                        continue
                    
                    # Check for NaN/Inf
                    if np.any(np.isnan(test_data)) or np.any(np.isnan(pred_data)) or \
                       np.any(np.isinf(test_data)) or np.any(np.isinf(pred_data)):
                        constant_voxels += 1
                        continue
                    
                    try:
                        r_val, p_val = pearsonr(test_data, pred_data)
                        if np.isnan(r_val):
                            constant_voxels += 1
                            continue
                        r_per_voxel_masked[i] = r_val
                        p_per_voxel_masked[i] = p_val
                    except ValueError:
                        constant_voxels += 1
            
            # Calculate summary statistics
            mean_pearson_r = np.nanmean(r_per_voxel_masked)
            valid_p_values = p_per_voxel_masked[~np.isnan(p_per_voxel_masked)]
            mean_p_value = np.mean(valid_p_values) if len(valid_p_values) > 0 else np.nan
            num_valid_voxels = np.sum(~np.isnan(r_per_voxel_masked))
            
            # Record results
            results.append({
                'subject': sub_id,
                'mean_pearson_r': mean_pearson_r,
                'mean_p_value': mean_p_value,
                'masked_voxels': n_masked_voxels,
                'valid_voxels': num_valid_voxels,
                'constant_voxels_skipped': constant_voxels
            })
            
        except Exception as e:
            print(f"\nError processing {sub_id}: {e}")
            skipped_subjects.append(sub_id)
            continue
    
    # --- 5. Finalize and Save Results ---
    if skipped_subjects:
        print(f"\nSkipped {len(skipped_subjects)} subjects due to errors.")
    
    if not results:
        print(f"\nError: No subjects processed successfully. No output file created.")
        sys.exit(1)
    
    df = pd.DataFrame(results)
    
    # Calculate grand averages
    grand_avg_r = df['mean_pearson_r'].mean()
    grand_avg_p = df['mean_p_value'].mean()
    
    # Add average row
    average_row = pd.DataFrame([{
        'subject': 'average',
        'mean_pearson_r': grand_avg_r,
        'mean_p_value': grand_avg_p,
        'masked_voxels': df['masked_voxels'].mean(),
        'valid_voxels': df['valid_voxels'].mean(),
        'constant_voxels_skipped': df['constant_voxels_skipped'].mean()
    }])
    df = pd.concat([df, average_row], ignore_index=True)
    
    # Construct output filename
    output_file = os.path.join(args.output_dir, f"{args.model_name}_{lang}_alpha{args.alpha}_wordContext.csv")
    df.to_csv(output_file, index=False, float_format='%.8f')
    print(f"\nResults saved to: {output_file}")
    print(f"Grand average Pearson r: {grand_avg_r:.4f}")


if __name__ == "__main__":
    main() 