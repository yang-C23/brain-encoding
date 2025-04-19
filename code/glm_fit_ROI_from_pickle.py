# glm_fit_ROI_from_pickle.py (Corrected Language Filtering)
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
import re # Import regex module

def extract_lang_from_path(filepath):
    """Extracts a 2-letter language code (e.g., 'en', 'cn') from the filepath."""
    # Try extracting from filename first (e.g., model_en_hrf.csv)
    basename = os.path.basename(filepath)
    match = re.search(r'[._-]([a-z]{2})[._-]', basename, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # If not in filename, try from directory structure (e.g., /path/to/en/hrf.csv)
    norm_path = os.path.normpath(filepath)
    path_parts = norm_path.split(os.sep)
    # Look for a 2-letter part
    for part in reversed(path_parts[:-1]): # Check parent dirs, not filename itself
        if len(part) == 2 and part.isalpha():
            return part.lower()

    print(f"Warning: Could not reliably determine 2-letter language code from path: {filepath}. Returning 'unknown'.")
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Language ROI encoding analysis from pre-masked Pickle files (Language Specific).')
    parser.add_argument('--hrf_csv', type=str, required=True,
                        help='Path to the HRF model CSV file (design matrix X). Must contain lang code (e.g., model_en_hrf.csv or in path).')
    parser.add_argument('--pickle_data_dir', type=str, required=True,
                        help='Base directory containing subject subdirectories with fmri_atlas_data.pkl files (e.g., data/dict_fMRI_atlas).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the results CSV file.')
    parser.add_argument('--alpha', type=float, default=1.0, # Note: RidgeCV finds best alpha, this isn't directly used unless RidgeCV alphas are restricted
                        help='Ridge regression regularization coefficient (informational, as RidgeCV selects best alpha).')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load Design Matrix (X) and Determine Language ---
    print(f"Loading design matrix: {args.hrf_csv}")
    try:
        X = pd.read_csv(args.hrf_csv, header=None).values  # shape: (n_timepoints, n_features)
        n_timepoints, n_features = X.shape
        print(f"  Design matrix shape: {X.shape}")
    except Exception as e:
        print(f"Error loading HRF CSV file: {e}")
        sys.exit(1)

    # Extract language code and HRF model name from the HRF file path
    lang = extract_lang_from_path(args.hrf_csv)
    if lang == 'unknown':
        print("Error: Could not determine language from HRF path. Please ensure the path or filename contains the 2-letter language code.")
        sys.exit(1)

    try:
        hrf_model = os.path.basename(args.hrf_csv).split('_')[0]
        print(f"  Detected Language: {lang.upper()}, HRF Model: {hrf_model}")
    except Exception as e:
        print(f"Warning: Could not parse model name from HRF path ({args.hrf_csv}): {e}")
        hrf_model = 'unknown_model'


    # --- 2. Find Subjects and Pickle Files for the target language ---
    print(f"\nSearching for subjects matching language '{lang.upper()}' in: {args.pickle_data_dir}")
    subject_pattern = os.path.join(args.pickle_data_dir, f'sub-{lang.upper()}*')
    # --- DEBUG PRINTS ---
    print(f"  Resolved pickle_data_dir: {os.path.abspath(args.pickle_data_dir)}")
    print(f"  Constructed glob pattern: {subject_pattern}")
    # --- END DEBUG PRINTS ---
    subject_dirs = sorted(glob.glob(subject_pattern))
    # --- DEBUG PRINTS ---
    print(f"  Result of glob.glob('{subject_pattern}'): {subject_dirs}")
    # --- END DEBUG PRINTS ---

    subjects = []
    subject_pickle_paths = {}
    for d in subject_dirs:
         # Double check the directory name starts correctly before adding
        sub_id = os.path.basename(d)
        if not sub_id.lower().startswith(f'sub-{lang}'):
             continue # Skip if somehow glob returned incorrect matches

        pickle_file = os.path.join(d, 'fmri_atlas_data.pkl')
        if os.path.isfile(pickle_file):
            subjects.append(sub_id)
            subject_pickle_paths[sub_id] = pickle_file
        else:
            print(f"  Pickle file 'fmri_atlas_data.pkl' not found in {d}, skipping directory.")


    if not subjects:
        print(f"\nError: No subjects found for language '{lang.upper()}' with 'fmri_atlas_data.pkl' in {args.pickle_data_dir}")
        sys.exit(1)

    print(f"\nFound {len(subjects)} subjects for language '{lang.upper()}'.")
    # print(f"Subjects found: {subjects}")


    # --- 3. Cross-Validation Setup ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_scaler = StandardScaler()
    # y_scaler will be defined per-fold

    results = []
    skipped_subjects = []

    # --- 4. Process Each Subject ---
    for sub_id in tqdm(subjects, desc=f'Processing {lang.upper()} subjects'):
        pickle_path = subject_pickle_paths[sub_id]

        try:
            # --- Load Masked BOLD Data (Y) ---
            with open(pickle_path, 'rb') as handle:
                masked_bold_data = pickle.load(handle) # Shape: (n_timepoints, n_masked_voxels)

            # Cast to float64 to match potential precision in NIfTI loading
            masked_bold_data = masked_bold_data.astype(np.float64)

            # print(f"\n  {sub_id}: Loaded masked BOLD data, shape: {masked_bold_data.shape}")

            # Validate time dimension
            if masked_bold_data.shape[0] != n_timepoints:
                print(f"\nError: Timepoints mismatch for {sub_id}. "
                      f"Pickle data has {masked_bold_data.shape[0]} timepoints, "
                      f"HRF CSV has {n_timepoints}. Skipping.")
                skipped_subjects.append(sub_id)
                continue

            n_masked_voxels = masked_bold_data.shape[1]
            if n_masked_voxels == 0:
                print(f"\nWarning: No masked voxels found for {sub_id} in {pickle_path}. Skipping.")
                skipped_subjects.append(sub_id)
                continue
            # print(f"  {sub_id}: Number of masked voxels: {n_masked_voxels}")


            # --- Cross-Validation Loop ---
            testY_all = []
            predY_all = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                # print(f"    Fold {fold+1}/{kf.get_n_splits()}...")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = masked_bold_data[train_idx, :], masked_bold_data[test_idx, :]

                # Standardize X
                X_train_scaled = X_scaler.fit_transform(X_train)
                X_test_scaled  = X_scaler.transform(X_test)

                # Standardize Y (masked BOLD data) per fold
                fold_y_scaler = StandardScaler()
                y_train_scaled = fold_y_scaler.fit_transform(y_train)

                if np.all(fold_y_scaler.scale_ > 1e-8):
                     y_test_scaled = fold_y_scaler.transform(y_test)
                else:
                     # print(f"    Warning: Constant training data detected for some voxels in fold {fold+1}. Using unscaled test data where needed.")
                     y_test_scaled = y_test.copy()
                     non_constant_mask = fold_y_scaler.scale_ > 1e-8
                     if np.any(non_constant_mask):
                         y_test_scaled[:, non_constant_mask] = fold_y_scaler.transform(y_test[:,non_constant_mask])


                # Train RidgeCV model
                # Consider a wider range or log-spaced alphas if needed
                model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                model.fit(X_train_scaled, y_train_scaled)
                # print(f"      Best alpha for fold {fold+1}: {model.alpha_}")

                # Predict
                y_pred_scaled = model.predict(X_test_scaled)

                # Collect scaled test ground truth and predictions
                testY_all.append(y_test_scaled)
                predY_all.append(y_pred_scaled)


            # --- Aggregate Fold Results and Calculate Metrics ---
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

                    # Simplified constant check, closer to glm_fit_ROI.py
                    if np.all(test_data == test_data[0]) or np.all(pred_data == pred_data[0]):
                        constant_voxels += 1
                        continue

                    # Additional check for NaN/Inf which might still occur
                    if np.any(np.isnan(test_data)) or np.any(np.isnan(pred_data)) or \
                       np.any(np.isinf(test_data)) or np.any(np.isinf(pred_data)):
                           constant_voxels += 1
                           continue

                    try:
                        r_val, p_val = pearsonr(test_data, pred_data)
                        # Check for NaN r_val which can happen even if std > 0 (e.g., perfect anti-correlation edge case?)
                        if np.isnan(r_val):
                            constant_voxels += 1
                            continue
                        r_per_voxel_masked[i] = r_val
                        p_per_voxel_masked[i] = p_val
                    except ValueError: # Catches issues if pearsonr fails
                         constant_voxels +=1


            # Calculate summary statistics for this subject
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


        except FileNotFoundError:
             print(f"\nError: Required pickle file not found for {sub_id} at {pickle_path}. Skipping.")
             skipped_subjects.append(sub_id)
             continue
        except Exception as e:
            print(f"\nError processing {sub_id}: {e}. Skipping.")
            # import traceback # Uncomment for debugging
            # traceback.print_exc() # Uncomment for debugging
            skipped_subjects.append(sub_id)
            continue

    # --- 5. Finalize and Save Results ---
    if skipped_subjects:
        print(f"\nSkipped {len(skipped_subjects)} subjects due to errors: {skipped_subjects}")

    if not results:
        print(f"\nError: No subjects for language '{lang.upper()}' were processed successfully. No output CSV file created.")
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

    # Construct output filename using determined language and model
    output_file = os.path.join(args.output_dir, f"{hrf_model}_{lang}_alpha{args.alpha}_langMasked_fromPickle.csv")
    df.to_csv(output_file, index=False, float_format='%.8f')
    print(f"\nResults for language '{lang.upper()}' saved to: {output_file}")


if __name__ == "__main__":
    main()