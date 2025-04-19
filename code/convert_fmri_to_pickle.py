import os
import argparse
import nibabel as nib
import numpy as np
import pickle
import glob
from tqdm import tqdm
import sys

def save_pickle(data, filename):
    """Saves data to a pickle file, creating directories if needed."""
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"  Saving data to {filename}")
    try:
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print("    Data saved successfully.")
    except Exception as e:
        print(f"    Error saving pickle file {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert subject NIfTI fMRI data with atlas mask to individual pickle files.')
    parser.add_argument('--derivatives_dir', type=str, required=True,
                        help='Root directory containing BIDS derivative data (e.g., fmriprep output).')
    parser.add_argument('--lang_atlas', type=str, required=True,
                        help='Path to the language probability atlas NIfTI file (e.g., SPM_LanA_n806.nii).')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Threshold for the language atlas probability map (default: 0.2).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base directory to save the output pickle files (e.g., data/dict_fMRI_atlas).')
    parser.add_argument('--bold_glob_pattern', type=str,
                        default='func/*task-lpp*_mergedTP-*_bold.nii.gz',
                        help='Glob pattern relative to subject directory to find the BOLD file(s). '
                             'Default finds merged LPP task runs.')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Ensure base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load Atlas and Create Mask ---
    print(f"Loading language atlas: {args.lang_atlas}")
    try:
        atlas_img = nib.load(args.lang_atlas)
        atlas_data = atlas_img.get_fdata()
        print(f"  Atlas original shape: {atlas_data.shape}")
    except Exception as e:
        print(f"Error loading atlas file: {e}")
        sys.exit(1)

    if len(atlas_data.shape) != 3:
        raise ValueError("Language atlas must be a 3D NIfTI file.")

    language_mask = (atlas_data >= args.threshold)
    n_masked_voxels_total = np.sum(language_mask)
    print(f"  Created mask shape: {language_mask.shape}")
    print(f"  Number of voxels in mask (threshold >= {args.threshold}): {n_masked_voxels_total}")
    if n_masked_voxels_total == 0:
        print("Warning: The mask is empty with the current threshold. No voxels will be extracted.")
        # Decide whether to exit or continue creating empty files
        # sys.exit(1)

    mask_1d = language_mask.ravel()

    # --- 2. Find Subjects ---
    subject_dirs = sorted(glob.glob(os.path.join(args.derivatives_dir, 'sub-*')))
    subjects = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]

    if not subjects:
        print(f"Error: No subject directories found matching 'sub-*' in {args.derivatives_dir}")
        sys.exit(1)

    print(f"\nFound {len(subjects)} potential subjects.")

    # --- 3. Process Each Subject ---
    processed_subjects = 0
    skipped_subjects = []

    for sub_id in tqdm(subjects, desc='Processing subjects'):
        bold_pattern = os.path.join(args.derivatives_dir, sub_id, args.bold_glob_pattern)
        # Use sorted glob to ensure consistent file selection if multiple match
        bold_files = sorted(glob.glob(bold_pattern))

        if not bold_files:
            # print(f"\nWarning: No BOLD file found for {sub_id} using pattern: {args.bold_glob_pattern}. Skipping.")
            skipped_subjects.append(sub_id)
            continue
        if len(bold_files) > 1:
            print(f"\nWarning: Multiple BOLD files found for {sub_id}, using the first one: {bold_files[0]}")
            # Add logic here if specific file selection is needed

        bold_file_path = bold_files[0]
        # Define the output path for this subject's pickle file
        output_pickle_path = os.path.join(args.output_dir, sub_id, 'fmri_atlas_data.pkl')

        try:
            # print(f"\nProcessing {sub_id}: {bold_file_path}")
            bold_img = nib.load(bold_file_path)
            bold_data = bold_img.get_fdata() # Shape: (X, Y, Z, T)
            # print(f"  Loaded BOLD data, shape: {bold_data.shape}")

            # Validate spatial dimensions
            if bold_data.shape[:3] != atlas_data.shape:
                print(f"\nError: Spatial dimensions mismatch for {sub_id}. "
                      f"BOLD shape {bold_data.shape[:3]} vs Atlas shape {atlas_data.shape}. Skipping.")
                skipped_subjects.append(sub_id)
                continue

            n_timepoints = bold_data.shape[-1]
            # print(f"  Number of timepoints: {n_timepoints}")

            # Apply mask
            # Reshape BOLD data: (X*Y*Z, T)
            bold_reshaped = bold_data.reshape(-1, n_timepoints)
            # Select voxels within the mask: (n_masked_voxels, T)
            masked_bold_data = bold_reshaped[mask_1d, :]
            # Transpose to desired format: (T, n_masked_voxels)
            masked_bold_data_t_vox = masked_bold_data.T

            # print(f"  Shape after masking and transpose: {masked_bold_data_t_vox.shape}")

            if masked_bold_data_t_vox.shape[1] != n_masked_voxels_total:
                 # This shouldn't happen if spatial dimensions match, but good sanity check
                 print(f"\nWarning: Mismatch in expected masked voxel count for {sub_id}. "
                       f"Expected {n_masked_voxels_total}, got {masked_bold_data_t_vox.shape[1]}.")

            # Save the NumPy array directly to the subject's pickle file
            save_pickle(masked_bold_data_t_vox.astype(np.float32), output_pickle_path)
            processed_subjects += 1


        except Exception as e:
            print(f"\nError processing {sub_id}: {e}. Skipping.")
            skipped_subjects.append(sub_id)
            continue

    # --- 4. Summary ---
    print(f"\n--- Processing Summary ---")
    print(f"Successfully processed {processed_subjects} subjects.")
    if skipped_subjects:
        print(f"Skipped {len(skipped_subjects)} subjects due to missing files or errors: {skipped_subjects}")
    print(f"Output saved in directory structure under: {args.output_dir}")

    # --- Verification (Optional) ---
    if processed_subjects > 0:
      first_processed_sub = next(sub for sub in subjects if sub not in skipped_subjects)
      first_output_file = os.path.join(args.output_dir, first_processed_sub, 'fmri_atlas_data.pkl')
      if os.path.exists(first_output_file):
          try:
              with open(first_output_file, 'rb') as handle:
                  first_data = pickle.load(handle)
              print("\nVerification of first processed subject:")
              print(f"  Loaded data from: {first_output_file}")
              print(f"  Data shape for {first_processed_sub}: {first_data.shape}")
              print(f"  Data type: {first_data.dtype}")
          except Exception as e:
              print(f"  Error verifying first output file: {e}")


if __name__ == "__main__":
    main()

