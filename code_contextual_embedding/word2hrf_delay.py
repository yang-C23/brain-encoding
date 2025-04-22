#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Align word embeddings with fMRI data using delay strategy.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="NPY file with columns [onset, duration, embedding...]")
    parser.add_argument("-t", "--tr", type=float, required=True,
                        help="TR in seconds")
    parser.add_argument("-n", "--nscans", type=int, required=True,
                        help="Total number of scans in the fMRI time series")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output CSV file containing aligned embeddings")
    parser.add_argument("--delay", type=int, default=2,
                        help="Number of TRs to delay the embeddings (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Verify args
    if args.tr <= 0:
        print("Error: TR must be positive")
        sys.exit(1)
    if args.nscans <= 0:
        print("Error: Number of scans must be positive")
        sys.exit(1)
    if args.delay < 0:
        print("Error: Delay must be non-negative")
        sys.exit(1)
    
    # Load word embeddings
    print(f"Loading word embeddings from {args.input}")
    try:
        word_data = np.load(args.input)
        # Format: [onset, duration, embedding_features...]
        onsets = word_data[:, 0]
        durations = word_data[:, 1]
        embeddings = word_data[:, 2:]
        embedding_dim = embeddings.shape[1]
        print(f"Loaded {len(onsets)} words with embedding dimension {embedding_dim}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Calculate scan times (in seconds)
    scan_times = np.arange(args.nscans) * args.tr
    endpoint = args.nscans * args.tr
    
    # Initialize design matrix (one row per scan, one column per embedding dimension)
    design_matrix = np.zeros((args.nscans, embedding_dim))
    
    # Track which scan indices have embeddings assigned
    scan_counts = np.zeros(args.nscans)
    
    # Assign embeddings to scans based on onset times
    for i in range(len(onsets)):
        onset = onsets[i]
        duration = durations[i]
        embedding = embeddings[i]
        
        # Find which scans this word falls into
        scan_indices = np.where((scan_times <= onset + duration) & (scan_times + args.tr > onset))[0]
        
        if len(scan_indices) > 0:
            for scan_idx in scan_indices:
                if scan_idx < args.nscans:  # Ensure index is valid
                    design_matrix[scan_idx] += embedding
                    scan_counts[scan_idx] += 1
    
    # Average embeddings for scans with multiple words
    nonzero_mask = scan_counts > 0
    design_matrix[nonzero_mask] = design_matrix[nonzero_mask] / scan_counts[nonzero_mask, np.newaxis]
    
    # Apply the delay shift
    delayed_matrix = np.zeros_like(design_matrix)
    if args.delay > 0:
        if args.delay < args.nscans:
            delayed_matrix[args.delay:] = design_matrix[:-args.delay]
        else:
            print(f"Warning: Delay ({args.delay} TRs) is greater than or equal to the number of scans ({args.nscans}). "
                  f"The resulting matrix will be all zeros.")
    else:
        delayed_matrix = design_matrix.copy()
    
    # Check if we have any non-zero values
    if np.all(delayed_matrix == 0):
        print("Warning: The resulting design matrix is all zeros. Check your parameters.")
    
    # Save to CSV
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(delayed_matrix).to_csv(args.output, index=False, header=False)
    print(f"Saved design matrix with shape {delayed_matrix.shape} to {args.output}")
    print(f"Applied delay of {args.delay} TRs ({args.delay * args.tr} seconds)")
    
    # Print stats
    words_assigned = np.sum(scan_counts > 0)
    print(f"Words assigned to at least one scan: {words_assigned} / {len(onsets)} ({words_assigned/len(onsets)*100:.1f}%)")
    print(f"Scans with at least one word: {np.sum(nonzero_mask)} / {args.nscans} ({np.sum(nonzero_mask)/args.nscans*100:.1f}%)")


if __name__ == "__main__":
    main() 