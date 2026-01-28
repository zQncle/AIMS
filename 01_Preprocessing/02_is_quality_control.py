"""
Step 2: Internal Standard (IS) Quality Control & Filtering
----------------------------------------------------------
This script filters the processed scans from Step 1 based on the presence 
and intensity of the Internal Standard (IS).

Logic:
1. Checks each scan for the IS peak (e.g., m/z 196.11).
2. Verifies if the IS intensity exceeds the minimum threshold (e.g., 1000).
3. Discards scans where IS is missing or too weak (indicating poor injection/instrument status).

Input:  .pkl files from Step 1
Output: .pkl files containing only valid scans
"""

import pickle
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= Configuration =================
# Input directory (Output from Step 1)
INPUT_DIR = "./processed_peaks_step1"
# Output directory for filtered data
OUTPUT_DIR = "./processed_peaks_step2"

# Internal Standard Parameters
TARGET_MZ = 196.11      # Theoretical m/z of IS
SEARCH_RANGE = 0.1      # Search window +/- 0.1 Da
MIN_IS_INTENSITY = 1000 # Minimum intensity threshold to be considered valid
# ===============================================

def has_internal_standard(mzs, ints):
    """
    Check if Internal Standard exists in the spectrum and meets intensity criteria.
    Returns:
        bool: True if valid IS found
        np.array: Array of IS intensities found (for statistics)
    """
    # Create mask for m/z range
    mask = (mzs >= TARGET_MZ - SEARCH_RANGE) & (mzs <= TARGET_MZ + SEARCH_RANGE)
    
    if np.any(mask):
        # Get intensities of matching peaks
        matched_ints = ints[mask]
        max_int = np.max(matched_ints)
        
        # Validation: Intensity must be above threshold
        if max_int > MIN_IS_INTENSITY:
            return True, matched_ints
            
    return False, []

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pkl_files = glob.glob(os.path.join(INPUT_DIR, "*.pkl"))
    print(f"Found {len(pkl_files)} processed sample files")
    print(f"Filter Criteria: m/z {TARGET_MZ} +/- {SEARCH_RANGE} Da, Intensity > {MIN_IS_INTENSITY}")

    # Global Statistics
    total_frames_processed = 0
    total_frames_kept = 0
    is_intensities_global = []
    
    # Process files with progress bar
    for file_path in tqdm(pkl_files, desc="Filtering by IS"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # data structure: {'filename':..., 'label':..., 'peaks': [frame1, frame2...]}
            original_frames = data['peaks']
            valid_frames = []
            
            # Iterate through each frame (Spectrum) in the sample
            for frame in original_frames:
                mzs = frame['mzs']
                ints = frame['areas'] # Note: Step 1 saved areas as 'areas'
                
                found, is_vals = has_internal_standard(mzs, ints)
                
                if found:
                    valid_frames.append(frame)
                    is_intensities_global.extend(is_vals)
            
            # Update Statistics
            total_frames_processed += len(original_frames)
            total_frames_kept += len(valid_frames)
            
            # Save only if the sample has valid frames left
            if len(valid_frames) > 0:
                # Update peaks list
                data['peaks'] = valid_frames
                
                # Save to new directory
                filename = os.path.basename(file_path)
                save_path = os.path.join(OUTPUT_DIR, filename)
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                # Sample completely removed (optional logging)
                # print(f"Sample discarded (No valid IS): {data['filename']}")
                pass

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # ================= Report =================
    print("\n" + "="*40)
    print("📊 QC Filtering Report")
    print(f"Total Scans Processed: {total_frames_processed}")
    print(f"✅ Valid Scans Kept:   {total_frames_kept}")
    
    if total_frames_processed > 0:
        drop_rate = 100 * (1 - total_frames_kept / total_frames_processed)
        print(f"🗑️ Rejection Rate:    {drop_rate:.2f}%")

    if len(is_intensities_global) > 0:
        avg_int = np.mean(is_intensities_global)
        print("-" * 40)
        print(f"📈 Internal Standard Statistics:")
        print(f"   Mean Intensity: {avg_int:.2e}")
        print(f"   Max Intensity:  {np.max(is_intensities_global):.2e}")
        print(f"   Min Intensity:  {np.min(is_intensities_global):.2e}")
    else:
        print("⚠️ WARNING: No valid Internal Standard found in any scan!")

    print(f"\nFiltered data saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
