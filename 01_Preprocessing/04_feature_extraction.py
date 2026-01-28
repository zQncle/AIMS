"""
Step 4: Deep Feature Deconvolution (Greedy Isotopic Blockout)
-------------------------------------------------------------
This script implements the "Greedy Isotopic Blockout Algorithm" to identify 
and extract metabolic features based on specific isotopic patterns.

Algorithm:
1. Identify Internal Standard (IS) for normalization.
2. Sort all peaks by intensity (Descending).
3. Greedy Search:
   - Pick the highest intensity peak (Candidate M).
   - Check for the presence of specific isotopic neighbors (M-2, M-1, M+1).
   - Validation: The M-2 peak must have > 50% intensity of M.
4. Blockout:
   - If a pattern is found, mask the region [M-2, M+1] to prevent re-use.
   - Sum intensities of the cluster and normalize to IS.
5. Save features with 3-decimal precision (float32).

Input:  Calibrated .pkl files from Step 3
Output: Processed .pkl files with reduced feature sets
"""

import pickle
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= Configuration =================
# Input directory (Output from Step 3)
INPUT_DIR = "./processed_peaks_step3_calibrated"
# Output directory for extracted features
OUTPUT_DIR = "./processed_peaks_step4_features"

# Internal Standard Parameters
IS_TARGET_MZ = 196.11
IS_SEARCH_TOL = 0.05 

# Feature Screening Parameters
MIN_MZ_THRESHOLD = 216.0    # Ignore peaks below this m/z
PATTERN_TOLERANCE = 0.03    # Tolerance for finding isotopic neighbors

# Core Filtering Logic
M_MINUS_2_RATIO = 0.5       # M-2 intensity must be > 0.5 * M

# Blockout Parameters
BLOCKOUT_MARGIN = 0.1       # Extra buffer zone for masking
# ===============================================

def find_best_peak_index(mzs, ints, target_mz, tol, used_mask):
    """
    Find the index of the strongest UNUSED peak within a given range.
    """
    mask_range = (mzs >= target_mz - tol) & (mzs <= target_mz + tol)
    valid_candidates_mask = mask_range & (~used_mask)
    
    candidate_indices = np.where(valid_candidates_mask)[0]
    
    if len(candidate_indices) > 0:
        # Return index of the highest intensity peak in the window
        best_rel_idx = np.argmax(ints[candidate_indices])
        return candidate_indices[best_rel_idx]
    
    return None

def blockout_range(mzs, center_mz, used_mask):
    """
    Apply Blockout: Mask the region covering [M-2 to M+1] to prevent
    redundant feature extraction.
    """
    lower_bound = center_mz - 2.0067 - BLOCKOUT_MARGIN
    upper_bound = center_mz + 1.00335 + BLOCKOUT_MARGIN
    
    indices_to_block = np.where((mzs >= lower_bound) & (mzs <= upper_bound))[0]
    used_mask[indices_to_block] = True

def process_frame(mzs, ints):
    """
    Process a single scan using the Greedy Blockout strategy.
    """
    used_mask = np.zeros(len(mzs), dtype=bool)

    # 1. Lock Internal Standard (IS)
    idx_is = find_best_peak_index(mzs, ints, IS_TARGET_MZ, IS_SEARCH_TOL, used_mask)
    if idx_is is None: return None
    is_intensity = ints[idx_is]
    if is_intensity < 100: return None
    
    # Blockout IS region immediately
    is_mz_val = mzs[idx_is]
    indices_is_block = np.where((mzs >= is_mz_val - 0.5) & (mzs <= is_mz_val + 0.5))[0]
    used_mask[indices_is_block] = True

    # 2. Identify Candidates
    valid_indices = np.where((mzs > MIN_MZ_THRESHOLD) & (~used_mask))[0]
    if len(valid_indices) == 0: return {}

    # Sort by intensity descending (Greedy approach)
    sorted_priority_indices = valid_indices[np.argsort(-ints[valid_indices])]

    features_dict = {}

    # 3. Greedy Loop
    for idx_m in sorted_priority_indices:
        if used_mask[idx_m]:
            continue

        mz_m = mzs[idx_m]
        int_m = ints[idx_m]

        # --- Check Isotopic Neighbors (Pattern Matching) ---
        # Look for M-2
        idx_m_minus_2 = find_best_peak_index(mzs, ints, mz_m - 2.0067, PATTERN_TOLERANCE, used_mask)
        if idx_m_minus_2 is None: continue
        int_m_minus_2 = ints[idx_m_minus_2]
        
        # CRITICAL CHECK: M-2 must be significant (> 50% of M)
        if int_m_minus_2 <= M_MINUS_2_RATIO * int_m: continue

        # Look for M-1
        idx_m_minus_1 = find_best_peak_index(mzs, ints, mz_m - 1.00335, PATTERN_TOLERANCE, used_mask)
        if idx_m_minus_1 is None: continue
        int_m_minus_1 = ints[idx_m_minus_1]

        # Look for M+1
        idx_m_plus_1 = find_best_peak_index(mzs, ints, mz_m + 1.00335, PATTERN_TOLERANCE, used_mask)
        if idx_m_plus_1 is None: continue
        int_m_plus_1 = ints[idx_m_plus_1]

        # --- Pattern Confirmed: Apply Blockout ---
        blockout_range(mzs, mz_m, used_mask)

        # --- Normalization ---
        sum_intensity = int_m_minus_2 + int_m_minus_1 + int_m + int_m_plus_1
        normalized_val = sum_intensity / is_intensity

        # --- Storage ---
        # Note: Using round(3) here acts as a preliminary binning.
        # Advanced alignment/merging should be handled in downstream matrix generation.
        mz_key = round(mz_m, 3)
        
        # Deduplication: Keep the strongest feature if collision occurs after rounding
        if mz_key not in features_dict:
            features_dict[mz_key] = normalized_val
        else:
            if normalized_val > features_dict[mz_key]:
                features_dict[mz_key] = normalized_val

    return features_dict

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pkl_files = glob.glob(os.path.join(INPUT_DIR, "*.pkl"))
    print(f"📂 Input Directory: {INPUT_DIR}")
    print(f"📄 Files to Process: {len(pkl_files)}")
    print(f"⚙️  Mode: Greedy Pattern Matching + Blockout (Float32)")

    total_frames_processed = 0
    total_features_extracted = 0

    for file_path in tqdm(pkl_files, desc="Extracting Features"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            frames = data['peaks']
            processed_frames = []

            for frame in frames:
                mzs = frame['mzs']
                ints = frame['areas']
                
                # Execute Greedy Blockout
                features_dict = process_frame(mzs, ints)
                
                if features_dict and len(features_dict) > 0:
                    sorted_keys = sorted(features_dict.keys())
                    sorted_vals = [features_dict[k] for k in sorted_keys]
                    
                    new_frame = {
                        'rt': frame['rt'],
                        # Save as float32 to reduce file size
                        'mzs': np.array(sorted_keys, dtype=np.float32), 
                        'intensities': np.array(sorted_vals, dtype=np.float32)
                    }
                    processed_frames.append(new_frame)
                    total_features_extracted += len(sorted_keys)

            if processed_frames:
                data['peaks'] = processed_frames
                
                save_name = os.path.basename(file_path)
                save_path = os.path.join(OUTPUT_DIR, save_name)
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
                    
            total_frames_processed += len(frames)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    print("\n" + "="*50)
    print(f"✅ Feature Extraction Complete!")
    print(f"Total Features Extracted: {total_features_extracted}")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
