"""
Step 3: Global Multiplicative Mass Calibration
----------------------------------------------
This script corrects mass drift (m/z shift) using the Internal Standard (IS) as a reference.

Methodology:
1. Identifies the observed m/z of the IS in each scan.
2. Calculates a scaling factor: Factor = Theoretical_m/z / Observed_m/z.
3. Applies this multiplicative factor to ALL m/z values in that scan.
   (Corrected_m/z = Original_m/z * Factor)

Why Multiplicative?
For time-of-flight or Orbitrap instruments, mass drift is often proportional 
to m/z (ppm-level error), making scaling more physically accurate than linear shifting.

Input:  Filtered .pkl files from Step 2
Output: Calibrated .pkl files
"""

import pickle
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= Configuration =================
# Input directory (Output from Step 2)
INPUT_DIR = "./processed_peaks_step2"
# Output directory for calibrated data
OUTPUT_DIR = "./processed_peaks_step3_calibrated"

# Internal Standard Parameters
TARGET_MZ = 196.11      # Theoretical m/z
SEARCH_RANGE = 0.1      # Search window +/- 0.1 Da

# Calibration Mode Switch
# True  = Apply correction (Multiply by factor)
# False = QC Check only (Do not modify data)
ENABLE_ALIGNMENT = True  
# ===============================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pkl_files = glob.glob(os.path.join(INPUT_DIR, "*.pkl"))
    
    mode_str = "ALIGNMENT (Scaling)" if ENABLE_ALIGNMENT else "QC CHECK ONLY"
    print(f"📂 Input Directory: {INPUT_DIR}")
    print(f"⚙️  Mode: [{mode_str}]")
    print(f"🎯 Target IS: {TARGET_MZ} m/z")

    # Statistics Containers
    stats_before_diff = []  # Absolute error before calibration (Da)
    stats_factors = []      # Calculated scaling factors
    
    # Process each file
    for file_path in tqdm(pkl_files, desc="Calibrating"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            frames = data['peaks']
            new_frames = []

            for frame in frames:
                mzs = frame['mzs']
                ints = frame['areas']
                
                # 1. Locate Internal Standard
                mask = (mzs >= TARGET_MZ - SEARCH_RANGE) & (mzs <= TARGET_MZ + SEARCH_RANGE)
                
                scale_factor = 1.0 # Default (No scaling)
                obs_mz = 0.0
                
                if np.any(mask):
                    # Pick the most intense peak as IS
                    subset_mzs = mzs[mask]
                    subset_ints = ints[mask]
                    idx_max = np.argmax(subset_ints)
                    obs_mz = subset_mzs[idx_max]
                    
                    # 2. Calculate Scaling Factor
                    # Factor = Target / Observed
                    # Example: Target 100, Obs 99 -> Factor = 1.0101...
                    if obs_mz != 0:
                        scale_factor = TARGET_MZ / obs_mz
                    
                    # Record Statistics
                    diff = obs_mz - TARGET_MZ
                    stats_before_diff.append(diff)
                    stats_factors.append(scale_factor)
                
                # 3. Apply Calibration (if enabled)
                if ENABLE_ALIGNMENT and obs_mz != 0:
                    # Multiplicative Correction
                    corrected_mzs = mzs * scale_factor
                else:
                    # Keep original
                    corrected_mzs = mzs

                # 4. Save Metadata
                frame['mzs'] = corrected_mzs
                frame['calibration_factor'] = scale_factor 
                frame['is_aligned'] = ENABLE_ALIGNMENT
                
                new_frames.append(frame)

            # Save Processed File
            data['peaks'] = new_frames
            
            filename = os.path.basename(file_path)
            save_path = os.path.join(OUTPUT_DIR, filename)
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    # ================= QC Report =================
    print("\n" + "="*50)
    print(f"📊 Calibration Report (Enabled: {ENABLE_ALIGNMENT})")
    
    if len(stats_before_diff) > 0:
        arr_diff = np.array(stats_before_diff)
        arr_factors = np.array(stats_factors)
        
        # Calculate PPM error
        ppm_before = (arr_diff / TARGET_MZ) * 1e6

        print(f"\n1️⃣  Drift Before Calibration:")
        print(f"   Mean Absolute Error: {np.mean(arr_diff):.6f} Da")
        print(f"   Mean PPM Error:      {np.mean(np.abs(ppm_before)):.2f} ppm")
        print(f"   Max Absolute Error:  {np.max(np.abs(arr_diff)):.6f} Da")

        print(f"\n2️⃣  Scaling Factors:")
        print(f"   Mean Factor: {np.mean(arr_factors):.8f}")
        print(f"   (Factor > 1: Instrument reading low; Factor < 1: Instrument reading high)")
        
        if ENABLE_ALIGNMENT:
            print(f"\n3️⃣  Status:")
            print(f"   🎉 All scans have been multiplicatively scaled to align IS to {TARGET_MZ} m/z")
    else:
        print("⚠️ WARNING: No Internal Standard found for statistics.")

    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
