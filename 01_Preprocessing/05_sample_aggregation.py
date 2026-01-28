"""
Step 5: Dynamic Frequency Aggregation & Occupancy Filtering
-----------------------------------------------------------
This script aggregates multi-frame scan data into a single representative 
metabolic profile for each sample.

Methodology:
1. Merge Features: Collect all extracted features from all valid scans of a sample.
2. Group by m/z: Calculate the mean intensity and occurrence frequency for each m/z.
3. Occupancy Filtering: 
   - Calculate Occupancy Rate = (Count of Appearance) / (Total Scans).
   - Filter out features with Occupancy Rate < Threshold (e.g., 5%).
   - This effectively removes sporadic noise and transient artifacts.

Input:  .pkl files from Step 4 (Feature Lists per Scan)
Output: .csv files per sample (Aggregated Feature List)
"""

import pickle
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= Configuration =================
# Input directory (Output from Step 4)
INPUT_DIR = "./processed_peaks_step4_features"
# Output directory for aggregated CSVs
OUTPUT_DIR = "./processed_peaks_step5_aggregated_csv"

# Core Parameter: Occupancy Rate Threshold
# 0.05 means a feature must appear in at least 5% of the scans to be kept.
MIN_OCCUPANCY_RATE = 0.05 
# ===============================================

def process_single_sample(file_path):
    """
    Read a sample .pkl, merge all frames, and apply occupancy filtering.
    Returns:
        DataFrame: Aggregated feature list (m/z, Intensity)
        str: Original filename (base)
        int: Total number of frames processed
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        filename = data['filename']
        frames = data['peaks'] # List of frames
        total_frames = len(frames)
        
        if total_frames == 0:
            return None, filename, 0

        # 1. Collect data from all frames
        # Flatten lists of m/z and intensities
        all_mzs = []
        all_ints = []
        
        for frame in frames:
            # Step 4 saved mzs/intensities as numpy arrays (float32)
            all_mzs.extend(frame['mzs'])
            all_ints.extend(frame['intensities'])
            
        if len(all_mzs) == 0:
            return None, filename, total_frames

        # 2. Build DataFrame
        df = pd.DataFrame({
            'm/z': all_mzs,
            'Intensity': all_ints
        })
        
        # 3. Aggregation (Core Logic)
        # Group by m/z (which was already rounded/binned in Step 4)
        # count: How many times this m/z appeared
        # mean:  Average intensity across valid scans
        agg_df = df.groupby('m/z')['Intensity'].agg(['mean', 'count']).reset_index()
        
        # 4. Occupancy Filtering
        agg_df['occupancy'] = agg_df['count'] / total_frames
        
        # Keep features that appear frequently enough (Occupancy > Threshold)
        final_df = agg_df[agg_df['occupancy'] > MIN_OCCUPANCY_RATE].copy()
        
        # 5. Formatting
        # Keep only m/z and Mean Intensity
        final_df = final_df[['m/z', 'mean']].rename(columns={'mean': 'Intensity'})
        
        # Sort by m/z
        final_df = final_df.sort_values(by='m/z')
        
        return final_df, filename, total_frames

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, "Error", 0

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pkl_files = glob.glob(os.path.join(INPUT_DIR, "*.pkl"))
    print(f"📂 Input Directory: {INPUT_DIR}")
    print(f"📄 Files to Process: {len(pkl_files)}")
    print(f"⚙️  Occupancy Threshold: > {MIN_OCCUPANCY_RATE*100}%")

    processed_count = 0
    
    for file_path in tqdm(pkl_files, desc="Aggregating Samples"):
        # Process individual sample
        df, original_filename, n_frames = process_single_sample(file_path)
        
        if df is not None and not df.empty:
            # Generate CSV filename (remove .mzML extension if present)
            base_name = os.path.splitext(original_filename)[0]
            # Ensure no path components remain
            base_name = os.path.basename(base_name)
            
            save_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
            
            # Save as CSV (ready for Matrix generation)
            df.to_csv(save_path, index=False)
            processed_count += 1
            
    print("\n" + "="*50)
    print(f"✅ Aggregation Complete!")
    print(f"Successfully converted: {processed_count} samples")
    print(f"CSV files saved in: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
