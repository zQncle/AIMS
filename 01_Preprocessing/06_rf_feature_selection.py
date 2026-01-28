"""
Step 6: Random Forest Feature Importance Screening
--------------------------------------------------
This script performs dimensionality reduction and biomarker discovery using 
Random Forest.

Methodology:
1. Load all aggregated CSVs and build a Feature Matrix (Rows=Samples, Cols=m/z).
2. Feature Alignment: Merge features that are extremely close (e.g., < 0.005 Da difference)
   to correct for rounding errors from previous steps.
3. Train a Random Forest Classifier to distinguish Healthy vs. Lung Cancer.
4. Extract Feature Importance scores and select the Top N (e.g., 500) most discriminative features.
5. Generate reports and visualizations (Boxplots of Top 9 biomarkers).

Input:  .csv files from Step 5
Output: 
  - top_biomarkers.csv (List of Top 500 features with stats)
  - feature_matrix.csv (Aligned full data matrix)
  - top_biomarkers_boxplot.png (Visualization)
"""

import pandas as pd
import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg') # Backend for server (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# ================= Configuration =================
# Input directory (Output from Step 5)
INPUT_DIR = "./processed_peaks_step5_aggregated_csv"
# Output directory for results
OUTPUT_DIR = "./results_step6_biomarkers"

# Number of top features to select
TOP_N = 500    

# Feature Alignment Tolerance (Da)
# Merges columns like 'mz_217.123' and 'mz_217.124'
ALIGNMENT_TOLERANCE = 0.005 
# ===============================================

def parse_label(filename):
    """
    Derive label from filename.
    Returns: 1 for Lung Cancer, 0 for Healthy, -1 for Unknown
    """
    name_lower = filename.lower()
    if 'lung' in name_lower:
        return 1
    elif 'health' in name_lower:
        return 0
    else:
        return -1 

def merge_close_features(df, tolerance=0.005):
    """
    Post-processing: Merge feature columns that are chemically identical 
    but separated due to binning/rounding artifacts.
    """
    print(f"🔄 Aligning features (Tolerance: {tolerance} Da)...")
    
    feat_cols = [c for c in df.columns if c.startswith("mz_")]
    meta_cols = [c for c in df.columns if not c.startswith("mz_")]
    
    # Parse m/z values
    mz_map = []
    for c in feat_cols:
        try:
            val = float(c.split('_')[1])
            mz_map.append({'mz': val, 'col': c})
        except:
            pass
    
    # Sort by m/z
    mz_map.sort(key=lambda x: x['mz'])
    
    merged_groups = [] 
    skip_cols = set()
    
    for i in range(len(mz_map)):
        curr = mz_map[i]
        if curr['col'] in skip_cols:
            continue
            
        group_cols = [curr['col']]
        group_mz_sum = curr['mz']
        
        # Look ahead for neighbors
        for j in range(i + 1, len(mz_map)):
            next_feat = mz_map[j]
            if next_feat['col'] in skip_cols:
                continue
            
            if (next_feat['mz'] - curr['mz']) < tolerance:
                group_cols.append(next_feat['col'])
                group_mz_sum += next_feat['mz']
                skip_cols.add(next_feat['col'])
            else:
                break
        
        # Calculate new merged name
        avg_mz = group_mz_sum / len(group_cols)
        new_col_name = f"mz_{avg_mz:.3f}"
        
        merged_groups.append({
            'new_name': new_col_name,
            'sources': group_cols
        })
    
    print(f"   Original Features: {len(feat_cols)} -> Merged: {len(merged_groups)}")
    
    # Build new DataFrame
    new_df = df[meta_cols].copy()
    
    for group in tqdm(merged_groups, desc="Merging columns"):
        new_name = group['new_name']
        cols_to_sum = group['sources']
        # Sum intensities of merged columns
        new_df[new_name] = df[cols_to_sum].sum(axis=1)
        
    return new_df

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ================= 1. Build Matrix =================
    print(f"1. Loading samples from {INPUT_DIR} ...")
    
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if len(csv_files) == 0:
        print("❌ Error: No CSV files found!")
        return

    data_list = []
    print(f"   Found {len(csv_files)} samples. Building matrix...")

    for i, file_path in enumerate(csv_files):
        filename = os.path.basename(file_path)
        label = parse_label(filename)
        
        if label == -1:
            continue
            
        df_sample = pd.read_csv(file_path)
        
        sample_dict = {
            'filename': filename,
            'label': label
        }
        
        # Convert rows to dictionary keys
        for _, row in df_sample.iterrows():
            mz_val = float(row['m/z'])
            # Create column name (temporary, will be aligned later)
            col_name = f"mz_{mz_val:.3f}"
            sample_dict[col_name] = row['Intensity']
            
        data_list.append(sample_dict)
        
        if (i+1) % 50 == 0:
            print(f"   Loaded {i+1}/{len(csv_files)} samples...")

    # Convert to DataFrame and fill missing values with 0
    df_matrix = pd.DataFrame(data_list).fillna(0)
    print(f"   ✅ Raw Matrix Shape: {df_matrix.shape}")

    # ================= 2. Feature Alignment =================
    # This step fixes "split peaks" caused by rounding differences
    df_aligned = merge_close_features(df_matrix, tolerance=ALIGNMENT_TOLERANCE)
    
    # Save the aligned matrix (Input for XGBoost)
    matrix_save_path = os.path.join(OUTPUT_DIR, "aligned_feature_matrix.csv")
    df_aligned.to_csv(matrix_save_path, index=False)
    print(f"   ✅ Aligned Matrix saved to: {matrix_save_path}")

    # ================= 3. Random Forest Screening =================
    print("-" * 40)
    print("2. Running Random Forest Feature Selection ...")
    
    feature_cols = [c for c in df_aligned.columns if c.startswith("mz_")]
    
    if len(feature_cols) == 0:
        print("❌ Error: No feature columns found after alignment!")
        return

    X = df_aligned[feature_cols].values
    y = df_aligned['label'].values
    
    # Initialize Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get Feature Importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1] # Descending sort
    
    # ================= 4. Report Top N Features =================
    actual_top_n = min(TOP_N, len(feature_cols))
    print(f"3. Extracting Top {actual_top_n} Biomarkers ...")
    
    top_features_data = []
    
    for i in range(actual_top_n):
        idx = indices[i]
        feat_name = feature_cols[idx]
        score = importances[idx]
        
        real_mz = feat_name.replace("mz_", "")
        
        # Calculate Group Means
        mean_health = df_aligned[df_aligned['label'] == 0][feat_name].mean()
        mean_lung = df_aligned[df_aligned['label'] == 1][feat_name].mean()
        
        trend = "UP" if mean_lung > mean_health else "DOWN"
        
        # Calculate Fold Change
        fc = mean_lung / (mean_health + 1e-9) 
        
        top_features_data.append({
            'Rank': i + 1,
            'm/z': real_mz,
            'Feature Name': feat_name,
            'Importance': score,
            'Trend': trend,
            'Mean (Health)': mean_health,
            'Mean (Lung)': mean_lung,
            'Fold Change': fc
        })
        
    df_top = pd.DataFrame(top_features_data)
    
    # Save Top Features List
    biomarkers_path = os.path.join(OUTPUT_DIR, "top_biomarkers.csv")
    df_top.to_csv(biomarkers_path, index=False)
    print(f"   ✅ Biomarker List saved to: {biomarkers_path}")
    print(df_top[['Rank', 'm/z', 'Importance', 'Trend']].head(10))

    # ================= 5. Visualization =================
    print("-" * 40)
    print("4. Generating Boxplots for Top 9 Features ...")
    
    top_9_names = df_top['Feature Name'].head(9).values
    
    if len(top_9_names) > 0:
        plt.figure(figsize=(15, 12))
        sns.set_style("whitegrid")
        
        for i, col in enumerate(top_9_names):
            mz_display = col.replace("mz_", "m/z ")
            
            plt.subplot(3, 3, i + 1)
            
            # Boxplot with scatter overlay
            sns.boxplot(x='label', y=col, data=df_aligned, 
                        hue='label', legend=False,
                        showfliers=False, 
                        palette={0: '#a8dadc', 1: '#ffb3c1'}) # Blue/Red
            
            sns.stripplot(x='label', y=col, data=df_aligned, 
                          color='black', alpha=0.3, size=3)
            
            imp_val = df_top.iloc[i]['Importance']
            plt.title(f"{mz_display}\n(Imp: {imp_val:.4f})", fontsize=12)
            plt.ylabel("Normalized Intensity")
            plt.xlabel("")
            plt.xticks([0, 1], ['Healthy', 'Lung Cancer'])
            
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "top_biomarkers_boxplot.png")
        plt.savefig(plot_path, dpi=300)
        print(f"   ✅ Plot saved to: {plot_path}")

    print("\n🎉 Analysis Complete!")

if __name__ == "__main__":
    main()
