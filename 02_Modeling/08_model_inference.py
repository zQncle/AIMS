"""
Step 8: Model Inference (Detection on New Samples)
--------------------------------------------------
This script applies the trained XGBoost model to new/test samples.

Methodology:
1. Load the trained model (from Step 7).
2. Load the Top 500 Biomarker list (from Step 6) to ensure feature alignment.
   (The model expects the exact same 500 m/z features in the same order).
3. Read new samples (.csv), extract the specific intensities for those 500 features.
4. Predict probabilities and assign class labels based on the Decision Threshold.
5. Output results to a CSV file.

Input:
  - Trained Model (xgb_lung_model_top500.pkl)
  - Feature List (top_biomarkers.csv)
  - New Sample CSVs in 'test_data/' folder
Output:
  - inference_results.csv
"""

import os
import numpy as np
import pandas as pd
from glob import glob
import joblib

# ================= Configuration =================
# Path to the Trained Model (Output from Step 7)
MODEL_PATH = "xgb_lung_model_top500.pkl"

# Feature List File (Output from Step 6)
# CRITICAL: Must match the features used during training!
BIOMARKER_FILE = "./results_step6_biomarkers/top_biomarkers.csv"
TOP_N_FEATURES = 500

# Input directory containing NEW samples (.csv)
# You need to create this folder and put test CSVs inside.
INPUT_FOLDER = "test_data"

# Output results file
OUTPUT_CSV = "inference_results.csv"

# Decision Threshold (Median Threshold from Step 7)
# Update this value based on Step 7's output for best accuracy.
DECISION_THRESHOLD = 0.6356 
# ===============================================

def load_target_features(csv_path, top_n):
    """Load the exact m/z features required by the model."""
    print(f"Loading Feature List: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.sort_values("Rank").head(top_n)
    
    # Extract m/z as float, rounded to 3 decimals
    target_mzs = df['m/z'].astype(float).round(3).values
    
    print(f"✅ Locked {len(target_mzs)} features for inference.")
    return target_mzs

def process_inference_sample(file_path, target_mzs):
    """
    Read a test sample and extract the specific feature vector.
    Returns None if file reading fails.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Build lookup dict: {mz: intensity}
        sample_dict = dict(zip(df['m/z'].astype(float).round(3), df['Intensity']))
        
        # Construct vector (fill 0.0 if missing)
        feature_vector = []
        for tmz in target_mzs:
            val = sample_dict.get(tmz, 0.0)
            feature_vector.append(val)
            
        return feature_vector
        
    except Exception as e:
        print(f"⚠️ Parsing failed: {os.path.basename(file_path)} - {e}")
        return None

def main():
    # ================= 1. Load Resources =================
    print(f"Loading Model: {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

    clf = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")

    # Load Target Features
    try:
        TARGET_MZS = load_target_features(BIOMARKER_FILE, TOP_N_FEATURES)
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return

    # ================= 2. Process Files =================
    if not os.path.exists(INPUT_FOLDER):
        print(f"⚠️ Input folder '{INPUT_FOLDER}' does not exist. Creating it...")
        os.makedirs(INPUT_FOLDER)
        print(f"❌ Please put .csv files into '{INPUT_FOLDER}' and run again.")
        return

    csv_files = sorted(glob(os.path.join(INPUT_FOLDER, "*.csv")))
    print(f"📂 Found {len(csv_files)} files in {INPUT_FOLDER}")

    if len(csv_files) == 0:
        print("❌ No CSV files found to test.")
        return

    X_new = []
    file_names = []

    for f in csv_files:
        vec = process_inference_sample(f, TARGET_MZS)
        if vec is not None:
            X_new.append(vec)
            file_names.append(os.path.basename(f))

    X_new = np.array(X_new)
    print(f"📌 Inference Matrix Shape: {X_new.shape}")

    # ================= 3. Inference =================
    print("🚀 Running Prediction...")
    # Get probability of Class 1 (Lung Cancer)
    y_prob = clf.predict_proba(X_new)[:, 1]

    # Apply Threshold
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    # ================= 4. Save Results =================
    df_out = pd.DataFrame({
        "file_name": file_names,
        "pred_label": y_pred,
        "pred_prob": y_prob,
    })

    # Map labels to text
    df_out["prediction_interpret"] = df_out["pred_label"].map({
        0: "Healthy",
        1: "Lung Cancer"
    })

    # Sort by probability descending (High risk first)
    df_out = df_out.sort_values(by="pred_prob", ascending=False)

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Results saved to: {OUTPUT_CSV}")

    # ================= 5. Statistics & Report =================
    n_total = len(df_out)
    n_pos = (df_out["pred_label"] == 1).sum()
    n_neg = n_total - n_pos

    print("\n📊 Summary:")
    print(f"   Total Samples: {n_total}")
    print(f"   Predicted Cancer:  {n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"   Predicted Healthy: {n_neg} ({n_neg/n_total*100:.1f}%)")

    # Check for Uncertain Samples (Borderline cases)
    # Define uncertainty zone as Threshold +/- 10%
    uncertain_mask = np.abs(df_out["pred_prob"] - DECISION_THRESHOLD) < 0.1
    uncertain = df_out[uncertain_mask]

    if len(uncertain) > 0:
        print(f"\n⚠️ Borderline Samples (Prob ≈ {DECISION_THRESHOLD} ± 0.1):")
        for _, row in uncertain.head(5).iterrows():
            print(f"   - {row['file_name']}: {row['pred_prob']:.4f}")
    else:
        print("\n✅ All predictions are high confidence.")

if __name__ == "__main__":
    main()
