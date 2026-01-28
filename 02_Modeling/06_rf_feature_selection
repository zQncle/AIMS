"""
Step 7: XGBoost Model Training & Evaluation (5-Fold CV)
-------------------------------------------------------
This script trains the final diagnostic classifier using the Top 500 features 
identified in Step 6.

Methodology:
1. Load the Top 500 Biomarker List.
2. Load training data for Healthy and Cancer cohorts.
   (Assumes data is organized in 'train_healthy' and 'train_cancer' folders).
3. Extract specific feature intensities for the Top 500 m/z values.
4. Perform 5-Fold Stratified Cross-Validation (CV) using XGBoost.
5. Evaluate performance (AUC, Accuracy) and determine the optimal decision threshold.
6. Generate publication-quality visualizations (Boxplots of AUC/ACC).
7. Save the final trained model for deployment.

Input:  
  - step6_top_biomarkers.csv
  - Sample CSVs in train_healthy/ and train_cancer/ folders
Output:
  - xgb_lung_model_top500.pkl (Trained Model)
  - figure_auc_boxplot.png (Performance Plot)
"""

import os
import numpy as np
import pandas as pd
from glob import glob
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from xgboost import XGBClassifier

# ================= Configuration =================
# Data Directories (You need to organize your Step 5 CSVs into these folders)
# Or modify the script to read from labels in filenames if preferred.
HEALTHY_DIR = "train_healthy" 
LUNG_DIR = "train_cancer"

# Feature List File (Output from Step 6)
BIOMARKER_FILE = "./results_step6_biomarkers/top_biomarkers.csv"
TOP_N_FEATURES = 500

# Training Parameters
N_SPLITS = 5
RANDOM_STATE = 42

# Plotting Configuration (Optional Font)
FONT_PATH = "./arial.ttf" # Change to a valid font path if available

# ================= 1. Load Top Biomarkers =================
def load_target_features(csv_path, top_n):
    print(f"Loading Feature List: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Sort by Rank and take Top N
    df = df.sort_values("Rank").head(top_n)
    
    # Extract target m/z values (Round to 3 decimals, float)
    target_mzs = df['m/z'].astype(float).round(3).values
    
    print(f"✅ Extracted Top {len(target_mzs)} features for training.")
    return target_mzs

# Load Features
TARGET_MZS = load_target_features(BIOMARKER_FILE, TOP_N_FEATURES)

# ================= 2. Feature Extraction Function =================
def process_single_sample(file_path, target_mzs):
    """
    Read sample CSV and extract intensities for target m/z values.
    Uses O(1) dictionary lookup for speed.
    """
    try:
        # Read sample data
        df = pd.read_csv(file_path)
        
        # Build lookup dictionary: {mz: intensity}
        # Round m/z keys to 3 decimals to match target features
        sample_dict = dict(zip(df['m/z'].astype(float).round(3), df['Intensity']))
        
        # Construct Feature Vector
        feature_vector = []
        for tmz in target_mzs:
            # Exact match lookup
            val = sample_dict.get(tmz, 0.0)
            feature_vector.append(val)
            
        return feature_vector
        
    except Exception as e:
        print(f"❌ Failed to read {os.path.basename(file_path)}: {e}")
        return [0.0] * len(target_mzs)

# ================= 3. Prepare Dataset (X, y) =================
X, y = [], []

def load_folder_data(path, label, target_mzs):
    if not os.path.exists(path):
        print(f"⚠️ Warning: Directory not found: {path}")
        return

    files = sorted(glob(os.path.join(path, "*.csv")))
    print(f"Processing {path} ({len(files)} files)...")
    
    for f in files:
        vec = process_single_sample(f, target_mzs)
        X.append(vec)
        y.append(label)

# Load Data
# 0 = Healthy, 1 = Cancer
load_folder_data(HEALTHY_DIR, 0, TARGET_MZS)
load_folder_data(LUNG_DIR, 1, TARGET_MZS)

X = np.array(X)
y = np.array(y)

print("-" * 30)
print(f"Final Dataset Shape X: {X.shape}")
print(f"Label Distribution y: {np.bincount(y)} (0=Healthy, 1=Cancer)")
print("-" * 30)

if len(X) == 0:
    print("❌ Error: No data loaded. Check input directories.")
    exit()

# ================= 4. Cross-Validation Training =================
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

tprs = []
aucs = []
accs = []
best_thresholds = []
mean_fpr = np.linspace(0, 1, 200)

fold = 1

for train_idx, val_idx in kf.split(X, y):
    print(f"🚀 Fold {fold}/{N_SPLITS} processing...")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist"  # Use 'auto' if 'hist' fails on older XGB versions
    )
    clf.fit(X_train, y_train)

    # Validation Predictions
    y_prob = clf.predict_proba(X_val)[:, 1]

    # Metrics
    auc = roc_auc_score(y_val, y_prob)
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)

    # Youden Index for Optimal Threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_th = thresholds[best_idx]
    best_thresholds.append(best_th)

    y_pred = (y_prob >= best_th).astype(int)
    acc = accuracy_score(y_val, y_pred)

    print(f"   -> ACC={acc:.4f}, AUC={auc:.4f}, BestTh={best_th:.4f}")

    # Interpolate ROC Curve
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

    aucs.append(auc)
    accs.append(acc)

    fold += 1

# ================= 5. Save Final Model =================
decision_threshold = np.median(best_thresholds)
print(f"\n🔥 Median Decision Threshold = {decision_threshold:.4f}")

print("Retraining Final Model on ALL data...")
final_clf = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    tree_method="hist"
)
final_clf.fit(X, y)

# Save Model
model_filename = "xgb_lung_model_top500.pkl"
joblib.dump(final_clf, model_filename)
print(f"🎉 Model saved: {model_filename}")

# ================= 6. Plotting (Publication Style) =================
# Font Setup
try:
    if os.path.exists(FONT_PATH):
        font_manager.fontManager.addfont(FONT_PATH)
        arial_font = font_manager.FontProperties(fname=FONT_PATH)
        font_name = arial_font.get_name()
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [font_name]
        print(f"✅ Custom font loaded: {font_name}")
    else:
        print(f"⚠️ Font not found: {FONT_PATH}, using default.")
except Exception as e:
    print(f"⚠️ Font loading error: {e}, using default.")

# Style Settings
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
AXIS_LW = 2.2
LINE_LW = 1.2
LABEL_FS = 14
TICK_FS = 12
TITLE_FS = 15

def plot_metric_boxplot(data, title, ylabel, filename, y_min):
    fig, ax = plt.subplots(figsize=(4.8, 5.8))

    # Boxplot
    ax.boxplot(
        data,
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=LINE_LW),
        medianprops=dict(color="#d62728", linewidth=2),
        whiskerprops=dict(color="black", linewidth=LINE_LW),
        capprops=dict(color="black", linewidth=LINE_LW)
    )

    # Jitter Scatter
    x_jitter = 1 + np.random.normal(0, 0.03, size=len(data))
    ax.scatter(
        x_jitter,
        data,
        s=70,
        color="#1f77b4" if "AUC" in ylabel else "#2ca02c",
        edgecolor="none",
        zorder=3,
        label="Individual folds"
    )

    # Mean +/- SD Error Bar
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.errorbar(
        1, mean_val, yerr=std_val, fmt="o", color="#ff7f0e",
        elinewidth=LINE_LW, capsize=6, markersize=8, label="Mean ± SD"
    )

    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xticks([])
    ax.set_ylim(y_min, 1.01)

    ax.legend(frameon=False, loc="lower right")

    # Spines
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(AXIS_LW)
    
    ax.tick_params(axis="both", width=AXIS_LW, length=6, labelsize=TICK_FS)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"📊 Plot saved: {filename}")

# Plot AUC
plot_metric_boxplot(aucs, "AUC (5-fold CV)", "AUC", "figure_auc_boxplot.png", 0.90)

# Plot Accuracy
plot_metric_boxplot(accs, "Accuracy (5-fold CV)", "Accuracy", "figure_acc_boxplot.png", 0.90)

print("\nAll tasks completed successfully!")
