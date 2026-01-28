# “Nano-Filter”-Integrated AIMS with Machine Learning: Fast Exhaled Breath Analysis for Lung Cancer Screening


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()

<img width="100%" alt="Workflow Overview" src="https://github.com/user-attachments/assets/b3518ac3-37ca-48a4-b689-46537828cfac" />

## 📖 Overview
This repository (**AIMS**) contains the source code for an **AI-driven metabolomics analysis framework** designed for high-precision lung cancer (LCa) screening.

Unlike conventional pipelines, this project integrates **physics-informed signal processing** with machine learning to reconstruct high-fidelity metabolic profiles from raw mass spectrometry (LC-MS) data. It addresses common challenges such as instrumental drift, isotopic interference, and high-dimensional noise.

## 🚀 Key Features

### 1. High-Fidelity Signal Reconstruction
* **Physics-Constrained Peak Picking**: Dynamic noise filtering based on instrument resolution ($R=70,000$) and FWHM constraints.
* **Temporal Filtering**: Automatic removal of unstable scans (RT < 0.5 min).
* **Global Multiplicative Calibration**: Non-linear mass drift correction using Internal Standard (IS) references.

### 2. Deep Feature Deconvolution
* **Greedy Isotopic Blockout Algorithm**: A custom algorithm to identify and mask isotopic clusters ($M-2$ to $M+1$) within a strict mass tolerance (0.03 Da), eliminating redundant features.
* **Dynamic Frequency Aggregation**: Merges multi-frame scan data into a single representative profile with occupancy filtering (>5%) to remove sporadic noise.

### 3. Diagnostic Modeling
* **Biomarker Discovery**: Feature selection using **Random Forest-based Importance Screening** (Top 500 features).
* **Clinical Classifier**: An **XGBoost** model optimized via 5-fold stratified cross-validation, achieving robust performance (AUC > 0.99) in both internal and external validation cohorts.

## 📂 Repository Structure

The codebase is organized into two main modules containing 8 sequential steps:

```text
AIMS/
├── 01_Preprocessing/               # Phase I & II: Signal Processing
│   ├── 01_physics_peak_picking.py    # Raw .mzML processing & Physics-based filtering
│   ├── 02_is_quality_control.py      # Internal Standard (IS) filtering
│   ├── 03_mass_calibration.py        # Global multiplicative mass calibration
│   ├── 04_feature_extraction.py      # Greedy Isotopic Blockout algorithm
│   ├── 05_sample_aggregation.py      # Multi-frame merging & Occupancy filtering
│   └── 06_rf_feature_selection.py    # Random Forest for Biomarker Discovery
│
├── 02_Modeling/                    # Phase III: AI Modeling & Inference
│   ├── 07_model_training_cv.py       # XGBoost Training (5-Fold CV)
│   └── 08_model_inference.py         # Inference script for new samples
│
├── train_healthy/                  # Training Data (Healthy Controls)
├── train_cancer/                   # Training Data (Lung Cancer Patients)
├── test_data/                      # Inference Data (New Samples)
└── README.md
