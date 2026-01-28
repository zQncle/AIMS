# Physics-Informed Computational Pipeline for Lung Cancer Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()

## 📖 Overview
This repository contains the source code for an **AI-driven metabolomics analysis framework** designed for high-precision lung cancer (LCa) screening. 

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

## 🛠️ Workflow

The pipeline is organized into three main phases:

1.  **Phase I: Data Processing & Calibration**
    * Raw `.mzML` parsing -> Peak picking -> RT & IS filtering -> Mass calibration.
2.  **Phase II: Feature Extraction**
    * Isotopic pattern recognition (Greedy Blockout) -> Normalization (to IS) -> Sample aggregation.
3.  **Phase III: Machine Learning**
    * Feature alignment -> Matrix generation -> RF screening -> XGBoost training & evaluation.

## 📦 Installation & Requirements

```bash
# Clone the repository
git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)

# Install dependencies
pip install -r requirements.txt
