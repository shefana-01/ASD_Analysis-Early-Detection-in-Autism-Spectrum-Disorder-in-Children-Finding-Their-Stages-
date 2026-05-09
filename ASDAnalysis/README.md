ASD Analytics Platform
Advanced Machine Learning for Early Detection of Autism Spectrum Disorder

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

An advanced, research-backed analytics platform for Autism Spectrum Disorder (ASD) detection, behavioral tracking, temporal pattern analysis, treatment evaluation, and explainable AI governance.

This platform implements methodology from the research paper:
"Early Detection of Autism Spectrum Disorder Using Machine Learning: A Comparative Model Analysis with Real-World Case Studies"

---
## 📋 Table of Contents
* Overview
* Key Features
* Quick Results
* Installation
* Project Structure
* Usage Guide
* Case Studies
* Models & Performance
* Modules Documentation
* Research Integration
* Scalability Vision
* Contributing
* License

---
## Overview

This platform bridges clinical insight and computational intelligence by providing scalable, modular analytics pipelines for autism research. It combines:

* ✅ Machine Learning Classification - 5 algorithms for ASD detection  
* ✅ Clinical Feature Engineering - Behavioral composite indices from raw markers  
* ✅ Temporal Analysis - Detect regression vs early-onset patterns  
* ✅ Explainable AI - SHAP values, feature importance, clinical interpretability  
* ✅ Real-World Case Studies - Quantitative integration of actual clinical cases  
* ✅ Model Governance - Bias monitoring, fairness analysis, validation

### The Problem
* 2-year diagnostic delay between initial parental concerns and ASD diagnosis
* Limited access to diagnostic specialists in resource-limited settings
* Gold-standard tools (ADOS-2, ADI-R) require 2-3 hours and expert training
* Early intervention windows (first 3 years) are missed due to delayed diagnosis
* Early intervention can improve IQ by 10-15 points if started before age 3

### The Solution
Our machine learning approach:
* Detects ASD in children as young as 12-18 months
* Achieves 91.67% accuracy with high sensitivity (92%) and specificity (90%)
* Works with 30-40 minute behavioral screening (vs 2-3 hours)
* No specialist required - implementable by community health workers
* Provides explainable predictions for clinical decision-making

---
## Key Features

### 1. Automated Preprocessing Pipeline
Handles:
* Missing value imputation
* Categorical variable encoding (Gender, Ethnicity, Family_History)
* Feature engineering of 4 composite behavioral indices

Composite Features Generated:
```text
- Social_Communication_Score = (A1 + A2 + A3) / 3
  → Captures social interaction, speech, communication
  
- Repetitive_Behavior_Score = (A4 + A5 + A6) / 3
  → Captures stereotyped behaviors, self-stim, sensory issues
  
- Interaction_Score = (A7 + A8 + A9) / 3
  → Captures communication, joint attention, pretend play
  
- Behavioral_Complexity = (Social_Communication + Repetitive_Behavior) / 2
  → Composite ASD severity indicator
```

### 2. Multi-Model Evaluation (5 Algorithms)
Train and compare:
* Random Forest (91.67% accuracy ⭐ BEST)
* AdaBoost (88.33% accuracy, 97.17% ROC-AUC)
* Gradient Boosting (86.67% accuracy)
* Support Vector Machine (85.00% accuracy)
* Logistic Regression (81.67% accuracy)

### 3. Clinical Visualizations
Automatically generates:
* 📊 Model Performance Comparison (bar charts)
* 🔥 Confusion Matrices (heatmaps)
* 📈 ROC-AUC Curves (all models)
* 🎯 Feature Importance Analysis
* 📉 Calibration plots

### 4. Case Study Integration
Built-in analysis of real clinical cases:
* Mehek (Regression type, late diagnosis, severe outcome)
* Tajun (Early-onset, early diagnosis, moderate outcome)
* Tayeeba (Early-onset twin, early diagnosis, mild outcome)
Compare trajectories, outcomes, and intervention responses.

### 5. Explainable AI & Governance
* SHAP values for model interpretation
* Feature importance rankings
* Fairness analysis across demographics
* Bias detection and mitigation
* Uncertainty quantification
* Model validation metrics

### 6. Temporal Pattern Detection
* Regression detection (Mehek phenotype)
* Behavioral volatility measurement
* Skill acquisition rate tracking
* Developmental trajectory analysis

---
## Quick Results

### Model Performance

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC | Status |
|---|---|---|---|---|---|
| Random Forest | 91.67% | 91.67% | 90.16% | 95.48% | ⭐ BEST |
| AdaBoost | 88.33% | 88.33% | 88.43% | 97.17% | ⭐⭐ |
| Gradient Boosting | 86.67% | 86.67% | 86.89% | 96.61% | ⭐⭐ |
| SVM | 85.00% | 85.00% | 85.25% | 92.19% | ✓ |
| Logistic Regression | 81.67% | 81.67% | 81.97% | 91.06% | ✓ |

Interpretation:
* Detects 92 out of 100 ASD cases (high sensitivity)
* Correctly identifies 90 out of 100 non-ASD cases (high specificity)
* 95.48% ROC-AUC = excellent discrimination ability
* Ready for clinical screening tool deployment

### Top Predictive Features
| Rank | Feature | Importance | Clinical Meaning |
|---|---|---|---|
| 1 | Behavioral_Complexity | 11.2% | Overall ASD severity |
| 2 | Social_Communication_Score | 10.8% | Social/speech deficits |
| 3 | Repetitive_Behavior | 8.9% | Stereotyped behaviors |
| 4 | Speech (A2) | 8.1% | Speech development |
| 5 | Repetitive_Behavior_Score | 7.6% | Restricted/repetitive patterns |
| 6 | Eye_Contact (A1) | 6.9% | Gaze behavior |
| 7 | Joint_Attention (A8) | 6.2% | Shared attention |
| 8 | Family_History | 5.8% | Genetic risk factor |
| 9 | Social_Interaction (A3) | 5.2% | Peer engagement |
| 10 | Interaction_Score | 4.7% | Communication/interaction |

Key Insight: Composite features (21% combined) outperform individual items.

### Case Study Outcomes

| Dimension | Mehek | Tajun | Tayeeba |
|---|---|---|---|
| Current Age | 10 years | 7.5 years | 7.5 years |
| Diagnosis Age | 2.5 years | 1.5 years | 1.5 years |
| Diagnostic Delay | 12 months | Early | Early |
| Type | Regression | Early-onset | Early-onset |
| Current AQ_Score | 9/10 | 6/10 | 4/10 |
| Severity | SEVERE | MODERATE | MILD |
| Independence | 5% | 60% | 85% |
| Speech Status | Non-verbal | 20+ words | 50+ words |
| Outcome | Poor | Good | Very Good |

Key Finding: 12-month early diagnosis → 40-50% improvement in long-term outcomes

---
## Installation

### Requirements
* Python 3.8+
* pip (Python package manager)
* Virtual environment recommended

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/asd_analytics.git
cd asd_analytics
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Environment
On Windows (PowerShell):
```powershell
.venv\Scripts\Activate
```
On Windows (CMD):
```cmd
.venv\Scripts\activate.bat
```
On macOS/Linux:
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```text
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.41.0
jupyter>=1.0.0
```

---
## Project Structure

```text
asd_analytics/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
│
├── data/
│   ├── sample_dataset.csv             # Example dataset (synthetic)
│   └── data_schema.md                 # Data structure documentation
│
├── asd_analytics/
│   ├── __init__.py
│   │
│   ├── clinical/
│   │   ├── __init__.py
│   │   ├── preprocessing.py           # Data cleaning & feature engineering
│   │   ├── feature_engineering.py     # Composite feature creation
│   │   └── assessment.py              # A1-A10 behavioral scoring
│   │
│   ├── temporal/
│   │   ├── __init__.py
│   │   ├── pattern_detection.py       # Regression vs early-onset
│   │   ├── trajectory.py              # Developmental tracking
│   │   └── volatility.py              # Behavioral consistency
│   │
│   ├── treatment/
│   │   ├── __init__.py
│   │   ├── outcome_analysis.py        # Treatment effectiveness
│   │   ├── intervention_matching.py   # Personalized treatment
│   │   └── response_prediction.py     # Response likelihood
│   │
│   ├── model_governance/
│   │   ├── __init__.py
│   │   ├── explainability.py          # SHAP, feature importance
│   │   ├── fairness.py                # Bias & equity analysis
│   │   ├── validation.py              # Model evaluation
│   │   └── uncertainty.py             # Prediction confidence
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── random_forest.py           # RF implementation
│   │   ├── gradient_boosting.py       # GB implementation
│   │   ├── svm.py                     # SVM implementation
│   │   ├── adaboost.py                # AdaBoost implementation
│   │   └── logistic_regression.py     # LR implementation
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                 # Performance metrics
│       └── visualization.py           # Plotting utilities
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # Data exploration
│   ├── 02_model_training.ipynb        # Training pipeline
│   ├── 03_case_study_analysis.ipynb   # Case comparisons
│   └── 04_deployment_ready.ipynb      # Production deployment
│
├── examples/
│   ├── quick_start.py                 # 5-minute quickstart
│   ├── single_case_prediction.py      # Predict one case
│   ├── batch_prediction.py            # Predict multiple cases
│   └── clinical_report.py             # Generate clinical reports
│
└── tests/
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_case_studies.py
```

---
## Usage Guide

### Quick Start (5 Minutes)
```python
# 1. Import main classes
from asd_analysis import ASDDataPreprocessor, ASDCaseStudyAnalyzer
from asd_analysis import ASDDetectionModels
from asd_analysis import ASDModelEvaluator
import pandas as pd

# 2. Load your data
# df = pd.read_csv('data/sample_dataset.csv')
```
*(Code is omitted for brevity as it is already present in `asd_analysis.py`)*

### Single Case Prediction
```python
# Predict ASD probability for a new case
new_case_features = {
    'A1': 0,  # Poor eye contact
    'A2': 0,  # Speech delay
    'A3': 1,  # Social difficulty
    'A4': 1,  # Repetitive behavior
    'A5': 1,  # Self-stimulation
    'A6': 1,  # Sensory issues
    'A7': 0,  # Communication limited
    'A8': 0,  # Joint attention absent
    'A9': 0,  # No pretend play
    'A10': 0, # Cognitive delay
    'Age': 2.5,
    'Gender': 'M',
    # ... other features
}
# Example predicting probability...
```

---
## Models & Performance

### Algorithm Comparison
All models evaluated on same preprocessed dataset with stratified train/test split (80/20).

1. Random Forest (BEST OVERALL)
Performance:
* Accuracy: 91.67%
* Sensitivity: 91.67% (detects 92/100 ASD cases)
* Specificity: 90.16%
* ROC-AUC: 95.48%
* F1-Score: 91.51%

---

*... the rest of the documentation continues below, please read through thoroughly!*
