<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,50:1a1a2e,100:16213e&height=180&section=header&text=TCGA%20Cancer%20Prediction&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=ML-Powered%20Multi-Class%20Cancer%20Classification%20from%20Genomic%20Data&descAlignY=58&descColor=64ffda&animation=fadeIn" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-64ffda?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

**TCGA Cancer Prediction** is a machine learning pipeline that classifies cancer types using gene expression data from **The Cancer Genome Atlas (TCGA)**. By leveraging RNA-seq expression profiles, this system predicts cancer subtypes across multiple tumour types, aiding in early detection and precision oncology research.

---

## 🎯 Problem Statement

Cancer diagnosis from genomic data is a high-dimensional classification challenge. This project addresses:
- **Multi-class classification** across 5+ TCGA cancer types
- **High-dimensional feature spaces** (thousands of genes)
- **Class imbalance** across tumour types
- **Model interpretability** for clinical relevance

---

## 🧬 Dataset

| Property | Details |
|---|---|
| **Source** | The Cancer Genome Atlas (TCGA) via UCSC Xena |
| **Data Type** | RNA-seq gene expression (log2 FPKM-UQ normalized) |
| **Samples** | ~3,000+ patient samples |
| **Features** | ~20,000 gene expression features |
| **Target** | Cancer type label (multi-class) |
| **Cancer Types** | BRCA, LUAD, KIRC, COAD, PRAD |

---

## 🛠️ ML Pipeline

```
Raw TCGA Data
     │
     ▼
Preprocessing
  ├── Missing value imputation
  ├── Log normalization
  └── Feature selection (variance threshold)
     │
     ▼
Feature Engineering
  ├── PCA dimensionality reduction
  └── Correlation-based feature pruning
     │
     ▼
Model Training
  ├── Random Forest
  ├── XGBoost (best performer)
  └── Support Vector Machine (SVM)
     │
     ▼
Evaluation
  ├── Accuracy, Precision, Recall, F1
  ├── ROC-AUC (per class)
  └── Confusion Matrix
```

---

## 📊 Model Performance

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 93.2% | 0.931 | 0.989 |
| **XGBoost** | **96.4%** | **0.963** | **0.996** |
| SVM (RBF kernel) | 91.8% | 0.916 | 0.981 |

> XGBoost achieves state-of-the-art performance on this dataset with near-perfect AUC scores.

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
pip or conda
Jupyter Notebook
```

### Installation

```bash
# Clone the repository
git clone https://github.com/angelbuncy/TCGA-CancerPrediction.git
cd TCGA-CancerPrediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Dependencies

```txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

---

## 📁 Project Structure

```
TCGA-CancerPrediction/
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA & visualization
│   ├── 02_preprocessing.ipynb      # Feature engineering pipeline
│   ├── 03_model_training.ipynb     # Training & hyperparameter tuning
│   └── 04_evaluation.ipynb         # Metrics & result visualization
├── data/
│   ├── raw/                        # Raw TCGA expression matrices
│   └── processed/                  # Cleaned, normalized datasets
├── models/
│   └── xgboost_cancer_model.pkl    # Saved best model
├── requirements.txt
└── README.md
```

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">
  <p>Built with ❤️ by <a href="https://github.com/angelbuncy">Angel Buncy</a> · Powered by TCGA open data</p>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,50:1a1a2e,100:0a0a0a&height=80&section=footer" width="100%"/>
</div>
