<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0a,50:1a1a2e,100:16213e&height=180&section=header&text=TCGA%20Cancer%20Prediction&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=ML-Powered%20Multi-Class%20Cancer%20Classification%20from%20Genomic%20Data&descAlignY=58&descColor=64ffda&animation=fadeIn" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-64ffda?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

**TCGA Cancer Prediction** is a comprehensive machine learning pipeline that classifies cancer types using gene expression data from **The Cancer Genome Atlas (TCGA)**. The project covers the full ML lifecycle — exploratory data analysis, classification, regression, unsupervised clustering, and a Hidden Markov Model — all run from a single Python script against a real TCGA parquet dataset.

---

## 🎯 Problem Statement

Cancer diagnosis from genomic data is a high-dimensional challenge. This project addresses:
- **Multi-class cancer type classification** across 5 TCGA tumour types (BRCA, LUAD, KIRC, COAD, PRAD)
- **Gene expression regression** — predicting one gene's expression from the rest of the transcriptome
- **Unsupervised clustering** — recovering biological cancer subtypes without labels
- **Generative sequence modelling** — using a Hidden Markov Model to capture tissue-specific expression regimes
- **Biomarker discovery** — identifying the top predictive genes in a leakage-free CV pipeline

---

## 🧬 Dataset

| Property | Details |
|---|---|
| **Source** | The Cancer Genome Atlas (TCGA) |
| **Format** | Parquet (RNA-seq gene expression) |
| **Cancer Types** | BRCA, LUAD, KIRC, COAD, PRAD |
| **Target** | Cancer type label (multi-class) |

> The raw data file (`data/tcga.parquet`) is excluded from this repo due to GitHub's 100 MB file size limit. Download it separately and place it in the `data/` folder before running.

---

## 🛠️ Pipeline

All analyses are implemented in `tcga_ml_project.py` and run end-to-end:

```
tcga.parquet
     │
     ▼
Exploratory Data Analysis (EDA)
  ├── Class distribution
  ├── Top gene heatmap
  ├── PCA + t-SNE visualisation
  └── Correlation analysis
     │
     ▼
Classification (Multi-class cancer type prediction)
  ├── Logistic Regression ✅ Best
  ├── Random Forest
  ├── XGBoost
  ├── SVM
  └── Gaussian HMM (generative baseline)
     │
     ▼
Regression (Gene co-expression modelling)
  ├── Lasso ✅ Best
  ├── Ridge
  └── ElasticNet
     │
     ▼
Clustering (Unsupervised cancer subtype recovery)
  ├── Spectral Clustering ✅ Best
  ├── K-Means
  └── Agglomerative
     │
     ▼
Outputs (results + visualisations saved to outputs/)
```

---

## 📊 Results

### Classification

| Model | Test Accuracy | Macro F1 | AUC |
|---|---|---|---|
| **Logistic Regression** | **99.38%** | **0.9947** | **0.9999** |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| SVM | — | — | — |
| Gaussian HMM (4 states) | 96.27% | — | — |

> Logistic Regression is the top performer — tumour types are near-linearly separable in high-dimensional gene expression space. The HMM achieves 96.27% accuracy as an interpretable generative model, with latent states capturing tissue-specific expression regimes.

### Regression

| Model | R² | RMSE |
|---|---|---|
| **Lasso** | **0.9453** | **1.5658** |

> Target gene `gene_9176` is strongly co-regulated with the rest of the transcriptome. Lasso's sparsity makes it the most interpretable choice.

### Clustering

| Method | ARI | NMI | Cluster Accuracy |
|---|---|---|---|
| **Spectral** | **0.9925** | **0.9895** | **99.75%** |

> Unsupervised clustering recovers all 5 cancer types almost perfectly — confirming strong biological structure in the expression data.

### Biomarkers

Top predictive genes are saved in `outputs/top_biomarkers.csv`. Feature selection was performed **inside cross-validation** to ensure zero data leakage.

---

## 📁 Project Structure

```
TCGA-CancerPrediction/
├── tcga_ml_project.py          # Full ML pipeline (EDA → clf → reg → clustering → HMM)
├── requirements.txt            # Python dependencies
├── data/
│   └── tcga.parquet            # ⚠️ Not included (>100MB) — download separately
└── outputs/
    ├── summary.txt                     # Pipeline results summary
    ├── top_biomarkers.csv              # Top predictive genes
    ├── classification_results.csv      # Per-model classification metrics
    ├── all_classifiers_incl_hmm.csv    # All classifier comparison
    ├── regression_results.csv          # Regression metrics
    ├── clustering_results.csv          # Clustering metrics
    ├── hmm_state_selection.csv         # HMM state count evaluation
    ├── eda_class_distribution.png
    ├── eda_pca_tsne.png
    ├── eda_top_genes_heatmap.png
    ├── eda_correlation.png
    ├── eda_expression_hist.png
    ├── eda_pca_variance.png
    ├── clf_confusion_matrix.png
    ├── clf_feature_importance.png
    ├── cmp_classifiers.png
    ├── hmm_confusion_matrix.png
    ├── hmm_transition_matrices.png
    ├── reg_actual_vs_pred.png
    ├── cmp_regression.png
    ├── clu_elbow_silhouette.png
    ├── clu_tsne_comparison.png
    ├── clu_contingency.png
    └── cmp_clustering.png
```

---

## 🚀 Getting Started

### Prerequisites
```
Python 3.10+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/angelbuncy/TCGA-CancerPrediction.git
cd TCGA-CancerPrediction

# Install dependencies
pip install -r requirements.txt

# Place tcga.parquet in the data/ folder, then run
python tcga_ml_project.py
```

---

## 📄 License

Distributed under the MIT License.

---

<div align="center">
  <p>Built by <a href="https://github.com/angelbuncy">Angel Buncy</a> · Powered by TCGA open data</p>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,50:1a1a2e,100:0a0a0a&height=80&section=footer" width="100%"/>
</div>
