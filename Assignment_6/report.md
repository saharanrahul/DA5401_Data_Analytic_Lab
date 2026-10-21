# DA5401 — Data Analytics Lab: Assignment 6 Report

## Student Roll Number / Seed
- **Roll Number**: MA25C033
- **Student Seed**: `33` (used for the train/test split, the 4-fold stratified CV, Random Search and the Optuna TPE sampler; all Random Forest / Isolation Forest models use `random_state=42`)
- **Environment**: Python 3.12, scikit-learn 1.9.0

---

## Problem 1 — Density-Based Clustering

All features were standardized with `StandardScaler`. Silhouette scores are computed on non-noise observations only.

### 1. DBSCAN Results (every configuration)

| eps | min_samples | Num Clusters | Cluster Sizes | Noise Fraction | Silhouette |
| :---: | :---: | :---: | :--- | :---: | :---: |
| 0.12 | 5 | 4 | [340, 472, 491, 953] | 0.0792 | **0.5169** |
| 0.18 | 5 | 5 | [5, 7, 478, 492, 1311] | 0.0641 | 0.2680 |
| 0.24 | 8 | 3 | [490, 495, 1328] | 0.0559 | 0.4918 |
| 0.30 | 10 | 4 | [10, 493, 500, 1338] | 0.0445 | 0.4010 |

**Selected DBSCAN configuration:** `eps=0.12`, `min_samples=5` (highest silhouette among configurations with at least two clusters)
- Number of clusters: 4
- Cluster sizes: [340, 472, 491, 953]
- Noise fraction: 0.0792
- Silhouette score: 0.5169

### 2. HDBSCAN Results (every configuration, `min_samples=None`, Euclidean)

| min_cluster_size | Num Clusters | Cluster Sizes | Noise Fraction | Silhouette |
| :---: | :---: | :--- | :---: | :---: |
| 20 | 4 | [337, 481, 495, 950] | 0.0763 | **0.5159** |
| 40 | 3 | [475, 495, 1316] | 0.0669 | 0.4969 |
| 70 | 3 | [474, 495, 1305] | 0.0718 | 0.5001 |
| 100 | 3 | [473, 494, 1286] | 0.0804 | 0.5062 |

**Selected HDBSCAN configuration:** `min_cluster_size=20`, `min_samples=None` (highest silhouette among configurations with at least two clusters)
- Number of clusters: 4
- Cluster sizes: [337, 481, 495, 950]
- Noise fraction: 0.0763
- Silhouette score: 0.5159

### 3. Summary Table

| Method | Selected Hyperparameters | Num Clusters | Cluster Sizes | Noise Fraction | Silhouette Score |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **DBSCAN** | `eps=0.12`, `min_samples=5` | 4 | [340, 472, 491, 953] | 0.0792 | **0.5169** |
| **HDBSCAN** | `min_cluster_size=20` | 4 | [337, 481, 495, 950] | 0.0763 | **0.5159** |

---

## Problem 2 — Hyperparameter Tuning of Random Forest

80/20 stratified split (seed 33); 4-fold stratified CV on the training set; objective: ROC-AUC. The test set was used only once per method for the final evaluation.

### 1. Search Results

- **Grid Search**
  - Best hyperparameters: `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 150}`
  - Best mean CV ROC-AUC: 0.9561
  - Configurations evaluated: 8
  - Test ROC-AUC: 0.9407
  - Test F1: 0.8276

- **Random Search**
  - Best hyperparameters: `{'n_estimators': 200, 'min_samples_split': 2, 'max_features': 'log2', 'max_depth': None}`
  - Best mean CV ROC-AUC: 0.9558
  - Configurations evaluated: 8
  - Test ROC-AUC: 0.9416
  - Test F1: 0.8250

- **Bayesian Optimization (Optuna, TPE)**
  - Best hyperparameters: `{'n_estimators': 99, 'max_depth': 11, 'min_samples_split': 3, 'max_features': 0.5}`
  - Best mean CV ROC-AUC: 0.9540
  - Trials completed: 8
  - Test ROC-AUC: 0.9428
  - Test F1: 0.8160

### 2. Summary Table

| Search Method | Best CV ROC-AUC | Evaluated | Test ROC-AUC | Test F1 | Best Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Grid Search** | 0.9561 | 8 | 0.9407 | 0.8276 | `n_estimators=150, max_depth=None, min_samples_split=2, max_features='sqrt'` |
| **Random Search** | 0.9558 | 8 | 0.9416 | 0.8250 | `n_estimators=200, max_depth=None, min_samples_split=2, max_features='log2'` |
| **Bayesian Opt** | 0.9540 | 8 | 0.9428 | 0.8160 | `n_estimators=99, max_depth=11, min_samples_split=3, max_features=0.5` |

---

## Problem 3 — Unsupervised Anomaly Detection with PyOD

Settings: `contamination=0.10`, `n_neighbors=20`, `flag_fraction=0.10`; Isolation Forest with `n_estimators=200`, `random_state=42`. `record_id` was excluded from the features.

### 1. Flagged Counts and Top 10 Ranked Records

- **KNN**
  - Flagged observations: 300
  - Top 10 `record_id`: `REC-00035, REC-00259, REC-02873, REC-01662, REC-00950, REC-00036, REC-01369, REC-01112, REC-01941, REC-01065`
- **LOF**
  - Flagged observations: 300
  - Top 10 `record_id`: `REC-00672, REC-00325, REC-02855, REC-00275, REC-01666, REC-01794, REC-02524, REC-01175, REC-01229, REC-01537`
- **Isolation Forest**
  - Flagged observations: 300
  - Top 10 `record_id`: `REC-00129, REC-01247, REC-00863, REC-02414, REC-00049, REC-01590, REC-01634, REC-02386, REC-02408, REC-02925`

### 2. Pairwise Overlap Counts (flagged by both)

| Pair | Observations |
| :--- | :---: |
| KNN & LOF | 104 |
| KNN & Isolation Forest | 189 |
| LOF & Isolation Forest | 68 |

---

## Problem 4 — Semi-Supervised Anomaly Detection with Pseudo-Labeling

Thresholds: `low_threshold=0.10`, `high_threshold=0.90`. Evaluation labels were used only for the final evaluation.

### 1. Pseudo-Label Selection
- Pseudo-normal samples added (p ≤ 0.10): **1390**
- Pseudo-anomaly samples added (p ≥ 0.90): **112**
- Total unlabeled samples selected: 1502

### 2. Evaluation Results

| Model | Evaluation ROC-AUC | Evaluation F1 |
| :--- | :---: | :---: |
| **Initial Model (labeled only)** | 0.9796 | 0.8762 |
| **Retrained Model (labeled + pseudo-labeled)** | 0.9765 | **0.8922** |

**Observation:** Pseudo-labeling raised the F1 score (0.8762 → 0.8922) but slightly lowered ROC-AUC (0.9796 → 0.9765), so it did not improve every metric.