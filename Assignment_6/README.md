# DA5401 — Data Analytics Lab

## Assignment 6 — Density Clustering, Hyperparameter Tuning and Anomaly Detection

### Objective

This assignment covers four topics from Sessions 8 and 9:

1. **Density-based clustering** using DBSCAN and HDBSCAN.
2. **Hyperparameter tuning** of a Random Forest classifier using Grid Search, Random Search and Bayesian Optimization.
3. **Unsupervised anomaly detection** using the PyOD library.
4. **Semi-supervised anomaly detection** using pseudo-labeling.

The required functions must be **data-agnostic**. Public and private tests may use different datasets, feature names, sample sizes, seeds and candidate hyperparameters while preserving the stated function contracts.

---

## Submission Requirements

To ensure automated evaluation pipelines run successfully:
- **`assignment_submission.py`**: You must rename the provided template and submit your executable Python code strictly under this filename.
- **`report.md`**: Submit a markdown file containing the concise reports requested for Problems 1, 2, 3, and 4.
- **Compatibility**: Ensure all code is compatible with **Python 3.12**.

---

## Files Provided

- `assignment6_density_clustering.csv`
- `assignment6_rf_tuning.csv`
- `assignment6_anomaly_unlabeled.csv`
- `assignment6_anomaly_semisupervised.csv`

---

## Student-Specific Seed

Use the last two digits of your roll number:

```python
student_seed = int(last_two_digits_of_roll_number)
```

If the last two digits are `00`, use:

```python
student_seed = 0
```

Use this seed wherever a student-specific split or randomized search is required.

---

# Problem 1 — Density-Based Clustering: DBSCAN and HDBSCAN

## Context

Use: `assignment6_density_clustering.csv`

Standardize all features using `StandardScaler` before applying density-based clustering.

Use the implementations provided by scikit-learn:

```python
from sklearn.cluster import DBSCAN, HDBSCAN
```

---

## 1. DBSCAN

For one DBSCAN configuration, return:

- cluster labels;
- number of clusters, excluding noise label `-1`;
- fraction of observations labelled as noise;
- sorted cluster sizes, excluding noise;
- silhouette score.

Use Euclidean distance. The silhouette score must be computed **only on non-noise observations**. If fewer than two non-noise clusters are present, return `np.nan` for the silhouette score.

Evaluate the following configurations on the provided dataset:

```python
DBSCAN_CONFIGS = [
    {"eps": 0.12, "min_samples": 5},
    {"eps": 0.18, "min_samples": 5},
    {"eps": 0.24, "min_samples": 8},
    {"eps": 0.30, "min_samples": 10},
]
```

Your implementation must work for any supplied list of configurations. Do not hard-code these values.

---

## 2. HDBSCAN

For HDBSCAN, vary `min_cluster_size`.

Use:

```python
HDBSCAN_MIN_CLUSTER_SIZES = [20, 40, 70, 100]
```

Use:

```python
min_samples=None
metric="euclidean"
```

For each value, return the same outputs used for DBSCAN:

- cluster labels;
- number of clusters;
- noise fraction;
- sorted non-noise cluster sizes;
- silhouette score computed on non-noise observations only.

Your implementation must also work for other `min_cluster_size` values used by private tests.

---

## 3. Select Configurations

For both DBSCAN and HDBSCAN:

1. consider only configurations that produce at least two non-noise clusters;
2. select the configuration with the highest silhouette score;
3. if tied within normal floating-point precision, select the configuration with the lower noise fraction;

---

## 4. Report — Problem 1

Report only:

- DBSCAN results for every supplied configuration;
- selected DBSCAN configuration;
- HDBSCAN results for every supplied configuration;
- selected HDBSCAN configuration;
- for the selected solutions: number of clusters, cluster sizes, noise fraction and silhouette score.

---

# Problem 2 — Hyperparameter Tuning of a Random Forest

## Context

Use: `assignment6_rf_tuning.csv`

The target column is: `target`

Use:

```python
RandomForestClassifier
```

---

## 1. Train/Test Split

Create a stratified:
\[
80\% \text{ training},\qquad 20\% \text{ test}
\]
split using `student_seed`. The test set must remain untouched during all hyperparameter tuning.

Use 4-fold stratified cross-validation on the training data:
```python
StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=student_seed
)
```

Use **ROC-AUC** as the tuning objective.

All Random Forest models must use:
```python
random_state=42
class_weight="balanced"
n_jobs=-1
```

unless a search method itself controls parallel execution.

---

## 2. Grid Search

Use `GridSearchCV`.

For the provided dataset, use:

```python
RF_GRID = {
    "n_estimators": [80, 150],
    "max_depth": [None, 8],
    "min_samples_split": [2, 8],
    "max_features": ["sqrt"],
}
```

Return:

- best hyperparameters;
- best mean CV ROC-AUC;
- number of candidate configurations evaluated.

The function must work for an arbitrary supplied parameter grid.

---

## 3. Random Search

Use `RandomizedSearchCV`.

For the provided dataset, use:

```python
RF_RANDOM_SPACE = {
    "n_estimators": [60, 80, 100, 130, 160, 200],
    "max_depth": [None, 5, 8, 12],
    "min_samples_split": [2, 4, 8, 12],
    "max_features": ["sqrt", "log2", 0.6],
}
```

Use:

```python
n_iter = 8
random_state = student_seed
```

Return:

- best hyperparameters;
- best mean CV ROC-AUC;
- number of candidate configurations evaluated.

The function must work for arbitrary supplied search spaces and `n_iter`.

---

## 4. Bayesian Optimization

Use **Optuna**.

Use `TPESampler(seed=student_seed)`.

For the provided dataset, search:

- `n_estimators`: integer from 60 to 220;
- `max_depth`: integer from 3 to 15;
- `min_samples_split`: integer from 2 to 12;
- `max_features`: one of `"sqrt"`, `"log2"`, `0.5`, `0.8`.

For each trial:

1. construct a Random Forest using the suggested hyperparameters;
2. evaluate mean ROC-AUC using the same 4-fold stratified CV;
3. return the mean CV score to Optuna.

Use:

```python
n_trials = 8
```

Return:

- best hyperparameters;
- best mean CV ROC-AUC;
- number of completed trials.

The implementation must work with a different number of trials and different datasets.

---

## 5. Final Evaluation

For each search method, fit the selected Random Forest on the full training set and evaluate it once on the untouched test set.

Report:

- test ROC-AUC;
- test F1 score.

Do not use the test set to select hyperparameters.

---

## 6. Report — Problem 2

Report only:

- best CV ROC-AUC for Grid Search, Random Search and Bayesian Optimization;
- best hyperparameters for each;
- number of configurations/trials evaluated;
- final test ROC-AUC and F1 for each search method.

---

# Problem 3 — Unsupervised Anomaly Detection with PyOD

## Context

Use: `assignment6_anomaly_unlabeled.csv` 

In the provided dataset, record_id is an identifier and must not be used as a feature. Exclude it when preparing the feature matrix. The required anomaly-detection functions themselves must remain data-agnostic and must not assume that an identifier column named record_id exists. 

No anomaly labels are available in this problem.

Standardize all numerical sensor features using `StandardScaler`.

Use the **PyOD** library:

```python
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
```

Do not replace these models with scikit-learn equivalents.

---

## 1. Models

Use:

```python
contamination = 0.10
n_neighbors = 20
```

Construct:

- PyOD `KNN`;
- PyOD `LOF`;
- PyOD `IForest`.

For Isolation Forest use:

```python
n_estimators=200
random_state=42
```

The required functions must also work when private tests supply different `contamination` and `n_neighbors` values.

---

## 2. Anomaly Scores

Fit each model and obtain one anomaly score per observation.

Use PyOD's score convention:

> **larger score = more anomalous**

Return the score array for every model.

---

## 3. Convert Scores to Flags

Given an anomaly-score array and a supplied fraction \(f\), flag exactly the observations with the largest scores.

Use:

```python
n_flagged = ceil(f * n_samples)
```

If scores are tied at the boundary, resolve ties by the original row order, with the smaller row index ranked first.

For the provided dataset, use:

```python
flag_fraction = 0.10
```

---

## 4. Compare the Three Methods

For KNN, LOF and Isolation Forest report:

- number of flagged observations;
- top 10 `record_id` values ranked by anomaly score.

Also report the pairwise number of observations flagged by both methods:

- KNN and LOF;
- KNN and Isolation Forest;
- LOF and Isolation Forest.

Do not use any hidden labels to select or tune the anomaly detectors.

---

## 5. Report — Problem 3

Report only the requested rankings, flag counts and overlap counts.

---

# Problem 4 — Semi-Supervised Anomaly Detection with Pseudo-Labeling

## Context

Use: `assignment6_anomaly_semisupervised.csv`

This file contains the **same underlying observations** as Problem 3, but now a small amount of verified label information is available.

Columns:

- `record_id`: identifier;
- sensor columns: predictors;
- `set_type`: one of `labeled`, `unlabeled`, or `evaluation`;
- `label`: available only for `labeled` and `evaluation` rows.

Do not use evaluation labels during training or pseudo-label creation.

---

## 1. Prepare the Labeled, Unlabeled and Evaluation Sets

In the provided dataset, the column `set_type` indicates which rows belong to:

- `labeled`
- `unlabeled`
- `evaluation`

Use `set_type` only to separate the three subsets. The column `label` contains the verified class label for rows in the `labeled` and `evaluation` subsets. It is missing for rows in the `unlabeled` subset.

The column `record_id` is an identifier and must not be used as a predictive feature.

For the provided dataset:

- rows with `set_type == "labeled"` form the initial labeled training set;
- rows with `set_type == "unlabeled"` form the unlabeled pool used for pseudo-labeling;
- rows with `set_type == "evaluation"` form the final evaluation set.

Create:

- `X_labeled`
- `y_labeled`
- `X_unlabeled`
- `X_evaluation`
- `y_evaluation`

Use only the supplied numerical feature columns when constructing the feature matrices. The required preparation function must be data-agnostic. It must not assume that the identifier column is named `record_id`, or that the feature columns have any particular names. The feature columns will be supplied explicitly to the function.

Private tests may use different numerical feature names and may omit an identifier column entirely.

---

## 2. Initial Model

Train:

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
```

using only the labeled subset.

Use predicted probability of class 1 as the anomaly probability.

---

## 3. Select High-Confidence Pseudo-Labels

For the provided dataset, use:

```python
low_threshold = 0.10
high_threshold = 0.90
```

For an unlabeled observation:

- probability `<= low_threshold` → pseudo-label `0`;
- probability `>= high_threshold` → pseudo-label `1`;
- otherwise, do not pseudo-label it.

A pseudo-label must depend only on the model prediction, never on a hidden true label.

Return:

- selected unlabeled row indices;
- pseudo-labels;
- number of pseudo-normal samples;
- number of pseudo-anomaly samples.

The implementation must work with different thresholds.

---

## 4. Retrain

Train a **new** Random Forest with the same model settings using:

- all originally labeled observations;
- selected pseudo-labeled observations.

---

## 5. Evaluate Initial and Retrained Models

Evaluate both models on the evaluation set using:

- ROC-AUC;
- F1 score with probability threshold 0.5.

Report:

- initial ROC-AUC and F1;
- number of pseudo-normal and pseudo-anomaly samples added;
- retrained ROC-AUC and F1.

Pseudo-labeling is not guaranteed to improve every metric. Report the observed result.

---