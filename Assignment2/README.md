# DA5401 — Data Analytics Lab

## Assignment 2 — Reliable Model Evaluation with Pipelines

### Objective

The purpose of this assignment is to build **reliable end-to-end machine-learning workflows** in scikit-learn and evaluate them correctly on unseen data.

You are expected to apply the data inspection, preprocessing, pipeline construction, cross-validation, model evaluation, and classification-threshold concepts discussed in **Sessions 2 and 3**.

Preprocessing must be part of the modelling workflow. Your implementation should work correctly not only on the supplied datasets, but also on unseen data with similar columns and potentially different missing-value patterns, category frequencies, and feature ranges.

---

## Files Provided

You are given two datasets:

```text
regression_data.csv
classification_data.csv
```
---

## General Requirements

1. Use **Python, pandas, NumPy and scikit-learn**.
2. Use `random_state=42` wherever a random split or randomized operation is required.
3. Perform preprocessing using scikit-learn `Pipeline` and `ColumnTransformer`.
4. Inspect the supplied data before deciding how individual columns should be treated.
5. Your preprocessing should appropriately handle the issues discussed in **Session 2**, including:
   - missing values;
   - categorical variables;
   - numerical variables with substantially different scales; and
   - columns that should not be used as predictive features.
6. The exact preprocessing choices are yours, but they should be appropriate for the data.
7. Preprocessing parameters must be learned from the relevant training data only.
8. Do not preprocess the complete dataset before splitting or cross-validation.
9. The returned pipeline must accept the raw feature DataFrame directly.
10. Do not hard-code category values, imputation values, transformed feature counts, thresholds, predictions, or metric values.
11. Unless explicitly stated otherwise, use scikit-learn implementations for metrics.

### Important Note About the Pipeline Functions

The functions `build_regression_pipeline()` and `build_classification_pipeline()` only need to **construct and return an unfitted pipeline**.

During testing, the public and private tests will call standard scikit-learn methods on the returned pipeline. For example:

```python
pipeline = build_regression_pipeline(X_train)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

and for classification:

```python
pipeline = build_classification_pipeline(X_train)
pipeline.fit(X_train, y_train)
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

Therefore, the pipeline returned by your functions must work correctly with `.fit()`, `.predict()`, and, for classification, `.predict_proba()`.

---

# Problem 1 — Regression Evaluation

## Business Context

A service company wants to predict the **cost of completing a maintenance job** before the job begins. The estimate is used for budgeting and resource planning.

Use:

```text
regression_data.csv
```

The target is:

```text
job_cost
```

The dataset contains a mixture of numerical and categorical information collected when a job is registered. Some values are missing, features are measured on different scales, and at least one column behaves like an identifier rather than a useful predictive variable.

Different error measures emphasize different aspects of model performance. Your task is therefore not only to evaluate the model, but also to decide which evaluation metric is most meaningful for this business application.

## Task

### 1. Inspect and prepare the data

Separate the target from the predictors and inspect the input features.

Decide:

* which columns are numerical;
* which columns are categorical;
* which columns should not be used for prediction; and
* what preprocessing is appropriate for the different feature types.

Do not include a column merely because it is present in the dataset. Use only information that would reasonably be useful and available when the prediction is made.

### 2. Split the data

Create an **80% training / 20% test split** using:

```python
random_state=42
```

The test set must remain untouched until the final evaluation.

### 3. Build the preprocessing and regression pipeline

Create a scikit-learn `Pipeline` containing a `ColumnTransformer` and ending in:

```python
sklearn.linear_model.LinearRegression
```

Use appropriate preprocessing based on the principles discussed in Session 2.

The pipeline must:

* accept the raw feature DataFrame directly;
* learn all preprocessing parameters only from the data on which `.fit()` is called; and
* avoid passing missing values to the regression model.

### 4. Perform 5-fold cross-validation on the training data

Before fitting the final model, evaluate the **complete pipeline** using:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Perform cross-validation using the **training set only**.

Use **MAE** as the cross-validation metric and report:

* mean cross-validation MAE;
* standard deviation of cross-validation MAE.

The purpose of this step is to assess how consistent the pipeline's performance is across different training-validation folds.

The complete preprocessing + modelling pipeline must be evaluated inside each fold so that preprocessing is fitted separately within each training fold.

You may use scikit-learn's `cross_val_score()` or an equivalent scikit-learn cross-validation function.

### 5. Fit the final regression model and predict

After cross-validation:

1. fit the complete pipeline on the full training set using `.fit(X_train, y_train)`;
2. generate predictions for the untouched test set using `.predict(X_test)`.

The testing framework will also perform these operations directly on the pipeline returned by your function.

### 6. Evaluate on the test set

Calculate:

* Mean Absolute Error (`MAE`)
* Root Mean Squared Error (`RMSE`)
* Mean Absolute Percentage Error (`MAPE`)
* Coefficient of Determination (`R²`)

For `MAPE`, return the value produced by scikit-learn's `mean_absolute_percentage_error` directly. Do not multiply it by 100 inside the metric function.

### 7. Select the primary business metric

Based on the stated business context and the final test results, choose **one primary evaluation metric** from:

* MAE
* RMSE
* MAPE
* R²

Briefly justify why that metric is the most appropriate one for this application.

The metric used for cross-validation in Step 4 is specified only to provide a consistent way of assessing fold-to-fold performance. It does **not** imply that MAE must be chosen as the primary business metric.

---

## Required Functions — Problem 1

### `build_regression_pipeline`

```python
def build_regression_pipeline(X: pd.DataFrame):
    """
    Return an unfitted sklearn Pipeline that:
    1. preprocesses the raw columns appropriately using a ColumnTransformer, and
    2. ends in LinearRegression.
    """
```

### `regression_metrics`

```python
def regression_metrics(y_true, y_pred):
    """
    Return a dictionary with exactly these keys:
    'mae', 'rmse', 'mape', 'r2'
    """
```

The returned values must be computed using the corresponding scikit-learn metrics.

---

# Problem 2 — Classification Evaluation

## Business Context

A maintenance team wants to predict whether a machine is likely to experience a **critical failure within the next 30 days**.

Use:

```text
classification_data.csv
```

The target is:

```text
critical_failure_30d
```

The positive class (`1`) represents a machine that will experience a critical failure. Failures are relatively rare, so the dataset is **imbalanced**.

Missing a machine that is genuinely at risk can be costly. The maintenance team therefore requires:

> **Recall for the positive class must be at least 0.85. Among candidate thresholds satisfying this requirement, choose the threshold with the highest precision.**

## Task

### 1. Inspect and prepare the data

Separate the target from the predictors and inspect the feature types.

Construct suitable preprocessing using the principles discussed in Session 2.

As in Problem 1:

- preprocessing must be part of the scikit-learn pipeline;
- preprocessing parameters must be learned only from the relevant training data; and
- columns that should not be used for prediction should be excluded appropriately.

### 2. Create train, validation and test sets

Create a **70% / 15% / 15% train-validation-test split**.

Use:

```python
random_state=42
```

and preserve the class proportions through **stratified splitting**.

The three subsets have different roles:

- **Training set:** fit preprocessing and model parameters.
- **Validation set:** choose the classification threshold.
- **Test set:** perform the final evaluation after the threshold has been fixed.

The final test set must not be used for threshold selection.

### 3. Build the classification pipeline

Construct a scikit-learn `Pipeline` containing a `ColumnTransformer` and ending in:

```python
sklearn.linear_model.LogisticRegression
```

Use:

```python
max_iter=1000
```

The pipeline must accept raw input features directly.

### 4. Fit on the training set and obtain validation probabilities

Fit the pipeline on the training set using:

```python
pipeline.fit(X_train, y_train)
```

Obtain the estimated probability of the positive class on the validation set using:

```python
predict_proba()
```

Do not use `predict()` to select the operating threshold.

### 5. Select the operating threshold using the validation set

Implement a function that evaluates the supplied candidate thresholds.

For a candidate threshold `t`, define the predicted class as:

```python
y_pred = (y_prob >= t).astype(int)
```

A threshold is feasible when:

```text
recall >= 0.85
```

Among feasible thresholds:

1. choose the threshold with the **highest precision**;
2. if more than one threshold has the same precision, choose the **highest threshold** among them.

You may assume that the supplied candidate threshold list contains at least one feasible threshold.

Threshold selection must use **validation-set probabilities and labels only**.

### 6. Refit the final model

After the threshold has been selected and fixed:

1. combine the training and validation data;
2. construct or reuse the same complete pipeline;
3. fit the pipeline using the combined training + validation data.

Do **not** change the selected threshold after this point.

### 7. Perform the final test evaluation

Use the refitted pipeline to obtain positive-class probabilities for the untouched test set:

```python
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

Apply the previously selected threshold:

```python
y_pred = (y_prob >= selected_threshold).astype(int)
```

Report:

- `TN`
- `FP`
- `FN`
- `TP`
- Accuracy
- Precision
- Recall
- Specificity
- F1 score

The recall constraint of `0.85` applies to **threshold selection on the validation set**. The recall observed on the final test set may differ because the test set contains unseen observations.

### 8. Interpret the result

Briefly justify:

- why accuracy alone is insufficient for this imbalanced problem;
- why the selected threshold is appropriate for the stated recall requirement; and
- whether the final test results suggest that the selected operating point generalises reasonably.

---

## Required Functions — Problem 2

### `build_classification_pipeline`

```python
def build_classification_pipeline(X: pd.DataFrame):
    """
    Return an unfitted sklearn Pipeline that:
    1. preprocesses the raw columns appropriately using a ColumnTransformer, and
    2. ends in LogisticRegression(max_iter=1000).
    """
```

### `classification_metrics`

```python
def classification_metrics(y_true, y_pred, y_prob):
    """
    Return a dictionary with exactly these keys:
    'tn', 'fp', 'fn', 'tp',
    'accuracy', 'precision', 'recall',
    'specificity', 'f1', 'roc_auc'
    """
```

Use the positive class (`1`) when calculating precision, recall and F1.

Specificity must be calculated as:

```text
TN / (TN + FP)
```

### `select_threshold`

```python
def select_threshold(
    y_true,
    y_prob,
    candidate_thresholds,
    min_recall=0.85
):
    """
    Return a dictionary with exactly these keys:
    'threshold', 'precision', 'recall', 'f1'

    Select the feasible threshold with the highest precision.
    If precision is tied, select the highest threshold.
    """
```

Do not hard-code a threshold inside this function.

---

# Submission Structure

Submit:

```text
assignment_submission.py
report.md
```

## `assignment_submission.py`

This file must contain at least:

```text
build_regression_pipeline
regression_metrics
build_classification_pipeline
classification_metrics
select_threshold
```

You may define additional helper functions.

The file must be importable without automatically training models or printing large outputs.

If you include executable analysis code, place it inside:

```python
if __name__ == "__main__":
    ...
```

---

## `report.md`

**Keep the report as concise as possible.**

Do not describe your code, preprocessing pipeline, implementation steps, or exploratory analysis.

Include **only the requested numerical results and brief justification**.

### Problem 1

Report only:

- 5-fold cross-validation RMSE mean;
- 5-fold cross-validation RMSE standard deviation;
- test MAE;
- test RMSE;
- test MAPE;
- test R²;
- chosen primary business metric;
- brief justification for the chosen metric.

### Problem 2

Report only:

- selected validation threshold;
- validation precision at the selected threshold;
- validation recall at the selected threshold;
- final test `TN`, `FP`, `FN`, `TP`;
- final test accuracy;
- final test precision;
- final test recall;
- final test specificity;
- final test F1;
- very brief justification of the selected threshold and whether the final result appears to generalise reasonably.

---

# Important Rules

- Do not manually fill missing values before fitting the pipeline.
- Do not fit encoders, imputers, or scalers on the complete dataset.
- Do not assume every integer-valued column is a continuous numerical feature.
- Do not assume every column in the dataset should be used for prediction.
- Do not hard-code category levels based only on the supplied dataset.
