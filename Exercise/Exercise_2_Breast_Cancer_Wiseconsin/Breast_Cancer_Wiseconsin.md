# Exercise 2: Does Scaling Really Matter?

## Objective

Investigate the impact of feature scaling on the performance of the K-Nearest Neighbors (KNN) classifier using the Breast Cancer Wisconsin dataset available in scikit-learn.

---

## Dataset

The Breast Cancer Wisconsin dataset is a binary classification dataset used to predict whether a tumor is malignant or benign based on various numerical measurements extracted from digitized images of breast masses.

---

## Tasks

1. Load the Breast Cancer Wisconsin dataset.

2. Inspect the ranges of different numerical features:
   - Examine minimum and maximum values.
   - Compare feature scales.
   - Identify features with significantly different magnitudes.

3. Split the dataset into:
   - Training set
   - Validation set
   - Test set

4. Train a K-Nearest Neighbors (KNN) classifier using the original (unscaled) features.

5. Repeat the experiment after applying:
   - Standardization (`StandardScaler`)
   - Min-Max Scaling (`MinMaxScaler`)

6. Compare the validation performance of all three approaches:
   - Original Features
   - Standardized Features
   - Min-Max Scaled Features

---

## Questions to Answer

### 1. How different are the original feature scales?

Analyze the range of values for each feature and identify whether some features dominate others due to their larger numerical scales.

### 2. Does scaling change KNN performance?

Compare the validation accuracy of KNN before and after scaling.

### 3. Why does scaling affect KNN?

Explain how KNN relies on distance calculations and why features with larger magnitudes can disproportionately influence the distance metric.

### 4. Do Standardization and Min-Max Scaling produce identical results?

Compare the performance of the two scaling techniques and discuss any observed differences.

---

## Expected Learning Outcomes

By completing this exercise, you will learn how to:

- Analyze feature distributions and scales.
- Understand the importance of feature scaling.
- Apply StandardScaler and MinMaxScaler.
- Evaluate the impact of preprocessing on machine learning models.
- Understand why distance-based algorithms such as KNN are sensitive to feature scales.
- Compare different normalization and standardization techniques.

---

## Algorithms and Techniques Used

- K-Nearest Neighbors (KNN)
- StandardScaler
- MinMaxScaler
- Train / Validation / Test Split
- Classification Performance Evaluation

