# DA5401 — Data Analytics Lab

## Assignment 5 — Clustering Techniques

### Objective

This assignment has two problems:

1. **K-Means clustering from scratch** — implement the main K-Means steps yourself and use them to compare different values of \(K\).
2. **Agglomerative hierarchical clustering** — use SciPy's built-in hierarchical clustering functions and compare different linkage criteria.

---

## Files Provided

- `assignment5_kmeans_operating_regimes.csv`
- `assignment5_hierarchical_linkages.csv`

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

For both problems, create your student-specific subset using:

```python
rng = np.random.default_rng(student_seed)
indices = rng.choice(len(df), size=n_rows, replace=False)
student_df = df.iloc[indices].reset_index(drop=True)
```

Use the same `student_seed` wherever a seed is required.

---

# Problem 1 — K-Means Clustering from Scratch

## Context

The dataset contains numerical operating measurements collected from an industrial process. The aim is to identify groups of operating conditions that behave similarly.

Use:

`assignment5_kmeans_operating_regimes.csv`

---

## 1. Create Your Student-Specific Dataset

Select exactly **1200 rows** using the student-specific sampling procedure above.

Standardize all features using `StandardScaler`.

All K-Means calculations must be performed on the standardized data.

---

## 2. Implement K-Means from Scratch

For the K-Means algorithm itself, use only:

- NumPy;
- pandas;
- Python standard library.

Do **not** use `sklearn.cluster.KMeans` or another built-in K-Means implementation.

You may use `StandardScaler` for scaling and `silhouette_score` for evaluation.

### 2.1 Initialize Centroids

For a given number of clusters \(K\), choose \(K\) distinct observations as the initial centroids using:

```python
rng = np.random.default_rng(seed)
indices = rng.choice(len(X), size=K, replace=False)
```

The selected observations themselves are the initial centroids.

### 2.2 Assignment Step

Assign every observation to its nearest centroid using squared Euclidean distance. If an observation is equally distant from two centroids, assign it to the cluster with the smaller cluster index.

### 2.3 Update Step

For each cluster, recompute the centroid. If a cluster is empty after the assignment step, keep its previous centroid unchanged.

### 2.4 Repeat Until Convergence

Repeat the assignment and update steps until either:

- the largest Euclidean movement of any centroid is less than `tol`; or
- `max_iter` iterations have been completed.

The final result must contain:

- cluster labels;
- final centroids;
- total within-cluster squared distance;
- number of iterations performed.

The total within-cluster squared distance is:

\[
\sum_{k=1}^{K}
\sum_{x_i\in C_k}
\|x_i-\mu_k\|^2.
\]

---

## 3. Compare Different Values of \(K\)

Evaluate:

\[
K \in \{2,3,4,5,6\}.
\]

For each value of \(K\):

1. run your from-scratch K-Means using `student_seed`;
2. calculate the total within-cluster squared distance;
3. calculate the silhouette score using `sklearn.metrics.silhouette_score`.

Select the \(K\) with the **highest silhouette score**.

If two values are tied within normal floating-point precision, choose the smaller \(K\).

---

> **Generality requirement:** Your K-Means implementation must be data-agnostic. 
It should work on any similar numerical dataset with an arbitrary number of observations and numerical features, provided \(K\) is valid. 
Do not hard-code feature names, the number of features, dataset size, cluster assignments, or results specific to the provided dataset.

## 4. Report — Problem 1

Report only:

- `student_seed`;
- total within-cluster squared distance for each \(K\);
- silhouette score for each \(K\);
- selected \(K\);
- cluster sizes for the selected \(K\).

No additional interpretation is required.

---

# Problem 2 — Hierarchical Clustering with Different Linkages

## Context

The second dataset contains numerical observations whose grouping structure can change depending on the linkage criterion.

Use:

`assignment5_hierarchical_linkages.csv`

---

## 1. Create Your Student-Specific Dataset

Select exactly **600 rows** using the same student-specific sampling procedure.

Standardize both features using `StandardScaler`.

---

## 2. Required Library and Built-In Functions

For this problem, use **SciPy's hierarchical clustering implementation**.

For each linkage method, construct the hierarchy using:

```python
linkage(X_scaled, method=linkage_method, metric="euclidean")
```

and obtain cluster assignments using:

```python
fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
```

The cluster labels returned by `fcluster` may begin at 1; this is acceptable.

---

## 3. Compare Linkage Methods

Use exactly:

```python
LINKAGE_METHODS = ["single", "complete", "average"]
```

and:

```python
n_clusters = 4
```

For each linkage method:

1. generate the linkage matrix using SciPy;
2. obtain the 4-cluster partition using `fcluster`;
3. calculate the silhouette score using Euclidean distance;
4. calculate the size of each cluster.

For comparison, use **only**:

- silhouette score;
- cluster sizes.

Do not calculate or report any other clustering metric.

---

## 4. Select a Linkage Method

Select the linkage method with the highest silhouette score.

If two methods are tied within normal floating-point precision, use this tie-break order:

1. `single`
2. `complete`
3. `average`

---

## 5. Report — Problem 2

Report only:

- `student_seed`;
- silhouette score for each linkage method;
- cluster sizes for each linkage method;
- selected linkage method.

No additional interpretation or additional metrics are required.

---

# Submission

Submit:

- `assignment_submission.py`
- `report.md`

Your Python file must be importable without automatically executing the complete analysis.

You may place the full workflow inside:

```python
if __name__ == "__main__":
    ...
```

But YOU MUST SPECIFY THE SEEDS YOU HAVE USED TO GENERATE THE RESULTS IN THE report.md

---

# Important Rules

- Do not use a built-in K-Means implementation in Problem 1.
- Use SciPy's `linkage` and `fcluster` in Problem 2.
- Use standardized data for both problems.
- Do not hard-code candidate \(K\) values, linkage values, cluster labels, scores, or expected outputs inside the required functions.
- Public and private tests may use different numerical datasets, different seeds, different candidate \(K\) values, and different linkage lists while preserving the stated function contracts.
