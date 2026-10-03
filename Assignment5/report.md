# DA5401 Assignment 5 — Report

**student_seed = 33**

This seed was used for both problems: for the student-specific subset sampling and for the K-Means centroid initialization.

---

## Problem 1 — K-Means from Scratch

| K | Total within-cluster squared distance | Silhouette score |
|---|---|---|
| 2 | 5102.344736423929 | 0.30835531667528904 |
| 3 | 3364.225486927464 | 0.39458187850132825 |
| 4 | 1737.283677021142 | 0.5164268043075194 |
| 5 | 1670.3546764589912 | 0.42368426485344657 |
| 6 | 1621.6749073765332 | 0.4029605211930016 |

Selected K: **4**

Cluster sizes for K = 4: [279, 296, 300, 325]

---

## Problem 2 — Hierarchical Clustering (n_clusters = 4)

| Linkage | Silhouette score | Cluster sizes |
|---|---|---|
| single | 0.4890119950463914 | [1, 135, 140, 324] |
| complete | 0.706349311426041 | [136, 140, 148, 176] |
| average | 0.706349311426041 | [136, 140, 148, 176] |

Selected linkage: **complete**