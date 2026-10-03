"""
DA5401 — Data Analytics Lab
Assignment 5 — Clustering Techniques

Student submission template.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster


# ============================================================
# Shared utility
# ============================================================

def student_subset(df, n_rows, seed):
    """
    Return the student-specific subset using the required seeded sampling rule.
    """
 


# ============================================================
# Problem 1 — K-Means from scratch
# ============================================================

def initialize_centroids(X, k, seed):
    """
    Select k distinct observations from X as initial centroids using
    np.random.default_rng(seed).
    """
 


def assign_clusters(X, centroids):
    """
    Assign each observation to its nearest centroid using squared
    Euclidean distance.

    If distances tie, assign to the smaller cluster index.
    """
 


def update_centroids(X, labels, previous_centroids):
    """
    Recompute each centroid as the mean of its assigned observations.

    If a cluster is empty, keep its previous centroid unchanged.
    """
 


def sum_squared_distances(X, labels, centroids):
    """
    Return the total within-cluster squared distance.
    """
 


def kmeans_from_scratch(X, k, seed, max_iter=100, tol=1e-4):
    """
    Run K-Means from scratch.

    Stop when either:
    - the largest Euclidean centroid movement is less than tol; or
    - max_iter iterations have been completed.

    Return exactly:
        {
            "labels": ...,
            "centroids": ...,
            "sum_squared_distances": ...,
            "n_iter": ...
        }

    The implementation must be data-agnostic and must not use
    sklearn.cluster.KMeans or another built-in K-Means implementation.
    """
 


def evaluate_k_values(X, k_values, seed):
    """
    For each supplied k:
    - run kmeans_from_scratch;
    - compute total within-cluster squared distance;
    - compute silhouette score.

    Return a list of dictionaries in the same order as k_values.

    Each dictionary must contain exactly:
        {
            "k": ...,
            "sum_squared_distances": ...,
            "silhouette": ...
        }
    """
 


# ============================================================
# Problem 2 — Hierarchical clustering using SciPy
# ============================================================

def hierarchical_clustering(X, linkage_method, n_clusters):
    """
    Use SciPy's linkage and fcluster functions.

    Required behaviour:
        linkage(X, method=linkage_method, metric="euclidean")
        fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    Return:
        linkage_matrix, labels
    """
 


def compare_linkages(X, linkage_methods, n_clusters):
    """
    Compare each supplied linkage method using only:
    - silhouette score;
    - cluster sizes.

    Return a list of dictionaries in the same order as linkage_methods.

    Each dictionary must contain exactly:
        {
            "linkage": ...,
            "silhouette": ...,
            "cluster_sizes": ...
        }

    cluster_sizes must be a sorted list of integers in ascending order.
    """
 


def select_best_linkage(results):
    """
    Select the linkage method with the highest silhouette score.

    Tie-break order:
        1. single
        2. complete
        3. average

    Return the selected linkage method as a string.
    """
 