"""
Public tests for DA5401 Assignment 5 — Clustering Techniques

Place this file in the same directory as:
    assignment_submission.py
    assignment5_kmeans_operating_regimes.csv
    assignment5_hierarchical_linkages.csv

Run:
    pytest -q public_tests.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

from assignment_submission import (
    student_subset,
    initialize_centroids,
    assign_clusters,
    update_centroids,
    sum_squared_distances,
    kmeans_from_scratch,
    hierarchical_clustering,
    compare_linkages,
    select_best_linkage,
)

DATA_DIR = Path(__file__).resolve().parent


# ============================================================
# Problem 1 — K-Means from scratch
# ============================================================

def test_student_subset_is_reproducible_and_seed_dependent():
    df = pd.DataFrame({
        "x": np.arange(30),
        "y": np.arange(100, 130),
    })

    a = student_subset(df, n_rows=10, seed=7)
    b = student_subset(df, n_rows=10, seed=7)
    c = student_subset(df, n_rows=10, seed=8)

    expected_idx = np.random.default_rng(7).choice(
        len(df), size=10, replace=False
    )
    expected = df.iloc[expected_idx].reset_index(drop=True)

    pd.testing.assert_frame_equal(a, expected)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_initialize_centroids_matches_required_random_choice():
    X = np.arange(40, dtype=float).reshape(10, 4)

    result = initialize_centroids(X, k=3, seed=11)

    idx = np.random.default_rng(11).choice(
        len(X), size=3, replace=False
    )
    expected = X[idx]

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 4)
    assert np.allclose(result, expected)


def test_assign_clusters_uses_nearest_centroid_and_tie_breaks_low():
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.0],
        [9.0, 0.0],
    ])

    centroids = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [10.0, 0.0],
    ])

    labels = assign_clusters(X, centroids)

    assert isinstance(labels, np.ndarray)
    assert np.array_equal(labels, np.array([0, 0, 1, 2]))


def test_update_centroids_keeps_previous_for_empty_cluster():
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [8.0, 2.0],
        [10.0, 2.0],
    ])

    labels = np.array([0, 0, 2, 2])

    previous = np.array([
        [1.0, 0.0],
        [5.0, 5.0],
        [9.0, 2.0],
    ])

    result = update_centroids(X, labels, previous)

    expected = np.array([
        [1.0, 0.0],
        [5.0, 5.0],
        [9.0, 2.0],
    ])

    assert result.shape == previous.shape
    assert np.allclose(result, expected)


def test_sum_squared_distances_known_example():
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [8.0, 0.0],
        [10.0, 0.0],
    ])

    labels = np.array([0, 0, 1, 1])

    centroids = np.array([
        [1.0, 0.0],
        [9.0, 0.0],
    ])

    result = sum_squared_distances(X, labels, centroids)

    assert result == pytest.approx(4.0)


def test_kmeans_from_scratch_on_simple_two_cluster_data():
    X = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [-0.1, 0.2],
        [10.0, 10.0],
        [10.2, 9.9],
        [9.8, 10.1],
    ])

    result = kmeans_from_scratch(
        X, k=2, seed=3, max_iter=100, tol=1e-8
    )

    assert set(result.keys()) == {
        "labels",
        "centroids",
        "sum_squared_distances",
        "n_iter",
    }
    assert result["labels"].shape == (6,)
    assert result["centroids"].shape == (2, 2)
    assert 1 <= result["n_iter"] <= 100

    expected_ssd = sum(
        np.sum(
            (X[i] - result["centroids"][result["labels"][i]]) ** 2
        )
        for i in range(len(X))
    )

    assert result["sum_squared_distances"] == pytest.approx(expected_ssd)
    assert len(set(result["labels"][:3])) == 1
    assert len(set(result["labels"][3:])) == 1
    assert result["labels"][0] != result["labels"][3]


# ============================================================
# Problem 2 — Hierarchical clustering
# ============================================================

def test_hierarchical_clustering_respects_requested_cluster_count():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [5.0, 0.0],
        [5.1, 0.1],
        [0.0, 5.0],
        [0.2, 5.1],
        [5.0, 5.0],
        [5.2, 5.1],
    ])

    for n_clusters in (3, 4):
        Z, labels = hierarchical_clustering(
            X,
            linkage_method="average",
            n_clusters=n_clusters,
        )

        assert isinstance(Z, np.ndarray)
        assert Z.shape == (len(X) - 1, 4)
        assert labels.shape == (len(X),)
        assert len(np.unique(labels)) == n_clusters


def test_hierarchical_clustering_matches_scipy():
    X = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [3.0, 3.0],
        [3.2, 3.1],
        [8.0, 0.0],
        [8.2, 0.1],
    ])

    Z, labels = hierarchical_clustering(
        X,
        linkage_method="complete",
        n_clusters=3,
    )

    expected_Z = linkage(
        X,
        method="complete",
        metric="euclidean",
    )
    expected_labels = fcluster(
        expected_Z,
        t=3,
        criterion="maxclust",
    )

    assert isinstance(Z, np.ndarray)
    assert Z.shape == (len(X) - 1, 4)
    assert np.allclose(Z, expected_Z)
    assert np.array_equal(labels, expected_labels)


def test_compare_linkages_cluster_sizes_are_correct_and_sorted():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],

        [4.0, 0.0],
        [4.1, 0.0],
        [4.2, 0.1],

        [0.0, 5.0],
        [0.1, 5.0],
        [0.2, 5.1],
        [0.3, 4.9],
        [0.4, 5.0],
    ])

    results = compare_linkages(
        X,
        linkage_methods=["complete"],
        n_clusters=3,
    )

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]

    assert set(result.keys()) == {
        "linkage",
        "silhouette",
        "cluster_sizes",
    }
    assert result["linkage"] == "complete"

    Z = linkage(
        X,
        method="complete",
        metric="euclidean",
    )
    labels = fcluster(
        Z,
        t=3,
        criterion="maxclust",
    )

    expected_sizes = sorted(
        int(v)
        for v in np.unique(labels, return_counts=True)[1]
    )

    assert result["cluster_sizes"] == expected_sizes
    assert result["cluster_sizes"] == sorted(result["cluster_sizes"])
    assert sum(result["cluster_sizes"]) == len(X)

    assert result["silhouette"] == pytest.approx(
        silhouette_score(X, labels, metric="euclidean")
    )


def test_problem2_workflow_on_assignment_dataset():
    df = pd.read_csv(
        DATA_DIR / "assignment5_hierarchical_linkages.csv"
    )

    sub = student_subset(
        df,
        n_rows=600,
        seed=17,
    )

    X = StandardScaler().fit_transform(sub)

    methods = [
        "single",
        "complete",
        "average",
    ]

    results = compare_linkages(
        X,
        linkage_methods=methods,
        n_clusters=4,
    )

    assert isinstance(results, list)
    assert len(results) == 3
    assert [r["linkage"] for r in results] == methods

    for r in results:
        assert set(r.keys()) == {
            "linkage",
            "silhouette",
            "cluster_sizes",
        }

        expected_Z = linkage(
            X,
            method=r["linkage"],
            metric="euclidean",
        )
        expected_labels = fcluster(
            expected_Z,
            t=4,
            criterion="maxclust",
        )

        expected_silhouette = silhouette_score(
            X,
            expected_labels,
            metric="euclidean",
        )

        expected_sizes = sorted(
            int(v)
            for v in np.unique(
                expected_labels,
                return_counts=True,
            )[1]
        )

        assert r["silhouette"] == pytest.approx(expected_silhouette)
        assert r["cluster_sizes"] == expected_sizes
        assert sum(r["cluster_sizes"]) == 600

    selected = select_best_linkage(results)

    tie_order = {
        "single": 0,
        "complete": 1,
        "average": 2,
    }

    expected_selected = sorted(
        results,
        key=lambda r: (
            -r["silhouette"],
            tie_order[r["linkage"]],
        ),
    )[0]["linkage"]

    assert selected == expected_selected
