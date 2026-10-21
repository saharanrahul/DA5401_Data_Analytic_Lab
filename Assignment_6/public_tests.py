"""
Public tests for DA5401 Assignment 6.

Place this file in the same directory as:
    assignment_submission.py
    assignment6_density_clustering.csv
    assignment6_rf_tuning.csv
    assignment6_anomaly_unlabeled.csv
    assignment6_anomaly_semisupervised.csv

Run:
    pytest -q public_tests.py

PyOD and Optuna must be installed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest

from assignment_submission import (
    cluster_summary,
    run_dbscan,
    evaluate_dbscan_configs,
    run_hdbscan,
    evaluate_hdbscan_configs,
    select_density_result,
    build_random_forest,
    run_grid_search,
    run_random_search,
    run_bayesian_search,
    classification_scores,
    build_pyod_models,
    fit_pyod_scores,
    flag_top_fraction,
    compare_pyod_models,
    prepare_semisupervised_sets,
    build_semisupervised_rf,
    select_pseudo_labels,
    retrain_with_pseudo_labels,
    evaluate_semisupervised_models,
)

DATA_DIR = Path(__file__).resolve().parent


# ============================================================
# Problem 1 — Density clustering (Tests 1–5)
# ============================================================

def test_01_cluster_summary_excludes_noise_and_sorts_sizes():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [3.0, 3.0],
        [3.1, 3.0],
        [3.2, 3.1],
        [9.0, 9.0],
    ])
    labels = np.array([0, 0, 1, 1, 1, -1])

    result = cluster_summary(X, labels)

    assert set(result.keys()) == {
        "n_clusters",
        "noise_fraction",
        "cluster_sizes",
        "silhouette",
    }
    assert result["n_clusters"] == 2
    assert result["noise_fraction"] == pytest.approx(1 / 6)
    assert result["cluster_sizes"] == [2, 3]

    mask = labels != -1
    expected = silhouette_score(X[mask], labels[mask])
    assert result["silhouette"] == pytest.approx(expected)


def test_02_cluster_summary_returns_nan_when_fewer_than_two_clusters():
    X = np.array([[0.0], [0.1], [0.2], [9.0]])
    labels = np.array([0, 0, 0, -1])

    result = cluster_summary(X, labels)

    assert result["n_clusters"] == 1
    assert np.isnan(result["silhouette"])


def test_03_run_dbscan_matches_sklearn():
    X = np.array([
        [0.0, 0.0],
        [0.05, 0.02],
        [0.10, -0.03],
        [2.0, 2.0],
        [2.05, 2.03],
        [2.10, 1.98],
        [8.0, 8.0],
    ])

    result = run_dbscan(X, eps=0.18, min_samples=2)

    expected_labels = DBSCAN(
        eps=0.18,
        min_samples=2,
        metric="euclidean",
    ).fit_predict(X)

    assert result["eps"] == pytest.approx(0.18)
    assert result["min_samples"] == 2
    assert np.array_equal(result["labels"], expected_labels)

    expected_summary = cluster_summary(X, expected_labels)
    for key in ("n_clusters", "noise_fraction", "cluster_sizes"):
        assert result[key] == expected_summary[key]
    assert result["silhouette"] == pytest.approx(
        expected_summary["silhouette"]
    )


def test_04_evaluate_density_configs_and_hdbscan_match_sklearn():
    X = np.array([
        [0.0, 0.0],
        [0.05, 0.00],
        [0.10, 0.05],
        [3.0, 3.0],
        [3.05, 3.05],
        [3.10, 2.95],
        [6.0, 0.0],
        [6.05, 0.05],
        [6.10, -0.05],
        [9.0, 9.0],
    ])

    configs = [
        {"eps": 0.15, "min_samples": 2},
        {"eps": 0.25, "min_samples": 2},
    ]
    db_results = evaluate_dbscan_configs(X, configs)

    assert len(db_results) == 2
    assert [r["eps"] for r in db_results] == [0.15, 0.25]

    h_results = evaluate_hdbscan_configs(
        X,
        min_cluster_sizes=[2, 3],
    )
    assert len(h_results) == 2
    assert [r["min_cluster_size"] for r in h_results] == [2, 3]

    direct = run_hdbscan(X, min_cluster_size=2, min_samples=None)
    expected_labels = HDBSCAN(
        min_cluster_size=2,
        min_samples=None,
        metric="euclidean",
    ).fit_predict(X)
    assert np.array_equal(direct["labels"], expected_labels)


def test_05_select_density_result_uses_silhouette_then_noise_then_order():
    results = [
    {
        "eps": 0.2,
        "n_clusters": 2,
        "noise_fraction": 0.15,
        "silhouette": 0.52,
    },
    {
        "eps": 0.3,
        "n_clusters": 3,
        "noise_fraction": 0.18,
        "silhouette": 0.68,
    },
    {
        "eps": 0.4,
        "n_clusters": 2,
        "noise_fraction": 0.05,
        "silhouette": 0.57,
    },
    ]

    selected = select_density_result(results)

    assert selected is results[1]


# ============================================================
# Problem 2 — Random Forest tuning (Tests 6–10)
# ============================================================

def _small_classification_data():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(180, 5))
    linear = 1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.6 * X[:, 2]
    p = 1 / (1 + np.exp(-linear))
    y = (rng.random(180) < p).astype(int)
    return X, y


def test_06_build_random_forest_required_defaults_and_supplied_params():
    model = build_random_forest(
        n_estimators=37,
        max_depth=5,
        min_samples_split=4,
        max_features="log2",
    )

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 37
    assert model.max_depth == 5
    assert model.min_samples_split == 4
    assert model.max_features == "log2"
    assert model.random_state == 42
    assert model.class_weight == "balanced"
    assert model.n_jobs == -1


def test_07_run_grid_search_matches_sklearn():
    X, y = _small_classification_data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

    grid = {
        "n_estimators": [15, 25],
        "max_depth": [3, None],
    }

    result = run_grid_search(X, y, grid, cv)

    expected = GridSearchCV(
        build_random_forest(),
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    expected.fit(X, y)

    assert set(result.keys()) == {
        "best_params",
        "best_cv_score",
        "n_evaluated",
    }
    assert result["best_params"] == expected.best_params_
    assert result["best_cv_score"] == pytest.approx(expected.best_score_)
    assert result["n_evaluated"] == len(expected.cv_results_["params"])


def test_08_run_random_search_matches_sklearn():
    X, y = _small_classification_data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)

    space = {
        "n_estimators": [12, 18, 24, 30],
        "max_depth": [2, 4, None],
        "min_samples_split": [2, 5],
    }

    result = run_random_search(
        X,
        y,
        param_distributions=space,
        n_iter=4,
        cv=cv,
        seed=13,
    )

    expected = RandomizedSearchCV(
        build_random_forest(),
        param_distributions=space,
        n_iter=4,
        scoring="roc_auc",
        cv=cv,
        random_state=13,
        n_jobs=-1,
    )
    expected.fit(X, y)

    assert result["best_params"] == expected.best_params_
    assert result["best_cv_score"] == pytest.approx(expected.best_score_)
    assert result["n_evaluated"] == 4


def test_09_run_bayesian_search_returns_valid_trials_and_bounds():
    X, y = _small_classification_data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)

    result = run_bayesian_search(
        X,
        y,
        n_trials=3,
        cv=cv,
        seed=17,
    )

    assert set(result.keys()) == {
        "best_params",
        "best_cv_score",
        "n_evaluated",
    }
    assert result["n_evaluated"] == 3
    assert 0.0 <= result["best_cv_score"] <= 1.0

    p = result["best_params"]
    assert 60 <= p["n_estimators"] <= 220
    assert 3 <= p["max_depth"] <= 15
    assert 2 <= p["min_samples_split"] <= 12
    assert p["max_features"] in ["sqrt", "log2", 0.5, 0.8]


def test_10_classification_scores_uses_roc_auc_and_threshold_point_five():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.10, 0.70, 0.80, 0.40, 0.95, 0.20])

    result = classification_scores(y_true, y_prob)

    expected_pred = (y_prob >= 0.5).astype(int)

    assert set(result.keys()) == {"roc_auc", "f1"}
    assert result["roc_auc"] == pytest.approx(
        roc_auc_score(y_true, y_prob)
    )
    assert result["f1"] == pytest.approx(
        f1_score(y_true, expected_pred)
    )


# ============================================================
# Problem 3 — PyOD anomaly detection (Tests 11–15)
# ============================================================

def test_11_build_pyod_models_returns_required_pyod_classes():
    models = build_pyod_models(
        contamination=0.08,
        n_neighbors=7,
    )

    assert set(models.keys()) == {"KNN", "LOF", "IForest"}
    assert isinstance(models["KNN"], KNN)
    assert isinstance(models["LOF"], LOF)
    assert isinstance(models["IForest"], IForest)

    assert models["KNN"].contamination == pytest.approx(0.08)
    assert models["LOF"].contamination == pytest.approx(0.08)
    assert models["IForest"].contamination == pytest.approx(0.08)


def test_12_fit_pyod_scores_returns_decision_scores_and_correct_shape():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 3))

    model = KNN(contamination=0.10, n_neighbors=5)
    scores = fit_pyod_scores(model, X)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (80,)
    assert np.isfinite(scores).all()
    assert np.allclose(scores, model.decision_scores_)


def test_13_flag_top_fraction_exact_count_ranking_and_tie_break():
    scores = np.array([1.0, 5.0, 3.0, 8.0, 2.0])

    flags = flag_top_fraction(scores, fraction=0.40)

    assert flags.sum() == 2
    assert np.array_equal(
        flags,
        np.array([0, 1, 0, 1, 0])
    )


def test_14_compare_pyod_models_output_structure_and_flag_count():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(100, 4))

    result = compare_pyod_models(
        X,
        contamination=0.10,
        n_neighbors=6,
        flag_fraction=0.10,
    )

    assert set(result.keys()) == {"KNN", "LOF", "IForest"}

    for name in result:
        assert set(result[name].keys()) == {"scores", "flags"}
        assert result[name]["scores"].shape == (100,)
        assert result[name]["flags"].shape == (100,)
        assert set(np.unique(result[name]["flags"])).issubset({0, 1})
        assert int(result[name]["flags"].sum()) == 10


def test_15_pyod_models_work_on_assignment_dataset_without_identifier():
    df = pd.read_csv(
        DATA_DIR / "assignment6_anomaly_unlabeled.csv"
    )
    feature_cols = [
        c for c in df.columns
        if c != "record_id"
    ]
    X = StandardScaler().fit_transform(df[feature_cols].iloc[:250])

    result = compare_pyod_models(
        X,
        contamination=0.08,
        n_neighbors=10,
        flag_fraction=0.08,
    )

    for name in ("KNN", "LOF", "IForest"):
        assert np.isfinite(result[name]["scores"]).all()
        assert int(result[name]["flags"].sum()) == 20


# ============================================================
# Problem 4 — Semi-supervised pseudo-labeling (Tests 16–20)
# ============================================================

def test_16_prepare_semisupervised_sets_uses_set_type_and_features_only():
    df = pd.DataFrame({
        "record_id": ["a", "b", "c", "d", "e"],
        "f1": [1, 2, 3, 4, 5],
        "f2": [6, 7, 8, 9, 10],
        "set_type": [
            "labeled",
            "unlabeled",
            "evaluation",
            "labeled",
            "evaluation",
        ],
        "label": [0, np.nan, 1, 1, 0],
    })

    result = prepare_semisupervised_sets(
        df,
        feature_columns=["f1", "f2"],
    )

    X_l, y_l, X_u, X_e, y_e = result

    assert list(X_l.columns) == ["f1", "f2"]
    assert X_l.shape == (2, 2)
    assert np.array_equal(np.asarray(y_l), np.array([0, 1]))
    assert X_u.shape == (1, 2)
    assert X_e.shape == (2, 2)
    assert np.array_equal(np.asarray(y_e), np.array([1, 0]))


def test_17_build_semisupervised_rf_has_required_settings():
    model = build_semisupervised_rf()

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200
    assert model.class_weight == "balanced"
    assert model.random_state == 42


class _FixedProbabilityModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        p = self.probs[:len(X)]
        return np.column_stack([1 - p, p])


def test_18_select_pseudo_labels_respects_thresholds_and_order():
    X = np.zeros((6, 2))
    model = _FixedProbabilityModel(
        [0.02, 0.10, 0.35, 0.90, 0.97, 0.60]
    )

    result = select_pseudo_labels(
        model,
        X,
        low_threshold=0.10,
        high_threshold=0.90,
    )

    assert set(result.keys()) == {
        "selected_indices",
        "pseudo_labels",
        "n_pseudo_normal",
        "n_pseudo_anomaly",
    }

    assert np.array_equal(
        result["selected_indices"],
        np.array([0, 1, 3, 4]),
    )
    assert np.array_equal(
        result["pseudo_labels"],
        np.array([0, 0, 1, 1]),
    )
    assert result["n_pseudo_normal"] == 2
    assert result["n_pseudo_anomaly"] == 2


def test_19_retrain_with_pseudo_labels_returns_fitted_required_rf():
    rng = np.random.default_rng(91)

    X_l = pd.DataFrame(rng.normal(size=(40, 3)))
    y_l = np.array([0] * 30 + [1] * 10)
    X_u = pd.DataFrame(rng.normal(size=(20, 3)))

    selected = np.array([1, 4, 7, 10])
    pseudo = np.array([0, 0, 1, 1])

    model = retrain_with_pseudo_labels(
        X_l,
        y_l,
        X_u,
        selected,
        pseudo,
    )

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200
    assert model.random_state == 42

    prob = model.predict_proba(X_u.iloc[:5])[:, 1]
    assert prob.shape == (5,)
    assert np.isfinite(prob).all()


def test_20_semisupervised_evaluation_matches_sklearn_metrics():
    rng = np.random.default_rng(101)

    X_train = rng.normal(size=(100, 4))
    y_train = np.array([0] * 70 + [1] * 30)

    X_eval = rng.normal(size=(40, 4))
    y_eval = np.array([0, 1] * 20)

    initial = RandomForestClassifier(
        n_estimators=25,
        random_state=1,
    ).fit(X_train, y_train)

    retrained = RandomForestClassifier(
        n_estimators=25,
        random_state=2,
    ).fit(X_train, y_train)

    result = evaluate_semisupervised_models(
        initial,
        retrained,
        X_eval,
        y_eval,
    )

    p1 = initial.predict_proba(X_eval)[:, 1]
    p2 = retrained.predict_proba(X_eval)[:, 1]

    assert set(result.keys()) == {
        "initial_roc_auc",
        "initial_f1",
        "retrained_roc_auc",
        "retrained_f1",
    }

    assert result["initial_roc_auc"] == pytest.approx(
        roc_auc_score(y_eval, p1)
    )
    assert result["initial_f1"] == pytest.approx(
        f1_score(y_eval, (p1 >= 0.5).astype(int))
    )
    assert result["retrained_roc_auc"] == pytest.approx(
        roc_auc_score(y_eval, p2)
    )
    assert result["retrained_f1"] == pytest.approx(
        f1_score(y_eval, (p2 >= 0.5).astype(int))
    )
