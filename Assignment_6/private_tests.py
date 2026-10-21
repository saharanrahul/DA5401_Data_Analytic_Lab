"""
50 hard "private-style" tests for DA5401 Assignment 6.
Run:  pytest -q private_tests.py
(CSV files must be next to this file with the original names.)
"""
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import randint
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

import assignment_submission as sub

pytestmark = pytest.mark.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent


# ------------------------------------------------------------------ helpers
def blobs(n=150, seed=0, centers=3):
    X, _ = make_blobs(n_samples=n, centers=centers, cluster_std=0.5, random_state=seed)
    return StandardScaler().fit_transform(X)


def clf_data(n=160, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    lin = 1.4 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    return X, y


class FixedProb:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, X):
        p = self.p[: len(X)]
        return np.column_stack([1 - p, p])


def semi_df(n=120, seed=5):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    lin = 1.5 * df["a"] - df["b"]
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * lin))).astype(int)
    st = np.array(["labeled"] * 40 + ["unlabeled"] * 50 + ["evaluation"] * 30)
    df["grp"] = st
    df["target"] = np.where(st == "unlabeled", np.nan, y).astype(float)
    return df


# ============================================================ P1 (14 tests)
def test_p1_01_all_noise():
    X = np.random.default_rng(0).normal(size=(5, 2))
    r = sub.cluster_summary(X, np.full(5, -1))
    assert r["n_clusters"] == 0 and r["cluster_sizes"] == []
    assert r["noise_fraction"] == pytest.approx(1.0)
    assert np.isnan(r["silhouette"])


def test_p1_02_no_noise_sizes_ascending():
    X = np.vstack([np.zeros((4, 2)) + np.arange(4)[:, None] * 0.01,
                   np.ones((2, 2)) * 5 + np.arange(2)[:, None] * 0.01])
    labels = np.array([1, 1, 1, 1, 0, 0])
    r = sub.cluster_summary(X, labels)
    assert r["noise_fraction"] == 0.0
    assert r["cluster_sizes"] == [2, 4]


def test_p1_03_noncontiguous_labels():
    X = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5], [5.2, 5.1], [20, 20]], float)
    r = sub.cluster_summary(X, np.array([7, 7, 3, 3, 3, -1]))
    assert r["n_clusters"] == 2 and r["cluster_sizes"] == [2, 3]


def test_p1_04_silhouette_uses_only_non_noise():
    X = np.array([[0, 0], [0.2, 0], [0, 0.2], [4, 4], [4.2, 4], [4, 4.2], [30, -30]], float)
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    r = sub.cluster_summary(X, labels)
    m = labels != -1
    assert r["silhouette"] == pytest.approx(silhouette_score(X[m], labels[m]))
    # treating noise as an extra cluster would give a different value
    assert abs(r["silhouette"] - silhouette_score(X, labels)) > 1e-3


def test_p1_05_accepts_list_and_dataframe_inputs():
    X = [[0, 0], [0.1, 0], [3, 3], [3.1, 3], [9, 9]]
    r1 = sub.cluster_summary(X, [0, 0, 1, 1, -1])
    r2 = sub.cluster_summary(pd.DataFrame(X), pd.Series([0, 0, 1, 1, -1]))
    assert r1["cluster_sizes"] == r2["cluster_sizes"] == [2, 2]
    assert r1["silhouette"] == pytest.approx(r2["silhouette"])


def test_p1_06_return_types():
    X = blobs()
    r = sub.cluster_summary(X, DBSCAN(eps=0.3, min_samples=5).fit_predict(X))
    assert type(r["n_clusters"]) is int
    assert type(r["noise_fraction"]) is float
    assert all(type(v) is int for v in r["cluster_sizes"])
    assert r["cluster_sizes"] == sorted(r["cluster_sizes"])


def test_p1_07_run_dbscan_matches_sklearn_and_exact_keys():
    X = blobs(seed=1)
    r = sub.run_dbscan(X, 0.25, 6)
    exp = DBSCAN(eps=0.25, min_samples=6, metric="euclidean").fit_predict(X)
    assert set(r) == {"eps", "min_samples", "labels", "n_clusters",
                      "noise_fraction", "cluster_sizes", "silhouette"}
    assert np.array_equal(r["labels"], exp)
    assert r["eps"] == 0.25 and r["min_samples"] == 6


def test_p1_08_run_dbscan_no_clusters():
    X = np.random.default_rng(2).normal(size=(60, 2))
    r = sub.run_dbscan(X, 1e-6, 5)
    assert (r["labels"] == -1).all() and r["n_clusters"] == 0
    assert np.isnan(r["silhouette"]) and r["noise_fraction"] == 1.0


def test_p1_09_evaluate_dbscan_configs_order_and_empty():
    X = blobs(seed=2)
    cfgs = [{"eps": 0.4, "min_samples": 4}, {"eps": 0.1, "min_samples": 8},
            {"eps": 0.25, "min_samples": 5}]
    res = sub.evaluate_dbscan_configs(X, cfgs)
    assert [(r["eps"], r["min_samples"]) for r in res] == [(0.4, 4), (0.1, 8), (0.25, 5)]
    assert sub.evaluate_dbscan_configs(X, []) == []


def test_p1_10_run_hdbscan_keys_and_min_samples_passthrough():
    X = blobs(seed=3)
    r = sub.run_hdbscan(X, 10, min_samples=3)
    exp = HDBSCAN(min_cluster_size=10, min_samples=3, metric="euclidean").fit_predict(X)
    assert set(r) == {"min_cluster_size", "min_samples", "labels", "n_clusters",
                      "noise_fraction", "cluster_sizes", "silhouette"}
    assert r["min_samples"] == 3 and np.array_equal(r["labels"], exp)
    assert sub.run_hdbscan(X, 10)["min_samples"] is None


def test_p1_11_evaluate_hdbscan_configs_order_and_min_samples_none():
    X = blobs(seed=4)
    res = sub.evaluate_hdbscan_configs(X, [15, 5, 10])
    assert [r["min_cluster_size"] for r in res] == [15, 5, 10]
    assert all(r["min_samples"] is None for r in res)


def test_p1_12_select_tie_within_float_precision_prefers_lower_noise():
    results = [
        {"n_clusters": 2, "noise_fraction": 0.10, "silhouette": 0.5},
        {"n_clusters": 3, "noise_fraction": 0.05, "silhouette": 0.5 + 1e-13},
        {"n_clusters": 2, "noise_fraction": 0.02, "silhouette": 0.4},
    ]
    assert sub.select_density_result(results) is results[1]
    # higher silhouette only by noise-level epsilon but higher noise: keep lower-noise first
    results2 = [
        {"n_clusters": 2, "noise_fraction": 0.05, "silhouette": 0.5},
        {"n_clusters": 2, "noise_fraction": 0.10, "silhouette": 0.5 + 1e-13},
    ]
    assert sub.select_density_result(results2) is results2[0]


def test_p1_13_select_full_tie_keeps_input_order():
    results = [
        {"id": "a", "n_clusters": 2, "noise_fraction": 0.1, "silhouette": 0.6},
        {"id": "b", "n_clusters": 2, "noise_fraction": 0.1, "silhouette": 0.6},
        {"id": "c", "n_clusters": 2, "noise_fraction": 0.1 + 1e-13, "silhouette": 0.6},
    ]
    assert sub.select_density_result(results)["id"] == "a"


def test_p1_14_select_ignores_invalid_and_returns_none():
    bad = [
        {"n_clusters": 1, "noise_fraction": 0.0, "silhouette": 0.99},
        {"n_clusters": 3, "noise_fraction": 0.0, "silhouette": float("nan")},
        {"n_clusters": 2, "noise_fraction": 0.0, "silhouette": None},
    ]
    assert sub.select_density_result(bad) is None
    assert sub.select_density_result([]) is None
    good = bad + [{"n_clusters": 2, "noise_fraction": 0.3, "silhouette": 0.1}]
    snapshot = [dict(r) for r in good]
    assert sub.select_density_result(good) is good[3]
    assert good == snapshot  # not mutated


# ============================================================ P2 (11 tests)
def test_p2_15_build_rf_defaults_and_passthrough():
    m = sub.build_random_forest()
    assert (m.random_state, m.class_weight, m.n_jobs) == (42, "balanced", -1)
    m = sub.build_random_forest(criterion="entropy", max_samples=0.5, bootstrap=True)
    assert m.criterion == "entropy" and m.max_samples == 0.5
    assert (m.random_state, m.class_weight, m.n_jobs) == (42, "balanced", -1)


def test_p2_16_grid_list_of_dicts_counts_candidates():
    X, y = clf_data()
    cv = StratifiedKFold(3, shuffle=True, random_state=1)
    grid = [{"n_estimators": [8, 12]}, {"max_depth": [2, 3, 4], "n_estimators": [10]}]
    r = sub.run_grid_search(X, y, grid, cv)
    assert r["n_evaluated"] == 5 and isinstance(r["best_params"], dict)


def test_p2_17_grid_best_score_is_recomputable_mean_cv_auc():
    X, y = clf_data()
    cv = StratifiedKFold(3, shuffle=True, random_state=2)
    r = sub.run_grid_search(X, y, {"n_estimators": [10, 20], "max_depth": [2, None]}, cv)
    manual = cross_val_score(sub.build_random_forest(**r["best_params"]), X, y,
                             cv=cv, scoring="roc_auc").mean()
    assert r["best_cv_score"] == pytest.approx(manual)
    assert type(r["best_cv_score"]) is float and type(r["n_evaluated"]) is int


def test_p2_18_grid_accepts_integer_cv():
    X, y = clf_data()
    r = sub.run_grid_search(X, y, {"n_estimators": [8, 12]}, 3)
    assert r["n_evaluated"] == 2 and 0 <= r["best_cv_score"] <= 1


def test_p2_19_random_search_reproducible_and_n_iter_respected():
    X, y = clf_data()
    cv = StratifiedKFold(3, shuffle=True, random_state=4)
    space = {"n_estimators": [8, 10, 12, 14, 16], "max_depth": [2, 3, 4, None],
             "min_samples_split": [2, 4, 6]}
    a = sub.run_random_search(X, y, space, 3, cv, 21)
    b = sub.run_random_search(X, y, space, 3, cv, 21)
    assert a == b and a["n_evaluated"] == 3


def test_p2_20_random_search_n_iter_larger_than_grid():
    X, y = clf_data()
    cv = StratifiedKFold(3, shuffle=True, random_state=4)
    r = sub.run_random_search(X, y, {"n_estimators": [8, 12], "max_depth": [2, 3]}, 10, cv, 0)
    assert r["n_evaluated"] == 4


def test_p2_21_random_search_scipy_distribution():
    X, y = clf_data()
    cv = StratifiedKFold(3, shuffle=True, random_state=4)
    r = sub.run_random_search(X, y, {"n_estimators": randint(8, 20),
                                      "max_depth": [2, 3, None]}, 3, cv, 5)
    assert r["n_evaluated"] == 3 and 8 <= r["best_params"]["n_estimators"] < 20


def test_p2_22_bayesian_reproducible_same_seed():
    X, y = clf_data(120)
    cv = StratifiedKFold(3, shuffle=True, random_state=6)
    a = sub.run_bayesian_search(X, y, 3, cv, 11)
    b = sub.run_bayesian_search(X, y, 3, cv, 11)
    assert a == b


def test_p2_23_bayesian_score_matches_manual_cv_and_param_keys():
    X, y = clf_data(120)
    cv = StratifiedKFold(3, shuffle=True, random_state=6)
    r = sub.run_bayesian_search(X, y, 4, cv, 3)
    assert set(r["best_params"]) == {"n_estimators", "max_depth", "min_samples_split", "max_features"}
    assert r["n_evaluated"] == 4
    manual = cross_val_score(sub.build_random_forest(**r["best_params"]), X, y,
                             cv=cv, scoring="roc_auc").mean()
    assert r["best_cv_score"] == pytest.approx(manual)


def test_p2_24_bayesian_param_bounds_over_many_trials():
    X, y = clf_data(100)
    r = sub.run_bayesian_search(X, y, 6, 3, 8)
    p = r["best_params"]
    assert 60 <= p["n_estimators"] <= 220 and 3 <= p["max_depth"] <= 15
    assert 2 <= p["min_samples_split"] <= 12
    assert p["max_features"] in ["sqrt", "log2", 0.5, 0.8]


def test_p2_25_classification_scores_threshold_boundary_and_lists():
    r = sub.classification_scores([0, 1, 1, 0], [0.5, 0.5, 0.2, 0.1])
    assert r["f1"] == pytest.approx(0.5)  # >=0.5 counts as class 1
    assert r["roc_auc"] == pytest.approx(roc_auc_score([0, 1, 1, 0], [0.5, 0.5, 0.2, 0.1]))
    assert type(r["roc_auc"]) is float and type(r["f1"]) is float


def test_p2_26_classification_scores_perfect_and_series_inputs():
    y = pd.Series([0, 0, 1, 1], index=[9, 5, 2, 7])
    p = pd.Series([0.1, 0.2, 0.8, 0.9], index=[9, 5, 2, 7])
    r = sub.classification_scores(y, p)
    assert r["roc_auc"] == pytest.approx(1.0) and r["f1"] == pytest.approx(1.0)


# ============================================================ P3 (11 tests)
def outlier_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 2))
    out = np.array([[15, 15], [-15, 15], [15, -15], [-15, -15], [0, 22]], float)
    return np.vstack([X, out])


def test_p3_27_build_models_parameters():
    m = sub.build_pyod_models(0.07, 9)
    assert m["KNN"].n_neighbors == 9 and m["LOF"].n_neighbors == 9
    assert m["IForest"].n_estimators == 200 and m["IForest"].random_state == 42
    assert all(m[k].contamination == pytest.approx(0.07) for k in m)


@pytest.mark.parametrize("name", ["KNN", "LOF", "IForest"])
def test_p3_28_larger_score_is_more_anomalous(name):
    X = outlier_data()
    res = sub.compare_pyod_models(X, 0.05, 20, 0.05)
    top5 = set(np.argsort(-res[name]["scores"])[:5])
    assert len(top5 & set(range(200, 205))) >= 4


def test_p3_29_fit_pyod_scores_returns_float_array_and_fits_model():
    X = np.random.default_rng(1).normal(size=(60, 3))
    model = LOF(contamination=0.1, n_neighbors=8)
    s = sub.fit_pyod_scores(model, pd.DataFrame(X))
    assert isinstance(s, np.ndarray) and s.dtype == float and s.shape == (60,)
    assert np.allclose(s, model.decision_scores_)


def test_p3_30_flag_ceil_non_integer():
    f = sub.flag_top_fraction(np.arange(7, dtype=float), 0.3)
    assert f.sum() == 3 and list(np.flatnonzero(f)) == [4, 5, 6]


def test_p3_31_flag_all_equal_scores_smaller_index_first():
    f = sub.flag_top_fraction(np.ones(10), 0.3)
    assert list(np.flatnonzero(f)) == [0, 1, 2]


def test_p3_32_flag_tie_at_boundary():
    f = sub.flag_top_fraction(np.array([5, 5, 5, 1.0]), 0.5)
    assert list(f) == [1, 1, 0, 0]
    f = sub.flag_top_fraction(np.array([2, 9, 9, 9, 1.0]), 0.4)  # k=2 -> idx1, idx2
    assert list(f) == [0, 1, 1, 0, 0]


def test_p3_33_flag_fraction_extremes_and_tiny_fraction():
    s = np.array([3, 1, 2, 5, 4.0])
    assert sub.flag_top_fraction(s, 0).sum() == 0
    assert sub.flag_top_fraction(s, 1).sum() == 5
    assert sub.flag_top_fraction(np.arange(10.0), 0.001).sum() == 1


def test_p3_34_flag_dtype_list_input_negative_and_no_mutation():
    s = [-3.0, -1.0, -2.0, -5.0]
    copy = list(s)
    f = sub.flag_top_fraction(s, 0.5)
    assert np.issubdtype(f.dtype, np.integer) and set(np.unique(f)) <= {0, 1}
    assert list(f) == [0, 1, 1, 0] and s == copy


def test_p3_35_compare_matches_direct_pyod_fits():
    X = outlier_data()
    res = sub.compare_pyod_models(X, 0.1, 15, 0.1)
    knn = KNN(contamination=0.1, n_neighbors=15).fit(X)
    lof = LOF(contamination=0.1, n_neighbors=15).fit(X)
    iso = IForest(contamination=0.1, n_estimators=200, random_state=42).fit(X)
    assert np.allclose(res["KNN"]["scores"], knn.decision_scores_)
    assert np.allclose(res["LOF"]["scores"], lof.decision_scores_)
    assert np.allclose(res["IForest"]["scores"], iso.decision_scores_)


def test_p3_36_compare_deterministic_and_dataframe_input():
    X = outlier_data()
    a = sub.compare_pyod_models(X, 0.1, 10, 0.1)
    b = sub.compare_pyod_models(pd.DataFrame(X), 0.1, 10, 0.1)
    for k in a:
        assert np.allclose(a[k]["scores"], b[k]["scores"])
        assert np.array_equal(a[k]["flags"], b[k]["flags"])


def test_p3_37_flag_count_depends_on_flag_fraction_not_contamination():
    X = outlier_data()
    res = sub.compare_pyod_models(X, 0.30, 12, 0.04)
    n = len(X)
    for k in res:
        assert int(res[k]["flags"].sum()) == int(np.ceil(0.04 * n))


# ============================================================ P4 (10 tests)
def test_p4_38_prepare_custom_names_no_identifier():
    df = semi_df()
    Xl, yl, Xu, Xe, ye = sub.prepare_semisupervised_sets(df, ["c", "a"], "grp", "target")
    assert list(Xl.columns) == ["c", "a"] and list(Xu.columns) == ["c", "a"]
    assert (Xl.shape, Xu.shape, Xe.shape) == ((40, 2), (50, 2), (30, 2))
    assert len(yl) == 40 and len(ye) == 30


def test_p4_39_prepare_values_order_and_label_dtype():
    df = semi_df()
    Xl, yl, Xu, Xe, ye = sub.prepare_semisupervised_sets(df, ["a", "b", "c"], "grp", "target")
    exp = df[df["grp"] == "labeled"]
    assert np.allclose(np.asarray(Xl, float), exp[["a", "b", "c"]].to_numpy())
    assert np.array_equal(np.asarray(yl), exp["target"].to_numpy().astype(int))
    assert np.issubdtype(np.asarray(yl).dtype, np.integer)
    assert np.issubdtype(np.asarray(ye).dtype, np.integer)
    assert not np.isnan(np.asarray(Xu, float)).any()


def test_p4_40_prepare_ignores_other_columns_and_does_not_mutate():
    df = semi_df()
    df["junk"] = "x"
    snap = df.copy(deep=True)
    out = sub.prepare_semisupervised_sets(df, ["a"], "grp", "target")
    assert list(out[0].columns) == ["a"]
    pd.testing.assert_frame_equal(df, snap)


def test_p4_41_select_real_rf_matches_independent_rule():
    df = semi_df()
    Xl, yl, Xu, Xe, ye = sub.prepare_semisupervised_sets(df, ["a", "b", "c"], "grp", "target")
    model = sub.build_semisupervised_rf().fit(Xl, yl)
    r = sub.select_pseudo_labels(model, Xu, 0.2, 0.8)
    p = model.predict_proba(Xu)[:, 1]
    exp_idx = np.flatnonzero((p <= 0.2) | (p >= 0.8))
    assert np.array_equal(r["selected_indices"], exp_idx)
    assert np.array_equal(r["pseudo_labels"], (p[exp_idx] >= 0.8).astype(int))
    assert r["n_pseudo_normal"] + r["n_pseudo_anomaly"] == len(exp_idx)


def test_p4_42_select_boundaries_inclusive_and_none_selected():
    X = np.zeros((4, 1))
    r = sub.select_pseudo_labels(FixedProb([0.1, 0.9, 0.5, 0.11]), X, 0.1, 0.9)
    assert list(r["selected_indices"]) == [0, 1] and list(r["pseudo_labels"]) == [0, 1]
    r = sub.select_pseudo_labels(FixedProb([0.4, 0.5, 0.6]), np.zeros((3, 1)), 0.1, 0.9)
    assert len(r["selected_indices"]) == 0 and len(r["pseudo_labels"]) == 0
    assert r["n_pseudo_normal"] == 0 and r["n_pseudo_anomaly"] == 0
    r = sub.select_pseudo_labels(FixedProb([0.0, 1.0, 0.05, 0.95]), np.zeros((4, 1)), 0.1, 0.9)
    assert list(r["selected_indices"]) == [0, 1, 2, 3]
    assert (r["n_pseudo_normal"], r["n_pseudo_anomaly"]) == (2, 2)


def test_p4_43_select_returns_positional_indices_for_shuffled_index():
    X = pd.DataFrame({"v": [1, 2, 3, 4]}, index=[40, 7, 99, 3])
    r = sub.select_pseudo_labels(FixedProb([0.01, 0.5, 0.99, 0.5]), X, 0.1, 0.9)
    assert list(r["selected_indices"]) == [0, 2]
    assert np.issubdtype(np.asarray(r["selected_indices"]).dtype, np.integer)


def test_p4_44_retrain_equals_reference_forest():
    df = semi_df()
    Xl, yl, Xu, Xe, ye = sub.prepare_semisupervised_sets(df, ["a", "b", "c"], "grp", "target")
    sel = np.array([0, 3, 4, 9, 20])
    pl = np.array([0, 1, 1, 0, 1])
    m = sub.retrain_with_pseudo_labels(Xl, yl, Xu, sel, pl)
    ref = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    ref.fit(np.vstack([np.asarray(Xl), np.asarray(Xu)[sel]]), np.concatenate([np.asarray(yl), pl]))
    assert np.allclose(m.predict_proba(np.asarray(Xe)), ref.predict_proba(np.asarray(Xe)))


def test_p4_45_retrain_shuffled_index_uses_positions_and_empty_selection():
    df = semi_df()
    Xl, yl, Xu, Xe, ye = sub.prepare_semisupervised_sets(df, ["a", "b", "c"], "grp", "target")
    Xu2 = Xu.sample(frac=1.0, random_state=1)  # non-default, shuffled index
    sel, pl = np.array([1, 5, 8]), np.array([1, 0, 0])
    m = sub.retrain_with_pseudo_labels(Xl, yl, Xu2, sel, pl)
    ref = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    ref.fit(np.vstack([np.asarray(Xl), np.asarray(Xu2)[sel]]), np.concatenate([np.asarray(yl), pl]))
    assert np.allclose(m.predict_proba(np.asarray(Xe)), ref.predict_proba(np.asarray(Xe)))
    m0 = sub.retrain_with_pseudo_labels(Xl, yl, Xu, np.array([], int), np.array([], int))
    init = sub.build_semisupervised_rf().fit(np.asarray(Xl), np.asarray(yl))
    assert np.allclose(m0.predict_proba(np.asarray(Xe)), init.predict_proba(np.asarray(Xe)))


def test_p4_46_retrain_new_object_and_inputs_not_mutated():
    rng = np.random.default_rng(0)
    Xl, yl = rng.normal(size=(30, 3)), np.array([0] * 20 + [1] * 10)
    Xu = rng.normal(size=(15, 3))
    snaps = (Xl.copy(), yl.copy(), Xu.copy())
    init = sub.build_semisupervised_rf().fit(Xl, yl)
    m = sub.retrain_with_pseudo_labels(Xl, yl, Xu, np.array([2, 6]), np.array([0, 1]))
    assert m is not init and isinstance(m, RandomForestClassifier)
    assert (m.n_estimators, m.class_weight, m.random_state) == (200, "balanced", 42)
    for a, b in zip((Xl, yl, Xu), snaps):
        assert np.array_equal(a, b)


def test_p4_47_evaluate_uses_class1_column_and_half_threshold():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(120, 3)), np.array([0, 1] * 60)
    m1 = RandomForestClassifier(n_estimators=20, random_state=0).fit(X[:80], y[:80])
    m2 = RandomForestClassifier(n_estimators=20, random_state=5).fit(X[:80], y[:80])
    ye = pd.Series(y[80:], index=np.arange(80, 120)[::-1])
    r = sub.evaluate_semisupervised_models(m1, m2, X[80:], ye)
    p1, p2 = m1.predict_proba(X[80:])[:, 1], m2.predict_proba(X[80:])[:, 1]
    assert r["initial_roc_auc"] == pytest.approx(roc_auc_score(ye, p1))
    assert r["retrained_f1"] == pytest.approx(f1_score(ye, (p2 >= 0.5).astype(int)))


# ============================================ dataset golden tests (3 tests)
@pytest.fixture(scope="module")
def density_X():
    return StandardScaler().fit_transform(pd.read_csv(DATA / "assignment6_density_clustering.csv").values)


def test_g_48_density_dataset_selection(density_X):
    db = sub.evaluate_dbscan_configs(density_X, [
        {"eps": 0.12, "min_samples": 5}, {"eps": 0.18, "min_samples": 5},
        {"eps": 0.24, "min_samples": 8}, {"eps": 0.30, "min_samples": 10}])
    sel = sub.select_density_result(db)
    assert (sel["eps"], sel["min_samples"], sel["n_clusters"]) == (0.12, 5, 4)
    assert sel["cluster_sizes"] == [340, 472, 491, 953]
    assert sel["noise_fraction"] == pytest.approx(0.0791837, abs=1e-6)
    assert sel["silhouette"] == pytest.approx(0.5169081, abs=1e-5)
    hd = sub.evaluate_hdbscan_configs(density_X, [20, 40, 70, 100])
    hsel = sub.select_density_result(hd)
    assert hsel["min_cluster_size"] == 20 and hsel["n_clusters"] == 4
    assert hsel["silhouette"] == pytest.approx(0.5159, abs=2e-3)


def test_g_49_pyod_dataset_counts_and_overlaps():
    df = pd.read_csv(DATA / "assignment6_anomaly_unlabeled.csv")
    feats = [c for c in df.columns if c != "record_id"]
    X = StandardScaler().fit_transform(df[feats].values)
    res = sub.compare_pyod_models(X, 0.10, 20, 0.10)
    assert all(int(res[k]["flags"].sum()) == 300 for k in res)
    ov = lambda a, b: int((res[a]["flags"] & res[b]["flags"]).sum())
    assert (ov("KNN", "LOF"), ov("KNN", "IForest"), ov("LOF", "IForest")) == (104, 189, 68)
    top = lambda k: df["record_id"].iloc[np.argsort(-res[k]["scores"], kind="stable")[:3]].tolist()
    assert top("KNN") == ["REC-00035", "REC-00259", "REC-02873"]


def test_g_50_semisupervised_dataset_end_to_end():
    df = pd.read_csv(DATA / "assignment6_anomaly_semisupervised.csv")
    feats = [c for c in df.columns if c not in ("record_id", "set_type", "label")]
    Xl, yl, Xu, Xe, ye = sub.prepare_semisupervised_sets(df, feats, "set_type", "label")
    assert len(Xl) + len(Xu) + len(Xe) == len(df)
    m0 = sub.build_semisupervised_rf().fit(Xl, yl)
    ps = sub.select_pseudo_labels(m0, Xu, 0.10, 0.90)
    assert (ps["n_pseudo_normal"], ps["n_pseudo_anomaly"]) == (1390, 112)
    assert ps["selected_indices"].max() < len(Xu)
    m1 = sub.retrain_with_pseudo_labels(Xl, yl, Xu, ps["selected_indices"], ps["pseudo_labels"])
    r = sub.evaluate_semisupervised_models(m0, m1, Xe, ye)
    assert r["initial_roc_auc"] == pytest.approx(0.97964, abs=2e-3)
    assert r["retrained_f1"] == pytest.approx(0.89216, abs=2e-2)


# ------------------------------------------------ bonus (51st) regression test
def test_bonus_one_class_model_gives_zero_class1_probability():
    rng = np.random.default_rng(0)
    m = RandomForestClassifier(n_estimators=10, random_state=0).fit(rng.normal(size=(20, 2)), np.zeros(20, int))
    r = sub.select_pseudo_labels(m, rng.normal(size=(6, 2)), 0.1, 0.9)
    assert r["n_pseudo_anomaly"] == 0 and r["n_pseudo_normal"] == 6
