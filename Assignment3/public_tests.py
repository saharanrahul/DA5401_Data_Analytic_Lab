"""
Public tests for DA5401 Assignment 3.

Place this file in the same directory as:
    assignment_submission.py
    decision_tree_classification.csv
    random_forest_regression.csv

Run:
    pytest -q public_tests.py

These are PUBLIC tests only. Private tests may use different data
satisfying the contracts stated in the assignment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from assignment_submission import (
    gini_impurity,
    best_split,
    build_tree,
    predict_one,
    predict_tree,
    evaluate_depths,
    select_depth_behavior,
    build_tree_regression_pipeline,
    build_random_forest_pipeline,
    regression_metrics,
)


DATA_DIR = Path(__file__).resolve().parent


def _has_column_transformer(pipe):
    return any(
        isinstance(step, ColumnTransformer)
        for _, step in pipe.steps
    )


# ============================================================
# Problem 1 — 6 public tests
# ============================================================

def test_gini_impurity():
    assert gini_impurity([]) == pytest.approx(0.0)
    assert gini_impurity([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini_impurity([0, 0, 1, 1]) == pytest.approx(0.5)
    assert gini_impurity([0, 0, 0, 1]) == pytest.approx(0.375)


def test_best_split_known_case():
    X = pd.DataFrame({
        "x1": [0.0, 1.0, 2.0, 3.0],
        "x2": [5.0, 5.0, 5.0, 5.0],
    })
    y = pd.Series([0, 0, 1, 1])

    result = best_split(X, y)

    assert result is not None
    assert set(result.keys()) == {"feature", "threshold", "gain"}
    assert result["feature"] == "x1"
    assert result["threshold"] == pytest.approx(1.5)
    assert result["gain"] == pytest.approx(0.5)


def test_best_split_tie_breaks_by_column_order():
    # Both columns produce exactly the same best split.
    # The earlier column in X.columns must be chosen.
    X = pd.DataFrame({
        "zeta": [0.0, 1.0, 2.0, 3.0],
        "alpha": [0.0, 1.0, 2.0, 3.0],
    })
    y = pd.Series([0, 0, 1, 1])

    result = best_split(X, y)

    assert result is not None
    assert result["feature"] == "zeta"
    assert result["threshold"] == pytest.approx(1.5)


def test_build_tree_and_prediction():
    X = pd.DataFrame({
        "temperature": [10.0, 12.0, 14.0, 30.0, 32.0, 35.0],
        "load": [1.0, 2.0, 1.5, 2.0, 1.0, 1.5],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])

    tree = build_tree(
        X,
        y,
        max_depth=1,
        min_samples_split=2,
    )

    assert set(tree.keys()) == {
        "feature", "threshold", "left", "right"
    }
    assert set(tree["left"].keys()) == {"prediction"}
    assert set(tree["right"].keys()) == {"prediction"}

    one_pred = predict_one(tree, X.iloc[0])
    assert one_pred in (0, 1)

    preds = predict_tree(tree, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.array_equal(preds, y.to_numpy())


def test_evaluate_depths_on_clean_numeric_data():
    # This also checks that the implementation is data-agnostic:
    # arbitrary numerical feature names are used.
    X_train = pd.DataFrame({
        "sensor_A": [0, 1, 2, 3, 4, 5, 6, 7],
        "sensor_B": [2, 2, 1, 1, 3, 3, 4, 4],
    }, dtype=float)
    y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    X_val = pd.DataFrame({
        "sensor_A": [0.5, 2.5, 4.5, 6.5],
        "sensor_B": [2.0, 1.0, 3.0, 4.0],
    })
    y_val = pd.Series([0, 0, 1, 1])

    depths = [1, 2, None]

    results = evaluate_depths(
        X_train,
        y_train,
        X_val,
        y_val,
        depths,
        min_samples_split=2,
    )

    assert isinstance(results, list)
    assert len(results) == len(depths)
    assert [r["max_depth"] for r in results] == depths

    for result in results:
        assert {
            "max_depth",
            "train_accuracy",
            "validation_accuracy",
            "tree",
        }.issubset(result.keys())

        train_pred = predict_tree(result["tree"], X_train)
        val_pred = predict_tree(result["tree"], X_val)

        expected_train_acc = np.mean(
            train_pred == y_train.to_numpy()
        )
        expected_val_acc = np.mean(
            val_pred == y_val.to_numpy()
        )

        assert result["train_accuracy"] == pytest.approx(
            expected_train_acc
        )
        assert result["validation_accuracy"] == pytest.approx(
            expected_val_acc
        )


def test_select_depth_behavior_and_tie_breaking():
    results = [
        {
            "max_depth": 1,
            "train_accuracy": 0.60,
            "validation_accuracy": 0.55,
            "tree": None,
        },
        {
            "max_depth": 2,
            "train_accuracy": 0.60,
            "validation_accuracy": 0.80,
            "tree": None,
        },
        {
            "max_depth": 3,
            "train_accuracy": 0.90,
            "validation_accuracy": 0.80,
            "tree": None,
        },
        {
            "max_depth": None,
            "train_accuracy": 0.95,
            "validation_accuracy": 0.75,
            "tree": None,
        },
    ]

    result = select_depth_behavior(results)

    assert set(result.keys()) == {
        "underfit_depth",
        "overfit_depth",
        "best_depth",
    }

    # Lowest training accuracy is tied at depths 1 and 2:
    # choose the shallower depth.
    assert result["underfit_depth"] == 1

    # Largest train-validation gap occurs at None.
    assert result["overfit_depth"] is None

    # Highest validation accuracy is tied at 2 and 3:
    # choose the shallower depth.
    assert result["best_depth"] == 2


# ============================================================
# Problem 2 — 4 public tests
# ============================================================

def test_tree_regression_pipeline_structure():
    df = pd.read_csv(DATA_DIR / "random_forest_regression.csv")
    X = df.drop(columns=["maintenance_duration_hours"])

    pipe = build_tree_regression_pipeline(X)

    assert isinstance(pipe, Pipeline)
    assert _has_column_transformer(pipe)
    assert isinstance(pipe.steps[-1][1], DecisionTreeRegressor)
    assert pipe.steps[-1][1].random_state == 42

    # The assignment says not to tune the baseline tree.
    assert pipe.steps[-1][1].max_depth is None
    assert pipe.steps[-1][1].min_samples_split == 2


def test_random_forest_pipeline_structure():
    df = pd.read_csv(DATA_DIR / "random_forest_regression.csv")
    X = df.drop(columns=["maintenance_duration_hours"])

    pipe = build_random_forest_pipeline(
        X,
        n_estimators=50,
    )

    assert isinstance(pipe, Pipeline)
    assert _has_column_transformer(pipe)
    assert isinstance(pipe.steps[-1][1], RandomForestRegressor)
    assert pipe.steps[-1][1].n_estimators == 50
    assert pipe.steps[-1][1].random_state == 42

    # Other main complexity parameters should remain at defaults.
    assert pipe.steps[-1][1].max_depth is None
    assert pipe.steps[-1][1].min_samples_split == 2


def test_regression_pipelines_fit_raw_provided_data():
    df = pd.read_csv(DATA_DIR / "random_forest_regression.csv")
    X = df.drop(columns=["maintenance_duration_hours"])
    y = df["maintenance_duration_hours"]

    # The supplied data contain categorical values and missing values.
    # A valid pipeline should accept the raw DataFrame directly.
    X_train = X.iloc[:350].copy()
    y_train = y.iloc[:350].copy()
    X_test = X.iloc[350:375].copy()

    tree_pipe = build_tree_regression_pipeline(X_train)
    tree_pipe.fit(X_train, y_train)
    tree_pred = tree_pipe.predict(X_test)

    forest_pipe = build_random_forest_pipeline(
        X_train,
        n_estimators=10,
    )
    forest_pipe.fit(X_train, y_train)
    forest_pred = forest_pipe.predict(X_test)

    assert tree_pred.shape == (25,)
    assert forest_pred.shape == (25,)
    assert np.isfinite(tree_pred).all()
    assert np.isfinite(forest_pred).all()


def test_regression_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0])

    result = regression_metrics(y_true, y_pred)

    assert set(result.keys()) == {"mae", "rmse", "r2"}

    assert result["mae"] == pytest.approx(
        mean_absolute_error(y_true, y_pred)
    )
    assert result["rmse"] == pytest.approx(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )
    assert result["r2"] == pytest.approx(
        r2_score(y_true, y_pred)
    )
