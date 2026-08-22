"""
Public tests for DA5401 Assignment 2 — Reliable Model Evaluation with Pipelines

Place this file in the same folder as:
    assignment_submission.py
    regression_data.csv
    classification_data.csv

Run:
    pytest -q public_tests.py

There are 10 public tests:
    5 for Problem 1 (Regression)
    5 for Problem 2 (Classification)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from assignment_submission import (
    build_regression_pipeline,
    regression_metrics,
    build_classification_pipeline,
    classification_metrics,
    select_threshold,
)


DATA_DIR = Path(__file__).resolve().parent


def load_regression_data():
    df = pd.read_csv(DATA_DIR / "regression_data.csv")
    X = df.drop(columns=["job_cost"])
    y = df["job_cost"]
    return X, y


def load_classification_data():
    df = pd.read_csv(DATA_DIR / "classification_data.csv")
    X = df.drop(columns=["critical_failure_30d"])
    y = df["critical_failure_30d"]
    return X, y


def has_column_transformer(pipe):
    return any(
        isinstance(step, ColumnTransformer)
        for _, step in pipe.steps
    )


# ============================================================
# Problem 1 — Regression: 5 tests
# ============================================================

def test_regression_pipeline_structure():
    X, _ = load_regression_data()
    pipe = build_regression_pipeline(X)

    assert isinstance(pipe, Pipeline)
    assert has_column_transformer(pipe)
    assert isinstance(pipe.steps[-1][1], LinearRegression)


def test_regression_pipeline_fit_and_predict():
    X, y = load_regression_data()

    # Use a small subset to keep the public test quick.
    X_small = X.iloc[:120].copy()
    y_small = y.iloc[:120].copy()

    pipe = build_regression_pipeline(X_small)
    pipe.fit(X_small, y_small)
    y_pred = pipe.predict(X_small.iloc[:10])

    assert y_pred.shape == (10,)
    assert np.isfinite(y_pred).all()


def test_regression_pipeline_works_in_cross_validation():
    X, y = load_regression_data()

    X_small = X.iloc[:250].copy()
    y_small = y.iloc[:250].copy()

    pipe = build_regression_pipeline(X_small)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        pipe,
        X_small,
        y_small,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_regression_metrics_keys():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 180.0, 330.0, 390.0])

    result = regression_metrics(y_true, y_pred)

    assert set(result.keys()) == {
        "mae", "rmse", "mape", "r2"
    }


def test_regression_metrics_values():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 180.0, 330.0, 390.0])

    result = regression_metrics(y_true, y_pred)

    assert result["mae"] == pytest.approx(
        mean_absolute_error(y_true, y_pred)
    )
    assert result["rmse"] == pytest.approx(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )
    assert result["mape"] == pytest.approx(
        mean_absolute_percentage_error(y_true, y_pred)
    )
    assert result["r2"] == pytest.approx(
        r2_score(y_true, y_pred)
    )


# ============================================================
# Problem 2 — Classification: 5 tests
# ============================================================

def test_classification_pipeline_structure():
    X, _ = load_classification_data()
    pipe = build_classification_pipeline(X)

    assert isinstance(pipe, Pipeline)
    assert has_column_transformer(pipe)
    assert isinstance(pipe.steps[-1][1], LogisticRegression)
    assert pipe.steps[-1][1].max_iter == 1000


def test_classification_pipeline_fit_and_predict_proba():
    X, y = load_classification_data()

    # Use enough rows to ensure both classes are present.
    X_small = X.iloc[:400].copy()
    y_small = y.iloc[:400].copy()

    pipe = build_classification_pipeline(X_small)
    pipe.fit(X_small, y_small)

    y_prob = pipe.predict_proba(X_small.iloc[:20])

    assert y_prob.shape == (20, 2)
    assert np.isfinite(y_prob).all()
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))


def test_classification_pipeline_handles_unseen_inputs():
    X, y = load_classification_data()

    X_train = X.iloc[:400].copy()
    y_train = y.iloc[:400].copy()

    pipe = build_classification_pipeline(X_train)
    pipe.fit(X_train, y_train)

    X_new = X.iloc[400:402].copy()

    # Simulate legitimate unseen variations.
    X_new.loc[X_new.index[0], "plant_section"] = "New-Section"
    X_new.loc[X_new.index[1], "vibration_mm_s"] = np.nan
    X_new.loc[X_new.index[0], "machine_id"] = "M-NEW-001"

    y_prob = pipe.predict_proba(X_new)[:, 1]

    assert y_prob.shape == (2,)
    assert np.isfinite(y_prob).all()
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))


def test_classification_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.10, 0.70, 0.20, 0.80, 0.40, 0.90])

    result = classification_metrics(
        y_true, y_pred, y_prob
    )

    expected_keys = {
        "tn", "fp", "fn", "tp",
        "accuracy", "precision", "recall",
        "specificity", "f1", "roc_auc",
    }
    assert set(result.keys()) == expected_keys

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred
    ).ravel()

    assert result["tn"] == tn
    assert result["fp"] == fp
    assert result["fn"] == fn
    assert result["tp"] == tp

    assert result["accuracy"] == pytest.approx(
        accuracy_score(y_true, y_pred)
    )
    assert result["precision"] == pytest.approx(
        precision_score(y_true, y_pred)
    )
    assert result["recall"] == pytest.approx(
        recall_score(y_true, y_pred)
    )
    assert result["specificity"] == pytest.approx(
        tn / (tn + fp)
    )
    assert result["f1"] == pytest.approx(
        f1_score(y_true, y_pred)
    )
    assert result["roc_auc"] == pytest.approx(
        roc_auc_score(y_true, y_prob)
    )


def test_select_threshold():
    # Threshold 0.30:
    #   recall = 1.00, precision = 0.60
    #
    # Threshold 0.50:
    #   recall = 1.00, precision = 0.75
    #
    # Threshold 0.70:
    #   recall = 2/3, so it is infeasible.
    #
    # Therefore threshold 0.50 must be selected.

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.10, 0.40, 0.60, 0.90, 0.65, 0.80])

    result = select_threshold(
        y_true,
        y_prob,
        candidate_thresholds=[0.30, 0.50, 0.70],
        min_recall=0.85,
    )

    assert set(result.keys()) == {
        "threshold", "precision", "recall", "f1"
    }

    assert result["threshold"] == pytest.approx(0.50)

    expected_pred = (y_prob >= 0.50).astype(int)

    assert result["precision"] == pytest.approx(
        precision_score(y_true, expected_pred)
    )
    assert result["recall"] == pytest.approx(
        recall_score(y_true, expected_pred)
    )
    assert result["f1"] == pytest.approx(
        f1_score(y_true, expected_pred)
    )
