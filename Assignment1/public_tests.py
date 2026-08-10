"""Public tests for DA5401 Assignment 1.

Place this file in the same folder as assignment_submission.py and run:

    pytest -q public_tests.py
"""

import numpy as np
import pandas as pd
import pytest

import assignment_submission as sub


REQUIRED_FUNCTIONS = [
    "linear_predict",
    "linear_loss",
    "linear_gradients",
    "fit_linear_regression",
    "predict_linear_regression",
    "fit_preprocessor",
    "transform_features",
    "sigmoid",
    "logistic_loss",
    "logistic_gradients",
    "fit_logistic_regression",
    "predict_proba_logistic",
    "predict_logistic",
]


def test_required_functions_exist():
    for name in REQUIRED_FUNCTIONS:
        assert hasattr(sub, name), f"Missing required function: {name}"
        assert callable(getattr(sub, name)), f"{name} must be callable"


# -----------------------------------------------------------------------------
# Linear regression tests
# -----------------------------------------------------------------------------


def test_linear_predict_values_and_shape():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0]])
    weights = np.array([2.0, -1.0])
    bias = 0.5

    result = sub.linear_predict(X, weights, bias)
    expected = np.array([0.5, 2.5, -6.5])

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_linear_loss_is_mean_squared_error():
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([0.0, 2.0, 5.0])

    result = sub.linear_loss(y_true, y_pred)
    expected = 2.0 / 3.0

    assert isinstance(result, (float, np.floating))
    assert np.isfinite(result)
    assert result == pytest.approx(expected)


def test_linear_gradients():
    X = np.array([[1.0, 0.0], [0.0, 2.0]])
    y = np.array([1.0, 2.0])
    weights = np.zeros(2)
    bias = 0.0

    dw, db = sub.linear_gradients(X, y, weights, bias)

    expected_dw = np.array([-1.0, -4.0])
    expected_db = -3.0

    assert isinstance(dw, np.ndarray)
    assert dw.shape == (2,)
    np.testing.assert_allclose(dw, expected_dw, rtol=1e-10, atol=1e-10)
    assert db == pytest.approx(expected_db)


def test_fit_linear_regression_returns_required_model():
    X = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [1.0, 0.0, 1.0, 0.0],
        }
    )
    y = pd.Series([1.0, 3.0, 5.0, 7.0])

    model = sub.fit_linear_regression(
        X,
        y,
        learning_rate=0.05,
        max_iter=3000,
        tol=1e-12,
    )

    assert isinstance(model, dict)
    for key in ("weights", "bias", "loss_history"):
        assert key in model

    assert isinstance(model["weights"], np.ndarray)
    assert model["weights"].shape == (2,)
    assert isinstance(model["bias"], (float, np.floating))
    assert isinstance(model["loss_history"], list)
    assert 1 <= len(model["loss_history"]) <= 3000
    assert np.all(np.isfinite(model["loss_history"]))

    predictions = sub.predict_linear_regression(model, X)
    assert predictions.shape == (4,)
    assert np.mean((predictions - y.to_numpy()) ** 2) < 1e-3


def test_linear_model_is_not_dataset_specific():
    X = np.array(
        [
            [1.0, 2.0, -1.0],
            [2.0, 0.0, 1.0],
            [3.0, 1.0, 2.0],
            [4.0, -1.0, 0.0],
            [5.0, 2.0, 1.0],
        ]
    )
    true_weights = np.array([1.5, -2.0, 0.5])
    true_bias = -0.75
    y = X @ true_weights + true_bias

    model = sub.fit_linear_regression(
        X,
        y,
        learning_rate=0.01,
        max_iter=10000,
        tol=1e-12,
    )
    predictions = sub.predict_linear_regression(model, X)

    assert predictions.shape == (5,)
    assert np.mean((predictions - y) ** 2) < 1e-4


# -----------------------------------------------------------------------------
# Logistic preprocessing tests
# -----------------------------------------------------------------------------


def test_preprocessor_identifies_and_orders_features():
    X = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Mumbai", "Chennai", "Delhi"],
            "income": [10.0, 20.0, 30.0],
        }
    )

    pre = sub.fit_preprocessor(X)

    assert pre["original_columns"] == ["age", "city", "income"]
    assert pre["numerical_columns"] == ["age", "income"]
    assert pre["categorical_columns"] == ["city"]
    assert pre["categories"]["city"] == ["Chennai", "Delhi", "Mumbai"]
    assert pre["encoded_feature_names"] == [
        "age",
        "city=Chennai",
        "city=Delhi",
        "city=Mumbai",
        "income",
    ]


def test_transform_features_exact_encoding_and_order():
    X_train = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Mumbai", "Chennai", "Delhi"],
            "income": [10.0, 20.0, 30.0],
        }
    )
    pre = sub.fit_preprocessor(X_train)

    X_new = pd.DataFrame(
        {
            "income": [50.0, 60.0],
            "city": ["Delhi", "Mumbai"],
            "age": [25, 35],
        }
    )

    encoded = sub.transform_features(X_new, pre)

    expected = np.array(
        [
            [25.0, 0.0, 1.0, 0.0, 50.0],
            [35.0, 0.0, 0.0, 1.0, 60.0],
        ]
    )

    assert isinstance(encoded, np.ndarray)
    assert encoded.dtype.kind == "f"
    assert encoded.shape == (2, 5)
    np.testing.assert_allclose(encoded, expected)


# -----------------------------------------------------------------------------
# Logistic regression tests
# -----------------------------------------------------------------------------


def test_sigmoid_values_shape_and_stability():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    result = sub.sigmoid(z)

    assert isinstance(result, np.ndarray)
    assert result.shape == z.shape
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert result[0] < 1e-12
    assert result[2] == pytest.approx(0.5)
    assert result[-1] > 1.0 - 1e-12


def test_logistic_loss_with_clipping():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    p = np.array([0.0, 1.0, 0.8, 0.2])

    result = sub.logistic_loss(y, p, epsilon=1e-12)

    p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
    expected = -np.mean(
        y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe)
    )

    assert isinstance(result, (float, np.floating))
    assert np.isfinite(result)
    assert result == pytest.approx(expected)


def test_logistic_gradients():
    X = np.array([[1.0, 0.0], [0.0, 2.0]])
    y = np.array([1.0, 0.0])
    weights = np.zeros(2)
    bias = 0.0

    dw, db = sub.logistic_gradients(X, y, weights, bias)

    expected_dw = np.array([-0.25, 0.5])
    expected_db = 0.0

    assert isinstance(dw, np.ndarray)
    assert dw.shape == (2,)
    np.testing.assert_allclose(dw, expected_dw, rtol=1e-10, atol=1e-10)
    assert db == pytest.approx(expected_db)


def test_fit_logistic_regression_with_categorical_feature():
    X = pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "group": ["A", "A", "B", "B", "C", "C"],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = sub.fit_logistic_regression(
        X,
        y,
        learning_rate=0.1,
        max_iter=5000,
        tol=1e-10,
        epsilon=1e-12,
    )

    assert isinstance(model, dict)
    for key in ("weights", "bias", "preprocessor", "loss_history"):
        assert key in model

    assert isinstance(model["weights"], np.ndarray)
    assert isinstance(model["bias"], (float, np.floating))
    assert isinstance(model["preprocessor"], dict)
    assert isinstance(model["loss_history"], list)
    assert np.all(np.isfinite(model["loss_history"]))

    probabilities = sub.predict_proba_logistic(model, X)
    predictions = sub.predict_logistic(model, X)

    assert probabilities.shape == (6,)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert predictions.shape == (6,)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert np.mean(predictions == y.to_numpy()) >= 5.0 / 6.0


def test_predict_logistic_uses_threshold_rule():
    X = pd.DataFrame(
        {
            "x": [-2.0, 0.0, 2.0],
            "type": ["A", "A", "B"],
        }
    )
    y = np.array([0, 0, 1])

    model = sub.fit_logistic_regression(
        X,
        y,
        learning_rate=0.1,
        max_iter=3000,
        tol=1e-10,
    )
    probabilities = sub.predict_proba_logistic(model, X)
    predictions = sub.predict_logistic(model, X, threshold=0.7)

    expected = (probabilities >= 0.7).astype(int)
    np.testing.assert_array_equal(predictions, expected)


def test_invalid_threshold_raises_value_error():
    X = pd.DataFrame({"x": [-1.0, 1.0]})
    y = np.array([0, 1])
    model = sub.fit_logistic_regression(X, y, max_iter=10)

    with pytest.raises(ValueError):
        sub.predict_logistic(model, X, threshold=-0.1)

    with pytest.raises(ValueError):
        sub.predict_logistic(model, X, threshold=1.1)
