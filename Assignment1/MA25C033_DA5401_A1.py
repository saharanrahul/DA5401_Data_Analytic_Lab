import numpy as np
import pandas as pd

def linear_predict(X, weights, bias):
    X_arr = np.asarray(X, dtype=float)
    predictions = X_arr @ weights + bias
    return predictions.ravel()
    

def linear_loss(y_true, y_pred):
    squared_errors = (y_pred - y_true) ** 2
    return float(np.mean(squared_errors))

def linear_gradients(X, y, weights, bias):
    n = X.shape[0]
    y_pred = linear_predict(X, weights, bias)
    error = y_pred - y

    
    dw = (2 / n) * (X.T @ error)
    db = float((2 / n) * np.sum(error))

    return dw, db

def fit_linear_regression(X, y, learning_rate=0.01, max_iter=1000, tol=1e-8):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if tol < 0:
        raise ValueError("tol cannot be negative")

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()

    if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("Data contains NaN or infinite values")

    if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X contains zero rows or zero columns")
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y contain different number of observations")

    n_features = X_arr.shape[1]
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    loss_history = []
    prev_loss = None

    for _ in range(max_iter):
        y_pred = linear_predict(X_arr, weights, bias)
        current_loss = linear_loss(y_arr, y_pred)
        loss_history.append(current_loss)
        if prev_loss is not None and abs(prev_loss - current_loss) <= tol:
            break

        prev_loss = current_loss
        dw, db = linear_gradients(X_arr, y_arr, weights, bias)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return{
         "weights": weights,
         "bias": bias,
        "loss_history": loss_history
        }
    
def predict_linear_regression(model, X): 
     weights = model["weights"]
     bias = model["bias"]
     return linear_predict(X, weights, bias)

def fit_preprocessor(X):
    original_columns = list(X.columns)
    numerical_columns = []
    categorical_columns = []
    categories = {}
    encoded_feature_names = []

    for col in original_columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numerical_columns.append(col)
            encoded_feature_names.append(col)
        else:
            categorical_columns.append(col)
            unique_cats = sorted([str(val)for val in X[col].dropna().unique()])
            categories[col] = unique_cats
            for cat in unique_cats:
                encoded_feature_names.append(f"{col}={cat}")
    return{
        "original_columns": original_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "categories": categories,
        "encoded_feature_names": encoded_feature_names
        }

def transform_features(X, preprocessor):
    missiong_columns = set(preprocessor["original_columns"]) - set(X.columns)
    if missiong_columns:
        raise ValueError("Prediction dataset is missing required original columns")

    n_samples = len(X)
    n_encoded_features = len(preprocessor["encoded_feature_names"])

    X_encoded = np.zeros((n_samples, n_encoded_features), dtype=float)
    feature_to_index = {name: idx for idx, name in enumerate(preprocessor["encoded_feature_names"])}

    for col in preprocessor["original_columns"]:
        if col in preprocessor["numerical_columns"]:
            col_index = feature_to_index[col]
            X_encoded[:, col_index] = np.asarray(X[col], dtype=float)
        else:
            col_data = X[col].astype(str).values
            for cat in preprocessor["categories"][col]:
                encoded_name = f"{col}={cat}"
                if encoded_name in feature_to_index:
                    col_index = feature_to_index[encoded_name]
                    X_encoded[:, col_index] = (col_data == cat).astype(float)
                    
    return X_encoded

def sigmoid(z):
    Z_arr = np.asarray(z, dtype=float)
    out = np.zeros_like(Z_arr)

    pos_mask = Z_arr >= 0
    neg_mask = ~pos_mask

    out[pos_mask] = 1.0 / (1.0 + np.exp(-Z_arr[pos_mask]))
    exp_neg = np.exp(Z_arr[neg_mask])
    out[neg_mask] = exp_neg / (1.0 + exp_neg)
    
    return out

def logistic_loss(y_true, probabilities, epsilon=1e-12):
    if not (0 < epsilon < 0.5):
        raise ValueError("epsilon must be strictly between 0 and 0.5")

    p_clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    bce = -np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))

    return float(bce)

def logistic_gradients(X_encoded, y, weights, bias):
    n = X_encoded.shape[0]
    z = X_encoded @ weights + bias
    p = sigmoid(z)
    error = p - y

    dw = (1.0 / n) * (X_encoded.T @ error)
    db = float((1.0 / n) * np.sum(error))

    return dw, db

def fit_logistic_regression(
    X,
    y,
    learning_rate=0.01,
    max_iter=1000,
    tol=1e-8,
    epsilon=1e-12,
):
    
    if learning_rate <=0:
        raise ValueError("learning_rate must be positive")
    if not isinstance(max_iter, int) or max_iter <=0:
        raise ValueError("max_iter must be a positive integer")
    if tol < 0:
        raise ValueError("tol cannot be negative")
    y_arr = np.asarray(y, dtype=float).ravel()
    if len(X) != len(y_arr):
        raise ValueError("X and y contain different numbers of observations")
    if not set(np.unique(y_arr)).issubset({0.0, 1.0}):
        raise ValueError("Logistic regression targets contain values other than 0 and 1")
    preprocessor = fit_preprocessor(X)
    X_encoded = transform_features(X, preprocessor)
    if not np.all(np.isfinite(X_encoded)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("Data contains NaN or infinite values")
    if X_encoded.shape[0] == 0 or X_encoded.shape[1] == 0:
        raise ValueError("X contains zero rows or zero columns")


    n_features = X_encoded.shape[1]
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0

    loss_history = []
    prev_loss = None

    for _ in range(max_iter):
        z = X_encoded @ weights + bias
        p = sigmoid(z)

        current_loss = logistic_loss(y_arr, p, epsilon)
        loss_history.append(current_loss)
        if prev_loss is not None and abs(prev_loss - current_loss)  <= tol:
            break
        prev_loss = current_loss
        dw, db = logistic_gradients(X_encoded, y_arr, weights, bias)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return {
        "weights": weights,
        "bias": bias,
        "preprocessor": preprocessor,
        "loss_history": loss_history,
        }

def predict_proba_logistic(model, X):
    X_encoded = transform_features(X, model["preprocessor"])
    z = X_encoded @ model["weights"] + model["bias"]
    return sigmoid(z).ravel()


def predict_logistic(model, X, threshold=0.5):
    if not ( 0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1 inclusive")
    probs = predict_proba_logistic(model, X)
    return (probs >= threshold).astype(int)