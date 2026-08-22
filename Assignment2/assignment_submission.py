
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


## Problem 1 - Regression

def build_regression_pipeline(X: pd.DataFrame):
    drop_cols = ["job_id"]
    feature_cols = [col for col  in X.columns if col not in drop_cols]


    numerical_cols = X[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X[feature_cols].select_dtypes(include=[object,"category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
        
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    return pipeline

def regression_metrics(y_true, y_pred):
    return {
        "mae" : mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# Problem 2 - Classification

def build_classification_pipeline(X: pd.DataFrame):

    drop_cols = ["machine_id"]
    feature_cols = [col for col in X.columns if col not in drop_cols]

    numerical_cols = X[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X[feature_cols].select_dtypes(include=[object, "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
    
        ],
        remainder="drop"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return pipeline

def classification_metrics(y_true, y_pred, y_prob):
   tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
   specificity = tn /(tn + fp) if (tn + fp) > 0 else 0.0

   return {
       "tn": int(tn),
       "fp": int(fp),
       "fn": int(fn),
       "tp": int(tp),
       "accuracy": accuracy_score(y_true, y_pred),
       "precision": precision_score(y_true, y_pred, zero_division=0),
       "recall": recall_score(y_true, y_pred, zero_division=0),
       "specificity" : specificity,
       "f1" : f1_score(y_true, y_pred, zero_division=0),
       "roc_auc": roc_auc_score(y_true, y_prob),
   }


def select_threshold(
    y_true,
    y_prob,
    candidate_thresholds,
    min_recall=0.85
):

    best_threshold = None
    best_precision = -1.0
    best_recall = 0.0
    best_f1 = 0.0


    for t in sorted(candidate_thresholds, reverse=True):
        y_pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if rec >= min_recall:

            if prec > best_precision:
                best_precision = prec
                best_threshold = t
                best_recall = rec
                best_f1 = f1
            elif prec == best_precision and best_threshold is not None and t > best_threshold:
                best_threshold = t
                best_recall = rec
                best_f1 = f1
            elif best_threshold is None:
                best_threshold = t
                best_precision = prec
                best_recall = rec
                best_f1 = f1

    if best_threshold is None:
        best_threshold = candidate_thresholds[0]
        y_pred = (y_prob >= best_threshold).astype(int)
        best_precision = precision_score(y_true, y_pred, zero_division=0)
        best_recall = recall_score(y_true, y_pred, zero_division=0)
        best_f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
    "threshold": float(best_threshold),
    "precision": float(best_precision),
    "recall": float(best_recall),
    "f1": float(best_f1),
    }

if __name__=="__main__":
    
    print("Problem 1: Regression")
    df = pd.read_csv("regression_data.csv")
    X = df.drop(columns=["job_cost"])
    y = df["job_cost"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_regression_pipeline(X_train)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = -cross_val_score(
        pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
    )
    print("CV RMSE mean:", rmse_scores.mean())
    print("CV RMSE sttd:", rmse_scores.std())

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = regression_metrics(y_test, y_pred)
    print("Test metrics:", metrics)

    print()

    print("Problem 2: Classification")
    df2 = pd.read_csv("classification_data.csv")
    X2 = df2.drop(columns=["critical_failure_30d"])
    y2 = df2["critical_failure_30d"]
    X_train2, X_tmp, y_train2, y_tmp = train_test_split(
        X2, y2, test_size=0.30, random_state=42, stratify=y2
    )
    X_val, X_test2, y_val, y_test2 = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    pipe2 = build_classification_pipeline(X_train2)
    pipe2.fit(X_train2, y_train2)

    y_val_prob = pipe2.predict_proba(X_val)[:,1]


    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    result = select_threshold(y_val, y_val_prob, candidate_thresholds, min_recall=0.85)
    print("Selected threshold info:", result)

    X_trainval = pd.concat([X_train2, X_val])
    y_trainval = pd.concat([y_train2, y_val])
    final_pipe = build_classification_pipeline(X_trainval)
    final_pipe.fit(X_trainval, y_trainval)

    Y_test_prob = final_pipe.predict_proba(X_test2)[:, 1]
    y_test_pred = (Y_test_prob >= result["threshold"]).astype(int)
    final_metrics = classification_metrics(y_test2, y_test_pred, Y_test_prob)
    print("Final test metrics:", final_metrics)