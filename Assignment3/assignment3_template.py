"""
DA5401 — Data Analytics Lab
Assignment 3 — Decision Trees and Random Forests

Submit this file as: assignment_submission.py

Notes
-----
Problem 1:
    Implement the decision-tree logic from scratch using only NumPy,
    pandas, and the Python standard library.

Problem 2:
    scikit-learn may be used as specified in the assignment.

Keep this file importable. Put any code that runs the complete
assignment workflow inside the `if __name__ == "__main__":` block.
"""

import numpy as np
import pandas as pd

# scikit-learn imports may be used for Problem 2.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression  # Remove if unused.
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


MAX_DEPTH_VALUES = [1, 2, 3, 4, 5, 6]
N_ESTIMATORS_VALUES = [10, 50, 100, 200]


# ============================================================
# Problem 1 — Decision Tree Classifier from Scratch
# ============================================================

def gini_impurity(y):
    """
    Return the Gini impurity of binary labels y.

    Requirements:
    - Return 0.0 for empty input.
    - Return 0.0 for a pure node.
    """


def best_split(X, y):
    """
    Search all features and all valid midpoint thresholds.

    Return:
        {
            "feature": <column name>,
            "threshold": <float>,
            "gain": <float>
        }

    Return None if no valid split with positive gain exists.

    Follow the tie-breaking rules given in the assignment.
    """


def build_tree(
    X,
    y,
    max_depth,
    min_samples_split=2,
    depth=0
):
    """
    Recursively build and return a binary classification tree.

    Internal node:
        {
            "feature": ...,
            "threshold": ...,
            "left": ...,
            "right": ...
        }

    Leaf:
        {
            "prediction": 0
        }
    or
        {
            "prediction": 1
        }
    """


def predict_one(tree, row):
    """
    Predict the class of one observation by traversing the tree.
    """


def predict_tree(tree, X):
    """
    Return a NumPy array containing one prediction for every row of X.
    """

def evaluate_depths(
    X_train,
    y_train,
    X_validation,
    y_validation,
    max_depth_values,
    min_samples_split=2
):
    """
    Train one tree for every value in max_depth_values.

    Return a list of dictionaries. Each dictionary must contain:
        {
            "max_depth": ...,
            "train_accuracy": ...,
            "validation_accuracy": ...,
            "tree": ...
        }

    Calculate accuracy using your own predictions.
    """


def select_depth_behavior(results):
    """
    Return:
        {
            "underfit_depth": ...,
            "overfit_depth": ...,
            "best_depth": ...
        }

    Use the definitions and tie-breaking rules in the assignment.
    """


# ============================================================
# Problem 2 — Random Forest Regression
# ============================================================

def build_tree_regression_pipeline(X):
    """
    Return an unfitted scikit-learn Pipeline that:
    1. preprocesses the raw predictors appropriately using a
       ColumnTransformer, and
    2. ends in DecisionTreeRegressor(random_state=42).

    The baseline tree should not be tuned.
    """


def build_random_forest_pipeline(X, n_estimators):
    """
    Return an unfitted scikit-learn Pipeline that:
    1. preprocesses the raw predictors appropriately using a
       ColumnTransformer, and
    2. ends in RandomForestRegressor(
           n_estimators=n_estimators,
           random_state=42
       ).

    Keep other RandomForestRegressor parameters at sklearn defaults
    unless the assignment explicitly states otherwise.
    """


def regression_metrics(y_true, y_pred):
    """
    Return exactly:
        {
            "mae": ...,
            "rmse": ...,
            "r2": ...
        }
    """

