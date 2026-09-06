import numpy as np
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics  import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split

MAX_DEPTH_VALUES = [1, 2, 3, 4, 5, 6]
N_ESTIMATORS_VALUES = [10, 50, 100, 200]
SPLIT_SEED = 33

# Problem 1 - Decision Tree Classifier from Scratch 

def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    p = np.mean(y)
    return 1.0 - (p**2 + (1 - p)**2)

def best_split(X, y):
    best_gain = -1.0
    best_feat = None
    best_thresh = None
    best_col_idx = None

    n_samples = len(y)
    current_gini = gini_impurity(y)
    cols = list(X.columns)

    for col_idx, col in enumerate(cols):
        x_col = X[col].values
        unique_vals = np.unique(x_col)
        if len(unique_vals) <= 1:
            continue
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for thresh in thresholds:
            left_mask = x_col <= thresh
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            w_gini = (len(y_left) / n_samples) * gini_impurity(y_left) + (len(y_right) / n_samples) * gini_impurity(y_right)
            gain = current_gini - w_gini

            if gain > best_gain + 1e-9:
                best_gain = gain
                best_feat = col
                best_thresh = thresh
                best_col_idx = col_idx
            

    if best_gain <= 0.0 or best_feat is None:
        return None

    return {
        "feature": best_feat,
        "threshold": float(best_thresh),
        "gain": float(best_gain)
    }

def build_tree(X, y, max_depth, min_samples_split=2, depth=0):
    n_samples = len(y)
    y_arr = np.asarray(y)
    n_classes = len(np.unique(y_arr))

    def make_leaf():    
        counts = np.bincount(y)
        c0 = counts[0] if len(counts) > 0 else 0 
        c1 = counts[1] if len(counts) > 1 else 0
        pred = 1 if c1 > c0 else 0 
        return{"prediction": pred}


    if n_classes == 1 or (max_depth is not None and depth == max_depth) or n_samples < min_samples_split:
        return make_leaf()

    split = best_split(X, y)
    if split is None:
        return make_leaf()


    
    feat = split["feature"]
    thresh = split["threshold"]

    left_mask = X[feat].values <= thresh
    right_mask = ~left_mask

    left_tree = build_tree(X[left_mask], y[left_mask], max_depth, min_samples_split, depth + 1)
    right_tree = build_tree(X[right_mask], y[right_mask], max_depth, min_samples_split, depth + 1)

    return {
        "feature": feat,
        "threshold": thresh,
        "left": left_tree,
        "right": right_tree
    }

def predict_one(tree, row):
    if "prediction" in tree:
        return tree["prediction"]
    feat = tree["feature"]
    thresh = tree["threshold"]
    if row[feat] <= thresh:
        return predict_one(tree["left"], row)
    else:
        return predict_one(tree["right"], row)

def predict_tree(tree, X):
    preds = [predict_one(tree, row) for _, row in X.iterrows()]
    return np.array(preds)

def evaluate_depths(X_train, y_train, X_validation, y_validation, max_depth_values, min_samples_split=2):
    results = []
    for d in max_depth_values:
        tree = build_tree(X_train, y_train, max_depth=d, min_samples_split=min_samples_split)
        train_preds = predict_tree(tree, X_train)
        val_preds = predict_tree(tree, X_validation)

        train_acc = np.mean(train_preds == y_train.values)
        val_acc = np.mean(val_preds == y_validation.values)

        results.append({
            "max_depth": d,
            "train_accuracy": float(train_acc),
            "validation_accuracy": float(val_acc),
            "tree": tree
        })
    return results

def select_depth_behavior(results):
    def safe_depth(d):
        return float('inf') if d is None else d

    underfit = min(results, key=lambda x: (x["train_accuracy"], safe_depth(x["max_depth"])))
    overfit = max(results, key=lambda x: (x["train_accuracy"] -x["validation_accuracy"], safe_depth(x["max_depth"])))
    best = max(results, key=lambda x: (x["validation_accuracy"], -safe_depth(x["max_depth"])))
    return {
        "underfit_depth": underfit["max_depth"],
        "overfit_depth": overfit["max_depth"],
        "best_depth": best["max_depth"]
    }

# Problem 2 - Random Forest Regression

def _make_preprocessor(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
    )
    return preprocessor

def build_tree_regression_pipeline(X):
    preprocessor = _make_preprocessor(X)

    return Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('model', DecisionTreeRegressor(random_state=42))
    ])

def build_random_forest_pipeline(X, n_estimators):
    preprocessor = _make_preprocessor(X)
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=42))
    ])
       

def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }


if __name__ == "__main__":
    print(f"Running Assignment 3 Workflow with Split Seed: {SPLIT_SEED}\n")

    df_class = pd.read_csv('decision_tree_classification.csv')
    X_class = df_class.drop(columns=['target'])
    y_class = df_class['target']


    rng_class = np.random.default_rng(SPLIT_SEED)
    indices_class = rng_class.permutation(len(df_class))

    train_end = int(0.70 * len(df_class))
    val_end = int(0.85 * len(df_class))

    X_train = X_class.iloc[indices_class[:train_end]].reset_index(drop=True)
    y_train = y_class.iloc[indices_class[:train_end]].reset_index(drop=True)
    X_val = X_class.iloc[indices_class[train_end:val_end]].reset_index(drop=True)
    y_val = y_class.iloc[indices_class[train_end:val_end]].reset_index(drop=True)
    X_test = X_class.iloc[indices_class[val_end:]].reset_index(drop=True)
    y_test = y_class.iloc[indices_class[val_end:]].reset_index(drop=True)

    depth_results = evaluate_depths(X_train, y_train, X_val, y_val, MAX_DEPTH_VALUES)
    depth_behavior = select_depth_behavior(depth_results)

    print("Problem 1 Results")
    for res in depth_results:
        print(f"Depth {res['max_depth']}: Train Acc = {res['train_accuracy']:.4f}, Val Acc = {res['validation_accuracy']:.4f}")
    print(f"Depth Behavior: {depth_behavior}")


    best_d = depth_behavior["best_depth"]
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    Y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    final_tree = build_tree(X_train_val, Y_train_val, max_depth=best_d)
    final_preds = predict_tree(final_tree, X_test)
    final_test_acc = np.mean(final_preds == y_test.values)
    print(f"Final Test Accuracy(Depth {best_d}): {final_test_acc:.4f}\n")


    df_reg = pd.read_csv('random_forest_regression.csv')
    X_reg = df_reg.drop(columns=['record_id', 'maintenance_duration_hours'])
    y_reg = df_reg['maintenance_duration_hours']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=SPLIT_SEED
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)

    baseline_pipe = build_tree_regression_pipeline(X_train_r)
    cv_base = cross_validate(baseline_pipe, X_train_r, y_train_r, cv=kf, scoring='neg_mean_absolute_error')
    base_mae_mean = -np.mean(cv_base['test_score'])
    base_mae_std = np.std(cv_base['test_score'])

    print('Problem 2 Results')
    print(f"Single Tree Baselne CV MAE: {base_mae_mean:.4f}± {base_mae_std:.4f}")

    rf_cv_results = []
    for n_est in N_ESTIMATORS_VALUES:
        rf_pipe = build_random_forest_pipeline(X_train_r, n_est)
        cv_rf = cross_validate(rf_pipe, X_train_r, y_train_r, cv=kf, scoring='neg_mean_absolute_error')
        mean_mae = -np.mean(cv_rf['test_score'])
        std_mae = np.std(cv_rf['test_score'])
        rf_cv_results.append((n_est, mean_mae, std_mae))
        print(f"Random Forest (n={n_est}) CV MAE: {mean_mae:.4f} ± {std_mae:.4f}")

    best_rf_config = min(rf_cv_results , key=lambda x: (x[1], x[0]))
    selected_n_estimators = best_rf_config[0]
    print(f"Selectted n_estimators: {selected_n_estimators}")

    baseline_pipe.fit(X_train_r, y_train_r)
    base_test_metrics = regression_metrics(y_test_r, baseline_pipe.predict(X_test_r))


    best_rf_pipe = build_random_forest_pipeline(X_train_r, selected_n_estimators)
    best_rf_pipe.fit(X_train_r, y_train_r)
    rf_test_metrics = regression_metrics(y_test_r, best_rf_pipe.predict(X_test_r))

    print(f"Single Tree Test Metrics: {base_test_metrics}")
    print(f"Selected Random Forest Test Metrics: {rf_test_metrics}")