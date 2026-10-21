

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

import optuna

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest


# ============================================================
# Problem 1 — Density-Based Clustering
# ============================================================

def cluster_summary(X, labels):
    labels = np.asarray(labels)
    X = np.asarray(X)
    n  = len(labels)
    mask = labels != -1
    non_noise = labels[mask]
    unique, counts = np.unique(non_noise, return_counts=True)
    n_clusters = int(len(unique))
    cluster_sizes = sorted(int(c) for c in counts)
    noise_fraction = float((~mask).sum() / n) if n > 0 else float("nan")

    silhouette = np.nan
    if n_clusters >= 2:
        silhouette = float(silhouette_score(X[mask], non_noise))

    return{
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "cluster_sizes": cluster_sizes,
        "silhouette": silhouette,
    }


def run_dbscan(X, eps, min_samples):
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)
    s = cluster_summary(X, labels)
    return {
        "eps": eps,
        "min_samples": min_samples,
        "labels": labels,
        "n_clusters": s["n_clusters"],
        "noise_fraction": s["noise_fraction"],
        "cluster_sizes": s["cluster_sizes"],
        "silhouette": s["silhouette"]
    }

 


def evaluate_dbscan_configs(X, configs):
  
    return[run_dbscan(X, cfg["eps"],cfg["min_samples"]) for cfg in configs]
  
 


def run_hdbscan(X, min_cluster_size, min_samples=None):
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        copy=False,
    ).fit_predict(X)
    s = cluster_summary(X, labels)
    return{
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "labels": labels,
        "n_clusters": s["n_clusters"],
        "noise_fraction": s["noise_fraction"],
        "cluster_sizes": s["cluster_sizes"],
        "silhouette": s["silhouette"]
    }
    
 


def evaluate_hdbscan_configs(X, min_cluster_sizes):
    return[run_hdbscan(X, m, min_samples=None) for m in min_cluster_sizes]
 


def select_density_result(results):
    best = None
    for r in results:
        sil = r.get("silhouette", np.nan)
        if r.get("n_clusters", 0) < 2 or sil is None or not np.isfinite(sil):
            continue
        if best is None:
            best = r
            continue
        if np.isclose(sil, best["silhouette"]):
            if (r["noise_fraction"] < best["noise_fraction"] and not np.isclose(r["noise_fraction"], best["noise_fraction"])):
                best = r
        elif sil > best["silhouette"]:
            best = r
    return best


# ============================================================
# Problem 2 — Random Forest Hyperparameter Tuning
# ============================================================

def build_random_forest(**params):
    kwargs = {"random_state": 42, "class_weight": "balanced","n_jobs": -1}
    kwargs.update(params)
    return RandomForestClassifier(**kwargs)

def _search_summary(search):
    return {
        "best_params": dict(search.best_params_),
        "best_cv_score": float(search.best_score_),
        "n_evaluated": int(len(search.cv_results_["params"])),

    }
 


def run_grid_search(X, y, param_grid, cv):
    search = GridSearchCV(
        estimator=build_random_forest(),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
    )
    search.fit(X, y)
    return _search_summary(search)


def run_random_search(X, y, param_distributions, n_iter, cv, seed):
    search = RandomizedSearchCV(
        estimator=build_random_forest(),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
    )
    search.fit(X, y)
    return _search_summary(search)


def run_bayesian_search(X, y, n_trials, cv, seed):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 220),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, 0.8]
            ),
        }
        model = build_random_forest(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    n_done = sum(
        t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
    )

    return {
        "best_params": dict(study.best_params),
        "best_cv_score": float(study.best_value),
        "n_evaluated": int(n_done),
    }


def classification_scores(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, (y_prob >= 0.5).astype(int))),
    }
 


# ============================================================
# Problem 3 — Unsupervised Anomaly Detection with PyOD
# ============================================================

def build_pyod_models(contamination, n_neighbors):
   return {
       "KNN": KNN(contamination=contamination, n_neighbors=n_neighbors),
       "LOF": LOF(contamination=contamination, n_neighbors=n_neighbors),
       "IForest": IForest(
           contamination=contamination, n_estimators=200, random_state=42
       ),
   }


def fit_pyod_scores(model, X):
    model.fit(np.asarray(X))
    return np.asarray(model.decision_scores_, dtype=float)

 


def flag_top_fraction(scores, fraction):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(np.ceil(fraction * n))
    k = max(0, min(k, n))
    order = np.argsort(-scores, kind="stable")
    flags = np.zeros(n, dtype=int)
    flags[order[:k]] = 1
    return flags
 


def compare_pyod_models(X, contamination, n_neighbors, flag_fraction):
    models = build_pyod_models(contamination, n_neighbors)
    out = {}
    for name in ["KNN", "LOF", "IForest"]:
        scores = fit_pyod_scores(models[name], X)
        out[name] = {
            "scores": scores,
            "flags": flag_top_fraction(scores, flag_fraction),
        }
    return out

# ============================================================
# Problem 4 — Semi-Supervised Anomaly Detection
# ============================================================

def prepare_semisupervised_sets(
    df,
    feature_columns,
    set_column ="set_type",
    label_column="label",
):
    feature_columns = list(feature_columns)
    labeled = df[df[set_column] == "labeled"]
    unlabeled = df[df[set_column] == "unlabeled"]
    evaluation = df[df[set_column] == "evaluation"]

    X_labeled = labeled[feature_columns]
    y_labeled = labeled[label_column].to_numpy().astype(int)
    X_unlabeled = unlabeled[feature_columns]
    X_evaluation = evaluation[feature_columns]
    y_evaluation = evaluation[label_column].to_numpy().astype(int)
    return X_labeled, y_labeled, X_unlabeled, X_evaluation, y_evaluation
 


def build_semisupervised_rf():
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
 

def _class1_proba(model, X):
    proba = model.predict_proba(X)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X))
    else:
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    
def select_pseudo_labels(model,X_unlabeled,low_threshold,high_threshold):
    p = _class1_proba(model, X_unlabeled)
    is_normal = p <= low_threshold
    is_anomaly = (p >= high_threshold) & ~is_normal
    selected = is_normal | is_anomaly

    selected_indices = np.flatnonzero(selected)
    pseudo_labels = is_anomaly[selected_indices].astype(int)
    return {
        "selected_indices": selected_indices,
        "pseudo_labels": pseudo_labels,
        "n_pseudo_normal": int((pseudo_labels == 0).sum()),
        "n_pseudo_anomaly": int((pseudo_labels == 1).sum()),
    }


def retrain_with_pseudo_labels(
    X_labeled,
    y_labeled,
    X_unlabeled,
    selected_indices,
    pseudo_labels,
):
    X_labeled = np.asarray(X_labeled)
    y_labeled = np.asarray(y_labeled)
    idx = np.asarray(selected_indices, dtype=int)
    if len(idx) > 0:
        X_new = np.asarray(X_unlabeled)[idx]
        X_train = np.vstack([X_labeled, X_new])
        y_train = np.concatenate([y_labeled, np.asarray(pseudo_labels).astype(int)])
    else:
        X_train, y_train = X_labeled, y_labeled
    model = build_semisupervised_rf()
    model.fit(X_train, y_train)
    return model

 


def evaluate_semisupervised_models(
    initial_model,
    retrained_model,
    X_evaluation,
    y_evaluation,
):
    init = classification_scores(y_evaluation, _class1_proba(initial_model, X_evaluation))
    retr = classification_scores(y_evaluation, _class1_proba(retrained_model, X_evaluation))

    return {
        "initial_roc_auc": init["roc_auc"],
        "initial_f1": init["f1"],
        "retrained_roc_auc": retr["roc_auc"],
        "retrained_f1": retr["f1"],
    }
if __name__ == "__main__":
    import os
    from math import ceil  # noqa: F401
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
 
    student_seed = 33
    d = os.path.dirname(os.path.abspath(__file__))
 
    # ---- Problem 1 ----
    X1 = StandardScaler().fit_transform(
        pd.read_csv(os.path.join(d, "assignment6_density_clustering.csv")).values
    )
    db = evaluate_dbscan_configs(X1, [
        {"eps": 0.12, "min_samples": 5}, {"eps": 0.18, "min_samples": 5},
        {"eps": 0.24, "min_samples": 8}, {"eps": 0.30, "min_samples": 10}])
    hd = evaluate_hdbscan_configs(X1, [20, 40, 70, 100])
    print("DBSCAN selected:", {k: v for k, v in select_density_result(db).items() if k != "labels"})
    print("HDBSCAN selected:", {k: v for k, v in select_density_result(hd).items() if k != "labels"})
 
    # ---- Problem 2 ----
    df2 = pd.read_csv(os.path.join(d, "assignment6_rf_tuning.csv"))
    X2, y2 = df2.drop(columns=["target"]), df2["target"]
    Xtr, Xte, ytr, yte = train_test_split(
        X2, y2, test_size=0.2, stratify=y2, random_state=student_seed)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=student_seed)
    grid = {"n_estimators": [80, 150], "max_depth": [None, 8],
            "min_samples_split": [2, 8], "max_features": ["sqrt"]}
    space = {"n_estimators": [60, 80, 100, 130, 160, 200],
             "max_depth": [None, 5, 8, 12], "min_samples_split": [2, 4, 8, 12],
             "max_features": ["sqrt", "log2", 0.6]}
    searches = {
        "Grid": run_grid_search(Xtr, ytr, grid, cv),
        "Random": run_random_search(Xtr, ytr, space, 8, cv, student_seed),
        "Bayesian": run_bayesian_search(Xtr, ytr, 8, cv, student_seed),
    }
    for name, res in searches.items():
        rf = build_random_forest(**res["best_params"]).fit(Xtr, ytr)
        print(name, res, classification_scores(yte, rf.predict_proba(Xte)[:, 1]))
 
    # ---- Problem 3 ----
    df3 = pd.read_csv(os.path.join(d, "assignment6_anomaly_unlabeled.csv"))
    feats3 = [c for c in df3.columns if c != "record_id"]
    X3 = StandardScaler().fit_transform(df3[feats3].values)
    cmp_ = compare_pyod_models(X3, 0.10, 20, 0.10)
    for name, r in cmp_.items():
        top = np.argsort(-r["scores"], kind="stable")[:10]
        print(name, "flagged:", int(r["flags"].sum()),
              "top10:", df3["record_id"].iloc[top].tolist())
    names = list(cmp_)
    for i in range(3):
        for j in range(i + 1, 3):
            print(names[i], names[j],
                  int((cmp_[names[i]]["flags"] & cmp_[names[j]]["flags"]).sum()))
 
    # ---- Problem 4 ----
    df4 = pd.read_csv(os.path.join(d, "assignment6_anomaly_semisupervised.csv"))
    feats4 = [c for c in df4.columns if c not in ("record_id", "set_type", "label")]
    Xl, yl, Xu, Xe, ye = prepare_semisupervised_sets(df4, feats4, "set_type", "label")
    m0 = build_semisupervised_rf().fit(Xl, yl)
    ps = select_pseudo_labels(m0, Xu, 0.10, 0.90)
    m1 = retrain_with_pseudo_labels(Xl, yl, Xu, ps["selected_indices"], ps["pseudo_labels"])
    print(ps["n_pseudo_normal"], ps["n_pseudo_anomaly"])
    print(evaluate_semisupervised_models(m0, m1, Xe, ye))
 
