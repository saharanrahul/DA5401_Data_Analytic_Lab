import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster


# ============================================================
# Shared utility
# ============================================================

def student_subset(df, n_rows, seed):
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=n_rows, replace=False)
    return df.iloc[indices].reset_index(drop=True)

 


# ============================================================
# Problem 1 — K-Means from scratch
# ============================================================

def initialize_centroids(X, k, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()
 


def assign_clusters(X, centroids):

    distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :,:]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels 


def update_centroids(X, labels, previous_centroids):
    k = len(previous_centroids)
    new_centroids = np.zeros_like(previous_centroids, dtype=float)
    for i in range(k):
        members = X[labels == i]
        if len(members) == 0:
            new_centroids[i] = previous_centroids[i]
        else:
            new_centroids[i] = np.mean(members, axis=0)
    return new_centroids

 
def sum_squared_distances(X, labels, centroids):
    ssd = 0.0
    for i in range(len(centroids)):
        members = X[labels == i]
        if len(members) > 0:
            ssd += np.sum((members - centroids[i]) ** 2)
    return float(ssd)


def kmeans_from_scratch(X, k, seed, max_iter=100, tol=1e-4):
    centroids = initialize_centroids(X, k, seed)
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, centroids)

        centroid_movements = np.linalg.norm(new_centroids - centroids, axis=1)
        max_movement = np.max(centroid_movements)

        centroids = new_centroids
        if max_movement < tol:
            break

    labels = assign_clusters(X, centroids)
    ssd = sum_squared_distances(X, labels, centroids)


    return{
        "labels": labels,
        "centroids": centroids,
        "sum_squared_distances": ssd,
        "n_iter": n_iter
    }
 


def evaluate_k_values(X, k_values, seed):
    results = []
    for k in k_values:
        res = kmeans_from_scratch(X, k, seed)
        if 2 <= len(np.unique(res["labels"])) <= len(X) - 1:
            sil = silhouette_score(X, res["labels"])
        else:
            sil = 0.0
        results.append({
            "k":k,
            "sum_squared_distances": float(res["sum_squared_distances"]),
            "silhouette": float(sil)
        })
    return results


# ============================================================
# Problem 2 — Hierarchical clustering using SciPy
# ============================================================

def hierarchical_clustering(X, linkage_method, n_clusters):
    linkage_matrix = linkage(X, method=linkage_method, metric="euclidean")
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    return linkage_matrix, labels
 


def compare_linkages(X, linkage_methods, n_clusters):
    results = []
    for method in linkage_methods:
        _, labels = hierarchical_clustering(X, method, n_clusters)
        sil = silhouette_score(X, labels, metric="euclidean")
        _,counts = np.unique(labels, return_counts=True)
        cluster_sizes = sorted([int(c) for c in counts])
        results.append({
            "linkage": method,
            "silhouette": float(sil),
            "cluster_sizes": cluster_sizes
        })
    return results


def select_best_linkage(results):
    tie_break_order = {"single": 0, "complete":1, "average":2}
    top = max(r["silhouette"] for r in results)
    tied = [r for r in results if top -r["silhouette"] < 1e-9]
    return min(tied, key=lambda r: tie_break_order.get(r["linkage"], len(tie_break_order)))["linkage"]


if __name__ == "__main__":
    student_seed = 33

    kmeans_df = pd.read_csv("assignment5_kmeans_operating_regimes.csv")
    sub1 = student_subset(kmeans_df, 1200, student_seed)
    scaler1 = StandardScaler()
    X1 = scaler1.fit_transform(sub1)

    k_values = [2, 3, 4, 5, 6]
    k_eval_results = evaluate_k_values(X1, k_values, student_seed)

    top = max(r["silhouette"] for r in k_eval_results)
    best_k_entry = min(
        (r for r in k_eval_results if top -r["silhouette"] < 1e-9),
        key=lambda r: r["k"],
    )
    selected_k = best_k_entry["k"]

    best_kmeans = kmeans_from_scratch(X1, selected_k, student_seed)
    _, p1_counts = np.unique(best_kmeans["labels"], return_counts=True)
    p1_cluster_size = sorted([int(c) for c in p1_counts])


    print("Problem 1 Evaluation Results:")
    for res in k_eval_results:
        print(res)
    print(f"Selected k: {selected_k}")
    print(f"Cluster size for selected K ({selected_k}): {p1_cluster_size}\n")


    hierarchical_df = pd.read_csv("assignment5_hierarchical_linkages.csv")
    sub2 = student_subset(hierarchical_df, 600, student_seed)
    scaler2= StandardScaler()
    X2 = scaler2.fit_transform(sub2)

    linkage_methods = ["single", "complete","average"]
    linkage_results = compare_linkages(X2, linkage_methods,n_clusters=4)
    selected_linkage = select_best_linkage(linkage_results)

    print("Problem 2 Linkage Results:")
    for res in linkage_results:
        print(res)
    print(f"Selected Linkage Method: {selected_linkage}")