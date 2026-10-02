import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset2.csv', header=None)
X = df.values
n = X.shape[0]

## part a) Run K-means clustring with K=4 using five different random initializations.

def kmeans(X, k, init_centers, max_iter=100, tol=1e-6):
    centers = init_centers.copy()
    error_history = []
    for it in range(max_iter):
        dists = np.sum((X[:,None,:] - centers[None,:,:]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        error = np.sum((X - centers[labels]) ** 2)
        error_history.append(error)
        new_centers = np.zeros_like(centers)
        for j in range(k):

            members = X[labels == j]
            if len(members) > 0:
                 new_centers[j] = members.mean(axis=0)
            else:
                 new_centers[j] = X[np.random.randint(0, len(X))]
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    return centers, labels, error_history
k  = 4
n_runs = 5
colors = ['tab:blue', 'tab:orange', 'tab:green','tab:red']
all_results = []
for run in range(n_runs):
    rng = np.random.RandomState(run)
    init_idx = rng.choice(n, size=k, replace=False)
    init_centers = X[init_idx]
    centers, labels, error_history = kmeans(X, k, init_centers)
    all_results.append((centers, labels, error_history))
    plt.figure(figsize=(5,4))
    plt.plot(range(1, len(error_history) +1), error_history, marker='o', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Error (sum of squared distances)')
    plt.title(f'K-means error vs iteration - run {run+1}')
    plt.tight_layout()
    plt.savefig(f'kmeans_error_run{run+1}.png',dpi=130)
    plt.close()
    plt.figure(figsize=(5, 4.5))
    for j in range(k):
        pts = X[labels == j]
        plt.scatter(pts[:,0], pts[:,1], s=10,color=colors[j], label=f'cluster {j}')
    plt.scatter(centers[:,0], centers[:,1], marker='X', s=150, color='black', edgecolor='white', linewidth=1, label='centers')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'K-means clusters (k=4) - run{run+1}\nfinal error={error_history[-1]:.3f}')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'kmeans_clusters_run{run+1}.png', dpi=130)
    plt.close()
    print(f"Run {run+1}: converged in {len(error_history)} iternations,final error = {error_history[-1]:.4f}")
plt.figure(figsize=(6,4.5))
for run in range(n_runs):
    _,_, error_history = all_results[run]
    plt.plot(range(1, len(error_history)+1), error_history, markersize=3, label=f'run{run+1}')
plt.xlabel('Iteration')
plt.ylabel('error (sum of squared distances)')
plt.title('K-means error vs iteration -all 5 runs')
plt.legend()
plt.tight_layout()
plt.tight_layout()
plt.savefig('kmeans_error_all_runs.png', dpi=130)
plt.close()
final_errors = [res[2][-1] for res in all_results]
best_run = int(np.argmin(final_errors))
print(f"\nBest run: {best_run+1} with final error = {final_errors[best_run]:.4f}")
print("Final errors across runs:", [f"{e:.4f}" for e in final_errors])

## part b) For K belongs {2,3,4,5}, run K-means using a fixed random initialization.

def kmeans(X, k, init_centers, max_iter=100, tol=1e-6):
    centers = init_centers.copy()
    for it in range(max_iter):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2,axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centers[j] = members.mean(axis=0)
            else:
                new_centers[j] = X[np.random.randint(0, len(X))]
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    return centers, labels
np.random.seed(1)
fixed_init_idx = np.random.choice(n, size=5, replace=False)
x_min, x_max = X[:,0].min(), X[:, 0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))              
grid_points = np.c_[xx.ravel(), yy.ravel()]
colors = ['tab:blue','tab:orange', 'tab:green', 'tab:purple', 'tab:red']
for k in [2, 3, 4, 5]:
    init_centers = X[fixed_init_idx[:k]]
    centers, labels = kmeans(X, k, init_centers)
    grid_dists = np.sum((grid_points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    grid_labels = np.argmin(grid_dists, axis=1)
    grid_labels = grid_labels.reshape(xx.shape)
    plt.figure(figsize=(5, 4.5))
    plt.contourf(xx, yy, grid_labels, levels=np.arange(k+1)-0.5, colors=colors[:k], alpha=0.3)
    for j in range(k):
        pts = X[labels == j]
        plt.scatter(pts[:, 0], pts[:, 1], s=10, color=colors[j])

    plt.scatter(centers[:,0], centers[:,1], marker='X', s=150, color='black', edgecolor='white', linewidth=1)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Voronoi regions, K = {k}')
    plt.tight_layout()
    plt.savefig(f'voronoi_k{k}.png', dpi=130)
    plt.close()


 ## Part C) Run spectral clustring ( K = 4) using a suitable kernel- based spectral embedding.

def rbf_kernel(X, sigma):
    sq_norms = np.sum(X**2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)
    gamma = 1.0 / (2 * sigma**2)
    return np.exp(-gamma * sq_dists)

def run_part_c():
    sigma = 0.3
    K = 4
    N_RESTARTS = 10
    SEED = 0

    K_mat = rbf_kernel(X, sigma)
    d = K_mat.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    K_norm = (d_inv_sqrt[:, None] * K_mat) * d_inv_sqrt[None, :]

    eigvals, eigvecs = np.linalg.eigh(K_norm)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    top_eigvecs = eigvecs[:, :K]
    top_eigvecs = top_eigvecs / (np.linalg.norm(top_eigvecs, axis=1, keepdims=True) + 1e-12)

    best_labels, best_sse = None, np.inf
    rng = np.random.RandomState(SEED)
    for _ in range(N_RESTARTS):
        init_idx = rng.choice(n, size=k, replace=False)
        init_centers = top_eigvecs[init_idx]
        centers, labels = kmeans(top_eigvecs, K, init_centers)
        sse = np.sum((top_eigvecs - centers[labels]) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_labels = labels
        
    labels = best_labels
    print("Cluster size:", np.bincount(labels, minlength=k))
    colors = ['tab:blue', 'tab:orange', 'tab:green','tab:red']
    plt.figure(figsize=(5,5))
    for j in range(K):
        pts = X[labels == j]
        plt.scatter(pts[:,0], pts[:,1], s=10, color=colors[j], label=f'cluster {j}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Spectral clustring (RBF kernel, sigma={sigma})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('spectral_clustring.png', dpi=130)
    plt.close()

    return top_eigvecs, K

top_eigvecs, K = run_part_c()


## Part D) Apply the direct eigenvector assignment rule l = argmax using the top K = 4.
labels_d = np.argmax(top_eigvecs, axis=1)
print("cluster size:", np.bincount(labels_d))
colors = ['tab:blue','tab:orange','tab:green', 'tab:red']
plt.figure(figsize=(5,5))
for j in range(K):
    pts = X[labels_d == j]
    plt.scatter(pts[:,0], pts[:,1], s=10, color=colors[j],label=f'cluster{j}')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Argmax eigenvector cluster assignment')
plt.legend()
plt.tight_layout()
plt.savefig('partd_clusters.png', dpi=130)
plt.close()


