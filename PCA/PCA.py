import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('Dataset1.CSV', header=0)
X = df[['x1', 'x2']].values
y = df['classLabel'].values

# Part a) Using PCA from Scratch try to find Variance for each Principle Components
def pca(X, n_components=2):
    N = X.shape[0]
    X_mean = np.mean(X ,axis=0)
    X_centered = X - X_mean
    cov_mat = (X_centered.T @ X_centered) / (N - 1)
    print(cov_mat.shape)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    explained_variance_ratio = eigvals / np.sum(eigvals)
    X_pca = X_centered @ eigvecs[:, :n_components]
    return X_pca, eigvals, explained_variance_ratio
    
X_pca, eigvals, explained_variance_ratio = pca(X, n_components=2)
print(f"Eigenvalues (Variances): {eigvals}")
print(f"Variance Rations : {explained_variance_ratio * 100}%")
print(f"PC1 Variance: {eigvals[0] :.4f} ({explained_variance_ratio[0]*100:.2f}%)")
print(f"PC2 Varianve: {eigvals[1]:.4f} ({explained_variance_ratio[1]*100:.2f}%)\n")


## Part b) Using KPCA for each Kernel, plot the projection of each point in the dataset onto the top-2 principal components.

def linear_kernel(X):
    return X @ X.T
    
def poly_kernel(X, degree=3, coef0=1):
    return (X @ X.T + coef0) ** degree

def rbf_kernel(X, sigma=1.0):
    sq_norms = np.sum(X ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None,:] - 2 * (X @ X.T)
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-sq_dists / (2 * sigma ** 2))
    
def sigmoid_kernel(X, gamma=0.5, coef0=1):
    return np.tanh(gamma * (X @ X.T) + coef0)

def kernel_pca(K, n_components=2):
    n = K.shape[0]
    one_n = np.ones((n,n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n 
    eigvals, eigvecs = np.linalg.eigh(K_centered)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals_top = np.clip(eigvals[:n_components], 1e-12, None)
    alphas = eigvecs[:,:n_components] / np.sqrt(eigvals_top)
    X_kpca = K_centered @ alphas
    return X_kpca, eigvals[:n_components]

def plot_kpca(X_kpca, y, title, fname):
    plt.figure(figsize=(5, 4.5))
    for label in np.unique(y):
        mask = (y == label)
        plt.scatter(X_kpca[mask, 0], X_kpca[mask, 1], s=12, label=f'class {label}')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()
    
k_lin = linear_kernel(X)
X_lin, ev_lin = kernel_pca(k_lin)
plot_kpca(X_lin, y, 'Linear kernel', 'kpca_fig1_linear.png')
print(f"[Linear kernel] Top-2 Eigenvalues: {ev_lin}")

k_sig = sigmoid_kernel(X)
X_sig, ev_sig = kernel_pca(k_sig)
plot_kpca(X_sig, y, 'Sigmoid kernel', 'kpca_fig1_sigmoid.png')
print(f"[sigmoid kernel] Top-2 eigenvalues: {ev_sig}")

for sigma in [0.1, 0.5, 1.0, 2.9, 5.0]:
    k_rbf = rbf_kernel(X, sigma=sigma)
    X_rbf, ev_rbf = kernel_pca(k_rbf)
    plot_kpca(X_rbf, y, f'RBF kernel(sigma={sigma})',f'kpca_fig2_sigma{sigma}.png')
    print(f"[RBF sigma={sigma}] Top-2 Eigenvalyes: {ev_rbf}")

for degree in [2, 3, 4, 5, 6]:
    k_poly = poly_kernel(X, degree=degree, coef0=1)
    X_poly , ev_poly = kernel_pca(k_poly)
    plot_kpca(X_poly, y, f'polynomial kernel(degree={degree})',f'kpca_fig3_poly_deg{degree}.png')
    print(f"[Poly degree={degree}] Top-2 Eigenvalyes: {ev_poly}")

print()

# Part c) find a kernel for this problem which with just using projections along top-2 principal components will linearly separable dataset as per the class labels specified?

def myself_kernel_matrix(X):
    r2 = X[:, 0] ** 2 + X[:,1] ** 2
    r4 = r2 ** 2
    k = r2[:, None] * r2[None,:] + r4[:, None] * r4[None, :]
    return k

def myself_define_pca(K, n_components=2):
    n = K.shape[0]
    one_n = np.full((n,n), 1.0 / n)
    Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    eigval, eigvec = np.linalg.eigh(Kc)
    order = np.argsort(eigval)[:: -1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    eigval_top = eigval[: n_components]
    eigvec_top = eigvec[: , :n_components]
    eigval_top = np.where(eigval_top > 1e-12, eigval_top, 1e-12)
    alphas = np.zeros_like(eigvec_top)
    for k in range(n_components):
        alphas[:, k] = eigvec_top[:, k] / np.sqrt(eigval_top[k])

    Z = Kc @ alphas 
    return Z, eigval_top

K_myself = myself_kernel_matrix(X)
Z_myself, myself_eigvals = myself_define_pca(K_myself, n_components=2)
print(f"Myself Kernel Top-2 Eigvalues: {myself_eigvals}")
print(f"project Feature Matrix Shape Z: {Z_myself.shape}")

plt.figure(figsize=(6, 5))
for label in np.unique(y):
    mask = (y == label)
    plt.scatter(Z_myself[mask, 0], Z_myself[mask, 1], s=15, label=f'class {int(label)}')
plt.xlabel('Kernel PC 1')
plt.ylabel('Kenrel PC 2')
plt.title('Top Kernel PCA | myself Radial kernel')
plt.legend()
plt.tight_layout()
plt.savefig('kpca_fig4_custom_radial.png', dpi=130)
plt.close()
  
    

