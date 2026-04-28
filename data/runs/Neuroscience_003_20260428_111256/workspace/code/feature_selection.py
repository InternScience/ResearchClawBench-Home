"""
Feature scoring methods for trajectory-preserving feature selection.

Implements:
  - variance score (HVF baseline)
  - Spearman correlation with pseudotime (annotated_age)
  - F-statistic across cell-cycle phases (ANOVA)
  - Laplacian Score (kNN-graph based unsupervised)
  - Graph-smoothness score = neighborhood mean preservation (DELVE-style dynamic-feature score):
      score(f) = corr( f(x), mean_kNN(f)(x) ); features whose value is preserved across
      the kNN graph carry coherent dynamic information; pure-noise features score ~0.
  - Composite "DynScore" = Graph-smoothness * |Spearman with pseudotime|
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, f_oneway
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags


def variance_score(X):
    return X.var(axis=0)


def spearman_pseudotime(X, t):
    n_features = X.shape[1]
    out = np.zeros(n_features)
    for j in range(n_features):
        r, _ = spearmanr(X[:, j], t)
        out[j] = 0.0 if np.isnan(r) else abs(r)
    return out


def anova_f_phase(X, phases):
    groups = pd.Categorical(phases)
    n_features = X.shape[1]
    out = np.zeros(n_features)
    by = [np.where(groups.codes == c)[0] for c in range(len(groups.categories))]
    for j in range(n_features):
        try:
            stat, p = f_oneway(*[X[idx, j] for idx in by])
            out[j] = 0.0 if np.isnan(stat) else stat
        except Exception:
            out[j] = 0.0
    return out


def build_knn_graph(X, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, idx = nn.kneighbors(X)
    # drop self
    return idx[:, 1:], dists[:, 1:]


def laplacian_score(X, knn_idx, sigma=None):
    """
    He et al. 2005 Laplacian Score. Lower = better (we'll return negative so higher=better).
    """
    n, d = X.shape
    rows = np.repeat(np.arange(n), knn_idx.shape[1])
    cols = knn_idx.ravel()
    if sigma is None:
        sigma = 1.0
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    W = ((W + W.T) > 0).astype(float)  # symmetrize
    D = np.array(W.sum(axis=1)).ravel()
    L = diags(D) - W
    Dsum = D.sum()
    out = np.zeros(d)
    for j in range(d):
        f = X[:, j]
        # remove weighted mean
        fbar = (D * f).sum() / Dsum
        fc = f - fbar
        num = fc @ (L @ fc)
        den = (D * fc * fc).sum()
        out[j] = num / den if den > 1e-12 else np.inf
    # higher = better -> return negative laplacian score (so larger is more structured)
    return -out


def graph_smoothness(X, knn_idx):
    """
    For each feature, compute Pearson correlation between its value at cell i
    and the average value among i's k nearest neighbors. High value = the
    feature varies smoothly along the graph (coherent dynamic feature) rather
    than being pure noise.
    """
    n, d = X.shape
    out = np.zeros(d)
    for j in range(d):
        f = X[:, j]
        f_nb = f[knn_idx].mean(axis=1)
        if f.std() < 1e-12 or f_nb.std() < 1e-12:
            out[j] = 0.0
            continue
        out[j] = np.corrcoef(f, f_nb)[0, 1]
    return out


def composite_dyn_score(graph_smooth, spearman_abs):
    # both >= 0 (clip negatives in graph_smooth)
    g = np.clip(graph_smooth, 0, None)
    s = np.clip(spearman_abs, 0, None)
    return g * s
