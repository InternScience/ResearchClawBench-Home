#!/usr/bin/env python3
"""
Structure Generation Workflow
Generate novel material structures/lattice parameters using generative models.
Uses data from File 2 of the M-AI-Synth dataset.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Parse data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'M-AI-Synth__Materials_AI_Dataset_.txt')
with open(data_path, 'r') as f:
    content = f.read()

sections = content.split('# 文件')
file2_content = sections[2]  # structure_generation.py

lines = [l.strip() for l in file2_content.split('\n') if l.strip().startswith('[')]
param_a = np.array(json.loads(lines[0]))  # 100 values
param_b = np.array(json.loads(lines[1]))  # 100 values

print(f"Param A: shape={param_a.shape}, range=[{param_a.min():.4f}, {param_a.max():.4f}], mean={param_a.mean():.4f}")
print(f"Param B: shape={param_b.shape}, range=[{param_b.min():.4f}, {param_b.max():.4f}], mean={param_b.mean():.4f}")

# Stack into 2D
data_2d = np.column_stack([param_a, param_b])

# 1. Kernel Density Estimation
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

kde_a = gaussian_kde(param_a)
kde_b = gaussian_kde(param_b)
kde_2d = gaussian_kde(data_2d.T)

# Generate samples from KDE
n_gen = 200
np.random.seed(42)
samples_2d = kde_2d.resample(n_gen).T
gen_a = samples_2d[:, 0]
gen_b = samples_2d[:, 1]

# 2. Gaussian Mixture Model
from sklearn.mixture import GaussianMixture

# Find best number of components
bic_scores = []
aic_scores = []
n_components_range = range(1, 11)
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42, covariance_type='full')
    gmm.fit(data_2d)
    bic_scores.append(gmm.bic(data_2d))
    aic_scores.append(gmm.aic(data_2d))

best_n_bic = n_components_range[np.argmin(bic_scores)]
best_n_aic = n_components_range[np.argmin(aic_scores)]
print(f"Best GMM components (BIC): {best_n_bic}, (AIC): {best_n_aic}")

gmm_best = GaussianMixture(n_components=best_n_bic, random_state=42, covariance_type='full')
gmm_best.fit(data_2d)
gmm_samples, _ = gmm_best.sample(n_gen)

# 3. Statistical comparison
from scipy.stats import ks_2samp, wasserstein_distance

# KS test for 1D marginals
ks_a_kde = ks_2samp(param_a, gen_a)
ks_b_kde = ks_2samp(param_b, gen_b)
ks_a_gmm = ks_2samp(param_a, gmm_samples[:, 0])
ks_b_gmm = ks_2samp(param_b, gmm_samples[:, 1])

# Wasserstein distance for 1D
w_a_kde = wasserstein_distance(param_a, gen_a)
w_b_kde = wasserstein_distance(param_b, gen_b)
w_a_gmm = wasserstein_distance(param_a, gmm_samples[:, 0])
w_b_gmm = wasserstein_distance(param_a, gmm_samples[:, 1])

# Pearson correlation comparison
real_corr = np.corrcoef(param_a, param_b)[0, 1]
kde_corr = np.corrcoef(gen_a, gen_b)[0, 1]
gmm_corr = np.corrcoef(gmm_samples[:, 0], gmm_samples[:, 1])[0, 1]

results = {
    'data_summary': {
        'n_samples': int(len(param_a)),
        'param_a': {
            'mean': float(param_a.mean()),
            'std': float(param_a.std()),
            'min': float(param_a.min()),
            'max': float(param_a.max())
        },
        'param_b': {
            'mean': float(param_b.mean()),
            'std': float(param_b.std()),
            'min': float(param_b.min()),
            'max': float(param_b.max())
        },
        'correlation': float(real_corr)
    },
    'gmm_selection': {
        'best_n_bic': int(best_n_bic),
        'best_n_aic': int(best_n_aic),
        'bic_scores': [float(b) for b in bic_scores],
        'aic_scores': [float(a) for a in aic_scores]
    },
    'generation_quality': {
        'KDE': {
            'ks_a_statistic': float(ks_a_kde.statistic),
            'ks_a_pvalue': float(ks_a_kde.pvalue),
            'ks_b_statistic': float(ks_b_kde.statistic),
            'ks_b_pvalue': float(ks_b_kde.pvalue),
            'wasserstein_a': float(w_a_kde),
            'wasserstein_b': float(w_b_kde),
            'correlation': float(kde_corr)
        },
        'GMM': {
            'ks_a_statistic': float(ks_a_gmm.statistic),
            'ks_a_pvalue': float(ks_a_gmm.pvalue),
            'ks_b_statistic': float(ks_b_gmm.statistic),
            'ks_b_pvalue': float(ks_b_gmm.pvalue),
            'wasserstein_a': float(w_a_gmm),
            'wasserstein_b': float(w_b_gmm),
            'correlation': float(gmm_corr)
        }
    },
    'generated_samples': {
        'KDE': {'param_a': gen_a.tolist(), 'param_b': gen_b.tolist()},
        'GMM': {'param_a': gmm_samples[:, 0].tolist(), 'param_b': gmm_samples[:, 1].tolist()}
    }
}

output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'structure_generation_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== Structure Generation Results ===")
print(f"Real correlation: {real_corr:.4f}")
print(f"KDE correlation: {kde_corr:.4f}")
print(f"GMM correlation: {gmm_corr:.4f}")
print(f"\nKS test p-values (KDE): A={ks_a_kde.pvalue:.4f}, B={ks_b_kde.pvalue:.4f}")
print(f"KS test p-values (GMM): A={ks_a_gmm.pvalue:.4f}, B={ks_b_gmm.pvalue:.4f}")
print(f"\nWasserstein distances (KDE): A={w_a_kde:.6f}, B={w_b_kde:.6f}")
print(f"Wasserstein distances (GMM): A={w_a_gmm:.6f}, B={w_b_gmm:.6f}")
print(f"\nResults saved to outputs/structure_generation_results.json")
