"""
Data-Driven De Novo Design of Super-Adhesive Hydrogels
Main Analysis Pipeline

This script:
1. Loads and integrates all hydrogel datasets
2. Performs exploratory data analysis
3. Trains Random Forest and Gaussian Process models
4. Performs Bayesian optimization to discover optimal compositions
5. Generates interpretability artifacts (feature importance, SHAP)
6. Saves all figures and intermediate results
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
from pathlib import Path

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Interpretability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_000_20260415_094527')
DATA_DIR = WORKSPACE / 'data'
OUTPUTS_DIR = WORKSPACE / 'outputs'
REPORT_IMAGES_DIR = WORKSPACE / 'report' / 'images'

for d in [OUTPUTS_DIR, REPORT_IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Feature columns (monomer compositions)
FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET_COL = 'Glass (kPa)_10s'

print("=" * 80)
print("PHASE 1: Data Integration & Exploration")
print("=" * 80)

# ============================================================
# Load primary verified dataset (184 samples)
# ============================================================
df_primary = pd.read_excel(DATA_DIR / '184_verified_Original Data_ML_20230926.xlsx')
print(f"\nPrimary dataset: {df_primary.shape}")
print(f"Columns: {list(df_primary.columns)}")

# Extract features and target
X_primary = df_primary[FEATURE_COLS].values
y_primary = df_primary[TARGET_COL].values

# Handle any NaN in target
valid_mask = ~np.isnan(y_primary)
X_clean = X_primary[valid_mask]
y_clean = y_primary[valid_mask]
print(f"Valid samples (with target): {X_clean.shape[0]}")
print(f"Target range: [{y_clean.min():.2f}, {y_clean.max():.2f}] kPa")
print(f"Target mean ± std: {y_clean.mean():.2f} ± {y_clean.std():.2f} kPa")

# Save cleaned data info
data_info = {
    'primary_samples': int(X_clean.shape[0]),
    'n_features': len(FEATURE_COLS),
    'feature_names': FEATURE_COLS,
    'target_name': TARGET_COL,
    'target_mean_kPa': float(y_clean.mean()),
    'target_std_kPa': float(y_clean.std()),
    'target_min_kPa': float(y_clean.min()),
    'target_max_kPa': float(y_clean.max()),
    'target_median_kPa': float(np.median(y_clean)),
    'samples_above_1MPa': int((y_clean > 1000).sum()),
}
with open(OUTPUTS_DIR / 'data_info.json', 'w') as f:
    json.dump(data_info, f, indent=2)
print(f"\nData info saved to outputs/data_info.json")

# ============================================================
# Load optimization round data
# ============================================================
df_opt_rounds = pd.read_excel(DATA_DIR / 'ML_ei&pred (1&2&3rounds)_20240408.xlsx')
print(f"\nOptimization rounds dataset: {df_opt_rounds.shape}")

# Merge primary + optimization data for full trajectory analysis
# The optimization data has predicted values; combine with primary for full picture
df_opt_features = df_opt_rounds[FEATURE_COLS].values
y_opt_pred = pd.to_numeric(df_opt_rounds['Glass (kPa)_max'], errors='coerce').values

valid_opt = ~np.isnan(y_opt_pred)
df_opt_features = df_opt_features[valid_opt]
y_opt_pred = y_opt_pred[valid_opt]
print(f"Optimization samples: {df_opt_features.shape[0]}")
print(f"Optimization target range: [{y_opt_pred.min():.2f}, {y_opt_pred.max():.2f}] kPa")

# ============================================================
# Exploratory Data Analysis - Figure 1: Feature Distributions & Correlations
# ============================================================
print("\n--- Generating Figure 1: Feature Distributions ---")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

# Feature distributions (histograms)
for i, col in enumerate(FEATURE_COLS):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    sns.histplot(df_primary[col], bins=20, kde=True, color='steelblue', ax=ax)
    ax.set_xlabel(col.replace('-', '\n'), fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f'{col}\n(mean={df_primary[col].mean():.3f})', fontsize=10)
    ax.tick_params(labelsize=8)

# Target distribution
ax_tgt = fig.add_subplot(gs[0, 2])
sns.histplot(y_clean, bins=25, kde=True, color='darkred', ax=ax_tgt)
ax_tgt.axvline(1000, color='green', linestyle='--', linewidth=2, label='1 MPa threshold')
ax_tgt.set_xlabel('Glass Adhesion (kPa)', fontsize=10)
ax_tgt.set_ylabel('Count', fontsize=9)
ax_tgt.set_title('Adhesive Strength Distribution', fontsize=11, fontweight='bold')
ax_tgt.legend(fontsize=9)
ax_tgt.tick_params(labelsize=8)

# Feature correlation heatmap
ax_corr = fig.add_subplot(gs[1:, 2:])
corr_data = df_primary[FEATURE_COLS + [TARGET_COL]].copy()
corr_data[TARGET_COL] = y_primary  # use original with NaN handled
corr_matrix = corr_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, ax=ax_corr, vmin=-1, vmax=1,
            xticklabels=[c.replace('-', '\n') for c in FEATURE_COLS] + ['Adhesion'],
            yticklabels=[c.replace('-', '\n') for c in FEATURE_COLS] + ['Adhesion'])
ax_corr.set_title('Feature-Target Correlation Matrix', fontsize=12, fontweight='bold')
ax_corr.tick_params(labelsize=7, rotation=45)

plt.savefig(REPORT_IMAGES_DIR / 'figure1_data_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure1_data_overview.png")

# ============================================================
# Feature-Target Scatter Plots - Figure 2
# ============================================================
print("\n--- Generating Figure 2: Feature vs Target Relationships ---")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(FEATURE_COLS):
    ax = axes[i]
    scatter = ax.scatter(X_clean[:, i], y_clean, c=y_clean, cmap='viridis', 
                         alpha=0.7, edgecolors='none', s=40)
    # Add trend line
    z = np.polyfit(X_clean[:, i], y_clean, 1)
    p = np.poly1d(z)
    x_line = np.linspace(X_clean[:, i].min(), X_clean[:, i].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
    ax.axhline(1000, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(col.replace('-', '\n'), fontsize=10)
    ax.set_ylabel('Adhesion (kPa)', fontsize=10)
    corr_coef = np.corrcoef(X_clean[:, i], y_clean)[0, 1]
    ax.set_title(f'{col}\nr = {corr_coef:.3f}', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=8)

plt.colorbar(scatter, ax=axes, label='Adhesion (kPa)', shrink=0.8)
plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'figure2_feature_target.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure2_feature_target.png")

# ============================================================
# Composition Space Visualization - Figure 3
# ============================================================
print("\n--- Generating Figure 3: Composition Space ---")

fig = plt.figure(figsize=(14, 10))

# Ternary-like plot: Nucleophilic vs Hydrophobic vs Acidic
ax1 = fig.add_subplot(221)
sc1 = ax1.scatter(X_clean[:, 0], X_clean[:, 1], c=y_clean, cmap='plasma', 
                   s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
ax1.set_xlabel('Nucleophilic-HEA', fontsize=11)
ax1.set_ylabel('Hydrophobic-BA', fontsize=11)
ax1.set_title('Nucleophilic vs Hydrophobic\n(colored by adhesion)', fontsize=12, fontweight='bold')
plt.colorbar(sc1, ax=ax1, label='Adhesion (kPa)')

# Cationic vs Aromatic
ax2 = fig.add_subplot(222)
sc2 = ax2.scatter(X_clean[:, 3], X_clean[:, 4], c=y_clean, cmap='plasma',
                   s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
ax2.set_xlabel('Cationic-ATAC', fontsize=11)
ax2.set_ylabel('Aromatic-PEA', fontsize=11)
ax2.set_title('Cationic vs Aromatic\n(colored by adhesion)', fontsize=12, fontweight='bold')
plt.colorbar(sc2, ax=ax2, label='Adhesion (kPa)')

# Stacked bar chart of top-10 performing compositions
top_idx = np.argsort(y_clean)[-10:]
ax3 = fig.add_subplot(212)
x_pos = np.arange(len(top_idx))
width = 0.12
colors_bar = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
for j, col in enumerate(FEATURE_COLS):
    vals = X_clean[top_idx, j]
    ax3.bar(x_pos + j*width, vals, width, label=col.replace('-', '\n'), color=colors_bar[j])
ax3.set_xticks(x_pos + width*2.5)
ax3.set_xticklabels([f'#{int(i+1)}\n{y_clean[top_idx[i]]:.0f}kPa' for i in range(len(top_idx))], fontsize=8)
ax3.set_ylabel('Monomer Fraction', fontsize=11)
ax3.set_title('Top-10 Performing Hydrogel Compositions', fontsize=12, fontweight='bold')
ax3.legend(fontsize=7, loc='upper right', ncol=3)
ax3.set_ylim(0, 1)
ax3.axhline(1.0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(REPORT_IMAGES_DIR / 'figure3_composition_space.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure3_composition_space.png")

print("\nPhase 1 complete!")
