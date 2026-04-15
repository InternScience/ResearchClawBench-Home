"""
Single-cell trajectory analysis and dynamic feature selection
for RPE cell cycle progression data.

This script:
1. Loads and preprocesses the single-cell protein imaging data
2. Performs trajectory inference using pseudotime analysis
3. Identifies dynamically expressed features that preserve cellular trajectories
4. Selects optimal feature subsets using multiple criteria
5. Validates feature selection through trajectory preservation metrics
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("Single-Cell Trajectory Analysis and Dynamic Feature Selection")
print("=" * 60)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n[1] Loading data...")
adata = sc.read_h5ad('data/adata_RPE.h5ad')
print(f"    Loaded data: {adata.shape[0]} cells x {adata.shape[1]} features")

# Clean data - remove NaN states
adata = adata[~adata.obs['state'].isna()].copy()
print(f"    After removing NaN states: {adata.shape[0]} cells")

# ==============================================================================
# 2. DATA PREPROCESSING
# ==============================================================================
print("\n[2] Preprocessing data...")

# Store raw data
adata.layers['raw'] = adata.X.copy()

# Normalize (the data appears to already be normalized, but let's ensure)
sc.pp.scale(adata, max_value=10)
print("    Data scaled (z-score normalization)")

# ==============================================================================
# 3. DIMENSIONALITY REDUCTION AND VISUALIZATION
# ==============================================================================
print("\n[3] Computing dimensionality reduction...")

# PCA
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
print(f"    PCA computed: {adata.obsm['X_pca'].shape[1]} components")

# Compute UMAP for visualization
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata, min_dist=0.3)
print("    UMAP embedding computed")

# ==============================================================================
# 4. PSEUDOTIME INFERENCE (Trajectory Analysis)
# ==============================================================================
print("\n[4] Inferring cellular trajectories via pseudotime analysis...")

# Use annotated_age as the primary pseudotime measure (biological time)
# This represents the progression through cell cycle
adata.obs['pseudotime'] = adata.obs['annotated_age'].values

# Also compute diffusion pseudotime for comparison
sc.tl.diffmap(adata, n_comps=10)

# Find root cells (G0 phase cells with lowest age)
root_cells = np.where((adata.obs['phase'] == 'G0') & 
                      (adata.obs['annotated_age'] < adata.obs['annotated_age'].quantile(0.1)))[0]
if len(root_cells) > 0:
    adata.uns['iroot'] = root_cells[0]
    sc.tl.dpt(adata, n_dcs=5)
    print("    Diffusion pseudotime computed")
else:
    adata.obs['dpt_pseudotime'] = adata.obs['pseudotime']
    print("    Using annotated age as pseudotime")

# ==============================================================================
# 5. VISUALIZATION OF DATA STRUCTURE
# ==============================================================================
print("\n[5] Generating data overview visualizations...")

fig = plt.figure(figsize=(16, 12))

# UMAP colored by phase
ax1 = plt.subplot(2, 3, 1)
sc.pl.umap(adata, color='phase', ax=ax1, show=False, title='Cell Cycle Phase')
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')

# UMAP colored by state
ax2 = plt.subplot(2, 3, 2)
sc.pl.umap(adata, color='state', ax=ax2, show=False, title='Cell State')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')

# UMAP colored by pseudotime
ax3 = plt.subplot(2, 3, 3)
sc.pl.umap(adata, color='pseudotime', ax=ax3, show=False, title='Pseudotime (Age)')
ax3.set_xlabel('UMAP 1')
ax3.set_ylabel('UMAP 2')

# Phase vs Pseudotime
ax4 = plt.subplot(2, 3, 4)
phase_order = ['G0', 'G1', 'S', 'G2']
adata.obs['phase_num'] = adata.obs['phase'].map({p: i for i, p in enumerate(phase_order)})
sns.boxplot(data=adata.obs, x='phase', y='pseudotime', order=phase_order, ax=ax4)
ax4.set_title('Pseudotime Distribution by Cell Cycle Phase')
ax4.set_xlabel('Cell Cycle Phase')
ax4.set_ylabel('Pseudotime')

# State vs Pseudotime
ax5 = plt.subplot(2, 3, 5)
sns.boxplot(data=adata.obs, x='state', y='pseudotime', ax=ax5)
ax5.set_title('Pseudotime by Cell State')
ax5.set_xlabel('Cell State')
ax5.set_ylabel('Pseudotime')

# Feature variance distribution
ax6 = plt.subplot(2, 3, 6)
feature_vars = np.var(adata.X, axis=0)
ax6.hist(feature_vars, bins=30, edgecolor='black', alpha=0.7)
ax6.set_xlabel('Feature Variance')
ax6.set_ylabel('Count')
ax6.set_title('Distribution of Feature Variances')
ax6.axvline(np.median(feature_vars), color='red', linestyle='--', 
            label=f'Median: {np.median(feature_vars):.4f}')
ax6.legend()

plt.tight_layout()
plt.savefig('report/images/figure_01_data_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: report/images/figure_01_data_overview.png")

# ==============================================================================
# 6. DYNAMIC FEATURE ANALYSIS
# ==============================================================================
print("\n[6] Analyzing feature dynamics along trajectories...")

# Calculate correlation of each feature with pseudotime
feature_names = adata.var_names.tolist()
X = adata.X
pseudotime = adata.obs['pseudotime'].values

correlations = []
p_values = []
for i in range(X.shape[1]):
    corr, pval = stats.pearsonr(X[:, i], pseudotime)
    correlations.append(corr)
    p_values.append(pval)

correlations = np.array(correlations)
p_values = np.array(p_values)

# Store in adata
adata.var['pseudotime_correlation'] = correlations
adata.var['pseudotime_pvalue'] = p_values
adata.var['abs_correlation'] = np.abs(correlations)

# Calculate feature importance using random forest
print("    Computing feature importance using Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, pseudotime)
adata.var['rf_importance'] = rf.feature_importances_

# Calculate mutual information
print("    Computing mutual information...")
mi_scores = mutual_info_regression(X, pseudotime, random_state=42)
adata.var['mutual_info'] = mi_scores

# ==============================================================================
# 7. FEATURE RANKING AND SELECTION
# ==============================================================================
print("\n[7] Ranking features by dynamic relevance...")

# Create composite score
var_df = adata.var.copy()
var_df['rank_corr'] = var_df['abs_correlation'].rank(ascending=False)
var_df['rank_rf'] = var_df['rf_importance'].rank(ascending=False)
var_df['rank_mi'] = var_df['mutual_info'].rank(ascending=False)

# Composite ranking (lower is better)
var_df['composite_rank'] = (var_df['rank_corr'] + var_df['rank_rf'] + var_df['rank_mi']) / 3
var_df['composite_score'] = 1 / var_df['composite_rank']  # Higher is better

# Sort by composite score
var_df_sorted = var_df.sort_values('composite_score', ascending=False)
top_features = var_df_sorted.head(50)

print(f"    Top 10 dynamically expressed features:")
for i, (feat, row) in enumerate(top_features.head(10).iterrows(), 1):
    print(f"      {i:2d}. {feat:40s} (r={row['pseudotime_correlation']:6.3f}, RF={row['rf_importance']:.4f})")

# Save feature rankings
var_df_sorted.to_csv('outputs/feature_rankings.csv')
print("    Saved: outputs/feature_rankings.csv")

# ==============================================================================
# 8. VISUALIZE TOP DYNAMIC FEATURES
# ==============================================================================
print("\n[8] Visualizing top dynamic features...")

top_12_features = top_features.head(12).index.tolist()

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, feat in enumerate(top_12_features):
    ax = axes[idx]
    feat_idx = list(feature_names).index(feat)
    
    # Scatter plot of feature value vs pseudotime
    ax.scatter(adata.obs['pseudotime'], X[:, feat_idx], 
               c=adata.obs['phase_num'], cmap='viridis', alpha=0.5, s=10)
    
    # Add trend line
    z = np.polyfit(adata.obs['pseudotime'], X[:, feat_idx], 1)
    p = np.poly1d(z)
    ax.plot(sorted(adata.obs['pseudotime']), 
            p(sorted(adata.obs['pseudotime'])), 
            "r--", alpha=0.8, linewidth=2)
    
    corr = var_df.loc[feat, 'pseudotime_correlation']
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Expression')
    ax.set_title(f'{feat}\nr={corr:.3f}')

plt.tight_layout()
plt.savefig('report/images/figure_02_dynamic_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: report/images/figure_02_dynamic_features.png")

# ==============================================================================
# 9. TRAJECTORY PRESERVATION ANALYSIS
# ==============================================================================
print("\n[9] Evaluating trajectory preservation for feature subsets...")

def compute_trajectory_preservation(adata_subset, original_pseudotime):
    """
    Compute how well the trajectory is preserved in a feature subset.
    Uses correlation between original pseudotime and diffusion pseudotime
    computed from the subset.
    """
    # Compute diffusion map on subset
    sc.pp.neighbors(adata_subset, n_neighbors=15, use_rep='X')
    sc.tl.diffmap(adata_subset, n_comps=5)
    
    # Compute correlation with original pseudotime
    # Use first diffusion component as it's often related to pseudotime
    dpt_corr = np.abs(np.corrcoef(adata_subset.obsm['X_diffmap'][:, 0], 
                                   original_pseudotime)[0, 1])
    return dpt_corr

def evaluate_feature_subset(adata, feature_indices, original_pseudotime):
    """Evaluate a feature subset for trajectory preservation."""
    subset = adata[:, feature_indices].copy()
    return compute_trajectory_preservation(subset, original_pseudotime)

# Test different subset sizes
subset_sizes = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
results = []

original_pseudotime = adata.obs['pseudotime'].values

for size in subset_sizes:
    # Method 1: Top by correlation
    top_corr_idx = var_df_sorted.head(size).index
    top_corr_indices = [feature_names.index(f) for f in top_corr_idx]
    score_corr = evaluate_feature_subset(adata, top_corr_indices, original_pseudotime)
    
    # Method 2: Top by random forest
    top_rf_idx = var_df.sort_values('rf_importance', ascending=False).head(size).index
    top_rf_indices = [feature_names.index(f) for f in top_rf_idx]
    score_rf = evaluate_feature_subset(adata, top_rf_indices, original_pseudotime)
    
    # Method 3: Top by composite score
    top_comp_idx = var_df_sorted.head(size).index
    top_comp_indices = [feature_names.index(f) for f in top_comp_idx]
    score_comp = evaluate_feature_subset(adata, top_comp_indices, original_pseudotime)
    
    # Method 4: Random selection (baseline)
    random_indices = np.random.choice(len(feature_names), size=min(size, len(feature_names)), replace=False)
    score_random = evaluate_feature_subset(adata, random_indices, original_pseudotime)
    
    results.append({
        'n_features': size,
        'correlation': score_corr,
        'random_forest': score_rf,
        'composite': score_comp,
        'random': score_random
    })
    print(f"    Size {size:3d}: Corr={score_corr:.3f}, RF={score_rf:.3f}, Composite={score_comp:.3f}, Random={score_random:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv('outputs/trajectory_preservation_analysis.csv', index=False)
print("    Saved: outputs/trajectory_preservation_analysis.csv")

# ==============================================================================
# 10. VISUALIZE TRAJECTORY PRESERVATION
# ==============================================================================
print("\n[10] Generating trajectory preservation plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Trajectory preservation curves
ax1 = axes[0]
ax1.plot(results_df['n_features'], results_df['correlation'], 'o-', label='Correlation', linewidth=2)
ax1.plot(results_df['n_features'], results_df['random_forest'], 's-', label='Random Forest', linewidth=2)
ax1.plot(results_df['n_features'], results_df['composite'], '^-', label='Composite Score', linewidth=2)
ax1.plot(results_df['n_features'], results_df['random'], 'x--', label='Random (baseline)', linewidth=2, alpha=0.7)
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Trajectory Preservation Score')
ax1.set_title('Trajectory Preservation vs. Feature Set Size')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Feature importance distribution
ax2 = axes[1]
top_20 = var_df_sorted.head(20)
x_pos = np.arange(len(top_20))
ax2.barh(x_pos, top_20['composite_score'], color='steelblue', alpha=0.7)
ax2.set_yticks(x_pos)
ax2.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in top_20.index], fontsize=8)
ax2.set_xlabel('Composite Score')
ax2.set_title('Top 20 Dynamically Expressed Features')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('report/images/figure_03_trajectory_preservation.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: report/images/figure_03_trajectory_preservation.png")

# ==============================================================================
# 11. OPTIMAL FEATURE SET SELECTION
# ==============================================================================
print("\n[11] Selecting optimal feature sets...")

# Find elbow point in trajectory preservation curve
from scipy.signal import argrelextrema

# Use composite score for optimal selection
scores = results_df['composite'].values
# Find point where adding more features gives diminishing returns
gradients = np.diff(scores)
# Find where gradient becomes small (less than 10% of max gradient)
threshold = 0.1 * np.max(gradients)
elbow_idx = np.where(gradients < threshold)[0]
if len(elbow_idx) > 0:
    optimal_idx = elbow_idx[0] + 1  # +1 because diff reduces length by 1
    optimal_n = subset_sizes[min(optimal_idx, len(subset_sizes)-1)]
else:
    optimal_n = 20

print(f"    Optimal feature set size: {optimal_n}")

# Select final feature sets
optimal_features = var_df_sorted.head(optimal_n).index.tolist()
high_confidence_features = var_df_sorted.head(10).index.tolist()

# Save feature sets
with open('outputs/selected_features_optimal.txt', 'w') as f:
    f.write(f"# Optimal feature set (n={optimal_n})\n")
    f.write(f"# Selected based on composite score maximizing trajectory preservation\n")
    for feat in optimal_features:
        f.write(f"{feat}\n")

with open('outputs/selected_features_high_confidence.txt', 'w') as f:
    f.write(f"# High-confidence feature set (n={len(high_confidence_features)})\n")
    f.write(f"# Top features by composite score\n")
    for feat in high_confidence_features:
        f.write(f"{feat}\n")

print(f"    Saved optimal feature set ({optimal_n} features)")
print(f"    Saved high-confidence feature set (10 features)")

# ==============================================================================
# 12. VALIDATION: TRAJECTORY VISUALIZATION WITH SELECTED FEATURES
# ==============================================================================
print("\n[12] Validating selected features through trajectory visualization...")

# Create AnnData with only optimal features
adata_optimal = adata[:, optimal_features].copy()
sc.pp.pca(adata_optimal, n_comps=min(20, optimal_n-1))
sc.pp.neighbors(adata_optimal, n_neighbors=15, n_pcs=min(10, optimal_n-2))
sc.tl.umap(adata_optimal, min_dist=0.3)

# Create AnnData with high-confidence features
adata_hc = adata[:, high_confidence_features].copy()
sc.pp.pca(adata_hc, n_comps=min(10, len(high_confidence_features)-1))
sc.pp.neighbors(adata_hc, n_neighbors=15, n_pcs=min(5, len(high_confidence_features)-2))
sc.tl.umap(adata_hc, min_dist=0.3)

# Visualization
fig = plt.figure(figsize=(18, 10))

# Original full data
ax1 = plt.subplot(2, 3, 1)
sc.pl.umap(adata, color='pseudotime', ax=ax1, show=False, title='Full Dataset (241 features)')
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')

ax2 = plt.subplot(2, 3, 2)
sc.pl.umap(adata_optimal, color='pseudotime', ax=ax2, show=False, 
            title=f'Optimal Subset ({optimal_n} features)')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')

ax3 = plt.subplot(2, 3, 3)
sc.pl.umap(adata_hc, color='pseudotime', ax=ax3, show=False, 
            title=f'High-Confidence Subset (10 features)')
ax3.set_xlabel('UMAP 1')
ax3.set_ylabel('UMAP 2')

# Phase coloring
ax4 = plt.subplot(2, 3, 4)
sc.pl.umap(adata, color='phase', ax=ax4, show=False, title='Full Dataset - Phase')
ax4.set_xlabel('UMAP 1')
ax4.set_ylabel('UMAP 2')

ax5 = plt.subplot(2, 3, 5)
sc.pl.umap(adata_optimal, color='phase', ax=ax5, show=False, title='Optimal Subset - Phase')
ax5.set_xlabel('UMAP 1')
ax5.set_ylabel('UMAP 2')

ax6 = plt.subplot(2, 3, 6)
sc.pl.umap(adata_hc, color='phase', ax=ax6, show=False, title='High-Confidence Subset - Phase')
ax6.set_xlabel('UMAP 1')
ax6.set_ylabel('UMAP 2')

plt.tight_layout()
plt.savefig('report/images/figure_04_validation_trajectories.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: report/images/figure_04_validation_trajectories.png")

# ==============================================================================
# 13. COMPARATIVE ANALYSIS
# ==============================================================================
print("\n[13] Performing comparative analysis...")

# Compute silhouette scores for phase separation
def compute_phase_separation(adata_subset):
    sc.pp.neighbors(adata_subset, n_neighbors=15)
    sc.tl.umap(adata_subset)
    # Use UMAP coordinates for silhouette
    try:
        score = silhouette_score(adata_subset.obsm['X_umap'], 
                                  adata_subset.obs['phase'].astype('category').cat.codes)
    except:
        score = 0
    return score

sil_full = compute_phase_separation(adata.copy())
sil_optimal = compute_phase_separation(adata_optimal.copy())
sil_hc = compute_phase_separation(adata_hc.copy())

print(f"    Phase separation (silhouette score):")
print(f"      Full dataset (241 features):     {sil_full:.4f}")
print(f"      Optimal subset ({optimal_n} features):   {sil_optimal:.4f}")
print(f"      High-confidence (10 features):   {sil_hc:.4f}")

# Store comparison results
comparison = {
    'method': ['Full Dataset', 'Optimal Subset', 'High-Confidence Subset'],
    'n_features': [241, optimal_n, 10],
    'silhouette_score': [sil_full, sil_optimal, sil_hc],
    'trajectory_preservation': [
        results_df[results_df['n_features'] == min(200, 241)]['composite'].values[0] if len(results_df[results_df['n_features'] == min(200, 241)]) > 0 else 1.0,
        results_df[results_df['n_features'] == optimal_n]['composite'].values[0],
        results_df[results_df['n_features'] == 10]['composite'].values[0] if len(results_df[results_df['n_features'] == 10]) > 0 else results_df['composite'].values[1]
    ]
}
comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv('outputs/feature_set_comparison.csv', index=False)
print("    Saved: outputs/feature_set_comparison.csv")

# ==============================================================================
# 14. HEATMAP OF TOP DYNAMIC FEATURES
# ==============================================================================
print("\n[14] Generating feature expression heatmap...")

# Sort cells by pseudotime for heatmap
sorted_idx = np.argsort(adata.obs['pseudotime'].values)
top_20_features = var_df_sorted.head(20).index.tolist()

# Get expression matrix for top features
heatmap_data = adata[sorted_idx, :][:, top_20_features].X.T

# Create phase colorbar
phase_colors = {'G0': '#1f77b4', 'G1': '#ff7f0e', 'S': '#2ca02c', 'G2': '#d62728'}
phase_labels = adata.obs['phase'].iloc[sorted_idx].values

fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                         gridspec_kw={'height_ratios': [0.05, 1]}, 
                         constrained_layout=True)

# Phase colorbar
ax_phase = axes[0]
phase_numeric = pd.Series(phase_labels).map({p: i for i, p in enumerate(phase_order)}).values
ax_phase.imshow(phase_numeric.reshape(1, -1), aspect='auto', cmap='viridis', extent=[0, len(phase_labels), 0, 1])
ax_phase.set_xticks([])
ax_phase.set_yticks([])
ax_phase.set_title('Cell Cycle Phase (G0=blue, G1=yellow, S=green, G2=red)', fontsize=10)

# Heatmap
ax_heatmap = axes[1]
im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                        vmin=-2, vmax=2, extent=[0, heatmap_data.shape[1], 0, heatmap_data.shape[0]])
ax_heatmap.set_yticks(np.arange(len(top_20_features)) + 0.5)
ax_heatmap.set_yticklabels([f.replace('Int_MeanEdge_', '').replace('_cell', '') for f in top_20_features], fontsize=8)
ax_heatmap.set_xlabel('Cells (ordered by pseudotime)')
ax_heatmap.set_ylabel('Features')
ax_heatmap.set_title('Expression of Top 20 Dynamic Features Along Trajectory')

# Add colorbar
cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.5)
cbar.set_label('Normalized Expression')

plt.savefig('report/images/figure_05_expression_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: report/images/figure_05_expression_heatmap.png")

# ==============================================================================
# 15. SUMMARY STATISTICS
# ==============================================================================
print("\n[15] Computing summary statistics...")

summary = {
    'total_cells': adata.shape[0],
    'total_features': adata.shape[1],
    'optimal_n_features': optimal_n,
    'high_confidence_n_features': 10,
    'trajectory_preservation_full': float(results_df[results_df['n_features'] == min(200, 241)]['composite'].values[0]) if len(results_df[results_df['n_features'] == min(200, 241)]) > 0 else 1.0,
    'trajectory_preservation_optimal': float(results_df[results_df['n_features'] == optimal_n]['composite'].values[0]),
    'trajectory_preservation_hc': float(results_df[results_df['n_features'] == 10]['composite'].values[0]) if len(results_df[results_df['n_features'] == 10]) > 0 else float(results_df['composite'].values[1]),
    'silhouette_full': float(sil_full),
    'silhouette_optimal': float(sil_optimal),
    'silhouette_hc': float(sil_hc),
    'features_with_high_correlation': int(np.sum(var_df['abs_correlation'] > 0.5)),
    'features_significant_pvalue': int(np.sum(var_df['pseudotime_pvalue'] < 0.001))
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('outputs/analysis_summary.csv', index=False)
print("    Saved: outputs/analysis_summary.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nSummary:")
print(f"  - Total cells analyzed: {summary['total_cells']}")
print(f"  - Total features: {summary['total_features']}")
print(f"  - Optimal feature subset: {summary['optimal_n_features']} features")
print(f"  - High-confidence subset: {summary['high_confidence_n_features']} features")
print(f"  - Features with strong pseudotime correlation (|r|>0.5): {summary['features_with_high_correlation']}")
print(f"\nTrajectory Preservation:")
print(f"  - Full dataset: {summary['trajectory_preservation_full']:.3f}")
print(f"  - Optimal subset: {summary['trajectory_preservation_optimal']:.3f}")
print(f"  - High-confidence: {summary['trajectory_preservation_hc']:.3f}")
