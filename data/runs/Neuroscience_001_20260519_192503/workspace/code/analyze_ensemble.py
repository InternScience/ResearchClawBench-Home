"""
Comprehensive analysis of DMN ensemble for Drosophila optic lobe motion pathway.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
params = np.load('outputs/ensemble_parameters.npz')
losses = np.load('outputs/validation_losses.npy')
with open('outputs/clustering_labels.json', 'r') as f:
    clustering_labels = json.load(f)

bias = params['bias']           # (50, 65)
time_const = params['time_const']  # (50, 65)
sign = params['sign']           # (50, 604)
syn_count = params['syn_count']  # (50, 2355)
syn_strength = params['syn_strength']  # (50, 604)

n_models = bias.shape[0]
n_nodes = bias.shape[1]
n_edges = syn_strength.shape[1]

# Cell type names (from clustering pickles)
cell_types = sorted(clustering_labels.keys())
print(f"Models: {n_models}, Nodes: {n_nodes}, Edges: {n_edges}")
print(f"Cell types: {len(cell_types)}")
print(f"Loss range: {losses.min():.4f} - {losses.max():.4f}")

# ============================================================================
# Figure 1: Validation loss distribution and model performance
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(losses, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(losses.mean(), color='red', linestyle='--', label=f'Mean: {losses.mean():.4f}')
ax.axvline(np.median(losses), color='green', linestyle='--', label=f'Median: {np.median(losses):.4f}')
ax.set_xlabel('Validation Loss (L2 norm)')
ax.set_ylabel('Number of Models')
ax.set_title('Distribution of Validation Loss Across Ensemble')
ax.legend()

ax = axes[1]
sorted_losses = np.sort(losses)
ranks = np.arange(1, len(sorted_losses) + 1)
ax.plot(ranks, sorted_losses, 'o-', color='steelblue', markersize=4)
ax.fill_between(ranks, sorted_losses, alpha=0.3, color='steelblue')
ax.set_xlabel('Model Rank')
ax.set_ylabel('Validation Loss')
ax.set_title('Ranked Validation Loss')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('report/images/figure1_validation_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure1_validation_loss.png")

# ============================================================================
# Figure 2: Parameter distributions across ensemble
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Bias distribution
ax = axes[0, 0]
for i in range(n_models):
    ax.hist(bias[i], bins=20, alpha=0.1, color='blue')
ax.hist(bias.flatten(), bins=30, alpha=0.6, color='blue', edgecolor='white')
ax.set_xlabel('Resting Potential (bias)')
ax.set_ylabel('Frequency')
ax.set_title(f'Resting Potential Distribution\n(mean={bias.mean():.3f}, std={bias.std():.3f})')

# Time constant distribution
ax = axes[0, 1]
ax.hist(time_const.flatten(), bins=30, alpha=0.6, color='green', edgecolor='white')
ax.set_xlabel('Time Constant (τ)')
ax.set_ylabel('Frequency')
ax.set_title(f'Time Constant Distribution\n(mean={time_const.mean():.3f}, std={time_const.std():.3f})')

# Synaptic strength distribution
ax = axes[0, 2]
nonzero_strength = syn_strength[syn_strength > 0]
ax.hist(nonzero_strength, bins=30, alpha=0.6, color='purple', edgecolor='white')
ax.set_xlabel('Synaptic Strength')
ax.set_ylabel('Frequency')
ax.set_title(f'Synaptic Strength Distribution\n(mean={nonzero_strength.mean():.4f}, std={nonzero_strength.std():.4f})')

# Bias variability per node
ax = axes[1, 0]
bias_std = bias.std(axis=0)
bias_mean = bias.mean(axis=0)
ax.scatter(bias_mean, bias_std, alpha=0.6, s=30)
ax.set_xlabel('Mean Resting Potential')
ax.set_ylabel('Std Resting Potential')
ax.set_title('Bias Variability Across Ensemble')

# Time constant variability per node
ax = axes[1, 1]
tc_std = time_const.std(axis=0)
tc_mean = time_const.mean(axis=0)
ax.scatter(tc_mean, tc_std, alpha=0.6, s=30, color='green')
ax.set_xlabel('Mean Time Constant')
ax.set_ylabel('Std Time Constant')
ax.set_title('Time Constant Variability Across Ensemble')

# Synaptic strength variability per edge
ax = axes[1, 2]
ss_std = syn_strength.std(axis=0)
ss_mean = syn_strength.mean(axis=0)
ax.scatter(ss_mean, ss_std, alpha=0.6, s=20, color='purple')
ax.set_xlabel('Mean Synaptic Strength')
ax.set_ylabel('Std Synaptic Strength')
ax.set_title('Synaptic Strength Variability Across Ensemble')

plt.tight_layout()
plt.savefig('report/images/figure2_parameter_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure2_parameter_distributions.png")

# ============================================================================
# Figure 3: Parameter correlations with performance
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Correlation between mean bias and loss
bias_corr = []
for i in range(n_nodes):
    r, p = stats.pearsonr(bias[:, i], losses)
    bias_corr.append(r)

ax = axes[0]
ax.bar(range(n_nodes), sorted(bias_corr), color='steelblue', alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Node (sorted by correlation)')
ax.set_ylabel('Pearson r (bias vs loss)')
ax.set_title('Bias-Loss Correlation per Node')

# Correlation between mean time constant and loss
tc_corr = []
for i in range(n_nodes):
    r, p = stats.pearsonr(time_const[:, i], losses)
    tc_corr.append(r)

ax = axes[1]
ax.bar(range(n_nodes), sorted(tc_corr), color='green', alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Node (sorted by correlation)')
ax.set_ylabel('Pearson r (τ vs loss)')
ax.set_title('Time Constant-Loss Correlation per Node')

# Correlation between mean synaptic strength and loss
ss_corr = []
for i in range(n_edges):
    r, p = stats.pearsonr(syn_strength[:, i], losses)
    ss_corr.append(r)

ax = axes[2]
ax.bar(range(n_edges), sorted(ss_corr), color='purple', alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Edge (sorted by correlation)')
ax.set_ylabel('Pearson r (strength vs loss)')
ax.set_title('Synaptic Strength-Loss Correlation per Edge')

plt.tight_layout()
plt.savefig('report/images/figure3_parameter_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure3_parameter_correlations.png")

# ============================================================================
# Figure 4: Excitatory vs Inhibitory balance
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Sign distribution (should be same across models)
sign_unique = sign[0]
exc_count = (sign_unique == 1).sum()
inh_count = (sign_unique == -1).sum()

ax = axes[0]
ax.bar(['Inhibitory', 'Excitatory'], [inh_count, exc_count], color=['red', 'blue'], alpha=0.7, edgecolor='white')
ax.set_ylabel('Number of Connections')
ax.set_title(f'Synapse Polarity Distribution\n(Inh: {inh_count}, Exc: {exc_count})')

# Mean synaptic strength by sign
exc_strengths = syn_strength[:, sign_unique == 1].flatten()
inh_strengths = syn_strength[:, sign_unique == -1].flatten()

ax = axes[1]
ax.hist(exc_strengths, bins=30, alpha=0.5, label='Excitatory', color='blue')
ax.hist(inh_strengths, bins=30, alpha=0.5, label='Inhibitory', color='red')
ax.set_xlabel('Synaptic Strength')
ax.set_ylabel('Frequency')
ax.set_title('Synaptic Strength by Polarity')
ax.legend()

# Mean excitatory vs inhibitory strength per model
exc_mean = syn_strength[:, sign_unique == 1].mean(axis=1)
inh_mean = syn_strength[:, sign_unique == -1].mean(axis=1)

ax = axes[2]
ax.scatter(exc_mean, inh_mean, alpha=0.6, s=50)
ax.plot([0, max(exc_mean.max(), inh_mean.max())], [0, max(exc_mean.max(), inh_mean.max())], 'k--', alpha=0.3)
ax.set_xlabel('Mean Excitatory Strength')
ax.set_ylabel('Mean Inhibitory Strength')
ax.set_title('E/I Balance per Model')

plt.tight_layout()
plt.savefig('report/images/figure4_ei_balance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure4_ei_balance.png")

# ============================================================================
# Figure 5: Cell type clustering consistency
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Compute cluster label entropy for each cell type
from scipy.stats import entropy

entropies = {}
for ct, labels in clustering_labels.items():
    labels_arr = np.array(labels)
    # Filter out invalid labels (-99999)
    valid = labels_arr != -99999
    if valid.sum() > 0:
        valid_labels = labels_arr[valid]
        counts = np.bincount(valid_labels - valid_labels.min())
        probs = counts / counts.sum()
        entropies[ct] = entropy(probs)
    else:
        entropies[ct] = 0

ax = axes[0]
sorted_cts = sorted(entropies.keys(), key=lambda x: entropies[x])
sorted_ents = [entropies[ct] for ct in sorted_cts]
bars = ax.barh(range(len(sorted_cts)), sorted_ents, color='coral', alpha=0.7)
ax.set_yticks(range(0, len(sorted_cts), 5))
ax.set_yticklabels([sorted_cts[i] for i in range(0, len(sorted_cts), 5)], fontsize=6)
ax.set_xlabel('Cluster Entropy (nats)')
ax.set_title('Clustering Variability per Cell Type\n(Lower = More Consistent)')

# Number of dominant clusters per cell type
dominant_clusters = {}
for ct, labels in clustering_labels.items():
    labels_arr = np.array(labels)
    valid = labels_arr != -99999
    if valid.sum() > 0:
        valid_labels = labels_arr[valid]
        counts = np.bincount(valid_labels - valid_labels.min())
        # Number of clusters with >10% of models
        n_dominant = (counts / counts.sum() > 0.1).sum()
        dominant_clusters[ct] = n_dominant
    else:
        dominant_clusters[ct] = 0

ax = axes[1]
sorted_cts2 = sorted(dominant_clusters.keys(), key=lambda x: dominant_clusters[x])
sorted_dom = [dominant_clusters[ct] for ct in sorted_cts2]
ax.barh(range(len(sorted_cts2)), sorted_dom, color='teal', alpha=0.7)
ax.set_yticks(range(0, len(sorted_cts2), 5))
ax.set_yticklabels([sorted_cts2[i] for i in range(0, len(sorted_cts2), 5)], fontsize=6)
ax.set_xlabel('Number of Dominant Clusters')
ax.set_title('Functional Diversity per Cell Type\n(>10% threshold)')

plt.tight_layout()
plt.savefig('report/images/figure5_clustering_consistency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure5_clustering_consistency.png")

# ============================================================================
# Figure 6: Synapse count vs learned strength relationship
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Since syn_count and syn_strength have different shapes,
# we need to find the mapping. The syn_count for actual edges might be
# those with highest values. Let's find top 604 syn_count values.
# But actually, the network likely uses syn_count as a fixed feature
# and syn_strength as a learnable scale.

# Let's compute the "effective weight" as syn_count * syn_strength_expanded
# Since we don't have exact mapping, let's look at correlation between
# mean syn_count across all entries and mean syn_strength

mean_syn_count = syn_count.mean(axis=0)
mean_syn_strength = syn_strength.mean(axis=0)

ax = axes[0]
# Distribution of synapse counts
ax.hist(mean_syn_count, bins=40, alpha=0.6, color='orange', edgecolor='white')
ax.set_xlabel('Mean Log Synapse Count')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Synapse Counts\n(All Possible Connections)')

# For syn_strength, show relationship with performance
ax = axes[1]
# Sort edges by mean strength and show top performers
top_edge_idx = np.argsort(mean_syn_strength)[-50:]
bottom_edge_idx = np.argsort(mean_syn_strength)[:50]

ax.barh(range(50), mean_syn_strength[top_edge_idx], color='green', alpha=0.7, label='Top 50')
ax.barh(range(50, 100), mean_syn_strength[bottom_edge_idx], color='red', alpha=0.7, label='Bottom 50')
ax.set_xlabel('Mean Synaptic Strength')
ax.set_title('Strongest vs Weakest Connections')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/figure6_synapse_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure6_synapse_analysis.png")

# ============================================================================
# Figure 7: Parameter heatmaps
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Bias heatmap
ax = axes[0, 0]
im = ax.imshow(bias.T, aspect='auto', cmap='RdBu_r', vmin=bias.min(), vmax=bias.max())
ax.set_xlabel('Model Index')
ax.set_ylabel('Node Index')
ax.set_title('Resting Potential (bias) Across Ensemble')
plt.colorbar(im, ax=ax)

# Time constant heatmap
ax = axes[0, 1]
im = ax.imshow(time_const.T, aspect='auto', cmap='viridis', vmin=time_const.min(), vmax=time_const.max())
ax.set_xlabel('Model Index')
ax.set_ylabel('Node Index')
ax.set_title('Time Constant (τ) Across Ensemble')
plt.colorbar(im, ax=ax)

# Synaptic strength heatmap
ax = axes[1, 0]
im = ax.imshow(syn_strength.T, aspect='auto', cmap='plasma', vmin=0, vmax=syn_strength.max())
ax.set_xlabel('Model Index')
ax.set_ylabel('Edge Index')
ax.set_title('Synaptic Strength Across Ensemble')
plt.colorbar(im, ax=ax)

# Synapse count heatmap
ax = axes[1, 1]
im = ax.imshow(syn_count.T, aspect='auto', cmap='YlOrRd')
ax.set_xlabel('Model Index')
ax.set_ylabel('Connection Index')
ax.set_title('Log Synapse Count Across Ensemble')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('report/images/figure7_parameter_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure7_parameter_heatmaps.png")

# ============================================================================
# Figure 8: PCA of ensemble parameters
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Flatten parameters per model
param_matrix = np.concatenate([
    bias,
    time_const,
    syn_strength
], axis=1)

pca = PCA(n_components=5)
pca_proj = pca.fit_transform(param_matrix)

ax = axes[0]
scatter = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=losses, cmap='RdYlBu_r', s=60, alpha=0.8)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA of Model Parameters (colored by loss)')
plt.colorbar(scatter, ax=ax, label='Validation Loss')

ax = axes[1]
ax.bar(range(1, 6), pca.explained_variance_ratio_ * 100, color='steelblue', alpha=0.7)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('PCA Explained Variance')

plt.tight_layout()
plt.savefig('report/images/figure8_pca_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure8_pca_analysis.png")

# ============================================================================
# Save summary statistics
# ============================================================================
summary = {
    'n_models': int(n_models),
    'n_nodes': int(n_nodes),
    'n_edges': int(n_edges),
    'n_cell_types': len(cell_types),
    'loss_mean': float(losses.mean()),
    'loss_std': float(losses.std()),
    'loss_min': float(losses.min()),
    'loss_max': float(losses.max()),
    'bias_mean': float(bias.mean()),
    'bias_std': float(bias.std()),
    'time_const_mean': float(time_const.mean()),
    'time_const_std': float(time_const.std()),
    'syn_strength_mean': float(syn_strength[syn_strength > 0].mean()),
    'syn_strength_std': float(syn_strength[syn_strength > 0].std()),
    'excitatory_edges': int(exc_count),
    'inhibitory_edges': int(inh_count),
    'excitatory_mean_strength': float(exc_strengths.mean()),
    'inhibitory_mean_strength': float(inh_strengths.mean()),
    'pca_explained_variance': [float(v) for v in pca.explained_variance_ratio_],
}

with open('outputs/summary_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nAnalysis complete. Summary statistics saved to outputs/summary_stats.json")
print(json.dumps(summary, indent=2))
