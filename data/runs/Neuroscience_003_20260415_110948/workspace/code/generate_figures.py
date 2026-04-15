#!/usr/bin/env python3
"""
Generate all figures for the research report.
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Neuroscience_003_20260415_110948'
DATA_PATH = os.path.join(WORKSPACE, 'data', 'adata_RPE.h5ad')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
REPORT_IMG_DIR = os.path.join(WORKSPACE, 'report', 'images')

os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# Load data
adata = sc.read_h5ad(DATA_PATH)
X = adata.X.copy()
age = adata.obs['annotated_age'].values
phase = adata.obs['phase'].values
state = adata.obs['state'].values
batch = adata.obs['batch'].values
feature_names = list(adata.var_names)

# Load feature scores
feature_scores_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_scores.csv'))
with open(os.path.join(OUTPUT_DIR, 'selection_results.json'), 'r') as f:
    selection_results = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'optimal_features.json'), 'r') as f:
    optimal_info = json.load(f)

optimal_k = optimal_info['k']
optimal_features = optimal_info['features']
optimal_indices = [feature_names.index(f) for f in optimal_features]

# ============================================================
# FIGURE 1: Data Overview - Feature distribution & cell state landscape
# ============================================================
print("Generating Figure 1: Data Overview...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Data Overview: Single-Cell Protein Imaging of RPE Cells', fontsize=16, fontweight='bold')

# 1a. Phase distribution
ax = axes[0, 0]
phase_counts = adata.obs['phase'].value_counts()
colors_phase = {'G0': '#8B5CF6', 'G1': '#3B82F6', 'S': '#10B981', 'G2': '#F59E0B'}
bars = ax.bar(phase_counts.index, phase_counts.values, 
              color=[colors_phase.get(p, '#999') for p in phase_counts.index], edgecolor='white')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
ax.set_title('Cell Cycle Phase Distribution', fontsize=12)
ax.set_ylabel('Cell Count')
ax.set_xlabel('Phase')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 1b. State distribution
ax = axes[0, 1]
state_counts = adata.obs['state'].value_counts()
colors_state = {'cycling': '#10B981', 'arrested': '#EF4444', 'nan': '#9CA3AF'}
bars = ax.bar(state_counts.index, state_counts.values,
              color=[colors_state.get(s, '#999') for s in state_counts.index], edgecolor='white')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
ax.set_title('Cell State Distribution', fontsize=12)
ax.set_ylabel('Cell Count')
ax.set_xlabel('State')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 1c. Annotated age distribution
ax = axes[0, 2]
ax.hist(age, bins=50, color='#6366F1', edgecolor='white', alpha=0.8)
ax.axvline(age.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={age.mean():.1f}')
ax.set_title('Annotated Age (Trajectory Variable)', fontsize=12)
ax.set_xlabel('Age')
ax.set_ylabel('Cell Count')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 1d. Batch distribution
ax = axes[1, 0]
batch_counts = adata.obs['batch'].value_counts()
bars = ax.bar(batch_counts.index, batch_counts.values,
              color=['#3B82F6', '#F59E0B'], edgecolor='white')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
ax.set_title('Batch Distribution', fontsize=12)
ax.set_ylabel('Cell Count')
ax.set_xlabel('Batch')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 1e. Top 5 features violin plot by phase
ax = axes[1, 1]
top5_features = feature_scores_df.head(5)['feature'].values
df_violin = pd.DataFrame(X[:, [feature_names.index(f) for f in top5_features]], columns=top5_features)
df_violin['phase'] = phase
df_melt = df_violin.melt(id_vars='phase', var_name='Feature', value_name='Expression')
sns.violinplot(data=df_melt, x='Feature', y='Expression', hue='phase', ax=ax, 
               palette=colors_phase, inner='quartile', cut=0, legend=False)
ax.set_title('Top 5 Dynamic Features by Phase', fontsize=12)
ax.set_ylabel('Expression')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 1f. Feature expression heatmap (top 15 features, subsampled cells)
ax = axes[1, 2]
np.random.seed(42)
sample_idx = np.random.choice(adata.n_obs, size=min(500, adata.n_obs), replace=False)
top15_features = feature_scores_df.head(15)['feature'].values
top15_indices = [feature_names.index(f) for f in top15_features]
X_sub = X[np.ix_(sample_idx, top15_indices)]

# Sort by age
sort_idx = np.argsort(age[sample_idx])
X_sub_sorted = X_sub[sort_idx]
age_sorted = age[sample_idx][sort_idx]
phase_sorted = phase[sample_idx][sort_idx]

im = ax.imshow(X_sub_sorted.T, aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_yticks(range(len(top15_features)))
short_names = [f.replace('Int_Med_', '').replace('Int_Intg_', '').replace('AreaShape_', '') 
               for f in top15_features]
ax.set_yticklabels([n[:20] for n in short_names], fontsize=7)
ax.set_xlabel('Cells (sorted by age)')
ax.set_title('Heatmap: Top 15 Features\n(cells sorted by trajectory age)', fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig1_data_overview.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig1_data_overview.png")

# ============================================================
# FIGURE 2: Feature Dynamics Analysis
# ============================================================
print("Generating Figure 2: Feature Dynamics Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Dynamics: Identifying Trajectory-Informative Markers', fontsize=16, fontweight='bold')

# 2a. Age correlation vs Kruskal-Wallis statistic scatter
ax = axes[0, 0]
sc = ax.scatter(np.abs(feature_scores_df['age_spearman_corr']), 
                feature_scores_df['kw_statistic'],
                c=feature_scores_df['dynamic_score'], cmap='YlOrRd', s=30, alpha=0.7, edgecolors='none')
# Highlight top features
top20 = feature_scores_df.head(20)
ax.scatter(np.abs(top20['age_spearman_corr']), top20['kw_statistic'],
           c='red', s=80, marker='*', zorder=5, label='Top 20')
for _, row in top20.head(8).iterrows():
    short_name = row['feature'].replace('Int_Med_', '').replace('Int_Intg_', '').replace('Int_Std_', '').replace('AreaShape_', '')[:15]
    ax.annotate(short_name, (abs(row['age_spearman_corr']), row['kw_statistic']),
                fontsize=7, textcoords="offset points", xytext=(5, 5))
ax.set_xlabel('Absolute Spearman Correlation with Age')
ax.set_ylabel('Kruskal-Wallis H Statistic (across phases)')
ax.set_title('Feature Dynamics: Correlation vs Phase-Dependence', fontsize=12)
ax.legend(fontsize=9)
plt.colorbar(sc, ax=ax, label='Dynamic Score', fraction=0.046, pad=0.04)

# 2b. Dynamic score distribution
ax = axes[0, 1]
ax.hist(feature_scores_df['dynamic_score'], bins=50, color='#6366F1', edgecolor='white', alpha=0.8)
threshold_line = feature_scores_df.iloc[optimal_k-1]['dynamic_score']
ax.axvline(threshold_line, color='red', linestyle='--', linewidth=2, label=f'Top {optimal_k} threshold={threshold_line:.3f}')
ax.set_xlabel('Composite Dynamic Score')
ax.set_ylabel('Feature Count')
ax.set_title(f'Dynamic Score Distribution (selecting top {optimal_k})', fontsize=12)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 2c. Top 15 features: mean expression across phases
ax = axes[1, 0]
top15 = feature_scores_df.head(15)['feature'].values
top15_indices = [feature_names.index(f) for f in top15]
phase_order = ['G0', 'G1', 'S', 'G2']
mean_expr = np.zeros((len(top15), len(phase_order)))
for i, idx in enumerate(top15_indices):
    for j, p in enumerate(phase_order):
        mean_expr[i, j] = X[phase == p, idx].mean()

im = ax.imshow(mean_expr, aspect='auto', cmap='coolwarm', vmin=-mean_expr.std()*2, vmax=mean_expr.std()*2)
ax.set_yticks(range(len(top15)))
short_names = [f.replace('Int_Med_', '').replace('Int_Intg_', '').replace('Int_Std_', '').replace('AreaShape_', '').replace('Int_MeanEdge_', '') for f in top15]
ax.set_yticklabels([n[:18] for n in short_names], fontsize=8)
ax.set_xticks(range(len(phase_order)))
ax.set_xticklabels(phase_order)
ax.set_xlabel('Cell Cycle Phase')
ax.set_title('Mean Expression of Top 15 Features by Phase', fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 2d. Age-binned expression profiles for top 8 features
ax = axes[1, 1]
top8 = feature_scores_df.head(8)['feature'].values
top8_indices = [feature_names.index(f) for f in top8]
age_bins = pd.cut(age, bins=10)
unique_bins = age_bins.categories

for idx, feat_idx in enumerate(top8_indices):
    binned_means = []
    for b_idx, bin_interval in enumerate(unique_bins):
        mask = (age_bins == bin_interval)
        if mask.sum() > 0:
            binned_means.append(X[mask, feat_idx].mean())
        else:
            binned_means.append(np.nan)
    
    short_name = top8[idx].replace('Int_Med_', '').replace('Int_Intg_', '').replace('Int_Std_', '').replace('AreaShape_', '')[:15]
    ax.plot(range(len(unique_bins)), binned_means, 'o-', label=short_name, linewidth=2, markersize=6)

ax.set_xlabel('Age Bin')
ax.set_ylabel('Mean Expression')
ax.set_title('Age-Binned Expression Profiles (Top 8 Features)', fontsize=12)
ax.legend(fontsize=7, loc='upper left', ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig2_feature_dynamics.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig2_feature_dynamics.png")

# ============================================================
# FIGURE 3: Trajectory Preservation Comparison
# ============================================================
print("Generating Figure 3: Trajectory Preservation Comparison...")

# Compute embeddings for full features, optimal subset, and random subset
def compute_embedding(feature_indices, perplexity=30):
    X_sub = X[:, feature_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    pca = PCA(n_components=min(30, len(feature_indices)))
    X_pca = pca.fit_transform(X_scaled)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    return tsne.fit_transform(X_pca[:, :min(15, X_pca.shape[1])])

all_indices = list(range(X.shape[1]))
np.random.seed(42)
random_indices = np.random.choice(all_indices, size=optimal_k, replace=False).tolist()

print("  Computing full-feature embedding...")
emb_full = compute_embedding(all_indices)
print("  Computing optimal-feature embedding...")
emb_optimal = compute_embedding(optimal_indices)
print("  Computing random-feature embedding...")
emb_random = compute_embedding(random_indices)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Trajectory Preservation: Embedding Comparison', fontsize=16, fontweight='bold')

# Color maps
cmap_age = plt.cm.viridis
cmap_phase = colors_phase

# 3a. Full features colored by age
ax = axes[0]
sc = ax.scatter(emb_full[:, 0], emb_full[:, 1], c=age, cmap=cmap_age, s=8, alpha=0.6)
corr_full, _ = stats.spearmanr(emb_full[:, 0], age)
sil_full = silhouette_score(emb_full, phase)
ax.set_title(f'All {X.shape[1]} Features\nAge corr={corr_full:+.3f}, Silhouette={sil_full:.3f}', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.colorbar(sc, ax=ax, label='Age', fraction=0.046, pad=0.04)

# 3b. Optimal features colored by age
ax = axes[1]
sc = ax.scatter(emb_optimal[:, 0], emb_optimal[:, 1], c=age, cmap=cmap_age, s=8, alpha=0.6)
corr_opt, _ = stats.spearmanr(emb_optimal[:, 0], age)
sil_opt = silhouette_score(emb_optimal, phase)
ax.set_title(f'Top {optimal_k} Dynamic Features\nAge corr={corr_opt:+.3f}, Silhouette={sil_opt:.3f}', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.colorbar(sc, ax=ax, label='Age', fraction=0.046, pad=0.04)

# 3c. Random features colored by age
ax = axes[2]
sc = ax.scatter(emb_random[:, 0], emb_random[:, 1], c=age, cmap=cmap_age, s=8, alpha=0.6)
corr_rand, _ = stats.spearmanr(emb_random[:, 0], age)
sil_rand = silhouette_score(emb_random, phase)
ax.set_title(f'{optimal_k} Random Features\nAge corr={corr_rand:+.3f}, Silhouette={sil_rand:.3f}', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.colorbar(sc, ax=ax, label='Age', fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig3_trajectory_preservation.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig3_trajectory_preservation.png")

# ============================================================
# FIGURE 4: Phase-colored embeddings & k-ablation curve
# ============================================================
print("Generating Figure 4: Phase Structure & Ablation Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Cell Cycle Phase Structure and Feature Selection Ablation', fontsize=16, fontweight='bold')

# 4a. Optimal embedding colored by phase
ax = axes[0]
phase_colors_arr = [colors_phase.get(p, '#999') for p in phase]
for p in phase_order:
    mask = phase == p
    ax.scatter(emb_optimal[mask, 0], emb_optimal[mask, 1], 
               c=colors_phase[p], s=8, alpha=0.6, label=p)
ax.set_title(f'Phase Structure (Top {optimal_k} Features)', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(fontsize=10)

# 4b. Optimal embedding colored by state
ax = axes[1]
state_colors = {'cycling': '#10B981', 'arrested': '#EF4444', 'nan': '#9CA3AF'}
for s in ['cycling', 'arrested', 'nan']:
    mask = state == s
    ax.scatter(emb_optimal[mask, 0], emb_optimal[mask, 1],
               c=state_colors[s], s=8, alpha=0.6, label=s)
ax.set_title(f'State Structure (Top {optimal_k} Features)', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(fontsize=10)

# 4c. Ablation curve: k vs metrics
ax = axes[2]
k_vals = list(selection_results.keys())
k_ints = [int(k) for k in k_vals]
age_corrs = [abs(selection_results[k]['tsne_age_corr_1']) for k in k_vals]
sil_phases = [selection_results[k]['silhouette_phase'] for k in k_vals]

ax.plot(k_ints, age_corrs, 'o-', color='#6366F1', linewidth=2, markersize=8, label='|Age Correlation|')
ax.plot(k_ints, sil_phases, 's-', color='#F59E0B', linewidth=2, markersize=8, label='Silhouette (Phase)')
ax.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Selected k={optimal_k}')
ax.set_xlabel('Number of Selected Features (k)')
ax.set_ylabel('Score')
ax.set_title('Ablation: Feature Count vs Trajectory Quality', fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig4_phase_ablation.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig4_phase_ablation.png")

# ============================================================
# FIGURE 5: Biological Interpretation of Selected Features
# ============================================================
print("Generating Figure 5: Biological Interpretation...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Biological Interpretation of Selected Dynamic Features', fontsize=16, fontweight='bold')

# Categorize selected features
cell_cycle_markers = ['cycA', 'cycB1', 'cycD1', 'cycE', 'CDK2', 'Cdt1', 'Skp2', 'PCNA', 'DNA']
checkpoint_stress = ['pH2AX', 'pCHK1', 'p53', 'pp53', 'p21', 'pp21', 'p27']
signaling = ['AKT', 'ERK', 'S6', 'STAT3', 'RSK', 'GSK3b', 'YAP', 'bCat']
regulatory = ['RB', 'pRB', 'E2F1', 'Fra1', 'cJun', 'Cdh1', 'p16']

selected_set = set(optimal_features)

def count_category(category_list, selected_set):
    count = 0
    for feat in selected_set:
        for marker in category_list:
            if marker in feat:
                count += 1
                break
    return count

categories = ['Cell Cycle\nMarkers', 'Checkpoint/\nStress', 'Signaling\nPathways', 'Regulatory\nFactors']
cat_lists = [cell_cycle_markers, checkpoint_stress, signaling, regulatory]
counts = [count_category(cl, selected_set) for cl in cat_lists]

# 5a. Category pie chart
ax = axes[0, 0]
colors_cat = ['#6366F1', '#EF4444', '#10B981', '#F59E0B']
wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1d', 
                                    colors=colors_cat, startangle=90, textprops={'fontsize': 10})
for t in autotexts:
    t.set_fontweight('bold')
ax.set_title('Functional Categories of Selected Features', fontsize=12)

# 5b. Top 10 features: correlation with age
ax = axes[0, 1]
top10 = feature_scores_df.head(10)
short_names = [f.replace('Int_Med_', '').replace('Int_Intg_', '').replace('Int_Std_', '').replace('AreaShape_', '').replace('Int_MeanEdge_', '') for f in top10['feature']]
colors_bar = ['#6366F1' if c > 0 else '#EF4444' for c in top10['age_spearman_corr']]
bars = ax.barh(range(len(top10)), top10['age_spearman_corr'], color=colors_bar, edgecolor='white')
ax.set_yticks(range(len(top10)))
ax.set_yticklabels([n[:22] for n in short_names], fontsize=9)
ax.set_xlabel('Spearman Correlation with Age')
ax.set_title('Top 10 Features: Age Correlation', fontsize=12)
ax.axvline(0, color='black', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 5c. Phase-specific expression of key cell cycle features
ax = axes[1, 0]
key_features = ['Int_Med_cycA_nuc', 'Int_Med_CDK2_nuc', 'Int_Med_cycB1_nuc', 'Int_Med_DNA_cell']
key_indices = [feature_names.index(f) for f in key_features]
key_short = ['Cyclin A', 'CDK2', 'Cyclin B1', 'DNA Content']

x_pos = np.arange(len(phase_order))
width = 0.2
for i, (feat_idx, label) in enumerate(zip(key_indices, key_short)):
    means = [X[phase == p, feat_idx].mean() for p in phase_order]
    stds = [X[phase == p, feat_idx].std() for p in phase_order]
    ax.bar(x_pos + i*width, means, width, label=label, yerr=stds, capsize=3, alpha=0.8)

ax.set_xticks(x_pos + width*1.5)
ax.set_xticklabels(phase_order)
ax.set_ylabel('Mean Expression')
ax.set_xlabel('Cell Cycle Phase')
ax.set_title('Key Cell Cycle Features Across Phases', fontsize=12)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 5d. Feature importance via permutation-style analysis
ax = axes[1, 1]
# Use variance explained by each feature individually
var_explained = []
for idx in optimal_indices:
    lr = LinearRegression()
    lr.fit(X[:, idx:idx+1], age)
    pred = lr.predict(X[:, idx:idx+1])
    ss_res = np.sum((age - pred) ** 2)
    ss_tot = np.sum((age - age.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    var_explained.append(max(r2, 0))

top15_opt_idx = np.argsort(var_explained)[-15:]
top15_var = [var_explained[i] for i in top15_opt_idx]
top15_names = [optimal_features[i].replace('Int_Med_', '').replace('Int_Intg_', '').replace('Int_Std_', '').replace('AreaShape_', '')[:18] for i in top15_opt_idx]

colors_var = plt.cm.viridis(np.linspace(0.3, 0.9, len(top15_var)))
ax.barh(range(len(top15_var)), top15_var, color=colors_var, edgecolor='white')
ax.set_yticks(range(len(top15_var)))
ax.set_yticklabels(top15_names, fontsize=9)
ax.set_xlabel('Variance Explained (R²)')
ax.set_title('Individual Feature Importance for Age Prediction', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig5_biological_interpretation.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig5_biological_interpretation.png")

# ============================================================
# FIGURE 6: Batch Effect Assessment
# ============================================================
print("Generating Figure 6: Batch Effect Assessment...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Batch Effect Assessment', fontsize=16, fontweight='bold')

# 6a. Full features colored by batch
ax = axes[0]
batch_colors = {'1': '#3B82F6', '2': '#F59E0B'}
for b in ['1', '2']:
    mask = batch == b
    ax.scatter(emb_full[mask, 0], emb_full[mask, 1],
               c=batch_colors[b], s=8, alpha=0.5, label=f'Batch {b}')
ax.set_title(f'Full Features ({X.shape[1]})', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(fontsize=10)

# 6b. Optimal features colored by batch
ax = axes[1]
for b in ['1', '2']:
    mask = batch == b
    ax.scatter(emb_optimal[mask, 0], emb_optimal[mask, 1],
               c=batch_colors[b], s=8, alpha=0.5, label=f'Batch {b}')
ax.set_title(f'Optimal Features ({optimal_k})', fontsize=12)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(fontsize=10)

# 6c. Batch mixing score comparison
ax = axes[2]
# Calculate batch mixing using local neighborhood entropy
from sklearn.neighbors import NearestNeighbors

def batch_mixing_score(embedding, batch_labels, n_neighbors=15):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    scores = []
    for i in range(len(embedding)):
        neighbor_batches = batch_labels[indices[i, 1:]]
        # Entropy-based mixing score
        unique, counts = np.unique(neighbor_batches, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(unique))
        scores.append(entropy / max_entropy if max_entropy > 0 else 0)
    return np.mean(scores)

mix_full = batch_mixing_score(emb_full, batch)
mix_optimal = batch_mixing_score(emb_optimal, batch)
mix_random = batch_mixing_score(emb_random, batch)

methods = ['Full\nFeatures', f'Optimal\n({optimal_k} features)', 'Random\nFeatures']
mix_scores = [mix_full, mix_optimal, mix_random]
colors_mix = ['#3B82F6', '#10B981', '#9CA3AF']

bars = ax.bar(methods, mix_scores, color=colors_mix, edgecolor='white', width=0.5)
for bar, score in zip(bars, mix_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Batch Mixing Score (0-1)')
ax.set_title('Batch Mixing Comparison\n(higher = better mixing)', fontsize=12)
ax.set_ylim(0, max(mix_scores) + 0.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig6_batch_effect.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  -> Saved fig6_batch_effect.png")

# ============================================================
# Save comparison table
# ============================================================
comparison_table = {
    "full_features": {
        "n_features": int(X.shape[1]),
        "age_correlation": float(corr_full),
        "silhouette_phase": float(sil_full),
        "batch_mixing": float(mix_full)
    },
    "optimal_features": {
        "n_features": int(optimal_k),
        "features": optimal_features,
        "age_correlation": float(corr_opt),
        "silhouette_phase": float(sil_opt),
        "batch_mixing": float(mix_optimal)
    },
    "random_features": {
        "n_features": int(optimal_k),
        "age_correlation": float(corr_rand),
        "silhouette_phase": float(sil_rand),
        "batch_mixing": float(mix_random)
    }
}

with open(os.path.join(OUTPUT_DIR, 'comparison_table.json'), 'w') as f:
    json.dump(comparison_table, f, indent=2)

print("\nAll figures generated successfully!")
print(f"Figures saved to: {REPORT_IMG_DIR}")
