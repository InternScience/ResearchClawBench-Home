"""
Visualization and analysis script for KA-GNN results.
Generates all figures for the report.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_000_20260416_104049'
OUT_DIR = os.path.join(ROOT, 'outputs')
IMG_DIR = os.path.join(ROOT, 'report', 'images')
os.makedirs(IMG_DIR, exist_ok=True)

# Load results
with open(os.path.join(OUT_DIR, 'results.json')) as f:
    results = json.load(f)
with open(os.path.join(OUT_DIR, 'curves.json')) as f:
    curves = json.load(f)
with open(os.path.join(OUT_DIR, 'param_counts.json')) as f:
    param_counts = json.load(f)

DATASETS = ['bace', 'bbbp', 'clintox', 'hiv', 'muv']
MODELS = ['GCN', 'GAT', 'KA-GCN', 'KA-GAT']

# ============ Figure 1: Main comparison bar chart ============
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(DATASETS))
width = 0.18
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for i, model in enumerate(MODELS):
    means = [results[ds][model]['test_auc_mean'] for ds in DATASETS]
    stds = [results[ds][model]['test_auc_std'] for ds in DATASETS]
    bars = ax.bar(x + i*width, means, width, yerr=stds, label=model, 
                  color=colors[i], capsize=3, alpha=0.85)

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Test ROC-AUC', fontsize=12)
ax.set_title('KA-GNN vs Baseline GNN Models: Molecular Property Prediction', fontsize=14)
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(DATASETS, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'main_comparison.png'), dpi=150)
plt.close()

# ============ Figure 2: Heatmap of AUC values ============
heatmap_data = []
for ds in DATASETS:
    row = [results[ds][model]['test_auc_mean'] for model in MODELS]
    heatmap_data.append(row)

heatmap_array = np.array(heatmap_data)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(heatmap_array, annot=True, fmt='.3f', xticklabels=MODELS, 
            yticklabels=DATASETS, cmap='YlOrRd', vmin=0.5, vmax=1.0,
            ax=ax, linewidths=0.5)
ax.set_title('Test ROC-AUC Heatmap Across Datasets and Models', fontsize=13)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Dataset', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'auc_heatmap.png'), dpi=150)
plt.close()

# ============ Figure 3: Training curves ============
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    for model in MODELS:
        # Average across runs
        all_val_aucs = curves[ds][model]['val_aucs']
        max_len = max(len(run) for run in all_val_aucs)
        avg_aucs = []
        for ep in range(max_len):
            vals = [run[ep] for run in all_val_aucs if ep < len(run)]
            avg_aucs.append(np.mean(vals))
        ax.plot(range(len(avg_aucs)), avg_aucs, label=model, color=colors[MODELS.index(model)])
    ax.set_title(ds.upper(), fontsize=11)
    ax.set_xlabel('Epoch', fontsize=9)
    if idx == 0: ax.set_ylabel('Val ROC-AUC', fontsize=9)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
plt.suptitle('Validation AUC Training Curves', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Figure 4: Parameter count comparison ============
fig, ax = plt.subplots(figsize=(6, 4))
models_list = list(param_counts.keys())
params = list(param_counts.values())
bars = ax.bar(models_list, params, color=colors, alpha=0.85)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Number of Parameters', fontsize=12)
ax.set_title('Model Complexity Comparison', fontsize=13)
for bar, p in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
            f'{p:,}', ha='center', va='bottom', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'param_comparison.png'), dpi=150)
plt.close()

# ============ Figure 5: Per-run variance box plot ============
fig, ax = plt.subplots(figsize=(10, 6))
data_for_boxplot = []
labels_for_boxplot = []
for ds in DATASETS:
    for model in MODELS:
        per_run = results[ds][model]['per_run_test']
        data_for_boxplot.append(per_run)
        labels_for_boxplot.append(f'{ds}-{model}')

bp = ax.boxplot(data_for_boxplot, patch_artist=True, widths=0.6)
for i, patch in enumerate(bp['boxes']):
    model_idx = i % 4
    patch.set_facecolor(colors[model_idx])
    patch.set_alpha(0.7)

ax.set_xticklabels(labels_for_boxplot, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Test ROC-AUC', fontsize=12)
ax.set_title('Per-Run Performance Distribution', fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'per_run_boxplot.png'), dpi=150)
plt.close()

# ============ Figure 6: Relative performance improvement ============
fig, ax = plt.subplots(figsize=(8, 5))
baseline_models = ['GCN', 'GAT']
ka_models = ['KA-GCN', 'KA-GAT']

for bm, km in zip(baseline_models, ka_models):
    improvements = []
    for ds in DATASETS:
        baseline_auc = results[ds][bm]['test_auc_mean']
        ka_auc = results[ds][km]['test_auc_mean']
        improvements.append((ka_auc - baseline_auc) * 100)
    ax.bar([f'{ds}\n({km} vs {bm})' for ds in DATASETS], improvements,
           color=colors[MODELS.index(km)], alpha=0.7, label=f'{km} vs {bm}')

ax.set_ylabel('AUC Improvement (%)', fontsize=12)
ax.set_title('Relative Performance: KA-GNN vs Baseline GNN', fontsize=13)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'relative_improvement.png'), dpi=150)
plt.close()

# ============ Figure 7: Fourier KAN layer architecture diagram (text-based) ============
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.axis('off')

# Input layer
ax.add_patch(plt.Rectangle((0.5, 5.5), 1.5, 1.5, fill=True, facecolor='#4C72B0', alpha=0.7))
ax.text(1.25, 6.25, 'Input\nx', ha='center', va='center', fontsize=10, fontweight='bold')

# Fourier basis
for k in range(4):
    ypos = 6.5 - k*0.6
    ax.add_patch(plt.Rectangle((3, ypos-0.25), 2.5, 0.5, fill=True, facecolor='#C44E52', alpha=0.7))
    ax.text(4.25, ypos, f'cos(2π·{k+1}·x)+sin(2π·{k+1}·x)', ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(3, ypos), xytext=(2, 6.25),
                arrowprops=dict(arrowstyle='->', color='gray'))

# Coefficient weighting
ax.add_patch(plt.Rectangle((6, 4), 1.5, 3, fill=True, facecolor='#55A868', alpha=0.7))
ax.text(6.75, 5.5, 'Fourier\nCoeffs\nα_k, β_k', ha='center', va='center', fontsize=9, fontweight='bold')

for k in range(4):
    ypos = 6.5 - k*0.6
    ax.annotate('', xy=(6, 5.5), xytext=(5.5, ypos),
                arrowprops=dict(arrowstyle='->', color='gray'))

# Residual
ax.add_patch(plt.Rectangle((6, 1), 1.5, 2, fill=True, facecolor='#8172B2', alpha=0.7))
ax.text(6.75, 2, 'Residual\nW·x + b', ha='center', va='center', fontsize=9, fontweight='bold')
ax.annotate('', xy=(6, 2), xytext=(2, 6.25),
            arrowprops=dict(arrowstyle='->', color='gray', linestyle='dashed'))

# Output
ax.add_patch(plt.Rectangle((8.5, 3), 1.5, 3, fill=True, facecolor='#DDAA33', alpha=0.7))
ax.text(9.25, 4.5, 'Output\nΣ(α·cos+β·sin)\n+W·x+b', ha='center', va='center', fontsize=9, fontweight='bold')
ax.annotate('', xy=(8.5, 5.5), xytext=(7.5, 5.5),
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('', xy=(8.5, 3.5), xytext=(7.5, 2),
            arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_title('Fourier-Based Kolmogorov-Arnold Network (KAN) Layer Architecture', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'kan_architecture.png'), dpi=150)
plt.close()

# ============ Figure 8: Dataset statistics ============
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
ds_stats = {
    'bace': {'total': 1513, 'pos': 691, 'neg': 822},
    'bbbp': {'total': 2039, 'pos': 1560, 'neg': 479},
    'clintox': {'total': 1477, 'pos': 112, 'neg': 1365},
    'hiv': {'total': 5000, 'pos': 1443, 'neg': 3557},
    'muv': {'total': 5000, 'pos': 27, 'neg': 4973},
}

for idx, ds in enumerate(DATASETS):
    ax = axes[idx]
    stats = ds_stats[ds]
    bars = ax.bar(['Negative', 'Positive'], [stats['neg'], stats['pos']], 
                  color=['#4C72B0', '#C44E52'], alpha=0.8)
    ax.set_title(f'{ds.upper()}\n(n={stats["total"]})', fontsize=11)
    ax.set_ylabel('Count' if idx==0 else '', fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 10, f'{int(h)}',
                ha='center', va='bottom', fontsize=9)

plt.suptitle('Dataset Class Distribution', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'dataset_stats.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Save summary table ============
summary = {}
for ds in DATASETS:
    summary[ds] = {}
    for model in MODELS:
        r = results[ds][model]
        summary[ds][model] = {
            'test_auc': f"{r['test_auc_mean']:.4f}±{r['test_auc_std']:.4f}",
            'val_auc': f"{r['val_auc_mean']:.4f}±{r['val_auc_std']:.4f}",
            'params': r['num_params']
        }

with open(os.path.join(OUT_DIR, 'summary_table.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("All figures generated!")
print(f"Images saved to: {IMG_DIR}")
for fname in os.listdir(IMG_DIR):
    print(f"  {fname}")