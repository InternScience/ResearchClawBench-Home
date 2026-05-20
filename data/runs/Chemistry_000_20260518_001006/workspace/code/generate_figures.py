"""
Generate figures for KA-GNN research report.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

os.makedirs('report/images', exist_ok=True)

# Load results
with open('outputs/results.json') as f:
    results = json.load(f)

datasets = ['BACE', 'BBBP', 'ClinTox', 'HIV', 'MUV']
models = ['KA-GNN', 'Baseline GCN', 'Baseline GAT']
colors = {'KA-GNN': '#2196F3', 'Baseline GCN': '#FF9800', 'Baseline GAT': '#4CAF50'}
hatches = {'KA-GNN': '//', 'Baseline GCN': '..', 'Baseline GAT': 'xx'}

# ============================================================
# Figure 1: AUC comparison across datasets
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(datasets))
width = 0.25

for i, model in enumerate(models):
    aucs = [results[d][model]['auc'] for d in datasets]
    bars = ax.bar(x + i * width, aucs, width, label=model, color=colors[model], 
                  edgecolor='black', linewidth=0.5, hatch=hatches[model])
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

ax.set_ylabel('ROC-AUC')
ax.set_title('Model Performance Comparison Across MoleculeNet Datasets')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend(loc='lower left')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Random')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/auc_comparison.png')
plt.close()
print("Figure 1 saved: auc_comparison.png")

# ============================================================
# Figure 2: Training time comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for i, model in enumerate(models):
    times = [results[d][model]['time'] for d in datasets]
    bars = ax.bar(x + i * width, times, width, label=model, color=colors[model],
                  edgecolor='black', linewidth=0.5, hatch=hatches[model])

ax.set_ylabel('Training Time (seconds)')
ax.set_title('Training Time Comparison Across Datasets')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/time_comparison.png')
plt.close()
print("Figure 2 saved: time_comparison.png")

# ============================================================
# Figure 3: Parameter count comparison (log scale)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

params = [results['BACE'][m]['params'] for m in models]
bars = ax.bar(models, params, color=[colors[m] for m in models], 
              edgecolor='black', linewidth=0.5, width=0.5)

for bar, p in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1000,
            f'{p:,}', ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Number of Parameters')
ax.set_title('Model Parameter Count Comparison')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/params_comparison.png')
plt.close()
print("Figure 3 saved: params_comparison.png")

# ============================================================
# Figure 4: Training curves for KA-GNN vs Baselines on BACE
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, model in enumerate(models):
    ax = axes[idx]
    history = results['BACE'][model].get('history', {})
    if 'test_auc' in history:
        ax.plot(history['test_auc'], label='Test AUC', color='blue', linewidth=2)
    if 'val_auc' in history:
        ax.plot(history['val_auc'], label='Val AUC', color='orange', linestyle='--', linewidth=2)
    if 'train_loss' in history:
        ax2 = ax.twinx()
        ax2.plot(history['train_loss'], label='Train Loss', color='red', alpha=0.5, linewidth=1)
        ax2.set_ylabel('Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title(f'{model} on BACE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.suptitle('Training Dynamics Comparison on BACE Dataset', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/training_curves_bace.png')
plt.close()
print("Figure 4 saved: training_curves_bace.png")

# ============================================================
# Figure 5: AUC relative to GCN baseline (improvement over baseline)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for i, dataset in enumerate(datasets):
    baseline_auc = results[dataset]['Baseline GCN']['auc']
    ka_auc = results[dataset]['KA-GNN']['auc']
    gat_auc = results[dataset]['Baseline GAT']['auc']
    
    ka_delta = ka_auc - baseline_auc
    gat_delta = gat_auc - baseline_auc
    
    bars1 = ax.bar(i - 0.15, ka_delta, 0.3, label='KA-GNN vs GCN' if i == 0 else '',
                   color=colors['KA-GNN'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(i + 0.15, gat_delta, 0.3, label='GAT vs GCN' if i == 0 else '',
                   color=colors['Baseline GAT'], edgecolor='black', linewidth=0.5)
    
    for bar, val in [(bars1, ka_delta), (bars2, gat_delta)]:
        for b, v in zip(bar, [val]):
            y_pos = b.get_height() + 0.005 if b.get_height() >= 0 else b.get_height() - 0.02
            ax.text(b.get_x() + b.get_width()/2., y_pos, f'{v:+.3f}', 
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

ax.set_ylabel('Δ AUC (Improvement over Baseline GCN)')
ax.set_title('Performance Improvement Relative to MLP-based GCN')
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/improvement_over_baseline.png')
plt.close()
print("Figure 5 saved: improvement_over_baseline.png")

# ============================================================
# Figure 6: Summary radar / heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Build heatmap data
auc_matrix = np.array([[results[d][m]['auc'] for d in datasets] for m in models])

im = ax.imshow(auc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models)

for i in range(len(models)):
    for j in range(len(datasets)):
        text_color = 'white' if auc_matrix[i, j] < 0.6 else 'black'
        ax.text(j, i, f'{auc_matrix[i, j]:.3f}', ha='center', va='center', 
                fontsize=11, fontweight='bold', color=text_color)

ax.set_title('ROC-AUC Performance Heatmap')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('ROC-AUC')

plt.tight_layout()
plt.savefig('report/images/auc_heatmap.png')
plt.close()
print("Figure 6 saved: auc_heatmap.png")

# ============================================================
# Figure 7: Efficiency plot (AUC vs Time)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

for model in models:
    for dataset in datasets:
        auc = results[dataset][model]['auc']
        time = results[dataset][model]['time']
        ax.scatter(time, auc, s=200, color=colors[model], edgecolors='black',
                   linewidth=0.5, alpha=0.7, zorder=5)
        ax.annotate(f'{dataset}', (time, auc), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, alpha=0.8)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[m], edgecolor='black', label=m) for m in models]
ax.legend(handles=legend_elements, loc='lower right')

ax.set_xlabel('Training Time (seconds)')
ax.set_ylabel('ROC-AUC')
ax.set_title('Model Efficiency: Predictive Performance vs Training Time')
ax.grid(alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('report/images/efficiency_scatter.png')
plt.close()
print("Figure 7 saved: efficiency_scatter.png")

print("\nAll figures generated successfully!")
