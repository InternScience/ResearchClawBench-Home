"""Generate all figures for the KA-GNN report"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.makedirs('report/images', exist_ok=True)

with open('outputs/experiment_results.json') as f:
    results = json.load(f)

# ---- Figure 1: Main comparison bar chart ----
fig, ax = plt.subplots(figsize=(10, 6))
datasets = ['bace', 'bbbp', 'clintox', 'hiv', 'muv']
models = ['KA-GNN', 'GCN', 'GAT']
colors = {'KA-GNN': '#e74c3c', 'GCN': '#3498db', 'GAT': '#2ecc71'}
x = np.arange(len(datasets))
width = 0.25

for i, model in enumerate(models):
    means, stds = [], []
    for ds in datasets:
        r = [x for x in results if x['dataset'] == ds and x['model_name'] == model]
        if r:
            means.append(r[0]['mean_auc'])
            stds.append(r[0]['std_auc'])
        else:
            means.append(0.5)
            stds.append(0)
    bars = ax.bar(x + i * width, means, width, yerr=stds, label=model,
                  color=colors[model], capsize=3, alpha=0.85)

ax.set_ylabel('ROC-AUC', fontsize=13)
ax.set_xlabel('Dataset', fontsize=13)
ax.set_title('Molecular Property Prediction: KA-GNN vs Baselines', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([d.upper() for d in datasets], fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0.4, 1.05)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/main_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1: main_comparison.png")

# ---- Figure 2: Dataset overview ----
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
dataset_info = {
    'BACE': (1513, 1, 'BACE-1 Inhibition'),
    'BBBP': (2039, 1, 'Blood-Brain Barrier'),
    'ClinTox': (1477, 2, 'Clinical Toxicity'),
    'HIV': (41127, 1, 'HIV Inhibition'),
    'MUV': (93087, 17, 'Virtual Screening'),
}
for ax, (name, (n, tasks, desc)) in zip(axes, dataset_info.items()):
    ax.bar(['Compounds'], [n], color='#3498db', alpha=0.7)
    ax.set_title(f'{name}\n{desc}\n({tasks} task{"s" if tasks>1 else ""})', fontsize=9)
    ax.set_ylabel('Count')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.suptitle('Dataset Overview', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2: dataset_overview.png")

# ---- Figure 3: Per-fold results heatmap ----
fig, ax = plt.subplots(figsize=(8, 5))
heatmap_data = []
for ds in datasets:
    for model in models:
        r = [x for x in results if x['dataset'] == ds and x['model_name'] == model]
        if r:
            for fold_idx, auc in enumerate(r[0]['fold_aucs']):
                heatmap_data.append({'Dataset': ds.upper(), 'Model': model,
                                    f'Fold {fold_idx+1}': auc})

# Create pivot for heatmap
rows = []
for ds in datasets:
    for model in models:
        r = [x for x in results if x['dataset'] == ds and x['model_name'] == model]
        if r:
            rows.append({'': f'{ds.upper()}-{model}', 'Mean AUC': r[0]['mean_auc']})
df_heat = pd.DataFrame(rows)
pivot = df_heat.pivot_table(index='', values='Mean AUC')
# Better: grouped bar
fig, ax = plt.subplots(figsize=(12, 6))
row_labels = [f'{d.upper()}\n{m}' for d in datasets for m in models]
aucs = []
for d in datasets:
    for m in models:
        r = [x for x in results if x['dataset'] == d and x['model_name'] == m]
        aucs.append(r[0]['mean_auc'] if r else 0.5)

y_pos = np.arange(len(row_labels))
colors_list = [colors[m] for d in datasets for m in models]
ax.barh(y_pos, aucs, color=colors_list, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(row_labels, fontsize=9)
ax.set_xlabel('ROC-AUC', fontsize=12)
ax.set_title('Per-Dataset Model Performance', fontsize=14, fontweight='bold')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlim(0.4, 1.05)
for i, v in enumerate(aucs):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('report/images/performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3: performance_heatmap.png")

# ---- Figure 4: KAN architecture diagram (schematic) ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
ax.set_aspect('equal')
ax.axis('off')

# Draw architecture boxes
boxes = [
    (1, 2.5, 'Input\n(SMILES)', '#e8e8e8'),
    (3, 2.5, 'Graph\nFeaturization', '#ffeaa7'),
    (5, 2.5, 'KAN-GNN\nMessage Passing', '#fab1a0'),
    (7, 2.5, 'Global\nPooling', '#dfe6e9'),
    (9, 2.5, 'Prediction', '#74b9ff'),
]
for x, y, label, color in boxes:
    rect = plt.Rectangle((x-0.7, y-0.6), 1.4, 1.2, facecolor=color, edgecolor='black', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

# Arrows
for i in range(len(boxes)-1):
    ax.annotate('', xy=(boxes[i+1][0]-0.7, boxes[i+1][1]),
                xytext=(boxes[i][0]+0.7, boxes[i][1]),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2d3436'))

# KAN detail box
rect = plt.Rectangle((3.8, 0.3), 2.4, 1.2, facecolor='#fd79a8', edgecolor='black', linewidth=1, alpha=0.8, zorder=2)
ax.add_patch(rect)
ax.text(5, 0.9, 'KAN Module:\nFourier-based univariate\nactivation functions', ha='center', va='center', fontsize=8, color='white', fontweight='bold', zorder=3)
ax.annotate('', xy=(5, 1.5), xytext=(5, 1.9), arrowprops=dict(arrowstyle='->', lw=1.5, color='#d63031'))

ax.set_title('KA-GNN Architecture Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4: architecture.png")

# ---- Figure 5: Improvement analysis ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: KA-GNN advantage on smaller datasets
small_ds = ['bace', 'bbbp', 'clintox']
large_ds = ['hiv', 'muv']
for ax_idx, (ds_group, title) in enumerate([(small_ds, 'Small Datasets (< 2K)'), (large_ds, 'Large Datasets (> 4K)')]):
    ax = axes[ax_idx]
    for model in models:
        aucs = []
        for ds in ds_group:
            r = [x for x in results if x['dataset'] == ds and x['model_name'] == model]
            aucs.append(r[0]['mean_auc'] if r else 0.5)
        ax.plot(ds_group, aucs, 'o-', label=model, color=colors[model], linewidth=2, markersize=8)
    ax.set_ylabel('ROC-AUC', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Performance by Dataset Size', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/size_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5: size_analysis.png")

# ---- Figure 6: Model parameter comparison ----
fig, ax = plt.subplots(figsize=(8, 5))
# Approximate parameter counts from test run
param_counts = {'KA-GNN': 66561, 'GCN': 8193, 'GAT': 8321}
bars = ax.bar(param_counts.keys(), param_counts.values(), color=[colors[m] for m in param_counts.keys()], alpha=0.8)
ax.set_ylabel('Number of Parameters', fontsize=12)
ax.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
for bar, v in zip(bars, param_counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, f'{v:,}', ha='center', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/param_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6: param_comparison.png")

print("\nAll figures generated!")
