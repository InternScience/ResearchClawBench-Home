"""
Generate all figures for the research report.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc

os.makedirs('report/images', exist_ok=True)

# Load results
with open('outputs/experiment_results.json', 'r') as f:
    results = json.load(f)

with open('outputs/multi_seed_results.json', 'r') as f:
    multi_seed = json.load(f)

# ============================================================
# Figure 1: Architecture Diagram
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(5, 7.5, 'KA-GNN Architecture', fontsize=16, fontweight='bold', ha='center')

# Input layer
ax.add_patch(plt.Rectangle((0.5, 6), 2, 1, fill=False, edgecolor='blue', linewidth=2))
ax.text(1.5, 6.5, 'Molecular\nGraph', fontsize=10, ha='center', va='center')

ax.annotate('', xy=(3.5, 6.5), xytext=(2.5, 6.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax.add_patch(plt.Rectangle((3.5, 6), 1.5, 1, fill=False, edgecolor='green', linewidth=2))
ax.text(4.25, 6.5, 'Node\nEncoder', fontsize=9, ha='center', va='center')

for i in range(3):
    y = 5.5 - i * 1.2
    ax.add_patch(plt.Rectangle((6, y), 2, 0.8, fill=False, edgecolor='red', linewidth=2))
    ax.text(7, y + 0.4, f'KAN-Conv {i+1}\n(Fourier basis)', fontsize=8, ha='center', va='center')
    if i < 2:
        ax.annotate('', xy=(7, y - 0.2), xytext=(7, y + 0.8),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.annotate('', xy=(6, 5.1), xytext=(5, 6.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax.add_patch(plt.Rectangle((6, 1.5), 2, 0.8, fill=False, edgecolor='purple', linewidth=2))
ax.text(7, 1.9, 'Global Pooling\n(Mean + Add)', fontsize=9, ha='center', va='center')

ax.annotate('', xy=(7, 2.3), xytext=(7, 3.1),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax.add_patch(plt.Rectangle((6, 0.3), 2, 0.8, fill=False, edgecolor='orange', linewidth=2))
ax.text(7, 0.7, 'KAN Readout\n(Fourier MLP)', fontsize=9, ha='center', va='center')

ax.annotate('', xy=(7, 1.1), xytext=(7, 1.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax.add_patch(plt.Rectangle((8.5, 0.3), 1, 0.8, fill=False, edgecolor='black', linewidth=2))
ax.text(9, 0.7, 'Property\nPrediction', fontsize=9, ha='center', va='center')

ax.annotate('', xy=(8.5, 0.7), xytext=(8, 0.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax.text(0.5, 0.5, 'Legend:', fontsize=10, fontweight='bold')
ax.text(0.5, 0.1, 'Blue: Input | Green: Encoder | Red: KAN-Conv | Purple: Pooling | Orange: Readout', 
        fontsize=8)

plt.tight_layout()
plt.savefig('report/images/architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved architecture.png")

# ============================================================
# Figure 2: Main Results Comparison (Bar chart with multi-seed stats)
# ============================================================
datasets = ['BACE', 'BBBP', 'ClinTox']
gcn_means, gcn_stds = [], []
kagcn_means, kagcn_stds = [], []

for ds_key in ['bace', 'bbbp', 'clintox']:
    if ds_key in multi_seed:
        seeds_res = multi_seed[ds_key]
        gcn_vals = [seeds_res[s]['GCN']['test_roc_auc'] for s in seeds_res if 'GCN' in seeds_res[s]]
        kagcn_vals = [seeds_res[s]['KA-GNN']['test_roc_auc'] for s in seeds_res if 'KA-GNN' in seeds_res[s]]
        gcn_means.append(np.mean(gcn_vals))
        gcn_stds.append(np.std(gcn_vals))
        kagcn_means.append(np.mean(kagcn_vals))
        kagcn_stds.append(np.std(kagcn_vals))

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(datasets))
width = 0.35

bars1 = ax.bar(x - width/2, gcn_means, width, label='GCN (Baseline)', 
               color='#4C72B0', yerr=gcn_stds, capsize=5, alpha=0.85)
bars2 = ax.bar(x + width/2, kagcn_means, width, label='KA-GNN (Ours)', 
               color='#DD8452', yerr=kagcn_stds, capsize=5, alpha=0.85)

ax.set_ylabel('Test ROC-AUC', fontsize=12)
ax.set_xlabel('Dataset', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(fontsize=11)
ax.set_ylim(0.4, 1.0)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/main_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved main_results.png")

# ============================================================
# Figure 3: Learning Curves
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, ds_key in enumerate(['bace', 'bbbp', 'clintox']):
    if ds_key not in results:
        continue
    ax = axes[idx]
    for model_name, color in [('GCN', '#4C72B0'), ('KA-GNN', '#DD8452')]:
        if model_name in results[ds_key]:
            res = results[ds_key][model_name]
            epochs = range(1, len(res['val_aucs']) + 1)
            ax.plot(epochs, res['val_aucs'], color=color, linewidth=2, label=model_name)
    ax.set_title(ds_key.upper(), fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val ROC-AUC')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved learning_curves.png")

# ============================================================
# Figure 4: Parameter Count vs Performance
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
colors_ds = {'bace': '#4C72B0', 'bbbp': '#55A868', 'clintox': '#C44E52'}
markers = {'GCN': 'o', 'KA-GNN': 's'}

for ds_key in ['bace', 'bbbp', 'clintox']:
    if ds_key not in multi_seed:
        continue
    for model_name in ['GCN', 'KA-GNN']:
        params_list, aucs_list = [], []
        for seed_key, res in multi_seed[ds_key].items():
            if model_name in res:
                params_list.append(res[model_name]['num_parameters'])
                aucs_list.append(res[model_name]['test_roc_auc'])
        if params_list:
            ax.scatter(params_list, aucs_list, c=colors_ds[ds_key], marker=markers[model_name], 
                      s=100, alpha=0.7, label=f"{model_name} ({ds_key.upper()})")

ax.set_xlabel('Number of Parameters', fontsize=12)
ax.set_ylabel('Test ROC-AUC', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('report/images/params_vs_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved params_vs_performance.png")

# ============================================================
# Figure 5: Training Time Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
models = ['GCN', 'KA-GNN']
times_data = {'bace': [], 'bbbp': [], 'clintox': []}

for ds_key in ['bace', 'bbbp', 'clintox']:
    if ds_key in multi_seed:
        for model_name in models:
            t = [multi_seed[ds_key][s][model_name]['training_time'] 
                 for s in multi_seed[ds_key] if model_name in multi_seed[ds_key][s]]
            times_data[ds_key].append(t)

x = np.arange(len(models))
width = 0.25
colors = ['#4C72B0', '#55A868', '#C44E52']

for i, ds_key in enumerate(['bace', 'bbbp', 'clintox']):
    if times_data[ds_key]:
        means = [np.mean(t) for t in times_data[ds_key]]
        stds = [np.std(t) for t in times_data[ds_key]]
        ax.bar(x + i*width, means, width, label=ds_key.upper(), 
               color=colors[i], yerr=stds, capsize=3, alpha=0.85)

ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_xticks(x + width)
ax.set_xticklabels(models)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('report/images/training_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved training_time.png")

# ============================================================
# Figure 6: Multi-seed Distribution (Box plot)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
data_to_plot, tick_labels, colors_plot = [], [], []

for ds_key in ['bace', 'bbbp', 'clintox']:
    if ds_key not in multi_seed:
        continue
    for model_name in ['GCN', 'KA-GNN']:
        aucs = [multi_seed[ds_key][s][model_name]['test_roc_auc'] 
                for s in multi_seed[ds_key] if model_name in multi_seed[ds_key][s]]
        if aucs:
            data_to_plot.append(aucs)
            tick_labels.append(f"{ds_key.upper()}\n{model_name}")
            colors_plot.append('#4C72B0' if model_name == 'GCN' else '#DD8452')

bp = ax.boxplot(data_to_plot, labels=tick_labels, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors_plot):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Test ROC-AUC', fontsize=12)
ax.set_xlabel('Dataset x Model', fontsize=12)
ax.set_ylim(0.4, 1.0)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('report/images/multiseed_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved multiseed_distribution.png")

# ============================================================
# Figure 7: ROC Curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for ds_key in ['bace', 'bbbp', 'clintox']:
    if ds_key not in results:
        continue
    for model_name, color in [('GCN', '#4C72B0'), ('KA-GNN', '#DD8452')]:
        if model_name in results[ds_key]:
            res = results[ds_key][model_name]
            labels = res['test_labels']
            preds = res['test_preds']
            # Use roc_curve's output for auc calculation
            fpr, tpr, thresholds = roc_curve(labels, preds)
            roc_auc = sklearn_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2, 
                   label=f"{model_name} - {ds_key.upper()} (AUC={roc_auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves', fontsize=14)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved roc_curves.png")

# ============================================================
# Figure 8: Fourier Basis Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Fourier basis functions
x_vals = np.linspace(-2, 2, 200)
for k in range(1, 6):
    axes[0].plot(x_vals, np.sin(k * x_vals), linewidth=1.5, alpha=0.7, label=f'sin({k}x)')
axes[0].set_title('Fourier Basis Functions (Sine)', fontsize=12)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Right: Combined Fourier approximation example
x_vals2 = np.linspace(-np.pi, np.pi, 200)
target = np.tanh(x_vals2)
approx = np.zeros_like(x_vals2)
for k in range(1, 6):
    coeff = 4 / (np.pi * k) if k % 2 == 1 else 0
    approx += coeff * np.sin(k * x_vals2)
axes[1].plot(x_vals2, target, 'k-', linewidth=2, label='tanh(x)')
axes[1].plot(x_vals2, approx, 'r--', linewidth=2, label='Fourier approx (G=5)')
axes[1].set_title('Fourier Approximation Example', fontsize=12)
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fourier_basis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fourier_basis.png")

print("\nAll figures saved to report/images/")
