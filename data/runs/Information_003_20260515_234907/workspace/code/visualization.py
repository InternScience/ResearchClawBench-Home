"""Generate all figures for the DIDS-MFL report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

IMG_DIR = 'report/images'
os.makedirs(IMG_DIR, exist_ok=True)

with open('outputs/all_results.json') as f:
    results = json.load(f)

# Figure 1: Data distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Binary
ax = axes[0]
binary = results['data_statistics']
ax.bar(['Benign', 'Attack'], [binary['benign'], binary['attack']], 
       color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_title('Binary Class Distribution', fontweight='bold')
ax.set_ylabel('Number of Flows')
for i, v in enumerate([binary['benign'], binary['attack']]):
    ax.text(i, v + 2000, f'{v:,}\n({v/binary["total_flows"]*100:.1f}%)', 
            ha='center', fontweight='bold')

# Multi-class
ax = axes[1]
mc = results['multiclass_classification']
classes = mc['class_names']
f1s = [mc['per_class_f1'][f'class_{i}'] for i in range(len(classes))]
colors = ['#e74c3c' if f1 < 0.7 else '#f39c12' if f1 < 0.9 else '#2ecc71' for f1 in f1s]
bars = ax.bar(range(len(classes)), f1s, color=colors, edgecolor='black')
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45)
ax.set_title('Per-Class F1 Score (Multi-class)', fontweight='bold')
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold 0.7')
ax.legend()

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig1_data_distribution.png', bbox_inches='tight')
plt.close()
print('Figure 1 done')

# Figure 2: Binary classification comparison
fig, ax = plt.subplots(figsize=(10, 6))
bin_res = results['binary_classification']
methods = list(bin_res.keys())
metrics = ['accuracy', 'f1', 'precision', 'recall']
x = np.arange(len(methods))
width = 0.2

for i, metric in enumerate(metrics):
    values = [bin_res[m][metric] for m in methods]
    bars = ax.bar(x + i*width - width*1.5, values, width, label=metric.capitalize(), edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_title('Binary Classification: Method Comparison', fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0.8, 1.01)
ax.legend(loc='lower right')
for i, metric in enumerate(metrics):
    for j, m in enumerate(methods):
        v = bin_res[m][metric]
        ax.text(j + i*width - width*1.5, v - 0.03, f'{v:.4f}', ha='center', 
                fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig2_binary_comparison.png', bbox_inches='tight')
plt.close()
print('Figure 2 done')

# Figure 3: Multi-class confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
cm = np.array(mc['confusion_matrix'])
cm_norm = cm / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=mc['class_names'], yticklabels=mc['class_names'],
            ax=ax, vmin=0, vmax=1)
ax.set_title('Normalized Confusion Matrix (Multi-class)', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig3_confusion_matrix.png', bbox_inches='tight')
plt.close()
print('Figure 3 done')

# Figure 4: Feature importance
fig, ax = plt.subplots(figsize=(12, 5))
mi = results['feature_importance']
top5 = results['top5_features']
sorted_idx = np.argsort(mi)[::-1]
colors = ['#e74c3c' if i in top5 else '#3498db' for i in sorted_idx]
ax.bar(range(len(mi)), np.array(mi)[sorted_idx], color=colors, edgecolor='black')
ax.set_xticks(range(len(mi)))
ax.set_xticklabels([f'F{i}' for i in sorted_idx], rotation=45, fontsize=8)
ax.set_title('Feature Importance (Mutual Information)', fontweight='bold')
ax.set_ylabel('Mutual Information')
ax.axhline(y=np.mean(mi), color='green', linestyle='--', alpha=0.7, label=f'Mean MI: {np.mean(mi):.3f}')
ax.legend()
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig4_feature_importance.png', bbox_inches='tight')
plt.close()
print('Figure 4 done')

# Figure 5: PCA/ICA disentanglement visualization
from data_pipeline import load_data, extract_flat_features, statistical_disentanglement
data = load_data()
features, labels, attacks, _, _, _ = extract_flat_features(data)
X_ica, _, pca, _ = statistical_disentanglement(features, n_components=20)

from sklearn.decomposition import PCA as PCAviz
pca_viz = PCAviz(n_components=2)
X_pca_2d = pca_viz.fit_transform(X_ica)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Binary coloring
ax = axes[0]
for lbl, color, name in [(0, '#2ecc71', 'Benign'), (1, '#e74c3c', 'Attack')]:
    mask = labels == lbl
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], c=color, label=name, 
               alpha=0.3, s=5, rasterized=True)
ax.set_title('Disentangled Feature Space (Binary)', fontweight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(markerscale=5)

# Attack-type coloring
ax = axes[1]
attack_names = {0:'A0',1:'A1',2:'Benign',3:'A3',4:'A4',5:'A5',6:'A6',7:'A7',8:'A8',9:'A9'}
cmap = plt.cm.tab10
for a in sorted(set(attacks)):
    mask = attacks == a
    ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], c=[cmap(a/10)], 
               label=attack_names[a], alpha=0.3, s=5, rasterized=True)
ax.set_title('Disentangled Feature Space (Multi-class)', fontweight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(markerscale=5, ncol=2, fontsize=7)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig5_disentanglement.png', bbox_inches='tight')
plt.close()
print('Figure 5 done')

# Figure 6: Unknown attack detection per class
fig, ax = plt.subplots(figsize=(10, 5))
ua = results['unknown_attack']['per_class']
classes_ua = [f'Class {r["held_out_class"]}' for r in ua]
f1_ua = [r['f1'] for r in ua]
ax.bar(classes_ua, f1_ua, color='#e74c3c', edgecolor='black')
ax.set_title('Unknown Attack Detection (F1 per Held-Out Class)', fontweight='bold')
ax.set_ylabel('F1 Score')
ax.set_ylim(0, max(f1_ua)*1.3)
for i, v in enumerate(f1_ua):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=8)
ax.axhline(y=results['unknown_attack']['f1'], color='blue', linestyle='--', 
           label=f'Average F1: {results["unknown_attack"]["f1"]:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig6_unknown_attack.png', bbox_inches='tight')
plt.close()
print('Figure 6 done')

# Figure 7: Few-shot analysis - per-class F1 vs sample count
fig, ax1 = plt.subplots(figsize=(10, 5))

class_counts = {}
for a in attacks:
    class_counts[a] = class_counts.get(a, 0) + 1

few_shot_classes = [0, 1, 4, 5, 9]
well_represented = [3, 6, 7, 8]
benign_class = 2

all_cls = list(range(10))
counts = [class_counts.get(c, 0) for c in all_cls]
f1s_per_class = [mc['per_class_f1'][f'class_{c}'] for c in all_cls]

# Bar chart for sample counts
colors_bar = ['#e74c3c' if c in few_shot_classes else '#2ecc71' for c in all_cls]
ax1.bar([f'C{c}' for c in all_cls], counts, color=colors_bar, alpha=0.6, edgecolor='black', label='Sample Count')
ax1.set_ylabel('Number of Samples', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_yscale('log')

# Line for F1
ax2 = ax1.twinx()
ax2.plot([f'C{c}' for c in all_cls], f1s_per_class, 'o-', color='red', linewidth=2, markersize=8, label='F1 Score')
ax2.set_ylabel('F1 Score', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 1.1)

ax1.set_title('Class Sample Count vs Detection F1 Score', fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig7_few_shot_analysis.png', bbox_inches='tight')
plt.close()
print('Figure 7 done')

# Figure 8: DIDS-MFL architecture diagram (schematic)
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6)
ax.axis('off')

# Draw boxes for the architecture
boxes = [
    (0.5, 4.5, 2.5, 1.2, 'Traffic Flow\nData', '#3498db'),
    (3.5, 4.5, 2.5, 1.2, 'Statistical\nDisentanglement\n(PCA+ICA+MI)', '#2ecc71'),
    (6.5, 4.5, 2.5, 1.2, 'Dynamic Graph\nConstruction', '#f39c12'),
    (9.5, 4.5, 2.2, 1.2, 'Disentangled\nGNN', '#e74c3c'),
    (6.5, 2.0, 2.5, 1.2, 'Representational\nDisentanglement\n(Covariance Reg.)', '#9b59b6'),
    (9.5, 2.0, 2.2, 1.2, 'Multi-Scale\nFusion\n(Few-Shot)', '#1abc9c'),
]

for x, y, w, h, label, color in boxes:
    rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=color, alpha=0.7,
                          edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrows
arrows = [
    (3.0, 5.1, 3.5, 5.1),  # data -> stat disent
    (6.0, 5.1, 6.5, 5.1),  # stat -> graph
    (9.0, 5.1, 9.5, 5.1),  # graph -> gnn
    (8.0, 4.3, 8.0, 3.2),  # graph -> rep disent (down)
    (9.0, 3.2, 9.5, 3.2),  # rep -> fusion
]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

ax.set_title('DIDS-MFL Architecture Overview', fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig8_architecture.png', bbox_inches='tight')
plt.close()
print('Figure 8 done')

print('\nAll figures generated successfully!')
