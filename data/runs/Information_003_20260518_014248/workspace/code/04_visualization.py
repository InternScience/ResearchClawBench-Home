"""
Phase 4: Generate all result figures and analysis.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import DIDS_MFL, BaselineMLP

# Load data and results
print("Loading data...")
data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', map_location='cpu', weights_only=False)
msg = data.msg.numpy()
label = data.label.numpy()
attack = data.attack.numpy()

# Load results
with open('outputs/full_results.json', 'r') as f:
    results = json.load(f)

with open('outputs/comparison_table.json', 'r') as f:
    comparison = json.load(f)

# Load embeddings
dids_emb = np.load('outputs/dids_embeddings.npy')
mlp_emb = np.load('outputs/mlp_embeddings.npy')

# Split data (same as training)
t = data.t.numpy()
split_idx = int(len(msg) * 0.7)
indices = np.argsort(t)
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

y_test_bin = label[test_idx]
y_test_mul = attack[test_idx]

# Attack type names
attack_names = {
    0: 'Reconnaissance', 1: 'Backdoor', 2: 'Benign', 3: 'DoS', 4: 'Exploits',
    5: 'Generic', 6: 'Fuzzers', 7: 'Analysis', 8: 'Shellcode', 9: 'Worms'
}

# ===================== Figure 3: Main Results Comparison =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 3a: Binary Classification Comparison
ax = axes[0]
methods = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'MLP', 'DIDS-MFL']
binary_f1 = [
    comparison.get('LogisticRegression_Binary', {}).get('f1_macro', 0),
    comparison.get('RandomForest_Binary', {}).get('f1_macro', 0),
    comparison.get('GradientBoosting_Binary', {}).get('f1_macro', 0),
    comparison.get('MLP_Binary', {}).get('f1_macro', 0),
    comparison.get('DIDSMFL_Binary', {}).get('f1_macro', 0),
]
colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
bars = ax.bar(methods, binary_f1, color=colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, binary_f1):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('Binary Classification\n(Benign vs Attack)', fontsize=13, fontweight='bold')
ax.set_ylim(0.97, 1.005)
ax.axhline(y=max(binary_f1), color='gray', linestyle='--', alpha=0.5)

# 3b: Multi-class Classification Comparison
ax = axes[1]
multi_f1 = [
    comparison.get('LogisticRegression_Multi', {}).get('f1_macro', 0),
    comparison.get('RandomForest_Multi', {}).get('f1_macro', 0),
    comparison.get('GradientBoosting_Multi', {}).get('f1_macro', 0),
    comparison.get('MLP_Multi', {}).get('f1_macro', 0),
    comparison.get('DIDSMFL_Multi', {}).get('f1_macro', 0),
]
bars = ax.bar(methods, multi_f1, color=colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, multi_f1):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('Multi-class Classification\n(10 attack types)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.axhline(y=max(multi_f1), color='gray', linestyle='--', alpha=0.5)

# 3c: Ablation Study
ax = axes[2]
ablation_methods = ['SD Only', 'No Graph\nDiffusion', 'No Rep.\nDisentangle', 'Full\nDIDS-MFL']
ablation_bin_f1 = [
    results['ablation']['SD_Only']['binary']['f1_macro'],
    results['ablation']['No_Graph']['binary']['f1_macro'],
    results['ablation']['No_RepDis']['binary']['f1_macro'],
    results['ablation']['Full_DIDSMFL']['binary']['f1_macro'],
]
ablation_mul_f1 = [
    results['ablation']['SD_Only']['multi']['f1_macro'],
    results['ablation']['No_Graph']['multi']['f1_macro'],
    results['ablation']['No_RepDis']['multi']['f1_macro'],
    results['ablation']['Full_DIDSMFL']['multi']['f1_macro'],
]
x_pos = np.arange(len(ablation_methods))
width = 0.35
bars1 = ax.bar(x_pos - width/2, ablation_bin_f1, width, label='Binary F1', 
               color='#3498db', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x_pos + width/2, ablation_mul_f1, width, label='Multi-class F1', 
               color='#e74c3c', edgecolor='black', linewidth=0.8)
for bar, val in zip(bars1, ablation_bin_f1):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, ablation_mul_f1):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(ablation_methods, fontsize=10)
ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_title('Ablation Study', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('report/images/main_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/main_results.png")

# ===================== Figure 4: Disentanglement Visualization (t-SNE) =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

np.random.seed(42)
n_sample = min(3000, len(y_test_bin))
idx = np.random.choice(len(y_test_bin), n_sample, replace=False)

# 4a: MLP embeddings colored by class
ax = axes[0, 0]
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
mlp_emb_2d = tsne.fit_transform(mlp_emb[idx])
scatter = ax.scatter(mlp_emb_2d[:, 0], mlp_emb_2d[:, 1], 
                     c=y_test_bin[idx], cmap='RdYlGn_r', s=5, alpha=0.6, edgecolors='none')
ax.set_title('MLP Embeddings\n(colored by class)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_ticklabels(['Benign', 'Attack'])

# 4b: DIDS-MFL embeddings colored by class
ax = axes[0, 1]
dids_emb_2d = tsne.fit_transform(dids_emb[idx])
scatter = ax.scatter(dids_emb_2d[:, 0], dids_emb_2d[:, 1], 
                     c=y_test_bin[idx], cmap='RdYlGn_r', s=5, alpha=0.6, edgecolors='none')
ax.set_title('DIDS-MFL Embeddings\n(colored by class)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_ticklabels(['Benign', 'Attack'])

# 4c: MLP embeddings colored by attack type
ax = axes[1, 0]
scatter = ax.scatter(mlp_emb_2d[:, 0], mlp_emb_2d[:, 1], 
                     c=y_test_mul[idx], cmap='tab10', s=5, alpha=0.6, edgecolors='none')
ax.set_title('MLP Embeddings\n(colored by attack type)', fontsize=13, fontweight='bold')

# 4d: DIDS-MFL embeddings colored by attack type
ax = axes[1, 1]
scatter = ax.scatter(dids_emb_2d[:, 0], dids_emb_2d[:, 1], 
                     c=y_test_mul[idx], cmap='tab10', s=5, alpha=0.6, edgecolors='none')
ax.set_title('DIDS-MFL Embeddings\n(colored by attack type)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/disentanglement_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/disentanglement_tsne.png")

# ===================== Figure 5: Confusion Matrix =====================
# Re-run DIDS-MFL to get predictions for confusion matrix
print("Generating confusion matrix...")
X_test_t = torch.FloatTensor(msg[test_idx])

# Load trained model (re-create and predict)
dids_model = DIDS_MFL(input_dim=40, num_groups=5, memory_size=64, 
                       hidden_dim=64, num_heads=3, head_dim=32,
                       num_hops=2, num_classes=10)
# We already have the embeddings and can compute predictions from them
# Actually we need predictions, let's use the saved comparison data

# Create confusion matrix from multi-class results
# For this, we'll create a synthetic confusion matrix based on per-class F1
per_class_f1 = results['DIDSMFL_Multi']['per_class'] if 'DIDSMFL_Multi' in results else {}

# Since we don't have the actual confusion matrix saved, let's create it from the data
# Use RF predictions as baseline and DIDS predictions from the comparison

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 5a: Create a per-class performance comparison
ax = axes[0]
class_ids = sorted([int(k) for k in per_class_f1.keys()])
class_names = [attack_names.get(c, f'Type-{c}') for c in class_ids]
class_f1_vals = [per_class_f1[str(c)]['f1'] for c in class_ids]
class_supports = [per_class_f1[str(c)]['support'] for c in class_ids]

colors_f1 = ['#2ecc71' if f > 0.5 else '#e67e22' if f > 0.2 else '#e74c3c' for f in class_f1_vals]
bars = ax.barh(range(len(class_ids)), class_f1_vals, color=colors_f1, 
               edgecolor='black', linewidth=0.5, height=0.6)
ax.set_yticks(range(len(class_ids)))
ax.set_yticklabels(class_names, fontsize=10)
ax.set_xlabel('F1 Score', fontsize=12)
ax.set_title('Per-Class F1 Score\n(DIDS-MFL Multi-class)', fontsize=13, fontweight='bold')
for i, (bar, val, sup) in enumerate(zip(bars, class_f1_vals, class_supports)):
    ax.text(val + 0.01, i, f'{val:.3f} (n={sup})', va='center', fontsize=9)
ax.set_xlim(0, 1.1)

# 5b: Few-shot detection performance
ax = axes[1]
few_shot = results.get('few_shot', {})
if few_shot:
    fs_classes = sorted([int(k) for k in few_shot.keys()])
    fs_names = [attack_names.get(c, f'Type-{c}') for c in fs_classes]
    fs_f1 = [few_shot[str(c)]['f1'] for c in fs_classes]
    fs_precision = [few_shot[str(c)]['precision'] for c in fs_classes]
    fs_recall = [few_shot[str(c)]['recall'] for c in fs_classes]
    
    x_pos = np.arange(len(fs_classes))
    width = 0.25
    ax.bar(x_pos - width, fs_f1, width, label='F1', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos, fs_precision, width, label='Precision', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + width, fs_recall, width, label='Recall', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fs_names, fontsize=10, rotation=15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Few-Shot Attack Detection\n(<1000 training samples)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('report/images/per_class_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/per_class_performance.png")

# ===================== Figure 6: Feature Importance =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

importances_rf = np.load('outputs/feature_importances_rf.npy')
importances_gb = np.load('outputs/feature_importances_gb.npy')

# 6a: RF Feature Importance
ax = axes[0]
top_k = 15
top_idx_rf = np.argsort(importances_rf)[-top_k:][::-1]
ax.barh(range(top_k), importances_rf[top_idx_rf][::-1], 
        color=plt.cm.viridis(np.linspace(0.2, 0.8, top_k)), edgecolor='black', linewidth=0.5)
ax.set_yticks(range(top_k))
ax.set_yticklabels([f'Feature {i}' for i in top_idx_rf[::-1]], fontsize=9)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Random Forest\nFeature Importance (Top-15)', fontsize=13, fontweight='bold')

# 6b: GB Feature Importance
ax = axes[1]
top_idx_gb = np.argsort(importances_gb)[-top_k:][::-1]
ax.barh(range(top_k), importances_gb[top_idx_gb][::-1], 
        color=plt.cm.plasma(np.linspace(0.2, 0.8, top_k)), edgecolor='black', linewidth=0.5)
ax.set_yticks(range(top_k))
ax.set_yticklabels([f'Feature {i}' for i in top_idx_gb[::-1]], fontsize=9)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Gradient Boosting\nFeature Importance (Top-15)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/feature_importance.png")

# ===================== Figure 7: Ablation Results Detail =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 7a: Binary ablation heatmap-style
ax = axes[0]
ablation_data = {
    'SD Only': results['ablation']['SD_Only']['binary'],
    'No Graph\nDiffusion': results['ablation']['No_Graph']['binary'],
    'No Rep.\nDisentangle': results['ablation']['No_RepDis']['binary'],
    'Full DIDS-MFL': results['ablation']['Full_DIDSMFL']['binary'],
}
metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
metric_labels = ['Accuracy', 'F1 (macro)', 'F1 (weighted)', 'Precision', 'Recall']
methods_list = list(ablation_data.keys())
data_matrix = np.array([[ablation_data[m].get(metric, 0) for metric in metrics] 
                        for m in methods_list])

im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0.98, vmax=1.0)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metric_labels, fontsize=10, rotation=30, ha='right')
ax.set_yticks(range(len(methods_list)))
ax.set_yticklabels(methods_list, fontsize=10)
for i in range(len(methods_list)):
    for j in range(len(metrics)):
        ax.text(j, i, f'{data_matrix[i, j]:.4f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title('Ablation: Binary Classification', fontsize=13, fontweight='bold')

# 7b: Multi-class ablation heatmap-style
ax = axes[1]
ablation_data_m = {
    'SD Only': results['ablation']['SD_Only']['multi'],
    'No Graph\nDiffusion': results['ablation']['No_Graph']['multi'],
    'No Rep.\nDisentangle': results['ablation']['No_RepDis']['multi'],
    'Full DIDS-MFL': results['ablation']['Full_DIDSMFL']['multi'],
}
data_matrix_m = np.array([[ablation_data_m[m].get(metric, 0) for metric in metrics] 
                          for m in methods_list])

im = ax.imshow(data_matrix_m, cmap='YlOrRd', aspect='auto', vmin=0.2, vmax=1.0)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metric_labels, fontsize=10, rotation=30, ha='right')
ax.set_yticks(range(len(methods_list)))
ax.set_yticklabels(methods_list, fontsize=10)
for i in range(len(methods_list)):
    for j in range(len(metrics)):
        ax.text(j, i, f'{data_matrix_m[i, j]:.4f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title('Ablation: Multi-class Classification', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/ablation_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/ablation_results.png")

# ===================== Figure 8: Training Curves =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Load training history
try:
    dids_hist = np.load('outputs/dids_training_history.npy', allow_pickle=True).item()
    mlp_hist = np.load('outputs/mlp_training_history.npy', allow_pickle=True).item()
except:
    dids_hist = None
    mlp_hist = None

if dids_hist is not None and mlp_hist is not None:
    # 8a: Loss curves
    ax = axes[0]
    epochs_dids = range(1, len(dids_hist['loss']) + 1)
    epochs_mlp = range(1, len(mlp_hist['loss']) + 1)
    ax.plot(epochs_dids, dids_hist['loss'], label='DIDS-MFL', color='#e74c3c', linewidth=2)
    ax.plot(epochs_mlp, mlp_hist['loss'], label='MLP', color='#3498db', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training Loss Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 8b: DIDS-MFL loss components
    ax = axes[1]
    ax.plot(epochs_dids, dids_hist['binary_loss'], label='Binary CE', color='#e74c3c', linewidth=2)
    ax.plot(epochs_dids, dids_hist['multi_loss'], label='Multi-class CE', color='#3498db', linewidth=2)
    if 'contrastive_loss' in dids_hist:
        ax.plot(epochs_dids, dids_hist['contrastive_loss'], label='Contrastive', color='#2ecc71', linewidth=2)
    if 'diversity_loss' in dids_hist:
        ax.plot(epochs_dids, dids_hist['diversity_loss'], label='Diversity', color='#f39c12', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('DIDS-MFL Loss Components', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/training_curves.png")

# ===================== Figure 9: Unknown Attack Detection =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

unknown = results.get('unknown_attack', {})

# 9a: Unknown detection rate
ax = axes[0]
if unknown:
    u_classes = sorted([int(k) for k in unknown.keys()])
    u_names = [attack_names.get(c, f'Type-{c}') for c in u_classes]
    u_det_rates = [unknown[str(c)]['unknown_detection_rate'] for c in u_classes]
    u_det_counts = [unknown[str(c)]['detected_as_unknown'] for c in u_classes]
    u_test_counts = [unknown[str(c)]['test_count'] for c in u_classes]
    
    colors_u = ['#2ecc71' if r > 0.4 else '#e67e22' if r > 0.2 else '#e74c3c' for r in u_det_rates]
    bars = ax.bar(u_names, u_det_rates, color=colors_u, edgecolor='black', linewidth=0.5)
    for bar, rate, count, total in zip(bars, u_det_rates, u_det_counts, u_test_counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{rate:.1%}\n({count}/{total})', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Unknown Detection Rate', fontsize=12)
    ax.set_title('Unknown Attack Detection\n(Leave-One-Class-Out)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend(fontsize=10)

# 9b: Class distribution in training vs test
ax = axes[1]
train_dist = np.bincount(attack[train_idx], minlength=10)
test_dist = np.bincount(attack[test_idx], minlength=10)
x_pos = np.arange(10)
width = 0.35
ax.bar(x_pos - width/2, train_dist / train_dist.sum(), width, label='Train', 
       color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.bar(x_pos + width/2, test_dist / test_dist.sum(), width, label='Test', 
       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([attack_names.get(i, f'T{i}') for i in range(10)], fontsize=8, rotation=30, ha='right')
ax.set_ylabel('Proportion', fontsize=12)
ax.set_title('Class Distribution\nTrain vs Test (temporal split)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('report/images/unknown_attack_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/unknown_attack_detection.png")

# ===================== Figure 10: Statistical Disentanglement Visualization =====================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Load group assignments
group_assignments = np.load('outputs/dids_group_assignments.npy')

# 10a: Group assignment heatmap (mean per attack type)
ax = axes[0]
n_groups = group_assignments.shape[1]
group_means = np.zeros((10, n_groups))
for a in range(10):
    mask = test_idx  # all test samples
    atk_mask = attack[test_idx] == a
    if atk_mask.sum() > 0:
        group_means[a] = group_assignments[atk_mask].mean(axis=0)

im = ax.imshow(group_means, cmap='YlOrRd', aspect='auto')
ax.set_yticks(range(10))
ax.set_yticklabels([attack_names.get(i, f'T{i}') for i in range(10)], fontsize=9)
ax.set_xticks(range(n_groups))
ax.set_xticklabels([f'Group {i}' for i in range(n_groups)], fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title('Statistical Disentanglement:\nGroup Assignment by Attack Type', fontsize=12, fontweight='bold')

# 10b: Group assignment distribution for benign vs attack
ax = axes[1]
benign_groups = group_assignments[y_test_bin == 0].mean(axis=0)
attack_groups = group_assignments[y_test_bin == 1].mean(axis=0)
x_pos = np.arange(n_groups)
width = 0.35
ax.bar(x_pos - width/2, benign_groups, width, label='Benign', color='#2ecc71', edgecolor='black')
ax.bar(x_pos + width/2, attack_groups, width, label='Attack', color='#e74c3c', edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'G{i}' for i in range(n_groups)], fontsize=10)
ax.set_ylabel('Mean Assignment Weight', fontsize=11)
ax.set_title('Group Weights:\nBenign vs Attack', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)

# 10c: t-SNE of disentangled features colored by group
ax = axes[2]
np.random.seed(42)
n_vis = min(2000, group_assignments.shape[0])
vis_idx = np.random.choice(group_assignments.shape[0], n_vis, replace=False)
dominant_group = group_assignments[vis_idx].argmax(axis=1)
scatter = ax.scatter(dids_emb[vis_idx, 0], dids_emb[vis_idx, 1], 
                     c=dominant_group, cmap='Set2', s=5, alpha=0.6, edgecolors='none')
ax.set_title('DIDS-MFL Embeddings\n(colored by dominant group)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_groups))
cbar.set_ticklabels([f'Group {i}' for i in range(n_groups)], fontsize=9)

plt.tight_layout()
plt.savefig('report/images/disentanglement_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/disentanglement_analysis.png")

print("\nAll figures generated successfully!")
