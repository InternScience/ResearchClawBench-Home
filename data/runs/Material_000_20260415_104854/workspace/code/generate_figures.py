"""
Generate all figures for the research report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import os
import torch

os.makedirs("report/images", exist_ok=True)

# Load results
with open("outputs/experiment_results.json") as f:
    results = json.load(f)

# Load predictions
with open("outputs/candidate_predictions.json") as f:
    pred_data = json.load(f)

cand_probs = np.array([p['probability'] for p in pred_data])
cand_labels = np.array([p['true_label'] for p in pred_data])
cand_preds = np.array([p['predicted'] for p in pred_data])

# Also load individual model predictions if available
gnn_probs = np.array([p.get('gnn_probability', 0) for p in pred_data]) if 'gnn_probability' in pred_data[0] else None
rf_probs = np.load("outputs/rf_predictions.npy") if os.path.exists("outputs/rf_predictions.npy") else None
gb_probs = np.load("outputs/gb_predictions.npy") if os.path.exists("outputs/gb_predictions.npy") else None

# Load data for analysis
finetune_data = torch.load("data/finetune_data.pt", map_location="cpu", weights_only=False)
pretrain_data = torch.load("data/pretrain_data.pt", map_location="cpu", weights_only=False)
candidate_data = torch.load("data/candidate_data.pt", map_location="cpu", weights_only=False)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150
})

# ============================================================
# Figure 1: Data Overview
# ============================================================
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# 1a: Dataset sizes
ax1 = fig.add_subplot(gs[0, 0])
datasets = ['Pre-training\n(Unlabeled)', 'Fine-tuning\n(Labeled)', 'Candidate\n(Unlabeled)']
sizes = [len(pretrain_data), len(finetune_data), len(candidate_data)]
colors = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax1.bar(datasets, sizes, color=colors, edgecolor='white', linewidth=1.5)
for bar, size in zip(bars, sizes):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
             str(size), ha='center', va='bottom', fontweight='bold', fontsize=12)
ax1.set_ylabel('Number of Samples')
ax1.set_title('Dataset Composition')
ax1.set_ylim(0, max(sizes) * 1.15)

# 1b: Class distribution in fine-tuning data
ax2 = fig.add_subplot(gs[0, 1])
ft_labels = [int(finetune_data[i].y.item()) for i in range(len(finetune_data))]
pos_count = sum(ft_labels)
neg_count = len(ft_labels) - pos_count
labels_pie = ['Non-Altermagnetic', 'Altermagnetic']
sizes_pie = [neg_count, pos_count]
colors_pie = ['#E57373', '#64B5F6']
wedges, texts, autotexts = ax2.pie(sizes_pie, labels=labels_pie, colors=colors_pie,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 10})
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight('bold')
ax2.set_title('Fine-tuning Class Distribution')

# 1c: Graph size distribution
ax3 = fig.add_subplot(gs[0, 2])
pos_idx = [i for i in range(len(finetune_data)) if finetune_data[i].y.item() == 1]
neg_idx = [i for i in range(len(finetune_data)) if finetune_data[i].y.item() == 0]
pos_sizes = [finetune_data[i].x.shape[0] for i in pos_idx]
neg_sizes = [finetune_data[i].x.shape[0] for i in neg_idx]
bins = np.arange(2, 26, 2)
ax3.hist(neg_sizes, bins=bins, alpha=0.6, label='Non-Altermagnetic', color='#E57373', density=True)
ax3.hist(pos_sizes, bins=bins, alpha=0.6, label='Altermagnetic', color='#64B5F6', density=True)
ax3.set_xlabel('Number of Nodes')
ax3.set_ylabel('Density')
ax3.set_title('Graph Size Distribution')
ax3.legend()

# 1d: Element frequency comparison
ax4 = fig.add_subplot(gs[1, :2])
elem_to_idx = finetune_data.elem_to_idx
idx_to_elem = {v: k for k, v in elem_to_idx.items()}

def get_elem_freq(dataset, indices):
    freq = np.zeros(28)
    total = 0
    for idx in indices:
        item = dataset[idx]
        x = item.x.numpy()
        ei = np.argmax(x, axis=-1)
        total += len(ei)
        for e in ei:
            freq[e] += 1
    return freq / max(total, 1)

pos_freq = get_elem_freq(finetune_data, pos_idx)
neg_freq = get_elem_freq(finetune_data, neg_idx)
elem_names = [idx_to_elem[i] for i in range(28)]
x_pos = np.arange(28)
width = 0.35

bars1 = ax4.bar(x_pos - width/2, neg_freq, width, label='Non-Altermagnetic', color='#E57373', alpha=0.8)
bars2 = ax4.bar(x_pos + width/2, pos_freq, width, label='Altermagnetic', color='#64B5F6', alpha=0.8)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(elem_names, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Element Frequency')
ax4.set_title('Element Composition Comparison')
ax4.legend()

# 1e: Edge attribute distribution
ax5 = fig.add_subplot(gs[1, 2])
pos_edge_attrs = []
neg_edge_attrs = []
for idx in pos_idx[:50]:
    item = finetune_data[idx]
    if item.edge_attr is not None:
        pos_edge_attrs.append(item.edge_attr.numpy().flatten())
for idx in neg_idx[:50]:
    item = finetune_data[idx]
    if item.edge_attr is not None:
        neg_edge_attrs.append(item.edge_attr.numpy().flatten())

if pos_edge_attrs and neg_edge_attrs:
    pos_ea = np.concatenate(pos_edge_attrs)
    neg_ea = np.concatenate(neg_edge_attrs)
    ax5.hist(neg_ea, bins=30, alpha=0.6, label='Non-Altermagnetic', color='#E57373', density=True)
    ax5.hist(pos_ea, bins=30, alpha=0.6, label='Altermagnetic', color='#64B5F6', density=True)
ax5.set_xlabel('Edge Attribute Value')
ax5.set_ylabel('Density')
ax5.set_title('Edge Attribute Distribution')
ax5.legend()

plt.savefig("report/images/fig1_data_overview.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig1_data_overview.png")

# ============================================================
# Figure 2: Training Results
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 2a: Pre-training loss curve
ax = axes[0]
pretrain_losses = results['pretrain_losses']
ax.plot(range(1, len(pretrain_losses)+1), pretrain_losses, 'b-', linewidth=2, label='Contrastive Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Self-Supervised Pre-Training')
ax.legend()
ax.grid(True, alpha=0.3)

# 2b: Validation metrics over time
ax = axes[1]
val_history = results.get('val_metrics_history', [])
if val_history:
    epochs = range(1, len(val_history)+1)
    roc_aucs = [m['roc_auc'] for m in val_history]
    f1s = [m['f1'] for m in val_history]
    accs = [m['accuracy'] for m in val_history]
    ax.plot(epochs, roc_aucs, 'b-', linewidth=2, label='ROC-AUC')
    ax.plot(epochs, f1s, 'r-', linewidth=2, label='F1 Score')
    ax.plot(epochs, accs, 'g-', linewidth=2, label='Accuracy')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('Validation Metrics During Fine-Tuning')
ax.legend()
ax.grid(True, alpha=0.3)

# 2c: Method comparison bar chart
ax = axes[2]
methods = ['GNN\nEnsemble', 'Random\nForest', 'Gradient\nBoosting', 'Combined\n(GNN+RF+GB)']
roc_values = [
    results.get('gnn_ensemble_roc', results.get('ensemble_mean_roc', 0.5)),
    results.get('handcrafted_rf_roc', 0.5),
    results.get('handcrafted_gb_roc', 0.5),
    results.get('combined_roc', 0.5)
]
bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax.bar(methods, roc_values, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, roc_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
ax.set_ylabel('ROC-AUC')
ax.set_title('Method Comparison on Candidate Set')
ax.legend()
ax.set_ylim(0.4, max(roc_values) * 1.15 if max(roc_values) > 0.5 else 0.6)

plt.savefig("report/images/fig2_training_results.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig2_training_results.png")

# ============================================================
# Figure 3: ROC and PR Curves
# ============================================================
from sklearn.metrics import roc_curve, precision_recall_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3a: ROC Curve
ax = axes[0]
fpr, tpr, _ = roc_curve(cand_labels, cand_probs)
roc_auc_val = auc(fpr, tpr)
ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'Combined Model (AUC={roc_auc_val:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# 3b: Precision-Recall Curve
ax = axes[1]
precision_arr, recall_arr, _ = precision_recall_curve(cand_labels, cand_probs)
pr_auc_val = auc(recall_arr, precision_arr)
ax.plot(recall_arr, precision_arr, 'b-', linewidth=2.5, label=f'Combined Model (AUC={pr_auc_val:.4f})')
baseline = sum(cand_labels) / len(cand_labels)
ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, label=f'Baseline ({baseline:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

plt.savefig("report/images/fig3_roc_pr_curves.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig3_roc_pr_curves.png")

# ============================================================
# Figure 4: Prediction Analysis
# ============================================================
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

# 4a: Probability distributions
ax1 = fig.add_subplot(gs[0, 0])
tp_probs = cand_probs[cand_labels == 1]
tn_probs = cand_probs[cand_labels == 0]
ax1.hist(tn_probs, bins=50, alpha=0.6, label=f'Non-Altermagnetic (n={len(tn_probs)})', color='#E57373', density=True)
ax1.hist(tp_probs, bins=50, alpha=0.6, label=f'Altermagnetic (n={len(tp_probs)})', color='#64B5F6', density=True)
ax1.axvline(x=results.get('optimal_threshold', 0.5), color='green', linestyle='--', linewidth=2, label=f'Threshold ({results.get("optimal_threshold", 0.5):.2f})')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Density')
ax1.set_title('Prediction Probability Distribution')
ax1.legend()

# 4b: Confusion matrix
ax2 = fig.add_subplot(gs[0, 1])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(cand_labels, cand_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Pred Negative', 'Pred Positive'],
            yticklabels=['True Negative', 'True Positive'])
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
ax2.set_title('Confusion Matrix')

# 4c: Top predictions ranking
ax3 = fig.add_subplot(gs[1, 0])
sorted_idx = np.argsort(cand_probs)[::-1]
top_n = 50
top_probs = cand_probs[sorted_idx[:top_n]]
top_labels = cand_labels[sorted_idx[:top_n]]
colors_top = ['#64B5F6' if l == 1 else '#E57373' for l in top_labels]
ax3.bar(range(top_n), top_probs, color=colors_top, edgecolor='white', linewidth=0.5)
ax3.axhline(y=results.get('optimal_threshold', 0.5), color='green', linestyle='--', linewidth=2, label='Threshold')
ax3.set_xlabel('Rank')
ax3.set_ylabel('Predicted Probability')
ax3.set_title(f'Top {top_n} Candidates by Predicted Probability')
ax3.legend()
blue_patch = plt.Rectangle((0, 0), 1, 1, fc='#64B5F6', label='True Altermagnetic')
red_patch = plt.Rectangle((0, 0), 1, 1, fc='#E57373', label='Non-Altermagnetic')
ax3.legend(handles=[blue_patch, red_patch])

# 4d: Discovery statistics
ax4 = fig.add_subplot(gs[1, 1])
categories = ['Total\nCandidates', 'Predicted\nAltermagnets', 'True\nPositives', 'False\nPositives']
values = [
    len(cand_labels),
    int(sum(cand_preds)),
    int(sum(cand_labels[cand_preds == 1])),
    int(sum((1 - cand_labels)[cand_preds == 1]))
]
cat_colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF5722']
bars = ax4.bar(categories, values, color=cat_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
ax4.set_ylabel('Count')
ax4.set_title('Discovery Statistics')

plt.savefig("report/images/fig4_prediction_analysis.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig4_prediction_analysis.png")

# ============================================================
# Figure 5: Feature Importance & Ablation
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5a: Feature importance from RF
ax = axes[0]
if os.path.exists("outputs/experiment_results.json"):
    with open("outputs/experiment_results.json") as f:
        res = json.load(f)
    fi = res.get('feature_importances', [])
    if fi:
        feature_names = ['n_nodes', 'n_edges', 'avg_degree', 'edge_density', 'n_unique',
                        'max_degree', 'min_degree', 'degree_std', 'degree_mean',
                        'n_magnetic', 'n_nonmagnetic', 'magnetic_ratio',
                        'n_rare_earth', 'rare_earth_ratio',
                        'n_halogen', 'halogen_ratio',
                        'n_chalcogen', 'chalcogen_ratio']
        elem_to_idx_map = {'Fe': 0, 'Co': 1, 'Ni': 2, 'Mn': 3, 'Cr': 4, 'V': 5, 'Ti': 6, 
                          'Nd': 7, 'Pr': 8, 'Sm': 9, 'Gd': 10, 'Ho': 11, 'Er': 12, 'Yb': 13,
                          'O': 14, 'F': 15, 'Cl': 16, 'Br': 17, 'I': 18, 'S': 19, 'Se': 20,
                          'Te': 21, 'B': 22, 'C': 23, 'N': 24, 'P': 25, 'Si': 26, 'H': 27}
        idx_to_elem = {v: k for k, v in elem_to_idx_map.items()}
        for idx in range(28):
            feature_names.append(f'elem_present_{idx_to_elem[idx]}')
        for idx in range(28):
            feature_names.append(f'elem_frac_{idx_to_elem[idx]}')
        feature_names.extend(['edge_mean_0', 'edge_mean_1', 'edge_std_0', 'edge_std_1'])
        
        fi_arr = np.array(fi)
        top_idx = np.argsort(fi_arr)[-15:][::-1]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = fi_arr[top_idx]
        
        ax.barh(range(len(top_names)), top_vals, color='#2196F3', edgecolor='white')
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=9)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Features by Importance (Random Forest)')
        ax.invert_yaxis()

# 5b: Performance vs training epochs (ablation)
ax = axes[1]
if val_history:
    epochs_list = list(range(1, len(val_history)+1))
    roc_list = [m['roc_auc'] for m in val_history]
    ax.plot(epochs_list, roc_list, 'b-o', markersize=3, linewidth=1.5)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    best_epoch = np.argmax(roc_list) + 1
    best_roc = max(roc_list)
    ax.annotate(f'Best: {best_roc:.4f}\n@ epoch {best_epoch}',
                xy=(best_epoch, best_roc), xytext=(best_epoch*0.6, best_roc*0.7),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('Training Epoch')
ax.set_ylabel('ROC-AUC')
ax.set_title('Model Performance vs Training Duration')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig("report/images/fig5_feature_ablation.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig5_feature_ablation.png")

# ============================================================
# Figure 6: Element Analysis Deep Dive
# ============================================================
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 3, hspace=0.3, wspace=0.3)

# 6a: Magnetic element presence
ax1 = fig.add_subplot(gs[0, 0])
magnetic_elems = {'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'V', 'Gd', 'Nd', 'Sm', 'Ho', 'Er'}

def count_magnetic(dataset, indices):
    counts = []
    for idx in indices:
        item = dataset[idx]
        x = item.x.numpy()
        ei = np.argmax(x, axis=-1)
        elems = set(list(finetune_data.elem_to_idx.keys())[e] for e in ei)
        mag_count = len(elems.intersection(magnetic_elems))
        counts.append(mag_count)
    return counts

pos_mag = count_magnetic(finetune_data, pos_idx)
neg_mag = count_magnetic(finetune_data, neg_idx[:500])

ax1.hist(neg_mag, bins=np.arange(-0.5, 8.5, 1), alpha=0.6, label='Non-Altermagnetic', color='#E57373', density=True)
ax1.hist(pos_mag, bins=np.arange(-0.5, 8.5, 1), alpha=0.6, label='Altermagnetic', color='#64B5F6', density=True)
ax1.set_xlabel('Number of Magnetic Elements')
ax1.set_ylabel('Density')
ax1.set_title('Magnetic Element Count Distribution')
ax1.legend()

# 6b: Element combination analysis
ax2 = fig.add_subplot(gs[0, 1])
categories_comb = ['Mag+RE+Hal\n(All Three)', 'Mag+RE\nOnly', 'Mag+Hal\nOnly', 'RE+Hal\nOnly', 'Single\nCategory', 'None']
counts_comb = [0, 0, 0, 0, 0, 0]
rare_earth = {'Nd', 'Pr', 'Sm', 'Gd', 'Ho', 'Er', 'Yb'}
halogens = {'O', 'F', 'Cl', 'Br', 'I'}

for idx in pos_idx:
    item = finetune_data[idx]
    x = item.x.numpy()
    ei = np.argmax(x, axis=-1)
    elems = set(list(finetune_data.elem_to_idx.keys())[e] for e in ei)
    
    has_mag = bool(elems.intersection(magnetic_elems))
    has_re = bool(elems.intersection(rare_earth))
    has_hal = bool(elems.intersection(halogens))
    
    n_cats = sum([has_mag, has_re, has_hal])
    if n_cats == 3: counts_comb[0] += 1
    elif has_mag and has_re: counts_comb[1] += 1
    elif has_mag and has_hal: counts_comb[2] += 1
    elif has_re and has_hal: counts_comb[3] += 1
    elif n_cats == 1: counts_comb[4] += 1
    else: counts_comb[5] += 1

comb_colors = ['#1B5E20', '#2E7D32', '#4CAF50', '#81C784', '#C8E6C9', '#E8F5E9']
bars = ax2.bar(categories_comb, counts_comb, color=comb_colors, edgecolor='white')
for bar, val in zip(bars, counts_comb):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
ax2.set_ylabel('Count')
ax2.set_title('Element Category Combinations\nin Altermagnetic Samples')
ax2.tick_params(axis='x', rotation=30)

# 6c: Candidate discovery breakdown
ax3 = fig.add_subplot(gs[0, 2])
cand_pos_idx = [i for i in range(len(candidate_data)) if candidate_data[i].y.item() == 1]
cand_neg_idx = [i for i in range(len(candidate_data)) if candidate_data[i].y.item() == 0]

# Show which candidates were correctly identified
correct_tp = sum(1 for i in cand_pos_idx if cand_preds[i] == 1)
missed_fn = sum(1 for i in cand_pos_idx if cand_preds[i] == 0)
false_fp = sum(1 for i in cand_neg_idx if cand_preds[i] == 1)
correct_tn = sum(1 for i in cand_neg_idx if cand_preds[i] == 0)

disc_categories = ['True Positives\n(Correctly Found)', 'False Negatives\n(Missed)', 'False Positives\n(False Alarm)', 'True Negatives\n(Correctly Rejected)']
disc_values = [correct_tp, missed_fn, false_fp, correct_tn]
disc_colors = ['#4CAF50', '#FF5722', '#FFC107', '#2196F3']
bars = ax3.bar(disc_categories, disc_values, color=disc_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, disc_values):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax3.set_ylabel('Count')
ax3.set_title('Candidate Classification Breakdown')
ax3.tick_params(axis='x', rotation=15)

plt.savefig("report/images/fig6_element_analysis.png", bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig6_element_analysis.png")

print("\nAll figures saved to report/images/")
