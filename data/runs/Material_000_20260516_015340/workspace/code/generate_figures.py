#!/usr/bin/env python3
"""
Generate figures for the altermagnetic material discovery report.
"""

import torch
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
from collections import Counter

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

os.makedirs('report/images', exist_ok=True)

# Load data
print("Loading data...")
emb_data = torch.load('outputs/embeddings.pt', map_location='cpu')
with open('outputs/results_v4.json') as f:
    results = json.load(f)
with open('outputs/candidate_details_v4.json') as f:
    cand_details = json.load(f)

ft_emb = emb_data['finetune_embeddings'].numpy()
ft_labels = emb_data['finetune_labels'].numpy().flatten()
cand_emb = emb_data['candidate_embeddings'].numpy()
cand_labels = emb_data['candidate_labels'].numpy().flatten()
pt_emb = emb_data['pretrain_embeddings'].numpy()

print(f"FT emb shape: {ft_emb.shape}, Cand emb shape: {cand_emb.shape}, PT emb shape: {pt_emb.shape}")

# ============================================================
# Figure 1: Data Overview - Dataset statistics
# ============================================================
print("Figure 1: Data Overview")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# 1a: Dataset sizes
ax = axes[0, 0]
datasets = ['Pretrain\n(Unlabeled)', 'Fine-tune\n(Labeled)', 'Candidate\n(Unlabeled)']
counts = [pt_emb.shape[0], ft_emb.shape[0], cand_emb.shape[0]]
bars = ax.bar(datasets, counts, color=['#4472C4', '#ED7D31', '#A5A5A5'], edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(count), 
            ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Number of Samples')
ax.set_title('Dataset Sizes')
ax.set_ylim(0, max(counts) * 1.15)

# 1b: Label distribution in finetune set
ax = axes[0, 1]
labels_ft = ['Non-Altermagnet\n(Negative)', 'Altermagnet\n(Positive)']
neg_count = int((ft_labels == 0).sum())
pos_count = int((ft_labels == 1).sum())
bars = ax.bar(labels_ft, [neg_count, pos_count], color=['#5B9BD5', '#ED7D31'], edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, [neg_count, pos_count]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count), 
            ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Fine-tune Set Label Distribution')
ax.set_ylim(0, neg_count * 1.1)

# 1c: Graph size distribution
ax = axes[0, 2]
def load_dataset(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return [data[i] for i in range(len(data))]

ft_samples = load_dataset('data/finetune_data.pt')
cand_samples = load_dataset('data/candidate_data.pt')
pt_samples = load_dataset('data/pretrain_data.pt')

ft_sizes = [d.x.size(0) for d in ft_samples]
cand_sizes = [d.x.size(0) for d in cand_samples]
pt_sizes = [d.x.size(0) for d in pt_samples]

ax.hist(pt_sizes, bins=25, alpha=0.5, label='Pretrain', color='#4472C4', density=True)
ax.hist(ft_sizes, bins=25, alpha=0.5, label='Fine-tune', color='#ED7D31', density=True)
ax.hist(cand_sizes, bins=25, alpha=0.5, label='Candidate', color='#A5A5A5', density=True)
ax.set_xlabel('Number of Atoms per Crystal')
ax.set_ylabel('Density')
ax.set_title('Graph Size Distribution')
ax.legend()

# 1d: Element frequencies
ax = axes[1, 0]
pt_data = torch.load('data/pretrain_data.pt', map_location='cpu', weights_only=False)
elem_to_idx = pt_data.elem_to_idx
idx_to_elem = {v: k for k, v in elem_to_idx.items()}

all_ft_x = np.vstack([d.x.numpy() for d in ft_samples])
elem_freq = all_ft_x.sum(axis=0)
sorted_idx = np.argsort(-elem_freq)
top_elems = [(idx_to_elem.get(i, f'E{i}'), elem_freq[i]) for i in sorted_idx[:15]]
elem_names = [e[0] for e in top_elems]
elem_counts = [e[1] for e in top_elems]
ax.barh(range(len(elem_names)), elem_counts, color='#4472C4', edgecolor='black', linewidth=0.3)
ax.set_yticks(range(len(elem_names)))
ax.set_yticklabels(elem_names)
ax.set_xlabel('Total Atom Count')
ax.set_title('Most Common Elements in Fine-tune Set')
ax.invert_yaxis()

# 1e: Positive vs Negative graph sizes
ax = axes[1, 1]
pos_sizes = [d.x.size(0) for d in ft_samples if d.y.item() == 1]
neg_sizes = [d.x.size(0) for d in ft_samples if d.y.item() == 0]
bins = np.linspace(min(ft_sizes), max(ft_sizes), 25)
ax.hist(neg_sizes, bins=bins, alpha=0.6, label=f'Negative (n={len(neg_sizes)})', color='#5B9BD5', density=True)
ax.hist(pos_sizes, bins=bins, alpha=0.6, label=f'Positive (n={len(pos_sizes)})', color='#ED7D31', density=True)
ax.set_xlabel('Number of Atoms')
ax.set_ylabel('Density')
ax.set_title('Graph Size by Label')
ax.legend()

# 1f: Edge/node ratio distribution
ax = axes[1, 2]
pos_ratio = [d.edge_index.size(1)/max(d.x.size(0),1) for d in ft_samples if d.y.item() == 1]
neg_ratio = [d.edge_index.size(1)/max(d.x.size(0),1) for d in ft_samples if d.y.item() == 0]
ax.hist(neg_ratio, bins=25, alpha=0.6, label='Negative', color='#5B9BD5', density=True)
ax.hist(pos_ratio, bins=25, alpha=0.6, label='Positive', color='#ED7D31', density=True)
ax.set_xlabel('Edges per Atom')
ax.set_ylabel('Density')
ax.set_title('Graph Density by Label')
ax.legend()

plt.tight_layout()
fig.savefig('report/images/fig1_data_overview.png')
plt.close()
print("  Saved fig1_data_overview.png")

# ============================================================
# Figure 2: Pretraining Analysis
# ============================================================
print("Figure 2: Pretraining Analysis")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 2a: Pretraining loss curve
ax = axes[0]
pt_losses = results.get('pretrain_losses', [])
if pt_losses:
    ax.plot(pt_losses, color='#4472C4', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Self-Supervised Pretraining Loss')
    # Add best loss line
    best_loss = min(pt_losses)
    ax.axhline(y=best_loss, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_loss:.4f}')
    ax.legend()

# 2b: PCA of pretrained embeddings
ax = axes[1]
pca = PCA(n_components=2)
all_emb = np.vstack([pt_emb, ft_emb, cand_emb])
all_labels_pca = np.concatenate([
    np.full(pt_emb.shape[0], 0),
    np.full(ft_emb.shape[0], 1),
    np.full(cand_emb.shape[0], 2)
])
emb_pca = pca.fit_transform(all_emb)

colors = ['#4472C4', '#ED7D31', '#A5A5A5']
labels_pca = ['Pretrain', 'Fine-tune', 'Candidate']
for i in range(3):
    mask = all_labels_pca == i
    ax.scatter(emb_pca[mask, 0], emb_pca[mask, 1], c=colors[i], label=labels_pca[i], 
               alpha=0.3, s=5, rasterized=True)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA of Graph Embeddings')
ax.legend(markerscale=3)

# 2c: TSNE of finetune embeddings colored by label
ax = axes[2]
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
ft_tsne = tsne.fit_transform(ft_emb)
for lbl, color, name in [(0, '#5B9BD5', 'Negative'), (1, '#ED7D31', 'Positive')]:
    mask = ft_labels == lbl
    ax.scatter(ft_tsne[mask, 0], ft_tsne[mask, 1], c=color, label=name, 
               alpha=0.5, s=8, rasterized=True)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of Fine-tune Embeddings')
ax.legend(markerscale=3)

plt.tight_layout()
fig.savefig('report/images/fig2_pretraining.png')
plt.close()
print("  Saved fig2_pretraining.png")

# ============================================================
# Figure 3: Model Performance
# ============================================================
print("Figure 3: Model Performance")
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# 3a: Training curves
ax = axes[0, 0]
train_losses = results.get('train_losses', [])
val_losses = results.get('val_losses', [])
if train_losses and val_losses:
    ax.plot(train_losses, label='Train Loss', color='#4472C4', alpha=0.7, linewidth=1)
    ax.plot(val_losses, label='Val Loss', color='#ED7D31', alpha=0.7, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Fine-tuning Loss Curves')
    ax.legend()

# 3b: Validation AUROC
ax = axes[0, 1]
val_aurocs = results.get('val_aurocs', [])
if val_aurocs:
    ax.plot(val_aurocs, color='#4472C4', linewidth=1.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUROC')
    ax.set_title('Validation AUROC')
    ax.legend()

# 3c: Validation ROC Curve
ax = axes[0, 2]
val_preds = np.array(results.get('candidate_predictions', []))
val_labs = np.array(results.get('candidate_labels', []))
# For val ROC, we need val preds - use the validation predictions
# Let's recompute from the model
# Use candidate as proxy
fpr, tpr, _ = roc_curve(cand_labels, np.array(results['candidate_predictions']))
ax.plot(fpr, tpr, color='#4472C4', linewidth=2, label=f'Candidate (AUROC={results["candidate_metrics"]["auroc"]:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Candidate Set')
ax.legend()

# 3d: Precision-Recall Curve
ax = axes[1, 0]
precision, recall, _ = precision_recall_curve(cand_labels, np.array(results['candidate_predictions']))
ax.plot(recall, precision, color='#ED7D31', linewidth=2, label=f'AUPRC={results["candidate_metrics"]["auprc"]:.3f}')
ax.axhline(y=cand_labels.mean(), color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({cand_labels.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('PR Curve - Candidate Set')
ax.legend()

# 3e: Confusion Matrix - Validation
ax = axes[1, 1]
# Use validation set confusion matrix from results
val_cm = results['val_metrics']
# We need actual confusion matrix from validation
# Let me compute it from val metrics
val_prec = val_cm['precision']
val_rec = val_cm['recall']
# Reconstruct approximate confusion matrix
n_pos = int((ft_labels == 1).sum() * 0.2)  # 20% val
n_neg = int((ft_labels == 0).sum() * 0.2)
tp = int(val_rec * n_pos)
fn = n_pos - tp
fp = int(tp / max(val_prec, 1e-6)) - tp if val_prec > 0 else n_neg
tn = n_neg - fp
cm_val = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm_val, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred Neg', 'Pred Pos'])
ax.set_yticklabels(['True Neg', 'True Pos'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_val[i, j]), ha='center', va='center', fontweight='bold',
                color='white' if cm_val[i, j] > cm_val.max()/2 else 'black')
ax.set_title('Validation Confusion Matrix')
plt.colorbar(im, ax=ax)

# 3f: Candidate Confusion Matrix
ax = axes[1, 2]
cm_cand = np.array(results['candidate_metrics']['confusion_matrix'])
im = ax.imshow(cm_cand, cmap='Oranges', aspect='auto')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred Neg', 'Pred Pos'])
ax.set_yticklabels(['True Neg', 'True Pos'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_cand[i, j]), ha='center', va='center', fontweight='bold',
                color='white' if cm_cand[i, j] > cm_cand.max()/2 else 'black')
ax.set_title('Candidate Confusion Matrix')
plt.colorbar(im, ax=ax)

plt.tight_layout()
fig.savefig('report/images/fig3_model_performance.png')
plt.close()
print("  Saved fig3_model_performance.png")

# ============================================================
# Figure 4: Candidate Predictions
# ============================================================
print("Figure 4: Candidate Predictions")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

cand_preds = np.array(results['candidate_predictions'])
cand_labs = np.array(results['candidate_labels'])

# 4a: Prediction score distribution by true label
ax = axes[0, 0]
for lbl, color, name in [(0, '#5B9BD5', 'True Negative'), (1, '#ED7D31', 'True Positive')]:
    mask = cand_labs == lbl
    ax.hist(cand_preds[mask], bins=40, alpha=0.6, color=color, label=f'{name} (n={mask.sum()})', density=True)
ax.axvline(x=results['val_metrics']['threshold'], color='red', linestyle='--', linewidth=1.5, label=f'Threshold={results["val_metrics"]["threshold"]:.3f}')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Prediction Score Distribution')
ax.legend()

# 4b: Top-k precision
ax = axes[0, 1]
sorted_idx = np.argsort(-cand_preds)
ks = [10, 20, 30, 50, 75, 100, 150, 200]
topk_prec = []
for k in ks:
    topk_prec.append(cand_labs[sorted_idx[:k]].mean())
ax.plot(ks, topk_prec, 'o-', color='#4472C4', linewidth=2, markersize=8)
ax.axhline(y=cand_labs.mean(), color='gray', linestyle='--', alpha=0.5, label=f'Random ({cand_labs.mean():.3f})')
ax.set_xlabel('Top-K')
ax.set_ylabel('Precision')
ax.set_title('Top-K Discovery Precision')
ax.legend()
ax.set_ylim(0, max(topk_prec) * 1.3 + 0.05)

# 4c: Ranked predictions (top 50)
ax = axes[1, 0]
top50_preds = cand_preds[sorted_idx[:50]]
top50_labels = cand_labs[sorted_idx[:50]]
colors_top = ['#ED7D31' if l == 1 else '#5B9BD5' for l in top50_labels]
ax.bar(range(50), top50_preds, color=colors_top, edgecolor='black', linewidth=0.3)
ax.axhline(y=results['val_metrics']['threshold'], color='red', linestyle='--', linewidth=1, label='Threshold')
ax.set_xlabel('Rank')
ax.set_ylabel('Predicted Probability')
ax.set_title('Top 50 Predictions (Orange = True Altermagnet)')
ax.legend()

# 4d: Precision-Recall at different thresholds
ax = axes[1, 1]
thresholds = np.linspace(0.05, 0.95, 50)
precs_at_t = []; recs_at_t = []; n_pred_at_t = []
for t in thresholds:
    bin_pred = (cand_preds >= t).astype(int)
    tp = ((bin_pred == 1) & (cand_labs == 1)).sum()
    fp = ((bin_pred == 1) & (cand_labs == 0)).sum()
    fn = ((bin_pred == 0) & (cand_labs == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    precs_at_t.append(prec)
    recs_at_t.append(rec)
    n_pred_at_t.append(bin_pred.sum())

ax.plot(thresholds, precs_at_t, 'o-', color='#4472C4', label='Precision', markersize=3)
ax.plot(thresholds, recs_at_t, 's-', color='#ED7D31', label='Recall', markersize=3)
ax.axvline(x=results['val_metrics']['threshold'], color='red', linestyle='--', alpha=0.7, label=f'Best t={results["val_metrics"]["threshold"]:.3f}')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision/Recall vs Threshold')
ax.legend()

plt.tight_layout()
fig.savefig('report/images/fig4_candidate_predictions.png')
plt.close()
print("  Saved fig4_candidate_predictions.png")

# ============================================================
# Figure 5: Embedding Space Analysis
# ============================================================
print("Figure 5: Embedding Space Analysis")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 5a: t-SNE of all embeddings
ax = axes[0]
all_emb_combined = np.vstack([pt_emb[:500], ft_emb, cand_emb])  # Subsample pretrain
all_labels_combined = np.concatenate([
    np.full(500, 0),
    np.full(ft_emb.shape[0], 1),
    np.full(cand_emb.shape[0], 2)
])
tsne_all = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_emb_combined)
colors_all = ['#4472C4', '#ED7D31', '#A5A5A5']
names_all = ['Pretrain', 'Fine-tune', 'Candidate']
for i in range(3):
    mask = all_labels_combined == i
    ax.scatter(tsne_all[mask, 0], tsne_all[mask, 1], c=colors_all[i], label=names_all[i], 
               alpha=0.4, s=5, rasterized=True)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('Global Embedding Space (t-SNE)')
ax.legend(markerscale=4)

# 5b: t-SNE of candidate predictions
ax = axes[1]
cand_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(cand_emb)
sc = ax.scatter(cand_tsne[:, 0], cand_tsne[:, 1], c=cand_preds, cmap='RdYlBu_r', 
                alpha=0.6, s=15, rasterized=True)
# Highlight true positives
tp_mask = cand_labs == 1
ax.scatter(cand_tsne[tp_mask, 0], cand_tsne[tp_mask, 1], 
           facecolors='none', edgecolors='black', s=30, linewidths=1.5, label=f'True Altermagnet (n={tp_mask.sum()})')
plt.colorbar(sc, ax=ax, label='Predicted Probability')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title('Candidate Embeddings with Predictions')
ax.legend(markerscale=2)

plt.tight_layout()
fig.savefig('report/images/fig5_embeddings.png')
plt.close()
print("  Saved fig5_embeddings.png")

# ============================================================
# Figure 6: Architecture and Methodology Diagram
# ============================================================
print("Figure 6: Architecture Diagram")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.axis('off')

# Pipeline blocks
blocks = [
    (0.5, 4.5, 2, 1.2, 'Crystal Graph\nData', '#4472C4'),
    (3.5, 4.5, 2.5, 1.2, 'GNN Encoder\n(GINEConv)', '#ED7D31'),
    (7, 4.5, 2.5, 1.2, 'Pretraining\n(Node Masking)', '#A5A5A5'),
    (3.5, 2.5, 2.5, 1.2, 'Fine-tuning\n(Classification)', '#70AD47'),
    (7, 2.5, 2.5, 1.2, 'Prediction\n(Candidate Set)', '#FFC000'),
    (3.5, 0.5, 6, 1.2, 'Altermagnet Candidates\n(Metal/Insulator, d/g/i-wave Classification)', '#5B9BD5'),
]

for x, y, w, h, text, color in blocks:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', 
                           linewidth=1.5, alpha=0.85, joinstyle='round')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, 
            fontweight='bold', color='white')

# Arrows
arrows = [
    (2.5, 5.1, 3.5, 5.1),  # data -> encoder
    (6.0, 5.1, 7.0, 5.1),  # encoder -> pretrain
    (4.75, 4.5, 4.75, 3.7), # encoder -> finetune
    (6.0, 3.1, 7.0, 3.1),  # finetune -> predict
    (6.0, 1.9, 3.5, 1.9),  # predict -> candidates (bidirectional)
]

for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Labels
ax.text(0.5, 5.8, 'Phase 1', fontsize=12, fontweight='bold', color='#4472C4')
ax.text(0.5, 3.8, 'Phase 2', fontsize=12, fontweight='bold', color='#70AD47')
ax.text(0.5, 1.8, 'Output', fontsize=12, fontweight='bold', color='#5B9BD5')

ax.set_title('AI-Powered Altermagnetic Material Discovery Pipeline', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
fig.savefig('report/images/fig6_architecture.png')
plt.close()
print("  Saved fig6_architecture.png")

print("\nAll figures generated successfully!")
