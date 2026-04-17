"""
Generate all figures for the report
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.manifold import TSNE

WS = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_000_20260416_204143'
OUT = f'{WS}/outputs'
IMG = f'{WS}/report/images'
os.makedirs(IMG, exist_ok=True)

# Load data
results = json.load(open(f'{OUT}/evaluation_results.json'))
ce = np.load(f'{OUT}/candidate_embeddings.npy')
cd_y = np.load(f'{OUT}/candidate_true_labels.npy')
fe = np.load(f'{OUT}/finetune_embeddings.npy')
fl = np.load(f'{OUT}/finetune_labels.npy')
pt_losses = json.load(open(f'{OUT}/pretrain_losses.json'))

# Load all probs
probs = {}
for name in ['candidate_probs_gnn', 'candidate_probs_random', 'candidate_probs_gb', 
             'candidate_probs_rf', 'candidate_probs_hybrid']:
    try:
        probs[name] = np.load(f'{OUT}/{name}.npy')
    except:
        pass

best_probs = np.load(f'{OUT}/candidate_probs.npy')

# Load histories
pt_hists = []
ri_hists = []
for i in range(5):
    try:
        pt_hists.append(json.load(open(f'{OUT}/pt_hist_{i}.json')))
    except: pass
    try:
        ri_hists.append(json.load(open(f'{OUT}/ri_hist_{i}.json')))
    except: pass

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

# ============================================================
# Figure 1: Data Overview
# ============================================================
import torch, sys, types
dp = types.ModuleType('data_prepare')
class R:
    def __init__(self,*a,**k): pass
    def __setstate__(self,s): self.__dict__.update(s)
dp.RealisticCrystalDataset = R
sys.modules['data_prepare'] = dp

pt_data = torch.load(f'{WS}/data/pretrain_data.pt', weights_only=False).data_list
ft_data = torch.load(f'{WS}/data/finetune_data.pt', weights_only=False).data_list
cd_data = torch.load(f'{WS}/data/candidate_data.pt', weights_only=False).data_list

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1a: Node count distribution
for ax, data, name in [(axes[0,0], pt_data, 'Pre-train (5000)'), 
                         (axes[0,1], ft_data, 'Fine-tune (2000)'),
                         (axes[0,2], cd_data, 'Candidate (1000)')]:
    nodes = [d.x.size(0) for d in data]
    ax.hist(nodes, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_title(f'{name}\nNode Count Distribution')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Count')

# 1b: Edge count distribution
for ax, data, name in [(axes[1,0], pt_data, 'Pre-train'), 
                         (axes[1,1], ft_data, 'Fine-tune'),
                         (axes[1,2], cd_data, 'Candidate')]:
    edges = [d.edge_index.size(1) for d in data]
    ax.hist(edges, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax.set_title(f'{name}\nEdge Count Distribution')
    ax.set_xlabel('Number of Edges')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig(f'{IMG}/fig1_data_overview.png', bbox_inches='tight')
plt.close()
print("Fig 1 done")

# ============================================================
# Figure 2: Label Distribution
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = [
    ('Pre-train', [d.y.item() for d in pt_data]),
    ('Fine-tune', [d.y.item() for d in ft_data]),
    ('Candidate', [d.y.item() for d in cd_data])
]

colors = ['#2ecc71', '#e74c3c']
for ax, (name, labels) in zip(axes, datasets):
    pos = sum(labels)
    neg = len(labels) - pos
    bars = ax.bar(['Negative (0)', 'Positive (1)'], [neg, pos], color=colors, edgecolor='black')
    ax.set_title(f'{name} Dataset\n(n={len(labels)})')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, [neg, pos]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{val}\n({val/len(labels)*100:.1f}%)', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(f'{IMG}/fig2_label_distribution.png', bbox_inches='tight')
plt.close()
print("Fig 2 done")

# ============================================================
# Figure 3: Pre-training Loss Curve
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(pt_losses)+1), pt_losses, 'b-o', markersize=4, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Self-Supervised Pre-training Loss\n(Graph Property Prediction)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMG}/fig3_pretrain_loss.png', bbox_inches='tight')
plt.close()
print("Fig 3 done")

# ============================================================
# Figure 4: Fine-tuning Training Curves
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training loss
for i, h in enumerate(pt_hists):
    axes[0].plot(h['tl'], alpha=0.7, label=f'PT Model {i+1}')
for i, h in enumerate(ri_hists):
    axes[0].plot(h['tl'], '--', alpha=0.5, label=f'RI Model {i+1}')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Validation AUC
for i, h in enumerate(pt_hists):
    axes[1].plot(h['auc'], alpha=0.7, label=f'PT Model {i+1}')
for i, h in enumerate(ri_hists):
    axes[1].plot(h['auc'], '--', alpha=0.5, label=f'RI Model {i+1}')
axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC-ROC')
axes[1].set_title('Validation AUC-ROC'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# Validation AP
for i, h in enumerate(pt_hists):
    axes[2].plot(h['ap'], alpha=0.7, label=f'PT Model {i+1}')
for i, h in enumerate(ri_hists):
    axes[2].plot(h['ap'], '--', alpha=0.5, label=f'RI Model {i+1}')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Average Precision')
axes[2].set_title('Validation Average Precision'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG}/fig4_training_curves.png', bbox_inches='tight')
plt.close()
print("Fig 4 done")

# ============================================================
# Figure 5: ROC and PR Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

approach_names = {
    'candidate_probs_gnn': 'Pre-trained GIN',
    'candidate_probs_random': 'Random Init GIN',
    'candidate_probs_gb': 'Gradient Boosting',
    'candidate_probs_rf': 'Random Forest',
    'candidate_probs_hybrid': 'Hybrid (GNN+ML)'
}
colors_map = {'candidate_probs_gnn': 'blue', 'candidate_probs_random': 'red', 
              'candidate_probs_gb': 'green', 'candidate_probs_rf': 'orange',
              'candidate_probs_hybrid': 'purple'}

# ROC
for key, name in approach_names.items():
    if key in probs:
        fpr, tpr, _ = roc_curve(cd_y, probs[key])
        auc = roc_auc_score(cd_y, probs[key])
        axes[0].plot(fpr, tpr, color=colors_map[key], linewidth=2, label=f'{name} (AUC={auc:.3f})')
axes[0].plot([0,1],[0,1],'k--',alpha=0.5,label='Random')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve - Candidate Screening'); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# PR
base_rate = cd_y.sum() / len(cd_y)
for key, name in approach_names.items():
    if key in probs:
        prec, rec, _ = precision_recall_curve(cd_y, probs[key])
        ap = average_precision_score(cd_y, probs[key])
        axes[1].plot(rec, prec, color=colors_map[key], linewidth=2, label=f'{name} (AP={ap:.3f})')
axes[1].axhline(y=base_rate, color='gray', linestyle=':', alpha=0.5, label=f'Baseline ({base_rate:.3f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG}/fig5_roc_pr_curves.png', bbox_inches='tight')
plt.close()
print("Fig 5 done")

# ============================================================
# Figure 6: t-SNE Embeddings
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Finetune embeddings
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
fe_2d = tsne.fit_transform(fe)

scatter = axes[0].scatter(fe_2d[:, 0], fe_2d[:, 1], c=fl, cmap='RdYlGn_r', 
                           alpha=0.5, s=10, edgecolors='none')
# Highlight positives
pos_mask = fl == 1
axes[0].scatter(fe_2d[pos_mask, 0], fe_2d[pos_mask, 1], c='red', s=30, 
                marker='*', edgecolors='black', linewidths=0.5, label='Altermagnet', zorder=5)
axes[0].set_title('Fine-tune Data Embeddings (t-SNE)')
axes[0].legend(); axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')

# Candidate embeddings
tsne2 = TSNE(n_components=2, random_state=42, perplexity=30)
ce_2d = tsne2.fit_transform(ce)

scatter2 = axes[1].scatter(ce_2d[:, 0], ce_2d[:, 1], c=cd_y, cmap='RdYlGn_r',
                            alpha=0.5, s=10, edgecolors='none')
pos_mask2 = cd_y == 1
axes[1].scatter(ce_2d[pos_mask2, 0], ce_2d[pos_mask2, 1], c='red', s=30,
                marker='*', edgecolors='black', linewidths=0.5, label='True Altermagnet', zorder=5)
axes[1].set_title('Candidate Data Embeddings (t-SNE)')
axes[1].legend(); axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig(f'{IMG}/fig6_tsne_embeddings.png', bbox_inches='tight')
plt.close()
print("Fig 6 done")

# ============================================================
# Figure 7: Prediction Score Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Best approach
for ax, (name, p) in zip(axes, [('Best (Random GIN Ensemble)', best_probs),
                                  ('Pre-trained GIN Ensemble', probs.get('candidate_probs_gnn', best_probs))]):
    pos_probs = p[cd_y == 1]
    neg_probs = p[cd_y == 0]
    ax.hist(neg_probs, bins=30, alpha=0.6, color='steelblue', label=f'Non-AM (n={len(neg_probs)})', density=True)
    ax.hist(pos_probs, bins=30, alpha=0.6, color='coral', label=f'Altermagnet (n={len(pos_probs)})', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(f'Score Distribution\n{name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMG}/fig7_score_distribution.png', bbox_inches='tight')
plt.close()
print("Fig 7 done")

# ============================================================
# Figure 8: Confusion Matrix
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, name in zip(axes, ['Pretrained_GIN_Ensemble', 'Random_GIN_Ensemble']):
    if name in results:
        cm = np.array(results[name]['cm'])
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(f'Confusion Matrix\n{name.replace("_"," ")}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Non-AM', 'AM']); ax.set_yticklabels(['Non-AM', 'AM'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                       color='white' if cm[i,j] > cm.max()/2 else 'black')
        plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(f'{IMG}/fig8_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("Fig 8 done")

# ============================================================
# Figure 9: Discovery Rate Curve
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

for name, p, color, ls in [('Pre-trained GIN', probs.get('candidate_probs_gnn', best_probs), 'blue', '-'),
                             ('Random Init GIN', probs.get('candidate_probs_random', best_probs), 'red', '--'),
                             ('Gradient Boosting', probs.get('candidate_probs_gb', best_probs), 'green', '-.'),
                             ('Random Forest', probs.get('candidate_probs_rf', best_probs), 'orange', ':'),
                             ('Hybrid', probs.get('candidate_probs_hybrid', best_probs), 'purple', '-')]:
    si = np.argsort(-p)
    ks = range(1, len(cd_y)+1)
    cumulative = np.cumsum(cd_y[si])
    total_pos = cd_y.sum()
    discovery_rate = cumulative / total_pos
    ax.plot(ks, discovery_rate, color=color, linestyle=ls, linewidth=2, label=name)

# Random baseline
random_rate = np.arange(1, len(cd_y)+1) / len(cd_y)
ax.plot(ks, random_rate, 'k--', alpha=0.5, label='Random')

ax.set_xlabel('Number of Candidates Screened (K)')
ax.set_ylabel('Discovery Rate (Fraction of True Altermagnets Found)')
ax.set_title('Cumulative Discovery Rate Curve')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 300)
plt.tight_layout()
plt.savefig(f'{IMG}/fig9_discovery_rate.png', bbox_inches='tight')
plt.close()
print("Fig 9 done")

# ============================================================
# Figure 10: Model Comparison Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

model_names = []
aucs = []
aps = []
f1s = []

for name in ['Pretrained_GIN_Ensemble', 'Random_GIN_Ensemble', 'GradientBoosting', 'RandomForest', 'Hybrid_GNN_ML']:
    if name in results:
        model_names.append(name.replace('_', '\n'))
        aucs.append(results[name]['auc_roc'])
        aps.append(results[name]['auc_pr'])
        f1s.append(results[name]['f1'])

x = np.arange(len(model_names))
w = 0.6

axes[0].bar(x, aucs, w, color=['blue','red','green','orange','purple'], alpha=0.7, edgecolor='black')
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].set_ylabel('AUC-ROC'); axes[0].set_title('AUC-ROC Comparison')
axes[0].set_xticks(x); axes[0].set_xticklabels(model_names, fontsize=9)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(x, aps, w, color=['blue','red','green','orange','purple'], alpha=0.7, edgecolor='black')
base_ap = cd_y.sum() / len(cd_y)
axes[1].axhline(y=base_ap, color='gray', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Average Precision'); axes[1].set_title('Average Precision Comparison')
axes[1].set_xticks(x); axes[1].set_xticklabels(model_names, fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(x, f1s, w, color=['blue','red','green','orange','purple'], alpha=0.7, edgecolor='black')
axes[2].set_ylabel('F1 Score'); axes[2].set_title('F1 Score Comparison')
axes[2].set_xticks(x); axes[2].set_xticklabels(model_names, fontsize=9)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{IMG}/fig10_model_comparison.png', bbox_inches='tight')
plt.close()
print("Fig 10 done")

# ============================================================
# Figure 11: Feature Importance
# ============================================================
try:
    fi = json.load(open(f'{OUT}/feature_importance.json'))
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [x[0] for x in sorted_fi]
    vals = [x[1] for x in sorted_fi]
    ax.barh(range(len(names)), vals, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importance\n(Top 15 Hand-Crafted Features)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{IMG}/fig11_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("Fig 11 done")
except Exception as e:
    print(f"Fig 11 skipped: {e}")

# ============================================================
# Figure 12: Top Candidates Visualization
# ============================================================
preds = json.load(open(f'{OUT}/all_candidate_predictions.json'))
top50 = preds[:50]

fig, ax = plt.subplots(figsize=(12, 6))
ranks = [p['rank'] for p in top50]
probs_vals = [p['prob'] for p in top50]
true_labels = [p['true'] for p in top50]
colors_bar = ['red' if t==1 else 'steelblue' for t in true_labels]

ax.bar(range(len(top50)), probs_vals, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Rank')
ax.set_ylabel('Predicted Probability')
ax.set_title('Top 50 Candidate Materials by Predicted Altermagnet Probability\n(Red = True Altermagnet, Blue = Non-Altermagnet)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{IMG}/fig12_top_candidates.png', bbox_inches='tight')
plt.close()
print("Fig 12 done")

print("\n=== ALL FIGURES GENERATED ===")
