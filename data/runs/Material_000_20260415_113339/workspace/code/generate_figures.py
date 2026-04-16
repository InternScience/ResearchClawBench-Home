import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import torch
import os

os.makedirs('report/images', exist_ok=True)

with open('outputs/evaluation_results.json', 'r') as f:
    results = json.load(f)

cand_labels = np.load('outputs/cand_labels.npy')
probs_gnn_pre = np.load('outputs/probs_gnn_pretrained.npy')
probs_gnn_scr = np.load('outputs/probs_gnn_scratch.npy')
probs_rf = np.load('outputs/probs_rf.npy')
probs_gb = np.load('outputs/probs_gb.npy')
probs_svm = np.load('outputs/probs_svm.npy')
probs_ens2 = np.load('outputs/probs_ensemble_weighted.npy')

with open('outputs/pretrain_loss.json', 'r') as f:
    pt_loss = json.load(f)
with open('outputs/ft_metrics_pretrained.json', 'r') as f:
    ft_pre = json.load(f)
with open('outputs/ft_metrics_scratch.json', 'r') as f:
    ft_scr = json.load(f)

ft_features = np.load('outputs/ft_features.npy')
ft_labels = np.load('outputs/ft_labels.npy')

model_keys = ['gnn_pretrained', 'gnn_scratch', 'random_forest', 'gradient_boosting', 'svm', 'ensemble_weighted']
model_names_short = ['GNN (Pre)', 'GNN (Scratch)', 'RF', 'GB', 'SVM', 'Ensemble']
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']

all_probs = {
    'gnn_pretrained': probs_gnn_pre,
    'gnn_scratch': probs_gnn_scr,
    'random_forest': probs_rf,
    'gradient_boosting': probs_gb,
    'svm': probs_svm,
    'ensemble_weighted': probs_ens2,
}

# Figure 1: Data Overview
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
datasets = ['Finetune\n(Train)', 'Candidate\n(Test)']
pos_counts = [int(ft_labels.sum()), int(cand_labels.sum())]
neg_counts = [int(len(ft_labels) - ft_labels.sum()), int(len(cand_labels) - cand_labels.sum())]
x = np.arange(len(datasets))
width = 0.35
bars1 = axes[0].bar(x - width/2, pos_counts, width, label='Altermagnetic', color='#e74c3c', alpha=0.8)
bars2 = axes[0].bar(x + width/2, neg_counts, width, label='Non-altermagnetic', color='#3498db', alpha=0.8)
axes[0].set_ylabel('Count'); axes[0].set_title('(a) Class Distribution')
axes[0].set_xticks(x); axes[0].set_xticklabels(datasets); axes[0].legend(fontsize=8)
for bar in bars1+bars2:
    axes[0].text(bar.get_x()+bar.get_width()/2., bar.get_height()+20, f'{int(bar.get_height())}', ha='center', fontsize=9)

finetune_ds = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
pos_nodes = [finetune_ds.data_list[i].x.shape[0] for i in range(len(finetune_ds.data_list)) if ft_labels[i]==1]
neg_nodes = [finetune_ds.data_list[i].x.shape[0] for i in range(len(finetune_ds.data_list)) if ft_labels[i]==0]
axes[1].hist(neg_nodes, bins=20, alpha=0.6, label='Negative', color='#3498db', density=True)
axes[1].hist(pos_nodes, bins=15, alpha=0.6, label='Positive', color='#e74c3c', density=True)
axes[1].set_xlabel('Number of Nodes'); axes[1].set_ylabel('Density'); axes[1].set_title('(b) Graph Size Distribution')
axes[1].legend(fontsize=8)

elem_names = ['Fe','Co','Ni','Mn','Cr','V','Ti','Nd','Pr','Sm','Gd','Ho','Er','Yb','O','F','Cl','Br','I','S','Se','Te','B','C','N','P','Si','H']
pos_mask = ft_labels == 1; neg_mask = ft_labels == 0
pos_elem = ft_features[pos_mask, 23:].mean(axis=0); neg_elem = ft_features[neg_mask, 23:].mean(axis=0)
x_pos = np.arange(28)
axes[2].bar(x_pos-0.2, pos_elem, 0.4, label='Positive', color='#e74c3c', alpha=0.8)
axes[2].bar(x_pos+0.2, neg_elem, 0.4, label='Negative', color='#3498db', alpha=0.8)
axes[2].set_xticks(x_pos); axes[2].set_xticklabels(elem_names, rotation=45, fontsize=7)
axes[2].set_ylabel('Mean Fraction'); axes[2].set_title('(c) Element Composition'); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 1 saved")

# Figure 2: Training Curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(pt_loss, color='#2c3e50', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('(a) Pre-training Loss'); axes[0].grid(True, alpha=0.3)

axes[1].plot([m['val_f1'] for m in ft_pre], label='Pre-trained', color='#e74c3c', linewidth=2)
axes[1].plot([m['val_f1'] for m in ft_scr], label='From Scratch', color='#3498db', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation F1'); axes[1].set_title('(b) Fine-tuning Validation F1')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot([m['val_ap'] for m in ft_pre], label='Pre-trained', color='#e74c3c', linewidth=2)
axes[2].plot([m['val_ap'] for m in ft_scr], label='From Scratch', color='#3498db', linewidth=2)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Validation AP'); axes[2].set_title('(c) Fine-tuning Validation AP')
axes[2].legend(); axes[2].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig2_training_curves.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 2 saved")

# Figure 3: ROC and PR Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for key, name, color in zip(model_keys, model_names_short, colors):
    fpr, tpr, _ = roc_curve(cand_labels, all_probs[key])
    auc_val = roc_auc_score(cand_labels, all_probs[key])
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', color=color, linewidth=1.5)
axes[0].plot([0,1],[0,1],'k--',alpha=0.3); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('(a) ROC Curves'); axes[0].legend(fontsize=7, loc='lower right'); axes[0].grid(True, alpha=0.3)

for key, name, color in zip(model_keys, model_names_short, colors):
    prec_arr, rec_arr, _ = precision_recall_curve(cand_labels, all_probs[key])
    ap_val = average_precision_score(cand_labels, all_probs[key])
    axes[1].plot(rec_arr, prec_arr, label=f'{name} (AP={ap_val:.3f})', color=color, linewidth=1.5)
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('(b) Precision-Recall Curves')
axes[1].legend(fontsize=7, loc='upper right'); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig3_roc_pr_curves.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 3 saved")

# Figure 4: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
aucs = [results[k]['auc_roc'] for k in model_keys]
bars = axes[0].bar(model_names_short, aucs, color=colors, alpha=0.8)
axes[0].set_ylabel('AUC-ROC'); axes[0].set_title('(a) AUC-ROC'); axes[0].set_ylim(0.4, 0.55)
for bar, val in zip(bars, aucs): axes[0].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.003, f'{val:.3f}', ha='center', fontsize=8)
axes[0].axhline(y=0.5, color='k', linestyle='--', alpha=0.3); axes[0].grid(True, alpha=0.3, axis='y')

aps = [results[k]['average_precision'] for k in model_keys]
bars = axes[1].bar(model_names_short, aps, color=colors, alpha=0.8)
axes[1].set_ylabel('Average Precision'); axes[1].set_title('(b) Average Precision')
pos_ratio = cand_labels.sum()/len(cand_labels)
axes[1].axhline(y=pos_ratio, color='k', linestyle='--', alpha=0.3, label=f'Random ({pos_ratio:.3f})')
for bar, val in zip(bars, aps): axes[1].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.001, f'{val:.3f}', ha='center', fontsize=8)
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, axis='y')

f1s = [results[k]['f1_0.5'] for k in model_keys]
bars = axes[2].bar(model_names_short, f1s, color=colors, alpha=0.8)
axes[2].set_ylabel('F1 Score (threshold=0.5)'); axes[2].set_title('(c) F1 Score')
for bar, val in zip(bars, f1s): axes[2].text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.002, f'{val:.3f}', ha='center', fontsize=8)
axes[2].grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('report/images/fig4_model_comparison.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 4 saved")

# Figure 5: Top-K
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
k_values = [10, 20, 30, 43, 50, 100]
for key, name, color in zip(model_keys, model_names_short, colors):
    axes[0].plot(k_values, [results[key]['topk'][f'top_{k}']['P@K'] for k in k_values], marker='o', label=name, color=color, linewidth=1.5)
axes[0].set_xlabel('K'); axes[0].set_ylabel('Precision@K'); axes[0].set_title('(a) Precision@K')
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

for key, name, color in zip(model_keys, model_names_short, colors):
    axes[1].plot(k_values, [results[key]['topk'][f'top_{k}']['R@K'] for k in k_values], marker='o', label=name, color=color, linewidth=1.5)
axes[1].set_xlabel('K'); axes[1].set_ylabel('Recall@K'); axes[1].set_title('(b) Recall@K')
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig5_topk_performance.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 5 saved")

# Figure 6: Probability Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
pos_c = cand_labels == 1; neg_c = cand_labels == 0
axes[0].hist(probs_gnn_pre[neg_c], bins=30, alpha=0.6, label='True Negative', color='#3498db', density=True)
axes[0].hist(probs_gnn_pre[pos_c], bins=15, alpha=0.6, label='True Positive', color='#e74c3c', density=True)
axes[0].set_xlabel('Predicted Probability'); axes[0].set_ylabel('Density'); axes[0].set_title('(a) GNN (Pre-trained)')
axes[0].legend(); axes[0].axvline(x=0.5, color='k', linestyle='--', alpha=0.3)

axes[1].hist(probs_ens2[neg_c], bins=30, alpha=0.6, label='True Negative', color='#3498db', density=True)
axes[1].hist(probs_ens2[pos_c], bins=15, alpha=0.6, label='True Positive', color='#e74c3c', density=True)
axes[1].set_xlabel('Predicted Probability'); axes[1].set_ylabel('Density'); axes[1].set_title('(b) Ensemble')
axes[1].legend(); axes[1].axvline(x=0.5, color='k', linestyle='--', alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig6_prob_distribution.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 6 saved")

# Figure 7: Feature Importance
fig, ax = plt.subplots(figsize=(8, 6))
feat_imp = results['feature_importance']
names = list(feat_imp.keys())[:12]
values = [feat_imp[n] for n in names]
sorted_pairs = sorted(zip(names, values), key=lambda x: x[1])
ax.barh([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], color='#2ecc71', alpha=0.8)
ax.set_xlabel('Feature Importance'); ax.set_title('Top Features (Random Forest)'); ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout(); plt.savefig('report/images/fig7_feature_importance.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 7 saved")

# Figure 8: Discovery Thresholds
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for key, name, color in zip(model_keys, model_names_short, colors):
    probs = all_probs[key]
    pl = []; rl = []
    for t in thresholds:
        pred = (probs > t).astype(int)
        tp = int(((pred==1)&(cand_labels==1)).sum()); fp = int(((pred==1)&(cand_labels==0)).sum()); fn = int(((pred==0)&(cand_labels==1)).sum())
        pl.append(tp/(tp+fp) if (tp+fp)>0 else 0); rl.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    axes[0].plot(thresholds, pl, marker='o', label=name, color=color, linewidth=1.5)
    axes[1].plot(thresholds, rl, marker='o', label=name, color=color, linewidth=1.5)
axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('Precision'); axes[0].set_title('(a) Precision vs Threshold')
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Recall'); axes[1].set_title('(b) Recall vs Threshold')
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig8_discovery_thresholds.png', dpi=150, bbox_inches='tight'); plt.close()
print("Figure 8 saved")

print("\nAll figures saved!")
