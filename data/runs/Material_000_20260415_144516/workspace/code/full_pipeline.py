"""
Complete Altermagnetic Materials Discovery Pipeline
Generates results, figures, and analysis for the report.
"""
import os, sys, json, numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prepare

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

np.random.seed(42)
print("="*60)
print("Altermagnetic Materials Discovery Pipeline")
print("="*60)

# Load data
print("\nLoading data...")
pretrain_raw = torch.load('data/pretrain_data.pt', weights_only=False)
finetune_raw = torch.load('data/finetune_data.pt', weights_only=False)
candidate_raw = torch.load('data/candidate_data.pt', weights_only=False)

pretrain_data = pretrain_raw.data_list
finetune_data = finetune_raw.data_list
candidate_data = candidate_raw.data_list

print(f"Pretrain: {len(pretrain_data)} samples")
print(f"Finetune: {len(finetune_data)} samples")
print(f"Candidate: {len(candidate_data)} samples")

# Extract labels
finetune_labels = [d.y.item() for d in finetune_data]
candidate_labels = [d.y.item() for d in candidate_data]
print(f"Finetune positives: {sum(finetune_labels)}/{len(finetune_labels)} ({sum(finetune_labels)/len(finetune_labels)*100:.1f}%)")
print(f"Candidate positives: {sum(candidate_labels)}/{len(candidate_labels)} ({sum(candidate_labels)/len(candidate_labels)*100:.1f}%)")

# Feature extraction
def extract_features(data_list):
    """Extract comprehensive graph features."""
    features = []
    for d in data_list:
        x = d.x.numpy()
        # Node feature statistics
        node_mean = x.mean(axis=0)
        node_std = x.std(axis=0)
        node_max = x.max(axis=0)
        node_min = x.min(axis=0)
        
        # Graph structure features
        num_nodes = d.num_nodes
        num_edges = d.edge_index.shape[1]
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        
        # Edge features
        if d.edge_attr is not None:
            ea = d.edge_attr.numpy()
            edge_mean = ea.mean(axis=0)
            edge_std = ea.std(axis=0)
        else:
            edge_mean = np.zeros(2)
            edge_std = np.zeros(2)
        
        # Combine features
        feat = np.concatenate([
            node_mean, node_std, node_max, node_min,
            [num_nodes, num_edges, avg_degree],
            edge_mean, edge_std
        ])
        features.append(feat)
    return np.array(features)

print("\nExtracting features...")
finetune_feats = extract_features(finetune_data)
candidate_feats = extract_features(candidate_data)
print(f"Feature dimension: {finetune_feats.shape[1]}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    finetune_feats, finetune_labels, 
    test_size=0.2, random_state=42, stratify=finetune_labels
)

# Train models
print("\nTraining models...")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_val_probs = rf.predict_proba(X_val)[:, 1]
rf_val_preds = (rf_val_probs > 0.5).astype(int)
rf_val_f1 = f1_score(y_val, rf_val_preds)
rf_val_auc = roc_auc_score(y_val, rf_val_probs)
print(f"RF - Val F1: {rf_val_f1:.4f}, AUC: {rf_val_auc:.4f}")

# Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_val_probs = lr.predict_proba(X_val)[:, 1]
lr_val_preds = (lr_val_probs > 0.5).astype(int)
lr_val_f1 = f1_score(y_val, lr_val_preds)
lr_val_auc = roc_auc_score(y_val, lr_val_probs)
print(f"LR - Val F1: {lr_val_f1:.4f}, AUC: {lr_val_auc:.4f}")

# Select best model
if rf_val_f1 > lr_val_f1:
    best_model = rf
    model_name = "Random Forest"
    print(f"\nSelected: {model_name}")
else:
    best_model = lr
    model_name = "Logistic Regression"
    print(f"\nSelected: {model_name}")

# Discovery on candidates
candidate_probs = best_model.predict_proba(candidate_feats)[:, 1]
candidate_preds = (candidate_probs > 0.5).astype(int)

# Top-k analysis
top_k = 50
top_indices = np.argsort(candidate_probs)[::-1][:top_k]
top_labels = [candidate_labels[i] for i in top_indices]
tp_at_k = sum(top_labels)

print(f"\n{'='*60}")
print("DISCOVERY RESULTS")
print(f"{'='*60}")
print(f"\nTop-{top_k} Discoveries:")
for i, idx in enumerate(top_indices[:10]):
    print(f"  {i+1:2d}. Candidate {idx:3d}: prob={candidate_probs[idx]:.4f}, true={int(candidate_labels[idx])}")

print(f"\nTop-{top_k} Statistics:")
print(f"  True Positives: {tp_at_k}/{top_k} ({tp_at_k/top_k*100:.1f}%)")
print(f"  Precision@{top_k}: {tp_at_k/top_k:.4f}")
print(f"  Recall@{top_k}: {tp_at_k/sum(candidate_labels):.4f}")

print(f"\nOverall Performance:")
print(f"  Accuracy: {(candidate_preds == np.array(candidate_labels)).mean():.4f}")
print(f"  Precision: {precision_score(candidate_labels, candidate_preds, zero_division=0):.4f}")
print(f"  Recall: {recall_score(candidate_labels, candidate_preds, zero_division=0):.4f}")
print(f"  F1: {f1_score(candidate_labels, candidate_preds, zero_division=0):.4f}")
print(f"  AUC: {roc_auc_score(candidate_labels, candidate_probs):.4f}")
print(f"  AUPRC: {average_precision_score(candidate_labels, candidate_probs):.4f}")

# Save results
os.makedirs('outputs', exist_ok=True)
results = {
    'model_type': model_name,
    'candidate_probs': [float(x) for x in candidate_probs],
    'candidate_labels': [int(x) for x in candidate_labels],
    'top_k_indices': [int(x) for x in top_indices],
    'top_k_labels': [int(x) for x in top_labels],
    'true_positives_at_k': int(tp_at_k),
    'precision_at_k': float(tp_at_k/top_k),
    'recall_at_k': float(tp_at_k/sum(candidate_labels)),
    'overall_accuracy': float((candidate_preds == np.array(candidate_labels)).mean()),
    'overall_precision': float(precision_score(candidate_labels, candidate_preds, zero_division=0)),
    'overall_recall': float(recall_score(candidate_labels, candidate_preds, zero_division=0)),
    'overall_f1': float(f1_score(candidate_labels, candidate_preds, zero_division=0)),
    'overall_auc': float(roc_auc_score(candidate_labels, candidate_probs)),
    'overall_auprc': float(average_precision_score(candidate_labels, candidate_probs)),
}

with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to outputs/results.json")

# Generate visualizations
print("\nGenerating visualizations...")
os.makedirs('report/images', exist_ok=True)

# Figure 1: Data distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Dataset distribution
datasets = ['Pretrain', 'Finetune', 'Candidate']
positives = [sum([d.y.item() for d in pretrain_data]), sum(finetune_labels), sum(candidate_labels)]
negatives = [len(pretrain_data) - positives[0], len(finetune_data) - positives[1], len(candidate_data) - positives[2]]

x = np.arange(len(datasets))
width = 0.35
axes[0].bar(x - width/2, negatives, width, label='Non-altermagnet', color='steelblue')
axes[0].bar(x + width/2, positives, width, label='Altermagnet', color='coral')
axes[0].set_ylabel('Number of Samples')
axes[0].set_title('Dataset Distribution')
axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets)
axes[0].legend()
axes[0].set_yscale('log')

# Node distribution
node_counts = [d.num_nodes for d in finetune_data]
axes[1].hist([d.num_nodes for d in finetune_data if d.y.item() == 0], bins=30, alpha=0.7, label='Non-altermagnet', color='steelblue')
axes[1].hist([d.num_nodes for d in finetune_data if d.y.item() == 1], bins=30, alpha=0.7, label='Altermagnet', color='coral')
axes[1].set_xlabel('Number of Nodes')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Graph Size Distribution (Finetune)')
axes[1].legend()

# Edge distribution
axes[2].hist([d.edge_index.shape[1] for d in finetune_data if d.y.item() == 0], bins=30, alpha=0.7, label='Non-altermagnet', color='steelblue')
axes[2].hist([d.edge_index.shape[1] for d in finetune_data if d.y.item() == 1], bins=30, alpha=0.7, label='Altermagnet', color='coral')
axes[2].set_xlabel('Number of Edges')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Edge Count Distribution (Finetune)')
axes[2].legend()

plt.tight_layout()
plt.savefig('report/images/fig1_data_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_data_distribution.png")

# Figure 2: ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve
fpr, tpr, _ = roc_curve(candidate_labels, candidate_probs)
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc_score(candidate_labels, candidate_probs):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve on Candidates')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Precision-Recall curve
precision_vals, recall_vals, _ = precision_recall_curve(candidate_labels, candidate_probs)
axes[1].plot(recall_vals, precision_vals, 'g-', linewidth=2, 
             label=f'PR curve (AUPRC = {average_precision_score(candidate_labels, candidate_probs):.3f})')
axes[1].axhline(sum(candidate_labels)/len(candidate_labels), color='r', linestyle='--', 
                label=f'Baseline ({sum(candidate_labels)/len(candidate_labels):.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve on Candidates')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig2_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_roc_pr_curves.png")

# Figure 3: Confusion matrix
cm = confusion_matrix(candidate_labels, candidate_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Altermagnet', 'Altermagnet'],
            yticklabels=['Non-Altermagnet', 'Altermagnet'],
            ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix on Candidates')
plt.tight_layout()
plt.savefig('report/images/fig3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_confusion_matrix.png")

# Figure 4: Discovery ranking
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Probability distribution
axes[0].hist(candidate_probs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(0.5, color='r', linestyle='--', label='Decision threshold')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Predicted Probabilities')
axes[0].legend()

# Top-k discovery
top_k_range = range(10, 201, 10)
precisions = []
recalls = []
for k in top_k_range:
    top_k_idx = np.argsort(candidate_probs)[::-1][:k]
    top_k_labs = [candidate_labels[i] for i in top_k_idx]
    tp = sum(top_k_labs)
    precisions.append(tp / k)
    recalls.append(tp / sum(candidate_labels))

axes[1].plot(top_k_range, precisions, 'b-', linewidth=2, marker='o', markersize=4, label='Precision')
axes[1].plot(top_k_range, recalls, 'r-', linewidth=2, marker='s', markersize=4, label='Recall')
axes[1].axvline(50, color='g', linestyle='--', alpha=0.5, label='Top-50')
axes[1].set_xlabel('Top-k Candidates')
axes[1].set_ylabel('Score')
axes[1].set_title('Precision and Recall vs. Top-k')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig4_discovery_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_discovery_analysis.png")

# Figure 5: Feature importance (if RF)
if model_name == "Random Forest":
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_names = [f'NodeMean_{i}' for i in range(28)] + \
                   [f'NodeStd_{i}' for i in range(28)] + \
                   [f'NodeMax_{i}' for i in range(28)] + \
                   [f'NodeMin_{i}' for i in range(28)] + \
                   ['NumNodes', 'NumEdges', 'AvgDegree', 'EdgeMean_0', 'EdgeMean_1', 'EdgeStd_0', 'EdgeStd_1']
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    ax.barh(range(20), importances[indices], color='steelblue')
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Feature Importances (Random Forest)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('report/images/fig5_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_feature_importance.png")

print(f"\n{'='*60}")
print("Pipeline completed successfully!")
print(f"{'='*60}")
