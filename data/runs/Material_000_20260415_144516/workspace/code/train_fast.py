"""Fast Altermagnetic Discovery"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prepare

np.random.seed(42); torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
print("Loading data...")
ptr = torch.load('data/pretrain_data.pt', weights_only=False).data_list
ft = torch.load('data/finetune_data.pt', weights_only=False).data_list
cand = torch.load('data/candidate_data.pt', weights_only=False).data_list
print(f"Pretrain: {len(ptr)}, Finetune: {len(ft)}, Candidate: {len(cand)}")

# Simple feature-based model (fast baseline)
print("\nTraining feature-based classifier...")

# Extract simple graph features
def extract_features(data_list):
    features = []
    labels = []
    for d in data_list:
        # Node feature statistics
        x = d.x.numpy()
        feats = [
            x.mean(axis=0),  # mean node features
            x.std(axis=0),   # std node features
            x.max(axis=0),   # max node features
            [d.num_nodes],   # graph size
            [d.edge_index.shape[1]],  # number of edges
        ]
        features.append(np.concatenate(feats))
        if hasattr(d, 'y'):
            labels.append(d.y.item())
    return np.array(features), np.array(labels) if labels else None

# Extract features
train_feats, train_labels = extract_features(ft)
cand_feats, cand_labels = extract_features(cand)

# Split train/val
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

X_train, X_val, y_train, y_val = train_test_split(train_feats, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Train classifiers
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Evaluate on validation
val_probs_rf = rf.predict_proba(X_val)[:, 1]
val_preds_rf = (val_probs_rf > 0.5).astype(int)
val_f1_rf = f1_score(y_val, val_preds_rf)
print(f"RF Val F1: {val_f1_rf:.4f}, AUC: {roc_auc_score(y_val, val_probs_rf):.4f}")

# Train Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

val_probs_lr = lr.predict_proba(X_val)[:, 1]
val_preds_lr = (val_probs_lr > 0.5).astype(int)
val_f1_lr = f1_score(y_val, val_preds_lr)
print(f"LR Val F1: {val_f1_lr:.4f}, AUC: {roc_auc_score(y_val, val_probs_lr):.4f}")

# Use best model for discovery
best_model = rf if val_f1_rf > val_f1_lr else lr
print(f"\nUsing {'RF' if val_f1_rf > val_f1_lr else 'LR'} for discovery")

# Discovery
cand_probs = best_model.predict_proba(cand_feats)[:, 1]
cand_preds = (cand_probs > 0.5).astype(int)

# Top-k evaluation
k = 50
top_k_idx = np.argsort(cand_probs)[::-1][:k]
top_k_labels = [cand_labels[i] for i in top_k_idx]
tp_at_k = sum(top_k_labels)

print(f"\nDiscovery Results (top-{k}):")
print(f"  True Positives: {tp_at_k}/{k} ({tp_at_k/k*100:.1f}%)")
print(f"  Precision@{k}: {tp_at_k/k:.4f}")
print(f"  Recall@{k}: {tp_at_k/sum(cand_labels):.4f}")

# Overall metrics
print(f"\nOverall Performance:")
print(f"  Accuracy: {(cand_preds == cand_labels).mean():.4f}")
print(f"  Precision: {tp_at_k/k:.4f}")
print(f"  Recall: {tp_at_k/sum(cand_labels):.4f}")
print(f"  F1: {f1_score(cand_labels, cand_preds):.4f}")
print(f"  AUC: {roc_auc_score(cand_labels, cand_probs):.4f}")
print(f"  AUPRC: {average_precision_score(cand_labels, cand_probs):.4f}")

# Save results
os.makedirs('outputs', exist_ok=True)
results = {
    'model_type': 'RF' if val_f1_rf > val_f1_lr else 'LR',
    'candidate_probs': cand_probs.tolist(),
    'candidate_labels': [int(x) for x in cand_labels],
    'top_k_indices': top_k_idx.tolist(),
    'top_k_labels': [int(x) for x in top_k_labels],
    'true_positives_at_k': int(tp_at_k),
    'precision_at_k': tp_at_k/k,
    'recall_at_k': tp_at_k/sum(cand_labels),
    'overall_accuracy': float((cand_preds == cand_labels).mean()),
    'overall_f1': float(f1_score(cand_labels, cand_preds)),
    'overall_auc': float(roc_auc_score(cand_labels, cand_probs)),
    'overall_auprc': float(average_precision_score(cand_labels, cand_probs)),
    'val_f1_rf': val_f1_rf,
    'val_auc_rf': float(roc_auc_score(y_val, val_probs_rf)),
    'val_f1_lr': val_f1_lr,
    'val_auc_lr': float(roc_auc_score(y_val, val_probs_lr))
}

with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to outputs/results.json")
