"""
Altermagnetic Material Discovery (v3)
- Supervised pre-training on pretrain labels
- Fine-tuning with class imbalance handling
- Multiple baselines for comparison
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool, AttentionalAggregation
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets():
    pretrain = torch.load('data/pretrain_data.pt', map_location='cpu', weights_only=False)
    finetune = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
    candidate = torch.load('data/candidate_data.pt', map_location='cpu', weights_only=False)
    return pretrain.data_list, finetune.data_list, candidate.data_list

# ---------------------------------------------------------------------------
# Feature extraction for baselines
# ---------------------------------------------------------------------------

def extract_graph_features(data_list):
    """Extract simple graph-level features: mean, max, sum of node and edge features."""
    features = []
    labels = []
    for d in data_list:
        x = d.x.numpy()
        e = d.edge_attr.numpy() if d.edge_attr is not None else np.zeros((1, 2))
        
        node_mean = x.mean(axis=0)
        node_max = x.max(axis=0)
        node_min = x.min(axis=0)
        node_sum = x.sum(axis=0)
        node_std = x.std(axis=0)
        
        edge_mean = e.mean(axis=0)
        edge_max = e.max(axis=0)
        edge_min = e.min(axis=0)
        edge_sum = e.sum(axis=0)
        
        # Graph stats
        num_nodes = x.shape[0]
        num_edges = e.shape[0]
        density = num_edges / max(1, num_nodes * (num_nodes - 1))
        
        feat = np.concatenate([
            node_mean, node_max, node_min, node_sum, node_std,
            edge_mean, edge_max, edge_min, edge_sum,
            [num_nodes, num_edges, density]
        ])
        features.append(feat)
        if hasattr(d, 'y') and d.y is not None:
            labels.append(d.y.item())
        else:
            labels.append(0)
    
    return np.array(features), np.array(labels)

# ---------------------------------------------------------------------------
# GNN Encoder
# ---------------------------------------------------------------------------

class CrystalEncoder(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=128, out_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ))
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, data, return_graph_emb=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        edge_emb = self.edge_emb(edge_attr) if edge_attr is not None else None
        x = self.node_emb(x)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = bn(x)
            x = F.relu(x)
        
        graph_emb = self.pool(x, batch)
        
        if return_graph_emb:
            return graph_emb
        
        z = self.projector(graph_emb)
        z = F.normalize(z, p=2, dim=1)
        return z

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class AltermagnetClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        emb = self.encoder(data, return_graph_emb=True)
        out = self.mlp(emb)
        return out.squeeze(-1)

# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = batch.y.float().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 0.5,
        'pr_auc': float(average_precision_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 0.5,
    }
    return metrics, all_probs, all_labels, all_preds

def train_epoch_classifier(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        labels = batch.y.float()
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    
    print("Loading datasets...")
    pretrain_list, finetune_list, candidate_list = load_datasets()
    print(f"Pretrain: {len(pretrain_list)}, Finetune: {len(finetune_list)}, Candidate: {len(candidate_list)}")
    
    for d in finetune_list:
        d.y = d.y.float()
    for d in candidate_list:
        d.y = d.y.float()
    for d in pretrain_list:
        d.y = d.y.float()
    
    # Split finetune
    n = len(finetune_list)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:int(0.7*n)]
    val_idx = indices[int(0.7*n):int(0.85*n)]
    test_idx = indices[int(0.85*n):]
    
    train_data = [finetune_list[i] for i in train_idx]
    val_data = [finetune_list[i] for i in val_idx]
    test_data = [finetune_list[i] for i in test_idx]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    train_pos = sum([d.y.item() for d in train_data])
    val_pos = sum([d.y.item() for d in val_data])
    test_pos = sum([d.y.item() for d in test_data])
    print(f"Positives - Train: {train_pos}, Val: {val_pos}, Test: {test_pos}")
    
    # Oversample positives
    pos_samples = [d for d in train_data if d.y.item() == 1.0]
    neg_samples = [d for d in train_data if d.y.item() == 0.0]
    oversample_ratio = min(len(neg_samples) // max(1, len(pos_samples)), 10)
    train_data_balanced = neg_samples + pos_samples * oversample_ratio
    random.shuffle(train_data_balanced)
    print(f"Balanced train: {len(train_data_balanced)} (pos: {sum([d.y.item() for d in train_data_balanced])})")
    
    # Combine pretrain + balanced finetune for supervised pretraining
    combined_train = pretrain_list + train_data_balanced
    random.shuffle(combined_train)
    print(f"Combined train (pretrain + finetune): {len(combined_train)}")
    
    pretrain_loader = PyGDataLoader(combined_train, batch_size=128, shuffle=True, num_workers=0)
    train_loader = PyGDataLoader(train_data_balanced, batch_size=64, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
    candidate_loader = PyGDataLoader(candidate_list, batch_size=64, shuffle=False, num_workers=0)
    
    # ========================
    # Baseline: Random Forest on hand-crafted features
    # ========================
    print("\n=== Baseline: Random Forest ===")
    X_train, y_train = extract_graph_features(train_data)
    X_val, y_val = extract_graph_features(val_data)
    X_test, y_test = extract_graph_features(test_data)
    X_cand, y_cand = extract_graph_features(candidate_list)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    rf_test_probs = rf.predict_proba(X_test)[:, 1]
    rf_test_preds = (rf_test_probs >= 0.5).astype(int)
    rf_test_auc = roc_auc_score(y_test, rf_test_probs)
    rf_test_f1 = f1_score(y_test, rf_test_preds, zero_division=0)
    print(f"RF Test AUC: {rf_test_auc:.4f}, F1: {rf_test_f1:.4f}")
    
    rf_cand_probs = rf.predict_proba(X_cand)[:, 1]
    rf_top50 = np.argsort(rf_cand_probs)[::-1][:50]
    rf_discovered = int(y_cand[rf_top50].sum())
    print(f"RF Top-50 discovered: {rf_discovered}, Precision@50: {rf_discovered/50:.4f}")
    
    # ========================
    # GNN Training
    # ========================
    encoder = CrystalEncoder(node_dim=28, edge_dim=2, hidden_dim=128, out_dim=64, num_layers=3).to(device)
    classifier = AltermagnetClassifier(encoder, hidden_dim=128).to(device)
    
    # Phase 1: Supervised pre-training on combined data
    print("\n=== Phase 1: Supervised Pre-training ===")
    pos_weight = len(combined_train) / max(1, sum([d.y.item() for d in combined_train])) - 1
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    for epoch in range(1, 31):
        loss = train_epoch_classifier(classifier, pretrain_loader, optimizer, device, criterion)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Pretrain Loss: {loss:.4f}")
    
    # Phase 2: Fine-tuning
    print("\n=== Phase 2: Fine-tuning ===")
    pos_weight = len(train_data_balanced) / max(1, sum([d.y.item() for d in train_data_balanced])) - 1
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    optimizer = torch.optim.AdamW([
        {'params': classifier.encoder.parameters(), 'lr': 5e-5},
        {'params': classifier.mlp.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_auc = 0
    best_state = None
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(1, 101):
        train_loss = train_epoch_classifier(classifier, train_loader, optimizer, device, criterion)
        
        if epoch % 5 == 0 or epoch == 1:
            train_metrics, _, _, _ = evaluate_classifier(classifier, train_loader, device)
            val_metrics, _, _, _ = evaluate_classifier(classifier, val_loader, device)
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            
            if val_metrics['roc_auc'] > best_val_auc:
                best_val_auc = val_metrics['roc_auc']
                best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
                print(f"  --> New best val AUC: {best_val_auc:.4f}")
            
            scheduler.step(val_metrics['roc_auc'])
    
    if best_state is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Test evaluation
    print("\n=== Test Set Evaluation ===")
    test_metrics, test_probs, test_labels, test_preds = evaluate_classifier(classifier, test_loader, device)
    print(f"Test AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Candidate screening
    print("\n=== Candidate Screening ===")
    classifier.eval()
    all_candidate_probs = []
    all_candidate_labels = []
    with torch.no_grad():
        for batch in candidate_loader:
            batch = batch.to(device)
            logits = classifier(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch.y.float().cpu().numpy()
            all_candidate_probs.extend(probs.tolist())
            all_candidate_labels.extend(labels.tolist())
    
    all_candidate_probs = np.array(all_candidate_probs)
    all_candidate_labels = np.array(all_candidate_labels)
    
    top_k = 50
    top_indices = np.argsort(all_candidate_probs)[::-1][:top_k]
    top_probs = all_candidate_probs[top_indices]
    top_labels = all_candidate_labels[top_indices]
    
    discovered = int(top_labels.sum())
    precision_at_50 = discovered / top_k
    recall_at_50 = discovered / max(1, all_candidate_labels.sum())
    print(f"GNN Top-{top_k}: {discovered} true positives discovered")
    print(f"GNN Precision@50: {precision_at_50:.4f}")
    print(f"GNN Recall@50: {recall_at_50:.4f}")
    
    # Save results
    with open('outputs/metrics.json', 'w') as f:
        json.dump({
            'gnn_test': test_metrics,
            'rf_test_auc': float(rf_test_auc),
            'rf_test_f1': float(rf_test_f1),
            'val_best_auc': float(best_val_auc),
        }, f, indent=2)
    
    candidate_results = {
        'gnn_precision_at_50': float(precision_at_50),
        'gnn_recall_at_50': float(recall_at_50),
        'rf_precision_at_50': float(rf_discovered / 50),
        'rf_recall_at_50': float(rf_discovered / max(1, all_candidate_labels.sum())),
        'num_true_positives_in_candidate': int(all_candidate_labels.sum())
    }
    with open('outputs/candidate_results.json', 'w') as f:
        json.dump(candidate_results, f, indent=2)
    
    # ========================
    # Figures
    # ========================
    print("\n=== Generating Figures ===")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_test_probs)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f'GNN (AUC = {test_metrics["roc_auc"]:.3f})')
    plt.plot(rf_fpr, rf_tpr, lw=2, linestyle='--', label=f'RF (AUC = {rf_test_auc:.3f})')
    plt.plot([0,1], [0,1], 'k:', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Altermagnet Classification')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('report/images/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # PR Curve
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_probs)
    rf_precision_curve, rf_recall_curve, _ = precision_recall_curve(y_test, rf_test_probs)
    plt.figure(figsize=(7,6))
    plt.plot(recall_curve, precision_curve, lw=2, label=f'GNN (AP = {test_metrics["pr_auc"]:.3f})')
    plt.plot(rf_recall_curve, rf_precision_curve, lw=2, linestyle='--', label=f'RF (AP = {average_precision_score(y_test, rf_test_probs):.3f})')
    plt.axhline(y=test_labels.mean(), color='r', linestyle=':', label=f'Baseline ({test_labels.mean():.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('report/images/pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-AM', 'AM'], yticklabels=['Non-AM', 'AM'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - GNN (Test Set)')
    plt.savefig('report/images/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Candidate Score Distribution
    plt.figure(figsize=(8,5))
    bins = np.linspace(0, 1, 31)
    plt.hist(all_candidate_probs[all_candidate_labels==0], bins=bins, alpha=0.6, label='Non-AM', color='steelblue')
    plt.hist(all_candidate_probs[all_candidate_labels==1], bins=bins, alpha=0.6, label='AM (True)', color='crimson')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Candidate Score Distribution (GNN)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('report/images/candidate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Top-50 Candidates
    plt.figure(figsize=(12,6))
    colors = ['crimson' if l == 1 else 'steelblue' for l in top_labels]
    plt.bar(range(top_k), top_probs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.xlabel('Candidate Rank')
    plt.ylabel('Predicted Probability')
    plt.title(f'Top-{top_k} Predicted Altermagnets (Red = True Positive)')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('report/images/top_candidates.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Precision@K and Recall@K
    k_values = list(range(10, min(201, len(candidate_list)+1), 10))
    gnn_precisions = []
    gnn_recalls = []
    rf_precisions = []
    rf_recalls = []
    sorted_idx = np.argsort(all_candidate_probs)[::-1]
    rf_sorted_idx = np.argsort(rf_cand_probs)[::-1]
    for k in k_values:
        gnn_precisions.append(float(all_candidate_labels[sorted_idx[:k]].mean()))
        gnn_recalls.append(float(all_candidate_labels[sorted_idx[:k]].sum() / max(1, all_candidate_labels.sum())))
        rf_precisions.append(float(y_cand[rf_sorted_idx[:k]].mean()))
        rf_recalls.append(float(y_cand[rf_sorted_idx[:k]].sum() / max(1, y_cand.sum())))
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(k_values, gnn_precisions, 'b-o', label='GNN Precision@K')
    ax1.plot(k_values, rf_precisions, 'b--s', label='RF Precision@K')
    ax1.set_xlabel('K (Number of Top Candidates)')
    ax1.set_ylabel('Precision@K', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(k_values, gnn_recalls, 'r-o', label='GNN Recall@K')
    ax2.plot(k_values, rf_recalls, 'r--s', label='RF Recall@K')
    ax2.set_ylabel('Recall@K', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Precision and Recall at K (Candidate Set)')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.9))
    plt.grid(True, alpha=0.3)
    plt.savefig('report/images/precision_recall_at_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Training curves
    epochs_tracked = [i*5 for i in range(1, len(val_metrics_history)+1)]
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_tracked, [m['roc_auc'] for m in train_metrics_history], 'o-', label='Train AUC')
    plt.plot(epochs_tracked, [m['roc_auc'] for m in val_metrics_history], 's-', label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC During Fine-tuning')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(epochs_tracked, [m['f1'] for m in train_metrics_history], 'o-', label='Train F1')
    plt.plot(epochs_tracked, [m['f1'] for m in val_metrics_history], 's-', label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score During Fine-tuning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('report/images/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Data distribution
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    datasets = [pretrain_list, finetune_list, candidate_list]
    names = ['Pre-train\n(5000)', 'Fine-tune\n(2000)', 'Candidate\n(1000)']
    colors_pie = ['lightgreen', 'lightcoral']
    for ax, ds, name in zip(axes, datasets, names):
        if hasattr(ds[0], 'y') and ds[0].y is not None:
            counts = [sum([d.y.item()==0 for d in ds]), sum([d.y.item()==1 for d in ds])]
            ax.pie(counts, labels=['Non-AM', 'AM'], autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax.set_title(name)
    plt.suptitle('Dataset Label Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('report/images/data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    torch.save(classifier.state_dict(), 'outputs/classifier_best.pt')
    
    print("\n=== All done! ===")

if __name__ == '__main__':
    main()
