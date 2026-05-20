"""
Altermagnetic Material Discovery (v5)
- Train encoder on pretrain data (supervised)
- Freeze encoder, train classifier on finetune data
- Ensemble with feature-based model
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool, AttentionalAggregation
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)
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

def load_datasets():
    pretrain = torch.load('data/pretrain_data.pt', map_location='cpu', weights_only=False)
    finetune = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
    candidate = torch.load('data/candidate_data.pt', map_location='cpu', weights_only=False)
    return pretrain.data_list, finetune.data_list, candidate.data_list

def extract_graph_features(data_list):
    features = []
    labels = []
    for d in data_list:
        x = d.x.numpy()
        e = d.edge_attr.numpy() if d.edge_attr is not None else np.zeros((1, 2))
        node_mean = x.mean(axis=0)
        node_max = x.max(axis=0)
        node_sum = x.sum(axis=0)
        node_std = x.std(axis=0)
        edge_mean = e.mean(axis=0)
        edge_sum = e.sum(axis=0)
        num_nodes = x.shape[0]
        num_edges = e.shape[0]
        density = num_edges / max(1, num_nodes * (num_nodes - 1))
        feat = np.concatenate([node_mean, node_max, node_sum, node_std, edge_mean, edge_sum, [num_nodes, num_edges, density]])
        features.append(feat)
        labels.append(d.y.item() if hasattr(d, 'y') and d.y is not None else 0)
    return np.array(features), np.array(labels)

# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=128, num_layers=2):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1)))
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_emb = self.edge_emb(edge_attr) if edge_attr is not None else None
        x = self.node_emb(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = bn(x)
            x = F.relu(x)
        return self.pool(x, batch)

class Classifier(nn.Module):
    def __init__(self, encoder, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, data):
        emb = self.encoder(data)
        return self.head(emb).squeeze(-1)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    probs, labels, preds = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)
        p = torch.sigmoid(logit).cpu().numpy()
        l = batch.y.float().cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(l.tolist())
        preds.extend((p >= 0.5).astype(int).tolist())
    probs = np.array(probs); labels = np.array(labels); preds = np.array(preds)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        'pr_auc': float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
    }, probs, labels, preds

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logit = model(batch)
        loss = criterion(logit, batch.y.float())
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
    
    pretrain_list, finetune_list, candidate_list = load_datasets()
    for d in pretrain_list + finetune_list + candidate_list:
        d.y = d.y.float()
    
    print(f"Pretrain: {len(pretrain_list)}, Finetune: {len(finetune_list)}, Candidate: {len(candidate_list)}")
    
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
    
    # Oversample
    pos = [d for d in train_data if d.y.item() == 1.0]
    neg = [d for d in train_data if d.y.item() == 0.0]
    train_bal = neg + pos * 10
    random.shuffle(train_bal)
    
    pretrain_loader = PyGDataLoader(pretrain_list, batch_size=128, shuffle=True, num_workers=0)
    train_loader = PyGDataLoader(train_bal, batch_size=64, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)
    candidate_loader = PyGDataLoader(candidate_list, batch_size=64, shuffle=False, num_workers=0)
    
    encoder = Encoder(node_dim=28, edge_dim=2, hidden_dim=128, num_layers=2).to(device)
    classifier = Classifier(encoder, hidden_dim=128).to(device)
    
    # Phase 1: Pre-train encoder + classifier on pretrain data
    print("\n=== Phase 1: Supervised Pre-training on Pretrain Data ===")
    pos_w = len(pretrain_list) / max(1, sum([d.y.item() for d in pretrain_list])) - 1
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    for epoch in range(1, 31):
        loss = train_epoch(classifier, pretrain_loader, optimizer, criterion, device)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    
    # Phase 2: Freeze encoder, fine-tune classifier head only
    print("\n=== Phase 2: Fine-tuning (Frozen Encoder) ===")
    for param in classifier.encoder.parameters():
        param.requires_grad = False
    
    pos_w = len(train_bal) / max(1, sum([d.y.item() for d in train_bal])) - 1
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(classifier.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_auc = 0
    best_state = None
    for epoch in range(1, 101):
        loss = train_epoch(classifier, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0:
            metrics, _, _, _ = eval_model(classifier, val_loader, device)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {metrics['roc_auc']:.4f} | Val F1: {metrics['f1']:.4f}")
            if metrics['roc_auc'] > best_val_auc:
                best_val_auc = metrics['roc_auc']
                best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
                print(f"  --> New best: {best_val_auc:.4f}")
            scheduler.step(metrics['roc_auc'])
    
    if best_state is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Test
    print("\n=== Test Set Evaluation ===")
    test_metrics, test_probs, test_labels, test_preds = eval_model(classifier, test_loader, device)
    print(f"Test AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    
    # Candidates
    print("\n=== Candidate Screening ===")
    classifier.eval()
    cand_probs = []
    cand_labels = []
    with torch.no_grad():
        for batch in candidate_loader:
            batch = batch.to(device)
            logit = classifier(batch)
            p = torch.sigmoid(logit).cpu().numpy()
            l = batch.y.float().cpu().numpy()
            cand_probs.extend(p.tolist())
            cand_labels.extend(l.tolist())
    cand_probs = np.array(cand_probs)
    cand_labels = np.array(cand_labels)
    
    top50 = np.argsort(cand_probs)[::-1][:50]
    discovered = int(cand_labels[top50].sum())
    print(f"Top-50 discovered: {discovered}, Precision@50: {discovered/50:.4f}")
    
    # Save
    with open('outputs/metrics.json', 'w') as f:
        json.dump({'test': test_metrics, 'val_best_auc': float(best_val_auc)}, f, indent=2)
    
    with open('outputs/candidate_results.json', 'w') as f:
        json.dump({
            'precision_at_50': float(discovered/50),
            'recall_at_50': float(discovered/max(1, cand_labels.sum())),
            'num_true_positives_in_candidate': int(cand_labels.sum())
        }, f, indent=2)
    
    # Figures
    print("\n=== Generating Figures ===")
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f'Test ROC (AUC = {test_metrics["roc_auc"]:.3f})')
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('report/images/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_probs)
    plt.figure(figsize=(7,6))
    plt.plot(recall_curve, precision_curve, lw=2, label=f'Test PR (AP = {test_metrics["pr_auc"]:.3f})')
    plt.axhline(y=test_labels.mean(), color='r', linestyle='--', label=f'Baseline ({test_labels.mean():.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('report/images/pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Non-AM', 'AM'], yticklabels=['Non-AM', 'AM'])
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('report/images/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8,5))
    bins = np.linspace(0, 1, 31)
    plt.hist(cand_probs[cand_labels==0], bins=bins, alpha=0.6, label='Non-AM', color='steelblue')
    plt.hist(cand_probs[cand_labels==1], bins=bins, alpha=0.6, label='AM', color='crimson')
    plt.xlabel('Predicted Probability'); plt.ylabel('Count')
    plt.title('Candidate Score Distribution')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('report/images/candidate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12,6))
    colors = ['crimson' if l == 1 else 'steelblue' for l in cand_labels[top50]]
    plt.bar(range(50), cand_probs[top50], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.xlabel('Candidate Rank'); plt.ylabel('Predicted Probability')
    plt.title('Top-50 Predicted Altermagnets (Red = True Positive)')
    plt.ylim(0, 1.05); plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('report/images/top_candidates.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    k_values = list(range(10, min(201, len(candidate_list)+1), 10))
    precisions, recalls = [], []
    sorted_idx = np.argsort(cand_probs)[::-1]
    for k in k_values:
        precisions.append(float(cand_labels[sorted_idx[:k]].mean()))
        recalls.append(float(cand_labels[sorted_idx[:k]].sum() / max(1, cand_labels.sum())))
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(k_values, precisions, 'b-o', label='Precision@K')
    ax1.set_xlabel('K'); ax1.set_ylabel('Precision@K', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(k_values, recalls, 'r-s', label='Recall@K')
    ax2.set_ylabel('Recall@K', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Precision and Recall at K')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.9))
    plt.grid(True, alpha=0.3)
    plt.savefig('report/images/precision_recall_at_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, ds, name in zip(axes, [pretrain_list, finetune_list, candidate_list], ['Pre-train\n(5000)', 'Fine-tune\n(2000)', 'Candidate\n(1000)']):
        counts = [sum([d.y.item()==0 for d in ds]), sum([d.y.item()==1 for d in ds])]
        ax.pie(counts, labels=['Non-AM', 'AM'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
        ax.set_title(name)
    plt.suptitle('Dataset Label Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('report/images/data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    torch.save(classifier.state_dict(), 'outputs/classifier_best.pt')
    print("\n=== All done! ===")

if __name__ == '__main__':
    main()
