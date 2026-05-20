#!/usr/bin/env python3
"""
Altermagnetic Material Discovery - Improved Pipeline

Key improvements:
1. Edge-feature-aware GNN encoder (GIN with edge features)
2. Better self-supervised pretraining: edge prediction + node masking
3. Focal loss for class imbalance
4. More robust evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import dropout_edge, add_self_loops
import numpy as np
import json
import os
from collections import Counter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'hidden_dim': 256,
    'num_layers': 5,
    'dropout': 0.3,
    'pretrain_epochs': 300,
    'finetune_epochs': 400,
    'batch_size': 64,
    'lr': 0.0005,
    'weight_decay': 1e-4,
    'pos_weight': 20.0,
    'mask_ratio': 0.4,
    'seed': 42,
    'device': 'cpu',
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# ============================================================
# Data Loading
# ============================================================
def load_dataset(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    samples = []
    for i in range(len(data)):
        samples.append(data[i])
    return samples

# ============================================================
# Edge-feature-aware GNN Encoder
# ============================================================
class EdgeGNNEncoder(nn.Module):
    """GNN encoder using GINEConv (supports edge features)."""
    def __init__(self, input_dim, edge_dim, hidden_dim, num_layers=5, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.node_encoder(x)

        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = None

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if i > 0:
                h_new = h_new + h  # skip connection
            h = h_new

        return h

# ============================================================
# Pretraining: Node masking + Edge prediction
# ============================================================
class PretrainHeads(nn.Module):
    """Multi-task pretraining heads."""
    def __init__(self, hidden_dim, node_output_dim, edge_output_dim):
        super().__init__()
        # Node feature reconstruction head
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_output_dim),
        )
        # Edge prediction head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_output_dim),
        )

    def forward_node(self, h):
        return self.node_head(h)

    def forward_edge(self, h, edge_index):
        src, dst = edge_index
        h_edge = torch.cat([h[src], h[dst]], dim=-1)
        return self.edge_head(h_edge)

# ============================================================
# Classification Head
# ============================================================
class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, h, batch):
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_add = global_add_pool(h, batch)
        h_cat = torch.cat([h_mean, h_max, h_add], dim=1)
        return self.net(h_cat).squeeze(-1)

# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ============================================================
# Pretraining epoch
# ============================================================
def pretrain_epoch(model, pretrain_heads, loader, optimizer, mask_ratio, device):
    model.train()
    pretrain_heads.train()
    total_loss = 0.0
    n_batches = 0

    for data in loader:
        data = data.to(device)
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)

        # Node masking
        num_nodes = x.size(0)
        mask = torch.rand(num_nodes, device=device) < mask_ratio
        if mask.all():
            mask[0] = False

        x_masked = x.clone()
        original_node_feats = x[mask].clone()
        x_masked[mask] = 0.0

        # Edge perturbation: drop some edges, add negative edges
        num_edges = edge_index.size(1)
        edge_mask = torch.rand(num_edges, device=device) < 0.15  # drop 15% edges
        edge_index_used = edge_index[:, ~edge_mask]
        edge_attr_used = edge_attr[~edge_mask] if edge_attr is not None else None

        # Forward
        h = model(x_masked, edge_index_used, edge_attr_used, batch)

        # Node loss
        node_pred = pretrain_heads.forward_node(h[mask])
        node_loss = F.binary_cross_entropy_with_logits(node_pred, original_node_feats)

        # Edge loss on dropped edges (positive) + random negatives
        pos_edge_index = edge_index[:, edge_mask]
        num_pos = pos_edge_index.size(1)

        # Sample negative edges
        num_neg = min(num_pos, num_nodes * 3)
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
        neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)

        pos_edge_pred = pretrain_heads.forward_edge(h, pos_edge_index)
        neg_edge_pred = pretrain_heads.forward_edge(h, neg_edge_index)

        pos_labels = torch.ones(num_pos, device=device)
        neg_labels = torch.zeros(num_neg, device=device)

        all_edge_pred = torch.cat([pos_edge_pred, neg_edge_pred], dim=0)
        all_edge_labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Edge prediction is binary classification per edge dimension (or just use first dim)
        edge_loss = F.binary_cross_entropy_with_logits(
            all_edge_pred[:, 0], all_edge_labels
        ) if all_edge_pred.size(1) > 0 else torch.tensor(0.0, device=device)

        # For multi-dim edge attr, also reconstruct the dropped edge attributes
        edge_recon_loss = 0.0
        if edge_attr is not None and num_pos > 0:
            edge_recon_loss = F.mse_loss(pos_edge_pred, edge_attr[edge_mask])

        loss = node_loss + 0.3 * edge_loss + 0.1 * edge_recon_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(pretrain_heads.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

# ============================================================
# Finetuning epoch
# ============================================================
def finetune_epoch(model, classifier, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
        classifier.train()
    else:
        model.eval()
        classifier.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0

    for data in loader:
        data = data.to(device)
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)
        labels = data.y.float()

        with torch.set_grad_enabled(train):
            h = model(x, edge_index, edge_attr, batch)
            logits = classifier(h, batch)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, np.array(all_preds), np.array(all_labels)

# ============================================================
# Main
# ============================================================
def main():
    device = CONFIG['device']
    print(f"Using device: {device}")

    # Load data
    print("\n=== Loading Datasets ===")
    pretrain_samples = load_dataset('data/pretrain_data.pt')
    finetune_samples = load_dataset('data/finetune_data.pt')
    candidate_samples = load_dataset('data/candidate_data.pt')

    print(f"Pretrain: {len(pretrain_samples)}")
    print(f"Finetune: {len(finetune_samples)}")
    print(f"Candidate: {len(candidate_samples)}")

    ft_labels = [int(s.y.item()) for s in finetune_samples]
    print(f"Finetune label dist: {Counter(ft_labels)}")

    input_dim = pretrain_samples[0].x.size(1)
    edge_dim = pretrain_samples[0].edge_attr.size(1) if pretrain_samples[0].edge_attr is not None else 1
    print(f"Input dim: {input_dim}, Edge dim: {edge_dim}")

    # Split finetune data with stratification
    ft_indices = list(range(len(finetune_samples)))
    ft_y = np.array(ft_labels)
    train_idx, val_idx = train_test_split(
        ft_indices, test_size=0.2, random_state=CONFIG['seed'], stratify=ft_y
    )
    train_samples = [finetune_samples[i] for i in train_idx]
    val_samples = [finetune_samples[i] for i in val_idx]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Train labels: {Counter([int(s.y.item()) for s in train_samples])}")
    print(f"Val labels: {Counter([int(s.y.item()) for s in val_samples])}")

    # DataLoaders
    pretrain_loader = DataLoader(pretrain_samples, batch_size=CONFIG['batch_size'], shuffle=True)
    train_loader = DataLoader(train_samples, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=CONFIG['batch_size'], shuffle=False)
    candidate_loader = DataLoader(candidate_samples, batch_size=CONFIG['batch_size'], shuffle=False)

    # Build model
    model = EdgeGNNEncoder(
        input_dim=input_dim,
        edge_dim=edge_dim,
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
    ).to(device)

    pretrain_heads = PretrainHeads(CONFIG['hidden_dim'], input_dim, edge_dim).to(device)

    # ---------- Phase 1: Pretraining ----------
    print("\n=== Phase 1: Self-Supervised Pretraining ===")
    pretrain_optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(pretrain_heads.parameters()),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )
    scheduler_pt = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=CONFIG['pretrain_epochs'])

    best_pt_loss = float('inf')
    pt_losses = []

    for epoch in range(1, CONFIG['pretrain_epochs'] + 1):
        loss = pretrain_epoch(model, pretrain_heads, pretrain_loader, pretrain_optimizer,
                              CONFIG['mask_ratio'], device)
        scheduler_pt.step()
        pt_losses.append(loss)

        if loss < best_pt_loss:
            best_pt_loss = loss
            torch.save(model.state_dict(), 'outputs/pretrained_encoder_v2.pt')

        if epoch % 30 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CONFIG['pretrain_epochs']}: Loss = {loss:.6f}")

    print(f"Best pretrain loss: {best_pt_loss:.6f}")
    model.load_state_dict(torch.load('outputs/pretrained_encoder_v2.pt', map_location=device))

    # ---------- Phase 2: Fine-tuning ----------
    print("\n=== Phase 2: Fine-tuning ===")
    classifier = ClassifierHead(CONFIG['hidden_dim'], CONFIG['dropout']).to(device)

    # Use focal loss with class-specific weighting
    criterion = FocalLoss(alpha=0.8, gamma=2.0)

    ft_optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=CONFIG['lr'] * 0.1, weight_decay=CONFIG['weight_decay']
    )
    scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ft_optimizer, mode='max', factor=0.5, patience=30
    )

    best_val_auroc = 0.0
    best_val_auprc = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_aurocs = []
    val_auprcs = []

    for epoch in range(1, CONFIG['finetune_epochs'] + 1):
        train_loss, train_preds, train_labs = finetune_epoch(
            model, classifier, train_loader, ft_optimizer, criterion, device, train=True
        )
        val_loss, val_preds, val_labs = finetune_epoch(
            model, classifier, val_loader, None, criterion, device, train=False
        )

        val_auroc = roc_auc_score(val_labs, val_preds) if len(np.unique(val_labs)) > 1 else 0.5
        val_auprc = average_precision_score(val_labs, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)
        val_auprcs.append(val_auprc)

        scheduler_ft.step(val_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_auprc = val_auprc
            best_epoch = epoch
            torch.save({
                'encoder': model.state_dict(),
                'classifier': classifier.state_dict(),
            }, 'outputs/best_model_v2.pt')

        if epoch % 40 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val AUROC={val_auroc:.4f}, Val AUPRC={val_auprc:.4f}")

    print(f"\nBest epoch: {best_epoch}, AUROC={best_val_auroc:.4f}, AUPRC={best_val_auprc:.4f}")

    # Load best model
    best_ckpt = torch.load('outputs/best_model_v2.pt', map_location=device)
    model.load_state_dict(best_ckpt['encoder'])
    classifier.load_state_dict(best_ckpt['classifier'])

    # ---------- Validation Evaluation ----------
    print("\n=== Validation Set Evaluation ===")
    _, val_preds, val_labs = finetune_epoch(
        model, classifier, val_loader, None, criterion, device, train=False
    )

    val_auroc = roc_auc_score(val_labs, val_preds)
    val_auprc = average_precision_score(val_labs, val_preds)

    precisions, recalls, thresholds = precision_recall_curve(val_labs, val_preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]

    val_pred_binary = (val_preds >= best_threshold).astype(int)
    val_acc = accuracy_score(val_labs, val_pred_binary)
    val_prec = precision_score(val_labs, val_pred_binary, zero_division=0)
    val_rec = recall_score(val_labs, val_pred_binary, zero_division=0)

    print(f"  AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")
    print(f"  Threshold: {best_threshold:.4f}, F1: {best_f1:.4f}")
    print(f"  Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")
    cm = confusion_matrix(val_labs, val_pred_binary)
    print(f"  Confusion Matrix:\n{cm}")

    # ---------- Candidate Prediction ----------
    print("\n=== Candidate Prediction ===")
    model.eval()
    classifier.eval()

    candidate_preds = []
    candidate_labels = []

    for data in candidate_loader:
        data = data.to(device)
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)
        labels = data.y.float()

        with torch.no_grad():
            h = model(x, edge_index, edge_attr, batch)
            logits = classifier(h, batch)
            probs = torch.sigmoid(logits)

        candidate_preds.extend(probs.cpu().numpy().tolist())
        candidate_labels.extend(labels.cpu().numpy().tolist())

    candidate_preds = np.array(candidate_preds)
    candidate_labels = np.array(candidate_labels)

    # Binary predictions
    candidate_binary = (candidate_preds >= best_threshold).astype(int)
    n_pred_pos = candidate_binary.sum()
    n_true_pos = int(candidate_labels.sum())

    print(f"  Total: {len(candidate_preds)}, Predicted positive: {n_pred_pos}, True positive: {n_true_pos}")

    cand_auroc = roc_auc_score(candidate_labels, candidate_preds)
    cand_auprc = average_precision_score(candidate_labels, candidate_preds)
    cand_prec = precision_score(candidate_labels, candidate_binary, zero_division=0)
    cand_rec = recall_score(candidate_labels, candidate_binary, zero_division=0)
    cand_f1 = f1_score(candidate_labels, candidate_binary, zero_division=0)
    cand_acc = accuracy_score(candidate_labels, candidate_binary)

    print(f"  AUROC: {cand_auroc:.4f}, AUPRC: {cand_auprc:.4f}")
    print(f"  Accuracy: {cand_acc:.4f}, Precision: {cand_prec:.4f}")
    print(f"  Recall: {cand_rec:.4f}, F1: {cand_f1:.4f}")
    cm2 = confusion_matrix(candidate_labels, candidate_binary)
    print(f"  Confusion Matrix:\n{cm2}")

    # Sort by probability
    sorted_indices = np.argsort(-candidate_preds)
    sorted_probs = candidate_preds[sorted_indices]
    sorted_labels = candidate_labels[sorted_indices]

    # Top-k analysis
    for k in [20, 50, 100]:
        top_k_correct = int(sorted_labels[:k].sum())
        print(f"  Top-{k} precision: {top_k_correct}/{k} = {top_k_correct/k:.2%}")

    # Print top predictions
    print("\n=== Top 30 Predicted Altermagnets ===")
    for i in range(30):
        idx = int(sorted_indices[i])
        prob = float(sorted_probs[i])
        label = bool(sorted_labels[i])
        marker = "✓" if label else "✗"
        print(f"  #{idx:4d}: p={prob:.4f} [{marker}]")

    # ---------- Save Results ----------
    results = {
        'config': {k: v for k, v in CONFIG.items() if k != 'device'},
        'pretrain_best_loss': best_pt_loss,
        'pretrain_losses': pt_losses,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aurocs': val_aurocs,
        'val_auprcs': val_auprcs,
        'best_val_auroc': float(best_val_auroc),
        'best_val_auprc': float(best_val_auprc),
        'best_epoch': best_epoch,
        'val_metrics': {
            'auroc': float(val_auroc),
            'auprc': float(val_auprc),
            'f1': float(best_f1),
            'accuracy': float(val_acc),
            'precision': float(val_prec),
            'recall': float(val_rec),
            'threshold': float(best_threshold),
            'confusion_matrix': cm.tolist(),
        },
        'candidate_metrics': {
            'auroc': float(cand_auroc),
            'auprc': float(cand_auprc),
            'accuracy': float(cand_acc),
            'precision': float(cand_prec),
            'recall': float(cand_rec),
            'f1': float(cand_f1),
            'n_predicted_positive': int(n_pred_pos),
            'n_true_positive': n_true_pos,
            'confusion_matrix': cm2.tolist(),
        },
        'candidate_predictions': candidate_preds.tolist(),
        'candidate_labels': candidate_labels.tolist(),
        'candidate_binary': candidate_binary.tolist(),
        'top50_precision': float(sorted_labels[:50].mean()),
        'top100_precision': float(sorted_labels[:100].mean()),
    }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Candidate details
    candidate_details = []
    for i in range(len(candidate_preds)):
        candidate_details.append({
            'index': i,
            'probability': float(candidate_preds[i]),
            'predicted_altermagnet': bool(candidate_binary[i]),
            'true_altermagnet': bool(candidate_labels[i]),
        })
    candidate_details.sort(key=lambda x: x['probability'], reverse=True)

    with open('outputs/candidate_details_v2.json', 'w') as f:
        json.dump(candidate_details, f, indent=2)

    print("\n=== Training Complete ===")
    return results

if __name__ == '__main__':
    main()
