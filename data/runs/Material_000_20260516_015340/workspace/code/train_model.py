#!/usr/bin/env python3
"""
Altermagnetic Material Discovery using Self-Supervised GNN Pre-training + Fine-tuning.

Pipeline:
1. Self-supervised pre-training on unlabeled crystal graphs via node masking
2. Fine-tuning on labeled altermagnet data (imbalanced: ~5% positive)
3. Prediction on candidate materials
4. Evaluation and analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_edge
import numpy as np
import json
import os
import sys
from collections import Counter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'hidden_dim': 128,
    'num_layers': 4,
    'dropout': 0.3,
    'pretrain_epochs': 200,
    'finetune_epochs': 300,
    'batch_size': 64,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'pos_weight_ratio': 15.0,  # class imbalance handling
    'mask_ratio': 0.3,          # node masking ratio for pretraining
    'seed': 42,
    'device': 'cpu',
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# ============================================================
# Dataset Loading
# ============================================================
def load_dataset(path):
    """Load a RealisticCrystalDataset and convert to list of PyG Data objects."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    samples = []
    for i in range(len(data)):
        item = data[i]
        samples.append(item)
    return samples

# ============================================================
# Graph Encoder
# ============================================================
class GraphEncoder(nn.Module):
    """Stacked GCN encoder with skip connections and pooling."""
    def __init__(self, input_dim, hidden_dim, num_layers=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input projection
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if i > 0:
                h_new = h_new + h  # skip connection
            h = h_new
        return h

# ============================================================
# Node Feature Decoder (for pre-training)
# ============================================================
class NodeDecoder(nn.Module):
    """Decode node features from latent representations."""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h):
        return self.net(h)

# ============================================================
# Classifier Head (for fine-tuning)
# ============================================================
class ClassifierHead(nn.Module):
    """Graph-level classifier."""
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, batch):
        # Global pooling: mean + max
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_cat = torch.cat([h_mean, h_max], dim=1)
        return self.net(h_cat).squeeze(-1)

# ============================================================
# Self-Supervised Pre-training
# ============================================================
def pretrain_epoch(model, decoder, loader, optimizer, mask_ratio, device):
    model.train()
    decoder.train()
    total_loss = 0.0
    n_batches = 0

    for data in loader:
        data = data.to(device)
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # Create masked version
        num_nodes = x.size(0)
        mask = torch.rand(num_nodes, device=device) < mask_ratio
        # Don't mask all nodes
        if mask.all():
            mask[0] = False

        x_masked = x.clone()
        original_features = x[mask].clone()
        x_masked[mask] = 0.0  # Mask with zeros

        # Forward
        h = model(x_masked, edge_index, batch)

        # Decode masked nodes
        pred = decoder(h[mask])

        # Loss: binary cross entropy per element
        loss = F.binary_cross_entropy_with_logits(pred, original_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

# ============================================================
# Fine-tuning
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
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        labels = data.y.float()

        with torch.set_grad_enabled(train):
            h = model(x, edge_index, batch)
            logits = classifier(h, batch)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, np.array(all_preds), np.array(all_labels)

# ============================================================
# Collate function for PyG DataLoader
# ============================================================
class SimpleBatch:
    """Simple batch class to hold batched graph data."""
    pass

def simple_collate(data_list):
    """Collate a list of PyG Data objects into a batch."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(data_list)

# ============================================================
# Main Training Pipeline
# ============================================================
def main():
    device = CONFIG['device']
    print(f"Using device: {device}")

    # ---------- Load Data ----------
    print("\n=== Loading Datasets ===")
    pretrain_samples = load_dataset('data/pretrain_data.pt')
    finetune_samples = load_dataset('data/finetune_data.pt')
    candidate_samples = load_dataset('data/candidate_data.pt')

    print(f"Pretrain: {len(pretrain_samples)} samples")
    print(f"Finetune: {len(finetune_samples)} samples")
    print(f"Candidate: {len(candidate_samples)} samples")

    # Analyze finetune labels
    ft_labels = [int(s.y.item()) for s in finetune_samples]
    print(f"Finetune label distribution: {Counter(ft_labels)}")

    # Node feature dimension
    input_dim = pretrain_samples[0].x.size(1)
    print(f"Node feature dim: {input_dim}")

    # ---------- Split finetune data ----------
    from sklearn.model_selection import train_test_split
    ft_indices = list(range(len(finetune_samples)))
    ft_y = np.array(ft_labels)
    train_idx, val_idx = train_test_split(
        ft_indices, test_size=0.2, random_state=CONFIG['seed'], stratify=ft_y
    )
    train_samples = [finetune_samples[i] for i in train_idx]
    val_samples = [finetune_samples[i] for i in val_idx]

    train_labels = [int(s.y.item()) for s in train_samples]
    val_labels = [int(s.y.item()) for s in val_samples]
    print(f"Train labels: {Counter(train_labels)}")
    print(f"Val labels: {Counter(val_labels)}")

    # ---------- DataLoaders ----------
    pretrain_loader = DataLoader(pretrain_samples, batch_size=CONFIG['batch_size'], shuffle=True)
    train_loader = DataLoader(train_samples, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=CONFIG['batch_size'], shuffle=False)
    candidate_loader = DataLoader(candidate_samples, batch_size=CONFIG['batch_size'], shuffle=False)

    # ---------- Build Model ----------
    encoder = GraphEncoder(
        input_dim=input_dim,
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
    ).to(device)

    # ---------- Phase 1: Self-Supervised Pre-training ----------
    print("\n=== Phase 1: Self-Supervised Pre-training ===")
    decoder = NodeDecoder(CONFIG['hidden_dim'], input_dim).to(device)
    pretrain_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )

    best_pretrain_loss = float('inf')
    pretrain_losses = []

    for epoch in range(1, CONFIG['pretrain_epochs'] + 1):
        loss = pretrain_epoch(encoder, decoder, pretrain_loader, pretrain_optimizer,
                              CONFIG['mask_ratio'], device)
        pretrain_losses.append(loss)

        if loss < best_pretrain_loss:
            best_pretrain_loss = loss
            torch.save(encoder.state_dict(), 'outputs/pretrained_encoder.pt')

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CONFIG['pretrain_epochs']}: Loss = {loss:.6f}")

    print(f"Best pretrain loss: {best_pretrain_loss:.6f}")

    # Load best pretrained encoder
    encoder.load_state_dict(torch.load('outputs/pretrained_encoder.pt', map_location=device))

    # ---------- Phase 2: Fine-tuning ----------
    print("\n=== Phase 2: Fine-tuning ===")
    classifier = ClassifierHead(CONFIG['hidden_dim'], CONFIG['dropout']).to(device)

    # Weighted BCE for class imbalance
    pos_weight = torch.tensor([CONFIG['pos_weight_ratio']], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    finetune_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=CONFIG['lr'] * 0.5, weight_decay=CONFIG['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        finetune_optimizer, mode='max', factor=0.5, patience=20
    )

    best_val_auroc = 0.0
    train_losses = []
    val_losses = []
    val_aurocs = []

    for epoch in range(1, CONFIG['finetune_epochs'] + 1):
        train_loss, train_preds, train_labs = finetune_epoch(
            encoder, classifier, train_loader, finetune_optimizer, criterion, device, train=True
        )
        val_loss, val_preds, val_labs = finetune_epoch(
            encoder, classifier, val_loader, None, criterion, device, train=False
        )

        val_auroc = roc_auc_score(val_labs, val_preds) if len(np.unique(val_labs)) > 1 else 0.5

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)

        scheduler.step(val_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
            }, 'outputs/best_model.pt')

        if epoch % 30 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUROC={val_auroc:.4f}")

    print(f"\nBest Val AUROC: {best_val_auroc:.4f}")

    # Load best model
    best_ckpt = torch.load('outputs/best_model.pt', map_location=device)
    encoder.load_state_dict(best_ckpt['encoder'])
    classifier.load_state_dict(best_ckpt['classifier'])

    # ---------- Final Validation Evaluation ----------
    print("\n=== Validation Set Evaluation ===")
    _, val_preds, val_labs = finetune_epoch(
        encoder, classifier, val_loader, None, criterion, device, train=False
    )

    val_auroc = roc_auc_score(val_labs, val_preds)
    val_auprc = average_precision_score(val_labs, val_preds)

    # Find best threshold on validation set
    precisions, recalls, thresholds = precision_recall_curve(val_labs, val_preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]

    val_pred_binary = (val_preds >= best_threshold).astype(int)
    val_acc = accuracy_score(val_labs, val_pred_binary)
    val_prec = precision_score(val_labs, val_pred_binary, zero_division=0)
    val_rec = recall_score(val_labs, val_pred_binary, zero_division=0)

    print(f"  AUROC: {val_auroc:.4f}")
    print(f"  AUPRC: {val_auprc:.4f}")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  F1: {best_f1:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall: {val_rec:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(val_labs, val_pred_binary)}")

    # ---------- Phase 3: Candidate Prediction ----------
    print("\n=== Phase 3: Candidate Prediction ===")
    encoder.eval()
    classifier.eval()

    candidate_preds = []
    candidate_labels = []
    node_counts = []

    for data in candidate_loader:
        data = data.to(device)
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        labels = data.y.float()

        with torch.no_grad():
            h = encoder(x, edge_index, batch)
            logits = classifier(h, batch)
            probs = torch.sigmoid(logits)

        candidate_preds.extend(probs.cpu().numpy().tolist())
        candidate_labels.extend(labels.cpu().numpy().tolist())

    candidate_preds = np.array(candidate_preds)
    candidate_labels = np.array(candidate_labels)

    # Predict using best threshold
    candidate_binary = (candidate_preds >= best_threshold).astype(int)
    n_predicted_pos = candidate_binary.sum()

    print(f"  Total candidates: {len(candidate_preds)}")
    print(f"  Predicted positive (altermagnets): {n_predicted_pos}")
    print(f"  True positives in candidate set: {int(candidate_labels.sum())}")

    # Evaluate against hidden labels
    cand_auroc = roc_auc_score(candidate_labels, candidate_preds)
    cand_auprc = average_precision_score(candidate_labels, candidate_preds)
    cand_prec = precision_score(candidate_labels, candidate_binary, zero_division=0)
    cand_rec = recall_score(candidate_labels, candidate_binary, zero_division=0)
    cand_f1 = f1_score(candidate_labels, candidate_binary, zero_division=0)
    cand_acc = accuracy_score(candidate_labels, candidate_binary)

    print(f"\n  Candidate Set Metrics:")
    print(f"    AUROC: {cand_auroc:.4f}")
    print(f"    AUPRC: {cand_auprc:.4f}")
    print(f"    Accuracy: {cand_acc:.4f}")
    print(f"    Precision: {cand_prec:.4f}")
    print(f"    Recall: {cand_rec:.4f}")
    print(f"    F1: {cand_f1:.4f}")
    cm = confusion_matrix(candidate_labels, candidate_binary)
    print(f"    Confusion Matrix:\n{cm}")

    # ---------- Save Results ----------
    results = {
        'config': CONFIG,
        'pretrain_best_loss': best_pretrain_loss,
        'pretrain_losses': pretrain_losses,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aurocs': val_aurocs,
        'best_val_auroc': best_val_auroc,
        'val_metrics': {
            'auroc': float(val_auroc),
            'auprc': float(val_auprc),
            'f1': float(best_f1),
            'accuracy': float(val_acc),
            'precision': float(val_prec),
            'recall': float(val_rec),
            'threshold': float(best_threshold),
        },
        'candidate_metrics': {
            'auroc': float(cand_auroc),
            'auprc': float(cand_auprc),
            'accuracy': float(cand_acc),
            'precision': float(cand_prec),
            'recall': float(cand_rec),
            'f1': float(cand_f1),
            'n_predicted_positive': int(n_predicted_pos),
            'n_true_positive': int(candidate_labels.sum()),
            'confusion_matrix': cm.tolist(),
        },
        'candidate_predictions': candidate_preds.tolist(),
        'candidate_labels': candidate_labels.tolist(),
        'candidate_binary': candidate_binary.tolist(),
    }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save detailed candidate list
    candidate_details = []
    for i, (prob, label, binary) in enumerate(zip(
        candidate_preds, candidate_labels, candidate_binary
    )):
        candidate_details.append({
            'index': i,
            'probability': float(prob),
            'predicted_altermagnet': bool(binary),
            'true_altermagnet': bool(label),
        })

    # Sort by probability descending
    candidate_details.sort(key=lambda x: x['probability'], reverse=True)

    with open('outputs/candidate_details.json', 'w') as f:
        json.dump(candidate_details, f, indent=2)

    # Print top predictions
    print("\n=== Top 20 Predicted Altermagnets ===")
    for d in candidate_details[:20]:
        marker = "✓" if d['true_altermagnet'] else "✗"
        print(f"  #{d['index']:4d}: p={d['probability']:.4f} [{marker}]")

    # Count how many in top-50 are true
    top50 = candidate_details[:50]
    top50_correct = sum(1 for d in top50 if d['true_altermagnet'])
    print(f"\n  Top-50 precision: {top50_correct}/50 = {top50_correct/50:.2%}")

    # Also try more aggressive threshold: top-50 by probability
    top50_by_prob = [d for d in candidate_details[:50]]
    top50_correct_by_prob = sum(1 for d in top50_by_prob if d['true_altermagnet'])
    print(f"  Top-50 by probability precision: {top50_correct_by_prob}/50 = {top50_correct_by_prob/50:.2%}")

    print("\n=== Training Complete ===")
    return results

if __name__ == '__main__':
    main()
