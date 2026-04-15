"""
AI-Powered Search Engine for Altermagnetic Materials Discovery

This module implements a graph neural network-based approach to discover
altermagnetic materials from crystal structure data.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                             average_precision_score, confusion_matrix,
                             classification_report, precision_score, recall_score, f1_score)
from tqdm import tqdm
import pickle
import json

# Add parent directory to path for data_prepare module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prepare

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CrystalDataset(Dataset):
    """Wrapper for crystal structure data."""
    
    def __init__(self, data_list, has_labels=True):
        self.data_list = data_list
        self.has_labels = has_labels
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None
        
        if self.has_labels and hasattr(data, 'y'):
            y = data.y.float()
            return x, edge_index, edge_attr, y, data.num_nodes
        else:
            return x, edge_index, edge_attr, None, data.num_nodes


def collate_fn(batch):
    """Custom collate function for batching graphs."""
    x_list, edge_index_list, edge_attr_list, y_list, num_nodes_list = [], [], [], [], []
    
    node_offset = 0
    for x, edge_index, edge_attr, y, num_nodes in batch:
        x_list.append(x)
        # Adjust edge indices for batching
        edge_index_list.append(edge_index + node_offset)
        edge_attr_list.append(edge_attr)
        y_list.append(y)
        num_nodes_list.append(num_nodes)
        node_offset += num_nodes
    
    x_batch = torch.cat(x_list, dim=0)
    edge_index_batch = torch.cat(edge_index_list, dim=1)
    edge_attr_batch = torch.cat(edge_attr_list, dim=0) if edge_attr_list[0] is not None else None
    y_batch = torch.stack([y for y in y_list if y is not None]) if y_list[0] is not None else None
    batch_idx = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(num_nodes_list))
    
    return x_batch, edge_index_batch, edge_attr_batch, y_batch, batch_idx


class MessagePassingLayer(nn.Module):
    """Message passing layer for crystal graphs."""
    
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Edge network
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        Returns:
            Updated node features
        """
        src, dst = edge_index
        
        # Compute messages
        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.edge_mlp(edge_features)
        
        # Aggregate messages
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        
        # Update nodes
        node_features = torch.cat([x, aggregated], dim=-1)
        x_new = self.node_mlp(node_features)
        
        return x_new + x  # Residual connection


class CrystalEncoder(nn.Module):
    """Graph Neural Network encoder for crystal structures."""
    
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, batch_idx):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch_idx: Batch indices for each node [num_nodes]
        Returns:
            Graph-level representation [batch_size, hidden_dim]
        """
        # Embed features
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Message passing
        for mp_layer, layer_norm in zip(self.mp_layers, self.layer_norms):
            x_new = mp_layer(x, edge_index, edge_attr)
            x = layer_norm(x_new)
            x = self.dropout(x)
        
        # Global pooling (mean + max)
        batch_size = batch_idx.max().item() + 1
        pooled = torch.zeros(batch_size, self.hidden_dim * 2, device=x.device)
        
        for i in range(batch_size):
            mask = batch_idx == i
            if mask.any():
                x_i = x[mask]
                pooled[i, :self.hidden_dim] = x_i.mean(dim=0)
                pooled[i, self.hidden_dim:] = x_i.max(dim=0)[0]
        
        return pooled


class ContrastivePretrainer(nn.Module):
    """Self-supervised pretraining using contrastive learning."""
    
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(encoder.hidden_dim * 2, encoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder.hidden_dim, projection_dim)
        )
        
    def forward(self, x, edge_index, edge_attr, batch_idx):
        h = self.encoder(x, edge_index, edge_attr, batch_idx)
        z = F.normalize(self.projection(h), dim=1)
        return z
    
    def nt_xent_loss(self, z1, z2, temperature=0.5):
        """Normalized temperature-scaled cross entropy loss."""
        batch_size = z1.size(0)
        
        # Concatenate projections
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        # Positive pairs
        pos_sim = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ])
        
        # Negative pairs
        neg_sim = sim_matrix[~mask].view(2 * batch_size, -1)
        
        # Compute loss
        loss = -torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=1))).mean()
        
        return loss


class AltermagnetClassifier(nn.Module):
    """Classifier for altermagnetic material discovery."""
    
    def __init__(self, encoder, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder.hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, edge_attr, batch_idx):
        h = self.encoder(x, edge_index, edge_attr, batch_idx)
        logits = self.classifier(h)
        return logits.squeeze(-1)
    
    def predict_proba(self, x, edge_index, edge_attr, batch_idx):
        logits = self.forward(x, edge_index, edge_attr, batch_idx)
        return torch.sigmoid(logits)


def augment_graph(x, edge_index, edge_attr, aug_ratio=0.1):
    """Apply graph augmentation for contrastive learning."""
    # Node feature masking
    mask = torch.rand(x.size(0), x.size(1)) > aug_ratio
    x_aug = x * mask.float().to(x.device)
    
    # Edge dropout
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > aug_ratio
    keep_mask = keep_mask.to(edge_index.device)
    
    edge_index_aug = edge_index[:, keep_mask]
    edge_attr_aug = edge_attr[keep_mask] if edge_attr is not None else None
    
    return x_aug, edge_index_aug, edge_attr_aug


def pretrain_epoch(model, dataloader, optimizer, device, aug_ratio=0.1):
    """Train for one epoch with contrastive learning."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        x, edge_index, edge_attr, y, batch_idx = batch
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        batch_idx = batch_idx.to(device)
        
        # Create two augmented views
        x1, edge_index1, edge_attr1 = augment_graph(x, edge_index, edge_attr, aug_ratio)
        x2, edge_index2, edge_attr2 = augment_graph(x, edge_index, edge_attr, aug_ratio)
        
        # Forward pass
        z1 = model(x1, edge_index1, edge_attr1, batch_idx)
        z2 = model(x2, edge_index2, edge_attr2, batch_idx)
        
        # Compute contrastive loss
        loss = model.nt_xent_loss(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_classifier_epoch(model, dataloader, optimizer, criterion, device, class_weights=None):
    """Train classifier for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        x, edge_index, edge_attr, y, batch_idx = batch
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        y = y.to(device)
        batch_idx = batch_idx.to(device)
        
        # Forward pass
        logits = model(x, edge_index, edge_attr, batch_idx)
        
        # Compute loss with class weights
        if class_weights is not None:
            weights = class_weights[y.long()].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=weights)
        else:
            loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.sigmoid(logits) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(dataloader), accuracy


def evaluate_classifier(model, dataloader, device):
    """Evaluate classifier."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, edge_index, edge_attr, y, batch_idx = batch
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device) if edge_attr is not None else None
            batch_idx = batch_idx.to(device)
            
            probs = model.predict_proba(x, edge_index, edge_attr, batch_idx)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if y is not None:
                all_labels.extend(y.cpu().numpy())
    
    metrics = {}
    if len(all_labels) > 0:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics['accuracy'] = (all_preds == all_labels).mean()
        metrics['precision'] = precision_score(all_labels, all_preds, zero_division=0)
        metrics['recall'] = recall_score(all_labels, all_preds, zero_division=0)
        metrics['f1'] = f1_score(all_labels, all_preds, zero_division=0)
        metrics['auc'] = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        metrics['auprc'] = average_precision_score(all_labels, all_probs)
    
    return metrics, all_probs, all_labels


def main():
    """Main training pipeline."""
    print("="*60)
    print("Altermagnetic Materials Discovery with GNN")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    pretrain_dataset_raw = torch.load('data/pretrain_data.pt', weights_only=False)
    finetune_dataset_raw = torch.load('data/finetune_data.pt', weights_only=False)
    candidate_dataset_raw = torch.load('data/candidate_data.pt', weights_only=False)
    
    # Create datasets
    pretrain_dataset = CrystalDataset(pretrain_dataset_raw.data_list, has_labels=True)
    finetune_dataset = CrystalDataset(finetune_dataset_raw.data_list, has_labels=True)
    candidate_dataset = CrystalDataset(candidate_dataset_raw.data_list, has_labels=True)
    
    print(f"Pretrain dataset: {len(pretrain_dataset)} samples")
    print(f"Finetune dataset: {len(finetune_dataset)} samples")
    print(f"Candidate dataset: {len(candidate_dataset)} samples")
    
    # Split finetune into train/val
    train_size = int(0.8 * len(finetune_dataset))
    val_size = len(finetune_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        finetune_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    candidate_loader = DataLoader(
        candidate_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    
    # Initialize model
    print("\nInitializing model...")
    encoder = CrystalEncoder(node_dim=28, edge_dim=2, hidden_dim=128, num_layers=4).to(device)
    
    # Stage 1: Self-supervised pretraining
    print("\n" + "="*60)
    print("Stage 1: Self-Supervised Pretraining")
    print("="*60)
    
    pretrainer = ContrastivePretrainer(encoder, projection_dim=128).to(device)
    optimizer_pre = AdamW(pretrainer.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler_pre = CosineAnnealingLR(optimizer_pre, T_max=50)
    
    pretrain_losses = []
    num_pretrain_epochs = 30
    
    for epoch in range(num_pretrain_epochs):
        loss = pretrain_epoch(pretrainer, pretrain_loader, optimizer_pre, device)
        scheduler_pre.step()
        pretrain_losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_pretrain_epochs}, Loss: {loss:.4f}")
    
    # Save pretrained encoder
    os.makedirs('outputs', exist_ok=True)
    torch.save(encoder.state_dict(), 'outputs/pretrained_encoder.pt')
    print("Pretrained encoder saved to outputs/pretrained_encoder.pt")
    
    # Stage 2: Fine-tuning
    print("\n" + "="*60)
    print("Stage 2: Fine-tuning Classifier")
    print("="*60)
    
    classifier = AltermagnetClassifier(encoder, hidden_dim=128).to(device)
    optimizer_cls = AdamW(classifier.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler_cls = CosineAnnealingLR(optimizer_cls, T_max=100)
    criterion = nn.BCEWithLogitsLoss()
    
    # Compute class weights for imbalance
    train_labels = [finetune_dataset_raw.data_list[i].y.item() for i in train_dataset.indices]
    pos_weight = sum(train_labels) / len(train_labels)
    neg_weight = 1 - pos_weight
    class_weights = torch.tensor([neg_weight / pos_weight * 10])  # Upweight positive class
    
    print(f"Class weights: {class_weights.item():.4f}")
    
    best_val_f1 = 0
    train_losses = []
    val_metrics_history = []
    num_finetune_epochs = 100
    
    for epoch in range(num_finetune_epochs):
        train_loss, train_acc = train_classifier_epoch(
            classifier, train_loader, optimizer_cls, criterion, device, class_weights
        )
        scheduler_cls.step()
        train_losses.append(train_loss)
        
        # Validation
        val_metrics, _, _ = evaluate_classifier(classifier, val_loader, device)
        val_metrics_history.append(val_metrics)
        
        if val_metrics.get('f1', 0) > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(classifier.state_dict(), 'outputs/best_classifier.pt')
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_finetune_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Acc: {val_metrics.get('accuracy', 0):.4f}, "
                  f"Val F1: {val_metrics.get('f1', 0):.4f}, "
                  f"Val AUC: {val_metrics.get('auc', 0):.4f}")
    
    print(f"\nBest validation F1: {best_val_f1:.4f}")
    
    # Load best model
    classifier.load_state_dict(torch.load('outputs/best_classifier.pt'))
    
    # Stage 3: Candidate discovery
    print("\n" + "="*60)
    print("Stage 3: Candidate Discovery")
    print("="*60)
    
    classifier.eval()
    candidate_probs = []
    candidate_labels = []
    
    with torch.no_grad():
        for batch in candidate_loader:
            x, edge_index, edge_attr, y, batch_idx = batch
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device) if edge_attr is not None else None
            batch_idx = batch_idx.to(device)
            
            probs = classifier.predict_proba(x, edge_index, edge_attr, batch_idx)
            candidate_probs.extend(probs.cpu().numpy())
            if y is not None:
                candidate_labels.extend(y.cpu().numpy())
    
    candidate_probs = np.array(candidate_probs)
    
    # Rank candidates
    top_k = 50
    top_indices = np.argsort(candidate_probs)[::-1][:top_k]
    
    print(f"\nTop {top_k} candidate discoveries:")
    for i, idx in enumerate(top_indices[:10]):
        prob = candidate_probs[idx]
        print(f"  {i+1}. Candidate {idx}: {prob:.4f}")
    
    # Evaluation
    if len(candidate_labels) > 0:
        candidate_labels = np.array(candidate_labels)
        discovered = candidate_labels[top_indices]
        true_positives = discovered.sum()
        
        print(f"\nDiscovery Statistics:")
        print(f"  True positives in top {top_k}: {int(true_positives)}")
        print(f"  Precision @ {top_k}: {true_positives / top_k:.4f}")
        print(f"  Recall @ {top_k}: {true_positives / candidate_labels.sum():.4f}")
        
        # Overall metrics
        preds = (candidate_probs > 0.5).astype(int)
        print(f"\nOverall Candidate Evaluation:")
        print(f"  Accuracy: {(preds == candidate_labels).mean():.4f}")
        print(f"  Precision: {precision_score(candidate_labels, preds, zero_division=0):.4f}")
        print(f"  Recall: {recall_score(candidate_labels, preds, zero_division=0):.4f}")
        print(f"  F1: {f1_score(candidate_labels, preds, zero_division=0):.4f}")
        print(f"  AUC: {roc_auc_score(candidate_labels, candidate_probs):.4f}")
    
    # Save results
    results = {
        'candidate_probs': candidate_probs.tolist(),
        'candidate_labels': candidate_labels.tolist() if len(candidate_labels) > 0 else [],
        'top_indices': top_indices.tolist(),
        'train_losses': train_losses,
        'pretrain_losses': pretrain_losses,
        'val_metrics': val_metrics_history
    }
    
    with open('outputs/discovery_results.json', 'w') as f:
        json.dump(results, f)
    
    print("\nResults saved to outputs/discovery_results.json")
    
    return results


if __name__ == '__main__':
    results = main()
