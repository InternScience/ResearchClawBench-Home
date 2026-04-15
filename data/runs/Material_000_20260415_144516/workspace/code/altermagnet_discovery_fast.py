"""
AI-Powered Search Engine for Altermagnetic Materials Discovery
Optimized version with reduced complexity for faster execution.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)
import json
from collections import defaultdict

# Add parent directory to path for data_prepare module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_prepare

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CrystalEncoder(nn.Module):
    """Graph Neural Network encoder with message passing."""
    
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=64, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial node embedding
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        
        # Message passing layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
            batch: [num_nodes] - batch assignment
        Returns:
            Graph-level embeddings [batch_size, hidden_dim*2]
        """
        x = F.relu(self.node_embed(x))
        edge_attr = F.relu(self.edge_embed(edge_attr))
        
        # Message passing
        for conv, bn in zip(self.convs, self.batch_norms):
            src, dst = edge_index
            # Aggregate messages
            messages = torch.cat([x[src], x[dst], edge_attr], dim=-1)
            messages = F.relu(conv(messages))
            
            # Aggregate to destination nodes
            out = torch.zeros_like(x)
            out.index_add_(0, dst, messages)
            x = bn(x + out)  # Residual + BatchNorm
        
        # Global pooling
        num_graphs = batch.max().item() + 1
        output = torch.zeros(num_graphs, self.hidden_dim * 2, device=x.device)
        
        for i in range(num_graphs):
            mask = batch == i
            if mask.any():
                x_i = x[mask]
                output[i, :self.hidden_dim] = x_i.mean(dim=0)
                output[i, self.hidden_dim:] = x_i.max(dim=0)[0]
        
        return output


class Pretrainer(nn.Module):
    """Self-supervised pretraining with property prediction."""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Linear(encoder.hidden_dim * 2, 1)
    
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr, batch)
        return self.predictor(h).squeeze()


class Classifier(nn.Module):
    """Altermagnet classifier."""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(encoder.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr, batch)
        return self.fc(h).squeeze()
    
    def predict_proba(self, x, edge_index, edge_attr, batch):
        logits = self.forward(x, edge_index, edge_attr, batch)
        return torch.sigmoid(logits)


def create_batch(data_list, device):
    """Create a batch from list of Data objects."""
    x_list, edge_list, edge_attr_list, y_list = [], [], [], []
    batch_idx = []
    node_offset = 0
    
    for i, data in enumerate(data_list):
        x_list.append(data.x.float())
        edge_list.append(data.edge_index + node_offset)
        if data.edge_attr is not None:
            edge_attr_list.append(data.edge_attr.float())
        if hasattr(data, 'y') and data.y is not None:
            y_list.append(data.y.float())
        batch_idx.extend([i] * data.num_nodes)
        node_offset += data.num_nodes
    
    x = torch.cat(x_list, dim=0).to(device)
    edge_index = torch.cat(edge_list, dim=1).to(device)
    edge_attr = torch.cat(edge_attr_list, dim=0).to(device) if edge_attr_list else None
    batch = torch.tensor(batch_idx, dtype=torch.long).to(device)
    y = torch.stack(y_list).to(device) if y_list else None
    
    return x, edge_index, edge_attr, batch, y


def pretrain(model, data_list, epochs=20, lr=1e-3, batch_size=32):
    """Self-supervised pretraining using graph property prediction (node count)."""
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()
    
    # Create target: predict number of nodes (structural property)
    targets = torch.tensor([d.num_nodes for d in data_list], dtype=torch.float).to(device)
    
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        indices = np.random.permutation(len(data_list))
        
        for i in range(0, len(data_list), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_data = [data_list[j] for j in batch_idx]
            
            x, edge_index, edge_attr, batch, _ = create_batch(batch_data, device)
            pred = model(x, edge_index, edge_attr, batch)
            target = targets[batch_idx]
            
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(data_list) // batch_size)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  Pretrain Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def finetune(model, train_data, val_data, epochs=50, lr=5e-4, batch_size=16, pos_weight=10.0):
    """Fine-tune classifier."""
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    train_losses = []
    val_metrics = []
    best_f1 = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        indices = np.random.permutation(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_data = [train_data[j] for j in batch_idx]
            
            x, edge_index, edge_attr, batch, y = create_batch(batch_data, device)
            logits = model(x, edge_index, edge_attr, batch)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / (len(train_data) // batch_size)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_probs, val_labels = [], []
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i+batch_size]
                x, edge_index, edge_attr, batch, y = create_batch(batch_data, device)
                probs = model.predict_proba(x, edge_index, edge_attr, batch)
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        val_preds = (val_probs > 0.5).astype(int)
        
        metrics = {
            'accuracy': (val_preds == val_labels).mean(),
            'precision': precision_score(val_labels, val_preds, zero_division=0),
            'recall': recall_score(val_labels, val_preds, zero_division=0),
            'f1': f1_score(val_labels, val_preds, zero_division=0),
            'auc': roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5,
            'auprc': average_precision_score(val_labels, val_probs)
        }
        val_metrics.append(metrics)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'outputs/best_model.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"  Finetune Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, "
                  f"Val F1: {metrics['f1']:.4f}, Val AUC: {metrics['auc']:.4f}")
    
    return train_losses, val_metrics, best_f1


def main():
    """Main pipeline."""
    print("="*60)
    print("Altermagnetic Materials Discovery")
    print("="*60)
    
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    pretrain_raw = torch.load('data/pretrain_data.pt', weights_only=False)
    finetune_raw = torch.load('data/finetune_data.pt', weights_only=False)
    candidate_raw = torch.load('data/candidate_data.pt', weights_only=False)
    
    pretrain_data = pretrain_raw.data_list
    finetune_data = finetune_raw.data_list
    candidate_data = candidate_raw.data_list
    
    print(f"   Pretrain: {len(pretrain_data)} samples")
    print(f"   Finetune: {len(finetune_data)} samples")
    print(f"   Candidate: {len(candidate_data)} samples")
    
    # Count labels
    finetune_labels = [d.y.item() for d in finetune_data]
    print(f"   Finetune positives: {sum(finetune_labels)}/{len(finetune_labels)} ({sum(finetune_labels)/len(finetune_labels)*100:.1f}%)")
    
    candidate_labels = [d.y.item() for d in candidate_data]
    print(f"   Candidate positives: {sum(candidate_labels)}/{len(candidate_labels)} ({sum(candidate_labels)/len(candidate_labels)*100:.1f}%)")
    
    # Split finetune data
    np.random.seed(42)
    indices = np.random.permutation(len(finetune_data))
    split = int(0.8 * len(finetune_data))
    train_idx, val_idx = indices[:split], indices[split:]
    train_data = [finetune_data[i] for i in train_idx]
    val_data = [finetune_data[i] for i in val_idx]
    
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize model
    print("\n2. Initializing model...")
    encoder = CrystalEncoder(node_dim=28, edge_dim=2, hidden_dim=64, num_layers=3).to(device)
    
    # Stage 1: Pretraining
    print("\n3. Self-supervised pretraining...")
    pretrainer = Pretrainer(encoder).to(device)
    pretrain_losses = pretrain(pretrainer, pretrain_data, epochs=20, lr=1e-3, batch_size=32)
    
    # Stage 2: Fine-tuning
    print("\n4. Fine-tuning classifier...")
    classifier = Classifier(encoder).to(device)
    
    # Calculate class weight
    train_labels = [d.y.item() for d in train_data]
    pos_ratio = sum(train_labels) / len(train_labels)
    pos_weight = (1 - pos_ratio) / pos_ratio * 5  # Upweight positive class
    print(f"   Using pos_weight: {pos_weight:.2f}")
    
    train_losses, val_metrics, best_f1 = finetune(
        classifier, train_data, val_data, 
        epochs=50, lr=5e-4, batch_size=16, pos_weight=pos_weight
    )
    
    print(f"\n   Best validation F1: {best_f1:.4f}")
    
    # Load best model
    classifier.load_state_dict(torch.load('outputs/best_model.pt'))
    
    # Stage 3: Discovery
    print("\n5. Candidate discovery...")
    classifier.eval()
    
    candidate_probs = []
    with torch.no_grad():
        for i in range(0, len(candidate_data), 16):
            batch_data = candidate_data[i:i+16]
            x, edge_index, edge_attr, batch, _ = create_batch(batch_data, device)
            probs = classifier.predict_proba(x, edge_index, edge_attr, batch)
            candidate_probs.extend(probs.cpu().numpy())
    
    candidate_probs = np.array(candidate_probs)
    
    # Rank candidates
    top_k = 50
    top_indices = np.argsort(candidate_probs)[::-1][:top_k]
    
    print(f"\n   Top {top_k} discoveries:")
    for i, idx in enumerate(top_indices[:10]):
        true_label = candidate_labels[idx]
        print(f"   {i+1}. Candidate {idx}: prob={candidate_probs[idx]:.4f}, true_label={int(true_label)}")
    
    # Evaluate discovery
    discovered_labels = [candidate_labels[i] for i in top_indices]
    true_positives = sum(discovered_labels)
    
    print(f"\n   Discovery Statistics:")
    print(f"   - True positives in top {top_k}: {true_positives}")
    print(f"   - Precision @ {top_k}: {true_positives/top_k:.4f}")
    print(f"   - Recall @ {top_k}: {true_positives/sum(candidate_labels):.4f}")
    
    # Overall metrics
    predictions = (candidate_probs > 0.5).astype(int)
    print(f"\n   Overall Performance:")
    print(f"   - Accuracy: {(predictions == np.array(candidate_labels)).mean():.4f}")
    print(f"   - Precision: {precision_score(candidate_labels, predictions, zero_division=0):.4f}")
    print(f"   - Recall: {recall_score(candidate_labels, predictions, zero_division=0):.4f}")
    print(f"   - F1: {f1_score(candidate_labels, predictions, zero_division=0):.4f}")
    print(f"   - AUC: {roc_auc_score(candidate_labels, candidate_probs):.4f}")
    
    # Save results
    results = {
        'pretrain_losses': pretrain_losses,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'candidate_probs': candidate_probs.tolist(),
        'candidate_labels': candidate_labels,
        'top_indices': top_indices.tolist(),
        'true_positives_at_k': int(true_positives),
        'precision_at_k': true_positives/top_k,
        'recall_at_k': true_positives/sum(candidate_labels),
        'overall_auc': roc_auc_score(candidate_labels, candidate_probs),
        'overall_f1': f1_score(candidate_labels, predictions, zero_division=0)
    }
    
    with open('outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n   Results saved to outputs/results.json")
    
    return results


if __name__ == '__main__':
    results = main()
