import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader, Batch
import numpy as np
import json
import os

# ============================================================
# 1. Data Loading
# ============================================================

def load_dataset(path):
    """Load a RealisticCrystalDataset from .pt file."""
    dataset = torch.load(path, map_location='cpu', weights_only=False)
    return dataset

def get_data_list(dataset):
    """Return the list of Data objects."""
    return dataset.data_list

# ============================================================
# 2. GNN Model: Pre-training Encoder + Classification Head
# ============================================================

class GNNEncoder(nn.Module):
    """Graph Neural Network encoder with multiple GINEConv layers."""
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(nn_layer, edge_dim=hidden_dim, train_eps=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Embed nodes and edges
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        
        # GNN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=e)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res  # residual
        
        # Global pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        graph_repr = torch.cat([h_mean, h_max, h_sum], dim=-1)
        
        return graph_repr, h  # graph-level and node-level representations


class AltermagnetClassifier(nn.Module):
    """Full model: encoder + classification head."""
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.encoder = GNNEncoder(node_dim, edge_dim, hidden_dim, num_layers, dropout)
        repr_dim = hidden_dim * 3  # mean + max + sum pooling
        
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        graph_repr, node_repr = self.encoder(x, edge_index, edge_attr, batch)
        logits = self.classifier(graph_repr)
        return logits, graph_repr, node_repr


# ============================================================
# 3. Self-Supervised Pre-training (Graph Contrastive Learning)
# ============================================================

def augment_graph(data, drop_rate=0.1, noise_std=0.1):
    """Create an augmented view of a graph for contrastive learning."""
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()
    
    # Add Gaussian noise to node features
    x = x + torch.randn_like(x) * noise_std
    
    # Randomly drop edges
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_rate
    # Keep at least 1 edge
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    edge_index = edge_index[:, keep_mask]
    edge_attr = edge_attr[keep_mask]
    
    # Randomly mask node features
    mask = torch.rand(x.size(0)) > drop_rate
    x[~mask] = 0
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def pretrain_contrastive(encoder, dataset, epochs=100, batch_size=128, lr=1e-3, 
                         temperature=0.07, device='cpu', save_path=None):
    """Pre-train encoder with graph contrastive learning (SimCLR-style)."""
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Projection head for contrastive learning
    repr_dim = 128 * 3  # hidden_dim * 3
    proj_head = nn.Sequential(
        nn.Linear(repr_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)
    
    data_list = get_data_list(dataset)
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(epochs):
        encoder.train()
        proj_head.train()
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(data_list))
        
        for i in range(0, len(data_list), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Create two augmented views
            view1_list = [augment_graph(data_list[idx]) for idx in batch_indices]
            view2_list = [augment_graph(data_list[idx]) for idx in batch_indices]
            
            batch1 = Batch.from_data_list(view1_list).to(device)
            batch2 = Batch.from_data_list(view2_list).to(device)
            
            # Get representations
            repr1, _ = encoder(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
            repr2, _ = encoder(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
            
            # Project
            z1 = proj_head(repr1)
            z2 = proj_head(repr2)
            
            # Normalize
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            
            # NT-Xent loss
            batch_size_curr = z1.size(0)
            z = torch.cat([z1, z2], dim=0)  # [2N, dim]
            sim = torch.mm(z, z.t()) / temperature  # [2N, 2N]
            
            # Mask out self-similarity
            mask = torch.eye(2 * batch_size_curr, device=device).bool()
            sim.masked_fill_(mask, -1e9)
            
            # Positive pairs: (i, i+N) and (i+N, i)
            labels = torch.cat([torch.arange(batch_size_curr, device=device) + batch_size_curr,
                               torch.arange(batch_size_curr, device=device)], dim=0)
            
            loss = F.cross_entropy(sim, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                torch.save(encoder.state_dict(), save_path)
        
        if (epoch + 1) % 20 == 0:
            print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return loss_history


# ============================================================
# 4. Fine-tuning for Classification
# ============================================================

def finetune_classifier(model, dataset, epochs=150, batch_size=64, lr=1e-3,
                        pos_weight=10.0, device='cpu', save_path=None):
    """Fine-tune the classifier on labeled data."""
    model = model.to(device)
    
    data_list = get_data_list(dataset)
    labels = torch.tensor([d.y.item() for d in data_list], dtype=torch.float32)
    num_pos = labels.sum().item()
    num_neg = len(labels) - num_pos
    
    # Compute class weights
    weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device) * pos_weight
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0
    train_metrics = []
    
    # Train/val split (80/20)
    n = len(data_list)
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle training data
        np.random.shuffle(train_data)
        
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch = Batch.from_data_list(batch_data).to(device)
            
            logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            target = batch.y.float().view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(logits, target, 
                                                       pos_weight=weight)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy().flatten())
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_labels = []
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i+batch_size]
                batch = Batch.from_data_list(batch_data).to(device)
                logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
                val_preds.extend(preds)
                val_labels.extend(batch.y.cpu().numpy().flatten())
        
        from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
        
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_ap = average_precision_score(val_labels, val_preds) if sum(val_labels) > 0 else 0
        
        train_metrics.append({
            'epoch': epoch,
            'train_loss': epoch_loss / max(len(train_data) // batch_size, 1),
            'train_f1': train_f1,
            'val_f1': val_f1,
            'val_ap': val_ap,
        })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % 25 == 0:
            print(f"Finetune Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Val AP: {val_ap:.4f}")
    
    return train_metrics


# ============================================================
# 5. Inference on Candidate Data
# ============================================================

def predict_candidates(model, dataset, batch_size=64, device='cpu'):
    """Predict altermagnet probability for candidate materials."""
    model.eval()
    model = model.to(device)
    
    data_list = get_data_list(dataset)
    all_probs = []
    all_logits = []
    all_reprs = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i+batch_size]
            batch = Batch.from_data_list(batch_data).to(device)
            logits, graph_repr, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_reprs.extend(graph_repr.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy().flatten())
    
    return np.array(all_probs), np.array(all_logits), np.array(all_reprs), np.array(all_labels)


# ============================================================
# 6. Main Pipeline
# ============================================================

if __name__ == '__main__':
    device = 'cpu'
    
    # Load datasets
    print("Loading datasets...")
    pretrain_dataset = load_dataset('data/pretrain_data.pt')
    finetune_dataset = load_dataset('data/finetune_data.pt')
    candidate_dataset = load_dataset('data/candidate_data.pt')
    
    print(f"Pretrain: {len(pretrain_dataset)} samples")
    print(f"Finetune: {len(finetune_dataset)} samples")
    print(f"Candidate: {len(candidate_dataset)} samples")
    
    # Create model
    model = AltermagnetClassifier(
        node_dim=28, edge_dim=2, hidden_dim=128, num_layers=4, dropout=0.1
    )
    
    # Step 1: Pre-training
    print("\n=== Step 1: Self-supervised pre-training ===")
    pretrain_loss = pretrain_contrastive(
        model.encoder, pretrain_dataset, 
        epochs=80, batch_size=128, lr=1e-3,
        device=device, save_path='outputs/pretrained_encoder.pt'
    )
    
    # Save pretrain loss history
    np.save('outputs/pretrain_loss_history.npy', pretrain_loss)
    
    # Step 2: Fine-tuning
    print("\n=== Step 2: Fine-tuning for classification ===")
    # Reload best pre-trained encoder
    model.encoder.load_state_dict(torch.load('outputs/pretrained_encoder.pt', map_location=device))
    
    finetune_metrics = finetune_classifier(
        model, finetune_dataset,
        epochs=150, batch_size=64, lr=5e-4,
        pos_weight=10.0, device=device, save_path='outputs/best_model.pt'
    )
    
    # Save finetune metrics
    with open('outputs/finetune_metrics.json', 'w') as f:
        json.dump(finetune_metrics, f, indent=2)
    
    # Step 3: Prediction on candidates
    print("\n=== Step 3: Predicting candidates ===")
    model.load_state_dict(torch.load('outputs/best_model.pt', map_location=device))
    
    probs, logits, reprs, true_labels = predict_candidates(
        model, candidate_dataset, device=device
    )
    
    # Save results
    np.save('outputs/candidate_probs.npy', probs)
    np.save('outputs/candidate_logits.npy', logits)
    np.save('outputs/candidate_reprs.npy', reprs)
    np.save('outputs/candidate_true_labels.npy', true_labels)
    
    # Compute evaluation metrics
    from sklearn.metrics import (roc_auc_score, average_precision_score, 
                                  f1_score, precision_score, recall_score,
                                  classification_report, confusion_matrix)
    
    pred_labels = (probs > 0.5).astype(int)
    
    auc = roc_auc_score(true_labels, probs)
    ap = average_precision_score(true_labels, probs)
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    
    print(f"\n=== Candidate Evaluation ===")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(true_labels, pred_labels)}")
    print(f"\nClassification Report:\n{classification_report(true_labels, pred_labels)}")
    
    # Save evaluation results
    eval_results = {
        'auc_roc': float(auc),
        'average_precision': float(ap),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'num_true_positives': int(true_labels.sum()),
        'num_predicted_positives': int(pred_labels.sum()),
        'num_candidates': len(true_labels),
    }
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Top candidates
    top_k = 50
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_results = []
    for rank, idx in enumerate(top_indices):
        top_results.append({
            'rank': rank + 1,
            'index': int(idx),
            'probability': float(probs[idx]),
            'true_label': int(true_labels[idx]),
            'is_correct': bool(pred_labels[idx] == true_labels[idx]),
        })
    
    with open('outputs/top_candidates.json', 'w') as f:
        json.dump(top_results, f, indent=2)
    
    # Discovery metrics at different thresholds
    discovery_metrics = {}
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_at_t = (probs > threshold).astype(int)
        tp = ((pred_at_t == 1) & (true_labels == 1)).sum()
        fp = ((pred_at_t == 1) & (true_labels == 0)).sum()
        fn = ((pred_at_t == 0) & (true_labels == 1)).sum()
        tn = ((pred_at_t == 0) & (true_labels == 0)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        discovery_metrics[f'threshold_{threshold}'] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'precision': float(prec),
            'recall': float(rec),
        }
    
    with open('outputs/discovery_metrics.json', 'w') as f:
        json.dump(discovery_metrics, f, indent=2)
    
    print("\nDone! All results saved to outputs/")
