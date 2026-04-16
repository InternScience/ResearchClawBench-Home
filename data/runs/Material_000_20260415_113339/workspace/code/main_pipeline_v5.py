import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool, Set2Set
from torch_geometric.data import Data, Batch
import numpy as np
import json
import os
import pickle
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                              f1_score, precision_score, recall_score,
                              confusion_matrix)

# ============================================================
# 1. Data Loading
# ============================================================

def load_dataset(path):
    dataset = torch.load(path, map_location='cpu', weights_only=False)
    return dataset

def get_data_list(dataset):
    return dataset.data_list

# ============================================================
# 2. Enhanced GNN Model
# ============================================================

class CrystalGNN(nn.Module):
    """Enhanced GNN with attention-based pooling and multi-scale representations."""
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=64, num_layers=5, dropout=0.3, heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node embedding with positional encoding
        self.node_emb = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Edge embedding
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # GNN layers with different architectures
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.skip_lins = nn.ModuleList()
        
        for i in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            conv = GINEConv(nn_layer, edge_dim=hidden_dim, train_eps=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            self.skip_lins.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        
        # Multi-layer GNN with skip connections
        for conv, norm, drop, skip_lin in zip(self.convs, self.norms, self.dropouts, self.skip_lins):
            h_res = skip_lin(h)
            h = conv(h, edge_index, edge_attr=e)
            h = norm(h)
            h = F.gelu(h)
            h = drop(h)
            h = h + h_res
        
        # Multi-pooling strategy
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        
        # Attention-weighted pooling
        attn_weights = self.attn_pool(h)  # [num_nodes, 1]
        attn_weights = torch.softmax(attn_weights, dim=0)  # normalize per graph handled below
        # Need to normalize per graph
        attn_scores = attn_weights.squeeze(-1)
        # Manual scatter for per-graph softmax
        batch_sizes = torch.bincount(batch)
        attn_norm = torch.zeros_like(attn_scores)
        for b in range(batch_sizes.size(0)):
            mask = (batch == b)
            if mask.sum() > 0:
                attn_norm[mask] = F.softmax(attn_scores[mask], dim=0)
        h_attn = global_add_pool(h * attn_norm.unsqueeze(-1), batch)
        
        graph_repr = torch.cat([h_mean, h_max, h_sum, h_attn], dim=-1)
        logits = self.classifier(graph_repr)
        
        return logits, graph_repr, h


# ============================================================
# 3. Self-Supervised Pre-training (Multi-task)
# ============================================================

def augment_graph(data, drop_rate=0.2, noise_std=0.2, feature_mask_rate=0.3):
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()
    
    # Gaussian noise
    x = x + torch.randn_like(x) * noise_std
    
    # Edge dropping
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_rate
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    edge_index = edge_index[:, keep_mask]
    edge_attr = edge_attr[keep_mask]
    
    # Feature masking
    node_mask = torch.rand(x.size(0)) > feature_mask_rate
    x[~node_mask] = 0
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def save_model_safe(state_dict, path):
    numpy_state = {}
    for k, v in state_dict.items():
        numpy_state[k] = v.cpu().numpy()
    with open(path, 'wb') as f:
        pickle.dump(numpy_state, f)


def load_model_safe(path):
    with open(path, 'rb') as f:
        numpy_state = pickle.load(f)
    state_dict = {}
    for k, v in numpy_state.items():
        state_dict[k] = torch.from_numpy(v)
    return state_dict


def pretrain_multitask(model, dataset, epochs=100, batch_size=128, lr=1e-3,
                       temperature=0.07, device='cpu', save_path=None):
    """Multi-task pre-training: contrastive + node reconstruction + graph property prediction."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)
    
    hidden_dim = 64
    repr_dim = hidden_dim * 4
    
    # Projection head for contrastive learning
    proj_head = nn.Sequential(
        nn.Linear(repr_dim, hidden_dim * 2),
        nn.LayerNorm(hidden_dim * 2),
        nn.GELU(),
        nn.Linear(hidden_dim * 2, 128),
    ).to(device)
    
    # Node feature reconstruction head
    node_recon_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 28),
    ).to(device)
    
    # Graph property prediction head (num_nodes, num_edges)
    graph_prop_head = nn.Sequential(
        nn.Linear(repr_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 2),
    ).to(device)
    
    data_list = get_data_list(dataset)
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        proj_head.train()
        node_recon_head.train()
        graph_prop_head.train()
        epoch_loss = 0
        num_batches = 0
        
        indices = torch.randperm(len(data_list))
        
        for i in range(0, len(data_list), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            view1_list = [augment_graph(data_list[idx]) for idx in batch_indices]
            view2_list = [augment_graph(data_list[idx]) for idx in batch_indices]
            orig_list = [data_list[idx] for idx in batch_indices]
            
            batch1 = Batch.from_data_list(view1_list).to(device)
            batch2 = Batch.from_data_list(view2_list).to(device)
            batch_orig = Batch.from_data_list(orig_list).to(device)
            
            # Forward pass
            _, repr1, node_repr1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
            _, repr2, node_repr2 = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
            
            # 1. Contrastive loss
            z1 = proj_head(repr1)
            z2 = proj_head(repr2)
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            
            bs = z1.size(0)
            z = torch.cat([z1, z2], dim=0)
            sim = torch.mm(z, z.t()) / temperature
            mask = torch.eye(2 * bs, device=device).bool()
            sim.masked_fill_(mask, -1e9)
            labels = torch.cat([torch.arange(bs, device=device) + bs,
                               torch.arange(bs, device=device)], dim=0)
            contrastive_loss = F.cross_entropy(sim, labels)
            
            # 2. Node reconstruction loss
            recon_x = node_recon_head(node_repr1)
            recon_loss = F.mse_loss(recon_x, batch_orig.x)
            
            # 3. Graph property prediction loss
            graph_props = graph_prop_head(repr1)
            num_nodes_per_graph = torch.bincount(batch1.batch).float().unsqueeze(-1)
            num_edges_per_graph = torch.bincount(batch1.batch[edge_index_to_batch(batch1.edge_index, batch1.batch)]).float().unsqueeze(-1) if batch1.edge_index.size(1) > 0 else torch.zeros(bs, 1, device=device)
            # Simplified: just predict num_nodes
            target_props = torch.cat([
                torch.bincount(batch1.batch).float().unsqueeze(-1),
                torch.zeros(bs, 1, device=device),  # placeholder for edges
            ], dim=-1)
            prop_loss = F.mse_loss(graph_props, target_props)
            
            # Combined loss
            loss = contrastive_loss + 0.5 * recon_loss + 0.1 * prop_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + 
                                           list(proj_head.parameters()) +
                                           list(node_recon_head.parameters()) +
                                           list(graph_prop_head.parameters()), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                save_model_safe(model.state_dict(), save_path)
        
        if (epoch + 1) % 20 == 0:
            print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return loss_history


def edge_index_to_batch(edge_index, node_batch):
    """Map edge_index to batch indices."""
    return node_batch[edge_index[0]]


# ============================================================
# 4. Fine-tuning with Advanced Strategies
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def finetune_classifier(model, dataset, epochs=300, batch_size=32, lr=1e-4,
                        device='cpu', save_path=None):
    model = model.to(device)
    
    data_list = get_data_list(dataset)
    labels = np.array([d.y.item() for d in data_list])
    
    # Stratified k-fold cross validation for better evaluation
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    # Use more positives for training - 90/10 split
    train_pos = pos_idx[:int(0.9 * len(pos_idx))]
    val_pos = pos_idx[int(0.9 * len(pos_idx)):]
    train_neg = neg_idx[:int(0.9 * len(neg_idx))]
    val_neg = neg_idx[int(0.9 * len(neg_idx)):]
    
    train_idx = np.concatenate([train_pos, train_neg])
    val_idx = np.concatenate([val_pos, val_neg])
    
    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    
    print(f"Train: {len(train_pos)} pos, {len(train_neg)} neg")
    print(f"Val: {len(val_pos)} pos, {len(val_neg)} neg")
    
    # Multiple loss functions
    focal_criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    optimizer = torch.optim.AdamW([
        {'params': model.convs.parameters(), 'lr': lr * 0.1},  # Lower LR for GNN layers
        {'params': model.classifier.parameters(), 'lr': lr},  # Higher LR for classifier
        {'params': model.node_emb.parameters(), 'lr': lr * 0.5},
        {'params': model.edge_emb.parameters(), 'lr': lr * 0.5},
        {'params': model.attn_pool.parameters(), 'lr': lr},
        {'params': model.skip_lins.parameters(), 'lr': lr * 0.1},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    best_f1 = 0
    best_ap = 0
    best_checkpoint_state = None
    best_threshold = 0.5
    train_metrics = []
    
    for epoch in range(epochs):
        model.train()
        
        # Aggressive oversampling of positive examples
        num_pos = len(train_pos)
        num_neg = len(train_neg)
        oversample_ratio = min(num_neg // num_pos, 30)
        
        train_epoch_data = list(train_data)
        pos_data = [data_list[i] for i in train_pos]
        for _ in range(oversample_ratio - 1):
            train_epoch_data.extend(pos_data)
        
        np.random.shuffle(train_epoch_data)
        
        epoch_loss = 0
        all_preds = []
        all_labels_list = []
        
        for i in range(0, len(train_epoch_data), batch_size):
            batch_data = train_epoch_data[i:i+batch_size]
            batch = Batch.from_data_list(batch_data).to(device)
            
            logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.float().view(-1, 1)
            
            loss = focal_criterion(logits, target)
            
            # Add label smoothing
            smooth_target = target * 0.9 + 0.05
            bce_loss = F.binary_cross_entropy_with_logits(logits, smooth_target)
            loss = loss + 0.3 * bce_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels_list.extend(target.cpu().numpy().flatten())
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = []
            val_labels = []
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i+batch_size]
                batch = Batch.from_data_list(batch_data).to(device)
                logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs)
                val_labels.extend(batch.y.cpu().numpy().flatten())
        
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        
        # Find optimal threshold
        bt = 0.5
        bvf1 = 0
        for thresh in np.arange(0.05, 0.95, 0.05):
            val_preds_t = (val_probs > thresh).astype(int)
            val_f1_t = f1_score(val_labels, val_preds_t, zero_division=0)
            if val_f1_t > bvf1:
                bvf1 = val_f1_t
                bt = thresh
        
        val_preds = (val_probs > bt).astype(int)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_ap = average_precision_score(val_labels, val_probs) if sum(val_labels) > 0 else 0
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        
        train_f1 = f1_score(all_labels_list, all_preds, zero_division=0)
        
        train_metrics.append({
            'epoch': epoch,
            'train_loss': epoch_loss / max(len(train_epoch_data) // batch_size, 1),
            'train_f1': float(train_f1),
            'val_f1': float(val_f1),
            'val_ap': float(val_ap),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'best_threshold': float(bt),
        })
        
        if val_f1 > best_f1 or (val_f1 == best_f1 and val_ap > best_ap):
            best_f1 = val_f1
            best_ap = val_ap
            best_threshold = bt
            best_checkpoint_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            print(f"Finetune Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f} (thresh={bt:.2f}), "
                  f"Val AP: {val_ap:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}")
    
    # Save best model
    if save_path and best_checkpoint_state is not None:
        numpy_state = {}
        for k, v in best_checkpoint_state.items():
            numpy_state[k] = v.numpy()
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model_state_dict': numpy_state,
                'best_threshold': float(best_threshold),
                'val_f1': float(best_f1),
                'val_ap': float(best_ap),
            }, f)
    
    return train_metrics


# ============================================================
# 5. Inference
# ============================================================

def predict_candidates(model, dataset, batch_size=64, device='cpu'):
    model.eval()
    model = model.to(device)
    
    data_list = get_data_list(dataset)
    all_probs = []
    all_reprs = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i+batch_size]
            batch = Batch.from_data_list(batch_data).to(device)
            logits, graph_repr, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_reprs.extend(graph_repr.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy().flatten())
    
    return np.array(all_probs), np.array(all_reprs), np.array(all_labels)


# ============================================================
# 6. Baseline: Train from scratch (no pre-training)
# ============================================================

def train_from_scratch(model, dataset, epochs=300, batch_size=32, lr=2e-4,
                       device='cpu', save_path=None):
    """Train model from scratch for comparison."""
    model = model.to(device)
    
    data_list = get_data_list(dataset)
    labels = np.array([d.y.item() for d in data_list])
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    train_pos = pos_idx[:int(0.9 * len(pos_idx))]
    val_pos = pos_idx[int(0.9 * len(pos_idx)):]
    train_neg = neg_idx[:int(0.9 * len(neg_idx))]
    val_neg = neg_idx[int(0.9 * len(neg_idx)):]
    
    train_data = [data_list[i] for i in np.concatenate([train_pos, train_neg])]
    val_data = [data_list[i] for i in np.concatenate([val_pos, val_neg])]
    
    print(f"[Scratch] Train: {len(train_pos)} pos, {len(train_neg)} neg")
    print(f"[Scratch] Val: {len(val_pos)} pos, {len(val_neg)} neg")
    
    focal_criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    best_f1 = 0
    best_ap = 0
    best_checkpoint_state = None
    best_threshold = 0.5
    metrics = []
    
    for epoch in range(epochs):
        model.train()
        
        num_pos = len(train_pos)
        num_neg = len(train_neg)
        oversample_ratio = min(num_neg // num_pos, 30)
        
        train_epoch_data = list(train_data)
        pos_data = [data_list[i] for i in train_pos]
        for _ in range(oversample_ratio - 1):
            train_epoch_data.extend(pos_data)
        
        np.random.shuffle(train_epoch_data)
        
        epoch_loss = 0
        for i in range(0, len(train_epoch_data), batch_size):
            batch_data = train_epoch_data[i:i+batch_size]
            batch = Batch.from_data_list(batch_data).to(device)
            
            logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.float().view(-1, 1)
            
            loss = focal_criterion(logits, target)
            smooth_target = target * 0.9 + 0.05
            bce_loss = F.binary_cross_entropy_with_logits(logits, smooth_target)
            loss = loss + 0.3 * bce_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = []
            val_labels = []
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i+batch_size]
                batch = Batch.from_data_list(batch_data).to(device)
                logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs)
                val_labels.extend(batch.y.cpu().numpy().flatten())
        
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        
        bt = 0.5
        bvf1 = 0
        for thresh in np.arange(0.05, 0.95, 0.05):
            val_preds_t = (val_probs > thresh).astype(int)
            val_f1_t = f1_score(val_labels, val_preds_t, zero_division=0)
            if val_f1_t > bvf1:
                bvf1 = val_f1_t
                bt = thresh
        
        val_preds = (val_probs > bt).astype(int)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_ap = average_precision_score(val_labels, val_probs) if sum(val_labels) > 0 else 0
        
        metrics.append({'epoch': epoch, 'val_f1': float(val_f1), 'val_ap': float(val_ap), 'best_threshold': float(bt)})
        
        if val_f1 > best_f1 or (val_f1 == best_f1 and val_ap > best_ap):
            best_f1 = val_f1
            best_ap = val_ap
            best_threshold = bt
            best_checkpoint_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 50 == 0:
            print(f"[Scratch] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, "
                  f"Val F1: {val_f1:.4f} (thresh={bt:.2f}), Val AP: {val_ap:.4f}")
    
    if save_path and best_checkpoint_state is not None:
        numpy_state = {}
        for k, v in best_checkpoint_state.items():
            numpy_state[k] = v.numpy()
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model_state_dict': numpy_state,
                'best_threshold': float(best_threshold),
                'val_f1': float(best_f1),
                'val_ap': float(best_ap),
            }, f)
    
    return metrics


# ============================================================
# 7. Main Pipeline
# ============================================================

if __name__ == '__main__':
    device = 'cpu'
    
    print("Loading datasets...")
    pretrain_dataset = load_dataset('data/pretrain_data.pt')
    finetune_dataset = load_dataset('data/finetune_data.pt')
    candidate_dataset = load_dataset('data/candidate_data.pt')
    
    print(f"Pretrain: {len(pretrain_dataset)} samples")
    print(f"Finetune: {len(finetune_dataset)} samples")
    print(f"Candidate: {len(candidate_dataset)} samples")
    
    # ===== Model with pre-training =====
    model_pretrained = CrystalGNN(
        node_dim=28, edge_dim=2, hidden_dim=64, num_layers=5, dropout=0.3
    )
    
    # Step 1: Pre-training
    print("\n=== Step 1: Self-supervised pre-training ===")
    pretrain_loss = pretrain_multitask(
        model_pretrained, pretrain_dataset, 
        epochs=100, batch_size=128, lr=1e-3,
        device=device, save_path='outputs/pretrained_encoder.pkl'
    )
    
    # Step 2: Fine-tuning
    print("\n=== Step 2: Fine-tuning (pre-trained) ===")
    model_pretrained.load_state_dict(load_model_safe('outputs/pretrained_encoder.pkl'))
    
    finetune_metrics = finetune_classifier(
        model_pretrained, finetune_dataset,
        epochs=300, batch_size=32, lr=1e-4,
        device=device, save_path='outputs/best_model_pretrained.pkl'
    )
    
    # ===== Model without pre-training (baseline) =====
    print("\n=== Step 2b: Training from scratch (baseline) ===")
    model_scratch = CrystalGNN(
        node_dim=28, edge_dim=2, hidden_dim=64, num_layers=5, dropout=0.3
    )
    
    scratch_metrics = train_from_scratch(
        model_scratch, finetune_dataset,
        epochs=300, batch_size=32, lr=2e-4,
        device=device, save_path='outputs/best_model_scratch.pkl'
    )
    
    # ===== Step 3: Prediction on candidates =====
    print("\n=== Step 3: Predicting candidates ===")
    
    # Load pretrained model
    with open('outputs/best_model_pretrained.pkl', 'rb') as f:
        checkpoint_pre = pickle.load(f)
    state_dict_pre = {k: torch.from_numpy(v) for k, v in checkpoint_pre['model_state_dict'].items()}
    model_pretrained.load_state_dict(state_dict_pre)
    threshold_pre = checkpoint_pre.get('best_threshold', 0.5)
    
    # Load scratch model
    with open('outputs/best_model_scratch.pkl', 'rb') as f:
        checkpoint_scr = pickle.load(f)
    state_dict_scr = {k: torch.from_numpy(v) for k, v in checkpoint_scr['model_state_dict'].items()}
    model_scratch.load_state_dict(state_dict_scr)
    threshold_scr = checkpoint_scr.get('best_threshold', 0.5)
    
    # Predict with pretrained model
    probs_pre, reprs_pre, true_labels = predict_candidates(model_pretrained, candidate_dataset, device=device)
    
    # Predict with scratch model
    probs_scr, reprs_scr, _ = predict_candidates(model_scratch, candidate_dataset, device=device)
    
    # Save results
    np.save('outputs/candidate_probs_pretrained.npy', probs_pre)
    np.save('outputs/candidate_probs_scratch.npy', probs_scr)
    np.save('outputs/candidate_true_labels.npy', true_labels)
    
    # Evaluate pretrained model
    pred_labels_pre = (probs_pre > threshold_pre).astype(int)
    auc_pre = roc_auc_score(true_labels, probs_pre)
    ap_pre = average_precision_score(true_labels, probs_pre)
    f1_pre = f1_score(true_labels, pred_labels_pre)
    prec_pre = precision_score(true_labels, pred_labels_pre, zero_division=0)
    rec_pre = recall_score(true_labels, pred_labels_pre, zero_division=0)
    
    print(f"\n=== Pretrained Model (threshold={threshold_pre:.3f}) ===")
    print(f"AUC-ROC: {auc_pre:.4f}, AP: {ap_pre:.4f}, F1: {f1_pre:.4f}, Prec: {prec_pre:.4f}, Rec: {rec_pre:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(true_labels, pred_labels_pre)}")
    
    # Evaluate scratch model
    pred_labels_scr = (probs_scr > threshold_scr).astype(int)
    auc_scr = roc_auc_score(true_labels, probs_scr)
    ap_scr = average_precision_score(true_labels, probs_scr)
    f1_scr = f1_score(true_labels, pred_labels_scr)
    prec_scr = precision_score(true_labels, pred_labels_scr, zero_division=0)
    rec_scr = recall_score(true_labels, pred_labels_scr, zero_division=0)
    
    print(f"\n=== Scratch Model (threshold={threshold_scr:.3f}) ===")
    print(f"AUC-ROC: {auc_scr:.4f}, AP: {ap_scr:.4f}, F1: {f1_scr:.4f}, Prec: {prec_scr:.4f}, Rec: {rec_scr:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(true_labels, pred_labels_scr)}")
    
    # Also evaluate at 0.5 threshold
    for name, probs, threshold in [("Pretrained", probs_pre, 0.5), ("Scratch", probs_scr, 0.5)]:
        pred = (probs > threshold).astype(int)
        f1_t = f1_score(true_labels, pred)
        prec_t = precision_score(true_labels, pred, zero_division=0)
        rec_t = recall_score(true_labels, pred, zero_division=0)
        print(f"\n=== {name} at threshold=0.5 ===")
        print(f"F1: {f1_t:.4f}, Prec: {prec_t:.4f}, Rec: {rec_t:.4f}")
    
    # Save all evaluation results
    eval_results = {
        'pretrained': {
            'auc_roc': float(auc_pre), 'average_precision': float(ap_pre),
            'f1_score': float(f1_pre), 'precision': float(prec_pre), 'recall': float(rec_pre),
            'best_threshold': float(threshold_pre),
        },
        'scratch': {
            'auc_roc': float(auc_scr), 'average_precision': float(ap_scr),
            'f1_score': float(f1_scr), 'precision': float(prec_scr), 'recall': float(rec_scr),
            'best_threshold': float(threshold_scr),
        },
        'num_true_positives': int(true_labels.sum()),
        'num_candidates': len(true_labels),
    }
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Top-K metrics for both models
    for model_name, probs in [("pretrained", probs_pre), ("scratch", probs_scr)]:
        topk_metrics = {}
        for k in [10, 20, 30, 43, 50, 100]:
            top_k_idx = np.argsort(probs)[::-1][:k]
            tp_at_k = int(true_labels[top_k_idx].sum())
            total_pos = int(true_labels.sum())
            topk_metrics[f'top_{k}'] = {
                'true_positives_in_top_k': tp_at_k,
                'total_positives': total_pos,
                'recall_at_k': float(tp_at_k / total_pos) if total_pos > 0 else 0,
                'precision_at_k': float(tp_at_k / k),
            }
        with open(f'outputs/topk_metrics_{model_name}.json', 'w') as f:
            json.dump(topk_metrics, f, indent=2)
        
        print(f"\n{model_name} Top-K Metrics:")
        for k, v in topk_metrics.items():
            print(f"  {k}: TP={v['true_positives_in_top_k']}, P@K={v['precision_at_k']:.4f}, R@K={v['recall_at_k']:.4f}")
    
    # Save training metrics
    with open('outputs/finetune_metrics.json', 'w') as f:
        json.dump(finetune_metrics, f)
    with open('outputs/scratch_metrics.json', 'w') as f:
        json.dump(scratch_metrics, f)
    with open('outputs/pretrain_loss.json', 'w') as f:
        json.dump(pretrain_loss, f)
    
    # Discovery metrics
    for model_name, probs in [("pretrained", probs_pre), ("scratch", probs_scr)]:
        discovery_metrics = {}
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred_at_t = (probs > threshold).astype(int)
            tp = int(((pred_at_t == 1) & (true_labels == 1)).sum())
            fp = int(((pred_at_t == 1) & (true_labels == 0)).sum())
            fn = int(((pred_at_t == 0) & (true_labels == 1)).sum())
            tn = int(((pred_at_t == 0) & (true_labels == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            discovery_metrics[f'threshold_{threshold}'] = {
                'true_positives': tp, 'false_positives': fp,
                'precision': float(prec), 'recall': float(rec), 'f1': float(f1_t),
            }
        with open(f'outputs/discovery_metrics_{model_name}.json', 'w') as f:
            json.dump(discovery_metrics, f, indent=2)
    
    # Top candidates from pretrained model
    top_indices = np.argsort(probs_pre)[::-1][:50]
    top_results = []
    for rank, idx in enumerate(top_indices):
        top_results.append({
            'rank': rank + 1,
            'index': int(idx),
            'probability': float(probs_pre[idx]),
            'true_label': int(true_labels[idx]),
        })
    with open('outputs/top_candidates.json', 'w') as f:
        json.dump(top_results, f, indent=2)
    
    print("\nDone!")
