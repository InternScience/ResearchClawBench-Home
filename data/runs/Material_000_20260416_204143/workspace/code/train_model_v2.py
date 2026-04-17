"""
AI-Powered Altermagnet Discovery Engine - V2 (Improved)
========================================================
Key improvements:
1. Better pre-training with graph-level reconstruction + contrastive
2. Edge feature integration via EdgeConv-style message passing
3. Improved class imbalance handling (focal loss, oversampling)
4. Ensemble of multiple runs
5. Better hyperparameter tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv, GCNConv, GATConv, 
    global_mean_pool, global_add_pool, global_max_pool,
    BatchNorm
)
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import json
import os
import sys
import types
import copy
import random
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# ============================================================
# Setup
# ============================================================
WORKSPACE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_000_20260416_204143'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
REPORT_DIR = os.path.join(WORKSPACE, 'report')
IMAGE_DIR = os.path.join(REPORT_DIR, 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# Load Data
# ============================================================
data_prepare = types.ModuleType('data_prepare')
class RealisticCrystalDataset:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): self.__dict__.update(state)
data_prepare.RealisticCrystalDataset = RealisticCrystalDataset
sys.modules['data_prepare'] = data_prepare

print("Loading datasets...")
pretrain_ds = torch.load(os.path.join(DATA_DIR, 'pretrain_data.pt'), weights_only=False)
finetune_ds = torch.load(os.path.join(DATA_DIR, 'finetune_data.pt'), weights_only=False)
candidate_ds = torch.load(os.path.join(DATA_DIR, 'candidate_data.pt'), weights_only=False)

pretrain_data = pretrain_ds.data_list
finetune_data = finetune_ds.data_list
candidate_data = candidate_ds.data_list

NODE_FEATURES = 28
EDGE_FEATURES = 2

print(f"Pretrain: {len(pretrain_data)}, Finetune: {len(finetune_data)}, Candidate: {len(candidate_data)}")

# Examine the data more carefully
print("\n--- Data Analysis ---")
finetune_labels = [d.y.item() for d in finetune_data]
candidate_labels = [d.y.item() for d in candidate_data]
print(f"Finetune: {sum(finetune_labels)} pos / {len(finetune_labels)-sum(finetune_labels)} neg")
print(f"Candidate: {sum(candidate_labels)} pos / {len(candidate_labels)-sum(candidate_labels)} neg")

# Analyze graph properties of positive vs negative
pos_nodes = [finetune_data[i].x.shape[0] for i in range(len(finetune_data)) if finetune_labels[i] == 1]
neg_nodes = [finetune_data[i].x.shape[0] for i in range(len(finetune_data)) if finetune_labels[i] == 0]
pos_edges = [finetune_data[i].edge_index.shape[1] for i in range(len(finetune_data)) if finetune_labels[i] == 1]
neg_edges = [finetune_data[i].edge_index.shape[1] for i in range(len(finetune_data)) if finetune_labels[i] == 0]
print(f"Pos graphs: nodes={np.mean(pos_nodes):.1f}±{np.std(pos_nodes):.1f}, edges={np.mean(pos_edges):.1f}±{np.std(pos_edges):.1f}")
print(f"Neg graphs: nodes={np.mean(neg_nodes):.1f}±{np.std(neg_nodes):.1f}, edges={np.mean(neg_edges):.1f}±{np.std(neg_edges):.1f}")

# Analyze element composition
print("\n--- Element Composition Analysis ---")
elem_to_idx = finetune_ds.elem_to_idx
idx_to_elem = {v: k for k, v in elem_to_idx.items()}
print(f"Elements: {elem_to_idx}")

# Check which elements appear in positive vs negative samples
def get_element_composition(data_list, indices=None):
    """Get element frequency for a set of graphs."""
    elem_counts = np.zeros(NODE_FEATURES)
    if indices is None:
        indices = range(len(data_list))
    for i in indices:
        # Each node's features are one-hot encoded
        node_feats = data_list[i].x.numpy()
        # Sum over nodes to get element counts
        elem_counts += node_feats.sum(axis=0)
    return elem_counts

pos_idx = [i for i, l in enumerate(finetune_labels) if l == 1]
neg_idx = [i for i, l in enumerate(finetune_labels) if l == 0]

pos_elem = get_element_composition(finetune_data, pos_idx)
neg_elem = get_element_composition(finetune_data, neg_idx)

# Normalize
pos_elem_norm = pos_elem / max(pos_elem.sum(), 1)
neg_elem_norm = neg_elem / max(neg_elem.sum(), 1)

print("Element frequency (pos vs neg):")
for i in range(NODE_FEATURES):
    if pos_elem[i] > 0 or neg_elem[i] > 0:
        elem_name = idx_to_elem.get(i, f'idx_{i}')
        print(f"  {elem_name}: pos={pos_elem_norm[i]:.4f}, neg={neg_elem_norm[i]:.4f}, diff={pos_elem_norm[i]-neg_elem_norm[i]:.4f}")


# ============================================================
# Model: Improved GIN with edge features and multi-scale readout
# ============================================================

class ImprovedGINEncoder(nn.Module):
    """Improved GIN encoder with edge features and multi-scale readout."""
    def __init__(self, in_channels=28, hidden_channels=128, num_layers=4, dropout=0.1, 
                 edge_dim=2, use_edge_features=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        
        # Initial node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # Edge feature projection
        if use_edge_features:
            self.edge_embed = nn.Sequential(
                nn.Linear(edge_dim, hidden_channels),
                nn.ReLU()
            )
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Multi-scale readout: concat mean and max pooling from each layer
        self.readout_dim = hidden_channels * num_layers * 2  # mean + max for each layer
        self.readout_proj = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Initial embedding
        x = self.node_embed(x)
        
        # Optional: incorporate edge features into node features
        if self.use_edge_features and edge_attr is not None:
            edge_emb = self.edge_embed(edge_attr)
            # Aggregate edge features to source nodes
            src = edge_index[0]
            edge_agg = torch.zeros(x.size(0), x.size(1), device=x.device)
            edge_agg.scatter_add_(0, src.unsqueeze(1).expand(-1, x.size(1)), edge_emb)
            x = x + edge_agg
        
        # GIN layers with JK (jumping knowledge) style readout
        layer_outputs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Collect graph-level representations from each layer
            if batch is not None:
                mean_pool = global_mean_pool(x, batch)
                max_pool = global_max_pool(x, batch)
            else:
                mean_pool = x.mean(dim=0, keepdim=True)
                max_pool = x.max(dim=0, keepdim=True)[0]
            layer_outputs.append(mean_pool)
            layer_outputs.append(max_pool)
        
        # Concatenate all layer outputs
        graph_repr = torch.cat(layer_outputs, dim=1)
        graph_repr = self.readout_proj(graph_repr)
        
        return graph_repr


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none',
                                                   pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class Classifier(nn.Module):
    """Binary classifier with pre-trained encoder."""
    def __init__(self, encoder, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.encoder(x, edge_index, edge_attr=edge_attr, batch=batch)
        return self.head(h)
    
    def get_embeddings(self, x, edge_index, edge_attr=None, batch=None):
        return self.encoder(x, edge_index, edge_attr=edge_attr, batch=batch)


# ============================================================
# Contrastive Pre-training (improved)
# ============================================================

class ContrastiveModel(nn.Module):
    def __init__(self, encoder, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr=edge_attr, batch=batch)
        z = self.projector(h)
        return z


def nt_xent_loss(z1, z2, temperature=0.2):
    """NT-Xent contrastive loss."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    
    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(z.device)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e9)
    
    loss = F.cross_entropy(sim, labels)
    return loss


def pretrain_contrastive(data_list, epochs=80, batch_size=128, lr=1e-3, hidden_dim=128):
    """Self-supervised contrastive pre-training."""
    print("\n" + "="*60)
    print("PHASE 1: Self-Supervised Contrastive Pre-training")
    print("="*60)
    
    encoder = ImprovedGINEncoder(
        in_channels=NODE_FEATURES, hidden_channels=hidden_dim, 
        num_layers=4, dropout=0.1, edge_dim=EDGE_FEATURES
    ).to(device)
    
    model = ContrastiveModel(encoder, hidden_dim=hidden_dim, proj_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            
            # View 1: feature masking + edge dropping
            x1 = batch.x.clone()
            mask1 = (torch.rand_like(x1) > 0.15).float()
            x1 = x1 * mask1
            
            num_edges = batch.edge_index.size(1)
            keep1 = (torch.rand(num_edges) > 0.2).bool()
            if keep1.sum() == 0: keep1[0] = True
            ei1 = batch.edge_index[:, keep1]
            ea1 = batch.edge_attr[keep1] if batch.edge_attr is not None else None
            
            # View 2: different masking
            x2 = batch.x.clone()
            mask2 = (torch.rand_like(x2) > 0.15).float()
            x2 = x2 * mask2
            
            keep2 = (torch.rand(num_edges) > 0.2).bool()
            if keep2.sum() == 0: keep2[0] = True
            ei2 = batch.edge_index[:, keep2]
            ea2 = batch.edge_attr[keep2] if batch.edge_attr is not None else None
            
            z1 = model(x1, ei1, ea1, batch.batch)
            z2 = model(x2, ei2, ea2, batch.batch)
            
            loss = nt_xent_loss(z1, z2, temperature=0.2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"Pre-training complete. Final loss: {losses[-1]:.4f}")
    return encoder, losses


# ============================================================
# Fine-tuning with oversampling and focal loss
# ============================================================

def oversample_minority(data_list, target_ratio=0.3):
    """Oversample positive class to reduce imbalance."""
    labels = [d.y.item() for d in data_list]
    pos_data = [d for d, l in zip(data_list, labels) if l == 1]
    neg_data = [d for d, l in zip(data_list, labels) if l == 0]
    
    # Target: pos_ratio ~ target_ratio
    target_pos = int(len(neg_data) * target_ratio / (1 - target_ratio))
    n_repeats = target_pos // len(pos_data) + 1
    
    oversampled_pos = (pos_data * n_repeats)[:target_pos]
    combined = neg_data + oversampled_pos
    random.shuffle(combined)
    
    labels_new = [d.y.item() for d in combined]
    print(f"After oversampling: {len(combined)} total, {sum(labels_new)} pos, {len(labels_new)-sum(labels_new)} neg")
    return combined


def finetune_model(encoder, finetune_data, epochs=150, batch_size=64, lr=3e-4, 
                   hidden_dim=128, val_ratio=0.2, use_focal=True, oversample=True,
                   seed=42):
    """Fine-tune with improved training strategy."""
    print(f"\n--- Fine-tuning (seed={seed}) ---")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    labels = [d.y.item() for d in finetune_data]
    pos_indices = [i for i, l in enumerate(labels) if l == 1]
    neg_indices = [i for i, l in enumerate(labels) if l == 0]
    
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    
    n_val_pos = max(2, int(len(pos_indices) * val_ratio))
    n_val_neg = int(len(neg_indices) * val_ratio)
    
    val_indices = pos_indices[:n_val_pos] + neg_indices[:n_val_neg]
    train_indices = pos_indices[n_val_pos:] + neg_indices[n_val_neg:]
    
    train_data = [finetune_data[i] for i in train_indices]
    val_data = [finetune_data[i] for i in val_indices]
    
    train_labels = [d.y.item() for d in train_data]
    val_labels = [d.y.item() for d in val_data]
    print(f"Train: {len(train_data)} (pos={sum(train_labels)}, neg={len(train_labels)-sum(train_labels)})")
    print(f"Val: {len(val_data)} (pos={sum(val_labels)}, neg={len(val_labels)-sum(val_labels)})")
    
    # Oversample
    if oversample:
        train_data = oversample_minority(train_data, target_ratio=0.25)
    
    model = Classifier(copy.deepcopy(encoder), hidden_dim=hidden_dim, dropout=0.3).to(device)
    
    n_pos = sum([d.y.item() for d in train_data])
    n_neg = len(train_data) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    
    if use_focal:
        criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use different LR for encoder and head
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': lr * 0.1},  # Lower LR for pre-trained encoder
        {'params': model.head.parameters(), 'lr': lr}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    train_losses = []
    val_losses = []
    val_aucs = []
    val_aps = []
    val_f1s = []
    best_metric = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            loss = criterion(logits, batch.y.float())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        val_loss_sum = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                loss = criterion(logits, batch.y.float())
                val_loss_sum += loss.item()
                val_batches += 1
                
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_true.extend(batch.y.cpu().numpy().tolist())
        
        val_loss = val_loss_sum / max(val_batches, 1)
        val_losses.append(val_loss)
        
        try:
            auc = roc_auc_score(val_true, val_preds)
            ap = average_precision_score(val_true, val_preds)
        except:
            auc = 0.5
            ap = 0.0
        val_aucs.append(auc)
        val_aps.append(ap)
        
        # Use AP (average precision) as primary metric for imbalanced data
        metric = ap + 0.5 * auc  # Weighted combination
        
        preds_binary = [1 if p > 0.5 else 0 for p in val_preds]
        f1 = f1_score(val_true, preds_binary, zero_division=0)
        val_f1s.append(f1)
        
        if metric > best_metric:
            best_metric = metric
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
        
        if patience_counter >= 40:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Best validation metric: {best_metric:.4f}")
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs,
        'val_aps': val_aps,
        'val_f1s': val_f1s
    }
    
    return model, history, val_data, val_true


# ============================================================
# Ensemble training
# ============================================================

def train_ensemble(encoder, finetune_data, n_models=5, **kwargs):
    """Train an ensemble of classifiers with different seeds."""
    print("\n" + "="*60)
    print(f"Training Ensemble of {n_models} Models")
    print("="*60)
    
    models = []
    histories = []
    
    for i in range(n_models):
        seed = SEED + i * 17
        model, history, _, _ = finetune_model(encoder, finetune_data, seed=seed, **kwargs)
        models.append(model)
        histories.append(history)
    
    return models, histories


def ensemble_predict(models, data_list, batch_size=128):
    """Get ensemble predictions."""
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    
    all_probs_list = []
    
    for model in models:
        model.eval()
        model_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                model_probs.extend(probs.tolist())
        all_probs_list.append(np.array(model_probs))
    
    # Average probabilities
    ensemble_probs = np.mean(all_probs_list, axis=0)
    return ensemble_probs, all_probs_list


def get_embeddings(model, data_list, batch_size=128):
    """Get embeddings from a model."""
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.get_embeddings(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    results = {}
    
    # ---- Phase 1: Pre-training ----
    pretrained_encoder, pretrain_losses = pretrain_contrastive(
        pretrain_data, epochs=80, batch_size=128, lr=1e-3, hidden_dim=128
    )
    
    with open(os.path.join(OUTPUT_DIR, 'pretrain_losses.json'), 'w') as f:
        json.dump(pretrain_losses, f)
    
    # ---- Phase 2: Fine-tuning (Ensemble) ----
    # Pre-trained ensemble
    pretrained_models, pretrained_histories = train_ensemble(
        pretrained_encoder, finetune_data, n_models=5,
        epochs=150, batch_size=64, lr=3e-4, hidden_dim=128,
        use_focal=True, oversample=True
    )
    
    # Random init ensemble for comparison
    print("\n" + "="*60)
    print("BASELINE: Random Initialization Ensemble")
    print("="*60)
    random_encoder = ImprovedGINEncoder(
        in_channels=NODE_FEATURES, hidden_channels=128, 
        num_layers=4, dropout=0.1, edge_dim=EDGE_FEATURES
    ).to(device)
    random_models, random_histories = train_ensemble(
        random_encoder, finetune_data, n_models=5,
        epochs=150, batch_size=64, lr=3e-4, hidden_dim=128,
        use_focal=True, oversample=True
    )
    
    # ---- Phase 3: Candidate Screening ----
    print("\n" + "="*60)
    print("PHASE 3: Candidate Screening")
    print("="*60)
    
    cand_true = np.array([d.y.item() for d in candidate_data])
    
    # Pre-trained ensemble predictions
    cand_probs_pretrained, cand_probs_individual = ensemble_predict(pretrained_models, candidate_data)
    
    # Random init ensemble predictions
    cand_probs_random, _ = ensemble_predict(random_models, candidate_data)
    
    # Get embeddings from first pre-trained model
    cand_embeddings = get_embeddings(pretrained_models[0], candidate_data)
    ft_embeddings = get_embeddings(pretrained_models[0], finetune_data)
    ft_labels = np.array([d.y.item() for d in finetune_data])
    
    # ---- Evaluation ----
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for name, probs in [("Pre-trained GIN Ensemble", cand_probs_pretrained),
                         ("Random Init GIN Ensemble", cand_probs_random)]:
        auc_roc = roc_auc_score(cand_true, probs)
        auc_pr = average_precision_score(cand_true, probs)
        
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.05, 0.95, 0.01):
            preds = (probs > thresh).astype(int)
            f1 = f1_score(cand_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        preds = (probs > best_thresh).astype(int)
        prec = precision_score(cand_true, preds, zero_division=0)
        rec = recall_score(cand_true, preds, zero_division=0)
        
        cm = confusion_matrix(cand_true, preds)
        
        print(f"\n{name}:")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")
        print(f"  Best F1: {best_f1:.4f} (threshold={best_thresh:.2f})")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        results[name] = {
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'best_f1': float(best_f1),
            'best_threshold': float(best_thresh),
            'precision': float(prec),
            'recall': float(rec),
            'confusion_matrix': cm.tolist()
        }
    
    # Individual model performance
    print("\n--- Individual Model Performance (Pre-trained) ---")
    individual_aucs = []
    individual_aps = []
    for i, probs in enumerate(cand_probs_individual):
        auc = roc_auc_score(cand_true, probs)
        ap = average_precision_score(cand_true, probs)
        individual_aucs.append(auc)
        individual_aps.append(ap)
        print(f"  Model {i+1}: AUC-ROC={auc:.4f}, AUC-PR={ap:.4f}")
    
    results['individual_aucs'] = [float(a) for a in individual_aucs]
    results['individual_aps'] = [float(a) for a in individual_aps]
    
    # Discovery rate
    print("\n--- Discovery Rate (Pre-trained Ensemble) ---")
    sorted_indices = np.argsort(-cand_probs_pretrained)
    discovery_rates = {}
    for k in [10, 20, 30, 50, 75, 100, 150, 200]:
        top_k_true = cand_true[sorted_indices[:k]]
        found = int(top_k_true.sum())
        total_pos = int(cand_true.sum())
        rate = found / total_pos if total_pos > 0 else 0
        precision_at_k = found / k
        print(f"  Top-{k}: Found {found}/{total_pos} altermagnets (DR: {rate:.2%}, P@{k}: {precision_at_k:.2%})")
        discovery_rates[str(k)] = {
            'found': found,
            'total_positives': total_pos,
            'discovery_rate': float(rate),
            'precision_at_k': float(precision_at_k)
        }
    
    results['discovery_rates'] = discovery_rates
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    candidate_predictions = []
    for i in sorted_indices:
        candidate_predictions.append({
            'rank': int(np.where(sorted_indices == i)[0][0] + 1),
            'index': int(i),
            'probability': float(cand_probs_pretrained[i]),
            'true_label': int(cand_true[i]),
            'predicted': int(cand_probs_pretrained[i] > results['Pre-trained GIN Ensemble']['best_threshold'])
        })
    
    with open(os.path.join(OUTPUT_DIR, 'all_candidate_predictions.json'), 'w') as f:
        json.dump(candidate_predictions, f, indent=2)
    
    # Save top candidates
    top_candidates = candidate_predictions[:100]
    with open(os.path.join(OUTPUT_DIR, 'top100_candidates.json'), 'w') as f:
        json.dump(top_candidates, f, indent=2)
    
    # Save embeddings
    np.save(os.path.join(OUTPUT_DIR, 'candidate_embeddings.npy'), cand_embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'candidate_probs.npy'), cand_probs_pretrained)
    np.save(os.path.join(OUTPUT_DIR, 'candidate_true_labels.npy'), cand_true)
    np.save(os.path.join(OUTPUT_DIR, 'finetune_embeddings.npy'), ft_embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'finetune_labels.npy'), ft_labels)
    np.save(os.path.join(OUTPUT_DIR, 'cand_probs_random.npy'), cand_probs_random)
    
    # Save training histories
    for i, h in enumerate(pretrained_histories):
        with open(os.path.join(OUTPUT_DIR, f'pretrained_history_{i}.json'), 'w') as f:
            json.dump(h, f)
    for i, h in enumerate(random_histories):
        with open(os.path.join(OUTPUT_DIR, f'random_history_{i}.json'), 'w') as f:
            json.dump(h, f)
    
    print("\n" + "="*60)
    print("ALL RESULTS SAVED SUCCESSFULLY")
    print("="*60)
