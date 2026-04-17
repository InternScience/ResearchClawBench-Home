"""
AI-Powered Altermagnet Discovery Engine - Fast Version
======================================================
Optimized for CPU execution with reduced epochs but maintained quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_max_pool
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

WORKSPACE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_000_20260416_204143'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device('cpu')
print(f"Using device: {device}")

# Load Data
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

finetune_labels = [d.y.item() for d in finetune_data]
candidate_labels = [d.y.item() for d in candidate_data]
print(f"Finetune: {sum(finetune_labels)} pos / {len(finetune_labels)-sum(finetune_labels)} neg")
print(f"Candidate: {sum(candidate_labels)} pos / {len(candidate_labels)-sum(candidate_labels)} neg")

# ============================================================
# Models
# ============================================================

class GINEncoder(nn.Module):
    def __init__(self, in_channels=28, hidden_channels=64, num_layers=3, dropout=0.1, edge_dim=2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        self.node_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        self.edge_embed = nn.Linear(edge_dim, hidden_channels)
        
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
        
        # JK-style readout
        self.readout_dim = hidden_channels * num_layers * 2
        self.readout_proj = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_embed(x)
        
        if edge_attr is not None:
            edge_emb = self.edge_embed(edge_attr)
            src = edge_index[0]
            edge_agg = torch.zeros(x.size(0), x.size(1), device=x.device)
            edge_agg.scatter_add_(0, src.unsqueeze(1).expand(-1, x.size(1)), edge_emb)
            x = x + edge_agg
        
        layer_outputs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if batch is not None:
                layer_outputs.append(global_mean_pool(x, batch))
                layer_outputs.append(global_max_pool(x, batch))
            else:
                layer_outputs.append(x.mean(dim=0, keepdim=True))
                layer_outputs.append(x.max(dim=0, keepdim=True)[0])
        
        graph_repr = torch.cat(layer_outputs, dim=1)
        graph_repr = self.readout_proj(graph_repr)
        return graph_repr


class GCNEncoder(nn.Module):
    def __init__(self, in_channels=28, hidden_channels=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_ch, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
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
    def __init__(self, encoder, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
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
# Contrastive Pre-training
# ============================================================

def nt_xent_loss(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)])
    mask = torch.eye(2 * batch_size, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    loss = F.cross_entropy(sim, labels)
    return loss


def pretrain_contrastive(data_list, epochs=40, batch_size=256, lr=1e-3, hidden_dim=64):
    print("\n" + "="*60)
    print("PHASE 1: Self-Supervised Contrastive Pre-training")
    print("="*60)
    
    encoder = GINEncoder(in_channels=NODE_FEATURES, hidden_channels=hidden_dim, 
                         num_layers=3, dropout=0.1, edge_dim=EDGE_FEATURES)
    
    projector = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 32)
    )
    
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    
    losses = []
    encoder.train()
    projector.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in loader:
            # View 1
            x1 = batch.x * (torch.rand_like(batch.x) > 0.15).float()
            num_edges = batch.edge_index.size(1)
            keep1 = (torch.rand(num_edges) > 0.2)
            if keep1.sum() == 0: keep1[0] = True
            ei1 = batch.edge_index[:, keep1]
            ea1 = batch.edge_attr[keep1] if batch.edge_attr is not None else None
            
            # View 2
            x2 = batch.x * (torch.rand_like(batch.x) > 0.15).float()
            keep2 = (torch.rand(num_edges) > 0.2)
            if keep2.sum() == 0: keep2[0] = True
            ei2 = batch.edge_index[:, keep2]
            ea2 = batch.edge_attr[keep2] if batch.edge_attr is not None else None
            
            h1 = encoder(x1, ei1, ea1, batch.batch)
            h2 = encoder(x2, ei2, ea2, batch.batch)
            z1 = projector(h1)
            z2 = projector(h2)
            
            loss = nt_xent_loss(z1, z2, temperature=0.2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print(f"Pre-training complete. Final loss: {losses[-1]:.4f}")
    return encoder, losses


# ============================================================
# Fine-tuning
# ============================================================

def oversample_minority(data_list, target_ratio=0.3):
    labels = [d.y.item() for d in data_list]
    pos_data = [d for d, l in zip(data_list, labels) if l == 1]
    neg_data = [d for d, l in zip(data_list, labels) if l == 0]
    target_pos = int(len(neg_data) * target_ratio / (1 - target_ratio))
    n_repeats = target_pos // len(pos_data) + 1
    oversampled_pos = (pos_data * n_repeats)[:target_pos]
    combined = neg_data + oversampled_pos
    random.shuffle(combined)
    return combined


def finetune_model(encoder, finetune_data, epochs=80, batch_size=64, lr=3e-4, 
                   hidden_dim=64, val_ratio=0.2, seed=42):
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
    
    # Oversample
    train_data_os = oversample_minority(train_data, target_ratio=0.25)
    
    model = Classifier(copy.deepcopy(encoder), hidden_dim=hidden_dim, dropout=0.3)
    
    n_pos = sum([d.y.item() for d in train_data_os])
    n_neg = len(train_data_os) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': lr * 0.2},
        {'params': model.head.parameters(), 'lr': lr}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_loader = DataLoader(train_data_os, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    train_losses, val_losses, val_aucs, val_aps, val_f1s = [], [], [], [], []
    best_metric = 0
    best_model_state = None
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        nb = 0
        for batch in train_loader:
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            loss = criterion(logits, batch.y.float())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        
        scheduler.step()
        train_losses.append(epoch_loss / max(nb, 1))
        
        model.eval()
        vp, vt = [], []
        vl = 0
        vnb = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                vl += criterion(logits, batch.y.float()).item()
                vnb += 1
                vp.extend(torch.sigmoid(logits).numpy().tolist())
                vt.extend(batch.y.numpy().tolist())
        
        val_losses.append(vl / max(vnb, 1))
        
        try:
            auc = roc_auc_score(vt, vp)
            ap = average_precision_score(vt, vp)
        except:
            auc, ap = 0.5, 0.0
        val_aucs.append(auc)
        val_aps.append(ap)
        
        f1 = f1_score(vt, [1 if p > 0.5 else 0 for p in vp], zero_division=0)
        val_f1s.append(f1)
        
        metric = ap + 0.3 * auc
        if metric > best_metric:
            best_metric = metric
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, TL: {train_losses[-1]:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
        
        if patience >= 30:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    history = {
        'train_losses': train_losses, 'val_losses': val_losses,
        'val_aucs': val_aucs, 'val_aps': val_aps, 'val_f1s': val_f1s
    }
    return model, history, val_data, vt


def ensemble_predict(models, data_list, batch_size=256):
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    all_probs_list = []
    for model in models:
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                probs.extend(torch.sigmoid(logits).numpy().tolist())
        all_probs_list.append(np.array(probs))
    ensemble_probs = np.mean(all_probs_list, axis=0)
    return ensemble_probs, all_probs_list


def get_embeddings(model, data_list, batch_size=256):
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            emb = model.get_embeddings(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embs.append(emb.numpy())
    return np.concatenate(embs, axis=0)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    results = {}
    HIDDEN = 64
    
    # Phase 1: Pre-training
    pretrained_encoder, pretrain_losses = pretrain_contrastive(
        pretrain_data, epochs=40, batch_size=256, lr=1e-3, hidden_dim=HIDDEN
    )
    
    with open(os.path.join(OUTPUT_DIR, 'pretrain_losses.json'), 'w') as f:
        json.dump(pretrain_losses, f)
    
    # Phase 2: Fine-tune ensemble (3 models)
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning Pre-trained Ensemble")
    print("="*60)
    
    pretrained_models = []
    pretrained_histories = []
    for i in range(3):
        seed = SEED + i * 17
        print(f"\nModel {i+1}/3 (seed={seed})")
        model, history, val_data, val_true = finetune_model(
            pretrained_encoder, finetune_data, epochs=80, batch_size=64, 
            lr=3e-4, hidden_dim=HIDDEN, seed=seed
        )
        pretrained_models.append(model)
        pretrained_histories.append(history)
    
    # Random init ensemble
    print("\n" + "="*60)
    print("BASELINE: Random Init Ensemble")
    print("="*60)
    
    random_models = []
    random_histories = []
    for i in range(3):
        seed = SEED + i * 17
        print(f"\nRandom Model {i+1}/3 (seed={seed})")
        random_encoder = GINEncoder(in_channels=NODE_FEATURES, hidden_channels=HIDDEN, 
                                     num_layers=3, dropout=0.1, edge_dim=EDGE_FEATURES)
        model, history, _, _ = finetune_model(
            random_encoder, finetune_data, epochs=80, batch_size=64,
            lr=3e-4, hidden_dim=HIDDEN, seed=seed
        )
        random_models.append(model)
        random_histories.append(history)
    
    # GCN baseline (single model)
    print("\n" + "="*60)
    print("BASELINE: GCN (no pre-training)")
    print("="*60)
    gcn_encoder = GCNEncoder(in_channels=NODE_FEATURES, hidden_channels=HIDDEN, num_layers=3, dropout=0.1)
    # Need a wrapper that matches Classifier interface
    class GCNClassifier(nn.Module):
        def __init__(self, encoder, hidden_dim=64, dropout=0.3):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
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
    
    # Fine-tune GCN
    gcn_model_obj = GCNClassifier(gcn_encoder, hidden_dim=HIDDEN)
    # Use same training loop manually
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    labels = [d.y.item() for d in finetune_data]
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    n_vp = max(2, int(len(pos_idx) * 0.2))
    n_vn = int(len(neg_idx) * 0.2)
    val_idx = pos_idx[:n_vp] + neg_idx[:n_vn]
    train_idx = pos_idx[n_vp:] + neg_idx[n_vn:]
    gcn_train = oversample_minority([finetune_data[i] for i in train_idx], 0.25)
    gcn_val = [finetune_data[i] for i in val_idx]
    
    n_p = sum([d.y.item() for d in gcn_train])
    n_n = len(gcn_train) - n_p
    gcn_criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=torch.tensor([n_n/max(n_p,1)]))
    gcn_optimizer = torch.optim.Adam(gcn_model_obj.parameters(), lr=3e-4, weight_decay=1e-4)
    gcn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gcn_optimizer, T_max=80)
    
    gcn_train_loader = DataLoader(gcn_train, batch_size=64, shuffle=True)
    gcn_val_loader = DataLoader(gcn_val, batch_size=64, shuffle=False)
    
    best_gcn_metric = 0
    best_gcn_state = None
    gcn_history = {'val_aucs': [], 'val_aps': []}
    
    for epoch in range(80):
        gcn_model_obj.train()
        for batch in gcn_train_loader:
            logits = gcn_model_obj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            loss = gcn_criterion(logits, batch.y.float())
            gcn_optimizer.zero_grad()
            loss.backward()
            gcn_optimizer.step()
        gcn_scheduler.step()
        
        gcn_model_obj.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in gcn_val_loader:
                logits = gcn_model_obj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
                vp.extend(torch.sigmoid(logits).numpy().tolist())
                vt.extend(batch.y.numpy().tolist())
        try:
            auc = roc_auc_score(vt, vp)
            ap = average_precision_score(vt, vp)
        except:
            auc, ap = 0.5, 0.0
        gcn_history['val_aucs'].append(auc)
        gcn_history['val_aps'].append(ap)
        
        metric = ap + 0.3 * auc
        if metric > best_gcn_metric:
            best_gcn_metric = metric
            best_gcn_state = copy.deepcopy(gcn_model_obj.state_dict())
        
        if (epoch + 1) % 20 == 0:
            print(f"  GCN Epoch {epoch+1}/80, AUC: {auc:.4f}, AP: {ap:.4f}")
    
    if best_gcn_state:
        gcn_model_obj.load_state_dict(best_gcn_state)
    
    # Phase 3: Candidate Screening
    print("\n" + "="*60)
    print("PHASE 3: Candidate Screening")
    print("="*60)
    
    cand_true = np.array([d.y.item() for d in candidate_data])
    
    cand_probs_pretrained, cand_probs_individual = ensemble_predict(pretrained_models, candidate_data)
    cand_probs_random, _ = ensemble_predict(random_models, candidate_data)
    
    # GCN predictions
    gcn_model_obj.eval()
    gcn_probs = []
    gcn_loader = DataLoader(candidate_data, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in gcn_loader:
            logits = gcn_model_obj(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            gcn_probs.extend(torch.sigmoid(logits).numpy().tolist())
    cand_probs_gcn = np.array(gcn_probs)
    
    # Get embeddings
    cand_embeddings = get_embeddings(pretrained_models[0], candidate_data)
    ft_embeddings = get_embeddings(pretrained_models[0], finetune_data)
    ft_labels = np.array([d.y.item() for d in finetune_data])
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for name, probs in [("Pre-trained GIN Ensemble", cand_probs_pretrained),
                         ("Random Init GIN Ensemble", cand_probs_random),
                         ("GCN (no pre-training)", cand_probs_gcn)]:
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
        print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}")
        print(f"  CM: {cm.tolist()}")
        
        results[name] = {
            'auc_roc': float(auc_roc), 'auc_pr': float(auc_pr),
            'best_f1': float(best_f1), 'best_threshold': float(best_thresh),
            'precision': float(prec), 'recall': float(rec),
            'confusion_matrix': cm.tolist()
        }
    
    # Individual model performance
    individual_aucs = []
    individual_aps = []
    for i, probs in enumerate(cand_probs_individual):
        auc = roc_auc_score(cand_true, probs)
        ap = average_precision_score(cand_true, probs)
        individual_aucs.append(float(auc))
        individual_aps.append(float(ap))
        print(f"  Individual Model {i+1}: AUC={auc:.4f}, AP={ap:.4f}")
    results['individual_aucs'] = individual_aucs
    results['individual_aps'] = individual_aps
    
    # Discovery rates
    print("\n--- Discovery Rate ---")
    sorted_idx = np.argsort(-cand_probs_pretrained)
    discovery_rates = {}
    for k in [10, 20, 30, 50, 75, 100, 150, 200]:
        found = int(cand_true[sorted_idx[:k]].sum())
        total = int(cand_true.sum())
        dr = found / total
        pk = found / k
        print(f"  Top-{k}: {found}/{total} (DR={dr:.2%}, P@{k}={pk:.2%})")
        discovery_rates[str(k)] = {'found': found, 'total': total, 'dr': float(dr), 'pk': float(pk)}
    results['discovery_rates'] = discovery_rates
    
    # Save everything
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    preds_list = []
    for i in sorted_idx:
        preds_list.append({
            'rank': int(np.where(sorted_idx == i)[0][0] + 1),
            'index': int(i),
            'probability': float(cand_probs_pretrained[i]),
            'true_label': int(cand_true[i])
        })
    with open(os.path.join(OUTPUT_DIR, 'all_candidate_predictions.json'), 'w') as f:
        json.dump(preds_list, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'top100_candidates.json'), 'w') as f:
        json.dump(preds_list[:100], f, indent=2)
    
    # Save arrays
    np.save(os.path.join(OUTPUT_DIR, 'candidate_embeddings.npy'), cand_embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'candidate_probs.npy'), cand_probs_pretrained)
    np.save(os.path.join(OUTPUT_DIR, 'candidate_true_labels.npy'), cand_true)
    np.save(os.path.join(OUTPUT_DIR, 'finetune_embeddings.npy'), ft_embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'finetune_labels.npy'), ft_labels)
    np.save(os.path.join(OUTPUT_DIR, 'cand_probs_random.npy'), cand_probs_random)
    np.save(os.path.join(OUTPUT_DIR, 'cand_probs_gcn.npy'), cand_probs_gcn)
    
    # Save histories
    for i, h in enumerate(pretrained_histories):
        with open(os.path.join(OUTPUT_DIR, f'pretrained_history_{i}.json'), 'w') as f:
            json.dump(h, f)
    for i, h in enumerate(random_histories):
        with open(os.path.join(OUTPUT_DIR, f'random_history_{i}.json'), 'w') as f:
            json.dump(h, f)
    with open(os.path.join(OUTPUT_DIR, 'gcn_history.json'), 'w') as f:
        json.dump(gcn_history, f)
    
    print("\n" + "="*60)
    print("ALL DONE")
    print("="*60)
