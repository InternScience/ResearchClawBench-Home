#!/usr/bin/env python3
"""
Altermagnetic Discovery - Enhanced GNN with Positional Encodings

Key enhancements:
1. Laplacian Positional Encodings (LPE) to capture structural/symmetry info
2. Multi-task self-supervised pretraining (node + edge)
3. Deeper GNN with proper residual connections
4. Cosine annealing + warmup
5. Cross-validation for robust evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj, dropout_edge, add_self_loops
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

CONFIG = {
    'hidden_dim': 256,
    'num_layers': 6,
    'num_pe': 16,  # Laplacian PE dim
    'dropout': 0.3,
    'pretrain_epochs': 150,
    'finetune_epochs': 300,
    'batch_size': 64,
    'lr': 0.0003,
    'weight_decay': 1e-4,
    'seed': 42,
    'device': 'cpu',
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

def load_dataset(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return [data[i] for i in range(len(data))]

def compute_lpe(data, k=16):
    """Compute Laplacian Positional Encodings for a single graph."""
    N = data.x.size(0)
    if N <= 1:
        return torch.zeros(N, k)
    
    edge_index = data.edge_index
    edge_weight = None
    
    # Use edge features as weights if available
    if data.edge_attr is not None:
        edge_weight = data.edge_attr[:, 0]  # Use first edge feature dimension
    
    # Compute normalized Laplacian eigenvectors
    lap_index, lap_weight = get_laplacian(edge_index, edge_weight, normalization='sym')
    
    # Build sparse -> dense Laplacian
    lap_dense = torch.zeros(N, N)
    for i in range(lap_index.size(1)):
        u, v = lap_index[0, i].item(), lap_index[1, i].item()
        lap_dense[u, v] = lap_weight[i].item()
    
    # Eigendecomposition
    try:
        eigvals, eigvecs = torch.linalg.eigh(lap_dense)
        # Take k smallest non-trivial eigenvectors
        k_actual = min(k, N)
        pe = eigvecs[:, :k_actual]
        # Pad if needed
        if k_actual < k:
            pe = torch.cat([pe, torch.zeros(N, k - k_actual)], dim=-1)
        # Random sign flip for stability
        sign = torch.rand(k) * 2 - 1
        sign = torch.where(sign >= 0, 1.0, -1.0)
        pe = pe * sign.unsqueeze(0)
        return pe.float()
    except:
        return torch.zeros(N, k)

def add_pe_to_dataset(samples, k=16):
    """Add Laplacian PE to all samples."""
    enhanced = []
    for data in samples:
        pe = compute_lpe(data, k)
        new_x = torch.cat([data.x.float(), pe], dim=-1)
        enhanced.append(Data(
            x=new_x,
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            y=data.y.clone() if hasattr(data, 'y') else torch.tensor([0]),
        ))
    return enhanced

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, num_layers=6, dropout=0.3):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.linears = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        
        for conv, norm, linear in zip(self.convs, self.norms, self.linears):
            h_res = h
            h = conv(h, edge_index, e)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + linear(h_res)  # residual connection
        
        return h

class MultiTaskPretrain(nn.Module):
    def __init__(self, hidden_dim, node_dim, edge_dim):
        super().__init__()
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, node_dim),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, edge_dim),
        )
    
    def forward_node(self, h):
        return self.node_head(h)
    
    def forward_edge(self, h, edge_index):
        src, dst = edge_index
        return self.edge_head(torch.cat([h[src], h[dst]], dim=-1))

class Classifier(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )
    
    def forward(self, h, batch):
        h_cat = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch),
            global_add_pool(h, batch),
        ], dim=-1)
        return self.net(h_cat).squeeze(-1)

def pretrain_epoch(model, pretrain_heads, loader, optimizer, device, orig_node_dim):
    model.train(); pretrain_heads.train()
    total_loss = 0.0; n = 0
    
    for data in loader:
        data = data.to(device)
        x = data.x; ei = data.edge_index; ea = data.edge_attr
        batch = data.batch
        
        # 1. Node masking
        N = x.size(0)
        mask_ratio = 0.35
        node_mask = torch.rand(N, device=device) < mask_ratio
        if node_mask.sum() == 0:
            node_mask[0] = True
        
        x_masked = x.clone()
        orig_nodes = x[node_mask].clone()
        # Only mask original features (not PE), but we mask everything for simplicity
        x_masked[node_mask] = 0.0 
        
        # 2. Edge dropping for edge prediction
        E = ei.size(1)
        edge_drop_ratio = 0.2
        edge_mask = torch.rand(E, device=device) < edge_drop_ratio
        ei_used = ei[:, ~edge_mask]
        ea_used = ea[~edge_mask] if ea is not None else None
        ei_dropped = ei[:, edge_mask]
        
        # Forward
        h = model(x_masked, ei_used, ea_used, batch)
        
        # Node loss
        # Only predict original node features (without PE portion)
        orig_no_pe = orig_nodes[:, :orig_node_dim]
        node_pred = pretrain_heads.forward_node(h[node_mask])  # [N_masked, orig_node_dim]
        node_loss = F.binary_cross_entropy_with_logits(node_pred, orig_no_pe)
        
        # Edge loss: predict edge features of dropped edges
        edge_loss = torch.tensor(0.0, device=device)
        if ei_dropped.size(1) > 0:
            edge_pred = pretrain_heads.forward_edge(h, ei_dropped)
            edge_true = ea[edge_mask] if ea is not None else torch.ones(ei_dropped.size(1), 1, device=device)
            edge_loss = F.mse_loss(edge_pred, edge_true)
        
        loss = node_loss + 0.5 * edge_loss
        
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(pretrain_heads.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item(); n += 1
    
    return total_loss / max(n, 1)

def finetune_epoch(model, classifier, loader, optimizer, criterion, device, train=True):
    if train: model.train(); classifier.train()
    else: model.eval(); classifier.eval()
    
    total_loss = 0.0; all_preds = []; all_labels = []; n = 0
    
    for data in loader:
        data = data.to(device)
        x = data.x; ei = data.edge_index; ea = data.edge_attr
        batch = data.batch; labels = data.y.float()
        
        with torch.set_grad_enabled(train):
            h = model(x, ei, ea, batch)
            logits = classifier(h, batch)
            loss = criterion(logits, labels)
        
        if train:
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        n += 1
    
    return total_loss / max(n, 1), np.array(all_preds), np.array(all_labels)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()

def main():
    device = CONFIG['device']
    print(f"Device: {device}")
    
    print("\n=== Loading Data ===")
    pretrain_samples_raw = load_dataset('data/pretrain_data.pt')
    finetune_samples_raw = load_dataset('data/finetune_data.pt')
    candidate_samples_raw = load_dataset('data/candidate_data.pt')
    
    ft_labels = [int(s.y.item()) for s in finetune_samples_raw]
    cand_labels_true = [int(s.y.item()) for s in candidate_samples_raw]
    print(f"Pretrain: {len(pretrain_samples_raw)}, Finetune: {len(finetune_samples_raw)}, Candidate: {len(candidate_samples_raw)}")
    print(f"Finetune labels: {Counter(ft_labels)}")
    print(f"Candidate true labels: {Counter(cand_labels_true)}")
    
    # Add PE
    print("\n=== Computing Laplacian Positional Encodings ===")
    pretrain_samples = add_pe_to_dataset(pretrain_samples_raw, CONFIG['num_pe'])
    finetune_samples = add_pe_to_dataset(finetune_samples_raw, CONFIG['num_pe'])
    candidate_samples = add_pe_to_dataset(candidate_samples_raw, CONFIG['num_pe'])
    
    input_dim = pretrain_samples[0].x.size(1)
    edge_dim = pretrain_samples[0].edge_attr.size(1)
    print(f"Input dim: {input_dim}, Edge dim: {edge_dim}")
    
    # Train/val split
    ft_idx = list(range(len(finetune_samples)))
    train_idx, val_idx = train_test_split(ft_idx, test_size=0.2, random_state=CONFIG['seed'], stratify=ft_labels)
    train_samples = [finetune_samples[i] for i in train_idx]
    val_samples = [finetune_samples[i] for i in val_idx]
    
    # Balance training via oversampling
    pos_tr = [s for s in train_samples if s.y.item() == 1]
    neg_tr = [s for s in train_samples if s.y.item() == 0]
    target_pos = len(neg_tr) // 3
    oversampled = (pos_tr * (target_pos // max(len(pos_tr), 1)))[:target_pos]
    train_balanced = neg_tr + oversampled
    print(f"Balanced train: {len(train_balanced)} (neg={len(neg_tr)}, pos={len(oversampled)})")
    
    pretrain_loader = DataLoader(pretrain_samples, batch_size=CONFIG['batch_size'], shuffle=True)
    train_loader = DataLoader(train_balanced, batch_size=min(CONFIG['batch_size'], len(train_balanced)), shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=CONFIG['batch_size'], shuffle=False)
    candidate_loader = DataLoader(candidate_samples, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Build model
    model = GNNEncoder(input_dim, edge_dim, CONFIG['hidden_dim'], CONFIG['num_layers'], CONFIG['dropout']).to(device)
    pretrain_heads = MultiTaskPretrain(CONFIG['hidden_dim'], input_dim - CONFIG['num_pe'], edge_dim).to(device)
    
    # Phase 1: Pretraining
    print("\n=== Phase 1: Pretraining ===")
    pt_opt = torch.optim.AdamW(
        list(model.parameters()) + list(pretrain_heads.parameters()),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )
    pt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pt_opt, T_max=CONFIG['pretrain_epochs'])
    
    best_pt = float('inf')
    for epoch in range(1, CONFIG['pretrain_epochs'] + 1):
        loss = pretrain_epoch(model, pretrain_heads, pretrain_loader, pt_opt, device, input_dim - CONFIG['num_pe'])
        pt_scheduler.step()
        if loss < best_pt:
            best_pt = loss
            torch.save(model.state_dict(), 'outputs/pretrained_pe.pt')
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Loss={loss:.6f}")
    print(f"Best pretrain loss: {best_pt:.6f}")
    
    model.load_state_dict(torch.load('outputs/pretrained_pe.pt', map_location=device))
    
    # Phase 2: Fine-tuning with cross-validation
    print("\n=== Phase 2: Fine-tuning ===")
    
    # Use stratified k-fold for robust evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG['seed'])
    fold_results = []
    
    all_ft_idx = np.array(range(len(finetune_samples)))
    all_ft_y = np.array(ft_labels)
    
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(all_ft_idx, all_ft_y)):
        print(f"\n--- Fold {fold+1}/5 ---")
        
        # Reset model weights
        model.load_state_dict(torch.load('outputs/pretrained_pe.pt', map_location=device))
        classifier = Classifier(CONFIG['hidden_dim'], CONFIG['dropout']).to(device)
        
        # Prepare fold data (with balancing)
        fold_tr_samples = [finetune_samples[i] for i in tr_idx]
        fold_vl_samples = [finetune_samples[i] for i in vl_idx]
        
        pos_fold = [s for s in fold_tr_samples if s.y.item() == 1]
        neg_fold = [s for s in fold_tr_samples if s.y.item() == 0]
        target_pos_fold = len(neg_fold) // 3
        os_fold = (pos_fold * (target_pos_fold // max(len(pos_fold), 1)))[:target_pos_fold]
        fold_tr_balanced = neg_fold + os_fold
        
        fold_tr_loader = DataLoader(fold_tr_balanced, batch_size=min(CONFIG['batch_size'], len(fold_tr_balanced)), shuffle=True)
        fold_vl_loader = DataLoader(fold_vl_samples, batch_size=CONFIG['batch_size'], shuffle=False)
        
        # Compute pos_weight for this fold
        pos_count = len(os_fold)
        neg_count = len(neg_fold)
        pos_w = torch.tensor([neg_count / max(pos_count, 1)], device=device)
        criterion = FocalLoss(alpha=0.8, gamma=2.0)
        
        ft_opt = torch.optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()),
            lr=CONFIG['lr'] * 0.5, weight_decay=CONFIG['weight_decay']
        )
        
        best_fold_auroc = 0.0
        for epoch in range(1, 150):  # Fewer epochs per fold
            tr_loss, _, _ = finetune_epoch(model, classifier, fold_tr_loader, ft_opt, criterion, device, True)
            vl_loss, vl_preds, vl_labs = finetune_epoch(model, classifier, fold_vl_loader, None, criterion, device, False)
            vl_auroc = roc_auc_score(vl_labs, vl_preds) if len(np.unique(vl_labs)) > 1 else 0.5
            
            if vl_auroc > best_fold_auroc:
                best_fold_auroc = vl_auroc
            
            if epoch % 50 == 0:
                print(f"    Epoch {epoch}: Val AUROC={vl_auroc:.4f}")
        
        fold_results.append(best_fold_auroc)
        print(f"  Fold {fold+1} best AUROC: {best_fold_auroc:.4f}")
    
    print(f"\nCross-validation AUROCs: {[f'{x:.4f}' for x in fold_results]}")
    print(f"Mean CV AUROC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    
    # Final training on all training data
    print("\n=== Final Training ===")
    model.load_state_dict(torch.load('outputs/pretrained_pe.pt', map_location=device))
    classifier = Classifier(CONFIG['hidden_dim'], CONFIG['dropout']).to(device)
    
    pos_count_final = len(oversampled)
    neg_count_final = len(neg_tr)
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    
    ft_opt = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=CONFIG['lr'] * 0.5, weight_decay=CONFIG['weight_decay']
    )
    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=CONFIG['finetune_epochs'])
    
    best_auroc = 0.0
    for epoch in range(1, CONFIG['finetune_epochs'] + 1):
        tr_loss, _, _ = finetune_epoch(model, classifier, train_loader, ft_opt, criterion, device, True)
        val_loss, val_preds, val_labs = finetune_epoch(model, classifier, val_loader, None, criterion, device, False)
        val_auroc = roc_auc_score(val_labs, val_preds) if len(np.unique(val_labs)) > 1 else 0.5
        ft_scheduler.step()
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'encoder': model.state_dict(),
                'classifier': classifier.state_dict(),
            }, 'outputs/best_model_pe.pt')
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: TrLoss={tr_loss:.4f}, ValLoss={val_loss:.4f}, AUROC={val_auroc:.4f}")
    
    print(f"Best AUROC: {best_auroc:.4f}")
    
    # Load best and evaluate
    ckpt = torch.load('outputs/best_model_pe.pt', map_location=device)
    model.load_state_dict(ckpt['encoder']); classifier.load_state_dict(ckpt['classifier'])
    
    # Validation
    print("\n=== Validation ===")
    _, val_preds, val_labs = finetune_epoch(model, classifier, val_loader, None, criterion, device, False)
    val_auroc = roc_auc_score(val_labs, val_preds)
    val_auprc = average_precision_score(val_labs, val_preds)
    precs, recs, threshs = precision_recall_curve(val_labs, val_preds)
    f1s = 2 * precs * recs / (precs + recs + 1e-10)
    best_thresh = threshs[np.argmax(f1s)]
    val_bin = (val_preds >= best_thresh).astype(int)
    
    print(f"  AUROC={val_auroc:.4f}, AUPRC={val_auprc:.4f}")
    print(f"  F1={f1_score(val_labs, val_bin, zero_division=0):.4f}")
    print(f"  Acc={accuracy_score(val_labs, val_bin):.4f}")
    cm_v = confusion_matrix(val_labs, val_bin)
    print(f"  CM:\n{cm_v}")
    
    # Candidate Prediction
    print("\n=== Candidate Prediction ===")
    model.eval(); classifier.eval()
    cand_preds = []; cand_labels = []
    for data in candidate_loader:
        data = data.to(device)
        x = data.x; ei = data.edge_index; ea = data.edge_attr
        batch = data.batch; labels = data.y.float()
        with torch.no_grad():
            h = model(x, ei, ea, batch)
            logits = classifier(h, batch)
            probs = torch.sigmoid(logits)
        cand_preds.extend(probs.cpu().numpy().tolist())
        cand_labels.extend(labels.cpu().numpy().tolist())
    
    cand_preds = np.array(cand_preds); cand_labels = np.array(cand_labels)
    cand_bin = (cand_preds >= best_thresh).astype(int)
    
    n_pred = int(cand_bin.sum()); n_true = int(cand_labels.sum())
    c_auroc = roc_auc_score(cand_labels, cand_preds)
    c_auprc = average_precision_score(cand_labels, cand_preds)
    
    print(f"  Predicted +: {n_pred}, True +: {n_true}")
    print(f"  AUROC={c_auroc:.4f}, AUPRC={c_auprc:.4f}")
    cm2 = confusion_matrix(cand_labels, cand_bin)
    print(f"  CM:\n{cm2}")
    
    sorted_idx = np.argsort(-cand_preds)
    for k in [20, 50, 100]:
        correct = int(cand_labels[sorted_idx[:k]].sum())
        print(f"  Top-{k}: {correct}/{k} = {correct/k:.2%}")
    
    print("\nTop 30 Predictions:")
    for i in range(min(30, len(sorted_idx))):
        idx = int(sorted_idx[i])
        p = float(cand_preds[idx]); l = bool(cand_labels[idx])
        print(f"  #{idx:4d}: p={p:.4f} [{'✓' if l else '✗'}]")
    
    # Save
    os.makedirs('outputs', exist_ok=True)
    results = {
        'config': CONFIG,
        'cv_aurocs': fold_results,
        'cv_mean': float(np.mean(fold_results)),
        'cv_std': float(np.std(fold_results)),
        'val_auroc': float(val_auroc),
        'val_metrics': {
            'auroc': float(val_auroc), 'auprc': float(val_auprc),
            'f1': float(f1_score(val_labs, val_bin, zero_division=0)),
            'accuracy': float(accuracy_score(val_labs, val_bin)),
            'precision': float(precision_score(val_labs, val_bin, zero_division=0)),
            'recall': float(recall_score(val_labs, val_bin, zero_division=0)),
            'threshold': float(best_thresh),
        },
        'candidate_metrics': {
            'auroc': float(c_auroc), 'auprc': float(c_auprc),
            'n_predicted': n_pred, 'n_true': n_true,
            'confusion_matrix': cm2.tolist(),
        },
        'top20': float(cand_labels[sorted_idx[:20]].mean()),
        'top50': float(cand_labels[sorted_idx[:50]].mean()),
        'top100': float(cand_labels[sorted_idx[:100]].mean()),
        'candidate_predictions': cand_preds.tolist(),
        'candidate_labels': cand_labels.tolist(),
    }
    with open('outputs/results_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
