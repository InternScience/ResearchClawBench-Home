#!/usr/bin/env python3
"""
Altermagnetic Discovery - Simplified & Robust Pipeline

Key changes:
1. Use graph-level feature extraction for pretraining
2. Node feature statistics as graph-level features
3. GNN with edge features, robust batching
4. Simpler but more reliable approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import json
import os
from collections import Counter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'hidden_dim': 256,
    'num_layers': 4,
    'dropout': 0.3,
    'pretrain_epochs': 100,
    'finetune_epochs': 300,
    'batch_size': 64,
    'lr': 0.0005,
    'weight_decay': 1e-4,
    'seed': 42,
    'device': 'cpu',
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

def load_dataset(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return [data[i] for i in range(len(data))]

class GNNEncoder(nn.Module):
    """GNN encoder using GINEConv for graph-level embeddings."""
    def __init__(self, input_dim, edge_dim, hidden_dim, num_layers=4, dropout=0.3):
        super().__init__()
        self.node_enc = nn.Linear(input_dim, hidden_dim)
        self.edge_enc = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch=None):
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if i > 0:
                h_new = h_new + h
            h = h_new
        return h

class AutoencoderDecoder(nn.Module):
    """Decode node features from latent representations for pretraining."""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, h):
        return self.net(h)

class Classifier(nn.Module):
    """Graph classification head."""
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
        h_cat = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch),
            global_add_pool(h, batch),
        ], dim=-1)
        return self.net(h_cat).squeeze(-1)

def pretrain_epoch(model, decoder, loader, optimizer, device):
    model.train(); decoder.train()
    total_loss = 0.0; n = 0
    for data in loader:
        data = data.to(device)
        x = data.x.float()
        ei = data.edge_index
        ea = data.edge_attr.float()
        batch = data.batch

        # Mask random nodes
        num_nodes = x.size(0)
        mask_ratio = 0.35
        mask = torch.rand(num_nodes, device=device) < mask_ratio
        if mask.sum() == 0:
            mask[0] = True
        
        x_masked = x.clone()
        orig = x[mask].clone()
        x_masked[mask] = 0.0
        
        h = model(x_masked, ei, ea, batch)
        pred = decoder(h[mask])
        loss = F.binary_cross_entropy_with_logits(pred, orig)
        
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)

def compute_graph_embeddings(model, loader, device):
    """Compute graph-level embeddings for all samples."""
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = data.x.float(); ei = data.edge_index; ea = data.edge_attr.float()
            batch = data.batch
            h = model(x, ei, ea, batch)
            # Pool to graph level
            graph_h = torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
                global_add_pool(h, batch),
            ], dim=-1)
            embeddings.append(graph_h.cpu())
            if hasattr(data, 'y'):
                labels.append(data.y.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0) if labels else None
    return embeddings, labels

def finetune_epoch(model, classifier, loader, optimizer, criterion, device, train=True):
    if train: model.train(); classifier.train()
    else: model.eval(); classifier.eval()
    total_loss = 0.0; all_preds = []; all_labels = []; n = 0
    for data in loader:
        data = data.to(device)
        x = data.x.float(); ei = data.edge_index; ea = data.edge_attr.float()
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

def main():
    device = CONFIG['device']
    torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])
    print(f"Device: {device}")

    print("\n=== Loading Data ===")
    pretrain_samples = load_dataset('data/pretrain_data.pt')
    finetune_samples = load_dataset('data/finetune_data.pt')
    candidate_samples = load_dataset('data/candidate_data.pt')
    print(f"Pretrain: {len(pretrain_samples)}, Finetune: {len(finetune_samples)}, Candidate: {len(candidate_samples)}")

    ft_labels = [int(s.y.item()) for s in finetune_samples]
    cand_labels_true = [int(s.y.item()) for s in candidate_samples]
    print(f"Finetune labels: {Counter(ft_labels)}")
    print(f"Candidate true labels: {Counter(cand_labels_true)}")

    input_dim = pretrain_samples[0].x.size(1)
    edge_dim = pretrain_samples[0].edge_attr.size(1)
    print(f"Dims: node={input_dim}, edge={edge_dim}")

    # Train/val split
    ft_idx = list(range(len(finetune_samples)))
    train_idx, val_idx = train_test_split(ft_idx, test_size=0.2, random_state=CONFIG['seed'], stratify=ft_labels)
    train_samples = [finetune_samples[i] for i in train_idx]
    val_samples = [finetune_samples[i] for i in val_idx]

    # Balance: oversample positives
    pos_tr = [s for s in train_samples if s.y.item() == 1]
    neg_tr = [s for s in train_samples if s.y.item() == 0]
    n_pos = len(pos_tr)
    n_neg = len(neg_tr)
    # Oversample to ~20% positive rate
    target_pos = n_neg // 4
    oversampled = (pos_tr * (target_pos // max(n_pos, 1)))[:target_pos]
    train_balanced = neg_tr + oversampled
    print(f"Train balanced: {len(train_balanced)} (neg={n_neg}, pos_os={len(oversampled)})")

    pretrain_loader = DataLoader(pretrain_samples, batch_size=CONFIG['batch_size'], shuffle=True)
    train_loader = DataLoader(train_balanced, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=CONFIG['batch_size'], shuffle=False)
    candidate_loader = DataLoader(candidate_samples, batch_size=CONFIG['batch_size'], shuffle=False)

    # Model
    model = GNNEncoder(input_dim, edge_dim, CONFIG['hidden_dim'], CONFIG['num_layers'], CONFIG['dropout']).to(device)
    decoder = AutoencoderDecoder(CONFIG['hidden_dim'], input_dim).to(device)

    # Phase 1: Pretraining
    print("\n=== Phase 1: Self-Supervised Pretraining ===")
    pt_opt = torch.optim.AdamW(list(model.parameters()) + list(decoder.parameters()),
                                lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    best_pt_loss = float('inf')
    for epoch in range(1, CONFIG['pretrain_epochs'] + 1):
        loss = pretrain_epoch(model, decoder, pretrain_loader, pt_opt, device)
        if loss < best_pt_loss:
            best_pt_loss = loss
            torch.save(model.state_dict(), 'outputs/pretrained_v4.pt')
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Loss={loss:.6f}")
    print(f"Best pretrain loss: {best_pt_loss:.6f}")
    model.load_state_dict(torch.load('outputs/pretrained_v4.pt', map_location=device))

    # Phase 2: Fine-tuning
    print("\n=== Phase 2: Fine-tuning ===")
    classifier = Classifier(CONFIG['hidden_dim'], CONFIG['dropout']).to(device)
    
    # Compute class weights
    pos_count = sum(1 for s in train_balanced if s.y.item() == 1)
    neg_count = sum(1 for s in train_balanced if s.y.item() == 0)
    pos_weight = neg_count / max(pos_count, 1)
    print(f"Pos weight: {pos_weight:.2f}")
    
    pos_w = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    
    ft_opt = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()),
                                lr=CONFIG['lr'] * 0.5, weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ft_opt, mode='max', factor=0.5, patience=40)
    
    best_auroc = 0.0; best_epoch = 0
    for epoch in range(1, CONFIG['finetune_epochs'] + 1):
        tr_loss, _, _ = finetune_epoch(model, classifier, train_loader, ft_opt, criterion, device, True)
        val_loss, val_preds, val_labs = finetune_epoch(model, classifier, val_loader, None, criterion, device, False)
        val_auroc = roc_auc_score(val_labs, val_preds) if len(np.unique(val_labs)) > 1 else 0.5
        scheduler.step(val_auroc)
        if val_auroc > best_auroc:
            best_auroc = val_auroc; best_epoch = epoch
            torch.save({'encoder': model.state_dict(), 'classifier': classifier.state_dict()}, 'outputs/best_model_v4.pt')
        if epoch % 40 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: TrLoss={tr_loss:.4f}, ValLoss={val_loss:.4f}, AUROC={val_auroc:.4f}")
    print(f"Best epoch {best_epoch}, AUROC={best_auroc:.4f}")

    ckpt = torch.load('outputs/best_model_v4.pt', map_location=device)
    model.load_state_dict(ckpt['encoder']); classifier.load_state_dict(ckpt['classifier'])

    # Validation
    print("\n=== Validation Evaluation ===")
    _, val_preds, val_labs = finetune_epoch(model, classifier, val_loader, None, criterion, device, False)
    val_auroc = roc_auc_score(val_labs, val_preds)
    val_auprc = average_precision_score(val_labs, val_preds)
    precs, recs, threshs = precision_recall_curve(val_labs, val_preds)
    f1s = 2 * precs * recs / (precs + recs + 1e-10)
    best_idx = np.argmax(f1s)
    best_thresh = threshs[best_idx]
    val_bin = (val_preds >= best_thresh).astype(int)
    print(f"  AUROC={val_auroc:.4f}, AUPRC={val_auprc:.4f}, Thresh={best_thresh:.4f}")
    print(f"  F1={f1_score(val_labs, val_bin, zero_division=0):.4f}")
    print(f"  Acc={accuracy_score(val_labs, val_bin):.4f}")
    print(f"  Prec={precision_score(val_labs, val_bin, zero_division=0):.4f}")
    print(f"  Rec={recall_score(val_labs, val_bin, zero_division=0):.4f}")
    print(f"  CM:\n{confusion_matrix(val_labs, val_bin)}")

    # Candidate Prediction
    print("\n=== Candidate Prediction ===")
    model.eval(); classifier.eval()
    cand_preds = []; cand_labels = []
    for data in candidate_loader:
        data = data.to(device)
        x = data.x.float(); ei = data.edge_index; ea = data.edge_attr.float()
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
    c_f1 = f1_score(cand_labels, cand_bin, zero_division=0)
    c_acc = accuracy_score(cand_labels, cand_bin)
    c_prec = precision_score(cand_labels, cand_bin, zero_division=0)
    c_rec = recall_score(cand_labels, cand_bin, zero_division=0)
    
    print(f"  Predicted +: {n_pred}, True +: {n_true}")
    print(f"  AUROC={c_auroc:.4f}, AUPRC={c_auprc:.4f}")
    print(f"  F1={c_f1:.4f}, Acc={c_acc:.4f}")
    print(f"  Prec={c_prec:.4f}, Rec={c_rec:.4f}")
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
    
    # Also compute and save graph embeddings for visualization
    print("\n=== Computing Graph Embeddings ===")
    pretrain_emb, _ = compute_graph_embeddings(model, pretrain_loader, device)
    finetune_emb, ft_l = compute_graph_embeddings(model, DataLoader(finetune_samples, batch_size=CONFIG['batch_size'], shuffle=False), device)
    candidate_emb, cd_l = compute_graph_embeddings(model, candidate_loader, device)
    
    torch.save({
        'pretrain_embeddings': pretrain_emb,
        'finetune_embeddings': finetune_emb,
        'finetune_labels': ft_l,
        'candidate_embeddings': candidate_emb,
        'candidate_labels': cd_l,
    }, 'outputs/embeddings.pt')

    results = {
        'config': CONFIG,
        'best_val_auroc': float(val_auroc),
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
            'accuracy': float(c_acc), 'precision': float(c_prec),
            'recall': float(c_rec), 'f1': float(c_f1),
            'n_predicted': n_pred, 'n_true': n_true,
            'confusion_matrix': cm2.tolist(),
        },
        'top20': float(cand_labels[sorted_idx[:20]].mean()),
        'top50': float(cand_labels[sorted_idx[:50]].mean()),
        'top100': float(cand_labels[sorted_idx[:100]].mean()),
        'candidate_predictions': cand_preds.tolist(),
        'candidate_labels': cand_labels.tolist(),
    }
    with open('outputs/results_v4.json', 'w') as f:
        json.dump(results, f, indent=2)

    details = [{
        'index': i,
        'probability': float(cand_preds[i]),
        'predicted_altermagnet': bool(cand_bin[i]),
        'true_altermagnet': bool(cand_labels[i]),
    } for i in range(len(cand_preds))]
    details.sort(key=lambda x: x['probability'], reverse=True)
    with open('outputs/candidate_details_v4.json', 'w') as f:
        json.dump(details, f, indent=2)

    print("\nDone!")
    return results

if __name__ == '__main__':
    main()
