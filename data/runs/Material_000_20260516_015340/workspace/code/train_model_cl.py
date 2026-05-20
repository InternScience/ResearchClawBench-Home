#!/usr/bin/env python3
"""
Altermagnetic Discovery - Contrastive Pretraining + Supervised Fine-tuning

Uses graph contrastive learning (GraphCL-style) for self-supervised pretraining:
- Node dropping
- Edge perturbation  
- Subgraph sampling
to create multiple views of the same graph, then contrastive loss.

Then fine-tune with balanced sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import dropout_edge, subgraph, add_self_loops
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
    'finetune_epochs': 250,
    'batch_size': 64,
    'lr': 0.0005,
    'weight_decay': 1e-4,
    'temperature': 0.1,
    'seed': 42,
    'device': 'cpu',
    'pos_weight': 5.0,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

def load_dataset(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return [data[i] for i in range(len(data))]

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, num_layers=4, dropout=0.3):
        super().__init__()
        self.node_enc = nn.Linear(input_dim, hidden_dim)
        if edge_dim > 0:
            self.edge_enc = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_enc = None
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim if edge_dim > 0 else None, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr) if self.edge_enc is not None and edge_attr is not None else None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, e)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if i > 0:
                h_new = h_new + h
            h = h_new
        return self.pool(h, batch) if batch is not None else h

def augment_graph(data, drop_node_ratio=0.1, drop_edge_ratio=0.1):
    """Create augmented view of a graph."""
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone() if data.edge_attr is not None else None
    
    # Node dropping: randomly mask node features
    num_nodes = x.size(0)
    node_mask = torch.rand(num_nodes) < drop_node_ratio
    if node_mask.any():
        x[node_mask] = 0.0
    
    # Edge dropping
    edge_index, edge_attr_dropped = dropout_edge(edge_index, p=drop_edge_ratio, force_undirected=False)
    if edge_attr is not None and edge_attr_dropped is not None:
        edge_attr = edge_attr_dropped
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class Classifier(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, h):
        return self.net(h).squeeze(-1)

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, input_dim, hidden_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, h):
        return F.normalize(self.net(h), dim=-1)

def contrastive_loss(z1, z2, temperature=0.1):
    """NT-Xent loss between two views."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    
    # Cosine similarity
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]
    
    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    
    # Positive pairs: (i, i+B) and (i+B, i)
    pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device, dtype=torch.bool)
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = True
        pos_mask[i + batch_size, i] = True
    
    # Compute loss
    exp_sim = torch.exp(sim)
    pos_sim = exp_sim[pos_mask].view(2 * batch_size, 1)
    neg_sum = exp_sim.masked_fill(pos_mask, 0).sum(dim=1, keepdim=True) + exp_sim.masked_fill(mask, 0).sum(dim=1, keepdim=True)
    
    loss = -torch.log(pos_sim / (pos_sim + neg_sum)).mean()
    return loss

def pretrain_epoch(encoder, proj_head, loader, optimizer, device):
    encoder.train(); proj_head.train()
    total_loss = 0.0; n = 0
    
    for data in loader:
        # Create two views
        data1 = augment_graph(data, drop_node_ratio=0.15, drop_edge_ratio=0.15).to(device)
        data2 = augment_graph(data, drop_node_ratio=0.15, drop_edge_ratio=0.15).to(device)
        
        x1, ei1, ea1 = data1.x.float(), data1.edge_index, (data1.edge_attr.float() if data1.edge_attr is not None else None)
        x2, ei2, ea2 = data2.x.float(), data2.edge_index, (data2.edge_attr.float() if data2.edge_attr is not None else None)
        
        # Encoder produces graph-level embeddings
        h1 = encoder(x1, ei1, ea1)
        h2 = encoder(x2, ei2, ea2)
        
        # Project
        z1 = proj_head(h1)
        z2 = proj_head(h2)
        
        loss = contrastive_loss(z1, z2, CONFIG['temperature'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item(); n += 1
    
    return total_loss / max(n, 1)

class SimpleBatch:
    pass

def collate_graphs(data_list):
    """Simple manual batch collation that handles edge_attr properly."""
    xs = []; eis = []; eas = []; batches = []
    node_offset = 0
    for i, d in enumerate(data_list):
        xs.append(d.x)
        eis.append(d.edge_index + node_offset)
        if d.edge_attr is not None:
            eas.append(d.edge_attr)
        batches.append(torch.full((d.x.size(0),), i, dtype=torch.long))
        node_offset += d.x.size(0)
    
    x = torch.cat(xs, dim=0)
    ei = torch.cat(eis, dim=1)
    ea = torch.cat(eas, dim=0) if eas else None
    batch = torch.cat(batches, dim=0)
    return Data(x=x, edge_index=ei, edge_attr=ea, batch=batch)

def finetune_epoch(encoder, classifier, loader, optimizer, criterion, device, train=True):
    if train: encoder.train(); classifier.train()
    else: encoder.eval(); classifier.eval()
    
    total_loss = 0.0; all_preds = []; all_labels = []; n = 0
    
    for data in loader:
        data = data.to(device)
        x = data.x.float(); ei = data.edge_index
        ea = data.edge_attr.float() if data.edge_attr is not None else None
        labels = data.y.float()
        
        with torch.set_grad_enabled(train):
            h = encoder(x, ei, ea)
            logits = classifier(h)
            loss = criterion(logits, labels)
        
        if train:
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        n += 1
    
    return total_loss / max(n, 1), np.array(all_preds), np.array(all_labels)

def main():
    device = CONFIG['device']
    print(f"Device: {device}")
    
    print("\n=== Loading Data ===")
    pretrain_samples = load_dataset('data/pretrain_data.pt')
    finetune_samples = load_dataset('data/finetune_data.pt')
    candidate_samples = load_dataset('data/candidate_data.pt')
    print(f"Pretrain: {len(pretrain_samples)}, Finetune: {len(finetune_samples)}, Candidate: {len(candidate_samples)}")
    
    ft_labels = [int(s.y.item()) for s in finetune_samples]
    print(f"Finetune labels: {Counter(ft_labels)}")
    
    input_dim = pretrain_samples[0].x.size(1)
    edge_dim = pretrain_samples[0].edge_attr.size(1) if pretrain_samples[0].edge_attr is not None else 0
    print(f"Dims: node={input_dim}, edge={edge_dim}")
    
    # Split with stratification
    ft_idx = list(range(len(finetune_samples)))
    train_idx, val_idx = train_test_split(ft_idx, test_size=0.2, random_state=CONFIG['seed'], stratify=ft_labels)
    train_samples = [finetune_samples[i] for i in train_idx]
    val_samples = [finetune_samples[i] for i in val_idx]
    
    # Balance training set via oversampling
    pos_tr = [s for s in train_samples if s.y.item() == 1]
    neg_tr = [s for s in train_samples if s.y.item() == 0]
    n_oversample = max(len(neg_tr) // 4, len(pos_tr) * 3)  # aim for ~20% positive
    oversampled = (pos_tr * (n_oversample // max(len(pos_tr), 1)))[:n_oversample]
    train_balanced = neg_tr + oversampled
    print(f"Balanced train: {len(train_balanced)} (neg={len(neg_tr)}, pos_os={len(oversampled)})")
    
    pretrain_loader = DataLoader(pretrain_samples, batch_size=CONFIG['batch_size'], shuffle=True)
    train_loader = DataLoader(train_balanced, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=CONFIG['batch_size'], shuffle=False)
    candidate_loader = DataLoader(candidate_samples, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Model
    encoder = GNNEncoder(input_dim, edge_dim, CONFIG['hidden_dim'], CONFIG['num_layers'], CONFIG['dropout']).to(device)
    proj_head = ProjectionHead(CONFIG['hidden_dim'], CONFIG['hidden_dim'], 128).to(device)
    
    # Phase 1: Contrastive Pretraining
    print("\n=== Phase 1: Contrastive Pretraining ===")
    pt_opt = torch.optim.AdamW(list(encoder.parameters()) + list(proj_head.parameters()), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    for epoch in range(1, CONFIG['pretrain_epochs'] + 1):
        # Use custom batching for pretraining
        loss_sum = 0.0; n_batches = 0
        encoder.train(); proj_head.train()
        
        indices = np.random.permutation(len(pretrain_samples))
        for start in range(0, len(indices), CONFIG['batch_size']):
            end = min(start + CONFIG['batch_size'], len(indices))
            batch_samples = [pretrain_samples[i] for i in indices[start:end]]
            
            # Create views
            views1 = [augment_graph(s, 0.15, 0.15) for s in batch_samples]
            views2 = [augment_graph(s, 0.15, 0.15) for s in batch_samples]
            
            # Forward both views
            h1s = []; h2s = []
            for v1, v2 in zip(views1, views2):
                v1 = v1.to(device); v2 = v2.to(device)
                h1 = encoder(v1.x.float(), v1.edge_index, v1.edge_attr.float() if v1.edge_attr is not None else None)
                h2 = encoder(v2.x.float(), v2.edge_index, v2.edge_attr.float() if v2.edge_attr is not None else None)
                h1s.append(h1); h2s.append(h2)
            
            z1 = proj_head(torch.stack(h1s))
            z2 = proj_head(torch.stack(h2s))
            
            loss = contrastive_loss(z1, z2, CONFIG['temperature'])
            
            pt_opt.zero_grad(); loss.backward(); pt_opt.step()
            loss_sum += loss.item(); n_batches += 1
        
        avg_loss = loss_sum / max(n_batches, 1)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Loss={avg_loss:.6f}")
    
    # Save pretrained encoder
    torch.save(encoder.state_dict(), 'outputs/pretrained_cl.pt')
    
    # Phase 2: Fine-tuning
    print("\n=== Phase 2: Fine-tuning ===")
    classifier = Classifier(CONFIG['hidden_dim'], CONFIG['dropout']).to(device)
    
    pos_w = torch.tensor([CONFIG['pos_weight']], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    
    ft_opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=CONFIG['lr'] * 0.5, weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ft_opt, mode='max', factor=0.5, patience=30)
    
    best_auroc = 0.0; best_epoch = 0
    for epoch in range(1, CONFIG['finetune_epochs'] + 1):
        tr_loss, _, _ = finetune_epoch(encoder, classifier, train_loader, ft_opt, criterion, device, True)
        val_loss, val_preds, val_labs = finetune_epoch(encoder, classifier, val_loader, None, criterion, device, False)
        
        val_auroc = roc_auc_score(val_labs, val_preds) if len(np.unique(val_labs)) > 1 else 0.5
        scheduler.step(val_auroc)
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc; best_epoch = epoch
            torch.save({'encoder': encoder.state_dict(), 'classifier': classifier.state_dict()}, 'outputs/best_model_cl.pt')
        
        if epoch % 40 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: TrLoss={tr_loss:.4f}, ValLoss={val_loss:.4f}, AUROC={val_auroc:.4f}")
    
    print(f"Best epoch {best_epoch}, AUROC={best_auroc:.4f}")
    
    ckpt = torch.load('outputs/best_model_cl.pt', map_location=device)
    encoder.load_state_dict(ckpt['encoder']); classifier.load_state_dict(ckpt['classifier'])
    
    # Validation
    print("\n=== Validation Evaluation ===")
    _, val_preds, val_labs = finetune_epoch(encoder, classifier, val_loader, None, criterion, device, False)
    val_auroc = roc_auc_score(val_labs, val_preds)
    val_auprc = average_precision_score(val_labs, val_preds)
    
    precs, recs, threshs = precision_recall_curve(val_labs, val_preds)
    f1s = 2 * precs * recs / (precs + recs + 1e-10)
    best_thresh = threshs[np.argmax(f1s)]
    val_bin = (val_preds >= best_thresh).astype(int)
    
    print(f"  AUROC={val_auroc:.4f}, AUPRC={val_auprc:.4f}")
    print(f"  F1={f1_score(val_labs, val_bin, zero_division=0):.4f}")
    print(f"  Acc={accuracy_score(val_labs, val_bin):.4f}")
    print(f"  Prec={precision_score(val_labs, val_bin, zero_division=0):.4f}")
    print(f"  Rec={recall_score(val_labs, val_bin, zero_division=0):.4f}")
    print(f"  CM:\n{confusion_matrix(val_labs, val_bin)}")
    
    # Candidate Prediction
    print("\n=== Candidate Prediction ===")
    encoder.eval(); classifier.eval()
    cand_preds = []; cand_labels = []
    for data in candidate_loader:
        data = data.to(device)
        x = data.x.float(); ei = data.edge_index
        ea = data.edge_attr.float() if data.edge_attr is not None else None
        labels = data.y.float()
        with torch.no_grad():
            h = encoder(x, ei, ea)
            logits = classifier(h)
            probs = torch.sigmoid(logits)
        cand_preds.extend(probs.cpu().numpy().tolist())
        cand_labels.extend(labels.cpu().numpy().tolist())
    
    cand_preds = np.array(cand_preds); cand_labels = np.array(cand_labels)
    cand_bin = (cand_preds >= best_thresh).astype(int)
    
    n_pred = int(cand_bin.sum()); n_true = int(cand_labels.sum())
    print(f"  Predicted +: {n_pred}, True +: {n_true}")
    
    c_auroc = roc_auc_score(cand_labels, cand_preds)
    c_auprc = average_precision_score(cand_labels, cand_preds)
    c_f1 = f1_score(cand_labels, cand_bin, zero_division=0)
    c_acc = accuracy_score(cand_labels, cand_bin)
    c_prec = precision_score(cand_labels, cand_bin, zero_division=0)
    c_rec = recall_score(cand_labels, cand_bin, zero_division=0)
    
    print(f"  AUROC={c_auroc:.4f}, AUPRC={c_auprc:.4f}")
    print(f"  F1={c_f1:.4f}, Acc={c_acc:.4f}, Prec={c_prec:.4f}, Rec={c_rec:.4f}")
    print(f"  CM:\n{confusion_matrix(cand_labels, cand_bin)}")
    
    sorted_idx = np.argsort(-cand_preds)
    for k in [20, 50, 100]:
        correct = int(cand_labels[sorted_idx[:k]].sum())
        print(f"  Top-{k}: {correct}/{k} = {correct/k:.2%}")
    
    print("\nTop 30:")
    for i in range(min(30, len(sorted_idx))):
        idx = int(sorted_idx[i])
        p = float(cand_preds[idx]); l = bool(cand_labels[idx])
        print(f"  #{idx:4d}: p={p:.4f} [{'✓' if l else '✗'}]")
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
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
        },
        'top50_precision': float(cand_labels[sorted_idx[:50]].mean()),
        'candidate_predictions': cand_preds.tolist(),
        'candidate_labels': cand_labels.tolist(),
        'candidate_binary': cand_bin.tolist(),
    }
    with open('outputs/results_cl.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    details = [{'index': i, 'probability': float(cand_preds[i]), 'predicted_altermagnet': bool(cand_bin[i]), 'true_altermagnet': bool(cand_labels[i])} for i in range(len(cand_preds))]
    details.sort(key=lambda x: x['probability'], reverse=True)
    with open('outputs/candidate_details_cl.json', 'w') as f:
        json.dump(details, f, indent=2)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
