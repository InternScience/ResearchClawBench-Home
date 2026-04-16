import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np
import json
import os
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
# 2. GNN Model - Smaller for speed
# ============================================================

class CrystalGNN(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=32, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(nn_layer, edge_dim=hidden_dim, train_eps=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = dropout
        repr_dim = hidden_dim * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=e)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res
        
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        graph_repr = torch.cat([h_mean, h_max, h_sum], dim=-1)
        logits = self.classifier(graph_repr)
        
        return logits, graph_repr, h


# ============================================================
# 3. Pre-training (Contrastive + Reconstruction)
# ============================================================

def augment_graph(data, drop_rate=0.15, noise_std=0.15):
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()
    
    x = x + torch.randn_like(x) * noise_std
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_rate
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    edge_index = edge_index[:, keep_mask]
    edge_attr = edge_attr[keep_mask]
    node_mask = torch.rand(x.size(0)) > 0.2
    x[~node_mask] = 0
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def pretrain(model, dataset, epochs=50, batch_size=256, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    hidden_dim = 32
    repr_dim = hidden_dim * 3
    
    proj_head = nn.Sequential(
        nn.Linear(repr_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
    ).to(device)
    
    node_recon = nn.Linear(hidden_dim, 28).to(device)
    
    data_list = get_data_list(dataset)
    best_loss = float('inf')
    best_state = None
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        proj_head.train()
        node_recon.train()
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
            
            _, repr1, node1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
            _, repr2, _ = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
            
            # Contrastive
            z1 = F.normalize(proj_head(repr1), dim=-1)
            z2 = F.normalize(proj_head(repr2), dim=-1)
            bs = z1.size(0)
            z = torch.cat([z1, z2], dim=0)
            sim = torch.mm(z, z.t()) / 0.07
            mask = torch.eye(2*bs, device=device).bool()
            sim.masked_fill_(mask, -1e9)
            labels = torch.cat([torch.arange(bs, device=device)+bs, torch.arange(bs, device=device)])
            cl_loss = F.cross_entropy(sim, labels)
            
            # Reconstruction
            recon_loss = F.mse_loss(node_recon(node1), batch_orig.x)
            
            loss = cl_loss + 0.5 * recon_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(proj_head.parameters())+list(node_recon.parameters()), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return loss_history, best_state


# ============================================================
# 4. Fine-tuning
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def finetune(model, dataset, epochs=150, batch_size=64, lr=2e-4, device='cpu'):
    model = model.to(device)
    
    data_list = get_data_list(dataset)
    labels = np.array([d.y.item() for d in data_list])
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    train_pos = pos_idx[:int(0.85 * len(pos_idx))]
    val_pos = pos_idx[int(0.85 * len(pos_idx)):]
    train_neg = neg_idx[:int(0.85 * len(neg_idx))]
    val_neg = neg_idx[int(0.85 * len(neg_idx)):]
    
    train_data = [data_list[i] for i in np.concatenate([train_pos, train_neg])]
    val_data = [data_list[i] for i in np.concatenate([val_pos, val_neg])]
    
    print(f"Train: {len(train_pos)} pos, {len(train_neg)} neg | Val: {len(val_pos)} pos, {len(val_neg)} neg")
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0
    best_ap = 0
    best_state = None
    best_threshold = 0.5
    metrics = []
    
    for epoch in range(epochs):
        model.train()
        
        # Oversample positives
        oversample = min(len(train_neg) // len(train_pos), 20)
        epoch_data = list(train_data)
        pos_data = [data_list[i] for i in train_pos]
        for _ in range(oversample - 1):
            epoch_data.extend(pos_data)
        np.random.shuffle(epoch_data)
        
        epoch_loss = 0
        for i in range(0, len(epoch_data), batch_size):
            batch = Batch.from_data_list(epoch_data[i:i+batch_size]).to(device)
            logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.float().view(-1, 1)
            
            loss = criterion(logits, target)
            # Label smoothing BCE
            smooth = target * 0.9 + 0.05
            loss = loss + 0.3 * F.binary_cross_entropy_with_logits(logits, smooth)
            
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
                batch = Batch.from_data_list(val_data[i:i+batch_size]).to(device)
                logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                val_labels.extend(batch.y.cpu().numpy().flatten())
        
        vp = np.array(val_probs)
        vl = np.array(val_labels)
        
        # Find best threshold
        bt, bvf1 = 0.5, 0
        for t in np.arange(0.05, 0.95, 0.05):
            vf = f1_score(vl, (vp > t).astype(int), zero_division=0)
            if vf > bvf1:
                bvf1 = vf
                bt = t
        
        vf1 = f1_score(vl, (vp > bt).astype(int), zero_division=0)
        vap = average_precision_score(vl, vp) if vl.sum() > 0 else 0
        vprec = precision_score(vl, (vp > bt).astype(int), zero_division=0)
        vrec = recall_score(vl, (vp > bt).astype(int), zero_division=0)
        
        metrics.append({
            'epoch': epoch, 'val_f1': float(vf1), 'val_ap': float(vap),
            'val_precision': float(vprec), 'val_recall': float(vrec),
            'best_threshold': float(bt),
        })
        
        if vf1 > best_f1 or (vf1 == best_f1 and vap > best_ap):
            best_f1 = vf1
            best_ap = vap
            best_threshold = bt
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val F1: {vf1:.4f} (t={bt:.2f}), AP: {vap:.4f}, P: {vprec:.4f}, R: {vrec:.4f}")
    
    return metrics, best_state, best_threshold


# ============================================================
# 5. Inference
# ============================================================

def predict(model, dataset, batch_size=128, device='cpu'):
    model.eval()
    model = model.to(device)
    data_list = get_data_list(dataset)
    all_probs, all_reprs, all_labels = [], [], []
    
    with torch.no_grad():
        for i in range(0, len(data_list), batch_size):
            batch = Batch.from_data_list(data_list[i:i+batch_size]).to(device)
            logits, reprs, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_reprs.extend(reprs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy().flatten())
    
    return np.array(all_probs), np.array(all_reprs), np.array(all_labels)


# ============================================================
# 6. Main
# ============================================================

if __name__ == '__main__':
    device = 'cpu'
    
    print("Loading datasets...")
    pretrain_ds = load_dataset('data/pretrain_data.pt')
    finetune_ds = load_dataset('data/finetune_data.pt')
    candidate_ds = load_dataset('data/candidate_data.pt')
    
    # ===== Pretrained Model =====
    model_pre = CrystalGNN(node_dim=28, edge_dim=2, hidden_dim=32, num_layers=3, dropout=0.2)
    
    print("\n=== Pre-training ===")
    pt_loss, pt_state = pretrain(model_pre, pretrain_ds, epochs=50, batch_size=256, lr=1e-3)
    
    print("\n=== Fine-tuning (pretrained) ===")
    model_pre.load_state_dict(pt_state)
    ft_pre, ft_state_pre, ft_thresh_pre = finetune(model_pre, finetune_ds, epochs=150, batch_size=64, lr=2e-4)
    
    # ===== Scratch Model =====
    print("\n=== Training from scratch ===")
    model_scr = CrystalGNN(node_dim=28, edge_dim=2, hidden_dim=32, num_layers=3, dropout=0.2)
    ft_scr, ft_state_scr, ft_thresh_scr = finetune(model_scr, finetune_ds, epochs=150, batch_size=64, lr=3e-4)
    
    # ===== Prediction =====
    print("\n=== Predicting candidates ===")
    model_pre.load_state_dict(ft_state_pre)
    probs_pre, reprs_pre, true_labels = predict(model_pre, candidate_ds)
    
    model_scr.load_state_dict(ft_state_scr)
    probs_scr, reprs_scr, _ = predict(model_scr, candidate_ds)
    
    # Evaluate
    results = {}
    for name, probs, thresh in [("pretrained", probs_pre, ft_thresh_pre), 
                                 ("scratch", probs_scr, ft_thresh_scr)]:
        pred = (probs > thresh).astype(int)
        pred05 = (probs > 0.5).astype(int)
        
        r = {
            'auc_roc': float(roc_auc_score(true_labels, probs)),
            'average_precision': float(average_precision_score(true_labels, probs)),
            'f1_optimal': float(f1_score(true_labels, pred)),
            'precision_optimal': float(precision_score(true_labels, pred, zero_division=0)),
            'recall_optimal': float(recall_score(true_labels, pred, zero_division=0)),
            'optimal_threshold': float(thresh),
            'f1_0.5': float(f1_score(true_labels, pred05)),
            'precision_0.5': float(precision_score(true_labels, pred05, zero_division=0)),
            'recall_0.5': float(recall_score(true_labels, pred05, zero_division=0)),
            'confusion_optimal': confusion_matrix(true_labels, pred).tolist(),
            'confusion_0.5': confusion_matrix(true_labels, pred05).tolist(),
        }
        
        # Top-K
        topk = {}
        for k in [10, 20, 30, 43, 50, 100]:
            idx = np.argsort(probs)[::-1][:k]
            tp = int(true_labels[idx].sum())
            total = int(true_labels.sum())
            topk[f'top_{k}'] = {'TP': tp, 'P@K': float(tp/k), 'R@K': float(tp/total) if total > 0 else 0}
        r['topk'] = topk
        
        # Discovery at thresholds
        disc = {}
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pt = (probs > t).astype(int)
            tp = int(((pt==1)&(true_labels==1)).sum())
            fp = int(((pt==1)&(true_labels==0)).sum())
            fn = int(((pt==0)&(true_labels==1)).sum())
            p = tp/(tp+fp) if (tp+fp)>0 else 0
            rc = tp/(tp+fn) if (tp+fn)>0 else 0
            disc[f'threshold_{t}'] = {'TP':tp,'FP':fp,'FN':fn,'precision':float(p),'recall':float(rc)}
        r['discovery'] = disc
        
        results[name] = r
        
        print(f"\n=== {name.upper()} (thresh={thresh:.3f}) ===")
        print(f"AUC: {r['auc_roc']:.4f}, AP: {r['average_precision']:.4f}")
        print(f"F1: {r['f1_optimal']:.4f}, P: {r['precision_optimal']:.4f}, R: {r['recall_optimal']:.4f}")
        print(f"At 0.5: F1: {r['f1_0.5']:.4f}, P: {r['precision_0.5']:.4f}, R: {r['recall_0.5']:.4f}")
        for k, v in topk.items():
            print(f"  {k}: TP={v['TP']}, P@K={v['P@K']:.4f}, R@K={v['R@K']:.4f}")
    
    results['meta'] = {
        'num_true_positives': int(true_labels.sum()),
        'num_candidates': len(true_labels),
    }
    
    # Save results
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    np.save('outputs/candidate_probs_pretrained.npy', probs_pre)
    np.save('outputs/candidate_probs_scratch.npy', probs_scr)
    np.save('outputs/candidate_true_labels.npy', true_labels)
    
    with open('outputs/ft_metrics_pretrained.json', 'w') as f:
        json.dump(ft_pre, f)
    with open('outputs/ft_metrics_scratch.json', 'w') as f:
        json.dump(ft_scr, f)
    with open('outputs/pretrain_loss.json', 'w') as f:
        json.dump(pt_loss, f)
    
    # Top candidates
    top_idx = np.argsort(probs_pre)[::-1][:50]
    top_cands = [{'rank': r+1, 'index': int(i), 'prob': float(probs_pre[i]), 'true_label': int(true_labels[i])} for r, i in enumerate(top_idx)]
    with open('outputs/top_candidates.json', 'w') as f:
        json.dump(top_cands, f, indent=2)
    
    print("\nDone!")
