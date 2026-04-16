import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
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
# 2. GNN Model
# ============================================================

class GNNEncoder(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.node_emb = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            conv = GINEConv(nn_layer, edge_dim=hidden_dim, train_eps=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        
        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            h_res = h
            h = conv(h, edge_index, edge_attr=e)
            h = norm(h)
            h = F.gelu(h)
            h = drop(h)
            h = h + h_res
        
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        graph_repr = torch.cat([h_mean, h_max, h_sum], dim=-1)
        
        return graph_repr, h


class AltermagnetClassifier(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = GNNEncoder(node_dim, edge_dim, hidden_dim, num_layers, dropout)
        repr_dim = hidden_dim * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        graph_repr, node_repr = self.encoder(x, edge_index, edge_attr, batch)
        logits = self.classifier(graph_repr)
        return logits, graph_repr, node_repr


# ============================================================
# 3. Self-Supervised Pre-training
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
    
    mask = torch.rand(x.size(0)) > drop_rate
    x[~mask] = 0
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def save_model_safe(state_dict, path):
    """Save model state dict safely using pickle instead of torch.save to avoid zip issues."""
    # Convert tensors to numpy for safe serialization
    numpy_state = {}
    for k, v in state_dict.items():
        numpy_state[k] = v.cpu().numpy()
    with open(path, 'wb') as f:
        pickle.dump(numpy_state, f)


def load_model_safe(path):
    """Load model state dict from pickle format."""
    with open(path, 'rb') as f:
        numpy_state = pickle.load(f)
    state_dict = {}
    for k, v in numpy_state.items():
        state_dict[k] = torch.from_numpy(v)
    return state_dict


def pretrain_combined(encoder, dataset, epochs=80, batch_size=128, lr=1e-3,
                      temperature=0.07, device='cpu', save_path=None):
    encoder = encoder.to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    repr_dim = 128 * 3
    proj_head = nn.Sequential(
        nn.Linear(repr_dim, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Linear(256, 128),
    ).to(device)
    
    node_recon_head = nn.Sequential(
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 28),
    ).to(device)
    
    data_list = get_data_list(dataset)
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(epochs):
        encoder.train()
        proj_head.train()
        node_recon_head.train()
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
            
            repr1, node_repr1 = encoder(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
            repr2, node_repr2 = encoder(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
            
            z1 = proj_head(repr1)
            z2 = proj_head(repr2)
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            
            batch_size_curr = z1.size(0)
            z = torch.cat([z1, z2], dim=0)
            sim = torch.mm(z, z.t()) / temperature
            mask = torch.eye(2 * batch_size_curr, device=device).bool()
            sim.masked_fill_(mask, -1e9)
            labels = torch.cat([torch.arange(batch_size_curr, device=device) + batch_size_curr,
                               torch.arange(batch_size_curr, device=device)], dim=0)
            contrastive_loss = F.cross_entropy(sim, labels)
            
            recon_x = node_recon_head(node_repr1)
            recon_loss = F.mse_loss(recon_x, batch_orig.x)
            
            loss = contrastive_loss + 0.5 * recon_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + 
                                           list(proj_head.parameters()) +
                                           list(node_recon_head.parameters()), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                save_model_safe(encoder.state_dict(), save_path)
        
        if (epoch + 1) % 20 == 0:
            print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return loss_history


# ============================================================
# 4. Fine-tuning with Focal Loss and Oversampling
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


def finetune_classifier(model, dataset, epochs=200, batch_size=32, lr=2e-4,
                        device='cpu', save_path=None):
    model = model.to(device)
    
    data_list = get_data_list(dataset)
    labels = np.array([d.y.item() for d in data_list])
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    train_pos = pos_idx[:int(0.8 * len(pos_idx))]
    val_pos = pos_idx[int(0.8 * len(pos_idx)):]
    train_neg = neg_idx[:int(0.8 * len(neg_idx))]
    val_neg = neg_idx[int(0.8 * len(neg_idx)):]
    
    train_data = [data_list[i] for i in train_idx] if False else [data_list[i] for i in np.concatenate([train_pos, train_neg])]
    val_data = [data_list[i] for i in np.concatenate([val_pos, val_neg])]
    
    print(f"Train: {len(train_pos)} pos, {len(train_neg)} neg")
    print(f"Val: {len(val_pos)} pos, {len(val_neg)} neg")
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    best_f1 = 0
    best_ap = 0
    best_checkpoint_state = None
    best_threshold = 0.5
    train_metrics = []
    
    for epoch in range(epochs):
        model.train()
        
        num_pos = len(train_pos)
        num_neg = len(train_neg)
        oversample_ratio = min(num_neg // num_pos, 20)
        
        train_epoch_data = list(train_data)
        pos_data = [data_list[i] for i in train_pos]
        for _ in range(oversample_ratio - 1):
            train_epoch_data.extend(pos_data)
        
        np.random.shuffle(train_epoch_data)
        
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for i in range(0, len(train_epoch_data), batch_size):
            batch_data = train_epoch_data[i:i+batch_size]
            batch = Batch.from_data_list(batch_data).to(device)
            
            logits, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y.float().view(-1, 1)
            
            loss = criterion(logits, target)
            
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
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
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
        
        if (epoch + 1) % 25 == 0:
            print(f"Finetune Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, "
                  f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f} (thresh={bt:.2f}), "
                  f"Val AP: {val_ap:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}")
    
    # Save best model using pickle format
    if save_path and best_checkpoint_state is not None:
        save_data = {
            'model_state_dict': best_checkpoint_state,
            'best_threshold': float(best_threshold),
            'val_f1': float(best_f1),
            'val_ap': float(best_ap),
        }
        # Convert to numpy for safe serialization
        numpy_save = {}
        for k, v in save_data['model_state_dict'].items():
            numpy_save[k] = v.numpy()
        save_data['model_state_dict'] = numpy_save
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
    
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
# 6. Main Pipeline
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
    
    model = AltermagnetClassifier(
        node_dim=28, edge_dim=2, hidden_dim=128, num_layers=4, dropout=0.2
    )
    
    # Step 1: Pre-training
    print("\n=== Step 1: Self-supervised pre-training ===")
    pretrain_loss = pretrain_combined(
        model.encoder, pretrain_dataset, 
        epochs=80, batch_size=128, lr=1e-3,
        device=device, save_path='outputs/pretrained_encoder.pkl'
    )
    
    # Step 2: Fine-tuning
    print("\n=== Step 2: Fine-tuning for classification ===")
    encoder_state = load_model_safe('outputs/pretrained_encoder.pkl')
    model.encoder.load_state_dict(encoder_state)
    
    finetune_metrics = finetune_classifier(
        model, finetune_dataset,
        epochs=200, batch_size=32, lr=2e-4,
        device=device, save_path='outputs/best_model.pkl'
    )
    
    # Save metrics as JSON (small files)
    with open('outputs/finetune_metrics.json', 'w') as f:
        json.dump(finetune_metrics, f, indent=2)
    
    # Step 3: Prediction
    print("\n=== Step 3: Predicting candidates ===")
    with open('outputs/best_model.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Convert back to tensors
    state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        state_dict[k] = torch.from_numpy(v)
    model.load_state_dict(state_dict)
    best_threshold = checkpoint.get('best_threshold', 0.5)
    print(f"Using threshold: {best_threshold:.3f}")
    
    probs, reprs, true_labels = predict_candidates(
        model, candidate_dataset, device=device
    )
    
    # Save results as numpy (smaller than torch)
    np.save('outputs/candidate_probs.npy', probs)
    np.save('outputs/candidate_true_labels.npy', true_labels)
    # Save reprs as compressed
    np.savez_compressed('outputs/candidate_reprs.npz', reprs=reprs)
    
    # Evaluate
    pred_labels = (probs > best_threshold).astype(int)
    
    auc = roc_auc_score(true_labels, probs)
    ap = average_precision_score(true_labels, probs)
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    
    print(f"\n=== Candidate Evaluation (threshold={best_threshold:.3f}) ===")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(true_labels, pred_labels)}")
    
    # Also at 0.5
    pred_labels_05 = (probs > 0.5).astype(int)
    f1_05 = f1_score(true_labels, pred_labels_05)
    prec_05 = precision_score(true_labels, pred_labels_05, zero_division=0)
    rec_05 = recall_score(true_labels, pred_labels_05, zero_division=0)
    print(f"\n=== At threshold=0.5 ===")
    print(f"F1: {f1_05:.4f}, Precision: {prec_05:.4f}, Recall: {rec_05:.4f}")
    
    # Save evaluation results
    eval_results = {
        'auc_roc': float(auc),
        'average_precision': float(ap),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'best_threshold': float(best_threshold),
        'num_true_positives': int(true_labels.sum()),
        'num_predicted_positives': int(pred_labels.sum()),
        'num_candidates': len(true_labels),
        'f1_at_0.5': float(f1_05),
        'precision_at_0.5': float(prec_05),
        'recall_at_0.5': float(rec_05),
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
        })
    with open('outputs/top_candidates.json', 'w') as f:
        json.dump(top_results, f, indent=2)
    
    # Discovery metrics
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
            'false_negatives': fn, 'true_negatives': tn,
            'precision': float(prec), 'recall': float(rec), 'f1': float(f1_t),
        }
    with open('outputs/discovery_metrics.json', 'w') as f:
        json.dump(discovery_metrics, f, indent=2)
    
    # Top-K metrics
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
    with open('outputs/topk_metrics.json', 'w') as f:
        json.dump(topk_metrics, f, indent=2)
    
    print(f"\nTop-K Metrics:")
    for k, v in topk_metrics.items():
        print(f"  {k}: TP={v['true_positives_in_top_k']}, Precision@K={v['precision_at_k']:.4f}, Recall@K={v['recall_at_k']:.4f}")
    
    # Save pretrain loss
    with open('outputs/pretrain_loss.json', 'w') as f:
        json.dump(pretrain_loss, f)
    
    print("\nDone!")
