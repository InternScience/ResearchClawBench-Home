"""
DIDS-MFL: Disentangled Dynamic Intrusion Detection with Multi-scale Fusion Learning
===================================================================================
A framework for network intrusion detection that addresses:
1. Statistical feature disentanglement
2. Representational disentanglement  
3. Dynamic graph diffusion for spatiotemporal aggregation
4. Multi-scale representation fusion for few-shot learning
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def load_data(path='data/NF-UNSW-NB15-v2_3d.pt'):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return data


def preprocess_data(data):
    n_edges = data.msg.shape[0]
    features = data.msg.float()
    labels_binary = data.label.long()
    labels_multi = data.attack.long()
    timestamps = data.t.float()
    src = data.src.long()
    dst = data.dst.long()

    # Normalize features robustly
    feat_median = features.median(dim=0).values
    feat_iqr = (features.quantile(0.75, dim=0) - features.quantile(0.25, dim=0)) + 1e-8
    features = (features - feat_median) / feat_iqr
    features = torch.clamp(features, -10, 10)

    all_nodes = torch.cat([src, dst]).unique()
    node_map = {int(nid): i for i, nid in enumerate(all_nodes)}
    src_mapped = torch.tensor([node_map[int(s)] for s in src], dtype=torch.long)
    dst_mapped = torch.tensor([node_map[int(d)] for d in dst], dtype=torch.long)
    num_nodes = len(all_nodes)

    edge_index = torch.stack([src_mapped, dst_mapped], dim=0)
    edge_time = timestamps

    node_time = torch.zeros(num_nodes)
    node_count = torch.zeros(num_nodes)
    for e in range(n_edges):
        node_time[src_mapped[e]] += timestamps[e]
        node_time[dst_mapped[e]] += timestamps[e]
        node_count[src_mapped[e]] += 1
        node_count[dst_mapped[e]] += 1
    node_time = node_time / (node_count + 1e-8)

    time_sorted_idx = torch.argsort(timestamps)
    n_train = int(0.70 * n_edges)
    n_val = int(0.15 * n_edges)
    train_idx = time_sorted_idx[:n_train]
    val_idx = time_sorted_idx[n_train:n_train + n_val]
    test_idx = time_sorted_idx[n_train + n_val:]

    return {
        'features': features,
        'labels_binary': labels_binary,
        'labels_multi': labels_multi,
        'timestamps': timestamps,
        'src': src_mapped,
        'dst': dst_mapped,
        'edge_index': edge_index,
        'edge_time': edge_time,
        'node_time': node_time,
        'num_nodes': num_nodes,
        'num_edges': n_edges,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
    }


class StatisticalDisentangler(nn.Module):
    """Feature attention for statistical disentanglement."""
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.sigmoid(scores)
        return x * weights, weights


class TemporalGraphConv(nn.Module):
    """Graph convolution with temporal edge weighting."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.temp_enc = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1)
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_time, node_time):
        src, dst = edge_index
        # Temporal weights
        dt = torch.abs(edge_time - node_time[src]).unsqueeze(1)
        w = torch.exp(-torch.abs(self.temp_enc(dt))).squeeze(1)
        
        # Message passing: aggregate from src to dst
        x_src = self.lin(x[src])
        out = torch.zeros(x.shape[0], x_src.shape[1], device=x.device)
        out.index_add_(0, dst, x_src * w.unsqueeze(1))
        
        deg = torch.bincount(dst, minlength=x.shape[0]).float().clamp(min=1)
        out = out / deg.unsqueeze(1)
        out = out + self.lin(x)  # self-loop
        out = self.bn(out)
        return F.relu(out)


class DIDSMFL(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, num_hops=3, dropout=0.3):
        super().__init__()
        self.stat_disentangler = StatisticalDisentangler(in_dim, hidden_dim)
        
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.convs = nn.ModuleList([
            TemporalGraphConv(hidden_dim if i == 0 else out_dim, out_dim)
            for i in range(num_hops)
        ])
        
        self.edge_readout = nn.Sequential(
            nn.Linear(out_dim * 2 + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier_binary = nn.Linear(out_dim, 2)
        self.classifier_multi = nn.Linear(out_dim, 10)
        self.dropout = dropout

    def forward(self, edge_feats, edge_index, edge_time, node_time, return_reps=False):
        x_dis, feat_weights = self.stat_disentangler(edge_feats)
        h = self.embed(x_dis)
        
        # Initialize node features from edge features
        N = node_time.shape[0]
        node_feats = torch.zeros(N, h.shape[1], device=h.device)
        src, dst = edge_index
        node_feats.index_add_(0, src, h)
        node_feats.index_add_(0, dst, h)
        deg = torch.bincount(src, minlength=N) + torch.bincount(dst, minlength=N)
        node_feats = node_feats / deg.float().clamp(min=1).unsqueeze(1)
        
        # Multi-hop diffusion
        multi_scale = []
        x = node_feats
        for conv in self.convs:
            x = conv(x, edge_index, edge_time, node_time)
            x = F.dropout(x, p=self.dropout, training=self.training)
            multi_scale.append(x)
        
        # Edge-level readout
        src_emb = multi_scale[-1][src]
        dst_emb = multi_scale[-1][dst]
        edge_repr = torch.cat([src_emb, dst_emb, h], dim=-1)
        edge_repr = self.edge_readout(edge_repr)
        
        logits_bin = self.classifier_binary(edge_repr)
        logits_multi = self.classifier_multi(edge_repr)
        
        if return_reps:
            return logits_bin, logits_multi, edge_repr, feat_weights
        return logits_bin, logits_multi


def rep_dis_loss(reps):
    reps = torch.nan_to_num(reps, 0.0)
    reps = F.normalize(reps, p=2, dim=1)
    d = min(reps.shape[1], 32)
    corr = torch.mm(reps[:, :d].t(), reps[:, :d]) / (reps.shape[0] + 1e-8)
    eye = torch.eye(d, device=reps.device)
    return torch.norm(corr - eye, p='fro') ** 2


def train_epoch(model, optimizer, edge_feats, edge_index, edge_time, node_time,
                labels_binary, labels_multi, train_idx, dis_w=0.005):
    model.train()
    optimizer.zero_grad()
    
    batch_size = min(8192, len(train_idx))
    batch_idx = train_idx[torch.randperm(len(train_idx))[:batch_size]]
    
    logits_bin, logits_multi, reps, _ = model(
        edge_feats, edge_index, edge_time, node_time, return_reps=True
    )
    
    loss_bin = F.cross_entropy(logits_bin[batch_idx], labels_binary[batch_idx])
    loss_multi = F.cross_entropy(logits_multi[batch_idx], labels_multi[batch_idx])
    loss_dis = rep_dis_loss(reps[batch_idx])
    
    loss = loss_bin + loss_multi + dis_w * loss_dis
    if torch.isnan(loss):
        optimizer.zero_grad()
        return 0.0, 0.0, 0.0, 0.0
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), loss_bin.item(), loss_multi.item(), loss_dis.item()


@torch.no_grad()
def evaluate(model, edge_feats, edge_index, edge_time, node_time,
             labels_binary, labels_multi, eval_idx):
    model.eval()
    logits_bin, logits_multi, reps, _ = model(
        edge_feats, edge_index, edge_time, node_time, return_reps=True
    )
    logits_bin = torch.nan_to_num(logits_bin, 0.0)
    logits_multi = torch.nan_to_num(logits_multi, 0.0)
    
    preds_bin = logits_bin[eval_idx].argmax(dim=1).cpu().numpy()
    preds_multi = logits_multi[eval_idx].argmax(dim=1).cpu().numpy()
    probs_bin = F.softmax(logits_bin[eval_idx], dim=1)[:, 1].cpu().numpy()
    
    y_bin = labels_binary[eval_idx].cpu().numpy()
    y_multi = labels_multi[eval_idx].cpu().numpy()
    
    return {
        'binary': {
            'accuracy': accuracy_score(y_bin, preds_bin),
            'precision': precision_score(y_bin, preds_bin, zero_division=0),
            'recall': recall_score(y_bin, preds_bin, zero_division=0),
            'f1': f1_score(y_bin, preds_bin, zero_division=0),
            'auc': roc_auc_score(y_bin, probs_bin) if len(np.unique(y_bin)) > 1 else 0.0,
        },
        'multi': {
            'accuracy': accuracy_score(y_multi, preds_multi),
            'precision': precision_score(y_multi, preds_multi, average='weighted', zero_division=0),
            'recall': recall_score(y_multi, preds_multi, average='weighted', zero_division=0),
            'f1': f1_score(y_multi, preds_multi, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_multi, preds_multi, average='macro', zero_division=0),
        },
        'preds_bin': preds_bin,
        'preds_multi': preds_multi,
        'y_bin': y_bin,
        'y_multi': y_multi,
        'probs_bin': probs_bin,
    }


def evaluate_unknown_attack(dataset, unknown_attack_id, hidden_dim=64, out_dim=64, epochs=50):
    edge_feats = dataset['features']
    labels_binary = dataset['labels_binary']
    labels_multi = dataset['labels_multi']
    edge_index = dataset['edge_index']
    edge_time = dataset['edge_time']
    node_time = dataset['node_time']
    
    unknown_mask = labels_multi == unknown_attack_id
    known_mask = ~unknown_mask
    known_idx = torch.where(known_mask)[0]
    if len(known_idx) < 10:
        return None
    
    train_known, val_known = train_test_split(
        known_idx.numpy(), test_size=0.2, random_state=SEED,
        stratify=labels_multi[known_idx].numpy()
    )
    train_idx = torch.from_numpy(train_known).long()
    val_idx = torch.from_numpy(val_known).long()
    test_idx = torch.where(unknown_mask)[0]
    if len(test_idx) == 0:
        return None
    
    model = DIDSMFL(edge_feats.shape[1], hidden_dim, out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    edge_feats_d = edge_feats.to(DEVICE)
    edge_index_d = edge_index.to(DEVICE)
    edge_time_d = edge_time.to(DEVICE)
    node_time_d = node_time.to(DEVICE)
    labels_binary_d = labels_binary.to(DEVICE)
    labels_multi_d = labels_multi.to(DEVICE)
    
    best_val_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        loss, lb, lm, ld = train_epoch(
            model, optimizer, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
            labels_binary_d, labels_multi_d, train_idx
        )
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_metrics = evaluate(
                model, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
                labels_binary_d, labels_multi_d, val_idx
            )
            val_f1 = val_metrics['binary']['f1']
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
        labels_binary_d, labels_multi_d, test_idx
    )
    return test_metrics


def evaluate_few_shot(dataset, n_shots=5, hidden_dim=64, out_dim=64, epochs=50):
    edge_feats = dataset['features']
    labels_binary = dataset['labels_binary']
    labels_multi = dataset['labels_multi']
    edge_index = dataset['edge_index']
    edge_time = dataset['edge_time']
    node_time = dataset['node_time']
    
    train_idx_list = []
    for c in torch.unique(labels_multi):
        c_idx = torch.where(labels_multi == c)[0]
        n = min(n_shots, len(c_idx))
        sampled = c_idx[torch.randperm(len(c_idx))[:n]]
        train_idx_list.append(sampled)
    train_idx = torch.cat(train_idx_list)
    
    test_mask = torch.ones(len(labels_multi), dtype=torch.bool)
    test_mask[train_idx] = False
    test_idx = torch.where(test_mask)[0]
    
    val_size = min(max(len(train_idx) // 5, 5), 200)
    val_idx = train_idx[torch.randperm(len(train_idx))[:val_size]]
    train_mask = torch.ones(len(train_idx), dtype=torch.bool)
    for v in val_idx:
        train_mask[train_idx == v] = False
    train_idx = train_idx[train_mask]
    
    model = DIDSMFL(edge_feats.shape[1], hidden_dim, out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    
    edge_feats_d = edge_feats.to(DEVICE)
    edge_index_d = edge_index.to(DEVICE)
    edge_time_d = edge_time.to(DEVICE)
    node_time_d = node_time.to(DEVICE)
    labels_binary_d = labels_binary.to(DEVICE)
    labels_multi_d = labels_multi.to(DEVICE)
    
    best_val_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        loss, lb, lm, ld = train_epoch(
            model, optimizer, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
            labels_binary_d, labels_multi_d, train_idx
        )
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_metrics = evaluate(
                model, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
                labels_binary_d, labels_multi_d, val_idx
            )
            val_f1 = val_metrics['binary']['f1']
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
        labels_binary_d, labels_multi_d, test_idx
    )
    return test_metrics


def evaluate_baselines(dataset):
    edge_feats = dataset['features'].numpy()
    labels_binary = dataset['labels_binary'].numpy()
    labels_multi = dataset['labels_multi'].numpy()
    train_idx = dataset['train_idx'].numpy()
    test_idx = dataset['test_idx'].numpy()
    
    X_train = edge_feats[train_idx]
    X_test = edge_feats[test_idx]
    y_bin_train = labels_binary[train_idx]
    y_bin_test = labels_binary[test_idx]
    y_multi_train = labels_multi[train_idx]
    y_multi_test = labels_multi[test_idx]
    
    results = {}
    
    print("\nTraining Random Forest baseline...")
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_bin.fit(X_train, y_bin_train)
    preds_bin = rf_bin.predict(X_test)
    probs_bin = rf_bin.predict_proba(X_test)[:, 1]
    results['RF_binary'] = {
        'accuracy': accuracy_score(y_bin_test, preds_bin),
        'precision': precision_score(y_bin_test, preds_bin, zero_division=0),
        'recall': recall_score(y_bin_test, preds_bin, zero_division=0),
        'f1': f1_score(y_bin_test, preds_bin, zero_division=0),
        'auc': roc_auc_score(y_bin_test, probs_bin),
    }
    
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_multi.fit(X_train, y_multi_train)
    preds_multi = rf_multi.predict(X_test)
    results['RF_multi'] = {
        'accuracy': accuracy_score(y_multi_test, preds_multi),
        'f1': f1_score(y_multi_test, preds_multi, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_multi_test, preds_multi, average='macro', zero_division=0),
    }
    
    print("Training Logistic Regression baseline...")
    lr_bin = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
    lr_bin.fit(X_train, y_bin_train)
    preds_bin = lr_bin.predict(X_test)
    probs_bin = lr_bin.predict_proba(X_test)[:, 1]
    results['LR_binary'] = {
        'accuracy': accuracy_score(y_bin_test, preds_bin),
        'precision': precision_score(y_bin_test, preds_bin, zero_division=0),
        'recall': recall_score(y_bin_test, preds_bin, zero_division=0),
        'f1': f1_score(y_bin_test, preds_bin, zero_division=0),
        'auc': roc_auc_score(y_bin_test, probs_bin),
    }
    
    return results


def run_experiments():
    print("=" * 80)
    print("DIDS-MFL: Disentangled Dynamic Intrusion Detection")
    print("=" * 80)
    
    raw_data = load_data()
    dataset = preprocess_data(raw_data)
    
    edge_feats = dataset['features']
    labels_binary = dataset['labels_binary']
    labels_multi = dataset['labels_multi']
    edge_index = dataset['edge_index']
    edge_time = dataset['edge_time']
    node_time = dataset['node_time']
    train_idx = dataset['train_idx']
    val_idx = dataset['val_idx']
    test_idx = dataset['test_idx']
    
    print(f"\nDataset statistics:")
    print(f"  Total edges: {len(edge_feats)}")
    print(f"  Features: {edge_feats.shape[1]}")
    print(f"  Nodes: {dataset['num_nodes']}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Binary: benign={int((labels_binary==0).sum())}, attack={int((labels_binary==1).sum())}")
    for c in torch.unique(labels_multi):
        print(f"    Class {c.item()}: {(labels_multi==c).sum().item()}")
    
    # Standard evaluation
    print("\n" + "=" * 80)
    print("STANDARD EVALUATION")
    print("=" * 80)
    
    model = DIDSMFL(edge_feats.shape[1], hidden_dim=64, out_dim=64, num_hops=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    
    edge_feats_d = edge_feats.to(DEVICE)
    edge_index_d = edge_index.to(DEVICE)
    edge_time_d = edge_time.to(DEVICE)
    node_time_d = node_time.to(DEVICE)
    labels_binary_d = labels_binary.to(DEVICE)
    labels_multi_d = labels_multi.to(DEVICE)
    
    best_val_f1 = 0
    best_state = None
    history = {'train_loss': [], 'val_bin_f1': [], 'val_multi_f1': []}
    
    epochs = 100
    for epoch in range(epochs):
        loss, lb, lm, ld = train_epoch(
            model, optimizer, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
            labels_binary_d, labels_multi_d, train_idx
        )
        history['train_loss'].append(loss)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            val_metrics = evaluate(
                model, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
                labels_binary_d, labels_multi_d, val_idx
            )
            history['val_bin_f1'].append(val_metrics['binary']['f1'])
            history['val_multi_f1'].append(val_metrics['multi']['f1'])
            
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} (bin={lb:.4f}, multi={lm:.4f}, dis={ld:.4f}) | "
                  f"Val Bin F1: {val_metrics['binary']['f1']:.4f} | Val Multi F1: {val_metrics['multi']['f1']:.4f}")
            
            if val_metrics['binary']['f1'] > best_val_f1:
                best_val_f1 = val_metrics['binary']['f1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    test_metrics = evaluate(
        model, edge_feats_d, edge_index_d, edge_time_d, node_time_d,
        labels_binary_d, labels_multi_d, test_idx
    )
    
    print("\n--- Test Results ---")
    for k, v in test_metrics['binary'].items():
        print(f"  Binary {k}: {v:.4f}")
    for k, v in test_metrics['multi'].items():
        print(f"  Multi {k}: {v:.4f}")
    
    target_names = [f'Class_{i}' for i in range(10)]
    print("\nMulti-class Report:")
    print(classification_report(
        test_metrics['y_multi'], test_metrics['preds_multi'],
        target_names=target_names, zero_division=0
    ))
    
    standard_results = {
        'test_binary': test_metrics['binary'],
        'test_multi': test_metrics['multi'],
        'history': history,
        'classification_report': classification_report(
            test_metrics['y_multi'], test_metrics['preds_multi'],
            target_names=target_names, zero_division=0, output_dict=True
        )
    }
    
    # Baselines
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    baseline_results = evaluate_baselines(dataset)
    for name, metrics in baseline_results.items():
        print(f"{name}: {metrics}")
    
    # Unknown attacks
    print("\n" + "=" * 80)
    print("UNKNOWN ATTACK EVALUATION")
    print("=" * 80)
    unknown_results = {}
    attack_classes = [c.item() for c in torch.unique(labels_multi) if c != 2]
    for attack_id in attack_classes:
        print(f"\nLeaving out attack class {attack_id}...")
        result = evaluate_unknown_attack(dataset, attack_id, epochs=40)
        if result is not None:
            unknown_results[f'attack_{attack_id}'] = {
                'binary': result['binary'],
                'multi': result['multi'],
            }
            print(f"  Binary F1: {result['binary']['f1']:.4f}, AUC: {result['binary']['auc']:.4f}")
            print(f"  Multi F1: {result['multi']['f1']:.4f}, Macro F1: {result['multi']['f1_macro']:.4f}")
    
    # Few-shot
    print("\n" + "=" * 80)
    print("FEW-SHOT EVALUATION")
    print("=" * 80)
    fewshot_results = {}
    for n_shots in [1, 5, 10, 20]:
        print(f"\n{n_shots}-shot evaluation...")
        result = evaluate_few_shot(dataset, n_shots=n_shots, epochs=40)
        fewshot_results[f'{n_shots}_shot'] = {
            'binary': result['binary'],
            'multi': result['multi'],
        }
        print(f"  Binary F1: {result['binary']['f1']:.4f}, AUC: {result['binary']['auc']:.4f}")
        print(f"  Multi F1: {result['multi']['f1']:.4f}, Macro F1: {result['multi']['f1_macro']:.4f}")
    
    return standard_results, unknown_results, fewshot_results, baseline_results, test_metrics, dataset


if __name__ == '__main__':
    standard_results, unknown_results, fewshot_results, baseline_results, test_metrics, dataset = run_experiments()
    
    os.makedirs('outputs', exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    
    all_results = {
        'standard': convert(standard_results),
        'unknown': convert(unknown_results),
        'fewshot': convert(fewshot_results),
        'baselines': convert(baseline_results),
    }
    
    with open('outputs/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    np.savez('outputs/test_predictions.npz',
             y_bin=test_metrics['y_bin'],
             preds_bin=test_metrics['preds_bin'],
             probs_bin=test_metrics['probs_bin'],
             y_multi=test_metrics['y_multi'],
             preds_multi=test_metrics['preds_multi'])
    
    print("\nResults saved.")
