"""
Training and Evaluation Pipeline for DIDS-MFL

Handles:
1. Data loading and preprocessing
2. Train/validation/test splitting with temporal ordering
3. Known attack evaluation
4. Unknown attack evaluation (zero-shot generalization)
5. Few-shot attack evaluation
6. Baseline comparisons
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

ATTACK_NAMES = {
    0: 'Analysis', 1: 'Backdoor', 2: 'Benign', 3: 'DoS',
    4: 'Exploits', 5: 'Fuzzers', 6: 'Generic', 7: 'Reconnaissance',
    8: 'Shellcode', 9: 'Worms'
}


def to_np(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.array(x)


def load_and_preprocess_data(data_path):
    print("Loading data...")
    data = torch.load(data_path, weights_only=False)
    store = data.stores[0]
    
    msg = store['msg']
    label = store['label']
    attack = store['attack']
    src = store['src']
    dst = store['dst']
    t = store['t']
    
    all_nodes = torch.cat([src, dst])
    unique_nodes = torch.unique(all_nodes)
    node_map = torch.zeros(unique_nodes.max().item() + 1, dtype=torch.long)
    node_map[unique_nodes] = torch.arange(len(unique_nodes))
    new_src = node_map[src]
    new_dst = node_map[dst]
    
    edge_index = torch.stack([new_src, new_dst], dim=0)
    
    feature_mean = msg.mean(dim=0)
    feature_std = msg.std(dim=0) + 1e-8
    msg_norm = (msg - feature_mean) / feature_std
    
    print(f"Data loaded: {msg.shape[0]} samples, {msg.shape[1]} features")
    print(f"Nodes: {len(unique_nodes)}, Edges: {edge_index.shape[1]}")
    
    return {
        'x': msg_norm,
        'y_binary': label,
        'y_multi': attack,
        'edge_index': edge_index,
        't': t,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'num_nodes': len(unique_nodes),
    }


def temporal_split(data_dict, train_ratio=0.6, val_ratio=0.2):
    t = data_dict['t']
    n = len(t)
    sorted_idx = torch.argsort(t)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_idx = sorted_idx[:train_end]
    val_idx = sorted_idx[train_end:val_end]
    test_idx = sorted_idx[val_end:]
    return train_idx, val_idx, test_idx


class EdgeClassifier(nn.Module):
    def __init__(self, feature_dim=40, hidden_dim=128, num_classes=10, num_factors=8):
        super().__init__()
        self.num_factors = num_factors
        
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.factor_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_factors),
        )
        
        self.representation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ) for _ in range(3)
        ])
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1),
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.multiclass_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
    def forward(self, x, edge_index=None, t=None):
        h = self.encoder(x)
        factor_weights = F.softmax(self.factor_encoder(h), dim=-1)
        rep = self.representation(h)
        
        scale_features = [branch(rep) for branch in self.scale_branches]
        concat = torch.cat(scale_features, dim=-1)
        gates = self.fusion_gate(concat)
        stacked = torch.stack(scale_features, dim=1)
        fused = (gates.unsqueeze(-1) * stacked).sum(dim=1)
        
        binary_logit = self.binary_head(fused)
        multiclass_logit = self.multiclass_head(fused)
        
        return binary_logit, multiclass_logit, factor_weights, rep
    
    def compute_loss(self, binary_logit, multiclass_logit, factor_weights, rep,
                     binary_labels, multiclass_labels, alpha_mi=0.1, alpha_dec=0.05):
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logit.squeeze(-1), binary_labels.float()
        )
        multiclass_loss = F.cross_entropy(multiclass_logit, multiclass_labels)
        
        batch_size = factor_weights.size(0)
        mean_w = factor_weights.mean(dim=0, keepdim=True)
        centered = factor_weights - mean_w
        corr = torch.matmul(centered.t(), centered) / batch_size
        mask = ~torch.eye(corr.size(0), device=corr.device).bool()
        mi_loss = (corr[mask] ** 2).mean()
        
        rep_norm = F.normalize(rep, p=2, dim=-1)
        mean_r = rep_norm.mean(dim=0, keepdim=True)
        c_rep = rep_norm - mean_r
        corr_rep = torch.matmul(c_rep.t(), c_rep) / (batch_size - 1)
        dim = corr_rep.size(0)
        mask_r = ~torch.eye(dim, device=corr_rep.device).bool()
        dec_loss = (corr_rep[mask_r] ** 2).mean()
        
        total_loss = (binary_loss + multiclass_loss + 
                     alpha_mi * mi_loss + alpha_dec * dec_loss)
        
        return total_loss, {
            'binary': binary_loss.item(),
            'multiclass': multiclass_loss.item(),
            'mi_reg': mi_loss.item(),
            'dec_reg': dec_loss.item(),
        }


def train_model(model, data_dict, train_idx, val_idx, 
                epochs=50, batch_size=2048, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    x = data_dict['x'].to(device)
    y_binary = data_dict['y_binary'].to(device)
    y_multi = data_dict['y_multi'].to(device)
    
    best_val_f1 = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_f1_binary': [], 'val_f1_multi': []}
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_idx), device=device)
        train_shuffled = train_idx[perm]
        
        epoch_loss = 0
        n_batches = 0
        
        for start in range(0, len(train_shuffled), batch_size):
            end = min(start + batch_size, len(train_shuffled))
            batch_idx = train_shuffled[start:end]
            
            batch_x = x[batch_idx]
            batch_y_bin = y_binary[batch_idx]
            batch_y_multi = y_multi[batch_idx]
            
            optimizer.zero_grad()
            binary_logit, multiclass_logit, factor_weights, rep = model(batch_x)
            
            loss, loss_dict = model.compute_loss(
                binary_logit, multiclass_logit, factor_weights, rep,
                batch_y_bin, batch_y_multi
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_x = x[val_idx]
            val_y_bin = y_binary[val_idx]
            val_y_multi = y_multi[val_idx]
            
            val_bin_logit, val_multi_logit, _, _ = model(val_x)
            
            val_pred_bin = (val_bin_logit.squeeze(-1) > 0).long()
            val_pred_multi = val_multi_logit.argmax(dim=-1)
            
            val_f1_bin = f1_score(val_y_bin.cpu(), val_pred_bin.cpu(), average='binary')
            val_f1_multi = f1_score(val_y_multi.cpu(), val_pred_multi.cpu(), 
                                     average='weighted', zero_division=0)
            val_loss = epoch_loss / max(n_batches, 1)
        
        history['train_loss'].append(epoch_loss / max(n_batches, 1))
        history['val_loss'].append(val_loss)
        history['val_f1_binary'].append(val_f1_bin)
        history['val_f1_multi'].append(val_f1_multi)
        
        if val_f1_multi > best_val_f1:
            best_val_f1 = val_f1_multi
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_loss/max(n_batches,1):.4f} | "
                  f"Val F1 Binary: {val_f1_bin:.4f} | "
                  f"Val F1 Multi: {val_f1_multi:.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


def evaluate_model(model, data_dict, test_idx, device='cpu'):
    model.eval()
    x = data_dict['x'].to(device)
    y_binary = data_dict['y_binary'].to(device)
    y_multi = data_dict['y_multi'].to(device)
    
    with torch.no_grad():
        test_x = x[test_idx]
        test_y_bin = y_binary[test_idx]
        test_y_multi = y_multi[test_idx]
        
        bin_logit, multi_logit, _, _ = model(test_x)
        
        pred_bin = (bin_logit.squeeze(-1) > 0).long()
        pred_multi = multi_logit.argmax(dim=-1)
        
        results = {
            'binary': {
                'accuracy': accuracy_score(test_y_bin.cpu(), pred_bin.cpu()),
                'precision': precision_score(test_y_bin.cpu(), pred_bin.cpu(), zero_division=0),
                'recall': recall_score(test_y_bin.cpu(), pred_bin.cpu(), zero_division=0),
                'f1': f1_score(test_y_bin.cpu(), pred_bin.cpu(), zero_division=0),
            },
            'multiclass': {
                'accuracy': accuracy_score(test_y_multi.cpu(), pred_multi.cpu()),
                'f1_weighted': f1_score(test_y_multi.cpu(), pred_multi.cpu(), 
                                        average='weighted', zero_division=0),
                'f1_macro': f1_score(test_y_multi.cpu(), pred_multi.cpu(), 
                                     average='macro', zero_division=0),
            }
        }
        
        per_class_f1 = f1_score(test_y_multi.cpu(), pred_multi.cpu(), 
                                average=None, zero_division=0)
        results['per_class_f1'] = {}
        for cls_idx, f1_val in enumerate(per_class_f1):
            cls_name = ATTACK_NAMES.get(cls_idx, f'Class_{cls_idx}')
            results['per_class_f1'][cls_name] = float(f1_val)
        
        cm = confusion_matrix(test_y_multi.cpu(), pred_multi.cpu())
        results['confusion_matrix'] = cm.tolist()
        
        try:
            prob_bin = torch.sigmoid(bin_logit.squeeze(-1)).cpu()
            results['binary']['roc_auc'] = roc_auc_score(test_y_bin.cpu(), prob_bin)
        except:
            results['binary']['roc_auc'] = 0.0
        
    return results


def create_unknown_attack_split(data_dict, train_idx, val_idx, test_idx, 
                                 unknown_classes=[0, 1, 9]):
    y_multi = data_dict['y_multi']
    mask_train_known = ~torch.isin(y_multi[train_idx], torch.tensor(unknown_classes))
    train_known_idx = train_idx[mask_train_known]
    mask_test_unknown = torch.isin(y_multi[test_idx], torch.tensor(unknown_classes))
    test_unknown_idx = test_idx[mask_test_unknown]
    test_known_idx = test_idx[~mask_test_unknown]
    return train_known_idx, test_known_idx, test_unknown_idx


def create_few_shot_split(data_dict, train_idx, val_idx, test_idx,
                           few_shot_classes=[1, 8, 9], shots_per_class=10):
    y_multi = data_dict['y_multi']
    few_shot_train = []
    normal_train = []
    
    for idx in train_idx:
        cls = y_multi[idx].item()
        if cls in few_shot_classes:
            few_shot_train.append(idx)
        else:
            normal_train.append(idx)
    
    few_shot_selected = []
    for cls in few_shot_classes:
        cls_indices = [i for i in few_shot_train if y_multi[i].item() == cls]
        if len(cls_indices) > shots_per_class:
            selected = random.sample(cls_indices, shots_per_class)
        else:
            selected = cls_indices
        few_shot_selected.extend(selected)
    
    fs_train_idx = torch.tensor(normal_train + few_shot_selected, dtype=torch.long)
    return fs_train_idx


def train_svm_baseline(data_dict, train_idx, test_idx):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    
    x_np = to_np(data_dict['x'])
    y_bin = to_np(data_dict['y_binary'])
    y_multi = to_np(data_dict['y_multi'])
    train_np = to_np(train_idx).astype(int)
    test_np = to_np(test_idx).astype(int)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_np[train_np])
    X_test = scaler.transform(x_np[test_np])
    
    svm_bin = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_bin.fit(X_train, y_bin[train_np])
    pred_bin = svm_bin.predict(X_test)
    
    max_train = 15000
    if len(train_np) > max_train:
        sample_idx = np.random.choice(len(train_np), max_train, replace=False)
        X_train_sub = X_train[sample_idx]
        y_train_sub = y_multi[train_np[sample_idx]]
    else:
        X_train_sub = X_train
        y_train_sub = y_multi[train_np]
    
    svm_multi = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_multi.fit(X_train_sub, y_train_sub)
    pred_multi = svm_multi.predict(X_test)
    
    results = {
        'binary': {
            'accuracy': accuracy_score(y_bin[test_np], pred_bin),
            'precision': precision_score(y_bin[test_np], pred_bin, zero_division=0),
            'recall': recall_score(y_bin[test_np], pred_bin, zero_division=0),
            'f1': f1_score(y_bin[test_np], pred_bin, zero_division=0),
        },
        'multiclass': {
            'accuracy': accuracy_score(y_multi[test_np], pred_multi),
            'f1_weighted': f1_score(y_multi[test_np], pred_multi, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_multi[test_np], pred_multi, average='macro', zero_division=0),
        },
        'per_class_f1': {},
    }
    
    per_class_f1 = f1_score(y_multi[test_np], pred_multi, average=None, zero_division=0)
    for cls_idx, f1_val in enumerate(per_class_f1):
        cls_name = ATTACK_NAMES.get(cls_idx, f'Class_{cls_idx}')
        results['per_class_f1'][cls_name] = float(f1_val)
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dict = load_and_preprocess_data('data/NF-UNSW-NB15-v2_3d.pt')
    train_idx, val_idx, test_idx = temporal_split(data_dict)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    os.makedirs('outputs', exist_ok=True)
    
    # ====== Main Model Training ======
    print("\n=== Training DIDS-MFL ===")
    model = EdgeClassifier(
        feature_dim=data_dict['x'].shape[1],
        hidden_dim=128,
        num_classes=10,
    )
    
    model, history = train_model(
        model, data_dict, train_idx, val_idx,
        epochs=60, batch_size=4096, lr=1e-3, device=device
    )
    
    with open('outputs/training_history.json', 'w') as f:
        json.dump(history, f)
    
    # ====== Evaluate on Normal Test Set ======
    print("\n=== Evaluating DIDS-MFL (Normal Test) ===")
    results_normal = evaluate_model(model, data_dict, test_idx, device)
    print(f"Binary F1: {results_normal['binary']['f1']:.4f}")
    print(f"Multi-class F1 (weighted): {results_normal['multiclass']['f1_weighted']:.4f}")
    print(f"Multi-class F1 (macro): {results_normal['multiclass']['f1_macro']:.4f}")
    print("Per-class F1:", results_normal['per_class_f1'])
    
    # ====== Unknown Attack Evaluation ======
    print("\n=== Evaluating Unknown Attacks ===")
    unknown_classes = [0, 1, 9]
    train_known, test_known, test_unknown = create_unknown_attack_split(
        data_dict, train_idx, val_idx, test_idx, unknown_classes
    )
    
    model_unknown = EdgeClassifier(
        feature_dim=data_dict['x'].shape[1],
        hidden_dim=128,
        num_classes=10,
    )
    model_unknown, _ = train_model(
        model_unknown, data_dict, train_known, val_idx,
        epochs=40, batch_size=4096, lr=1e-3, device=device
    )
    
    if len(test_unknown) > 0:
        results_unknown = evaluate_model(model_unknown, data_dict, test_unknown, device)
        print(f"Unknown Attack Binary F1: {results_unknown['binary']['f1']:.4f}")
        print(f"Unknown Attack Multi-class F1: {results_unknown['multiclass']['f1_macro']:.4f}")
    else:
        results_unknown = {'binary': {'f1': 0}, 'multiclass': {'f1_macro': 0}}
    
    # ====== Few-shot Evaluation ======
    print("\n=== Evaluating Few-shot Attacks ===")
    few_shot_classes = [1, 8, 9]
    fs_train_idx = create_few_shot_split(
        data_dict, train_idx, val_idx, test_idx,
        few_shot_classes=few_shot_classes, shots_per_class=15
    )
    
    model_fs = EdgeClassifier(
        feature_dim=data_dict['x'].shape[1],
        hidden_dim=128,
        num_classes=10,
    )
    model_fs, _ = train_model(
        model_fs, data_dict, fs_train_idx, val_idx,
        epochs=40, batch_size=4096, lr=1e-3, device=device
    )
    
    results_fewshot = evaluate_model(model_fs, data_dict, test_idx, device)
    print(f"Few-shot Binary F1: {results_fewshot['binary']['f1']:.4f}")
    print(f"Few-shot Multi-class F1: {results_fewshot['multiclass']['f1_macro']:.4f}")
    print("Few-shot Per-class F1:", results_fewshot['per_class_f1'])
    
    # ====== SVM Baseline ======
    print("\n=== Training SVM Baseline ===")
    results_svm = train_svm_baseline(data_dict, train_idx, test_idx)
    print(f"SVM Binary F1: {results_svm['binary']['f1']:.4f}")
    print(f"SVM Multi-class F1: {results_svm['multiclass']['f1_weighted']:.4f}")
    
    # ====== Save All Results ======
    all_results = {
        'dids_mfl_normal': results_normal,
        'dids_mfl_unknown': results_unknown,
        'dids_mfl_fewshot': results_fewshot,
        'svm_baseline': results_svm,
        'training_history': history,
    }
    
    with open('outputs/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n=== All results saved to outputs/ ===")
    print("Done!")


if __name__ == '__main__':
    main()
