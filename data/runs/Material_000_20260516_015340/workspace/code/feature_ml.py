#!/usr/bin/env python3
"""
Altermagnetic Discovery - Feature Engineering + Traditional ML Approach

Strategy:
1. Extract rich graph-level features from each crystal:
   - Element composition statistics
   - Graph topology features (degree stats, clustering, etc.)
   - Graph spectral features
   - GNN embeddings from pretrained model
2. Use XGBoost/LightGBM for classification (robust to small data, class imbalance)
3. Ensemble multiple models
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import DataLoader
import numpy as np
import json
import os
import sys
from collections import Counter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'seed': 42,
    'device': 'cpu',
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

def load_dataset(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return [data[i] for i in range(len(data))]

def extract_graph_features(samples):
    """Extract rich graph-level features from each crystal graph."""
    features = []
    
    for data in samples:
        x = data.x.numpy()  # [N, 28] one-hot
        ei = data.edge_index.numpy()  # [2, E]
        ea = data.edge_attr.numpy()  # [E, 2]
        y = data.y.item()
        
        N = x.shape[0]  # number of atoms
        E = ei.shape[1]  # number of bonds
        
        feat = []
        
        # 1. Basic counts
        feat.append(N)  # num atoms
        feat.append(E)  # num edges
        feat.append(E / max(N, 1))  # edges per atom
        
        # 2. Element composition (which elements are present)
        elem_counts = x.sum(axis=0)  # [28]
        feat.extend(elem_counts.tolist())  # 28 features
        
        # Normalized composition
        elem_frac = elem_counts / max(N, 1)
        feat.extend(elem_frac.tolist())  # 28 features
        
        # 3. Degree statistics
        degrees = np.bincount(ei[0], minlength=N)
        feat.append(float(np.mean(degrees)))
        feat.append(float(np.std(degrees)))
        feat.append(float(np.max(degrees)))
        feat.append(float(np.min(degrees)))
        
        # Degree distribution percentiles
        for p in [25, 50, 75]:
            feat.append(float(np.percentile(degrees, p)))
        
        # 4. Edge feature statistics
        if ea is not None:
            for dim in range(ea.shape[1]):
                feat.append(float(np.mean(ea[:, dim])))
                feat.append(float(np.std(ea[:, dim])))
                feat.append(float(np.min(ea[:, dim])))
                feat.append(float(np.max(ea[:, dim])))
        
        # 5. Graph density
        max_edges = N * (N - 1) / 2
        density = E / max(max_edges, 1)
        feat.append(density)
        
        # 6. Node diversity (how many unique element types)
        elem_present = (elem_counts > 0).sum()
        feat.append(float(elem_present))
        feat.append(elem_present / max(N, 1))
        
        # 7. Adjacency matrix spectral features (approximate)
        # Use degree-based features as proxy
        deg_sq = degrees ** 2
        feat.append(float(np.sum(deg_sq)))
        feat.append(float(np.sum(deg_sq)) / max(N, 1))
        
        # 8. Number of connected components approximation
        # (Use degree-0 count)
        isolated = (degrees == 0).sum()
        feat.append(float(isolated))
        
        # 9. Maximum element fraction
        feat.append(float(np.max(elem_frac)))
        
        # 10. Element entropy
        nonzero_frac = elem_frac[elem_frac > 0]
        entropy = -np.sum(nonzero_frac * np.log(nonzero_frac + 1e-10))
        feat.append(entropy)
        
        # 11. Pairwise element co-occurrence features
        # For each pair of elements, count edges connecting them
        # Simplified: for each node, compute neighbor element distribution
        src = ei[0]; dst = ei[1]
        node_to_elem = np.argmax(x, axis=1)  # [N]
        
        # Average neighbor element type 
        neighbor_elems = node_to_elem[dst]
        unique_pairs = 0
        for i in range(N):
            neighbors_i = neighbor_elems[src == i]
            unique_pairs += len(np.unique(neighbors_i))
        feat.append(unique_pairs / max(E, 1))
        
        features.append(feat)
    
    return np.array(features), np.array([s.y.item() for s in samples])

def main():
    device = CONFIG['device']
    print(f"Device: {device}")
    
    print("\n=== Loading Data ===")
    pretrain_samples = load_dataset('data/pretrain_data.pt')
    finetune_samples = load_dataset('data/finetune_data.pt')
    candidate_samples = load_dataset('data/candidate_data.pt')
    
    print(f"Pretrain: {len(pretrain_samples)}, Finetune: {len(finetune_samples)}, Candidate: {len(candidate_samples)}")
    
    # Also load pretrained embeddings if available
    use_pretrained_emb = os.path.exists('outputs/embeddings.pt')
    if use_pretrained_emb:
        emb_data = torch.load('outputs/embeddings.pt', map_location='cpu')
        print("Loaded pretrained embeddings")
    
    # Extract features
    print("\n=== Extracting Graph Features ===")
    ft_feat, ft_labels = extract_graph_features(finetune_samples)
    cand_feat, cand_labels = extract_graph_features(candidate_samples)
    pt_feat, _ = extract_graph_features(pretrain_samples)
    
    print(f"Finetune features shape: {ft_feat.shape}")
    print(f"Candidate features shape: {cand_feat.shape}")
    print(f"Pretrain features shape: {pt_feat.shape}")
    print(f"Finetune labels: {Counter(ft_labels.tolist())}")
    print(f"Candidate labels: {Counter(cand_labels.tolist())}")
    
    # Combine with GNN embeddings if available
    if use_pretrained_emb:
        ft_emb = emb_data['finetune_embeddings'].numpy()
        cand_emb = emb_data['candidate_embeddings'].numpy()
        ft_feat = np.concatenate([ft_feat, ft_emb], axis=1)
        cand_feat = np.concatenate([cand_feat, cand_emb], axis=1)
        print(f"Combined features shape: {ft_feat.shape}")
    
    # Standardize
    scaler = StandardScaler()
    ft_feat = scaler.fit_transform(ft_feat)
    cand_feat = scaler.transform(cand_feat)
    
    # Train/val split
    train_idx, val_idx = train_test_split(
        range(len(ft_feat)), test_size=0.2, random_state=CONFIG['seed'], stratify=ft_labels
    )
    X_train, X_val = ft_feat[train_idx], ft_feat[val_idx]
    y_train, y_val = ft_labels[train_idx], ft_labels[val_idx]
    
    print(f"\nTrain: {len(X_train)} (pos={y_train.sum()}), Val: {len(X_val)} (pos={y_val.sum()})")
    
    # Try multiple models with class weighting
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=CONFIG['seed'], n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=5, random_state=CONFIG['seed']
        ),
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', max_iter=1000, C=0.1, random_state=CONFIG['seed']
        ),
    }
    
    best_model = None
    best_auroc = 0.0
    best_name = None
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Compute sample weights for gradient boosting
        if name == 'GradientBoosting':
            sample_weight = np.where(y_train == 1, 19.0, 1.0)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        
        # Validation
        if hasattr(model, 'predict_proba'):
            val_probs = model.predict_proba(X_val)[:, 1]
        else:
            val_probs = model.decision_function(X_val)
            val_probs = 1 / (1 + np.exp(-val_probs))
        
        val_auroc = roc_auc_score(y_val, val_probs)
        val_auprc = average_precision_score(y_val, val_probs)
        
        precs, recs, threshs = precision_recall_curve(y_val, val_probs)
        f1s = 2 * precs * recs / (precs + recs + 1e-10)
        best_thresh = threshs[np.argmax(f1s)]
        val_bin = (val_probs >= best_thresh).astype(int)
        
        print(f"  AUROC={val_auroc:.4f}, AUPRC={val_auprc:.4f}")
        print(f"  F1={f1_score(y_val, val_bin, zero_division=0):.4f}")
        print(f"  CM:\n{confusion_matrix(y_val, val_bin)}")
        
        # Candidate evaluation
        if hasattr(model, 'predict_proba'):
            cand_probs = model.predict_proba(cand_feat)[:, 1]
        else:
            cand_probs = model.decision_function(cand_feat)
            cand_probs = 1 / (1 + np.exp(-cand_probs))
        
        cand_auroc = roc_auc_score(cand_labels, cand_probs)
        cand_bin = (cand_probs >= best_thresh).astype(int)
        n_pred = cand_bin.sum()
        
        sorted_idx = np.argsort(-cand_probs)
        top20 = cand_labels[sorted_idx[:20]].mean()
        top50 = cand_labels[sorted_idx[:50]].mean()
        
        print(f"  Candidate AUROC={cand_auroc:.4f}, Predicted +: {n_pred}")
        print(f"  Top-20: {top20:.2%}, Top-50: {top50:.2%}")
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_model = model
            best_name = name
            best_val_probs = val_probs
            best_cand_probs = cand_probs
            best_thresh_final = best_thresh
    
    print(f"\n=== Best Model: {best_name} (AUROC={best_auroc:.4f}) ===")
    
    # Final evaluation with best model
    cand_bin_final = (best_cand_probs >= best_thresh_final).astype(int)
    sorted_idx = np.argsort(-best_cand_probs)
    
    print(f"\n=== Final Candidate Results ===")
    n_pred = int(cand_bin_final.sum())
    n_true = int(cand_labels.sum())
    c_auroc = roc_auc_score(cand_labels, best_cand_probs)
    c_auprc = average_precision_score(cand_labels, best_cand_probs)
    
    print(f"  Predicted +: {n_pred}, True +: {n_true}")
    print(f"  AUROC={c_auroc:.4f}, AUPRC={c_auprc:.4f}")
    print(f"  CM:\n{confusion_matrix(cand_labels, cand_bin_final)}")
    
    for k in [20, 50, 100]:
        correct = int(cand_labels[sorted_idx[:k]].sum())
        print(f"  Top-{k}: {correct}/{k} = {correct/k:.2%}")
    
    # Print top predictions
    print("\nTop 30 Predictions:")
    for i in range(min(30, len(sorted_idx))):
        idx = int(sorted_idx[i])
        p = float(best_cand_probs[idx]); l = bool(cand_labels[idx])
        print(f"  #{idx:4d}: p={p:.4f} [{'✓' if l else '✗'}]")
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    results = {
        'best_model': best_name,
        'val_auroc': float(best_auroc),
        'candidate_metrics': {
            'auroc': float(c_auroc), 'auprc': float(c_auprc),
            'n_predicted': n_pred, 'n_true': n_true,
            'confusion_matrix': confusion_matrix(cand_labels, cand_bin_final).tolist(),
        },
        'top20': float(cand_labels[sorted_idx[:20]].mean()),
        'top50': float(cand_labels[sorted_idx[:50]].mean()),
        'top100': float(cand_labels[sorted_idx[:100]].mean()),
        'candidate_predictions': best_cand_probs.tolist(),
        'candidate_labels': cand_labels.tolist(),
        'candidate_binary': cand_bin_final.tolist(),
    }
    with open('outputs/results_ml.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    details = [{
        'index': i,
        'probability': float(best_cand_probs[i]),
        'predicted_altermagnet': bool(cand_bin_final[i]),
        'true_altermagnet': bool(cand_labels[i]),
    } for i in range(len(best_cand_probs))]
    details.sort(key=lambda x: x['probability'], reverse=True)
    with open('outputs/candidate_details_ml.json', 'w') as f:
        json.dump(details, f, indent=2)
    
    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        top_feats = np.argsort(-importances)[:20]
        print("\nTop 20 Features:")
        for i, idx in enumerate(top_feats):
            print(f"  {i+1}. Feature {idx}: importance={importances[idx]:.6f}")
    
    print("\nDone!")
    return results

if __name__ == '__main__':
    main()
