"""
Altermagnetic Material Discovery - Boosted Trees Approach
- Extract rich graph-level features
- Train XGBoost and LightGBM with cross-validation
- Evaluate on candidates
"""

import os
import random
import numpy as np
import torch
import json
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def load_datasets():
    finetune = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
    candidate = torch.load('data/candidate_data.pt', map_location='cpu', weights_only=False)
    return finetune.data_list, candidate.data_list

def extract_features(data_list):
    features = []
    labels = []
    for d in data_list:
        x = d.x.numpy()
        e = d.edge_attr.numpy() if d.edge_attr is not None else np.zeros((1, 2))
        
        # Element composition (dims 0-27 are one-hot elements)
        elem_counts = x.sum(axis=0)  # [28]
        elem_frac = elem_counts / max(1, elem_counts.sum())
        
        # Binary presence of each element
        elem_present = (elem_counts > 0).astype(float)
        
        # Graph stats
        n_nodes = x.shape[0]
        n_edges = e.shape[0]
        density = n_edges / max(1, n_nodes * (n_nodes - 1))
        avg_degree = 2 * n_edges / max(1, n_nodes)
        
        # Edge stats
        edge_mean = e.mean(axis=0)
        edge_std = e.std(axis=0)
        edge_min = e.min(axis=0)
        edge_max = e.max(axis=0)
        
        # Number of unique elements
        n_unique_elems = elem_present.sum()
        
        # Most common element fraction
        max_elem_frac = elem_frac.max()
        
        # Entropy of element distribution
        p = elem_frac[elem_frac > 0]
        entropy = -np.sum(p * np.log(p + 1e-10))
        
        feat = np.concatenate([
            elem_counts,
            elem_frac,
            elem_present,
            edge_mean, edge_std, edge_min, edge_max,
            [n_nodes, n_edges, density, avg_degree, n_unique_elems, max_elem_frac, entropy]
        ])
        features.append(feat)
        labels.append(d.y.item() if hasattr(d, 'y') and d.y is not None else 0)
    return np.array(features), np.array(labels)

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    
    finetune_list, candidate_list = load_datasets()
    for d in finetune_list + candidate_list:
        d.y = d.y.float()
    
    X_all, y_all = extract_features(finetune_list)
    X_cand, y_cand = extract_features(candidate_list)
    
    print(f"Features shape: {X_all.shape}")
    print(f"Class distribution: {np.bincount(y_all.astype(int))}")
    
    # Try XGBoost
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=20, random_state=SEED, n_jobs=-1,
            eval_metric='logloss', use_label_encoder=False
        )
    except ImportError:
        xgb_model = None
        print("XGBoost not available")
    
    # Try LightGBM
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=SEED, n_jobs=-1,
            verbose=-1
        )
    except ImportError:
        lgb_model = None
        print("LightGBM not available")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    results = {}
    all_models = {}
    
    for name, model in [('XGBoost', xgb_model), ('LightGBM', lgb_model)]:
        if model is None:
            continue
        aucs = []
        f1s = []
        all_probs = []
        all_labels = []
        all_preds = []
        
        for train_idx, val_idx in skf.split(X_all, y_all):
            model.fit(X_all[train_idx], y_all[train_idx])
            probs = model.predict_proba(X_all[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            aucs.append(roc_auc_score(y_all[val_idx], probs))
            f1s.append(f1_score(y_all[val_idx], preds, zero_division=0))
            all_probs.extend(probs)
            all_labels.extend(y_all[val_idx])
            all_preds.extend(preds)
        
        results[name] = {
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
        }
        print(f"{name}: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}, F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        
        # Train on full data for candidate prediction
        model.fit(X_all, y_all)
        all_models[name] = model
        
        cand_probs = model.predict_proba(X_cand)[:, 1]
        top50 = np.argsort(cand_probs)[::-1][:50]
        discovered = int(y_cand[top50].sum())
        results[name]['precision_at_50'] = float(discovered / 50)
        results[name]['recall_at_50'] = float(discovered / max(1, y_cand.sum()))
        print(f"  Top-50 discovered: {discovered}, Precision@50: {discovered/50:.4f}")
    
    # Feature importance
    if 'LightGBM' in all_models:
        import lightgbm as lgb
        lgb.plot_importance(all_models['LightGBM'], max_num_features=20, figsize=(10,6))
        plt.title('Top 20 Feature Importances (LightGBM)')
        plt.tight_layout()
        plt.savefig('report/images/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save results
    with open('outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate comparison figures
    fig_names = []
    
    # If we have both models, create comparison plots
    if len(all_models) >= 1:
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_mean'])
        best_model = all_models[best_model_name]
        
        # ROC and PR from CV predictions
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_probs = []
        cv_labels = []
        cv_preds = []
        for train_idx, val_idx in skf.split(X_all, y_all):
            best_model.fit(X_all[train_idx], y_all[train_idx])
            probs = best_model.predict_proba(X_all[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            cv_probs.extend(probs.tolist())
            cv_labels.extend(y_all[val_idx].tolist())
            cv_preds.extend(preds.tolist())
        
        cv_probs = np.array(cv_probs)
        cv_labels = np.array(cv_labels)
        cv_preds = np.array(cv_preds)
        
        # ROC
        fpr, tpr, _ = roc_curve(cv_labels, cv_probs)
        plt.figure(figsize=(7,6))
        plt.plot(fpr, tpr, lw=2, label=f'{best_model_name} (AUC = {roc_auc_score(cv_labels, cv_probs):.3f})')
        plt.plot([0,1], [0,1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('report/images/roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # PR
        precision_curve, recall_curve, _ = precision_recall_curve(cv_labels, cv_probs)
        plt.figure(figsize=(7,6))
        plt.plot(recall_curve, precision_curve, lw=2, label=f'{best_model_name} (AP = {average_precision_score(cv_labels, cv_probs):.3f})')
        plt.axhline(y=cv_labels.mean(), color='r', linestyle='--', label=f'Baseline ({cv_labels.mean():.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig('report/images/pr_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Confusion matrix
        cm = confusion_matrix(cv_labels, cv_preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Non-AM', 'AM'], yticklabels=['Non-AM', 'AM'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('report/images/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Candidate predictions
        best_model.fit(X_all, y_all)
        cand_probs = best_model.predict_proba(X_cand)[:, 1]
        
        plt.figure(figsize=(8,5))
        bins = np.linspace(0, 1, 31)
        plt.hist(cand_probs[y_cand==0], bins=bins, alpha=0.6, label='Non-AM', color='steelblue')
        plt.hist(cand_probs[y_cand==1], bins=bins, alpha=0.6, label='AM', color='crimson')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Candidate Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('report/images/candidate_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Top-50
        top50 = np.argsort(cand_probs)[::-1][:50]
        top_labels = y_cand[top50]
        plt.figure(figsize=(12,6))
        colors = ['crimson' if l == 1 else 'steelblue' for l in top_labels]
        plt.bar(range(50), cand_probs[top50], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.xlabel('Candidate Rank')
        plt.ylabel('Predicted Probability')
        plt.title('Top-50 Predicted Altermagnets (Red = True Positive)')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig('report/images/top_candidates.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Precision/Recall @ K
        k_values = list(range(10, min(201, len(candidate_list)+1), 10))
        precisions = []
        recalls = []
        sorted_idx = np.argsort(cand_probs)[::-1]
        for k in k_values:
            precisions.append(float(y_cand[sorted_idx[:k]].mean()))
            recalls.append(float(y_cand[sorted_idx[:k]].sum() / max(1, y_cand.sum())))
        
        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.plot(k_values, precisions, 'b-o', label='Precision@K')
        ax1.set_xlabel('K')
        ax1.set_ylabel('Precision@K', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(k_values, recalls, 'r-s', label='Recall@K')
        ax2.set_ylabel('Recall@K', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.title('Precision and Recall at K')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.9))
        plt.grid(True, alpha=0.3)
        plt.savefig('report/images/precision_recall_at_k.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Model comparison
        if len(results) > 1:
            plt.figure(figsize=(8,5))
            names = list(results.keys())
            aucs = [results[n]['auc_mean'] for n in names]
            stds = [results[n]['auc_std'] for n in names]
            x = np.arange(len(names))
            plt.bar(x, aucs, yerr=stds, capsize=4, alpha=0.8, color=['steelblue', 'crimson'])
            plt.xticks(x, names)
            plt.ylabel('ROC-AUC')
            plt.title('Model Comparison (5-Fold CV)')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('report/images/model_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Data distribution
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for ax, ds, name in zip(axes, [finetune_list, candidate_list], ['Fine-tune\n(2000)', 'Candidate\n(1000)']):
        counts = [sum([d.y.item()==0 for d in ds]), sum([d.y.item()==1 for d in ds])]
        ax.pie(counts, labels=['Non-AM', 'AM'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
        ax.set_title(name)
    plt.suptitle('Dataset Label Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('report/images/data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n=== All done! ===")

if __name__ == '__main__':
    main()
