"""
Altermagnetic Material Discovery (v4)
- Focus on finetune data only (pretrain has domain shift)
- Multiple baselines: LR, MLP, Simple GNN
- Cross-validation for robust evaluation
- Ensemble of best models
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool, AttentionalAggregation
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_datasets():
    finetune = torch.load('data/finetune_data.pt', map_location='cpu', weights_only=False)
    candidate = torch.load('data/candidate_data.pt', map_location='cpu', weights_only=False)
    return finetune.data_list, candidate.data_list

def extract_graph_features(data_list):
    features = []
    labels = []
    for d in data_list:
        x = d.x.numpy()
        e = d.edge_attr.numpy() if d.edge_attr is not None else np.zeros((1, 2))
        
        node_mean = x.mean(axis=0)
        node_max = x.max(axis=0)
        node_min = x.min(axis=0)
        node_sum = x.sum(axis=0)
        node_std = x.std(axis=0)
        
        edge_mean = e.mean(axis=0)
        edge_max = e.max(axis=0)
        edge_sum = e.sum(axis=0)
        
        num_nodes = x.shape[0]
        num_edges = e.shape[0]
        density = num_edges / max(1, num_nodes * (num_nodes - 1))
        
        feat = np.concatenate([
            node_mean, node_max, node_min, node_sum, node_std,
            edge_mean, edge_max, edge_sum,
            [num_nodes, num_edges, density]
        ])
        features.append(feat)
        labels.append(d.y.item() if hasattr(d, 'y') and d.y is not None else 0)
    return np.array(features), np.array(labels)

# ---------------------------------------------------------------------------
# Simple GNN
# ---------------------------------------------------------------------------

class SimpleGNN(nn.Module):
    def __init__(self, node_dim=28, edge_dim=2, hidden_dim=64, num_layers=3):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_emb = self.edge_emb(edge_attr) if edge_attr is not None else None
        x = self.node_emb(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = bn(x)
            x = F.relu(x)
        x = self.pool(x, batch)
        return self.classifier(x).squeeze(-1)

@torch.no_grad()
def eval_gnn(model, loader, device):
    model.eval()
    probs, labels, preds = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)
        p = torch.sigmoid(logit).cpu().numpy()
        l = batch.y.float().cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(l.tolist())
        preds.extend((p >= 0.5).astype(int).tolist())
    probs = np.array(probs)
    labels = np.array(labels)
    preds = np.array(preds)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        'pr_auc': float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        'mcc': float(matthews_corrcoef(labels, preds)),
    }, probs, labels, preds

def train_gnn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logit = model(batch)
        loss = criterion(logit, batch.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    
    finetune_list, candidate_list = load_datasets()
    for d in finetune_list:
        d.y = d.y.float()
    for d in candidate_list:
        d.y = d.y.float()
    
    print(f"Finetune: {len(finetune_list)}, Candidate: {len(candidate_list)}")
    
    # Extract features for sklearn models
    X_all, y_all = extract_graph_features(finetune_list)
    X_cand, y_cand = extract_graph_features(candidate_list)
    
    # Cross-validation for baselines
    print("\n=== Cross-validation for Baselines ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=SEED, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=SEED)
    }
    
    baseline_results = {}
    for name, model in models.items():
        aucs = []
        f1s = []
        for train_idx, val_idx in skf.split(X_all, y_all):
            model.fit(X_all[train_idx], y_all[train_idx])
            probs = model.predict_proba(X_all[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            aucs.append(roc_auc_score(y_all[val_idx], probs))
            f1s.append(f1_score(y_all[val_idx], preds, zero_division=0))
        baseline_results[name] = {'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs), 'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s)}
        print(f"{name}: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}, F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    # Train best baseline on full data and evaluate on candidates
    best_baseline_name = max(baseline_results, key=lambda k: baseline_results[k]['auc_mean'])
    print(f"\nBest baseline: {best_baseline_name}")
    best_baseline = models[best_baseline_name]
    best_baseline.fit(X_all, y_all)
    baseline_cand_probs = best_baseline.predict_proba(X_cand)[:, 1]
    baseline_top50 = np.argsort(baseline_cand_probs)[::-1][:50]
    baseline_discovered = int(y_cand[baseline_top50].sum())
    print(f"Baseline Top-50 discovered: {baseline_discovered}, Precision@50: {baseline_discovered/50:.4f}")
    
    # Cross-validation for GNN
    print("\n=== Cross-validation for GNN ===")
    gnn_aucs = []
    gnn_f1s = []
    gnn_val_probs_all = []
    gnn_val_labels_all = []
    gnn_val_preds_all = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(finetune_list)), y_all)):
        train_data = [finetune_list[i] for i in train_idx]
        val_data = [finetune_list[i] for i in val_idx]
        
        # Oversample
        pos = [d for d in train_data if d.y.item() == 1.0]
        neg = [d for d in train_data if d.y.item() == 0.0]
        ratio = min(len(neg) // max(1, len(pos)), 8)
        train_bal = neg + pos * ratio
        random.shuffle(train_bal)
        
        train_loader = PyGDataLoader(train_bal, batch_size=64, shuffle=True, num_workers=0)
        val_loader = PyGDataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
        
        model = SimpleGNN(node_dim=28, edge_dim=2, hidden_dim=64, num_layers=3).to(device)
        pos_w = len(train_bal) / max(1, sum([d.y.item() for d in train_bal])) - 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        best_auc = 0
        best_state = None
        
        for epoch in range(1, 101):
            train_gnn(model, train_loader, optimizer, criterion, device)
            if epoch % 5 == 0:
                metrics, _, _, _ = eval_gnn(model, val_loader, device)
                if metrics['roc_auc'] > best_auc:
                    best_auc = metrics['roc_auc']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                scheduler.step(metrics['roc_auc'])
        
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
        metrics, probs, labels, preds = eval_gnn(model, val_loader, device)
        gnn_aucs.append(metrics['roc_auc'])
        gnn_f1s.append(metrics['f1'])
        gnn_val_probs_all.extend(probs.tolist())
        gnn_val_labels_all.extend(labels.tolist())
        gnn_val_preds_all.extend(preds.tolist())
        print(f"Fold {fold+1}: AUC = {metrics['roc_auc']:.4f}, F1 = {metrics['f1']:.4f}")
    
    print(f"GNN CV: AUC = {np.mean(gnn_aucs):.4f} ± {np.std(gnn_aucs):.4f}, F1 = {np.mean(gnn_f1s):.4f} ± {np.std(gnn_f1s):.4f}")
    
    # Train final GNN on full data
    print("\n=== Training Final GNN ===")
    pos = [d for d in finetune_list if d.y.item() == 1.0]
    neg = [d for d in finetune_list if d.y.item() == 0.0]
    ratio = min(len(neg) // max(1, len(pos)), 8)
    train_bal = neg + pos * ratio
    random.shuffle(train_bal)
    
    final_model = SimpleGNN(node_dim=28, edge_dim=2, hidden_dim=64, num_layers=3).to(device)
    pos_w = len(train_bal) / max(1, sum([d.y.item() for d in train_bal])) - 1
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_loader = PyGDataLoader(train_bal, batch_size=64, shuffle=True, num_workers=0)
    
    for epoch in range(1, 151):
        loss = train_gnn(final_model, train_loader, optimizer, criterion, device)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    
    # Evaluate on candidates
    candidate_loader = PyGDataLoader(candidate_list, batch_size=64, shuffle=False, num_workers=0)
    final_model.eval()
    gnn_cand_probs = []
    gnn_cand_labels = []
    with torch.no_grad():
        for batch in candidate_loader:
            batch = batch.to(device)
            logit = final_model(batch)
            p = torch.sigmoid(logit).cpu().numpy()
            l = batch.y.float().cpu().numpy()
            gnn_cand_probs.extend(p.tolist())
            gnn_cand_labels.extend(l.tolist())
    
    gnn_cand_probs = np.array(gnn_cand_probs)
    gnn_cand_labels = np.array(gnn_cand_labels)
    
    gnn_top50 = np.argsort(gnn_cand_probs)[::-1][:50]
    gnn_discovered = int(gnn_cand_labels[gnn_top50].sum())
    print(f"GNN Top-50 discovered: {gnn_discovered}, Precision@50: {gnn_discovered/50:.4f}")
    
    # Ensemble: average GNN and best baseline
    ensemble_probs = 0.5 * gnn_cand_probs + 0.5 * baseline_cand_probs
    ensemble_top50 = np.argsort(ensemble_probs)[::-1][:50]
    ensemble_discovered = int(y_cand[ensemble_top50].sum())
    print(f"Ensemble Top-50 discovered: {ensemble_discovered}, Precision@50: {ensemble_discovered/50:.4f}")
    
    # Save results
    results = {
        'baselines': baseline_results,
        'gnn_cv_auc_mean': float(np.mean(gnn_aucs)),
        'gnn_cv_auc_std': float(np.std(gnn_aucs)),
        'gnn_cv_f1_mean': float(np.mean(gnn_f1s)),
        'gnn_cv_f1_std': float(np.std(gnn_f1s)),
        'best_baseline': best_baseline_name,
        'baseline_precision_at_50': float(baseline_discovered / 50),
        'baseline_recall_at_50': float(baseline_discovered / max(1, y_cand.sum())),
        'gnn_precision_at_50': float(gnn_discovered / 50),
        'gnn_recall_at_50': float(gnn_discovered / max(1, gnn_cand_labels.sum())),
        'ensemble_precision_at_50': float(ensemble_discovered / 50),
        'ensemble_recall_at_50': float(ensemble_discovered / max(1, y_cand.sum())),
        'num_true_positives_in_candidate': int(y_cand.sum())
    }
    with open('outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.savez('outputs/predictions.npz',
             gnn_probs=gnn_cand_probs,
             baseline_probs=baseline_cand_probs,
             ensemble_probs=ensemble_probs,
             labels=y_cand)
    
    # ========================
    # Figures
    # ========================
    print("\n=== Generating Figures ===")
    
    # ROC from CV
    fpr, tpr, _ = roc_curve(gnn_val_labels_all, gnn_val_probs_all)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f'GNN CV (AUC = {roc_auc_score(gnn_val_labels_all, gnn_val_probs_all):.3f})')
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - GNN Cross-validation')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('report/images/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # PR from CV
    precision_curve, recall_curve, _ = precision_recall_curve(gnn_val_labels_all, gnn_val_probs_all)
    plt.figure(figsize=(7,6))
    plt.plot(recall_curve, precision_curve, lw=2, label=f'GNN CV (AP = {average_precision_score(gnn_val_labels_all, gnn_val_probs_all):.3f})')
    plt.axhline(y=np.mean(gnn_val_labels_all), color='r', linestyle='--', label=f'Baseline ({np.mean(gnn_val_labels_all):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('report/images/pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix from CV
    cm = confusion_matrix(gnn_val_labels_all, gnn_val_preds_all)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-AM', 'AM'], yticklabels=['Non-AM', 'AM'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - GNN (5-Fold CV)')
    plt.savefig('report/images/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Candidate distributions
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for ax, probs, name in zip(axes, [gnn_cand_probs, baseline_cand_probs, ensemble_probs], ['GNN', 'Best Baseline', 'Ensemble']):
        bins = np.linspace(0, 1, 31)
        ax.hist(probs[y_cand==0], bins=bins, alpha=0.6, label='Non-AM', color='steelblue')
        ax.hist(probs[y_cand==1], bins=bins, alpha=0.6, label='AM (True)', color='crimson')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle('Candidate Score Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig('report/images/candidate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Top-50 ensemble
    plt.figure(figsize=(12,6))
    top_probs = ensemble_probs[ensemble_top50]
    top_labels = y_cand[ensemble_top50]
    colors = ['crimson' if l == 1 else 'steelblue' for l in top_labels]
    plt.bar(range(50), top_probs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.xlabel('Candidate Rank')
    plt.ylabel('Predicted Probability')
    plt.title(f'Top-50 Predicted Altermagnets (Ensemble) - Red = True Positive')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('report/images/top_candidates.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Precision@K / Recall@K
    k_values = list(range(10, min(201, len(candidate_list)+1), 10))
    gnn_prec, gnn_rec = [], []
    base_prec, base_rec = [], []
    ens_prec, ens_rec = [], []
    gnn_sorted = np.argsort(gnn_cand_probs)[::-1]
    base_sorted = np.argsort(baseline_cand_probs)[::-1]
    ens_sorted = np.argsort(ensemble_probs)[::-1]
    for k in k_values:
        gnn_prec.append(float(y_cand[gnn_sorted[:k]].mean()))
        gnn_rec.append(float(y_cand[gnn_sorted[:k]].sum() / max(1, y_cand.sum())))
        base_prec.append(float(y_cand[base_sorted[:k]].mean()))
        base_rec.append(float(y_cand[base_sorted[:k]].sum() / max(1, y_cand.sum())))
        ens_prec.append(float(y_cand[ens_sorted[:k]].mean()))
        ens_rec.append(float(y_cand[ens_sorted[:k]].sum() / max(1, y_cand.sum())))
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(k_values, gnn_prec, 'b-o', label='GNN Prec@K', markersize=4)
    ax1.plot(k_values, base_prec, 'b--s', label='Baseline Prec@K', markersize=4)
    ax1.plot(k_values, ens_prec, 'b:^', label='Ensemble Prec@K', markersize=4)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Precision@K', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(k_values, gnn_rec, 'r-o', label='GNN Rec@K', markersize=4)
    ax2.plot(k_values, base_rec, 'r--s', label='Baseline Rec@K', markersize=4)
    ax2.plot(k_values, ens_rec, 'r:^', label='Ensemble Rec@K', markersize=4)
    ax2.set_ylabel('Recall@K', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Precision and Recall at K (Candidate Set)')
    fig.legend(loc='upper right', bbox_to_anchor=(0.92,0.92), fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig('report/images/precision_recall_at_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Model comparison bar chart
    plt.figure(figsize=(8,5))
    model_names = list(baseline_results.keys()) + ['GNN']
    auc_means = [baseline_results[m]['auc_mean'] for m in model_names[:-1]] + [np.mean(gnn_aucs)]
    auc_stds = [baseline_results[m]['auc_std'] for m in model_names[:-1]] + [np.std(gnn_aucs)]
    x = np.arange(len(model_names))
    plt.bar(x, auc_means, yerr=auc_stds, capsize=4, alpha=0.8, color=['steelblue']*4 + ['crimson'])
    plt.xticks(x, model_names, rotation=15, ha='right')
    plt.ylabel('ROC-AUC')
    plt.title('Model Comparison (5-Fold CV)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('report/images/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Data distribution
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    names = ['Fine-tune\n(2000)', 'Candidate\n(1000)']
    datasets = [finetune_list, candidate_list]
    for ax, ds, name in zip(axes, datasets, names):
        counts = [sum([d.y.item()==0 for d in ds]), sum([d.y.item()==1 for d in ds])]
        ax.pie(counts, labels=['Non-AM', 'AM'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
        ax.set_title(name)
    plt.suptitle('Dataset Label Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig('report/images/data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    torch.save(final_model.state_dict(), 'outputs/gnn_final.pt')
    
    print("\n=== All done! ===")

if __name__ == '__main__':
    main()
