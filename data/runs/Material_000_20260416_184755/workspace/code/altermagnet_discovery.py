#!/usr/bin/env python3
"""
AI-Powered Altermagnetic Material Discovery
Graph Neural Network for predicting altermagnetic materials from crystal structures
"""

import sys
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool, LayerNorm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_000_20260416_184755'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUTS_DIR = os.path.join(WORKSPACE, 'outputs')
REPORT_DIR = os.path.join(WORKSPACE, 'report')
IMAGES_DIR = os.path.join(REPORT_DIR, 'images')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

class RealisticCrystalDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, 
                 log=True, _indices=None, force_reload=False, num_samples=5000,
                 pos_ratio=0.0, elem_to_idx=None, node_features=None, data_list=None,
                 has_download=False, has_process=False):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.num_samples = num_samples
        self.pos_ratio = pos_ratio
        self.elem_to_idx = elem_to_idx or {}
        self.node_features = node_features
        self.data_list = data_list or []
        self.has_download = has_download
        self.has_process = has_process
        self._indices = _indices
    
    def len(self):
        return len(self.data_list) if self.data_list else self.num_samples
    
    def get(self, idx):
        if self.data_list and idx < len(self.data_list):
            return self.data_list[idx]
        return None
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    @property
    def raw_file_names(self):
        return ['raw.pt']

def load_dataset(pt_path):
    class MockDataPrepare:
        RealisticCrystalDataset = RealisticCrystalDataset
    sys.modules['data_prepare'] = MockDataPrepare()
    return torch.load(pt_path, weights_only=False)

class GNN_Encoder(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=64, output_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_dim = output_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
            x = self.dropout(x)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        return self.output_proj(x)

class Classifier(nn.Module):
    def __init__(self, encoder, hidden_dim=32):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, edge_index, batch=None):
        return self.classifier(self.encoder(x, edge_index, batch))
    
    def predict_proba(self, x, edge_index, batch=None):
        return torch.sigmoid(self.forward(x, edge_index, batch))

def finetune_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', class_weight=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss, best_model_state = float('inf'), None
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss, epoch_train_correct, epoch_train_total = 0, 0, 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.float()
            pos_weight = torch.tensor([class_weight], device=device) if class_weight else None
            loss = F.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1), pos_weight=pos_weight if pos_weight is not None else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float().squeeze()
            epoch_train_correct += (preds == labels).sum().item()
            epoch_train_total += labels.size(0)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_correct / epoch_train_total
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        model.eval()
        epoch_val_loss, epoch_val_correct, epoch_val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.float()
                loss = F.binary_cross_entropy_with_logits(logits, labels.unsqueeze(1))
                epoch_val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float().squeeze()
                epoch_val_correct += (preds == labels).sum().item()
                epoch_val_total += labels.size(0)
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = epoch_val_correct / epoch_val_total
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return {'train_losses': train_losses, 'val_losses': val_losses, 
            'train_accs': train_accs, 'val_accs': val_accs, 'best_val_loss': best_val_loss}

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            probs = model.predict_proba(batch.x, batch.edge_index, batch.batch)
            all_probs.extend(probs.squeeze().cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    }, all_labels, all_probs

def plot_data_overview(pretrain_data, finetune_data, candidate_data, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sizes = [len(pretrain_data.data_list), len(finetune_data.data_list), len(candidate_data.data_list)]
    names = ['Pre-train\n(Unlabeled)', 'Fine-tune\n(Labeled)', 'Candidate\n(Prediction)']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    axes[0, 0].bar(names, sizes, color=colors)
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title('Dataset Sizes')
    for i, v in enumerate(sizes):
        axes[0, 0].text(i, v + 50, str(v), ha='center')
    
    ft_labels = [int(d.y.item()) for d in finetune_data.data_list]
    axes[0, 1].pie([len(ft_labels)-sum(ft_labels), sum(ft_labels)], labels=['Negative', 'Positive'], 
                   colors=['#95a5a6', '#e74c3c'], autopct='%1.1f%%')
    axes[0, 1].set_title(f'Fine-tune Labels (Total: {len(ft_labels)})')
    
    cand_labels = [int(d.y.item()) for d in candidate_data.data_list]
    axes[1, 0].pie([len(cand_labels)-sum(cand_labels), sum(cand_labels)], labels=['Negative', 'Positive'], 
                   colors=['#95a5a6', '#27ae60'], autopct='%1.1f%%')
    axes[1, 0].set_title(f'Candidate True Labels (Positives: {sum(cand_labels)})')
    
    stats = [{'dataset': n, 'num_nodes': d.x.shape[0]} for n, dl in [('Pre-train', pretrain_data.data_list[:500]), ('Fine-tune', finetune_data.data_list), ('Candidate', candidate_data.data_list)] for d in dl]
    sns.boxplot(data=pd.DataFrame(stats), x='dataset', y='num_nodes', ax=axes[1, 1], palette=colors)
    axes[1, 1].set_ylabel('Number of Nodes')
    axes[1, 1].set_title('Graph Size Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curves(log, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(log['train_losses']) + 1)
    axes[0].plot(epochs, log['train_losses'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, log['val_losses'], 'r--', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, log['train_accs'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, log['val_accs'], 'r--', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_results(df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df['predicted_prob'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
    axes[0, 0].set_xlabel('Predicted Probability'); axes[0, 0].set_title('Prediction Distribution')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    top20 = df.nlargest(20, 'predicted_prob')
    colors = ['#27ae60' if t else '#e74c3c' for t in top20['true_label']]
    axes[0, 1].barh(range(20), top20['predicted_prob'], color=colors)
    axes[0, 1].set_yticks(range(20)); axes[0, 1].set_yticklabels([f"#{i}" for i in range(20)])
    axes[0, 1].set_xlabel('Probability'); axes[0, 1].set_title('Top 20 Candidates')
    axes[0, 1].invert_yaxis()
    
    cm = confusion_matrix(df['true_label'], df['predicted_class'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Pred'); axes[1, 0].set_ylabel('True'); axes[1, 0].set_title('Confusion Matrix')
    
    axes[1, 1].boxplot([df[df['true_label']==0]['predicted_prob'], df[df['true_label']==1]['predicted_prob']],
                       labels=['True Neg', 'True Pos'])
    axes[1, 1].set_ylabel('Probability'); axes[1, 1].set_title('By True Label')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_features(data, save_path):
    sample = data.data_list[:100]
    vals = [float(d.x[:, j].mean()) for d in sample if d.x is not None for j in range(min(10, d.x.shape[1]))]
    feat_df = pd.DataFrame({'feature': list(range(10)) * len(sample), 'value': vals[:1000]})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.violinplot(data=feat_df, x='feature', y='value', ax=axes[0], palette='viridis')
    axes[0].set_title('Feature Distributions')
    
    all_vals = np.concatenate([d.x.flatten().cpu().numpy() for d in sample if d.x is not None])
    axes[1].hist(all_vals, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[1].set_title('Overall Value Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("="*60 + "\nAI-Powered Altermagnetic Material Discovery\n" + "="*60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    pretrain = load_dataset(os.path.join(DATA_DIR, 'pretrain_data.pt'))
    finetune = load_dataset(os.path.join(DATA_DIR, 'finetune_data.pt'))
    candidate = load_dataset(os.path.join(DATA_DIR, 'candidate_data.pt'))
    print(f"Pre-train: {len(pretrain.data_list)}, Fine-tune: {len(finetune.data_list)}, Candidate: {len(candidate.data_list)}")
    
    # Save stats
    stats = {
        'pretrain': {'samples': len(pretrain.data_list)},
        'finetune': {'samples': len(finetune.data_list), 'pos': sum(int(d.y.item()) for d in finetune.data_list)},
        'candidate': {'samples': len(candidate.data_list), 'true_pos': sum(int(d.y.item()) for d in candidate.data_list)}
    }
    with open(os.path.join(OUTPUTS_DIR, 'data_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Plots
    print("Generating plots...")
    plot_data_overview(pretrain, finetune, candidate, os.path.join(IMAGES_DIR, 'data_overview.png'))
    plot_features(pretrain, os.path.join(IMAGES_DIR, 'feature_analysis.png'))
    
    # Prepare loaders
    pyg_data = [Data(x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y) for d in finetune.data_list]
    idx = np.random.permutation(len(pyg_data))
    split = int(0.8 * len(pyg_data))
    train_loader = DataLoader([pyg_data[i] for i in idx[:split]], batch_size=32, shuffle=True)
    val_loader = DataLoader([pyg_data[i] for i in idx[split:]], batch_size=32)
    
    pos_ratio = sum(int(d.y.item()) for d in finetune.data_list) / len(finetune.data_list)
    class_weight = (1 - pos_ratio) / pos_ratio
    print(f"Class weight: {class_weight:.2f}")
    
    # Model
    encoder = GNN_Encoder(input_dim=28, hidden_dim=64, output_dim=64, num_layers=3)
    model = Classifier(encoder, hidden_dim=32).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    log = finetune_model(model, train_loader, val_loader, epochs=20, lr=0.001, device=device, class_weight=class_weight)
    
    with open(os.path.join(OUTPUTS_DIR, 'model_training_log.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] if isinstance(v, list) else float(v) for k, v in log.items()}, f, indent=2)
    plot_training_curves(log, os.path.join(IMAGES_DIR, 'training_curves.png'))
    
    # Evaluate
    print("\nEvaluating...")
    metrics, val_labels, val_probs = evaluate_model(model, val_loader, device)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    plot_roc_curve(val_labels, val_probs, os.path.join(IMAGES_DIR, 'roc_curve.png'))
    
    # Predict candidates
    print("\nPredicting candidates...")
    cand_pyg = [Data(x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y) for d in candidate.data_list]
    cand_loader = DataLoader(cand_pyg, batch_size=32)
    
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b in cand_loader:
            b = b.to(device)
            probs.extend(model.predict_proba(b.x, b.edge_index, b.batch).squeeze().cpu().numpy())
            labels.extend(b.y.cpu().numpy())
    
    df = pd.DataFrame({'id': range(len(probs)), 'prob': probs, 'pred': (np.array(probs) > 0.5).astype(int), 'true': labels})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUTPUTS_DIR, 'predictions.csv'), index=False)
    
    tp_disc = sum((df['pred']==1) & (df['true']==1))
    tp_total = sum(df['true']==1)
    rate = tp_disc / tp_total if tp_total else 0
    top50_tp = sum(df.head(50)['true']==1)
    
    print(f"\nDiscovery: Found {tp_disc}/{tp_total} true positives (rate={rate:.4f})")
    print(f"Top 50 contains {top50_tp} true positives")
    
    eval_res = {
        'val_metrics': {k: float(v) for k, v in metrics.items()},
        'discovery': {'found': int(tp_disc), 'total': int(tp_total), 'rate': float(rate), 'top50': int(top50_tp)}
    }
    with open(os.path.join(OUTPUTS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_res, f, indent=2)
    
    plot_results(df, os.path.join(IMAGES_DIR, 'results_comparison.png'))
    
    print("\n" + "="*60)
    print("Complete! Artifacts saved to outputs/ and report/images/")
    print("="*60)

if __name__ == '__main__':
    main()
