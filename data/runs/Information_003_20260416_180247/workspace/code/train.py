#!/usr/bin/env python3
"""
Training script for DIDS-MFL intrusion detection model.
Handles data loading, training loops, and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

from dids_mfl import DIDS_MFL, compute_class_distribution


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_path: str):
    """Load temporal data from .pt file."""
    data = torch.load(data_path, weights_only=False)
    return data


def create_edge_dataset(temporal_data):
    """Create dataset for edge-level classification."""
    store = temporal_data.stores[0]
    X = store.msg
    y_binary = store.label
    y_multiclass = store.attack
    return X, y_binary, y_multiclass


def create_few_shot_split(X, y_multiclass, shots_per_class=5):
    """Create few-shot learning split."""
    unique_classes = torch.unique(y_multiclass)
    support_indices = []
    query_indices = []
    
    for cls in unique_classes:
        cls_mask = (y_multiclass == cls)
        cls_indices = torch.where(cls_mask)[0]
        n_support = min(shots_per_class, len(cls_indices))
        support_idx = cls_indices[:n_support]
        query_idx = cls_indices[n_support:]
        support_indices.append(support_idx)
        query_indices.append(query_idx)
    
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices) if len(query_indices) > 0 else torch.tensor([])
    return support_indices, query_indices


def train_epoch(model, loader, optimizer, criterion, device, task='binary'):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    for batch in loader:
        X_batch, y_batch = batch
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long()
        
        num_edges = X_batch.shape[0]
        num_nodes = max(50, num_edges // 2)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
        edge_attr = X_batch.clone()
        time = torch.randn(num_edges, device=device)
        
        optimizer.zero_grad()
        
        if task == 'binary':
            logits = model(X_batch, edge_index, edge_attr, time, task='binary')
        else:
            logits = model(X_batch, edge_index, edge_attr, time, task='multiclass')
        
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)
        
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss, all_preds, all_labels


def evaluate(model, loader, criterion, device, task='binary'):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_logits = []
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch = batch
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).long()
            
            num_edges = X_batch.shape[0]
            num_nodes = max(50, num_edges // 2)
            edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
            edge_attr = X_batch.clone()
            time = torch.randn(num_edges, device=device)
            
            if task == 'binary':
                logits = model(X_batch, edge_index, edge_attr, time, task='binary')
            else:
                logits = model(X_batch, edge_index, edge_attr, time, task='multiclass')
            
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_logits.extend(F.softmax(logits, dim=-1).cpu().numpy())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss, all_preds, all_labels, all_logits


def compute_metrics(y_true, y_pred, y_prob=None, average='weighted'):
    """Compute classification metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    
    if y_prob is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        metrics['roc_auc'] = float(auc(fpr, tpr))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def plot_roc_curve(y_true, y_prob, save_path, title='ROC Curve'):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return roc_auc


def plot_training_curves(train_losses, val_losses, save_path, title='Training Curves'):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_class_distribution(y, save_path, title='Class Distribution'):
    """Plot and save class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique.astype(float), counts.astype(float), color='steelblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(unique)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    config = {
        'data_path': '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Information_003_20260416_180247/data/NF-UNSW-NB15-v2_3d.pt',
        'output_dir': '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Information_003_20260416_180247/outputs',
        'images_dir': '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Information_003_20260416_180247/report/images',
        'hidden_dim': 128,
        'num_factors': 8,
        'batch_size': 256,
        'epochs': 15,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'seed': 42,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['images_dir'], exist_ok=True)
    set_seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    temporal_data = load_data(config['data_path'])
    X, y_binary, y_multiclass = create_edge_dataset(temporal_data)
    
    print(f"Data shape: X={X.shape}, y_binary={y_binary.shape}, y_multiclass={y_multiclass.shape}")
    print(f"Binary class distribution: {compute_class_distribution(y_binary)}")
    print(f"Multiclass distribution: {compute_class_distribution(y_multiclass)}")
    
    data_overview = {
        'num_samples': int(X.shape[0]),
        'num_features': int(X.shape[1]),
        'binary_classes': int(y_binary.max().item() + 1),
        'multiclass_classes': int(y_multiclass.max().item() + 1),
        'binary_distribution': {str(k): round(v, 4) for k, v in compute_class_distribution(y_binary).items()},
        'multiclass_distribution': {str(k): round(v, 4) for k, v in compute_class_distribution(y_multiclass).items()},
    }
    
    with open(os.path.join(config['output_dir'], 'data_overview.json'), 'w') as f:
        json.dump(data_overview, f, indent=2)
    
    plot_class_distribution(
        y_binary.numpy(), 
        os.path.join(config['images_dir'], 'binary_class_distribution.png'),
        'Binary Classification: Benign vs Attack'
    )
    
    attack_names = ['Normal', 'DoS', 'Probe', 'U2R', 'R2L', 'DDoS', 'Bot', 'Web', 'Exploit', 'Shellcode']
    plot_class_distribution(
        y_multiclass.numpy(),
        os.path.join(config['images_dir'], 'multiclass_class_distribution.png'),
        'Multi-class Attack Type Distribution'
    )
    
    print("Splitting data...")
    n_samples = len(X)
    indices = torch.randperm(n_samples)
    
    train_end = int(config['train_ratio'] * n_samples)
    val_end = int((config['train_ratio'] + config['val_ratio']) * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_bin_train, y_multi_train = X[train_idx], y_binary[train_idx], y_multiclass[train_idx]
    X_val, y_bin_val, y_multi_val = X[val_idx], y_binary[val_idx], y_multiclass[val_idx]
    X_test, y_bin_test, y_multi_test = X[test_idx], y_binary[test_idx], y_multiclass[test_idx]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    train_dataset = TensorDataset(X_train, y_bin_train)
    val_dataset = TensorDataset(X_val, y_bin_val)
    test_dataset = TensorDataset(X_test, y_bin_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print("Initializing model...")
    model = DIDS_MFL(
        input_dim=X.shape[1],
        hidden_dim=config['hidden_dim'],
        num_classes=10,
        num_factors=config['num_factors']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(config['epochs']):
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, criterion, device, task='binary')
        val_loss, _, _, _ = evaluate(model, val_loader, criterion, device, task='binary')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    plot_training_curves(
        train_losses, val_losses,
        os.path.join(config['images_dir'], 'training_curves.png'),
        'Binary Classification Training Curves'
    )
    
    print("\n=== Binary Classification Results ===")
    test_loss, y_bin_pred, y_bin_true, y_bin_prob = evaluate(model, test_loader, criterion, device, task='binary')
    
    binary_metrics = compute_metrics(y_bin_true, y_bin_pred, y_bin_prob)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {binary_metrics['accuracy']:.4f}")
    print(f"Precision: {binary_metrics['precision']:.4f}")
    print(f"Recall: {binary_metrics['recall']:.4f}")
    print(f"F1 Score: {binary_metrics['f1']:.4f}")
    if 'roc_auc' in binary_metrics:
        print(f"ROC-AUC: {binary_metrics['roc_auc']:.4f}")
    
    with open(os.path.join(config['output_dir'], 'binary_metrics.json'), 'w') as f:
        json.dump(binary_metrics, f, indent=2)
    
    plot_confusion_matrix(
        y_bin_true, y_bin_pred, ['Benign', 'Attack'],
        os.path.join(config['images_dir'], 'binary_confusion_matrix.png'),
        'Binary Classification Confusion Matrix'
    )
    
    if 'roc_auc' in binary_metrics:
        plot_roc_curve(
            y_bin_true, y_bin_prob,
            os.path.join(config['images_dir'], 'binary_roc_curve.png'),
            'Binary Classification ROC Curve'
        )
    
    print("\n=== Multi-class Classification Training ===")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_dataset_multi = TensorDataset(X_train, y_multi_train)
    val_dataset_multi = TensorDataset(X_val, y_multi_val)
    test_dataset_multi = TensorDataset(X_test, y_multi_test)
    
    train_loader_multi = DataLoader(train_dataset_multi, batch_size=config['batch_size'], shuffle=True)
    val_loader_multi = DataLoader(val_dataset_multi, batch_size=config['batch_size'], shuffle=False)
    test_loader_multi = DataLoader(test_dataset_multi, batch_size=config['batch_size'], shuffle=False)
    
    train_losses_multi = []
    val_losses_multi = []
    best_val_loss_multi = float('inf')
    best_model_state_multi = None
    
    for epoch in range(config['epochs']):
        train_loss, _, _ = train_epoch(model, train_loader_multi, optimizer, criterion, device, task='multiclass')
        val_loss, _, _, _ = evaluate(model, val_loader_multi, criterion, device, task='multiclass')
        
        train_losses_multi.append(train_loss)
        val_losses_multi.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss_multi:
            best_val_loss_multi = val_loss
            best_model_state_multi = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    if best_model_state_multi is not None:
        model.load_state_dict(best_model_state_multi)
    
    plot_training_curves(
        train_losses_multi, val_losses_multi,
        os.path.join(config['images_dir'], 'multiclass_training_curves.png'),
        'Multi-class Classification Training Curves'
    )
    
    print("\n=== Multi-class Classification Results ===")
    test_loss_multi, y_multi_pred, y_multi_true, y_multi_prob = evaluate(
        model, test_loader_multi, criterion, device, task='multiclass'
    )
    
    multi_metrics = compute_metrics(y_multi_true, y_multi_pred, average='weighted')
    print(f"Test Loss: {test_loss_multi:.4f}")
    print(f"Accuracy: {multi_metrics['accuracy']:.4f}")
    print(f"Precision: {multi_metrics['precision']:.4f}")
    print(f"Recall: {multi_metrics['recall']:.4f}")
    print(f"F1 Score: {multi_metrics['f1']:.4f}")
    
    print("\nPer-class F1 scores:")
    class_f1 = f1_score(y_multi_true, y_multi_pred, average=None, zero_division=0)
    per_class_f1_dict = {}
    for i, f1 in enumerate(class_f1):
        print(f"  {attack_names[i]}: {f1:.4f}")
        per_class_f1_dict[attack_names[i]] = float(f1)
    
    multi_metrics_full = {
        'overall': multi_metrics,
        'per_class_f1': per_class_f1_dict
    }
    with open(os.path.join(config['output_dir'], 'multiclass_metrics.json'), 'w') as f:
        json.dump(multi_metrics_full, f, indent=2)
    
    plot_confusion_matrix(
        y_multi_true, y_multi_pred, attack_names,
        os.path.join(config['images_dir'], 'multiclass_confusion_matrix.png'),
        'Multi-class Classification Confusion Matrix'
    )
    
    print("\n=== Few-shot Learning Evaluation ===")
    support_idx, query_idx = create_few_shot_split(X, y_multiclass, shots_per_class=5)
    
    print(f"Support set size: {len(support_idx)}, Query set size: {len(query_idx)}")
    
    few_shot_metrics = None
    if len(query_idx) > 0:
        X_support = X[support_idx].to(device)
        y_support_multi = y_multiclass[support_idx].to(device)
        
        few_shot_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        few_shot_dataset = TensorDataset(X_support, y_support_multi)
        few_shot_loader = DataLoader(few_shot_dataset, batch_size=min(16, len(support_idx)), shuffle=True)
        
        model.train()
        for fs_epoch in range(10):
            for batch in few_shot_loader:
                X_batch, y_batch = batch
                num_edges = X_batch.shape[0]
                num_nodes = max(50, num_edges // 2)
                edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
                edge_attr = X_batch.clone()
                time = torch.randn(num_edges, device=device)
                
                few_shot_optimizer.zero_grad()
                logits = model(X_batch, edge_index, edge_attr, time, task='multiclass')
                loss = criterion(logits, y_batch)
                loss.backward()
                few_shot_optimizer.step()
        
        X_query = X[query_idx].to(device)
        y_query_multi = y_multiclass[query_idx]
        
        model.eval()
        with torch.no_grad():
            num_edges = X_query.shape[0]
            num_nodes = max(50, num_edges // 2)
            edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
            edge_attr = X_query.clone()
            time = torch.randn(num_edges, device=device)
            
            logits = model(X_query, edge_index, edge_attr, time, task='multiclass')
            y_query_pred = torch.argmax(logits, dim=-1).cpu().numpy()
        
        few_shot_metrics = compute_metrics(y_query_multi.numpy(), y_query_pred, average='weighted')
        print(f"Few-shot Accuracy: {few_shot_metrics['accuracy']:.4f}")
        print(f"Few-shot F1 Score: {few_shot_metrics['f1']:.4f}")
        
        with open(os.path.join(config['output_dir'], 'fewshot_metrics.json'), 'w') as f:
            json.dump(few_shot_metrics, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(config['output_dir'], 'dids_mfl_model.pt'))
    
    summary = {
        'config': config,
        'binary_metrics': binary_metrics,
        'multiclass_metrics': multi_metrics_full,
        'data_overview': data_overview,
    }
    if few_shot_metrics is not None:
        summary['fewshot_metrics'] = few_shot_metrics
    
    with open(os.path.join(config['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Training Complete ===")
    print(f"Results saved to: {config['output_dir']}")
    print(f"Figures saved to: {config['images_dir']}")


if __name__ == '__main__':
    main()
