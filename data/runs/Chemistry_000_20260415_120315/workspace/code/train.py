"""
Training script for KA-GNN and baseline models on molecular property prediction.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_utils import get_dataloader, load_dataset, MoleculeDataset, collate_fn
from kagnn_model import KAGNN, GCN

# Set random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed()

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        atom_features = batch['atom_features'].to(device)
        bond_features = batch['bond_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_idx = batch['batch'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(atom_features, bond_features, edge_index, batch_idx)
        
        # Handle multi-task (ClinTox has 2 labels)
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # For multi-task, use first task for metrics
                preds = torch.sigmoid(outputs[:, 0])
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels[:, 0].cpu().numpy())
            else:
                preds = torch.sigmoid(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Filter valid labels
    valid_mask = ~np.isnan(all_labels)
    if valid_mask.sum() > 0:
        all_preds = all_preds[valid_mask]
        all_labels = all_labels[valid_mask]
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    else:
        auc = 0.5
        acc = 0.5
    
    return avg_loss, auc, acc

def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            atom_features = batch['atom_features'].to(device)
            bond_features = batch['bond_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            batch_idx = batch['batch'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(atom_features, bond_features, edge_index, batch_idx)
            
            # Handle multi-task
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs[:, 0])
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels[:, 0].cpu().numpy())
            else:
                loss = criterion(outputs, labels.squeeze())
                preds = torch.sigmoid(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Filter valid labels
    valid_mask = ~np.isnan(all_labels)
    if valid_mask.sum() > 0:
        all_preds = all_preds[valid_mask]
        all_labels = all_labels[valid_mask]
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    else:
        auc = 0.5
        acc = 0.5
    
    return avg_loss, auc, acc, all_preds, all_labels

def train_model(model_name, dataset_name, config, device='cpu'):
    """
    Train a model on a dataset.
    
    Args:
        model_name: 'kagnn', 'gcn', or 'mlp_gnn'
        dataset_name: 'bace', 'bbbp', 'clintox', 'hiv', 'muv'
        config: Dictionary with hyperparameters
        device: 'cpu' or 'cuda'
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load data
    graphs, df = load_dataset(dataset_name)
    
    # Split data
    indices = list(range(len(graphs)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, 
                                           stratify=[g.label if isinstance(g.label, (int, float, np.integer)) else g.label[0] for g in graphs])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    train_dataset = MoleculeDataset(train_graphs)
    val_dataset = MoleculeDataset(val_graphs)
    test_dataset = MoleculeDataset(test_graphs)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    num_classes = 2 if dataset_name == 'clintox' else 1
    
    if model_name == 'kagnn':
        model = KAGNN(
            node_feature_dim=7,
            edge_feature_dim=7,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            use_kan=True,
            num_frequencies=config.get('num_frequencies', 8),
            omega=config.get('omega', 1.0)
        )
    elif model_name == 'gcn':
        model = GCN(
            node_feature_dim=7,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=num_classes
        )
    elif model_name == 'mlp_gnn':
        model = KAGNN(
            node_feature_dim=7,
            edge_feature_dim=7,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=num_classes,
            use_kan=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Loss and optimizer
    if num_classes > 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-5))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    max_patience = config.get('patience', 30)
    
    history = {
        'train_loss': [],
        'train_auc': [],
        'train_acc': [],
        'val_loss': [],
        'val_auc': [],
        'val_acc': []
    }
    
    for epoch in range(config['epochs']):
        train_loss, train_auc, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            os.makedirs('outputs/models', exist_ok=True)
            torch.save(model.state_dict(), f'outputs/models/{model_name}_{dataset_name}_best.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'outputs/models/{model_name}_{dataset_name}_best.pt', map_location=device))
    test_loss, test_auc, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBest validation AUC: {best_val_auc:.4f} (epoch {best_epoch+1})")
    print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")
    
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'best_epoch': best_epoch,
        'history': history
    }
    
    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    with open(f'outputs/results/{model_name}_{dataset_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, history

def run_experiments(datasets=None, models=None, device='cpu'):
    """Run experiments on all datasets and models."""
    
    if datasets is None:
        datasets = ['bace', 'bbbp', 'clintox', 'hiv', 'muv']
    
    if models is None:
        models = ['kagnn', 'gcn', 'mlp_gnn']
    
    # Common hyperparameters
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 20,
        'num_frequencies': 8,
        'omega': 1.0
    }
    
    all_results = {}
    
    for dataset in datasets:
        all_results[dataset] = {}
        for model in models:
            try:
                results, history = train_model(model, dataset, config, device)
                all_results[dataset][model] = results
            except Exception as e:
                print(f"Error training {model} on {dataset}: {e}")
                import traceback
                traceback.print_exc()
                all_results[dataset][model] = {'error': str(e)}
    
    # Save combined results
    with open('outputs/results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run experiments
    datasets = ['bace', 'bbbp', 'clintox', 'hiv', 'muv']
    models = ['kagnn', 'gcn', 'mlp_gnn']
    
    results = run_experiments(datasets=datasets, models=models, device=device)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        for model in models:
            if model in results[dataset] and 'test_auc' in results[dataset][model]:
                test_auc = results[dataset][model]['test_auc']
                print(f"  {model:12s}: Test AUC = {test_auc:.4f}")
