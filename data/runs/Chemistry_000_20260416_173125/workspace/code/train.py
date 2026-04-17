"""
Main training and evaluation pipeline for KA-GNN molecular property prediction.
Trains both GCN-MLP (baseline) and KA-GNN on all 5 MoleculeNet datasets.
"""
import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import smiles_to_graph, get_atom_feature_dim
from models import GCN_MLP, KA_GNN, count_parameters

# Configuration
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cpu')
SEED = 42

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---- Dataset Loading ----

def load_bace():
    df = pd.read_csv(os.path.join(DATA_DIR, 'bace.csv'))
    smiles = df['smiles'].tolist()
    labels = df['label'].values.astype(float)
    return smiles, labels, 1, 'BACE', 'scaffold'

def load_bbbp():
    df = pd.read_csv(os.path.join(DATA_DIR, 'bbbp.csv'))
    smiles = df['smiles'].tolist()
    labels = df['label'].values.astype(float)
    return smiles, labels, 1, 'BBBP', 'scaffold'

def load_clintox():
    df = pd.read_csv(os.path.join(DATA_DIR, 'clintox.csv'))
    smiles = df['smiles'].tolist()
    # Multi-task: FDA_APPROVED and CT_TOX
    labels = df[['FDA_APPROVED', 'CT_TOX']].values.astype(float)
    return smiles, labels, 2, 'ClinTox', 'random'

def load_hiv():
    df = pd.read_csv(os.path.join(DATA_DIR, 'hiv.csv'))
    smiles = df['smiles'].tolist()
    labels = df['label'].values.astype(float)
    return smiles, labels, 1, 'HIV', 'scaffold'

def load_muv():
    df = pd.read_csv(os.path.join(DATA_DIR, 'muv.csv'))
    smiles = df['smiles'].tolist()
    task_cols = [c for c in df.columns if c.startswith('MUV-')]
    labels = df[task_cols].values.astype(float)
    return smiles, labels, len(task_cols), 'MUV', 'random'

# ---- Scaffold Split ----

def scaffold_split(smiles_list, labels, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Scaffold-based split for molecular datasets."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            except:
                scaffold = smi
        else:
            scaffold = smi
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(i)
    
    # Sort scaffolds by size (largest first for determinism)
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    
    n = len(smiles_list)
    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))
    
    train_idx, val_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_cutoff:
            train_idx.extend(scaffold_set)
        elif len(train_idx) + len(val_idx) + len(scaffold_set) <= val_cutoff:
            val_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)
    
    # Ensure test set has both classes - if not, swap some samples
    if isinstance(labels, np.ndarray) and labels.ndim == 1:
        test_labels = labels[test_idx] if len(test_idx) > 0 else np.array([])
        val_labels = labels[val_idx] if len(val_idx) > 0 else np.array([])
        
        if len(test_labels) > 0 and len(np.unique(test_labels)) < 2:
            # Fall back to random split to ensure both classes
            print("  WARNING: Scaffold split produced single-class test set. Using stratified random split.")
            return None, None, None
    
    return train_idx, val_idx, test_idx

def random_split(n, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Random split."""
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))
    return (indices[:train_cutoff].tolist(), 
            indices[train_cutoff:val_cutoff].tolist(), 
            indices[val_cutoff:].tolist())

def stratified_split(labels_1d, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Stratified random split ensuring both classes in all sets."""
    n = len(labels_1d)
    indices = np.arange(n)
    
    # First split: train+val vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=labels_1d)
    
    # Second split: train vs val
    val_relative = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_relative, random_state=seed, 
        stratify=labels_1d[train_val_idx])
    
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

# ---- Graph Dataset Creation ----

def create_graph_dataset(smiles_list, labels, num_tasks):
    """Convert SMILES to graph dataset."""
    graphs = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        if num_tasks == 1:
            y = float(labels[i])
        else:
            y = labels[i]
        
        graph = smiles_to_graph(smi)
        if graph is not None:
            if num_tasks == 1:
                graph.y = torch.tensor([[y]], dtype=torch.float)
            else:
                graph.y = torch.tensor([y], dtype=torch.float)
            graphs.append(graph)
            valid_indices.append(i)
    
    return graphs, valid_indices

# ---- Training ----

def train_epoch(model, loader, optimizer, criterion, num_tasks=1):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch)
        
        y = batch.y
        if num_tasks == 1:
            y = y.view(-1, 1)
        else:
            y = y.view(-1, num_tasks)
        
        # Handle NaN labels (for multi-task datasets like MUV)
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        
        loss = criterion(out[mask], y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)

@torch.no_grad()
def evaluate(model, loader, num_tasks=1, metric='roc_auc'):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        preds = torch.sigmoid(out)
        
        y = batch.y
        if num_tasks == 1:
            y = y.view(-1, 1)
        else:
            y = y.view(-1, num_tasks)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    scores = []
    for task_i in range(num_tasks):
        mask = ~np.isnan(all_labels[:, task_i])
        if mask.sum() < 5:
            continue
        y_true = all_labels[mask, task_i]
        y_pred = all_preds[mask, task_i]
        
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            continue
        
        try:
            if metric == 'roc_auc':
                score = roc_auc_score(y_true, y_pred)
            elif metric == 'prc_auc':
                score = average_precision_score(y_true, y_pred)
            scores.append(score)
        except:
            continue
    
    return np.mean(scores) if scores else 0.0, all_preds, all_labels

# ---- Main Pipeline ----

def run_experiment(dataset_loader, hidden_dim=64, num_layers=3, num_frequencies=8,
                   epochs=50, batch_size=64, lr=1e-3, patience=10):
    """Run full experiment on one dataset."""
    set_seed(SEED)
    
    smiles, labels, num_tasks, name, split_type = dataset_loader()
    print(f"\n{'='*60}")
    print(f"Dataset: {name} | Tasks: {num_tasks} | Molecules: {len(smiles)}")
    print(f"Split: {split_type}")
    
    # Determine metric
    metric = 'prc_auc' if name == 'MUV' else 'roc_auc'
    metric_name = 'PRC-AUC' if metric == 'prc_auc' else 'ROC-AUC'
    
    # Create graphs
    print("Creating molecular graphs...")
    graphs, valid_indices = create_graph_dataset(smiles, labels, num_tasks)
    print(f"Valid graphs: {len(graphs)}/{len(smiles)}")
    
    # Get labels for valid indices
    if num_tasks == 1:
        valid_labels = labels[valid_indices]
    else:
        valid_labels = labels[valid_indices]
    
    # Split
    if split_type == 'scaffold':
        valid_smiles = [smiles[i] for i in valid_indices]
        train_idx, val_idx, test_idx = scaffold_split(
            valid_smiles, valid_labels)
        
        # If scaffold split failed (single class), fall back to stratified
        if train_idx is None:
            if num_tasks == 1:
                train_idx, val_idx, test_idx = stratified_split(valid_labels.astype(int))
            else:
                train_idx, val_idx, test_idx = random_split(len(graphs))
    else:
        if num_tasks == 1:
            try:
                train_idx, val_idx, test_idx = stratified_split(valid_labels.astype(int))
            except:
                train_idx, val_idx, test_idx = random_split(len(graphs))
        else:
            train_idx, val_idx, test_idx = random_split(len(graphs))
    
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    
    # Verify class balance
    if num_tasks == 1:
        train_labels = valid_labels[train_idx]
        test_labels = valid_labels[test_idx]
        print(f"Train class dist: {np.bincount(train_labels.astype(int))}")
        print(f"Test class dist: {np.bincount(test_labels.astype(int))}")
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    # For large datasets, subsample for efficiency on CPU
    max_train = 5000
    if len(train_graphs) > max_train:
        np.random.seed(SEED)
        idx = np.random.choice(len(train_graphs), max_train, replace=False)
        train_graphs = [train_graphs[i] for i in idx]
        print(f"Subsampled training to {max_train} for CPU efficiency")
    
    max_eval = 2000
    if len(test_graphs) > max_eval:
        np.random.seed(SEED)
        idx = np.random.choice(len(test_graphs), max_eval, replace=False)
        test_graphs = [test_graphs[i] for i in idx]
    if len(val_graphs) > max_eval:
        np.random.seed(SEED)
        idx = np.random.choice(len(val_graphs), max_eval, replace=False)
        val_graphs = [val_graphs[i] for i in idx]
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    
    in_dim = get_atom_feature_dim()
    results = {}
    
    for model_name, ModelClass, kwargs in [
        ('GCN-MLP', GCN_MLP, {'in_features': in_dim, 'hidden_dim': hidden_dim, 
                                'num_layers': num_layers, 'num_tasks': num_tasks}),
        ('KA-GNN', KA_GNN, {'in_features': in_dim, 'hidden_dim': hidden_dim, 
                             'num_layers': num_layers, 'num_tasks': num_tasks,
                             'num_frequencies': num_frequencies}),
    ]:
        set_seed(SEED)
        print(f"\n--- Training {model_name} ---")
        model = ModelClass(**kwargs).to(DEVICE)
        n_params = count_parameters(model)
        print(f"Parameters: {n_params:,}")
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_score = -1
        best_test_score = 0
        best_epoch = 0
        no_improve = 0
        train_losses = []
        val_scores = []
        test_scores = []
        best_preds = None
        best_labels_arr = None
        fourier_coeffs = None
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, num_tasks)
            val_score, _, _ = evaluate(model, val_loader, num_tasks, metric)
            test_score, test_preds, test_labels_arr = evaluate(model, test_loader, num_tasks, metric)
            
            scheduler.step(val_score)
            
            train_losses.append(loss)
            val_scores.append(val_score)
            test_scores.append(test_score)
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_test_score = test_score
                best_epoch = epoch
                best_preds = test_preds.copy()
                best_labels_arr = test_labels_arr.copy()
                no_improve = 0
                # Save Fourier coefficients for KA-GNN
                if model_name == 'KA-GNN' and hasattr(model, 'get_fourier_coefficients'):
                    fourier_coeffs = model.get_fourier_coefficients()
            else:
                no_improve += 1
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val {metric_name}: {val_score:.4f} | Test {metric_name}: {test_score:.4f}")
            
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        elapsed = time.time() - start_time
        
        # If no improvement was ever recorded, use last epoch
        if best_preds is None:
            best_preds = test_preds
            best_labels_arr = test_labels_arr
        
        print(f"  Best epoch: {best_epoch} | Best Val {metric_name}: {best_val_score:.4f} | Best Test {metric_name}: {best_test_score:.4f}")
        print(f"  Training time: {elapsed:.1f}s")
        
        results[model_name] = {
            'best_val_score': best_val_score,
            'best_test_score': best_test_score,
            'best_epoch': best_epoch,
            'n_params': n_params,
            'training_time': elapsed,
            'train_losses': train_losses,
            'val_scores': val_scores,
            'test_scores': test_scores,
            'test_preds': best_preds,
            'test_labels': best_labels_arr,
            'metric': metric_name,
        }
        
        if fourier_coeffs is not None:
            results[model_name]['fourier_coefficients'] = fourier_coeffs
    
    return name, results

def main():
    all_results = {}
    
    datasets = [
        load_bace,
        load_bbbp,
        load_clintox,
        load_hiv,
        load_muv,
    ]
    
    for loader in datasets:
        name, results = run_experiment(
            loader,
            hidden_dim=64,
            num_layers=3,
            num_frequencies=8,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            patience=15,
        )
        all_results[name] = results
    
    # Save summary results
    summary = {}
    for dataset_name, results in all_results.items():
        summary[dataset_name] = {}
        for model_name, r in results.items():
            summary[dataset_name][model_name] = {
                'test_score': round(r['best_test_score'], 4),
                'val_score': round(r['best_val_score'], 4),
                'best_epoch': r['best_epoch'],
                'n_params': r['n_params'],
                'training_time': round(r['training_time'], 1),
                'metric': r['metric'],
            }
    
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results for plotting
    detailed = {}
    for dataset_name, results in all_results.items():
        detailed[dataset_name] = {}
        for model_name, r in results.items():
            detailed[dataset_name][model_name] = {
                'train_losses': r['train_losses'],
                'val_scores': r['val_scores'],
                'test_scores': r['test_scores'],
                'test_preds': r['test_preds'].tolist(),
                'test_labels': r['test_labels'].tolist(),
                'metric': r['metric'],
            }
            if 'fourier_coefficients' in r:
                fc = r['fourier_coefficients']
                fc_info = {}
                for layer_name, coeffs in fc.items():
                    fc_info[layer_name] = {
                        'a0_shape': list(coeffs['a0'].shape),
                        'a_cos_shape': list(coeffs['a_cos'].shape),
                        'freq_scale': coeffs['freq_scale'],
                        'a0_mean_abs': float(np.mean(np.abs(coeffs['a0']))),
                        'a_cos_mean_abs': float(np.mean(np.abs(coeffs['a_cos']))),
                        'b_sin_mean_abs': float(np.mean(np.abs(coeffs['b_sin']))),
                    }
                detailed[dataset_name][model_name]['fourier_info'] = fc_info
    
    with open(os.path.join(OUTPUT_DIR, 'results_detailed.json'), 'w') as f:
        json.dump(detailed, f, indent=2)
    
    # Save Fourier coefficients for interpretability
    for dataset_name, results in all_results.items():
        if 'KA-GNN' in results and 'fourier_coefficients' in results['KA-GNN']:
            fc = results['KA-GNN']['fourier_coefficients']
            fc_save = {}
            for layer_name, coeffs in fc.items():
                fc_save[layer_name] = {
                    'a0': coeffs['a0'].tolist(),
                    'a_cos_magnitude': np.sqrt(np.sum(coeffs['a_cos']**2, axis=-1)).tolist(),
                    'b_sin_magnitude': np.sqrt(np.sum(coeffs['b_sin']**2, axis=-1)).tolist(),
                    'freq_scale': coeffs['freq_scale'],
                }
            with open(os.path.join(OUTPUT_DIR, f'fourier_coeffs_{dataset_name}.json'), 'w') as f:
                json.dump(fc_save, f)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Dataset':<12} {'Model':<10} {'Metric':<10} {'Test Score':<12} {'Params':<10} {'Time(s)':<8}")
    print("-"*62)
    for dataset_name in all_results:
        for model_name in all_results[dataset_name]:
            r = all_results[dataset_name][model_name]
            print(f"{dataset_name:<12} {model_name:<10} {r['metric']:<10} {r['best_test_score']:<12.4f} {r['n_params']:<10,} {r['training_time']:<8.1f}")
    
    return all_results

if __name__ == '__main__':
    all_results = main()
