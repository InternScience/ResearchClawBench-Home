"""
Training and evaluation script for KA-GNN and GCN baseline on molecular property datasets.
"""

import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, auc

warnings.filterwarnings('ignore')

# Add code directory to path
sys.path.insert(0, os.path.dirname(__file__))

from kagcn import (
    KAGNN, GCNBaseline, smiles_to_graph, scaffold_split,
    get_atom_features, get_bond_features
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================
# Dataset wrapper
# ============================================================

class MolecularDataset(Dataset):
    def __init__(self, smiles_list, labels, transform=None):
        self.smiles_list = smiles_list
        self.labels = labels
        self.transform = transform
        self.graphs = []
        self.valid_indices = []
        
        for idx, smi in enumerate(smiles_list):
            g = smiles_to_graph(smi)
            if g is not None:
                self.graphs.append(g)
                self.valid_indices.append(idx)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        data = self.graphs[idx]
        label = self.labels[self.valid_indices[idx]]
        data.y = torch.tensor(label, dtype=torch.float)
        return data


def collate_fn(batch):
    return Batch.from_data_list(batch)


# ============================================================
# Training loop
# ============================================================

def safe_roc_auc(labels, preds):
    """Compute ROC-AUC safely, returning 0.5 if not computable."""
    try:
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return 0.5
        return roc_auc_score(labels, preds)
    except:
        return 0.5


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.view(-1), batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        preds = torch.sigmoid(out).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(batch.y.detach().cpu().numpy().flatten())
    
    avg_loss = total_loss / len(loader.dataset)
    roc_auc = safe_roc_auc(all_labels, all_preds)
    
    return avg_loss, roc_auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out.view(-1), batch.y)
        
        total_loss += loss.item() * batch.num_graphs
        preds = torch.sigmoid(out).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(batch.y.detach().cpu().numpy().flatten())
    
    avg_loss = total_loss / len(loader.dataset)
    roc_auc = safe_roc_auc(all_labels, all_preds)
    acc = accuracy_score(all_labels, [p > 0.5 for p in all_preds])
    try:
        f1 = f1_score(all_labels, [p > 0.5 for p in all_preds], zero_division=0)
    except:
        f1 = 0.0
    
    return avg_loss, roc_auc, acc, f1, all_preds, all_labels


def run_experiment(dataset_name, smiles_list, labels, num_tasks=1, 
                   hidden_dim=128, num_layers=3, grid_size=5,
                   epochs=50, batch_size=64, lr=1e-3, seed=42):
    """Run a full experiment for one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(smiles_list)}, Tasks: {num_tasks}")
    print(f"Config: hidden={hidden_dim}, layers={num_layers}, grid={grid_size}, "
          f"epochs={epochs}, bs={batch_size}, lr={lr}")
    
    # Try scaffold split, fall back to random if it creates bad splits
    np.random.seed(seed)
    train_idx, val_idx, test_idx = scaffold_split(smiles_list, seed=seed)
    
    # Check split quality - ensure each split has both classes
    train_labels_set = set(labels[i] for i in train_idx)
    val_labels_set = set(labels[i] for i in val_idx)
    test_labels_set = set(labels[i] for i in test_idx)
    
    if len(val_labels_set) < 2 or len(test_labels_set) < 2 or len(train_labels_set) < 2:
        print("Scaffold split created imbalanced splits, using random split instead")
        indices = np.arange(len(smiles_list))
        np.random.shuffle(indices)
        n = len(indices)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train+n_val].tolist()
        test_idx = indices[n_train+n_val:].tolist()
    
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    train_smiles = [smiles_list[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_smiles = [smiles_list[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_smiles = [smiles_list[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    train_ds = MolecularDataset(train_smiles, train_labels)
    val_ds = MolecularDataset(val_smiles, val_labels)
    test_ds = MolecularDataset(test_smiles, test_labels)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        print("Warning: empty split, skipping")
        return None
    
    # Determine input dimensions from first sample
    sample = train_ds[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1] if sample.edge_attr.dim() > 1 else 1
    
    print(f"Node features: {node_in_dim}, Edge features: {edge_in_dim}")
    
    # Compute class weights for imbalanced datasets
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    if pos_count > 0:
        pos_weight = neg_count / pos_count
    else:
        pos_weight = 1.0
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))
    
    results = {}
    
    for model_name, model_cls in [('GCN', GCNBaseline), ('KA-GNN', KAGNN)]:
        print(f"\n--- Training {model_name} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if model_name == 'KA-GNN':
            model = model_cls(
                node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim, num_layers=num_layers,
                grid_size=grid_size, dropout=0.2, num_tasks=num_tasks
            ).to(DEVICE)
        else:
            model = model_cls(
                node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim, num_layers=num_layers,
                dropout=0.2, num_tasks=num_tasks
            ).to(DEVICE)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        
        train_losses, val_losses, val_aucs = [], [], []
        best_val_auc = 0
        best_state = None
        patience_counter = 0
        early_stop_patience = 15
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_auc, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            
            scheduler.step(val_auc)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
                      f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} | Patience: {patience_counter}")
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        # Load best model and evaluate on test
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)
        test_loss, test_auc, test_acc, test_f1, test_preds, test_labels_arr = evaluate(
            model, test_loader, criterion, DEVICE
        )
        
        # Also get final validation metrics
        _, final_val_auc, final_val_acc, final_val_f1, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )
        
        print(f"\n{model_name} Results:")
        print(f"  Val ROC-AUC:   {final_val_auc:.4f}")
        print(f"  Test ROC-AUC:  {test_auc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1:       {test_f1:.4f}")
        print(f"  Training Time: {training_time:.1f}s")
        
        # PR-AUC
        try:
            precision, recall, _ = precision_recall_curve(test_labels_arr, test_preds)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = 0.0
        
        results[model_name] = {
            'val_roc_auc': float(final_val_auc),
            'test_roc_auc': float(test_auc),
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'test_pr_auc': float(pr_auc),
            'training_time': float(training_time),
            'num_parameters': int(num_params),
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'val_aucs': [float(x) for x in val_aucs],
            'test_preds': [float(x) for x in test_preds],
            'test_labels': [int(x) for x in test_labels_arr],
            'best_epoch': int(len(train_losses)),
        }
    
    return results


# ============================================================
# Main execution
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['bace', 'bbbp', 'clintox', 'hiv'],
                        help='Datasets to run (muv is very large, run separately)')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    all_results = {}
    
    for ds_name in args.datasets:
        filepath = f'data/{ds_name}.csv'
        df = pd.read_csv(filepath)
        
        # Find SMILES and label columns
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        
        # Determine label column(s)
        if ds_name == 'clintox':
            label_col = 'CT_TOX'  # Clinical toxicity
            num_tasks = 1
        elif ds_name == 'muv':
            # Use first MUV task
            label_col = 'MUV-466'
            num_tasks = 1
        else:
            label_col = 'label'
            num_tasks = 1
        
        smiles_list = df[smiles_col].tolist()
        labels = df[label_col].fillna(0).astype(float).tolist()
        
        result = run_experiment(
            dataset_name=ds_name,
            smiles_list=smiles_list,
            labels=labels,
            num_tasks=num_tasks,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            grid_size=args.grid_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed
        )
        
        if result is not None:
            all_results[ds_name] = result
    
    # Save results
    with open('outputs/experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to outputs/experiment_results.json")
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Dataset':<12} {'Model':<10} {'Val AUC':<10} {'Test AUC':<10} {'Test Acc':<10} {'Time(s)':<10}")
    print("-" * 62)
    for ds_name, res in all_results.items():
        for model_name in ['GCN', 'KA-GNN']:
            if model_name in res:
                r = res[model_name]
                print(f"{ds_name:<12} {model_name:<10} {r['val_roc_auc']:<10.4f} "
                      f"{r['test_roc_auc']:<10.4f} {r['test_accuracy']:<10.4f} "
                      f"{r['training_time']:<10.1f}")
