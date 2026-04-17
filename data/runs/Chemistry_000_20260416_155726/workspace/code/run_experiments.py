"""
Fast Training and Evaluation Script for KA-GNNs
Optimized for quick execution with reduced epochs and smaller models.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

from ka_gnn import mol_to_graph, GCN, GAT, KAGNN, train_epoch, evaluate

DEVICE = torch.device('cpu')
HIDDEN_FEATURES = 32
NUM_LAYERS = 2
NUM_FOURIER_TERMS = 4
DROPOUT = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.01

DATASETS = ['bace', 'bbbp', 'clintox', 'hiv']
DATA_DIR = '../data'
OUTPUT_DIR = '../outputs'
IMAGES_DIR = '../report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_dataset(name, max_samples=1000):
    """Load a dataset with optional subsampling."""
    filepath = os.path.join(DATA_DIR, f'{name}.csv')
    df = pd.read_csv(filepath)
    
    # Subsample for speed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    graphs = []
    labels = []
    
    label_col = 'label'
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row[label_col]
        graph = mol_to_graph(smiles, int(label))
        if graph is not None:
            graphs.append(graph)
            labels.append(int(label))
    
    print(f"  Loaded {len(graphs)} graphs for {name}")
    return graphs, np.array(labels)


def create_data_loaders(graphs, batch_size=BATCH_SIZE):
    """Create train/val/test loaders."""
    indices = list(range(len(graphs)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, 
                                           stratify=[g.y.item() for g in graphs])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, 
                                         stratify=[graphs[i].y.item() for i in temp_idx])
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def compute_metrics(scores, labels):
    """Compute evaluation metrics."""
    metrics = {}
    
    if len(np.unique(labels)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(labels, scores)
        except:
            metrics['roc_auc'] = 0.5
    else:
        metrics['roc_auc'] = 0.5
    
    try:
        metrics['pr_auc'] = average_precision_score(labels, scores)
    except:
        metrics['pr_auc'] = 0.5
    
    predictions = (scores > 0.5).astype(float)
    metrics['accuracy'] = (predictions == labels).mean()
    
    if len(np.unique(labels)) > 1:
        metrics['f1'] = f1_score(labels, predictions, zero_division=0)
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels, predictions)
    else:
        metrics['f1'] = 0.0
        metrics['balanced_accuracy'] = 0.5
    
    return metrics


def train_model_fast(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train model quickly."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, use_edge_attr=False)
        val_results = evaluate(model, val_loader, criterion, DEVICE, use_edge_attr=False)
        
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_results['loss'])
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_results['accuracy'])
        
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return history


def run_single_experiment(dataset_name, model_class, model_name, model_kwargs):
    """Run experiment on single dataset/model."""
    graphs, _ = load_dataset(dataset_name)
    if len(graphs) == 0:
        return None
    
    train_loader, val_loader, test_loader = create_data_loaders(graphs)
    in_features = graphs[0].x.shape[1]
    
    model = model_class(in_features=in_features, **model_kwargs).to(DEVICE)
    history = train_model_fast(model, train_loader, val_loader)
    
    criterion = nn.BCEWithLogitsLoss()
    test_results = evaluate(model, test_loader, criterion, DEVICE, use_edge_attr=False)
    test_metrics = compute_metrics(test_results['scores'], test_results['labels'])
    
    print(f"  {model_name}: ROC-AUC={test_metrics['roc_auc']:.4f}, Acc={test_metrics['accuracy']:.4f}")
    
    return {
        'dataset': dataset_name,
        'model': model_name,
        'test_metrics': test_metrics,
        'history': history,
        'test_scores': test_results['scores'].tolist(),
        'test_labels': test_results['labels'].tolist()
    }


def main():
    """Run all experiments."""
    all_results = []
    
    models = [
        ('GCN', GCN, {'hidden_features': HIDDEN_FEATURES, 'num_layers': NUM_LAYERS, 'dropout': DROPOUT}),
        ('GAT', GAT, {'hidden_features': HIDDEN_FEATURES, 'num_layers': NUM_LAYERS, 'num_heads': 2, 'dropout': DROPOUT}),
        ('KA-GNN', KAGNN, {'hidden_features': HIDDEN_FEATURES, 'num_layers': NUM_LAYERS, 
                           'num_fourier_terms': NUM_FOURIER_TERMS, 'dropout': DROPOUT}),
    ]
    
    for dataset in DATASETS:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*50}")
        
        for model_name, model_class, model_kwargs in models:
            result = run_single_experiment(dataset, model_class, model_name, model_kwargs)
            if result:
                all_results.append(result)
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary
    generate_summary_and_plots(all_results)
    
    return all_results


def generate_summary_and_plots(results):
    """Generate summary table and plots."""
    df = pd.DataFrame([{
        'Dataset': r['dataset'],
        'Model': r['model'],
        'ROC-AUC': r['test_metrics']['roc_auc'],
        'PR-AUC': r['test_metrics']['pr_auc'],
        'Accuracy': r['test_metrics']['accuracy'],
        'F1': r['test_metrics']['f1']
    } for r in results])
    
    df.to_csv(os.path.join(OUTPUT_DIR, 'results_summary.csv'), index=False)
    
    # Plot 1: ROC-AUC comparison
    plt.figure(figsize=(10, 6))
    pivot = df.pivot(index='Dataset', columns='Model', values='ROC-AUC')
    ax = pivot.plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.xlabel('Dataset')
    plt.ylabel('ROC-AUC')
    plt.title('Model Comparison: ROC-AUC Across Datasets')
    plt.legend(loc='lower right')
    plt.xticks(rotation=0)
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'roc_auc_comparison.png'), dpi=150)
    plt.close()
    
    # Plot 2: Heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', center=0.75, vmin=0.5, vmax=1.0)
    plt.title('ROC-AUC Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'performance_heatmap.png'), dpi=150)
    plt.close()
    
    # Plot 3: Learning curves
    if results:
        sample = results[0]
        h = sample['history']
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(h['train_losses'], label='Train', alpha=0.8)
        plt.plot(h['val_losses'], label='Val', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"{sample['dataset']} - {sample['model']}")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(h['train_accs'], label='Train', alpha=0.8)
        plt.plot(h['val_accs'], label='Val', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'learning_curves.png'), dpi=150)
        plt.close()
    
    # Plot 4: Class distribution
    plt.figure(figsize=(12, 4))
    for i, ds in enumerate(DATASETS):
        ds_res = [r for r in results if r['dataset'] == ds]
        if ds_res:
            labels = np.array(ds_res[0]['test_labels'])
            plt.subplot(1, len(DATASETS), i+1)
            plt.bar(['Neg', 'Pos'], [(labels==0).sum(), (labels==1).sum()], 
                    color=['#3498db', '#e74c3c'])
            plt.title(ds.upper())
            plt.xticks(fontsize=8)
    plt.suptitle('Class Distribution (Test Sets)')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'class_distribution.png'), dpi=150)
    plt.close()
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Plots saved to {IMAGES_DIR}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ds in DATASETS:
        print(f"\n{ds.upper()}:")
        ds_df = df[df['Dataset']==ds]
        for _, row in ds_df.iterrows():
            print(f"  {row['Model']}: ROC-AUC={row['ROC-AUC']:.3f}, Acc={row['Accuracy']:.3f}")


if __name__ == '__main__':
    main()
