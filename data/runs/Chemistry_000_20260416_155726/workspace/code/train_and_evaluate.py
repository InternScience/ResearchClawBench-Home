"""
Training and Evaluation Script for KA-GNNs on Molecular Property Prediction

This script:
1. Loads and preprocesses all datasets (BACE, BBBP, ClinTox, HIV, MUV)
2. Trains baseline models (GCN, GAT) and KA-GNN
3. Evaluates performance using ROC-AUC, PR-AUC, accuracy, F1
4. Saves results and generates visualizations
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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import our models
from ka_gnn import (
    mol_to_graph, GCN, GAT, KAGNN, 
    train_epoch, evaluate, FourierKANLayer
)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

HIDDEN_FEATURES = 64
NUM_LAYERS = 3
NUM_FOURIER_TERMS = 8
DROPOUT = 0.3
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

DATASETS = ['bace', 'bbbp', 'clintox', 'hiv', 'muv']
DATA_DIR = '../data'
OUTPUT_DIR = '../outputs'
IMAGES_DIR = '../report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_dataset(name):
    """Load a dataset and convert to graph representations."""
    filepath = os.path.join(DATA_DIR, f'{name}.csv')
    df = pd.read_csv(filepath)
    
    graphs = []
    labels = []
    
    # Determine label column(s)
    if name == 'clintox':
        # ClinTox has two tasks: FDA_APPROVED and CT_TOX
        label_cols = ['FDA_APPROVED', 'CT_TOX']
        for idx, row in df.iterrows():
            smiles = row['smiles']
            graph = mol_to_graph(smiles)
            if graph is not None:
                for col in label_cols:
                    label = row[col]
                    if not pd.isna(label):
                        g = graph.clone()
                        g.y = torch.tensor([float(label)], dtype=torch.float)
                        g.task = col
                        graphs.append(g)
                        labels.append(float(label))
    elif name == 'muv':
        # MUV has multiple tasks
        task_cols = [c for c in df.columns if c.startswith('MUV-')]
        mol_col = 'mol_id'
        smiles_col = 'smiles'
        
        for idx, row in df.iterrows():
            smiles = row[smiles_col]
            graph = mol_to_graph(smiles)
            if graph is not None:
                for col in task_cols:
                    label = row[col]
                    if not pd.isna(label):
                        g = graph.clone()
                        g.y = torch.tensor([float(label)], dtype=torch.float)
                        g.task = col
                        graphs.append(g)
                        labels.append(float(label))
    else:
        # Single task datasets
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
    """Create train/val/test loaders with 80/10/10 split."""
    indices = list(range(len(graphs)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=[g.y.item() for g in graphs])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=[graphs[i].y.item() for i in temp_idx])
    
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
    
    # ROC-AUC (only if both classes present)
    if len(np.unique(labels)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(labels, scores)
        except:
            metrics['roc_auc'] = 0.5
    else:
        metrics['roc_auc'] = 0.5
    
    # PR-AUC
    try:
        metrics['pr_auc'] = average_precision_score(labels, scores)
    except:
        metrics['pr_auc'] = 0.5
    
    # Accuracy, F1, Balanced Accuracy
    predictions = (scores > 0.5).astype(float)
    metrics['accuracy'] = (predictions == labels).mean()
    
    if len(np.unique(labels)) > 1:
        metrics['f1'] = f1_score(labels, predictions, zero_division=0)
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels, predictions)
    else:
        metrics['f1'] = 0.0
        metrics['balanced_accuracy'] = 0.5
    
    return metrics


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, use_edge_attr=False):
    """Train model and return training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, 
                                           use_edge_attr=use_edge_attr)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_results = evaluate(model, val_loader, criterion, DEVICE, use_edge_attr=use_edge_attr)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }


def run_experiment(dataset_name, model_class, model_name, **model_kwargs):
    """Run a complete experiment on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()} | Model: {model_name}")
    print(f"{'='*60}")
    
    # Load data
    graphs, labels = load_dataset(dataset_name)
    if len(graphs) == 0:
        print(f"  No valid graphs for {dataset_name}")
        return None
    
    # Create loaders
    train_loader, val_loader, test_loader = create_data_loaders(graphs)
    
    # Get input dimensions
    sample_graph = graphs[0]
    in_features = sample_graph.x.shape[1]
    edge_features = sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') else 0
    
    # Initialize model
    # Check if model class expects edge_features parameter
    import inspect
    sig = inspect.signature(model_class.__init__)
    params = sig.parameters
    
    if 'edge_features' in params:
        model = model_class(in_features=in_features, edge_features=edge_features, **model_kwargs)
        actual_use_edge_attr = True
    else:
        model = model_class(in_features=in_features, **model_kwargs)
        actual_use_edge_attr = False
    
    model = model.to(DEVICE)
    
    # Train
    model.use_edge_attr = actual_use_edge_attr
    
    history = train_model(model, train_loader, val_loader, use_edge_attr=actual_use_edge_attr)
    
    # Evaluate on test set
    criterion = nn.BCEWithLogitsLoss()
    test_results = evaluate(model, test_loader, criterion, DEVICE, use_edge_attr=actual_use_edge_attr)
    test_metrics = compute_metrics(test_results['scores'], test_results['labels'])
    
    print(f"  Test Metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")
    
    return {
        'dataset': dataset_name,
        'model': model_name,
        'test_metrics': test_metrics,
        'history': history,
        'test_scores': test_results['scores'].tolist(),
        'test_labels': test_results['labels'].tolist()
    }


def main():
    """Main experiment runner."""
    all_results = []
    
    # Models to compare
    models = [
        ('GCN', GCN, {'hidden_features': HIDDEN_FEATURES, 'num_layers': NUM_LAYERS, 'dropout': DROPOUT}),
        ('GAT', GAT, {'hidden_features': HIDDEN_FEATURES, 'num_layers': NUM_LAYERS, 'num_heads': 4, 'dropout': DROPOUT}),
        ('KA-GNN', KAGNN, {'hidden_features': HIDDEN_FEATURES, 'num_layers': NUM_LAYERS, 
                           'num_fourier_terms': NUM_FOURIER_TERMS, 'dropout': DROPOUT}),
    ]
    
    # Run experiments
    for dataset in DATASETS:
        for model_name, model_class, model_kwargs in models:
            result = run_experiment(dataset, model_class, model_name, **model_kwargs)
            if result is not None:
                all_results.append(result)
    
    # Save results
    results_file = os.path.join(OUTPUT_DIR, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate summary table
    generate_summary_table(all_results)
    
    # Generate visualizations
    generate_visualizations(all_results)
    
    return all_results


def generate_summary_table(results):
    """Generate a summary table of results."""
    rows = []
    for r in results:
        row = {
            'Dataset': r['dataset'],
            'Model': r['model'],
            'ROC-AUC': r['test_metrics']['roc_auc'],
            'PR-AUC': r['test_metrics']['pr_auc'],
            'Accuracy': r['test_metrics']['accuracy'],
            'F1': r['test_metrics']['f1'],
            'Balanced Acc': r['test_metrics']['balanced_accuracy']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_file = os.path.join(OUTPUT_DIR, 'results_summary.csv')
    df.to_csv(csv_file, index=False)
    print(f"Summary table saved to {csv_file}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Group by dataset
    for dataset in DATASETS:
        dataset_df = df[df['Dataset'] == dataset]
        print(f"\n{dataset.upper()}:")
        print(dataset_df[['Model', 'ROC-AUC', 'PR-AUC', 'Accuracy']].to_string(index=False))
    
    return df


def generate_visualizations(results):
    """Generate visualization plots."""
    
    # 1. ROC-AUC comparison bar plot
    plt.figure(figsize=(12, 8))
    
    df = pd.DataFrame([{
        'Dataset': r['dataset'],
        'Model': r['model'],
        'ROC-AUC': r['test_metrics']['roc_auc']
    } for r in results])
    
    # Pivot for grouped bar plot
    pivot_df = df.pivot(index='Dataset', columns='Model', values='ROC-AUC')
    
    # Plot
    ax = pivot_df.plot(kind='bar', figsize=(12, 8), width=0.8)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('ROC-AUC', fontsize=12)
    plt.title('Model Comparison: ROC-AUC Across Datasets', fontsize=14)
    plt.legend(title='Model', loc='upper right')
    plt.xticks(rotation=0)
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'roc_auc_comparison.png'), dpi=150)
    plt.close()
    print("Saved: roc_auc_comparison.png")
    
    # 2. Learning curves for first dataset/model combination
    if len(results) > 0:
        sample_result = results[0]
        history = sample_result['history']
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Train Loss', alpha=0.8)
        plt.plot(history['val_losses'], label='Val Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Training Curves ({sample_result['dataset']}, {sample_result['model']})")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accs'], label='Train Acc', alpha=0.8)
        plt.plot(history['val_accs'], label='Val Acc', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f"Accuracy Curves ({sample_result['dataset']}, {sample_result['model']})")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'learning_curves.png'), dpi=150)
        plt.close()
        print("Saved: learning_curves.png")
    
    # 3. Heatmap of ROC-AUC scores
    plt.figure(figsize=(10, 6))
    pivot_df = df.pivot(index='Dataset', columns='Model', values='ROC-AUC')
    
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                center=0.75, vmin=0.5, vmax=1.0)
    plt.title('ROC-AUC Heatmap: Model Performance Across Datasets', fontsize=12)
    plt.xlabel('Model', fontsize=11)
    plt.ylabel('Dataset', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'performance_heatmap.png'), dpi=150)
    plt.close()
    print("Saved: performance_heatmap.png")
    
    # 4. Data overview - class distribution
    plt.figure(figsize=(15, 4))
    
    for i, dataset in enumerate(DATASETS):
        dataset_results = [r for r in results if r['dataset'] == dataset]
        if len(dataset_results) > 0:
            labels = np.array(dataset_results[0]['test_labels'])
            
            plt.subplot(1, len(DATASETS), i+1)
            plt.bar(['Negative', 'Positive'], 
                    [(labels == 0).sum(), (labels == 1).sum()],
                    color=['#3498db', '#e74c3c'])
            plt.title(f'{dataset.upper()}\nTest Set', fontsize=10)
            plt.ylabel('Count')
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
    
    plt.suptitle('Class Distribution Across Datasets (Test Sets)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'class_distribution.png'), dpi=150)
    plt.close()
    print("Saved: class_distribution.png")
    
    # 5. Multi-metric radar chart for best performing dataset
    if len(results) >= 3:
        # Pick one dataset for detailed comparison
        sample_dataset = results[0]['dataset']
        dataset_results = [r for r in results if r['dataset'] == sample_dataset]
        
        metrics_to_plot = ['ROC-AUC', 'PR-AUC', 'Accuracy', 'F1', 'Balanced Acc']
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        colors = {'GCN': '#3498db', 'GAT': '#2ecc71', 'KA-GNN': '#e74c3c'}
        
        for result in dataset_results:
            model = result['model']
            values = [
                result['test_metrics']['roc_auc'],
                result['test_metrics']['pr_auc'],
                result['test_metrics']['accuracy'],
                result['test_metrics']['f1'],
                result['test_metrics']['balanced_accuracy']
            ]
            values += values[:1]  # Complete the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors.get(model, 'gray'))
            ax.fill(angles, values, alpha=0.15, color=colors.get(model, 'gray'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_plot, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'Multi-Metric Comparison ({sample_dataset.upper()})', fontsize=12, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'radar_comparison.png'), dpi=150)
        plt.close()
        print("Saved: radar_comparison.png")
    
    print("\nAll visualizations saved!")


if __name__ == '__main__':
    results = main()
