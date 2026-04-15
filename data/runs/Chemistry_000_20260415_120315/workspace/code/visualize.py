"""
Visualization utilities for KA-GNN experiments.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def plot_dataset_statistics(datasets, output_dir='report/images'):
    """Generate overview plots for datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    from data_utils import load_dataset
    
    stats = []
    for dataset_name in datasets:
        try:
            graphs, df = load_dataset(dataset_name)
            
            # Calculate statistics
            num_molecules = len(graphs)
            num_atoms = [g.atom_features.shape[0] for g in graphs]
            num_edges = [g.edge_index.shape[1] // 2 for g in graphs if g.valid]  # Undirected
            
            # Labels
            labels = [g.label for g in graphs]
            if isinstance(labels[0], list):
                labels = [l[0] for l in labels]  # Use first task for multi-task
            
            stats.append({
                'Dataset': dataset_name.upper(),
                'Molecules': num_molecules,
                'Avg Atoms': np.mean(num_atoms),
                'Avg Bonds': np.mean(num_edges),
                'Positive %': np.mean(labels) * 100
            })
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
    
    stats_df = pd.DataFrame(stats)
    
    # Plot 1: Dataset sizes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Number of molecules
    ax = axes[0, 0]
    ax.bar(stats_df['Dataset'], stats_df['Molecules'], color='steelblue', edgecolor='black')
    ax.set_ylabel('Number of Molecules')
    ax.set_title('Dataset Sizes')
    ax.set_yscale('log')
    for i, v in enumerate(stats_df['Molecules']):
        ax.text(i, v * 1.2, str(int(v)), ha='center', va='bottom', fontsize=10)
    
    # Average atoms per molecule
    ax = axes[0, 1]
    ax.bar(stats_df['Dataset'], stats_df['Avg Atoms'], color='coral', edgecolor='black')
    ax.set_ylabel('Average Number of Atoms')
    ax.set_title('Average Molecule Size (Atoms)')
    for i, v in enumerate(stats_df['Avg Atoms']):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Average bonds per molecule
    ax = axes[1, 0]
    ax.bar(stats_df['Dataset'], stats_df['Avg Bonds'], color='lightgreen', edgecolor='black')
    ax.set_ylabel('Average Number of Bonds')
    ax.set_title('Average Molecule Size (Bonds)')
    for i, v in enumerate(stats_df['Avg Bonds']):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Class distribution
    ax = axes[1, 1]
    ax.bar(stats_df['Dataset'], stats_df['Positive %'], color='mediumpurple', edgecolor='black')
    ax.set_ylabel('Positive Class (%)')
    ax.set_title('Class Imbalance (Positive %)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Balanced')
    ax.legend()
    for i, v in enumerate(stats_df['Positive %']):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dataset statistics to {output_dir}/dataset_statistics.png")
    return stats_df

def plot_training_curves(results_dir='outputs/results', output_dir='report/images'):
    """Plot training curves for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and f != 'all_results.json']
    
    if not result_files:
        print("No result files found")
        return
    
    # Group by dataset
    datasets = set()
    for f in result_files:
        parts = f.replace('.json', '').split('_')
        dataset = parts[-1]
        datasets.add(dataset)
    
    datasets = sorted(datasets)
    
    for dataset in datasets:
        # Get results for this dataset
        kagnn_file = f'kagnn_{dataset}.json'
        gcn_file = f'gcn_{dataset}.json'
        mlp_file = f'mlp_gnn_{dataset}.json'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for model_name, file_name in [('KA-GNN', kagnn_file), ('GCN', gcn_file), ('MLP-GNN', mlp_file)]:
            file_path = os.path.join(results_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'history' in data:
                    history = data['history']
                    epochs = range(1, len(history['train_loss']) + 1)
                    
                    # Plot loss
                    axes[0].plot(epochs, history['train_loss'], label=f'{model_name} (train)', alpha=0.7)
                    axes[0].plot(epochs, history['val_loss'], label=f'{model_name} (val)', 
                               linestyle='--', alpha=0.7)
                    
                    # Plot AUC
                    axes[1].plot(epochs, history['train_auc'], label=f'{model_name} (train)', alpha=0.7)
                    axes[1].plot(epochs, history['val_auc'], label=f'{model_name} (val)', 
                               linestyle='--', alpha=0.7)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{dataset.upper()}: Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ROC-AUC')
        axes[1].set_title(f'{dataset.upper()}: Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_curves_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved training curves to {output_dir}")

def plot_performance_comparison(results_dir='outputs/results', output_dir='report/images'):
    """Plot performance comparison across datasets and models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    all_results_file = os.path.join(results_dir, 'all_results.json')
    if not os.path.exists(all_results_file):
        print("No combined results file found")
        return
    
    with open(all_results_file, 'r') as f:
        all_results = json.load(f)
    
    # Prepare data for plotting
    datasets = []
    models = []
    test_aucs = []
    
    for dataset, model_results in all_results.items():
        for model, results in model_results.items():
            if 'test_auc' in results:
                datasets.append(dataset.upper())
                models.append(model.upper().replace('_', '-'))
                test_aucs.append(results['test_auc'])
    
    df = pd.DataFrame({
        'Dataset': datasets,
        'Model': models,
        'Test AUC': test_aucs
    })
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Dataset', columns='Model', values='Test AUC')
    
    # Plot 1: Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu', 
               cbar_kws={'label': 'Test ROC-AUC'}, ax=ax, vmin=0.5, vmax=1.0)
    ax.set_title('Model Performance Comparison (Test ROC-AUC)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(pivot_df.index))
    width = 0.25
    
    for i, col in enumerate(pivot_df.columns):
        ax.bar(x + i * width, pivot_df[col], width, label=col, alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Test ROC-AUC')
    ax.set_title('Model Performance Comparison Across Datasets')
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot_df.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Summary table as figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = pivot_df.round(3)
    table = ax.table(cellText=table_data.values,
                    rowLabels=table_data.index,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#4472C4']*len(table_data.columns),
                    colWidths=[0.2]*len(table_data.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Test ROC-AUC Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved performance comparisons to {output_dir}")
    return df

def plot_kagnn_architecture(output_dir='report/images'):
    """Create a diagram illustrating KA-GNN architecture."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'KA-GNN Architecture', fontsize=18, fontweight='bold', ha='center')
    
    # Input
    ax.add_patch(plt.Rectangle((0.5, 7.5), 2, 1, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(1.5, 8, 'Molecular Graph\n(SMILES)', ha='center', va='center', fontsize=10)
    
    # Arrow
    ax.annotate('', xy=(3.5, 8), xytext=(2.5, 8),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    # Node/Edge Embedding with KAN
    ax.add_patch(plt.Rectangle((3.5, 7.5), 2, 1, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.text(4.5, 8.3, 'KAN Embedding', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.5, 7.7, 'Fourier Basis', ha='center', va='center', fontsize=9, style='italic')
    
    # Arrow
    ax.annotate('', xy=(6.5, 8), xytext=(5.5, 8),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    # Message Passing with KAN
    ax.add_patch(plt.Rectangle((6.5, 7.5), 2, 1, facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax.text(7.5, 8.3, 'KAN Message Passing', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.5, 7.7, 'φ(x) = Σ [a_k·cos(kωx) + b_k·sin(kωx)]', ha='center', va='center', fontsize=8)
    
    # Stacked layers indication
    for i in range(3):
        y_pos = 6.2 - i * 1.2
        ax.add_patch(plt.Rectangle((2.5, y_pos), 5, 1, facecolor='lightgreen', 
                                  edgecolor='black', linewidth=1.5, alpha=0.7))
        ax.text(5, y_pos + 0.5, f'Message Passing Layer {i+1}', ha='center', va='center', fontsize=10)
    
    # Arrow down
    ax.annotate('', xy=(5, 6.2), xytext=(5, 7.5),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    # Readout with KAN
    ax.add_patch(plt.Rectangle((2.5, 2), 5, 1, facecolor='plum', edgecolor='black', linewidth=2))
    ax.text(5, 2.5, 'KAN Readout (Pooling + Classification)', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(5, 2), xytext=(5, 2.9),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    # Output
    ax.add_patch(plt.Rectangle((3, 0.5), 4, 1, facecolor='gold', edgecolor='black', linewidth=2))
    ax.text(5, 1, 'Property Prediction\n(Toxicity / Bioactivity)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Key innovation box
    ax.add_patch(plt.Rectangle((7.5, 3), 2.2, 2.5, facecolor='wheat', edgecolor='darkred', linewidth=2))
    ax.text(8.6, 5.2, 'Key Innovation:', ha='center', va='center', fontsize=10, fontweight='bold', color='darkred')
    ax.text(8.6, 4.5, 'Fourier-based KAN', ha='center', va='center', fontsize=9)
    ax.text(8.6, 4.0, 'replaces MLP', ha='center', va='center', fontsize=9)
    ax.text(8.6, 3.5, 'for stronger', ha='center', va='center', fontsize=9)
    ax.text(8.6, 3.0, 'expressiveness', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kagnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved KA-GNN architecture diagram to {output_dir}/kagnn_architecture.png")

def plot_fourier_basis(output_dir='report/images'):
    """Illustrate Fourier basis functions used in KAN."""
    os.makedirs(output_dir, exist_ok=True)
    
    x = np.linspace(-2, 2, 500)
    omega = 1.0
    num_freqs = 4
    
    fig, axes = plt.subplots(2, num_freqs, figsize=(14, 6))
    
    for k in range(1, num_freqs + 1):
        # Cosine basis
        cos_vals = np.cos(k * omega * x)
        axes[0, k-1].plot(x, cos_vals, 'b-', linewidth=2)
        axes[0, k-1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, k-1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        axes[0, k-1].set_title(f'cos({k}ωx)', fontsize=12)
        axes[0, k-1].set_ylim(-1.5, 1.5)
        axes[0, k-1].grid(True, alpha=0.3)
        if k == 1:
            axes[0, k-1].set_ylabel('Cosine Basis', fontsize=12)
        
        # Sine basis
        sin_vals = np.sin(k * omega * x)
        axes[1, k-1].plot(x, sin_vals, 'r-', linewidth=2)
        axes[1, k-1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, k-1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        axes[1, k-1].set_title(f'sin({k}ωx)', fontsize=12)
        axes[1, k-1].set_ylim(-1.5, 1.5)
        axes[1, k-1].grid(True, alpha=0.3)
        axes[1, k-1].set_xlabel('x', fontsize=12)
        if k == 1:
            axes[1, k-1].set_ylabel('Sine Basis', fontsize=12)
    
    plt.suptitle('Fourier Basis Functions in Kolmogorov-Arnold Networks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fourier_basis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Fourier basis diagram to {output_dir}/fourier_basis.png")

def generate_all_visualizations():
    """Generate all visualizations for the report."""
    print("Generating visualizations...")
    
    datasets = ['bace', 'bbbp', 'clintox', 'hiv', 'muv']
    
    # Dataset statistics
    print("\n1. Dataset statistics...")
    stats_df = plot_dataset_statistics(datasets)
    print(stats_df.to_string(index=False))
    
    # Architecture diagram
    print("\n2. KA-GNN architecture diagram...")
    plot_kagnn_architecture()
    
    # Fourier basis
    print("\n3. Fourier basis illustration...")
    plot_fourier_basis()
    
    # Training curves and performance (only if results exist)
    if os.path.exists('outputs/results/all_results.json'):
        print("\n4. Training curves...")
        plot_training_curves()
        
        print("\n5. Performance comparison...")
        plot_performance_comparison()
    else:
        print("\n4-5. Skipping result visualizations (training not yet run)")
    
    print("\nAll visualizations generated!")

if __name__ == '__main__':
    generate_all_visualizations()
