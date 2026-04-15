"""
Visualization and analysis script for the diffusion model results.
Generates all figures needed for the research report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from scipy.spatial.distance import cdist


def load_data():
    """Load training history and evaluation results."""
    history = np.load('outputs/training_history.npz')
    
    with open('outputs/evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    structures = np.load('outputs/predicted_structures.npz')
    
    with open('outputs/data_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return history, eval_results, structures, metadata


def figure_training_curves(history, save_path='report/images/fig01_training_curves.png'):
    """Figure 1: Training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = history['epoch']
    
    # Total loss
    axes[0].plot(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Protein vs Ligand loss
    axes[1].plot(epochs, history['protein_loss'], 'r-', linewidth=2, label='Protein CA Loss')
    axes[1].plot(epochs, history['ligand_loss'], 'g-', linewidth=2, label='Ligand Loss')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss (MSE)', fontsize=12)
    axes[1].set_title('Component-wise Loss', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[2].plot(epochs, history['lr'], 'purple', linewidth=2, label='Learning Rate')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Cosine Annealing LR Schedule', fontsize=14)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def figure_rmsd_distribution(eval_results, save_path='report/images/fig02_rmsd_distribution.png'):
    """Figure 2: RMSD distribution across samples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_samples = len(eval_results['protein_rmsds'])
    sample_indices = np.arange(1, n_samples + 1)
    
    # Bar plot of RMSDs
    colors_p = plt.cm.Reds(np.linspace(0.3, 0.8, n_samples))
    colors_l = plt.cm.Blues(np.linspace(0.3, 0.8, n_samples))
    
    bars_p = axes[0].bar(sample_indices, eval_results['protein_rmsds'], 
                         color=colors_p, edgecolor='darkred', linewidth=1.5)
    axes[0].axhline(y=eval_results['mean_protein_rmsd'], color='red', 
                    linestyle='--', linewidth=2, label=f"Mean: {eval_results['mean_protein_rmsd']:.3f} Å")
    axes[0].set_xlabel('Sample Index', fontsize=12)
    axes[0].set_ylabel('CA-RMSD (Å)', fontsize=12)
    axes[0].set_title('Protein CA-RMSD per Sample', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    bars_l = axes[1].bar(sample_indices, eval_results['ligand_rmsds'],
                         color=colors_l, edgecolor='darkblue', linewidth=1.5)
    axes[1].axhline(y=eval_results['mean_ligand_rmsd'], color='blue',
                    linestyle='--', linewidth=2, label=f"Mean: {eval_results['mean_ligand_rmsd']:.3f} Å")
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('Heavy-Atom RMSD (Å)', fontsize=12)
    axes[1].set_title('Ligand Heavy-Atom RMSD per Sample', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def figure_structure_overlay(structures, save_path='report/images/fig03_structure_overlay.png'):
    """Figure 3: 2D projection of predicted vs true protein structure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    true_protein = structures['true_protein_coords'][0]  # (N_res, 3)
    best_sample_idx = 0  # Use first sample as representative
    
    pred_protein = structures['pred_protein_coords'][best_sample_idx]  # (N_res, 3)
    true_ligand = structures['true_ligand_coords'][0]
    pred_ligand = structures['pred_ligand_coords'][best_sample_idx]
    
    # Project to XY plane for protein
    ax = axes[0]
    sc_true = ax.scatter(true_protein[:, 0], true_protein[:, 1], 
                         c=np.arange(len(true_protein)), cmap='viridis', 
                         s=30, alpha=0.8, label='True (NMR)', edgecolors='none')
    sc_pred = ax.scatter(pred_protein[:, 0], pred_protein[:, 1], 
                         c=np.arange(len(pred_protein)), cmap='plasma',
                         s=20, alpha=0.6, label='Predicted (Diffusion)', 
                         marker='x', linewidths=1.5)
    
    # Draw backbone connections for true structure
    ax.plot(true_protein[:, 0], true_protein[:, 1], 'k-', alpha=0.3, linewidth=0.5)
    ax.plot(pred_protein[:, 0], pred_protein[:, 1], 'r--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_title(f'Protein CA Structure Overlay\n(FKBP12, 161 residues)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Ligand overlay
    ax = axes[1]
    heavy_mask_true = np.array([t not in ['H'] for t in ['H']*194])  # placeholder
    # Use all atoms for visualization
    ax.scatter(true_ligand[:, 0], true_ligand[:, 1], 
               c='green', s=40, alpha=0.7, label='True (FK506)', edgecolors='darkgreen')
    ax.scatter(pred_ligand[:, 0], pred_ligand[:, 1], 
               c='orange', s=30, alpha=0.5, label='Predicted', 
               marker='o', edgecolors='darkorange')
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_title(f'Ligand Structure Overlay\n(FK506, 194 atoms)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def figure_distance_comparison(structures, save_path='report/images/fig04_distance_comparison.png'):
    """Figure 4: Pairwise distance matrix comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    true_protein = structures['true_protein_coords'][0]
    pred_protein = structures['pred_protein_coords'][0]
    true_ligand = structures['true_ligand_coords'][0]
    pred_ligand = structures['pred_ligand_coords'][0]
    
    # Subsample for visualization (take every 5th residue)
    step = 5
    tp_sub = true_protein[::step]
    pp_sub = pred_protein[::step]
    
    # Distance matrices
    dist_true_p = cdist(tp_sub, tp_sub)
    dist_pred_p = cdist(pp_sub, pp_sub)
    diff_p = np.abs(dist_true_p - dist_pred_p)
    
    tl_sub = true_ligand[::3]
    pl_sub = pred_ligand[::3]
    
    dist_true_l = cdist(tl_sub, tl_sub)
    dist_pred_l = cdist(pl_sub, pl_sub)
    diff_l = np.abs(dist_true_l - dist_pred_l)
    
    # Protein true distances
    im0 = axes[0, 0].imshow(dist_true_p, cmap='magma', aspect='auto')
    axes[0, 0].set_title('True Protein CA Distances', fontsize=12)
    axes[0, 0].set_xlabel('Residue Index (subsampled)')
    axes[0, 0].set_ylabel('Residue Index (subsampled)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Protein predicted distances
    im1 = axes[0, 1].imshow(dist_pred_p, cmap='magma', aspect='auto')
    axes[0, 1].set_title('Predicted Protein CA Distances', fontsize=12)
    axes[0, 1].set_xlabel('Residue Index (subsampled)')
    axes[0, 1].set_ylabel('Residue Index (subsampled)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Protein distance difference
    im2 = axes[1, 0].imshow(diff_p, cmap='coolwarm', aspect='auto', vmin=0, vmax=5)
    axes[1, 0].set_title('Protein Distance Error', fontsize=12)
    axes[1, 0].set_xlabel('Residue Index (subsampled)')
    axes[1, 0].set_ylabel('Residue Index (subsampled)')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Ligand distance difference
    im3 = axes[1, 1].imshow(diff_l, cmap='coolwarm', aspect='auto', vmin=0, vmax=5)
    axes[1, 1].set_title('Ligand Distance Error', fontsize=12)
    axes[1, 1].set_xlabel('Atom Index (subsampled)')
    axes[1, 1].set_ylabel('Atom Index (subsampled)')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def figure_3d_projection(structures, save_path='report/images/fig05_3d_projection.png'):
    """Figure 5: 3D-like projections (XZ and YZ views)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    true_protein = structures['true_protein_coords'][0]
    pred_protein = structures['pred_protein_coords'][0]
    true_ligand = structures['true_ligand_coords'][0]
    pred_ligand = structures['pred_ligand_coords'][0]
    
    # XZ view - protein
    ax = axes[0, 0]
    ax.plot(true_protein[:, 0], true_protein[:, 2], 'b-', alpha=0.5, linewidth=1.5, label='True')
    ax.plot(pred_protein[:, 0], pred_protein[:, 2], 'r--', alpha=0.5, linewidth=1.5, label='Predicted')
    ax.scatter(true_protein[:, 0], true_protein[:, 2], c='blue', s=15, alpha=0.7)
    ax.scatter(pred_protein[:, 0], pred_protein[:, 2], c='red', s=10, alpha=0.5, marker='x')
    ax.set_xlabel('X (Å)', fontsize=11)
    ax.set_ylabel('Z (Å)', fontsize=11)
    ax.set_title('Protein XZ Projection', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # YZ view - protein
    ax = axes[0, 1]
    ax.plot(true_protein[:, 1], true_protein[:, 2], 'b-', alpha=0.5, linewidth=1.5, label='True')
    ax.plot(pred_protein[:, 1], pred_protein[:, 2], 'r--', alpha=0.5, linewidth=1.5, label='Predicted')
    ax.scatter(true_protein[:, 1], true_protein[:, 2], c='blue', s=15, alpha=0.7)
    ax.scatter(pred_protein[:, 1], pred_protein[:, 2], c='red', s=10, alpha=0.5, marker='x')
    ax.set_xlabel('Y (Å)', fontsize=11)
    ax.set_ylabel('Z (Å)', fontsize=11)
    ax.set_title('Protein YZ Projection', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # XZ view - ligand
    ax = axes[1, 0]
    ax.scatter(true_ligand[:, 0], true_ligand[:, 2], c='green', s=25, alpha=0.7, label='True')
    ax.scatter(pred_ligand[:, 0], pred_ligand[:, 2], c='orange', s=20, alpha=0.5, label='Predicted')
    ax.set_xlabel('X (Å)', fontsize=11)
    ax.set_ylabel('Z (Å)', fontsize=11)
    ax.set_title('Ligand XZ Projection', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # YZ view - ligand
    ax = axes[1, 1]
    ax.scatter(true_ligand[:, 1], true_ligand[:, 2], c='green', s=25, alpha=0.7, label='True')
    ax.scatter(pred_ligand[:, 1], pred_ligand[:, 2], c='orange', s=20, alpha=0.5, label='Predicted')
    ax.set_xlabel('Y (Å)', fontsize=11)
    ax.set_ylabel('Z (Å)', fontsize=11)
    ax.set_title('Ligand YZ Projection', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def figure_architecture_diagram(save_path='report/images/fig06_architecture.png'):
    """Figure 6: Model architecture schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    protein_color = '#E8F5E9'
    ligand_color = '#E3F2FD'
    cross_color = '#FFF3E0'
    diffusion_color = '#FCE4EC'
    output_color = '#F3E5F5'
    
    # Input boxes
    ax.add_patch(plt.Rectangle((0.5, 6), 3, 2.5, facecolor=protein_color, edgecolor='#4CAF50', linewidth=2))
    ax.text(2, 7.5, 'Protein Input\n\n• Amino Acid Sequence\n• One-hot Encoding\n• Positional Encoding',
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((0.5, 2), 3, 2.5, facecolor=ligand_color, edgecolor='#2196F3', linewidth=2))
    ax.text(2, 3.5, 'Ligand Input\n\n• Atom Types\n• Coordinates\n• Adjacency Matrix',
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Encoder boxes
    ax.add_patch(plt.Rectangle((5, 6.5), 3, 2, facecolor=protein_color, edgecolor='#388E3C', linewidth=2))
    ax.text(6.5, 7.5, 'Protein Encoder\n\nTransformer\n(4 layers, 8 heads)',
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((5, 2), 3, 2.5, facecolor=ligand_color, edgecolor='#1976D2', linewidth=2))
    ax.text(6.5, 3.5, 'Ligand Encoder\n\nMessage Passing GNN\n(4 layers)',
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Cross attention
    ax.add_patch(plt.Rectangle((9.5, 4.5), 3, 3, facecolor=cross_color, edgecolor='#FF9800', linewidth=2))
    ax.text(11, 6.5, 'Cross-Attention\n\nProtein ↔ Ligand\nMulti-head Attention',
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Denoising blocks
    ax.add_patch(plt.Rectangle((13.5, 4.5), 2.5, 3, facecolor=diffusion_color, edgecolor='#E91E63', linewidth=2))
    ax.text(14.75, 6.5, 'Denoising\nBlocks × 3\n\nCoord Update +\nTimestep Embedding',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Output
    ax.add_patch(plt.Rectangle((13.5, 0.5), 2.5, 2.5, facecolor=output_color, edgecolor='#9C27B0', linewidth=2))
    ax.text(14.75, 2, 'Output\n\nPredicted 3D\nCoordinates\n(Protein + Ligand)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='#333333')
    ax.annotate('', xy=(5, 7.5), xytext=(3.5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 3.5), xytext=(3.5, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(9.5, 6.5), xytext=(8, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(9.5, 5.5), xytext=(8, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(13.5, 6.5), xytext=(12.5, 6.5), arrowprops=arrow_props)
    ax.annotate('', xy=(14.75, 3), xytext=(14.75, 4.5), arrowprops=arrow_props)
    
    # Diffusion process annotation
    ax.add_patch(plt.Rectangle((5, 0.5), 4, 1.5, facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2))
    ax.text(7, 1.25, 'Diffusion Process\n\nForward: Add Gaussian Noise\nReverse: Learn Denoising (x₀ prediction)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.annotate('', xy=(5, 2), xytext=(7, 2), arrowprops=dict(arrowstyle='->', lw=1.5, color='#F57F17', linestyle='--'))
    
    ax.set_title('Unified Diffusion-Based Biomolecular Complex Structure Prediction Architecture',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Generating figures...")
    
    history, eval_results, structures, metadata = load_data()
    
    figure_training_curves(history)
    figure_rmsd_distribution(eval_results)
    figure_structure_overlay(structures)
    figure_distance_comparison(structures)
    figure_3d_projection(structures)
    figure_architecture_diagram()
    
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
