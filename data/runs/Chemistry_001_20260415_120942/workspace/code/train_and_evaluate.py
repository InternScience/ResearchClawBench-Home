"""
Training and evaluation pipeline for the unified biomolecular complex structure prediction framework.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import torch.nn.functional as F

from data_loader import ProteinStructure, LigandStructure, BiomolecularComplex
from graph_encoder import HeterogeneousGraphEncoder
from diffusion_model import DiffusionModel, compute_rmsd


class BiomolecularComplexDataset(Dataset):
    """Dataset for biomolecular complexes."""
    
    def __init__(self, protein_path, ligand_path, num_augmentations=10):
        self.protein_path = protein_path
        self.ligand_path = ligand_path
        self.complex = BiomolecularComplex(protein_path, ligand_path)
        self.num_augmentations = num_augmentations
        
        # Extract features
        self.protein_features = torch.FloatTensor(self.complex.protein.get_residue_features())
        self.ligand_features = torch.FloatTensor(self.complex.ligand.get_atom_features())
        
        # True coordinates
        self.protein_coords_true = torch.FloatTensor(self.complex.protein_coords)
        self.ligand_coords_true = torch.FloatTensor(self.complex.ligand_coords)
        
        # Center coordinates
        self.center = torch.cat([self.protein_coords_true, self.ligand_coords_true], dim=0).mean(dim=0)
        self.protein_coords_true = self.protein_coords_true - self.center
        self.ligand_coords_true = self.ligand_coords_true - self.center
        
        # Pad or truncate ligand features to fixed size
        self.max_ligand_atoms = 100
        if self.ligand_features.size(0) < self.max_ligand_atoms:
            padding = torch.zeros(self.max_ligand_atoms - self.ligand_features.size(0), 
                                 self.ligand_features.size(1))
            self.ligand_features = torch.cat([self.ligand_features, padding], dim=0)
            
            padding_coords = torch.zeros(self.max_ligand_atoms - self.ligand_coords_true.size(0), 3)
            self.ligand_coords_true = torch.cat([self.ligand_coords_true, padding_coords], dim=0)
        else:
            self.ligand_features = self.ligand_features[:self.max_ligand_atoms]
            self.ligand_coords_true = self.ligand_coords_true[:self.max_ligand_atoms]
        
    def __len__(self):
        return self.num_augmentations
    
    def __getitem__(self, idx):
        # Add small noise for data augmentation
        if idx > 0:
            noise_scale = 0.1 * (idx / self.num_augmentations)
            protein_coords = self.protein_coords_true + torch.randn_like(self.protein_coords_true) * noise_scale
            ligand_coords = self.ligand_coords_true + torch.randn_like(self.ligand_coords_true) * noise_scale
        else:
            protein_coords = self.protein_coords_true
            ligand_coords = self.ligand_coords_true
        
        return {
            'protein_features': self.protein_features,
            'protein_coords': protein_coords,
            'ligand_features': self.ligand_features,
            'ligand_coords': ligand_coords,
            'protein_coords_true': self.protein_coords_true,
            'ligand_coords_true': self.ligand_coords_true
        }


def train_diffusion_model(dataset, num_epochs=100, batch_size=1, lr=1e-4, device='cpu'):
    """Train the diffusion model."""
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get dimensions from data
    sample = dataset[0]
    protein_nodes = sample['protein_features'].size(0)
    ligand_nodes = sample['ligand_features'].size(0)
    
    # Initialize model
    model = DiffusionModel(
        protein_nodes=protein_nodes,
        ligand_nodes=ligand_nodes,
        hidden_dim=128,
        num_layers=4,
        timesteps=500
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            protein_features = batch['protein_features'].to(device)
            protein_coords_true = batch['protein_coords_true'].to(device)
            ligand_features = batch['ligand_features'].to(device)
            ligand_coords_true = batch['ligand_coords_true'].to(device)
            
            # Sample random timestep
            t = torch.randint(0, model.timesteps, (batch_size,), device=device)
            
            # Add noise
            protein_noise = torch.randn_like(protein_coords_true)
            ligand_noise = torch.randn_like(ligand_coords_true)
            
            protein_coords_noisy = model.add_noise(protein_coords_true, t, protein_noise)
            ligand_coords_noisy = model.add_noise(ligand_coords_true, t, ligand_noise)
            
            # Predict noise
            protein_noise_pred, ligand_noise_pred = model(
                protein_coords_noisy, ligand_coords_noisy,
                protein_features, ligand_features, t
            )
            
            # Compute loss
            loss_protein = F.mse_loss(protein_noise_pred, protein_noise)
            loss_ligand = F.mse_loss(ligand_noise_pred, ligand_noise)
            loss = loss_protein + loss_ligand
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return model, losses


def evaluate_model(model, dataset, device='cpu', num_samples=5):
    """Evaluate the model and generate predictions."""
    
    model.eval()
    sample = dataset[0]
    
    protein_features = sample['protein_features'].unsqueeze(0).to(device)
    ligand_features = sample['ligand_features'].unsqueeze(0).to(device)
    protein_coords_true = sample['protein_coords_true'].unsqueeze(0).to(device)
    ligand_coords_true = sample['ligand_coords_true'].unsqueeze(0).to(device)
    
    results = {
        'protein_rmsd': [],
        'ligand_rmsd': [],
        'protein_predictions': [],
        'ligand_predictions': []
    }
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample structure
            protein_pred, ligand_pred = model.sample(
                protein_features, ligand_features, device=device
            )
            
            # Compute RMSD
            protein_rmsd = compute_rmsd(protein_pred, protein_coords_true, align=True)
            ligand_rmsd = compute_rmsd(ligand_pred, ligand_coords_true, align=True)
            
            results['protein_rmsd'].append(protein_rmsd.item())
            results['ligand_rmsd'].append(ligand_rmsd.item())
            results['protein_predictions'].append(protein_pred.cpu().numpy())
            results['ligand_predictions'].append(ligand_pred.cpu().numpy())
    
    return results


def visualize_structures(dataset, results, output_dir='report/images'):
    """Generate visualizations of structures."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get true structure
    sample = dataset[0]
    protein_true = sample['protein_coords_true'].numpy()
    ligand_true = sample['ligand_coords_true'].numpy()
    
    # 1. Overlay visualization
    fig = plt.figure(figsize=(15, 5))
    
    # True structure
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(protein_true[:, 0], protein_true[:, 1], protein_true[:, 2], 
                c='blue', alpha=0.6, s=20, label='Protein')
    ax1.scatter(ligand_true[:, 0], ligand_true[:, 1], ligand_true[:, 2], 
                c='red', alpha=0.8, s=30, label='Ligand')
    ax1.set_title('True Structure')
    ax1.legend()
    
    # Predicted structure (best sample)
    best_idx = np.argmin(results['ligand_rmsd'])
    protein_pred = results['protein_predictions'][best_idx][0]
    ligand_pred = results['ligand_predictions'][best_idx][0]
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(protein_pred[:, 0], protein_pred[:, 1], protein_pred[:, 2], 
                c='blue', alpha=0.6, s=20, label='Protein')
    ax2.scatter(ligand_pred[:, 0], ligand_pred[:, 1], ligand_pred[:, 2], 
                c='red', alpha=0.8, s=30, label='Ligand')
    ax2.set_title('Predicted Structure (Best)')
    ax2.legend()
    
    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(protein_true[:, 0], protein_true[:, 1], protein_true[:, 2], 
                c='blue', alpha=0.4, s=20, label='Protein (True)')
    ax3.scatter(protein_pred[:, 0], protein_pred[:, 1], protein_pred[:, 2], 
                c='cyan', alpha=0.4, s=20, label='Protein (Pred)')
    ax3.scatter(ligand_true[:, 0], ligand_true[:, 1], ligand_true[:, 2], 
                c='red', alpha=0.6, s=30, label='Ligand (True)')
    ax3.scatter(ligand_pred[:, 0], ligand_pred[:, 1], ligand_pred[:, 2], 
                c='orange', alpha=0.6, s=30, label='Ligand (Pred)')
    ax3.set_title('Overlay Comparison')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/structure_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. RMSD distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].bar(range(len(results['protein_rmsd'])), results['protein_rmsd'])
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('RMSD (Å)')
    axes[0].set_title('Protein RMSD Distribution')
    axes[0].axhline(y=np.mean(results['protein_rmsd']), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(results["protein_rmsd"]):.2f} Å')
    axes[0].legend()
    
    axes[1].bar(range(len(results['ligand_rmsd'])), results['ligand_rmsd'])
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('RMSD (Å)')
    axes[1].set_title('Ligand RMSD Distribution')
    axes[1].axhline(y=np.mean(results['ligand_rmsd']), color='r', linestyle='--',
                    label=f'Mean: {np.mean(results["ligand_rmsd"]):.2f} Å')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rmsd_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def main():
    """Main execution function."""
    
    # Paths
    protein_path = "data/sample/2l3r/2l3r_protein.pdb"
    ligand_path = "data/sample/2l3r/2l3r_ligand.sdf"
    
    # Create dataset
    print("Loading data...")
    dataset = BiomolecularComplexDataset(protein_path, ligand_path, num_augmentations=20)
    
    print(f"Protein nodes: {dataset.protein_features.size(0)}")
    print(f"Ligand nodes: {dataset.ligand_features.size(0)}")
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, losses = train_diffusion_model(
        dataset, num_epochs=50, batch_size=1, lr=1e-3, device=device
    )
    
    # Save model
    os.makedirs('outputs', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/diffusion_model.pt')
    print("Model saved to outputs/diffusion_model.pt")
    
    # Save training loss
    np.save('outputs/training_losses.npy', losses)
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('report/images/training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, dataset, device=device, num_samples=10)
    
    # Print statistics
    print(f"\nProtein RMSD: {np.mean(results['protein_rmsd']):.4f} ± {np.std(results['protein_rmsd']):.4f} Å")
    print(f"Ligand RMSD: {np.mean(results['ligand_rmsd']):.4f} ± {np.std(results['ligand_rmsd']):.4f} Å")
    
    # Save results
    results_summary = {
        'protein_rmsd_mean': float(np.mean(results['protein_rmsd'])),
        'protein_rmsd_std': float(np.std(results['protein_rmsd'])),
        'ligand_rmsd_mean': float(np.mean(results['ligand_rmsd'])),
        'ligand_rmsd_std': float(np.std(results['ligand_rmsd'])),
        'protein_rmsd_values': [float(x) for x in results['protein_rmsd']],
        'ligand_rmsd_values': [float(x) for x in results['ligand_rmsd']]
    }
    
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_structures(dataset, results)
    
    print("\nDone!")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
