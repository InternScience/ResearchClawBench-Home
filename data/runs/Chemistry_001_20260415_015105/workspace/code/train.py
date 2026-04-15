"""
Training script for the unified diffusion-based biomolecular complex structure prediction model.

Trains on the 2L3R FKBP12-FK506 complex data.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_model import BioComplexDiffusionModel
from data_preprocessing import (
    parse_pdb, parse_sdf, sequence_to_onehot, 
    compute_adjacency_matrix, center_coordinates
)


def prepare_training_data(protein_path: str, ligand_path: str, device: str = 'cpu'):
    """Load and prepare training data from PDB and SDF files."""
    protein = parse_pdb(protein_path)
    ligand = parse_sdf(ligand_path)
    
    # Center coordinates
    centered_ca, _ = center_coordinates(protein['ca_coords'])
    centered_lig, _ = center_coordinates(ligand['atom_coords'])
    
    # Convert to tensors
    protein_coords = torch.from_numpy(centered_ca).unsqueeze(0).to(device)  # (1, N_res, 3)
    ligand_coords = torch.from_numpy(centered_lig).unsqueeze(0).to(device)  # (1, N_atoms, 3)
    
    onehot = sequence_to_onehot(protein['sequence'])
    protein_onehot = torch.from_numpy(onehot).unsqueeze(0).to(device)  # (1, N_res, 20)
    
    atom_types = torch.from_numpy(ligand['atom_type_indices']).unsqueeze(0).to(device)  # (1, N_atoms)
    adj = compute_adjacency_matrix(ligand['n_atoms'], ligand['bonds'])
    ligand_adj = torch.from_numpy(adj).unsqueeze(0).to(device)  # (1, N_atoms, N_atoms)
    
    return {
        'protein_coords': protein_coords,
        'ligand_coords': ligand_coords,
        'protein_onehot': protein_onehot,
        'ligand_atom_types': atom_types,
        'ligand_adj': ligand_adj,
        'n_residues': protein['n_ca_atoms'],
        'n_atoms': ligand['n_atoms'],
        'protein_raw_coords': protein['ca_coords'],
        'ligand_raw_coords': ligand['atom_coords'],
        'protein_sequence': protein['sequence'],
        'ligand_atom_types_list': ligand['atom_types'],
    }


def train_model(data: dict, config: dict) -> dict:
    """Train the diffusion model."""
    device = config.get('device', 'cpu')
    n_epochs = config.get('n_epochs', 500)
    lr = config.get('lr', 1e-3)
    n_timesteps = config.get('n_timesteps', 100)
    d_model = config.get('d_model', 128)
    save_interval = config.get('save_interval', 50)
    
    # Initialize model
    model = BioComplexDiffusionModel(d_model=d_model, n_timesteps=n_timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training loop
    training_history = {
        'epoch': [],
        'total_loss': [],
        'protein_loss': [],
        'ligand_loss': [],
        'lr': [],
    }
    
    best_loss = float('inf')
    best_state = None
    
    print(f"Starting training for {n_epochs} epochs...")
    print(f"Device: {device}, d_model: {d_model}, timesteps: {n_timesteps}")
    print(f"Learning rate: {lr}")
    print(f"Protein residues: {data['n_residues']}, Ligand atoms: {data['n_atoms']}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(
            data['protein_coords'],
            data['ligand_coords'],
            data['protein_onehot'],
            data['ligand_atom_types'],
            data['ligand_adj'],
        )
        
        loss = output['total_loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Record history
        training_history['epoch'].append(epoch)
        training_history['total_loss'].append(loss.item())
        training_history['protein_loss'].append(output['protein_loss'].item())
        training_history['ligand_loss'].append(output['ligand_loss'].item())
        training_history['lr'].append(scheduler.get_last_lr()[0])
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Prot: {output['protein_loss'].item():.6f} | "
                  f"Lig: {output['ligand_loss'].item():.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {elapsed:.1f}s")
    
    print("-" * 60)
    print(f"Training complete. Best loss: {best_loss:.6f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, training_history


def evaluate_model(model: BioComplexDiffusionModel, data: dict, 
                   config: dict) -> dict:
    """Evaluate the trained model by generating samples and computing metrics."""
    device = config.get('device', 'cpu')
    n_samples = config.get('eval_n_samples', 5)
    
    model.eval()
    
    with torch.no_grad():
        # Generate samples
        pred_protein, pred_ligand = model.sample(
            data['protein_onehot'],
            data['ligand_atom_types'],
            data['ligand_adj'],
            data['n_residues'],
            data['n_atoms'],
            n_samples=n_samples,
        )
    
    # Compute RMSD for each sample
    true_protein = data['protein_coords'].cpu().numpy()  # (1, N_res, 3)
    true_ligand = data['ligand_coords'].cpu().numpy()  # (1, N_atoms, 3)
    
    protein_rmsds = []
    ligand_rmsds = []
    
    for i in range(n_samples):
        p_pred = pred_protein[i].cpu().numpy()
        l_pred = pred_ligand[i].cpu().numpy()
        
        # Simple RMSD (without alignment for now)
        p_diff = p_pred - true_protein[0]
        p_rmsd = np.sqrt(np.mean(p_diff ** 2))
        protein_rmsds.append(float(p_rmsd))
        
        l_diff = l_pred - true_ligand[0]
        l_rmsd = np.sqrt(np.mean(l_diff ** 2))
        ligand_rmsds.append(float(l_rmsd))
    
    results = {
        'protein_rmsds': protein_rmsds,
        'ligand_rmsds': ligand_rmsds,
        'mean_protein_rmsd': float(np.mean(protein_rmsds)),
        'std_protein_rmsd': float(np.std(protein_rmsds)),
        'mean_ligand_rmsd': float(np.mean(ligand_rmsds)),
        'std_ligand_rmsd': float(np.std(ligand_rmsds)),
        'best_protein_rmsd': float(np.min(protein_rmsds)),
        'best_ligand_rmsd': float(np.min(ligand_rmsds)),
        'pred_protein_coords': pred_protein.cpu().numpy(),
        'pred_ligand_coords': pred_ligand.cpu().numpy(),
    }
    
    return results


def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 500,
        'lr': 5e-4,
        'n_timesteps': 100,
        'd_model': 128,
        'save_interval': 50,
        'eval_n_samples': 5,
    }
    
    print("=" * 60)
    print("Unified Diffusion-Based Biomolecular Complex Structure Prediction")
    print("=" * 60)
    
    # Prepare data
    print("\nPreparing training data...")
    data = prepare_training_data(
        'data/sample/2l3r/2l3r_protein.pdb',
        'data/sample/2l3r/2l3r_ligand.sdf',
        device=config['device']
    )
    print(f"Data loaded: {data['n_residues']} protein residues, {data['n_atoms']} ligand atoms")
    
    # Train
    print("\nTraining model...")
    model, history = train_model(data, config)
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, data, config)
    
    print(f"\nEvaluation Results:")
    print(f"  Protein RMSD: {results['mean_protein_rmsd']:.4f} ± {results['std_protein_rmsd']:.4f} Å")
    print(f"  Ligand RMSD:  {results['mean_ligand_rmsd']:.4f} ± {results['std_ligand_rmsd']:.4f} Å")
    print(f"  Best Protein RMSD: {results['best_protein_rmsd']:.4f} Å")
    print(f"  Best Ligand RMSD:  {results['best_ligand_rmsd']:.4f} Å")
    
    # Save outputs
    print("\nSaving outputs...")
    
    # Save training history
    np.savez('outputs/training_history.npz',
             epoch=np.array(history['epoch']),
             total_loss=np.array(history['total_loss']),
             protein_loss=np.array(history['protein_loss']),
             ligand_loss=np.array(history['ligand_loss']),
             lr=np.array(history['lr']))
    
    # Save evaluation results
    eval_results = {
        'protein_rmsds': results['protein_rmsds'],
        'ligand_rmsds': results['ligand_rmsds'],
        'mean_protein_rmsd': results['mean_protein_rmsd'],
        'std_protein_rmsd': results['std_protein_rmsd'],
        'mean_ligand_rmsd': results['mean_ligand_rmsd'],
        'std_ligand_rmsd': results['std_ligand_rmsd'],
        'best_protein_rmsd': results['best_protein_rmsd'],
        'best_ligand_rmsd': results['best_ligand_rmsd'],
    }
    
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Save predicted coordinates
    np.savez('outputs/predicted_structures.npz',
             pred_protein_coords=results['pred_protein_coords'],
             pred_ligand_coords=results['pred_ligand_coords'],
             true_protein_coords=data['protein_coords'].cpu().numpy(),
             true_ligand_coords=data['ligand_coords'].cpu().numpy())
    
    # Save model state
    torch.save(model.state_dict(), 'outputs/model_checkpoint.pt')
    
    # Save config
    with open('outputs/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("All outputs saved to outputs/")
    
    return model, history, results


if __name__ == '__main__':
    main()
