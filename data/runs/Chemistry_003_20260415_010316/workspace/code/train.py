"""
Training and evaluation scripts for the LES model on all three datasets.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from code.les_model import LESModel, ShortRangeOnlyModel, LESModelWithChargeEmbedding, compute_forces
from code.data_utils import (
    load_random_charges, load_charged_dimer, load_ag3_chargestates,
    compute_coulomb_energy, compute_coulomb_forces, compute_lj_energy, compute_lj_forces
)


# ============================================================
# Dataset classes
# ============================================================

class RandomChargesDataset(torch.utils.data.Dataset):
    """Dataset for random_charges with computed reference energies/forces."""
    
    def __init__(self, data, lj_eps=0.01, lj_sigma=1.0):
        self.data = data
        self.lj_eps = lj_eps
        self.lj_sigma = lj_sigma
        
        # Precompute energies and forces
        self.energies = []
        self.forces_list = []
        for d in data:
            E_coul = compute_coulomb_energy(d['positions'], d['true_charges'])
            F_coul = compute_coulomb_forces(d['positions'], d['true_charges'])
            E_lj = compute_lj_energy(d['positions'], epsilon_lj=lj_eps, sigma_lj=lj_sigma)
            F_lj = compute_lj_forces(d['positions'], epsilon_lj=lj_eps, sigma_lj=lj_sigma)
            self.energies.append(E_coul + E_lj)
            self.forces_list.append(F_coul + F_lj)
        
        self.energies = np.array(self.energies)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            'positions': torch.tensor(d['positions'], dtype=torch.float32),
            'true_charges': torch.tensor(d['true_charges'], dtype=torch.float32),
            'element_indices': torch.zeros(d['natoms'], dtype=torch.long),  # All same element
            'energy': torch.tensor(self.energies[idx], dtype=torch.float32),
            'forces': torch.tensor(self.forces_list[idx], dtype=torch.float32),
            'total_charge': torch.tensor(0.0, dtype=torch.float32),
        }


class ChargedDimerDataset(torch.utils.data.Dataset):
    """Dataset for charged_dimer."""
    
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        # Element indices: C=0, H=1
        elem_map = {'C': 0, 'H': 1}
        elem_idx = torch.tensor([elem_map[s] for s in d['species']], dtype=torch.long)
        
        return {
            'positions': torch.tensor(d['positions'], dtype=torch.float32),
            'element_indices': elem_idx,
            'energy': torch.tensor(d['energy'], dtype=torch.float32),
            'forces': torch.tensor(d['forces'], dtype=torch.float32),
            'separation': torch.tensor(d['separation'], dtype=torch.float32),
            'total_charge': torch.tensor(0.0, dtype=torch.float32),  # Neutral overall
        }


class Ag3ChargestatesDataset(torch.utils.data.Dataset):
    """Dataset for ag3_chargestates."""
    
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        # Element indices: Ag=0
        elem_idx = torch.zeros(d['natoms'], dtype=torch.long)
        
        return {
            'positions': torch.tensor(d['positions'], dtype=torch.float32),
            'element_indices': elem_idx,
            'energy': torch.tensor(d['energy'], dtype=torch.float32),
            'forces': torch.tensor(d['forces'], dtype=torch.float32),
            'charge_state': torch.tensor(d['charge_state'], dtype=torch.float32),
            'total_charge': torch.tensor(float(d['total_charge']), dtype=torch.float32),
        }


# ============================================================
# Training functions
# ============================================================

def train_model(model, dataset, n_epochs=200, lr=1e-3, batch_size=1,
                force_weight=0.1, energy_weight=1.0, device='cpu',
                use_charge_state=False, log_interval=10):
    """Train a model on a dataset.
    
    Args:
        model: LES model
        dataset: PyTorch dataset
        n_epochs: number of training epochs
        lr: learning rate
        batch_size: batch size (1 for variable-size structures)
        force_weight: weight for force loss
        energy_weight: weight for energy loss
        device: torch device
        use_charge_state: whether to pass charge_state as total_charge
        log_interval: logging interval
    
    Returns:
        Training history dict
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    history = {
        'train_energy_mae': [],
        'train_force_mae': [],
        'train_total_loss': [],
    }
    
    for epoch in range(n_epochs):
        model.train()
        epoch_energy_mae = 0.0
        epoch_force_mae = 0.0
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            positions = batch['positions'][0].to(device)  # (N, 3)
            elem_idx = batch['element_indices'][0].to(device)  # (N,)
            ref_energy = batch['energy'].to(device)  # (1,)
            ref_forces = batch['forces'][0].to(device)  # (N, 3)
            
            if use_charge_state:
                total_charge = batch['charge_state'].to(device)
            else:
                total_charge = batch['total_charge'].to(device)
            
            positions.requires_grad_(True)
            
            # Forward pass
            pred_energy, latent_charges = model(
                positions, elem_idx, total_charge=total_charge.item()
            )
            
            # Compute forces via autograd
            pred_forces = -torch.autograd.grad(
                pred_energy, positions, create_graph=True
            )[0]
            
            # Loss
            energy_loss = F.mse_loss(pred_energy, ref_energy)
            force_loss = F.mse_loss(pred_forces, ref_forces)
            loss = energy_weight * energy_loss + force_weight * force_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                energy_mae = torch.abs(pred_energy - ref_energy).item()
                force_mae = torch.abs(pred_forces - ref_forces).mean().item()
            
            epoch_energy_mae += energy_mae
            epoch_force_mae += force_mae
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if n_batches > 0:
            epoch_energy_mae /= n_batches
            epoch_force_mae /= n_batches
            epoch_loss /= n_batches
        
        history['train_energy_mae'].append(epoch_energy_mae)
        history['train_force_mae'].append(epoch_force_mae)
        history['train_total_loss'].append(epoch_loss)
        
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"E_MAE={epoch_energy_mae:.6f}, "
                  f"F_MAE={epoch_force_mae:.6f}, "
                  f"Loss={epoch_loss:.6f}")
    
    return history


def evaluate_model(model, dataset, device='cpu', use_charge_state=False):
    """Evaluate a model on a dataset.
    
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_pred_energies = []
    all_ref_energies = []
    all_pred_forces = []
    all_ref_forces = []
    all_latent_charges = []
    
    with torch.no_grad():
        for batch in dataloader:
            positions = batch['positions'][0].to(device)
            elem_idx = batch['element_indices'][0].to(device)
            ref_energy = batch['energy'].to(device)
            ref_forces = batch['forces'][0].to(device)
            
            if use_charge_state:
                total_charge = batch['charge_state'].to(device)
            else:
                total_charge = batch['total_charge'].to(device)
            
            pred_energy, latent_charges = model(
                positions, elem_idx, total_charge=total_charge.item()
            )
            
            # Compute forces
            positions_grad = positions.clone().requires_grad_(True)
            pred_energy_grad, _ = model(
                positions_grad, elem_idx, total_charge=total_charge.item()
            )
            pred_forces = -torch.autograd.grad(pred_energy_grad, positions_grad)[0]
            
            all_pred_energies.append(pred_energy.item())
            all_ref_energies.append(ref_energy.item())
            all_pred_forces.append(pred_forces.cpu().numpy())
            all_ref_forces.append(ref_forces.cpu().numpy())
            if latent_charges is not None:
                all_latent_charges.append(latent_charges.cpu().numpy())
    
    pred_energies = np.array(all_pred_energies)
    ref_energies = np.array(all_ref_energies)
    pred_forces = np.concatenate(all_pred_forces, axis=0)
    ref_forces = np.concatenate(all_ref_forces, axis=0)
    
    energy_mae = np.abs(pred_energies - ref_energies).mean()
    energy_rmse = np.sqrt(((pred_energies - ref_energies)**2).mean())
    force_mae = np.abs(pred_forces - ref_forces).mean()
    force_rmse = np.sqrt(((pred_forces - ref_forces)**2).mean())
    
    results = {
        'energy_mae': float(energy_mae),
        'energy_rmse': float(energy_rmse),
        'force_mae': float(force_mae),
        'force_rmse': float(force_rmse),
        'pred_energies': pred_energies,
        'ref_energies': ref_energies,
        'pred_forces': pred_forces,
        'ref_forces': ref_forces,
    }
    
    if all_latent_charges:
        results['latent_charges'] = all_latent_charges
    
    return results


import torch.nn.functional as F


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ============================================================
    # Experiment 1: Random Charges - Charge Recovery
    # ============================================================
    print("\n" + "="*60)
    print("Experiment 1: Random Charges - Charge Recovery")
    print("="*60)
    
    rc_data = load_random_charges('data/random_charges.xyz')
    rc_dataset = RandomChargesDataset(rc_data, lj_eps=0.01, lj_sigma=1.0)
    
    # Split into train/test
    n_train = int(0.8 * len(rc_dataset))
    n_test = len(rc_dataset) - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(
        rc_dataset, [n_train, n_test]
    )
    
    # Train LES model
    print("\nTraining LES model on random_charges...")
    les_model = LESModel(n_elements=1, n_rbf=16, cutoff=8.0, n_hidden=64, n_layers=3)
    les_history = train_model(
        les_model, train_dataset, n_epochs=100, lr=1e-3,
        force_weight=0.1, device=device
    )
    
    # Evaluate
    print("\nEvaluating LES model...")
    les_results = evaluate_model(les_model, test_dataset, device=device)
    print(f"  Energy MAE: {les_results['energy_mae']:.6f}")
    print(f"  Force MAE: {les_results['force_mae']:.6f}")
    
    # Check charge recovery
    print("\nChecking charge recovery...")
    for i in range(min(5, len(test_dataset))):
        batch = test_dataset[i]
        positions = batch['positions'].to(device)
        elem_idx = batch['element_indices'].to(device)
        true_charges = batch['true_charges'].numpy()
        
        with torch.no_grad():
            _, latent_charges = les_model(positions, elem_idx, total_charge=0.0)
        
        if latent_charges is not None:
            lc = latent_charges.cpu().numpy()
            # Correlate latent charges with true charges
            correlation = np.corrcoef(lc, true_charges)[0, 1]
            print(f"  Structure {i}: charge correlation = {correlation:.4f}")
    
    # Train short-range only model
    print("\nTraining short-range only model on random_charges...")
    sr_model = ShortRangeOnlyModel(n_elements=1, n_rbf=16, cutoff=8.0, n_hidden=64, n_layers=3)
    sr_history = train_model(
        sr_model, train_dataset, n_epochs=100, lr=1e-3,
        force_weight=0.1, device=device
    )
    
    sr_results = evaluate_model(sr_model, test_dataset, device=device)
    print(f"  Energy MAE: {sr_results['energy_mae']:.6f}")
    print(f"  Force MAE: {sr_results['force_mae']:.6f}")
    
    # Save results
    results_exp1 = {
        'les_energy_mae': les_results['energy_mae'],
        'les_force_mae': les_results['force_mae'],
        'sr_energy_mae': sr_results['energy_mae'],
        'sr_force_mae': sr_results['force_mae'],
    }
    with open('outputs/exp1_random_charges_results.json', 'w') as f:
        json.dump(results_exp1, f, indent=2)
    
    print("\nExperiment 1 complete!")
