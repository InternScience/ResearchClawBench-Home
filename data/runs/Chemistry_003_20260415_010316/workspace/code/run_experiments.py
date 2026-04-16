"""
Main experiment runner for all three benchmark datasets.
Generates results, figures, and saves outputs.
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from code.efficient_les import (
    EfficientLESModel, EfficientShortRangeModel, EfficientLESWithChargeEmbedding
)
from code.data_utils import (
    load_random_charges, load_charged_dimer, load_ag3_chargestates,
    compute_coulomb_energy, compute_coulomb_forces, compute_lj_energy, compute_lj_forces
)

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# ============================================================
# Utility functions
# ============================================================

def compute_reference_random_charges(data, lj_eps=0.01, lj_sigma=1.0):
    """Compute reference energies and forces for random_charges dataset."""
    energies = []
    forces_list = []
    for d in data:
        E_coul = compute_coulomb_energy(d['positions'], d['true_charges'])
        F_coul = compute_coulomb_forces(d['positions'], d['true_charges'])
        E_lj = compute_lj_energy(d['positions'], epsilon_lj=lj_eps, sigma_lj=lj_sigma)
        F_lj = compute_lj_forces(d['positions'], epsilon_lj=lj_eps, sigma_lj=lj_sigma)
        energies.append(E_coul + E_lj)
        forces_list.append(F_coul + F_lj)
    return np.array(energies), forces_list


def train_epoch(model, data_list, optimizer, force_weight=0.1, energy_weight=1.0, 
                use_charge_state=False, device='cpu'):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_e_mae = 0.0
    total_f_mae = 0.0
    n = len(data_list)
    
    indices = np.random.permutation(n)
    
    for idx in indices:
        d = data_list[idx]
        positions = torch.tensor(d['positions'], dtype=torch.float32, device=device)
        ref_energy = torch.tensor([d['energy']], dtype=torch.float32, device=device)
        ref_forces = torch.tensor(d['forces'], dtype=torch.float32, device=device)
        
        # Element indices
        if 'element_indices' in d:
            elem_idx = torch.tensor(d['element_indices'], dtype=torch.long, device=device)
        elif 'species' in d:
            elem_map = {'X': 0, 'C': 0, 'H': 1, 'Ag': 0}
            elem_idx = torch.tensor([elem_map.get(s, 0) for s in d['species']], 
                                     dtype=torch.long, device=device)
        else:
            elem_idx = torch.zeros(d['natoms'], dtype=torch.long, device=device)
        
        total_charge = 0.0
        if use_charge_state and 'charge_state' in d:
            total_charge = float(d['charge_state'])
        elif 'total_charge' in d:
            total_charge = float(d['total_charge'])
        
        positions.requires_grad_(True)
        
        pred_energy, latent_charges = model(positions, elem_idx, total_charge=total_charge)
        
        # Compute forces
        pred_forces = -torch.autograd.grad(pred_energy, positions, create_graph=True)[0]
        
        # Loss
        energy_loss = F.mse_loss(pred_energy.unsqueeze(0), ref_energy)
        force_loss = F.mse_loss(pred_forces, ref_forces)
        loss = energy_weight * energy_loss + force_weight * force_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            e_mae = torch.abs(pred_energy - ref_energy).item()
            f_mae = torch.abs(pred_forces - ref_forces).mean().item()
        
        total_loss += loss.item()
        total_e_mae += e_mae
        total_f_mae += f_mae
    
    return total_loss / n, total_e_mae / n, total_f_mae / n


def evaluate(model, data_list, use_charge_state=False, device='cpu'):
    """Evaluate model on dataset."""
    model.eval()
    pred_energies = []
    ref_energies = []
    pred_forces_list = []
    ref_forces_list = []
    latent_charges_list = []
    
    with torch.no_grad():
        for d in data_list:
            positions = torch.tensor(d['positions'], dtype=torch.float32, device=device)
            ref_energy = d['energy']
            ref_forces = d['forces']
            
            if 'element_indices' in d:
                elem_idx = torch.tensor(d['element_indices'], dtype=torch.long, device=device)
            elif 'species' in d:
                elem_map = {'X': 0, 'C': 0, 'H': 1, 'Ag': 0}
                elem_idx = torch.tensor([elem_map.get(s, 0) for s in d['species']], 
                                         dtype=torch.long, device=device)
            else:
                elem_idx = torch.zeros(d['natoms'], dtype=torch.long, device=device)
            
            total_charge = 0.0
            if use_charge_state and 'charge_state' in d:
                total_charge = float(d['charge_state'])
            elif 'total_charge' in d:
                total_charge = float(d['total_charge'])
            
            pred_energy, latent_charges = model(positions, elem_idx, total_charge=total_charge)
            
            # Compute forces with grad
            positions_grad = positions.clone().requires_grad_(True)
            pred_e_grad, _ = model(positions_grad, elem_idx, total_charge=total_charge)
            pred_forces = -torch.autograd.grad(pred_e_grad, positions_grad)[0]
            
            pred_energies.append(pred_energy.item())
            ref_energies.append(ref_energy)
            pred_forces_list.append(pred_forces.cpu().numpy())
            ref_forces_list.append(ref_forces)
            if latent_charges is not None:
                latent_charges_list.append(latent_charges.cpu().numpy())
    
    pred_energies = np.array(pred_energies)
    ref_energies = np.array(ref_energies)
    pred_forces = np.concatenate(pred_forces_list)
    ref_forces = np.concatenate(ref_forces_list)
    
    results = {
        'energy_mae': float(np.abs(pred_energies - ref_energies).mean()),
        'energy_rmse': float(np.sqrt(((pred_energies - ref_energies)**2).mean())),
        'force_mae': float(np.abs(pred_forces - ref_forces).mean()),
        'force_rmse': float(np.sqrt(((pred_forces - ref_forces)**2).mean())),
        'pred_energies': pred_energies,
        'ref_energies': ref_energies,
    }
    
    if latent_charges_list:
        results['latent_charges'] = latent_charges_list
    
    return results


# ============================================================
# Experiment 1: Random Charges - Charge Recovery
# ============================================================
print("\n" + "="*60)
print("Experiment 1: Random Charges - Charge Recovery")
print("="*60)

rc_data_raw = load_random_charges('data/random_charges.xyz')
rc_energies, rc_forces = compute_reference_random_charges(rc_data_raw)

# Prepare data with computed energies/forces
rc_data = []
for i, d in enumerate(rc_data_raw):
    rc_data.append({
        'positions': d['positions'],
        'forces': rc_forces[i],
        'energy': rc_energies[i],
        'true_charges': d['true_charges'],
        'natoms': d['natoms'],
        'species': d['species'],
        'element_indices': np.zeros(d['natoms'], dtype=int),
        'total_charge': 0.0,
    })

# Split train/test
np.random.seed(42)
n = len(rc_data)
perm = np.random.permutation(n)
n_train = int(0.8 * n)
train_idx = perm[:n_train]
test_idx = perm[n_train:]
rc_train = [rc_data[i] for i in train_idx]
rc_test = [rc_data[i] for i in test_idx]

print(f"Train: {len(rc_train)}, Test: {len(rc_test)}")
print(f"Energy range: {rc_energies.min():.2f} to {rc_energies.max():.2f}")

# Train LES model
print("\nTraining LES model...")
les_model_rc = EfficientLESModel(
    n_elements=1, n_rbf=8, cutoff=8.0, n_hidden=32, n_mp_layers=2
).to(device)

optimizer = optim.Adam(les_model_rc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

les_rc_history = {'loss': [], 'e_mae': [], 'f_mae': []}
for epoch in range(150):
    loss, e_mae, f_mae = train_epoch(
        les_model_rc, rc_train, optimizer, force_weight=0.01, device=device
    )
    scheduler.step()
    les_rc_history['loss'].append(loss)
    les_rc_history['e_mae'].append(e_mae)
    les_rc_history['f_mae'].append(f_mae)
    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, E_MAE={e_mae:.4f}, F_MAE={f_mae:.4f}")

# Evaluate LES
les_rc_results = evaluate(les_model_rc, rc_test, device=device)
print(f"\nLES Results: E_MAE={les_rc_results['energy_mae']:.4f}, F_MAE={les_rc_results['force_mae']:.4f}")

# Check charge recovery
print("\nCharge Recovery Analysis:")
charge_correlations = []
for d in rc_test:
    positions = torch.tensor(d['positions'], dtype=torch.float32, device=device)
    elem_idx = torch.zeros(d['natoms'], dtype=torch.long, device=device)
    true_charges = d['true_charges']
    
    with torch.no_grad():
        _, latent_charges = les_model_rc(positions, elem_idx, total_charge=0.0)
    
    if latent_charges is not None:
        lc = latent_charges.cpu().numpy()
        corr = np.corrcoef(lc, true_charges)[0, 1]
        charge_correlations.append(corr)

charge_correlations = np.array(charge_correlations)
print(f"  Mean charge correlation: {charge_correlations.mean():.4f} ± {charge_correlations.std():.4f}")
print(f"  Min/Max correlation: {charge_correlations.min():.4f} / {charge_correlations.max():.4f}")

# Train short-range only model
print("\nTraining Short-Range only model...")
sr_model_rc = EfficientShortRangeModel(
    n_elements=1, n_rbf=8, cutoff=8.0, n_hidden=32, n_mp_layers=2
).to(device)

optimizer = optim.Adam(sr_model_rc.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

sr_rc_history = {'loss': [], 'e_mae': [], 'f_mae': []}
for epoch in range(150):
    loss, e_mae, f_mae = train_epoch(
        sr_model_rc, rc_train, optimizer, force_weight=0.01, device=device
    )
    scheduler.step()
    sr_rc_history['loss'].append(loss)
    sr_rc_history['e_mae'].append(e_mae)
    sr_rc_history['f_mae'].append(f_mae)
    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, E_MAE={e_mae:.4f}, F_MAE={f_mae:.4f}")

sr_rc_results = evaluate(sr_model_rc, rc_test, device=device)
print(f"\nSR Results: E_MAE={sr_rc_results['energy_mae']:.4f}, F_MAE={sr_rc_results['force_mae']:.4f}")

# Save experiment 1 results
exp1_results = {
    'les_energy_mae': les_rc_results['energy_mae'],
    'les_force_mae': les_rc_results['force_mae'],
    'sr_energy_mae': sr_rc_results['energy_mae'],
    'sr_force_mae': sr_rc_results['force_mae'],
    'charge_correlation_mean': float(charge_correlations.mean()),
    'charge_correlation_std': float(charge_correlations.std()),
}
with open('outputs/exp1_results.json', 'w') as f:
    json.dump({k: v for k, v in exp1_results.items() if not isinstance(v, np.ndarray)}, f, indent=2)

print("\nExperiment 1 complete!")


# ============================================================
# Experiment 2: Charged Dimer - Binding Energy Curve
# ============================================================
print("\n" + "="*60)
print("Experiment 2: Charged Dimer - Binding Energy Curve")
print("="*60)

cd_data_raw = load_charged_dimer('data/charged_dimer.xyz')
cd_data = []
for d in cd_data_raw:
    elem_map = {'C': 0, 'H': 1}
    cd_data.append({
        'positions': d['positions'],
        'forces': d['forces'],
        'energy': d['energy'],
        'natoms': d['natoms'],
        'species': d['species'],
        'element_indices': np.array([elem_map[s] for s in d['species']]),
        'separation': d['separation'],
        'total_charge': 0.0,
    })

# Split train/test by separation
np.random.seed(42)
n = len(cd_data)
perm = np.random.permutation(n)
n_train = int(0.8 * n)
cd_train = [cd_data[i] for i in perm[:n_train]]
cd_test = [cd_data[i] for i in perm[n_train:]]

print(f"Train: {len(cd_train)}, Test: {len(cd_test)}")

# Train LES model
print("\nTraining LES model on charged dimer...")
les_model_cd = EfficientLESModel(
    n_elements=2, n_rbf=8, cutoff=6.0, n_hidden=32, n_mp_layers=2
).to(device)

optimizer = optim.Adam(les_model_cd.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

les_cd_history = {'loss': [], 'e_mae': [], 'f_mae': []}
for epoch in range(200):
    loss, e_mae, f_mae = train_epoch(
        les_model_cd, cd_train, optimizer, force_weight=0.05, device=device
    )
    scheduler.step()
    les_cd_history['loss'].append(loss)
    les_cd_history['e_mae'].append(e_mae)
    les_cd_history['f_mae'].append(f_mae)
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, E_MAE={e_mae:.4f}, F_MAE={f_mae:.4f}")

les_cd_results = evaluate(les_model_cd, cd_test, device=device)
print(f"\nLES Results: E_MAE={les_cd_results['energy_mae']:.4f}, F_MAE={les_cd_results['force_mae']:.4f}")

# Train SR model
print("\nTraining SR model on charged dimer...")
sr_model_cd = EfficientShortRangeModel(
    n_elements=2, n_rbf=8, cutoff=6.0, n_hidden=32, n_mp_layers=2
).to(device)

optimizer = optim.Adam(sr_model_cd.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

sr_cd_history = {'loss': [], 'e_mae': [], 'f_mae': []}
for epoch in range(200):
    loss, e_mae, f_mae = train_epoch(
        sr_model_cd, cd_train, optimizer, force_weight=0.05, device=device
    )
    scheduler.step()
    sr_cd_history['loss'].append(loss)
    sr_cd_history['e_mae'].append(e_mae)
    sr_cd_history['f_mae'].append(f_mae)
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, E_MAE={e_mae:.4f}, F_MAE={f_mae:.4f}")

sr_cd_results = evaluate(sr_model_cd, cd_test, device=device)
print(f"\nSR Results: E_MAE={sr_cd_results['energy_mae']:.4f}, F_MAE={sr_cd_results['force_mae']:.4f}")

# Evaluate on all data for binding curve
les_cd_all = evaluate(les_model_cd, cd_data, device=device)
sr_cd_all = evaluate(sr_model_cd, cd_data, device=device)

exp2_results = {
    'les_energy_mae': les_cd_results['energy_mae'],
    'les_force_mae': les_cd_results['force_mae'],
    'sr_energy_mae': sr_cd_results['energy_mae'],
    'sr_force_mae': sr_cd_results['force_mae'],
}
with open('outputs/exp2_results.json', 'w') as f:
    json.dump(exp2_results, f, indent=2)

print("\nExperiment 2 complete!")


# ============================================================
# Experiment 3: Ag3 Charge States - PES Discrimination
# ============================================================
print("\n" + "="*60)
print("Experiment 3: Ag3 Charge States - PES Discrimination")
print("="*60)

ag_data_raw = load_ag3_chargestates('data/ag3_chargestates.xyz')
ag_data = []
for d in ag_data_raw:
    ag_data.append({
        'positions': d['positions'],
        'forces': d['forces'],
        'energy': d['energy'],
        'natoms': d['natoms'],
        'species': d['species'],
        'element_indices': np.zeros(d['natoms'], dtype=int),
        'charge_state': d['charge_state'],
        'total_charge': float(d['total_charge']),
        'bond_lengths': d['bond_lengths'],
    })

# Split train/test
np.random.seed(42)
n = len(ag_data)
perm = np.random.permutation(n)
n_train = int(0.8 * n)
ag_train = [ag_data[i] for i in perm[:n_train]]
ag_test = [ag_data[i] for i in perm[n_train:]]

print(f"Train: {len(ag_train)}, Test: {len(ag_test)}")

# Train LES model (without charge embedding)
print("\nTraining LES model (no charge embedding) on Ag3...")
les_model_ag = EfficientLESModel(
    n_elements=1, n_rbf=8, cutoff=6.0, n_hidden=32, n_mp_layers=2
).to(device)

optimizer = optim.Adam(les_model_ag.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

les_ag_history = {'loss': [], 'e_mae': [], 'f_mae': []}
for epoch in range(200):
    loss, e_mae, f_mae = train_epoch(
        les_model_ag, ag_train, optimizer, force_weight=0.05, 
        use_charge_state=True, device=device
    )
    scheduler.step()
    les_ag_history['loss'].append(loss)
    les_ag_history['e_mae'].append(e_mae)
    les_ag_history['f_mae'].append(f_mae)
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, E_MAE={e_mae:.4f}, F_MAE={f_mae:.4f}")

les_ag_results = evaluate(les_model_ag, ag_test, use_charge_state=True, device=device)
print(f"\nLES Results: E_MAE={les_ag_results['energy_mae']:.4f}, F_MAE={les_ag_results['force_mae']:.4f}")

# Train SR model (without charge embedding)
print("\nTraining SR model (no charge embedding) on Ag3...")
sr_model_ag = EfficientShortRangeModel(
    n_elements=1, n_rbf=8, cutoff=6.0, n_hidden=32, n_mp_layers=2
).to(device)

optimizer = optim.Adam(sr_model_ag.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

sr_ag_history = {'loss': [], 'e_mae': [], 'f_mae': []}
for epoch in range(200):
    loss, e_mae, f_mae = train_epoch(
        sr_model_ag, ag_train, optimizer, force_weight=0.05,
        use_charge_state=True, device=device
    )
    scheduler.step()
    sr_ag_history['loss'].append(loss)
    sr_ag_history['e_mae'].append(e_mae)
    sr_ag_history['f_mae'].append(f_mae)
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, E_MAE={e_mae:.4f}, F_MAE={f_mae:.4f}")

sr_ag_results = evaluate(sr_model_ag, ag_test, use_charge_state=True, device=device)
print(f"\nSR Results: E_MAE={sr_ag_results['energy_mae']:.4f}, F_MAE={sr_ag_results['force_mae']:.4f}")

# Train LES with charge embedding
print("\nTraining LES+ChargeEmbedding model on Ag3...")
les_ce_model_ag = EfficientLESWithChargeEmbedding(
    n_elements=1, n_rbf=8, cutoff=6.0, n_hidden=32, n_mp_layers=2
).to(device)

optimizer = optim.Adam(les_ce_model_ag.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

ce_ag_history = {'loss': [], 'e_mae': [], 'f_mae': []}
for epoch in range(200):
    loss, e_mae, f_mae = train_epoch(
        les_ce_model_ag, ag_train, optimizer, force_weight=0.05,
        use_charge_state=True, device=device
    )
    scheduler.step()
    ce_ag_history['loss'].append(loss)
    ce_ag_history['e_mae'].append(e_mae)
    ce_ag_history['f_mae'].append(f_mae)
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}: loss={loss:.4f}, E_MAE={e_mae:.4f}, F_MAE={f_mae:.4f}")

ce_ag_results = evaluate(les_ce_model_ag, ag_test, use_charge_state=True, device=device)
print(f"\nLES+CE Results: E_MAE={ce_ag_results['energy_mae']:.4f}, F_MAE={ce_ag_results['force_mae']:.4f}")

# Evaluate on all data for PES analysis
les_ag_all = evaluate(les_model_ag, ag_data, use_charge_state=True, device=device)
sr_ag_all = evaluate(sr_model_ag, ag_data, use_charge_state=True, device=device)
ce_ag_all = evaluate(les_ce_model_ag, ag_data, use_charge_state=True, device=device)

exp3_results = {
    'les_energy_mae': les_ag_results['energy_mae'],
    'les_force_mae': les_ag_results['force_mae'],
    'sr_energy_mae': sr_ag_results['energy_mae'],
    'sr_force_mae': sr_ag_results['force_mae'],
    'ce_energy_mae': ce_ag_results['energy_mae'],
    'ce_force_mae': ce_ag_results['force_mae'],
}
with open('outputs/exp3_results.json', 'w') as f:
    json.dump(exp3_results, f, indent=2)

print("\nExperiment 3 complete!")


# ============================================================
# Save all data for figure generation
# ============================================================
print("\nSaving data for figure generation...")

# Save all results
all_results = {
    'exp1': {
        'les': les_rc_results,
        'sr': sr_rc_results,
        'charge_correlations': charge_correlations.tolist(),
        'les_history': les_rc_history,
        'sr_history': sr_rc_history,
    },
    'exp2': {
        'les': les_cd_all,
        'sr': sr_cd_all,
        'les_test': les_cd_results,
        'sr_test': sr_cd_results,
        'les_history': les_cd_history,
        'sr_history': sr_cd_history,
        'separations': [d['separation'] for d in cd_data],
    },
    'exp3': {
        'les': les_ag_all,
        'sr': sr_ag_all,
        'ce': ce_ag_all,
        'les_test': les_ag_results,
        'sr_test': sr_ag_results,
        'ce_test': ce_ag_results,
        'les_history': les_ag_history,
        'sr_history': sr_ag_history,
        'ce_history': ce_ag_history,
    },
}

# Convert numpy arrays for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open('outputs/all_results.json', 'w') as f:
    json.dump(all_results, f, cls=NumpyEncoder, indent=2)

# Also save as torch checkpoints
torch.save({
    'les_rc': les_model_rc.state_dict(),
    'sr_rc': sr_model_rc.state_dict(),
    'les_cd': les_model_cd.state_dict(),
    'sr_cd': sr_model_cd.state_dict(),
    'les_ag': les_model_ag.state_dict(),
    'sr_ag': sr_model_ag.state_dict(),
    'ce_ag': les_ce_model_ag.state_dict(),
}, 'outputs/model_checkpoints.pt')

print("\nAll experiments complete! Data saved to outputs/")
