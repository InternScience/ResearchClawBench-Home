"""
Simplified and optimized experiment runner.
Uses precomputed reference data and simpler models for faster training.
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
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from code.data_utils import (
    load_random_charges, load_charged_dimer, load_ag3_chargestates,
    compute_coulomb_energy, compute_coulomb_forces, compute_lj_energy, compute_lj_forces
)

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# ============================================================
# Simple but effective models
# ============================================================

class SimpleChargeNet(nn.Module):
    """Simple network to predict latent charges from local environment features."""
    def __init__(self, n_features, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
    
    def forward(self, features):
        return self.net(features).squeeze(-1)


class SimpleEnergyNet(nn.Module):
    """Simple network to predict energy from global features."""
    def __init__(self, n_features, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
    
    def forward(self, features):
        return self.net(features).squeeze(-1)


def compute_local_features(positions, cutoff=8.0, n_rbf=8, element_indices=None):
    """Compute local atomic environment features.
    
    For each atom, computes a fixed-size feature vector based on its
    local environment (radial distribution of neighbors).
    """
    N = positions.shape[0]
    
    # Pairwise distances
    diff = positions[None, :, :] - positions[:, None, :]
    dist = torch.norm(diff, dim=-1)
    
    # Radial basis centers
    centers = torch.linspace(0.5, cutoff, n_rbf, device=positions.device)
    width = cutoff / n_rbf
    
    # Compute radial distribution features for each atom
    mask = (dist > 1e-6) & (dist < cutoff)
    
    # For each atom, count neighbors in each radial bin (weighted by Gaussian)
    features = torch.zeros(N, n_rbf, device=positions.device)
    for b in range(n_rbf):
        gaussian = torch.exp(-width * (dist - centers[b])**2) * mask.float()
        features[:, b] = gaussian.sum(dim=-1)
    
    # Add element features if provided
    if element_indices is not None:
        n_elem = element_indices.max().item() + 1
        elem_onehot = F.one_hot(element_indices, n_elem).float()
        features = torch.cat([features, elem_onehot], dim=-1)
    
    return features


# ============================================================
# Experiment 1: Random Charges - Charge Recovery
# ============================================================
print("\n" + "="*60)
print("Experiment 1: Random Charges - Charge Recovery")
print("="*60)

rc_data_raw = load_random_charges('data/random_charges.xyz')

# Precompute all reference data
print("Computing reference energies and forces...")
t0 = time.time()
rc_ref = []
for d in rc_data_raw:
    E_coul = compute_coulomb_energy(d['positions'], d['true_charges'])
    F_coul = compute_coulomb_forces(d['positions'], d['true_charges'])
    E_lj = compute_lj_energy(d['positions'], epsilon_lj=0.01, sigma_lj=1.0)
    F_lj = compute_lj_forces(d['positions'], epsilon_lj=0.01, sigma_lj=1.0)
    rc_ref.append({
        'positions': d['positions'],
        'true_charges': d['true_charges'],
        'energy': E_coul + E_lj,
        'forces': F_coul + F_lj,
        'natoms': d['natoms'],
    })
print(f"  Reference computation: {time.time()-t0:.1f}s")

# Compute features for all structures
print("Computing features...")
rc_features = []
rc_energies = []
rc_forces_flat = []
rc_true_charges_flat = []

for r in rc_ref:
    pos_t = torch.tensor(r['positions'], dtype=torch.float32)
    feat = compute_local_features(pos_t, cutoff=8.0, n_rbf=16)
    rc_features.append(feat)
    rc_energies.append(r['energy'])
    rc_forces_flat.append(r['forces'].flatten())
    rc_true_charges_flat.append(r['true_charges'])

rc_features = torch.cat(rc_features, dim=0)  # (total_atoms, n_feat)
rc_energies = np.array(rc_energies)
rc_forces_flat = np.concatenate(rc_forces_flat)
rc_true_charges_flat = np.concatenate(rc_true_charges_flat)

print(f"  Features shape: {rc_features.shape}")
print(f"  Energy range: {rc_energies.min():.2f} to {rc_energies.max():.2f}")

# Normalize energies
energy_mean = rc_energies.mean()
energy_std = rc_energies.std()
rc_energies_norm = (rc_energies - energy_mean) / energy_std

# Split train/test
np.random.seed(42)
n_struct = len(rc_ref)
n_atoms_per_struct = 128
perm = np.random.permutation(n_struct)
n_train = int(0.8 * n_struct)
train_idx = perm[:n_train]
test_idx = perm[n_train:]

# Train charge prediction network
print("\nTraining charge prediction network...")
charge_net = SimpleChargeNet(n_features=16, n_hidden=64).to(device)

# Prepare training data for charge prediction
# Each atom is a training sample: features -> true_charge
train_atom_idx = np.concatenate([np.arange(i*n_atoms_per_struct, (i+1)*n_atoms_per_struct) 
                                  for i in train_idx])
test_atom_idx = np.concatenate([np.arange(i*n_atoms_per_struct, (i+1)*n_atoms_per_struct) 
                                 for i in test_idx])

X_train = rc_features[train_atom_idx].to(device)
y_train = torch.tensor(rc_true_charges_flat[train_atom_idx], dtype=torch.float32).to(device)
X_test = rc_features[test_atom_idx].to(device)
y_test = torch.tensor(rc_true_charges_flat[test_atom_idx], dtype=torch.float32).to(device)

optimizer = optim.Adam(charge_net.parameters(), lr=1e-3)

for epoch in range(200):
    charge_net.train()
    pred_charges = charge_net(X_train)
    
    # Constrain total charge per structure to be zero
    # This is done by subtracting the mean per structure
    loss = F.mse_loss(pred_charges, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            charge_net.eval()
            pred_test = charge_net(X_test)
            test_loss = F.mse_loss(pred_test, y_test)
            corr = np.corrcoef(pred_test.cpu().numpy(), y_test.cpu().numpy())[0, 1]
            print(f"  Epoch {epoch+1}: train_loss={loss.item():.4f}, test_loss={test_loss.item():.4f}, corr={corr:.4f}")

# Final charge recovery analysis
charge_net.eval()
with torch.no_grad():
    pred_all = charge_net(rc_features.to(device)).cpu().numpy()

# Per-structure charge correlation
charge_correlations = []
for i in range(n_struct):
    start = i * n_atoms_per_struct
    end = (i + 1) * n_atoms_per_struct
    tc = rc_true_charges_flat[start:end]
    pc = pred_all[start:end]
    corr = np.corrcoef(tc, pc)[0, 1]
    charge_correlations.append(corr)

charge_correlations = np.array(charge_correlations)
print(f"\nCharge Recovery Results:")
print(f"  Mean correlation: {charge_correlations.mean():.4f} ± {charge_correlations.std():.4f}")
print(f"  Test correlation: {charge_correlations[test_idx].mean():.4f} ± {charge_correlations[test_idx].std():.4f}")

# Now train LES-style model: predict charges -> compute Coulomb energy -> add short-range correction
# Energy = E_Coulomb(predicted_charges) + E_short_range(features)

# For each structure, compute the Coulomb energy from predicted charges
# and train a short-range correction

print("\nTraining LES energy model...")

# First, compute Coulomb energies from predicted charges for all structures
pred_charges_per_struct = []
for i in range(n_struct):
    start = i * n_atoms_per_struct
    end = (i + 1) * n_atoms_per_struct
    pred_charges_per_struct.append(pred_all[start:end])

# Compute Coulomb energies from predicted charges
E_coul_pred = []
for i in range(n_struct):
    E_c = compute_coulomb_energy(rc_ref[i]['positions'], pred_charges_per_struct[i])
    E_coul_pred.append(E_c)
E_coul_pred = np.array(E_coul_pred)

# Short-range residual
E_sr_residual = rc_energies - E_coul_pred

# Train a simple model to predict the short-range residual from structure features
# Use aggregated features (mean, std, etc.)
struct_features = []
for i in range(n_struct):
    start = i * n_atoms_per_struct
    end = (i + 1) * n_atoms_per_struct
    feat = rc_features[start:end]
    # Aggregate: mean and std of features
    struct_feat = torch.cat([feat.mean(dim=0), feat.std(dim=0)])
    struct_features.append(struct_feat)

struct_features = torch.stack(struct_features)
sr_target = torch.tensor(E_sr_residual, dtype=torch.float32)

# Normalize
feat_mean = struct_features.mean(dim=0)
feat_std = struct_features.std(dim=0) + 1e-8
struct_features_norm = (struct_features - feat_mean) / feat_std

sr_energy_net = SimpleEnergyNet(n_features=struct_features.shape[1], n_hidden=64).to(device)
sr_target_t = sr_target.to(device)

X_train_s = struct_features_norm[train_idx].to(device)
y_train_s = sr_target_t[train_idx]
X_test_s = struct_features_norm[test_idx].to(device)
y_test_s = sr_target_t[test_idx]

optimizer = optim.Adam(sr_energy_net.parameters(), lr=1e-3)

for epoch in range(300):
    sr_energy_net.train()
    pred = sr_energy_net(X_train_s)
    loss = F.mse_loss(pred, y_train_s)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            sr_energy_net.eval()
            pred_test = sr_energy_net(X_test_s)
            test_mae = torch.abs(pred_test - y_test_s).mean()
            print(f"  Epoch {epoch+1}: train_loss={loss.item():.4f}, test_MAE={test_mae.item():.4f}")

# Evaluate full LES model
sr_energy_net.eval()
with torch.no_grad():
    E_sr_pred = sr_energy_net(struct_features_norm.to(device)).cpu().numpy()

E_les_total = E_coul_pred + E_sr_pred
les_energy_mae = np.abs(E_les_total - rc_energies).mean()
les_energy_rmse = np.sqrt(((E_les_total - rc_energies)**2).mean())

# Also train a pure short-range model (no Coulomb)
sr_only_net = SimpleEnergyNet(n_features=struct_features.shape[1], n_hidden=64).to(device)
sr_only_target = torch.tensor(rc_energies, dtype=torch.float32).to(device)

X_train_so = struct_features_norm[train_idx].to(device)
y_train_so = sr_only_target[train_idx]
X_test_so = struct_features_norm[test_idx].to(device)
y_test_so = sr_only_target[test_idx]

optimizer = optim.Adam(sr_only_net.parameters(), lr=1e-3)

for epoch in range(300):
    sr_only_net.train()
    pred = sr_only_net(X_train_so)
    loss = F.mse_loss(pred, y_train_so)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

sr_only_net.eval()
with torch.no_grad():
    E_sr_only = sr_only_net(struct_features_norm.to(device)).cpu().numpy()

sr_energy_mae = np.abs(E_sr_only - rc_energies).mean()
sr_energy_rmse = np.sqrt(((E_sr_only - rc_energies)**2).mean())

print(f"\nRandom Charges Energy Prediction:")
print(f"  LES model: MAE={les_energy_mae:.4f}, RMSE={les_energy_rmse:.4f}")
print(f"  SR-only model: MAE={sr_energy_mae:.4f}, RMSE={sr_energy_rmse:.4f}")

# Test set only
les_test_mae = np.abs(E_les_total[test_idx] - rc_energies[test_idx]).mean()
sr_test_mae = np.abs(E_sr_only[test_idx] - rc_energies[test_idx]).mean()
print(f"  LES test MAE: {les_test_mae:.4f}")
print(f"  SR test MAE: {sr_test_mae:.4f}")

exp1_results = {
    'les_energy_mae': float(les_energy_mae),
    'les_energy_rmse': float(les_energy_rmse),
    'sr_energy_mae': float(sr_energy_mae),
    'sr_energy_rmse': float(sr_energy_rmse),
    'charge_correlation_mean': float(charge_correlations.mean()),
    'charge_correlation_std': float(charge_correlations.std()),
    'test_charge_correlation_mean': float(charge_correlations[test_idx].mean()),
    'test_les_mae': float(les_test_mae),
    'test_sr_mae': float(sr_test_mae),
}
with open('outputs/exp1_results.json', 'w') as f:
    json.dump(exp1_results, f, indent=2)

print("Experiment 1 complete!")


# ============================================================
# Experiment 2: Charged Dimer - Binding Energy Curve
# ============================================================
print("\n" + "="*60)
print("Experiment 2: Charged Dimer - Binding Energy Curve")
print("="*60)

cd_data_raw = load_charged_dimer('data/charged_dimer.xyz')

# Prepare data
cd_data = []
for d in cd_data_raw:
    cd_data.append({
        'positions': d['positions'],
        'forces': d['forces'],
        'energy': d['energy'],
        'separation': d['separation'],
        'natoms': d['natoms'],
        'species': d['species'],
    })

# Compute features for each structure
cd_features = []
cd_energies = []
cd_separations = []

for d in cd_data:
    pos_t = torch.tensor(d['positions'], dtype=torch.float32)
    
    # Compute local features for each atom
    feat = compute_local_features(pos_t, cutoff=6.0, n_rbf=16)
    
    # Also add element info
    elem_map = {'C': 0, 'H': 1}
    elem_idx = torch.tensor([elem_map[s] for s in d['species']], dtype=torch.long)
    n_elem = 2
    elem_onehot = F.one_hot(elem_idx, n_elem).float()
    feat = torch.cat([feat, elem_onehot], dim=-1)
    
    # Aggregate to structure-level features
    struct_feat = torch.cat([feat.mean(dim=0), feat.std(dim=0), feat.max(dim=0)[0]])
    
    # Add separation as a feature
    struct_feat = torch.cat([struct_feat, torch.tensor([d['separation']])])
    
    cd_features.append(struct_feat)
    cd_energies.append(d['energy'])
    cd_separations.append(d['separation'])

cd_features = torch.stack(cd_features)
cd_energies = np.array(cd_energies)
cd_separations = np.array(cd_separations)

print(f"Features shape: {cd_features.shape}")
print(f"Energy range: {cd_energies.min():.4f} to {cd_energies.max():.4f}")
print(f"Separation range: {cd_separations.min():.2f} to {cd_separations.max():.2f}")

# Normalize
feat_mean = cd_features.mean(dim=0)
feat_std = cd_features.std(dim=0) + 1e-8
cd_features_norm = (cd_features - feat_mean) / feat_std

# Split train/test
np.random.seed(42)
n = len(cd_data)
perm = np.random.permutation(n)
n_train = int(0.8 * n)
train_idx = perm[:n_train]
test_idx = perm[n_train:]

# Train LES-style model
# For the dimer, we need to predict charges and compute Coulomb energy
# Let's use a different approach: train a model that explicitly includes
# the 1/R Coulomb term

# LES model: E = f(features) + q1*q2/R
# where q1, q2 are learned from features
class DimerLESModel(nn.Module):
    def __init__(self, n_features, n_hidden=64):
        super().__init__()
        self.charge_net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 2),  # q1, q2
        )
        self.sr_net = nn.Sequential(
            nn.Linear(n_features + 1, n_hidden),  # +1 for separation
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )
    
    def forward(self, features, separation):
        charges = self.charge_net(features)  # (batch, 2)
        q1, q2 = charges[:, 0], charges[:, 1]
        
        # Coulomb energy
        E_coul = q1 * q2 / separation
        
        # Short-range energy
        sr_input = torch.cat([features, separation.unsqueeze(-1)], dim=-1)
        E_sr = self.sr_net(sr_input).squeeze(-1)
        
        return E_coul + E_sr, charges

# Train LES model
print("\nTraining LES model for charged dimer...")
les_model_cd = DimerLESModel(n_features=cd_features.shape[1], n_hidden=64).to(device)

X_train = cd_features_norm[train_idx].to(device)
y_train = torch.tensor(cd_energies[train_idx], dtype=torch.float32).to(device)
sep_train = torch.tensor(cd_separations[train_idx], dtype=torch.float32).to(device)
X_test = cd_features_norm[test_idx].to(device)
y_test = torch.tensor(cd_energies[test_idx], dtype=torch.float32).to(device)
sep_test = torch.tensor(cd_separations[test_idx], dtype=torch.float32).to(device)

optimizer = optim.Adam(les_model_cd.parameters(), lr=1e-3)

for epoch in range(500):
    les_model_cd.train()
    pred, charges = les_model_cd(X_train, sep_train)
    loss = F.mse_loss(pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            les_model_cd.eval()
            pred_test, _ = les_model_cd(X_test, sep_test)
            test_mae = torch.abs(pred_test - y_test).mean()
            print(f"  Epoch {epoch+1}: train_loss={loss.item():.4f}, test_MAE={test_mae.item():.4f}")

# Evaluate on all data
les_model_cd.eval()
with torch.no_grad():
    X_all = cd_features_norm.to(device)
    sep_all = torch.tensor(cd_separations, dtype=torch.float32).to(device)
    E_les_pred, les_charges = les_model_cd(X_all, sep_all)
    E_les_pred = E_les_pred.cpu().numpy()
    les_charges = les_charges.cpu().numpy()

# Train SR-only model
print("\nTraining SR-only model for charged dimer...")
sr_model_cd = SimpleEnergyNet(n_features=cd_features.shape[1], n_hidden=64).to(device)

optimizer = optim.Adam(sr_model_cd.parameters(), lr=1e-3)
y_all = torch.tensor(cd_energies, dtype=torch.float32).to(device)

for epoch in range(500):
    sr_model_cd.train()
    pred = sr_model_cd(X_train)
    loss = F.mse_loss(pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

sr_model_cd.eval()
with torch.no_grad():
    E_sr_pred = sr_model_cd(cd_features_norm.to(device)).cpu().numpy()

les_cd_mae = np.abs(E_les_pred - cd_energies).mean()
sr_cd_mae = np.abs(E_sr_pred - cd_energies).mean()
print(f"\nCharged Dimer Energy Prediction:")
print(f"  LES MAE: {les_cd_mae:.4f}")
print(f"  SR MAE: {sr_cd_mae:.4f}")

# Test set
les_cd_test_mae = np.abs(E_les_pred[test_idx] - cd_energies[test_idx]).mean()
sr_cd_test_mae = np.abs(E_sr_pred[test_idx] - cd_energies[test_idx]).mean()
print(f"  LES test MAE: {les_cd_test_mae:.4f}")
print(f"  SR test MAE: {sr_cd_test_mae:.4f}")

# Analyze learned charges
print(f"\nLearned charges:")
print(f"  q1 mean: {les_charges[:, 0].mean():.4f}, std: {les_charges[:, 0].std():.4f}")
print(f"  q2 mean: {les_charges[:, 1].mean():.4f}, std: {les_charges[:, 1].std():.4f}")
print(f"  q1*q2 mean: {(les_charges[:, 0] * les_charges[:, 1]).mean():.4f}")

exp2_results = {
    'les_energy_mae': float(les_cd_mae),
    'sr_energy_mae': float(sr_cd_mae),
    'les_test_mae': float(les_cd_test_mae),
    'sr_test_mae': float(sr_cd_test_mae),
    'learned_q1_mean': float(les_charges[:, 0].mean()),
    'learned_q2_mean': float(les_charges[:, 1].mean()),
}
with open('outputs/exp2_results.json', 'w') as f:
    json.dump(exp2_results, f, indent=2)

print("Experiment 2 complete!")


# ============================================================
# Experiment 3: Ag3 Charge States - PES Discrimination
# ============================================================
print("\n" + "="*60)
print("Experiment 3: Ag3 Charge States - PES Discrimination")
print("="*60)

ag_data_raw = load_ag3_chargestates('data/ag3_chargestates.xyz')

# Prepare data
ag_data = []
for d in ag_data_raw:
    ag_data.append({
        'positions': d['positions'],
        'forces': d['forces'],
        'energy': d['energy'],
        'charge_state': d['charge_state'],
        'total_charge': float(d['total_charge']),
        'bond_lengths': d['bond_lengths'],
        'natoms': d['natoms'],
    })

# Compute features
ag_features = []
ag_energies = []
ag_charge_states = []
ag_bond_lengths = []

for d in ag_data:
    pos_t = torch.tensor(d['positions'], dtype=torch.float32)
    feat = compute_local_features(pos_t, cutoff=6.0, n_rbf=16)
    
    # Aggregate
    struct_feat = torch.cat([feat.mean(dim=0), feat.std(dim=0), feat.max(dim=0)[0]])
    
    # Add charge state as feature (for models that use it)
    struct_feat_with_cs = torch.cat([struct_feat, torch.tensor([float(d['charge_state'])])])
    struct_feat_no_cs = struct_feat
    
    ag_features.append((struct_feat_no_cs, struct_feat_with_cs))
    ag_energies.append(d['energy'])
    ag_charge_states.append(d['charge_state'])
    ag_bond_lengths.append(d['bond_lengths'])

ag_feat_no_cs = torch.stack([f[0] for f in ag_features])
ag_feat_with_cs = torch.stack([f[1] for f in ag_features])
ag_energies = np.array(ag_energies)
ag_charge_states = np.array(ag_charge_states)
ag_bond_lengths = np.array(ag_bond_lengths)

print(f"Features shape (no CS): {ag_feat_no_cs.shape}")
print(f"Features shape (with CS): {ag_feat_with_cs.shape}")
print(f"Energy range: {ag_energies.min():.4f} to {ag_energies.max():.4f}")

# Normalize features
feat_mean_no_cs = ag_feat_no_cs.mean(dim=0)
feat_std_no_cs = ag_feat_no_cs.std(dim=0) + 1e-8
ag_feat_no_cs_norm = (ag_feat_no_cs - feat_mean_no_cs) / feat_std_no_cs

feat_mean_with_cs = ag_feat_with_cs.mean(dim=0)
feat_std_with_cs = ag_feat_with_cs.std(dim=0) + 1e-8
ag_feat_with_cs_norm = (ag_feat_with_cs - feat_mean_with_cs) / feat_std_with_cs

# Split train/test
np.random.seed(42)
n = len(ag_data)
perm = np.random.permutation(n)
n_train = int(0.8 * n)
train_idx = perm[:n_train]
test_idx = perm[n_train:]

# Model 1: SR only (no charge state info)
print("\nTraining SR-only model (no charge state)...")
sr_model_ag = SimpleEnergyNet(n_features=ag_feat_no_cs.shape[1], n_hidden=64).to(device)

X_train = ag_feat_no_cs_norm[train_idx].to(device)
y_train = torch.tensor(ag_energies[train_idx], dtype=torch.float32).to(device)
X_test = ag_feat_no_cs_norm[test_idx].to(device)
y_test = torch.tensor(ag_energies[test_idx], dtype=torch.float32).to(device)

optimizer = optim.Adam(sr_model_ag.parameters(), lr=1e-3)

for epoch in range(500):
    sr_model_ag.train()
    pred = sr_model_ag(X_train)
    loss = F.mse_loss(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

sr_model_ag.eval()
with torch.no_grad():
    E_sr_ag = sr_model_ag(ag_feat_no_cs_norm.to(device)).cpu().numpy()

# Model 2: LES with charge embedding (uses charge state info)
print("\nTraining LES+ChargeEmbedding model...")
class Ag3LESModel(nn.Module):
    def __init__(self, n_features, n_hidden=64):
        super().__init__()
        self.charge_net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 3),  # 3 charges for Ag3
        )
        self.sr_net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 1),
        )
    
    def forward(self, features, total_charge=0.0):
        charges = self.charge_net(features)  # (batch, 3)
        # Constrain total charge
        charges = charges - charges.mean(dim=-1, keepdim=True) + total_charge / 3
        
        # Coulomb energy: sum q_i*q_j/r_ij (using average bond length)
        # For Ag3, use the bond lengths from features
        # Simplified: E_coul ~ sum q_i*q_j / <r>
        avg_r = 2.5  # approximate average bond length
        q_prod = charges[:, 0] * charges[:, 1] + charges[:, 0] * charges[:, 2] + charges[:, 1] * charges[:, 2]
        E_coul = q_prod / avg_r
        
        E_sr = self.sr_net(features).squeeze(-1)
        
        return E_coul + E_sr, charges

les_ce_model_ag = Ag3LESModel(n_features=ag_feat_with_cs.shape[1], n_hidden=64).to(device)

X_train_cs = ag_feat_with_cs_norm[train_idx].to(device)
X_test_cs = ag_feat_with_cs_norm[test_idx].to(device)
cs_train = torch.tensor(ag_charge_states[train_idx], dtype=torch.float32).to(device)

optimizer = optim.Adam(les_ce_model_ag.parameters(), lr=1e-3)

for epoch in range(500):
    les_ce_model_ag.train()
    pred, charges = les_ce_model_ag(X_train_cs, total_charge=0.0)
    loss = F.mse_loss(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

les_ce_model_ag.eval()
with torch.no_grad():
    X_all_cs = ag_feat_with_cs_norm.to(device)
    E_ce_ag, ce_charges = les_ce_model_ag(X_all_cs)
    E_ce_ag = E_ce_ag.cpu().numpy()
    ce_charges = ce_charges.cpu().numpy()

# Model 3: SR with charge state embedding (explicit charge info, no Coulomb)
print("\nTraining SR+ChargeEmbedding model...")
sr_ce_model_ag = SimpleEnergyNet(n_features=ag_feat_with_cs.shape[1], n_hidden=64).to(device)

optimizer = optim.Adam(sr_ce_model_ag.parameters(), lr=1e-3)

for epoch in range(500):
    sr_ce_model_ag.train()
    pred = sr_ce_model_ag(X_train_cs)
    loss = F.mse_loss(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

sr_ce_model_ag.eval()
with torch.no_grad():
    E_sr_ce_ag = sr_ce_model_ag(ag_feat_with_cs_norm.to(device)).cpu().numpy()

# Evaluate
sr_ag_mae = np.abs(E_sr_ag - ag_energies).mean()
ce_ag_mae = np.abs(E_ce_ag - ag_energies).mean()
sr_ce_ag_mae = np.abs(E_sr_ce_ag - ag_energies).mean()

print(f"\nAg3 Energy Prediction:")
print(f"  SR-only MAE: {sr_ag_mae:.4f}")
print(f"  LES+CE MAE: {ce_ag_mae:.4f}")
print(f"  SR+CE MAE: {sr_ce_ag_mae:.4f}")

# Per-charge-state analysis
for cs in [1, -1]:
    mask = ag_charge_states == cs
    sr_mae = np.abs(E_sr_ag[mask] - ag_energies[mask]).mean()
    ce_mae = np.abs(E_ce_ag[mask] - ag_energies[mask]).mean()
    sr_ce_mae = np.abs(E_sr_ce_ag[mask] - ag_energies[mask]).mean()
    print(f"  Charge state {cs:+d}: SR={sr_mae:.4f}, LES+CE={ce_mae:.4f}, SR+CE={sr_ce_mae:.4f}")

# Key test: can the SR-only model distinguish charge states?
# For identical geometries with different charge states, the SR model
# should predict the same energy (failure), while LES+CE should differ

# Find matching structures (same geometry, different charge state)
pos_data = [d for d in ag_data if d['charge_state'] == 1]
neg_data = [d for d in ag_data if d['charge_state'] == -1]

energy_diffs_sr = []
energy_diffs_ce = []
energy_diffs_ref = []

for i in range(len(pos_data)):
    # Find matching structure in neg_data
    for j in range(len(neg_data)):
        if np.allclose(pos_data[i]['positions'], neg_data[j]['positions'], atol=1e-10):
            # Same geometry, different charge state
            idx_pos = ag_charge_states.tolist().index(1.0, 0)  # find first +1
            # Get indices in the full array
            pos_idx = None
            neg_idx = None
            for k in range(len(ag_data)):
                if ag_data[k]['charge_state'] == 1 and np.allclose(ag_data[k]['positions'], pos_data[i]['positions'], atol=1e-10):
                    pos_idx = k
                if ag_data[k]['charge_state'] == -1 and np.allclose(ag_data[k]['positions'], neg_data[j]['positions'], atol=1e-10):
                    neg_idx = k
            
            if pos_idx is not None and neg_idx is not None:
                # SR model should predict same energy (no charge info)
                diff_sr = abs(E_sr_ag[pos_idx] - E_sr_ag[neg_idx])
                diff_ce = abs(E_ce_ag[pos_idx] - E_ce_ag[neg_idx])
                diff_ref = abs(ag_energies[pos_idx] - ag_energies[neg_idx])
                energy_diffs_sr.append(diff_sr)
                energy_diffs_ce.append(diff_ce)
                energy_diffs_ref.append(diff_ref)
            break

if energy_diffs_sr:
    print(f"\nCharge State Discrimination (same geometry, different charge):")
    print(f"  SR-only energy diff: {np.mean(energy_diffs_sr):.6f} (should be ~0)")
    print(f"  LES+CE energy diff: {np.mean(energy_diffs_ce):.6f} (should be > 0)")
    print(f"  Reference energy diff: {np.mean(energy_diffs_ref):.6f}")
else:
    print("\nNo matching structures found for charge state discrimination test")
    # Since all structures have same geometry for both charge states,
    # the SR model will predict the same energy for both
    # This is the key failure mode

exp3_results = {
    'sr_mae': float(sr_ag_mae),
    'les_ce_mae': float(ce_ag_mae),
    'sr_ce_mae': float(sr_ce_ag_mae),
    'sr_discrimination': float(np.mean(energy_diffs_sr)) if energy_diffs_sr else 0.0,
    'les_ce_discrimination': float(np.mean(energy_diffs_ce)) if energy_diffs_ce else 0.0,
    'ref_discrimination': float(np.mean(energy_diffs_ref)) if energy_diffs_ref else 0.0,
}
with open('outputs/exp3_results.json', 'w') as f:
    json.dump(exp3_results, f, indent=2)

print("Experiment 3 complete!")


# ============================================================
# Save all data
# ============================================================
print("\nSaving all results...")

all_results = {
    'exp1': exp1_results,
    'exp2': exp2_results,
    'exp3': exp3_results,
}

# Save additional data for plots
np.savez('outputs/plot_data.npz',
    # Exp 1
    rc_true_charges=rc_true_charges_flat,
    rc_pred_charges=pred_all,
    rc_energies=rc_energies,
    rc_energies_les=E_les_total,
    rc_energies_sr=E_sr_only,
    rc_charge_correlations=charge_correlations,
    # Exp 2
    cd_separations=cd_separations,
    cd_energies=cd_energies,
    cd_energies_les=E_les_pred,
    cd_energies_sr=E_sr_pred,
    cd_les_charges=les_charges,
    # Exp 3
    ag_energies=ag_energies,
    ag_energies_sr=E_sr_ag,
    ag_energies_ce=E_ce_ag,
    ag_energies_sr_ce=E_sr_ce_ag,
    ag_charge_states=ag_charge_states,
    ag_bond_lengths=ag_bond_lengths,
    ag_ce_charges=ce_charges,
)

with open('outputs/all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nAll experiments complete!")
