#!/usr/bin/env python3
"""
Comprehensive analysis of the random charges dataset.
Computes Coulomb energies and forces using direct summation (no PBC),
and analyzes the relationship between charge configurations and energies.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import re

WORKDIR = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Chemistry_003_20260416_180425"

def parse_random_charges(filepath):
    """Parse the random_charges.xyz file."""
    frames = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        n_atoms = int(line)
        comment = lines[i+1].strip()
        i += 2
        
        # Extract true charges
        tq_match = re.search(r'true_charges="([^"]+)"', comment)
        true_charges = np.array([float(x) for x in tq_match.group(1).split()])
        
        positions = []
        for j in range(n_atoms):
            parts = lines[i].split()
            i += 1
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        frames.append({
            'positions': np.array(positions),
            'charges': true_charges,
            'n_atoms': n_atoms
        })
    
    return frames

def compute_coulomb_energy_forces(positions, charges, sigma_lj=1.0, epsilon_lj=0.01):
    """
    Compute Coulomb energy and forces for a non-periodic system.
    Also includes repulsive LJ term to prevent collapse.
    
    E_coulomb = sum_{i<j} q_i * q_j / r_ij  (in appropriate units)
    E_LJ_rep = sum_{i<j} 4*epsilon * (sigma/r_ij)^12  (repulsive only)
    """
    n = len(positions)
    energy_coulomb = 0.0
    energy_lj = 0.0
    forces = np.zeros_like(positions)
    
    # Conversion factor: e^2 / (4*pi*eps0) in eV*Å
    # k_e = 14.3996 eV*Å (for charges in units of e)
    k_e = 14.3996
    
    for i in range(n):
        for j in range(i+1, n):
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)
            rhat = rij / r
            
            # Coulomb
            e_c = k_e * charges[i] * charges[j] / r
            energy_coulomb += e_c
            f_c = k_e * charges[i] * charges[j] / (r**2) * rhat
            forces[i] -= f_c  # force on i from j
            forces[j] += f_c
            
            # Repulsive LJ
            sr6 = (sigma_lj / r)**6
            sr12 = sr6**2
            e_lj = 4 * epsilon_lj * sr12
            energy_lj += e_lj
            f_lj = 4 * epsilon_lj * 12 * sr12 / r * rhat
            forces[i] -= f_lj
            forces[j] += f_lj
    
    return energy_coulomb, energy_lj, forces

# Parse data
print("Parsing random charges data...")
frames = parse_random_charges(os.path.join(WORKDIR, "data/random_charges.xyz"))
print(f"Loaded {len(frames)} frames")

# Compute energies for all frames (this is O(N^2) per frame)
print("Computing Coulomb energies and forces...")
coulomb_energies = []
lj_energies = []
total_energies = []
all_forces = []

for idx, frame in enumerate(frames):
    e_c, e_lj, forces = compute_coulomb_energy_forces(
        frame['positions'], frame['charges']
    )
    coulomb_energies.append(e_c)
    lj_energies.append(e_lj)
    total_energies.append(e_c + e_lj)
    all_forces.append(forces)
    if (idx + 1) % 10 == 0:
        print(f"  Frame {idx+1}/{len(frames)}: E_coulomb={e_c:.4f}, E_LJ={e_lj:.4f}, E_total={e_c+e_lj:.4f}")

coulomb_energies = np.array(coulomb_energies)
lj_energies = np.array(lj_energies)
total_energies = np.array(total_energies)

print(f"\nCoulomb energy: mean={coulomb_energies.mean():.4f}, std={coulomb_energies.std():.4f}")
print(f"LJ energy: mean={lj_energies.mean():.4f}, std={lj_energies.std():.4f}")
print(f"Total energy: mean={total_energies.mean():.4f}, std={total_energies.std():.4f}")

# Analyze charge distributions
print("\nCharge analysis:")
charges = frames[0]['charges']
print(f"  Total charge: {charges.sum()}")
print(f"  Positive charges: {np.sum(charges > 0)}")
print(f"  Negative charges: {np.sum(charges < 0)}")

# Analyze pairwise distances
print("\nPairwise distance analysis:")
min_dists = []
for frame in frames:
    pos = frame['positions']
    n = len(pos)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(pos[j] - pos[i]))
    min_dists.append(min(dists))
print(f"  Min pairwise distance across frames: {min(min_dists):.4f} Å")
print(f"  Mean min pairwise distance: {np.mean(min_dists):.4f} Å")

# Save results
results = {
    'n_frames': len(frames),
    'n_atoms': frames[0]['n_atoms'],
    'coulomb_energy_mean': float(coulomb_energies.mean()),
    'coulomb_energy_std': float(coulomb_energies.std()),
    'lj_energy_mean': float(lj_energies.mean()),
    'lj_energy_std': float(lj_energies.std()),
    'total_energy_mean': float(total_energies.mean()),
    'total_energy_std': float(total_energies.std()),
    'min_pairwise_distance': float(min(min_dists)),
}
with open(os.path.join(WORKDIR, "outputs/random_charges_analysis.json"), 'w') as f:
    json.dump(results, f, indent=2)

# Save computed energies and forces for later use
np.savez(os.path.join(WORKDIR, "outputs/random_charges_computed.npz"),
         coulomb_energies=coulomb_energies,
         lj_energies=lj_energies,
         total_energies=total_energies)

print("\nResults saved.")

# ===== FIGURES =====

# Figure 1: Energy distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(coulomb_energies, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Coulomb Energy (eV)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Coulomb Energy Distribution', fontsize=13)
axes[0].axvline(coulomb_energies.mean(), color='red', linestyle='--', label=f'Mean={coulomb_energies.mean():.1f}')
axes[0].legend()

axes[1].hist(lj_energies, bins=20, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('LJ Repulsive Energy (eV)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('LJ Repulsive Energy Distribution', fontsize=13)

axes[2].hist(total_energies, bins=20, color='seagreen', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Total Energy (eV)', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('Total Energy Distribution', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/random_charges_energy_dist.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: random_charges_energy_dist.png")

# Figure 2: Charge configuration visualization (first frame)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
pos = frames[0]['positions']
q = frames[0]['charges']
pos_mask = q > 0
neg_mask = q < 0
ax.scatter(pos[pos_mask, 0], pos[pos_mask, 1], c='red', s=30, alpha=0.7, label='+1e', edgecolors='darkred')
ax.scatter(pos[neg_mask, 0], pos[neg_mask, 1], c='blue', s=30, alpha=0.7, label='-1e', edgecolors='darkblue')
ax.set_xlabel('x (Å)', fontsize=12)
ax.set_ylabel('y (Å)', fontsize=12)
ax.set_title('Random Charges Configuration (Frame 1, xy projection)', fontsize=13)
ax.legend(fontsize=11)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/random_charges_config.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: random_charges_config.png")

# Figure 3: Force magnitude distribution
force_mags = []
for forces in all_forces:
    force_mags.extend(np.linalg.norm(forces, axis=1))
force_mags = np.array(force_mags)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(force_mags, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
ax.set_xlabel('Force Magnitude (eV/Å)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Atomic Force Magnitudes', fontsize=13)
ax.axvline(force_mags.mean(), color='red', linestyle='--', label=f'Mean={force_mags.mean():.2f} eV/Å')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/random_charges_force_dist.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: random_charges_force_dist.png")

# Figure 4: Coulomb vs LJ energy correlation
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(coulomb_energies, lj_energies, c='teal', s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Coulomb Energy (eV)', fontsize=12)
ax.set_ylabel('LJ Repulsive Energy (eV)', fontsize=12)
ax.set_title('Coulomb vs LJ Energy', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/random_charges_energy_correlation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: random_charges_energy_correlation.png")

print("\nAll random charges analysis complete.")
