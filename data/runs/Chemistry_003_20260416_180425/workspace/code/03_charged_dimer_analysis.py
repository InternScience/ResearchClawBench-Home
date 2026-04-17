#!/usr/bin/env python3
"""
Analysis of the charged dimer dataset.
Two CH3+ and CH3- dimers at various separations.
Demonstrates the need for long-range interactions in binding energy curves.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import re

WORKDIR = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Chemistry_003_20260416_180425"

def parse_charged_dimer(filepath):
    """Parse the charged_dimer.xyz file."""
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
        
        energy_match = re.search(r'energy=([-\d.eE+]+)', comment)
        energy = float(energy_match.group(1)) if energy_match else None
        
        positions = []
        forces = []
        species = []
        for j in range(n_atoms):
            parts = lines[i].split()
            i += 1
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        
        frames.append({
            'n_atoms': n_atoms,
            'species': species,
            'positions': np.array(positions),
            'forces': np.array(forces),
            'energy': energy,
        })
    
    return frames

# Parse data
print("Parsing charged dimer data...")
frames = parse_charged_dimer(os.path.join(WORKDIR, "data/charged_dimer.xyz"))
print(f"Loaded {len(frames)} frames")

# Analyze structure: 8 atoms = 2 CH3 groups (C + 3H each)
# First 4 atoms: C + 3H (dimer 1)
# Last 4 atoms: C + 3H (dimer 2)

# Compute dimer separations and properties
separations = []
energies = []
com1_list = []
com2_list = []
internal_dist1 = []
internal_dist2 = []

for frame in frames:
    pos = frame['positions']
    sp = frame['species']
    
    # Dimer 1: atoms 0-3 (C + 3H)
    # Dimer 2: atoms 4-7 (C + 3H)
    mol1 = pos[:4]
    mol2 = pos[4:]
    
    # Center of mass (approximate, using C position as reference)
    com1 = mol1[0]  # Carbon position
    com2 = mol2[0]  # Carbon position
    
    # Also compute actual COM with masses
    mass_C = 12.0
    mass_H = 1.0
    masses1 = np.array([mass_C, mass_H, mass_H, mass_H])
    masses2 = np.array([mass_C, mass_H, mass_H, mass_H])
    com1_real = np.average(mol1, axis=0, weights=masses1)
    com2_real = np.average(mol2, axis=0, weights=masses2)
    
    sep = np.linalg.norm(com2_real - com1_real)
    separations.append(sep)
    energies.append(frame['energy'])
    
    # Internal C-H distances
    for h_idx in [1, 2, 3]:
        internal_dist1.append(np.linalg.norm(mol1[h_idx] - mol1[0]))
        internal_dist2.append(np.linalg.norm(mol2[h_idx] - mol2[0]))

separations = np.array(separations)
energies = np.array(energies)

print(f"\nSeparation range: [{separations.min():.3f}, {separations.max():.3f}] Å")
print(f"Energy range: [{energies.min():.4f}, {energies.max():.4f}]")
print(f"Internal C-H distance (mol1): {np.mean(internal_dist1):.3f} ± {np.std(internal_dist1):.3f} Å")
print(f"Internal C-H distance (mol2): {np.mean(internal_dist2):.3f} ± {np.std(internal_dist2):.3f} Å")

# Sort by separation
sort_idx = np.argsort(separations)
sep_sorted = separations[sort_idx]
e_sorted = energies[sort_idx]

# Compute Coulomb interaction between +1 and -1 charged molecules
# E_coulomb = -k_e / r (for +1 and -1 charges)
k_e = 14.3996  # eV*Å
coulomb_energy = -k_e / sep_sorted

# Estimate the "short-range" contribution (subtract Coulomb from total)
# The asymptotic energy should approach zero or a constant at large separation
e_inf = e_sorted[-1]  # approximate asymptotic energy
binding_energy = e_sorted - e_inf

print(f"\nAsymptotic energy (largest separation): {e_inf:.4f}")
print(f"Binding energy range: [{binding_energy.min():.4f}, {binding_energy.max():.4f}]")

# Typical short-range cutoff
cutoff_typical = 5.0  # Å

# Save results
results = {
    'n_frames': len(frames),
    'separation_range': [float(separations.min()), float(separations.max())],
    'energy_range': [float(energies.min()), float(energies.max())],
    'asymptotic_energy': float(e_inf),
    'mean_ch_distance': float(np.mean(internal_dist1)),
    'typical_cutoff': cutoff_typical,
}
with open(os.path.join(WORKDIR, "outputs/charged_dimer_analysis.json"), 'w') as f:
    json.dump(results, f, indent=2)

np.savez(os.path.join(WORKDIR, "outputs/charged_dimer_computed.npz"),
         separations=separations,
         energies=energies,
         sep_sorted=sep_sorted,
         e_sorted=e_sorted,
         binding_energy=binding_energy,
         coulomb_energy=coulomb_energy)

# ===== FIGURES =====

# Figure 1: Binding energy curve
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(sep_sorted, e_sorted, c='steelblue', s=50, zorder=5, edgecolors='black', linewidth=0.5, label='DFT Energy')

# Coulomb fit
r_fine = np.linspace(sep_sorted.min(), sep_sorted.max(), 200)
# Shift Coulomb to match asymptotic behavior
coulomb_fine = -k_e / r_fine + e_inf + k_e / sep_sorted[-1]
ax.plot(r_fine, coulomb_fine, 'r--', linewidth=2, label=r'$-k_e/r$ (Coulomb, shifted)', alpha=0.8)

# Mark typical cutoff
ax.axvline(cutoff_typical, color='gray', linestyle=':', linewidth=2, label=f'Typical cutoff ({cutoff_typical} Å)')

ax.set_xlabel('Dimer Separation (Å)', fontsize=14)
ax.set_ylabel('Energy (eV)', fontsize=14)
ax.set_title('Charged Dimer Binding Energy Curve', fontsize=15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/charged_dimer_binding_curve.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: charged_dimer_binding_curve.png")

# Figure 2: Binding energy with Coulomb decomposition
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Raw energy vs separation
axes[0].scatter(sep_sorted, e_sorted, c='steelblue', s=40, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('Dimer Separation (Å)', fontsize=12)
axes[0].set_ylabel('Total Energy (eV)', fontsize=12)
axes[0].set_title('Total Energy vs Separation', fontsize=13)
axes[0].grid(True, alpha=0.3)

# Right: Energy relative to asymptotic
axes[1].scatter(sep_sorted, binding_energy, c='coral', s=40, edgecolors='black', linewidth=0.5, label='Binding Energy')
# Plot -ke/r relative contribution
coulomb_relative = -k_e / sep_sorted + k_e / sep_sorted[-1]
axes[1].plot(sep_sorted, coulomb_relative, 'b-', linewidth=2, alpha=0.7, label=r'$-k_e/r$ (shifted)')
axes[1].axvline(cutoff_typical, color='gray', linestyle=':', linewidth=2, label=f'Cutoff = {cutoff_typical} Å')
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Dimer Separation (Å)', fontsize=12)
axes[1].set_ylabel('Binding Energy (eV)', fontsize=12)
axes[1].set_title('Binding Energy vs Separation', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/charged_dimer_decomposition.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: charged_dimer_decomposition.png")

# Figure 3: Force analysis
fig, ax = plt.subplots(figsize=(10, 6))
force_on_mol1 = []
force_on_mol2 = []
for idx in sort_idx:
    f = frames[idx]
    # Net force on molecule 1 (atoms 0-3)
    net_f1 = f['forces'][:4].sum(axis=0)
    net_f2 = f['forces'][4:].sum(axis=0)
    force_on_mol1.append(np.linalg.norm(net_f1))
    force_on_mol2.append(np.linalg.norm(net_f2))

force_on_mol1 = np.array(force_on_mol1)
force_on_mol2 = np.array(force_on_mol2)

ax.scatter(sep_sorted, force_on_mol1, c='steelblue', s=40, alpha=0.7, label='|F| on Mol 1', edgecolors='black', linewidth=0.5)
ax.scatter(sep_sorted, force_on_mol2, c='coral', s=40, alpha=0.7, label='|F| on Mol 2', edgecolors='black', linewidth=0.5)

# Coulomb force magnitude
coulomb_force = k_e / sep_sorted**2
ax.plot(sep_sorted, coulomb_force, 'g--', linewidth=2, label=r'$k_e/r^2$ (Coulomb force)')
ax.axvline(cutoff_typical, color='gray', linestyle=':', linewidth=2, label=f'Cutoff = {cutoff_typical} Å')

ax.set_xlabel('Dimer Separation (Å)', fontsize=12)
ax.set_ylabel('Force Magnitude (eV/Å)', fontsize=12)
ax.set_title('Net Force on Each Molecule vs Separation', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/charged_dimer_forces.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: charged_dimer_forces.png")

# Figure 4: Short-range vs long-range energy contributions
fig, ax = plt.subplots(figsize=(10, 6))

# Beyond-cutoff energy contribution
beyond_cutoff_mask = sep_sorted > cutoff_typical
within_cutoff_mask = sep_sorted <= cutoff_typical

# Coulomb energy beyond cutoff
coulomb_beyond = np.zeros_like(sep_sorted)
coulomb_beyond[beyond_cutoff_mask] = -k_e / sep_sorted[beyond_cutoff_mask]

ax.fill_between(sep_sorted, 0, binding_energy, alpha=0.3, color='steelblue', label='Total binding energy')
ax.fill_between(sep_sorted[beyond_cutoff_mask], 0, 
                coulomb_relative[beyond_cutoff_mask], 
                alpha=0.3, color='red', label='Coulomb beyond cutoff')
ax.axvline(cutoff_typical, color='gray', linestyle=':', linewidth=2, label=f'Cutoff = {cutoff_typical} Å')
ax.set_xlabel('Dimer Separation (Å)', fontsize=12)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.set_title('Short-Range vs Long-Range Energy Contributions', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/charged_dimer_sr_lr.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: charged_dimer_sr_lr.png")

print("\nAll charged dimer analysis complete.")
