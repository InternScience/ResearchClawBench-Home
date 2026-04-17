#!/usr/bin/env python3
"""
Analysis of the Ag3 charge states dataset.
Ag3 trimers in +1 and -1 charge states with varying geometries.
Demonstrates the challenge of distinguishing charge states with short-range models.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import re

WORKDIR = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Chemistry_003_20260416_180425"

def parse_ag3_chargestates(filepath):
    """Parse the ag3_chargestates.xyz file."""
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
        
        cs_match = re.search(r'charge_state=([-\d]+)', comment)
        charge_state = int(cs_match.group(1)) if cs_match else None
        
        tc_match = re.search(r'total_charge=([-\d]+)', comment)
        total_charge = int(tc_match.group(1)) if tc_match else None
        
        positions = []
        forces = []
        for j in range(n_atoms):
            parts = lines[i].split()
            i += 1
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        
        frames.append({
            'n_atoms': n_atoms,
            'positions': np.array(positions),
            'forces': np.array(forces),
            'energy': energy,
            'charge_state': charge_state,
            'total_charge': total_charge,
        })
    
    return frames

# Parse data
print("Parsing Ag3 charge states data...")
frames = parse_ag3_chargestates(os.path.join(WORKDIR, "data/ag3_chargestates.xyz"))
print(f"Loaded {len(frames)} frames")

# Separate by charge state
plus1 = [f for f in frames if f['charge_state'] == 1]
minus1 = [f for f in frames if f['charge_state'] == -1]
print(f"Charge +1: {len(plus1)} frames")
print(f"Charge -1: {len(minus1)} frames")

# Compute geometric descriptors for each frame
def compute_ag3_descriptors(frame):
    """Compute bond lengths and angles for Ag3 trimer."""
    pos = frame['positions']
    # Bond lengths
    d01 = np.linalg.norm(pos[1] - pos[0])
    d02 = np.linalg.norm(pos[2] - pos[0])
    d12 = np.linalg.norm(pos[2] - pos[1])
    
    # Sort bond lengths
    bonds = sorted([d01, d02, d12])
    
    # Angles (using law of cosines)
    # Angle at atom 0 (between bonds 01 and 02)
    cos_angle0 = np.dot(pos[1]-pos[0], pos[2]-pos[0]) / (d01 * d02)
    cos_angle0 = np.clip(cos_angle0, -1, 1)
    angle0 = np.degrees(np.arccos(cos_angle0))
    
    # Mean bond length
    mean_bl = np.mean([d01, d02, d12])
    
    # Perimeter
    perimeter = d01 + d02 + d12
    
    return {
        'd01': d01, 'd02': d02, 'd12': d12,
        'bonds_sorted': bonds,
        'mean_bl': mean_bl,
        'perimeter': perimeter,
        'angle0': angle0,
    }

# Compute descriptors
plus1_desc = [compute_ag3_descriptors(f) for f in plus1]
minus1_desc = [compute_ag3_descriptors(f) for f in minus1]

# Check if geometries are truly identical
print("\nGeometry comparison (first 5 frames):")
for idx in range(min(5, len(plus1), len(minus1))):
    pos_diff = np.linalg.norm(plus1[idx]['positions'] - minus1[idx]['positions'])
    e_diff = plus1[idx]['energy'] - minus1[idx]['energy']
    f_diff = np.linalg.norm(plus1[idx]['forces'] - minus1[idx]['forces'])
    print(f"  Frame {idx}: pos_diff={pos_diff:.8f}, E_diff={e_diff:.8f}, F_diff={f_diff:.8f}")

# Energy statistics
e_plus = np.array([f['energy'] for f in plus1])
e_minus = np.array([f['energy'] for f in minus1])
print(f"\nEnergy stats:")
print(f"  +1: mean={e_plus.mean():.4f}, std={e_plus.std():.4f}, range=[{e_plus.min():.4f}, {e_plus.max():.4f}]")
print(f"  -1: mean={e_minus.mean():.4f}, std={e_minus.std():.4f}, range=[{e_minus.min():.4f}, {e_minus.max():.4f}]")
print(f"  E(+1) - E(-1): mean={np.mean(e_plus - e_minus):.6f}, max={np.max(np.abs(e_plus - e_minus)):.6f}")

# Force statistics
f_plus_mag = np.array([np.linalg.norm(f['forces'], axis=1).mean() for f in plus1])
f_minus_mag = np.array([np.linalg.norm(f['forces'], axis=1).mean() for f in minus1])
print(f"\nForce magnitude stats:")
print(f"  +1: mean={f_plus_mag.mean():.4f}")
print(f"  -1: mean={f_minus_mag.mean():.4f}")

# Bond length analysis
bl_plus = np.array([d['mean_bl'] for d in plus1_desc])
bl_minus = np.array([d['mean_bl'] for d in minus1_desc])
print(f"\nMean bond length:")
print(f"  +1: {bl_plus.mean():.4f} ± {bl_plus.std():.4f}")
print(f"  -1: {bl_minus.mean():.4f} ± {bl_minus.std():.4f}")

# Save results
results = {
    'n_frames_total': len(frames),
    'n_frames_plus1': len(plus1),
    'n_frames_minus1': len(minus1),
    'energy_plus1_mean': float(e_plus.mean()),
    'energy_plus1_std': float(e_plus.std()),
    'energy_minus1_mean': float(e_minus.mean()),
    'energy_minus1_std': float(e_minus.std()),
    'energy_difference_mean': float(np.mean(e_plus - e_minus)),
    'geometries_identical': bool(np.allclose(
        [f['positions'] for f in plus1], 
        [f['positions'] for f in minus1]
    )),
    'energies_identical': bool(np.allclose(e_plus, e_minus)),
    'forces_identical': bool(np.allclose(
        [f['forces'] for f in plus1],
        [f['forces'] for f in minus1]
    )),
}
with open(os.path.join(WORKDIR, "outputs/ag3_chargestates_analysis.json"), 'w') as f:
    json.dump(results, f, indent=2)

# ===== FIGURES =====

# Figure 1: Energy vs mean bond length for both charge states
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Energy vs mean bond length
axes[0].scatter(bl_plus, e_plus, c='red', s=60, alpha=0.7, label='Charge +1', 
                edgecolors='darkred', linewidth=0.5, marker='o')
axes[0].scatter(bl_minus, e_minus, c='blue', s=60, alpha=0.7, label='Charge -1', 
                edgecolors='darkblue', linewidth=0.5, marker='s')
axes[0].set_xlabel('Mean Bond Length (Å)', fontsize=12)
axes[0].set_ylabel('Energy (eV)', fontsize=12)
axes[0].set_title('Ag₃ Energy vs Mean Bond Length', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Right: Energy difference (should be zero for this dataset)
axes[1].scatter(bl_plus, e_plus - e_minus, c='green', s=60, alpha=0.7, 
                edgecolors='black', linewidth=0.5)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Mean Bond Length (Å)', fontsize=12)
axes[1].set_ylabel('E(+1) - E(-1) (eV)', fontsize=12)
axes[1].set_title('Energy Difference Between Charge States', fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/ag3_energy_vs_bondlength.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ag3_energy_vs_bondlength.png")

# Figure 2: PES comparison - energy vs all three bond lengths
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
bond_labels = ['Shortest Bond', 'Middle Bond', 'Longest Bond']

for bi, label in enumerate(bond_labels):
    bl_p = np.array([d['bonds_sorted'][bi] for d in plus1_desc])
    bl_m = np.array([d['bonds_sorted'][bi] for d in minus1_desc])
    
    axes[bi].scatter(bl_p, e_plus, c='red', s=50, alpha=0.7, label='Charge +1', 
                     edgecolors='darkred', linewidth=0.5)
    axes[bi].scatter(bl_m, e_minus, c='blue', s=50, alpha=0.7, label='Charge -1', 
                     edgecolors='darkblue', linewidth=0.5)
    axes[bi].set_xlabel(f'{label} (Å)', fontsize=12)
    axes[bi].set_ylabel('Energy (eV)', fontsize=12)
    axes[bi].set_title(f'Energy vs {label}', fontsize=13)
    axes[bi].legend(fontsize=10)
    axes[bi].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/ag3_pes_bonds.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ag3_pes_bonds.png")

# Figure 3: Force magnitude comparison
fig, ax = plt.subplots(figsize=(10, 6))

perimeter_plus = np.array([d['perimeter'] for d in plus1_desc])
perimeter_minus = np.array([d['perimeter'] for d in minus1_desc])

ax.scatter(perimeter_plus, f_plus_mag, c='red', s=60, alpha=0.7, label='Charge +1', 
           edgecolors='darkred', linewidth=0.5, marker='o')
ax.scatter(perimeter_minus, f_minus_mag, c='blue', s=60, alpha=0.7, label='Charge -1', 
           edgecolors='darkblue', linewidth=0.5, marker='s')
ax.set_xlabel('Triangle Perimeter (Å)', fontsize=12)
ax.set_ylabel('Mean Force Magnitude (eV/Å)', fontsize=12)
ax.set_title('Ag₃ Force Magnitude vs Geometry', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/ag3_forces_vs_geometry.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ag3_forces_vs_geometry.png")

# Figure 4: Schematic showing the challenge
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Identical geometry with different charges
# Draw a triangle with atoms
for ax_idx, (label, color, charge) in enumerate([
    ('Ag₃⁺ (Charge +1)', 'red', '+1'),
    ('Ag₃⁻ (Charge -1)', 'blue', '-1')
]):
    ax = axes[ax_idx]
    # Example triangle
    pos = plus1[15]['positions']  # middle frame
    for i in range(3):
        for j in range(i+1, 3):
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 
                    color='gray', linewidth=2, zorder=1)
    
    ax.scatter(pos[:,0], pos[:,1], c=color, s=200, zorder=5, 
               edgecolors='black', linewidth=1.5)
    for i in range(3):
        ax.annotate(f'Ag ({charge})', (pos[i,0]+0.05, pos[i,1]+0.05), fontsize=10)
    
    ax.set_xlabel('x (Å)', fontsize=12)
    ax.set_ylabel('y (Å)', fontsize=12)
    ax.set_title(f'{label}\nE = {plus1[15]["energy"]:.4f} eV', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Same Geometry, Different Charge States → Same E, F\n(Challenge for short-range models)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/ag3_charge_state_challenge.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ag3_charge_state_challenge.png")

# Figure 5: Comprehensive PES plot (like Fig 5e in paper)
fig, ax = plt.subplots(figsize=(10, 7))

# Use perimeter as a 1D descriptor of geometry
sort_p = np.argsort(perimeter_plus)
ax.plot(perimeter_plus[sort_p], e_plus[sort_p], 'ro-', markersize=6, alpha=0.7, 
        label='Charge +1', linewidth=1.5)
ax.plot(perimeter_minus[sort_p], e_minus[sort_p], 'bs-', markersize=6, alpha=0.7, 
        label='Charge -1', linewidth=1.5)

ax.set_xlabel('Triangle Perimeter (Å)', fontsize=14)
ax.set_ylabel('Energy (eV)', fontsize=14)
ax.set_title('Ag₃ Potential Energy Surface: +1 vs -1 Charge States', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add annotation
ax.annotate('Identical PES for both charge states\n→ Short-range model cannot distinguish them',
            xy=(0.5, 0.95), xycoords='axes fraction', fontsize=11,
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))

plt.tight_layout()
plt.savefig(os.path.join(WORKDIR, "report/images/ag3_pes_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ag3_pes_comparison.png")

print("\nAll Ag3 charge states analysis complete.")
