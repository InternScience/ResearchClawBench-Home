#!/usr/bin/env python3
"""Generate all figures for the BioDiffuseNet report."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path

base_dir = Path(__file__).parent.parent
output_dir = base_dir / "outputs"
img_dir = base_dir / "report" / "images"
img_dir.mkdir(parents=True, exist_ok=True)

# Load data
protein_coords = np.load(output_dir / 'protein_coords.npy')
pred_protein_coords = np.load(output_dir / 'pred_protein_coords.npy')
ligand_coords = np.load(output_dir / 'ligand_coords.npy')
pred_ligand_coords = np.load(output_dir / 'pred_ligand_coords.npy')
protein_dist_matrix = np.load(output_dir / 'protein_dist_matrix.npy')
interface_residues = np.load(output_dir / 'interface_residues.npy')

with open(output_dir / 'analysis_results.json') as f:
    results = json.load(f)

# --- Figure 1: 3D Structure Overlay ---
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(protein_coords[:,0], protein_coords[:,1], protein_coords[:,2], 
            c='green', s=20, alpha=0.7, label='Reference')
ax1.scatter(pred_protein_coords[:,0], pred_protein_coords[:,1], pred_protein_coords[:,2],
            c='blue', s=20, alpha=0.5, label='Predicted')
# Draw CA backbone
ax1.plot(protein_coords[:,0], protein_coords[:,1], protein_coords[:,2], 'g-', alpha=0.3, lw=0.5)
ax1.plot(pred_protein_coords[:,0], pred_protein_coords[:,1], pred_protein_coords[:,2], 'b-', alpha=0.3, lw=0.5)
ax1.set_xlabel('X (Å)')
ax1.set_ylabel('Y (Å)')
ax1.set_zlabel('Z (Å)')
ax1.set_title('FKBP12 Protein Backbone (CA atoms)')
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(ligand_coords[:,0], ligand_coords[:,1], ligand_coords[:,2],
            c='red', s=15, alpha=0.7, label='Reference FK506')
ax2.scatter(pred_ligand_coords[:,0], pred_ligand_coords[:,1], pred_ligand_coords[:,2],
            c='orange', s=15, alpha=0.5, label='Predicted FK506')
ax2.set_xlabel('X (Å)')
ax2.set_ylabel('Y (Å)')
ax2.set_zlabel('Z (Å)')
ax2.set_title('FK506 Ligand Conformation')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(img_dir / 'figure1_3d_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# --- Figure 2: Distance Matrix and Interface ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im = axes[0].imshow(protein_dist_matrix, cmap='viridis', origin='lower')
axes[0].set_xlabel('Residue Index')
axes[0].set_ylabel('Residue Index')
axes[0].set_title('Protein CA Distance Matrix')
plt.colorbar(im, ax=axes[0], label='Distance (Å)')

# Interface residue positions
interface_mask = np.zeros(len(protein_coords))
interface_mask[interface_residues.astype(int)] = 1
axes[1].bar(range(len(interface_mask)), interface_mask, color='steelblue', width=1.0)
axes[1].set_xlabel('Residue Index')
axes[1].set_ylabel('Interface (1=yes)')
axes[1].set_title(f'Binding Interface Residues ({len(interface_residues)} of {len(protein_coords)})')

plt.tight_layout()
plt.savefig(img_dir / 'figure2_distance_interface.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# --- Figure 3: Evaluation Metrics ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Per-residue RMSD
per_res_rmsd = np.sqrt(np.sum((protein_coords - pred_protein_coords)**2, axis=1))
axes[0].plot(per_res_rmsd, 'b-', alpha=0.7)
axes[0].axhline(y=np.mean(per_res_rmsd), color='r', linestyle='--', label=f'Mean={np.mean(per_res_rmsd):.2f} Å')
axes[0].set_xlabel('Residue Index')
axes[0].set_ylabel('RMSD (Å)')
axes[0].set_title('Per-Residue Backbone RMSD')
axes[0].legend()

# Per-atom ligand RMSD
per_atom_rmsd = np.sqrt(np.sum((ligand_coords - pred_ligand_coords)**2, axis=1))
axes[1].plot(per_atom_rmsd, 'r-', alpha=0.7)
axes[1].axhline(y=np.mean(per_atom_rmsd), color='b', linestyle='--', label=f'Mean={np.mean(per_atom_rmsd):.2f} Å')
axes[1].set_xlabel('Atom Index')
axes[1].set_ylabel('RMSD (Å)')
axes[1].set_title('Per-Atom Ligand RMSD')
axes[1].legend()

# Metrics summary bar chart
metrics = results['evaluation_metrics']
names = ['Protein\nRMSD (Å)', 'Ligand\nRMSD (Å)', 'Contact\nAccuracy']
values = [metrics['protein_rmsd_angstrom'], metrics['ligand_rmsd_angstrom'], metrics['contact_accuracy']]
colors = ['#2196F3', '#F44336', '#4CAF50']
bars = axes[2].bar(names, values, color=colors, edgecolor='black')
axes[2].set_title('Evaluation Metrics Summary')
for bar, val in zip(bars, values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(img_dir / 'figure3_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# --- Figure 4: Diffusion Process Illustration ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
np.random.seed(42)

timesteps = [0, 250, 500, 999]
for idx, t in enumerate(timesteps):
    alpha_cumprod = np.cos((t/1000 * 0.999 + 0.001) * np.pi / 2)**2
    noise_level = np.sqrt(1 - alpha_cumprod)
    signal_level = np.sqrt(alpha_cumprod)
    
    noisy_coords = signal_level * protein_coords[:30] + noise_level * np.random.randn(30, 3) * 10
    
    axes[idx].scatter(noisy_coords[:,0], noisy_coords[:,1], c=np.arange(30), cmap='coolwarm', s=30)
    axes[idx].set_title(f't={t} (noise={noise_level:.2f})')
    axes[idx].set_xlim(-40, 40)
    axes[idx].set_ylim(-40, 40)
    axes[idx].set_aspect('equal')

plt.suptitle('Forward Diffusion Process on Protein Coordinates', fontsize=13)
plt.tight_layout()
plt.savefig(img_dir / 'figure4_diffusion_process.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# --- Figure 5: Architecture Diagram ---
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Input boxes
inputs = [('Protein\nSequence', 1, 6, '#2196F3'), ('Nucleic Acid\nSequence', 3.5, 6, '#4CAF50'), 
          ('Small Molecule\nStructure', 6, 6, '#F44336')]
for label, x, y, color in inputs:
    rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

# Encoder boxes
encoders = [('Protein\nEncoder', 1, 4.5, '#2196F3'), ('Nucleic Acid\nEncoder', 3.5, 4.5, '#4CAF50'),
            ('Molecule\nEncoder', 6, 4.5, '#F44336')]
for label, x, y, color in encoders:
    rect = plt.Rectangle((x-0.7, y-0.35), 1.4, 0.7, facecolor=color, alpha=0.5, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

# Arrows from inputs to encoders
for x in [1, 3.5, 6]:
    ax.annotate('', xy=(x, 4.85), xytext=(x, 5.6), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Cross-attention block
rect = plt.Rectangle((2.5, 3.2), 3, 0.8, facecolor='#FF9800', alpha=0.4, edgecolor='#FF9800', linewidth=2)
ax.add_patch(rect)
ax.text(4, 3.6, 'Cross-Attention\nInteraction Module', ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows to cross-attention
for x in [1, 3.5, 6]:
    ax.annotate('', xy=(4, 4.0), xytext=(x, 4.15), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Diffusion block
rect = plt.Rectangle((2.5, 1.8), 3, 0.9, facecolor='#9C27B0', alpha=0.4, edgecolor='#9C27B0', linewidth=2)
ax.add_patch(rect)
ax.text(4, 2.25, 'SE(3)-Equivariant\nDenoising Network\n(Diffusion)', ha='center', va='center', fontsize=9, fontweight='bold')

ax.annotate('', xy=(4, 2.7), xytext=(4, 3.2), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Output
rect = plt.Rectangle((2.5, 0.5), 3, 0.8, facecolor='#00BCD4', alpha=0.4, edgecolor='#00BCD4', linewidth=2)
ax.add_patch(rect)
ax.text(4, 0.9, '3D Complex\nStructure', ha='center', va='center', fontsize=10, fontweight='bold')

ax.annotate('', xy=(4, 1.3), xytext=(4, 1.8), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Noise input
rect = plt.Rectangle((8, 1.8), 2, 0.9, facecolor='#607D8B', alpha=0.4, edgecolor='#607D8B', linewidth=2)
ax.add_patch(rect)
ax.text(9, 2.25, 'Gaussian\nNoise x_T', ha='center', va='center', fontsize=9, fontweight='bold')
ax.annotate('', xy=(4.5, 2.25), xytext=(8, 2.25), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='dashed'))

# Time embedding
rect = plt.Rectangle((8, 3.2), 2, 0.8, facecolor='#795548', alpha=0.4, edgecolor='#795548', linewidth=2)
ax.add_patch(rect)
ax.text(9, 3.6, 'Time\nEmbedding', ha='center', va='center', fontsize=9, fontweight='bold')
ax.annotate('', xy=(5.5, 3.6), xytext=(8, 3.6), arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='dashed'))

ax.set_title('BioDiffuseNet Architecture', fontsize=14, fontweight='bold', pad=20)
plt.savefig(img_dir / 'figure5_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# --- Figure 6: Complex Structure with Interface ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Protein backbone
non_interface = [i for i in range(len(protein_coords)) if i not in interface_residues]
ax.scatter(protein_coords[non_interface,0], protein_coords[non_interface,1], protein_coords[non_interface,2],
           c='lightblue', s=30, alpha=0.6, label='Non-interface residues')
ax.scatter(protein_coords[interface_residues.astype(int),0], 
           protein_coords[interface_residues.astype(int),1],
           protein_coords[interface_residues.astype(int),2],
           c='blue', s=60, alpha=0.9, label='Interface residues')
ax.plot(protein_coords[:,0], protein_coords[:,1], protein_coords[:,2], 'b-', alpha=0.2, lw=0.5)

# Ligand
ax.scatter(ligand_coords[:,0], ligand_coords[:,1], ligand_coords[:,2],
           c='red', s=40, alpha=0.8, label='FK506 ligand')

ax.set_xlabel('X (Å)')
ax.set_ylabel('Y (Å)')
ax.set_zlabel('Z (Å)')
ax.set_title('FKBP12–FK506 Complex with Binding Interface')
ax.legend(fontsize=9, loc='upper left')

plt.savefig(img_dir / 'figure6_complex_interface.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

print("\nAll figures generated successfully!")
