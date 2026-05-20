#!/usr/bin/env python3
"""
Structural alignment analysis for protein complexes 7xg4 and 6n40.

This script:
1. Parses PDB structures
2. Analyzes chain composition and lengths
3. Loads pairwise alignment results from TMalign/USalign
4. Generates visualizations
5. Saves intermediate results to outputs/
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import PDBParser, PDBIO, Select

# Paths
DATA_DIR = 'data'
OUTPUTS_DIR = 'outputs'
IMAGES_DIR = 'report/images'

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ============================================================
# 1. Parse PDB structures
# ============================================================
parser = PDBParser(QUIET=True)

structures = {}
chain_info = {}

for pdb_id in ['7xg4', '6n40']:
    filepath = os.path.join(DATA_DIR, f'{pdb_id}.pdb')
    structure = parser.get_structure(pdb_id, filepath)
    structures[pdb_id] = structure
    
    info = []
    for model in structure:
        for chain in model:
            residues = [r for r in chain.get_residues() if r.id[0] == ' ']
            ca_atoms = []
            coords = []
            for r in residues:
                if 'CA' in r:
                    ca_atoms.append(r['CA'])
                    coords.append(r['CA'].get_coord())
            info.append({
                'pdb_id': pdb_id,
                'chain_id': chain.id,
                'num_residues': len(residues),
                'num_ca': len(ca_atoms),
                'coords': np.array(coords) if coords else np.array([]),
                'is_protein': len(ca_atoms) > 0
            })
    chain_info[pdb_id] = info

# Save chain info summary
summary = {}
for pdb_id, info_list in chain_info.items():
    summary[pdb_id] = [
        {'chain_id': c['chain_id'], 'num_residues': c['num_residues'], 
         'num_ca': c['num_ca'], 'is_protein': c['is_protein']}
        for c in info_list
    ]

with open(os.path.join(OUTPUTS_DIR, 'chain_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("Chain summary saved.")

# ============================================================
# 2. Pairwise alignment results (from TMalign runs)
# ============================================================
# These were collected manually and hardcoded here
pairwise_results = [
    {'query': '6n40:A', 'target': '7xg4:A', 'query_len': 726, 'target_len': 241, 
     'aligned_len': 151, 'rmsd': 5.91, 'seq_id': 0.093, 'tm_score_1': 0.15662, 'tm_score_2': 0.36498},
    {'query': '6n40:A', 'target': '7xg4:B', 'query_len': 726, 'target_len': 219, 
     'aligned_len': 108, 'rmsd': 5.91, 'seq_id': 0.056, 'tm_score_1': 0.11336, 'tm_score_2': 0.29124},
    {'query': '6n40:A', 'target': '7xg4:C', 'query_len': 726, 'target_len': 329, 
     'aligned_len': 172, 'rmsd': 6.41, 'seq_id': 0.087, 'tm_score_1': 0.17152, 'tm_score_2': 0.31430},
    {'query': '6n40:A', 'target': '7xg4:D', 'query_len': 726, 'target_len': 331, 
     'aligned_len': 172, 'rmsd': 6.38, 'seq_id': 0.081, 'tm_score_1': 0.17226, 'tm_score_2': 0.31532},
    {'query': '6n40:A', 'target': '7xg4:E', 'query_len': 726, 'target_len': 324, 
     'aligned_len': 169, 'rmsd': 6.38, 'seq_id': 0.071, 'tm_score_1': 0.16926, 'tm_score_2': 0.31465},
    {'query': '6n40:A', 'target': '7xg4:F', 'query_len': 726, 'target_len': 324, 
     'aligned_len': 168, 'rmsd': 6.33, 'seq_id': 0.071, 'tm_score_1': 0.16862, 'tm_score_2': 0.31348},
    {'query': '6n40:A', 'target': '7xg4:G', 'query_len': 726, 'target_len': 280, 
     'aligned_len': 165, 'rmsd': 6.46, 'seq_id': 0.061, 'tm_score_1': 0.16590, 'tm_score_2': 0.34586},
    {'query': '6n40:A', 'target': '7xg4:H', 'query_len': 726, 'target_len': 234, 
     'aligned_len': 128, 'rmsd': 5.92, 'seq_id': 0.055, 'tm_score_1': 0.13239, 'tm_score_2': 0.31361},
    {'query': '6n40:A', 'target': '7xg4:L', 'query_len': 726, 'target_len': 594, 
     'aligned_len': 243, 'rmsd': 8.68, 'seq_id': 0.074, 'tm_score_1': 0.20066, 'tm_score_2': 0.23192},
]

with open(os.path.join(OUTPUTS_DIR, 'pairwise_alignments.json'), 'w') as f:
    json.dump(pairwise_results, f, indent=2)

# Oligomeric alignment result
oligomeric_result = {
    'mode': 'USalign oligomeric (mm=1)',
    'query': '6n40:A',
    'target': '7xg4:L:A:B:C:D:E:F:G:H:I:J:K',
    'query_len': 726,
    'target_len': 3009,
    'aligned_len': 225,
    'rmsd': 8.28,
    'seq_id': 0.071,
    'tm_score_query_norm': 0.19411,
    'tm_score_target_norm': 0.06066,
    'chain_correspondence': '6n40:A mapped to all 7xg4 chains (no specific pairing)'
}

with open(os.path.join(OUTPUTS_DIR, 'oligomeric_alignment.json'), 'w') as f:
    json.dump(oligomeric_result, f, indent=2)

print("Alignment results saved.")

# ============================================================
# 3. Generate Figures
# ============================================================

# --- Figure 1: Chain composition overview ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 7xg4 chains
chains_7xg4 = [c for c in chain_info['7xg4'] if c['is_protein']]
chain_ids = [c['chain_id'] for c in chains_7xg4]
chain_lengths = [c['num_residues'] for c in chains_7xg4]
colors = plt.cm.tab10(np.linspace(0, 1, len(chain_ids)))

axes[0].bar(chain_ids, chain_lengths, color=colors, edgecolor='black')
axes[0].set_xlabel('Chain ID')
axes[0].set_ylabel('Number of Residues')
axes[0].set_title('7XG4 Chain Composition (Protein Chains)')
axes[0].set_ylim(0, max(chain_lengths) * 1.15)
for i, v in enumerate(chain_lengths):
    axes[0].text(i, v + 10, str(v), ha='center', va='bottom', fontsize=9)

# 6n40
axes[1].bar(['A'], [chain_info['6n40'][0]['num_residues']], color='steelblue', edgecolor='black')
axes[1].set_xlabel('Chain ID')
axes[1].set_ylabel('Number of Residues')
axes[1].set_title('6N40 Chain Composition (Monomer)')
axes[1].text(0, chain_info['6n40'][0]['num_residues'] + 10, 
             str(chain_info['6n40'][0]['num_residues']), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig1_chain_composition.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# --- Figure 2: Pairwise TM-scores ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

target_chains = [r['target'].split(':')[1] for r in pairwise_results]
tm_scores_1 = [r['tm_score_1'] for r in pairwise_results]  # normalized by query (6n40)
tm_scores_2 = [r['tm_score_2'] for r in pairwise_results]  # normalized by target

axes[0].bar(target_chains, tm_scores_1, color='coral', edgecolor='black')
axes[0].axhline(y=0.17, color='gray', linestyle='--', label='Random similarity threshold (0.17)')
axes[0].axhline(y=0.30, color='red', linestyle='--', label='Structural similarity threshold (0.30)')
axes[0].axhline(y=0.50, color='green', linestyle='--', label='Same fold threshold (0.50)')
axes[0].set_xlabel('7XG4 Chain')
axes[0].set_ylabel('TM-score (normalized by 6N40 length)')
axes[0].set_title('Pairwise TM-scores: 6N40 vs 7XG4 Chains')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].set_ylim(0, 0.6)
for i, v in enumerate(tm_scores_1):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

axes[1].bar(target_chains, tm_scores_2, color='skyblue', edgecolor='black')
axes[1].axhline(y=0.17, color='gray', linestyle='--', label='Random similarity (0.17)')
axes[1].axhline(y=0.30, color='red', linestyle='--', label='Structural similarity (0.30)')
axes[1].axhline(y=0.50, color='green', linestyle='--', label='Same fold (0.50)')
axes[1].set_xlabel('7XG4 Chain')
axes[1].set_ylabel('TM-score (normalized by target chain length)')
axes[1].set_title('Pairwise TM-scores: 7XG4 Chains vs 6N40')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].set_ylim(0, 0.6)
for i, v in enumerate(tm_scores_2):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig2_tm_scores.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# --- Figure 3: RMSD and Aligned Length ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

rmsds = [r['rmsd'] for r in pairwise_results]
aligned_lens = [r['aligned_len'] for r in pairwise_results]

axes[0].bar(target_chains, rmsds, color='lightgreen', edgecolor='black')
axes[0].set_xlabel('7XG4 Chain')
axes[0].set_ylabel('RMSD (Å)')
axes[0].set_title('Pairwise RMSD: 6N40 vs 7XG4 Chains')
axes[0].set_ylim(0, max(rmsds) * 1.15)
for i, v in enumerate(rmsds):
    axes[0].text(i, v + 0.15, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

axes[1].bar(target_chains, aligned_lens, color='mediumpurple', edgecolor='black')
axes[1].set_xlabel('7XG4 Chain')
axes[1].set_ylabel('Aligned Residues')
axes[1].set_title('Pairwise Alignment Length: 6N40 vs 7XG4 Chains')
axes[1].set_ylim(0, max(aligned_lens) * 1.15)
for i, v in enumerate(aligned_lens):
    axes[1].text(i, v + 5, str(v), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig3_rmsd_and_length.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# --- Figure 4: 3D Superimposition visualization (best alignment: 6n40 vs 7xg4:L) ---
# We need to apply the rotation matrix to 6n40 coordinates
# Rotation matrix from TMalign:
# t = [255.8927873969, 200.8630753399, 239.8295382249]
# u = [[0.7008336186, -0.5015290980, -0.5072482656],
#      [-0.7103842130, -0.5552172828, -0.4325367486],
#      [-0.0647032384, 0.6634774546, -0.7453932910]]

# Load coordinates
structure_6n40 = structures['6n40']
structure_7xg4 = structures['7xg4']

coords_6n40 = []
for model in structure_6n40:
    for chain in model:
        for residue in chain:
            if residue.id[0] == ' ' and 'CA' in residue:
                coords_6n40.append(residue['CA'].get_coord())
coords_6n40 = np.array(coords_6n40)

# Get 7xg4 chain L coordinates
coords_7xg4_L = []
for model in structure_7xg4:
    for chain in model:
        if chain.id == 'L':
            for residue in chain:
                if residue.id[0] == ' ' and 'CA' in residue:
                    coords_7xg4_L.append(residue['CA'].get_coord())
coords_7xg4_L = np.array(coords_7xg4_L)

# Apply rotation matrix
t = np.array([255.8927873969, 200.8630753399, 239.8295382249])
U = np.array([
    [0.7008336186, -0.5015290980, -0.5072482656],
    [-0.7103842130, -0.5552172828, -0.4325367486],
    [-0.0647032384, 0.6634774546, -0.7453932910]
])

coords_6n40_rot = (U @ coords_6n40.T).T + t

fig = plt.figure(figsize=(14, 5))

# Before superimposition
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(coords_6n40[:, 0], coords_6n40[:, 1], coords_6n40[:, 2], 
            c='blue', s=5, alpha=0.5, label='6N40 (query)')
ax1.scatter(coords_7xg4_L[:, 0], coords_7xg4_L[:, 1], coords_7xg4_L[:, 2], 
            c='red', s=5, alpha=0.5, label='7XG4:L (target)')
ax1.set_title('Before Superimposition')
ax1.set_xlabel('X (Å)')
ax1.set_ylabel('Y (Å)')
ax1.set_zlabel('Z (Å)')
ax1.legend()

# After superimposition
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(coords_6n40_rot[:, 0], coords_6n40_rot[:, 1], coords_6n40_rot[:, 2], 
            c='blue', s=5, alpha=0.5, label='6N40 (rotated)')
ax2.scatter(coords_7xg4_L[:, 0], coords_7xg4_L[:, 1], coords_7xg4_L[:, 2], 
            c='red', s=5, alpha=0.5, label='7XG4:L (target)')
ax2.set_title('After Superimposition (TM-align)')
ax2.set_xlabel('X (Å)')
ax2.set_ylabel('Y (Å)')
ax2.set_zlabel('Z (Å)')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig4_3d_superimposition.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# --- Figure 5: Alignment quality summary ---
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(target_chains))
width = 0.25

bars1 = ax.bar(x - width, tm_scores_1, width, label='TM-score (query-norm)', color='coral', edgecolor='black')
bars2 = ax.bar(x, [r/10 for r in rmsds], width, label='RMSD/10 (Å)', color='lightgreen', edgecolor='black')
bars3 = ax.bar(x + width, [a/500 for a in aligned_lens], width, label='Aligned length/500', color='mediumpurple', edgecolor='black')

ax.set_xlabel('7XG4 Chain')
ax.set_ylabel('Normalized Score')
ax.set_title('Alignment Quality Metrics: 6N40 vs 7XG4 Protein Chains')
ax.set_xticks(x)
ax.set_xticklabels(target_chains)
ax.legend()
ax.axhline(y=0.17, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=0.30, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig5_alignment_quality_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# --- Figure 6: Sequence identity vs TM-score scatter ---
fig, ax = plt.subplots(figsize=(8, 6))

seq_ids = [r['seq_id'] for r in pairwise_results]
ax.scatter(seq_ids, tm_scores_1, s=150, c='steelblue', edgecolors='black', alpha=0.7)

for i, chain in enumerate(target_chains):
    ax.annotate(f'7XG4:{chain}', (seq_ids[i], tm_scores_1[i]), 
                textcoords="offset points", xytext=(5, 5), fontsize=9)

ax.axhline(y=0.17, color='gray', linestyle='--', label='Random similarity (0.17)')
ax.axhline(y=0.30, color='red', linestyle='--', label='Structural similarity (0.30)')
ax.set_xlabel('Sequence Identity (aligned region)')
ax.set_ylabel('TM-score (normalized by query length)')
ax.set_title('Sequence Identity vs Structural Similarity')
ax.legend()
ax.set_xlim(0, max(seq_ids) * 1.3)
ax.set_ylim(0, 0.35)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig6_seqid_vs_tmscore.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ============================================================
# 4. Save superimposition data
# ============================================================
np.savez(os.path.join(OUTPUTS_DIR, 'superimposition_coords.npz'),
         coords_6n40=coords_6n40,
         coords_6n40_rot=coords_6n40_rot,
         coords_7xg4_L=coords_7xg4_L,
         rotation_matrix=U,
         translation_vector=t)

print("Superimposition data saved.")
print("\nAll analysis complete!")
