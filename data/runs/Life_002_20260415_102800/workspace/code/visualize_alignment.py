#!/usr/bin/env python3
"""
Visualization of structural alignment results for 7xg4 vs 6n40
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from Bio import PDB

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, "data")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report", "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ── Load results ───────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "pairwise_chain_alignments.json")) as f:
    pairwise = json.load(f)
with open(os.path.join(OUTPUT_DIR, "complex_alignments.json")) as f:
    complex_res = json.load(f)
with open(os.path.join(OUTPUT_DIR, "best_chain_correspondence.json")) as f:
    best_matches = json.load(f)
with open(os.path.join(OUTPUT_DIR, "chain_summary.json")) as f:
    chain_summary = json.load(f)

# ── Figure 1: TM-score heatmap for pairwise chain alignments ──────────
query_chains = sorted(set(k.split('_vs_')[0] for k in pairwise.keys()))
target_chains = sorted(set(k.split('_vs_')[1] for k in pairwise.keys()))

# TM-score normalized by target
tm_target_matrix = np.zeros((len(query_chains), len(target_chains)))
rmsd_matrix = np.zeros((len(query_chains), len(target_chains)))
for i, qc in enumerate(query_chains):
    for j, tc in enumerate(target_chains):
        key = f"{qc}_vs_{tc}"
        if key in pairwise and 'error' not in pairwise[key]:
            tm_target_matrix[i, j] = pairwise[key]['tm_norm_target']
            rmsd_matrix[i, j] = pairwise[key]['rmsd']
        else:
            tm_target_matrix[i, j] = 0
            rmsd_matrix[i, j] = 0

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# TM-score heatmap
ax1 = axes[0]
sns.heatmap(tm_target_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=target_chains, yticklabels=query_chains,
            ax=ax1, vmin=0, vmax=0.3, linewidths=0.5)
ax1.set_title('TM-score (normalized by target)', fontsize=13, fontweight='bold')
ax1.set_xlabel('6n40 Chain', fontsize=11)
ax1.set_ylabel('7xg4 Chain', fontsize=11)

# RMSD heatmap
ax2 = axes[1]
sns.heatmap(rmsd_matrix, annot=True, fmt='.1f', cmap='YlGnBu_r',
            xticklabels=target_chains, yticklabels=query_chains,
            ax=ax2, linewidths=0.5)
ax2.set_title('RMSD (Å)', fontsize=13, fontweight='bold')
ax2.set_xlabel('6n40 Chain', fontsize=11)
ax2.set_ylabel('7xg4 Chain', fontsize=11)

plt.suptitle('Pairwise Chain-Level Structural Alignment: 7xg4 vs 6n40',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig1_pairwise_tm_rmsd_heatmap.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ── Figure 2: Bar chart of best TM-scores per query chain ─────────────
fig, ax = plt.subplots(figsize=(12, 6))

chains = sorted(best_matches.keys())
tm_scores = [best_matches[c]['tm_norm_target'] for c in chains]
rmsd_vals = [best_matches[c]['rmsd'] for c in chains]

# Color by chain type
protein_chains = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']
nucleic_chains = ['I', 'J', 'K']
colors = []
for c in chains:
    if c in nucleic_chains:
        colors.append('#e74c3c')  # red for nucleic acid
    else:
        colors.append('#3498db')  # blue for protein

x = np.arange(len(chains))
bars = ax.bar(x, tm_scores, color=colors, edgecolor='black', linewidth=0.5, width=0.6)

# Add TM-score values on bars
for i, (bar, tm) in enumerate(zip(bars, tm_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{tm:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add RMSD as text below
for i, (c, rmsd) in enumerate(zip(chains, rmsd_vals)):
    ax.text(i, -0.015, f'RMSD={rmsd:.1f}Å', ha='center', va='top', fontsize=8, color='gray')

ax.set_xticks(x)
ax.set_xticklabels([f'{c}\n→ 6n40_A' for c in chains], fontsize=9)
ax.set_ylabel('TM-score (normalized by target)', fontsize=12)
ax.set_title('Best Chain-Level TM-scores: 7xg4 Chains vs 6n40 Chain A',
             fontsize=14, fontweight='bold')
ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='TM=0.5 (significant)')
ax.axhline(y=0.17, color='orange', linestyle='--', alpha=0.5, label='TM=0.17 (random)')

# Legend
prot_patch = mpatches.Patch(color='#3498db', label='Protein chains')
na_patch = mpatches.Patch(color='#e74c3c', label='Nucleic acid chains')
ax.legend(handles=[prot_patch, na_patch,
                   plt.Line2D([0], [0], color='green', linestyle='--', alpha=0.5),
                   plt.Line2D([0], [0], color='orange', linestyle='--', alpha=0.5)],
          labels=['Protein chains', 'Nucleic acid chains',
                  'TM=0.5 (significant)', 'TM=0.17 (random)'],
          loc='upper right', fontsize=9)

ax.set_ylim(-0.03, max(tm_scores) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig2_best_tm_scores_barplot.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ── Figure 3: Complex-level alignment comparison ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# TM-scores comparison
modes = ['all_chains', 'protein_chains_only']
mode_labels = ['All Chains\n(12 vs 1)', 'Protein Chains Only\n(9 vs 1)']
tm_q = [complex_res[m]['tm_norm_query'] if 'error' not in complex_res[m] else 0 for m in modes]
tm_t = [complex_res[m]['tm_norm_target'] if 'error' not in complex_res[m] else 0 for m in modes]
rmsd_v = [complex_res[m]['rmsd'] if 'error' not in complex_res[m] else 0 for m in modes]

x = np.arange(len(modes))
width = 0.3
axes[0].bar(x - width/2, tm_q, width, label='TM (norm. query)', color='#2ecc71', edgecolor='black', linewidth=0.5)
axes[0].bar(x + width/2, tm_t, width, label='TM (norm. target)', color='#9b59b6', edgecolor='black', linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(mode_labels, fontsize=10)
axes[0].set_ylabel('TM-score', fontsize=11)
axes[0].set_title('Complex-Level TM-scores', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].axhline(y=0.5, color='green', linestyle='--', alpha=0.4)
axes[0].axhline(y=0.17, color='orange', linestyle='--', alpha=0.4)

# Add value labels
for i, (tq, tt) in enumerate(zip(tm_q, tm_t)):
    axes[0].text(i - width/2, tq + 0.005, f'{tq:.3f}', ha='center', fontsize=9)
    axes[0].text(i + width/2, tt + 0.005, f'{tt:.3f}', ha='center', fontsize=9)

# RMSD comparison
axes[1].bar(x, rmsd_v, width=0.5, color='#e67e22', edgecolor='black', linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(mode_labels, fontsize=10)
axes[1].set_ylabel('RMSD (Å)', fontsize=11)
axes[1].set_title('Complex-Level RMSD', fontsize=12, fontweight='bold')
for i, r in enumerate(rmsd_v):
    axes[1].text(i, r + 0.1, f'{r:.2f}Å', ha='center', fontsize=10, fontweight='bold')

# Chain composition
chain_types = ['Protein\n(7xg4)', 'Nucleic Acid\n(7xg4)', 'Protein\n(6n40)']
chain_counts = [9, 3, 1]
residue_counts = [
    sum(chain_summary[f'7xg4_{c}']['length'] for c in ['A','B','C','D','E','F','G','H','L']),
    sum(chain_summary[f'7xg4_{c}']['length'] for c in ['I','J','K']),
    chain_summary['6n40_A']['length']
]
colors_comp = ['#3498db', '#e74c3c', '#2ecc71']

axes[2].bar(chain_types, chain_counts, color=colors_comp, edgecolor='black', linewidth=0.5, alpha=0.7)
ax2_twin = axes[2].twinx()
axes[2].set_ylabel('Number of Chains', fontsize=11, color='black')
ax2_twin.set_ylabel('Total Residues', fontsize=11, color='gray')

# Add residue counts as line
ax2_twin.plot(chain_types, residue_counts, 'D-', color='gray', markersize=8, linewidth=2)
for i, rc in enumerate(residue_counts):
    ax2_twin.annotate(str(rc), (i, rc), textcoords="offset points",
                      xytext=(0, 10), ha='center', fontsize=9, color='gray')

axes[2].set_title('Complex Composition', fontsize=12, fontweight='bold')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
ax2_twin.spines['top'].set_visible(False)

plt.suptitle('Complex-Level Structural Alignment: 7xg4 vs 6n40',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig3_complex_alignment_comparison.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ── Figure 4: Superimposition visualization ───────────────────────────
# Parse structures and apply superimposition for best-matching chain pair
from tmtools import tm_align as tm_align_func

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M',
}

def parse_chain_coords(pdb_path, chain_id):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('str', pdb_path)
    coords = []
    seq = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if PDB.is_aa(residue, standard=False) and 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                        rn = residue.get_resname().strip()
                        seq.append(THREE_TO_ONE.get(rn, 'X'))
        break
    return np.array(coords, dtype=np.float64), ''.join(seq)

# Best match: L vs A (highest TM-score normalized by target)
best_q_chain = 'L'
best_t_chain = 'A'

q_coords, q_seq = parse_chain_coords(os.path.join(DATA_DIR, '7xg4.pdb'), best_q_chain)
t_coords, t_seq = parse_chain_coords(os.path.join(DATA_DIR, '6n40.pdb'), best_t_chain)

result = tm_align_func(x=q_coords, y=t_coords, seqx=q_seq, seqy=t_seq)

# Apply rotation and translation to query
rot = np.array(result.u)
trans = np.array(result.t)
q_transformed = (q_coords @ rot.T) + trans

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Before superimposition
ax = axes[0]
ax.scatter(t_coords[:, 0], t_coords[:, 1], c='#2ecc71', s=8, alpha=0.5, label=f'6n40 chain {best_t_chain}')
ax.scatter(q_coords[:, 0], q_coords[:, 1], c='#e74c3c', s=8, alpha=0.5, label=f'7xg4 chain {best_q_chain}')
ax.set_xlabel('X (Å)', fontsize=11)
ax.set_ylabel('Y (Å)', fontsize=11)
ax.set_title('Before Superimposition (XY projection)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# After superimposition
ax = axes[1]
ax.scatter(t_coords[:, 0], t_coords[:, 1], c='#2ecc71', s=8, alpha=0.5, label=f'6n40 chain {best_t_chain}')
ax.scatter(q_transformed[:, 0], q_transformed[:, 1], c='#e74c3c', s=8, alpha=0.5, label=f'7xg4 chain {best_q_chain} (transformed)')
ax.set_xlabel('X (Å)', fontsize=11)
ax.set_ylabel('Y (Å)', fontsize=11)
ax.set_title(f'After Superimposition (TM={result.tm_norm_chain2:.3f}, RMSD={result.rmsd:.2f}Å)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(f'Structural Superimposition: 7xg4 Chain {best_q_chain} vs 6n40 Chain {best_t_chain}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig4_superimposition_before_after.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ── Figure 5: Distance distribution after alignment ───────────────────
# Compute per-residue distances after superimposition for best chain pair
distances = np.linalg.norm(q_transformed - t_coords[:len(q_transformed)], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of distances
ax = axes[0]
ax.hist(distances, bins=40, color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.axvline(x=np.mean(distances), color='red', linestyle='--', label=f'Mean={np.mean(distances):.2f}Å')
ax.axvline(x=np.median(distances), color='orange', linestyle='--', label=f'Median={np.median(distances):.2f}Å')
ax.set_xlabel('Cα Distance after Superimposition (Å)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Per-Residue Distance Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Per-residue distance profile
ax = axes[1]
ax.plot(range(len(distances)), distances, color='#3498db', linewidth=0.8, alpha=0.8)
ax.fill_between(range(len(distances)), distances, alpha=0.2, color='#3498db')
ax.axhline(y=np.mean(distances), color='red', linestyle='--', alpha=0.6,
           label=f'Mean={np.mean(distances):.2f}Å')
ax.axhline(y=5.0, color='orange', linestyle=':', alpha=0.6, label='5Å threshold')
ax.set_xlabel('Residue Index (aligned)', fontsize=11)
ax.set_ylabel('Cα Distance (Å)', fontsize=11)
ax.set_title('Per-Residue Distance Profile', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(f'Alignment Quality: 7xg4 Chain {best_q_chain} vs 6n40 Chain {best_t_chain}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig5_distance_distribution.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# ── Figure 6: Multi-panel superimposition for top protein chains ─────
top_protein_chains = ['C', 'D', 'E', 'F', 'G', 'L']  # top protein chains by TM-score
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, q_chain in enumerate(top_protein_chains):
    ax = axes[idx]
    q_c, q_s = parse_chain_coords(os.path.join(DATA_DIR, '7xg4.pdb'), q_chain)
    t_c, t_s = parse_chain_coords(os.path.join(DATA_DIR, '6n40.pdb'), 'A')

    res = tm_align_func(x=q_c, y=t_c, seqx=q_s, seqy=t_s)
    rot_m = np.array(res.u)
    trans_v = np.array(res.t)
    q_trans = (q_c @ rot_m.T) + trans_v

    ax.scatter(t_c[:, 0], t_c[:, 1], c='#2ecc71', s=4, alpha=0.3, label=f'6n40_A')
    ax.scatter(q_trans[:, 0], q_trans[:, 1], c='#e74c3c', s=4, alpha=0.3,
               label=f'7xg4_{q_chain} (transformed)')
    ax.set_title(f'Chain {q_chain}: TM={res.tm_norm_chain2:.3f}, RMSD={res.rmsd:.1f}Å',
                 fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)

plt.suptitle('Superimposition of Top 6xg4 Protein Chains onto 6n40 Chain A (XY projection)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig6_multipanel_superimposition.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

# ── Figure 7: Alignment summary radar/spider chart ───────────────────
# Compare alignment metrics across chains
fig, ax = plt.subplots(figsize=(10, 8))

# For each protein chain, plot TM-score and 1/RMSD as a grouped comparison
protein_chain_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']
tm_vals = [best_matches[c]['tm_norm_target'] for c in protein_chain_list]
rmsd_vals = [best_matches[c]['rmsd'] for c in protein_chain_list]
# Normalize RMSD to a comparable scale (inverse, capped)
inv_rmsd = [1.0 / max(r, 0.1) for r in rmsd_vals]
inv_rmsd_norm = [r / max(inv_rmsd) * max(tm_vals) for r in inv_rmsd]

x = np.arange(len(protein_chain_list))
width = 0.35

bars1 = ax.bar(x - width/2, tm_vals, width, label='TM-score (norm. target)',
               color='#3498db', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, inv_rmsd_norm, width, label='1/RMSD (normalized)',
               color='#e67e22', edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(protein_chain_list, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Chain-Level Alignment Quality: TM-score vs Inverse RMSD',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig7_tm_vs_inv_rmsd.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

print("\nAll figures generated successfully!")
