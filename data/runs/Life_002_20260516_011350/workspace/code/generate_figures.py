#!/usr/bin/env python3
"""Generate figures for the structural alignment report."""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

os.makedirs('report/images', exist_ok=True)

# Load results
with open('outputs/alignment_results.json', 'r') as f:
    results = json.load(f)

protein_chains = results['protein_chains_7xg4']
nucleic_chains = results['nucleic_chains_7xg4']

# Parse nucleic chain lengths from PDB
with open('data/7xg4.pdb', 'r') as f:
    pdb_lines = f.readlines()

nucleic_lengths = {}
for c in nucleic_chains:
    count = 0
    for line in pdb_lines:
        if (line.startswith('ATOM') or line.startswith('HETATM')) and line[21:22].strip() == c:
            atom_name = line[12:16].strip()
            if atom_name == 'P':
                count += 1
    nucleic_lengths[c] = count

# ============================================================
# Figure 1: Data Overview - Chain Composition
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 7xg4 chain composition - all chains
all_chain_ids_7xg4 = protein_chains + nucleic_chains
all_lengths = []
all_colors = []
all_labels = []
for c in protein_chains:
    all_lengths.append(results['chain_lengths_7xg4'][c])
    all_colors.append('#4472C4')
    all_labels.append(f'{c}\n(protein)')
for c in nucleic_chains:
    all_lengths.append(nucleic_lengths.get(c, 0))
    all_colors.append('#ED7D31')
    all_labels.append(f'{c}\n(nucleic)')

ax1 = axes[0]
x_pos = np.arange(len(all_lengths))
bars1 = ax1.bar(x_pos, all_lengths, color=all_colors, edgecolor='black', linewidth=0.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(all_labels, fontsize=8)
ax1.set_ylabel('Number of Residues', fontsize=11)
ax1.set_title('7xg4: Type IV-A CRISPR-Cas Complex\nChain Composition', fontweight='bold', fontsize=12)
ax1.set_ylim(0, max(all_lengths) * 1.2)

# Add legend
legend_elements = [mpatches.Patch(facecolor='#4472C4', label='Protein chain (CA atoms)'),
                   mpatches.Patch(facecolor='#ED7D31', label='Nucleic acid chain (P atoms)')]
ax1.legend(handles=legend_elements, fontsize=8, loc='upper right')

for bar, val in zip(bars1, all_lengths):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3, str(val),
             ha='center', va='bottom', fontsize=7)

# 6n40
ax2 = axes[1]
chain_len_6n40 = results['chain_lengths_6n40']['A']
bars2 = ax2.bar([0], [chain_len_6n40], color='#4472C4', edgecolor='black', linewidth=0.5, width=0.4)
ax2.set_xticks([0])
ax2.set_xticklabels(['A (protein)'], fontsize=10)
ax2.set_ylabel('Number of Residues', fontsize=11)
ax2.set_title('6n40: MmpL3 Transporter\nChain Composition', fontweight='bold', fontsize=12)
ax2.set_ylim(0, chain_len_6n40 * 1.2)
ax2.text(0, chain_len_6n40 + 10, str(chain_len_6n40), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/figure1_chain_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: figure1_chain_composition.png")

# ============================================================
# Figure 2: TM-score Heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

chain_ids_1 = protein_chains
chain_ids_2 = results['chain_6n40']

tm_matrix = np.zeros((len(chain_ids_1), len(chain_ids_2)))
for pr in results['all_pairwise']:
    i = chain_ids_1.index(pr['chain_1'])
    j = chain_ids_2.index(pr['chain_2'])
    tm_matrix[i, j] = pr['tm_score_avg']

cmap = LinearSegmentedColormap.from_list('tm_cmap', ['#f7fbff', '#deebf7', '#9ecae1', '#4292c6', '#2171b5', '#08519c', '#08306b'])
im = ax.imshow(tm_matrix, cmap=cmap, aspect='auto', vmin=0.15, vmax=0.3)

ax.set_xticks(range(len(chain_ids_2)))
ax.set_xticklabels([f'6n40:{c}' for c in chain_ids_2], fontsize=11)
ax.set_yticks(range(len(chain_ids_1)))
ax.set_yticklabels([f'7xg4:{c}' for c in chain_ids_1], fontsize=11)

for i in range(len(chain_ids_1)):
    for j in range(len(chain_ids_2)):
        val = tm_matrix[i, j]
        text_color = 'white' if val > 0.22 else 'black'
        ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=10, color=text_color, fontweight='bold')

ax.set_title('TM-score Matrix: 7xg4 vs 6n40\n(Protein Chain Pairs)', fontweight='bold', fontsize=14)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Average TM-score', fontsize=11)

ax.text(0.5, -0.15, 'TM-score < 0.5: No significant structural similarity detected\n'
        '(random baseline TM-score ≈ 0.17)',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('report/images/figure2_tm_score_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: figure2_tm_score_heatmap.png")

# ============================================================
# Figure 3: Per-Chain Alignment Quality
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

all_pr = results['all_pairwise']
chains_list = [pr['chain_1'] for pr in all_pr]
tm_scores = [pr['tm_score_avg'] for pr in all_pr]
rmsds = [pr['rmsd'] for pr in all_pr]
coverages = [pr['coverage_1'] for pr in all_pr]

idx = np.argsort(tm_scores)[::-1]
chains_sorted = [chains_list[i] for i in idx]
tm_sorted = [tm_scores[i] for i in idx]
rmsd_sorted = [rmsds[i] for i in idx]
cov_sorted = [coverages[i] for i in idx]

colors = ['#08306b' if t > 0.25 else '#4292c6' if t > 0.22 else '#9ecae1' if t > 0.19 else '#deebf7' for t in tm_sorted]

# TM-score panel
ax = axes[0]
x_pos = np.arange(len(tm_sorted))
bars = ax.bar(x_pos, tm_sorted, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0.17, color='red', linestyle='--', linewidth=1.5, label='Random baseline (0.17)')
ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1.5, label='Significance (0.5)')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{c}' for c in chains_sorted], fontsize=10)
ax.set_xlabel('7xg4 Chain', fontsize=10)
ax.set_ylabel('Average TM-score', fontsize=11)
ax.set_title('TM-score by Chain', fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
for bar, val in zip(bars, tm_sorted):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{val:.3f}',
            ha='center', va='bottom', fontsize=7)

# RMSD panel
ax = axes[1]
bars = ax.bar(x_pos, rmsd_sorted, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{c}' for c in chains_sorted], fontsize=10)
ax.set_xlabel('7xg4 Chain', fontsize=10)
ax.set_ylabel('RMSD (Å)', fontsize=11)
ax.set_title('RMSD by Chain', fontweight='bold')
for bar, val in zip(bars, rmsd_sorted):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2, f'{val:.1f}',
            ha='center', va='bottom', fontsize=7)

# Coverage panel
ax = axes[2]
bars = ax.bar(x_pos, [c*100 for c in cov_sorted], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{c}' for c in chains_sorted], fontsize=10)
ax.set_xlabel('7xg4 Chain', fontsize=10)
ax.set_ylabel('Coverage (%)', fontsize=11)
ax.set_title('Alignment Coverage', fontweight='bold')
for bar, val in zip(bars, [c*100 for c in cov_sorted]):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{val:.0f}%',
            ha='center', va='bottom', fontsize=7)

plt.suptitle('Per-Chain Alignment Quality: 7xg4 Chains → 6n40 Chain A', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('report/images/figure3_alignment_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: figure3_alignment_quality.png")

# ============================================================
# Figure 4: Distance Distribution & Per-Residue Analysis
# ============================================================

import sys
sys.path.insert(0, 'code')
from structural_alignment import parse_pdb, compute_d0

chains_7xg4 = parse_pdb('data/7xg4.pdb')
chains_6n40 = parse_pdb('data/6n40.pdb')

best_chain = 'H'
best_result = None
for pr in all_pr:
    if pr['chain_1'] == best_chain:
        best_result = pr
        break

if best_result:
    coords1 = chains_7xg4[best_chain]
    coords2 = chains_6n40['A']
    
    a1 = best_result['aligned_indices_1']
    a2 = best_result['aligned_indices_2']
    
    c1_aligned = coords1[a1]
    c2_aligned = coords2[a2]
    
    R = np.array(best_result['rotation_matrix'])
    t = np.array(best_result['translation_vector'])
    
    c1_rot = (R @ c1_aligned.T).T + t
    dists = np.sqrt(np.sum((c1_rot - c2_aligned) ** 2, axis=1))
    
    L = len(c1_aligned)
    d0 = compute_d0(L)
    contributions = 1.0 / (1.0 + (dists / d0) ** 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distance histogram
    ax = axes[0]
    ax.hist(dists, bins=40, color='#4472C4', edgecolor='black', alpha=0.8, linewidth=0.5)
    ax.axvline(x=np.mean(dists), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(dists):.1f} Å')
    ax.axvline(x=np.median(dists), color='orange', linestyle='--', linewidth=2, label=f'Median = {np.median(dists):.1f} Å')
    ax.axvline(x=d0, color='green', linestyle=':', linewidth=2, label=f'd₀ = {d0:.1f} Å')
    ax.set_xlabel('Cα Distance After Superposition (Å)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Residue Distance Distribution\n7xg4 Chain {best_chain} (CSF5) → 6n40 Chain A (MmpL3)', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    
    # Per-residue contribution
    ax = axes[1]
    ax.plot(range(len(contributions)), contributions, 'o-', markersize=2, linewidth=1, color='#08519c', alpha=0.7)
    ax.axhline(y=np.mean(contributions), color='red', linestyle='--', linewidth=1.5, 
               label=f'Mean contribution = {np.mean(contributions):.3f}')
    ax.set_xlabel('Aligned Residue Pair Index', fontsize=11)
    ax.set_ylabel('TM-score Contribution', fontsize=11)
    ax.set_title(f'Per-Residue TM-score Contribution\n7xg4 Chain {best_chain} (CSF5) → 6n40 Chain A (MmpL3)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('report/images/figure4_distance_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved: figure4_distance_analysis.png")

# ============================================================
# Figure 5: TM-score vs RMSD Scatter
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

all_tm = [pr['tm_score_avg'] for pr in all_pr]
all_rmsd = [pr['rmsd'] for pr in all_pr]
sizes = [pr['length_1'] * 0.8 for pr in all_pr]
labels_list = [f"7xg4:{pr['chain_1']}" for pr in all_pr]

scatter = ax.scatter(all_rmsd, all_tm, s=sizes, c=range(len(all_tm)), 
                     cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)

for i, (x, y, label) in enumerate(zip(all_rmsd, all_tm, labels_list)):
    ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1.5, label='TM-score = 0.5 (significant similarity)')
ax.axhline(y=0.17, color='red', linestyle='--', linewidth=1.5, label='TM-score = 0.17 (random baseline)')
ax.set_xlabel('RMSD (Å)', fontsize=12)
ax.set_ylabel('Average TM-score', fontsize=12)
ax.set_title('TM-score vs RMSD: All 7xg4 Chains → 6n40 Chain A', fontweight='bold', fontsize=14)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(12, 30)
ax.set_ylim(0.15, 0.30)

textstr = ('Key Observations:\n'
           '• All TM-scores < 0.5: no significant structural similarity\n'
           '• All TM-scores > 0.17: above random but marginal\n'
           '• Best: Chain H (CSF5) TM=0.264, RMSD=15.3 Å\n'
           '• Structurally unrelated protein complexes')
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.savefig('report/images/figure5_tm_vs_rmsd.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: figure5_tm_vs_rmsd.png")

# ============================================================
# Figure 6: Superimposition Visualization
# ============================================================

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

coords_h = chains_7xg4['H']
coords_a = chains_6n40['A']

n_sample = min(200, len(coords_h), len(coords_a))
idx_h = np.linspace(0, len(coords_h)-1, n_sample).astype(int)
idx_a = np.linspace(0, len(coords_a)-1, n_sample).astype(int)

coords_h_sub = coords_h[idx_h]
coords_a_sub = coords_a[idx_a]

# Before alignment
ax1.scatter(coords_h_sub[:, 0], coords_h_sub[:, 1], c='#2166AC', s=15, alpha=0.6, label='7xg4 Chain H (CSF5)', marker='o')
ax1.scatter(coords_a_sub[:, 0], coords_a_sub[:, 1], c='#B2182B', s=15, alpha=0.6, label='6n40 Chain A (MmpL3)', marker='^')
ax1.set_xlabel('X (Å)', fontsize=10)
ax1.set_ylabel('Y (Å)', fontsize=10)
ax1.set_title(f'Before Alignment (XY Projection)', fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_aspect('equal')

# After alignment
if best_result:
    a1 = best_result['aligned_indices_1']
    a2 = best_result['aligned_indices_2']
    
    c1_ali = chains_7xg4[best_chain][a1]
    c2_ali = chains_6n40['A'][a2]
    
    R_mat = np.array(best_result['rotation_matrix'])
    t_vec = np.array(best_result['translation_vector'])
    c1_rot = (R_mat @ c1_ali.T).T + t_vec
    
    n_s = min(200, len(c1_rot))
    idx = np.linspace(0, len(c1_rot)-1, n_s).astype(int)
    
    ax2.scatter(c1_rot[idx, 0], c1_rot[idx, 1], c='#2166AC', s=15, alpha=0.6, label='7xg4 Chain H (rotated)', marker='o')
    ax2.scatter(c2_ali[idx, 0], c2_ali[idx, 1], c='#B2182B', s=15, alpha=0.6, label='6n40 Chain A', marker='^')
    
    for k in range(0, n_s, max(1, n_s // 15)):
        ax2.plot([c1_rot[idx[k], 0], c2_ali[idx[k], 0]], 
                 [c1_rot[idx[k], 1], c2_ali[idx[k], 1]], 
                 'gray', alpha=0.3, linewidth=0.5)
    
    ax2.set_xlabel('X (Å)', fontsize=10)
    ax2.set_ylabel('Y (Å)', fontsize=10)
    ax2.set_title(f'After Optimal Superposition (XY Projection)', fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_aspect('equal')

plt.suptitle(f'Structural Superimposition: 7xg4 Chain H → 6n40 Chain A\n'
             f'TM-score = {best_result["tm_score_avg"]:.3f}, RMSD = {best_result["rmsd"]:.1f} Å', 
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('report/images/figure6_superimposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: figure6_superimposition.png")

print("\nAll figures saved to report/images/")
