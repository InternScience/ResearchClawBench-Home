#!/usr/bin/env python3
"""
Generate figures for the structural alignment report.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

os.makedirs('report/images', exist_ok=True)

# Load results
with open('outputs/alignment_results.json', 'r') as f:
    results = json.load(f)

# ============================================================
# Figure 1: TM-score heatmap (full alignment matrix)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

q_chains = []
t_chains = []
tm_data = []

for row in results['full_alignment_matrix']:
    q_chains.append(row['query_chain'])
    for t_id, vals in row['targets'].items():
        if t_id not in t_chains:
            t_chains.append(t_id)

tm_matrix = np.zeros((len(q_chains), len(t_chains)))
for i, row in enumerate(results['full_alignment_matrix']):
    for j, t_id in enumerate(t_chains):
        tm_matrix[i, j] = row['targets'][t_id]['tm_avg']

cmap = LinearSegmentedColormap.from_list('tm', ['#2c3e50', '#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])
im = ax.imshow(tm_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=0.5)

ax.set_xticks(range(len(t_chains)))
ax.set_xticklabels([f'6n40:{c}' for c in t_chains], fontsize=11, fontweight='bold')
ax.set_yticks(range(len(q_chains)))
ax.set_yticklabels([f'7xg4:{c}' for c in q_chains], fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(q_chains)):
    for j in range(len(t_chains)):
        val = tm_matrix[i, j]
        color = 'white' if val < 0.25 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, color=color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='TM-score (avg)')
ax.set_title('Pairwise Chain TM-Score Matrix\n7xg4 (Query) vs 6n40 (Target)', fontsize=14, fontweight='bold')
ax.set_xlabel('Target Chains (6n40)', fontsize=12)
ax.set_ylabel('Query Chains (7xg4)', fontsize=12)

plt.tight_layout()
plt.savefig('report/images/fig1_tm_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: report/images/fig1_tm_heatmap.png")

# ============================================================
# Figure 2: Bar chart of pairwise TM-scores
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

pairwise = results['pairwise_chain_alignments']
# Filter protein chains only
protein_pairs = [p for p in pairwise if p['mol_type'] == 'protein']

chains = [p['query_chain'] for p in protein_pairs]
tm_q = [p['tm_score_query_norm'] for p in protein_pairs]
tm_t = [p['tm_score_target_norm'] for p in protein_pairs]
rmsds = [p['rmsd'] for p in protein_pairs]
aligned = [p['n_aligned'] for p in protein_pairs]

x = np.arange(len(chains))
width = 0.35

# TM-scores
bars1 = axes[0].bar(x - width/2, tm_q, width, label='TM (7xg4 norm)', color='#3498db', alpha=0.85)
bars2 = axes[0].bar(x + width/2, tm_t, width, label='TM (6n40 norm)', color='#e74c3c', alpha=0.85)
axes[0].set_xlabel('7xg4 Chain', fontsize=12)
axes[0].set_ylabel('TM-score', fontsize=12)
axes[0].set_title('TM-Score by Chain Pair', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(chains, fontsize=11)
axes[0].legend(fontsize=10)
axes[0].set_ylim(0, 0.5)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Significance threshold')

# RMSD and aligned residues
ax2 = axes[1]
color1 = '#2ecc71'
color2 = '#9b59b6'

bars3 = ax2.bar(x - width/2, rmsds, width, label='RMSD (A)', color=color1, alpha=0.85)
ax2.set_xlabel('7xg4 Chain', fontsize=12)
ax2.set_ylabel('RMSD (Angstrom)', fontsize=12, color=color1)
ax2.set_title('RMSD and Aligned Residues', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(chains, fontsize=11)

ax3 = ax2.twinx()
bars4 = ax3.bar(x + width/2, aligned, width, label='Aligned residues', color=color2, alpha=0.85)
ax3.set_ylabel('Aligned Residues', fontsize=12, color=color2)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('report/images/fig2_pairwise_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: report/images/fig2_pairwise_bars.png")

# ============================================================
# Figure 3: Chain composition and structure overview
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 7xg4 chain lengths
q_info = results['query_info']
q_chains_sorted = sorted(q_info.keys())
q_lengths = [q_info[c]['n_residues'] for c in q_chains_sorted]
q_types = [q_info[c]['mol_type'] for c in q_chains_sorted]
colors = ['#3498db' if t == 'protein' else '#e74c3c' for t in q_types]

axes[0].bar(q_chains_sorted, q_lengths, color=colors, alpha=0.85, edgecolor='white')
axes[0].set_xlabel('Chain ID', fontsize=12)
axes[0].set_ylabel('Number of Residues', fontsize=12)
axes[0].set_title('7xg4 Chain Composition', fontsize=13, fontweight='bold')
legend_patches = [mpatches.Patch(color='#3498db', label='Protein'),
                  mpatches.Patch(color='#e74c3c', label='Nucleic Acid')]
axes[0].legend(handles=legend_patches, fontsize=10)

# 6n40 chain lengths
t_info = results['target_info']
t_chains_sorted = sorted(t_info.keys())
t_lengths = [t_info[c]['n_residues'] for c in t_chains_sorted]
t_types = [t_info[c]['mol_type'] for c in t_chains_sorted]
colors2 = ['#3498db' if t == 'protein' else '#e74c3c' for t in t_types]

axes[1].bar(t_chains_sorted, t_lengths, color=colors2, alpha=0.85, edgecolor='white')
axes[1].set_xlabel('Chain ID', fontsize=12)
axes[1].set_ylabel('Number of Residues', fontsize=12)
axes[1].set_title('6n40 Chain Composition', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/fig3_chain_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: report/images/fig3_chain_composition.png")

# ============================================================
# Figure 4: Complex-level summary
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

cs = results['complex_scores']
metrics = ['TM-score\n(query norm)', 'TM-score\n(target norm)', 'TM-score\n(average)',
           'Coverage\n(query)', 'Coverage\n(target)']
values = [cs['tm_score_normalized_by_query'], cs['tm_score_normalized_by_target'],
          cs['tm_score_avg'], cs['coverage_query'], cs['coverage_target']]
bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

bars = ax.bar(metrics, values, color=bar_colors, alpha=0.85, edgecolor='white', width=0.6)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Complex-Level Alignment Summary\n7xg4 vs 6n40', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(values) * 1.3)
ax.axhline(y=0.17, color='red', linestyle='--', alpha=0.5, label='Random threshold (~0.17)')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig4_complex_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: report/images/fig4_complex_summary.png")

# ============================================================
# Figure 5: Alignment length distribution
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

protein_pairs_sorted = sorted(protein_pairs, key=lambda x: x['n_aligned'], reverse=True)
chains_sorted = [p['query_chain'] for p in protein_pairs_sorted]
n_query = [p['n_query'] for p in protein_pairs_sorted]
n_aligned = [p['n_aligned'] for p in protein_pairs_sorted]
n_unaligned = [q - a for q, a in zip(n_query, n_aligned)]

x = np.arange(len(chains_sorted))
ax.bar(x, n_aligned, label='Aligned residues', color='#2ecc71', alpha=0.85)
ax.bar(x, n_unaligned, bottom=n_aligned, label='Unaligned residues', color='#ecf0f1', alpha=0.85, edgecolor='#bdc3c7')

ax.set_xticks(x)
ax.set_xticklabels([f'7xg4:{c}' for c in chains_sorted], fontsize=11)
ax.set_xlabel('Query Chain', fontsize=12)
ax.set_ylabel('Number of Residues', fontsize=12)
ax.set_title('Alignment Coverage per Chain\n(Aligned vs Unaligned Residues)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('report/images/fig5_alignment_coverage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: report/images/fig5_alignment_coverage.png")

print("\nAll figures generated successfully!")
