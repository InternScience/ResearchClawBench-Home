"""
Generate all figures for the protein complex structural alignment report.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# Load results
with open('outputs/alignment_results.json', 'r') as f:
    alignment_results = json.load(f)

with open('outputs/complex_alignment.json', 'r') as f:
    complex_result = json.load(f)

with open('outputs/summary_statistics.json', 'r') as f:
    summary = json.load(f)

with open('outputs/chain_info.json', 'r') as f:
    chain_info = json.load(f)

os.makedirs('report/images', exist_ok=True)

# ============================================================
# Figure 1: TM-score bar chart for pairwise chain alignments
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

chains_7xg4 = [r['query_chain'].split('_')[1] for r in alignment_results]
tm_scores = [r['tm_score'] for r in alignment_results]
rmsd_values = [r['rmsd'] for r in alignment_results]

colors = ['#2196F3' if s < 0.2 else '#FF9800' if s < 0.4 else '#4CAF50' for s in tm_scores]
bars = ax.bar(range(len(chains_7xg4)), tm_scores, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, tm_scores)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(chains_7xg4)))
ax.set_xticklabels([f'Chain {c}' for c in chains_7xg4], fontsize=10)
ax.set_ylabel('TM-score', fontsize=12)
ax.set_xlabel('7xg4 Chain (vs 6n40 Chain A)', fontsize=12)
ax.set_title('Pairwise Chain-level Structural Alignment: TM-scores\n(7xg4 individual chains vs 6n40 Chain A)', fontsize=13, fontweight='bold')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='TM=0.5 (similar fold threshold)')
ax.set_ylim(0, max(tm_scores) * 1.25)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/tm_score_barchain.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/tm_score_barchain.png")

# ============================================================
# Figure 2: RMSD bar chart for pairwise chain alignments
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

colors_rmsd = ['#FF5722' if r > 8 else '#FF9800' if r > 6 else '#FFC107' for r in rmsd_values]
bars = ax.bar(range(len(chains_7xg4)), rmsd_values, color=colors_rmsd, edgecolor='black', linewidth=0.5)

for i, (bar, rmsd) in enumerate(zip(bars, rmsd_values)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            f'{rmsd:.2f}A', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(chains_7xg4)))
ax.set_xticklabels([f'Chain {c}' for c in chains_7xg4], fontsize=10)
ax.set_ylabel('RMSD (A)', fontsize=12)
ax.set_xlabel('7xg4 Chain (vs 6n40 Chain A)', fontsize=12)
ax.set_title('Pairwise Chain-level Structural Alignment: RMSD\n(7xg4 individual chains vs 6n40 Chain A)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/rmsd_barchain.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/rmsd_barchain.png")

# ============================================================
# Figure 3: TM-score vs Aligned residues scatter
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

n_aligned_list = [r['n_aligned'] for r in alignment_results]
chain_labels = [r['query_chain'].split('_')[1] for r in alignment_results]

scatter = ax.scatter(n_aligned_list, tm_scores, c=rmsd_values, cmap='viridis_r', 
                     s=150, edgecolors='black', linewidth=1.5, zorder=5)

for i, label in enumerate(chain_labels):
    ax.annotate(f'Chain {label}', (n_aligned_list[i], tm_scores[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=9, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('RMSD (A)', fontsize=11)

ax.set_xlabel('Number of Aligned Residues', fontsize=12)
ax.set_ylabel('TM-score', fontsize=12)
ax.set_title('TM-score vs Alignment Coverage\n(7xg4 chains vs 6n40 Chain A)', fontsize=13, fontweight='bold')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='TM=0.5 threshold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/tm_vs_aligned.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/tm_vs_aligned.png")

# ============================================================
# Figure 4: Complex-level comparison overview
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: TM-score comparison
categories = ['Best Chain\n(7xg4_L vs 6n40_A)', 'Mean Chain\n(all pairs)', 'Full Complex\n(all chains)']
tm_vals = [summary['best_chain_tm'], summary['mean_tm_score'], summary['complex_tm_score']]
comp_colors = ['#4CAF50', '#FFC107', '#2196F3']

bars = axes[0].bar(categories, tm_vals, color=comp_colors, edgecolor='black', linewidth=0.5, width=0.5)
for bar, val in zip(bars, tm_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_ylabel('TM-score', fontsize=12)
axes[0].set_title('TM-score Comparison', fontsize=13, fontweight='bold')
axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Similar fold (0.5)')
axes[0].set_ylim(0, max(tm_vals) * 1.3)
axes[0].legend(fontsize=9)
axes[0].grid(axis='y', alpha=0.3)

# Right: RMSD comparison
rmsd_vals = [summary['best_chain_rmsd'], summary['mean_rmsd'], summary['complex_rmsd']]
bars = axes[1].bar(categories, rmsd_vals, color=comp_colors, edgecolor='black', linewidth=0.5, width=0.5)
for bar, val in zip(bars, rmsd_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 f'{val:.2f}A', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[1].set_ylabel('RMSD (A)', fontsize=12)
axes[1].set_title('RMSD Comparison', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Structural Alignment Summary: Pairwise vs Complex-level', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('report/images/alignment_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/alignment_summary.png")

# ============================================================
# Figure 5: Chain length distribution
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

chain_names_7xg4 = sorted(chain_info['7xg4'].keys())
chain_lengths_7xg4 = [chain_info['7xg4'][c]['n_residues'] for c in chain_names_7xg4]

chain_names_6n40 = sorted(chain_info['6n40'].keys())
chain_lengths_6n40 = [chain_info['6n40'][c]['n_residues'] for c in chain_names_6n40]

x_7xg4 = np.arange(len(chain_names_7xg4))
x_6n40 = np.arange(len(chain_names_6n40))

bars1 = ax.bar(x_7xg4 - 0.2, chain_lengths_7xg4, 0.4, label='7xg4', color='#2196F3', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_6n40 + 0.2, chain_lengths_6n40, 0.4, label='6n40', color='#FF5722', edgecolor='black', linewidth=0.5)

for bar, length in zip(bars1, chain_lengths_7xg4):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            str(length), ha='center', va='bottom', fontsize=8, fontweight='bold')

for bar, length in zip(bars2, chain_lengths_6n40):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            str(length), ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(range(max(len(chain_names_7xg4), len(chain_names_6n40))))
ax.set_xticklabels([f'Chain {c}' for c in chain_names_7xg4] + [''] * max(0, len(chain_names_6n40) - len(chain_names_7xg4)), fontsize=10)
ax.set_ylabel('Number of Residues', fontsize=12)
ax.set_title('Chain Length Distribution: 7xg4 vs 6n40', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/chain_lengths.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/chain_lengths.png")

# ============================================================
# Figure 6: TM-score distribution histogram
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(tm_scores, bins=9, color='#2196F3', edgecolor='black', alpha=0.7, density=False)
ax.axvline(x=np.mean(tm_scores), color='red', linestyle='-', linewidth=2, label=f'Mean = {np.mean(tm_scores):.4f}')
ax.axvline(x=np.median(tm_scores), color='green', linestyle='--', linewidth=2, label=f'Median = {np.median(tm_scores):.4f}')

ax.set_xlabel('TM-score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Pairwise Chain-level TM-scores', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/tm_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/tm_distribution.png")

# ============================================================
# Figure 7: Heatmap-style table of alignment metrics
# ============================================================
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

# Create table data
table_data = []
for r in alignment_results:
    table_data.append([
        r['query_chain'],
        r['target_chain'],
        str(r['query_length']),
        str(r['target_length']),
        str(r['n_aligned']),
        f"{r['tm_score']:.4f}",
        f"{r['rmsd']:.4f}",
        f"{r['computation_time_sec']:.3f}s"
    ])

columns = ['Query', 'Target', 'Q Len', 'T Len', 'Aligned', 'TM-score', 'RMSD (A)', 'Time']

table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Color the TM-score column
for i in range(len(table_data)):
    cell = table[i+1, 5]
    tm_val = float(table_data[i][5])
    if tm_val >= 0.17:
        cell.set_facecolor('#E8F5E9')
    else:
        cell.set_facecolor('#FFEBEE')

ax.set_title('Complete Pairwise Alignment Results Table', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('report/images/alignment_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/alignment_table.png")

print("\nAll figures generated successfully!")
