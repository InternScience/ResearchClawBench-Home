#!/usr/bin/env python3
"""
Additional analysis and visualizations for the structural alignment report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

OUTPUTS_DIR = 'outputs'
IMAGES_DIR = 'report/images'

os.makedirs(IMAGES_DIR, exist_ok=True)

# Load data
with open(os.path.join(OUTPUTS_DIR, 'pairwise_alignments.json')) as f:
    pairwise_results = json.load(f)

with open(os.path.join(OUTPUTS_DIR, 'oligomeric_alignment.json')) as f:
    oligomeric_result = json.load(f)

# ============================================================
# Figure 7: Rotation matrix heatmap
# ============================================================
U = np.array([
    [0.7008336186, -0.5015290980, -0.5072482656],
    [-0.7103842130, -0.5552172828, -0.4325367486],
    [-0.0647032384, 0.6634774546, -0.7453932910]
])
t = np.array([255.8927873969, 200.8630753399, 239.8295382249])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Rotation matrix
im1 = axes[0].imshow(U, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_xticks([0, 1, 2])
axes[0].set_yticks([0, 1, 2])
axes[0].set_xticklabels(['X', 'Y', 'Z'])
axes[0].set_yticklabels(['X\'', 'Y\'', 'Z\''])
axes[0].set_title('Rotation Matrix (U)')
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f'{U[i,j]:.3f}', ha='center', va='center', 
                    color='white' if abs(U[i,j]) > 0.5 else 'black', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Translation vector
axes[1].bar(['X', 'Y', 'Z'], t, color=['crimson', 'forestgreen', 'royalblue'], edgecolor='black')
axes[1].set_ylabel('Translation (Å)')
axes[1].set_title('Translation Vector (t)')
for i, v in enumerate(t):
    axes[1].text(i, v + 3, f'{v:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig7_transformation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

# ============================================================
# Figure 8: Monomeric vs Oligomeric alignment comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

target_chains = [r['target'].split(':')[1] for r in pairwise_results]
tm_scores = [r['tm_score_1'] for r in pairwise_results]

# Best monomeric
best_monomeric_idx = np.argmax(tm_scores)
best_monomeric_chain = target_chains[best_monomeric_idx]
best_monomeric_tm = tm_scores[best_monomeric_idx]

# Oligomeric
oligomeric_tm = oligomeric_result['tm_score_query_norm']

bars = ax.bar(target_chains + ['Oligomeric\n(USalign)'], 
              tm_scores + [oligomeric_tm],
              color=['coral']*len(target_chains) + ['navy'],
              edgecolor='black')

# Highlight best monomeric
bars[best_monomeric_idx].set_color('gold')
ax.axhline(y=0.17, color='gray', linestyle='--', alpha=0.7, label='Random (0.17)')
ax.axhline(y=0.30, color='red', linestyle='--', alpha=0.7, label='Structural similarity (0.30)')
ax.set_xlabel('Alignment Mode / Target Chain')
ax.set_ylabel('TM-score (normalized by 6N40 length)')
ax.set_title('Monomeric vs Oligomeric Alignment Comparison')
ax.legend()
ax.set_ylim(0, 0.35)

for i, v in enumerate(tm_scores + [oligomeric_tm]):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig8_monomeric_vs_oligomeric.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 8 saved.")

# ============================================================
# Figure 9: Conceptual pipeline diagram
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')

# Boxes
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_box(ax, x, y, w, h, text, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

draw_box(ax, 0.5, 1.5, 2, 1, 'Input\nStructures\n(7XG4, 6N40)', 'lightblue')
draw_box(ax, 3.5, 1.5, 2, 1, 'Chain\nExtraction\n& Parsing', 'lightgreen')
draw_box(ax, 6.5, 1.5, 2, 1, 'Pairwise\n& Oligomeric\nAlignment\n(TM-align, USalign)', 'lightyellow')
draw_box(ax, 9.5, 1.5, 2, 1, 'Output:\nTM-scores,\nRMSD, Vectors', 'lightcoral')

# Arrows
ax.annotate('', xy=(3.4, 2.0), xytext=(2.6, 2.0),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(6.4, 2.0), xytext=(5.6, 2.0),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(9.4, 2.0), xytext=(8.6, 2.0),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_title('Structural Alignment Pipeline for Protein Complex Comparison', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig9_pipeline.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 9 saved.")

# ============================================================
# Figure 10: Coverage fraction analysis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

target_chains = [r['target'].split(':')[1] for r in pairwise_results]
coverage_query = [r['aligned_len'] / r['query_len'] for r in pairwise_results]
coverage_target = [r['aligned_len'] / r['target_len'] for r in pairwise_results]

x = np.arange(len(target_chains))
width = 0.35

bars1 = ax.bar(x - width/2, coverage_query, width, label='Coverage (6N40)', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, coverage_target, width, label='Coverage (7XG4 chain)', color='darkorange', edgecolor='black')

ax.set_xlabel('7XG4 Chain')
ax.set_ylabel('Alignment Coverage (fraction)')
ax.set_title('Alignment Coverage: Fraction of Structure Aligned')
ax.set_xticks(x)
ax.set_xticklabels(target_chains)
ax.legend()
ax.set_ylim(0, 1.0)

for i, (v1, v2) in enumerate(zip(coverage_query, coverage_target)):
    ax.text(i - width/2, v1 + 0.02, f'{v1:.2f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2, v2 + 0.02, f'{v2:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig10_coverage.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure 10 saved.")

# ============================================================
# Save summary statistics
# ============================================================
summary_stats = {
    'query': '6n40 (MMPL3 membrane protein, Mycobacterium smegmatis)',
    'target': '7xg4 (Type IV-A CRISPR-Cas complex, Pseudomonas aeruginosa)',
    'query_chains': 1,
    'target_protein_chains': 9,
    'target_total_chains': 12,
    'best_monomeric_alignment': {
        'target_chain': best_monomeric_chain,
        'tm_score_query_norm': float(best_monomeric_tm),
        'tm_score_target_norm': float(pairwise_results[best_monomeric_idx]['tm_score_2']),
        'rmsd': float(pairwise_results[best_monomeric_idx]['rmsd']),
        'aligned_length': int(pairwise_results[best_monomeric_idx]['aligned_len']),
        'sequence_identity': float(pairwise_results[best_monomeric_idx]['seq_id'])
    },
    'oligomeric_alignment': oligomeric_result,
    'interpretation': 'All TM-scores are below 0.30, indicating no meaningful structural similarity between 6N40 and any chain of 7XG4. The best match (7XG4:L, TM-score=0.201) is still in the random structural similarity range.',
    'method_comparison': {
        'tmalign': 'Monomeric alignment, best TM-score=0.201',
        'usalign_oligomeric': 'Oligomeric alignment (mm=1), TM-score=0.194',
        'conclusion': 'Both methods consistently report no significant structural homology'
    }
}

with open(os.path.join(OUTPUTS_DIR, 'summary_statistics.json'), 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("Summary statistics saved.")
print("\nAll additional analysis complete!")
