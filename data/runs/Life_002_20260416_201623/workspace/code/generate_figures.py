#!/usr/bin/env python3
"""
Generate all figures for the research report.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

WORKSPACE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_002_20260416_201623'
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
IMAGE_DIR = os.path.join(WORKSPACE, 'report', 'images')
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load data
with open(os.path.join(OUTPUT_DIR, 'structure_summary.json')) as f:
    summary = json.load(f)

with open(os.path.join(OUTPUT_DIR, 'alignment_results.json')) as f:
    alignment = json.load(f)

with open(os.path.join(OUTPUT_DIR, 'intra_complex_alignment.json')) as f:
    intra = json.load(f)

with open(os.path.join(OUTPUT_DIR, 'inter_chain_contacts.json')) as f:
    contacts = json.load(f)

with open(os.path.join(OUTPUT_DIR, 'chain_features.json')) as f:
    features = json.load(f)

with open(os.path.join(OUTPUT_DIR, '3di_sequences.json')) as f:
    threedi = json.load(f)

# ============ Figure 1: Structure Overview ============
print("Generating Figure 1: Structure Overview...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 7XG4 chain sizes
chains_7xg4 = summary['query']['chain_details']
chain_ids = sorted(chains_7xg4.keys())
sizes = [chains_7xg4[c]['n_residues'] for c in chain_ids]
types = [chains_7xg4[c]['type'] for c in chain_ids]
colors = ['#2196F3' if t == 'protein' else '#FF9800' for t in types]

bars = axes[0].bar(chain_ids, sizes, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Chain ID', fontsize=12)
axes[0].set_ylabel('Number of Residues', fontsize=12)
axes[0].set_title('7XG4 - Type IV-A CRISPR-Cas Complex\n(P. aeruginosa)', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=10)

# Add molecule labels
molecule_map = {
    'A': 'Csf1', 'B': 'Csf3', 'C': 'Csf2', 'D': 'Csf2', 'E': 'Csf2',
    'F': 'Csf2', 'G': 'Csf2', 'H': 'Csf5', 'I': 'crRNA', 'J': 'NTS',
    'K': 'TS', 'L': 'Csf4'
}
for bar, cid in zip(bars, chain_ids):
    if cid in molecule_map:
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    molecule_map[cid], ha='center', va='bottom', fontsize=7, rotation=45)

protein_patch = mpatches.Patch(color='#2196F3', label='Protein')
na_patch = mpatches.Patch(color='#FF9800', label='Nucleic Acid')
axes[0].legend(handles=[protein_patch, na_patch], loc='upper right')

# 6N40 chain size
chains_6n40 = summary['target']['chain_details']
chain_ids_6n40 = sorted(chains_6n40.keys())
sizes_6n40 = [chains_6n40[c]['n_residues'] for c in chain_ids_6n40]
axes[1].bar(chain_ids_6n40, sizes_6n40, color='#4CAF50', edgecolor='black', linewidth=0.5, width=0.4)
axes[1].set_xlabel('Chain ID', fontsize=12)
axes[1].set_ylabel('Number of Residues', fontsize=12)
axes[1].set_title('6N40 - MmpL3 Membrane Protein\n(M. smegmatis)', fontsize=13, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=10)
axes[1].text(0, sizes_6n40[0] + 10, f'726 residues\nMmpL family', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure1_structure_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure1_structure_overview.png")

# ============ Figure 2: TM-score Heatmap (Intra-complex) ============
print("Generating Figure 2: Intra-complex TM-score Heatmap...")

chain_ids_intra = intra['chain_ids']
tm_matrix = np.array(intra['tm_score_matrix'])

fig, ax = plt.subplots(figsize=(10, 8))

# Custom colormap
cmap = LinearSegmentedColormap.from_list('tm_cmap', ['#FFFFFF', '#BBDEFB', '#2196F3', '#1565C0', '#0D47A1'])

mask = np.zeros_like(tm_matrix, dtype=bool)
np.fill_diagonal(mask, True)

sns.heatmap(tm_matrix, annot=True, fmt='.3f', cmap=cmap, vmin=0, vmax=1,
            xticklabels=chain_ids_intra, yticklabels=chain_ids_intra,
            mask=mask, ax=ax, linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'TM-score', 'shrink': 0.8},
            annot_kws={'size': 8})

ax.set_title('Intra-complex Chain TM-score Matrix (7XG4)', fontsize=14, fontweight='bold')
ax.set_xlabel('Chain', fontsize=12)
ax.set_ylabel('Chain', fontsize=12)

# Highlight CSF2 subunits
for i, cid in enumerate(chain_ids_intra):
    if cid in ['C', 'D', 'E', 'F', 'G']:
        ax.get_xticklabels()[i].set_color('red')
        ax.get_yticklabels()[i].set_color('red')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure2_intra_tm_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure2_intra_tm_heatmap.png")

# ============ Figure 3: Cross-complex TM-scores ============
print("Generating Figure 3: Cross-complex TM-scores...")

fig, ax = plt.subplots(figsize=(10, 6))

# Get all pairwise TM-scores between 7XG4 chains and 6N40 chain A
pair_data = alignment['pairwise_details']
chain_labels = []
tm_scores_norm1 = []
tm_scores_norm2 = []
tm_scores_avg = []

for key in sorted(pair_data.keys()):
    chain_labels.append(key.split('-')[0])
    tm_scores_norm1.append(pair_data[key]['tm_score_norm1'])
    tm_scores_norm2.append(pair_data[key]['tm_score_norm2'])
    tm_scores_avg.append(pair_data[key]['tm_score_avg'])

x = np.arange(len(chain_labels))
width = 0.25

bars1 = ax.bar(x - width, tm_scores_norm1, width, label='TM-score (norm by query)', color='#2196F3', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, tm_scores_norm2, width, label='TM-score (norm by target)', color='#FF9800', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, tm_scores_avg, width, label='TM-score (average)', color='#4CAF50', edgecolor='black', linewidth=0.5)

ax.set_xlabel('7XG4 Chain', fontsize=12)
ax.set_ylabel('TM-score', fontsize=12)
ax.set_title('Structural Alignment: 7XG4 Chains vs 6N40 Chain A', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(chain_labels)
ax.legend(fontsize=10)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Significance threshold')
ax.axhline(y=0.17, color='gray', linestyle=':', alpha=0.5, label='Random similarity')
ax.set_ylim(0, 0.6)

# Add text annotations
ax.text(len(chain_labels) - 0.5, 0.51, 'TM > 0.5: same fold', fontsize=9, color='red', alpha=0.7)
ax.text(len(chain_labels) - 0.5, 0.18, 'TM ~ 0.17: random', fontsize=9, color='gray', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure3_cross_complex_tm.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure3_cross_complex_tm.png")

# ============ Figure 4: Chain Features Comparison ============
print("Generating Figure 4: Chain Features Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Separate 7XG4 and 6N40 features
chain_ids_7xg4_feat = [k for k in features.keys() if not k.startswith('6N40')]
chain_ids_6n40_feat = [k for k in features.keys() if k.startswith('6N40')]

all_chain_ids = chain_ids_7xg4_feat + chain_ids_6n40_feat
all_labels = chain_ids_7xg4_feat + [k.replace('6N40_', '6N40:') for k in chain_ids_6n40_feat]
all_colors = ['#2196F3'] * len(chain_ids_7xg4_feat) + ['#4CAF50'] * len(chain_ids_6n40_feat)

# Radius of gyration
rg_vals = [features[k]['radius_of_gyration'] for k in all_chain_ids]
axes[0, 0].bar(range(len(all_labels)), rg_vals, color=all_colors, edgecolor='black', linewidth=0.5)
axes[0, 0].set_xticks(range(len(all_labels)))
axes[0, 0].set_xticklabels(all_labels, rotation=45, ha='right')
axes[0, 0].set_ylabel('Radius of Gyration (Å)')
axes[0, 0].set_title('Radius of Gyration')

# Number of residues
nres_vals = [features[k]['n_residues'] for k in all_chain_ids]
axes[0, 1].bar(range(len(all_labels)), nres_vals, color=all_colors, edgecolor='black', linewidth=0.5)
axes[0, 1].set_xticks(range(len(all_labels)))
axes[0, 1].set_xticklabels(all_labels, rotation=45, ha='right')
axes[0, 1].set_ylabel('Number of Residues')
axes[0, 1].set_title('Chain Length')

# Contact density
cd_vals = [features[k]['contact_density'] for k in all_chain_ids]
axes[1, 0].bar(range(len(all_labels)), cd_vals, color=all_colors, edgecolor='black', linewidth=0.5)
axes[1, 0].set_xticks(range(len(all_labels)))
axes[1, 0].set_xticklabels(all_labels, rotation=45, ha='right')
axes[1, 0].set_ylabel('Contact Density')
axes[1, 0].set_title('Intra-chain Contact Density (8Å cutoff)')

# End-to-end distance
e2e_vals = [features[k]['end_to_end_distance'] for k in all_chain_ids]
axes[1, 1].bar(range(len(all_labels)), e2e_vals, color=all_colors, edgecolor='black', linewidth=0.5)
axes[1, 1].set_xticks(range(len(all_labels)))
axes[1, 1].set_xticklabels(all_labels, rotation=45, ha='right')
axes[1, 1].set_ylabel('End-to-End Distance (Å)')
axes[1, 1].set_title('End-to-End Distance')

# Add legend
blue_patch = mpatches.Patch(color='#2196F3', label='7XG4')
green_patch = mpatches.Patch(color='#4CAF50', label='6N40')
fig.legend(handles=[blue_patch, green_patch], loc='upper center', ncol=2, fontsize=12)

plt.suptitle('Structural Features Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure4_chain_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure4_chain_features.png")

# ============ Figure 5: Inter-chain Contact Map ============
print("Generating Figure 5: Inter-chain Contact Map...")

# Build contact matrix
all_7xg4_chains = sorted(summary['query']['chain_details'].keys())
n_chains = len(all_7xg4_chains)
contact_matrix = np.zeros((n_chains, n_chains))

for key, val in contacts.items():
    c1, c2 = key.split('-')
    if c1 in all_7xg4_chains and c2 in all_7xg4_chains:
        i = all_7xg4_chains.index(c1)
        j = all_7xg4_chains.index(c2)
        contact_matrix[i, j] = val['n_contacts']
        contact_matrix[j, i] = val['n_contacts']

fig, ax = plt.subplots(figsize=(10, 8))

# Use log scale for better visualization
contact_display = np.where(contact_matrix > 0, np.log10(contact_matrix + 1), 0)

sns.heatmap(contact_display, annot=contact_matrix.astype(int), fmt='d',
            cmap='YlOrRd', xticklabels=all_7xg4_chains, yticklabels=all_7xg4_chains,
            ax=ax, linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'log10(contacts + 1)', 'shrink': 0.8},
            annot_kws={'size': 9})

ax.set_title('Inter-chain Contact Map (7XG4)\nCα-Cα distance < 8Å', fontsize=14, fontweight='bold')
ax.set_xlabel('Chain', fontsize=12)
ax.set_ylabel('Chain', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure5_contact_map.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure5_contact_map.png")

# ============ Figure 6: 3Di Structural Alphabet Composition ============
print("Generating Figure 6: 3Di Structural Alphabet Composition...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

alphabet = 'ACDEFGHIKLMNPQRSTVWY'

for idx, (struct_name, struct_data) in enumerate(threedi.items()):
    ax = axes[idx]
    chain_ids_3di = sorted(struct_data.keys())
    
    compositions = {}
    for cid in chain_ids_3di:
        seq = struct_data[cid]['3di_sequence']
        comp = {c: seq.count(c) / len(seq) * 100 for c in alphabet}
        compositions[cid] = comp
    
    # Stacked bar chart
    bottom = np.zeros(len(chain_ids_3di))
    colors_3di = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for i, letter in enumerate(alphabet):
        values = [compositions[cid].get(letter, 0) for cid in chain_ids_3di]
        ax.bar(chain_ids_3di, values, bottom=bottom, label=letter, color=colors_3di[i], width=0.6)
        bottom += values
    
    ax.set_xlabel('Chain', fontsize=12)
    ax.set_ylabel('Composition (%)', fontsize=12)
    ax.set_title(f'{struct_name} - 3Di Alphabet Composition', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    
    if idx == 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2, title='3Di State')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure6_3di_composition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure6_3di_composition.png")

# ============ Figure 7: Superimposition Visualization ============
print("Generating Figure 7: Superimposition Visualization...")

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

parser = PDBParser(QUIET=True)
struct1 = parser.get_structure('7xg4', os.path.join(WORKSPACE, 'data', '7xg4.pdb'))
struct2 = parser.get_structure('6n40', os.path.join(WORKSPACE, 'data', '6n40.pdb'))

# Get CA coordinates for best matching chains
best_pair = alignment['best_chain_pair']
R = np.array(alignment['superimposition']['rotation_matrix'])
t = np.array(alignment['superimposition']['translation_vector'])

# Get coords for 7XG4 chain A (best match)
chain_7xg4_A = struct1[0]['A']
coords_7xg4 = []
for res in chain_7xg4_A.get_residues():
    if is_aa(res, standard=True) and 'CA' in res:
        coords_7xg4.append(res['CA'].get_vector().get_array())
coords_7xg4 = np.array(coords_7xg4)

# Get coords for 6N40 chain A
chain_6n40_A = struct2[0]['A']
coords_6n40 = []
for res in chain_6n40_A.get_residues():
    if is_aa(res, standard=True) and 'CA' in res:
        coords_6n40.append(res['CA'].get_vector().get_array())
coords_6n40 = np.array(coords_6n40)

# Apply superimposition to 7XG4
coords_7xg4_aligned = (R @ coords_7xg4.T).T + t

fig = plt.figure(figsize=(16, 6))

# 2D projections
projections = [
    ('XY', 0, 1, 'X (Å)', 'Y (Å)'),
    ('XZ', 0, 2, 'X (Å)', 'Z (Å)'),
    ('YZ', 1, 2, 'Y (Å)', 'Z (Å)'),
]

for idx, (name, d1, d2, xlabel, ylabel) in enumerate(projections):
    ax = fig.add_subplot(1, 3, idx + 1)
    ax.scatter(coords_7xg4_aligned[:, d1], coords_7xg4_aligned[:, d2], 
              s=3, alpha=0.5, c='#2196F3', label='7XG4 Chain A (aligned)')
    ax.scatter(coords_6n40[:, d1], coords_6n40[:, d2], 
              s=3, alpha=0.5, c='#4CAF50', label='6N40 Chain A')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'{name} Projection', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, markerscale=3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Structural Superimposition (TM-score = {best_pair["tm_score"]:.4f})',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure7_superimposition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure7_superimposition.png")

# ============ Figure 8: CSF2 Homolog Cluster Analysis ============
print("Generating Figure 8: CSF2 Homolog Cluster...")

# CSF2 chains are C, D, E, F, G
csf2_chains = ['C', 'D', 'E', 'F', 'G']
csf2_idx = [intra['chain_ids'].index(c) for c in csf2_chains]
csf2_matrix = np.array(intra['tm_score_matrix'])[np.ix_(csf2_idx, csf2_idx)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap
sns.heatmap(csf2_matrix, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0.8, vmax=1.0,
            xticklabels=csf2_chains, yticklabels=csf2_chains,
            ax=axes[0], linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'TM-score', 'shrink': 0.8},
            annot_kws={'size': 11})
axes[0].set_title('CSF2 Subunit TM-score Matrix', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Chain', fontsize=12)
axes[0].set_ylabel('Chain', fontsize=12)

# Dendrogram-like visualization using distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Convert TM-scores to distances
dist_matrix = 1 - csf2_matrix
np.fill_diagonal(dist_matrix, 0)
dist_matrix_sym = (dist_matrix + dist_matrix.T) / 2
condensed = squareform(dist_matrix_sym)
Z = linkage(condensed, method='average')

dendrogram(Z, labels=csf2_chains, ax=axes[1], color_threshold=0.05,
          above_threshold_color='#2196F3')
axes[1].set_title('CSF2 Subunit Hierarchical Clustering', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Chain', fontsize=12)
axes[1].set_ylabel('1 - TM-score (distance)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure8_csf2_cluster.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure8_csf2_cluster.png")

# ============ Figure 9: Secondary Structure Composition ============
print("Generating Figure 9: Secondary Structure Composition...")

fig, ax = plt.subplots(figsize=(12, 6))

all_struct_chains = []
helix_fracs = []
strand_fracs = []
coil_fracs = []
bar_colors = []

for struct_name, struct_data in threedi.items():
    for cid in sorted(struct_data.keys()):
        ss = struct_data[cid]['ss_composition']
        all_struct_chains.append(f"{struct_name}:{cid}")
        helix_fracs.append(ss['helix'] * 100)
        strand_fracs.append(ss['strand'] * 100)
        coil_fracs.append(ss['coil'] * 100)
        bar_colors.append('#2196F3' if struct_name == '7XG4' else '#4CAF50')

x = np.arange(len(all_struct_chains))
width = 0.25

ax.bar(x - width, helix_fracs, width, label='α-Helix', color='#E53935', edgecolor='black', linewidth=0.5)
ax.bar(x, strand_fracs, width, label='β-Strand', color='#FDD835', edgecolor='black', linewidth=0.5)
ax.bar(x + width, coil_fracs, width, label='Coil', color='#78909C', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Chain', fontsize=12)
ax.set_ylabel('Fraction (%)', fontsize=12)
ax.set_title('Secondary Structure Composition', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(all_struct_chains, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure9_ss_composition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure9_ss_composition.png")

# ============ Figure 10: Distance Distribution ============
print("Generating Figure 10: Distance Distribution after alignment...")

from Bio.PDB.Polypeptide import is_aa
import tmtools

# Re-run alignment for best pair to get detailed distances
parser = PDBParser(QUIET=True)
struct1 = parser.get_structure('7xg4', os.path.join(WORKSPACE, 'data', '7xg4.pdb'))
struct2 = parser.get_structure('6n40', os.path.join(WORKSPACE, 'data', '6n40.pdb'))

def get_coords_seq(chain):
    coords = []
    seq = []
    for res in chain.get_residues():
        if is_aa(res, standard=True) and 'CA' in res:
            coords.append(res['CA'].get_vector().get_array())
            try:
                from Bio.Data.IUPACData import protein_letters_3to1
                resname_cap = res.get_resname().strip().capitalize()
                seq.append(protein_letters_3to1.get(resname_cap, 'X'))
            except:
                seq.append('X')
    return np.array(coords), ''.join(seq)

coords1, seq1 = get_coords_seq(struct1[0]['A'])
coords2, seq2 = get_coords_seq(struct2[0]['A'])

result = tmtools.tm_align(coords1, coords2, seq1, seq2)

# Apply transformation
R_best = result.u
t_best = result.t
coords1_aligned = (R_best @ coords1.T).T + t_best

# Get aligned residue pairs from alignment
aln1 = result.seqxA
aln2 = result.seqyA

distances = []
pos1 = 0
pos2 = 0
for a, b in zip(aln1, aln2):
    if a != '-' and b != '-':
        d = np.linalg.norm(coords1_aligned[pos1] - coords2[pos2])
        distances.append(d)
    if a != '-':
        pos1 += 1
    if b != '-':
        pos2 += 1

distances = np.array(distances)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(distances, bins=50, color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.8)
axes[0].axvline(x=np.median(distances), color='red', linestyle='--', label=f'Median: {np.median(distances):.1f}Å')
axes[0].axvline(x=np.mean(distances), color='orange', linestyle='--', label=f'Mean: {np.mean(distances):.1f}Å')
axes[0].set_xlabel('Cα-Cα Distance (Å)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distance Distribution of Aligned Residues', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)

# Cumulative distribution
sorted_dist = np.sort(distances)
cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
axes[1].plot(sorted_dist, cdf, color='#2196F3', linewidth=2)
axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
axes[1].axvline(x=5.0, color='red', linestyle='--', alpha=0.5, label='5Å threshold')
frac_under_5 = np.sum(distances < 5.0) / len(distances) * 100
axes[1].text(5.5, 0.3, f'{frac_under_5:.1f}% < 5Å', fontsize=10, color='red')
axes[1].set_xlabel('Cα-Cα Distance (Å)', fontsize=12)
axes[1].set_ylabel('Cumulative Fraction', fontsize=12)
axes[1].set_title('Cumulative Distance Distribution', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)

plt.suptitle(f'Alignment Quality: 7XG4 Chain A vs 6N40 Chain A\n(TM-score = {(result.tm_norm_chain1 + result.tm_norm_chain2)/2:.4f}, {len(distances)} aligned residues)',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'figure10_distance_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure10_distance_distribution.png")

print("\nAll figures generated successfully!")
print(f"Figures saved to: {IMAGE_DIR}")
