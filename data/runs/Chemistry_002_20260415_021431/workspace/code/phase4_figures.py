"""
Phase 4: Generate all figures for the report
"""
import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# Create output directory
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
with open('outputs/interface_analysis.json', 'r') as f:
    interface_data = json.load(f)

with open('outputs/pdb_analysis.json', 'r') as f:
    pdb_stats = json.load(f)

with open('outputs/validation_results.json', 'r') as f:
    validation_results = json.load(f)

with open('outputs/mutation_details.json', 'r') as f:
    mutation_details = json.load(f)

skempi_df = pd.read_csv('outputs/barnase_barstar_skempi.csv', sep=',')

# Parse PDB file for structure visualization
def parse_pdb(pdb_path):
    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                record = {
                    'serial': int(line[6:11]),
                    'name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21],
                    'res_seq': int(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                }
                atoms.append(record)
    return atoms

atoms = parse_pdb('data/1brs_AD.pdb')

# Calculate dDeltaG for SKEMPI entries
R = 1.987
T = 298.15

def calculate_ddg(affinity_mut, affinity_wt):
    try:
        kd_mut = float(affinity_mut)
        kd_wt = float(affinity_wt)
        if kd_mut > 0 and kd_wt > 0:
            return R * T * np.log(kd_mut / kd_wt) / 1000
    except:
        pass
    return np.nan

skempi_df['ddG_kcal_mol'] = skempi_df.apply(
    lambda row: calculate_ddg(row['Affinity_mut_parsed'], row['Affinity_wt_parsed']), axis=1
)

# Get interface residues set
interface_res_set = set()
for pair in interface_data['interface_pairs']:
    res1 = pair['res1']
    res2 = pair['res2']
    interface_res_set.add((res1[0], res1[1]))
    interface_res_set.add((res2[0], res2[1]))

# ============================================================
# Figure 1: Structure Overview - Chain positions and interface
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Extract coordinates by chain
chain_a = [a for a in atoms if a['chain'] == 'A']
chain_d = [a for a in atoms if a['chain'] == 'D']

# Plot chain A (barnase)
ax = axes[0]
ca_x = [a['x'] for a in chain_a]
ca_y = [a['y'] for a in chain_a]
ca_z = [a['z'] for a in chain_a]

# Color by residue type
colors_by_res = {'positive': '#FF6B6B', 'negative': '#4ECDC4', 'polar': '#95E1D3', 'hydrophobic': '#F38181'}
residue_types = {
    'ARG': 'positive', 'LYS': 'positive', 'HIS': 'positive',
    'ASP': 'negative', 'GLU': 'negative',
    'SER': 'polar', 'THR': 'polar', 'ASN': 'polar', 'GLN': 'polar', 'TYR': 'polar',
    'ALA': 'hydrophobic', 'VAL': 'hydrophobic', 'LEU': 'hydrophobic', 'ILE': 'hydrophobic',
    'MET': 'hydrophobic', 'PHE': 'hydrophobic', 'TRP': 'hydrophobic', 'PRO': 'hydrophobic',
    'GLY': 'polar', 'CYS': 'hydrophobic'
}

for res_type, color in colors_by_res.items():
    mask = [residue_types.get(a['res_name'], 'polar') == res_type for a in chain_a]
    ax.scatter([a['x'] for i, a in enumerate(chain_a) if mask[i]],
               [a['y'] for i, a in enumerate(chain_a) if mask[i]],
               c=color, s=3, alpha=0.6, label=res_type)

ax.set_title('Barnase (Chain A)\nColored by Residue Type', fontsize=12, fontweight='bold')
ax.set_xlabel('X (Å)', fontsize=10)
ax.set_ylabel('Y (Å)', fontsize=10)
ax.legend(loc='upper right', fontsize=8, markerscale=3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Plot chain D (barstar)
ax = axes[1]
for res_type, color in colors_by_res.items():
    mask = [residue_types.get(a['res_name'], 'polar') == res_type for a in chain_d]
    ax.scatter([a['x'] for i, a in enumerate(chain_d) if mask[i]],
               [a['y'] for i, a in enumerate(chain_d) if mask[i]],
               c=color, s=3, alpha=0.6, label=res_type)

ax.set_title('Barstar (Chain D)\nColored by Residue Type', fontsize=12, fontweight='bold')
ax.set_xlabel('X (Å)', fontsize=10)
ax.set_ylabel('Y (Å)', fontsize=10)
ax.legend(loc='upper right', fontsize=8, markerscale=3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure1_structure_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: structure overview")

# ============================================================
# Figure 2: Interface Contact Map
# ============================================================
# Build contact map between chains
chain_a_residues = sorted(set((a['res_name'], a['res_seq']) for a in atoms if a['chain'] == 'A'))
chain_d_residues = sorted(set((a['res_name'], a['res_seq']) for a in atoms if a['chain'] == 'D'))

# Create contact matrix
contact_matrix = np.zeros((len(chain_a_residues), len(chain_d_residues)))

for a_atom in atoms:
    if a_atom['chain'] != 'A':
        continue
    for d_atom in atoms:
        if d_atom['chain'] != 'D':
            continue
        dist = np.sqrt((a_atom['x']-d_atom['x'])**2 + (a_atom['y']-d_atom['y'])**2 + (a_atom['z']-d_atom['z'])**2)
        if dist < 5.0:
            # Find indices
            a_idx = next((i for i, r in enumerate(chain_a_residues) if r[1] == a_atom['res_seq']), None)
            d_idx = next((i for i, r in enumerate(chain_d_residues) if r[1] == a_atom['res_seq']), None)
            if a_idx is not None and d_idx is not None:
                contact_matrix[a_idx][d_idx] = max(contact_matrix[a_idx][d_idx], 1)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(contact_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Barstar (Chain D) Residue Index', fontsize=11)
ax.set_ylabel('Barnase (Chain A) Residue Index', fontsize=11)
ax.set_title('Barnase-Barstar Interface Contact Map\n(Distance cutoff: 5.0 Å)', fontsize=12, fontweight='bold')

# Add labels for interface regions
ax.axhline(y=-0.5, color='red', linewidth=2, linestyle='--', alpha=0.5)
ax.axvline(x=-0.5, color='red', linewidth=2, linestyle='--', alpha=0.5)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Contact (1 = atoms within 5.0 Å)', fontsize=10)

plt.tight_layout()
plt.savefig('report/images/figure2_contact_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: contact map")

# ============================================================
# Figure 3: SKEMPI dDeltaG Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall distribution
ax = axes[0]
ddg_values = skempi_df['ddG_kcal_mol'].dropna().values
ax.hist(ddg_values, bins=30, color='#2196F3', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect (ΔΔG = 0)')
ax.axvline(x=np.mean(ddg_values), color='green', linestyle='-', linewidth=2, 
           label=f'Mean ΔΔG = {np.mean(ddg_values):.2f} kcal/mol')
ax.set_xlabel('ΔΔG (kcal/mol)', fontsize=11)
ax.set_ylabel('Number of Mutations', fontsize=11)
ax.set_title('Distribution of Binding Energy Changes\nUpon Mutation (Barnase-Barstar)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Interface vs Non-interface
ax = axes[1]
import re

def parse_mutation(mut_str):
    if ',' in str(mut_str):
        return [parse_single_mutation(m) for m in str(mut_str).split(',')]
    return [parse_single_mutation(mut_str)]

def parse_single_mutation(mut_str):
    mut_str = str(mut_str).strip()
    match = re.match(r'([A-Z])([A-Z])(\d+)([A-Z])', mut_str)
    if match:
        return {'wt_res': match.group(1), 'chain': match.group(2), 
                'res_seq': int(match.group(3)), 'mut_res': match.group(4)}
    return None

def is_interface_mutation(mutations_list):
    for m in mutations_list:
        if m and (m['chain'], m['res_seq']) in interface_res_set:
            return True
    return False

unique_entries = skempi_df.drop_duplicates(subset=['Mutation(s)_cleaned'])
interface_mask = unique_entries['Mutation(s)_cleaned'].apply(
    lambda x: is_interface_mutation(parse_mutation(x))
)

interface_ddg = unique_entries[interface_mask]['ddG_kcal_mol'].dropna().values
non_interface_ddg = unique_entries[~interface_mask]['ddG_kcal_mol'].dropna().values

ax.hist(interface_ddg, bins=20, color='#FF6B6B', edgecolor='white', alpha=0.7, label=f'Interface (n={len(interface_ddg)})')
ax.hist(non_interface_ddg, bins=20, color='#4ECDC4', edgecolor='white', alpha=0.7, label=f'Non-interface (n={len(non_interface_ddg)})')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No effect')
ax.set_xlabel('ΔΔG (kcal/mol)', fontsize=11)
ax.set_ylabel('Number of Mutations', fontsize=11)
ax.set_title('ΔΔG Distribution: Interface vs Non-Interface Mutations', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure3_ddg_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: dDeltaG distribution")

# ============================================================
# Figure 4: Interface Residue Analysis - Hotspot Identification
# ============================================================
# Calculate per-residue interaction energy proxy (number of contacts)
res_contacts = defaultdict(int)
for a_atom in atoms:
    if a_atom['chain'] != 'A':
        continue
    for d_atom in atoms:
        if d_atom['chain'] != 'D':
            continue
        dist = np.sqrt((a_atom['x']-d_atom['x'])**2 + (a_atom['y']-d_atom['y'])**2 + (a_atom['z']-d_atom['z'])**2)
        if dist < 4.0:
            res_contacts[(a_atom['res_name'], a_atom['res_seq'])] += 1

# Sort by contact count
sorted_contacts = sorted(res_contacts.items(), key=lambda x: x[1], reverse=True)
top_residues = sorted_contacts[:15]

fig, ax = plt.subplots(figsize=(12, 6))
res_labels = [f"{r[0]}{r[1]}" for r, c in top_residues]
contact_counts = [c for r, c in top_residues]

colors = ['#FF6B6B' if c > 20 else '#FFA07A' if c > 10 else '#FFD93D' for c in contact_counts]
bars = ax.barh(range(len(res_labels)), contact_counts, color=colors, edgecolor='white')

ax.set_yticks(range(len(res_labels)))
ax.set_yticklabels(res_labels, fontsize=10)
ax.set_xlabel('Number of Atomic Contacts (< 4.0 Å)', fontsize=11)
ax.set_title('Top Interface Residues by Contact Count\n(Barnase Chain A → Barstar Chain D)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, contact_counts)):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, str(val), 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/figure4_interface_hotspots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: interface hotspots")

# ============================================================
# Figure 5: Mutation Effect vs Structural Position
# ============================================================
# For each mutation, get average ddG and compare with distance to interface center
unique_entries = skempi_df.drop_duplicates(subset=['Mutation(s)_cleaned'])

# Calculate centroid of interface
interface_coords = []
for atom in atoms:
    key = (atom['chain'], atom['res_seq'])
    if key in interface_res_set:
        interface_coords.append([atom['x'], atom['y'], atom['z']])

if interface_coords:
    interface_centroid = np.mean(interface_coords, axis=0)
else:
    interface_centroid = np.array([0, 0, 0])

# For each mutated residue, calculate distance to interface centroid
mutation_distances = []
for idx, row in unique_entries.iterrows():
    mut_str = row['Mutation(s)_cleaned']
    parsed = parse_mutation(mut_str)
    ddg = row['ddG_kcal_mol']
    
    if parsed and parsed[0] and not np.isnan(ddg):
        for m in parsed:
            if m:
                # Find coordinates of this residue
                res_atoms = [a for a in atoms if a['chain'] == m['chain'] and a['res_seq'] == m['res_seq']]
                if res_atoms:
                    res_centroid = np.mean([[a['x'], a['y'], a['z']] for a in res_atoms], axis=0)
                    dist = np.linalg.norm(res_centroid - interface_centroid)
                    mutation_distances.append({
                        'mutation': mut_str,
                        'ddG': ddg,
                        'distance_to_interface': dist,
                        'is_interface': (m['chain'], m['res_seq']) in interface_res_set
                    })

mut_df = pd.DataFrame(mutation_distances)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot interface vs non-interface
interface_mask = mut_df['is_interface']
ax.scatter(mut_df[~interface_mask]['distance_to_interface'], 
           mut_df[~interface_mask]['ddG'],
           c='#4ECDC4', s=50, alpha=0.7, label='Non-interface', edgecolors='white', linewidth=0.5)
ax.scatter(mut_df[interface_mask]['distance_to_interface'], 
           mut_df[interface_mask]['ddG'],
           c='#FF6B6B', s=80, alpha=0.8, label='Interface', edgecolors='white', linewidth=0.5, marker='^')

# Add trend line
from scipy import stats
if len(mut_df) > 2:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        mut_df['distance_to_interface'], mut_df['ddG'])
    x_line = np.linspace(mut_df['distance_to_interface'].min(), 
                         mut_df['distance_to_interface'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k--', alpha=0.5, 
            label=f'Trend (r={r_value:.3f}, p={p_value:.4f})')

ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Distance from Interface Centroid (Å)', fontsize=11)
ax.set_ylabel('ΔΔG (kcal/mol)', fontsize=11)
ax.set_title('Mutation Effect vs Distance from Interface\nBarnase-Barstar Complex', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure5_mutation_vs_distance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: mutation vs distance")

# ============================================================
# Figure 6: HADDOCK Scoring Function Components Analysis
# ============================================================
# Analyze the relationship between residue properties and mutation effects
# Group mutations by residue type

aa_properties = {
    'ALA': ('Small', 'Hydrophobic'), 'ARG': ('Large', 'Charged+'),
    'ASN': ('Medium', 'Polar'), 'ASP': ('Medium', 'Charged-'),
    'CYS': ('Medium', 'Hydrophobic'), 'GLN': ('Large', 'Polar'),
    'GLU': ('Large', 'Charged-'), 'GLY': ('Small', 'Hydrophobic'),
    'HIS': ('Medium', 'Charged+'), 'ILE': ('Medium', 'Hydrophobic'),
    'LEU': ('Medium', 'Hydrophobic'), 'LYS': ('Large', 'Charged+'),
    'MET': ('Large', 'Hydrophobic'), 'PHE': ('Large', 'Hydrophobic'),
    'PRO': ('Medium', 'Hydrophobic'), 'SER': ('Small', 'Polar'),
    'THR': ('Medium', 'Polar'), 'TRP': ('Large', 'Hydrophobic'),
    'TYR': ('Large', 'Polar'), 'VAL': ('Medium', 'Hydrophobic')
}

# Categorize mutations
mutation_effects = []
for idx, row in unique_entries.iterrows():
    mut_str = row['Mutation(s)_cleaned']
    parsed = parse_mutation(mut_str)
    ddg = row['ddG_kcal_mol']
    
    if parsed and parsed[0] and not np.isnan(ddg):
        m = parsed[0]
        wt_prop = aa_properties.get(m['wt_res'], ('Unknown', 'Unknown'))
        mut_prop = aa_properties.get(m['mut_res'], ('Unknown', 'Unknown'))
        
        mutation_effects.append({
            'mutation': mut_str,
            'ddG': ddg,
            'wt_res': m['wt_res'],
            'mut_res': m['mut_res'],
            'wt_size': wt_prop[0],
            'wt_charge': wt_prop[1],
            'is_interface': (m['chain'], m['res_seq']) in interface_res_set
        })

me_df = pd.DataFrame(mutation_effects)

# Box plot by charge type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By wild-type charge
ax = axes[0]
charge_order = ['Charged+', 'Charged-', 'Polar', 'Hydrophobic']
charge_data = [me_df[me_df['wt_charge'] == c]['ddG'].dropna().values for c in charge_order]
bp = ax.boxplot(charge_data, labels=charge_order, patch_artist=True, notch=True)

colors_box = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('ΔΔG (kcal/mol)', fontsize=11)
ax.set_title('Mutation Effects by Wild-Type\nResidue Charge Type', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# By mutation location (interface vs non-interface)
ax = axes[1]
loc_data = [
    me_df[me_df['is_interface']]['ddG'].dropna().values,
    me_df[~me_df['is_interface']]['ddG'].dropna().values
]
bp2 = ax.boxplot(loc_data, labels=['Interface', 'Non-interface'], 
                 patch_artist=True, notch=True)

for patch, color in zip(bp2['boxes'], ['#FF6B6B', '#4ECDC4']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('ΔΔG (kcal/mol)', fontsize=11)
ax.set_title('Mutation Effects by\nStructural Location', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('report/images/figure6_residue_properties.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: residue properties")

# ============================================================
# Save summary statistics
# ============================================================
summary_stats = {
    'pdb_info': pdb_stats,
    'interface_info': {
        'n_pairs': interface_data['n_pairs'],
        'n_unique_residues': interface_data['n_unique_residues'],
        'atomic_contacts': pdb_stats['atomic_contacts']
    },
    'skempi_info': {
        'total_entries': len(skempi_df),
        'unique_mutations': len(unique_entries),
        'ddg_mean': float(np.mean(ddg_values)),
        'ddg_std': float(np.std(ddg_values)),
        'interface_mutations': int(interface_mask.sum()),
        'non_interface_mutations': int((~interface_mask).sum())
    }
}

with open('outputs/summary_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\nAll figures generated successfully!")
print("Figures saved to report/images/")
