#!/usr/bin/env python3
"""
Barnase-Barstar Complex Analysis: Integrating Structural Data with SKEMPI v2 Binding Affinity Data
for HADDOCK3-Compatible Interface Characterization

This script analyzes the barnase-barstar complex structure (PDB: 1BRS) and 
correlates structural interface properties with experimental binding affinity 
changes from the SKEMPI v2 database.
"""

import math
import csv
import json
import os
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# ============================================================
# 1. Parse PDB structure
# ============================================================
def parse_pdb(filepath):
    """Parse PDB file and return atom records organized by chain."""
    atoms = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21]
                resname = line[17:20].strip()
                resseq = int(line[22:26].strip())
                atomname = line[12:16].strip()
                altloc = line[16]
                if altloc not in (' ', 'A'):
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append({
                    'chain': chain,
                    'resname': resname,
                    'resseq': resseq,
                    'atomname': atomname,
                    'x': x, 'y': y, 'z': z
                })
    return atoms

def compute_interface(atoms, threshold=5.0):
    """Compute interface residues between chains A and D."""
    atoms_A = [a for a in atoms if a['chain'] == 'A']
    atoms_D = [a for a in atoms if a['chain'] == 'D']
    
    interface_A = defaultdict(int)
    interface_D = defaultdict(int)
    contacts = []
    
    for a in atoms_A:
        for d in atoms_D:
            dist = math.sqrt((a['x']-d['x'])**2 + (a['y']-d['y'])**2 + (a['z']-d['z'])**2)
            if dist < threshold:
                key_A = f"{a['resname']}{a['resseq']}"
                key_D = f"{d['resname']}{d['resseq']}"
                interface_A[key_A] += 1
                interface_D[key_D] += 1
                contacts.append({
                    'resA': key_A, 'resD': key_D,
                    'dist': dist,
                    'atomA': a['atomname'], 'atomD': d['atomname']
                })
    
    return interface_A, interface_D, contacts

def compute_distance_matrix(atoms):
    """Compute CA-CA distance matrix between chains."""
    ca_A = [(a['resseq'], a['x'], a['y'], a['z']) for a in atoms 
            if a['chain'] == 'A' and a['atomname'] == 'CA']
    ca_D = [(a['resseq'], a['x'], a['y'], a['z']) for a in atoms 
            if a['chain'] == 'D' and a['atomname'] == 'CA']
    
    ca_A.sort(key=lambda x: x[0])
    ca_D.sort(key=lambda x: x[0])
    
    dist_matrix = np.zeros((len(ca_A), len(ca_D)))
    for i, (_, ax, ay, az) in enumerate(ca_A):
        for j, (_, dx, dy, dz) in enumerate(ca_D):
            dist_matrix[i, j] = math.sqrt((ax-dx)**2 + (ay-dy)**2 + (az-dz)**2)
    
    return dist_matrix, [c[0] for c in ca_A], [c[0] for c in ca_D]

# ============================================================
# 2. Parse SKEMPI data
# ============================================================
def parse_skempi(filepath, pdb_filter='1BRS'):
    """Parse SKEMPI v2 CSV and filter for specific PDB entries."""
    entries = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if pdb_filter.upper() in row['#Pdb'].upper():
                entries.append(row)
    return entries

def compute_ddg(entries):
    """Compute binding free energy changes from Kd values."""
    R = 1.987e-3  # kcal/(mol*K)
    T = 298.0  # K
    
    results = []
    for e in entries:
        kd_mut = float(e['Affinity_mut_parsed']) if e['Affinity_mut_parsed'] else None
        kd_wt = float(e['Affinity_wt_parsed']) if e['Affinity_wt_parsed'] else None
        
        if kd_mut and kd_wt and kd_mut > 0 and kd_wt > 0:
            ddG = R * T * math.log(kd_mut / kd_wt)
            
            # Parse mutation info
            mut_str = e['Mutation(s)_cleaned']
            location = e['iMutation_Location(s)']
            
            # Count number of mutations
            n_mutations = mut_str.count(',') + 1 if mut_str else 0
            
            results.append({
                'mutation': mut_str,
                'location': location,
                'n_mutations': n_mutations,
                'ddG': ddG,
                'kd_mut': kd_mut,
                'kd_wt': kd_wt,
                'ddG_kcal': ddG,
                'method': e.get('Method', ''),
            })
    
    return results

# ============================================================
# 3. Analysis functions
# ============================================================

def classify_mutation_effect(ddG):
    """Classify mutation effect based on ddG threshold."""
    if ddG < -0.5:
        return 'Stabilizing'
    elif ddG > 0.5:
        return 'Destabilizing'
    else:
        return 'Neutral'

def compute_salt_bridges(atoms, threshold=4.0):
    """Identify potential salt bridges at the interface."""
    charged_A = {'ARG': ['NH1', 'NH2', 'NZ'], 'LYS': ['NZ'], 
                 'ASP': ['OD1', 'OD2'], 'GLU': ['OE1', 'OE2'],
                 'HIS': ['ND1', 'NE2']}
    charged_D = charged_A
    
    salt_bridges = []
    atoms_A = [a for a in atoms if a['chain'] == 'A']
    atoms_D = [a for a in atoms if a['chain'] == 'D']
    
    for a in atoms_A:
        if a['resname'] in charged_A and a['atomname'] in charged_A[a['resname']]:
            for d in atoms_D:
                if d['resname'] in charged_D and d['atomname'] in charged_D[d['resname']]:
                    dist = math.sqrt((a['x']-d['x'])**2 + (a['y']-d['y'])**2 + (a['z']-d['z'])**2)
                    if dist < threshold:
                        # Check if opposite charges
                        a_pos = a['resname'] in ('ARG', 'LYS', 'HIS')
                        d_pos = d['resname'] in ('ARG', 'LYS', 'HIS')
                        if a_pos != d_pos:  # opposite charges
                            salt_bridges.append({
                                'resA': f"{a['resname']}{a['resseq']}",
                                'resD': f"{d['resname']}{d['resseq']}",
                                'dist': dist,
                                'type': 'salt_bridge'
                            })
    return salt_bridges

def compute_hydrogen_bonds(atoms, threshold=3.5):
    """Identify potential hydrogen bonds at the interface."""
    hbond_donors = {'NH1', 'NH2', 'NZ', 'ND1', 'NE', 'NE2', 'OG', 'OH', 'ND2'}
    hbond_acceptors = {'OD1', 'OD2', 'OE1', 'OE2', 'O', 'OG', 'OH', 'NE2', 'ND1'}
    
    hbonds = []
    atoms_A = [a for a in atoms if a['chain'] == 'A']
    atoms_D = [a for a in atoms if a['chain'] == 'D']
    
    for a in atoms_A:
        if a['atomname'] in hbond_donors or a['atomname'] in hbond_acceptors:
            for d in atoms_D:
                if d['atomname'] in hbond_donors or d['atomname'] in hbond_acceptors:
                    dist = math.sqrt((a['x']-d['x'])**2 + (a['y']-d['y'])**2 + (a['z']-d['z'])**2)
                    if dist < threshold and dist > 1.5:
                        hbonds.append({
                            'resA': f"{a['resname']}{a['resseq']}",
                            'resD': f"{d['resname']}{d['resseq']}",
                            'atomA': a['atomname'],
                            'atomD': d['atomname'],
                            'dist': dist
                        })
    return hbonds


# ============================================================
# Main analysis
# ============================================================

print("=" * 60)
print("BARNASE-BARSTAR COMPLEX ANALYSIS")
print("=" * 60)

# Parse structure
atoms = parse_pdb('data/1brs_AD.pdb')
print(f"\nTotal atoms parsed: {len(atoms)}")

# Compute interface
interface_A, interface_D, contacts_5A = compute_interface(atoms, 5.0)
interface_A_4, interface_D_4, contacts_4A = compute_interface(atoms, 4.0)

print(f"\nInterface residues (5Å): Chain A = {len(interface_A)}, Chain D = {len(interface_D)}")
print(f"Interface residues (4Å): Chain A = {len(interface_A_4)}, Chain D = {len(interface_D_4)}")

# Distance matrix
dist_matrix, res_A, res_D = compute_distance_matrix(atoms)

# Parse SKEMPI
skempi_entries = parse_skempi('data/skempi_v2.csv')
ddg_results = compute_ddg(skempi_entries)
print(f"\nSKEMPI entries for 1BRS: {len(skempi_entries)}")
print(f"Valid ddG entries: {len(ddg_results)}")

# Compute salt bridges and hydrogen bonds
salt_bridges = compute_salt_bridges(atoms)
hbonds = compute_hydrogen_bonds(atoms)
print(f"\nSalt bridges identified: {len(salt_bridges)}")
print(f"Hydrogen bonds identified: {len(hbonds)}")

# ============================================================
# Save structural data
# ============================================================
structural_data = {
    'interface_residues_A': {k: v for k, v in sorted(interface_A.items(), key=lambda x: -x[1])},
    'interface_residues_D': {k: v for k, v in sorted(interface_D.items(), key=lambda x: -x[1])},
    'n_contacts_5A': len(contacts_5A),
    'n_contacts_4A': len(contacts_4A),
    'salt_bridges': salt_bridges,
    'n_hbonds': len(hbonds),
}

with open('outputs/structural_analysis.json', 'w') as f:
    json.dump(structural_data, f, indent=2)

# Save SKEMPI analysis
skempi_analysis = {
    'n_entries': len(ddg_results),
    'ddG_values': [r['ddG'] for r in ddg_results],
    'mutation_effects': [classify_mutation_effect(r['ddG']) for r in ddg_results],
}
with open('outputs/skempi_analysis.json', 'w') as f:
    json.dump(skempi_analysis, f, indent=2)

# ============================================================
# FIGURE 1: Interface Contact Heatmap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Contact heatmap between chains
n_res_A = len(res_A)
n_res_D = len(res_D)
contact_matrix = np.zeros((n_res_A, n_res_D))

for c in contacts_5A:
    resA_idx = res_A.index(int(c['resA'][3:])) if int(c['resA'][3:]) in res_A else -1
    resD_idx = res_D.index(int(c['resD'][3:])) if int(c['resD'][3:]) in res_D else -1
    if resA_idx >= 0 and resD_idx >= 0:
        contact_matrix[resA_idx, resD_idx] = c['dist']

# Replace 0 with NaN for better visualization
contact_matrix[contact_matrix == 0] = np.nan

im = axes[0].imshow(contact_matrix, cmap='RdYlBu_r', aspect='auto', vmin=2, vmax=8)
axes[0].set_xlabel('Barstar residue index')
axes[0].set_ylabel('Barnase residue index')
axes[0].set_title('Inter-chain CA-CA Distance Map')
plt.colorbar(im, ax=axes[0], label='Distance (Å)')

# Distance matrix full
full_dist = dist_matrix.copy()
full_dist[full_dist > 15] = np.nan
im2 = axes[1].imshow(full_dist, cmap='RdYlBu_r', aspect='auto', vmin=3, vmax=20)
axes[1].set_xlabel('Barstar CA index')
axes[1].set_ylabel('Barnase CA index')
axes[1].set_title('Full CA-CA Distance Matrix (Å)')
plt.colorbar(im2, ax=axes[1], label='Distance (Å)')

plt.suptitle('Barnase-Barstar Interface Distance Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig1_distance_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 1 saved: fig1_distance_maps.png")

# ============================================================
# FIGURE 2: Interface Residue Contact Frequency
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chain A contacts
res_A_sorted = sorted(interface_A.items(), key=lambda x: -x[1])
res_A_labels = [r[0] for r in res_A_sorted]
res_A_counts = [r[1] for r in res_A_sorted]

colors_A = ['#e74c3c' if c > 50 else '#f39c12' if c > 20 else '#3498db' for c in res_A_counts]
axes[0].barh(range(len(res_A_labels)), res_A_counts, color=colors_A, edgecolor='white', linewidth=0.5)
axes[0].set_yticks(range(len(res_A_labels)))
axes[0].set_yticklabels(res_A_labels, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Number of Atom Contacts (5Å)')
axes[0].set_title('Barnase (Chain A) Interface Residues')
axes[0].axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='20 contacts')
axes[0].legend()

# Chain D contacts
res_D_sorted = sorted(interface_D.items(), key=lambda x: -x[1])
res_D_labels = [r[0] for r in res_D_sorted]
res_D_counts = [r[1] for r in res_D_sorted]

colors_D = ['#e74c3c' if c > 50 else '#f39c12' if c > 20 else '#3498db' for c in res_D_counts]
axes[1].barh(range(len(res_D_labels)), res_D_counts, color=colors_D, edgecolor='white', linewidth=0.5)
axes[1].set_yticks(range(len(res_D_labels)))
axes[1].set_yticklabels(res_D_labels, fontsize=9)
axes[1].invert_yaxis()
axes[1].set_xlabel('Number of Atom Contacts (5Å)')
axes[1].set_title('Barstar (Chain D) Interface Residues')
axes[1].axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='20 contacts')
axes[1].legend()

legend_elements = [
    mpatches.Patch(color='#e74c3c', label='Hot spot (>50 contacts)'),
    mpatches.Patch(color='#f39c12', label='Warm (20-50 contacts)'),
    mpatches.Patch(color='#3498db', label='Peripheral (<20 contacts)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Interface Contact Frequency Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig2_interface_contacts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: fig2_interface_contacts.png")

# ============================================================
# FIGURE 3: SKEMPI ddG Distribution and Mutation Effects
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a: ddG distribution
ddGs = [r['ddG'] for r in ddg_results]
effects = [classify_mutation_effect(d) for d in ddGs]
effect_colors = {'Stabilizing': '#27ae60', 'Neutral': '#95a5a6', 'Destabilizing': '#e74c3c'}

ax = axes[0, 0]
for effect in ['Stabilizing', 'Neutral', 'Destabilizing']:
    vals = [d for d, e in zip(ddGs, effects) if e == effect]
    if vals:
        ax.hist(vals, bins=15, alpha=0.7, color=effect_colors[effect], label=f'{effect} (n={len(vals)})', edgecolor='white')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('ΔΔG (kcal/mol)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Binding Free Energy Changes')
ax.legend(fontsize=9)

# 3b: Single vs Double mutations
ax = axes[0, 1]
single = [r['ddG'] for r in ddg_results if r['n_mutations'] == 1]
double = [r['ddG'] for r in ddg_results if r['n_mutations'] == 2]
data_to_plot = [single, double]
bp = ax.boxplot(data_to_plot, labels=['Single\n(n={})'.format(len(single)), 'Double\n(n={})'.format(len(double))],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title('Single vs. Double Mutations')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 3c: ddG by location
ax = axes[1, 0]
location_ddg = defaultdict(list)
for r in ddg_results:
    loc = r['location']
    location_ddg[loc].append(r['ddG'])

# Sort by median ddG
loc_sorted = sorted(location_ddg.items(), key=lambda x: np.median(x[1]))
loc_labels = [l[0] for l in loc_sorted]
loc_data = [l[1] for l in loc_sorted]

positions = range(len(loc_labels))
bp = ax.boxplot(loc_data, positions=positions, patch_artist=True, widths=0.6, vert=True)
colors_loc = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(loc_labels)))
for patch, color in zip(bp['boxes'], colors_loc):
    patch.set_facecolor(color)
ax.set_xticks(positions)
ax.set_xticklabels(loc_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title('ΔΔG by Interface Location')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 3d: ddG vs number of mutations
ax = axes[1, 1]
n_muts = [r['n_mutations'] for r in ddg_results]
jitter = np.random.uniform(-0.1, 0.1, len(n_muts))
colors_scatter = [effect_colors[e] for e in effects]
ax.scatter([n + j for n, j in zip(n_muts, jitter)], ddGs, c=colors_scatter, alpha=0.6, edgecolors='white', linewidth=0.5, s=50)
ax.set_xlabel('Number of Mutations')
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title('ΔΔG vs. Number of Mutations')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

legend_elements = [mpatches.Patch(color=c, label=l) for l, c in effect_colors.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('SKEMPI v2 Binding Affinity Analysis for Barnase-Barstar', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig3_skempi_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: fig3_skempi_analysis.png")

# ============================================================
# FIGURE 4: Structural Property Correlation with ddG
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4a: ddG by mutation location (barstar vs barnase)
ax = axes[0, 0]
# Parse which chain the mutation is on
barnase_muts = []
barstar_muts = []
for r in ddg_results:
    mut = r['mutation']
    # Mutations are like KA25A (barnase) or DD39A (barstar)
    # In SKEMPI, barnase residues are labeled with original AA + position
    # We need to figure out which chain. Looking at the data, barnase is chain A
    # and barstar is chain D
    parts = mut.split(',')
    for p in parts:
        p = p.strip()
        if len(p) >= 4:
            # Check if residue is in barnase or barstar interface
            resseq = int(''.join(c for c in p[1:] if c.isdigit()))
            # Heuristic: barnase residues 1-110, barstar 1-89
            # But we need actual chain info. Let's use the interface data
            resname = p[0]
            if f"{ {'ALA':'ALA','ARG':'ARG','ASN':'ASN','ASP':'ASP','GLU':'GLU','GLN':'GLN',
                      'HIS':'HIS','ILE':'ILE','LYS':'LYS','PHE':'PHE','SER':'SER',
                      'THR':'THR','TRP':'TRP','TYR':'TYR','VAL':'VAL','GLY':'GLY',
                      'LEU':'LEU','PRO':'PRO','CYS':'CYS','MET':'MET'}.get(resname, resname)}{resseq}" in interface_A:
                barnase_muts.append(r['ddG'])
            elif f"{ {'ALA':'ALA','ARG':'ARG','ASN':'ASN','ASP':'ASP','GLU':'GLU','GLN':'GLN',
                      'HIS':'HIS','ILE':'ILE','LYS':'LYS','PHE':'PHE','SER':'SER',
                      'THR':'THR','TRP':'TRP','TYR':'TYR','VAL':'VAL','GLY':'GLY',
                      'LEU':'LEU','PRO':'PRO','CYS':'CYS','MET':'MET'}.get(resname, resname)}{resseq}" in interface_D:
                barstar_muts.append(r['ddG'])
            else:
                barnase_muts.append(r['ddG'])  # default to barnase

bp = ax.boxplot([barnase_muts, barstar_muts], 
                labels=['Barnase\nmutations', 'Barstar\nmutations'],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title('ΔΔG by Protein Chain')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 4b: Scatter of ddG vs interface contact count for mutated residues
ax = axes[0, 1]
mut_res_ddg = []
for r in ddg_results:
    mut = r['mutation']
    parts = mut.split(',')
    total_contacts = 0
    for p in parts:
        p = p.strip()
        resname = p[0]
        resseq = int(''.join(c for c in p[1:] if c.isdigit()))
        res_key = f"{resname}{resseq}"
        if res_key in interface_A:
            total_contacts += interface_A[res_key]
        if res_key in interface_D:
            total_contacts += interface_D[res_key]
    mut_res_ddg.append((total_contacts, r['ddG'], r['n_mutations']))

contacts_vals = [m[0] for m in mut_res_ddg]
ddg_vals = [m[1] for m in mut_res_ddg]
n_mut_vals = [m[2] for m in mut_res_ddg]

scatter = ax.scatter(contacts_vals, ddg_vals, c=n_mut_vals, cmap='coolwarm', 
                     s=60, alpha=0.7, edgecolors='white', linewidth=0.5, vmin=1, vmax=3)
plt.colorbar(scatter, ax=ax, label='Number of mutations')
ax.set_xlabel('Total Interface Contacts (5Å)')
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title('Interface Contact Count vs. ΔΔG')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Add trend line
if len(set(contacts_vals)) > 2:
    slope, intercept, r_val, p_val, std_err = stats.linregress(contacts_vals, ddg_vals)
    x_line = np.linspace(min(contacts_vals), max(contacts_vals), 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, 
            label=f'r={r_val:.3f}, p={p_val:.3f}')
    ax.legend(fontsize=9)

# 4c: ddG vs contact type (core, rim, surface)
ax = axes[1, 0]
location_groups = defaultdict(list)
for r in ddg_results:
    # Use primary location
    loc = r['location'].split(',')[0]
    location_groups[loc].append(r['ddG'])

loc_order = ['COR', 'SUP', 'RIM', 'SUR', 'INT']
loc_labels_plot = []
loc_data_plot = []
for loc in loc_order:
    if loc in location_groups:
        loc_labels_plot.append(loc)
        loc_data_plot.append(location_groups[loc])

colors_loc2 = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
bp = ax.boxplot(loc_data_plot, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors_loc2[:len(loc_labels_plot)]):
    patch.set_facecolor(color)
ax.set_xticklabels(loc_labels_plot, fontsize=10)
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title('ΔΔG by Interface Region Type')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 4d: Cumulative distribution of ddG
ax = axes[1, 1]
ddGs_sorted = np.sort(ddGs)
cdf = np.arange(1, len(ddGs_sorted) + 1) / len(ddGs_sorted)
ax.plot(ddGs_sorted, cdf, 'b-', linewidth=2)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='ΔΔG = 0')
ax.axvline(x=np.median(ddGs), color='green', linestyle='--', alpha=0.7, label=f'Median = {np.median(ddGs):.2f}')
ax.fill_between(ddGs_sorted, cdf, where=ddGs_sorted > 0, alpha=0.1, color='red')
ax.fill_between(ddGs_sorted, cdf, where=ddGs_sorted < 0, alpha=0.1, color='green')
ax.set_xlabel('ΔΔG (kcal/mol)')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('Cumulative Distribution of ΔΔG Values')
ax.legend(fontsize=9)

plt.suptitle('Structural Property Correlations with Binding Affinity', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig4_structural_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: fig4_structural_correlations.png")

# ============================================================
# FIGURE 5: Salt Bridge and H-bond Network
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5a: Salt bridge network
ax = axes[0]
sb_residues_A = set()
sb_residues_D = set()
for sb in salt_bridges:
    sb_residues_A.add(sb['resA'])
    sb_residues_D.add(sb['resD'])

# Create a simple network visualization
all_res_A = sorted(sb_residues_A)
all_res_D = sorted(sb_residues_D)
pos_A = {r: (0, i) for i, r in enumerate(all_res_A)}
pos_D = {r: (2, i) for i, r in enumerate(all_res_D)}

# Draw edges
for sb in salt_bridges:
    rA, rD = sb['resA'], sb['resD']
    if rA in pos_A and rD in pos_D:
        yA = pos_A[rA][1]
        yD = pos_D[rD][1]
        ax.plot([0, 2], [yA, yD], 'r-', alpha=0.3, linewidth=1)

# Draw nodes
for r, pos in pos_A.items():
    ax.scatter(pos[0], pos[1], s=200, c='#3498db', zorder=5, edgecolors='white')
    ax.text(pos[0]-0.15, pos[1], r, fontsize=8, ha='right', va='center')
for r, pos in pos_D.items():
    ax.scatter(pos[0], pos[1], s=200, c='#e74c3c', zorder=5, edgecolors='white')
    ax.text(pos[0]+0.15, pos[1], r, fontsize=8, ha='left', va='center')

ax.set_xlim(-1, 3)
ax.set_ylim(-1, max(len(all_res_A), len(all_res_D)))
ax.set_title('Salt Bridge Network')
ax.set_xticks([0, 2])
ax.set_xticklabels(['Barnase', 'Barstar'])
ax.axis('off')

# 5b: Hydrogen bond summary
ax = axes[1]
hbond_summary = defaultdict(int)
for hb in hbonds:
    key = f"{hb['resA']}-{hb['resD']}"
    hbond_summary[key] += 1

hb_sorted = sorted(hbond_summary.items(), key=lambda x: -x[1])[:20]
hb_labels = [h[0] for h in hb_sorted]
hb_counts = [h[1] for h in hb_sorted]

ax.barh(range(len(hb_labels)), hb_counts, color='#2ecc71', edgecolor='white')
ax.set_yticks(range(len(hb_labels)))
ax.set_yticklabels(hb_labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Number of H-bond atom pairs')
ax.set_title('Top 20 Hydrogen Bond Pairs')

plt.suptitle('Interfacial Non-covalent Interaction Network', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig5_interaction_network.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: fig5_interaction_network.png")

# ============================================================
# FIGURE 6: Comprehensive Summary - Mutation Impact by Region
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6a: Violin plot of ddG by location
ax = axes[0, 0]
# Filter to locations with enough data
loc_with_data = {k: v for k, v in location_groups.items() if len(v) >= 3}
if loc_with_data:
    locs_violin = list(loc_with_data.keys())
    data_violin = [loc_with_data[l] for l in locs_violin]
    parts = ax.violinplot(data_violin, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(locs_violin) + 1))
    ax.set_xticklabels(locs_violin, fontsize=9)
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title('ΔΔG Distribution by Interface Region')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 6b: Scatter of ddG for single mutations colored by location
ax = axes[0, 1]
single_muts = [r for r in ddg_results if r['n_mutations'] == 1]
for loc in ['COR', 'SUP', 'RIM', 'SUR', 'INT']:
    vals = [r['ddG'] for r in single_muts if loc in r['location']]
    if vals:
        loc_idx = ['COR', 'SUP', 'RIM', 'SUR', 'INT'].index(loc)
        jitter = np.random.uniform(-0.05, 0.05, len(vals))
        x_vals = [loc_idx + j for j in jitter]
        ax.scatter(x_vals, vals, alpha=0.6, s=50, label=f'{loc} (n={len(vals)})', edgecolors='white')
ax.set_ylabel('ΔΔG (kcal/mol)')
ax.set_title('Single Mutation Effects by Region')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels(['COR', 'SUP', 'RIM', 'SUR', 'INT'], fontsize=9)
ax.legend(fontsize=8)

# 6c: Mean ddG by location with error bars
ax = axes[1, 0]
loc_means = {}
loc_stds = {}
for loc in loc_order:
    if loc in location_groups and len(location_groups[loc]) > 1:
        loc_means[loc] = np.mean(location_groups[loc])
        loc_stds[loc] = np.std(location_groups[loc])

if loc_means:
    x_pos = range(len(loc_means))
    ax.bar(x_pos, [loc_means[l] for l in loc_means], 
           yerr=[loc_stds[l] for l in loc_means],
           color=colors_loc2[:len(loc_means)], capsize=5, edgecolor='white')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(loc_means.keys()), fontsize=10)
ax.set_ylabel('Mean ΔΔG ± SD (kcal/mol)')
ax.set_title('Mean ΔΔG by Interface Region')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 6d: Top destabilizing mutations
ax = axes[1, 1]
top_destab = sorted(ddg_results, key=lambda x: -x['ddG'])[:15]
y_pos = range(len(top_destab))
ax.barh(y_pos, [r['ddG'] for r in top_destab], color='#e74c3c', alpha=0.8, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['mutation']} ({r['location']})" for r in top_destab], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('ΔΔG (kcal/mol)')
ax.set_title('Top 15 Most Destabilizing Mutations')

plt.suptitle('Comprehensive Mutation Impact Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig6_mutation_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: fig6_mutation_impact.png")

# ============================================================
# FIGURE 7: 3D Structure Overview (2D Projection)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Project CA atoms onto 2D using PCA-like projection
ca_A = [(a['resseq'], a['x'], a['y'], a['z']) for a in atoms if a['chain'] == 'A' and a['atomname'] == 'CA']
ca_D = [(a['resseq'], a['x'], a['y'], a['z']) for a in atoms if a['chain'] == 'D' and a['atomname'] == 'CA']

# Simple projection: x vs z
ax.scatter([a[1] for a in ca_A], [a[3] for a in ca_A], c='#3498db', s=80, label='Barnase (Chain A)', 
           edgecolors='white', linewidth=0.5, zorder=5)
ax.scatter([d[1] for d in ca_D], [d[3] for d in ca_D], c='#e74c3c', s=80, label='Barstar (Chain D)',
           edgecolors='white', linewidth=0.5, zorder=5)

# Connect interface residues
for c in contacts_5A:
    resA_seq = int(c['resA'][3:])
    resD_seq = int(c['resD'][3:])
    a_coord = next((a for a in ca_A if a[0] == resA_seq), None)
    d_coord = next((d for d in ca_D if d[0] == resD_seq), None)
    if a_coord and d_coord:
        ax.plot([a_coord[1], d_coord[1]], [a_coord[3], d_coord[3]], 
                'gray', alpha=0.05, linewidth=0.3)

# Highlight key hot-spot residues
hot_spots_A = ['HIS102', 'ARG59', 'TYR103', 'ARG83']
hot_spots_D = ['TYR29', 'ASP35', 'ASP39', 'TRP38']

for hs in hot_spots_A:
    resseq = int(''.join(c for c in hs[3:] if c.isdigit()))
    a = next((a for a in ca_A if a[0] == resseq), None)
    if a:
        ax.scatter(a[1], a[3], c='#f1c40f', s=150, marker='*', zorder=10, edgecolors='black', linewidth=0.5)
        ax.annotate(hs, (a[1], a[3]), fontsize=7, ha='center', va='bottom', fontweight='bold')

for hs in hot_spots_D:
    resseq = int(''.join(c for c in hs[3:] if c.isdigit()))
    d = next((d for d in ca_D if d[0] == resseq), None)
    if d:
        ax.scatter(d[1], d[3], c='#f1c40f', s=150, marker='*', zorder=10, edgecolors='black', linewidth=0.5)
        ax.annotate(hs, (d[1], d[3]), fontsize=7, ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('X (Å)')
ax.set_ylabel('Z (Å)')
ax.set_title('Barnase-Barstar Complex Structure (XZ Projection)\nInterface hot-spot residues marked with stars')
ax.legend(fontsize=10)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('report/images/fig7_structure_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7 saved: fig7_structure_overview.png")

# ============================================================
# Save summary statistics
# ============================================================
summary = {
    'structure': {
        'pdb': '1BRS',
        'chain_A': 'Barnase (108 residues)',
        'chain_D': 'Barstar (87 residues)',
        'interface_residues_A': len(interface_A),
        'interface_residues_D': len(interface_D),
        'total_contacts_5A': len(contacts_5A),
        'total_contacts_4A': len(contacts_4A),
        'salt_bridges': len(salt_bridges),
        'hydrogen_bonds': len(hbonds),
    },
    'skempi': {
        'total_entries': len(ddg_results),
        'single_mutations': len([r for r in ddg_results if r['n_mutations'] == 1]),
        'double_mutations': len([r for r in ddg_results if r['n_mutations'] == 2]),
        'mean_ddG': float(np.mean(ddGs)),
        'std_ddG': float(np.std(ddGs)),
        'median_ddG': float(np.median(ddGs)),
        'n_stabilizing': len([e for e in effects if e == 'Stabilizing']),
        'n_neutral': len([e for e in effects if e == 'Neutral']),
        'n_destabilizing': len([e for e in effects if e == 'Destabilizing']),
    },
    'hot_spot_residues': {
        'barnase': ['HIS102', 'ARG59', 'TYR103', 'ARG83', 'GLU60'],
        'barstar': ['TYR29', 'ASP35', 'ASP39', 'TRP38', 'ASN33'],
    }
}

with open('outputs/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nOutputs saved to:")
print(f"  outputs/structural_analysis.json")
print(f"  outputs/skempi_analysis.json")
print(f"  outputs/summary_statistics.json")
print(f"  report/images/fig1_distance_maps.png")
print(f"  report/images/fig2_interface_contacts.png")
print(f"  report/images/fig3_skempi_analysis.png")
print(f"  report/images/fig4_structural_correlations.png")
print(f"  report/images/fig5_interaction_network.png")
print(f"  report/images/fig6_mutation_impact.png")
print(f"  report/images/fig7_structure_overview.png")
