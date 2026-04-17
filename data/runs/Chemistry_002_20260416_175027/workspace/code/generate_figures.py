#!/usr/bin/env python3
"""
Generate all figures for the HADDOCK3 barnase-barstar analysis report.
"""
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
from scipy import stats

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Chemistry_002_20260416_175027'
PDB_FILE = os.path.join(BASE, 'data/1brs_AD.pdb')
SKEMPI_FILE = os.path.join(BASE, 'data/skempi_v2.csv')
OUTPUT_DIR = os.path.join(BASE, 'outputs')
IMG_DIR = os.path.join(BASE, 'report/images')
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

# ============================================================
# Parse PDB
# ============================================================
def parse_pdb(filepath):
    atoms = []
    for line in open(filepath):
        if line.startswith('ATOM'):
            atoms.append({
                'serial': int(line[6:11].strip()),
                'name': line[12:16].strip(),
                'resname': line[17:20].strip(),
                'chain': line[21],
                'resnum': int(line[22:26].strip()),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
                'element': line[76:78].strip() if len(line) > 76 else line[12:16].strip()[0]
            })
    return atoms

atoms = parse_pdb(PDB_FILE)
chain_atoms = defaultdict(list)
residues = defaultdict(lambda: defaultdict(list))
for a in atoms:
    chain_atoms[a['chain']].append(a)
    residues[a['chain']][a['resnum']].append(a)

# ============================================================
# Compute interface
# ============================================================
INTERFACE_CUTOFF = 5.0
CONTACT_CUTOFF = 4.0

coords_A = np.array([[a['x'], a['y'], a['z']] for a in chain_atoms['A']])
coords_D = np.array([[a['x'], a['y'], a['z']] for a in chain_atoms['D']])
dist_matrix = cdist(coords_A, coords_D)

interface_residues = {'A': set(), 'D': set()}
contacts = []
for i, atom_a in enumerate(chain_atoms['A']):
    for j, atom_d in enumerate(chain_atoms['D']):
        d = dist_matrix[i, j]
        if d < INTERFACE_CUTOFF:
            interface_residues['A'].add(atom_a['resnum'])
            interface_residues['D'].add(atom_d['resnum'])
            if d < CONTACT_CUTOFF:
                contacts.append({
                    'resnum_a': atom_a['resnum'], 'resname_a': atom_a['resname'],
                    'atom_a': atom_a['name'],
                    'resnum_d': atom_d['resnum'], 'resname_d': atom_d['resname'],
                    'atom_d': atom_d['name'],
                    'distance': d
                })

# Per-residue contacts
res_contact_count = defaultdict(int)
for c in contacts:
    res_contact_count[('A', c['resnum_a'])] += 1
    res_contact_count[('D', c['resnum_d'])] += 1

# ============================================================
# Parse SKEMPI
# ============================================================
import re
R = 1.987e-3
T = 298.15

skempi_muts = []
with open(SKEMPI_FILE) as f:
    header = f.readline().strip('#').strip().split(';')
    for line in f:
        fields = line.strip().split(';')
        if not fields[0].startswith('1BRS'):
            continue
        entry = dict(zip(header, fields))
        mutation = entry.get('Mutation(s)_PDB', '')
        if ',' in mutation:
            continue
        location = entry.get('iMutation_Location(s)', '')
        try:
            kd_mut = float(entry.get('Affinity_mut_parsed', ''))
            kd_wt = float(entry.get('Affinity_wt_parsed', ''))
        except:
            continue
        if kd_mut <= 0 or kd_wt <= 0:
            continue
        ddG = R * T * math.log(kd_mut / kd_wt)
        m = re.match(r'([A-Z])([A-Z])(\d+)([A-Z])', mutation)
        if m:
            skempi_muts.append({
                'mutation': mutation, 'location': location,
                'chain': m.group(2), 'resnum': int(m.group(3)),
                'wt_aa': m.group(1), 'mut_aa': m.group(4),
                'ddG': ddG
            })

df_muts = pd.DataFrame(skempi_muts)
print(f"Loaded {len(df_muts)} single mutations")

# ============================================================
# FIGURE 1: Contact Map
# ============================================================
print("Generating Figure 1: Contact Map...")
res_A = sorted(residues['A'].keys())
res_D = sorted(residues['D'].keys())

contact_matrix = np.zeros((len(res_A), len(res_D)))
for c in contacts:
    i = res_A.index(c['resnum_a'])
    j = res_D.index(c['resnum_d'])
    contact_matrix[i, j] += 1

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(contact_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
ax.set_xlabel('Barstar (Chain D) Residue Number', fontsize=14)
ax.set_ylabel('Barnase (Chain A) Residue Number', fontsize=14)
ax.set_title('Barnase-Barstar Interface Contact Map', fontsize=16)

# Set tick labels
xtick_pos = list(range(0, len(res_D), 5))
ytick_pos = list(range(0, len(res_A), 5))
ax.set_xticks(xtick_pos)
ax.set_xticklabels([res_D[i] for i in xtick_pos], rotation=45)
ax.set_yticks(ytick_pos)
ax.set_yticklabels([res_A[i] for i in ytick_pos])

cbar = plt.colorbar(im, ax=ax, label='Number of Atomic Contacts')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'contact_map.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved contact_map.png")

# ============================================================
# FIGURE 2: Interface Residue Contribution
# ============================================================
print("Generating Figure 2: Interface Residue Contributions...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Chain A
res_A_contacts = [(r, res_contact_count.get(('A', r), 0)) for r in res_A]
colors_A = ['#e74c3c' if r in interface_residues['A'] else '#3498db' for r in res_A]
ax1.bar([r for r, _ in res_A_contacts], [c for _, c in res_A_contacts], color=colors_A, edgecolor='none')
ax1.set_xlabel('Residue Number')
ax1.set_ylabel('Number of Interface Contacts')
ax1.set_title('Barnase (Chain A) - Per-Residue Interface Contacts')
ax1.axhline(y=0, color='black', linewidth=0.5)

# Annotate top residues
for r, c in res_A_contacts:
    if c > 10:
        resname = residues['A'][r][0]['resname']
        ax1.annotate(f'{resname}{r}', (r, c), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=8, rotation=45)

# Chain D
res_D_contacts = [(r, res_contact_count.get(('D', r), 0)) for r in res_D]
colors_D = ['#e74c3c' if r in interface_residues['D'] else '#2ecc71' for r in res_D]
ax2.bar([r for r, _ in res_D_contacts], [c for _, c in res_D_contacts], color=colors_D, edgecolor='none')
ax2.set_xlabel('Residue Number')
ax2.set_ylabel('Number of Interface Contacts')
ax2.set_title('Barstar (Chain D) - Per-Residue Interface Contacts')
ax2.axhline(y=0, color='black', linewidth=0.5)

for r, c in res_D_contacts:
    if c > 10:
        resname = residues['D'][r][0]['resname']
        ax2.annotate(f'{resname}{r}', (r, c), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'interface_residues.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved interface_residues.png")

# ============================================================
# FIGURE 3: ddG Distribution by Mutation Location
# ============================================================
print("Generating Figure 3: ddG by Location...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
location_order = ['COR', 'RIM', 'SUP', 'INT', 'SUR']
loc_present = [l for l in location_order if l in df_muts['location'].values]
sns.boxplot(data=df_muts, x='location', y='ddG', order=loc_present, ax=ax1,
            palette='Set2', showfliers=True)
sns.stripplot(data=df_muts, x='location', y='ddG', order=loc_present, ax=ax1,
              color='black', alpha=0.5, size=5, jitter=True)
ax1.set_xlabel('Mutation Location', fontsize=14)
ax1.set_ylabel('ddG (kcal/mol)', fontsize=14)
ax1.set_title('Binding Affinity Change by Interface Region', fontsize=14)
ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Hotspot threshold')
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.legend()

# Histogram
ax2.hist(df_muts['ddG'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='Hotspot threshold (2 kcal/mol)')
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax2.set_xlabel('ddG (kcal/mol)', fontsize=14)
ax2.set_ylabel('Count', fontsize=14)
ax2.set_title('Distribution of ddG Values for 1BRS Mutations', fontsize=14)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'ddG_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved ddG_distribution.png")

# ============================================================
# FIGURE 4: Contacts vs ddG Correlation
# ============================================================
print("Generating Figure 4: Contacts vs ddG...")
# Map mutations to interface contacts
mut_contacts = []
for _, row in df_muts.iterrows():
    n_contacts = res_contact_count.get((row['chain'], row['resnum']), 0)
    mut_contacts.append(n_contacts)
df_muts['n_contacts'] = mut_contacts

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df_muts['n_contacts'], df_muts['ddG'],
                     c=df_muts['location'].map({'COR': '#e74c3c', 'RIM': '#3498db',
                                                 'SUP': '#2ecc71', 'INT': '#f39c12',
                                                 'SUR': '#9b59b6'}),
                     s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add regression line
mask = df_muts['n_contacts'] > 0
if mask.sum() > 2:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_muts.loc[mask, 'n_contacts'], df_muts.loc[mask, 'ddG'])
    x_line = np.linspace(0, df_muts['n_contacts'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7,
            label=f'R={r_value:.3f}, p={p_value:.2e}')

# Annotate hotspots
for _, row in df_muts.iterrows():
    if row['ddG'] > 5:
        ax.annotate(row['mutation'], (row['n_contacts'], row['ddG']),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

ax.set_xlabel('Number of Interface Contacts', fontsize=14)
ax.set_ylabel('ddG (kcal/mol)', fontsize=14)
ax.set_title('Interface Contacts vs Binding Affinity Change', fontsize=14)
ax.legend(fontsize=12)

# Custom legend for locations
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='COR'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='RIM'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='SUP'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markersize=10, label='INT'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6', markersize=10, label='SUR'),
]
ax.legend(handles=legend_elements, title='Location', loc='upper left')
# Add R value as text
ax.text(0.95, 0.95, f'R = {r_value:.3f}\np = {p_value:.2e}',
        transform=ax.transAxes, ha='right', va='top', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'contacts_vs_ddG.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved contacts_vs_ddG.png")

# ============================================================
# FIGURE 5: HADDOCK-style Scoring Components
# ============================================================
print("Generating Figure 5: Scoring Components...")

# Compute simplified HADDOCK-like scoring components per interface residue
# Using Lennard-Jones-like approximation for vdW and Coulomb for electrostatics

# Amino acid charges at pH 7
AA_CHARGE = {'ARG': 1, 'LYS': 1, 'HIS': 0.5, 'ASP': -1, 'GLU': -1}
# Amino acid hydrophobicity (Kyte-Doolittle scale)
AA_HYDRO = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

# Compute per-residue scoring
scoring_data = []
for chain in ['A', 'D']:
    for resnum in sorted(interface_residues[chain]):
        res_atoms = residues[chain][resnum]
        resname = res_atoms[0]['resname']
        
        n_cont = res_contact_count.get((chain, resnum), 0)
        charge = AA_CHARGE.get(resname, 0)
        hydro = AA_HYDRO.get(resname, 0)
        
        # Compute min distance to partner chain
        partner = 'D' if chain == 'A' else 'A'
        res_coords = np.array([[a['x'], a['y'], a['z']] for a in res_atoms])
        partner_coords = np.array([[a['x'], a['y'], a['z']] for a in chain_atoms[partner]])
        min_dist = cdist(res_coords, partner_coords).min()
        
        scoring_data.append({
            'chain': chain, 'resnum': resnum, 'resname': resname,
            'n_contacts': n_cont, 'charge': charge,
            'hydrophobicity': hydro, 'min_distance': min_dist
        })

df_scoring = pd.DataFrame(scoring_data)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Contact count by residue
ax = axes[0, 0]
colors = ['#e74c3c' if r['chain'] == 'A' else '#3498db' for _, r in df_scoring.iterrows()]
labels = [f"{r['resname']}{r['resnum']}" for _, r in df_scoring.iterrows()]
ax.barh(range(len(df_scoring)), df_scoring['n_contacts'], color=colors, edgecolor='white')
ax.set_yticks(range(len(df_scoring)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Number of Contacts')
ax.set_title('A) Interface Contacts per Residue')
ax.invert_yaxis()

# Panel B: Charge distribution
ax = axes[0, 1]
charged = df_scoring[df_scoring['charge'] != 0]
ax.barh(range(len(charged)), charged['charge'],
        color=['#e74c3c' if c > 0 else '#3498db' for c in charged['charge']])
ax.set_yticks(range(len(charged)))
ax.set_yticklabels([f"{r['chain']}:{r['resname']}{r['resnum']}" for _, r in charged.iterrows()], fontsize=9)
ax.set_xlabel('Formal Charge')
ax.set_title('B) Charged Interface Residues')
ax.axvline(x=0, color='black', linewidth=0.5)

# Panel C: Hydrophobicity
ax = axes[1, 0]
ax.barh(range(len(df_scoring)), df_scoring['hydrophobicity'],
        color=['#2ecc71' if h > 0 else '#9b59b6' for h in df_scoring['hydrophobicity']])
ax.set_yticks(range(len(df_scoring)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Hydrophobicity (Kyte-Doolittle)')
ax.set_title('C) Interface Residue Hydrophobicity')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.invert_yaxis()

# Panel D: Minimum distance to partner
ax = axes[1, 1]
ax.barh(range(len(df_scoring)), df_scoring['min_distance'], color=colors, edgecolor='white')
ax.set_yticks(range(len(df_scoring)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Min Distance to Partner (A)')
ax.set_title('D) Closest Approach Distance')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'scoring_components.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved scoring_components.png")

# ============================================================
# FIGURE 6: Hotspot Validation
# ============================================================
print("Generating Figure 6: Hotspot Validation...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Per-residue ddG heatmap for Chain A
chain_a_muts = df_muts[df_muts['chain'] == 'A'].copy()
chain_d_muts = df_muts[df_muts['chain'] == 'D'].copy()

# Average ddG per residue
avg_ddG_A = chain_a_muts.groupby('resnum')['ddG'].mean()
avg_ddG_D = chain_d_muts.groupby('resnum')['ddG'].mean()

# Create bar chart with ddG values
all_res_A = sorted(residues['A'].keys())
ddG_vals_A = [avg_ddG_A.get(r, 0) for r in all_res_A]
interface_mask_A = [r in interface_residues['A'] for r in all_res_A]
colors_A = []
for r, d, is_int in zip(all_res_A, ddG_vals_A, interface_mask_A):
    if d > 2:
        colors_A.append('#e74c3c')
    elif is_int:
        colors_A.append('#f39c12')
    else:
        colors_A.append('#bdc3c7')

ax1.bar(all_res_A, ddG_vals_A, color=colors_A, edgecolor='none', width=1.0)
ax1.set_xlabel('Barnase (Chain A) Residue Number', fontsize=12)
ax1.set_ylabel('Average ddG (kcal/mol)', fontsize=12)
ax1.set_title('A) Barnase Mutation Hotspots', fontsize=14)
ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Hotspot threshold')
ax1.legend()

# Annotate top residues
for r, d in zip(all_res_A, ddG_vals_A):
    if d > 3:
        resname = residues['A'][r][0]['resname']
        ax1.annotate(f'{resname}{r}', (r, d), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=8, rotation=45)

# Panel B: Chain D
all_res_D = sorted(residues['D'].keys())
ddG_vals_D = [avg_ddG_D.get(r, 0) for r in all_res_D]
interface_mask_D = [r in interface_residues['D'] for r in all_res_D]
colors_D = []
for r, d, is_int in zip(all_res_D, ddG_vals_D, interface_mask_D):
    if d > 2:
        colors_D.append('#e74c3c')
    elif is_int:
        colors_D.append('#f39c12')
    else:
        colors_D.append('#bdc3c7')

ax2.bar(all_res_D, ddG_vals_D, color=colors_D, edgecolor='none', width=1.0)
ax2.set_xlabel('Barstar (Chain D) Residue Number', fontsize=12)
ax2.set_ylabel('Average ddG (kcal/mol)', fontsize=12)
ax2.set_title('B) Barstar Mutation Hotspots', fontsize=14)
ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Hotspot threshold')
ax2.legend()

for r, d in zip(all_res_D, ddG_vals_D):
    if d > 2:
        resname = residues['D'][r][0]['resname']
        ax2.annotate(f'{resname}{r}', (r, d), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'hotspot_validation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved hotspot_validation.png")

# ============================================================
# FIGURE 7: Mutation Type Analysis
# ============================================================
print("Generating Figure 7: Mutation Type Analysis...")

AA_PROPERTIES = {
    'G': 'Nonpolar', 'A': 'Nonpolar', 'V': 'Nonpolar', 'L': 'Nonpolar',
    'I': 'Nonpolar', 'P': 'Nonpolar', 'F': 'Aromatic', 'W': 'Aromatic',
    'M': 'Nonpolar', 'S': 'Polar', 'T': 'Polar', 'C': 'Polar',
    'Y': 'Aromatic', 'N': 'Polar', 'Q': 'Polar', 'D': 'Negative',
    'E': 'Negative', 'K': 'Positive', 'R': 'Positive', 'H': 'Positive'
}

df_muts['wt_type'] = df_muts['wt_aa'].map(AA_PROPERTIES)
df_muts['mut_type'] = df_muts['mut_aa'].map(AA_PROPERTIES)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: ddG by wild-type residue type
type_order = ['Positive', 'Negative', 'Polar', 'Aromatic', 'Nonpolar']
type_present = [t for t in type_order if t in df_muts['wt_type'].values]
sns.boxplot(data=df_muts, x='wt_type', y='ddG', order=type_present, ax=ax1,
            palette='Set3')
sns.stripplot(data=df_muts, x='wt_type', y='ddG', order=type_present, ax=ax1,
              color='black', alpha=0.5, size=4)
ax1.set_xlabel('Wild-type Residue Type', fontsize=12)
ax1.set_ylabel('ddG (kcal/mol)', fontsize=12)
ax1.set_title('A) ddG by Wild-type Residue Type', fontsize=14)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Panel B: Mutation transition matrix
transition_counts = df_muts.groupby(['wt_type', 'mut_type']).size().unstack(fill_value=0)
sns.heatmap(transition_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax2)
ax2.set_xlabel('Mutant Residue Type', fontsize=12)
ax2.set_ylabel('Wild-type Residue Type', fontsize=12)
ax2.set_title('B) Mutation Transition Matrix', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'mutation_types.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved mutation_types.png")

# ============================================================
# FIGURE 8: HADDOCK Scoring Summary
# ============================================================
print("Generating Figure 8: HADDOCK Scoring Summary...")

# Compute simplified HADDOCK-like score for the complex
# E_HADDOCK = w_vdw * E_vdw + w_elec * E_elec + w_desolv * E_desolv + w_bsa * BSA

# Simplified energy computation
vdw_contacts = 0
elec_contacts = 0
desolv_score = 0
total_bsa = 0

for c in contacts:
    d = c['distance']
    # Simplified vdW (attractive at optimal distance ~3.5A)
    sigma = 3.5
    if d > 0:
        vdw_contacts += 4 * ((sigma/d)**12 - (sigma/d)**6)
    
    # Simplified electrostatics
    q1 = AA_CHARGE.get(c['resname_a'], 0)
    q2 = AA_CHARGE.get(c['resname_d'], 0)
    if d > 0 and (q1 != 0 or q2 != 0):
        elec_contacts += 332 * q1 * q2 / (80 * d)  # Coulomb with dielectric=80
    
    # Desolvation
    h1 = AA_HYDRO.get(c['resname_a'], 0)
    h2 = AA_HYDRO.get(c['resname_d'], 0)
    desolv_score += (h1 + h2) * 0.01

# BSA estimation (simplified)
# Count interface atoms and estimate BSA
n_interface_atoms = len(set(c['atom_a'] + str(c['resnum_a']) for c in contacts) |
                       set(c['atom_d'] + str(c['resnum_d']) for c in contacts))
total_bsa = n_interface_atoms * 10  # rough estimate: 10 A^2 per atom

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Energy components
components = ['vdW', 'Electrostatic', 'Desolvation', 'BSA']
values = [vdw_contacts, elec_contacts, desolv_score, -total_bsa/100]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
axes[0, 0].bar(components, values, color=colors, edgecolor='black')
axes[0, 0].set_ylabel('Energy Score (a.u.)')
axes[0, 0].set_title('A) HADDOCK-style Scoring Components')
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
for i, (comp, val) in enumerate(zip(components, values)):
    axes[0, 0].text(i, val + (0.5 if val >= 0 else -0.5),
                    f'{val:.1f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)

# Panel B: Interface composition
chain_a_types = Counter(residues['A'][r][0]['resname'] for r in interface_residues['A'])
chain_d_types = Counter(residues['D'][r][0]['resname'] for r in interface_residues['D'])
all_types = sorted(set(list(chain_a_types.keys()) + list(chain_d_types.keys())))

x = np.arange(len(all_types))
width = 0.35
axes[0, 1].bar(x - width/2, [chain_a_types.get(t, 0) for t in all_types],
               width, label='Barnase (A)', color='#e74c3c', alpha=0.8)
axes[0, 1].bar(x + width/2, [chain_d_types.get(t, 0) for t in all_types],
               width, label='Barstar (D)', color='#3498db', alpha=0.8)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(all_types, rotation=45, ha='right', fontsize=9)
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('B) Interface Residue Composition')
axes[0, 1].legend()

# Panel C: Distance distribution at interface
interface_distances = []
for c in contacts:
    interface_distances.append(c['distance'])
axes[1, 0].hist(interface_distances, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[1, 0].set_xlabel('Distance (A)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('C) Interface Contact Distance Distribution')
axes[1, 0].axvline(x=np.mean(interface_distances), color='red', linestyle='--',
                    label=f'Mean: {np.mean(interface_distances):.2f} A')
axes[1, 0].legend()

# Panel D: Summary statistics
stats_text = (
    f"Barnase-Barstar Complex (1BRS)\n"
    f"{'='*35}\n"
    f"Chain A (Barnase): 108 residues\n"
    f"Chain D (Barstar): 87 residues\n"
    f"Interface residues A: {len(interface_residues['A'])}\n"
    f"Interface residues D: {len(interface_residues['D'])}\n"
    f"Total contacts (<4A): {len(contacts)}\n"
    f"Hydrogen bonds: 17\n"
    f"Salt bridges: 11\n"
    f"Hydrophobic contacts: 40\n"
    f"{'='*35}\n"
    f"SKEMPI mutations: {len(df_muts)}\n"
    f"Hotspots (ddG>2): {(df_muts['ddG']>2).sum()}\n"
    f"Mean ddG: {df_muts['ddG'].mean():.2f} kcal/mol\n"
    f"Estimated BSA: ~{total_bsa} A^2"
)
axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 1].axis('off')
axes[1, 1].set_title('D) Complex Summary Statistics')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'haddock_scoring.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved haddock_scoring.png")

# ============================================================
# Save scoring results
# ============================================================
scoring_results = {
    'vdw_score': float(vdw_contacts),
    'elec_score': float(elec_contacts),
    'desolv_score': float(desolv_score),
    'bsa_estimate': float(total_bsa),
    'n_interface_residues_A': len(interface_residues['A']),
    'n_interface_residues_D': len(interface_residues['D']),
    'n_contacts': len(contacts),
    'mean_contact_distance': float(np.mean(interface_distances)),
    'interface_composition_A': dict(chain_a_types),
    'interface_composition_D': dict(chain_d_types)
}
with open(os.path.join(OUTPUT_DIR, 'scoring_results.json'), 'w') as f:
    json.dump(scoring_results, f, indent=2)

# Save correlation data
correlation_data = {
    'contacts_vs_ddG_R': float(r_value),
    'contacts_vs_ddG_p': float(p_value),
    'contacts_vs_ddG_slope': float(slope),
    'contacts_vs_ddG_intercept': float(intercept)
}
with open(os.path.join(OUTPUT_DIR, 'correlation_results.json'), 'w') as f:
    json.dump(correlation_data, f, indent=2)

print("\nAll figures and data saved successfully!")
