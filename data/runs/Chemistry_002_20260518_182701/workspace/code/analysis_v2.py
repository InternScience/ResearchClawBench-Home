#!/usr/bin/env python3
"""
HADDOCK3-inspired integrative modeling analysis for barnase-barstar (1BRS)
using SKEMPI 2.0 binding affinity data.

Version 2: Improved interface detection using all heavy atoms.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Paths
PDB_PATH = 'data/1brs_AD.pdb'
SKEMPI_PATH = 'data/skempi_v2.csv'
OUTPUT_DIR = 'outputs'
FIGURE_DIR = 'report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ============== PDB Parsing ==============

def parse_pdb(pdb_path):
    """Parse ATOM records from PDB."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = {
                    'record': line[0:6].strip(),
                    'serial': int(line[6:11].strip()),
                    'name': line[12:16].strip(),
                    'altLoc': line[16:17].strip(),
                    'resName': line[17:20].strip(),
                    'chainID': line[21:22].strip(),
                    'resSeq': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'element': line[76:78].strip() if len(line) > 76 else '',
                }
                atoms.append(atom)
    return pd.DataFrame(atoms)


def get_residue_centroids(df_atoms):
    """Compute centroids per residue (all atoms)."""
    groups = df_atoms.groupby(['chainID', 'resSeq', 'resName'])
    centroids = []
    for (chain, resseq, resname), group in groups:
        centroids.append({
            'chainID': chain,
            'resSeq': resseq,
            'resName': resname,
            'x': group['x'].mean(),
            'y': group['y'].mean(),
            'z': group['z'].mean(),
        })
    return pd.DataFrame(centroids)


def compute_interface_residues_all_atoms(df_atoms, chainA='A', chainD='D', cutoff=5.0):
    """Identify interface residues using all heavy atoms within cutoff."""
    atoms_A = df_atoms[df_atoms['chainID'] == chainA].copy()
    atoms_D = df_atoms[df_atoms['chainID'] == chainD].copy()
    
    interface_A = set()
    interface_D = set()
    
    # For efficiency, use numpy arrays
    coords_A = atoms_A[['x','y','z']].values
    coords_D = atoms_D[['x','y','z']].values
    resseq_A = atoms_A['resSeq'].values
    resseq_D = atoms_D['resSeq'].values
    
    # Check distances in blocks to avoid O(N^2) memory blowup
    block = 500
    for i_start in range(0, len(coords_A), block):
        i_end = min(i_start + block, len(coords_A))
        ca = coords_A[i_start:i_end]
        for j_start in range(0, len(coords_D), block):
            j_end = min(j_start + block, len(coords_D))
            cd = coords_D[j_start:j_end]
            # Compute pairwise distances for this block
            dists = np.sqrt(((ca[:, None, :] - cd[None, :, :])**2).sum(axis=2))
            mask = dists < cutoff
            if mask.any():
                rows, cols = np.where(mask)
                for r, c in zip(rows, cols):
                    interface_A.add(int(resseq_A[i_start + r]))
                    interface_D.add(int(resseq_D[j_start + c]))
    
    return sorted(interface_A), sorted(interface_D)


def compute_buried_surface_area(df_atoms, interface_A, interface_D, probe_radius=1.4):
    """Approximate BSA using residue surface area estimates."""
    residue_sasa = {
        'ALA': 115, 'CYS': 135, 'ASP': 150, 'GLU': 190, 'PHE': 210,
        'GLY': 75,  'HIS': 195, 'ILE': 175, 'LYS': 200, 'LEU': 170,
        'MET': 185, 'ASN': 160, 'PRO': 145, 'GLN': 180, 'ARG': 225,
        'SER': 115, 'THR': 140, 'VAL': 155, 'TRP': 255, 'TYR': 230,
    }
    
    def get_sasa(chain, interface_residues):
        chain_atoms = df_atoms[df_atoms['chainID'] == chain]
        total = 0
        buried = 0
        for resseq in chain_atoms['resSeq'].unique():
            resname = chain_atoms[chain_atoms['resSeq'] == resseq]['resName'].iloc[0]
            sasa = residue_sasa.get(resname, 150)
            total += sasa
            if resseq in interface_residues:
                buried += sasa * 0.5
        return total, buried
    
    total_A, buried_A = get_sasa('A', interface_A)
    total_D, buried_D = get_sasa('D', interface_D)
    return buried_A + buried_D


# ============== SKEMPI Processing ==============

def parse_skempi(skempi_path):
    df = pd.read_csv(skempi_path, sep=';')
    df_brs = df[df['#Pdb'].str.contains('1brs', case=False, na=False)].copy()
    return df_brs


def compute_ddg(row):
    try:
        Ka_mut = float(row['Affinity_mut_parsed'])
        Ka_wt = float(row['Affinity_wt_parsed'])
        R = 1.987e-3  # kcal/mol/K
        T = 298.0
        dG_mut = -R * T * np.log(Ka_mut)
        dG_wt = -R * T * np.log(Ka_wt)
        return (dG_mut - dG_wt) * 1000
    except Exception:
        return np.nan


def parse_mutation(mut_str):
    m = re.match(r'^([A-Z])([A-Z])(\d+)([A-Z])$', mut_str)
    if m:
        return {'wt': m.group(1), 'chain': m.group(2), 'resseq': int(m.group(3)), 'mut': m.group(4)}
    return None


def parse_mutations_combined(mut_str):
    parts = re.split(r'[,+]', mut_str)
    parsed = []
    for p in parts:
        p = p.strip()
        if p:
            r = parse_mutation(p)
            if r:
                parsed.append(r)
    return parsed


# ============== Structural Features ==============

def compute_distance_to_interface(parsed_mut, centroids, interface_A, interface_D):
    chain = parsed_mut['chain']
    resseq = parsed_mut['resseq']
    
    mut_res = centroids[(centroids['chainID'] == chain) & (centroids['resSeq'] == resseq)]
    if len(mut_res) == 0:
        return np.nan
    
    mx, my, mz = mut_res.iloc[0]['x'], mut_res.iloc[0]['y'], mut_res.iloc[0]['z']
    
    if chain == 'A':
        iface_residues = interface_D
        iface_chain = 'D'
    else:
        iface_residues = interface_A
        iface_chain = 'A'
    
    iface = centroids[(centroids['chainID'] == iface_chain) & 
                      (centroids['resSeq'].isin(iface_residues))]
    
    if len(iface) == 0:
        return np.nan
    
    dists = np.sqrt((iface['x']-mx)**2 + (iface['y']-my)**2 + (iface['z']-mz)**2)
    return dists.min()


def compute_haddock_like_score(parsed_mut, df_atoms, interface_A, interface_D):
    chain = parsed_mut['chain']
    resseq = parsed_mut['resseq']
    wt = parsed_mut['wt']
    mut = parsed_mut['mut']
    
    hydro = {
        'ALA': 1.8, 'CYS': 2.5, 'ASP': -3.5, 'GLU': -3.5, 'PHE': 2.8,
        'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LYS': -3.9, 'LEU': 3.8,
        'MET': 1.9, 'ASN': -3.5, 'PRO': -1.6, 'GLN': -3.5, 'ARG': -4.5,
        'SER': -0.8, 'THR': -0.7, 'VAL': 4.2, 'TRP': -0.9, 'TYR': -1.3,
    }
    
    aa1_to3 = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
               'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
               'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
               'W':'TRP','Y':'TYR'}
    
    try:
        wt3 = aa1_to3[wt]
        mut3 = aa1_to3[mut]
    except KeyError:
        return np.nan
    
    d_hydro = hydro.get(mut3, 0) - hydro.get(wt3, 0)
    
    if chain == 'A' and resseq in interface_A:
        is_interface = True
    elif chain == 'D' and resseq in interface_D:
        is_interface = True
    else:
        is_interface = False
    
    positive = {'R', 'K', 'H'}
    negative = {'D', 'E'}
    wt_charge = 1 if wt in positive else (-1 if wt in negative else 0)
    mut_charge = 1 if mut in positive else (-1 if mut in negative else 0)
    d_charge = mut_charge - wt_charge
    
    size = {'A':1,'C':2,'D':4,'E':5,'F':7,'G':0,'H':6,'I':4,'K':5,
            'L':4,'M':4,'N':4,'P':3,'Q':5,'R':7,'S':2,'T':3,'V':3,'W':8,'Y':7}
    d_size = size.get(mut, 3) - size.get(wt, 3)
    
    weight = 3.0 if is_interface else 0.3
    score = weight * (abs(d_hydro) + abs(d_charge) * 2 + abs(d_size) * 0.5)
    
    return score


def compute_neighbor_contacts(parsed_mut, df_atoms, cutoff=6.0):
    chain = parsed_mut['chain']
    resseq = parsed_mut['resseq']
    
    mut_atoms = df_atoms[(df_atoms['chainID'] == chain) & (df_atoms['resSeq'] == resseq)]
    if len(mut_atoms) == 0:
        return np.nan
    
    other_atoms = df_atoms[df_atoms['chainID'] != chain]
    
    contacts = 0
    for _, ma in mut_atoms.iterrows():
        dx = other_atoms['x'] - ma['x']
        dy = other_atoms['y'] - ma['y']
        dz = other_atoms['z'] - ma['z']
        dists = np.sqrt(dx**2 + dy**2 + dz**2)
        contacts += (dists < cutoff).sum()
    
    return contacts


def compute_solvent_accessibility(parsed_mut, df_atoms):
    chain = parsed_mut['chain']
    resseq = parsed_mut['resseq']
    
    res_atoms = df_atoms[(df_atoms['chainID'] == chain) & (df_atoms['resSeq'] == resseq)]
    if len(res_atoms) == 0:
        return np.nan
    
    ca = res_atoms[res_atoms['name'] == 'CA']
    if len(ca) == 0:
        ca = res_atoms.iloc[0:1]
    
    all_coords = df_atoms[df_atoms['chainID'] == chain][['x','y','z']].values
    center = all_coords.mean(axis=0)
    ca_pos = ca[['x','y','z']].values[0]
    dist_to_center = np.linalg.norm(ca_pos - center)
    
    max_dist = np.max(np.linalg.norm(all_coords - center, axis=1))
    rsa_approx = dist_to_center / max_dist if max_dist > 0 else 0
    
    return rsa_approx


# ============== Main Analysis ==============

def main():
    print("=== Parsing PDB ===")
    df_atoms = parse_pdb(PDB_PATH)
    centroids = get_residue_centroids(df_atoms)
    
    print("=== Computing Interface Residues (all atoms, 5Å cutoff) ===")
    interface_A, interface_D = compute_interface_residues_all_atoms(df_atoms, cutoff=5.0)
    print(f"Interface A (barnase): {len(interface_A)} residues: {interface_A}")
    print(f"Interface D (barstar): {len(interface_D)} residues: {interface_D}")
    
    bsa = compute_buried_surface_area(df_atoms, interface_A, interface_D)
    print(f"Approximate BSA: {bsa:.1f} Å²")
    
    print("\n=== Parsing SKEMPI ===")
    df_skempi = parse_skempi(SKEMPI_PATH)
    print(f"1BRS entries: {len(df_skempi)}")
    
    df_skempi['ddG_cal_mol'] = df_skempi.apply(compute_ddg, axis=1)
    df_skempi['ddG_kcal_mol'] = df_skempi['ddG_cal_mol'] / 1000.0
    
    df_skempi['parsed_mutations'] = df_skempi['Mutation(s)_PDB'].apply(parse_mutations_combined)
    df_skempi['n_mutations'] = df_skempi['parsed_mutations'].apply(len)
    
    df_single = df_skempi[df_skempi['n_mutations'] == 1].copy()
    print(f"Single mutations: {len(df_single)}")
    
    df_single['mut_chain'] = df_single['parsed_mutations'].apply(lambda x: x[0]['chain'] if x else None)
    df_single['mut_resseq'] = df_single['parsed_mutations'].apply(lambda x: x[0]['resseq'] if x else None)
    df_single['mut_wt'] = df_single['parsed_mutations'].apply(lambda x: x[0]['wt'] if x else None)
    df_single['mut_mut'] = df_single['parsed_mutations'].apply(lambda x: x[0]['mut'] if x else None)
    
    def is_interface(row):
        if row['mut_chain'] == 'A':
            return row['mut_resseq'] in interface_A
        elif row['mut_chain'] == 'D':
            return row['mut_resseq'] in interface_D
        return False
    
    df_single['is_interface'] = df_single.apply(is_interface, axis=1)
    
    print("\n=== Computing Structural Features ===")
    df_single['dist_to_interface'] = df_single['parsed_mutations'].apply(
        lambda x: compute_distance_to_interface(x[0], centroids, interface_A, interface_D) if x else np.nan)
    df_single['haddock_score'] = df_single['parsed_mutations'].apply(
        lambda x: compute_haddock_like_score(x[0], df_atoms, interface_A, interface_D) if x else np.nan)
    df_single['neighbor_contacts'] = df_single['parsed_mutations'].apply(
        lambda x: compute_neighbor_contacts(x[0], df_atoms) if x else np.nan)
    df_single['rsa_approx'] = df_single['parsed_mutations'].apply(
        lambda x: compute_solvent_accessibility(x[0], df_atoms) if x else np.nan)
    
    loc_map = {
        'COR': 'Core',
        'SUP': 'Support',
        'RIM': 'Rim',
        'SUR': 'Surface',
        'INT': 'Interface',
    }
    df_single['location_clean'] = df_single['iMutation_Location(s)'].apply(
        lambda x: loc_map.get(str(x).split(',')[0], str(x)))
    
    df_single = df_single.dropna(subset=['ddG_kcal_mol'])
    print(f"Final single-mutation dataset: {len(df_single)} entries")
    
    df_single.to_csv(os.path.join(OUTPUT_DIR, 'processed_mutations.csv'), index=False)
    
    # ============== Statistics ==============
    print("\n=== Statistics ===")
    print("\nΔΔG distribution:")
    print(df_single['ddG_kcal_mol'].describe())
    
    print("\nBy interface location:")
    print(df_single.groupby('is_interface')['ddG_kcal_mol'].agg(['count','mean','std','median']))
    
    print("\nBy mutation location class:")
    print(df_single.groupby('location_clean')['ddG_kcal_mol'].agg(['count','mean','std','median']))
    
    features = ['dist_to_interface', 'haddock_score', 'neighbor_contacts', 'rsa_approx']
    corr_results = []
    for feat in features:
        valid = df_single.dropna(subset=[feat, 'ddG_kcal_mol'])
        if len(valid) > 5:
            r, p = stats.pearsonr(valid[feat], valid['ddG_kcal_mol'])
            corr_results.append({'feature': feat, 'r': r, 'p': p, 'n': len(valid)})
            print(f"\n{feat} vs ddG: r={r:.3f}, p={p:.4f}, n={len(valid)}")
    
    pd.DataFrame(corr_results).to_csv(os.path.join(OUTPUT_DIR, 'correlations.csv'), index=False)
    
    # ============== Figures ==============
    print("\n=== Generating Figures ===")
    
    # Figure 1: Data overview
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    ax = axes[0]
    ax.hist(df_single['ddG_kcal_mol'], bins=25, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel(r'$\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Binding Affinity Changes')
    
    ax = axes[1]
    data_violin = [df_single[df_single['is_interface']==True]['ddG_kcal_mol'].values,
                   df_single[df_single['is_interface']==False]['ddG_kcal_mol'].values]
    positions = [1, 2]
    parts = ax.violinplot(data_violin, positions=positions, showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Interface', 'Non-interface'])
    ax.set_ylabel(r'$\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_title('ΔΔG by Interface Location')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure1_data_overview.png'))
    plt.close()
    
    # Figure 2: Structural features scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    valid = df_single.dropna(subset=['dist_to_interface', 'ddG_kcal_mol'])
    ax = axes[0,0]
    colors = ['crimson' if x else 'steelblue' for x in valid['is_interface']]
    ax.scatter(valid['dist_to_interface'], valid['ddG_kcal_mol'], c=colors, alpha=0.6, edgecolors='none')
    ax.set_xlabel('Distance to Interface (Å)')
    ax.set_ylabel(r'$\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_title('Distance to Interface vs ΔΔG')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # Add regression line
    if len(valid) > 3:
        z = np.polyfit(valid['dist_to_interface'], valid['ddG_kcal_mol'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['dist_to_interface'].min(), valid['dist_to_interface'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1, alpha=0.5)
    
    valid = df_single.dropna(subset=['haddock_score', 'ddG_kcal_mol'])
    ax = axes[0,1]
    colors = ['crimson' if x else 'steelblue' for x in valid['is_interface']]
    ax.scatter(valid['haddock_score'], valid['ddG_kcal_mol'], c=colors, alpha=0.6, edgecolors='none')
    ax.set_xlabel('HADDOCK-like Score')
    ax.set_ylabel(r'$\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_title('HADDOCK-like Score vs ΔΔG')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    if len(valid) > 3:
        z = np.polyfit(valid['haddock_score'], valid['ddG_kcal_mol'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['haddock_score'].min(), valid['haddock_score'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1, alpha=0.5)
    
    valid = df_single.dropna(subset=['neighbor_contacts', 'ddG_kcal_mol'])
    ax = axes[1,0]
    colors = ['crimson' if x else 'steelblue' for x in valid['is_interface']]
    ax.scatter(valid['neighbor_contacts'], valid['ddG_kcal_mol'], c=colors, alpha=0.6, edgecolors='none')
    ax.set_xlabel('Inter-chain Contacts')
    ax.set_ylabel(r'$\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_title('Inter-chain Contacts vs ΔΔG')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    if len(valid) > 3:
        z = np.polyfit(valid['neighbor_contacts'], valid['ddG_kcal_mol'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['neighbor_contacts'].min(), valid['neighbor_contacts'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1, alpha=0.5)
    
    valid = df_single.dropna(subset=['rsa_approx', 'ddG_kcal_mol'])
    ax = axes[1,1]
    colors = ['crimson' if x else 'steelblue' for x in valid['is_interface']]
    ax.scatter(valid['rsa_approx'], valid['ddG_kcal_mol'], c=colors, alpha=0.6, edgecolors='none')
    ax.set_xlabel('Approximate Relative SASA')
    ax.set_ylabel(r'$\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_title('Solvent Accessibility vs ΔΔG')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    if len(valid) > 3:
        z = np.polyfit(valid['rsa_approx'], valid['ddG_kcal_mol'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['rsa_approx'].min(), valid['rsa_approx'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1, alpha=0.5)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=8, label='Interface'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Non-interface')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure2_structural_correlations.png'), bbox_inches='tight')
    plt.close()
    
    # Figure 3: Location class comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    loc_order = ['Core', 'Support', 'Rim', 'Surface']
    data_to_plot = [df_single[df_single['location_clean']==loc]['ddG_kcal_mol'].values 
                    for loc in loc_order if loc in df_single['location_clean'].values]
    tick_labels = [loc for loc in loc_order if loc in df_single['location_clean'].values]
    
    bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True)
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel(r'$\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_title('Binding Affinity Changes by Mutation Location Class')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure3_location_classes.png'))
    plt.close()
    
    # Figure 4: Residue-level bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for idx, (chain, name, iface) in enumerate([('A', 'Barnase (chain A)', interface_A), 
                                                  ('D', 'Barstar (chain D)', interface_D)]):
        chain_data = df_single[df_single['mut_chain'] == chain].copy()
        if len(chain_data) == 0:
            continue
        
        res_ddg = chain_data.groupby('mut_resseq')['ddG_kcal_mol'].mean().to_dict()
        resseqs = sorted(chain_data['mut_resseq'].unique())
        
        values = [res_ddg.get(r, 0) for r in resseqs]
        colors = ['crimson' if r in iface else 'lightgray' for r in resseqs]
        
        ax = axes[idx]
        bars = ax.bar(range(len(resseqs)), values, color=colors, edgecolor='white', linewidth=0.3)
        step = max(1, len(resseqs)//10)
        ax.set_xticks(range(0, len(resseqs), step))
        ax.set_xticklabels([resseqs[i] for i in range(0, len(resseqs), step)], rotation=45)
        ax.set_xlabel('Residue Number')
        ax.set_ylabel(r'Mean $\Delta\Delta G$ (kcal/mol)')
        ax.set_title(name)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure4_residue_level_map.png'))
    plt.close()
    
    # Figure 5: Validation - predicted vs experimental
    valid = df_single.dropna(subset=['haddock_score', 'ddG_kcal_mol'])
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    X = valid[['haddock_score', 'dist_to_interface', 'neighbor_contacts']].fillna(
        valid[['haddock_score', 'dist_to_interface', 'neighbor_contacts']].median())
    y = valid['ddG_kcal_mol'].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_pred, c=['crimson' if x else 'steelblue' for x in valid['is_interface']], 
               alpha=0.6, edgecolors='none')
    lim = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lim, lim, 'k--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel(r'Experimental $\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_ylabel(r'Predicted $\Delta\Delta G_{\mathrm{bind}}$ (kcal/mol)')
    ax.set_title(f'Prediction Validation (R²={r2_score(y, y_pred):.3f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure5_validation.png'))
    plt.close()
    
    metrics = {
        'r2': float(r2_score(y, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'n_samples': int(len(y)),
        'coefficients': {name: float(coef) for name, coef in zip(X.columns, reg.coef_)},
        'intercept': float(reg.intercept_),
    }
    with open(os.path.join(OUTPUT_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n=== Model Metrics ===")
    print(json.dumps(metrics, indent=2))
    
    # Figure 6: Interface residue contact map
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ca_atoms = df_atoms[df_atoms['name'] == 'CA']
    ca_A = ca_atoms[ca_atoms['chainID'] == 'A'].sort_values('resSeq').reset_index(drop=True)
    ca_D = ca_atoms[ca_atoms['chainID'] == 'D'].sort_values('resSeq').reset_index(drop=True)
    
    contact_matrix = np.zeros((len(ca_A), len(ca_D)))
    for i, (_, ra) in enumerate(ca_A.iterrows()):
        for j, (_, rd) in enumerate(ca_D.iterrows()):
            dist = np.sqrt((ra['x']-rd['x'])**2 + (ra['y']-rd['y'])**2 + (ra['z']-rd['z'])**2)
            contact_matrix[i, j] = dist
    
    im = ax.imshow(contact_matrix, cmap='viridis_r', aspect='auto')
    ax.set_xlabel('Barstar Residue Number')
    ax.set_ylabel('Barnase Residue Number')
    ax.set_title('Inter-chain CA Distance Map (Å)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (Å)')
    
    iface_A_idx = [i for i, (_, r) in enumerate(ca_A.iterrows()) if r['resSeq'] in interface_A]
    iface_D_idx = [j for j, (_, r) in enumerate(ca_D.iterrows()) if r['resSeq'] in interface_D]
    
    for i in iface_A_idx:
        ax.axhline(i, color='red', alpha=0.1, linewidth=0.5)
    for j in iface_D_idx:
        ax.axvline(j, color='red', alpha=0.1, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure6_contact_map.png'))
    plt.close()
    
    # Figure 7: Energy terms breakdown by interface vs non-interface
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    valid = df_single.dropna(subset=['haddock_score'])
    ax = axes[0]
    ax.boxplot([valid[valid['is_interface']==True]['haddock_score'].values,
                valid[valid['is_interface']==False]['haddock_score'].values],
               tick_labels=['Interface', 'Non-interface'], patch_artist=True)
    ax.set_ylabel('HADDOCK-like Score')
    ax.set_title('Score Distribution')
    
    ax = axes[1]
    ax.boxplot([valid[valid['is_interface']==True]['neighbor_contacts'].values,
                valid[valid['is_interface']==False]['neighbor_contacts'].values],
               tick_labels=['Interface', 'Non-interface'], patch_artist=True)
    ax.set_ylabel('Inter-chain Contacts')
    ax.set_title('Contact Distribution')
    
    ax = axes[2]
    ax.boxplot([valid[valid['is_interface']==True]['dist_to_interface'].values,
                valid[valid['is_interface']==False]['dist_to_interface'].values],
               tick_labels=['Interface', 'Non-interface'], patch_artist=True)
    ax.set_ylabel('Distance to Interface (Å)')
    ax.set_title('Distance Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'figure7_feature_comparison.png'))
    plt.close()
    
    # Save summary
    summary = {
        'n_total_mutations': int(len(df_skempi)),
        'n_single_mutations': int(len(df_single)),
        'n_interface_mutations': int(df_single['is_interface'].sum()),
        'n_noninterface_mutations': int((~df_single['is_interface']).sum()),
        'interface_A_residues': interface_A,
        'interface_D_residues': interface_D,
        'approximate_BSA_A2': float(bsa),
        'mean_ddG_interface_kcal_mol': float(df_single[df_single['is_interface']]['ddG_kcal_mol'].mean()),
        'mean_ddG_noninterface_kcal_mol': float(df_single[~df_single['is_interface']]['ddG_kcal_mol'].mean()),
        'std_ddG_interface_kcal_mol': float(df_single[df_single['is_interface']]['ddG_kcal_mol'].std()),
        'std_ddG_noninterface_kcal_mol': float(df_single[~df_single['is_interface']]['ddG_kcal_mol'].std()),
    }
    with open(os.path.join(OUTPUT_DIR, 'summary_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
