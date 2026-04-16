#!/usr/bin/env python3
"""
Generate all figures for the HADDOCK-inspired barnase-barstar analysis report.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

os.makedirs('report/images', exist_ok=True)

# Load data
with open('outputs/interface_residues.json', 'r') as f:
    interface_data = json.load(f)

with open('outputs/mutation_data.json', 'r') as f:
    mutation_data = json.load(f)

with open('outputs/residue_ddG.json', 'r') as f:
    residue_ddG = json.load(f)

with open('outputs/residue_energy.json', 'r') as f:
    residue_energy = json.load(f)

with open('outputs/contacts.json', 'r') as f:
    contacts_data = json.load(f)

# ============================================================
# Figure 1: Interface Residue Map
# ============================================================

def fig1_interface_residue_map():
    """Plot interface residues on both chains with their properties."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Chain A
    ax = axes[0]
    a_residues = interface_data['chain_a']
    resnums_a = [r['resnum'] for r in a_residues]
    resnames_a = [r['resname'] for r in a_residues]
    
    # Color by property
    colors_a = []
    for rn in resnames_a:
        if rn in ['ARG', 'LYS']:
            colors_a.append('#e74c3c')  # positive
        elif rn in ['ASP', 'GLU']:
            colors_a.append('#3498db')  # negative
        elif rn in ['SER', 'THR', 'ASN', 'GLN', 'HIS', 'TYR']:
            colors_a.append('#2ecc71')  # polar
        else:
            colors_a.append('#f39c12')  # hydrophobic
    
    bars = ax.bar(range(len(resnums_a)), [1]*len(resnums_a), color=colors_a, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(resnums_a)))
    ax.set_xticklabels([f"{rn}{rn2}" for rn, rn2 in zip(resnames_a, resnums_a)], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Interface Residue')
    ax.set_title('Chain A (Barnase) Interface Residues')
    ax.set_yticks([])
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='Positive'),
        mpatches.Patch(facecolor='#3498db', label='Negative'),
        mpatches.Patch(facecolor='#2ecc71', label='Polar'),
        mpatches.Patch(facecolor='#f39c12', label='Hydrophobic')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Chain D
    ax = axes[1]
    d_residues = interface_data['chain_d']
    resnums_d = [r['resnum'] for r in d_residues]
    resnames_d = [r['resname'] for r in d_residues]
    
    colors_d = []
    for rn in resnames_d:
        if rn in ['ARG', 'LYS']:
            colors_d.append('#e74c3c')
        elif rn in ['ASP', 'GLU']:
            colors_d.append('#3498db')
        elif rn in ['SER', 'THR', 'ASN', 'GLN', 'HIS', 'TYR']:
            colors_d.append('#2ecc71')
        else:
            colors_d.append('#f39c12')
    
    bars = ax.bar(range(len(resnums_d)), [1]*len(resnums_d), color=colors_d, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(resnums_d)))
    ax.set_xticklabels([f"{rn}{rn2}" for rn, rn2 in zip(resnames_d, resnums_d)], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Interface Residue')
    ax.set_title('Chain D (Barstar) Interface Residues')
    ax.set_yticks([])
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('report/images/fig1_interface_residues.png')
    plt.close()
    print("Figure 1 saved: fig1_interface_residues.png")

# ============================================================
# Figure 2: ddG Distribution by Mutation Location
# ============================================================

def fig2_ddg_distribution():
    """Plot distribution of ddG values by structural location."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    ddG_all = [m['ddG'] for m in mutation_data]
    ddG_interface = [m['ddG'] for m in mutation_data if m['location_struct'] == 'interface']
    ddG_non_interface = [m['ddG'] for m in mutation_data if m['location_struct'] == 'non-interface']
    
    # Overall distribution
    ax = axes[0]
    ax.hist(ddG_all, bins=30, color='#2c3e50', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=np.mean(ddG_all), color='red', linestyle='--', label=f'Mean={np.mean(ddG_all):.2f}')
    ax.set_xlabel('ΔΔG (kcal/mol)')
    ax.set_ylabel('Count')
    ax.set_title('All Mutations')
    ax.legend()
    
    # Interface vs non-interface
    ax = axes[1]
    ax.hist(ddG_interface, bins=25, color='#e74c3c', alpha=0.7, label=f'Interface (n={len(ddG_interface)})', edgecolor='black', linewidth=0.5)
    ax.hist(ddG_non_interface, bins=25, color='#3498db', alpha=0.7, label=f'Non-interface (n={len(ddG_non_interface)})', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('ΔΔG (kcal/mol)')
    ax.set_ylabel('Count')
    ax.set_title('Interface vs Non-interface Mutations')
    ax.legend()
    
    # Box plot
    ax = axes[2]
    data_box = [ddG_interface, ddG_non_interface]
    bp = ax.boxplot(data_box, labels=['Interface', 'Non-interface'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#3498db')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('ΔΔG (kcal/mol)')
    ax.set_title('ΔΔG by Location')
    
    # Add statistical test
    from scipy import stats
    t_stat, p_val = stats.mannwhitneyu(ddG_interface, ddG_non_interface, alternative='greater')
    ax.text(0.5, 0.95, f'Mann-Whitney p={p_val:.4f}', transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('report/images/fig2_ddg_distribution.png')
    plt.close()
    print("Figure 2 saved: fig2_ddg_distribution.png")

# ============================================================
# Figure 3: Per-Residue ddG Heatmap
# ============================================================

def fig3_per_residue_ddg():
    """Plot per-residue average ddG as a bar chart."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Chain A
    ax = axes[0]
    a_keys = sorted([k for k in residue_ddG.keys() if k.startswith('A_')], key=lambda x: int(x.split('_')[1]))
    a_resnums = [int(k.split('_')[1]) for k in a_keys]
    a_ddG = [residue_ddG[k]['mean_ddG'] for k in a_keys]
    a_std = [residue_ddG[k]['std_ddG'] for k in a_keys]
    a_n = [residue_ddG[k]['n_mutations'] for k in a_keys]
    
    # Check if interface
    interface_a_resnums = set(r['resnum'] for r in interface_data['chain_a'])
    a_colors = ['#e74c3c' if rn in interface_a_resnums else '#3498db' for rn in a_resnums]
    
    bars = ax.bar(range(len(a_keys)), a_ddG, yerr=a_std, color=a_colors, edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(a_keys)))
    ax.set_xticklabels([f"A{rn}" for rn in a_resnums], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean ΔΔG (kcal/mol)')
    ax.set_title('Chain A (Barnase) - Per-Residue Mean ΔΔG')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Annotate number of mutations
    for i, (bar, n) in enumerate(zip(bars, a_n)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + a_std[i] + 0.2,
                f'n={n}', ha='center', va='bottom', fontsize=7)
    
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='Interface'),
        mpatches.Patch(facecolor='#3498db', label='Non-interface')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Chain D
    ax = axes[1]
    d_keys = sorted([k for k in residue_ddG.keys() if k.startswith('D_')], key=lambda x: int(x.split('_')[1]))
    d_resnums = [int(k.split('_')[1]) for k in d_keys]
    d_ddG = [residue_ddG[k]['mean_ddG'] for k in d_keys]
    d_std = [residue_ddG[k]['std_ddG'] for k in d_keys]
    d_n = [residue_ddG[k]['n_mutations'] for k in d_keys]
    
    interface_d_resnums = set(r['resnum'] for r in interface_data['chain_d'])
    d_colors = ['#e74c3c' if rn in interface_d_resnums else '#3498db' for rn in d_resnums]
    
    bars = ax.bar(range(len(d_keys)), d_ddG, yerr=d_std, color=d_colors, edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(d_keys)))
    ax.set_xticklabels([f"D{rn}" for rn in d_resnums], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean ΔΔG (kcal/mol)')
    ax.set_title('Chain D (Barstar) - Per-Residue Mean ΔΔG')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for i, (bar, n) in enumerate(zip(bars, d_n)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + d_std[i] + 0.2,
                f'n={n}', ha='center', va='bottom', fontsize=7)
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('report/images/fig3_per_residue_ddg.png')
    plt.close()
    print("Figure 3 saved: fig3_per_residue_ddg.png")

# ============================================================
# Figure 4: Interaction Energy vs Experimental ddG
# ============================================================

def fig4_energy_vs_ddg():
    """Correlate computed interaction energy with experimental ddG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect data points
    common_residues = set(residue_ddG.keys()) & set(residue_energy.keys())
    
    x_data = []
    y_data = []
    labels = []
    colors = []
    
    interface_a_resnums = set(r['resnum'] for r in interface_data['chain_a'])
    interface_d_resnums = set(r['resnum'] for r in interface_data['chain_d'])
    
    for key in common_residues:
        chain = key[0]
        resnum = int(key.split('_')[1])
        ddG = residue_ddG[key]['mean_ddG']
        energy = residue_energy[key]['interaction_energy']
        n_contacts = residue_energy[key]['n_contacts']
        
        x_data.append(abs(energy))
        y_data.append(ddG)
        labels.append(key)
        
        if chain == 'A' and resnum in interface_a_resnums:
            colors.append('#e74c3c')
        elif chain == 'D' and resnum in interface_d_resnums:
            colors.append('#e74c3c')
        else:
            colors.append('#3498db')
    
    # Scatter plot
    ax = axes[0]
    for x, y, c, l in zip(x_data, y_data, colors, labels):
        ax.scatter(x, y, c=c, s=80, edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(l.replace('_', ''), (x, y), fontsize=7, ha='left', va='bottom')
    
    # Fit line
    if len(x_data) > 2:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'k--', alpha=0.5, label=f'R²={r_value**2:.3f}, p={p_value:.4f}')
        ax.legend()
    
    ax.set_xlabel('|Interaction Energy| (a.u.)')
    ax.set_ylabel('Mean ΔΔG (kcal/mol)')
    ax.set_title('Interaction Energy vs Experimental ΔΔG')
    
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='Interface'),
        mpatches.Patch(facecolor='#3498db', label='Non-interface')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Contact count vs ddG
    ax = axes[1]
    x_contacts = []
    y_ddG = []
    c_contacts = []
    
    for key in common_residues:
        chain = key[0]
        resnum = int(key.split('_')[1])
        ddG = residue_ddG[key]['mean_ddG']
        n_contacts = residue_energy[key]['n_contacts']
        
        x_contacts.append(n_contacts)
        y_ddG.append(ddG)
        
        if chain == 'A' and resnum in interface_a_resnums:
            c_contacts.append('#e74c3c')
        elif chain == 'D' and resnum in interface_d_resnums:
            c_contacts.append('#e74c3c')
        else:
            c_contacts.append('#3498db')
    
    for x, y, c, l in zip(x_contacts, y_ddG, c_contacts, labels):
        ax.scatter(x, y, c=c, s=80, edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(l.replace('_', ''), (x, y), fontsize=7, ha='left', va='bottom')
    
    if len(x_contacts) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_contacts, y_ddG)
        x_fit = np.linspace(min(x_contacts), max(x_contacts), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, 'k--', alpha=0.5, label=f'R²={r_value**2:.3f}, p={p_value:.4f}')
        ax.legend()
    
    ax.set_xlabel('Number of Inter-chain Contacts')
    ax.set_ylabel('Mean ΔΔG (kcal/mol)')
    ax.set_title('Contact Count vs Experimental ΔΔG')
    
    plt.tight_layout()
    plt.savefig('report/images/fig4_energy_vs_ddg.png')
    plt.close()
    print("Figure 4 saved: fig4_energy_vs_ddg.png")

# ============================================================
# Figure 5: SKEMPI Location Classification Comparison
# ============================================================

def fig5_location_classification():
    """Compare SKEMPI location labels with structure-based classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parse SKEMPI locations
    skempi_locs = defaultdict(list)
    for m in mutation_data:
        # Parse the SKEMPI location string
        locs = m['location_skempi'].split(',')
        for loc in locs:
            skempi_locs[loc.strip()].append(m['ddG'])
    
    # Bar chart of mean ddG by SKEMPI location
    ax = axes[0]
    loc_order = ['COR', 'RIM', 'SUP', 'SUR', 'INT']
    loc_names = {'COR': 'Core', 'RIM': 'Rim', 'SUP': 'Support', 'SUR': 'Surface', 'INT': 'Interior'}
    loc_colors = {'COR': '#e74c3c', 'RIM': '#f39c12', 'SUP': '#2ecc71', 'SUR': '#3498db', 'INT': '#9b59b6'}
    
    means = []
    stds = []
    labels_plot = []
    colors_plot = []
    
    for loc in loc_order:
        if loc in skempi_locs and len(skempi_locs[loc]) > 0:
            means.append(np.mean(skempi_locs[loc]))
            stds.append(np.std(skempi_locs[loc]))
            labels_plot.append(f"{loc_names.get(loc, loc)}\n(n={len(skempi_locs[loc])})")
            colors_plot.append(loc_colors.get(loc, 'gray'))
    
    bars = ax.bar(range(len(means)), means, yerr=stds, color=colors_plot, edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot)
    ax.set_ylabel('Mean ΔΔG (kcal/mol)')
    ax.set_title('ΔΔG by SKEMPI Mutation Location')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Structure-based classification
    ax = axes[1]
    struct_locs = defaultdict(list)
    for m in mutation_data:
        struct_locs[m['location_struct']].append(m['ddG'])
    
    struct_means = []
    struct_stds = []
    struct_labels = []
    struct_colors = []
    
    for loc in ['interface', 'non-interface']:
        if loc in struct_locs:
            struct_means.append(np.mean(struct_locs[loc]))
            struct_stds.append(np.std(struct_locs[loc]))
            struct_labels.append(f"{loc.title()}\n(n={len(struct_locs[loc])})")
            struct_colors.append('#e74c3c' if loc == 'interface' else '#3498db')
    
    bars = ax.bar(range(len(struct_means)), struct_means, yerr=struct_stds, color=struct_colors, 
                  edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(struct_labels)))
    ax.set_xticklabels(struct_labels)
    ax.set_ylabel('Mean ΔΔG (kcal/mol)')
    ax.set_title('ΔΔG by Structure-Based Location')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('report/images/fig5_location_classification.png')
    plt.close()
    print("Figure 5 saved: fig5_location_classification.png")

# ============================================================
# Figure 6: Contact Map
# ============================================================

def fig6_contact_map():
    """Plot inter-chain contact map."""
    # Parse PDB to get residue info
    atoms = []
    with open('data/1brs_AD.pdb', 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms.append({
                    'chain': line[21],
                    'resseq': int(line[22:26]),
                    'resname': line[17:20].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54])
                })
    
    # Get unique residues
    chain_a_residues = sorted(set((a['resseq'], a['resname']) for a in atoms if a['chain'] == 'A'))
    chain_d_residues = sorted(set((a['resseq'], a['resname']) for a in atoms if a['chain'] == 'D'))
    
    # Build contact matrix
    contact_matrix = np.zeros((len(chain_a_residues), len(chain_d_residues)))
    
    coords_a = {}
    coords_d = {}
    
    for a in atoms:
        key = a['resseq']
        coord = np.array([a['x'], a['y'], a['z']])
        if a['chain'] == 'A':
            if key not in coords_a:
                coords_a[key] = []
            coords_a[key].append(coord)
        else:
            if key not in coords_d:
                coords_d[key] = []
            coords_d[key].append(coord)
    
    for i, (res_a, rn_a) in enumerate(chain_a_residues):
        for j, (res_d, rn_d) in enumerate(chain_d_residues):
            if res_a in coords_a and res_d in coords_d:
                ca = np.array(coords_a[res_a])
                cd = np.array(coords_d[res_d])
                # Compute minimum distance
                min_dist = np.inf
                for c1 in ca:
                    dists = np.sqrt(np.sum((cd - c1)**2, axis=1))
                    d = np.min(dists)
                    if d < min_dist:
                        min_dist = d
                if min_dist < 5.0:
                    contact_matrix[i, j] = min_dist
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mask zeros
    masked = np.ma.masked_where(contact_matrix == 0, contact_matrix)
    
    im = ax.imshow(masked, cmap='YlOrRd_r', aspect='auto', vmin=2.5, vmax=5.0)
    
    # Labels
    a_labels = [f"{rn}{res}" for res, rn in chain_a_residues]
    d_labels = [f"{rn}{res}" for res, rn in chain_d_residues]
    
    # Only show every nth label for readability
    step_a = max(1, len(a_labels) // 20)
    step_d = max(1, len(d_labels) // 20)
    
    ax.set_xticks(range(len(d_labels)))
    ax.set_xticklabels(d_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(a_labels)))
    ax.set_yticklabels(a_labels, fontsize=6)
    
    ax.set_xlabel('Chain D (Barstar) Residues')
    ax.set_ylabel('Chain A (Barnase) Residues')
    ax.set_title('Inter-chain Contact Map (distance < 5Å)')
    
    plt.colorbar(im, ax=ax, label='Minimum Distance (Å)', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('report/images/fig6_contact_map.png')
    plt.close()
    print("Figure 6 saved: fig6_contact_map.png")

# ============================================================
# Figure 7: HADDOCK Scoring Function Analysis
# ============================================================

def fig7_haddock_scoring():
    """Analyze HADDOCK-inspired scoring components."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Interface composition pie chart
    ax = axes[0]
    props = interface_data['properties']
    sizes = [props.get('polar_count', 0), props.get('nonpolar_count', 0), props.get('charged_count', 0)]
    # Adjust: charged are subset of polar
    polar_only = sizes[0] - sizes[2]
    sizes_adj = [polar_only, sizes[1], sizes[2]]
    labels = [f'Polar\n({polar_only})', f'Non-polar\n({sizes[1]})', f'Charged\n({sizes[2]})']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(sizes_adj, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 10})
    ax.set_title('Interface Composition')
    
    # Panel B: ddG by mutation type
    ax = axes[1]
    mut_types = defaultdict(list)
    for m in mutation_data:
        if m['mt'] == 'A':
            mut_types['Ala scan'].append(m['ddG'])
        elif m['wt'] in ['R', 'K'] and m['mt'] in ['D', 'E']:
            mut_types['Charge rev.'].append(m['ddG'])
        elif m['wt'] in ['D', 'E'] and m['mt'] in ['R', 'K']:
            mut_types['Charge rev.'].append(m['ddG'])
        elif m['mt'] == 'F' or m['mt'] == 'W' or m['mt'] == 'Y':
            mut_types['Aromatic'].append(m['ddG'])
        else:
            mut_types['Other'].append(m['ddG'])
    
    type_order = ['Ala scan', 'Charge rev.', 'Aromatic', 'Other']
    type_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    data_box = []
    labels_box = []
    colors_box = []
    for t, c in zip(type_order, type_colors):
        if t in mut_types and len(mut_types[t]) > 0:
            data_box.append(mut_types[t])
            labels_box.append(f"{t}\n(n={len(mut_types[t])})")
            colors_box.append(c)
    
    bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('ΔΔG (kcal/mol)')
    ax.set_title('ΔΔG by Mutation Type')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Panel C: Chain-specific ddG comparison
    ax = axes[2]
    ddG_a = [m['ddG'] for m in mutation_data if m['chain'] == 'A']
    ddG_d = [m['ddG'] for m in mutation_data if m['chain'] == 'D']
    
    bp = ax.boxplot([ddG_a, ddG_d], labels=[f'Chain A\n(n={len(ddG_a)})', f'Chain D\n(n={len(ddG_d)})'], 
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#3498db')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('ΔΔG (kcal/mol)')
    ax.set_title('ΔΔG by Chain')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    from scipy import stats
    if len(ddG_a) > 0 and len(ddG_d) > 0:
        t_stat, p_val = stats.mannwhitneyu(ddG_a, ddG_d)
        ax.text(0.5, 0.95, f'Mann-Whitney p={p_val:.4f}', transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('report/images/fig7_haddock_scoring.png')
    plt.close()
    print("Figure 7 saved: fig7_haddock_scoring.png")

# ============================================================
# Figure 8: Mutation Hotspot Map
# ============================================================

def fig8_hotspot_map():
    """Create a linear hotspot map showing ddG along the sequence."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    
    # Parse PDB for residue info
    atoms = []
    with open('data/1brs_AD.pdb', 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms.append({
                    'chain': line[21],
                    'resseq': int(line[22:26]),
                    'resname': line[17:20].strip()
                })
    
    chain_a_residues = sorted(set((a['resseq'], a['resname']) for a in atoms if a['chain'] == 'A'))
    chain_d_residues = sorted(set((a['resseq'], a['resname']) for a in atoms if a['chain'] == 'D'))
    
    interface_a_resnums = set(r['resnum'] for r in interface_data['chain_a'])
    interface_d_resnums = set(r['resnum'] for r in interface_data['chain_d'])
    
    # Chain A
    ax = axes[0]
    a_resnums_all = [r[0] for r in chain_a_residues]
    a_ddG_map = {}
    for key, val in residue_ddG.items():
        if key.startswith('A_'):
            a_ddG_map[int(key.split('_')[1])] = val['mean_ddG']
    
    x_vals = []
    y_vals = []
    c_vals = []
    
    for rn in a_resnums_all:
        if rn in a_ddG_map:
            x_vals.append(rn)
            y_vals.append(a_ddG_map[rn])
            c_vals.append('#e74c3c' if rn in interface_a_resnums else '#3498db')
    
    if x_vals:
        bars = ax.bar(x_vals, y_vals, color=c_vals, edgecolor='black', linewidth=0.5, width=1.5)
        for x, y, c in zip(x_vals, y_vals, c_vals):
            ax.annotate(f'{y:.1f}', (x, y), fontsize=7, ha='center', va='bottom')
    
    # Mark interface regions
    for rn in a_resnums_all:
        if rn in interface_a_resnums:
            ax.axvspan(rn-0.75, rn+0.75, alpha=0.1, color='red')
    
    ax.set_xlabel('Residue Number')
    ax.set_ylabel('Mean ΔΔG (kcal/mol)')
    ax.set_title('Chain A (Barnase) - Mutation Hotspot Map')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=2.0, color='orange', linestyle='--', linewidth=0.5, alpha=0.5, label='Hotspot threshold (2 kcal/mol)')
    ax.legend(fontsize=8)
    
    # Chain D
    ax = axes[1]
    d_resnums_all = [r[0] for r in chain_d_residues]
    d_ddG_map = {}
    for key, val in residue_ddG.items():
        if key.startswith('D_'):
            d_ddG_map[int(key.split('_')[1])] = val['mean_ddG']
    
    x_vals = []
    y_vals = []
    c_vals = []
    
    for rn in d_resnums_all:
        if rn in d_ddG_map:
            x_vals.append(rn)
            y_vals.append(d_ddG_map[rn])
            c_vals.append('#e74c3c' if rn in interface_d_resnums else '#3498db')
    
    if x_vals:
        bars = ax.bar(x_vals, y_vals, color=c_vals, edgecolor='black', linewidth=0.5, width=1.5)
        for x, y, c in zip(x_vals, y_vals, c_vals):
            ax.annotate(f'{y:.1f}', (x, y), fontsize=7, ha='center', va='bottom')
    
    for rn in d_resnums_all:
        if rn in interface_d_resnums:
            ax.axvspan(rn-0.75, rn+0.75, alpha=0.1, color='red')
    
    ax.set_xlabel('Residue Number')
    ax.set_ylabel('Mean ΔΔG (kcal/mol)')
    ax.set_title('Chain D (Barstar) - Mutation Hotspot Map')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=2.0, color='orange', linestyle='--', linewidth=0.5, alpha=0.5, label='Hotspot threshold (2 kcal/mol)')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('report/images/fig8_hotspot_map.png')
    plt.close()
    print("Figure 8 saved: fig8_hotspot_map.png")

# ============================================================
# Figure 9: Electrostatic Complementarity
# ============================================================

def fig9_electrostatics():
    """Analyze electrostatic complementarity at the interface."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Charge distribution at interface
    ax = axes[0]
    
    # Count charges by chain at interface
    a_charges = defaultdict(int)
    d_charges = defaultdict(int)
    
    charge_map = {'ARG': +1, 'LYS': +1, 'ASP': -1, 'GLU': -1, 'HIS': 0.5}
    
    for r in interface_data['chain_a']:
        rn = r['resname']
        if rn in charge_map:
            a_charges[charge_map[rn]] += 1
    
    for r in interface_data['chain_d']:
        rn = r['resname']
        if rn in charge_map:
            d_charges[charge_map[rn]] += 1
    
    # Stacked bar chart
    charge_types = [-1, 0.5, 1]
    charge_labels = ['Negative', 'Partial +', 'Positive']
    charge_colors = ['#3498db', '#9b59b6', '#e74c3c']
    
    a_vals = [a_charges.get(c, 0) for c in charge_types]
    d_vals = [d_charges.get(c, 0) for c in charge_types]
    
    x = np.arange(2)
    width = 0.35
    
    bottom_a = np.zeros(1)
    bottom_d = np.zeros(1)
    
    for i, (ct, cl, cc) in enumerate(zip(charge_types, charge_labels, charge_colors)):
        ax.bar([0], [a_vals[i]], width, bottom=bottom_a[0], color=cc, label=f'{cl} (A)', edgecolor='black', linewidth=0.5)
        ax.bar([1], [d_vals[i]], width, bottom=bottom_d[0], color=cc, label=f'{cl} (D)', edgecolor='black', linewidth=0.5)
        bottom_a[0] += a_vals[i]
        bottom_d[0] += d_vals[i]
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Chain A\n(Barnase)', 'Chain D\n(Barstar)'])
    ax.set_ylabel('Number of Residues')
    ax.set_title('Charged Residues at Interface')
    ax.legend(fontsize=8)
    
    # Panel B: ddG for charged vs non-charged residue mutations
    ax = axes[1]
    
    charged_residues = set()
    for r in interface_data['chain_a'] + interface_data['chain_d']:
        if r['resname'] in charge_map:
            charged_residues.add(f"{r['resname'][0]}_{r['resnum']}" if False else 
                                f"{'A' if r in interface_data['chain_a'] else 'D'}_{r['resnum']}")
    
    ddG_charged = []
    ddG_noncharged = []
    
    for m in mutation_data:
        key = f"{m['chain']}_{m['resnum']}"
        if key in charged_residues:
            ddG_charged.append(m['ddG'])
        else:
            ddG_noncharged.append(m['ddG'])
    
    bp = ax.boxplot([ddG_charged, ddG_noncharged], 
                    labels=[f'Charged\n(n={len(ddG_charged)})', f'Non-charged\n(n={len(ddG_noncharged)})'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#3498db')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('ΔΔG (kcal/mol)')
    ax.set_title('ΔΔG: Charged vs Non-charged Residue Mutations')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    from scipy import stats
    if len(ddG_charged) > 0 and len(ddG_noncharged) > 0:
        t_stat, p_val = stats.mannwhitneyu(ddG_charged, ddG_noncharged, alternative='greater')
        ax.text(0.5, 0.95, f'Mann-Whitney p={p_val:.4f}', transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('report/images/fig9_electrostatics.png')
    plt.close()
    print("Figure 9 saved: fig9_electrostatics.png")

# ============================================================
# Figure 10: HADDOCK Workflow Schematic
# ============================================================

def fig10_workflow():
    """Create a schematic of the HADDOCK-inspired analysis workflow."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'HADDOCK-Inspired Analysis Workflow', fontsize=16, ha='center', fontweight='bold')
    
    # Define boxes
    boxes = [
        (1, 5.5, 3, 1.2, 'PDB Structure\n(1BRS_AD)', '#3498db'),
        (5, 5.5, 3, 1.2, 'Interface Detection\n(d < 5Å)', '#2ecc71'),
        (9, 5.5, 3, 1.2, 'AIR Computation\n(HADDOCK-style)', '#e74c3c'),
        (1, 3.5, 3, 1.2, 'SKEMPI 2.0\nMutations', '#9b59b6'),
        (5, 3.5, 3, 1.2, 'ΔΔG Calculation\n(RT·ln(Kd_mut/Kd_wt))', '#f39c12'),
        (9, 3.5, 3, 1.2, 'Per-residue\nEnergy Decomposition', '#1abc9c'),
        (5, 1.5, 3, 1.2, 'Structure-Function\nCorrelation', '#e67e22'),
        (9, 1.5, 3, 1.2, 'Hotspot\nIdentification', '#c0392b'),
    ]
    
    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=2)
    
    # Row 1 arrows
    ax.annotate('', xy=(5, 6.1), xytext=(4, 6.1), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 6.1), xytext=(8, 6.1), arrowprops=arrow_props)
    
    # Row 2 arrows
    ax.annotate('', xy=(5, 4.1), xytext=(4, 4.1), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 4.1), xytext=(8, 4.1), arrowprops=arrow_props)
    
    # Vertical arrows
    ax.annotate('', xy=(2.5, 4.7), xytext=(2.5, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 4.7), xytext=(6.5, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(10.5, 4.7), xytext=(10.5, 5.5), arrowprops=arrow_props)
    
    # Row 3 arrows
    ax.annotate('', xy=(5, 2.1), xytext=(4, 2.1), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 2.1), xytext=(8, 2.1), arrowprops=arrow_props)
    
    # Vertical to row 3
    ax.annotate('', xy=(6.5, 2.7), xytext=(6.5, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(10.5, 2.7), xytext=(10.5, 3.5), arrowprops=arrow_props)
    
    plt.tight_layout()
    plt.savefig('report/images/fig10_workflow.png')
    plt.close()
    print("Figure 10 saved: fig10_workflow.png")

# ============================================================
# Run all figure generation
# ============================================================

if __name__ == '__main__':
    print("Generating all figures...")
    fig1_interface_residue_map()
    fig2_ddg_distribution()
    fig3_per_residue_ddg()
    fig4_energy_vs_ddg()
    fig5_location_classification()
    fig6_contact_map()
    fig7_haddock_scoring()
    fig8_hotspot_map()
    fig9_electrostatics()
    fig10_workflow()
    print("\nAll figures generated successfully!")
