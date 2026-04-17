#!/usr/bin/env python3
"""
HADDOCK3 Analysis of Barnase-Barstar Complex (1BRS)
Using SKEMPI 2.0 database for validation

This script:
1. Parses the 1BRS PDB structure
2. Identifies interface residues between barnase and barstar
3. Extracts mutation data from SKEMPI 2.0
4. Calculates binding affinity changes (ddG)
5. Generates visualizations
"""

import csv
import json
import os
from collections import defaultdict
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import protein_letters_3to1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figures
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

def parse_pdb_structure(pdb_path):
    """Parse PDB file and extract chain information."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('1BRS', pdb_path)
    
    chains = defaultdict(list)
    residues_info = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if residue.id[0] == ' ':  # Only standard amino acids
                    try:
                        res_name = protein_letters_3to1.get(residue.resname.strip(), 'X')
                        res_num = residue.id[1]
                        chains[chain_id].append((res_num, res_name))
                        
                        # Get CA atom coordinates
                        if 'CA' in residue:
                            ca_coord = residue['CA'].get_coord()
                            residues_info.append({
                                'chain': chain_id,
                                'resnum': res_num,
                                'resname': res_name,
                                'ca_coord': ca_coord.tolist()
                            })
                    except KeyError:
                        pass  # Skip non-standard residues
    
    return dict(chains), residues_info, structure


def find_interface_residues(structure, cutoff=5.0):
    """Identify interface residues between chains A and D."""
    interface_residues = {'A': set(), 'D': set()}
    
    atoms_A = []
    atoms_D = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Standard residues only
                    for atom in residue:
                        if chain.id == 'A':
                            atoms_A.append(atom)
                        elif chain.id == 'D':
                            atoms_D.append(atom)
    
    # Find atoms within cutoff distance
    for atom_A in atoms_A:
        for atom_D in atoms_D:
            dist = atom_A - atom_D
            if dist < cutoff:
                interface_residues['A'].add((atom_A.get_parent().id[1], atom_A.get_parent().resname))
                interface_residues['D'].add((atom_D.get_parent().id[1], atom_D.get_parent().resname))
    
    return interface_residues


def extract_skempi_data(skempi_path, target_pdb='1BRS'):
    """Extract mutation data from SKEMPI 2.0 for target PDB."""
    mutations = []
    
    with open(skempi_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            pdb_id = row['#Pdb'].upper()
            if target_pdb.upper() in pdb_id:
                # Calculate ddG = -RT * ln(Kd_mut / Kd_wt)
                # At 298K, RT ≈ 0.592 kcal/mol
                # ddG = -RT * ln(Kd_mut/Kd_wt) = RT * ln(Kd_wt/Kd_mut)
                try:
                    kd_mut = float(row['Affinity_mut_parsed'])
                    kd_wt = float(row['Affinity_wt_parsed'])
                    
                    # ddG in kcal/mol: positive means destabilizing, negative means stabilizing
                    R = 0.001987  # kcal/(mol*K)
                    T = 298  # Kelvin (standard temperature)
                    
                    if kd_mut > 0 and kd_wt > 0:
                        ddg = R * T * np.log(kd_mut / kd_wt)
                    else:
                        ddg = None
                except (ValueError, TypeError):
                    ddg = None
                
                mutations.append({
                    'pdb': row['#Pdb'],
                    'mutation_pdb': row['Mutation(s)_PDB'],
                    'mutation_cleaned': row['Mutation(s)_cleaned'],
                    'location': row['iMutation_Location(s)'],
                    'kd_mut': row['Affinity_mut_parsed'],
                    'kd_wt': row['Affinity_wt_parsed'],
                    'ddg': ddg,
                    'protein1': row['Protein 1'],
                    'protein2': row['Protein 2'],
                    'method': row['Method']
                })
    
    return mutations


def map_mutations_to_structure(mutations, interface_residues):
    """Map mutations to interface or non-interface locations."""
    mapped_mutations = []
    
    for mut in mutations:
        mutation_str = mut['mutation_cleaned']
        # Parse mutation string like "LI38G" -> residue L at position 38 mutated to G
        is_interface = False
        
        # Check if mutation position matches interface residues
        for res_num, res_name in interface_residues['A']:
            if str(res_num) in mutation_str:
                is_interface = True
                break
        
        for res_num, res_name in interface_residues['D']:
            if str(res_num) in mutation_str:
                is_interface = True
                break
        
        mapped_mutations.append({
            **mut,
            'is_interface': is_interface,
            'location_type': 'interface' if is_interface else 'non-interface'
        })
    
    return mapped_mutations


def save_intermediate_outputs(chains, residues_info, interface_residues, 
                              mutations, mapped_mutations, output_dir):
    """Save intermediate results to JSON/CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Structure statistics
    structure_stats = {
        'pdb_id': '1BRS',
        'chains': list(chains.keys()),
        'chain_lengths': {k: len(v) for k, v in chains.items()},
        'total_residues': sum(len(v) for v in chains.values()),
        'interface_residues': {
            'chain_A': len(interface_residues['A']),
            'chain_D': len(interface_residues['D']),
            'total': len(interface_residues['A']) + len(interface_residues['D'])
        }
    }
    
    with open(os.path.join(output_dir, 'structure_stats.json'), 'w') as f:
        json.dump(structure_stats, f, indent=2)
    
    # Interface residues
    interface_data = {
        'chain_A': [{'resnum': r[0], 'resname': r[1]} for r in interface_residues['A']],
        'chain_D': [{'resnum': r[0], 'resname': r[1]} for r in interface_residues['D']]
    }
    
    with open(os.path.join(output_dir, 'interface_residues.json'), 'w') as f:
        json.dump(interface_data, f, indent=2)
    
    # SKEMPI data
    with open(os.path.join(output_dir, 'skempi_1brs_data.csv'), 'w', newline='') as f:
        if mapped_mutations:
            writer = csv.DictWriter(f, fieldnames=mapped_mutations[0].keys())
            writer.writeheader()
            for mut in mapped_mutations:
                row = {k: (v if not isinstance(v, (list, dict)) else str(v)) for k, v in mut.items()}
                writer.writerow(row)
    
    # ddG distribution statistics
    ddg_values = [m['ddg'] for m in mapped_mutations if m['ddg'] is not None]
    ddg_stats = {
        'count': len(ddg_values),
        'mean': float(np.mean(ddg_values)) if ddg_values else None,
        'std': float(np.std(ddg_values)) if ddg_values else None,
        'min': float(np.min(ddg_values)) if ddg_values else None,
        'max': float(np.max(ddg_values)) if ddg_values else None,
        'interface_count': len([m for m in mapped_mutations if m['is_interface'] and m['ddg']]),
        'non_interface_count': len([m for m in mapped_mutations if not m['is_interface'] and m['ddg']])
    }
    
    with open(os.path.join(output_dir, 'ddg_distribution.json'), 'w') as f:
        json.dump(ddg_stats, f, indent=2)
    
    return structure_stats, ddg_stats


def create_figures(mapped_mutations, interface_residues, ddg_stats, images_dir):
    """Create publication-quality figures."""
    os.makedirs(images_dir, exist_ok=True)
    
    ddg_values = [m['ddg'] for m in mapped_mutations if m['ddg'] is not None]
    interface_ddg = [m['ddg'] for m in mapped_mutations if m['is_interface'] and m['ddg'] is not None]
    non_interface_ddg = [m['ddg'] for m in mapped_mutations if not m['is_interface'] and m['ddg'] is not None]
    
    # Figure 1: Data Overview - Mutation distribution by location
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    location_counts = defaultdict(int)
    for m in mapped_mutations:
        loc_type = 'Interface' if m['is_interface'] else 'Non-interface'
        location_counts[loc_type] += 1
    
    colors = ['#e74c3c', '#3498db']
    bars = ax1.bar(location_counts.keys(), location_counts.values(), color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Location Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Mutations', fontsize=12, fontweight='bold')
    ax1.set_title('1BRS Mutation Distribution in SKEMPI 2.0', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars, location_counts.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'fig1_data_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: ddG Distribution Histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Overall distribution
    n, bins, patches = ax2.hist(ddg_values, bins=20, color='#2ecc71', edgecolor='black', 
                                 alpha=0.7, linewidth=1.2, label=f'All mutations (n={len(ddg_values)})')
    
    # Add vertical lines for mean and zero
    mean_ddg = np.mean(ddg_values)
    ax2.axvline(mean_ddg, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean ddG = {mean_ddg:.2f} kcal/mol')
    ax2.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, label='No effect (ddG=0)')
    
    ax2.set_xlabel('ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Binding Affinity Change Distribution for 1BRS Mutations', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'fig2_ddg_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Interface vs Non-interface ddG comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    data_to_plot = [interface_ddg, non_interface_ddg]
    labels = [f'Interface\n(n={len(interface_ddg)})', f'Non-interface\n(n={len(non_interface_ddg)})']
    colors_box = ['#e74c3c', '#3498db']
    
    bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                     widths=0.6, whis=1.5, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.7)
    
    # Customize boxplot appearance
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    plt.setp(bp['medians'], color='black', linewidth=2.5)
    
    ax3.set_ylabel('ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
    ax3.set_title('Comparison of Binding Affinity Changes by Location', fontsize=14, fontweight='bold')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'fig3_interface_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Hot-spot analysis (mutations with large |ddG|)
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # Define hot-spots as |ddG| > 2 kcal/mol
    hotspot_threshold = 2.0
    hotspots = [(m['mutation_cleaned'], m['ddg'], m['is_interface']) 
                for m in mapped_mutations 
                if m['ddg'] is not None and abs(m['ddg']) > hotspot_threshold]
    
    # Sort by absolute ddG value
    hotspots.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Take top 15 hotspots
    top_hotspots = hotspots[:15]
    
    mutations_names = [h[0] for h in top_hotspots]
    ddg_vals = [h[1] for h in top_hotspots]
    is_interface = [h[2] for h in top_hotspots]
    
    # Color based on interface status
    bar_colors = ['#e74c3c' if iface else '#3498db' for iface in is_interface]
    
    y_pos = np.arange(len(mutations_names))
    bars = ax4.barh(y_pos, ddg_vals, color=bar_colors, edgecolor='black', linewidth=1.2)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(mutations_names, fontsize=10)
    ax4.set_xlabel('ΔΔG (kcal/mol)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Top 15 Hot-spot Mutations (|ΔΔG| > {hotspot_threshold} kcal/mol)', fontsize=14, fontweight='bold')
    ax4.axvline(0, color='black', linestyle='-', linewidth=1.5)
    ax4.axvline(hotspot_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(-hotspot_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', edgecolor='black', label='Interface mutation'),
                       Patch(facecolor='#3498db', edgecolor='black', label='Non-interface mutation')]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax4.invert_yaxis()  # Highest values at top
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'fig4_hotspot_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Interface residue mapping (schematic representation)
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    
    # Create a schematic representation of interface residues
    interface_A_list = sorted(list(interface_residues['A']), key=lambda x: x[0])
    interface_D_list = sorted(list(interface_residues['D']), key=lambda x: x[0])
    
    # Plot chain A residues
    ax5.scatter([r[0] for r in interface_A_list], [1]*len(interface_A_list), 
                c='#e74c3c', s=100, edgecolors='black', linewidth=1.5, label='Chain A (Barnase) interface')
    
    # Plot chain D residues
    ax5.scatter([r[0] for r in interface_D_list], [2]*len(interface_D_list), 
                c='#3498db', s=100, edgecolors='black', linewidth=1.5, label='Chain D (Barstar) interface')
    
    ax5.set_xlabel('Residue Number', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Chain', fontsize=12, fontweight='bold')
    ax5.set_yticks([1, 2])
    ax5.set_yticklabels(['Chain A (Barnase)', 'Chain D (Barstar)'])
    ax5.set_title('Interface Residue Mapping on 1BRS Structure', fontsize=14, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='x')
    
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'fig5_interface_mapping.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to {images_dir}")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("HADDOCK3 Analysis: Barnase-Barstar Complex (1BRS)")
    print("=" * 60)
    
    # File paths
    pdb_path = 'data/1brs_AD.pdb'
    skempi_path = 'data/skempi_v2.csv'
    output_dir = 'outputs'
    images_dir = 'report/images'
    
    # Step 1: Parse PDB structure
    print("\n[1/5] Parsing PDB structure...")
    chains, residues_info, structure = parse_pdb_structure(pdb_path)
    print(f"  - Found chains: {list(chains.keys())}")
    print(f"  - Chain lengths: {[(k, len(v)) for k, v in chains.items()]}")
    
    # Step 2: Identify interface residues
    print("\n[2/5] Identifying interface residues (cutoff: 5.0 Å)...")
    interface_residues = find_interface_residues(structure, cutoff=5.0)
    print(f"  - Chain A interface residues: {len(interface_residues['A'])}")
    print(f"  - Chain D interface residues: {len(interface_residues['D'])}")
    print(f"  - Interface residues (Chain A): {sorted([f'{r[1]}{r[0]}' for r in interface_residues['A']])}")
    print(f"  - Interface residues (Chain D): {sorted([f'{r[1]}{r[0]}' for r in interface_residues['D']])}")
    
    # Step 3: Extract SKEMPI data
    print("\n[3/5] Extracting SKEMPI 2.0 data for 1BRS...")
    mutations = extract_skempi_data(skempi_path, target_pdb='1BRS')
    print(f"  - Found {len(mutations)} mutations in SKEMPI 2.0")
    
    # Step 4: Map mutations to structure
    print("\n[4/5] Mapping mutations to interface/non-interface locations...")
    mapped_mutations = map_mutations_to_structure(mutations, interface_residues)
    interface_count = sum(1 for m in mapped_mutations if m['is_interface'])
    print(f"  - Interface mutations: {interface_count}")
    print(f"  - Non-interface mutations: {len(mapped_mutations) - interface_count}")
    
    # Step 5: Save intermediate outputs
    print("\n[5/5] Saving intermediate outputs and generating figures...")
    structure_stats, ddg_stats = save_intermediate_outputs(
        chains, residues_info, interface_residues, 
        mutations, mapped_mutations, output_dir
    )
    
    print(f"\n  Structure Statistics:")
    print(f"    - Total residues: {structure_stats['total_residues']}")
    print(f"    - Interface residues: {structure_stats['interface_residues']['total']}")
    
    print(f"\n  ddG Statistics:")
    print(f"    - Valid ddG values: {ddg_stats['count']}")
    print(f"    - Mean ddG: {ddg_stats['mean']:.3f} kcal/mol")
    print(f"    - Std ddG: {ddg_stats['std']:.3f} kcal/mol")
    print(f"    - Range: [{ddg_stats['min']:.3f}, {ddg_stats['max']:.3f}] kcal/mol")
    
    # Generate figures
    create_figures(mapped_mutations, interface_residues, ddg_stats, images_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"Figures saved to: {images_dir}/")


if __name__ == '__main__':
    main()
