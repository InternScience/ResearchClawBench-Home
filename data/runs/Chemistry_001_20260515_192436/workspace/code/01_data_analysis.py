#!/usr/bin/env python3
"""
Unified Deep Learning Framework for Biomolecular Complex Structure Prediction
Phase 1: Data Analysis and Structure Visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import json

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

def parse_pdb_file(filepath):
    """Parse PDB file and extract atom information"""
    atoms = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom = {
                    'serial': int(line[6:11].strip()),
                    'name': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21],
                    'resseq': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'element': line[76:78].strip()
                }
                atoms.append(atom)
    return atoms

def parse_sdf_file(filepath):
    """Parse SDF file and extract atom information"""
    atoms = []
    bonds = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the counts block (line 4)
    if len(lines) > 3:
        counts_line = lines[3]
        n_atoms = int(counts_line[0:3].strip())
        n_bonds = int(counts_line[3:6].strip())
        
        # Parse atoms (lines 4 to 4+n_atoms)
        for i in range(4, 4 + n_atoms):
            if i < len(lines):
                line = lines[i]
                atom = {
                    'x': float(line[0:10].strip()),
                    'y': float(line[10:20].strip()),
                    'z': float(line[20:30].strip()),
                    'element': line[31:34].strip()
                }
                atoms.append(atom)
        
        # Parse bonds (lines 4+n_atoms to 4+n_atoms+n_bonds)
        for i in range(4 + n_atoms, 4 + n_atoms + n_bonds):
            if i < len(lines):
                line = lines[i]
                bond = {
                    'atom1': int(line[0:3].strip()),
                    'atom2': int(line[3:6].strip()),
                    'type': int(line[6:9].strip())
                }
                bonds.append(bond)
    
    return atoms, bonds

def extract_backbone_atoms(atoms):
    """Extract CA atoms for protein backbone visualization"""
    return [a for a in atoms if a['name'] == 'CA']

def compute_distance_matrix(coords):
    """Compute pairwise distance matrix"""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = np.array(coords[i]) - np.array(coords[j])
                dist_matrix[i, j] = np.sqrt(np.sum(diff**2))
    return dist_matrix

def visualize_protein_structure(atoms, save_path):
    """Visualize protein 3D structure"""
    # Extract CA atoms for backbone
    ca_atoms = extract_backbone_atoms(atoms)
    coords = np.array([[a['x'], a['y'], a['z']] for a in ca_atoms])
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot backbone
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', linewidth=2, alpha=0.8)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='red', s=50, alpha=0.8)
    
    # Highlight start and end
    ax.scatter([coords[0, 0]], [coords[0, 1]], [coords[0, 2]], c='green', s=200, marker='*', label='N-terminus')
    ax.scatter([coords[-1, 0]], [coords[-1, 1]], [coords[-1, 2]], c='orange', s=200, marker='*', label='C-terminus')
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('FKBP12 Protein Structure (CA atoms)\nPDB ID: 2L3R')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return coords

def visualize_ligand_structure(atoms, bonds, save_path):
    """Visualize ligand 3D structure"""
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    elements = [a['element'] for a in atoms]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by element type
    element_colors = {
        'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'yellow',
        'H': 'white', 'F': 'green', 'Cl': 'green', 'Br': 'brown'
    }
    
    colors = [element_colors.get(e, 'gray') for e in elements]
    
    # Plot atoms
    for i, (coord, color) in enumerate(zip(coords, colors)):
        ax.scatter(coord[0], coord[1], coord[2], c=color, s=100, alpha=0.8)
    
    # Plot bonds
    for bond in bonds:
        if bond['atom1'] <= len(coords) and bond['atom2'] <= len(coords):
            start = coords[bond['atom1'] - 1]
            end = coords[bond['atom2'] - 1]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   'k-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('FK506 Ligand Structure\n(Experimental Conformation)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return coords

def analyze_protein_ligand_interaction(protein_atoms, ligand_atoms, save_path):
    """Analyze protein-ligand interaction distances"""
    # Get protein CA atoms
    ca_atoms = extract_backbone_atoms(protein_atoms)
    ca_coords = np.array([[a['x'], a['y'], a['z']] for a in ca_atoms])
    
    # Get ligand coordinates (non-H atoms)
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in ligand_atoms if a['element'] != 'H'])
    
    # Compute minimum distances from ligand to protein residues
    min_distances = []
    for lc in lig_coords:
        distances = np.sqrt(np.sum((ca_coords - lc)**2, axis=1))
        min_distances.append(np.min(distances))
    
    # Create distance heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distance distribution
    axes[0].hist(min_distances, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Minimum Distance (Å)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Ligand-Residue Distances')
    axes[0].axvline(x=np.mean(min_distances), color='red', linestyle='--', label=f'Mean: {np.mean(min_distances):.2f} Å')
    axes[0].legend()
    
    # Residue distance profile
    lig_com = np.mean(lig_coords, axis=0)
    residue_distances = np.sqrt(np.sum((ca_coords - lig_com)**2, axis=1))
    
    axes[1].plot(range(1, len(residue_distances)+1), residue_distances, 'o-', markersize=4)
    axes[1].set_xlabel('Residue Index')
    axes[1].set_ylabel('Distance to Ligand COM (Å)')
    axes[1].set_title('Residue Distance to Ligand Center of Mass')
    axes[1].axhline(y=5.0, color='red', linestyle='--', label='Binding threshold (5 Å)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return min_distances, residue_distances

def compute_protein_metrics(atoms, save_path):
    """Compute and visualize protein structural metrics"""
    ca_atoms = extract_backbone_atoms(atoms)
    coords = np.array([[a['x'], a['y'], a['z']] for a in ca_atoms])
    
    # Compute distance matrix
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
    
    # Compute secondary structure indicators (simplified)
    # Ramachandran-like analysis using C-alpha geometry
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Distance matrix heatmap
    im = axes[0].imshow(dist_matrix, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Residue Index')
    axes[0].set_ylabel('Residue Index')
    axes[0].set_title('Cα Distance Matrix')
    plt.colorbar(im, ax=axes[0], label='Distance (Å)')
    
    # Radius of gyration profile
    rg_values = []
    for i in range(3, n):
        com = np.mean(coords[:i+1], axis=0)
        rg = np.sqrt(np.mean(np.sum((coords[:i+1] - com)**2, axis=1)))
        rg_values.append(rg)
    
    axes[1].plot(range(3, n), rg_values, 'b-', linewidth=2)
    axes[1].set_xlabel('Residue Index (N_terminus)')
    axes[1].set_ylabel('Radius of Gyration (Å)')
    axes[1].set_title('Cumulative Radius of Gyration')
    
    # End-to-end distance
    e2e_distances = []
    for i in range(n):
        e2e = np.sqrt(np.sum((coords[i] - coords[0])**2))
        e2e_distances.append(e2e)
    
    axes[2].plot(range(n), e2e_distances, 'r-', linewidth=2)
    axes[2].set_xlabel('Residue Index')
    axes[2].set_ylabel('Distance to N-terminus (Å)')
    axes[2].set_title('End-to-End Distance Profile')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return dist_matrix, rg_values

def main():
    print("=" * 60)
    print("Biomolecular Complex Structure Prediction - Data Analysis")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading protein structure...")
    protein_atoms = parse_pdb_file('data/sample/2l3r/2l3r_protein.pdb')
    print(f"   Loaded {len(protein_atoms)} atoms")
    
    print("\n2. Loading ligand structure...")
    ligand_atoms, ligand_bonds = parse_sdf_file('data/sample/2l3r/2l3r_ligand.sdf')
    print(f"   Loaded {len(ligand_atoms)} atoms, {len(ligand_bonds)} bonds")
    
    # Analyze protein
    print("\n3. Analyzing protein structure...")
    ca_atoms = extract_backbone_atoms(protein_atoms)
    print(f"   Found {len(ca_atoms)} CA atoms")
    print(f"   Residue range: {ca_atoms[0]['resseq']} - {ca_atoms[-1]['resseq']}")
    
    # Visualizations
    print("\n4. Generating visualizations...")
    
    protein_coords = visualize_protein_structure(protein_atoms, 'report/images/protein_structure.png')
    print("   Saved protein structure visualization")
    
    ligand_coords = visualize_ligand_structure(ligand_atoms, ligand_bonds, 'report/images/ligand_structure.png')
    print("   Saved ligand structure visualization")
    
    min_dists, residue_dists = analyze_protein_ligand_interaction(protein_atoms, ligand_atoms, 
                                                                  'report/images/interaction_analysis.png')
    print("   Saved interaction analysis")
    
    dist_matrix, rg_values = compute_protein_metrics(protein_atoms, 'report/images/protein_metrics.png')
    print("   Saved protein metrics")
    
    # Save analysis results
    results = {
        'protein': {
            'total_atoms': len(protein_atoms),
            'ca_atoms': len(ca_atoms),
            'residue_range': [ca_atoms[0]['resseq'], ca_atoms[-1]['resseq']],
            'sequence_length': ca_atoms[-1]['resseq'] - ca_atoms[0]['resseq'] + 1,
            'mean_rg': float(np.mean(rg_values))
        },
        'ligand': {
            'total_atoms': len(ligand_atoms),
            'non_h_atoms': len([a for a in ligand_atoms if a['element'] != 'H']),
            'total_bonds': len(ligand_bonds)
        },
        'interaction': {
            'mean_min_distance': float(np.mean(min_dists)),
            'min_distance_std': float(np.std(min_dists)),
            'binding_residues': int(np.sum(np.array(residue_dists) < 5.0))
        }
    }
    
    with open('outputs/analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n5. Analysis complete!")
    print(f"   Results saved to outputs/analysis_results.json")
    print(f"   Figures saved to report/images/")
    
    return results

if __name__ == '__main__':
    results = main()
