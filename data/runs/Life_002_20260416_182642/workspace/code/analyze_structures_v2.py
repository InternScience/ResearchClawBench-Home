#!/usr/bin/env python3
"""
Improved protein complex structural alignment analysis.
Implements TM-score calculation and structural alignment between PDB structures
using an iterative superposition approach similar to TM-align.
"""

import numpy as np
import json
import os
from collections import defaultdict

# Workspace paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Life_002_20260416_182642"
DATA_DIR = os.path.join(WORKSPACE, "data")
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
REPORT_IMAGES_DIR = os.path.join(WORKSPACE, "report/images")

def parse_pdb(pdb_path):
    """Parse PDB file and extract atom coordinates organized by chain."""
    chains = defaultdict(list)
    residues = defaultdict(list)
    header_info = {}
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HEADER'):
                header_info['header'] = line[10:66].strip()
            elif line.startswith('TITLE'):
                if 'title' not in header_info:
                    header_info['title'] = ''
                header_info['title'] += line[10:66].strip()
            elif line.startswith('COMPND'):
                if 'compnd' not in header_info:
                    header_info['compnd'] = []
                header_info['compnd'].append(line[10:66].strip())
            elif line.startswith('EXPDTA'):
                header_info['expdta'] = line[10:66].strip()
            elif line.startswith('REMARK   2') and 'RESOLUTION' in line:
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.replace('.', '').replace('-', '').isdigit():
                            header_info['resolution'] = float(p)
                            break
                except:
                    pass
            elif line.startswith('ATOM'):
                try:
                    atom_serial = int(line[6:11])
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21].strip() if line[21].strip() else 'A'
                    residue_seq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
                    b_factor = float(line[60:66]) if line[60:66].strip() else 0.0
                    element = line[76:78].strip()
                    
                    atom_info = {
                        'serial': atom_serial,
                        'name': atom_name,
                        'residue': residue_name,
                        'chain': chain_id,
                        'resseq': residue_seq,
                        'x': x, 'y': y, 'z': z,
                        'occupancy': occupancy,
                        'b_factor': b_factor,
                        'element': element
                    }
                    chains[chain_id].append(atom_info)
                    residues[(chain_id, residue_seq)].append(atom_info)
                except ValueError:
                    continue
    
    return chains, residues, header_info

def get_ca_coordinates(chains):
    """Extract C-alpha coordinates for each chain."""
    ca_coords = {}
    for chain_id, atoms in chains.items():
        ca_atoms = [a for a in atoms if a['name'] == 'CA']
        if ca_atoms:
            coords = np.array([[a['x'], a['y'], a['z']] for a in ca_atoms])
            ca_coords[chain_id] = {
                'coords': coords,
                'residues': [a['residue'] for a in ca_atoms],
                'resseqs': [a['resseq'] for a in ca_atoms]
            }
    return ca_coords

def kabsch_rotation(P, Q):
    """Calculate the rotation matrix that minimizes RMSD between P and Q using Kabsch algorithm."""
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    R = np.dot(V, Wt)
    return R

def tm_score_formula(distances, L_norm):
    """Calculate TM-score from distances."""
    d0 = 1.24 * np.cbrt(L_norm - 15) - 1.8
    if d0 < 0.5:
        d0 = 0.5
    return np.sum(1.0 / (1.0 + (distances ** 2) / (d0 ** 2))) / L_norm, d0

def iterative_tm_superposition(coords1, coords2, max_iter=50, d0_scale=1.24):
    """
    Perform iterative TM-score maximization superposition.
    Implements a simplified version of the TM-align algorithm.
    """
    L1, L2 = len(coords1), len(coords2)
    L_min = min(L1, L2)
    L_max = max(L1, L2)
    
    if L_min < 3:
        return 0.0, np.eye(3), np.zeros(3), coords2.copy()
    
    # Normalize d0 based on the shorter structure
    d0 = d0_scale * np.cbrt(L_min - 15) - 1.8
    if d0 < 0.5:
        d0 = 0.5
    
    # Center both structures
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    c1 = coords1 - center1
    c2 = coords2 - center2
    
    # Initialize with identity rotation
    best_tm = 0.0
    best_R = np.eye(3)
    best_t = np.zeros(3)
    
    # Use all residues initially
    current_c1 = c1[:L_min].copy()
    current_c2 = c2[:L_min].copy()
    
    # Gap penalty for dynamic programming-like selection
    gap_open = -0.6
    
    for iteration in range(max_iter):
        if len(current_c1) < 3 or len(current_c2) < 3:
            break
        
        # Kabsch rotation
        R = kabsch_rotation(current_c1, current_c2)
        
        # Rotate full c2
        c2_rot = np.dot(c2, R.T)
        
        # Calculate distances for the aligned region
        if L1 == L2:
            dists = np.sqrt(np.sum((c1 - c2_rot) ** 2, axis=1))
        else:
            dists = np.sqrt(np.sum((c1[:L_min] - c2_rot[:L_min]) ** 2, axis=1))
        
        # Calculate TM-score
        tm, _ = tm_score_formula(dists, L_min)
        
        if tm > best_tm:
            best_tm = tm
            best_R = R.copy()
        
        # Select residues for next iteration based on distance threshold
        # Use adaptive threshold
        threshold = 1.5 * d0 * np.sqrt((1.0 / tm) - 1.0) if tm > 0 else 5.0
        threshold = min(threshold, 5.0)
        threshold = max(threshold, 1.0)
        
        mask = dists < threshold
        n_aligned = np.sum(mask)
        
        if n_aligned < 3:
            break
        
        # Update working sets
        if L1 == L2:
            current_c1 = c1[mask].copy()
            current_c2 = c2[mask].copy()
        else:
            # For unequal lengths, map indices properly
            indices = np.where(mask)[0]
            current_c1 = c1[indices].copy()
            current_c2 = c2[indices].copy()
        
        # Check convergence
        if n_aligned == L_min and iteration > 5:
            break
    
    # Final superposition with best rotation
    c2_final = np.dot(c2, best_R.T) + center1
    translation = center1 - np.dot(center2, best_R.T)
    
    # Calculate final TM-score and RMSD
    if L1 == L2:
        final_dists = np.sqrt(np.sum((coords1 - c2_final) ** 2, axis=1))
    else:
        final_dists = np.sqrt(np.sum((coords1[:L_min] - c2_final[:L_min]) ** 2, axis=1))
    
    final_tm, _ = tm_score_formula(final_dists, L_min)
    final_rmsd = np.sqrt(np.mean(final_dists ** 2))
    
    return final_tm, best_R, translation, c2_final, final_rmsd

def structural_alignment(chain1_ca, chain2_ca):
    """Perform structural alignment between two chains."""
    coords1 = chain1_ca['coords']
    coords2 = chain2_ca['coords']
    
    L1, L2 = len(coords1), len(coords2)
    L_min = min(L1, L2)
    L_max = max(L1, L2)
    
    # Run iterative superposition
    tm_score, R, t, coords2_superposed, rmsd = iterative_tm_superposition(coords1, coords2)
    
    # Alignment coverage
    coverage = L_min / L_max
    
    return {
        'tm_score': float(tm_score),
        'rmsd': float(rmsd),
        'rotation_matrix': R.tolist(),
        'translation_vector': t.tolist(),
        'aligned_length': int(L_min),
        'coverage': float(coverage),
        'query_length': L1,
        'target_length': L2
    }

def main():
    print("=" * 60)
    print("Protein Complex Structural Alignment Analysis (v2)")
    print("=" * 60)
    
    # Load structures
    pdb7xg4_path = os.path.join(DATA_DIR, "7xg4.pdb")
    pdb6n40_path = os.path.join(DATA_DIR, "6n40.pdb")
    
    print(f"\nLoading structure: {pdb7xg4_path}")
    chains_7xg4, residues_7xg4, header_7xg4 = parse_pdb(pdb7xg4_path)
    ca_7xg4 = get_ca_coordinates(chains_7xg4)
    
    print(f"Loading structure: {pdb6n40_path}")
    chains_6n40, residues_6n40, header_6n40 = parse_pdb(pdb6n40_path)
    ca_6n40 = get_ca_coordinates(chains_6n40)
    
    # Data overview statistics
    print("\n" + "=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    
    stats_7xg4 = {
        'pdb_id': '7xg4',
        'chains': list(ca_7xg4.keys()),
        'num_chains': len(ca_7xg4),
        'total_ca_atoms': sum(len(c['coords']) for c in ca_7xg4.values()),
        'chain_lengths': {k: len(v['coords']) for k, v in ca_7xg4.items()},
        'header': header_7xg4.get('title', ''),
        'method': header_7xg4.get('expdta', ''),
        'resolution': header_7xg4.get('resolution', None)
    }
    
    stats_6n40 = {
        'pdb_id': '6n40',
        'chains': list(ca_6n40.keys()),
        'num_chains': len(ca_6n40),
        'total_ca_atoms': sum(len(c['coords']) for c in ca_6n40.values()),
        'chain_lengths': {k: len(v['coords']) for k, v in ca_6n40.items()},
        'header': header_6n40.get('title', ''),
        'method': header_6n40.get('expdta', ''),
        'resolution': header_6n40.get('resolution', None)
    }
    
    print(f"\n7xg4 (Query):")
    print(f"  Chains: {stats_7xg4['chains']}")
    print(f"  Number of chains: {stats_7xg4['num_chains']}")
    print(f"  Total CA atoms: {stats_7xg4['total_ca_atoms']}")
    print(f"  Chain lengths: {stats_7xg4['chain_lengths']}")
    print(f"  Method: {stats_7xg4['method']}")
    if stats_7xg4['resolution']:
        print(f"  Resolution: {stats_7xg4['resolution']} Angstroms")
    
    print(f"\n6n40 (Target):")
    print(f"  Chains: {stats_6n40['chains']}")
    print(f"  Number of chains: {stats_6n40['num_chains']}")
    print(f"  Total CA atoms: {stats_6n40['total_ca_atoms']}")
    print(f"  Chain lengths: {stats_6n40['chain_lengths']}")
    print(f"  Method: {stats_6n40['method']}")
    if stats_6n40['resolution']:
        print(f"  Resolution: {stats_6n40['resolution']} Angstroms")
    
    # Save chain composition
    chain_composition = {'7xg4': stats_7xg4, '6n40': stats_6n40}
    with open(os.path.join(OUTPUTS_DIR, 'chain_composition.json'), 'w') as f:
        json.dump(chain_composition, f, indent=2)
    print(f"\nSaved chain composition to: {OUTPUTS_DIR}/chain_composition.json")
    
    # Structural alignment
    print("\n" + "=" * 60)
    print("STRUCTURAL ALIGNMENT RESULTS")
    print("=" * 60)
    
    alignment_results = {}
    
    for chain_7xg4, data_7xg4 in ca_7xg4.items():
        for chain_6n40, data_6n40 in ca_6n40.items():
            key = f"{chain_7xg4}_vs_{chain_6n40}"
            print(f"\nAligning chain {chain_7xg4} (7xg4) to chain {chain_6n40} (6n40)...")
            
            result = structural_alignment(data_7xg4, data_6n40)
            alignment_results[key] = result
            
            print(f"  TM-score: {result['tm_score']:.4f}")
            print(f"  RMSD: {result['rmsd']:.4f} Angstroms")
            print(f"  Aligned length: {result['aligned_length']} residues")
            print(f"  Coverage: {result['coverage']:.2%}")
    
    # Find best alignment
    best_key = max(alignment_results.keys(), key=lambda k: alignment_results[k]['tm_score'])
    best_result = alignment_results[best_key]
    
    print(f"\n{'=' * 60}")
    print(f"BEST ALIGNMENT: {best_key}")
    print(f"{'=' * 60}")
    print(f"  TM-score: {best_result['tm_score']:.4f}")
    print(f"  RMSD: {best_result['rmsd']:.4f} Angstroms")
    print(f"  Aligned length: {best_result['aligned_length']} residues")
    print(f"  Query length: {best_result['query_length']} residues")
    print(f"  Target length: {best_result['target_length']} residues")
    print(f"  Coverage: {best_result['coverage']:.2%}")
    
    # Save alignment results
    with open(os.path.join(OUTPUTS_DIR, 'alignment_results.json'), 'w') as f:
        json.dump({
            'query': '7xg4',
            'target': '6n40',
            'all_alignments': alignment_results,
            'best_alignment': {'pair': best_key, 'result': best_result}
        }, f, indent=2)
    print(f"\nSaved alignment results to: {OUTPUTS_DIR}/alignment_results.json")
    
    # Chain correspondence
    chain_correspondence = {
        'query_pdb': '7xg4',
        'target_pdb': '6n40',
        'correspondences': []
    }
    
    for chain_6n40 in ca_6n40.keys():
        best_chain_7xg4 = None
        best_tm = -1
        for chain_7xg4 in ca_7xg4.keys():
            key = f"{chain_7xg4}_vs_{chain_6n40}"
            if alignment_results[key]['tm_score'] > best_tm:
                best_tm = alignment_results[key]['tm_score']
                best_chain_7xg4 = chain_7xg4
        
        chain_correspondence['correspondences'].append({
            'query_chain': best_chain_7xg4,
            'target_chain': chain_6n40,
            'tm_score': best_tm
        })
    
    with open(os.path.join(OUTPUTS_DIR, 'chain_correspondence.json'), 'w') as f:
        json.dump(chain_correspondence, f, indent=2)
    print(f"Saved chain correspondence to: {OUTPUTS_DIR}/chain_correspondence.json")
    
    # Save TM-score result
    tm_score_result = {
        'query': '7xg4',
        'target': '6n40',
        'best_tm_score': best_result['tm_score'],
        'best_pair': best_key,
        'all_tm_scores': {k: v['tm_score'] for k, v in alignment_results.items()}
    }
    with open(os.path.join(OUTPUTS_DIR, 'tm_score_result.json'), 'w') as f:
        json.dump(tm_score_result, f, indent=2)
    print(f"Saved TM-score result to: {OUTPUTS_DIR}/tm_score_result.json")
    
    # Save RMSD result
    rmsd_result = {
        'query': '7xg4',
        'target': '6n40',
        'best_rmsd': best_result['rmsd'],
        'best_pair': best_key,
        'all_rmsds': {k: v['rmsd'] for k, v in alignment_results.items()}
    }
    with open(os.path.join(OUTPUTS_DIR, 'rmsd_result.json'), 'w') as f:
        json.dump(rmsd_result, f, indent=2)
    print(f"Saved RMSD result to: {OUTPUTS_DIR}/rmsd_result.json")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)
    
    # Figure 1: Data overview
    fig, ax = plt.subplots(figsize=(10, 6))
    pdb_ids = ['7xg4', '6n40']
    chain_counts = [stats_7xg4['num_chains'], stats_6n40['num_chains']]
    total_residues = [stats_7xg4['total_ca_atoms'], stats_6n40['total_ca_atoms']]
    
    x = np.arange(len(pdb_ids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, chain_counts, width, label='Number of chains', color='#3498db')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, total_residues, width, label='Total residues (CA)', color='#e74c3c')
    
    ax.set_xlabel('PDB ID')
    ax.set_ylabel('Number of chains', color='#3498db')
    ax2.set_ylabel('Total residues (CA)', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(pdb_ids)
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.title('Data Overview: Chain and Residue Composition', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'data_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_IMAGES_DIR}/data_overview.png")
    
    # Figure 2: TM-score heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    chains_7xg4_list = sorted(ca_7xg4.keys())
    chains_6n40_list = sorted(ca_6n40.keys())
    
    tm_matrix = np.zeros((len(chains_7xg4_list), len(chains_6n40_list)))
    for i, c1 in enumerate(chains_7xg4_list):
        for j, c2 in enumerate(chains_6n40_list):
            key = f"{c1}_vs_{c2}"
            tm_matrix[i, j] = alignment_results[key]['tm_score']
    
    sns.heatmap(tm_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=chains_6n40_list, yticklabels=chains_7xg4_list,
                ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('6n40 chains')
    ax.set_ylabel('7xg4 chains')
    ax.set_title('TM-score Matrix: 7xg4 vs 6n40 Chain Alignments', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'tm_score_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_IMAGES_DIR}/tm_score_heatmap.png")
    
    # Figure 3: RMSD comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    pairs = list(alignment_results.keys())
    rmsds = [alignment_results[p]['rmsd'] for p in pairs]
    tms = [alignment_results[p]['tm_score'] for p in pairs]
    
    x = np.arange(len(pairs))
    bars = ax.bar(x, rmsds, color='#9b59b6')
    ax.set_xlabel('Chain pair alignment')
    ax.set_ylabel('RMSD (Angstroms)')
    ax.set_title('RMSD for All Chain Pair Alignments', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    
    for bar, tm in zip(bars, tms):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}\n(TM={tm:.3f})', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'rmsd_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_IMAGES_DIR}/rmsd_comparison.png")
    
    # Figure 4: Structural superposition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    best_chain_7xg4 = best_key.split('_vs_')[0]
    best_chain_6n40 = best_key.split('_vs_')[1]
    
    coords_7xg4 = ca_7xg4[best_chain_7xg4]['coords']
    coords_6n40_orig = ca_6n40[best_chain_6n40]['coords']
    
    _, R, t, coords_6n40_superposed, _ = iterative_tm_superposition(coords_7xg4, coords_6n40_orig)
    
    L_viz = min(len(coords_7xg4), len(coords_6n40_orig))
    
    ax_xy = axes[0]
    ax_xy.scatter(coords_7xg4[:L_viz, 0], coords_7xg4[:L_viz, 1], 
                  c='blue', s=10, alpha=0.6, label='7xg4 (query)')
    ax_xy.scatter(coords_6n40_superposed[:L_viz, 0], coords_6n40_superposed[:L_viz, 1],
                  c='red', s=10, alpha=0.6, label='6n40 (superposed)')
    ax_xy.set_xlabel('X (Angstroms)')
    ax_xy.set_ylabel('Y (Angstroms)')
    ax_xy.set_title(f'Structural Superposition (XY)\nTM={best_result["tm_score"]:.3f}, RMSD={best_result["rmsd"]:.2f}A', fontsize=11)
    ax_xy.legend()
    ax_xy.set_aspect('equal')
    
    ax_xz = axes[1]
    ax_xz.scatter(coords_7xg4[:L_viz, 0], coords_7xg4[:L_viz, 2],
                  c='blue', s=10, alpha=0.6, label='7xg4 (query)')
    ax_xz.scatter(coords_6n40_superposed[:L_viz, 0], coords_6n40_superposed[:L_viz, 2],
                  c='red', s=10, alpha=0.6, label='6n40 (superposed)')
    ax_xz.set_xlabel('X (Angstroms)')
    ax_xz.set_ylabel('Z (Angstroms)')
    ax_xz.set_title('Structural Superposition (XZ)', fontsize=11)
    ax_xz.legend()
    ax_xz.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'alignment_superposition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_IMAGES_DIR}/alignment_superposition.png")
    
    # Figure 5: TM-score distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    tm_values = [r['tm_score'] for r in alignment_results.values()]
    
    ax.hist(tm_values, bins=10, edgecolor='black', color='#2ecc71', alpha=0.7)
    ax.axvline(best_result['tm_score'], color='red', linestyle='--', linewidth=2, 
               label=f'Best: {best_result["tm_score"]:.3f}')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=2, label='TM=0.5 (same fold)')
    ax.axvline(0.3, color='orange', linestyle=':', linewidth=2, label='TM=0.3 (remote homolog)')
    
    ax.set_xlabel('TM-score')
    ax.set_ylabel('Frequency')
    ax.set_title('TM-score Distribution for Chain Pair Alignments', fontsize=12, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'tm_score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {REPORT_IMAGES_DIR}/tm_score_distribution.png")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {OUTPUTS_DIR}")
    print(f"Figures saved to: {REPORT_IMAGES_DIR}")
    
    return alignment_results, best_result

if __name__ == "__main__":
    main()
