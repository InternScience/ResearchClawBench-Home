#!/usr/bin/env python3
"""
Structural Alignment Analysis of Protein Complexes
====================================================
Analyzes structural alignment between 7xg4 (type IV-A CRISPR-Cas complex)
and 6n40 (MmpL3 membrane protein) using US-align and TM-align.

Produces:
- Chain composition analysis
- Pairwise chain alignments
- Multimer alignment results
- Superimposition vectors
- TM scores
- Visualization plots
"""

import os
import subprocess
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_002_20260518_034243')
DATA_DIR = WORKSPACE / 'data'
OUTPUTS_DIR = WORKSPACE / 'outputs'
CODE_DIR = WORKSPACE / 'code'
REPORT_DIR = WORKSPACE / 'report'
IMAGES_DIR = REPORT_DIR / 'images'
TOOLS_DIR = WORKSPACE / 'tools' / 'USalign-master'

# Ensure directories exist
OUTPUTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


def parse_pdb_chains(pdb_path):
    """Parse PDB file and extract chain information."""
    chains = defaultdict(lambda: {'residues': [], 'atoms': 0, 'type': 'protein'})
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                chain_id = line[21].strip()
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                atom_name = line[12:16].strip()
                
                # Determine if RNA/DNA
                rna_dna = {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}
                if res_name in rna_dna:
                    chains[chain_id]['type'] = 'nucleic_acid'
                
                if res_num not in chains[chain_id]['residues']:
                    chains[chain_id]['residues'].append(res_num)
                chains[chain_id]['atoms'] += 1
    
    return dict(chains)


def run_alignment(tool, pdb1, pdb2, options, output_file=None):
    """Run structural alignment tool."""
    cmd = [str(tool), str(pdb1), str(pdb2)] + options
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            f.write('\n--- STDERR ---\n')
            f.write(result.stderr)
    
    return result.stdout, result.stderr


def parse_alignment_output(output):
    """Parse alignment output to extract key metrics."""
    metrics = {}
    
    for line in output.split('\n'):
        if 'TM-score=' in line and 'normalized by' in line:
            parts = line.split('=')
            score = float(parts[1].split('(')[0].strip())
            if 'Structure_1' in line:
                metrics['tm_score_struct1'] = score
            elif 'Structure_2' in line:
                metrics['tm_score_struct2'] = score
        elif 'Aligned length=' in line:
            parts = line.split(',')
            metrics['aligned_length'] = int(parts[0].split('=')[1].strip())
            metrics['rmsd'] = float(parts[1].split('=')[1].strip())
        elif 'Length of Structure_1' in line:
            metrics['length_struct1'] = int(line.split(':')[1].strip().split()[0])
        elif 'Length of Structure_2' in line:
            metrics['length_struct2'] = int(line.split(':')[1].strip().split()[0])
        elif 'Seq_ID=' in line:
            seq_id_str = line.split('Seq_ID=')[1].split('=')[0].strip()
            metrics['seq_id'] = seq_id_str
    
    return metrics


def extract_alignment_details(alignment_line):
    """Extract alignment details from FASTA-like alignment output."""
    details = []
    for char in alignment_line:
        if char == ':':
            details.append('close')  # d < 5.0 Angstrom
        elif char == '.':
            details.append('aligned')  # other aligned residues
        elif char == '*':
            details.append('gap')  # gap
        else:
            details.append(None)  # unaligned
    
    return details


def main():
    print("=" * 70)
    print("Structural Alignment Analysis of Protein Complexes")
    print("=" * 70)
    
    # =====================
    # 1. Parse PDB files
    # =====================
    print("\n[1] Parsing PDB files...")
    
    chains_7xg4 = parse_pdb_chains(DATA_DIR / '7xg4.pdb')
    chains_6n40 = parse_pdb_chains(DATA_DIR / '6n40.pdb')
    
    print("\n  7xg4.pdb (Type IV-A CRISPR-Cas complex):")
    total_res_7xg4 = 0
    total_atoms_7xg4 = 0
    for chain_id in sorted(chains_7xg4.keys()):
        info = chains_7xg4[chain_id]
        n_res = len(info['residues'])
        total_res_7xg4 += n_res
        total_atoms_7xg4 += info['atoms']
        print(f"    Chain {chain_id}: {n_res} residues, {info['atoms']} atoms ({info['type']})")
    print(f"    Total: {total_res_7xg4} residues, {total_atoms_7xg4} atoms, {len(chains_7xg4)} chains")
    
    print("\n  6n40.pdb (MmpL3 membrane protein):")
    total_res_6n40 = 0
    total_atoms_6n40 = 0
    for chain_id in sorted(chains_6n40.keys()):
        info = chains_6n40[chain_id]
        n_res = len(info['residues'])
        total_res_6n40 += n_res
        total_atoms_6n40 += info['atoms']
        print(f"    Chain {chain_id}: {n_res} residues, {info['atoms']} atoms ({info['type']})")
    print(f"    Total: {total_res_6n40} residues, {total_atoms_6n40} atoms, {len(chains_6n40)} chains")
    
    # Save chain composition
    chain_info = {
        '7xg4': {k: {'n_residues': len(v['residues']), 'n_atoms': v['atoms'], 'type': v['type']} 
                  for k, v in chains_7xg4.items()},
        '6n40': {k: {'n_residues': len(v['residues']), 'n_atoms': v['atoms'], 'type': v['type']} 
                  for k, v in chains_6n40.items()}
    }
    
    with open(OUTPUTS_DIR / 'chain_composition.json', 'w') as f:
        json.dump(chain_info, f, indent=2)
    
    # =====================
    # 2. Run multimer alignment (US-align -mm 1)
    # =====================
    print("\n[2] Running multimer alignment (US-align -mm 1)...")
    
    usalign = TOOLS_DIR / 'USalign'
    tmalign = TOOLS_DIR / 'TMalign'
    
    # Full multimer alignment
    stdout, stderr = run_alignment(
        usalign,
        DATA_DIR / '7xg4.pdb',
        DATA_DIR / '6n40.pdb',
        ['-mm', '1', '-ter', '0', '-m', str(OUTPUTS_DIR / 'multimer_matrix.txt')],
        OUTPUTS_DIR / 'usalign_multimer_output.txt'
    )
    
    multimer_metrics = parse_alignment_output(stdout)
    print(f"  Aligned length: {multimer_metrics.get('aligned_length', 'N/A')}")
    print(f"  RMSD: {multimer_metrics.get('rmsd', 'N/A')} Å")
    print(f"  TM-score (norm by 7xg4): {multimer_metrics.get('tm_score_struct1', 'N/A')}")
    print(f"  TM-score (norm by 6n40): {multimer_metrics.get('tm_score_struct2', 'N/A')}")
    print(f"  Sequence identity: {multimer_metrics.get('seq_id', 'N/A')}")
    
    # =====================
    # 3. Run protein-only alignment
    # =====================
    print("\n[3] Running protein-only alignment (US-align -mol prot -mm 1)...")
    
    stdout, stderr = run_alignment(
        usalign,
        DATA_DIR / '7xg4.pdb',
        DATA_DIR / '6n40.pdb',
        ['-mm', '1', '-ter', '0', '-mol', 'prot'],
        OUTPUTS_DIR / 'usalign_prot_output.txt'
    )
    
    prot_metrics = parse_alignment_output(stdout)
    print(f"  Aligned length: {prot_metrics.get('aligned_length', 'N/A')}")
    print(f"  RMSD: {prot_metrics.get('rmsd', 'N/A')} Å")
    print(f"  TM-score (norm by 7xg4): {prot_metrics.get('tm_score_struct1', 'N/A')}")
    print(f"  TM-score (norm by 6n40): {prot_metrics.get('tm_score_struct2', 'N/A')}")
    print(f"  Sequence identity: {prot_metrics.get('seq_id', 'N/A')}")
    
    # =====================
    # 4. Run TM-align for each protein chain of 7xg4 against 6n40
    # =====================
    print("\n[4] Running pairwise chain alignments (TM-align)...")
    
    protein_chains_7xg4 = [c for c in sorted(chains_7xg4.keys()) 
                           if chains_7xg4[c]['type'] == 'protein']
    
    chain_results = []
    
    for chain_id in protein_chains_7xg4:
        # Extract single chain to temp PDB
        chain_pdb = OUTPUTS_DIR / f'temp_7xg4_{chain_id}.pdb'
        
        # Write chain-specific PDB
        with open(DATA_DIR / '7xg4.pdb', 'r') as fin, open(chain_pdb, 'w') as fout:
            for line in fin:
                if line.startswith(('ATOM', 'HETATM')):
                    if line[21] == chain_id:
                        fout.write(line)
                elif line.startswith('END'):
                    fout.write('END\n')
        
        # Run TM-align
        stdout, stderr = run_alignment(
            tmalign,
            chain_pdb,
            DATA_DIR / '6n40.pdb',
            [],
            OUTPUTS_DIR / f'tmalign_chain_{chain_id}_vs_6n40.txt'
        )
        
        metrics = parse_alignment_output(stdout)
        metrics['chain'] = chain_id
        metrics['chain_length'] = len(chains_7xg4[chain_id]['residues'])
        
        chain_results.append(metrics)
        
        print(f"  Chain {chain_id} vs 6n40: TM-score={metrics.get('tm_score_struct1', 'N/A')}, "
              f"RMSD={metrics.get('rmsd', 'N/A')}, aligned={metrics.get('aligned_length', 'N/A')}")
        
        # Clean up temp file
        chain_pdb.unlink(missing_ok=True)
    
    # Save all chain results
    all_results = {
        'multimer_alignment': {
            'method': 'US-align -mm 1',
            'metrics': multimer_metrics,
            'description': 'Multimer-to-multimer alignment (7xg4 12 chains vs 6n40 1 chain)'
        },
        'protein_only_alignment': {
            'method': 'US-align -mol prot -mm 1',
            'metrics': prot_metrics,
            'description': 'Protein-only multimer alignment'
        },
        'chain_by_chain_alignment': {
            'method': 'TM-align',
            'results': chain_results,
            'description': 'Individual chain alignments of 7xg4 protein chains vs 6n40'
        },
        'chain_composition': chain_info
    }
    
    with open(OUTPUTS_DIR / 'alignment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # =====================
    # 5. Read rotation matrix
    # =====================
    print("\n[5] Reading rotation matrix...")
    
    with open(OUTPUTS_DIR / 'multimer_matrix.txt', 'r') as f:
        matrix_content = f.read()
    print(matrix_content)
    
    # Parse matrix
    matrix_lines = [l.strip() for l in matrix_content.split('\n') if l.strip() and l.strip()[0].isdigit()]
    translation = []
    rotation = []
    for line in matrix_lines:
        parts = line.split()
        if len(parts) >= 5:
            t = float(parts[1])
            u0 = float(parts[2])
            u1 = float(parts[3])
            u2 = float(parts[4])
            translation.append(t)
            rotation.append([u0, u1, u2])
    
    print(f"\n  Translation vector (Å): [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
    print(f"  Rotation matrix:")
    for row in rotation:
        print(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")
    
    # =====================
    # 6. Generate Figures
    # =====================
    print("\n[6] Generating figures...")
    
    # Figure 1: Chain composition comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 7xg4 chain lengths
    chains_sorted_7xg4 = sorted(chains_7xg4.items(), key=lambda x: x[0])
    chain_ids = [c[0] for c in chains_sorted_7xg4]
    chain_lengths = [len(c[1]['residues']) for c in chains_sorted_7xg4]
    chain_types = [c[1]['type'] for c in chains_sorted_7xg4]
    
    colors = ['#2196F3' if t == 'protein' else '#FF9800' for t in chain_types]
    
    axes[0].bar(chain_ids, chain_lengths, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_xlabel('Chain ID', fontsize=12)
    axes[0].set_ylabel('Number of Residues', fontsize=12)
    axes[0].set_title('7xg4 Chain Composition\n(Type IV-A CRISPR-Cas Complex)', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Add value labels
    for i, (cid, length) in enumerate(zip(chain_ids, chain_lengths)):
        axes[0].text(i, length + 5, str(length), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    protein_patch = mpatches.Patch(color='#2196F3', label='Protein')
    rna_patch = mpatches.Patch(color='#FF9800', label='Nucleic Acid')
    axes[0].legend(handles=[protein_patch, rna_patch], loc='upper right')
    
    # 6n40 chain length
    chains_sorted_6n40 = sorted(chains_6n40.items(), key=lambda x: x[0])
    chain_ids_6 = [c[0] for c in chains_sorted_6n40]
    chain_lengths_6 = [len(c[1]['residues']) for c in chains_sorted_6n40]
    
    axes[1].bar(chain_ids_6, chain_lengths_6, color='#4CAF50', edgecolor='white', linewidth=1.5)
    axes[1].set_xlabel('Chain ID', fontsize=12)
    axes[1].set_ylabel('Number of Residues', fontsize=12)
    axes[1].set_title('6n40 Chain Composition\n(MmpL3 Membrane Protein)', fontsize=13, fontweight='bold')
    
    for i, (cid, length) in enumerate(zip(chain_ids_6, chain_lengths_6)):
        axes[1].text(i, length + 5, str(length), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'figure1_chain_composition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure1_chain_composition.png")
    
    # Figure 2: Pairwise chain TM-scores
    fig, ax = plt.subplots(figsize=(12, 6))
    
    chain_ids_tm = [r['chain'] for r in chain_results]
    tm_scores_struct1 = [r.get('tm_score_struct1', 0) or 0 for r in chain_results]
    tm_scores_struct2 = [r.get('tm_score_struct2', 0) or 0 for r in chain_results]
    rmsd_values = [r.get('rmsd', 0) or 0 for r in chain_results]
    aligned_lengths = [r.get('aligned_length', 0) or 0 for r in chain_results]
    
    x = np.arange(len(chain_ids_tm))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, tm_scores_struct1, width, label='TM-score (norm. by 7xg4 chain)', 
                   color='#2196F3', edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, tm_scores_struct2, width, label='TM-score (norm. by 6n40)', 
                   color='#4CAF50', edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Chain ID (from 7xg4)', fontsize=12)
    ax.set_ylabel('TM-score', fontsize=12)
    ax.set_title('Pairwise Chain Alignment: 7xg4 Protein Chains vs 6n40\n(TM-align)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(chain_ids_tm)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(max(tm_scores_struct1), max(tm_scores_struct2)) * 1.2 + 0.01)
    
    # Add value labels
    for bar, val in zip(bars1, tm_scores_struct1):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, tm_scores_struct2):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add a horizontal line at 0.5 (threshold for fold-level similarity)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Fold similarity threshold (0.5)')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'figure2_chainwise_tmscores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure2_chainwise_tmscores.png")
    
    # Figure 3: Alignment summary heatmap (RMSD vs aligned length)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSD scatter
    colors_scatter = ['#2196F3' if r.get('tm_score_struct1', 0) and r.get('tm_score_struct1', 0) > 0.3 
                      else '#FF9800' if r.get('tm_score_struct1', 0) and r.get('tm_score_struct1', 0) > 0.1 
                      else '#F44336' for r in chain_results]
    
    axes[0].scatter(rmsd_values, aligned_lengths, c=colors_scatter, s=100, edgecolors='black', linewidth=0.5, zorder=3)
    
    for i, r in enumerate(chain_results):
        axes[0].annotate(r['chain'], (rmsd_values[i], aligned_lengths[i]), 
                        textcoords="offset points", xytext=(5, 5), fontsize=10, fontweight='bold')
    
    axes[0].set_xlabel('RMSD (Å)', fontsize=12)
    axes[0].set_ylabel('Aligned Length (residues)', fontsize=12)
    axes[0].set_title('RMSD vs Alignment Coverage', fontsize=13, fontweight='bold')
    
    # Color legend
    high_patch = mpatches.Patch(color='#2196F3', label='TM-score > 0.3')
    med_patch = mpatches.Patch(color='#FF9800', label='TM-score 0.1-0.3')
    low_patch = mpatches.Patch(color='#F44336', label='TM-score < 0.1')
    axes[0].legend(handles=[high_patch, med_patch, low_patch], loc='upper right')
    
    # Sequence identity vs TM-score
    seq_ids_numeric = []
    for r in chain_results:
        sid = r.get('seq_id', '0.000')
        if isinstance(sid, str) and '/' in sid:
            parts = sid.split('/')
            try:
                seq_ids_numeric.append(float(parts[0]) / float(parts[1]))
            except:
                seq_ids_numeric.append(0)
        else:
            seq_ids_numeric.append(float(sid) if sid else 0)
    
    axes[1].scatter(seq_ids_numeric, tm_scores_struct1, c='#9C27B0', s=100, edgecolors='black', 
                   linewidth=0.5, zorder=3)
    
    for i, r in enumerate(chain_results):
        axes[1].annotate(r['chain'], (seq_ids_numeric[i], tm_scores_struct1[i]), 
                        textcoords="offset points", xytext=(5, 5), fontsize=10, fontweight='bold')
    
    axes[1].set_xlabel('Sequence Identity', fontsize=12)
    axes[1].set_ylabel('TM-score (norm. by 7xg4 chain)', fontsize=12)
    axes[1].set_title('Sequence Identity vs TM-score', fontsize=13, fontweight='bold')
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'figure3_alignment_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure3_alignment_metrics.png")
    
    # Figure 4: Summary comparison of alignment methods
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['US-align\n(multimer)', 'US-align\n(protein-only)', 'TM-align\n(chain A)']
    
    # Get chain A result for TM-align
    chain_a_result = next((r for r in chain_results if r['chain'] == 'A'), {})
    
    tm_scores = [
        multimer_metrics.get('tm_score_struct2', 0) or 0,
        prot_metrics.get('tm_score_struct2', 0) or 0,
        chain_a_result.get('tm_score_struct2', 0) or 0
    ]
    
    rmsd_vals = [
        multimer_metrics.get('rmsd', 0) or 0,
        prot_metrics.get('rmsd', 0) or 0,
        chain_a_result.get('rmsd', 0) or 0
    ]
    
    aligned_lens = [
        multimer_metrics.get('aligned_length', 0) or 0,
        prot_metrics.get('aligned_length', 0) or 0,
        chain_a_result.get('aligned_length', 0) or 0
    ]
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, [t / max(tm_scores) if max(tm_scores) > 0 else 0 for t in tm_scores], 
           width, label='TM-score (normalized)', color='#2196F3', edgecolor='white')
    ax.bar(x, [r / max(rmsd_vals) if max(rmsd_vals) > 0 else 0 for r in rmsd_vals], 
           width, label='RMSD (normalized)', color='#FF9800', edgecolor='white')
    ax.bar(x + width, [a / max(aligned_lens) if max(aligned_lens) > 0 else 0 for a in aligned_lens], 
           width, label='Aligned Length (normalized)', color='#4CAF50', edgecolor='white')
    
    ax.set_xlabel('Alignment Method', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Comparison of Alignment Methods\n(7xg4 vs 6n40)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10)
    
    # Add actual values as text
    for i in range(len(methods)):
        ax.text(i - width, tm_scores[i]/max(tm_scores) + 0.02, f'{tm_scores[i]:.3f}', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i, rmsd_vals[i]/max(rmsd_vals) + 0.02, f'{rmsd_vals[i]:.1f}Å', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + width, aligned_lens[i]/max(aligned_lens) + 0.02, f'{aligned_lens[i]}', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'figure4_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure4_method_comparison.png")
    
    # Figure 5: Rotation matrix visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Translation vector
    axes[0].bar(['X', 'Y', 'Z'], translation, color=['#F44336', '#4CAF50', '#2196F3'], 
               edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Translation (Å)', fontsize=12)
    axes[0].set_title('Superimposition Translation Vector', fontsize=13, fontweight='bold')
    
    for i, val in enumerate(translation):
        axes[0].text(i, val + 2, f'{val:.2f} Å', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Rotation matrix as heatmap
    rotation_array = np.array(rotation)
    im = axes[1].imshow(rotation_array, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(['X\'', 'Y\'', 'Z\''])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['X', 'Y', 'Z'])
    axes[1].set_title('Rotation Matrix', fontsize=13, fontweight='bold')
    
    # Add value annotations
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{rotation_array[i, j]:.4f}', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white' if abs(rotation_array[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=axes[1], label='Value')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'figure5_transformation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure5_transformation.png")
    
    # Figure 6: Comprehensive summary
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel A: Structure sizes
    ax1 = fig.add_subplot(gs[0, 0])
    structures = ['7xg4\n(CRISPR-Cas)', '6n40\n(MmpL3)']
    total_residues = [total_res_7xg4, total_res_6n40]
    total_chains = [len(chains_7xg4), len(chains_6n40)]
    
    x_pos = np.arange(len(structures))
    bars = ax1.bar(x_pos, total_residues, color=['#2196F3', '#4CAF50'], edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Total Residues', fontsize=11)
    ax1.set_title('A. Structure Sizes', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(structures)
    
    for bar, val, nc in zip(bars, total_residues, total_chains):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                f'{val}\n({nc} chains)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel B: Best chain alignment
    ax2 = fig.add_subplot(gs[0, 1])
    best_chain = max(chain_results, key=lambda r: r.get('tm_score_struct1', 0) or 0)
    worst_chain = min(chain_results, key=lambda r: r.get('tm_score_struct1', 0) or 0)
    
    categories = ['Best\n(' + best_chain['chain'] + ')', 'Worst\n(' + worst_chain['chain'] + ')']
    best_tm = best_chain.get('tm_score_struct1', 0) or 0
    worst_tm = worst_chain.get('tm_score_struct1', 0) or 0
    
    ax2.bar(categories, [best_tm, worst_tm], color=['#4CAF50', '#F44336'], edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('TM-score', fontsize=11)
    ax2.set_title('B. Best vs Worst Chain Alignment', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(best_tm, worst_tm) * 1.3 + 0.01)
    
    ax2.text(0, best_tm + 0.005, f'{best_tm:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.text(1, worst_tm + 0.005, f'{worst_tm:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel C: All chain TM-scores
    ax3 = fig.add_subplot(gs[0, 2])
    all_tm = [r.get('tm_score_struct1', 0) or 0 for r in chain_results]
    chain_labels = [r['chain'] for r in chain_results]
    
    colors_tm = ['#4CAF50' if t > 0.3 else '#FF9800' if t > 0.1 else '#F44336' for t in all_tm]
    ax3.barh(chain_labels, all_tm, color=colors_tm, edgecolor='white', linewidth=1)
    ax3.set_xlabel('TM-score', fontsize=11)
    ax3.set_title('C. All Chain TM-scores', fontsize=12, fontweight='bold')
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Fold threshold')
    ax3.legend(fontsize=9)
    
    # Panel D: RMSD distribution
    ax4 = fig.add_subplot(gs[1, 0])
    all_rmsd = [r.get('rmsd', 0) or 0 for r in chain_results]
    ax4.hist(all_rmsd, bins=10, color='#9C27B0', edgecolor='white', alpha=0.8)
    ax4.set_xlabel('RMSD (Å)', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('D. RMSD Distribution', fontsize=12, fontweight='bold')
    ax4.axvline(x=np.mean(all_rmsd), color='red', linestyle='--', label=f'Mean: {np.mean(all_rmsd):.1f}Å')
    ax4.legend(fontsize=9)
    
    # Panel E: Alignment coverage
    ax5 = fig.add_subplot(gs[1, 1])
    coverage = [(r.get('aligned_length', 0) or 0) / (r.get('chain_length', 1) or 1) * 100 for r in chain_results]
    ax5.bar(chain_labels, coverage, color='#FF9800', edgecolor='white', linewidth=1)
    ax5.set_xlabel('Chain ID', fontsize=11)
    ax5.set_ylabel('Alignment Coverage (%)', fontsize=11)
    ax5.set_title('E. Alignment Coverage per Chain', fontsize=12, fontweight='bold')
    
    # Panel F: Key findings summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = (
        f"KEY FINDINGS\n"
        f"{'='*35}\n\n"
        f"Structures:\n"
        f"  • 7xg4: {total_res_7xg4} residues, {len(chains_7xg4)} chains\n"
        f"  • 6n40: {total_res_6n40} residues, {len(chains_6n40)} chain(s)\n\n"
        f"Multimer Alignment:\n"
        f"  • TM-score: {multimer_metrics.get('tm_score_struct2', 0):.4f}\n"
        f"  • RMSD: {multimer_metrics.get('rmsd', 0):.2f} Å\n"
        f"  • Aligned: {multimer_metrics.get('aligned_length', 0)} residues\n\n"
        f"Best Chain Pair:\n"
        f"  • Chain {best_chain['chain']}: TM={best_tm:.4f}\n"
        f"  • RMSD={best_chain.get('rmsd', 0):.1f} Å\n\n"
        f"Interpretation:\n"
        f"  • Low TM-scores indicate structural\n"
        f"    divergence between these complexes\n"
        f"  • Foldseek-Multimer can detect weak\n"
        f"    structural similarities at scale"
    )
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    fig.suptitle('Structural Alignment Analysis: 7xg4 vs 6n40\n(Foldseek-Multimer Benchmark)', 
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.savefig(IMAGES_DIR / 'figure6_comprehensive_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure6_comprehensive_summary.png")
    
    # =====================
    # 7. Print final summary
    # =====================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - outputs/alignment_results.json")
    print(f"  - outputs/chain_composition.json")
    print(f"  - outputs/multimer_matrix.txt")
    print(f"  - outputs/usalign_multimer_output.txt")
    print(f"  - outputs/usalign_prot_output.txt")
    print(f"\nFigures:")
    for img in sorted(IMAGES_DIR.glob('*.png')):
        print(f"  - {img.relative_to(WORKSPACE)}")


if __name__ == '__main__':
    main()
