#!/usr/bin/env python3
"""
Protein Complex Structural Alignment Analysis
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser, PDBIO, Select
import json
import subprocess
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
REPORT_IMG_DIR = "report/images"
USALIGN_PATH = "/tmp/USalign/USalign"
TMALIGN_PATH = "/tmp/USalign/TMalign"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

class StructureAnalyzer:
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure(os.path.basename(pdb_file), pdb_file)
        self.chains = {}
        self.chain_lengths = {}
        self.extract_chains()
        
    def extract_chains(self):
        for model in self.structure:
            for chain in model:
                chain_id = chain.id
                residues = []
                ca_atoms = []
                for residue in chain:
                    if residue.id[0] == ' ':
                        residues.append(residue)
                        if 'CA' in residue:
                            ca_atoms.append(residue['CA'].coord)
                        elif "C1'" in residue:
                            ca_atoms.append(residue["C1'"].coord)
                
                if ca_atoms:
                    self.chains[chain_id] = {
                        'residues': residues,
                        'ca_coords': np.array(ca_atoms),
                        'length': len(residues)
                    }
                    self.chain_lengths[chain_id] = len(residues)
    
    def get_chain_coords(self, chain_id):
        if chain_id in self.chains:
            return self.chains[chain_id]['ca_coords']
        return None
    
    def get_summary(self):
        return {
            'pdb_file': self.pdb_file,
            'num_chains': len(self.chains),
            'chain_ids': list(self.chains.keys()),
            'chain_lengths': self.chain_lengths,
            'total_residues': sum(self.chain_lengths.values())
        }


def extract_chain(pdb_file, chain_id, output_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('temp', pdb_file)
    
    class ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, ChainSelect())


def run_tmalign(structure1, structure2, chain1=None, chain2=None):
    if chain1 or chain2:
        temp_dir = f"{OUTPUT_DIR}/temp"
        os.makedirs(temp_dir, exist_ok=True)
        struct1_name = os.path.basename(structure1).replace('.pdb', '')
        struct2_name = os.path.basename(structure2).replace('.pdb', '')
        
        if chain1:
            temp1 = f"{temp_dir}/{struct1_name}_{chain1}.pdb"
            extract_chain(structure1, chain1, temp1)
            structure1 = temp1
        if chain2:
            temp2 = f"{temp_dir}/{struct2_name}_{chain2}.pdb"
            extract_chain(structure2, chain2, temp2)
            structure2 = temp2
    
    cmd = [TMALIGN_PATH, structure1, structure2]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_tmalign_output(result.stdout), result.stdout


def parse_tmalign_output(output):
    result = {'tm_score': None, 'rmsd': None, 'aligned_length': None}
    for line in output.split('\n'):
        if 'TM-score=' in line:
            parts = line.split('TM-score=')
            if len(parts) > 1:
                score_str = parts[1].split()[0]
                try:
                    result['tm_score'] = float(score_str)
                except:
                    pass
        if 'RMSD=' in line:
            parts = line.split('RMSD=')
            if len(parts) > 1:
                rmsd_str = parts[1].split(',')[0].strip()
                try:
                    result['rmsd'] = float(rmsd_str)
                except:
                    pass
        if 'Aligned length=' in line:
            parts = line.split('Aligned length=')
            if len(parts) > 1:
                len_str = parts[1].split(',')[0].strip()
                try:
                    result['aligned_length'] = int(len_str)
                except:
                    pass
    return result


def kabsch_alignment(coords1, coords2):
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    H = np.dot(coords1_centered.T, coords2_centered)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
    diag = np.diag([1, 1, d])
    rotation_matrix = np.dot(Vt.T, np.dot(diag, U.T))
    translation_vector = centroid2 - np.dot(centroid1, rotation_matrix)
    coords1_aligned = np.dot(coords1, rotation_matrix) + translation_vector
    rmsd = np.sqrt(np.mean(np.sum((coords1_aligned - coords2)**2, axis=1)))
    return rotation_matrix, translation_vector, rmsd, coords1_aligned


def calculate_tm_score(coords1, coords2, L_target=None):
    if len(coords1) != len(coords2):
        return None
    if L_target is None:
        L_target = len(coords1)
    if L_target <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * np.cbrt(L_target - 15) - 1.8
    if d0 <= 0:
        d0 = 0.5
    distances = np.sqrt(np.sum((coords1 - coords2)**2, axis=1))
    tm_scores = 1 / (1 + (distances / d0)**2)
    tm_score = np.sum(tm_scores) / L_target
    return tm_score


def perform_chain_alignments(analyzer1, analyzer2):
    results = []
    for chain1_id in analyzer1.chains:
        for chain2_id in analyzer2.chains:
            coords1 = analyzer1.get_chain_coords(chain1_id)
            coords2 = analyzer2.get_chain_coords(chain2_id)
            min_len = min(len(coords1), len(coords2))
            max_len = max(len(coords1), len(coords2))
            if min_len < 20 or max_len / min_len > 2.0:
                continue
            use_len = min_len
            coords1_trim = coords1[:use_len]
            coords2_trim = coords2[:use_len]
            R, T, rmsd, aligned = kabsch_alignment(coords1_trim, coords2_trim)
            tm_score = calculate_tm_score(aligned, coords2_trim, max_len)
            results.append({
                'chain1': chain1_id,
                'chain2': chain2_id,
                'length1': len(coords1),
                'length2': len(coords2),
                'aligned_length': use_len,
                'rmsd': rmsd,
                'tm_score': tm_score,
                'rotation_matrix': R.tolist(),
                'translation_vector': T.tolist()
            })
    return results


def visualize_structure_overview(analyzer1, analyzer2):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    chains1 = list(analyzer1.chain_lengths.keys())
    lengths1 = list(analyzer1.chain_lengths.values())
    chains2 = list(analyzer2.chain_lengths.keys())
    lengths2 = list(analyzer2.chain_lengths.values())
    
    ax1 = axes[0, 0]
    x1 = np.arange(len(chains1))
    x2 = np.arange(len(chains2))
    ax1.bar(x1 - 0.2, lengths1, 0.4, label='7xg4', color='steelblue')
    ax1.bar(x2 + 0.2, lengths2, 0.4, label='6n40', color='coral')
    ax1.set_xlabel('Chain ID')
    ax1.set_ylabel('Number of Residues')
    ax1.set_title('Chain Length Comparison')
    ax1.set_xticks(range(max(len(chains1), len(chains2))))
    labels = []
    for i in range(max(len(chains1), len(chains2))):
        if i < len(chains1) and i < len(chains2):
            labels.append(f"{chains1[i]}\n{chains2[i]}")
        elif i < len(chains1):
            labels.append(chains1[i])
        else:
            labels.append(chains2[i])
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.axis('off')
    summary1 = analyzer1.get_summary()
    summary2 = analyzer2.get_summary()
    info_text = f"""Structure Comparison Summary

7xg4 (Type IV-A CRISPR-Cas):
- PDB ID: 7xg4
- Chains: {summary1['num_chains']} ({', '.join(summary1['chain_ids'])})
- Total residues: {summary1['total_residues']}
- Structure type: Multi-chain complex
- Components: CSF1, CSF3, CSF2 (x5), CSF5, crRNA, NTS, TS, CSF4

6n40 (MMPL3 membrane protein):
- PDB ID: 6n40
- Chains: {summary2['num_chains']} ({', '.join(summary2['chain_ids'])})
- Total residues: {summary2['total_residues']}
- Structure type: Single-chain membrane protein
- Resolution: 3.31 Angstrom"""
    ax2.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax2.transAxes)
    
    ax3 = axes[1, 0]
    ax3.hist(lengths1, bins=10, alpha=0.7, label='7xg4', color='steelblue', edgecolor='black')
    ax3.hist(lengths2, bins=10, alpha=0.7, label='6n40', color='coral', edgecolor='black')
    ax3.set_xlabel('Chain Length (residues)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Chain Length Distribution')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = axes[1, 1]
    categories = ['7xg4\n(Multi-chain)', '6n40\n(Single-chain)']
    num_chains = [summary1['num_chains'], summary2['num_chains']]
    total_res = [summary1['total_residues'], summary2['total_residues']]
    x = np.arange(len(categories))
    width = 0.35
    ax4.bar(x - width/2, num_chains, width, label='Number of Chains', color='steelblue')
    ax4.set_ylabel('Number of Chains', color='steelblue')
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x + width/2, total_res, width, label='Total Residues', color='coral')
    ax4_twin.set_ylabel('Total Residues', color='coral')
    ax4_twin.tick_params(axis='y', labelcolor='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.set_title('Structural Complexity Comparison')
    
    plt.tight_layout()
    plt.savefig(f"{REPORT_IMG_DIR}/structure_overview.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/structure_overview.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved structure overview")


def visualize_alignment_results(alignment_results):
    if not alignment_results:
        print("No alignment results to visualize")
        return
    
    df = pd.DataFrame(alignment_results)
    chain1_ids = sorted(set(df['chain1']))
    chain2_ids = sorted(set(df['chain2']))
    
    tm_matrix = np.zeros((len(chain1_ids), len(chain2_ids)))
    rmsd_matrix = np.zeros((len(chain1_ids), len(chain2_ids)))
    
    for _, row in df.iterrows():
        i = chain1_ids.index(row['chain1'])
        j = chain2_ids.index(row['chain2'])
        tm_matrix[i, j] = row['tm_score']
        rmsd_matrix[i, j] = row['rmsd']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax1 = axes[0, 0]
    im = ax1.imshow(tm_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(chain2_ids)))
    ax1.set_yticks(range(len(chain1_ids)))
    ax1.set_xticklabels(chain2_ids)
    ax1.set_yticklabels(chain1_ids)
    ax1.set_xlabel('6n40 Chain ID')
    ax1.set_ylabel('7xg4 Chain ID')
    ax1.set_title('TM-score Heatmap (Chain-level)')
    plt.colorbar(im, ax=ax1, label='TM-score')
    for i in range(len(chain1_ids)):
        for j in range(len(chain2_ids)):
            text = ax1.text(j, i, f'{tm_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['rmsd'], df['tm_score'], 
                         s=df['aligned_length']/5, 
                         c=df['aligned_length'], 
                         cmap='viridis', alpha=0.6, edgecolors='black')
    ax2.set_xlabel('RMSD (Angstrom)')
    ax2.set_ylabel('TM-score')
    ax2.set_title('TM-score vs RMSD')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(left=0)
    plt.colorbar(scatter, ax=ax2, label='Aligned Length')
    
    ax3 = axes[1, 0]
    df_sorted = df.sort_values('tm_score', ascending=False).head(15)
    labels = [f"{r['chain1']}-{r['chain2']}" for _, r in df_sorted.iterrows()]
    bars = ax3.barh(range(len(labels)), df_sorted['tm_score'], color='steelblue')
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels)
    ax3.set_xlabel('TM-score')
    ax3.set_title('Top 15 Chain Pair Alignments')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, df_sorted['tm_score'])):
        ax3.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=8)
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""Chain-Level Alignment Statistics

Total chain pairs: {len(df)}

TM-score:
- Mean: {df['tm_score'].mean():.4f}
- Median: {df['tm_score'].median():.4f}
- Max: {df['tm_score'].max():.4f}
- Min: {df['tm_score'].min():.4f}

RMSD:
- Mean: {df['rmsd'].mean():.2f} A
- Min: {df['rmsd'].min():.2f} A
- Max: {df['rmsd'].max():.2f} A

TM-score > 0.5: {len(df[df['tm_score'] > 0.5])} ({100*len(df[df['tm_score'] > 0.5])/len(df):.1f}%)"""
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{REPORT_IMG_DIR}/chain_alignments.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/chain_alignments.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved chain alignment results")
    return df


def run_multimer_alignment():
    print("\n=== US-align Multimer Alignment ===")
    structure1 = f"{DATA_DIR}/7xg4.pdb"
    structure2 = f"{DATA_DIR}/6n40.pdb"
    output_file = f"{OUTPUT_DIR}/usalign_multimer_output.txt"
    
    cmd = [USALIGN_PATH, structure1, structure2, "-mm", "1", "-ter", "0", "-outfmt", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    with open(output_file, 'w') as f:
        f.write(result.stdout)
        f.write(result.stderr)
    
    result_data = parse_tmalign_output(result.stdout)
    print(f"TM-score: {result_data['tm_score']}")
    print(f"RMSD: {result_data['rmsd']}")
    return result_data, result.stdout


def main():
    print("="*60)
    print("Protein Complex Structural Alignment Analysis")
    print("="*60)
    
    print("\n=== Parsing PDB Structures ===")
    analyzer1 = StructureAnalyzer(f"{DATA_DIR}/7xg4.pdb")
    analyzer2 = StructureAnalyzer(f"{DATA_DIR}/6n40.pdb")
    
    summary1 = analyzer1.get_summary()
    summary2 = analyzer2.get_summary()
    
    print(f"\n7xg4: {summary1['num_chains']} chains, {summary1['total_residues']} residues")
    print(f"6n40: {summary2['num_chains']} chains, {summary2['total_residues']} residues")
    
    print("\n=== Generating Structure Overview ===")
    visualize_structure_overview(analyzer1, analyzer2)
    
    print("\n=== Chain-level Alignments ===")
    alignment_results = perform_chain_alignments(analyzer1, analyzer2)
    df_alignments = visualize_alignment_results(alignment_results)
    
    print("\n=== Multimer Alignment ===")
    multimer_result, multimer_output = run_multimer_alignment()
    
    with open(f"{OUTPUT_DIR}/alignment_results.json", 'w') as f:
        json.dump({
            'structure1_summary': summary1,
            'structure2_summary': summary2,
            'chain_alignments': alignment_results,
            'multimer_alignment': multimer_result
        }, f, indent=2)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to {OUTPUT_DIR}/")
    print(f"Figures saved to {REPORT_IMG_DIR}/")
    
    return analyzer1, analyzer2, alignment_results, multimer_result


if __name__ == "__main__":
    main()
