"""
Protein Complex Structural Alignment Analysis - v2
Uses tmtools (TM-align bindings) for accurate TM-score computation.
Aligns 7xg4.pdb (CRISPR-Cas complex) against 6n40.pdb (MMPL3 membrane protein).
"""

import numpy as np
import json
import os
import time
from collections import defaultdict
from tmtools import tm_align

def parse_pdb(pdb_path):
    """Parse PDB file and extract ATOM records by chain."""
    chains = defaultdict(list)
    seqs = defaultdict(str)
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    res_name = line[17:20].strip()
                    chain_id = line[21]
                    res_seq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    chains[chain_id].append({
                        'res_name': res_name,
                        'res_seq': res_seq,
                        'coords': np.array([x, y, z])
                    })
                    aa = three_to_one.get(res_name, 'X')
                    seqs[chain_id] += aa
    return dict(chains), dict(seqs)

def get_chain_coords(chain_data):
    """Extract coordinate array from chain data."""
    return np.array([r['coords'] for r in chain_data])

def kabsch(P, Q):
    """Kabsch algorithm for optimal rotation."""
    C = np.dot(np.transpose(P), Q)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(Wt.T, V.T)))
    D = np.diag([1, 1, d])
    U = np.dot(np.dot(Wt.T, D), V.T)
    return U

def tm_score_from_distances(d_i, L_target):
    """Compute TM-score from distances."""
    if L_target <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * (L_target - 15)**(1/3) - 1.8
    return np.mean(1.0 / (1.0 + (d_i / d0)**2))

def main():
    print("=" * 70)
    print("Protein Complex Structural Alignment Analysis")
    print("Using TM-align (tmtools) for accurate structural alignment")
    print("=" * 70)
    
    # Parse PDB files
    print("\n[1] Parsing PDB files...")
    chains_7xg4, seqs_7xg4 = parse_pdb('data/7xg4.pdb')
    chains_6n40, seqs_6n40 = parse_pdb('data/6n40.pdb')
    
    print(f"  7xg4 chains: {sorted(chains_7xg4.keys())}")
    for cid, cdata in sorted(chains_7xg4.items()):
        print(f"    Chain {cid}: {len(cdata)} residues, seq_len={len(seqs_7xg4[cid])}")
    
    print(f"  6n40 chains: {sorted(chains_6n40.keys())}")
    for cid, cdata in sorted(chains_6n40.items()):
        print(f"    Chain {cid}: {len(cdata)} residues, seq_len={len(seqs_6n40[cid])}")
    
    # Save chain info
    chain_info = {
        '7xg4': {cid: {'n_residues': len(cdata), 'sequence_length': len(seqs_7xg4[cid])} 
                 for cid, cdata in chains_7xg4.items()},
        '6n40': {cid: {'n_residues': len(cdata), 'sequence_length': len(seqs_6n40[cid])} 
                 for cid, cdata in chains_6n40.items()}
    }
    with open('outputs/chain_info.json', 'w') as f:
        json.dump(chain_info, f, indent=2)
    
    # Extract coordinates and sequences
    coords_7xg4 = {cid: get_chain_coords(cdata) for cid, cdata in chains_7xg4.items()}
    coords_6n40 = {cid: get_chain_coords(cdata) for cid, cdata in chains_6n40.items()}
    
    # Pairwise chain alignment using TM-align
    print("\n[2] Performing pairwise chain-level structural alignment (TM-align)...")
    alignment_results = []
    
    for chain_6n40_id, coords_target in sorted(coords_6n40.items()):
        L_target = len(coords_target)
        seq_target = seqs_6n40[chain_6n40_id]
        print(f"\n  Target: 6n40 Chain {chain_6n40_id} ({L_target} residues)")
        
        for chain_7xg4_id, coords_query in sorted(coords_7xg4.items()):
            L_query = len(coords_query)
            seq_query = seqs_7xg4[chain_7xg4_id]
            
            start_time = time.time()
            try:
                result = tm_align(coords_query, coords_target, seq_query, seq_target)
                
                tm_val = result.tm_norm_chain2  # TM-score normalized by target (chain2)
                rmsd_val = result.rmsd
                n_aligned = len(result.seqM.replace('-', '').replace(':', ''))
                
                # Get rotation and translation
                rot_matrix = result.u
                translation = result.t
                elapsed = time.time() - start_time
                ar = {
                    'query_chain': f'7xg4_{chain_7xg4_id}',
                    'target_chain': f'6n40_{chain_6n40_id}',
                    'query_length': L_query,
                    'target_length': L_target,
                    'n_aligned': int(n_aligned),
                    'tm_score': round(float(tm_val), 4),
                    'rmsd': round(float(rmsd_val), 4) if rmsd_val is not None else 999.0,
                    'rotation_matrix': rot_matrix.tolist() if hasattr(rot_matrix, 'tolist') else rot_matrix,
                    'translation_vector': translation.tolist() if hasattr(translation, 'tolist') else translation,
                    'computation_time_sec': round(elapsed, 4)
                }
                alignment_results.append(ar)
                
                print(f"    vs 7xg4 Chain {chain_7xg4_id} ({L_query} res): "
                      f"TM={tm_val:.4f}, RMSD={rmsd_val:.4f}A, aligned={n_aligned}, "
                      f"time={elapsed:.3f}s")
            except Exception as e:
                print(f"    vs 7xg4 Chain {chain_7xg4_id}: ERROR - {e}")
    
    # Save alignment results
    with open('outputs/alignment_results.json', 'w') as f:
        json.dump(alignment_results, f, indent=2)
    
    # Find best chain correspondence
    print("\n[3] Identifying best chain correspondences...")
    best_result = max(alignment_results, key=lambda x: x['tm_score'])
    print(f"  Best alignment: {best_result['query_chain']} -> {best_result['target_chain']}")
    print(f"    TM-score: {best_result['tm_score']:.4f}")
    print(f"    RMSD: {best_result['rmsd']:.4f} A")
    print(f"    Aligned residues: {best_result['n_aligned']}")
    
    # Full complex alignment
    print("\n[4] Performing full complex-level alignment...")
    all_coords_7xg4 = np.vstack([coords_7xg4[cid] for cid in sorted(coords_7xg4.keys())])
    all_coords_6n40 = np.vstack([coords_6n40[cid] for cid in sorted(coords_6n40.keys())])
    
    all_seq_7xg4 = ''.join([seqs_7xg4[cid] for cid in sorted(coords_7xg4.keys())])
    all_seq_6n40 = ''.join([seqs_6n40[cid] for cid in sorted(coords_6n40.keys())])
    
    start_time = time.time()
    complex_result = tm_align(all_coords_7xg4, all_coords_6n40, all_seq_7xg4, all_seq_6n40)
    complex_time = time.time() - start_time
    
    complex_output = {
        'query': '7xg4 (all chains)',
        'target': '6n40 (all chains)',
        'query_total_residues': int(len(all_coords_7xg4)),
        'target_total_residues': int(len(all_coords_6n40)),
        'n_aligned': int(len(complex_result.seqM.replace('-', '').replace(':', ''))),
        'tm_score': round(float(complex_result.tm_norm_chain2), 4),
        'rmsd': round(float(complex_result.rmsd), 4) if complex_result.rmsd is not None else 999.0,
        'rotation_matrix': complex_result.u.tolist() if hasattr(complex_result.u, 'tolist') else complex_result.u,
        'translation_vector': complex_result.t.tolist() if hasattr(complex_result.t, 'tolist') else complex_result.t,
        'computation_time_sec': round(complex_time, 4)
    }
    
    with open('outputs/complex_alignment.json', 'w') as f:
        json.dump(complex_output, f, indent=2)
    
    print(f"  Full complex: TM={complex_result.tm_norm_chain2:.4f}, RMSD={complex_result.rmsd:.4f}A, "
          f"aligned={len(complex_result.seqM.replace('-', '').replace(':', ''))}, time={complex_time:.3f}s")
    
    # Summary statistics
    print("\n[5] Computing summary statistics...")
    tm_scores = [r['tm_score'] for r in alignment_results]
    rmsd_values = [r['rmsd'] for r in alignment_results]
    
    summary = {
        'n_pairwise_comparisons': len(alignment_results),
        'mean_tm_score': round(float(np.mean(tm_scores)), 4),
        'std_tm_score': round(float(np.std(tm_scores)), 4),
        'max_tm_score': round(float(max(tm_scores)), 4),
        'min_tm_score': round(float(min(tm_scores)), 4),
        'mean_rmsd': round(float(np.mean(rmsd_values)), 4),
        'median_rmsd': round(float(np.median(rmsd_values)), 4),
        'complex_tm_score': round(float(complex_result.tm_norm_chain2), 4),
        'complex_rmsd': round(float(complex_result.rmsd), 4) if complex_result.rmsd is not None else 999.0,
        'best_chain_pair': f"{best_result['query_chain']} -> {best_result['target_chain']}",
        'best_chain_tm': best_result['tm_score'],
        'best_chain_rmsd': best_result['rmsd']
    }
    
    with open('outputs/summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Save superimposition vectors
    print("\n[6] Saving superimposition vectors...")
    superimposition = {
        'best_chain_pair': {
            'query': best_result['query_chain'],
            'target': best_result['target_chain'],
            'rotation_matrix': best_result['rotation_matrix'],
            'translation_vector': best_result['translation_vector']
        },
        'complex_level': {
            'rotation_matrix': complex_output['rotation_matrix'],
            'translation_vector': complex_output['translation_vector']
        }
    }
    with open('outputs/superimposition_vectors.json', 'w') as f:
        json.dump(superimposition, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Analysis complete. Results saved to outputs/")
    print("=" * 70)
    
    return alignment_results, complex_output, summary

if __name__ == '__main__':
    main()
