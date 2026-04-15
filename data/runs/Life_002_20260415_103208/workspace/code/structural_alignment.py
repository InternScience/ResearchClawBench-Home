"""
Protein Complex Structural Alignment Analysis
Aligns 7xg4.pdb (CRISPR-Cas complex) against 6n40.pdb (MMPL3 membrane protein)
using TM-score, Kabsch superposition, and chain-level correspondence mapping.
"""

import numpy as np
import json
import os
import time
from collections import defaultdict

# Parse PDB files manually for speed and control
def parse_pdb(pdb_path):
    """Parse PDB file and extract ATOM records by chain."""
    chains = defaultdict(list)
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":  # Only C-alpha atoms
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
    return dict(chains)

def get_chain_coords(chain_data):
    """Extract coordinate array from chain data."""
    return np.array([r['coords'] for r in chain_data])

def kabsch(P, Q):
    """
    Kabsch algorithm: find optimal rotation matrix U to minimize RMSD between P and Q.
    P, Q: (N, 3) arrays of coordinates (already centered).
    Returns: rotation matrix U (3x3)
    """
    C = np.dot(np.transpose(P), Q)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(Wt.T, V.T)))
    D = np.diag([1, 1, d])
    U = np.dot(np.dot(Wt.T, D), V.T)
    return U

def compute_rmsd(P, Q):
    """Compute RMSD between two aligned coordinate sets."""
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))

def tm_score(d_i, L_target):
    """
    Compute TM-score given distances and target length.
    TM-score = (1/L_target) * sum(1 / (1 + (d_i/d0)^2))
    d0 = 1.24 * (L_target - 15)^(1/3) - 1.8
    """
    if L_target <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * (L_target - 15)**(1/3) - 1.8
    return np.mean(1.0 / (1.0 + (d_i / d0)**2))

def tm_align_chains(coords1, coords2, max_iterations=50):
    """
    Simplified TM-align style iterative alignment.
    Returns: tm_score_value, rmsd, n_aligned, rotation_matrix, translation
    """
    L1, L2 = len(coords1), len(coords2)
    L_min = min(L1, L2)
    L_target = L2  # normalize by target
    
    if L_min < 3:
        return 0.0, 999.0, 0, np.eye(3), np.zeros(3)
    
    # Initialize with centering
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    
    P_centered = coords1 - center1
    Q_centered = coords2 - center2
    
    # Initial alignment using all available points (up to L_min)
    n_use = min(L1, L2)
    P = P_centered[:n_use]
    Q = Q_centered[:n_use]
    
    best_tm = 0.0
    best_U = np.eye(3)
    best_t = center2 - np.dot(center1, np.eye(3).T)
    best_n = n_use
    best_rmsd = compute_rmsd(P, Q)
    
    # Iterative refinement
    current_U = np.eye(3)
    current_t = center2 - center1
    weights = np.ones(n_use)
    
    for iteration in range(max_iterations):
        # Rotate and translate P
        P_rotated = np.dot(coords1, current_U.T) + current_t
        
        # Compute distances for all pairs
        distances = np.linalg.norm(P_rotated[:n_use] - Q_centered[:n_use], axis=1)
        
        # Compute TM-score with weights
        if L_target <= 15:
            d0 = 0.5
        else:
            d0 = 1.24 * (L_target - 15)**(1/3) - 1.8
        
        w = 1.0 / (1.0 + (distances / d0)**2)
        current_tm = np.mean(w)
        
        # Check convergence
        if abs(current_tm - best_tm) < 1e-6 and iteration > 5:
            break
        
        if current_tm > best_tm:
            best_tm = current_tm
            best_U = current_U.copy()
            best_t = current_t.copy()
            best_rmsd = np.sqrt(np.mean(distances**2))
            best_n = n_use
        
        # Update weights and recompute rotation
        weights = w
        
        # Weighted Kabsch
        P_w = coords1[:n_use] * np.sqrt(weights)[:, np.newaxis]
        Q_w = Q_centered[:n_use] * np.sqrt(weights)[:, np.newaxis]
        
        P_mean = np.mean(P_w, axis=0) / np.mean(weights)
        Q_mean = np.mean(Q_w, axis=0) / np.mean(weights)
        
        P_c = P_w - P_mean * np.sqrt(weights)[:, np.newaxis]
        Q_c = Q_w - Q_mean * np.sqrt(weights)[:, np.newaxis]
        
        C = np.dot(P_c.T, Q_c)
        V_s, S_s, Wt_s = np.linalg.svd(C)
        d_det = np.sign(np.linalg.det(np.dot(Wt_s.T, V_s.T)))
        D_mat = np.diag([1, 1, d_det])
        current_U = np.dot(np.dot(Wt_s.T, D_mat), V_s.T)
        current_t = Q_mean - np.dot(P_mean, current_U.T)
    
    # Final TM-score computation
    P_final = np.dot(coords1, best_U.T) + best_t
    final_distances = np.linalg.norm(P_final[:best_n] - Q_centered[:best_n], axis=1)
    final_tm = tm_score(final_distances, L_target)
    final_rmsd = compute_rmsd(P_final[:best_n], Q_centered[:best_n])
    
    return final_tm, final_rmsd, best_n, best_U, best_t

def compute_complex_tm_score(chain_results, L_target_total):
    """Compute overall complex TM-score from chain-level results."""
    if not chain_results:
        return 0.0
    total_score = sum(r['tm_score'] * r['n_aligned'] for r in chain_results)
    total_aligned = sum(r['n_aligned'] for r in chain_results)
    if total_aligned == 0:
        return 0.0
    return total_score * total_aligned / L_target_total if L_target_total > 0 else 0.0

def main():
    print("=" * 70)
    print("Protein Complex Structural Alignment Analysis")
    print("=" * 70)
    
    # Parse PDB files
    print("\n[1] Parsing PDB files...")
    chains_7xg4 = parse_pdb('data/7xg4.pdb')
    chains_6n40 = parse_pdb('data/6n40.pdb')
    
    print(f"  7xg4 chains: {sorted(chains_7xg4.keys())}")
    for cid, cdata in sorted(chains_7xg4.items()):
        print(f"    Chain {cid}: {len(cdata)} residues")
    
    print(f"  6n40 chains: {sorted(chains_6n40.keys())}")
    for cid, cdata in sorted(chains_6n40.items()):
        print(f"    Chain {cid}: {len(cdata)} residues")
    
    # Save chain info
    chain_info = {
        '7xg4': {cid: {'n_residues': len(cdata)} for cid, cdata in chains_7xg4.items()},
        '6n40': {cid: {'n_residues': len(cdata)} for cid, cdata in chains_6n40.items()}
    }
    with open('outputs/chain_info.json', 'w') as f:
        json.dump(chain_info, f, indent=2)
    
    # Extract coordinates
    print("\n[2] Extracting C-alpha coordinates...")
    coords_7xg4 = {}
    for cid, cdata in chains_7xg4.items():
        coords_7xg4[cid] = get_chain_coords(cdata)
    
    coords_6n40 = {}
    for cid, cdata in chains_6n40.items():
        coords_6n40[cid] = get_chain_coords(cdata)
    
    # Pairwise chain alignment
    print("\n[3] Performing pairwise chain-level structural alignment...")
    alignment_results = []
    
    for chain_6n40_id, coords_target in sorted(coords_6n40.items()):
        L_target = len(coords_target)
        print(f"\n  Target: 6n40 Chain {chain_6n40_id} ({L_target} residues)")
        
        for chain_7xg4_id, coords_query in sorted(coords_7xg4.items()):
            L_query = len(coords_query)
            
            start_time = time.time()
            tm_val, rmsd_val, n_aligned, rot_matrix, translation = tm_align_chains(
                coords_query, coords_target
            )
            elapsed = time.time() - start_time
            
            result = {
                'query_chain': f'7xg4_{chain_7xg4_id}',
                'target_chain': f'6n40_{chain_6n40_id}',
                'query_length': L_query,
                'target_length': L_target,
                'n_aligned': n_aligned,
                'tm_score': round(tm_val, 4),
                'rmsd': round(rmsd_val, 4),
                'rotation_matrix': rot_matrix.tolist(),
                'translation_vector': translation.tolist(),
                'computation_time_sec': round(elapsed, 4)
            }
            alignment_results.append(result)
            
            print(f"    vs 7xg4 Chain {chain_7xg4_id} ({L_query} res): "
                  f"TM={tm_val:.4f}, RMSD={rmsd_val:.4f}A, aligned={n_aligned}, "
                  f"time={elapsed:.3f}s")
    
    # Save alignment results
    with open('outputs/alignment_results.json', 'w') as f:
        json.dump(alignment_results, f, indent=2)
    
    # Find best chain correspondence
    print("\n[4] Identifying best chain correspondences...")
    best_result = max(alignment_results, key=lambda x: x['tm_score'])
    print(f"  Best alignment: {best_result['query_chain']} -> {best_result['target_chain']}")
    print(f"    TM-score: {best_result['tm_score']:.4f}")
    print(f"    RMSD: {best_result['rmsd']:.4f} A")
    print(f"    Aligned residues: {best_result['n_aligned']}")
    
    # Full complex alignment (concatenate all chains)
    print("\n[5] Performing full complex-level alignment...")
    all_coords_7xg4 = np.vstack([coords_7xg4[cid] for cid in sorted(coords_7xg4.keys())])
    all_coords_6n40 = np.vstack([coords_6n40[cid] for cid in sorted(coords_6n40.keys())])
    
    start_time = time.time()
    complex_tm, complex_rmsd, complex_n, complex_rot, complex_trans = tm_align_chains(
        all_coords_7xg4, all_coords_6n40
    )
    complex_time = time.time() - start_time
    
    complex_result = {
        'query': '7xg4 (all chains)',
        'target': '6n40 (all chains)',
        'query_total_residues': len(all_coords_7xg4),
        'target_total_residues': len(all_coords_6n40),
        'n_aligned': complex_n,
        'tm_score': round(complex_tm, 4),
        'rmsd': round(complex_rmsd, 4),
        'rotation_matrix': complex_rot.tolist(),
        'translation_vector': complex_trans.tolist(),
        'computation_time_sec': round(complex_time, 4)
    }
    
    with open('outputs/complex_alignment.json', 'w') as f:
        json.dump(complex_result, f, indent=2)
    
    print(f"  Full complex: TM={complex_tm:.4f}, RMSD={complex_rmsd:.4f}A, "
          f"aligned={complex_n}, time={complex_time:.3f}s")
    
    # Summary statistics
    print("\n[6] Computing summary statistics...")
    tm_scores = [r['tm_score'] for r in alignment_results]
    rmsd_values = [r['rmsd'] for r in alignment_results]
    
    summary = {
        'n_pairwise_comparisons': len(alignment_results),
        'mean_tm_score': round(np.mean(tm_scores), 4),
        'std_tm_score': round(np.std(tm_scores), 4),
        'max_tm_score': round(max(tm_scores), 4),
        'min_tm_score': round(min(tm_scores), 4),
        'mean_rmsd': round(np.mean(rmsd_values), 4),
        'median_rmsd': round(np.median(rmsd_values), 4),
        'complex_tm_score': round(complex_tm, 4),
        'complex_rmsd': round(complex_rmsd, 4),
        'best_chain_pair': f"{best_result['query_chain']} -> {best_result['target_chain']}",
        'best_chain_tm': best_result['tm_score'],
        'best_chain_rmsd': best_result['rmsd']
    }
    
    with open('outputs/summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("Analysis complete. Results saved to outputs/")
    print("=" * 70)
    
    return alignment_results, complex_result, summary

if __name__ == '__main__':
    main()
