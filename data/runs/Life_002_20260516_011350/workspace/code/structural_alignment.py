#!/usr/bin/env python3
"""
Optimized Structural Alignment: 7xg4 vs 6n40
Uses vectorized operations for speed.
"""

import numpy as np
import json
import os
from collections import defaultdict

def parse_pdb(filepath):
    chains = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    chain_id = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    chains[chain_id].append({
                        'res_num': res_num, 'res_name': res_name,
                        'x': x, 'y': y, 'z': z
                    })
    chain_coords = {}
    for cid, residues in chains.items():
        chain_coords[cid] = np.array([[r['x'], r['y'], r['z']] for r in residues])
    return chain_coords

def parse_pdb_all_atoms(filepath):
    chains = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                if atom_name in ['CA', 'P', "C4'"]:
                    chain_id = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    atom_type = 'protein' if atom_name == 'CA' else 'nucleic'
                    chains[chain_id].append({
                        'res_num': res_num, 'res_name': res_name,
                        'x': x, 'y': y, 'z': z,
                        'atom_type': atom_type, 'atom_name': atom_name
                    })
    chain_coords = {}
    chain_meta = {}
    for cid, residues in chains.items():
        chain_coords[cid] = np.array([[r['x'], r['y'], r['z']] for r in residues])
        n_protein = sum(1 for r in residues if r['atom_type'] == 'protein')
        n_nucleic = sum(1 for r in residues if r['atom_type'] == 'nucleic')
        chain_meta[cid] = {'type': 'protein' if n_protein > n_nucleic else 'nucleic', 'length': len(residues)}
    return chain_coords, chain_meta

def kabsch_rotation(P, Q):
    p_center = np.mean(P, axis=0)
    q_center = np.mean(Q, axis=0)
    Pc = P - p_center
    Qc = Q - q_center
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    t = q_center - R @ p_center
    return R, t

def compute_d0(L):
    return max(1.24 * (max(L, 15) - 15) ** (1.0/3.0) - 1.8, 0.5)

def tm_score_fast(coords1, coords2):
    n = len(coords1)
    if n < 3:
        return 0.0, 0.0, float('inf'), np.eye(3), np.zeros(3), None
    
    L1, L2 = len(coords1), len(coords2)
    d0_1 = compute_d0(L1)
    d0_2 = compute_d0(L2)
    
    R, t = kabsch_rotation(coords1, coords2)
    c1_rot = (R @ coords1.T).T + t
    dists = np.sqrt(np.sum((c1_rot - coords2) ** 2, axis=1))
    
    tm1 = float(np.mean(1.0 / (1.0 + (dists / d0_1) ** 2)))
    tm2 = float(np.mean(1.0 / (1.0 + (dists / d0_2) ** 2)))
    rmsd = float(np.sqrt(np.mean(dists ** 2)))
    
    return tm1, tm2, rmsd, R, t, dists

def gapless_align(coords1, coords2, step=None):
    """Gapless sliding window alignment."""
    if len(coords1) > len(coords2):
        coords1, coords2 = coords2, coords1
        swapped = True
    else:
        swapped = False
    
    n1, n2 = len(coords1), len(coords2)
    if step is None:
        step = max(1, (n2 - n1 + 1) // 30)
    
    best_tm = -1
    best_offset = 0
    
    for offset in range(0, n2 - n1 + 1, step):
        subset = coords2[offset:offset+n1]
        R, t = kabsch_rotation(coords1, subset)
        c1_rot = (R @ coords1.T).T + t
        dists = np.sqrt(np.sum((c1_rot - subset) ** 2, axis=1))
        d0 = compute_d0(n1)
        tm = float(np.mean(1.0 / (1.0 + (dists / d0) ** 2)))
        if tm > best_tm:
            best_tm = tm
            best_offset = offset
    
    # Local refinement
    for offset in range(max(0, best_offset - step), min(n2 - n1 + 1, best_offset + step + 1)):
        subset = coords2[offset:offset+n1]
        R, t = kabsch_rotation(coords1, subset)
        c1_rot = (R @ coords1.T).T + t
        dists = np.sqrt(np.sum((c1_rot - subset) ** 2, axis=1))
        d0 = compute_d0(n1)
        tm = float(np.mean(1.0 / (1.0 + (dists / d0) ** 2)))
        if tm > best_tm:
            best_tm = tm
            best_offset = offset
    
    a1 = list(range(n1))
    a2 = list(range(best_offset, best_offset + n1))
    
    if swapped:
        return a2, a1
    return a1, a2

def dp_alignment(score_matrix, gap_open=-0.6):
    """Fast DP alignment."""
    n1, n2 = score_matrix.shape
    dp = np.full((n1 + 1, n2 + 1), -np.inf)
    trace = np.zeros((n1 + 1, n2 + 1), dtype=np.int8)
    
    dp[0, 0] = 0
    for i in range(1, n1 + 1):
        dp[i, 0] = i * gap_open
        trace[i, 0] = 2
    for j in range(1, n2 + 1):
        dp[0, j] = j * gap_open
        trace[0, j] = 3
    
    for i in range(1, n1 + 1):
        dp_i = dp[i]
        dp_im1 = dp[i-1]
        score_i = score_matrix[i-1]
        trace_i = trace[i]
        
        for j in range(1, n2 + 1):
            diag = dp_im1[j-1] + score_i[j-1]
            up = dp_im1[j] + gap_open
            left = dp_i[j-1] + gap_open
            
            if diag >= up and diag >= left:
                dp_i[j] = diag
                trace_i[j] = 1
            elif up >= left:
                dp_i[j] = up
                trace_i[j] = 2
            else:
                dp_i[j] = left
                trace_i[j] = 3
    
    i, j = n1, n2
    a1, a2 = [], []
    while i > 0 or j > 0:
        t = trace[i, j]
        if t == 1:
            a1.append(i - 1)
            a2.append(j - 1)
            i -= 1
            j -= 1
        elif t == 2:
            i -= 1
        else:
            j -= 1
    
    a1.reverse()
    a2.reverse()
    return a1, a2

def tm_align_pairwise(coords1, coords2, max_iter=3):
    """TM-align with limited iterations for speed - designed for large proteins."""
    L1, L2 = len(coords1), len(coords2)
    if L1 < 5 or L2 < 5:
        return None
    
    d0_min = compute_d0(min(L1, L2))
    
    # Initial alignment: gapless
    a1, a2 = gapless_align(coords1, coords2)
    
    # Iterative refinement
    for iteration in range(max_iter):
        if len(a1) < 3:
            break
        
        c1 = coords1[a1]
        c2 = coords2[a2]
        R, t = kabsch_rotation(c1, c2)
        c1_rot = (R @ coords1.T).T + t
        
        # Build score matrix using chunked computation
        d0_sq = d0_min ** 2
        chunk_size = 200
        score = np.zeros((L1, L2))
        
        for i_start in range(0, L1, chunk_size):
            i_end = min(i_start + chunk_size, L1)
            c1_chunk = c1_rot[i_start:i_end]
            diff = c1_chunk[:, np.newaxis, :] - coords2[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            score[i_start:i_end, :] = 1.0 / (1.0 + dist_sq / d0_sq)
        
        new_a1, new_a2 = dp_alignment(score, gap_open=-0.6)
        
        if len(new_a1) < 3:
            break
        
        if new_a1 == a1 and new_a2 == a2:
            break
        
        a1, a2 = new_a1, new_a2
    
    if len(a1) >= 3:
        c1 = coords1[a1]
        c2 = coords2[a2]
        tm1, tm2, rmsd, R, t, dists = tm_score_fast(c1, c2)
        avg_tm = (tm1 + tm2) / 2.0
        
        return {
            'aligned_indices_1': a1,
            'aligned_indices_2': a2,
            'n_aligned': len(a1),
            'tm_score_1': tm1, 'tm_score_2': tm2,
            'tm_score_avg': avg_tm, 'rmsd': rmsd,
            'rotation_matrix': R.tolist(),
            'translation_vector': t.tolist(),
            'coverage_1': len(a1) / L1,
            'coverage_2': len(a2) / L2,
            'length_1': L1, 'length_2': L2,
        }
    
    return None


def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    
    print("=" * 60)
    print("STRUCTURAL ALIGNMENT: 7xg4 vs 6n40")
    print("=" * 60)
    
    print("\n[1] Parsing PDB files...")
    chains_7xg4 = parse_pdb('data/7xg4.pdb')
    chains_6n40 = parse_pdb('data/6n40.pdb')
    chains_all_7xg4, chain_meta_7xg4 = parse_pdb_all_atoms('data/7xg4.pdb')
    
    print(f"\n7xg4: {len(chains_all_7xg4)} chains")
    protein_chains = sorted([c for c, m in chain_meta_7xg4.items() if m['type'] == 'protein'])
    nucleic_chains = sorted([c for c, m in chain_meta_7xg4.items() if m['type'] == 'nucleic'])
    print(f"  Protein chains: {protein_chains}")
    print(f"  Nucleic acid chains: {nucleic_chains}")
    for c in sorted(chains_all_7xg4.keys()):
        m = chain_meta_7xg4[c]
        print(f"    {c} ({m['type']}): {m['length']} residues")
    
    print(f"\n6n40: {len(chains_6n40)} chain(s)")
    for c in sorted(chains_6n40.keys()):
        print(f"  Chain {c}: {len(chains_6n40[c])} CA atoms")
    
    # Run pairwise for protein chains
    print("\n[2] Running pairwise alignments (protein chains only)...")
    chain_ids_1 = protein_chains
    chain_ids_2 = sorted(chains_6n40.keys())
    
    all_pairwise = []
    total_pairs = len(chain_ids_1) * len(chain_ids_2)
    pair_count = 0
    
    for cid1 in chain_ids_1:
        for cid2 in chain_ids_2:
            pair_count += 1
            print(f"  [{pair_count}/{total_pairs}] 7xg4:{cid1} vs 6n40:{cid2} ...", end=' ', flush=True)
            result = tm_align_pairwise(chains_7xg4[cid1], chains_6n40[cid2])
            if result:
                all_pairwise.append({'chain_1': cid1, 'chain_2': cid2, **result})
                print(f"TM={result['tm_score_avg']:.4f}, RMSD={result['rmsd']:.2f}A, N={result['n_aligned']}")
            else:
                print("FAILED")
    
    all_pairwise.sort(key=lambda x: x['tm_score_avg'], reverse=True)
    
    # Greedy chain mapping
    used_2 = set()
    chain_mapping = {}
    for pr in all_pairwise:
        if pr['chain_1'] not in chain_mapping and pr['chain_2'] not in used_2:
            chain_mapping[pr['chain_1']] = {k: v for k, v in pr.items() if k not in ['chain_1', 'chain_2']}
            chain_mapping[pr['chain_1']]['target_chain'] = pr['chain_2']
            used_2.add(pr['chain_2'])
    
    unmatched_1 = [c for c in chain_ids_1 if c not in chain_mapping]
    unmatched_2 = [c for c in chain_ids_2 if c not in used_2]
    
    # Summary
    print("\n[3] RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Protein chains in 7xg4: {protein_chains}")
    print(f"  Nucleic acid chains in 7xg4 (not aligned): {nucleic_chains}")
    print(f"  Chain(s) in 6n40: {chain_ids_2}")
    print(f"\n  Greedy chain mapping ({len(chain_mapping)} mapped):")
    
    for c1 in sorted(chain_mapping.keys(), key=lambda x: chain_mapping[x]['tm_score_avg'], reverse=True):
        m = chain_mapping[c1]
        print(f"    7xg4:{c1} -> 6n40:{m['target_chain']}: "
              f"TM-avg={m['tm_score_avg']:.4f}, "
              f"TM1={m['tm_score_1']:.4f}, TM2={m['tm_score_2']:.4f}, "
              f"RMSD={m['rmsd']:.2f}A, "
              f"Aligned={m['n_aligned']}/{m['length_1']} "
              f"(cov1={m['coverage_1']:.1%}, cov2={m['coverage_2']:.1%})")
    
    print(f"\n  Unmatched 7xg4 chains: {unmatched_1}")
    print(f"  Unmatched 6n40 chains: {unmatched_2}")
    
    if chain_mapping:
        avg_tm = float(np.mean([m['tm_score_avg'] for m in chain_mapping.values()]))
        weighted_avg_tm = float(np.average([m['tm_score_avg'] for m in chain_mapping.values()],
                                           weights=[m['length_1'] for m in chain_mapping.values()]))
        avg_rmsd = float(np.mean([m['rmsd'] for m in chain_mapping.values()]))
        print(f"\n  Mean TM-score (chain avg): {avg_tm:.4f}")
        print(f"  Length-weighted mean TM-score: {weighted_avg_tm:.4f}")
        print(f"  Mean RMSD: {avg_rmsd:.2f} A")
    
    print(f"\n  Top 5 pairwise hits (all chains):")
    for i, pr in enumerate(all_pairwise[:5]):
        print(f"    {i+1}. 7xg4:{pr['chain_1']} -> 6n40:{pr['chain_2']}: "
              f"TM={pr['tm_score_avg']:.4f}, RMSD={pr['rmsd']:.2f}A, N={pr['n_aligned']}")
    
    print(f"\n  Bottom 5 pairwise hits:")
    for i, pr in enumerate(all_pairwise[-5:]):
        print(f"    {i+1}. 7xg4:{pr['chain_1']} -> 6n40:{pr['chain_2']}: "
              f"TM={pr['tm_score_avg']:.4f}, RMSD={pr['rmsd']:.2f}A, N={pr['n_aligned']}")
    
    # Statistical significance: TM-score ~0.17 is random
    random_baseline = 0.17
    significant = sum(1 for pr in all_pairwise if pr['tm_score_avg'] > 0.5)
    print(f"\n  Pairs with TM-score > 0.5 (significant similarity): {significant}/{len(all_pairwise)}")
    print(f"  Pairs with TM-score > random baseline (0.17): {sum(1 for pr in all_pairwise if pr['tm_score_avg'] > random_baseline)}/{len(all_pairwise)}")
    
    # Save results
    print("\n[4] Saving outputs...")
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    full_result = {
        'chain_mapping': convert(chain_mapping),
        'all_pairwise': convert(all_pairwise),
        'unmatched_1': unmatched_1,
        'unmatched_2': unmatched_2,
        'protein_chains_7xg4': protein_chains,
        'nucleic_chains_7xg4': nucleic_chains,
        'chain_6n40': chain_ids_2,
        'chain_lengths_7xg4': {c: len(chains_7xg4[c]) for c in chain_ids_1},
        'chain_lengths_6n40': {c: len(chains_6n40[c]) for c in chain_ids_2},
        'summary': {
            'n_mapped_chains': len(chain_mapping),
            'mean_tm_score': float(np.mean([m['tm_score_avg'] for m in chain_mapping.values()])) if chain_mapping else 0,
            'weighted_mean_tm_score': float(np.average([m['tm_score_avg'] for m in chain_mapping.values()],
                                                       weights=[m['length_1'] for m in chain_mapping.values()])) if chain_mapping else 0,
            'mean_rmsd': float(np.mean([m['rmsd'] for m in chain_mapping.values()])) if chain_mapping else float('inf'),
        }
    }
    
    with open('outputs/alignment_results.json', 'w') as f:
        json.dump(full_result, f, indent=2)
    
    # TM-score matrix
    tm_matrix = np.zeros((len(chain_ids_1), len(chain_ids_2)))
    for pr in all_pairwise:
        i = chain_ids_1.index(pr['chain_1'])
        j = chain_ids_2.index(pr['chain_2'])
        tm_matrix[i, j] = pr['tm_score_avg']
    
    np.savetxt('outputs/tm_score_matrix.csv', tm_matrix, delimiter=',', fmt='%.6f')
    with open('outputs/tm_score_matrix_labels.txt', 'w') as f:
        f.write(f"# Rows: 7xg4 chains\n")
        f.write(f"rows: {','.join(chain_ids_1)}\n")
        f.write(f"cols: {','.join(chain_ids_2)}\n")
    
    # Chain details
    chain_details = {}
    for c1, m in chain_mapping.items():
        chain_details[c1] = {
            'target_chain': m['target_chain'],
            'tm_score_1': m['tm_score_1'],
            'tm_score_2': m['tm_score_2'],
            'tm_score_avg': m['tm_score_avg'],
            'rmsd': m['rmsd'],
            'n_aligned': m['n_aligned'],
            'coverage_1': m['coverage_1'],
            'coverage_2': m['coverage_2'],
            'length_1': m['length_1'],
            'length_2': m['length_2'],
            'rotation_matrix': m['rotation_matrix'],
            'translation_vector': m['translation_vector'],
        }
    
    with open('outputs/per_chain_alignments.json', 'w') as f:
        json.dump(chain_details, f, indent=2)
    
    print("  -> outputs/alignment_results.json")
    print("  -> outputs/tm_score_matrix.csv")
    print("  -> outputs/tm_score_matrix_labels.txt")
    print("  -> outputs/per_chain_alignments.json")
    
    return full_result, chains_7xg4, chains_6n40

if __name__ == '__main__':
    result, chains_7xg4, chains_6n40 = main()
    print("\nDone!")
