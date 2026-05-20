"""
Compute XEB (Cross-Entropy Benchmarking) fidelity estimates for all available
circuit instances where both experimental counts and ideal amplitudes are provided.

The XEB fidelity is defined as:
    F_XEB = 2^n * <P(x_i)>_i - 1

where n is the number of qubits, P(x_i) = |<x_i|psi>|^2 is the ideal probability
of measured bitstring x_i, and the average is over the observed bitstrings
(weighted by their occurrence counts).
"""

import json
import os
import glob
import numpy as np
from pathlib import Path


def parse_ndr_from_filename(filename):
    """Extract N, d, r, and method from filename like N40_d12_r10_XEB_counts.json"""
    basename = os.path.basename(filename)
    # Format: N{N}_d{d}_r{r}_{method}_counts.json or N{N}_d{d}_r{r}_{method}_amplitudes.json
    parts = basename.replace('.json', '').split('_')
    N = int(parts[0][1:])  # N40 -> 40
    d = int(parts[1][1:])  # d12 -> 12
    r = int(parts[2][1:])  # r10 -> 10
    method = parts[3]  # XEB, MB, Transport, etc.
    return N, d, r, method


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_xeb_fidelity(counts_dict, amplitudes_dict, N):
    """
    Compute XEB fidelity from counts and ideal amplitudes.
    
    Args:
        counts_dict: dict mapping bitstring key -> occurrence count
        amplitudes_dict: dict mapping bitstring key -> complex amplitude string
        N: number of qubits
    
    Returns:
        dict with fidelity, uncertainty, total_counts, matched_keys, etc.
    """
    D = 2 ** N  # Hilbert space dimension
    
    # Match keys
    matched_keys = set(counts_dict.keys()) & set(amplitudes_dict.keys())
    
    if len(matched_keys) == 0:
        return {
            'fidelity': None,
            'uncertainty': None,
            'total_counts': 0,
            'matched_keys': 0,
            'mean_p': None,
            'std_p': None,
        }
    
    # Build matched pairs
    probs = []
    counts = []
    for key in matched_keys:
        amp_str = amplitudes_dict[key]
        # Parse complex number string like "(2.2367691094530178e-07+6.62922439010291e-07j)"
        amp_str = amp_str.strip().strip('()')
        amp = complex(amp_str)
        p = abs(amp) ** 2
        c = counts_dict[key]
        probs.append(p)
        counts.append(c)
    
    probs = np.array(probs)
    counts = np.array(counts)
    total_counts = int(counts.sum())
    
    # Counts-weighted mean probability
    mean_p = np.sum(counts * probs) / total_counts
    
    # Counts-weighted variance of probabilities
    if total_counts > 1:
        # Sample variance (using counts as weights)
        # var = sum(c_i * (p_i - mean_p)^2) / sum(c_i)
        var_p = np.sum(counts * (probs - mean_p) ** 2) / total_counts
        std_p = np.sqrt(var_p)
    else:
        var_p = 0.0
        std_p = 0.0
    
    # XEB fidelity
    fidelity = D * mean_p - 1.0
    
    # Standard error of the mean for weighted sample
    # SE(F_XEB) = D * SE(mean_p) = D * std_p / sqrt(total_counts)
    if total_counts > 1:
        uncertainty = D * std_p / np.sqrt(total_counts)
    else:
        uncertainty = float('inf')
    
    return {
        'fidelity': float(fidelity),
        'uncertainty': float(uncertainty),
        'total_counts': total_counts,
        'matched_keys': len(matched_keys),
        'mean_p': float(mean_p),
        'std_p': float(std_p),
        'D': D,
    }


def find_xeb_pairs():
    """Find all matching (counts, amplitudes) pairs for XEB method."""
    pairs = []
    
    # Search all results directories for XEB counts files
    results_pattern = 'data/results/**/N*_d*_r*_XEB_counts.json'
    for counts_path in glob.glob(results_pattern, recursive=True):
        # Derive corresponding amplitudes path
        # data/results/.../N40_d12_XEB/N40_d12_r10_XEB_counts.json
        # -> data/amplitudes/.../N40_d12_XEB/N40_d12_r10_XEB_amplitudes.json
        rel_path = os.path.relpath(counts_path, 'data/results')
        amplitudes_path = os.path.join('data/amplitudes', rel_path)
        amplitudes_path = amplitudes_path.replace('_counts.json', '_amplitudes.json')
        
        if os.path.exists(amplitudes_path):
            N, d, r, method = parse_ndr_from_filename(counts_path)
            pairs.append({
                'N': N,
                'd': d,
                'r': r,
                'method': method,
                'counts_path': counts_path,
                'amplitudes_path': amplitudes_path,
            })
    
    return pairs


def main():
    pairs = find_xeb_pairs()
    print(f"Found {len(pairs)} matching XEB (counts, amplitudes) pairs")
    
    results = []
    for pair in pairs:
        counts_dict = load_json(pair['counts_path'])
        amplitudes_dict = load_json(pair['amplitudes_path'])
        
        fidelity_info = compute_xeb_fidelity(
            counts_dict, amplitudes_dict, pair['N']
        )
        
        result = {
            'N': pair['N'],
            'd': pair['d'],
            'r': pair['r'],
            'method': pair['method'],
            **fidelity_info,
            'counts_path': pair['counts_path'],
            'amplitudes_path': pair['amplitudes_path'],
        }
        results.append(result)
        
        print(f"N={pair['N']:2d}, d={pair['d']:2d}, r={pair['r']:2d}: "
              f"F_XEB={fidelity_info['fidelity']:.6f} ± {fidelity_info['uncertainty']:.6f}, "
              f"counts={fidelity_info['total_counts']}, matched={fidelity_info['matched_keys']}")
    
    # Save results
    with open('outputs/xeb_fidelities.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved {len(results)} results to outputs/xeb_fidelities.json")
    
    # Summary statistics
    print("\n=== Summary by (N, d) ===")
    from collections import defaultdict
    by_nd = defaultdict(list)
    for r in results:
        by_nd[(r['N'], r['d'])].append(r['fidelity'])
    
    for (N, d), fids in sorted(by_nd.items()):
        mean_fid = np.mean(fids)
        std_fid = np.std(fids, ddof=1)
        print(f"N={N:2d}, d={d:2d}: mean_F_XEB={mean_fid:.6f} ± {std_fid:.6f} (n={len(fids)})")


if __name__ == '__main__':
    main()
