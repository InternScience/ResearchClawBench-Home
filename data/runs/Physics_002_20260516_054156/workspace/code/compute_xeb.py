#!/usr/bin/env python3
"""
Compute XEB fidelity estimates for RCS data on N=40 qubits.
"""

import json
import os
import glob
import numpy as np
from collections import defaultdict

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_xeb_fidelity(counts_path, amps_path):
    counts = load_json(counts_path)
    amps = load_json(amps_path)
    
    # Convert amplitudes to probabilities
    ideal_probs = {}
    for bs, amp_str in amps.items():
        # amp is complex string like "(a+bj)"
        amp = complex(amp_str)
        ideal_probs[bs] = abs(amp)**2
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0, 0.0
    
    # XEB: average over measured samples of (D * p(s) - 1) where D=2^N but here use subset normalization?
    # For verification subset, use the standard formula adapted to the matched probabilities
    # Typically F_XEB = (mean( D * p(s) for sampled s ) - 1) / (D - 1) but since D huge, approx mean(D p(s)) -1
    # But for small subset verification, it's the correlation between observed freq and ideal p
    
    # Standard implementation for XEB with partial probs:
    # sum over observed s: freq(s) * (D * p(s) - 1) / (D-1) approx
    # But since subset, we use the provided probs directly and normalize appropriately.
    
    # Common practical formula for XEB fidelity estimate:
    # F = ( sum_{samples} p_ideal(s) / num_samples - 1/(D) ) / (1 - 1/D) but D huge
    # Simplified: F_XEB ≈ mean( D * p_ideal(s) for s in samples ) - 1
    
    D = 2 ** 40  # full space, but we use approximation since p small
    
    xeb_values = []
    for bs, cnt in counts.items():
        if bs in ideal_probs:
            p = ideal_probs[bs]
            # Each occurrence contributes
            for _ in range(cnt):
                xeb_values.append(D * p - 1)
    
    if len(xeb_values) == 0:
        return 0.0, 0.0
    
    mean_xeb = np.mean(xeb_values)
    # Normalize: since D huge, F ≈ mean_xeb / (D * avg_p or something) wait:
    # Actually the unbiased estimator is F = mean(D*p(s) -1 )   since for Porter-Thomas, E[p]=1/D, var leads to F= mean(Dp-1)
    # For full, F_XEB = mean(D p(s) -1 )
    # But to get fidelity between 0 and 1, it's already the estimate (for chaotic, max F=1 when perfect)
    
    # In practice for these experiments, the XEB fidelity is reported as the mean value normalized.
    # Here we take F = mean(D * p - 1) but since D large and p~1/D, F around 0-1? No:
    # For ideal, p~ exp(-D p) Porter Thomas, E[D p(s)] =1 for sampled from ideal? Wait.
    
    # Correct standard XEB:
    # XEB = D * sum_s p_ideal(s) * p_exp(s) - 1
    # For samples, estimator is (D * mean p_ideal(sampled) - 1)
    # And for ideal sampling, E[XEB] =1, for uniform =0
    
    fidelity = mean_xeb   # since D*p mean gives the scale, F = mean(D p(s)) -1 
    # For perfect, mean(D p) =1 +1/D ~1, F~0? Wait let's correct.
    
    # Standard:
    # The linear XEB fidelity is F_XEB = (D * <p_ideal>_{samples} - 1) / (D - 1) ≈ <D p> -1
    # For perfect sampling from p_ideal, <p> = sum p^2 ~ 2/D , so D*<p> ~2, F~1
    # For uniform, <p>=1/D, F=0
    # Yes: F = mean_over_samples ( D * p_ideal(s) ) - 1
    
    fidelity = np.mean([D * ideal_probs.get(bs, 0.0) for bs in counts for _ in range(cnt)]) - 1
    std_err = np.std([D * ideal_probs.get(bs, 0.0) for bs in counts for _ in range(cnt)]) / np.sqrt(total_shots)
    
    return fidelity, std_err

def main():
    base_dir = 'data'
    results_dir = os.path.join(base_dir, 'results/N40_verification')
    amps_dir = os.path.join(base_dir, 'amplitudes/N40_verification')
    
    xeb_dirs = sorted(glob.glob(os.path.join(results_dir, 'N40_d*_XEB')))
    
    all_results = []
    
    for xeb_dir in xeb_dirs:
        d_str = os.path.basename(xeb_dir).split('_')[1]  # d10 etc
        d = int(d_str[1:])
        counts_files = sorted(glob.glob(os.path.join(xeb_dir, '*_counts.json')))
        
        for cf in counts_files:
            fname = os.path.basename(cf)
            r_part = fname.split('_r')[1].split('_')[0]
            r = int(r_part)
            af = cf.replace('results', 'amplitudes').replace('_counts.json', '_amplitudes.json')
            
            if not os.path.exists(af):
                continue
            
            fid, err = compute_xeb_fidelity(cf, af)
            all_results.append({
                'N': 40,
                'd': d,
                'r': r,
                'fidelity': fid,
                'std_err': err
            })
            print(f"N=40, d={d}, r={r}: F_XEB = {fid:.4f} ± {err:.4f}")
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/xeb_fidelities.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Saved outputs/xeb_fidelities.json")

if __name__ == '__main__':
    main()