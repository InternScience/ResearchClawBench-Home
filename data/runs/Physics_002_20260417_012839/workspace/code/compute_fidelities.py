#!/usr/bin/env python3
"""
XEB Fidelity Estimation for Random Quantum Circuit Sampling (RCS)

Computes linear cross-entropy benchmarking (XEB) fidelity:
  F_XEB = 2^N * <P(x_i)>_counts - 1

where P(x_i) = |amplitude_i|^2 is the ideal probability of bitstring x_i,
and the average is weighted by measurement counts.
"""

import json
import os
import glob
import numpy as np
import ast
import re
from collections import defaultdict

BASE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_002_20260417_012839"
DATA = os.path.join(BASE, "data")
OUTPUTS = os.path.join(BASE, "outputs")

def parse_bitstring_key(key_str):
    """Parse a tuple-string key like '(0, 1, 0, ...)' into a tuple."""
    return tuple(int(x) for x in re.findall(r'\d+', key_str))

def load_counts(filepath):
    """Load counts JSON: {bitstring_tuple_str: count}"""
    with open(filepath) as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        bits = parse_bitstring_key(k)
        result[bits] = v
    return result

def load_amplitudes(filepath):
    """Load amplitudes JSON: {bitstring_tuple_str: complex_str}"""
    with open(filepath) as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        bits = parse_bitstring_key(k)
        # Parse complex number from string
        amp = complex(v)
        prob = abs(amp) ** 2
        result[bits] = prob
    return result

def compute_xeb_fidelity(counts, ideal_probs, n_qubits):
    """
    Compute XEB fidelity:
    F_XEB = 2^N * sum(count_i * P(x_i)) / sum(count_i) - 1
    """
    total_counts = 0
    weighted_prob_sum = 0.0
    matched = 0
    
    for bitstring, count in counts.items():
        if bitstring in ideal_probs:
            weighted_prob_sum += count * ideal_probs[bitstring]
            total_counts += count
            matched += 1
    
    if total_counts == 0:
        return None, 0
    
    mean_prob = weighted_prob_sum / total_counts
    fxeb = (2 ** n_qubits) * mean_prob - 1.0
    return fxeb, matched

def compute_mb_probability(counts_filepath, ideal_bitstring_filepath):
    """
    Compute matched bitstring (MB) regression probability.
    This is the fraction of shots that match the ideal most-probable bitstring.
    """
    with open(counts_filepath) as f:
        counts_data = json.load(f)
    with open(ideal_bitstring_filepath) as f:
        ideal_bits = json.load(f)
    
    ideal_tuple = tuple(ideal_bits)
    ideal_key = str(ideal_tuple)
    
    # Parse all counts
    total_shots = 0
    matched_shots = 0
    for k, v in counts_data.items():
        bits = parse_bitstring_key(k)
        total_shots += v
        if bits == ideal_tuple:
            matched_shots += v
    
    if total_shots == 0:
        return None, 0
    
    return matched_shots / total_shots, total_shots

def compute_transport_1qrb_fidelity(counts_filepath, ideal_bitstring_filepath):
    """
    Compute Transport/1QRB fidelity.
    For transport experiments, we measure how well the qubit state is preserved.
    The fidelity is computed as the probability of measuring the ideal output bitstring.
    """
    with open(counts_filepath) as f:
        counts_data = json.load(f)
    with open(ideal_bitstring_filepath) as f:
        ideal_bits = json.load(f)
    
    ideal_tuple = tuple(ideal_bits)
    
    # Count total shots and ideal matches
    total_shots = 0
    ideal_count = 0
    
    # Also compute Hamming distance distribution
    hamming_distances = []
    
    for k, v in counts_data.items():
        bits = parse_bitstring_key(k)
        total_shots += v
        hd = sum(a != b for a, b in zip(bits, ideal_tuple))
        hamming_distances.extend([hd] * v)
        if bits == ideal_tuple:
            ideal_count += v
    
    if total_shots == 0:
        return None, 0, None
    
    # Exact match probability
    exact_prob = ideal_count / total_shots
    
    # Average Hamming distance (useful for estimating per-qubit error)
    avg_hd = np.mean(hamming_distances) if hamming_distances else None
    
    return exact_prob, total_shots, avg_hd

# ============================================================
# 1. XEB Fidelity Computation
# ============================================================
print("=" * 60)
print("Computing XEB Fidelities")
print("=" * 60)

xeb_results = defaultdict(list)

# Process N40_verification
for depth in [8, 10, 12, 14, 16, 18, 20]:
    amp_dir = os.path.join(DATA, f"amplitudes/N40_verification/N40_d{depth}_XEB")
    cnt_dir = os.path.join(DATA, f"results/N40_verification/N40_d{depth}_XEB")
    
    if not os.path.exists(amp_dir) or not os.path.exists(cnt_dir):
        continue
    
    for r in range(1, 51):
        amp_file = os.path.join(amp_dir, f"N40_d{depth}_r{r}_XEB_amplitudes.json")
        cnt_file = os.path.join(cnt_dir, f"N40_d{depth}_r{r}_XEB_counts.json")
        
        if not os.path.exists(amp_file) or not os.path.exists(cnt_file):
            continue
        
        counts = load_counts(cnt_file)
        ideal_probs = load_amplitudes(amp_file)
        fxeb, matched = compute_xeb_fidelity(counts, ideal_probs, 40)
        
        if fxeb is not None:
            xeb_results[("N40_verification", 40, depth)].append({
                'r': r, 'fxeb': fxeb, 'matched': matched
            })

# Process N_scan_depth12 (only N=16,24,32,40 have amplitudes)
for N in [16, 24, 32, 40]:
    amp_dir = os.path.join(DATA, f"amplitudes/N_scan_depth12/N{N}_d12_XEB")
    cnt_dir = os.path.join(DATA, f"results/N_scan_depth12/N{N}_d12_XEB")
    
    if not os.path.exists(amp_dir) or not os.path.exists(cnt_dir):
        continue
    
    for r in range(1, 51):
        amp_file = os.path.join(amp_dir, f"N{N}_d12_r{r}_XEB_amplitudes.json")
        cnt_file = os.path.join(cnt_dir, f"N{N}_d12_r{r}_XEB_counts.json")
        
        if not os.path.exists(amp_file) or not os.path.exists(cnt_file):
            continue
        
        counts = load_counts(cnt_file)
        ideal_probs = load_amplitudes(amp_file)
        fxeb, matched = compute_xeb_fidelity(counts, ideal_probs, N)
        
        if fxeb is not None:
            xeb_results[("N_scan_depth12", N, 12)].append({
                'r': r, 'fxeb': fxeb, 'matched': matched
            })

# Print XEB summary
print("\nXEB Fidelity Summary:")
print(f"{'Dataset':<20} {'N':>4} {'d':>4} {'n_inst':>7} {'mean_FXEB':>12} {'std_FXEB':>12} {'sem_FXEB':>12}")
print("-" * 80)

xeb_summary = {}
for key in sorted(xeb_results.keys()):
    dataset, N, d = key
    fxebs = [r['fxeb'] for r in xeb_results[key]]
    mean_f = np.mean(fxebs)
    std_f = np.std(fxebs, ddof=1)
    sem_f = std_f / np.sqrt(len(fxebs))
    n_inst = len(fxebs)
    print(f"{dataset:<20} {N:>4} {d:>4} {n_inst:>7} {mean_f:>12.6f} {std_f:>12.6f} {sem_f:>12.6f}")
    xeb_summary[f"{dataset}_N{N}_d{d}"] = {
        'dataset': dataset, 'N': N, 'd': d,
        'n_instances': n_inst,
        'mean_fxeb': float(mean_f),
        'std_fxeb': float(std_f),
        'sem_fxeb': float(sem_f),
        'per_instance': [{'r': r['r'], 'fxeb': float(r['fxeb']), 'matched': r['matched']} for r in xeb_results[key]]
    }

# Save XEB results
with open(os.path.join(OUTPUTS, "xeb_fidelity_results.json"), 'w') as f:
    json.dump(xeb_summary, f, indent=2)

print(f"\nSaved XEB results to {OUTPUTS}/xeb_fidelity_results.json")

# ============================================================
# 2. MB Regression Probability Computation
# ============================================================
print("\n" + "=" * 60)
print("Computing MB Regression Probabilities")
print("=" * 60)

mb_results = defaultdict(list)

# N40_verification
for depth in [8, 10, 12, 14, 16, 18, 20]:
    mb_dir = os.path.join(DATA, f"results/N40_verification/N40_d{depth}_MB")
    if not os.path.exists(mb_dir):
        continue
    for r in range(1, 51):
        cnt_file = os.path.join(mb_dir, f"N40_d{depth}_r{r}_MB_counts.json")
        ideal_file = os.path.join(mb_dir, f"N40_d{depth}_r{r}_MB_ideal_bitstring.json")
        if not os.path.exists(cnt_file) or not os.path.exists(ideal_file):
            continue
        prob, total = compute_mb_probability(cnt_file, ideal_file)
        if prob is not None:
            mb_results[("N40_verification", 40, depth)].append({
                'r': r, 'prob': prob, 'total_shots': total
            })

# N56_depths
for depth in [8, 10, 12, 14, 16, 18, 20, 24]:
    mb_dir = os.path.join(DATA, f"results/N56_depths/N56_d{depth}_MB")
    if not os.path.exists(mb_dir):
        continue
    for r in range(1, 51):
        cnt_file = os.path.join(mb_dir, f"N56_d{depth}_r{r}_MB_counts.json")
        ideal_file = os.path.join(mb_dir, f"N56_d{depth}_r{r}_MB_ideal_bitstring.json")
        if not os.path.exists(cnt_file) or not os.path.exists(ideal_file):
            continue
        prob, total = compute_mb_probability(cnt_file, ideal_file)
        if prob is not None:
            mb_results[("N56_depths", 56, depth)].append({
                'r': r, 'prob': prob, 'total_shots': total
            })

# N_scan_depth12
for N in [16, 24, 32, 40, 48, 56]:
    mb_dir = os.path.join(DATA, f"results/N_scan_depth12/N{N}_d12_MB")
    if not os.path.exists(mb_dir):
        continue
    for r in range(1, 51):
        cnt_file = os.path.join(mb_dir, f"N{N}_d12_r{r}_MB_counts.json")
        ideal_file = os.path.join(mb_dir, f"N{N}_d12_r{r}_MB_ideal_bitstring.json")
        if not os.path.exists(cnt_file) or not os.path.exists(ideal_file):
            continue
        prob, total = compute_mb_probability(cnt_file, ideal_file)
        if prob is not None:
            mb_results[("N_scan_depth12", N, 12)].append({
                'r': r, 'prob': prob, 'total_shots': total
            })

# Print MB summary
print("\nMB Regression Probability Summary:")
print(f"{'Dataset':<20} {'N':>4} {'d':>4} {'n_inst':>7} {'mean_prob':>12} {'std_prob':>12} {'sem_prob':>12}")
print("-" * 80)

mb_summary = {}
for key in sorted(mb_results.keys()):
    dataset, N, d = key
    probs = [r['prob'] for r in mb_results[key]]
    mean_p = np.mean(probs)
    std_p = np.std(probs, ddof=1)
    sem_p = std_p / np.sqrt(len(probs))
    n_inst = len(probs)
    print(f"{dataset:<20} {N:>4} {d:>4} {n_inst:>7} {mean_p:>12.6f} {std_p:>12.6f} {sem_p:>12.6f}")
    mb_summary[f"{dataset}_N{N}_d{d}"] = {
        'dataset': dataset, 'N': N, 'd': d,
        'n_instances': n_inst,
        'mean_prob': float(mean_p),
        'std_prob': float(std_p),
        'sem_prob': float(sem_p),
        'per_instance': [{'r': r['r'], 'prob': float(r['prob']), 'total_shots': r['total_shots']} for r in mb_results[key]]
    }

with open(os.path.join(OUTPUTS, "mb_probability_results.json"), 'w') as f:
    json.dump(mb_summary, f, indent=2)

print(f"\nSaved MB results to {OUTPUTS}/mb_probability_results.json")

# ============================================================
# 3. Transport/1QRB Fidelity Computation
# ============================================================
print("\n" + "=" * 60)
print("Computing Transport/1QRB Fidelities")
print("=" * 60)

transport_results = defaultdict(list)

# N40_verification Transport
for depth in [4, 16, 32, 48, 64, 96]:
    t_dir = os.path.join(DATA, f"results/N40_verification/N40_d{depth}_Transport_1QRB")
    if not os.path.exists(t_dir):
        continue
    for r in range(1, 11):
        cnt_file = os.path.join(t_dir, f"N40_d{depth}_r{r}_Transport_1QRB_counts.json")
        ideal_file = os.path.join(t_dir, f"N40_d{depth}_r{r}_Transport_1QRB_ideal_bitstring.json")
        if not os.path.exists(cnt_file) or not os.path.exists(ideal_file):
            continue
        prob, total, avg_hd = compute_transport_1qrb_fidelity(cnt_file, ideal_file)
        if prob is not None:
            transport_results[("N40_verification", 40, depth)].append({
                'r': r, 'prob': prob, 'total_shots': total, 'avg_hamming': float(avg_hd) if avg_hd else None
            })

# N56_depths Transport
for depth in [4, 16, 32, 48, 64, 96]:
    t_dir = os.path.join(DATA, f"results/N56_depths/N56_d{depth}_Transport_1QRB")
    if not os.path.exists(t_dir):
        continue
    for r in range(1, 11):
        cnt_file = os.path.join(t_dir, f"N56_d{depth}_r{r}_Transport_1QRB_counts.json")
        ideal_file = os.path.join(t_dir, f"N56_d{depth}_r{r}_Transport_1QRB_ideal_bitstring.json")
        if not os.path.exists(cnt_file) or not os.path.exists(ideal_file):
            continue
        prob, total, avg_hd = compute_transport_1qrb_fidelity(cnt_file, ideal_file)
        if prob is not None:
            transport_results[("N56_depths", 56, depth)].append({
                'r': r, 'prob': prob, 'total_shots': total, 'avg_hamming': float(avg_hd) if avg_hd else None
            })

# N_scan_depth12 Transport
for N in [16, 24, 32, 40, 48, 56]:
    for depth in [4, 16, 32, 48, 64, 96]:
        t_dir = os.path.join(DATA, f"results/N_scan_depth12/N{N}_d{depth}_Transport_1QRB")
        if not os.path.exists(t_dir):
            continue
        for r in range(1, 11):
            cnt_file = os.path.join(t_dir, f"N{N}_d{depth}_r{r}_Transport_1QRB_counts.json")
            ideal_file = os.path.join(t_dir, f"N{N}_d{depth}_r{r}_Transport_1QRB_ideal_bitstring.json")
            if not os.path.exists(cnt_file) or not os.path.exists(ideal_file):
                continue
            prob, total, avg_hd = compute_transport_1qrb_fidelity(cnt_file, ideal_file)
            if prob is not None:
                transport_results[("N_scan_depth12", N, depth)].append({
                    'r': r, 'prob': prob, 'total_shots': total, 'avg_hamming': float(avg_hd) if avg_hd else None
                })

# Print Transport summary
print("\nTransport/1QRB Summary:")
print(f"{'Dataset':<20} {'N':>4} {'d':>4} {'n_inst':>7} {'mean_prob':>12} {'std_prob':>12} {'avg_hd':>10}")
print("-" * 80)

transport_summary = {}
for key in sorted(transport_results.keys()):
    dataset, N, d = key
    probs = [r['prob'] for r in transport_results[key]]
    hds = [r['avg_hamming'] for r in transport_results[key] if r['avg_hamming'] is not None]
    mean_p = np.mean(probs)
    std_p = np.std(probs, ddof=1) if len(probs) > 1 else 0
    avg_hd = np.mean(hds) if hds else None
    n_inst = len(probs)
    print(f"{dataset:<20} {N:>4} {d:>4} {n_inst:>7} {mean_p:>12.6f} {std_p:>12.6f} {avg_hd:>10.4f}" if avg_hd else f"{dataset:<20} {N:>4} {d:>4} {n_inst:>7} {mean_p:>12.6f} {std_p:>12.6f} {'N/A':>10}")
    transport_summary[f"{dataset}_N{N}_d{d}"] = {
        'dataset': dataset, 'N': N, 'd': d,
        'n_instances': n_inst,
        'mean_prob': float(mean_p),
        'std_prob': float(std_p),
        'avg_hamming_distance': float(avg_hd) if avg_hd is not None else None,
        'per_instance': transport_results[key]
    }

with open(os.path.join(OUTPUTS, "transport_1qrb_results.json"), 'w') as f:
    json.dump(transport_summary, f, indent=2)

print(f"\nSaved Transport results to {OUTPUTS}/transport_1qrb_results.json")
print("\n=== All computations complete ===")
