#!/usr/bin/env python3
"""
XEB Fidelity Estimation for Random Quantum Circuit Sampling
Based on Google's quantum supremacy paper (Arute et al., Nature 2019)

F_XEB = 2^n * <P(x_i)>_i - 1
where P(x_i) = |psi(x_i)|^2 is the ideal probability of measured bitstring x_i.
"""

import json
import os
import re
import glob
import numpy as np
from collections import defaultdict

DATA_ROOT = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_002_20260415_145205/data"
AMPLITUDES_ROOT = os.path.join(DATA_ROOT, "amplitudes")
RESULTS_ROOT = os.path.join(DATA_ROOT, "results")
OUTPUT_ROOT = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_002_20260415_145205/outputs"
REPORT_IMG = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_002_20260415_145205/report/images"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(REPORT_IMG, exist_ok=True)


def parse_filename(fname):
    """Extract N, d, r from filename like N40_d10_r1_XEB_counts.json"""
    base = os.path.basename(fname)
    m = re.match(r'N(\d+)_d(\d+)_r(\d+)_XEB', base)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def load_amplitude_file(path):
    """Load amplitude JSON and return dict of bitstring -> ideal probability |amp|^2"""
    with open(path, 'r') as f:
        data = json.load(f)
    ideal_probs = {}
    for bitstr_key, amp_str in data.items():
        # Parse complex amplitude from string like "(3.978e-07+3.095e-08j)"
        amp_str_clean = amp_str.strip('()')
        amp = complex(amp_str_clean)
        prob = abs(amp) ** 2
        ideal_probs[bitstr_key] = prob
    return ideal_probs


def load_counts_file(path):
    """Load counts JSON and return dict of bitstring -> count"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def compute_xeb_fidelity(ideal_probs, counts, n_qubits):
    """
    Compute linear XEB fidelity.
    F_XEB = 2^n * <P(x_i)>_i - 1
    
    For weighted counts, we compute the counts-weighted average of ideal probabilities.
    """
    D = 2 ** n_qubits  # Hilbert space dimension
    
    total_count = 0
    weighted_prob_sum = 0.0
    
    for bitstr, count in counts.items():
        if bitstr in ideal_probs:
            weighted_prob_sum += count * ideal_probs[bitstr]
            total_count += count
    
    if total_count == 0:
        return 0.0, 0
    
    avg_prob = weighted_prob_sum / total_count
    f_xeb = D * avg_prob - 1.0
    
    return f_xeb, total_count


def compute_xeb_fidelity_with_stderr(ideal_probs, counts, n_qubits, n_bootstrap=200):
    """
    Compute XEB fidelity with bootstrap standard error.
    """
    D = 2 ** n_qubits
    
    # Build list of (bitstring, count) pairs that match
    matched_probs = []
    matched_counts = []
    for bitstr, count in counts.items():
        if bitstr in ideal_probs:
            matched_probs.append(ideal_probs[bitstr])
            matched_counts.append(count)
    
    if len(matched_probs) == 0:
        return 0.0, 0.0, 0
    
    matched_probs = np.array(matched_probs)
    matched_counts = np.array(matched_counts)
    
    # Expand counts into individual samples for bootstrap
    expanded_probs = np.repeat(matched_probs, matched_counts)
    total_samples = len(expanded_probs)
    
    if total_samples < 2:
        avg_prob = np.sum(matched_probs * matched_counts) / np.sum(matched_counts)
        f_xeb = D * avg_prob - 1.0
        return f_xeb, 0.0, total_samples
    
    # Point estimate
    avg_prob = np.mean(expanded_probs)
    f_xeb = D * avg_prob - 1.0
    
    # Bootstrap standard error
    rng = np.random.default_rng(42)
    bootstrap_fidelities = []
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(total_samples, size=total_samples, replace=True)
        boot_avg = np.mean(expanded_probs[sample_idx])
        boot_f = D * boot_avg - 1.0
        bootstrap_fidelities.append(boot_f)
    
    se = np.std(bootstrap_fidelities)
    
    return f_xeb, se, total_samples


def find_matching_files():
    """Find all (amplitude, counts) file pairs across all configurations."""
    pairs = []
    
    # Scan N40_verification for depth scan
    amp_base = os.path.join(AMPLITUDES_ROOT, "N40_verification")
    res_base = os.path.join(RESULTS_ROOT, "N40_verification")
    
    if os.path.exists(amp_base):
        for depth_dir in sorted(os.listdir(amp_base)):
            depth_path = os.path.join(amp_base, depth_dir)
            if not os.path.isdir(depth_path):
                continue
            for amp_file in sorted(glob.glob(os.path.join(depth_path, "*_amplitudes.json"))):
                info = parse_filename(amp_file.replace("_amplitudes.json", "_XEB"))
                if info is None:
                    continue
                N, d, r = info
                # Find matching counts file
                counts_file = amp_file.replace("/amplitudes/", "/results/").replace("_amplitudes.json", "_counts.json")
                if os.path.exists(counts_file):
                    pairs.append((N, d, r, amp_file, counts_file))
    
    # Scan N_scan_depth12 for N scan
    amp_base = os.path.join(AMPLITUDES_ROOT, "N_scan_depth12")
    res_base = os.path.join(RESULTS_ROOT, "N_scan_depth12")
    
    if os.path.exists(amp_base):
        for n_dir in sorted(os.listdir(amp_base)):
            n_path = os.path.join(amp_base, n_dir)
            if not os.path.isdir(n_path):
                continue
            for amp_file in sorted(glob.glob(os.path.join(n_path, "*_amplitudes.json"))):
                info = parse_filename(amp_file.replace("_amplitudes.json", "_XEB"))
                if info is None:
                    continue
                N, d, r = info
                counts_file = amp_file.replace("/amplitudes/", "/results/").replace("_amplitudes.json", "_counts.json")
                if os.path.exists(counts_file):
                    pairs.append((N, d, r, amp_file, counts_file))
    
    return pairs


def main():
    print("Finding matching amplitude/counts file pairs...")
    pairs = find_matching_files()
    print(f"Found {len(pairs)} matching pairs")
    
    # Group by (N, d)
    grouped = defaultdict(list)
    for N, d, r, amp_file, counts_file in pairs:
        grouped[(N, d)].append((r, amp_file, counts_file))
    
    # Compute fidelity for each instance
    results = []
    for (N, d), instances in sorted(grouped.items()):
        print(f"\nProcessing N={N}, d={d} ({len(instances)} instances)")
        fidelities = []
        errors = []
        
        for r, amp_file, counts_file in sorted(instances):
            ideal_probs = load_amplitude_file(amp_file)
            counts = load_counts_file(counts_file)
            
            f_xeb, se, n_samples = compute_xeb_fidelity_with_stderr(ideal_probs, counts, N)
            fidelities.append(f_xeb)
            errors.append(se)
            
            print(f"  r={r:2d}: F_XEB = {f_xeb:.6f} ± {se:.6f} (n_samples={n_samples}, n_matched={len(ideal_probs)})")
            
            results.append({
                'N': N,
                'd': d,
                'r': r,
                'f_xeb': f_xeb,
                'f_xeb_se': se,
                'n_samples': n_samples,
                'n_matched_bitstrings': len(ideal_probs)
            })
        
        mean_f = np.mean(fidelities)
        std_f = np.std(fidelities)
        mean_se = np.mean(errors)
        print(f"  Summary: F_XEB = {mean_f:.6f} ± {std_f:.6f} (across instances), avg SE = {mean_se:.6f}")
    
    # Save results
    output_path = os.path.join(OUTPUT_ROOT, "fidelity_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Compute summary statistics
    summary = {}
    for (N, d), instances in sorted(grouped.items()):
        fidelities = [r['f_xeb'] for r in results if r['N'] == N and r['d'] == d]
        errors = [r['f_xeb_se'] for r in results if r['N'] == N and r['d'] == d]
        summary[f"N{N}_d{d}"] = {
            'N': N,
            'd': d,
            'n_instances': len(fidelities),
            'mean_f_xeb': float(np.mean(fidelities)),
            'std_f_xeb': float(np.std(fidelities)),
            'mean_se': float(np.mean(errors)),
            'fidelities': fidelities
        }
    
    summary_path = os.path.join(OUTPUT_ROOT, "fidelity_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    return results, summary


if __name__ == "__main__":
    main()
