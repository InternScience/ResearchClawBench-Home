"""
Fidelity Estimation for Random Quantum Circuit Sampling (RCS)
=============================================================

Implements XEB (Cross-Entropy Benchmarking) fidelity estimation and
MB (Measurement-Based) regression probability for RCS experiments.

Key formulas:
- XEB Fidelity: F_XEB = 2^n * <P_ideal(x_i)> - 1
  where P_ideal(x_i) = |amplitude(x_i)|^2 and average is over measured bitstrings
- MB Success Probability: P_heavy = fraction of measured bitstrings that are "heavy"

Reference: Arute et al., "Quantum supremacy using a programmable superconducting processor"
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def parse_complex(s):
    """Parse a complex number from string representation."""
    return complex(s.strip())


def load_xeb_data(results_dir, amplitudes_dir, N, d, max_instances=None):
    """
    Load XEB counts and amplitudes for a given N and d.
    
    Returns:
        list of dicts with keys: 'r', 'counts', 'amplitudes', 'ideal_probs', 'total_shots'
    """
    counts_dir = Path(results_dir) / f"N{N}_verification" / f"N{N}_d{d}_XEB" if N == 40 else \
                 Path(results_dir).parent / "results" / f"N_scan_depth12" / f"N{N}_d{d}_XEB" if N in [16,24,32,48,56] and d == 12 else \
                 Path(results_dir).parent / "results" / f"N56_depths" / f"N{N}_d{d}_XEB" if N == 56 else None
    
    if N == 40:
        counts_path = Path(results_dir) / "N40_verification" / f"N40_d{d}_XEB"
        amps_path = Path(amplitudes_dir) / "N40_verification" / f"N40_d{d}_XEB"
    elif N == 56:
        counts_path = Path(results_dir) / "N56_depths" / f"N56_d{d}_XEB"
        amps_path = None  # No amplitudes for N56
    elif d == 12:
        counts_path = Path(results_dir) / "N_scan_depth12" / f"N{N}_d{d}_XEB"
        if N <= 40:
            amps_path = Path(amplitudes_dir) / "N_scan_depth12" / f"N{N}_d{d}_XEB"
        else:
            amps_path = None
    else:
        return []
    
    if not counts_path.exists():
        return []
    
    # Find all instance files
    count_files = sorted(counts_path.glob("*_XEB_counts.json"))
    if max_instances:
        count_files = count_files[:max_instances]
    
    results = []
    for cf in count_files:
        # Extract instance index r
        fname = cf.name
        # Parse r from filename like N40_d10_r1_XEB_counts.json
        parts = fname.replace("_XEB_counts.json", "").split("_")
        r = int([p for p in parts if p.startswith("r")][0][1:])
        
        with open(cf) as f:
            counts = json.load(f)
        
        # Load amplitudes if available
        if amps_path and amps_path.exists():
            amp_file = amps_path / f"N{N}_d{d}_r{r}_XEB_amplitudes.json"
            if amp_file.exists():
                with open(amp_file) as f:
                    amplitudes = json.load(f)
                
                # Convert amplitudes to ideal probabilities
                ideal_probs = {}
                for bitstring, amp_str in amplitudes.items():
                    amp = parse_complex(amp_str)
                    ideal_probs[bitstring] = abs(amp) ** 2
                
                total_shots = sum(counts.values())
                results.append({
                    'r': r,
                    'counts': counts,
                    'ideal_probs': ideal_probs,
                    'total_shots': total_shots
                })
            else:
                results.append({
                    'r': r,
                    'counts': counts,
                    'ideal_probs': None,
                    'total_shots': sum(counts.values())
                })
        else:
            results.append({
                'r': r,
                'counts': counts,
                'ideal_probs': None,
                'total_shots': sum(counts.values())
            })
    
    return results


def load_mb_data(results_dir, N, d, max_instances=None):
    """
    Load MB counts and ideal bitstrings for a given N and d.
    
    Returns:
        list of dicts with keys: 'r', 'counts', 'ideal_bitstring', 'total_shots'
    """
    if N == 40:
        mb_path = Path(results_dir) / "N40_verification" / f"N40_d{d}_MB"
    elif N == 56:
        mb_path = Path(results_dir) / "N56_depths" / f"N56_d{d}_MB"
    elif d == 12:
        mb_path = Path(results_dir) / "N_scan_depth12" / f"N{N}_d{d}_MB"
    else:
        return []
    
    if not mb_path.exists():
        return []
    
    count_files = sorted(mb_path.glob("*_MB_counts.json"))
    if max_instances:
        count_files = count_files[:max_instances]
    
    results = []
    for cf in count_files:
        fname = cf.name
        parts = fname.replace("_MB_counts.json", "").split("_")
        r = int([p for p in parts if p.startswith("r")][0][1:])
        
        ideal_file = cf.parent / f"N{N}_d{d}_r{r}_MB_ideal_bitstring.json"
        if not ideal_file.exists():
            continue
            
        with open(cf) as f:
            counts = json.load(f)
        with open(ideal_file) as f:
            ideal_bitstring = json.load(f)
        
        ideal_key = str(tuple(ideal_bitstring))
        ideal_count = counts.get(ideal_key, 0)
        total_shots = sum(counts.values())
        
        results.append({
            'r': r,
            'counts': counts,
            'ideal_bitstring': ideal_bitstring,
            'ideal_count': ideal_count,
            'total_shots': total_shots
        })
    
    return results


def compute_xeb_fidelity(xeb_data, n):
    """
    Compute XEB fidelity for a single instance.
    
    F_XEB = 2^n * sum(counts_i * P_ideal(x_i)) / sum(counts_i) - 1
    
    This is the linear XEB fidelity estimator.
    """
    if xeb_data['ideal_probs'] is None:
        return None, None
    
    n_qubits = n
    N = 2 ** n_qubits  # Hilbert space dimension
    
    counts = xeb_data['counts']
    ideal_probs = xeb_data['ideal_probs']
    total_shots = xeb_data['total_shots']
    
    # Compute weighted sum of ideal probabilities
    weighted_sum = 0.0
    for bitstring, count in counts.items():
        if bitstring in ideal_probs:
            weighted_sum += count * ideal_probs[bitstring]
    
    # XEB fidelity
    F_xeb = N * weighted_sum / total_shots - 1
    
    # Uncertainty estimate
    # Using bootstrap: std of individual estimates
    individual_estimates = []
    for bitstring, count in counts.items():
        if bitstring in ideal_probs:
            for _ in range(count):
                individual_estimates.append(N * ideal_probs[bitstring])
    
    if len(individual_estimates) > 1:
        std_individual = np.std(individual_estimates)
        std_f = std_individual / np.sqrt(len(individual_estimates))
    else:
        std_f = 0.0
    
    return F_xeb, std_f


def compute_mb_success_probability(mb_data):
    """
    Compute MB success probability (probability of observing the ideal bitstring).
    
    P_success = count(ideal) / total_shots
    """
    return mb_data['ideal_count'] / mb_data['total_shots']


def compute_heavy_output_probability(mb_data, ideal_probs=None):
    """
    Compute heavy output probability.
    
    A bitstring is "heavy" if its ideal probability is above the median.
    HOG = fraction of measured bitstrings that are heavy.
    """
    counts = mb_data['counts']
    total_shots = mb_data['total_shots']
    
    if ideal_probs is None:
        # Use count-based estimation
        # Simple version: just report the fraction of shots on the most probable bitstring
        return mb_data['ideal_count'] / total_shots
    
    # Find median probability
    all_probs = sorted(ideal_probs.values())
    median_prob = np.median(all_probs)
    
    # Count heavy bitstrings in measurement
    heavy_count = 0
    for bitstring, count in counts.items():
        if bitstring in ideal_probs and ideal_probs[bitstring] >= median_prob:
            heavy_count += count
    
    return heavy_count / total_shots


def analyze_all_configs(data_dir, output_dir):
    """
    Analyze all configurations and compute fidelity estimates.
    """
    results_dir = Path(data_dir) / "results"
    amplitudes_dir = Path(data_dir) / "amplitudes"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 1. N40 verification: varying depth
    print("Analyzing N40 verification (varying depth)...")
    n40_depths = [8, 10, 12, 14, 16, 18, 20]
    for d in n40_depths:
        xeb_data = load_xeb_data(results_dir, amplitudes_dir, 40, d)
        mb_data = load_mb_data(results_dir, 40, d)
        
        xeb_fidelities = []
        for xd in xeb_data:
            F, std = compute_xeb_fidelity(xd, 40)
            if F is not None:
                xeb_fidelities.append({'r': xd['r'], 'F_xeb': F, 'std': std})
        
        mb_probs = []
        for md in mb_data:
            p_success = compute_mb_success_probability(md)
            mb_probs.append({'r': md['r'], 'p_success': p_success})
        
        # Aggregate statistics
        if xeb_fidelities:
            F_values = [xf['F_xeb'] for xf in xeb_fidelities]
            F_mean = np.mean(F_values)
            F_std = np.std(F_values) / np.sqrt(len(F_values))
            
            all_results.append({
                'N': 40, 'd': d, 'method': 'XEB',
                'F_mean': F_mean, 'F_std': F_std,
                'n_instances': len(F_values),
                'individual': xeb_fidelities
            })
            print(f"  N=40, d={d}: F_XEB = {F_mean:.4f} ± {F_std:.4f} (n={len(F_values)})")
        
        if mb_probs:
            p_values = [mp['p_success'] for mp in mb_probs]
            p_mean = np.mean(p_values)
            p_std = np.std(p_values) / np.sqrt(len(p_values))
            
            all_results.append({
                'N': 40, 'd': d, 'method': 'MB',
                'F_mean': p_mean, 'F_std': p_std,
                'n_instances': len(p_values),
                'individual': mb_probs
            })
            print(f"  N=40, d={d}: P_MB = {p_mean:.6f} ± {p_std:.6f} (n={len(p_values)})")
    
    # 2. N scan at fixed depth d=12
    print("\nAnalyzing N scan (fixed depth d=12)...")
    n_values = [16, 24, 32, 40]
    for N in n_values:
        xeb_data = load_xeb_data(results_dir, amplitudes_dir, N, 12)
        mb_data = load_mb_data(results_dir, N, 12)
        
        xeb_fidelities = []
        for xd in xeb_data:
            F, std = compute_xeb_fidelity(xd, N)
            if F is not None:
                xeb_fidelities.append({'r': xd['r'], 'F_xeb': F, 'std': std})
        
        mb_probs = []
        for md in mb_data:
            p_success = compute_mb_success_probability(md)
            mb_probs.append({'r': md['r'], 'p_success': p_success})
        
        if xeb_fidelities:
            F_values = [xf['F_xeb'] for xf in xeb_fidelities]
            F_mean = np.mean(F_values)
            F_std = np.std(F_values) / np.sqrt(len(F_values))
            
            all_results.append({
                'N': N, 'd': 12, 'method': 'XEB',
                'F_mean': F_mean, 'F_std': F_std,
                'n_instances': len(F_values),
                'individual': xeb_fidelities
            })
            print(f"  N={N}, d=12: F_XEB = {F_mean:.4f} ± {F_std:.4f} (n={len(F_values)})")
        
        if mb_probs:
            p_values = [mp['p_success'] for mp in mb_probs]
            p_mean = np.mean(p_values)
            p_std = np.std(p_values) / np.sqrt(len(p_values))
            
            all_results.append({
                'N': N, 'd': 12, 'method': 'MB',
                'F_mean': p_mean, 'F_std': p_std,
                'n_instances': len(p_values),
                'individual': mb_probs
            })
            print(f"  N={N}, d=12: P_MB = {p_mean:.6f} ± {p_std:.6f} (n={len(p_values)})")
    
    # 3. N56 depth scan (no amplitudes, only MB)
    print("\nAnalyzing N56 depth scan (MB only)...")
    n56_depths = [8, 10, 12, 14, 16, 18, 20]
    for d in n56_depths:
        mb_data = load_mb_data(results_dir, 56, d)
        
        mb_probs = []
        for md in mb_data:
            p_success = compute_mb_success_probability(md)
            mb_probs.append({'r': md['r'], 'p_success': p_success})
        
        if mb_probs:
            p_values = [mp['p_success'] for mp in mb_probs]
            p_mean = np.mean(p_values)
            p_std = np.std(p_values) / np.sqrt(len(p_values))
            
            all_results.append({
                'N': 56, 'd': d, 'method': 'MB',
                'F_mean': p_mean, 'F_std': p_std,
                'n_instances': len(p_values),
                'individual': mb_probs
            })
            print(f"  N=56, d={d}: P_MB = {p_mean:.6f} ± {p_std:.6f} (n={len(p_values)})")
    
    # Save results
    # Remove 'individual' for JSON serialization
    save_results = []
    for r in all_results:
        r_save = {k: v for k, v in r.items() if k != 'individual'}
        save_results.append(r_save)
    
    with open(output_dir / "fidelity_results.json", 'w') as f:
        json.dump(save_results, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    data_dir = Path("data")
    output_dir = Path("outputs")
    
    results = analyze_all_configs(data_dir, output_dir)
    print("\nDone! Results saved to outputs/fidelity_results.json")
