#!/usr/bin/env python3
"""
XEB Fidelity Analysis for Random Quantum Circuit Sampling

This module implements the Cross-Entropy Benchmarking (XEB) fidelity estimation
workflow as described in the Google Quantum Supremacy paper (Nature 2019).

The XEB linear cross-entropy benchmarking fidelity is computed as:
    F_XEB = 2^n * <P(x_i)>_i - 1

where:
    - n is the number of qubits
    - P(x_i) is the ideal probability of bitstring x_i
    - the average is over the observed bitstrings

For uncertainty estimation, we use bootstrap resampling.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


def parse_bitstring_key(key: str) -> str:
    """Convert tuple-string format to simple bitstring if needed."""
    # Handle format like "(0, 1, 0, ...)" 
    match = re.match(r'\(([\d,\s]+)\)', key)
    if match:
        digits = match.group(1).replace(',', '').replace(' ', '')
        return digits
    return key


def compute_ideal_probability(amplitude_str: str) -> float:
    """
    Convert complex amplitude string to probability |amplitude|^2.
    
    Args:
        amplitude_str: String representation of complex number, e.g., "(3.97e-07+3.09e-08j)"
    
    Returns:
        Probability = |amplitude|^2
    """
    # Parse the complex number string
    amplitude_str = amplitude_str.strip()
    # Remove parentheses if present
    if amplitude_str.startswith('(') and amplitude_str.endswith(')'):
        amplitude_str = amplitude_str[1:-1]
    
    # Handle the complex number format
    try:
        amp = complex(amplitude_str)
        return abs(amp) ** 2
    except ValueError:
        # Try alternative parsing
        return 0.0


def load_counts_file(filepath: str) -> Dict[str, int]:
    """Load measurement counts from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_amplitudes_file(filepath: str) -> Dict[str, float]:
    """Load ideal probabilities from amplitudes JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert amplitudes to probabilities
    probs = {}
    for key, amp_str in data.items():
        probs[parse_bitstring_key(key)] = compute_ideal_probability(amp_str)
    
    return probs


def compute_xeb_fidelity(counts: Dict[str, int], ideal_probs: Dict[str, float], n_qubits: int) -> float:
    """
    Compute the XEB linear cross-entropy benchmarking fidelity.
    
    F_XEB = 2^n * <P(x_i)>_i - 1
    
    where the average is weighted by the observed counts.
    
    Args:
        counts: Dictionary mapping bitstrings to observed counts
        ideal_probs: Dictionary mapping bitstrings to ideal probabilities
        n_qubits: Number of qubits in the circuit
    
    Returns:
        XEB fidelity estimate
    """
    total_counts = sum(counts.values())
    if total_counts == 0:
        return 0.0
    
    # Compute weighted average of ideal probabilities
    weighted_prob_sum = 0.0
    for bitstring, count in counts.items():
        parsed_key = parse_bitstring_key(bitstring)
        if parsed_key in ideal_probs:
            weighted_prob_sum += ideal_probs[parsed_key] * count
    
    avg_prob = weighted_prob_sum / total_counts
    
    # Compute XEB fidelity
    fidelity = (2 ** n_qubits) * avg_prob - 1
    
    return fidelity


def bootstrap_xeb_fidelity(counts: Dict[str, int], ideal_probs: Dict[str, float], 
                           n_qubits: int, n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """
    Compute XEB fidelity with bootstrap uncertainty estimation.
    
    Args:
        counts: Dictionary mapping bitstrings to observed counts
        ideal_probs: Dictionary mapping bitstrings to ideal probabilities
        n_qubits: Number of qubits
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (mean_fidelity, std_fidelity)
    """
    np.random.seed(seed)
    
    # Expand counts to individual samples
    samples = []
    for bitstring, count in counts.items():
        parsed_key = parse_bitstring_key(bitstring)
        if parsed_key in ideal_probs:
            samples.extend([parsed_key] * count)
    
    if len(samples) == 0:
        return 0.0, 0.0
    
    samples = np.array(samples)
    n_samples = len(samples)
    
    # Bootstrap resampling
    fidelities = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resampled = samples[indices]
        
        # Compute fidelity for this bootstrap sample
        unique, counts_unique = np.unique(resampled, return_counts=True)
        resampled_counts = dict(zip(unique, counts_unique))
        
        fid = compute_xeb_fidelity(resampled_counts, ideal_probs, n_qubits)
        fidelities.append(fid)
    
    return np.mean(fidelities), np.std(fidelities)


def extract_config_from_path(filepath: str) -> Tuple[int, int, int, str]:
    """
    Extract N (qubits), d (depth), r (instance), and type from filepath.
    
    Example: N40_d10_XEB/N40_d10_r1_XEB_counts.json -> (40, 10, 1, 'XEB')
    """
    filename = os.path.basename(filepath)

    # Pattern: N{N}_d{d}_r{r}_{type}_counts.json or N{N}_d{d}_r{r}_{type}.json
    # Handle both _counts suffix and direct naming
    match = re.search(r'N(\d+)_d(\d+)_r(\d+)_(\w+?)(?:_counts)?\.json', filename)
    if match:
        n = int(match.group(1))
        d = int(match.group(2))
        r = int(match.group(3))
        config_type = match.group(4)
        return n, d, r, config_type

    # Alternative pattern for Transport circuits: N{N}_d{d}_{type}
    match = re.search(r'N(\d+)_d(\d+)_(\w+?)(?:_counts)?\.json', filename)
    if match:
        n = int(match.group(1))
        d = int(match.group(2))
        config_type = match.group(3)
        return n, d, 0, config_type
    
    raise ValueError(f"Cannot parse configuration from: {filepath}")


def find_all_xeb_configs(base_dir: str) -> List[Tuple[str, str, int, int, int]]:
    """
    Find all XEB configuration pairs (counts and amplitudes files).

    Returns:
        List of tuples: (counts_path, amplitudes_path, N, d, r)
    """
    base_path = Path(base_dir)
    configs = []

    results_dir = base_path / 'results'
    amplitudes_dir = base_path / 'amplitudes'

    # Search for counts files in results directory
    for counts_file in results_dir.rglob('*_counts.json'):
        try:
            n, d, r, config_type = extract_config_from_path(str(counts_file))
        except ValueError:
            continue

        if config_type not in ['XEB', 'MB']:
            continue

        # Find corresponding amplitudes file
        # The amplitudes file should be in parallel directory structure
        rel_path = counts_file.relative_to(results_dir)
        amplitudes_path = amplitudes_dir / rel_path.parent / counts_file.name.replace('_counts.json', '_amplitudes.json')

        # Only include XEB configurations (not MB)
        if config_type != 'XEB':
            continue
        
        if amplitudes_path.exists():
            configs.append((str(counts_file), str(amplitudes_path), n, d, r))
    
    return sorted(configs, key=lambda x: (x[2], x[3], x[4]))  # Sort by N, d, r


def process_all_configs(configs: List[Tuple[str, str, int, int, int]], 
                        n_bootstrap: int = 1000) -> List[Dict]:
    """
    Process all configurations and compute XEB fidelities.
    
    Returns:
        List of dictionaries with results for each configuration
    """
    results = []
    
    for counts_path, amps_path, n, d, r in configs:
        try:
            counts = load_counts_file(counts_path)
            ideal_probs = load_amplitudes_file(amps_path)
            
            # Compute point estimate
            fidelity = compute_xeb_fidelity(counts, ideal_probs, n)
            
            # Compute bootstrap uncertainty
            mean_fid, std_fid = bootstrap_xeb_fidelity(counts, ideal_probs, n, n_bootstrap)
            
            results.append({
                'N': n,
                'd': d,
                'r': r,
                'fidelity': fidelity,
                'mean_fidelity': mean_fid,
                'std_fidelity': std_fid,
                'counts_file': counts_path,
                'amplitudes_file': amps_path,
                'n_samples': sum(counts.values()),
                'n_matched_keys': len([k for k in counts.keys() if parse_bitstring_key(k) in ideal_probs])
            })
        except Exception as e:
            print(f"Error processing {counts_path}: {e}")
            results.append({
                'N': n,
                'd': d,
                'r': r,
                'fidelity': None,
                'mean_fidelity': None,
                'std_fidelity': None,
                'error': str(e)
            })
    
    return results


def aggregate_by_config(results: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """
    Aggregate results by (N, d) configuration, computing mean and std across instances.
    
    Returns:
        Dictionary mapping (N, d) to aggregated statistics
    """
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for r in results:
        if r['fidelity'] is not None:
            grouped[(r['N'], r['d'])].append(r)
    
    aggregated = {}
    for (n, d), configs in grouped.items():
        fidelities = [c['fidelity'] for c in configs]
        mean_fids = [c['mean_fidelity'] for c in configs]
        std_fids = [c['std_fidelity'] for c in configs]
        
        aggregated[(n, d)] = {
            'N': n,
            'd': d,
            'n_instances': len(configs),
            'mean_fidelity': np.mean(fidelities),
            'std_fidelity_across_instances': np.std(fidelities),
            'mean_bootstrap_std': np.mean(std_fids),
            'min_fidelity': np.min(fidelities),
            'max_fidelity': np.max(fidelities),
            'fidelities': fidelities
        }
    
    return aggregated


if __name__ == '__main__':
    # Test with sample data
    base_dir = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_002_20260416_221313/data'
    
    print("Finding XEB configurations...")
    configs = find_all_xeb_configs(base_dir)
    print(f"Found {len(configs)} XEB configurations")
    
    # Process a subset for testing
    test_configs = configs[:5]
    print("\nProcessing test configurations...")
    results = process_all_configs(test_configs, n_bootstrap=100)
    
    for r in results:
        print(f"N={r['N']}, d={r['d']}, r={r['r']}: F_XEB = {r['fidelity']:.6f} ± {r['std_fidelity']:.6f}")
