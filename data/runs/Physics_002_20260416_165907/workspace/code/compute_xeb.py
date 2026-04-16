import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_xeb(n_qubits, depth):
    # For a given N and depth, find all instances r
    amp_files = glob.glob(f'data/amplitudes/N{n_qubits}_verification/N{n_qubits}_d{depth}_XEB/*_amplitudes.json')
    if not amp_files:
        amp_files = glob.glob(f'data/amplitudes/N_scan_depth{depth}/N{n_qubits}_d{depth}_XEB/*_amplitudes.json')
    
    fidelities = []
    
    for amp_file in amp_files:
        # Extract r from filename
        basename = os.path.basename(amp_file)
        # e.g. N40_d10_r50_XEB_amplitudes.json
        parts = basename.split('_')
        r_part = [p for p in parts if p.startswith('r') and p[1:].isdigit()][0]
        r = int(r_part[1:])
        
        # Find corresponding counts file
        counts_file = f'data/results/N{n_qubits}_verification/N{n_qubits}_d{depth}_XEB/N{n_qubits}_d{depth}_r{r}_XEB_counts.json'
        if not os.path.exists(counts_file):
            counts_file = f'data/results/N_scan_depth{depth}/N{n_qubits}_d{depth}_XEB/N{n_qubits}_d{depth}_r{r}_XEB_counts.json'
            
        if not os.path.exists(counts_file):
            print(f"Warning: Missing counts file for {basename}")
            continue
            
        with open(amp_file, 'r') as f:
            amps = json.load(f)
            
        with open(counts_file, 'r') as f:
            counts = json.load(f)
            
        # Compute probabilities from amplitudes
        probs = {k: abs(complex(v))**2 for k, v in amps.items()}
        
        # Compute XEB for this instance
        # Formula: F_XEB = 2^N * mean(P(x_i)) - 1
        # where the mean is taken over the experimentally observed bitstrings
        
        # In our case, the experimental results only give a subset of bitstrings and their counts
        # We need to compute the weighted average of P(x_i)
        
        total_counts = sum(counts.values())
        
        if total_counts == 0:
            continue
            
        # Filter to only include keys that are in both
        valid_keys = set(probs.keys()).intersection(set(counts.keys()))
        
        if not valid_keys:
            print(f"Warning: No matching keys for r={r}")
            continue
            
        # Calculate sum of counts for valid keys
        valid_counts = sum(counts[k] for k in valid_keys)
        
        if valid_counts == 0:
            continue
            
        # Compute average probability weighted by counts
        avg_prob = sum(probs[k] * counts[k] for k in valid_keys) / valid_counts
        
        # Calculate XEB
        dim = 2**n_qubits
        f_xeb = dim * avg_prob - 1
        
        fidelities.append(f_xeb)
        
    if not fidelities:
        return None, None
        
    return np.mean(fidelities), np.std(fidelities) / np.sqrt(len(fidelities))

print("Testing N=40, d=10")
mean_f, std_f = compute_xeb(40, 10)
print(f"Fidelity: {mean_f} +/- {std_f}")
