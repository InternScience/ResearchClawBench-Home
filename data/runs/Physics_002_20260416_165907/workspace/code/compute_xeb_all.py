import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_xeb(amp_dir, res_dir, n_qubits, depth):
    fidelities = []
    
    amp_files = glob.glob(os.path.join(amp_dir, f'N{n_qubits}_d{depth}_XEB', '*_amplitudes.json'))
    
    for amp_file in amp_files:
        basename = os.path.basename(amp_file)
        parts = basename.split('_')
        r_part = [p for p in parts if p.startswith('r') and p[1:].isdigit()][0]
        
        counts_file = os.path.join(res_dir, f'N{n_qubits}_d{depth}_XEB', basename.replace('_amplitudes.json', '_counts.json'))
        
        if not os.path.exists(counts_file):
            continue
            
        with open(amp_file, 'r') as f:
            amps = json.load(f)
            
        with open(counts_file, 'r') as f:
            counts = json.load(f)
            
        probs = {k: abs(complex(v))**2 for k, v in amps.items()}
        
        valid_keys = set(probs.keys()).intersection(set(counts.keys()))
        if not valid_keys:
            continue
            
        valid_counts = sum(counts[k] for k in valid_keys)
        if valid_counts == 0:
            continue
            
        avg_prob = sum(probs[k] * counts[k] for k in valid_keys) / valid_counts
        
        dim = 2**n_qubits
        f_xeb = dim * avg_prob - 1
        
        fidelities.append(f_xeb)
        
    if not fidelities:
        return None, None
        
    return np.mean(fidelities), np.std(fidelities) / np.sqrt(len(fidelities))

def scan_depths(n_qubits=40):
    depths = [8, 10, 12, 14, 16, 18, 20]
    amp_dir = 'data/amplitudes/N40_verification'
    res_dir = 'data/results/N40_verification'
    
    f_means = []
    f_errs = []
    
    for d in depths:
        mean, err = compute_xeb(amp_dir, res_dir, n_qubits, d)
        if mean is not None:
            f_means.append(mean)
            f_errs.append(err)
            print(f"N={n_qubits}, d={d}: F = {mean:.4f} +/- {err:.4f}")
        else:
            print(f"N={n_qubits}, d={d}: No data")
            
    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(depths, f_means, yerr=f_errs, fmt='o-', capsize=5, capthick=2, label='Experimental XEB')
    plt.yscale('log')
    plt.xlabel('Circuit Depth (d)')
    plt.ylabel('XEB Fidelity')
    plt.title(f'XEB Fidelity vs Depth for N={n_qubits}')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig('outputs/xeb_vs_depth.png')
    plt.close()
    
    return depths, f_means, f_errs

def scan_n(depth=12):
    ns = [16, 24, 32, 40]
    amp_dir = 'data/amplitudes/N_scan_depth12'
    res_dir = 'data/results/N_scan_depth12'
    
    f_means = []
    f_errs = []
    
    for n in ns:
        mean, err = compute_xeb(amp_dir, res_dir, n, depth)
        if mean is not None:
            f_means.append(mean)
            f_errs.append(err)
            print(f"N={n}, d={depth}: F = {mean:.4f} +/- {err:.4f}")
        else:
            print(f"N={n}, d={depth}: No data")
            
    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(ns, f_means, yerr=f_errs, fmt='s-', capsize=5, capthick=2, color='red', label='Experimental XEB')
    plt.yscale('log')
    plt.xlabel('Number of Qubits (N)')
    plt.ylabel('XEB Fidelity')
    plt.title(f'XEB Fidelity vs Qubit Count for d={depth}')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig('outputs/xeb_vs_n.png')
    plt.close()
    
    return ns, f_means, f_errs

print("Scanning depths for N=40...")
d_depths, d_means, d_errs = scan_depths(40)

print("\nScanning N for d=12...")
n_ns, n_means, n_errs = scan_n(12)

# Save results
results = {
    'depth_scan': {
        'N': 40,
        'depths': d_depths,
        'fidelities': d_means,
        'errors': d_errs
    },
    'n_scan': {
        'd': 12,
        'Ns': n_ns,
        'fidelities': n_means,
        'errors': n_errs
    }
}

with open('outputs/xeb_results.json', 'w') as f:
    json.dump(results, f, indent=4)
