import json
import glob
import os
import numpy as np

def compute_mb_fidelity(n_qubits, depth):
    res_files = glob.glob(f'data/results/N{n_qubits}_verification/N{n_qubits}_d{depth}_MB/*_MB_counts.json')
    
    if not res_files:
        return None, None
        
    fidelities = []
    
    for res_file in res_files:
        ideal_file = res_file.replace('_counts.json', '_ideal_bitstring.json')
        
        if not os.path.exists(ideal_file):
            continue
            
        with open(res_file, 'r') as f:
            counts = json.load(f)
            
        with open(ideal_file, 'r') as f:
            ideal_data = json.load(f)
            
        # The ideal bitstring is a list of ints
        target_bitstring = str(tuple(ideal_data))
        
        total_counts = sum(counts.values())
        if total_counts == 0:
            continue
            
        target_count = counts.get(target_bitstring, 0)
        
        # For Mirror Benchmarking, fidelity is related to survival probability of the target state
        # F_MB = (2^N * P_success - 1) / (2^N - 1)
        p_success = target_count / total_counts
        
        dim = 2**n_qubits
        f_mb = (dim * p_success - 1) / (dim - 1) if dim > 1 else p_success
        
        fidelities.append(f_mb)
        
    if not fidelities:
        return None, None
        
    return np.mean(fidelities), np.std(fidelities) / np.sqrt(len(fidelities))

depths = [8, 10, 12, 14, 16, 18, 20]
mb_means = []
mb_errs = []

for d in depths:
    mean, err = compute_mb_fidelity(40, d)
    if mean is not None:
        mb_means.append(mean)
        mb_errs.append(err)
        print(f"MB N=40, d={d}: F = {mean:.4f} +/- {err:.4f}")
    else:
        print(f"MB N=40, d={d}: No data")

# Save to JSON
with open('outputs/xeb_results.json', 'r') as f:
    results = json.load(f)
    
results['mb_scan'] = {
    'N': 40,
    'depths': depths,
    'fidelities': mb_means,
    'errors': mb_errs
}

with open('outputs/xeb_results.json', 'w') as f:
    json.dump(results, f, indent=4)
