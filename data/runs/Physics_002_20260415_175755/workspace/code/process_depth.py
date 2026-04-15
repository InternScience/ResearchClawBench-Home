#!/usr/bin/env python3
"""Process one depth at a time"""
import json
import os
import sys
import numpy as np
import pandas as pd


def parse_key(key):
    key = key.strip()
    if key.startswith('(') and key.endswith(')'):
        return tuple(int(x.strip()) for x in key[1:-1].split(','))
    return tuple(int(b) for b in key)


def compute_xeb(amp_file, counts_file, n_qubits):
    with open(amp_file, 'r') as f:
        amp_data = json.load(f)
    with open(counts_file, 'r') as f:
        counts_data = json.load(f)
    
    probs = []
    counts = []
    
    for key, amp_str in amp_data.items():
        bitstring_tuple = parse_key(key)
        count_key = str(bitstring_tuple)
        if count_key in counts_data:
            amp = complex(amp_str.strip('()').replace('j', 'j'))
            probs.append(abs(amp) ** 2)
            counts.append(counts_data[count_key])
    
    if not probs:
        return None, None, 0, 0
    
    probs = np.array(probs)
    counts = np.array(counts)
    total = counts.sum()
    
    mean_p = (probs * counts).sum() / total
    fidelity = (2 ** n_qubits) * mean_p - 1
    
    avg_p = np.average(probs, weights=counts)
    var_p = np.average((probs - avg_p) ** 2, weights=counts)
    std_err = (2 ** n_qubits) * np.sqrt(var_p / total)
    
    return fidelity, std_err, total, len(probs)


def parse_name(filename):
    import re
    m = re.search(r'N(\d+)_d(\d+)_r(\d+)', filename)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


d = int(sys.argv[1])
base = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_002_20260415_175755'
amp_dir = f'{base}/data/amplitudes/N40_verification/N40_d{d}_XEB'
counts_dir = f'{base}/data/results/N40_verification/N40_d{d}_XEB'
out_dir = f'{base}/outputs'

results = []
files = sorted([f for f in os.listdir(amp_dir) if f.endswith('_amplitudes.json')])

for fname in files:
    N, d_val, r = parse_name(fname)
    counts_fname = fname.replace('_amplitudes.json', '_counts.json')
    amp_path = os.path.join(amp_dir, fname)
    counts_path = os.path.join(counts_dir, counts_fname)
    
    if os.path.exists(counts_path):
        fid, std, n_samp, n_match = compute_xeb(amp_path, counts_path, 40)
        if fid is not None:
            results.append({
                'N': 40, 'd': d, 'r': r,
                'fidelity': fid, 'fidelity_std': std,
                'n_samples': n_samp, 'n_matched': n_match,
                'experiment': 'depth_scan'
            })

df = pd.DataFrame(results)
df.to_csv(f'{out_dir}/depth_{d}_results.csv', index=False)
print(f'd={d}: {len(results)} instances, mean fidelity: {df["fidelity"].mean():.4f}')
