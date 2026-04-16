#!/usr/bin/env python3
"""
Compute per-instance fidelity tables for all (N, d, r) configurations.
"""

import json
import os
import re
import numpy as np
from collections import defaultdict

def parse_tuple_key(key_str):
    return key_str.strip()

def amplitude_to_prob(amp_str):
    amp_str = amp_str.strip()
    if amp_str.startswith('(') and amp_str.endswith(')'):
        amp_str = amp_str[1:-1]
    c = complex(amp_str)
    return abs(c)**2

def load_counts_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {parse_tuple_key(k): v for k, v in data.items()}

def load_amplitudes_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        key = parse_tuple_key(k)
        prob = amplitude_to_prob(v)
        result[key] = prob
    return result

def load_ideal_bitstring_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return str(tuple(data))

DATA_DIR = 'data'
all_instance_data = []

# ---- N40 XEB instances ----
for d in [8, 10, 12, 14, 16, 18, 20]:
    xeb_dir = os.path.join(DATA_DIR, 'results', 'N40_verification', f'N40_d{d}_XEB')
    amp_dir = os.path.join(DATA_DIR, 'amplitudes', 'N40_verification', f'N40_d{d}_XEB')
    
    if not os.path.isdir(xeb_dir) or not os.path.isdir(amp_dir):
        continue
    
    for cf in sorted(os.listdir(xeb_dir)):
        match = re.search(r'_r(\d+)_XEB_counts\.json', cf)
        if not match:
            continue
        r = int(match.group(1))
        
        counts = load_counts_json(os.path.join(xeb_dir, cf))
        amp_file = cf.replace('_counts.json', '_amplitudes.json')
        amps = load_amplitudes_json(os.path.join(amp_dir, amp_file))
        
        D = 2**40
        matched = [(k, counts[k], amps[k]) for k in counts if k in amps]
        if matched:
            total_c = sum(c for _, c, _ in matched)
            avg_p = sum(c * p for _, c, p in matched) / total_c
            f_xeb = D * avg_p - 1
        else:
            f_xeb = None
        
        all_instance_data.append({
            'N': 40, 'd': d, 'r': r, 'method': 'XEB',
            'fidelity': f_xeb, 'n_matched': len(matched),
            'total_counts_matched': total_c if matched else 0
        })

# ---- N-scan XEB instances ----
for n in [16, 24, 32, 40]:
    xeb_dir = os.path.join(DATA_DIR, 'results', 'N_scan_depth12', f'N{n}_d12_XEB')
    amp_dir = os.path.join(DATA_DIR, 'amplitudes', 'N_scan_depth12', f'N{n}_d12_XEB')
    
    if not os.path.isdir(xeb_dir) or not os.path.isdir(amp_dir):
        continue
    
    for cf in sorted(os.listdir(xeb_dir)):
        match = re.search(r'_r(\d+)_XEB_counts\.json', cf)
        if not match:
            continue
        r = int(match.group(1))
        
        counts = load_counts_json(os.path.join(xeb_dir, cf))
        amp_file = cf.replace('_counts.json', '_amplitudes.json')
        amps = load_amplitudes_json(os.path.join(amp_dir, amp_file))
        
        D = 2**n
        matched = [(k, counts[k], amps[k]) for k in counts if k in amps]
        if matched:
            total_c = sum(c for _, c, _ in matched)
            avg_p = sum(c * p for _, c, p in matched) / total_c
            f_xeb = D * avg_p - 1
        else:
            f_xeb = None
        
        all_instance_data.append({
            'N': n, 'd': 12, 'r': r, 'method': 'XEB',
            'fidelity': f_xeb, 'n_matched': len(matched),
            'total_counts_matched': total_c if matched else 0
        })

# ---- N40 MB instances ----
for d in [8, 10, 12, 14, 16, 18, 20]:
    mb_dir = os.path.join(DATA_DIR, 'results', 'N40_verification', f'N40_d{d}_MB')
    if not os.path.isdir(mb_dir):
        continue
    
    for cf in sorted(os.listdir(mb_dir)):
        if not cf.endswith('_MB_counts.json'):
            continue
        match = re.search(r'_r(\d+)_MB_counts\.json', cf)
        if not match:
            continue
        r = int(match.group(1))
        
        counts = load_counts_json(os.path.join(mb_dir, cf))
        ideal_path = os.path.join(mb_dir, cf.replace('_counts.json', '_ideal_bitstring.json'))
        if not os.path.exists(ideal_path):
            continue
        ideal_key = load_ideal_bitstring_json(ideal_path)
        
        total = sum(counts.values())
        ideal_count = counts.get(ideal_key, 0)
        p_survival = ideal_count / total if total > 0 else 0
        
        all_instance_data.append({
            'N': 40, 'd': d, 'r': r, 'method': 'MB',
            'p_survival': p_survival,
            'ideal_count': ideal_count, 'total_counts': total
        })

# ---- N-scan MB instances ----
for n in [16, 24, 32, 40, 48, 56]:
    mb_dir = os.path.join(DATA_DIR, 'results', 'N_scan_depth12', f'N{n}_d12_MB')
    if not os.path.isdir(mb_dir):
        continue
    
    for cf in sorted(os.listdir(mb_dir)):
        if not cf.endswith('_MB_counts.json'):
            continue
        match = re.search(r'_r(\d+)_MB_counts\.json', cf)
        if not match:
            continue
        r = int(match.group(1))
        
        counts = load_counts_json(os.path.join(mb_dir, cf))
        ideal_path = os.path.join(mb_dir, cf.replace('_counts.json', '_ideal_bitstring.json'))
        if not os.path.exists(ideal_path):
            continue
        ideal_key = load_ideal_bitstring_json(ideal_path)
        
        total = sum(counts.values())
        ideal_count = counts.get(ideal_key, 0)
        p_survival = ideal_count / total if total > 0 else 0
        
        all_instance_data.append({
            'N': n, 'd': 12, 'r': r, 'method': 'MB',
            'p_survival': p_survival,
            'ideal_count': ideal_count, 'total_counts': total
        })

# ---- N56 MB instances ----
for d in [8, 10, 12, 14, 16, 18, 20]:
    mb_dir = os.path.join(DATA_DIR, 'results', 'N56_depths', f'N56_d{d}_MB')
    if not os.path.isdir(mb_dir):
        continue
    
    for cf in sorted(os.listdir(mb_dir)):
        if not cf.endswith('_MB_counts.json'):
            continue
        match = re.search(r'_r(\d+)_MB_counts\.json', cf)
        if not match:
            continue
        r = int(match.group(1))
        
        counts = load_counts_json(os.path.join(mb_dir, cf))
        ideal_path = os.path.join(mb_dir, cf.replace('_counts.json', '_ideal_bitstring.json'))
        if not os.path.exists(ideal_path):
            continue
        ideal_key = load_ideal_bitstring_json(ideal_path)
        
        total = sum(counts.values())
        ideal_count = counts.get(ideal_key, 0)
        p_survival = ideal_count / total if total > 0 else 0
        
        all_instance_data.append({
            'N': 56, 'd': d, 'r': r, 'method': 'MB',
            'p_survival': p_survival,
            'ideal_count': ideal_count, 'total_counts': total
        })

# ---- Transport 1QRB instances ----
for n in [40]:
    for d in [4, 16, 32, 48, 64, 96]:
        t_dir = os.path.join(DATA_DIR, 'results', 'N40_verification', f'N{n}_d{d}_Transport_1QRB')
        if not os.path.isdir(t_dir):
            continue
        
        for cf in sorted(os.listdir(t_dir)):
            if not cf.endswith('_Transport_1QRB_counts.json'):
                continue
            match = re.search(r'_r(\d+)_Transport_1QRB_counts\.json', cf)
            if not match:
                continue
            r = int(match.group(1))
            
            counts = load_counts_json(os.path.join(t_dir, cf))
            ideal_path = os.path.join(t_dir, cf.replace('_counts.json', '_ideal_bitstring.json'))
            if not os.path.exists(ideal_path):
                continue
            ideal_key = load_ideal_bitstring_json(ideal_path)
            
            total = sum(counts.values())
            ideal_count = counts.get(ideal_key, 0)
            p_survival = ideal_count / total if total > 0 else 0
            
            all_instance_data.append({
                'N': n, 'd': d, 'r': r, 'method': 'Transport_1QRB',
                'p_survival': p_survival,
                'ideal_count': ideal_count, 'total_counts': total
            })

# Save instance-level data
with open('outputs/instance_fidelity_data.json', 'w') as f:
    json.dump(all_instance_data, f, indent=2)

print(f"Total instances processed: {len(all_instance_data)}")

# Print summary tables
print("\n=== Per-Configuration Fidelity Summary ===")
print("\nN40 XEB (depth scan):")
for d in [8, 10, 12, 14, 16, 18, 20]:
    vals = [x['fidelity'] for x in all_instance_data if x['N']==40 and x['d']==d and x['method']=='XEB' and x['fidelity'] is not None]
    if vals:
        print(f"  d={d}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}, range=[{min(vals):.4f}, {max(vals):.4f}]")

print("\nN40 MB (depth scan):")
for d in [8, 10, 12, 14, 16, 18, 20]:
    vals = [x['p_survival'] for x in all_instance_data if x['N']==40 and x['d']==d and x['method']=='MB']
    if vals:
        print(f"  d={d}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}, range=[{min(vals):.4f}, {max(vals):.4f}]")

print("\nN-scan XEB (d=12):")
for n in [16, 24, 32, 40]:
    vals = [x['fidelity'] for x in all_instance_data if x['N']==n and x['d']==12 and x['method']=='XEB' and x['fidelity'] is not None]
    if vals:
        print(f"  N={n}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}, range=[{min(vals):.4f}, {max(vals):.4f}]")

print("\nN-scan MB (d=12):")
for n in [16, 24, 32, 40, 48, 56]:
    vals = [x['p_survival'] for x in all_instance_data if x['N']==n and x['d']==12 and x['method']=='MB']
    if vals:
        print(f"  N={n}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}, range=[{min(vals):.4f}, {max(vals):.4f}]")

print("\nN56 MB (depth scan):")
for d in [8, 10, 12, 14, 16, 18, 20]:
    vals = [x['p_survival'] for x in all_instance_data if x['N']==56 and x['d']==d and x['method']=='MB']
    if vals:
        print(f"  d={d}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, n={len(vals)}, range=[{min(vals):.4f}, {max(vals):.4f}]")