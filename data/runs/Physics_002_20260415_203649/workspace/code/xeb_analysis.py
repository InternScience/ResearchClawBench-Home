#!/usr/bin/env python3
"""
RCS Fidelity Estimation Analysis - Complete Pipeline
Computes XEB fidelity, MB survival probability, Transport RB fidelity,
and gate-count error propagation models for RCS on arbitrary geometries.
"""

import json
import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# Utility functions
# ============================================================

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

# ============================================================
# XEB Fidelity Computation
# ============================================================

def compute_xeb_fidelity(counts_dict, amplitudes_dict, n_qubits):
    D = 2**n_qubits
    matched_probs = []
    matched_counts = []
    
    for bitstring_key, count in counts_dict.items():
        if bitstring_key in amplitudes_dict:
            prob = amplitudes_dict[bitstring_key]
            matched_probs.append(prob)
            matched_counts.append(count)
    
    if len(matched_probs) == 0:
        return None, 0, 0, 0
    
    total_matched_counts = sum(matched_counts)
    weighted_sum = sum(c * p for c, p in zip(matched_counts, matched_probs))
    avg_prob = weighted_sum / total_matched_counts
    
    f_xeb = D * avg_prob - 1
    
    probs_array = np.array(matched_probs)
    counts_array = np.array(matched_counts)
    weights = counts_array / total_matched_counts
    weighted_var = np.sum(weights * (probs_array - avg_prob)**2)
    se_prob = np.sqrt(weighted_var / len(matched_probs)) if len(matched_probs) > 1 else 0
    se_f_xeb = D * se_prob
    
    return f_xeb, se_f_xeb, len(matched_probs), total_matched_counts

# ============================================================
# MB Fidelity Computation
# ============================================================

def compute_mb_fidelity_single(mb_counts_path, mb_ideal_path, n_qubits):
    counts_data = load_counts_json(mb_counts_path)
    ideal_key = load_ideal_bitstring_json(mb_ideal_path)
    
    total_counts = sum(counts_data.values())
    ideal_count = counts_data.get(ideal_key, 0)
    p_survival = ideal_count / total_counts if total_counts > 0 else 0
    
    return {
        'p_survival': p_survival,
        'ideal_count': ideal_count,
        'total_counts': total_counts,
        'n_unique_bitstrings': len(counts_data)
    }

# ============================================================
# Transport 1QRB Fidelity Computation
# ============================================================

def compute_transport_fidelity_single(transport_counts_path, transport_ideal_path, n_qubits):
    counts_data = load_counts_json(transport_counts_path)
    ideal_key = load_ideal_bitstring_json(transport_ideal_path)
    
    total_counts = sum(counts_data.values())
    ideal_count = counts_data.get(ideal_key, 0)
    p_survival = ideal_count / total_counts if total_counts > 0 else 0
    
    return {
        'p_survival': p_survival,
        'ideal_count': ideal_count,
        'total_counts': total_counts,
        'n_unique_bitstrings': len(counts_data)
    }

# ============================================================
# Gate-count Error Propagation Model
# ============================================================

def gate_count_model(n_qubits, depth, e_1q=0.0016, e_2q=0.0062, e_readout=0.018):
    n_sq_gates = depth * n_qubits
    n_2q_gates = depth * (n_qubits // 2)
    n_readout = n_qubits
    
    f_sq = (1 - e_1q) ** n_sq_gates
    f_2q = (1 - e_2q) ** n_2q_gates
    f_ro = (1 - e_readout) ** n_readout
    
    f_predicted = f_sq * f_2q * f_ro
    return f_predicted

# ============================================================
# Batch Analysis Functions
# ============================================================

def analyze_xeb_depth_scan(base_dir, amp_base_dir, n_qubits, depths):
    all_results = {}
    
    for d in depths:
        xeb_dir = os.path.join(base_dir, f'N{n_qubits}_d{d}_XEB')
        if amp_base_dir is not None:
            amp_dir = os.path.join(amp_base_dir, f'N{n_qubits}_d{d}_XEB')
        else:
            amp_dir = None
        
        if not os.path.isdir(xeb_dir):
            continue
        if amp_dir is not None and not os.path.isdir(amp_dir):
            continue
        
        instance_fidelities = []
        
        counts_files = sorted([f for f in os.listdir(xeb_dir) 
                              if f.endswith('_XEB_counts.json')])
        
        for cf in counts_files:
            match = re.search(r'_r(\d+)_XEB_counts\.json', cf)
            if not match:
                continue
            
            counts_data = load_counts_json(os.path.join(xeb_dir, cf))
            
            if amp_dir is not None:
                amp_file = cf.replace('_counts.json', '_amplitudes.json')
                amp_path = os.path.join(amp_dir, amp_file)
                if not os.path.exists(amp_path):
                    continue
                amplitudes_data = load_amplitudes_json(amp_path)
            else:
                continue
            
            result = compute_xeb_fidelity(counts_data, amplitudes_data, n_qubits)
            if result[0] is not None:
                instance_fidelities.append(result[0])
        
        if len(instance_fidelities) > 0:
            mean_f = np.mean(instance_fidelities)
            se_mean = np.std(instance_fidelities) / np.sqrt(len(instance_fidelities))
            all_results[d] = {
                'mean_fidelity': mean_f,
                'se_fidelity': se_mean,
                'n_instances': len(instance_fidelities),
                'individual_fidelities': instance_fidelities
            }
    
    return all_results

def analyze_xeb_n_scan(results_base, amp_base, n_values, depth=12):
    all_results = {}
    
    for n in n_values:
        xeb_dir = os.path.join(results_base, f'N{n}_d{depth}_XEB')
        if amp_base is not None:
            amp_dir = os.path.join(amp_base, f'N{n}_d{depth}_XEB')
        else:
            amp_dir = None
        
        if not os.path.isdir(xeb_dir):
            continue
        if amp_dir is not None and not os.path.isdir(amp_dir):
            continue
        
        instance_fidelities = []
        
        counts_files = sorted([f for f in os.listdir(xeb_dir) 
                              if f.endswith('_XEB_counts.json')])
        
        for cf in counts_files:
            match = re.search(r'_r(\d+)_XEB_counts\.json', cf)
            if not match:
                continue
            
            counts_data = load_counts_json(os.path.join(xeb_dir, cf))
            
            if amp_dir is not None:
                amp_file = cf.replace('_counts.json', '_amplitudes.json')
                amp_path = os.path.join(amp_dir, amp_file)
                if not os.path.exists(amp_path):
                    continue
                amplitudes_data = load_amplitudes_json(amp_path)
            else:
                continue
            
            result = compute_xeb_fidelity(counts_data, amplitudes_data, n)
            if result[0] is not None:
                instance_fidelities.append(result[0])
        
        if len(instance_fidelities) > 0:
            mean_f = np.mean(instance_fidelities)
            se_mean = np.std(instance_fidelities) / np.sqrt(len(instance_fidelities))
            all_results[n] = {
                'mean_fidelity': mean_f,
                'se_fidelity': se_mean,
                'n_instances': len(instance_fidelities),
                'individual_fidelities': instance_fidelities
            }
    
    return all_results

def analyze_mb_depth_scan(base_dir, n_qubits, depths):
    all_results = {}
    
    for d in depths:
        mb_dir = os.path.join(base_dir, f'N{n_qubits}_d{d}_MB')
        if not os.path.isdir(mb_dir):
            continue
        
        survival_probs = []
        counts_files = sorted([f for f in os.listdir(mb_dir) 
                              if f.endswith('_MB_counts.json')])
        
        for cf in counts_files:
            match = re.search(r'_r(\d+)_MB_counts\.json', cf)
            if not match:
                continue
            
            counts_path = os.path.join(mb_dir, cf)
            ideal_path = os.path.join(mb_dir, cf.replace('_counts.json', '_ideal_bitstring.json'))
            
            if not os.path.exists(ideal_path):
                continue
            
            result = compute_mb_fidelity_single(counts_path, ideal_path, n_qubits)
            survival_probs.append(result['p_survival'])
        
        if len(survival_probs) > 0:
            mean_p = np.mean(survival_probs)
            se_p = np.std(survival_probs) / np.sqrt(len(survival_probs))
            all_results[d] = {
                'mean_survival': mean_p,
                'se_survival': se_p,
                'n_instances': len(survival_probs),
                'individual_survivals': survival_probs
            }
    
    return all_results

def analyze_mb_n_scan(base_dir, n_values, depth=12):
    all_results = {}
    
    for n in n_values:
        mb_dir = os.path.join(base_dir, f'N{n}_d{depth}_MB')
        if not os.path.isdir(mb_dir):
            continue
        
        survival_probs = []
        counts_files = sorted([f for f in os.listdir(mb_dir) 
                              if f.endswith('_MB_counts.json')])
        
        for cf in counts_files:
            match = re.search(r'_r(\d+)_MB_counts\.json', cf)
            if not match:
                continue
            
            counts_path = os.path.join(mb_dir, cf)
            ideal_path = os.path.join(mb_dir, cf.replace('_counts.json', '_ideal_bitstring.json'))
            
            if not os.path.exists(ideal_path):
                continue
            
            result = compute_mb_fidelity_single(counts_path, ideal_path, n)
            survival_probs.append(result['p_survival'])
        
        if len(survival_probs) > 0:
            mean_p = np.mean(survival_probs)
            se_p = np.std(survival_probs) / np.sqrt(len(survival_probs))
            all_results[n] = {
                'mean_survival': mean_p,
                'se_survival': se_p,
                'n_instances': len(survival_probs),
                'individual_survivals': survival_probs
            }
    
    return all_results

def analyze_transport_depth_scan(base_dir, n_qubits, depths_list):
    all_results = {}
    
    for d in depths_list:
        transport_dir = os.path.join(base_dir, f'N{n_qubits}_d{d}_Transport_1QRB')
        if not os.path.isdir(transport_dir):
            continue
        
        survival_probs = []
        counts_files = sorted([f for f in os.listdir(transport_dir) 
                              if f.endswith('_Transport_1QRB_counts.json')])
        
        for cf in counts_files:
            match = re.search(r'_r(\d+)_Transport_1QRB_counts\.json', cf)
            if not match:
                continue
            
            counts_path = os.path.join(transport_dir, cf)
            ideal_path = os.path.join(transport_dir, 
                                      cf.replace('_counts.json', '_ideal_bitstring.json'))
            
            if not os.path.exists(ideal_path):
                continue
            
            result = compute_transport_fidelity_single(counts_path, ideal_path, n_qubits)
            survival_probs.append(result['p_survival'])
        
        if len(survival_probs) > 0:
            mean_p = np.mean(survival_probs)
            se_p = np.std(survival_probs) / np.sqrt(len(survival_probs))
            all_results[d] = {
                'mean_survival': mean_p,
                'se_survival': se_p,
                'n_instances': len(survival_probs),
                'individual_survivals': survival_probs
            }
    
    return all_results

def analyze_transport_n_scan(base_dir, n_values, depths_list):
    all_results = {}
    
    for n in n_values:
        for d in depths_list:
            transport_dir = os.path.join(base_dir, f'N{n}_d{d}_Transport_1QRB')
            if not os.path.isdir(transport_dir):
                continue
        
            survival_probs = []
            counts_files = sorted([f for f in os.listdir(transport_dir) 
                                  if f.endswith('_Transport_1QRB_counts.json')])
        
            for cf in counts_files:
                match = re.search(r'_r(\d+)_Transport_1QRB_counts\.json', cf)
                if not match:
                    continue
                
                counts_path = os.path.join(transport_dir, cf)
                ideal_path = os.path.join(transport_dir, 
                                          cf.replace('_counts.json', '_ideal_bitstring.json'))
                
                if not os.path.exists(ideal_path):
                    continue
                
                result = compute_transport_fidelity_single(counts_path, ideal_path, n)
                survival_probs.append(result['p_survival'])
        
            if len(survival_probs) > 0:
                key = (n, d)
                mean_p = np.mean(survival_probs)
                se_p = np.std(survival_probs) / np.sqrt(len(survival_probs))
                all_results[key] = {
                    'mean_survival': mean_p,
                    'se_survival': se_p,
                    'n_instances': len(survival_probs),
                    'individual_survivals': survival_probs
                }
    
    return all_results

# ============================================================
# Run full analysis and save results
# ============================================================

if __name__ == '__main__':
    DATA_DIR = 'data'
    
    print("=" * 60)
    print("RCS Fidelity Estimation Analysis")
    print("=" * 60)
    
    # ---- N40 Verification (depth scan) ----
    print("\n--- N40 Verification: Depth Scan ---")
    n40_xeb = analyze_xeb_depth_scan(
        base_dir=os.path.join(DATA_DIR, 'results', 'N40_verification'),
        amp_base_dir=os.path.join(DATA_DIR, 'amplitudes', 'N40_verification'),
        n_qubits=40,
        depths=[8, 10, 12, 14, 16, 18, 20]
    )
    
    n40_mb = analyze_mb_depth_scan(
        base_dir=os.path.join(DATA_DIR, 'results', 'N40_verification'),
        n_qubits=40,
        depths=[8, 10, 12, 14, 16, 18, 20]
    )
    
    n40_transport = analyze_transport_depth_scan(
        base_dir=os.path.join(DATA_DIR, 'results', 'N40_verification'),
        n_qubits=40,
        depths_list=[4, 16, 32, 48, 64, 96]
    )
    
    # ---- N56 Depths (depth scan) ----
    print("\n--- N56 Depths: Depth Scan ---")
    n56_mb = analyze_mb_depth_scan(
        base_dir=os.path.join(DATA_DIR, 'results', 'N56_depths'),
        n_qubits=56,
        depths=[8, 10, 12, 14, 16, 18, 20]
    )
    
    n56_transport = analyze_transport_depth_scan(
        base_dir=os.path.join(DATA_DIR, 'results', 'N56_depths'),
        n_qubits=56,
        depths_list=[4, 16, 32, 48, 64, 96]
    )
    
    # ---- N-scan at depth 12 ----
    print("\n--- N-scan at Depth 12 ---")
    n_scan_xeb = analyze_xeb_n_scan(
        results_base=os.path.join(DATA_DIR, 'results', 'N_scan_depth12'),
        amp_base=os.path.join(DATA_DIR, 'amplitudes', 'N_scan_depth12'),
        n_values=[16, 24, 32, 40],
        depth=12
    )
    
    n_scan_mb = analyze_mb_n_scan(
        base_dir=os.path.join(DATA_DIR, 'results', 'N_scan_depth12'),
        n_values=[16, 24, 32, 40, 48, 56],
        depth=12
    )
    
    n_scan_transport = analyze_transport_n_scan(
        base_dir=os.path.join(DATA_DIR, 'results', 'N_scan_depth12'),
        n_values=[16, 24, 32, 40, 48, 56],
        depths_list=[4, 16, 32, 48, 64, 96]
    )
    
    # ---- Gate-count model predictions ----
    print("\n--- Gate-count Error Propagation Model ---")
    
    n40_model = {}
    for d in [8, 10, 12, 14, 16, 18, 20]:
        n40_model[d] = gate_count_model(40, d)
    
    n56_model = {}
    for d in [8, 10, 12, 14, 16, 18, 20]:
        n56_model[d] = gate_count_model(56, d)
    
    n_scan_model = {}
    for n in [16, 24, 32, 40, 48, 56]:
        n_scan_model[n] = gate_count_model(n, 12)
    
    # ---- Print results ----
    print("\n=== Results Summary ===")
    
    print("\nN40 XEB Fidelity (depth scan):")
    for d in sorted(n40_xeb.keys()):
        v = n40_xeb[d]
        print(f"  d={d}: F_XEB = {v['mean_fidelity']:.4f} +/- {v['se_fidelity']:.4f} ({v['n_instances']} instances)")
    
    print("\nN40 MB Survival Probability (depth scan):")
    for d in sorted(n40_mb.keys()):
        v = n40_mb[d]
        print(f"  d={d}: p_survival = {v['mean_survival']:.6f} +/- {v['se_survival']:.6f}")
    
    print("\nN40 Transport 1QRB Survival Probability:")
    for d in sorted(n40_transport.keys()):
        v = n40_transport[d]
        print(f"  d={d}: p_survival = {v['mean_survival']:.6f} +/- {v['se_survival']:.6f}")
    
    print("\nN40 Gate-count Model (depth scan):")
    for d in sorted(n40_model.keys()):
        print(f"  d={d}: F_pred = {n40_model[d]:.6f}")
    
    print("\nN56 MB Survival Probability (depth scan):")
    for d in sorted(n56_mb.keys()):
        v = n56_mb[d]
        print(f"  d={d}: p_survival = {v['mean_survival']:.6f} +/- {v['se_survival']:.6f}")
    
    print("\nN-scan XEB Fidelity (d=12):")
    for n in sorted(n_scan_xeb.keys()):
        v = n_scan_xeb[n]
        print(f"  N={n}: F_XEB = {v['mean_fidelity']:.4f} +/- {v['se_fidelity']:.4f}")
    
    print("\nN-scan MB Survival Probability (d=12):")
    for n in sorted(n_scan_mb.keys()):
        v = n_scan_mb[n]
        print(f"  N={n}: p_survival = {v['mean_survival']:.6f} +/- {v['se_survival']:.6f}")
    
    print("\nN-scan Gate-count Model (d=12):")
    for n in sorted(n_scan_model.keys()):
        print(f"  N={n}: F_pred = {n_scan_model[n]:.6f}")
    
    # Save all results as JSON
    results_to_save = {
        'n40_xeb': {str(d): v for d, v in n40_xeb.items()},
        'n40_mb': {str(d): v for d, v in n40_mb.items()},
        'n40_transport': {str(d): v for d, v in n40_transport.items()},
        'n40_model': {str(d): {'f_pred': v} for d, v in n40_model.items()},
        'n56_mb': {str(d): v for d, v in n56_mb.items()},
        'n56_transport': {str(d): v for d, v in n56_transport.items()},
        'n56_model': {str(d): {'f_pred': v} for d, v in n56_model.items()},
        'n_scan_xeb': {str(n): v for n, v in n_scan_xeb.items()},
        'n_scan_mb': {str(n): v for n, v in n_scan_mb.items()},
        'n_scan_transport': {str(key): v for key, v in n_scan_transport.items()},
        'n_scan_model': {str(n): {'f_pred': v} for n, v in n_scan_model.items()},
    }
    
    with open('outputs/fidelity_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    print("\nResults saved to outputs/fidelity_results.json")