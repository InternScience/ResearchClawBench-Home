#!/usr/bin/env python3
"""
Streamlined XEB Fidelity Analysis
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def parse_key(key):
    """Parse bitstring key."""
    key = key.strip()
    if key.startswith('(') and key.endswith(')'):
        return tuple(int(x.strip()) for x in key[1:-1].split(','))
    return tuple(int(b) for b in key)


def compute_xeb(amp_file, counts_file, n_qubits):
    """Compute XEB fidelity from amplitude and counts files."""
    # Load amplitudes
    with open(amp_file, 'r') as f:
        amp_data = json.load(f)
    
    # Load counts
    with open(counts_file, 'r') as f:
        counts_data = json.load(f)
    
    # Match and compute
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
    
    # Weighted mean
    mean_p = (probs * counts).sum() / total
    fidelity = (2 ** n_qubits) * mean_p - 1
    
    # Std error
    avg_p = np.average(probs, weights=counts)
    var_p = np.average((probs - avg_p) ** 2, weights=counts)
    std_err = (2 ** n_qubits) * np.sqrt(var_p / total)
    
    return fidelity, std_err, total, len(probs)


def parse_name(filename):
    """Extract N, d, r from filename."""
    m = re.search(r'N(\d+)_d(\d+)_r(\d+)', filename)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def process_directory(amp_dir, res_dir, exp_type):
    """Process all matching files in a directory pair."""
    results = []
    
    if not os.path.exists(amp_dir) or not os.path.exists(res_dir):
        return results
    
    for fname in os.listdir(amp_dir):
        if not fname.endswith('_amplitudes.json'):
            continue
        
        N, d, r = parse_name(fname)
        if N is None:
            continue
        
        counts_fname = fname.replace('_amplitudes.json', '_counts.json')
        amp_path = os.path.join(amp_dir, fname)
        counts_path = os.path.join(res_dir, counts_fname)
        
        if not os.path.exists(counts_path):
            continue
        
        try:
            fid, std, n_samp, n_match = compute_xeb(amp_path, counts_path, N)
            if fid is not None:
                results.append({
                    'N': N, 'd': d, 'r': r,
                    'fidelity': fid,
                    'fidelity_std': std,
                    'n_samples': n_samp,
                    'n_matched': n_match,
                    'experiment_type': exp_type
                })
        except Exception as e:
            print(f"  Error: {fname}: {e}")
    
    return results


def main():
    base = Path(__file__).parent.parent
    data_dir = base / 'data'
    out_dir = base / 'outputs'
    img_dir = base / 'report' / 'images'
    
    out_dir.mkdir(exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("XEB Fidelity Analysis")
    print("=" * 60)
    
    all_results = []
    
    # Process N40 verification (depth scan)
    print("\n[1] Processing N40 verification (depth scan)...")
    amp_base = data_dir / 'amplitudes' / 'N40_verification'
    res_base = data_dir / 'results' / 'N40_verification'
    
    if amp_base.exists():
        for subdir in sorted(os.listdir(amp_base)):
            if '_XEB' in subdir:
                amp_path = amp_base / subdir
                res_path = res_base / subdir
                results = process_directory(str(amp_path), str(res_path), 'N40_verification')
                all_results.extend(results)
                print(f"  {subdir}: {len(results)} instances")
    
    # Process N scan (fixed depth)
    print("\n[2] Processing N scan (depth=12)...")
    amp_base = data_dir / 'amplitudes' / 'N_scan_depth12'
    res_base = data_dir / 'results' / 'N_scan_depth12'
    
    if amp_base.exists():
        for subdir in sorted(os.listdir(amp_base)):
            if '_XEB' in subdir:
                amp_path = amp_base / subdir
                res_path = res_base / subdir
                results = process_directory(str(amp_path), str(res_path), 'N_scan_depth12')
                all_results.extend(results)
                print(f"  {subdir}: {len(results)} instances")
    
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("\nNo data processed!")
        return
    
    print(f"\nTotal instances processed: {len(df)}")
    
    # Save results
    df.to_csv(out_dir / 'xeb_results.csv', index=False)
    df.to_csv(img_dir / 'xeb_results.csv', index=False)
    
    # Generate plots
    print("\n[3] Generating plots...")
    
    # Plot 1: Fidelity vs Depth (N=40)
    df_n40 = df[(df['N'] == 40) & (df['experiment_type'] == 'N40_verification')]
    if not df_n40.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stats = df_n40.groupby('d').agg({'fidelity': ['mean', 'std']}).reset_index()
        stats.columns = ['d', 'mean', 'std']
        
        for d in sorted(df_n40['d'].unique()):
            data = df_n40[df_n40['d'] == d]['fidelity']
            ax.scatter([d]*len(data), data, alpha=0.4, s=40, color='steelblue')
        
        ax.errorbar(stats['d'], stats['mean'], yerr=stats['std'],
                   fmt='o-', color='darkred', linewidth=2.5, markersize=9, capsize=5)
        
        ax.set_xlabel('Circuit Depth (d)', fontsize=14)
        ax.set_ylabel('XEB Fidelity', fontsize=14)
        ax.set_title('XEB Fidelity vs Circuit Depth (N=40 qubits)', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(img_dir / 'fidelity_vs_depth.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fidelity_vs_depth.png")
    
    # Plot 2: Fidelity vs N (d=12)
    df_scan = df[df['experiment_type'] == 'N_scan_depth12']
    if not df_scan.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stats = df_scan.groupby('N').agg({'fidelity': ['mean', 'std']}).reset_index()
        stats.columns = ['N', 'mean', 'std']
        
        for N in sorted(df_scan['N'].unique()):
            data = df_scan[df_scan['N'] == N]['fidelity']
            ax.scatter([N]*len(data), data, alpha=0.4, s=40, color='steelblue')
        
        ax.errorbar(stats['N'], stats['mean'], yerr=stats['std'],
                   fmt='s-', color='darkgreen', linewidth=2.5, markersize=9, capsize=5)
        
        ax.set_xlabel('Number of Qubits (N)', fontsize=14)
        ax.set_ylabel('XEB Fidelity', fontsize=14)
        ax.set_title('XEB Fidelity vs Qubit Count (d=12)', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(img_dir / 'fidelity_vs_n_qubits.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: fidelity_vs_n_qubits.png")
    
    # Plot 3: Quantum-Classical Gap
    if not df_n40.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        stats = df_n40.groupby('d').agg({'fidelity': ['mean', 'std']}).reset_index()
        stats.columns = ['d', 'mean', 'std']
        
        depths = stats['d'].values
        
        ax.plot(depths, stats['mean'], 'o-', color='darkblue', linewidth=3, markersize=10,
                label='Experimental XEB Fidelity')
        ax.fill_between(depths, stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                        alpha=0.25, color='darkblue')
        
        # Classical threshold
        ax.axhline(y=0.001, color='red', linestyle='--', linewidth=2.5, 
                   label='Classical Approximation Limit (~0.001)')
        
        # Supremacy region
        ax.fill_between([depths.min()-1, depths.max()+1], 0.001, 1.0, 
                        alpha=0.1, color='green', label='Quantum Advantage Regime')
        
        ax.set_xlabel('Circuit Depth (d)', fontsize=14)
        ax.set_ylabel('XEB Fidelity', fontsize=14)
        ax.set_title('Quantum vs Classical: The Computational Gap (N=40 qubits)', fontsize=16)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([depths.min()-1, depths.max()+1])
        
        plt.tight_layout()
        plt.savefig(img_dir / 'quantum_classical_gap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: quantum_classical_gap.png")
    
    # Plot 4: Combined summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    if not df_n40.empty:
        ax = axes[0, 0]
        stats = df_n40.groupby('d')['fidelity'].agg(['mean', 'std']).reset_index()
        for d in sorted(df_n40['d'].unique()):
            data = df_n40[df_n40['d'] == d]['fidelity']
            ax.scatter([d]*len(data), data, alpha=0.3, s=30, color='steelblue')
        ax.errorbar(stats['d'], stats['mean'], yerr=stats['std'],
                   fmt='o-', color='darkred', linewidth=2, capsize=4)
        ax.set_xlabel('Depth (d)')
        ax.set_ylabel('XEB Fidelity')
        ax.set_title('(a) Fidelity vs Depth (N=40)')
        ax.grid(True, alpha=0.3)
    
    if not df_scan.empty:
        ax = axes[0, 1]
        stats = df_scan.groupby('N')['fidelity'].agg(['mean', 'std']).reset_index()
        for N in sorted(df_scan['N'].unique()):
            data = df_scan[df_scan['N'] == N]['fidelity']
            ax.scatter([N]*len(data), data, alpha=0.3, s=30, color='steelblue')
        ax.errorbar(stats['N'], stats['mean'], yerr=stats['std'],
                   fmt='s-', color='darkgreen', linewidth=2, capsize=4)
        ax.set_xlabel('Number of Qubits (N)')
        ax.set_ylabel('XEB Fidelity')
        ax.set_title('(b) Fidelity vs N (d=12)')
        ax.grid(True, alpha=0.3)
    
    if not df_n40.empty:
        ax = axes[1, 0]
        depths = sorted(df_n40['d'].unique())
        data_plot = [df_n40[df_n40['d'] == d]['fidelity'].values for d in depths]
        bp = ax.boxplot(data_plot, labels=depths, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Depth (d)')
        ax.set_ylabel('XEB Fidelity')
        ax.set_title('(c) Fidelity Distribution by Depth')
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.hist(df['fidelity'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(df['fidelity'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df["fidelity"].mean():.4f}')
    ax.set_xlabel('XEB Fidelity')
    ax.set_ylabel('Count')
    ax.set_title('(d) Overall Fidelity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(img_dir / 'combined_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: combined_summary.png")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    print(f"\nOverall: {len(df)} instances")
    print(f"Fidelity: {df['fidelity'].mean():.6f} ± {df['fidelity'].std():.6f}")
    print(f"Range: [{df['fidelity'].min():.6f}, {df['fidelity'].max():.6f}]")
    
    if not df_n40.empty:
        print("\nDepth Scan (N=40):")
        summary = df_n40.groupby('d')['fidelity'].agg(['mean', 'std', 'count'])
        print(summary.to_string())
        summary.to_csv(out_dir / 'depth_scan_summary.csv')
    
    if not df_scan.empty:
        print("\nN Scan (d=12):")
        summary = df_scan.groupby('N')['fidelity'].agg(['mean', 'std', 'count'])
        print(summary.to_string())
        summary.to_csv(out_dir / 'n_scan_summary.csv')
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
