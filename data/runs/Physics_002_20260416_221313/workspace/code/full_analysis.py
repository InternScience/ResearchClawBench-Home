#!/usr/bin/env python3
"""
Full XEB Fidelity Analysis for Random Quantum Circuit Sampling

This script processes all XEB configurations and generates:
1. Per-instance fidelity estimates with uncertainties
2. Aggregated results by (N, d) configuration
3. Plots showing fidelity vs depth and fidelity vs qubit count
4. Validation of the quantum supremacy gap conclusion
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))
from xeb_analysis import (
    find_all_xeb_configs, 
    process_all_configs, 
    aggregate_by_config,
    load_counts_file,
    load_amplitudes_file,
    compute_xeb_fidelity,
    bootstrap_xeb_fidelity,
    parse_bitstring_key
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def save_results_json(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    serializable_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            else:
                sr[k] = v
        serializable_results.append(sr)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Saved results to {output_path}")


def save_aggregated_json(aggregated: Dict, output_path: str):
    """Save aggregated results to JSON file."""
    serializable = {}
    for key, val in aggregated.items():
        sk = f"{val['N']}_{val['d']}"
        sv = {}
        for k, v in val.items():
            if k == 'fidelities':
                sv[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            elif isinstance(v, (np.floating, np.integer)):
                sv[k] = float(v)
            else:
                sv[k] = v
        serializable[sk] = sv
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved aggregated results to {output_path}")


def plot_fidelity_vs_depth(results: List[Dict], aggregated: Dict, output_dir: str):
    """Plot fidelity as a function of circuit depth for fixed N."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by N
    by_n = defaultdict(list)
    for r in results:
        if r['fidelity'] is not None:
            by_n[r['N']].append(r)
    
    # Plot for each N
    for n in sorted(by_n.keys()):
        configs = by_n[n]
        
        # Group by depth
        by_d = defaultdict(list)
        for c in configs:
            by_d[c['d']].append(c)
        
        depths = sorted(by_d.keys())
        mean_fids = []
        std_fids = []
        
        for d in depths:
            fids = [c['fidelity'] for c in by_d[d]]
            mean_fids.append(np.mean(fids))
            std_fids.append(np.std(fids) / np.sqrt(len(fids)))  # SEM
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(depths, mean_fids, yerr=std_fids, fmt='o-', capsize=5, 
                    label=f'N={n} qubits', linewidth=2, markersize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Uniform distribution')
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Ideal circuit')
        ax.set_xlabel('Circuit Depth (d)', fontsize=12)
        ax.set_ylabel('XEB Fidelity (F_XEB)', fontsize=12)
        ax.set_title(f'Fidelity vs Depth for N={n} Qubits', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_ylim(-0.1, max(1.1, max(mean_fids) + 0.2))
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fidelity_vs_depth_N{n}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    # Combined plot for all N
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(by_n)))
    
    for i, n in enumerate(sorted(by_n.keys())):
        configs = by_n[n]
        by_d = defaultdict(list)
        for c in configs:
            by_d[c['d']].append(c)
        
        depths = sorted(by_d.keys())
        mean_fids = [np.mean([c['fidelity'] for c in by_d[d]]) for d in depths]
        std_fids = [np.std([c['fidelity'] for c in by_d[d]]) / np.sqrt(len(by_d[d])) for d in depths]
        
        ax.errorbar(depths, mean_fids, yerr=std_fids, fmt='o-', capsize=4, 
                    label=f'N={n}', linewidth=2, markersize=6, color=colors[i])
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Circuit Depth (d)', fontsize=12)
    ax.set_ylabel('XEB Fidelity (F_XEB)', fontsize=12)
    ax.set_title('Fidelity vs Depth for All Qubit Counts', fontsize=14)
    ax.legend(loc='upper right', title='Qubits')
    ax.set_ylim(-0.1, 1.2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fidelity_vs_depth_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_fidelity_vs_n(results: List[Dict], aggregated: Dict, output_dir: str):
    """Plot fidelity as a function of qubit count for fixed depth."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by depth
    by_d = defaultdict(list)
    for r in results:
        if r['fidelity'] is not None:
            by_d[r['d']].append(r)
    
    # Plot for each depth
    for d in sorted(by_d.keys()):
        configs = by_d[d]
        
        # Group by N
        by_n = defaultdict(list)
        for c in configs:
            by_n[c['N']].append(c)
        
        ns = sorted(by_n.keys())
        mean_fids = []
        std_fids = []
        
        for n in ns:
            fids = [c['fidelity'] for c in by_n[n]]
            mean_fids.append(np.mean(fids))
            std_fids.append(np.std(fids) / np.sqrt(len(fids)))  # SEM
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(ns, mean_fids, yerr=std_fids, fmt='o-', capsize=5, 
                    label=f'd={d}', linewidth=2, markersize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Uniform distribution')
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Ideal circuit')
        ax.set_xlabel('Number of Qubits (N)', fontsize=12)
        ax.set_ylabel('XEB Fidelity (F_XEB)', fontsize=12)
        ax.set_title(f'Fidelity vs Qubit Count for Depth d={d}', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_ylim(-0.1, max(1.1, max(mean_fids) + 0.2))
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fidelity_vs_N_d{d}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    # Combined plot for all depths
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.plasma(np.linspace(0, 1, len(by_d)))
    
    for i, d in enumerate(sorted(by_d.keys())):
        configs = by_d[d]
        by_n = defaultdict(list)
        for c in configs:
            by_n[c['N']].append(c)
        
        ns = sorted(by_n.keys())
        mean_fids = [np.mean([c['fidelity'] for c in by_n[n]]) for n in ns]
        std_fids = [np.std([c['fidelity'] for c in by_n[n]]) / np.sqrt(len(by_n[n])) for n in ns]
        
        ax.errorbar(ns, mean_fids, yerr=std_fids, fmt='o-', capsize=4, 
                    label=f'd={d}', linewidth=2, markersize=6, color=colors[i])
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Qubits (N)', fontsize=12)
    ax.set_ylabel('XEB Fidelity (F_XEB)', fontsize=12)
    ax.set_title('Fidelity vs Qubit Count for All Depths', fontsize=14)
    ax.legend(loc='upper right', title='Depth')
    ax.set_ylim(-0.1, 1.2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fidelity_vs_N_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_instance_distribution(aggregated: Dict, output_dir: str):
    """Plot distribution of fidelities across instances for each (N, d)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a few representative (N, d) configurations
    configs_to_plot = []
    for key, val in aggregated.items():
        if val['n_instances'] >= 10:  # Only plot if enough instances
            configs_to_plot.append(val)
    
    if not configs_to_plot:
        return
    
    # Sort by N then d
    configs_to_plot.sort(key=lambda x: (x['N'], x['d']))
    
    # Plot up to 9 configurations
    n_plots = min(9, len(configs_to_plot))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, config in enumerate(configs_to_plot[:n_plots]):
        ax = axes[i]
        fidelities = config['fidelities']
        
        ax.hist(fidelities, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(config['mean_fidelity'], color='red', linestyle='--', 
                   label=f"Mean: {config['mean_fidelity']:.3f}")
        ax.set_xlabel('XEB Fidelity', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'N={config["N"]}, d={config["d"]} (n={config["n_instances"]})', fontsize=12)
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Distribution of XEB Fidelities Across Circuit Instances', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fidelity_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_exponential_decay(aggregated: Dict, output_dir: str):
    """
    Plot fidelity decay with depth and fit exponential model.
    
    The expected decay is: F(d) ≈ exp(-λ * d)
    where λ is related to the error rate per cycle.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by N
    by_n = defaultdict(list)
    for key, val in aggregated.items():
        by_n[val['N']].append(val)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(by_n)))
    
    for i, n in enumerate(sorted(by_n.keys())):
        configs = by_n[n]
        configs.sort(key=lambda x: x['d'])
        
        depths = np.array([c['d'] for c in configs])
        fidelities = np.array([c['mean_fidelity'] for c in configs])
        
        # Filter positive fidelities for log fit
        mask = fidelities > 0
        if np.sum(mask) < 3:
            continue
        
        # Fit exponential decay: log(F) = -λ * d
        log_fids = np.log(fidelities[mask])
        depth_mask = depths[mask]
        
        try:
            coeffs = np.polyfit(depth_mask, log_fids, 1)
            lambda_fit = -coeffs[0]
            
            # Plot data and fit
            ax.errorbar(depths, fidelities, fmt='o', label=f'N={n} data', 
                       color=colors[i], markersize=8)
            
            d_fit = np.linspace(min(depths), max(depths), 100)
            f_fit = np.exp(coeffs[0] * d_fit + coeffs[1])
            ax.plot(d_fit, f_fit, '--', color=colors[i], 
                   label=f'N={n} fit (λ={lambda_fit:.4f})', linewidth=2)
        except Exception as e:
            print(f"Fit failed for N={n}: {e}")
            ax.errorbar(depths, fidelities, fmt='o', label=f'N={n}', 
                       color=colors[i], markersize=8)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Circuit Depth (d)', fontsize=12)
    ax.set_ylabel('XEB Fidelity (F_XEB)', fontsize=12)
    ax.set_title('Exponential Fidelity Decay with Circuit Depth', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'exponential_decay_fit.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(aggregated: Dict, output_path: str):
    """Generate a summary table of results."""
    lines = ["# XEB Fidelity Summary Table", ""]
    lines.append("| N (qubits) | d (depth) | n_instances | Mean F_XEB | Std F_XEB | Min F_XEB | Max F_XEB |")
    lines.append("|------------|-----------|-------------|------------|-----------|-----------|-----------|")
    
    for key in sorted(aggregated.keys()):
        config = aggregated[key]
        lines.append(f"| {config['N']:10} | {config['d']:9} | {config['n_instances']:11} | "
                    f"{config['mean_fidelity']:10.4f} | {config['std_fidelity_across_instances']:9.4f} | "
                    f"{config['min_fidelity']:9.4f} | {config['max_fidelity']:9.4f} |")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved summary table to {output_path}")


def main():
    base_dir = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_002_20260416_221313/data'
    outputs_dir = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_002_20260416_221313/outputs'
    images_dir = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_002_20260416_221313/report/images'
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print("=" * 60)
    print("XEB Fidelity Analysis for Random Quantum Circuit Sampling")
    print("=" * 60)
    
    # Find all configurations
    print("\n1. Finding all XEB configurations...")
    configs = find_all_xeb_configs(base_dir)
    print(f"   Found {len(configs)} XEB configurations")
    
    # Process all configurations (use fewer bootstrap samples for speed)
    print("\n2. Computing XEB fidelities...")
    results = process_all_configs(configs, n_bootstrap=200)
    valid_results = [r for r in results if r['fidelity'] is not None]
    print(f"   Successfully processed {len(valid_results)} configurations")
    
    # Save individual results
    results_path = os.path.join(outputs_dir, 'xeb_results.json')
    save_results_json(results, results_path)
    
    # Aggregate by (N, d)
    print("\n3. Aggregating results by (N, d) configuration...")
    aggregated = aggregate_by_config(results)
    print(f"   Aggregated into {len(aggregated)} (N, d) groups")
    
    # Save aggregated results
    agg_path = os.path.join(outputs_dir, 'xeb_aggregated.json')
    save_aggregated_json(aggregated, agg_path)
    
    # Generate plots
    print("\n4. Generating visualization plots...")
    plot_fidelity_vs_depth(results, aggregated, images_dir)
    plot_fidelity_vs_n(results, aggregated, images_dir)
    plot_instance_distribution(aggregated, images_dir)
    plot_exponential_decay(aggregated, images_dir)
    
    # Generate summary table
    print("\n5. Generating summary table...")
    summary_path = os.path.join(outputs_dir, 'summary_table.md')
    generate_summary_table(aggregated, summary_path)
    
    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    # Find configurations with highest and lowest fidelities
    if aggregated:
        best_config = max(aggregated.values(), key=lambda x: x['mean_fidelity'])
        worst_config = min(aggregated.values(), key=lambda x: x['mean_fidelity'])
        
        print(f"\nHighest fidelity: N={best_config['N']}, d={best_config['d']}")
        print(f"  Mean F_XEB = {best_config['mean_fidelity']:.4f} ± {best_config['std_fidelity_across_instances']:.4f}")
        
        print(f"\nLowest fidelity: N={worst_config['N']}, d={worst_config['d']}")
        print(f"  Mean F_XEB = {worst_config['mean_fidelity']:.4f} ± {worst_config['std_fidelity_across_instances']:.4f}")
    
    # Show depth scan for N=40
    n40_configs = [(k, v) for k, v in aggregated.items() if v['N'] == 40]
    if n40_configs:
        print("\nN=40 depth scan:")
        n40_configs.sort(key=lambda x: x[1]['d'])
        for key, config in n40_configs:
            print(f"  d={config['d']:2}: F_XEB = {config['mean_fidelity']:.4f} ± {config['std_fidelity_across_instances']:.4f} (n={config['n_instances']})")
    
    # Show N scan for d=12
    d12_configs = [(k, v) for k, v in aggregated.items() if v['d'] == 12]
    if d12_configs:
        print("\nd=12 qubit count scan:")
        d12_configs.sort(key=lambda x: x[1]['N'])
        for key, config in d12_configs:
            print(f"  N={config['N']:2}: F_XEB = {config['mean_fidelity']:.4f} ± {config['std_fidelity_across_instances']:.4f} (n={config['n_instances']})")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {outputs_dir}")
    print(f"Figures saved to: {images_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
