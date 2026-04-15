#!/usr/bin/env python3
"""
XEB Fidelity Analysis for Random Quantum Circuit Sampling (RCS)

This script computes cross-entropy benchmarking (XEB) fidelities from experimental
measurement results and ideal amplitude distributions across different qubit counts (N),
circuit depths (d), and instance indices (r).

XEB Formula: F_XEB = 2^N * <P(x_i)>_i - 1

where:
- N is the number of qubits
- P(x_i) = |amplitude(x_i)|^2 is the ideal probability of bitstring x_i
- The average is over observed bitstrings from the experiment
"""

import json
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set publication-quality styling
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

WORKSPACE = os.environ.get('WORKSPACE', '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_002_20260415_115036')
RESULTS_DIR = os.path.join(WORKSPACE, 'data/results')
AMPLITUDES_DIR = os.path.join(WORKSPACE, 'data/amplitudes')
OUTPUTS_DIR = os.path.join(WORKSPACE, 'outputs')
REPORT_IMAGES_DIR = os.path.join(WORKSPACE, 'report/images')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)


def parse_complex_amplitude(amp_str):
    """Parse a complex amplitude string like '(-1.34e-06+2.45e-07j)' into a complex number."""
    amp_str = amp_str.strip()
    if amp_str.startswith('(') and amp_str.endswith(')'):
        amp_str = amp_str[1:-1]
    return complex(amp_str)


def compute_xeb_fidelity(counts_dict, amplitudes_dict, n_qubits):
    """
    Compute XEB fidelity for a single circuit instance.
    
    Args:
        counts_dict: dict mapping bitstring keys to measurement counts
        amplitudes_dict: dict mapping bitstring keys to complex amplitude strings
        n_qubits: number of qubits in the circuit
    
    Returns:
        fidelity: XEB fidelity estimate
        uncertainty: Standard error of the fidelity estimate
        n_matched: Number of matched bitstrings between counts and amplitudes
    """
    # Find common bitstrings
    common_keys = set(counts_dict.keys()) & set(amplitudes_dict.keys())
    n_matched = len(common_keys)
    
    if n_matched == 0:
        return 0.0, float('inf'), 0
    
    # Compute ideal probabilities for matched bitstrings
    probs = []
    total_counts = 0
    
    for key in common_keys:
        count = counts_dict[key]
        total_counts += count
        amp = parse_complex_amplitude(amplitudes_dict[key])
        prob = abs(amp) ** 2
        probs.append(prob)
    
    probs = np.array(probs)
    
    # XEB fidelity: F = 2^N * mean(P(x_i)) - 1
    mean_prob = np.mean(probs)
    fidelity = (2 ** n_qubits) * mean_prob - 1.0
    
    # Uncertainty: sigma_F = 2^N * std(P) / sqrt(n_samples)
    if n_matched > 1:
        std_prob = np.std(probs, ddof=1)
        uncertainty = (2 ** n_qubits) * std_prob / np.sqrt(n_matched)
    else:
        uncertainty = float('inf')
    
    return fidelity, uncertainty, n_matched


def discover_configurations():
    """
    Discover all available (N, d, r) configurations from the data directory.
    Returns a list of tuples: (n_qubits, depth, instance_idx, results_path, amplitudes_path)
    """
    configs = []
    
    # Pattern 1: N40_verification/N40_d{d}_XEB/N40_d{d}_r{r}_XEB_counts.json
    # Pattern 2: N_scan_depth12/N{N}_d12_XEB/N{N}_d12_r{r}_XEB_counts.json
    # Pattern 3: N56_depths/N56_d{d}_XEB/N56_d{d}_r{r}_XEB_counts.json
    
    patterns = [
        (RESULTS_DIR + '/N40_verification/N40_d{d}_XEB/N40_d{d}_r{r}_XEB_counts.json',
         AMPLITUDES_DIR + '/N40_verification/N40_d{d}_XEB/N40_d{d}_r{r}_XEB_amplitudes.json',
         40),
        (RESULTS_DIR + '/N_scan_depth12/N{N}_d12_XEB/N{N}_d12_r{r}_XEB_counts.json',
         AMPLITUDES_DIR + '/N_scan_depth12/N{N}_d12_XEB/N{N}_d12_r{r}_XEB_amplitudes.json',
         None),
        (RESULTS_DIR + '/N56_depths/N56_d{d}_XEB/N56_d{d}_r{r}_XEB_counts.json',
         AMPLITUDES_DIR + '/N56_depths/N56_d{d}_XEB/N56_d{d}_r{r}_XEB_amplitudes.json',
         56),
    ]
    
    for result_pattern, amp_pattern, fixed_n in patterns:
        # Find all matching result files
        if fixed_n is not None:
            # For fixed N patterns, we need to iterate over depths
            # Extract depth from pattern
            depth_match = re.search(r'd(\d+)', result_pattern)
            if depth_match:
                # This is a specific depth pattern, use glob directly
                result_files = glob.glob(result_pattern.replace('{d}', depth_match.group(1)).replace('{r}', '*'))
                for rf in result_files:
                    r_match = re.search(r'_r(\d+)_', rf)
                    if r_match:
                        r_idx = int(r_match.group(1))
                        af = amp_pattern.replace('{d}', depth_match.group(1)).replace('{r}', str(r_idx))
                        if os.path.exists(af):
                            configs.append((fixed_n, int(depth_match.group(1)), r_idx, rf, af))
            else:
                # Need to find all depths
                base_dir = os.path.dirname(result_pattern).replace('{d}', '*')
                depth_dirs = glob.glob(base_dir)
                for dd in depth_dirs:
                    d_match = re.search(r'_(\d+)_', os.path.basename(dd))
                    if d_match:
                        d = int(d_match.group(1))
                        rf_pattern = result_pattern.replace('{d}', str(d)).replace('{r}', '*')
                        result_files = glob.glob(rf_pattern)
                        for rf in result_files:
                            r_match = re.search(r'_r(\d+)_', rf)
                            if r_match:
                                r_idx = int(r_match.group(1))
                                af = amp_pattern.replace('{d}', str(d)).replace('{r}', str(r_idx))
                                if os.path.exists(af):
                                    configs.append((fixed_n, d, r_idx, rf, af))
        else:
            # For N-scan patterns, extract N from the pattern
            n_match = re.search(r'N(\d+)', result_pattern)
            if n_match:
                n = int(n_match.group(1))
                result_files = glob.glob(result_pattern.replace('{N}', str(n)).replace('{r}', '*'))
                for rf in result_files:
                    r_match = re.search(r'_r(\d+)_', rf)
                    if r_match:
                        r_idx = int(r_match.group(1))
                        af = amp_pattern.replace('{N}', str(n)).replace('{r}', str(r_idx))
                        if os.path.exists(af):
                            configs.append((n, 12, r_idx, rf, af))
            else:
                # Need to find all N values
                base_dir = os.path.dirname(result_pattern).replace('{N}', '*')
                n_dirs = glob.glob(base_dir)
                for nd in n_dirs:
                    n_match = re.search(r'N(\d+)', os.path.basename(nd))
                    if n_match:
                        n = int(n_match.group(1))
                        rf_pattern = result_pattern.replace('{N}', str(n)).replace('{r}', '*')
                        result_files = glob.glob(rf_pattern)
                        for rf in result_files:
                            r_match = re.search(r'_r(\d+)_', rf)
                            if r_match:
                                r_idx = int(r_match.group(1))
                                af = amp_pattern.replace('{N}', str(n)).replace('{r}', str(r_idx))
                                if os.path.exists(af):
                                    configs.append((n, 12, r_idx, rf, af))
    
    return configs


def process_all_data():
    """Process all discovered configurations and compute XEB fidelities."""
    print("Discovering configurations...")
    configs = discover_configurations()
    print(f"Found {len(configs)} configurations")
    
    results = []
    
    for n_qubits, depth, r_idx, results_path, amplitudes_path in configs:
        try:
            with open(results_path, 'r') as f:
                counts = json.load(f)
            with open(amplitudes_path, 'r') as f:
                amplitudes = json.load(f)
            
            fidelity, uncertainty, n_matched = compute_xeb_fidelity(counts, amplitudes, n_qubits)
            
            results.append({
                'N': n_qubits,
                'd': depth,
                'r': r_idx,
                'fidelity': fidelity,
                'uncertainty': uncertainty,
                'n_matched': n_matched,
                'n_total_counts': len(counts),
                'results_file': os.path.basename(results_path),
            })
        except Exception as e:
            print(f"Error processing N={n_qubits}, d={depth}, r={r_idx}: {e}")
            continue
    
    return results


def save_results(results):
    """Save results to JSON file."""
    output_path = os.path.join(OUTPUTS_DIR, 'xeb_fidelities.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return output_path


def generate_figures(results):
    """Generate publication-quality figures."""
    
    # Convert to numpy arrays for easier manipulation
    N_vals = np.array([r['N'] for r in results])
    d_vals = np.array([r['d'] for r in results])
    f_vals = np.array([r['fidelity'] for r in results])
    u_vals = np.array([r['uncertainty'] for r in results])
    
    # Figure 1: Fidelity vs Depth for N=40
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    n40_mask = N_vals == 40
    depths_unique = sorted(np.unique(d_vals[n40_mask]))
    
    n40_means = []
    n40_stds = []
    n40_sems = []
    
    for d in depths_unique:
        mask = (N_vals == 40) & (d_vals == d)
        fid_sub = f_vals[mask]
        # Filter out extreme outliers
        fid_sub = fid_sub[np.isfinite(fid_sub)]
        if len(fid_sub) > 0:
            n40_means.append(np.mean(fid_sub))
            n40_stds.append(np.std(fid_sub, ddof=1))
            n40_sems.append(np.std(fid_sub, ddof=1) / np.sqrt(len(fid_sub)))
        else:
            n40_means.append(0)
            n40_stds.append(0)
            n40_sems.append(0)
    
    ax1.errorbar(depths_unique, n40_means, yerr=n40_sems, fmt='o-', 
                 capsize=5, markersize=8, linewidth=2, color='#1f77b4',
                 label='N=40 (Experimental)')
    ax1.set_xlabel('Circuit Depth (d)')
    ax1.set_ylabel('XEB Fidelity')
    ax1.set_title('Fidelity vs Circuit Depth for N=40 Qubits')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add exponential decay fit
    if len(depths_unique) > 2:
        popt = np.polyfit(depths_unique, np.log(np.maximum(n40_means, 1e-10)), 1)
        fit_x = np.linspace(min(depths_unique), max(depths_unique), 100)
        fit_y = np.exp(popt[1]) * np.exp(popt[0] * fit_x)
        ax1.plot(fit_x, fit_y, '--', color='#ff7f0e', linewidth=2,
                 label=f'Exponential fit: exp({popt[1]:.3f} + {popt[0]:.3f}·d)')
        ax1.legend()
    
    fig1.savefig(os.path.join(REPORT_IMAGES_DIR, 'fidelity_vs_depth_N40.png'), 
                 bbox_inches='tight', dpi=150)
    plt.close(fig1)
    print("Saved: fidelity_vs_depth_N40.png")
    
    # Figure 2: Fidelity vs Qubit Count for d=12
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    d12_mask = d_vals == 12
    n_unique = sorted(np.unique(N_vals[d12_mask]))
    
    n_means = []
    n_sems = []
    
    for n in n_unique:
        mask = (N_vals == n) & (d_vals == 12)
        fid_sub = f_vals[mask]
        fid_sub = fid_sub[np.isfinite(fid_sub)]
        if len(fid_sub) > 0:
            n_means.append(np.mean(fid_sub))
            n_sems.append(np.std(fid_sub, ddof=1) / np.sqrt(len(fid_sub)))
        else:
            n_means.append(0)
            n_sems.append(0)
    
    ax2.errorbar(n_unique, n_means, yerr=n_sems, fmt='s-', 
                 capsize=5, markersize=8, linewidth=2, color='#2ca02c',
                 label='d=12 (Experimental)')
    ax2.set_xlabel('Number of Qubits (N)')
    ax2.set_ylabel('XEB Fidelity')
    ax2.set_title('Fidelity vs Qubit Count at Fixed Depth d=12')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add exponential decay fit
    if len(n_unique) > 2:
        popt = np.polyfit(n_unique, np.log(np.maximum(n_means, 1e-10)), 1)
        fit_x = np.linspace(min(n_unique), max(n_unique), 100)
        fit_y = np.exp(popt[1]) * np.exp(popt[0] * fit_x)
        ax2.plot(fit_x, fit_y, '--', color='#ff7f0e', linewidth=2,
                 label=f'Exponential fit: exp({popt[1]:.3f} + {popt[0]:.3f}·N)')
        ax2.legend()
    
    fig2.savefig(os.path.join(REPORT_IMAGES_DIR, 'fidelity_vs_qubit_count_d12.png'), 
                 bbox_inches='tight', dpi=150)
    plt.close(fig2)
    print("Saved: fidelity_vs_qubit_count_d12.png")
    
    # Figure 3: Combined comparison - Fidelity heatmap (N x d)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Create matrix of mean fidelities
    all_depths = sorted(np.unique(d_vals))
    all_n = sorted(np.unique(N_vals))
    
    fidelity_matrix = np.zeros((len(all_n), len(all_depths)))
    sem_matrix = np.zeros((len(all_n), len(all_depths)))
    count_matrix = np.zeros((len(all_n), len(all_depths)))
    
    for i, n in enumerate(all_n):
        for j, d in enumerate(all_depths):
            mask = (N_vals == n) & (d_vals == d)
            fid_sub = f_vals[mask]
            fid_sub = fid_sub[np.isfinite(fid_sub)]
            if len(fid_sub) > 0:
                fidelity_matrix[i, j] = np.mean(fid_sub)
                sem_matrix[i, j] = np.std(fid_sub, ddof=1) / np.sqrt(len(fid_sub))
                count_matrix[i, j] = len(fid_sub)
    
    im = ax3.imshow(fidelity_matrix, aspect='auto', cmap='viridis', 
                    extent=[min(all_depths)-0.5, max(all_depths)+0.5, 
                            min(all_n)-0.5, max(all_n)+0.5])
    ax3.set_xlabel('Circuit Depth (d)')
    ax3.set_ylabel('Number of Qubits (N)')
    ax3.set_title('XEB Fidelity Heatmap: N × d')
    ax3.set_xticks(all_depths)
    ax3.set_yticks(all_n)
    
    # Add text annotations
    for i in range(len(all_n)):
        for j in range(len(all_depths)):
            if count_matrix[i, j] > 0:
                text = ax3.text(j + min(all_depths), i + min(all_n),
                               f'{fidelity_matrix[i, j]:.4f}',
                               ha='center', va='center', fontsize=8,
                               color='white' if fidelity_matrix[i, j] > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax3, label='Mean XEB Fidelity')
    fig3.savefig(os.path.join(REPORT_IMAGES_DIR, 'fidelity_heatmap.png'), 
                 bbox_inches='tight', dpi=150)
    plt.close(fig3)
    print("Saved: fidelity_heatmap.png")
    
    # Figure 4: Instance-level fidelity distribution for representative configs
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Select representative configurations
    rep_configs = [
        (40, 8, 'N=40, d=8'),
        (40, 14, 'N=40, d=14'),
        (16, 12, 'N=16, d=12'),
        (56, 12, 'N=56, d=12'),
    ]
    
    for idx, (n, d, label) in enumerate(rep_configs):
        mask = (N_vals == n) & (d_vals == d)
        fid_sub = f_vals[mask]
        fid_sub = fid_sub[np.isfinite(fid_sub)]
        
        if len(fid_sub) > 0:
            axes[idx].hist(fid_sub, bins=15, alpha=0.7, edgecolor='black', color='#1f77b4')
            axes[idx].axvline(np.mean(fid_sub), color='red', linestyle='--', 
                             linewidth=2, label=f'Mean = {np.mean(fid_sub):.4f}')
            axes[idx].set_xlabel('XEB Fidelity')
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'{label}\n(n={len(fid_sub)} instances)')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    fig4.suptitle('Instance-Level Fidelity Distributions', fontsize=16, fontweight='bold')
    fig4.tight_layout()
    fig4.savefig(os.path.join(REPORT_IMAGES_DIR, 'fidelity_distributions.png'), 
                 bbox_inches='tight', dpi=150)
    plt.close(fig4)
    print("Saved: fidelity_distributions.png")
    
    # Save summary statistics
    summary = {
        'N40_depth_scan': {
            'depths': depths_unique,
            'mean_fidelities': n40_means,
            'std_fidelities': n40_stds,
            'sem_fidelities': n40_sems,
        },
        'd12_N_scan': {
            'qubit_counts': n_unique,
            'mean_fidelities': n_means,
            'sem_fidelities': n_sems,
        },
        'heatmap_data': {
            'qubit_counts': all_n,
            'depths': all_depths,
            'mean_fidelities': fidelity_matrix.tolist(),
            'sem_fidelities': sem_matrix.tolist(),
            'instance_counts': count_matrix.tolist(),
        }
    }
    
    summary_path = os.path.join(OUTPUTS_DIR, 'summary_statistics.json')
    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=convert_numpy)
    print(f"Summary saved to {summary_path}")


def main():
    print("=" * 60)
    print("XEB Fidelity Analysis for RCS")
    print("=" * 60)
    
    results = process_all_data()
    print(f"\nProcessed {len(results)} configurations successfully")
    
    if len(results) > 0:
        save_results(results)
        generate_figures(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        # Group by (N, d)
        from collections import defaultdict
        groups = defaultdict(list)
        for r in results:
            key = (r['N'], r['d'])
            groups[key].append(r['fidelity'])
        
        for (n, d), fids in sorted(groups.items()):
            fids = np.array(fids)
            fids = fids[np.isfinite(fids)]
            if len(fids) > 0:
                print(f"N={n:2d}, d={d:2d}: mean={np.mean(fids):.6f}, "
                      f"std={np.std(fids):.6f}, n={len(fids)}")
    else:
        print("ERROR: No results were processed!")


if __name__ == '__main__':
    main()
