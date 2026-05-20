#!/usr/bin/env python3
"""
Cross-Entropy Benchmarking (XEB) Fidelity Analysis for Random Quantum Circuit Sampling.

Implements the fidelity estimation workflow from the Google Quantum Supremacy paper
(Arute et al., Nature 2019) and related works.

Computes F_XEB = 2^N * <P(x_i)> - 1 for each (N, d, r) configuration,
where <P(x_i)> is the average ideal probability of measured bitstrings.

Analyzes:
  - N=40 verification: scan depth d = 8, 10, 12, 14, 16, 18, 20
  - N-scan at d=12: N = 16, 24, 32, 40
"""

import json
import os
import re
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
FIG_DIR = "report/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Utility Functions ──────────────────────────────────────────────────────

def parse_complex(s):
    """Parse a complex number string like '(1.23e-07+4.56e-08j)'."""
    # Remove outer parentheses and whitespace
    s = s.strip().strip('()')
    return complex(s)

def bitstring_key_to_tuple(key):
    """Convert a tuple-string key to a canonical string for matching.
    We keep it as-is since both files use the same string format."""
    return key

def compute_xeb_fidelity(counts_dict, amplitudes_dict, N):
    """
    Compute XEB fidelity for a single circuit instance.
    
    F_XEB = 2^N * sum(w_i * P_i) / sum(w_i) - 1
    
    where w_i are the observed counts and P_i are ideal probabilities.
    
    Args:
        counts_dict: Dict mapping bitstring strings to integer counts
        amplitudes_dict: Dict mapping bitstring strings to complex amplitude strings
        N: Number of qubits
    
    Returns:
        dict with fidelity, n_matched, total_counts, mean_prob, etc.
    """
    # Find matching bitstrings
    common_keys = set(counts_dict.keys()) & set(amplitudes_dict.keys())
    
    if len(common_keys) == 0:
        return {
            'fidelity': np.nan,
            'n_matched': 0,
            'total_counts': sum(counts_dict.values()),
            'mean_prob': np.nan,
        }
    
    # Compute probabilities from amplitudes
    probs = {}
    for key in common_keys:
        amp = parse_complex(amplitudes_dict[key])
        probs[key] = np.abs(amp) ** 2
    
    # Weighted mean probability
    total_counts = sum(counts_dict[key] for key in common_keys)
    weighted_prob_sum = sum(counts_dict[key] * probs[key] for key in common_keys)
    mean_prob = weighted_prob_sum / total_counts
    
    # XEB fidelity
    fidelity = (2 ** N) * mean_prob - 1
    
    return {
        'fidelity': fidelity,
        'n_matched': len(common_keys),
        'total_counts': total_counts,
        'mean_prob': mean_prob,
    }


def process_configuration(counts_dir, amplitudes_dir, N, d):
    """
    Process all instances for a given (N, d) configuration.
    
    Args:
        counts_dir: Directory containing *_counts.json files
        amplitudes_dir: Directory containing *_amplitudes.json files
        N: Number of qubits
        d: Circuit depth
    
    Returns:
        List of dicts, one per instance
    """
    results = []
    
    # Find all counts files
    counts_files = sorted(glob.glob(os.path.join(counts_dir, "*_counts.json")))
    
    for cf in counts_files:
        # Derive amplitudes filename
        basename = os.path.basename(cf)
        # e.g., N40_d10_r1_XEB_counts.json -> N40_d10_r1_XEB_amplitudes.json
        amp_basename = basename.replace("_counts.json", "_amplitudes.json")
        af = os.path.join(amplitudes_dir, amp_basename)
        
        if not os.path.exists(af):
            continue
        
        # Parse instance index
        match = re.search(r'_r(\d+)_', basename)
        r = int(match.group(1)) if match else -1
        
        # Load data
        with open(cf) as f:
            counts = json.load(f)
        with open(af) as f:
            amplitudes = json.load(f)
        
        # Compute XEB
        result = compute_xeb_fidelity(counts, amplitudes, N)
        result['N'] = N
        result['d'] = d
        result['r'] = r
        result['counts_file'] = cf
        result['amplitudes_file'] = af
        
        results.append(result)
    
    return results


def aggregate_results(results_list):
    """Aggregate per-instance results into per-configuration statistics."""
    agg = {}
    for r in results_list:
        key = (r['N'], r['d'])
        if key not in agg:
            agg[key] = []
        agg[key].append(r)
    
    summary = []
    for (N, d), instances in agg.items():
        fids = np.array([x['fidelity'] for x in instances if not np.isnan(x['fidelity'])])
        if len(fids) == 0:
            continue
        
        summary.append({
            'N': N,
            'd': d,
            'n_instances': len(fids),
            'fidelity_mean': np.mean(fids),
            'fidelity_std': np.std(fids, ddof=1),
            'fidelity_sem': np.std(fids, ddof=1) / np.sqrt(len(fids)),
            'fidelity_median': np.median(fids),
            'fidelity_min': np.min(fids),
            'fidelity_max': np.max(fids),
            'fidelity_q25': np.percentile(fids, 25),
            'fidelity_q75': np.percentile(fids, 75),
        })
    
    return sorted(summary, key=lambda x: (x['N'], x['d']))


# ── Main Analysis ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("XEB Fidelity Analysis")
    print("=" * 70)
    
    all_results = []
    
    # ── N=40 Verification (Depth Scan) ──────────────────────────────────
    print("\n--- N=40 Verification: Depth Scan ---")
    depths = [8, 10, 12, 14, 16, 18, 20]
    
    for d in depths:
        counts_dir = f"data/results/N40_verification/N40_d{d}_XEB"
        amp_dir = f"data/amplitudes/N40_verification/N40_d{d}_XEB"
        
        if not os.path.exists(counts_dir) or not os.path.exists(amp_dir):
            print(f"  d={d}: Missing data, skipping")
            continue
        
        results = process_configuration(counts_dir, amp_dir, N=40, d=d)
        all_results.extend(results)
        
        fids = [r['fidelity'] for r in results]
        valid_fids = [f for f in fids if not np.isnan(f)]
        print(f"  d={d:2d}: {len(valid_fids)}/{len(results)} valid, "
              f"F_XEB = {np.mean(valid_fids):.6f} ± {np.std(valid_fids, ddof=1):.6f}")
    
    # ── N-Scan at d=12 ─────────────────────────────────────────────────
    print("\n--- N-Scan at d=12 ---")
    N_values = [16, 24, 32, 40]
    
    for N in N_values:
        counts_dir = f"data/results/N_scan_depth12/N{N}_d12_XEB"
        amp_dir = f"data/amplitudes/N_scan_depth12/N{N}_d12_XEB"
        
        if not os.path.exists(counts_dir) or not os.path.exists(amp_dir):
            print(f"  N={N}: Missing data, skipping")
            continue
        
        results = process_configuration(counts_dir, amp_dir, N=N, d=12)
        all_results.extend(results)
        
        fids = [r['fidelity'] for r in results]
        valid_fids = [f for f in fids if not np.isnan(f)]
        print(f"  N={N:2d}: {len(valid_fids)}/{len(results)} valid, "
              f"F_XEB = {np.mean(valid_fids):.6f} ± {np.std(valid_fids, ddof=1):.6f}")
    
    # ── Aggregate ──────────────────────────────────────────────────────
    summary = aggregate_results(all_results)
    
    # Save all per-instance results
    with open(os.path.join(OUTPUT_DIR, "xeb_per_instance.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save aggregated summary
    with open(os.path.join(OUTPUT_DIR, "xeb_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    print(f"{'N':>4s} {'d':>4s} {'n':>4s} {'F_mean':>12s} {'F_std':>10s} {'F_sem':>10s} {'F_median':>12s}")
    print("-" * 60)
    for s in summary:
        print(f"{s['N']:4d} {s['d']:4d} {s['n_instances']:4d} "
              f"{s['fidelity_mean']:12.6f} {s['fidelity_std']:10.6f} "
              f"{s['fidelity_sem']:10.6f} {s['fidelity_median']:12.6f}")
    
    # ── Generate Figures ───────────────────────────────────────────────
    generate_figures(summary, all_results)
    
    print("\nAnalysis complete.")
    return summary, all_results


# ── Figure Generation ──────────────────────────────────────────────────────

def generate_figures(summary, all_results):
    """Generate all figures for the report."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })
    
    # Convert to dict for easy lookup
    summary_by_key = {(s['N'], s['d']): s for s in summary}
    
    # ── Figure 1: F_XEB vs Depth for N=40 ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel (a): Fidelity vs depth curve
    ax = axes[0]
    n40_depths = [s['d'] for s in summary if s['N'] == 40]
    n40_fids = [s['fidelity_mean'] for s in summary if s['N'] == 40]
    n40_sems = [s['fidelity_sem'] for s in summary if s['N'] == 40]
    
    # Sort by depth
    sort_idx = np.argsort(n40_depths)
    n40_depths = np.array(n40_depths)[sort_idx]
    n40_fids = np.array(n40_fids)[sort_idx]
    n40_sems = np.array(n40_sems)[sort_idx]
    
    # Exponential decay fit: F(d) = A * exp(-λ * d)
    def exp_decay(d, A, lam):
        return A * np.exp(-lam * d)
    
    try:
        popt, pcov = curve_fit(exp_decay, n40_depths, n40_fids, 
                               p0=[1.0, 0.1], sigma=n40_sems, absolute_sigma=True)
        A_fit, lam_fit = popt
        d_fine = np.linspace(n40_depths[0], n40_depths[-1], 100)
        ax.plot(d_fine, exp_decay(d_fine, A_fit, lam_fit), '--', 
                color='red', alpha=0.7, linewidth=1.5,
                label=f'Exp. fit: $A e^{{-\\lambda d}}$\nA={A_fit:.3f}, λ={lam_fit:.4f}')
        # Effective per-gate error
        # For N=40 random circuits, number of gates ≈ N * d
        # F ≈ exp(-ε * N * d) so ε ≈ λ / N
        eps_eff = lam_fit / 40
        ax.text(0.55, 0.95, f'ε_eff ≈ λ/N = {eps_eff:.6f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        print(f"  Fit warning: {e}")
    
    ax.errorbar(n40_depths, n40_fids, yerr=n40_sems, 
                fmt='o-', capsize=5, capthick=1.5, markersize=8,
                color='steelblue', linewidth=2, markerfacecolor='white',
                markeredgewidth=2, label='N=40 XEB')
    
    ax.set_xlabel('Circuit Depth $d$')
    ax.set_ylabel('XEB Fidelity $\\mathcal{F}_{\\rm XEB}$')
    ax.set_title('(a) Fidelity vs Depth (N=40)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    
    # Panel (b): Per-instance scatter with boxplot overlay
    ax = axes[1]
    instance_data_n40 = [r for r in all_results if r['N'] == 40]
    
    # Organize by depth
    depths_n40 = sorted(set(r['d'] for r in instance_data_n40))
    positions = []
    fidelities = []
    colors = []
    
    for i, d in enumerate(depths_n40):
        fids_d = [r['fidelity'] for r in instance_data_n40 
                  if r['d'] == d and not np.isnan(r['fidelity'])]
        positions.extend([i] * len(fids_d))
        fidelities.extend(fids_d)
    
    # Jitter x positions slightly for visibility
    positions_jittered = np.array(positions) + np.random.uniform(-0.15, 0.15, len(positions))
    
    ax.scatter(positions_jittered, fidelities, alpha=0.4, s=20, 
               color='steelblue', edgecolors='none')
    
    # Boxplots
    bp_data = []
    for d in depths_n40:
        fids_d = [r['fidelity'] for r in instance_data_n40 
                  if r['d'] == d and not np.isnan(r['fidelity'])]
        bp_data.append(fids_d)
    
    bp = ax.boxplot(bp_data, positions=range(len(depths_n40)), widths=0.4,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='lightblue', alpha=0.5),
                    medianprops=dict(color='red', linewidth=2))
    
    ax.set_xticks(range(len(depths_n40)))
    ax.set_xticklabels([str(d) for d in depths_n40])
    ax.set_xlabel('Circuit Depth $d$')
    ax.set_ylabel('XEB Fidelity $\\mathcal{F}_{\\rm XEB}$')
    ax.set_title('(b) Per-Instance Distribution (N=40)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_xeb_vs_depth_n40.png"))
    plt.close()
    print("  Saved: fig1_xeb_vs_depth_n40.png")
    
    # ── Figure 2: F_XEB vs N at d=12 ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel (a): Fidelity vs N
    ax = axes[0]
    nscan_Ns = [s['N'] for s in summary if s['d'] == 12]
    nscan_fids = [s['fidelity_mean'] for s in summary if s['d'] == 12]
    nscan_sems = [s['fidelity_sem'] for s in summary if s['d'] == 12]
    
    sort_idx = np.argsort(nscan_Ns)
    nscan_Ns = np.array(nscan_Ns)[sort_idx]
    nscan_fids = np.array(nscan_fids)[sort_idx]
    nscan_sems = np.array(nscan_sems)[sort_idx]
    
    # Exponential fit: F(N) = A * exp(-λ' * N) at fixed d=12
    try:
        popt_n, pcov_n = curve_fit(exp_decay, nscan_Ns.astype(float), nscan_fids,
                                    p0=[1.0, 0.02], sigma=nscan_sems, absolute_sigma=True)
        A_fit_n, lam_fit_n = popt_n
        N_fine = np.linspace(nscan_Ns[0], nscan_Ns[-1], 100)
        ax.plot(N_fine, exp_decay(N_fine, A_fit_n, lam_fit_n), '--',
                color='red', alpha=0.7, linewidth=1.5,
                label=f'Exp. fit: $A e^{{-\\lambda N}}$\nA={A_fit_n:.3f}, λ={lam_fit_n:.4f}')
        eps_eff_n = lam_fit_n / 12
        ax.text(0.55, 0.95, f'ε_eff ≈ λ/d = {eps_eff_n:.6f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        print(f"  Fit warning (N-scan): {e}")
    
    ax.errorbar(nscan_Ns, nscan_fids, yerr=nscan_sems,
                fmt='s-', capsize=5, capthick=1.5, markersize=8,
                color='darkorange', linewidth=2, markerfacecolor='white',
                markeredgewidth=2, label='d=12 XEB')
    
    ax.set_xlabel('Number of Qubits $N$')
    ax.set_ylabel('XEB Fidelity $\\mathcal{F}_{\\rm XEB}$')
    ax.set_title('(a) Fidelity vs Qubit Count (d=12)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    
    # Panel (b): Per-instance scatter
    ax = axes[1]
    instance_data_nscan = [r for r in all_results if r['d'] == 12]
    Ns_scan = sorted(set(r['N'] for r in instance_data_nscan))
    
    for i, N in enumerate(Ns_scan):
        fids_N = [r['fidelity'] for r in instance_data_nscan
                  if r['N'] == N and not np.isnan(r['fidelity'])]
        pos = np.ones(len(fids_N)) * i + np.random.uniform(-0.15, 0.15, len(fids_N))
        ax.scatter(pos, fids_N, alpha=0.4, s=20, color='darkorange', edgecolors='none')
    
    bp_data_n = []
    for N in Ns_scan:
        fids_N = [r['fidelity'] for r in instance_data_nscan
                  if r['N'] == N and not np.isnan(r['fidelity'])]
        bp_data_n.append(fids_N)
    
    ax.boxplot(bp_data_n, positions=range(len(Ns_scan)), widths=0.4,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor='navajowhite', alpha=0.5),
               medianprops=dict(color='red', linewidth=2))
    
    ax.set_xticks(range(len(Ns_scan)))
    ax.set_xticklabels([str(N) for N in Ns_scan])
    ax.set_xlabel('Number of Qubits $N$')
    ax.set_ylabel('XEB Fidelity $\\mathcal{F}_{\\rm XEB}$')
    ax.set_title('(b) Per-Instance Distribution (d=12)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_xeb_vs_n_d12.png"))
    plt.close()
    print("  Saved: fig2_xeb_vs_n_d12.png")
    
    # ── Figure 3: Combined 2D view ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Create a 2D grid: N vs d with fidelity as color
    all_N = sorted(set(s['N'] for s in summary))
    all_d = sorted(set(s['d'] for s in summary))
    
    fid_matrix = np.full((len(all_N), len(all_d)), np.nan)
    for i, N in enumerate(all_N):
        for j, d in enumerate(all_d):
            key = (N, d)
            if key in summary_by_key:
                fid_matrix[i, j] = summary_by_key[key]['fidelity_mean']
    
    im = ax.imshow(fid_matrix, aspect='auto', origin='lower',
                   cmap='viridis', interpolation='nearest')
    
    ax.set_xticks(range(len(all_d)))
    ax.set_xticklabels([str(d) for d in all_d])
    ax.set_yticks(range(len(all_N)))
    ax.set_yticklabels([str(N) for N in all_N])
    ax.set_xlabel('Circuit Depth $d$')
    ax.set_ylabel('Number of Qubits $N$')
    ax.set_title('XEB Fidelity $\\mathcal{F}_{\\rm XEB}$ Heatmap')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('$\\mathcal{F}_{\\rm XEB}$')
    
    # Annotate values
    for i, N in enumerate(all_N):
        for j, d in enumerate(all_d):
            if not np.isnan(fid_matrix[i, j]):
                ax.text(j, i, f'{fid_matrix[i,j]:.4f}', ha='center', va='center',
                       fontsize=8, color='white' if fid_matrix[i,j] < 0.5 else 'black')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_xeb_heatmap.png"))
    plt.close()
    print("  Saved: fig3_xeb_heatmap.png")
    
    # ── Figure 4: Log-scale fidelity vs Nd product ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    
    Nd_vals = []
    fid_vals = []
    sem_vals = []
    labels = []
    
    for s in summary:
        Nd = s['N'] * s['d']
        Nd_vals.append(Nd)
        fid_vals.append(s['fidelity_mean'])
        sem_vals.append(s['fidelity_sem'])
        labels.append(f"N={s['N']},d={s['d']}")
    
    Nd_vals = np.array(Nd_vals)
    fid_vals = np.array(fid_vals)
    sem_vals = np.array(sem_vals)
    
    # Separate N40 depth scan and N-scan
    n40_mask = np.array([s['N'] == 40 for s in summary])
    nscan_mask = np.array([s['d'] == 12 and s['N'] != 40 for s in summary])
    
    ax.errorbar(Nd_vals[n40_mask], fid_vals[n40_mask], yerr=sem_vals[n40_mask],
                fmt='o', capsize=4, markersize=8, color='steelblue',
                label='N=40 depth scan', markerfacecolor='white', markeredgewidth=2)
    
    ax.errorbar(Nd_vals[nscan_mask], fid_vals[nscan_mask], yerr=sem_vals[nscan_mask],
                fmt='s', capsize=4, markersize=8, color='darkorange',
                label='d=12 N-scan', markerfacecolor='white', markeredgewidth=2)
    
    # Exponential fit on all data (log scale)
    log_fids = np.log(np.maximum(fid_vals, 1e-10))
    valid = np.isfinite(log_fids)
    if np.sum(valid) > 2:
        coeffs = np.polyfit(Nd_vals[valid], log_fids[valid], 1)
        Nd_fine = np.linspace(Nd_vals.min(), Nd_vals.max(), 100)
        ax.plot(Nd_fine, np.exp(coeffs[1]) * np.exp(coeffs[0] * Nd_fine),
                '--', color='red', alpha=0.5, linewidth=1.5,
                label=f'$\\mathcal{{F}} \\approx e^{{{coeffs[0]:.4f} \\cdot Nd {coeffs[1]:+.3f}}}$')
    
    ax.set_xlabel('$N \\cdot d$ (qubit-cycles)')
    ax.set_ylabel('XEB Fidelity $\\mathcal{F}_{\\rm XEB}$')
    ax.set_title('Fidelity Scaling with Computational Volume $N \\cdot d$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_fidelity_vs_Nd.png"))
    plt.close()
    print("  Saved: fig4_fidelity_vs_Nd.png")
    
    # ── Figure 5: Error model analysis ─────────────────────────────────
    # Assuming F = exp(-ε * N * d) where ε is effective per-gate error
    # Linear fit: ln(F) = -ε * N * d + ln(F0)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    valid_mask = fid_vals > 0
    Nd_valid = Nd_vals[valid_mask]
    lnF_valid = np.log(fid_vals[valid_mask])
    
    if len(Nd_valid) > 2:
        coeffs_err = np.polyfit(Nd_valid, lnF_valid, 1)
        eps = -coeffs_err[0]
        F0 = np.exp(coeffs_err[1])
        
        ax.scatter(Nd_valid, lnF_valid, c=['steelblue' if m else 'darkorange' 
                   for m in n40_mask[valid_mask]], s=60, edgecolors='black', linewidth=0.5)
        
        Nd_line = np.linspace(Nd_valid.min(), Nd_valid.max(), 100)
        ax.plot(Nd_line, coeffs_err[0] * Nd_line + coeffs_err[1],
                'r-', linewidth=2, label=f'ln(F) = {-eps:.6f}·Nd + {coeffs_err[1]:.3f}')
        
        ax.text(0.05, 0.05, 
                f'Effective per-gate error rate:\nε = {eps:.6f}\nF₀ = {F0:.4f}\n'
                f'Gate fidelity ≈ 1 - ε = {1-eps:.6f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlabel('$N \\cdot d$ (qubit-cycles)')
    ax.set_ylabel('$\\ln(\\mathcal{F}_{\\rm XEB})$')
    ax.set_title('Error Model: $\\mathcal{F} = F_0 \\cdot e^{-\\varepsilon \\cdot N d}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_error_model.png"))
    plt.close()
    print("  Saved: fig5_error_model.png")
    
    # Save error model parameters
    error_model = {
        'method': 'ln(F) = ln(F0) - epsilon * N * d',
        'epsilon': float(eps),
        'F0': float(F0),
        'gate_fidelity': float(1 - eps),
        'r_squared': float(np.corrcoef(Nd_valid, lnF_valid)[0, 1] ** 2),
    }
    with open(os.path.join(OUTPUT_DIR, "error_model.json"), 'w') as f:
        json.dump(error_model, f, indent=2)
    print(f"  Saved: error_model.json (ε = {eps:.6f})")


if __name__ == "__main__":
    main()
