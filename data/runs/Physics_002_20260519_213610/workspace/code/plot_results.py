"""
Generate comparative plots for XEB fidelity analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150


def load_results():
    with open('outputs/xeb_fidelities.json', 'r') as f:
        return json.load(f)


def aggregate_by_nd(results):
    """Aggregate fidelity estimates by (N, d)."""
    by_nd = defaultdict(list)
    for r in results:
        by_nd[(r['N'], r['d'])].append(r['fidelity'])
    
    aggregated = []
    for (N, d), fids in sorted(by_nd.items()):
        fids = np.array(fids)
        aggregated.append({
            'N': N,
            'd': d,
            'mean_fidelity': float(np.mean(fids)),
            'std_fidelity': float(np.std(fids, ddof=1)),
            'sem_fidelity': float(np.std(fids, ddof=1) / np.sqrt(len(fids))),
            'n_instances': len(fids),
            'median_fidelity': float(np.median(fids)),
            'q25': float(np.percentile(fids, 25)),
            'q75': float(np.percentile(fids, 75)),
        })
    return aggregated


def plot_fidelity_vs_depth_for_N40(results, aggregated):
    """Plot mean XEB fidelity vs circuit depth for N=40."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter for N=40
    nd_data = [a for a in aggregated if a['N'] == 40]
    depths = [a['d'] for a in nd_data]
    means = [a['mean_fidelity'] for a in nd_data]
    sems = [a['sem_fidelity'] for a in nd_data]
    stds = [a['std_fidelity'] for a in nd_data]
    
    # Plot mean with error bars (SEM)
    ax.errorbar(depths, means, yerr=sems, fmt='o-', capsize=5, capthick=2,
                markersize=8, linewidth=2, color='#2E86AB', label='Mean F_XEB ± SEM')
    
    # Also plot individual instances as faint scatter
    n40_results = [r for r in results if r['N'] == 40]
    by_d = defaultdict(list)
    for r in n40_results:
        by_d[r['d']].append(r['fidelity'])
    
    for d, fids in by_d.items():
        jitter = np.random.normal(0, 0.15, len(fids))
        ax.scatter([d + j for j in jitter], fids, alpha=0.3, s=20, color='gray')
    
    # Fit exponential decay model: F = A * exp(-r * g) where g = N * d
    # For simplicity, fit F = A * exp(-k * d)
    if len(depths) >= 3:
        def exp_decay(d, A, k):
            return A * np.exp(-k * d)
        
        try:
            popt, pcov = curve_fit(exp_decay, depths, means, sigma=sems, 
                                   p0=[1.0, 0.05], bounds=([0, 0], [2, 1]))
            d_fit = np.linspace(min(depths), max(depths), 200)
            f_fit = exp_decay(d_fit, *popt)
            ax.plot(d_fit, f_fit, '--', color='#E94F37', linewidth=2,
                    label=f'Exp. fit: F = {popt[0]:.3f}·exp(-{popt[1]:.4f}·d)')
        except Exception as e:
            print(f"Fit failed: {e}")
    
    # Classical approximability line (F_XEB = 0)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, label='Classical limit (F_XEB = 0)')
    
    # Ideal limit
    ax.axhline(y=1, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Ideal limit (F_XEB = 1)')
    
    ax.set_xlabel('Circuit depth $d$', fontsize=13)
    ax.set_ylabel('XEB fidelity $\\mathcal{F}_{\\mathrm{XEB}}$', fontsize=13)
    ax.set_title('XEB fidelity vs. circuit depth for N = 40 qubits', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fidelity_vs_depth_N40.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved report/images/fidelity_vs_depth_N40.png")


def plot_fidelity_vs_N_for_d12(results, aggregated):
    """Plot mean XEB fidelity vs qubit count N for fixed depth d=12."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter for d=12
    nd_data = [a for a in aggregated if a['d'] == 12]
    Ns = [a['N'] for a in nd_data]
    means = [a['mean_fidelity'] for a in nd_data]
    sems = [a['sem_fidelity'] for a in nd_data]
    stds = [a['std_fidelity'] for a in nd_data]
    
    # Plot mean with error bars
    ax.errorbar(Ns, means, yerr=sems, fmt='s-', capsize=5, capthick=2,
                markersize=8, linewidth=2, color='#A23B72', label='Mean F_XEB ± SEM')
    
    # Individual instances
    d12_results = [r for r in results if r['d'] == 12]
    by_N = defaultdict(list)
    for r in d12_results:
        by_N[r['N']].append(r['fidelity'])
    
    for N, fids in by_N.items():
        jitter = np.random.normal(0, 0.2, len(fids))
        ax.scatter([N + j for j in jitter], fids, alpha=0.3, s=20, color='gray')
    
    # Exponential decay fit with N: F = A * exp(-k * N)
    if len(Ns) >= 3:
        def exp_decay(N, A, k):
            return A * np.exp(-k * N)
        
        try:
            popt, pcov = curve_fit(exp_decay, Ns, means, sigma=sems,
                                   p0=[1.0, 0.02], bounds=([0, 0], [2, 0.5]))
            N_fit = np.linspace(min(Ns), max(Ns), 200)
            f_fit = exp_decay(N_fit, *popt)
            ax.plot(N_fit, f_fit, '--', color='#E94F37', linewidth=2,
                    label=f'Exp. fit: F = {popt[0]:.3f}·exp(-{popt[1]:.4f}·N)')
        except Exception as e:
            print(f"Fit failed: {e}")
    
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, label='Classical limit (F_XEB = 0)')
    ax.axhline(y=1, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Ideal limit (F_XEB = 1)')
    
    ax.set_xlabel('Number of qubits $N$', fontsize=13)
    ax.set_ylabel('XEB fidelity $\\mathcal{F}_{\\mathrm{XEB}}$', fontsize=13)
    ax.set_title('XEB fidelity vs. qubit count for fixed depth $d$ = 12', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fidelity_vs_N_d12.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved report/images/fidelity_vs_N_d12.png")


def plot_all_instances_distribution(results):
    """Plot distribution/histogram of all fidelity estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    fidelities = [r['fidelity'] for r in results]
    
    # Histogram
    axes[0].hist(fidelities, bins=50, color='#2E86AB', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=0, color='black', linestyle=':', linewidth=2, label='Classical limit')
    axes[0].axvline(x=1, color='green', linestyle=':', linewidth=2, label='Ideal limit')
    axes[0].axvline(x=np.mean(fidelities), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean = {np.mean(fidelities):.3f}')
    axes[0].set_xlabel('XEB fidelity $\\mathcal{F}_{\\mathrm{XEB}}$', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of all fidelity estimates', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by (N, d)
    by_nd = defaultdict(list)
    for r in results:
        by_nd[(r['N'], r['d'])].append(r['fidelity'])
    
    labels = []
    data = []
    for (N, d), fids in sorted(by_nd.items()):
        labels.append(f'N={N}\nd={d}')
        data.append(fids)
    
    bp = axes[1].boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red'))
    for patch in bp['boxes']:
        patch.set_facecolor('#A23B72')
        patch.set_alpha(0.6)
    
    axes[1].axhline(y=0, color='black', linestyle=':', linewidth=1.5, label='Classical limit')
    axes[1].axhline(y=1, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Ideal limit')
    axes[1].set_ylabel('XEB fidelity $\\mathcal{F}_{\\mathrm{XEB}}$', fontsize=12)
    axes[1].set_title('Fidelity distribution by circuit configuration', fontsize=13)
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('report/images/fidelity_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved report/images/fidelity_distributions.png")


def plot_classical_gap_analysis(aggregated):
    """Plot the gap between experimental fidelity and classical approximability."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Group by N and plot fidelity vs depth with classical limit
    by_N = defaultdict(list)
    for a in aggregated:
        by_N[a['N']].append(a)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(by_N)))
    
    for i, (N, data) in enumerate(sorted(by_N.items())):
        data = sorted(data, key=lambda x: x['d'])
        depths = [a['d'] for a in data]
        means = [a['mean_fidelity'] for a in data]
        sems = [a['sem_fidelity'] for a in data]
        
        ax.errorbar(depths, means, yerr=sems, fmt='-o', capsize=4,
                    markersize=6, linewidth=2, color=colors[i],
                    label=f'N = {N}')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Classical approximability (F_XEB = 0)')
    ax.fill_between([0, 25], -0.5, 0, alpha=0.15, color='red', label='Classically simulable region')
    
    ax.set_xlabel('Circuit depth $d$', fontsize=13)
    ax.set_ylabel('XEB fidelity $\\mathcal{F}_{\\mathrm{XEB}}$', fontsize=13)
    ax.set_title('Experimental fidelity vs. classical approximability gap', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.3, 1.2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/classical_gap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved report/images/classical_gap_analysis.png")


def plot_gate_count_model(aggregated):
    """Plot fidelity as a function of total gate count g ~ N*d and fit exponential decay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Approximate gate count: g = N * d (assuming roughly one gate per qubit per cycle)
    gate_counts = [a['N'] * a['d'] for a in aggregated]
    means = [a['mean_fidelity'] for a in aggregated]
    sems = [a['sem_fidelity'] for a in aggregated]
    Ns = [a['N'] for a in aggregated]
    
    # Color by N
    scatter = ax.scatter(gate_counts, means, c=Ns, cmap='viridis', s=100, 
                         edgecolors='black', linewidth=0.5, zorder=3)
    ax.errorbar(gate_counts, means, yerr=sems, fmt='none', ecolor='gray', alpha=0.5, zorder=2)
    
    # Fit: F = exp(-r * g)  (assuming A ≈ 1)
    def exp_gate(g, r):
        return np.exp(-r * g)
    
    try:
        popt, pcov = curve_fit(exp_gate, gate_counts, means, sigma=sems,
                               p0=[0.001], bounds=([0], [0.1]))
        g_fit = np.linspace(min(gate_counts), max(gate_counts), 200)
        f_fit = exp_gate(g_fit, *popt)
        ax.plot(g_fit, f_fit, '--', color='#E94F37', linewidth=2,
                label=f'Fit: F = exp(-{popt[0]:.5f}·g)')
    except Exception as e:
        print(f"Gate count fit failed: {e}")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of qubits $N$', fontsize=12)
    
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, label='Classical limit')
    ax.set_xlabel('Approximate gate count $g \\approx N \\times d$', fontsize=13)
    ax.set_ylabel('XEB fidelity $\\mathcal{F}_{\\mathrm{XEB}}$', fontsize=13)
    ax.set_title('Fidelity decay vs. total gate count', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fidelity_vs_gate_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved report/images/fidelity_vs_gate_count.png")


def plot_instance_variability(results):
    """Plot individual instance fidelities to show variability across random circuit instances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # N=40, varying d
    ax = axes[0]
    n40 = [r for r in results if r['N'] == 40]
    by_d = defaultdict(list)
    for r in n40:
        by_d[r['d']].append(r['fidelity'])
    
    depths = sorted(by_d.keys())
    data = [by_d[d] for d in depths]
    
    bp = ax.boxplot(data, positions=depths, widths=1.2, patch_artist=True, 
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red'))
    for patch in bp['boxes']:
        patch.set_facecolor('#2E86AB')
        patch.set_alpha(0.6)
    
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
    ax.axhline(y=1, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Circuit depth $d$', fontsize=12)
    ax.set_ylabel('XEB fidelity $\\mathcal{F}_{\\mathrm{XEB}}$', fontsize=12)
    ax.set_title('Instance variability: N = 40', fontsize=13)
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3, axis='y')
    
    # d=12, varying N
    ax = axes[1]
    d12 = [r for r in results if r['d'] == 12]
    by_N = defaultdict(list)
    for r in d12:
        by_N[r['N']].append(r['fidelity'])
    
    Ns = sorted(by_N.keys())
    data = [by_N[N] for N in Ns]
    
    bp = ax.boxplot(data, positions=Ns, widths=3, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red'))
    for patch in bp['boxes']:
        patch.set_facecolor('#A23B72')
        patch.set_alpha(0.6)
    
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
    ax.axhline(y=1, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Number of qubits $N$', fontsize=12)
    ax.set_ylabel('XEB fidelity $\\mathcal{F}_{\\mathrm{XEB}}$', fontsize=12)
    ax.set_title('Instance variability: $d$ = 12', fontsize=13)
    ax.set_xticks(Ns)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('report/images/instance_variability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved report/images/instance_variability.png")


def main():
    results = load_results()
    aggregated = aggregate_by_nd(results)
    
    print(f"Loaded {len(results)} individual results")
    print(f"Aggregated into {len(aggregated)} (N, d) configurations")
    
    plot_fidelity_vs_depth_for_N40(results, aggregated)
    plot_fidelity_vs_N_for_d12(results, aggregated)
    plot_all_instances_distribution(results)
    plot_classical_gap_analysis(aggregated)
    plot_gate_count_model(aggregated)
    plot_instance_variability(results)
    
    # Save aggregated results
    with open('outputs/xeb_fidelities_aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    print("\nSaved aggregated results to outputs/xeb_fidelities_aggregated.json")


if __name__ == '__main__':
    main()
