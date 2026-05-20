"""
Visualization of RCS Fidelity Estimation Results
=================================================

Creates publication-quality figures for the RCS fidelity analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Set style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Colors
COLORS = {
    'XEB': '#2196F3',  # Blue
    'MB': '#FF5722',   # Orange-Red
    'theory': '#4CAF50',  # Green
    'gap': '#9C27B0',   # Purple
}

def load_results(output_dir):
    with open(Path(output_dir) / "fidelity_results.json") as f:
        return json.load(f)

def plot_fig1_xeb_vs_depth(results, output_dir):
    """
    Figure 1: XEB Fidelity vs Circuit Depth for N=40
    Shows how fidelity degrades with increasing depth.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter for N=40
    n40_xeb = [r for r in results if r['N'] == 40 and r['method'] == 'XEB']
    n40_mb = [r for r in results if r['N'] == 40 and r['method'] == 'MB']
    
    n40_xeb.sort(key=lambda x: x['d'])
    n40_mb.sort(key=lambda x: x['d'])
    
    depths_xeb = [r['d'] for r in n40_xeb]
    F_xeb = [r['F_mean'] for r in n40_xeb]
    F_xeb_std = [r['F_std'] for r in n40_xeb]
    
    depths_mb = [r['d'] for r in n40_mb]
    P_mb = [r['F_mean'] for r in n40_mb]
    P_mb_std = [r['F_std'] for r in n40_mb]
    
    # Left panel: Both metrics
    ax1 = axes[0]
    ax1.errorbar(depths_xeb, F_xeb, yerr=F_xeb_std, fmt='o-', color=COLORS['XEB'],
                capsize=5, capthick=2, linewidth=2, markersize=8, label=r'$\mathcal{F}_{\mathrm{XEB}}$ (Linear)')
    ax1.errorbar(depths_mb, P_mb, yerr=P_mb_std, fmt='s-', color=COLORS['MB'],
                capsize=5, capthick=2, linewidth=2, markersize=8, label=r'$P_{\mathrm{heavy}}$ (MB)')
    
    # Add classical approximation threshold
    # For N=40, a random classical sampler would give F_XEB ≈ 0
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Classical (random)')
    
    ax1.set_xlabel('Circuit Depth $d$')
    ax1.set_ylabel('Fidelity Metric')
    ax1.set_title(r'$\mathcal{F}_{\mathrm{XEB}}$ and $P_{\mathrm{heavy}}$ vs Depth ($N=40$)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-0.1, 1.0)
    
    # Right panel: Fidelity decay analysis
    ax2 = axes[1]
    
    # Fit exponential decay: F = A * exp(-alpha * d) + F_inf
    depths_arr = np.array(depths_xeb)
    F_arr = np.array(F_xeb)
    
    # Simple exponential fit
    from scipy.optimize import curve_fit
    
    def exp_decay(d, A, alpha, F_inf):
        return A * np.exp(-alpha * d) + F_inf
    
    try:
        popt, pcov = curve_fit(exp_decay, depths_arr, F_arr, p0=[0.5, 0.05, 0.2], maxfev=5000)
        d_fit = np.linspace(min(depths_arr), max(depths_arr), 100)
        F_fit = exp_decay(d_fit, *popt)
        ax2.plot(d_fit, F_fit, '--', color=COLORS['theory'], linewidth=2, label=f'Exp fit: $\\alpha={popt[1]:.3f}$')
        
        # Extrapolate to larger depths
        d_extrap = np.linspace(min(depths_arr), 40, 200)
        F_extrap = exp_decay(d_extrap, *popt)
        ax2.plot(d_extrap, F_extrap, ':', color=COLORS['gap'], linewidth=2, alpha=0.7, label='Extrapolation')
    except:
        pass
    
    ax2.errorbar(depths_xeb, F_xeb, yerr=F_xeb_std, fmt='o', color=COLORS['XEB'],
                capsize=5, capthick=2, markersize=8, label=r'$\mathcal{F}_{\mathrm{XEB}}$ data')
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhspan(-0.1, 0, alpha=0.1, color='red', label='Classical regime')
    
    ax2.set_xlabel('Circuit Depth $d$')
    ax2.set_ylabel(r'$\mathcal{F}_{\mathrm{XEB}}$')
    ax2.set_title(r'XEB Fidelity Decay and Extrapolation ($N=40$)')
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir).parent / "report" / "images" / "figure1_xeb_vs_depth.png")
    plt.close()
    print("Saved figure1_xeb_vs_depth.png")


def plot_fig2_xeb_vs_N(results, output_dir):
    """
    Figure 2: XEB Fidelity vs Number of Qubits for d=12
    Shows how fidelity degrades with system size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter for d=12
    d12_xeb = [r for r in results if r['d'] == 12 and r['method'] == 'XEB']
    d12_mb = [r for r in results if r['d'] == 12 and r['method'] == 'MB']
    
    d12_xeb.sort(key=lambda x: x['N'])
    d12_mb.sort(key=lambda x: x['N'])
    
    N_xeb = [r['N'] for r in d12_xeb]
    F_xeb = [r['F_mean'] for r in d12_xeb]
    F_xeb_std = [r['F_std'] for r in d12_xeb]
    
    N_mb = [r['N'] for r in d12_mb]
    P_mb = [r['F_mean'] for r in d12_mb]
    P_mb_std = [r['F_std'] for r in d12_mb]
    
    # Left panel: Both metrics
    ax1 = axes[0]
    ax1.errorbar(N_xeb, F_xeb, yerr=F_xeb_std, fmt='o-', color=COLORS['XEB'],
                capsize=5, capthick=2, linewidth=2, markersize=8, label=r'$\mathcal{F}_{\mathrm{XEB}}$')
    ax1.errorbar(N_mb, P_mb, yerr=P_mb_std, fmt='s-', color=COLORS['MB'],
                capsize=5, capthick=2, linewidth=2, markersize=8, label=r'$P_{\mathrm{heavy}}$')
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Classical')
    
    ax1.set_xlabel('Number of Qubits $N$')
    ax1.set_ylabel('Fidelity Metric')
    ax1.set_title(r'Fidelity vs System Size ($d=12$)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-0.1, 1.0)
    
    # Right panel: Scaling analysis
    ax2 = axes[1]
    
    N_arr = np.array(N_xeb)
    F_arr = np.array(F_xeb)
    
    # Fit: F = F0 * exp(-beta * N)
    def exp_N(N, F0, beta):
        return F0 * np.exp(-beta * N)
    
    try:
        popt, pcov = curve_fit(exp_N, N_arr, F_arr, p0=[1.0, 0.01], maxfev=5000)
        N_fit = np.linspace(min(N_arr), max(N_arr), 100)
        F_fit = exp_N(N_fit, *popt)
        ax2.plot(N_fit, F_fit, '--', color=COLORS['theory'], linewidth=2, 
                label=f'Exp fit: $\\beta={popt[1]:.4f}$')
        
        # Extrapolate
        N_extrap = np.linspace(min(N_arr), 100, 200)
        F_extrap = exp_N(N_extrap, *popt)
        ax2.plot(N_extrap, F_extrap, ':', color=COLORS['gap'], linewidth=2, alpha=0.7,
                label='Extrapolation')
    except:
        pass
    
    ax2.errorbar(N_xeb, F_xeb, yerr=F_xeb_std, fmt='o', color=COLORS['XEB'],
                capsize=5, capthick=2, markersize=8, label=r'$\mathcal{F}_{\mathrm{XEB}}$ data')
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhspan(-0.1, 0, alpha=0.1, color='red', label='Classical regime')
    
    ax2.set_xlabel('Number of Qubits $N$')
    ax2.set_ylabel(r'$\mathcal{F}_{\mathrm{XEB}}$')
    ax2.set_title(r'XEB Fidelity Scaling ($d=12$)')
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir).parent / "report" / "images" / "figure2_xeb_vs_N.png")
    plt.close()
    print("Saved figure2_xeb_vs_N.png")


def plot_fig3_mb_comparison(results, output_dir):
    """
    Figure 3: Comparison of XEB and MB metrics across configurations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: N40 depth scan - XEB vs MB
    n40_xeb = {r['d']: r for r in results if r['N'] == 40 and r['method'] == 'XEB'}
    n40_mb = {r['d']: r for r in results if r['N'] == 40 and r['method'] == 'MB'}
    
    depths = sorted(set(n40_xeb.keys()) & set(n40_mb.keys()))
    
    ax1 = axes[0]
    xeb_vals = [n40_xeb[d]['F_mean'] for d in depths]
    mb_vals = [n40_mb[d]['F_mean'] for d in depths]
    
    ax1.scatter(xeb_vals, mb_vals, s=100, c=COLORS['XEB'], edgecolors='black', 
               linewidth=1, zorder=5)
    
    # Add depth labels
    for i, d in enumerate(depths):
        ax1.annotate(f'd={d}', (xeb_vals[i], mb_vals[i]), 
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=9, color='gray')
    
    # Perfect correlation line
    ax1.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel(r'$\mathcal{F}_{\mathrm{XEB}}$')
    ax1.set_ylabel(r'$P_{\mathrm{heavy}}$ (MB)')
    ax1.set_title(r'XEB vs MB Fidelity ($N=40$)')
    ax1.set_xlim(-0.1, 1.0)
    ax1.set_ylim(-0.1, 1.0)
    ax1.set_aspect('equal')
    
    # Panel B: N56 depth scan - MB only
    n56_mb = [r for r in results if r['N'] == 56 and r['method'] == 'MB']
    n56_mb.sort(key=lambda x: x['d'])
    
    n56_depths = [r['d'] for r in n56_mb]
    n56_p = [r['F_mean'] for r in n56_mb]
    n56_p_std = [r['F_std'] for r in n56_mb]
    
    ax2 = axes[1]
    ax2.errorbar(n56_depths, n56_p, yerr=n56_p_std, fmt='o-', color=COLORS['MB'],
                capsize=5, capthick=2, linewidth=2, markersize=8)
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
               label='50% threshold')
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Circuit Depth $d$')
    ax2.set_ylabel(r'$P_{\mathrm{heavy}}$ (MB)')
    ax2.set_title(r'MB Fidelity ($N=56$)')
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.05, 1.0)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir).parent / "report" / "images" / "figure3_mb_comparison.png")
    plt.close()
    print("Saved figure3_mb_comparison.png")


def plot_fig4_quantum_advantage(results, output_dir):
    """
    Figure 4: Quantum Advantage Demonstration
    Shows the gap between experimental fidelity and classical approximability.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Fidelity landscape (N vs d)
    ax1 = axes[0]
    
    # Get all MB data points
    all_mb = [r for r in results if r['method'] == 'MB']
    
    # Create a grid
    N_vals = sorted(set(r['N'] for r in all_mb))
    d_vals = sorted(set(r['d'] for r in all_mb))
    
    # Create color map
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm
    
    # Scatter plot with size and color encoding
    for r in all_mb:
        color = plt.cm.RdYlGn(r['F_mean'])  # Red (low) to Green (high)
        ax1.scatter(r['N'], r['d'], s=r['F_mean']*300 + 50, c=[color], 
                   edgecolors='black', linewidth=0.5, alpha=0.8)
    
    ax1.set_xlabel('Number of Qubits $N$')
    ax1.set_ylabel('Circuit Depth $d$')
    ax1.set_title(r'MB Fidelity Landscape')
    ax1.set_xticks(N_vals)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label(r'$P_{\mathrm{heavy}}$')
    
    # Panel B: Projected quantum advantage
    ax2 = axes[1]
    
    # Theoretical classical approximation threshold
    # For a random classical sampler: F_XEB ≈ 0
    # For importance sampling: F_XEB ≈ 1/N (extremely small for large N)
    
    N_range = np.array([16, 24, 32, 40, 48, 56, 64, 72, 80, 100])
    
    # Classical approximation bounds (theoretical)
    # Random sampling: F_XEB = 0
    # Importance sampling: F_XEB ~ 1/sqrt(N) (very rough)
    F_classical_random = np.zeros_like(N_range, dtype=float)
    F_classical_importance = 1.0 / np.sqrt(N_range)
    
    # Interpolated experimental fidelity (MB proxy)
    # Fit exponential decay with N
    n40_d12 = next((r for r in results if r['N'] == 40 and r['d'] == 12 and r['method'] == 'MB'), None)
    
    # Use empirical decay rates from data
    d12_mb = [r for r in results if r['d'] == 12 and r['method'] == 'MB']
    if d12_mb:
        N_data = np.array([r['N'] for r in d12_mb])
        F_data = np.array([r['F_mean'] for r in d12_mb])
        
        # Fit
        def exp_N(N, a, b):
            return a * np.exp(-b * N)
        
        try:
            popt, _ = curve_fit(exp_N, N_data, F_data, p0=[1.0, 0.01])
            F_exp = exp_N(N_range, *popt)
        except:
            F_exp = np.interp(N_range, N_data, F_data)
    else:
        F_exp = np.zeros_like(N_range, dtype=float)
    
    ax2.fill_between(N_range, F_classical_random, F_exp, 
                    alpha=0.3, color=COLORS['gap'], label='Quantum advantage gap')
    ax2.plot(N_range, F_exp, 'o-', color=COLORS['XEB'], linewidth=2, markersize=8,
            label='Experimental (extrapolated)')
    ax2.plot(N_range, F_classical_random, '--', color='gray', linewidth=2,
            label='Classical random')
    ax2.plot(N_range, F_classical_importance, ':', color='gray', linewidth=1.5,
            label='Classical importance sampling')
    
    ax2.set_xlabel('Number of Qubits $N$')
    ax2.set_ylabel(r'$\mathcal{F}_{\mathrm{XEB}}$ (proxy)')
    ax2.set_title('Gap Between Experiment and Classical')
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.05, 1.0)
    ax2.set_xlim(10, 110)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir).parent / "report" / "images" / "figure4_quantum_advantage.png")
    plt.close()
    print("Saved figure4_quantum_advantage.png")


def plot_fig5_n56_depth_scan(results, output_dir):
    """
    Figure 5: N56 depth scan with MB fidelity.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n56_mb = [r for r in results if r['N'] == 56 and r['method'] == 'MB']
    n56_mb.sort(key=lambda x: x['d'])
    
    depths = [r['d'] for r in n56_mb]
    P_vals = [r['F_mean'] for r in n56_mb]
    P_std = [r['F_std'] for r in n56_mb]
    
    ax.errorbar(depths, P_vals, yerr=P_std, fmt='o-', color=COLORS['MB'],
               capsize=5, capthick=2, linewidth=2, markersize=8, label=r'$P_{\mathrm{heavy}}$ (MB)')
    
    # Add N40 MB for comparison
    n40_mb = [r for r in results if r['N'] == 40 and r['method'] == 'MB']
    n40_mb.sort(key=lambda x: x['d'])
    
    depths40 = [r['d'] for r in n40_mb]
    P40_vals = [r['F_mean'] for r in n40_mb]
    P40_std = [r['F_std'] for r in n40_mb]
    
    ax.errorbar(depths40, P40_vals, yerr=P40_std, fmt='s--', color=COLORS['XEB'],
               capsize=5, capthick=2, linewidth=2, markersize=8, label=r'$P_{\mathrm{heavy}}$ (MB, $N=40$)')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
              label='50% threshold')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Circuit Depth $d$')
    ax.set_ylabel(r'$P_{\mathrm{heavy}}$')
    ax.set_title(r'MB Fidelity: $N=56$ vs $N=40$')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.05, 1.0)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir).parent / "report" / "images" / "figure5_n56_comparison.png")
    plt.close()
    print("Saved figure5_n56_comparison.png")


def plot_fig6_fidelity_table(results, output_dir):
    """
    Figure 6: Summary table of all fidelity estimates.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['N', 'd', 'Method', r'$\mathcal{F}$ (mean)', r'$\mathcal{F}$ (std)', 'Instances']
    cell_text = []
    
    # Sort results
    sorted_results = sorted(results, key=lambda x: (x['N'], x['d'], x['method']))
    
    for r in sorted_results:
        cell_text.append([
            str(r['N']),
            str(r['d']),
            r['method'],
            f"{r['F_mean']:.4f}",
            f"{r['F_std']:.4f}",
            str(r['n_instances'])
        ])
    
    table = ax.table(cellText=cell_text, colLabels=headers, loc='center',
                    cellLoc='center', colLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color code by method
    for i, row in enumerate(cell_text):
        if row[2] == 'XEB':
            for j in range(len(headers)):
                table[i+1, j].set_facecolor('#E3F2FD')
        else:
            for j in range(len(headers)):
                table[i+1, j].set_facecolor('#FFF3E0')
    
    ax.set_title('Summary of Fidelity Estimates', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir).parent / "report" / "images" / "figure6_summary_table.png")
    plt.close()
    print("Saved figure6_summary_table.png")


if __name__ == "__main__":
    output_dir = Path("outputs")
    results = load_results(output_dir)
    
    # Create images directory
    images_dir = Path("report") / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating figures...")
    plot_fig1_xeb_vs_depth(results, output_dir)
    plot_fig2_xeb_vs_N(results, output_dir)
    plot_fig3_mb_comparison(results, output_dir)
    plot_fig4_quantum_advantage(results, output_dir)
    plot_fig5_n56_depth_scan(results, output_dir)
    plot_fig6_fidelity_table(results, output_dir)
    
    print("\nAll figures generated!")
