#!/usr/bin/env python3
"""
Generate figures for the H0 Distance Network analysis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from h0_analysis import (
    anchors, host_measurements_raw, sneia_calibrators,
    sbf_calibrators, hubble_flow_sneia, hubble_flow_sbf,
    host_group, c_km, compute_h0_stepwise, run_gls_distance_network,
    run_variant_analysis
)

# Style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.autolayout': True
})

os.makedirs('report/images', exist_ok=True)

# ============================================================
# Figure 1: Data Overview — Distance Ladder Schematic
# ============================================================

def fig1_data_overview():
    """Schematic diagram of the distance ladder / distance network."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Local Distance Network Architecture', fontsize=14, fontweight='bold', pad=20)
    
    # Rungs
    rungs = [
        (1, 'Geometric\nAnchors', ['N4258 (maser)\nμ=29.397±0.032', 'LMC (DEB)\nμ=18.477±0.024', 'MW (parallax)\nμ=0.0'], '#1f77b4'),
        (2, 'Primary\nIndicators', ['Cepheids\n(7 measurements)', 'TRGB\n(4 measurements)'], '#ff7f0e'),
        (3, 'Secondary\nCalibrators', ['SNe Ia\n(7 calibrators)', 'SBF\n(3 calibrators)'], '#2ca02c'),
        (4, 'Hubble Flow\n& H₀', ['SNe Ia: 5 SNe\nz=0.023–0.082', 'SBF: 3 galaxies\nz=0.023–0.045'], '#d62728'),
    ]
    
    y_positions = [8, 5.5, 3, 0.5]
    x_positions = [2.5, 7.5]
    
    for i, (rung, title, items, color) in enumerate(rungs):
        y = y_positions[i]
        # Rung box
        rect = FancyBboxPatch((0.3, y-0.8), 9.4, 2.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y + 1.5, f'Rung {rung}: {title}', fontsize=11, fontweight='bold', color=color, va='center')
        
        for j, item in enumerate(items):
            x = x_positions[j % 2]
            y_off = -0.3 if j >= 2 else 0.3
            ax.text(x, y + y_off, item, fontsize=8.5, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))
    
    # Arrows between rungs
    for i in range(3):
        y_top = y_positions[i] - 0.8
        y_bot = y_positions[i+1] + 1.7
        ax.annotate('', xy=(5, y_bot), xytext=(5, y_top),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax.text(9.5, 9.5, 'GLS Covariance-Weighted\nConsensus H₀', fontsize=9,
           ha='right', va='top', style='italic', color='purple',
           bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")


# ============================================================
# Figure 2: Host Distance Measurements
# ============================================================

def fig2_host_distances():
    """Host distances by indicator and anchor."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: host distances by method
    ax = axes[0]
    methods = ['Cepheid', 'TRGB']
    colors = {'Cepheid': '#1f77b4', 'TRGB': '#ff7f0e'}
    markers = {'N4258': 'o', 'LMC': 's'}
    
    hosts_seen = []
    y_ticks = []
    y_labels = []
    
    for i, (host, method, anchor, mu, err) in enumerate(host_measurements_raw):
        y = len(host_measurements_raw) - i
        if host not in hosts_seen:
            hosts_seen.append(host)
        ax.errorbar(mu, y, xerr=err, fmt=markers[anchor], 
                   color=colors[method], markersize=8, capsize=3,
                   label=f'{method} ({anchor})' if i < 2 else '')
        y_ticks.append(y)
        y_labels.append(f'{host}')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=8)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Distance Modulus μ (mag)')
    ax.set_title('Host Distance Measurements', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Right: GLS fitted host distances
    ax = axes[1]
    gls = run_gls_distance_network()
    hosts_gls = list(gls['hosts'].keys())
    
    mu_vals = [gls['hosts'][h]['mu'] for h in hosts_gls]
    mu_errs = [gls['hosts'][h]['err'] for h in hosts_gls]
    y_pos = range(len(hosts_gls))
    
    colors_host = []
    for h in hosts_gls:
        has_snia = any(h == host for host, _, _ in sneia_calibrators)
        has_sbf = any(h == host for host, _, _ in sbf_calibrators)
        if has_snia and has_sbf:
            colors_host.append('purple')
        elif has_snia:
            colors_host.append('#2ca02c')
        elif has_sbf:
            colors_host.append('#d62728')
        else:
            colors_host.append('gray')
    
    ax.errorbar(mu_vals, y_pos, xerr=mu_errs, fmt='o', markersize=8, 
               capsize=3, color='black', ecolor='gray')
    for i, (h, c) in enumerate(zip(hosts_gls, colors_host)):
        ax.plot(mu_vals[i], i, 'o', color=c, markersize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hosts_gls, fontsize=8)
    ax.set_xlabel('Distance Modulus μ (mag)')
    ax.set_title('GLS Fitted Host Distances', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=8, label='SNe Ia host'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label='SBF host'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Both'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('report/images/fig2_host_distances.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")


# ============================================================
# Figure 3: Hubble Diagram and H0 Fits
# ============================================================

def fig3_hubble_diagram():
    """Hubble diagram showing SNe Ia and SBF constraints."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: SNe Ia Hubble diagram
    ax = axes[0]
    gls = run_gls_distance_network()
    M_B = gls['M_B']
    H0_best = gls['H0']
    H0_err = gls['H0_err']
    
    # Calibrator SNe Ia
    cal_z = []
    cal_mu = []
    for host, mB, err_mB in sneia_calibrators:
        if host in gls['hosts']:
            mu_host = gls['hosts'][host]['mu']
            cal_mu.append(mu_host)
            cal_z.append(0.003)  # approximate low z for calibrators
    
    # Hubble flow SNe Ia
    z_hf = np.array([z for z, _, _, _ in hubble_flow_sneia])
    mB_hf = np.array([mB for _, mB, _, _ in hubble_flow_sneia])
    mu_hf = mB_hf - M_B
    err_hf = np.array([e for _, _, e, _ in hubble_flow_sneia])
    
    # Predicted mu for H0
    z_grid = np.logspace(-2, -0.9, 50)
    mu_pred = 5.0 * np.log10(c_km * z_grid / H0_best) + 25.0
    mu_pred_plus = 5.0 * np.log10(c_km * z_grid / (H0_best + H0_err)) + 25.0
    mu_pred_minus = 5.0 * np.log10(c_km * z_grid / (H0_best - H0_err)) + 25.0
    
    ax.fill_between(z_grid, mu_pred_minus, mu_pred_plus, alpha=0.2, color='#1f77b4')
    ax.plot(z_grid, mu_pred, 'b-', linewidth=2, label=f'H₀ = {H0_best:.1f} ± {H0_err:.1f}')
    ax.errorbar(z_hf, mu_hf, yerr=err_hf, fmt='ro', markersize=8, capsize=3, 
               label='Hubble Flow SNe Ia')
    
    ax.set_xscale('log')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Distance Modulus μ (mag)')
    ax.set_title('SNe Ia Hubble Diagram', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Right: SBF Hubble diagram
    ax = axes[1]
    M_SBF = gls['M_SBF']
    
    z_sbf_hf = np.array([z for z, _, _, _ in hubble_flow_sbf])
    mF_sbf_hf = np.array([mF for _, mF, _, _ in hubble_flow_sbf])
    mu_sbf_hf = mF_sbf_hf - M_SBF
    err_sbf_hf = np.array([e for _, _, e, _ in hubble_flow_sbf])
    
    mu_pred_sbf = 5.0 * np.log10(c_km * z_grid / H0_best) + 25.0
    
    ax.plot(z_grid, mu_pred_sbf, 'b-', linewidth=2, alpha=0.5)
    ax.errorbar(z_sbf_hf, mu_sbf_hf, yerr=err_sbf_hf, fmt='gs', markersize=8, capsize=3,
               label='Hubble Flow SBF')
    
    # SBF calibrator positions
    for host, mF, err_mF in sbf_calibrators:
        if host in gls['hosts']:
            mu_host = gls['hosts'][host]['mu']
            ax.plot(0.001, mu_host, 'gD', markersize=10, label='SBF Calibrator' if host == sbf_calibrators[0][0] else '')
    
    ax.set_xscale('log')
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Distance Modulus μ (mag)')
    ax.set_title('SBF Hubble Diagram', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig3_hubble_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")


# ============================================================
# Figure 4: H0 Variants Comparison
# ============================================================

def fig4_h0_variants():
    """H0 values from different analysis variants."""
    variants = run_variant_analysis()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = list(variants.keys())
    h0_vals = [variants[n]['H0'] for n in names]
    h0_errs = [variants[n]['H0_err'] for n in names]
    
    display_names = {
        'baseline': 'GLS Baseline',
        'SNeIa_only': 'SNe Ia Only',
        'Cepheids_only': 'Cepheids Only',
        'TRGB_only': 'TRGB Only',
        'N4258_only': 'N4258 Anchor Only',
        'LMC_only': 'LMC Anchor Only'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    y_pos = range(len(names))
    
    for i, (name, h0, err, c) in enumerate(zip(names, h0_vals, h0_errs, colors)):
        ax.errorbar(h0, i, xerr=err, fmt='o', color=c, markersize=10, capsize=5, linewidth=2)
        ax.text(h0 + err + 1, i, f'{h0:.1f}±{err:.1f}', fontsize=9, va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_names.get(n, n) for n in names], fontsize=10)
    ax.set_xlabel('H₀ (km s⁻¹ Mpc⁻¹)', fontsize=12)
    ax.set_title('H₀ Analysis Variants', fontweight='bold', fontsize=13)
    ax.axvline(x=73.04, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(73.04, len(names)-0.3, 'SH0ES 2022\n(73.04)', color='red', fontsize=8, ha='center')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('report/images/fig4_h0_variants.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved.")


# ============================================================
# Figure 5: Residual Analysis
# ============================================================

def fig5_residuals():
    """Residuals from GLS fit."""
    gls = run_gls_distance_network()
    residuals = np.array(gls['residuals'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: residual distribution
    ax = axes[0]
    ax.hist(residuals, bins=12, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    
    from scipy import stats
    x = np.linspace(-5, 5, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
    
    ax.set_xlabel('Normalized Residual')
    ax.set_ylabel('Density')
    ax.set_title('GLS Residual Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Right: residuals by type
    ax = axes[1]
    
    # Categorize residuals
    n_host = len(host_measurements_raw)
    n_snia_cal = len(sneia_calibrators)
    n_sbf_cal = len(sbf_calibrators)
    n_hf_snia = len(hubble_flow_sneia)
    n_hf_sbf = len(hubble_flow_sbf)
    
    categories = ['Host μ', 'SNe Ia Cal', 'SBF Cal', 'HF SNe Ia', 'HF SBF']
    starts = [0, n_host, n_host + n_snia_cal, n_host + n_snia_cal + n_sbf_cal, 
              n_host + n_snia_cal + n_sbf_cal + n_hf_snia]
    
    colors_cat = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (cat, start, c) in enumerate(zip(categories, starts, colors_cat)):
        if i < len(categories) - 1:
            end = starts[i+1]
        else:
            end = len(residuals)
        if start < len(residuals):
            ax.scatter(range(start, end), residuals[start:end], 
                      color=c, label=cat, s=30, zorder=3)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Normalized Residual')
    ax.set_title('GLS Residuals by Category', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig5_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 5 saved.")


# ============================================================
# Figure 6: Correlation / Network connectivity
# ============================================================

def fig6_network():
    """Show the connectivity of the distance network."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_title('Distance Network Connectivity', fontweight='bold', fontsize=14)
    
    # Layout
    anchor_y = 9
    indicator_y = 6
    host_y = 3
    calibrator_y = 0.5
    
    # Anchors
    anchor_positions = {
        'N4258': (1, anchor_y),
        'LMC': (5, anchor_y),
    }
    
    # Hosts
    unique_hosts = sorted(set(h for h, _, _, _, _ in host_measurements_raw) | 
                         set(h for h, _, _ in sneia_calibrators) |
                         set(h for h, _, _ in sbf_calibrators))
    
    host_positions = {}
    for i, h in enumerate(unique_hosts):
        host_positions[h] = (i * 1.0 + 0.5, host_y)
    
    # Draw anchors
    for name, (x, y) in anchor_positions.items():
        ax.plot(x, y, 'ks', markersize=12)
        ax.text(x, y-0.3, name, fontsize=9, ha='center', fontweight='bold')
    
    # Draw hosts
    for name, (x, y) in host_positions.items():
        is_snia = any(name == h for h, _, _ in sneia_calibrators)
        is_sbf = any(name == h for h, _, _ in sbf_calibrators)
        if is_snia and is_sbf:
            color = 'purple'
        elif is_snia:
            color = '#2ca02c'
        elif is_sbf:
            color = '#d62728'
        else:
            color = 'gray'
        ax.plot(x, y, 'o', color=color, markersize=10, markeredgecolor='black')
        ax.text(x, y-0.25, name, fontsize=7, ha='center')
    
    # Draw connections
    for host, method, anchor, _, _ in host_measurements_raw:
        if anchor in anchor_positions and host in host_positions:
            ax.plot([anchor_positions[anchor][0], host_positions[host][0]],
                   [anchor_positions[anchor][1], host_positions[host][1]],
                   '-', color='gray', alpha=0.3, linewidth=1)
    
    # SNe Ia calibrator box
    ax.text(9, 1.5, 'SNe Ia\nHubble Flow\n(5 SNe)', fontsize=8, ha='center',
           bbox=dict(boxstyle='round', facecolor='#d62728', alpha=0.2))
    ax.text(9, 5, 'SBF\nHubble Flow\n(3 galaxies)', fontsize=8, ha='center',
           bbox=dict(boxstyle='round', facecolor='purple', alpha=0.2))
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Anchor'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='SN Ia host'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='SBF host'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Both'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('report/images/fig6_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 6 saved.")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("Generating figures...")
    fig1_data_overview()
    fig2_host_distances()
    fig3_hubble_diagram()
    fig4_h0_variants()
    fig5_residuals()
    fig6_network()
    print("All figures saved to report/images/")
