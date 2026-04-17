#!/usr/bin/env python3
"""
Generate figures for the H0 Local Distance Network analysis.
Creates publication-quality plots for the research report.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)

def load_data(workspace):
    """Load analysis results."""
    with open(os.path.join(workspace, 'outputs/plot_data.json'), 'r') as f:
        plot_data = json.load(f)
    with open(os.path.join(workspace, 'outputs/h0_measurement.json'), 'r') as f:
        h0_result = json.load(f)
    with open(os.path.join(workspace, 'outputs/analysis_variants.json'), 'r') as f:
        variants = json.load(f)
    with open(os.path.join(workspace, 'outputs/cmb_comparison.json'), 'r') as f:
        cmb_comp = json.load(f)
    return plot_data, h0_result, variants, cmb_comp

def create_data_overview_fig(plot_data, save_path):
    """Figure 1: Data overview - host distances and SN calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Host distance moduli
    ax1 = axes[0]
    hosts = list(plot_data['host_mu'].keys())
    mus = [plot_data['host_mu'][h] for h in hosts]
    mu_errs = [plot_data['host_mu_err'][h] for h in hosts]
    
    y_pos = np.arange(len(hosts))
    ax1.barh(y_pos, mus, xerr=mu_errs, color='steelblue', alpha=0.7, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(hosts)
    ax1.set_xlabel('Distance Modulus $\\mu$ (mag)')
    ax1.set_title('(a) Host Galaxy Distance Moduli\nfrom Primary Indicators (Cepheids + TRGB)')
    ax1.invert_yaxis()
    ax1.axvline(x=np.mean(mus), color='red', linestyle='--', label='Mean')
    ax1.legend(loc='lower right')
    
    # Right panel: SN Ia absolute magnitude calibration
    ax2 = axes[1]
    calibs = plot_data['calibrators']
    hosts_cal = [c['host'] for c in calibs]
    M_Bs = [c['M_B'] for c in calibs]
    M_B_errs = [c['M_B_err'] for c in calibs]
    
    y_pos2 = np.arange(len(hosts_cal))
    ax2.barh(y_pos2, M_Bs, xerr=M_B_errs, color='coral', alpha=0.7, height=0.6)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(hosts_cal)
    ax2.set_xlabel('Absolute Magnitude $M_B$ (mag)')
    ax2.set_title('(b) SN Ia Absolute Magnitude Calibration')
    ax2.invert_yaxis()
    
    mean_M_B = np.mean(M_Bs)
    ax2.axvline(x=mean_M_B, color='red', linestyle='--', label=f'Mean: {mean_M_B:.2f}')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def create_hubble_diagram_fig(plot_data, h0_result, save_path):
    """Figure 2: Hubble diagram showing magnitude-redshift relation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    hf_data = plot_data['hf_data']
    zs = [d['z'] for d in hf_data]
    mBs = [d['mB'] for d in hf_data]
    mB_errs = [d['err'] for d in hf_data]
    
    # Plot data points
    ax.errorbar(zs, mBs, yerr=mB_errs, fmt='o', color='darkblue', 
                ecolor='gray', capsize=4, markersize=8, label='Hubble Flow SNe Ia')
    
    # Overplot best-fit model
    z_fit = np.linspace(0.02, 0.10, 100)
    H0 = plot_data['H0']
    M_B = h0_result['M_B']
    C_KM = 299792.458
    
    def model_m(z, H0, M_B):
        cz = C_KM * z
        mu = 5 * np.log10(cz / H0) + 25
        return M_B + mu
    
    m_fit = model_m(z_fit, H0, M_B)
    ax.plot(z_fit, m_fit, 'r-', linewidth=2, label=f'Best Fit: $H_0 = {H0:.1f}$ km/s/Mpc')
    
    # Add +/- 1 sigma bands
    H0_hi = H0 + h0_result['H0_stat_err']
    H0_lo = H0 - h0_result['H0_stat_err']
    m_hi = model_m(z_fit, H0_hi, M_B)
    m_lo = model_m(z_fit, H0_lo, M_B)
    ax.fill_between(z_fit, m_lo, m_hi, alpha=0.2, color='red', label='1$\\sigma$ uncertainty')
    
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Apparent Magnitude $m_B$')
    ax.set_title('Hubble Diagram: SN Ia Magnitude-Redshift Relation')
    ax.legend(loc='upper left')
    ax.invert_yaxis()  # Brighter objects at top
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def create_h0_comparison_fig(variants, cmb_comp, save_path):
    """Figure 3: H0 measurements comparison and tension with CMB."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Analysis variants
    ax1 = axes[0]
    variant_names = ['Baseline', 'Cepheids Only', 'TRGB Only', 'No PV Correction']
    H0_values = [cmb_comp['local_H0']]
    H0_errs = [cmb_comp['local_err']]
    
    for name, res in variants.items():
        H0_values.append(res['H0'])
        H0_errs.append(res['H0_err'])
    
    y_pos = np.arange(len(variant_names))
    ax1.barh(y_pos, H0_values, xerr=H0_errs, color=['steelblue', 'lightblue', 'lightgreen', 'lightcoral'], 
             alpha=0.7, height=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(variant_names)
    ax1.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax1.set_title('(a) H0 Measurements: Analysis Variants')
    ax1.set_xlim(60, 140)
    
    # Add Planck reference
    ax1.axvline(x=67.4, color='purple', linestyle='--', linewidth=2, label='Planck CMB')
    ax1.axvspan(67.4-0.5, 67.4+0.5, alpha=0.2, color='purple')
    ax1.legend(loc='lower right')
    
    # Right panel: Tension visualization
    ax2 = axes[1]
    
    # Plot local measurement distribution
    H0_local = cmb_comp['local_H0']
    err_local = cmb_comp['local_err']
    x = np.linspace(H0_local - 4*err_local, H0_local + 4*err_local, 200)
    pdf_local = np.exp(-0.5 * ((x - H0_local) / err_local)**2) / (err_local * np.sqrt(2*np.pi))
    ax2.plot(x, pdf_local, 'b-', linewidth=2, label=f'Local: {H0_local:.1f} ± {err_local:.1f}')
    ax2.fill_between(x, pdf_local, alpha=0.3, color='blue')
    
    # Plot Planck distribution
    H0_planck = cmb_comp['planck_H0']
    err_planck = cmb_comp['planck_err']
    pdf_planck = np.exp(-0.5 * ((x - H0_planck) / err_planck)**2) / (err_planck * np.sqrt(2*np.pi))
    ax2.plot(x, pdf_planck, 'purple', linewidth=2, label=f'Planck: {H0_planck:.1f} ± {err_planck:.1f}')
    ax2.fill_between(x, pdf_planck, alpha=0.3, color='purple')
    
    ax2.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title(f'(b) Hubble Tension: {cmb_comp["tension_sigma"]:.1f}$\\sigma$ Discrepancy')
    ax2.legend(loc='upper right')
    ax2.set_xlim(60, 130)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def create_distance_ladder_schematic(save_path):
    """Figure 4: Schematic diagram of the distance ladder."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a schematic distance ladder
    rungs = [
        ('Geometric Anchors', ['MW Parallaxes', 'LMC DEBs', 'NGC4258 Masers'], 0.8),
        ('Primary Indicators', ['Cepheids', 'TRGB', 'Miras'], 0.6),
        ('Secondary Calibrators', ['SN Ia', 'SBF'], 0.4),
        ('Hubble Flow', ['z > 0.02 SNe', 'cz vs m relation'], 0.2)
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, (label, items, y_pos) in enumerate(rungs):
        # Draw rung box
        rect = plt.Rectangle((0.1, y_pos - 0.08), 0.8, 0.12, 
                            facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Label
        ax.text(0.5, y_pos, label, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Items
        for j, item in enumerate(items):
            ax.text(0.15 + j*0.25, y_pos - 0.12, item, ha='left', va='top', fontsize=9)
    
    # Arrows between rungs
    for i in range(len(rungs)-1):
        y_start = rungs[i][2] - 0.08
        y_end = rungs[i+1][2] + 0.08
        ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start),
                   arrowprops=dict(arrowstyle='->', linewidth=2, color='gray'))
    
    # Add H0 result annotation
    ax.text(0.5, 0.05, '$\\Rightarrow H_0$ Measurement', ha='center', va='bottom',
           fontsize=14, fontweight='bold', color='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('The Cosmic Distance Ladder for H0 Measurement', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    workspace = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Astronomy_002_20260416_153612'
    images_dir = os.path.join(workspace, 'report/images')
    os.makedirs(images_dir, exist_ok=True)
    
    print("Loading analysis results...")
    plot_data, h0_result, variants, cmb_comp = load_data(workspace)
    
    print("\nGenerating figures...")
    create_data_overview_fig(plot_data, os.path.join(images_dir, 'data_overview.png'))
    create_hubble_diagram_fig(plot_data, h0_result, os.path.join(images_dir, 'hubble_diagram.png'))
    create_h0_comparison_fig(variants, cmb_comp, os.path.join(images_dir, 'h0_comparison.png'))
    create_distance_ladder_schematic(os.path.join(images_dir, 'distance_ladder_schematic.png'))
    
    print("\nAll figures generated!")

if __name__ == '__main__':
    main()
