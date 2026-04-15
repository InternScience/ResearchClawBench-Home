#!/usr/bin/env python3
"""
Analysis of Early Dark Energy (EDE) models and acoustic tension
between CMB and BAO measurements using DESI DR2 data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

# Set publication-quality plotting style
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['figure.dpi'] = 150

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================================
# DATA EXTRACTION FROM INPUT FILE
# ============================================================================

# ΛCDM (CMB+DESI) parameters
lcdm_params = {
    'omega_m': (0.3037, 0.0037),
    'H0': (68.12, 0.28),
    'sigma8': (0.8101, 0.0055),
    'ns': (0.9672, 0.0034),
    'ombh2': (0.02229, 0.00012),
    'ln10As': (3.056, 0.014),
    'tau': (0.0621, 0.0075)
}

# EDE (CMB+DESI) parameters
ede_params = {
    'omega_m': (0.2999, 0.0038),
    'H0': (70.9, 1.0),
    'sigma8': (0.8283, 0.0093),
    'f_EDE': (0.093, 0.031),
    'log10_ac': (-3.564, 0.075),
    'ns': (0.9817, 0.0063),
    'ombh2': (0.02241, 0.00018),
    'ln10As': (3.067, 0.017),
    'tau': (0.0582, 0.0074)
}

# w0wa (CMB+DESI) parameters
w0wa_params = {
    'omega_m': (0.353, 0.021),
    'H0': (63.5, 1.9),
    'sigma8': (0.780, 0.016),
    'w0': (-0.42, 0.21),
    'wa': (-1.75, 0.58),
    'ns': (0.9632, 0.0037),
    'ombh2': (0.02218, 0.00013),
    'ln10As': (3.037, 0.013),
    'tau': (0.0520, 0.0071)
}

# DESI BAO D_V/r_d data points (z, value, error)
desi_dvrd_points = np.array([
    (0.295, -0.020, 0.010),
    (0.510, -0.015, 0.008),
    (0.700, -0.012, 0.007),
    (0.934, -0.010, 0.006),
    (1.100, -0.005, 0.007),
    (1.320,  0.000, 0.008),
    (2.330,  0.010, 0.012)
])

# DESI BAO F_AP data points (z, value, error)
desi_fap_points = np.array([
    (0.295, -0.01, 0.02),
    (0.510,  0.00, 0.02),
    (0.700,  0.01, 0.02),
    (0.934,  0.02, 0.02),
    (1.100,  0.02, 0.02),
    (1.320,  0.02, 0.02),
    (2.330, -0.03, 0.04)
])

# Union3 SNe distance modulus residuals (z, value, error)
sne_mu_points = np.array([
    (0.1, -0.08, 0.10),
    (0.2, -0.12, 0.08),
    (0.3, -0.10, 0.07),
    (0.4, -0.07, 0.06),
    (0.5, -0.05, 0.05),
    (0.6, -0.02, 0.05),
    (0.7,  0.00, 0.05)
])

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_parameter_shifts():
    """Calculate parameter shifts between models."""
    shifts = {}
    
    # Common parameters
    common_params = ['omega_m', 'H0', 'sigma8', 'ns', 'ombh2', 'ln10As', 'tau']
    
    shifts['ede_vs_lcdm'] = {}
    shifts['w0wa_vs_lcdm'] = {}
    shifts['ede_vs_w0wa'] = {}
    
    for param in common_params:
        if param in lcdm_params and param in ede_params:
            delta = ede_params[param][0] - lcdm_params[param][0]
            sigma_combined = np.sqrt(ede_params[param][1]**2 + lcdm_params[param][1]**2)
            shifts['ede_vs_lcdm'][param] = {
                'delta': delta,
                'sigma': sigma_combined,
                'nsigma': abs(delta) / sigma_combined
            }
        
        if param in lcdm_params and param in w0wa_params:
            delta = w0wa_params[param][0] - lcdm_params[param][0]
            sigma_combined = np.sqrt(w0wa_params[param][1]**2 + lcdm_params[param][1]**2)
            shifts['w0wa_vs_lcdm'][param] = {
                'delta': delta,
                'sigma': sigma_combined,
                'nsigma': abs(delta) / sigma_combined
            }
        
        if param in ede_params and param in w0wa_params:
            delta = ede_params[param][0] - w0wa_params[param][0]
            sigma_combined = np.sqrt(ede_params[param][1]**2 + w0wa_params[param][1]**2)
            shifts['ede_vs_w0wa'][param] = {
                'delta': delta,
                'sigma': sigma_combined,
                'nsigma': abs(delta) / sigma_combined
            }
    
    return shifts

def save_json_outputs(shifts):
    """Save processed data to JSON."""
    output_data = {
        'lcdm_params': lcdm_params,
        'ede_params': ede_params,
        'w0wa_params': w0wa_params,
        'parameter_shifts': shifts,
        'desi_dvrd': desi_dvrd_points.tolist(),
        'desi_fap': desi_fap_points.tolist(),
        'sne_mu': sne_mu_points.tolist()
    }
    
    with open('outputs/parameter_constraints.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Saved outputs/parameter_constraints.json")

# ============================================================================
# FIGURE 1: Parameter Constraints Comparison
# ============================================================================

def plot_parameter_constraints():
    """Create triangle plot comparing key cosmological parameters."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    
    params = ['omega_m', 'H0', 'sigma8']
    param_labels = [r'$\Omega_m$', r'$H_0$ [km/s/Mpc]', r'$\sigma_8$']
    colors = {'ΛCDM': '#1f77b4', 'EDE': '#d62728', 'w₀wₐ': '#2ca02c'}
    
    models = {
        'ΛCDM': lcdm_params,
        'EDE': ede_params,
        'w₀wₐ': w0wa_params
    }
    
    for i, param_i in enumerate(params):
        for j, param_j in enumerate(params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: 1D marginalized distributions (show as error bars)
                y_offset = 0
                for model_name, model_params in models.items():
                    if param_i in model_params:
                        mu, sigma = model_params[param_i]
                        ax.errorbar(mu, y_offset, xerr=sigma, fmt='o', 
                                   color=colors[model_name], markersize=10,
                                   capsize=5, capthick=2, label=model_name)
                        y_offset += 1
                
                ax.set_xlabel(param_labels[i])
                ax.set_yticks([])
                ax.set_ylim(-0.5, 3)
                if i == 0:
                    ax.legend(loc='upper right')
                ax.axvline(x=models['ΛCDM'][param_i][0], color=colors['ΛCDM'], 
                          linestyle='--', alpha=0.3)
                
            elif i > j:
                # Lower triangle: 2D contours (show as ellipses)
                for model_name, model_params in models.items():
                    if param_i in model_params and param_j in model_params:
                        mu_i, sigma_i = model_params[param_i]
                        mu_j, sigma_j = model_params[param_j]
                        
                        # Draw 1-sigma ellipse
                        ellipse = Ellipse((mu_j, mu_i), 2*sigma_j, 2*sigma_i,
                                         facecolor='none', edgecolor=colors[model_name],
                                         linewidth=2, label=model_name)
                        ax.add_patch(ellipse)
                        
                        # Mark center
                        ax.plot(mu_j, mu_i, 'o', color=colors[model_name], markersize=6)
                
                ax.set_xlabel(param_labels[j])
                ax.set_ylabel(param_labels[i])
                
            else:
                # Upper triangle: hide
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('report/images/figure1_parameter_constraints.png', dpi=150, bbox_inches='tight')
    plt.savefig('report/images/figure1_parameter_constraints.pdf', bbox_inches='tight')
    print("Saved figure1_parameter_constraints.png")
    plt.close()

# ============================================================================
# FIGURE 2: EDE-specific Parameters
# ============================================================================

def plot_ede_parameters():
    """Plot EDE-specific parameters f_EDE and log10(a_c)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: f_EDE
    ax1 = axes[0]
    params_ede = ['f_EDE', 'log10_ac']
    labels = [r'$f_{\rm EDE}$', r'$\log_{10}(a_c)$']
    
    for idx, (param, label) in enumerate(zip(params_ede, labels)):
        ax = axes[idx]
        mu, sigma = ede_params[param]
        
        # Draw Gaussian approximation
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        y = np.exp(-0.5 * ((x - mu) / sigma)**2)
        ax.fill_between(x, y, alpha=0.3, color='#d62728')
        ax.plot(x, y, color='#d62728', linewidth=2)
        
        # Mark peak and 1-sigma region
        ax.axvline(mu, color='k', linestyle='-', linewidth=2, label=f'{mu:.3f} ± {sigma:.3f}')
        ax.axvspan(mu - sigma, mu + sigma, alpha=0.2, color='gray')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Likelihood (arbitrary units)')
        ax.set_ylim(0, 1.2)
        ax.legend(loc='upper right')
        ax.set_title(f'EDE Parameter: {label}')
    
    plt.tight_layout()
    plt.savefig('report/images/figure2_ede_parameters.png', dpi=150, bbox_inches='tight')
    plt.savefig('report/images/figure2_ede_parameters.pdf', bbox_inches='tight')
    print("Saved figure2_ede_parameters.png")
    plt.close()

# ============================================================================
# FIGURE 3: BAO Distance Residuals
# ============================================================================

def plot_bao_residuals():
    """Plot BAO distance residuals vs redshift."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # D_V/r_d residuals
    ax1 = axes[0]
    z_dv = desi_dvrd_points[:, 0]
    dv_residuals = desi_dvrd_points[:, 1]
    dv_errors = desi_dvrd_points[:, 2]
    
    ax1.errorbar(z_dv, dv_residuals, yerr=dv_errors, fmt='o', 
                color='#1f77b4', markersize=8, capsize=5, capthick=2,
                label='DESI DR2 BAO $D_V/r_d$')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Redshift $z$')
    ax1.set_ylabel(r'$\Delta(D_V/r_d)$ (relative to fiducial)')
    ax1.set_title('BAO Distance Residuals: $D_V/r_d$')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # F_AP residuals
    ax2 = axes[1]
    z_fap = desi_fap_points[:, 0]
    fap_residuals = desi_fap_points[:, 1]
    fap_errors = desi_fap_points[:, 2]
    
    ax2.errorbar(z_fap, fap_residuals, yerr=fap_errors, fmt='s',
                color='#ff7f0e', markersize=8, capsize=5, capthick=2,
                label='DESI DR2 BAO $F_{AP}$')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Redshift $z$')
    ax2.set_ylabel(r'$\Delta F_{AP}$ (relative to fiducial)')
    ax2.set_title('BAO AP Residuals: $F_{AP}$')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/figure3_bao_residuals.png', dpi=150, bbox_inches='tight')
    plt.savefig('report/images/figure3_bao_residuals.pdf', bbox_inches='tight')
    print("Saved figure3_bao_residuals.png")
    plt.close()

# ============================================================================
# FIGURE 4: SNe Distance Modulus Residuals
# ============================================================================

def plot_sne_residuals():
    """Plot Union3 SNe distance modulus residuals."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    z_sne = sne_mu_points[:, 0]
    mu_residuals = sne_mu_points[:, 1]
    mu_errors = sne_mu_points[:, 2]
    
    ax.errorbar(z_sne, mu_residuals, yerr=mu_errors, fmt='D',
               color='#2ca02c', markersize=8, capsize=5, capthick=2,
               label='Union3 SNe Ia')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5, label='Fiducial Model')
    
    # Add smoothed trend
    z_smooth = np.linspace(z_sne.min(), z_sne.max(), 100)
    # Simple polynomial fit for visualization
    coeffs = np.polyfit(z_sne, mu_residuals, 2, w=1/mu_errors)
    trend = np.polyval(coeffs, z_smooth)
    ax.plot(z_smooth, trend, 'r--', alpha=0.7, label='Trend')
    
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'$\Delta\mu$ (mag)')
    ax.set_title('Union3 Type Ia Supernova Distance Modulus Residuals')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/figure4_sne_residuals.png', dpi=150, bbox_inches='tight')
    plt.savefig('report/images/figure4_sne_residuals.pdf', bbox_inches='tight')
    print("Saved figure4_sne_residuals.png")
    plt.close()

# ============================================================================
# FIGURE 5: Model Comparison Summary
# ============================================================================

def plot_model_comparison(shifts):
    """Create comprehensive model comparison figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Key parameter comparisons
    params_to_plot = ['H0', 'omega_m', 'sigma8']
    param_labels_short = [r'$H_0$', r'$\Omega_m$', r'$\sigma_8$']
    
    for idx, (param, label) in enumerate(zip(params_to_plot, param_labels_short)):
        ax = fig.add_subplot(gs[0, idx])
        
        models = ['ΛCDM', 'EDE', 'w₀wₐ']
        values = [lcdm_params[param][0], ede_params[param][0], w0wa_params[param][0]]
        errors = [lcdm_params[param][1], ede_params[param][1], w0wa_params[param][1]]
        colors_list = ['#1f77b4', '#d62728', '#2ca02c']
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, values, yerr=errors, capsize=5, color=colors_list, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.set_ylabel(label)
        ax.set_title(f'{label} Constraints')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (v, e) in enumerate(zip(values, errors)):
            ax.text(i, v + e + 0.02 * max(values), f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Middle left: H0 vs Omega_m 2D comparison
    ax_middle = fig.add_subplot(gs[1, 0])
    colors_dict = {'ΛCDM': '#1f77b4', 'EDE': '#d62728', 'w₀wₐ': '#2ca02c'}
    
    for model_name, model_params, label in [('ΛCDM', lcdm_params, 'ΛCDM'),
                                            ('EDE', ede_params, 'EDE'),
                                            ('w₀wₐ', w0wa_params, 'w₀wₐ')]:
        h0_mu, h0_sigma = model_params['H0']
        om_mu, om_sigma = model_params['omega_m']
        
        ellipse = Ellipse((om_mu, h0_mu), 2*om_sigma, 2*h0_sigma,
                         facecolor='none', edgecolor=colors_dict[model_name],
                         linewidth=2.5, label=label)
        ax_middle.add_patch(ellipse)
        ax_middle.plot(om_mu, h0_mu, 'o', color=colors_dict[model_name], markersize=10)
    
    ax_middle.set_xlabel(r'$\Omega_m$')
    ax_middle.set_ylabel(r'$H_0$ [km/s/Mpc]')
    ax_middle.set_title(r'$H_0$ vs $\Omega_m$ (1$\sigma$ contours)')
    ax_middle.legend(loc='upper right')
    ax_middle.grid(True, alpha=0.3)
    
    # Middle center: EDE parameter space
    ax_ede = fig.add_subplot(gs[1, 1])
    fede_mu, fede_sigma = ede_params['f_EDE']
    logac_mu, logac_sigma = ede_params['log10_ac']
    
    ellipse_ede = Ellipse((logac_mu, fede_mu), 2*logac_sigma, 2*fede_sigma,
                         facecolor='#d62728', edgecolor='darkred', alpha=0.3, linewidth=2)
    ax_ede.add_patch(ellipse_ede)
    ax_ede.plot(logac_mu, fede_mu, 'o', color='darkred', markersize=12)
    ax_ede.set_xlabel(r'$\log_{10}(a_c)$')
    ax_ede.set_ylabel(r'$f_{\rm EDE}$')
    ax_ede.set_title('EDE Parameter Space')
    ax_ede.grid(True, alpha=0.3)
    
    # Middle right: w0wa parameters
    ax_w0wa = fig.add_subplot(gs[1, 2])
    w0_mu, w0_sigma = w0wa_params['w0']
    wa_mu, wa_sigma = w0wa_params['wa']
    
    ellipse_w0wa = Ellipse((w0_mu, wa_mu), 2*w0_sigma, 2*wa_sigma,
                          facecolor='#2ca02c', edgecolor='darkgreen', alpha=0.3, linewidth=2)
    ax_w0wa.add_patch(ellipse_w0wa)
    ax_w0wa.plot(w0_mu, wa_mu, 'o', color='darkgreen', markersize=12)
    ax_w0wa.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_w0wa.axvline(-1, color='k', linestyle='--', alpha=0.3)
    ax_w0wa.set_xlabel(r'$w_0$')
    ax_w0wa.set_ylabel(r'$w_a$')
    ax_w0wa.set_title('w₀wₐ Parameter Space')
    ax_w0wa.grid(True, alpha=0.3)
    
    # Bottom: Parameter shift significance
    ax_shifts = fig.add_subplot(gs[2, :])
    
    comparisons = [
        ('EDE vs ΛCDM', shifts['ede_vs_lcdm']),
        ('w₀wₐ vs ΛCDM', shifts['w0wa_vs_lcdm']),
        ('EDE vs w₀wₐ', shifts['ede_vs_w0wa'])
    ]
    
    y_pos = 0
    colors_comp = ['#d62728', '#2ca02c', '#9467bd']
    all_params = ['H0', 'omega_m', 'sigma8', 'ns', 'ombh2']
    
    for comp_idx, (comp_name, comp_data) in enumerate(comparisons):
        for param_idx, param in enumerate(all_params):
            if param in comp_data:
                nsigma = comp_data[param]['nsigma']
                ax_shifts.barh(y_pos, nsigma, color=colors_comp[comp_idx], alpha=0.7, 
                              height=0.6, edgecolor='black')
                ax_shifts.text(nsigma + 0.1, y_pos, f'{nsigma:.2f}σ', va='center', fontsize=9)
                y_pos += 1
        y_pos += 0.5  # Gap between comparisons
    
    # Set y-tick labels
    yticks = []
    yticklabels = []
    y_pos = 0
    for comp_name, comp_data in comparisons:
        for param in all_params:
            if param in comp_data:
                yticks.append(y_pos)
                yticklabels.append(f"{comp_name.split(' vs ')[0][:3]} vs {comp_name.split(' vs ')[1][:4]}: {param}")
                y_pos += 1
        y_pos += 0.5
    
    ax_shifts.set_yticks(yticks)
    ax_shifts.set_yticklabels(yticklabels, fontsize=9)
    ax_shifts.set_xlabel(r'Parameter Shift ($\Delta/\sigma_{\rm combined}$)')
    ax_shifts.set_title('Significance of Parameter Shifts Between Models')
    ax_shifts.axvline(1, color='k', linestyle='--', alpha=0.5, label='1σ')
    ax_shifts.axvline(2, color='k', linestyle=':', alpha=0.5, label='2σ')
    ax_shifts.legend(loc='lower right')
    ax_shifts.grid(True, alpha=0.3, axis='x')
    
    plt.savefig('report/images/figure5_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('report/images/figure5_model_comparison.pdf', bbox_inches='tight')
    print("Saved figure5_model_comparison.png")
    plt.close()

# ============================================================================
# FIGURE 6: Hubble Tension Analysis
# ============================================================================

def plot_hubble_tension():
    """Analyze Hubble tension in different models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: H0 comparison with SH0ES value
    ax1 = axes[0]
    
    # SH0ES measurement (typical value from literature)
    sh0es_h0 = 73.04
    sh0es_sigma = 1.04
    
    models = ['ΛCDM', 'EDE', 'w₀wₐ']
    h0_values = [lcdm_params['H0'][0], ede_params['H0'][0], w0wa_params['H0'][0]]
    h0_errors = [lcdm_params['H0'][1], ede_params['H0'][1], w0wa_params['H0'][1]]
    colors_list = ['#1f77b4', '#d62728', '#2ca02c']
    
    # Plot model constraints
    for i, (model, h0, err, color) in enumerate(zip(models, h0_values, h0_errors, colors_list)):
        ax1.errorbar(h0, i, xerr=err, fmt='o', color=color, markersize=12, 
                    capsize=8, capthick=2, label=model)
    
    # Plot SH0ES measurement
    ax1.axvline(sh0es_h0, color='purple', linestyle='--', linewidth=2, label=f'SH0ES: {sh0es_h0}±{sh0es_sigma}')
    ax1.axvspan(sh0es_h0 - sh0es_sigma, sh0es_h0 + sh0es_sigma, alpha=0.2, color='purple')
    
    ax1.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_title('Hubble Parameter: CMB+BAO vs SH0ES')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(60, 75)
    
    # Right: Tension significance
    ax2 = axes[1]
    
    tensions = []
    tension_labels = []
    for model, h0, err in zip(models, h0_values, h0_errors):
        tension = abs(h0 - sh0es_h0) / np.sqrt(err**2 + sh0es_sigma**2)
        tensions.append(tension)
        tension_labels.append(f'{model}\n({h0:.1f}±{err:.1f})')
    
    bars = ax2.bar(range(len(models)), tensions, color=colors_list, alpha=0.7, edgecolor='black')
    ax2.axhline(1, color='k', linestyle='--', alpha=0.5, label='1σ tension')
    ax2.axhline(2, color='k', linestyle=':', alpha=0.5, label='2σ tension')
    ax2.axhline(3, color='r', linestyle='--', alpha=0.5, label='3σ tension')
    
    # Add value labels
    for i, (bar, tension) in enumerate(zip(bars, tensions)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{tension:.1f}σ', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(tension_labels, fontsize=10)
    ax2.set_ylabel('Tension with SH0ES (σ)')
    ax2.set_title('Hubble Tension Significance')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(tensions) * 1.2)
    
    plt.tight_layout()
    plt.savefig('report/images/figure6_hubble_tension.png', dpi=150, bbox_inches='tight')
    plt.savefig('report/images/figure6_hubble_tension.pdf', bbox_inches='tight')
    print("Saved figure6_hubble_tension.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("EDE Acoustic Tension Analysis")
    print("="*60)
    
    # Calculate parameter shifts
    print("\n[1] Calculating parameter shifts...")
    shifts = calculate_parameter_shifts()
    
    # Save outputs
    print("\n[2] Saving JSON outputs...")
    save_json_outputs(shifts)
    
    # Generate figures
    print("\n[3] Generating figures...")
    plot_parameter_constraints()
    plot_ede_parameters()
    plot_bao_residuals()
    plot_sne_residuals()
    plot_model_comparison(shifts)
    plot_hubble_tension()
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print("\nKey Parameter Constraints:")
    print(f"  ΛCDM: H₀ = {lcdm_params['H0'][0]:.2f} ± {lcdm_params['H0'][1]:.2f} km/s/Mpc")
    print(f"  EDE:  H₀ = {ede_params['H0'][0]:.2f} ± {ede_params['H0'][1]:.2f} km/s/Mpc")
    print(f"  w₀wₐ: H₀ = {w0wa_params['H0'][0]:.2f} ± {w0wa_params['H0'][1]:.2f} km/s/Mpc")
    
    print("\nEDE-specific Parameters:")
    print(f"  f_EDE = {ede_params['f_EDE'][0]:.3f} ± {ede_params['f_EDE'][1]:.3f}")
    print(f"  log₁₀(a_c) = {ede_params['log10_ac'][0]:.3f} ± {ede_params['log10_ac'][1]:.3f}")
    
    print("\nParameter Shifts (significance):")
    for comp_name, comp_data in [('EDE vs ΛCDM', shifts['ede_vs_lcdm']),
                                  ('w₀wₐ vs ΛCDM', shifts['w0wa_vs_lcdm'])]:
        print(f"\n  {comp_name}:")
        for param in ['H0', 'omega_m', 'sigma8']:
            if param in comp_data:
                nsigma = comp_data[param]['nsigma']
                print(f"    {param}: {nsigma:.2f}σ")
    
    print("\n" + "="*60)
    print("Analysis complete! Check report/images/ for figures.")
    print("="*60)

if __name__ == '__main__':
    main()
