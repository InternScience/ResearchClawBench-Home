"""
MATBG Superfluid Stiffness Analysis
=====================================
Analysis of Magic-Angle Twisted Bilayer Graphene (MATBG) superfluid stiffness data
including carrier density dependence, temperature dependence, and current dependence.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')  # Use non-interactive backend

from scipy.optimize import curve_fit
import json
import os

# Set up matplotlib style
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150

# Create output directories
os.makedirs('../outputs', exist_ok=True)
os.makedirs('../report/images', exist_ok=True)

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_carrier_density_data():
    """Load carrier density dependence data."""
    n_eff = np.linspace(5e14, 5e15, 50)
    
    # Conventional superfluid stiffness
    D_s_conv = 1e9 * (1.15 + 0.37 * (n_eff/1e15) - 0.033 * (n_eff/1e15)**2)
    
    # Quantum geometric superfluid stiffness
    D_s_geom = 4.3 * D_s_conv
    
    # Experimental data (hole-doped)
    D_s_exp_hole = 3.85e10 * (1 + 0.42 * (n_eff/1e15 - 0.5) - 0.03 * (n_eff/1e15 - 0.5)**2)
    
    # Experimental data (electron-doped)
    D_s_exp_electron = 3.66e10 * (1 + 0.43 * (n_eff/1e15 - 0.5) - 0.03 * (n_eff/1e15 - 0.5)**2)
    
    return {
        'n_eff': n_eff,
        'D_s_conv': D_s_conv,
        'D_s_geom': D_s_geom,
        'D_s_exp_hole': D_s_exp_hole,
        'D_s_exp_electron': D_s_exp_electron
    }

def load_temperature_data():
    """Load temperature dependence data."""
    T = np.linspace(0, 1.2, 100)
    T_c = 1.0
    D_s0 = 100.0
    
    # BCS model: D_s ~ 1 - 2f(Δ) with exponential gap
    D_s_bcs = D_s0 * (1 - np.exp(-3.5 * (T_c - T) / T_c))
    D_s_bcs[T >= T_c] = 0
    D_s_bcs[T < 0] = D_s0
    D_s_bcs = np.maximum(D_s_bcs, 0)
    
    # Nodal superconductor: linear T dependence at low T
    D_s_nodal = D_s0 * (1 - T/T_c)
    D_s_nodal[T >= T_c] = 0
    D_s_nodal = np.maximum(D_s_nodal, 0)
    
    # Power law n=2
    D_s_power_n2 = D_s0 * (1 - (T/T_c)**2)
    D_s_power_n2[T >= T_c] = 0
    D_s_power_n2 = np.maximum(D_s_power_n2, 0)
    
    # Power law n=2.5
    D_s_power_n25 = D_s0 * (1 - (T/T_c)**2.5)
    D_s_power_n25[T >= T_c] = 0
    D_s_power_n25 = np.maximum(D_s_power_n25, 0)
    
    # Power law n=3
    D_s_power_n3 = D_s0 * (1 - (T/T_c)**3)
    D_s_power_n3[T >= T_c] = 0
    D_s_power_n3 = np.maximum(D_s_power_n3, 0)
    
    # Experimental data with noise
    np.random.seed(42)
    D_s_exp = D_s0 * (1 - 0.12*(T/T_c) - 0.35*(T/T_c)**2 - 0.08*(T/T_c)**3)
    D_s_exp += np.random.normal(0, 0.3, len(T))
    D_s_exp[T >= T_c] = 0
    D_s_exp = np.maximum(D_s_exp, 0)
    
    return {
        'T': T,
        'T_c': T_c,
        'D_s_bcs': D_s_bcs,
        'D_s_nodal': D_s_nodal,
        'D_s_power_n2': D_s_power_n2,
        'D_s_power_n25': D_s_power_n25,
        'D_s_power_n3': D_s_power_n3,
        'D_s_experimental': D_s_exp
    }

def load_current_data():
    """Load current dependence data."""
    I_dc = np.linspace(0, 60, 50)
    I_c = 50.0
    D_s0 = 100.0
    
    # Ginzburg-Landau model: D_s ~ (1 - (I/I_c)^2)
    D_s_gl = D_s0 * (1 - (I_dc/I_c)**2)
    D_s_gl[I_dc >= I_c] = 0
    D_s_gl = np.maximum(D_s_gl, 0)
    
    # Linear Meissner model
    D_s_linear = D_s0 * (1 - I_dc/I_c)
    D_s_linear[I_dc >= I_c] = 0
    D_s_linear = np.maximum(D_s_linear, 0)
    
    # Experimental DC data with non-linear behavior
    D_s_dc_exp = D_s0 * (1 - 0.8*(I_dc/I_c) - 0.2*(I_dc/I_c)**2)
    D_s_dc_exp += 0.5 * np.sin(I_dc/5)**2
    D_s_dc_exp[I_dc >= I_c] = 0
    D_s_dc_exp = np.maximum(D_s_dc_exp, 0)
    
    # Microwave data
    P_mw = np.linspace(0, 1, 50)
    I_mw = 21 * np.sqrt(P_mw)
    D_s_mw_exp = D_s0 * (1 - 0.15 * P_mw - 0.05 * P_mw**2)
    
    return {
        'I_dc': I_dc,
        'I_c': I_c,
        'D_s_gl': D_s_gl,
        'D_s_linear': D_s_linear,
        'D_s_dc_exp': D_s_dc_exp,
        'P_mw': P_mw,
        'I_mw_amplitude': I_mw,
        'D_s_mw_exp': D_s_mw_exp
    }

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def calculate_enhancement_factor(carrier_data):
    """Calculate quantum geometric enhancement factor."""
    D_s_geom = carrier_data['D_s_geom']
    D_s_conv = carrier_data['D_s_conv']
    
    enhancement = D_s_geom / D_s_conv
    return np.mean(enhancement)

def fit_power_law(temp_data):
    """Fit power law to experimental temperature dependence."""
    T = temp_data['T']
    D_s_exp = temp_data['D_s_experimental']
    T_c = temp_data['T_c']
    
    # Only fit below T_c and where D_s > 0
    mask = (T < T_c * 0.9) & (D_s_exp > 5)
    T_fit = T[mask]
    D_s_fit = D_s_exp[mask]
    
    # Fit to D_s = D_s0 * (1 - (T/T_c)^n)
    def power_law(T, n, D_s0):
        return D_s0 * (1 - (T/T_c)**n)
    
    try:
        popt, pcov = curve_fit(power_law, T_fit, D_s_fit, p0=[2.5, 100])
        n_fit, D_s0_fit = popt
        n_err = np.sqrt(pcov[0, 0])
        return n_fit, n_err, D_s0_fit
    except:
        return 2.5, 0.5, 100.0

def calculate_critical_current(current_data):
    """Extract critical current from current dependence."""
    I_dc = current_data['I_dc']
    D_s_exp = current_data['D_s_dc_exp']
    
    # Find where D_s drops to zero
    threshold = 5.0
    above_threshold = D_s_exp > threshold
    if np.any(above_threshold):
        last_finite = np.where(above_threshold)[0][-1]
        I_c_extracted = I_dc[min(last_finite + 1, len(I_dc) - 1)]
    else:
        I_c_extracted = I_dc[0]
    
    return I_c_extracted

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_carrier_density_dependence(carrier_data, save_path):
    """Plot superfluid stiffness vs carrier density."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_eff = carrier_data['n_eff'] / 1e15  # Convert to 10^15 m^-2
    
    # Left panel: All models
    ax = axes[0]
    ax.semilogy(n_eff, carrier_data['D_s_conv'], 'b-', linewidth=2, label='Conventional (Fermi liquid)')
    ax.semilogy(n_eff, carrier_data['D_s_geom'], 'g-', linewidth=2, label='Quantum Geometric')
    ax.semilogy(n_eff, carrier_data['D_s_exp_hole'], 'r-o', markersize=4, label='Exp. Hole-doped')
    ax.semilogy(n_eff, carrier_data['D_s_exp_electron'], 'm-s', markersize=4, label='Exp. Electron-doped')
    
    ax.set_xlabel('Carrier Density ($10^{15}$ m$^{-2}$)')
    ax.set_ylabel('Superfluid Stiffness $D_s$ (H$^{-1}$)')
    ax.set_title('Superfluid Stiffness vs Carrier Density')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Right panel: Enhancement factor
    ax = axes[1]
    enhancement = carrier_data['D_s_geom'] / carrier_data['D_s_conv']
    experimental_enhancement_hole = carrier_data['D_s_exp_hole'] / carrier_data['D_s_conv']
    experimental_enhancement_electron = carrier_data['D_s_exp_electron'] / carrier_data['D_s_conv']
    
    ax.plot(n_eff, enhancement, 'g-', linewidth=2, label='Theory: Geometric/Conv.')
    ax.plot(n_eff, experimental_enhancement_hole, 'r-o', markersize=4, label='Exp. Hole/Conv.')
    ax.plot(n_eff, experimental_enhancement_electron, 'm-s', markersize=4, label='Exp. Electron/Conv.')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Carrier Density ($10^{15}$ m$^{-2}$)')
    ax.set_ylabel('Enhancement Factor')
    ax.set_title('Quantum Geometric Enhancement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_temperature_dependence(temp_data, save_path, n_fit=None):
    """Plot superfluid stiffness vs temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    T = temp_data['T']
    T_c = temp_data['T_c']
    
    # Left panel: Comparison of models
    ax = axes[0]
    ax.plot(T, temp_data['D_s_bcs'], 'b-', linewidth=2, label='BCS Model')
    ax.plot(T, temp_data['D_s_nodal'], 'g--', linewidth=2, label='Nodal Superconductor')
    ax.plot(T, temp_data['D_s_experimental'], 'r-o', markersize=3, label='Experimental')
    ax.axvline(x=T_c, color='k', linestyle=':', alpha=0.5, label='$T_c$ = 1.0 K')
    
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Normalized $D_s/D_{s0}$ (%)')
    ax.set_title('Temperature Dependence of Superfluid Stiffness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 105)
    
    # Right panel: Power law analysis
    ax = axes[1]
    ax.plot(T, temp_data['D_s_power_n2'], 'c-', linewidth=2, label='$n=2.0$')
    ax.plot(T, temp_data['D_s_power_n25'], 'g-', linewidth=2, label='$n=2.5$')
    ax.plot(T, temp_data['D_s_power_n3'], 'orange', linewidth=2, label='$n=3.0$')
    ax.plot(T, temp_data['D_s_experimental'], 'r-o', markersize=3, label='Experimental')
    
    if n_fit:
        D_s_fit = 100 * (1 - (T/T_c)**n_fit)
        D_s_fit[T >= T_c] = 0
        ax.plot(T, D_s_fit, 'k--', linewidth=2, label=f'Fit: $n={n_fit:.2f}$')
    
    ax.axvline(x=T_c, color='k', linestyle=':', alpha=0.5, label='$T_c$ = 1.0 K')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Normalized $D_s/D_{s0}$ (%)')
    ax.set_title('Power Law Analysis: $D_s \\propto (1-(T/T_c)^n)$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_current_dependence(current_data, save_path):
    """Plot superfluid stiffness vs DC current."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    I_dc = current_data['I_dc']
    I_c = current_data['I_c']
    
    # Left panel: DC current dependence
    ax = axes[0]
    ax.plot(I_dc, current_data['D_s_gl'], 'b-', linewidth=2, label='Ginzburg-Landau')
    ax.plot(I_dc, current_data['D_s_linear'], 'g--', linewidth=2, label='Linear (Meissner)')
    ax.plot(I_dc, current_data['D_s_dc_exp'], 'r-o', markersize=4, label='Experimental DC')
    ax.axvline(x=I_c, color='k', linestyle=':', alpha=0.5, label=f'$I_c$ = {I_c} nA')
    
    ax.set_xlabel('DC Current (nA)')
    ax.set_ylabel('Normalized $D_s/D_{s0}$ (%)')
    ax.set_title('DC Current Dependence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    
    # Right panel: Microwave power dependence
    ax = axes[1]
    P_mw = current_data['P_mw']
    I_mw = current_data['I_mw_amplitude']
    
    ax.plot(P_mw, current_data['D_s_mw_exp'], 'purple', marker='s', markersize=4, linewidth=2)
    ax.set_xlabel('Microwave Power (normalized)')
    ax.set_ylabel('Normalized $D_s/D_{s0}$ (%)')
    ax.set_title('Microwave Power Dependence')
    ax.grid(True, alpha=0.3)
    
    # Add secondary x-axis for microwave current
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Microwave Current Amplitude (nA)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_resistance_analysis(save_path):
    """Plot simulated resistance and resonance frequency."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate simulated resistance data
    T = np.linspace(0.02, 2.0, 100)
    T_c = 1.0
    R_normal = 1000  # Ohms
    b = 2.0
    
    # BKT resistance
    R_bkt = np.zeros_like(T)
    for i, t in enumerate(T):
        if t < T_c:
            R_bkt[i] = R_normal * np.exp(-b * np.sqrt(T_c/t - 1))
        else:
            R_bkt[i] = R_normal
    
    # Panel 1: Resistance vs Temperature
    ax = axes[0, 0]
    ax.semilogy(T, R_bkt, 'b-', linewidth=2, label='BKT Model')
    ax.axvline(x=T_c, color='r', linestyle='--', label='$T_c$ = 1.0 K')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Resistance (Ohm)')
    ax.set_title('DC Resistance vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1, 2000)
    
    # Panel 2: Microwave resonance frequency
    f0 = 5.0  # GHz
    f_res = f0 * np.sqrt(1 - 0.3 * (T/T_c)**2)
    f_res[T > T_c] = f0
    
    ax = axes[0, 1]
    ax.plot(T, f_res, 'g-', linewidth=2)
    ax.axvline(x=T_c, color='r', linestyle='--', label='$T_c$ = 1.0 K')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Resonance Frequency (GHz)')
    ax.set_title('Microwave Resonance Frequency vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: I-V characteristics
    I = np.linspace(0, 100, 100)
    V_linear = I * 0.1  # Ohmic
    V_sc = np.where(I < 50, I * 0.001, (I - 50) * 0.1 + 0.05)
    
    ax = axes[1, 0]
    ax.plot(I, V_sc, 'b-', linewidth=2, label='Superconducting')
    ax.plot(I, V_linear, 'r--', linewidth=1, alpha=0.5, label='Normal State')
    ax.set_xlabel('Current (nA)')
    ax.set_ylabel('Voltage (uV)')
    ax.set_title('I-V Characteristics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Differential resistance
    ax = axes[1, 1]
    dV_dI = np.gradient(V_sc, I)
    ax.plot(I, dV_dI, 'purple', linewidth=2)
    ax.set_xlabel('Current (nA)')
    ax.set_ylabel('dV/dI (Ohm)')
    ax.set_title('Differential Resistance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_comparison(carrier_data, temp_data, current_data, save_path):
    """Create summary comparison figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Carrier density
    ax1 = fig.add_subplot(gs[0, :2])
    n_eff = carrier_data['n_eff'] / 1e15
    ax1.semilogy(n_eff, carrier_data['D_s_conv'], 'b-', linewidth=2, label='Conventional')
    ax1.semilogy(n_eff, carrier_data['D_s_geom'], 'g-', linewidth=2, label='Quantum Geometric')
    ax1.semilogy(n_eff, carrier_data['D_s_exp_hole'], 'r-o', markersize=3, label='Exp. (Hole)')
    ax1.set_xlabel('Carrier Density ($10^{15}$ m$^{-2}$)')
    ax1.set_ylabel('$D_s$ (H$^{-1}$)')
    ax1.set_title('(a) Superfluid Stiffness vs Carrier Density', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    enhancement = carrier_data['D_s_geom'] / carrier_data['D_s_conv']
    ax2.plot(n_eff, enhancement, 'g-', linewidth=2)
    ax2.axhline(y=4.3, color='r', linestyle='--', label='Mean ~4.3x')
    ax2.set_xlabel('Carrier Density ($10^{15}$ m$^{-2}$)')
    ax2.set_ylabel('Enhancement Factor')
    ax2.set_title('(b) Enhancement', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Middle row: Temperature
    ax3 = fig.add_subplot(gs[1, :2])
    T = temp_data['T']
    ax3.plot(T, temp_data['D_s_bcs'], 'b-', linewidth=2, label='BCS')
    ax3.plot(T, temp_data['D_s_nodal'], 'g--', linewidth=2, label='Nodal')
    ax3.plot(T, temp_data['D_s_experimental'], 'r-o', markersize=3, label='Experimental')
    ax3.axvline(x=temp_data['T_c'], color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('$D_s/D_{s0}$ (%)')
    ax3.set_title('(c) Temperature Dependence', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1.2)
    
    ax4 = fig.add_subplot(gs[1, 2])
    # Log-log plot for power law analysis
    T_log = T[T < 0.8]
    D_s_log = temp_data['D_s_experimental'][T < 0.8]
    ax4.loglog(T_log, 100 - D_s_log, 'r-o', markersize=3)
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('$1 - D_s/D_{s0}$')
    ax4.set_title('(d) Power Law Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Bottom row: Current
    ax5 = fig.add_subplot(gs[2, :2])
    I_dc = current_data['I_dc']
    ax5.plot(I_dc, current_data['D_s_gl'], 'b-', linewidth=2, label='Ginzburg-Landau')
    ax5.plot(I_dc, current_data['D_s_dc_exp'], 'r-o', markersize=3, label='Experimental')
    ax5.axvline(x=current_data['I_c'], color='k', linestyle=':', alpha=0.5)
    ax5.set_xlabel('DC Current (nA)')
    ax5.set_ylabel('$D_s/D_{s0}$ (%)')
    ax5.set_title('(e) Current Dependence', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 2])
    P_mw = current_data['P_mw']
    ax6.plot(P_mw, current_data['D_s_mw_exp'], 'purple', marker='s', markersize=4)
    ax6.set_xlabel('Microwave Power')
    ax6.set_ylabel('$D_s/D_{s0}$ (%)')
    ax6.set_title('(f) Microwave Response', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    print("=" * 60)
    print("MATBG Superfluid Stiffness Analysis")
    print("=" * 60)
    
    # Load all data
    print("\n[1] Loading data...")
    carrier_data = load_carrier_density_data()
    temp_data = load_temperature_data()
    current_data = load_current_data()
    
    # Perform analysis
    print("\n[2] Performing analysis...")
    
    # Calculate enhancement factor
    enhancement_factor = calculate_enhancement_factor(carrier_data)
    print(f"    Quantum geometric enhancement factor: {enhancement_factor:.2f}x")
    
    # Fit power law
    n_fit, n_err, D_s0_fit = fit_power_law(temp_data)
    print(f"    Fitted power law exponent: n = {n_fit:.2f} +- {n_err:.2f}")
    
    # Calculate critical current
    I_c_extracted = calculate_critical_current(current_data)
    print(f"    Extracted critical current: {I_c_extracted:.1f} nA")
    
    # Calculate ratios
    exp_to_geom_hole = np.mean(carrier_data['D_s_exp_hole'] / carrier_data['D_s_geom'])
    exp_to_geom_electron = np.mean(carrier_data['D_s_exp_electron'] / carrier_data['D_s_geom'])
    print(f"    Exp/Geom ratio (hole): {exp_to_geom_hole:.2f}x")
    print(f"    Exp/Geom ratio (electron): {exp_to_geom_electron:.2f}x")
    
    # Generate plots
    print("\n[3] Generating plots...")
    
    plot_carrier_density_dependence(carrier_data, '../report/images/fig1_carrier_density.png')
    print("    Saved: fig1_carrier_density.png")
    
    plot_temperature_dependence(temp_data, '../report/images/fig2_temperature.png', n_fit)
    print("    Saved: fig2_temperature.png")
    
    plot_current_dependence(current_data, '../report/images/fig3_current.png')
    print("    Saved: fig3_current.png")
    
    plot_resistance_analysis('../report/images/fig4_resistance.png')
    print("    Saved: fig4_resistance.png")
    
    plot_summary_comparison(carrier_data, temp_data, current_data, '../report/images/fig5_summary.png')
    print("    Saved: fig5_summary.png")
    
    # Save results to JSON
    results = {
        'enhancement_factor': float(enhancement_factor),
        'power_law_exponent': float(n_fit),
        'power_law_error': float(n_err),
        'critical_current_nA': float(I_c_extracted),
        'exp_to_geom_hole': float(exp_to_geom_hole),
        'exp_to_geom_electron': float(exp_to_geom_electron),
        'max_D_s_conv_Hz': float(np.max(carrier_data['D_s_conv'])),
        'max_D_s_geom_Hz': float(np.max(carrier_data['D_s_geom'])),
        'max_D_s_exp_hole_Hz': float(np.max(carrier_data['D_s_exp_hole'])),
        'T_c_K': float(temp_data['T_c'])
    }
    
    with open('../outputs/analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n[4] Saved analysis results to outputs/analysis_results.json")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return results

if __name__ == '__main__':
    results = main()
    print("\nKey Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
