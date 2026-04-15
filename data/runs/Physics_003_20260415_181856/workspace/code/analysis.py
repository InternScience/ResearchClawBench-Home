"""
Floquet-Bloch States Analysis for Monolayer Epitaxial Graphene
==============================================================

This script analyzes time-resolved and angle-resolved photoemission spectroscopy (tr-ARPES)
data to demonstrate the existence of Floquet-Bloch states (photon-dressed states) in 
monolayer epitaxial graphene under mid-infrared pump excitation.

Author: Research Analysis Pipeline
Date: 2026-04-15
"""

import numpy as np
import h5py
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

# Set matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150

# Paths
DATA_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_003_20260415_181856/data'
OUTPUT_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_003_20260415_181856/outputs'
FIGURE_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_003_20260415_181856/report/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Physical constants
HBAR = 6.582119569e-16  # eV·s
C = 299792458  # m/s

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_raw_data():
    """Load raw tr-ARPES data from HDF5 file."""
    with h5py.File(os.path.join(DATA_DIR, 'raw_trARPES_data.h5'), 'r') as f:
        data = {
            'energy_axis': np.array(f['energy_axis']),
            'kx_axis': np.array(f['kx_axis']),
            'polarization_angles': np.array(f['polarization_angles']),
            'pump_off_spectrum': np.array(f['pump_off_spectrum']),
            'time_delays': np.array(f['time_delays']),
        }
        # Load pump-on spectra for each angle
        for angle in data['polarization_angles']:
            data[f'pump_on_angle_{angle}'] = np.array(f[f'pump_on_angle_{angle}'])
    return data


def load_processed_data():
    """Load processed band data from JSON."""
    with open(os.path.join(DATA_DIR, 'processed_band_data.json'), 'r') as f:
        return json.load(f)


def load_polarization_data():
    """Load polarization dependence data from CSV."""
    return pd.read_csv(os.path.join(DATA_DIR, 'polarization_dependence_data.csv'))


# ============================================================================
# Analysis Functions
# ============================================================================

def calculate_difference_spectrum(pump_off, pump_on):
    """Calculate pump-induced difference spectrum."""
    return pump_on - pump_off


def fit_polarization_dependence(angles, intensities):
    """Fit polarization dependence to sinusoidal function."""
    def sinusoidal(theta, I0, A, phi):
        return I0 + A * np.cos(2 * (theta - phi))
    
    # Initial guess
    I0_guess = np.mean(intensities)
    A_guess = (np.max(intensities) - np.min(intensities)) / 2
    phi_guess = 0
    
    popt, pcov = curve_fit(sinusoidal, angles, intensities, 
                           p0=[I0_guess, A_guess, phi_guess])
    return popt, pcov, sinusoidal


def extract_replica_band_gap(replica_bands):
    """Calculate energy gap between replica bands."""
    # Separate positive and negative order replica bands
    positive_order = [rb for rb in replica_bands if rb['order'] > 0]
    negative_order = [rb for rb in replica_bands if rb['order'] < 0]
    
    if positive_order and negative_order:
        # Calculate average energy for each order
        avg_pos_energy = np.mean([rb['energy'] for rb in positive_order])
        avg_neg_energy = np.mean([rb['energy'] for rb in negative_order])
        return avg_pos_energy - avg_neg_energy
    return None


def calculate_floquet_energy(photon_energy_eV=0.248):  # 5 μm ~ 0.248 eV
    """Calculate Floquet energy spacing for 5 μm pump wavelength."""
    return photon_energy_eV


# ============================================================================
# Visualization Functions
# ============================================================================

def create_custom_cmap():
    """Create custom colormap for ARPES data."""
    colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#800000']
    return LinearSegmentedColormap.from_list('arpes', colors)


def plot_raw_spectra(data, save_path=None):
    """Plot raw ARPES spectra for pump-off and pump-on conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = create_custom_cmap()
    
    energy = data['energy_axis']
    kx = data['kx_axis']
    
    # Pump-off spectrum
    ax = axes[0, 0]
    im = ax.pcolormesh(kx, energy, data['pump_off_spectrum'], shading='auto', cmap=cmap)
    ax.axhline(y=-0.3, color='white', linestyle='--', linewidth=1.5, alpha=0.8, label='Dirac point')
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Pump-Off Spectrum (Equilibrium)')
    ax.set_xlim([kx.min(), kx.max()])
    ax.set_ylim([energy.min(), energy.max()])
    plt.colorbar(im, ax=ax, label='Intensity (arb. units)')
    ax.legend(loc='upper right')
    
    # Pump-on spectrum at 0 degrees
    ax = axes[0, 1]
    diff_0 = calculate_difference_spectrum(data['pump_off_spectrum'], data['pump_on_angle_0'])
    im = ax.pcolormesh(kx, energy, data['pump_on_angle_0'], shading='auto', cmap=cmap)
    ax.axhline(y=-0.3, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=-0.3 + 0.248, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8, label='n=+1 replica')
    ax.axhline(y=-0.3 - 0.248, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8, label='n=-1 replica')
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(r'Pump-On Spectrum ($\theta_p = 0°$)')
    ax.set_xlim([kx.min(), kx.max()])
    ax.set_ylim([energy.min(), energy.max()])
    plt.colorbar(im, ax=ax, label='Intensity (arb. units)')
    ax.legend(loc='upper right')
    
    # Difference spectrum at 0 degrees
    ax = axes[1, 0]
    vmax = np.abs(diff_0).max()
    im = ax.pcolormesh(kx, energy, diff_0, shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.axhline(y=-0.3, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(r'Pump-Induced Difference ($\theta_p = 0°$)')
    ax.set_xlim([kx.min(), kx.max()])
    ax.set_ylim([energy.min(), energy.max()])
    plt.colorbar(im, ax=ax, label='ΔIntensity')
    
    # Pump-on at 90 degrees
    ax = axes[1, 1]
    diff_90 = calculate_difference_spectrum(data['pump_off_spectrum'], data['pump_on_angle_90'])
    im = ax.pcolormesh(kx, energy, data['pump_on_angle_90'], shading='auto', cmap=cmap)
    ax.axhline(y=-0.3, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(r'Pump-On Spectrum ($\theta_p = 90°$)')
    ax.set_xlim([kx.min(), kx.max()])
    ax.set_ylim([energy.min(), energy.max()])
    plt.colorbar(im, ax=ax, label='Intensity (arb. units)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_floquet_bands(processed_data, save_path=None):
    """Plot Floquet-Bloch bands with replica band positions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    band_dispersion = processed_data['band_dispersion']
    replica_bands = processed_data['replica_bands']
    dirac_point = processed_data['dirac_point']
    
    # Extract main band
    energies = np.array([p['energy'] for p in band_dispersion])
    kxs = np.array([p['kx'] for p in band_dispersion])
    intensities = np.array([p['intensity'] for p in band_dispersion])
    
    # Left panel: Band dispersion with replica bands
    ax = axes[0]
    scatter = ax.scatter(kxs, energies, c=intensities, cmap='viridis', s=20, alpha=0.7, label='Main Dirac cone')
    
    # Plot replica bands
    colors = {'-1': 'blue', '1': 'red'}
    labels = {'-1': 'n = -1 replica', '1': 'n = +1 replica'}
    for rb in replica_bands:
        order = str(rb['order'])
        ax.scatter(rb['kx'], rb['energy'], c=colors[order], s=200, marker='*', 
                   edgecolors='black', linewidths=1, zorder=5, label=labels[order])
        labels[order] = None  # Only label once
    
    # Mark Dirac point
    ax.scatter(dirac_point[0], dirac_point[1], c='green', s=300, marker='X', 
               edgecolors='black', linewidths=2, zorder=5, label='Dirac point')
    
    # Add Floquet energy spacing
    floquet_E = 0.248  # 5 μm pump
    ax.axhline(y=dirac_point[1] + floquet_E, color='red', linestyle=':', alpha=0.5)
    ax.axhline(y=dirac_point[1] - floquet_E, color='blue', linestyle=':', alpha=0.5)
    
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Floquet-Bloch Band Structure')
    ax.legend(loc='upper right')
    ax.set_xlim([kxs.min(), kxs.max()])
    ax.set_ylim([energies.min(), energies.max()])
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Intensity')
    
    # Right panel: Schematic of Floquet sidebands
    ax = axes[1]
    k_range = np.linspace(-0.1, 0.1, 100)
    v_f = 6  # eV·Å (Fermi velocity)
    E_dirac = -0.3  # Dirac point energy
    
    # Main Dirac cone
    E_cone = E_dirac + v_f * np.abs(k_range)
    ax.plot(k_range, E_cone, 'k-', linewidth=2, label='n = 0 (main)')
    ax.plot(k_range, -E_cone + 2*E_dirac, 'k-', linewidth=2)
    
    # Floquet replicas
    E_replica_plus = E_cone + floquet_E
    E_replica_minus = E_cone - floquet_E
    ax.plot(k_range, E_replica_plus, 'r--', linewidth=1.5, label='n = +1')
    ax.plot(k_range, E_replica_minus, 'b--', linewidth=1.5, label='n = -1')
    ax.plot(k_range, -E_replica_plus + 2*E_dirac, 'r--', linewidth=1.5)
    ax.plot(k_range, -E_replica_minus + 2*E_dirac, 'b--', linewidth=1.5)
    
    # Mark Dirac point
    ax.scatter([0], [E_dirac], c='green', s=300, marker='X', 
               edgecolors='black', linewidths=2, zorder=5)
    
    # Mark replica band positions from data
    for rb in replica_bands:
        ax.scatter([rb['kx']], [rb['energy']], c='cyan', s=150, marker='o', 
                   edgecolors='black', linewidths=1, zorder=5, alpha=0.7)
    
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Schematic of Floquet Sidebands')
    ax.legend(loc='upper right')
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.6, 0.4])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_polarization_analysis(pol_data, save_path=None):
    """Analyze and plot polarization dependence of replica band intensity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    angles_deg = pol_data['angle_degrees'].values
    angles_rad = pol_data['angle_radians'].values
    intensities = pol_data['intensity'].values
    
    # Fit sinusoidal function
    popt, pcov, fit_func = fit_polarization_dependence(angles_rad, intensities)
    I0, A, phi = popt
    
    # Fine grid for smooth curve
    theta_fine = np.linspace(0, np.pi, 200)
    fit_curve = fit_func(theta_fine, *popt)
    
    # Left panel: Polar plot
    ax = axes[0]
    ax.plot(theta_fine, fit_curve, 'b-', linewidth=2, label=f'Fit: $I_0={I0:.3f}$, $A={A:.3f}$')
    ax.scatter(angles_rad, intensities, c='red', s=100, zorder=5, label='Experimental Data')
    ax.set_xlabel(r'$\theta_p$ (rad)')
    ax.set_ylabel('Replica Band Intensity')
    ax.set_title('Polarization Dependence (Cartesian)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, np.pi])
    
    # Right panel: Polar plot
    ax = axes[1], projection='polar'
    theta_fine_full = np.linspace(0, 2*np.pi, 400)
    fit_curve_full = fit_func(theta_fine_full, *popt)
    ax[0].plot(theta_fine_full, fit_curve_full, 'b-', linewidth=2)
    ax[0].scatter(angles_rad, intensities, c='red', s=100, zorder=5)
    # Mirror points for 180-360 degrees
    ax[0].scatter(angles_rad + np.pi, intensities, c='red', s=100, zorder=5)
    ax[0].set_theta_zero_location('E')
    ax[0].set_theta_direction(1)
    ax[0].set_title('Polarization Dependence (Polar)', pad=20)
    ax[0].set_ylim([I0 - 2*abs(A), I0 + 2*abs(A)])
    ax[0].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return popt, pcov


def plot_polarization_comparison(data, save_path=None):
    """Compare ARPES spectra at different polarization angles."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    cmap = create_custom_cmap()
    
    angles = data['polarization_angles']
    energy = data['energy_axis']
    kx = data['kx_axis']
    
    for idx, angle in enumerate(angles):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        pump_on = data[f'pump_on_angle_{angle}']
        diff = calculate_difference_spectrum(data['pump_off_spectrum'], pump_on)
        
        vmax = np.abs(diff).max()
        im = ax.pcolormesh(kx, energy, diff, shading='auto', cmap='RdBu_r', 
                           vmin=-vmax, vmax=vmax)
        ax.axhline(y=-0.3, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=-0.3 + 0.248, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=-0.3 - 0.248, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title(f'$\\theta_p = {angle}°$')
        ax.set_xlim([kx.min(), kx.max()])
        ax.set_ylim([energy.min(), energy.max()])
        plt.colorbar(im, ax=ax, label='ΔI')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_energy_distribution_curves(data, processed_data, save_path=None):
    """Plot energy distribution curves (EDCs) at specific momentum values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    energy = data['energy_axis']
    
    # Find indices near kx = 0
    kx = data['kx_axis']
    kx_center_idx = np.argmin(np.abs(kx))
    kx_range = slice(max(0, kx_center_idx - 5), min(len(kx), kx_center_idx + 5))
    
    # EDC at kx ≈ 0 for different polarization angles
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(data['polarization_angles'])))
    
    for idx, angle in enumerate(data['polarization_angles']):
        pump_on = data[f'pump_on_angle_{angle}']
        edc = np.mean(pump_on[:, kx_range], axis=1)
        ax.plot(energy, edc, label=f'$\\theta_p = {angle}°$', color=colors[idx], linewidth=1.5)
    
    ax.axvline(x=-0.3, color='black', linestyle='--', alpha=0.5, label='Dirac point')
    ax.axvline(x=-0.3 + 0.248, color='red', linestyle=':', alpha=0.5, label='n=+1')
    ax.axvline(x=-0.3 - 0.248, color='blue', linestyle=':', alpha=0.5, label='n=-1')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.set_title('Energy Distribution Curves at $k_x \\approx 0$')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim([energy.min(), energy.max()])
    ax.grid(True, alpha=0.3)
    
    # Momentum distribution curves (MDCs) at specific energies
    ax = axes[1]
    
    # Find indices near Dirac point and replica energies
    dirac_E = -0.3
    replica_E_plus = dirac_E + 0.248
    replica_E_minus = dirac_E - 0.248
    
    idx_dirac = np.argmin(np.abs(energy - dirac_E))
    idx_plus = np.argmin(np.abs(energy - replica_E_plus))
    idx_minus = np.argmin(np.abs(energy - replica_E_minus))
    
    pump_on_0 = data['pump_on_angle_0']
    
    ax.plot(kx, pump_on_0[idx_dirac, :], 'k-', linewidth=2, label=f'E = {dirac_E:.2f} eV (n=0)')
    ax.plot(kx, pump_on_0[idx_plus, :], 'r--', linewidth=2, label=f'E = {replica_E_plus:.2f} eV (n=+1)')
    ax.plot(kx, pump_on_0[idx_minus, :], 'b:', linewidth=2, label=f'E = {replica_E_minus:.2f} eV (n=-1)')
    
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.set_title('Momentum Distribution Curves ($\\theta_p = 0°$)')
    ax.legend(loc='upper right')
    ax.set_xlim([kx.min(), kx.max()])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    """Run complete analysis pipeline."""
    print("=" * 70)
    print("Floquet-Bloch States Analysis Pipeline")
    print("Monolayer Epitaxial Graphene - tr-ARPES Study")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data files...")
    raw_data = load_raw_data()
    processed_data = load_processed_data()
    pol_data = load_polarization_data()
    print("    ✓ Raw tr-ARPES data loaded")
    print(f"    ✓ Energy axis: {raw_data['energy_axis'].min():.2f} to {raw_data['energy_axis'].max():.2f} eV")
    print(f"    ✓ Momentum axis: {raw_data['kx_axis'].min():.3f} to {raw_data['kx_axis'].max():.3f} Å⁻¹")
    print(f"    ✓ Polarization angles: {raw_data['polarization_angles']}")
    print("    ✓ Processed band data loaded")
    print(f"    ✓ Dirac point: E = {processed_data['dirac_point'][1]:.3f} eV, kx = {processed_data['dirac_point'][0]:.3f} Å⁻¹")
    print(f"    ✓ Found {len(processed_data['replica_bands'])} replica bands")
    print("    ✓ Polarization dependence data loaded")
    
    # Calculate key quantities
    print("\n[2] Calculating Floquet-Bloch parameters...")
    floquet_E = calculate_floquet_energy()
    print(f"    ✓ Pump photon energy (5 μm): {floquet_E:.3f} eV")
    
    replica_gap = extract_replica_band_gap(processed_data['replica_bands'])
    if replica_gap:
        print(f"    ✓ Energy spacing between n=±1 replicas: {replica_gap:.3f} eV")
        print(f"    ✓ Expected spacing: {2*floquet_E:.3f} eV")
        print(f"    ✓ Agreement: {abs(replica_gap - 2*floquet_E)/floquet_E*100:.1f}%")
    
    # Generate figures
    print("\n[3] Generating figures...")
    
    print("    → Figure 1: Raw ARPES spectra")
    plot_raw_spectra(raw_data, save_path=os.path.join(FIGURE_DIR, 'figure_01_raw_spectra.png'))
    
    print("    → Figure 2: Floquet-Bloch band structure")
    plot_floquet_bands(processed_data, save_path=os.path.join(FIGURE_DIR, 'figure_02_floquet_bands.png'))
    
    print("    → Figure 3: Polarization dependence analysis")
    popt, pcov = plot_polarization_analysis(pol_data, 
                                            save_path=os.path.join(FIGURE_DIR, 'figure_03_polarization.png'))
    print(f"        Fit parameters: I₀ = {popt[0]:.4f}, A = {popt[1]:.4f}, φ = {np.degrees(popt[2]):.1f}°")
    
    print("    → Figure 4: Polarization angle comparison")
    plot_polarization_comparison(raw_data, save_path=os.path.join(FIGURE_DIR, 'figure_04_angle_comparison.png'))
    
    print("    → Figure 5: EDCs and MDCs")
    plot_energy_distribution_curves(raw_data, processed_data, 
                                    save_path=os.path.join(FIGURE_DIR, 'figure_05_edc_mdc.png'))
    
    # Save analysis results
    print("\n[4] Saving analysis results...")
    results = {
        'floquet_photon_energy_eV': floquet_E,
        'pump_wavelength_um': 5.0,
        'dirac_point': processed_data['dirac_point'],
        'replica_band_count': len(processed_data['replica_bands']),
        'replica_energy_spacing_eV': replica_gap,
        'polarization_fit': {
            'I0': float(popt[0]),
            'amplitude': float(popt[1]),
            'phase_rad': float(popt[2]),
            'phase_deg': float(np.degrees(popt[2]))
        },
        'replica_bands': processed_data['replica_bands']
    }
    
    with open(os.path.join(OUTPUT_DIR, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("    ✓ Analysis results saved to outputs/analysis_results.json")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()
