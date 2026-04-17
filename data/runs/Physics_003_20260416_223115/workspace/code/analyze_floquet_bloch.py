#!/usr/bin/env python3
"""
Analysis of Floquet-Bloch states in epitaxial graphene using tr-ARPES data.

This script processes raw tr-ARPES data, processed band data, and polarization
dependence measurements to characterize photon-dressed Floquet replica bands.
"""

import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Paths
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
REPORT_IMAGES_DIR = Path("report/images")

# Ensure output directories exist
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORT_IMAGES_DIR.mkdir(exist_ok=True)


def load_raw_trarpes_data(filepath):
    """Load raw tr-ARPES data from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        data = {
            'energy_axis': f['energy_axis'][:],
            'kx_axis': f['kx_axis'][:],
            'polarization_angles': f['polarization_angles'][:],
            'time_delays': f['time_delays'][:],
            'pump_off_spectrum': f['pump_off_spectrum'][:],
        }
        # Load all pump-on spectra
        for angle in data['polarization_angles']:
            key = f'pump_on_angle_{angle}'
            if key in f:
                data[f'pump_on_{angle}'] = f[key][:]
    return data


def load_processed_band_data(filepath):
    """Load processed band data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_polarization_data(filepath):
    """Load polarization dependence data from CSV file."""
    return pd.read_csv(filepath)


def plot_data_overview(raw_data, output_path):
    """Figure 1: Overview of pump-off and pump-on ARPES spectra."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    energy = raw_data['energy_axis']
    kx = raw_data['kx_axis']
    KX, ENERGY = np.meshgrid(kx, energy)
    
    # Panel A: Pump-off spectrum
    ax = axes[0, 0]
    im = ax.pcolormesh(KX, ENERGY, raw_data['pump_off_spectrum'], 
                       shading='gouraud', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('Momentum k_x (Å⁻¹)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Pump-Off: Equilibrium Dirac Cone')
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
    
    # Panel B: Pump-on spectrum (angle 0)
    ax = axes[0, 1]
    im = ax.pcolormesh(KX, ENERGY, raw_data['pump_on_0'], 
                       shading='gouraud', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('Momentum k_x (Å⁻¹)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Pump-On (θ=0°): Floquet Replica Bands')
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
    
    # Panel C: Difference spectrum (pump-on - pump-off)
    ax = axes[1, 0]
    diff = raw_data['pump_on_0'] - raw_data['pump_off_spectrum']
    im = ax.pcolormesh(KX, ENERGY, diff, shading='gouraud', 
                       cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax.set_xlabel('Momentum k_x (Å⁻¹)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Difference Spectrum: Pump-Induced Changes')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.colorbar(im, ax=ax, label='Δ Intensity (a.u.)')
    
    # Panel D: Energy distribution curves at Dirac point
    ax = axes[1, 1]
    dirac_idx = np.argmin(np.abs(kx))  # Find kx closest to 0
    ax.plot(energy, raw_data['pump_off_spectrum'][:, dirac_idx + 75], 
            'k-', label='Pump-Off', linewidth=2)
    ax.plot(energy, raw_data['pump_on_0'][:, dirac_idx + 75], 
            'r-', label='Pump-On (θ=0°)', linewidth=2)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('Energy Distribution Curves at k_x ≈ 0')
    ax.legend()
    ax.set_ylim([-0.6, 0.4])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_replica_bands(processed_data, raw_data, output_path):
    """Figure 2: Extracted replica band positions on dispersion."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    energy = raw_data['energy_axis']
    kx = raw_data['kx_axis']
    KX, ENERGY = np.meshgrid(kx, energy)
    
    # Background: pump-on spectrum
    im = ax.pcolormesh(KX, ENERGY, raw_data['pump_on_0'], 
                       shading='gouraud', cmap='viridis', vmin=0, vmax=0.7, alpha=0.6)
    
    # Plot main Dirac cone
    dirac_point = processed_data['dirac_point']
    ax.plot(dirac_point[1], dirac_point[0], 'wo', markersize=12, 
            markeredgecolor='black', markeredgewidth=2, label='Dirac Point')
    
    # Plot replica bands
    replica_bands = processed_data['replica_bands']
    colors = {'-1': 'blue', '1': 'red'}
    
    for replica in replica_bands:
        order = str(replica['order'])
        ax.plot(replica['kx'], replica['energy'], 
                'o', color=colors.get(order, 'green'), 
                markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                label=f'Replica n={order}' if f'n={order}' not in [t.get_label() for t in ax.get_legend_handles_labels()[1]] else "")
    
    # Add annotations for replica bands
    for replica in replica_bands:
        order = replica['order']
        ax.annotate(f'n={order}', 
                   xy=(replica['kx'], replica['energy']),
                   xytext=(replica['kx'] + 0.01*np.sign(replica['kx']), 
                          replica['energy'] + 0.05*np.sign(order)),
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_xlabel('Momentum k_x (Å⁻¹)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Floquet Replica Bands: Main Dirac Cone and Photon-Dressed States')
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_polarization_dependence(polarization_data, output_path):
    """Figure 3: Replica band intensity vs polarization angle."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    angles_deg = polarization_data['angle_degrees'].values
    angles_rad = polarization_data['angle_radians'].values
    intensities = polarization_data['intensity'].values
    
    # Scatter plot with error bars (using standard deviation as estimate)
    ax.scatter(angles_deg, intensities, s=100, c='darkblue', zorder=3, label='Measured')
    
    # Fit to cos²(θ) model expected for linearly polarized light coupling
    def cos2_model(theta, A, B, theta0):
        return A * np.cos(theta - theta0)**2 + B
    
    try:
        popt, pcov = curve_fit(cos2_model, angles_rad, intensities, 
                               p0=[0.05, 0.47, 0])
        theta_fit = np.linspace(-np.pi/2, 3*np.pi/2, 200)
        intensity_fit = cos2_model(theta_fit, *popt)
        ax.plot(np.degrees(theta_fit), intensity_fit, 'r-', linewidth=2, 
                label=f'Fit: A·cos²(θ-θ₀)+B\nA={popt[0]:.4f}, θ₀={np.degrees(popt[2]):.1f}°')
        
        # Save fit results
        fit_results = {
            'model': 'A*cos^2(theta - theta0) + B',
            'parameters': {
                'A': float(popt[0]),
                'B': float(popt[1]),
                'theta0_rad': float(popt[2]),
                'theta0_deg': float(np.degrees(popt[2]))
            },
            'covariance': pcov.tolist()
        }
        with open(OUTPUTS_DIR / 'polarization_fit_results.json', 'w') as f:
            json.dump(fit_results, f, indent=2)
        print(f"Saved fit results to: {OUTPUTS_DIR / 'polarization_fit_results.json'}")
    except Exception as e:
        print(f"Fit failed: {e}")
    
    ax.set_xlabel('Pump Polarization Angle θ_p (degrees)')
    ax.set_ylabel('Replica Band Intensity (a.u.)')
    ax.set_title('Polarization Dependence of Floquet Replica Band')
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_band_dispersion(processed_data, raw_data, output_path):
    """Figure 4: Full band dispersion with Floquet replicas."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    energy = raw_data['energy_axis']
    kx = raw_data['kx_axis']
    KX, ENERGY = np.meshgrid(kx, energy)
    
    # Background: pump-off spectrum
    im = ax.pcolormesh(KX, ENERGY, raw_data['pump_off_spectrum'], 
                       shading='gouraud', cmap='magma', vmin=0, vmax=0.8)
    
    # Extract and plot band dispersion from processed data
    band_dispersion = processed_data['band_dispersion']
    energies = [pt['energy'] for pt in band_dispersion]
    kxs = [pt['kx'] for pt in band_dispersion]
    intensities = [pt['intensity'] for pt in band_dispersion]
    
    # Scatter plot colored by intensity
    scatter = ax.scatter(kxs, energies, c=intensities, s=50, 
                         cmap='coolwarm', edgecolors='black', linewidth=0.5,
                         label='Extracted Band Dispersion')
    
    # Mark the Dirac point
    dirac_point = processed_data['dirac_point']
    ax.plot(dirac_point[1], dirac_point[0], 'w*', markersize=20, 
            markeredgecolor='gold', markeredgewidth=2, label='Dirac Point')
    
    # Mark replica bands
    for replica in processed_data['replica_bands']:
        ax.plot(replica['kx'], replica['energy'], 's', 
                color='cyan' if replica['order'] == -1 else 'yellow',
                markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                label=f"Floquet Replica n={replica['order']}")
    
    ax.set_xlabel('Momentum k_x (Å⁻¹)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Band Dispersion: Dirac Cone and Floquet-Bloch Replica Bands')
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, label='Fermi Level')
    ax.legend(loc='lower right')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Spectral Weight (a.u.)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_replica_band_analysis(processed_data, raw_data):
    """Compute quantitative analysis of replica bands."""
    replica_bands = processed_data['replica_bands']
    dirac_point = processed_data['dirac_point']
    
    # Calculate energy spacing from Dirac point
    results = {
        'dirac_point': {
            'energy_eV': dirac_point[0],
            'kx_Angstrom_inv': dirac_point[1]
        },
        'replica_bands': [],
        'photon_energy_estimate_eV': None,
        'analysis_notes': []
    }
    
    energy_spacings = []
    for replica in replica_bands:
        delta_energy = replica['energy'] - dirac_point[0]
        delta_kx = replica['kx'] - dirac_point[1]
        energy_spacings.append(abs(delta_energy))
        
        results['replica_bands'].append({
            'order': replica['order'],
            'energy_eV': replica['energy'],
            'kx_Angstrom_inv': replica['kx'],
            'intensity': replica['intensity'],
            'delta_energy_from_dirac_eV': delta_energy,
            'delta_kx_from_dirac_Angstrom_inv': delta_kx
        })
    
    # Estimate photon energy from replica spacing
    # For Floquet states, replica spacing should equal photon energy ħω
    if len(energy_spacings) >= 2:
        # Group by order
        n_minus_1 = [r for r in results['replica_bands'] if r['order'] == -1]
        n_plus_1 = [r for r in results['replica_bands'] if r['order'] == 1]
        
        if n_minus_1 and n_plus_1:
            avg_spacing_minus = np.mean([abs(r['delta_energy_from_dirac_eV']) for r in n_minus_1])
            avg_spacing_plus = np.mean([abs(r['delta_energy_from_dirac_eV']) for r in n_plus_1])
            photon_energy_estimate = (avg_spacing_minus + avg_spacing_plus) / 2
            results['photon_energy_estimate_eV'] = photon_energy_estimate
            results['analysis_notes'].append(
                f"Estimated photon energy from Floquet replica spacing: {photon_energy_estimate:.3f} eV"
            )
            
            # Compare with 5 μm pump wavelength
            # E = hc/λ = 1240 eV·nm / 5000 nm = 0.248 eV
            expected_photon_energy = 1.2398 / 5.0  # eV for 5 μm
            results['expected_photon_energy_eV'] = expected_photon_energy
            results['photon_energy_deviation_percent'] = abs(photon_energy_estimate - expected_photon_energy) / expected_photon_energy * 100
            results['analysis_notes'].append(
                f"Expected photon energy for 5 μm pump: {expected_photon_energy:.3f} eV"
            )
    
    return results


def save_data_overview(raw_data, processed_data, polarization_data, output_path):
    """Save overview of input data characteristics."""
    overview = {
        'raw_trarpes_data': {
            'energy_axis_range_eV': [float(raw_data['energy_axis'].min()), 
                                     float(raw_data['energy_axis'].max())],
            'energy_axis_points': int(len(raw_data['energy_axis'])),
            'kx_axis_range_Angstrom_inv': [float(raw_data['kx_axis'].min()), 
                                           float(raw_data['kx_axis'].max())],
            'kx_axis_points': int(len(raw_data['kx_axis'])),
            'polarization_angles_degrees': raw_data['polarization_angles'].tolist(),
            'time_delays_fs': raw_data['time_delays'].tolist(),
            'spectrum_shape': list(raw_data['pump_off_spectrum'].shape)
        },
        'processed_band_data': {
            'dirac_point': processed_data['dirac_point'],
            'n_replica_bands': len(processed_data['replica_bands']),
            'n_band_dispersion_points': len(processed_data['band_dispersion'])
        },
        'polarization_data': {
            'n_angles': len(polarization_data),
            'angle_range_degrees': [int(polarization_data['angle_degrees'].min()), 
                                    int(polarization_data['angle_degrees'].max())],
            'target_energy_eV': float(polarization_data['target_energy'].iloc[0]),
            'target_kx_Angstrom_inv': float(polarization_data['target_kx'].iloc[0])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(overview, f, indent=2)
    print(f"Saved: {output_path}")
    
    return overview


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Floquet-Bloch States Analysis in Epitaxial Graphene")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading data files...")
    raw_data = load_raw_trarpes_data(DATA_DIR / 'raw_trARPES_data.h5')
    processed_data = load_processed_band_data(DATA_DIR / 'processed_band_data.json')
    polarization_data = load_polarization_data(DATA_DIR / 'polarization_dependence_data.csv')
    print(f"  - Raw tr-ARPES data: {len(raw_data)} arrays loaded")
    print(f"  - Processed band data: {len(processed_data['replica_bands'])} replica bands")
    print(f"  - Polarization data: {len(polarization_data)} angles")
    
    # Save data overview
    print("\n[2/6] Saving data overview...")
    overview = save_data_overview(raw_data, processed_data, polarization_data, 
                                  OUTPUTS_DIR / 'data_overview.json')
    
    # Compute replica band analysis
    print("\n[3/6] Computing replica band analysis...")
    replica_analysis = compute_replica_band_analysis(processed_data, raw_data)
    with open(OUTPUTS_DIR / 'replica_band_analysis.json', 'w') as f:
        json.dump(replica_analysis, f, indent=2)
    print(f"  - Estimated photon energy: {replica_analysis.get('photon_energy_estimate_eV', 'N/A')} eV")
    if 'photon_energy_deviation_percent' in replica_analysis:
        print(f"  - Deviation from expected (5 μm): {replica_analysis['photon_energy_deviation_percent']:.1f}%")
    
    # Generate figures
    print("\n[4/6] Generating Figure 1: Data Overview...")
    plot_data_overview(raw_data, REPORT_IMAGES_DIR / 'fig1_data_overview.png')
    
    print("\n[5/6] Generating Figure 2: Replica Bands...")
    plot_replica_bands(processed_data, raw_data, REPORT_IMAGES_DIR / 'fig2_replica_bands.png')
    
    print("\n[6/6] Generating Figure 3: Polarization Dependence...")
    plot_polarization_dependence(polarization_data, REPORT_IMAGES_DIR / 'fig3_polarization_dependence.png')
    
    print("\n[7/6] Generating Figure 4: Band Dispersion...")
    plot_band_dispersion(processed_data, raw_data, REPORT_IMAGES_DIR / 'fig4_band_dispersion.png')
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Outputs saved to: {OUTPUTS_DIR.absolute()}")
    print(f"Figures saved to: {REPORT_IMAGES_DIR.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
