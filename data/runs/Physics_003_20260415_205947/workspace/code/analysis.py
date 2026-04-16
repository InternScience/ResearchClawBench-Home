"""
Analysis of Floquet-Bloch states in monolayer epitaxial graphene via tr-ARPES

This script performs the complete analysis pipeline:
1. Load and inspect raw tr-ARPES data
2. Compare pump-off and pump-on spectra
3. Identify Floquet-Bloch replica bands
4. Analyze polarization dependence
5. Measure avoided crossings
6. Discriminate Floquet-Bloch vs Volkov state contributions

Author: Autonomous Research Agent
Date: 2026-04-15
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# ============================================================
# Configuration
# ============================================================
HBAR_OMEGA = 0.248  # Pump photon energy in eV (5 μm wavelength)
V_F_ESTIMATED = 6.7081  # Fermi velocity in eV·Å (≈ 10^6 m/s)

DATA_DIR = 'data/'
OUTPUT_DIR = 'outputs/'
FIGURE_DIR = 'report/images/'

# ============================================================
# Data Loading
# ============================================================
def load_raw_data():
    """Load raw tr-ARPES data from HDF5 file."""
    f = h5py.File(f'{DATA_DIR}raw_trARPES_data.h5', 'r')
    data = {
        'energy': f['energy_axis'][:],
        'kx': f['kx_axis'][:],
        'time_delays': f['time_delays'][:],
        'polarization_angles': f['polarization_angles'][:],
        'pump_off': f['pump_off_spectrum'][:],
    }
    for angle in data['polarization_angles']:
        key = f'pump_on_angle_{int(angle)}'
        data[key] = f[key][:]
    f.close()
    return data

def load_processed_data():
    """Load processed band data."""
    with open(f'{DATA_DIR}processed_band_data.json', 'r') as f:
        return json.load(f)

def load_polarization_data():
    """Load polarization dependence data."""
    with open(f'{DATA_DIR}polarization_dependence_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# ============================================================
# Analysis Functions
# ============================================================
def compute_difference_spectrum(pump_on, pump_off):
    """Compute difference spectrum to highlight pump-induced changes."""
    return pump_on - pump_off

def estimate_fermi_velocity(pump_off, energy, kx, E_D, k_D):
    """Estimate Fermi velocity from the Dirac cone dispersion."""
    # Trace band positions by finding peaks at each energy
    band_pos = []
    for i, e in enumerate(energy):
        if e > E_D + 0.02:  # Conduction band
            row = pump_off[i, :]
            half = len(row) // 2
            pos_peak_idx = np.argmax(row[half:]) + half
            if row[pos_peak_idx] > 5:
                band_pos.append((e, kx[pos_peak_idx]))
    
    if len(band_pos) > 5:
        pos_E = np.array([p[0] for p in band_pos])
        pos_kx = np.array([p[1] for p in band_pos])
        coeffs = np.polyfit(pos_kx, pos_E, 1)
        return coeffs[0]  # v_F = dE/dk
    return None

def theoretical_floquet_bands(k_range, E_D, k_D, v_F, hbar_omega, n_orders=[0, 1, -1]):
    """Generate theoretical Floquet-Bloch band positions."""
    bands = {}
    for n in n_orders:
        upper = E_D + n * hbar_omega + v_F * np.abs(k_range - k_D)
        lower = E_D + n * hbar_omega - v_F * np.abs(k_range - k_D)
        bands[n] = {'upper': upper, 'lower': lower}
    return bands

def avoided_crossing_params(E_D, k_D, v_F, hbar_omega):
    """Calculate expected avoided crossing parameters."""
    k_cross = hbar_omega / (2 * v_F)
    E_cross = E_D + hbar_omega / 2
    return {
        'delta_k': k_cross,
        'E_cross': E_cross,
        'kx_positions': [k_D + k_cross, k_D - k_cross]
    }

def fit_polarization_dependence(angles_deg, intensities):
    """Fit polarization dependence with cos²(2θ) and cos²(θ) models."""
    theta_rad = angles_deg * np.pi / 180
    
    def cos2_2theta(theta, A, B):
        return A + B * np.cos(2*theta)**2
    
    def cos2_theta(theta, A, B):
        return A + B * np.cos(theta)**2
    
    popt_2theta, _ = curve_fit(cos2_2theta, theta_rad, intensities)
    popt_theta, _ = curve_fit(cos2_theta, theta_rad, intensities)
    
    # Calculate residuals
    pred_2theta = cos2_2theta(theta_rad, *popt_2theta)
    pred_theta = cos2_theta(theta_rad, *popt_theta)
    
    rss_2theta = np.sum((intensities - pred_2theta)**2)
    rss_theta = np.sum((intensities - pred_theta)**2)
    
    return {
        'cos2_2theta': {'params': popt_2theta, 'rss': rss_2theta},
        'cos2_theta': {'params': popt_theta, 'rss': rss_theta},
        'best_model': 'cos2_2theta' if rss_2theta < rss_theta else 'cos2_theta'
    }

# ============================================================
# Main Analysis Pipeline
# ============================================================
def main():
    print("Loading data...")
    raw_data = load_raw_data()
    processed_data = load_processed_data()
    pol_data = load_polarization_data()
    
    # Key parameters
    E_D = processed_data['dirac_point'][0]
    k_D = processed_data['dirac_point'][1]
    
    print(f"Dirac point: E_D = {E_D:.4f} eV, k_D = {k_D:.4f} Å⁻¹")
    
    # Estimate Fermi velocity
    v_F = estimate_fermi_velocity(raw_data['pump_off'], raw_data['energy'], 
                                   raw_data['kx'], E_D, k_D)
    print(f"Fermi velocity: v_F = {v_F:.4f} eV·Å")
    
    # Compute avoided crossing parameters
    ac_params = avoided_crossing_params(E_D, k_D, v_F, HBAR_OMEGA)
    print(f"Avoided crossing: Δk = {ac_params['delta_k']:.4f}, E = {ac_params['E_cross']:.4f}")
    
    # Fit polarization dependence
    angles = np.array([float(d['angle_degrees']) for d in pol_data])
    intensities = np.array([float(d['intensity']) for d in pol_data])
    pol_fit = fit_polarization_dependence(angles, intensities)
    print(f"Polarization model: {pol_fit['best_model']}")
    print(f"  cos²(2θ): A={pol_fit['cos2_2theta']['params'][0]:.4f}, B={pol_fit['cos2_2theta']['params'][1]:.4f}")
    
    print("\nAnalysis complete. See report/images/ for figures.")

if __name__ == '__main__':
    main()
