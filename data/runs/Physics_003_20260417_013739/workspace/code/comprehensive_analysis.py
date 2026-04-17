"""Comprehensive quantitative analysis of Floquet-Bloch states."""
import numpy as np
import h5py
import json
import pandas as pd

datadir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/data'
resultsdir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_003_20260417_013739/outputs'

# Load all data
with h5py.File(f'{datadir}/raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    pump_energy = f.attrs['pump_energy_eV']
    angles = f['polarization_angles'][:]
    
    pump_on_data = {}
    for angle in angles:
        pump_on_data[angle] = f[f'pump_on_angle_{angle}'][:]

with open(f'{datadir}/processed_band_data.json', 'r') as f:
    band_data = json.load(f)

df_pol = pd.read_csv(f'{datadir}/polarization_dependence_data.csv')

print("=" * 70)
print("COMPREHENSIVE ANALYSIS OF FLOQUET-BLOCH STATES")
print("=" * 70)

# 1. Dirac cone characterization
print("\n1. DIRAC CONE CHARACTERIZATION")
print("-" * 40)
dirac_E = -0.0427
print(f"Dirac point energy: {dirac_E:.4f} eV")
print(f"Pump photon energy: {pump_energy:.3f} eV")
print(f"Pump wavelength: 5.0 μm")

# Estimate Fermi velocity from the cone slope
# Find the peak kx position for each energy in pump-off
peak_kx_positive = []
peak_kx_negative = []
for i in range(len(energy)):
    row = pump_off[i, :]
    # Positive kx side
    pos_mask = kx > 0.01
    if np.any(pos_mask):
        pos_row = row[pos_mask]
        pos_kx = kx[pos_mask]
        peak_kx_positive.append((energy[i], pos_kx[np.argmax(pos_row)]))
    # Negative kx side
    neg_mask = kx < -0.01
    if np.any(neg_mask):
        neg_row = row[neg_mask]
        neg_kx = kx[neg_mask]
        peak_kx_negative.append((energy[i], neg_kx[np.argmax(neg_row)]))

# Estimate v_F from the slope of E vs |kx| near the Dirac point
# Use energies between -0.3 and -0.1 (below Dirac point, lower branch)
peak_pos = np.array(peak_kx_positive)
mask = (peak_pos[:, 0] > -0.3) & (peak_pos[:, 0] < -0.1)
if np.sum(mask) > 5:
    from scipy.stats import linregress
    E_sel = peak_pos[mask, 0]
    kx_sel = peak_pos[mask, 1]
    slope, intercept, r_value, p_value, std_err = linregress(kx_sel, E_sel)
    v_F = abs(slope)
    print(f"Estimated Fermi velocity (positive kx): {v_F:.2f} eV·Å")
    print(f"  (linear fit R² = {r_value**2:.4f})")

# 2. Replica band analysis
print("\n2. FLOQUET-BLOCH REPLICA BAND ANALYSIS")
print("-" * 40)
replica_bands = band_data['replica_bands']
print(f"Number of identified replica bands: {len(replica_bands)}")

for rb in replica_bands:
    print(f"  n={rb['order']:+d}: kx={rb['kx']:.4f} Å⁻¹, E={rb['energy']:.4f} eV, I={rb['intensity']:.4f}")

# Energy separation
n_minus_E = np.mean([rb['energy'] for rb in replica_bands if rb['order'] == -1])
n_plus_E = np.mean([rb['energy'] for rb in replica_bands if rb['order'] == +1])
separation = n_plus_E - n_minus_E
expected_separation = 2 * pump_energy

print(f"\nMeasured n=+1 average energy: {n_plus_E:.4f} eV")
print(f"Measured n=-1 average energy: {n_minus_E:.4f} eV")
print(f"Measured separation: {separation:.4f} eV")
print(f"Expected separation (2ℏω): {expected_separation:.4f} eV")
print(f"Deviation: {abs(separation - expected_separation):.6f} eV ({abs(separation - expected_separation)/expected_separation*100:.3f}%)")

# Midpoint (should be Dirac point energy)
midpoint = (n_plus_E + n_minus_E) / 2
print(f"Midpoint energy: {midpoint:.4f} eV (Dirac point: {dirac_E:.4f} eV)")

# 3. Intensity enhancement analysis
print("\n3. INTENSITY ENHANCEMENT ANALYSIS")
print("-" * 40)

for angle in angles:
    diff = pump_on_data[angle] - pump_off
    mean_diff = diff.mean()
    max_diff = diff.max()
    # Intensity at replica positions
    for rb in replica_bands:
        kx_idx = np.argmin(np.abs(kx - rb['kx']))
        E_idx = np.argmin(np.abs(energy - rb['energy']))
        I_off = pump_off[E_idx, kx_idx]
        I_on = pump_on_data[angle][E_idx, kx_idx]
        if angle == 0:
            print(f"  θ={angle}°, n={rb['order']:+d} at kx={rb['kx']:.3f}: "
                  f"I_off={I_off:.3f}, I_on={I_on:.3f}, ΔI={I_on-I_off:.3f} ({(I_on-I_off)/I_off*100:.1f}%)")

# 4. Polarization dependence analysis
print("\n4. POLARIZATION DEPENDENCE ANALYSIS")
print("-" * 40)
from scipy.optimize import curve_fit

def volkov_model(theta, I0, A, phi):
    return I0 + A * np.cos(2 * (theta - phi))**2

def lape_model(theta, I0, A, phi):
    return I0 + A * np.cos(theta - phi)**2

angles_rad = df_pol['angle_radians'].values
intensity = df_pol['intensity'].values

# Fit Volkov model
popt_v, pcov_v = curve_fit(volkov_model, angles_rad, intensity, p0=[0.5, 0.01, 0.0])
perr_v = np.sqrt(np.diag(pcov_v))
y_pred_v = volkov_model(angles_rad, *popt_v)
ss_res_v = np.sum((intensity - y_pred_v)**2)
ss_tot = np.sum((intensity - np.mean(intensity))**2)
r2_v = 1 - ss_res_v / ss_tot

# Fit LAPE model
popt_l, pcov_l = curve_fit(lape_model, angles_rad, intensity, p0=[0.5, 0.01, 0.0])
perr_l = np.sqrt(np.diag(pcov_l))
y_pred_l = lape_model(angles_rad, *popt_l)
ss_res_l = np.sum((intensity - y_pred_l)**2)
r2_l = 1 - ss_res_l / ss_tot

# AIC comparison (simplified)
n = len(intensity)
aic_v = n * np.log(ss_res_v / n) + 2 * 3
aic_l = n * np.log(ss_res_l / n) + 2 * 3

print(f"Volkov model: I₀={popt_v[0]:.6f}, A={popt_v[1]:.6f}, φ={np.degrees(popt_v[2]):.2f}°")
print(f"  R² = {r2_v:.6f}, AIC = {aic_v:.2f}")
print(f"LAPE model:   I₀={popt_l[0]:.6f}, A={popt_l[1]:.6f}, φ={np.degrees(popt_l[2]):.2f}°")
print(f"  R² = {r2_l:.6f}, AIC = {aic_l:.2f}")
print(f"\nVolkov model is {'preferred' if aic_v < aic_l else 'not preferred'} (ΔAIC = {aic_l - aic_v:.2f})")

# 5. Spectral weight transfer
print("\n5. SPECTRAL WEIGHT ANALYSIS")
print("-" * 40)
total_off = pump_off.sum()
for angle in angles:
    total_on = pump_on_data[angle].sum()
    ratio = total_on / total_off
    print(f"  θ={angle:3d}°: Total I_on/I_off = {ratio:.4f}")

# Save comprehensive results
comprehensive = {
    "dirac_cone": {
        "dirac_point_energy_eV": dirac_E,
        "estimated_fermi_velocity_eV_A": float(v_F) if 'v_F' in dir() else None,
    },
    "floquet_bloch_replicas": {
        "n_plus1_energy_eV": float(n_plus_E),
        "n_minus1_energy_eV": float(n_minus_E),
        "measured_separation_eV": float(separation),
        "expected_separation_eV": float(expected_separation),
        "deviation_percent": float(abs(separation - expected_separation)/expected_separation*100),
        "replica_bands": replica_bands,
    },
    "polarization_analysis": {
        "volkov_fit": {
            "I0": float(popt_v[0]),
            "A": float(popt_v[1]),
            "phi_deg": float(np.degrees(popt_v[2])),
            "R_squared": float(r2_v),
            "AIC": float(aic_v),
        },
        "lape_fit": {
            "I0": float(popt_l[0]),
            "A": float(popt_l[1]),
            "phi_deg": float(np.degrees(popt_l[2])),
            "R_squared": float(r2_l),
            "AIC": float(aic_l),
        },
        "preferred_model": "Volkov" if aic_v < aic_l else "LAPE",
        "delta_AIC": float(aic_l - aic_v),
    },
    "pump_parameters": {
        "wavelength_um": 5.0,
        "photon_energy_eV": float(pump_energy),
        "sample": "monolayer_epitaxial_graphene",
    }
}

with open(f'{resultsdir}/comprehensive_results.json', 'w') as f:
    json.dump(comprehensive, f, indent=2)
print("\nSaved comprehensive_results.json")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
