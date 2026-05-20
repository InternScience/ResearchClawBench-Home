"""
Analysis script for Floquet-Bloch states in monolayer epitaxial graphene
tr-ARPES data.
"""
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Paths
RAW_H5 = 'data/raw_trARPES_data.h5'
PROC_JSON = 'data/processed_band_data.json'
POL_CSV = 'data/polarization_dependence_data.csv'
OUT_DIR = 'outputs'
IMG_DIR = 'report/images'

# Load raw data
with h5py.File(RAW_H5, 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    angles = f['polarization_angles'][:]
    pump_on = {a: f[f'pump_on_angle_{a}'][:] for a in angles}
    time_delays = f['time_delays'][:]

# Average difference over all polarization angles
diffs = np.stack([pump_on[a] - pump_off for a in angles], axis=0)
avg_diff = diffs.mean(axis=0)

# ------------------------------------------------------------------
# 1. Figure 1: Overview spectra
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

im0 = axes[0].imshow(pump_off, aspect='auto', origin='lower',
                     extent=[kx.min(), kx.max(), energy.min(), energy.max()],
                     cmap='hot')
axes[0].set_title('(a) Pump-off spectrum')
axes[0].set_xlabel(r'$k_x$ (arb. units)')
axes[0].set_ylabel('Energy (eV)')
fig.colorbar(im0, ax=axes[0])

# Representative pump-on (0°)
im1 = axes[1].imshow(pump_on[0], aspect='auto', origin='lower',
                     extent=[kx.min(), kx.max(), energy.min(), energy.max()],
                     cmap='hot')
axes[1].set_title('(b) Pump-on (0°)')
axes[1].set_xlabel(r'$k_x$ (arb. units)')
fig.colorbar(im1, ax=axes[1])

# Average difference
vmax = np.max(np.abs(avg_diff))
im2 = axes[2].imshow(avg_diff, aspect='auto', origin='lower',
                     extent=[kx.min(), kx.max(), energy.min(), energy.max()],
                     cmap='bwr', vmin=-vmax, vmax=vmax)
axes[2].set_title('(c) Average difference (pump-on – pump-off)')
axes[2].set_xlabel(r'$k_x$ (arb. units)')
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig1_overview.png', dpi=300)
plt.close()

# ------------------------------------------------------------------
# 2. Extract peaks from average difference map
# ------------------------------------------------------------------
peaks_kx = []
peaks_E = []
peaks_order = []

for j, k in enumerate(kx):
    col = avg_diff[:, j]
    col_smooth = gaussian_filter1d(col, sigma=1)
    peaks, props = find_peaks(col_smooth, prominence=0.3, distance=8)
    for p in peaks:
        e = energy[p]
        order = int(round(e / 0.248))
        peaks_kx.append(k)
        peaks_E.append(e)
        peaks_order.append(order)

peaks_kx = np.array(peaks_kx)
peaks_E = np.array(peaks_E)
peaks_order = np.array(peaks_order)

# Save extracted peaks
peak_df = pd.DataFrame({
    'kx': peaks_kx,
    'energy': peaks_E,
    'order': peaks_order
})
peak_df.to_csv(f'{OUT_DIR}/extracted_peaks.csv', index=False)

# ------------------------------------------------------------------
# 3. Figure 2: Difference map with extracted peaks + EDCs
# ------------------------------------------------------------------
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

ax0 = fig.add_subplot(gs[0, 0])
vmax = np.max(np.abs(avg_diff))
im = ax0.imshow(avg_diff, aspect='auto', origin='lower',
                extent=[kx.min(), kx.max(), energy.min(), energy.max()],
                cmap='bwr', vmin=-vmax, vmax=vmax)
# Overlay peaks
for o in sorted(set(peaks_order)):
    mask = peaks_order == o
    ax0.scatter(peaks_kx[mask], peaks_E[mask], s=8, label=f'$n={o}$')
ax0.set_xlabel(r'$k_x$ (arb. units)')
ax0.set_ylabel('Energy (eV)')
ax0.set_title('(a) Average difference with extracted peaks')
ax0.legend(loc='upper left', fontsize=8)
fig.colorbar(im, ax=ax0)

ax1 = fig.add_subplot(gs[0, 1])
# EDCs at selected kx values
selected_kx = [0.0, 0.05, 0.10, 0.15]
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_kx)))
for target, c in zip(selected_kx, colors):
    j = np.argmin(np.abs(kx - target))
    col = avg_diff[:, j]
    ax1.plot(energy, col, color=c, label=f'$k_x={kx[j]:.3f}$')
ax1.axvline(0.248, color='r', ls='--', lw=1, label=r'$\pm\hbar\omega$')
ax1.axvline(-0.248, color='r', ls='--', lw=1)
ax1.set_xlabel('Energy (eV)')
ax1.set_ylabel('Difference intensity')
ax1.set_title('(b) EDCs at representative $k_x$')
ax1.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig2_replica_peaks.png', dpi=300)
plt.close()

# ------------------------------------------------------------------
# 4. Compute mean energy per order and sideband spacing
# ------------------------------------------------------------------
order_stats = {}
for o in sorted(set(peaks_order)):
    vals = peaks_E[peaks_order == o]
    order_stats[o] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'count': int(len(vals))}

# Linear fit of mean energy vs order
orders_arr = np.array([o for o in sorted(order_stats.keys())], dtype=float)
means_arr = np.array([order_stats[o]['mean'] for o in sorted(order_stats.keys())])
coeffs = np.polyfit(orders_arr, means_arr, 1)
spacing = coeffs[0]
offset = coeffs[1]
# Fit using only first-order sidebands (most reliable)
orders_arr_1 = np.array([-1, 0, 1], dtype=float)
means_arr_1 = np.array([order_stats[o]['mean'] for o in [-1, 0, 1]])
coeffs_1 = np.polyfit(orders_arr_1, means_arr_1, 1)
spacing_1 = coeffs_1[0]
offset_1 = coeffs_1[1]

with open(f'{OUT_DIR}/sideband_spacing.txt', 'w') as f:
    f.write(f'Linear fit E_n = {spacing:.4f} * n + {offset:.4f} (eV)\n')
    f.write(f'Extracted photon energy (spacing): {spacing:.4f} eV\n')
    f.write(f'Nominal pump photon energy: 0.248 eV\n')
    for o in sorted(order_stats.keys()):
        f.write(f'Order {o}: mean E = {order_stats[o]["mean"]:.4f} ± {order_stats[o]["std"]:.4f} eV (N={order_stats[o]["count"]})\n')
    f.write('\nFit using orders -1, 0, +1:\n')
    f.write(f'E_n = {spacing_1:.4f} * n + {offset_1:.4f} (eV)\n')
    f.write(f'Extracted photon energy (spacing_1): {spacing_1:.4f} eV\n')

# ------------------------------------------------------------------
# 5. Figure 3: Polarization dependence
# ------------------------------------------------------------------
pol_df = pd.read_csv(POL_CSV)
theta_deg = pol_df['angle_degrees'].values
theta = np.deg2rad(theta_deg)
I = pol_df['intensity'].values

def cos2_model(theta, A, B, theta0):
    return A + B * np.cos(theta - theta0)**2

def const_model(theta, C):
    return np.full_like(theta, C)

popt_cos2, _ = curve_fit(cos2_model, theta, I, p0=[0.5, 0.01, 0])
popt_const, _ = curve_fit(const_model, theta, I, p0=[0.5])

I_fit_cos2 = cos2_model(theta, *popt_cos2)
I_fit_const = const_model(theta, *popt_const)

chi2_cos2 = float(np.sum((I - I_fit_cos2)**2))
chi2_const = float(np.sum((I - I_fit_const)**2))

fit_result = {
    'cos2': {'A': float(popt_cos2[0]), 'B': float(popt_cos2[1]), 'theta0': float(popt_cos2[2]), 'chi2': chi2_cos2},
    'constant': {'C': float(popt_const[0]), 'chi2': chi2_const}
}
with open(f'{OUT_DIR}/polarization_fit.json', 'w') as f:
    json.dump(fit_result, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(theta_deg, I, 'o', color='C0', label='Data')
theta_dense = np.linspace(0, 180, 200)
ax.plot(theta_dense, cos2_model(np.deg2rad(theta_dense), *popt_cos2), '-', color='C1',
        label=f'Cos² fit: $A$={popt_cos2[0]:.4f}, $B$={popt_cos2[1]:.4f}')
ax.axhline(popt_const[0], color='C2', ls='--', label=f'Constant fit: $C$={popt_const[0]:.4f}')
ax.set_xlabel(r'Pump polarization angle $\theta_p$ (°)')
ax.set_ylabel('Replica intensity (arb. units)')
ax.set_title('(a) Polarization dependence of $n=+1$ replica')
ax.legend()

ax = axes[1]
res_cos2 = I - I_fit_cos2
res_const = I - I_fit_const
ax.plot(theta_deg, res_cos2, 'o-', color='C1', label=f'Cos² residual ($\\chi^2$={chi2_cos2:.2e})')
ax.plot(theta_deg, res_const, 's-', color='C2', label=f'Constant residual ($\\chi^2$={chi2_const:.2e})')
ax.axhline(0, color='k', ls='--', lw=0.5)
ax.set_xlabel(r'Pump polarization angle $\theta_p$ (°)')
ax.set_ylabel('Residual')
ax.set_title('(b) Fit residuals')
ax.legend()

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig3_polarization.png', dpi=300)
plt.close()

# ------------------------------------------------------------------
# 6. Figure 4: Sideband energy spacing and schematic dispersion
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) Mean energy vs order
ax = axes[0]
ax.errorbar(orders_arr, means_arr,
            yerr=[order_stats[o]['std'] for o in sorted(order_stats.keys())],
            fmt='o', capsize=4, color='C0')
ax.plot(orders_arr, np.polyval(coeffs_1, orders_arr), 'k--', lw=1.5,
        label=f'First-order fit: $E_n={spacing_1:.3f}n+{offset_1:.3f}$')
ax.plot(orders_arr, np.polyval(coeffs, orders_arr), 'k:', lw=1,
        label=f'All-order fit: $E_n={spacing:.3f}n+{offset:.3f}$')
ax.set_xlabel('Floquet order $n$')
ax.set_ylabel('Mean extracted energy (eV)')
ax.set_title(f'(a) Linear sideband spacing ($\\hbar\\omega={spacing_1:.3f}$ eV)')
ax.legend()

# (b) Replica dispersion E vs |kx| with theoretical lines
ax = axes[1]
# Visual estimate of Dirac cone slope from raw image: E ≈ 0.4 at |kx| ≈ 0.15
v_est = 0.4 / 0.15
for o in sorted(set(peaks_order)):
    mask = peaks_order == o
    ax.scatter(np.abs(peaks_kx[mask]), peaks_E[mask], s=10, label=f'$n={o}$')

kx_dense = np.linspace(0, 0.3, 100)
for o in [-2, -1, 0, 1, 2]:
    ax.plot(kx_dense,  v_est * kx_dense + o * 0.248, 'k--', lw=0.5, alpha=0.4)
    ax.plot(kx_dense, -v_est * kx_dense + o * 0.248, 'k--', lw=0.5, alpha=0.4)

ax.set_xlabel(r'$|k_x|$ (arb. units)')
ax.set_ylabel('Energy (eV)')
ax.set_title('(b) Replica band dispersion')
ax.legend(loc='upper left', fontsize=7)

plt.tight_layout()
plt.savefig(f'{IMG_DIR}/fig4_spacing_and_dispersion.png', dpi=300)
plt.close()

# ------------------------------------------------------------------
# 7. Summary printout
# ------------------------------------------------------------------
print('Analysis complete.')
print(f'Extracted sideband spacing: {spacing:.4f} eV (nominal 0.248 eV).')
print(f'First-order sideband spacing: {spacing_1:.4f} eV.')
print(f'Polarization cos2 amplitude B = {popt_cos2[1]:.4f} (<< constant term).')
print('Figures saved to report/images/')
print('Intermediate results saved to outputs/')
