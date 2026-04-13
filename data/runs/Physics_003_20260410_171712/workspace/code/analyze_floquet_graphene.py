import json, math, os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('.')
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

plt.style.use('seaborn-v0_8-whitegrid')

with h5py.File(DATA / 'raw_trARPES_data.h5', 'r') as f:
    energy = f['energy_axis'][:]
    kx = f['kx_axis'][:]
    pump_off = f['pump_off_spectrum'][:]
    angles = f['polarization_angles'][:]
    pump_on = {int(a): f[f'pump_on_angle_{int(a)}'][:] for a in angles}
    attrs = dict(f.attrs)

with open(DATA / 'processed_band_data.json', 'r') as fh:
    proc = json.load(fh)

pol = pd.read_csv(DATA / 'polarization_dependence_data.csv')

# Basic metrics
energy_step = float(np.mean(np.diff(energy)))
kx_step = float(np.mean(np.diff(kx)))
dirac_energy = float(proc['dirac_point'][1])
dirac_kx = float(proc['band_dispersion'][np.argmin(np.abs(np.array(proc['energy_axis']) - dirac_energy))]['kx'])
pump_energy = float(proc.get('pump_energy', attrs.get('pump_energy_eV', np.nan)))
replicas = pd.DataFrame(proc['replica_bands'])
main_disp = pd.DataFrame(proc['band_dispersion'])

# Linear fit away from Dirac point
fit_mask = (np.abs(main_disp['energy'] - dirac_energy) > 0.05) & (np.abs(main_disp['kx']) < 0.12)
fit_df = main_disp.loc[fit_mask].copy()
coef = np.polyfit(np.abs(fit_df['kx'].values), np.abs(fit_df['energy'].values - dirac_energy), 1)
velocity_eVA = float(coef[0])
expected_crossing_k = pump_energy / (2 * velocity_eVA) if velocity_eVA > 0 else np.nan
observed_replica_k = float(np.mean(np.abs(replicas['kx'])))

# Contrast maps and extracted quantities
summary = {
    'sample': str(attrs.get('sample', 'unknown')),
    'pump_wavelength_um': float(attrs.get('pump_wavelength_um', np.nan)),
    'pump_energy_eV': pump_energy,
    'energy_axis_minmax_eV': [float(energy.min()), float(energy.max())],
    'kx_axis_minmax_Ainv': [float(kx.min()), float(kx.max())],
    'energy_step_eV': energy_step,
    'kx_step_Ainv': kx_step,
    'dirac_energy_eV': dirac_energy,
    'replica_orders': sorted(replicas['order'].unique().tolist()),
    'replica_mean_abs_kx_Ainv': observed_replica_k,
    'estimated_velocity_eVA': velocity_eVA,
    'expected_crossing_k_Ainv': expected_crossing_k,
}

# Polarization fit: I = a + b cos(2 theta)
theta = pol['angle_radians'].values
I = pol['intensity'].values
X = np.column_stack([np.ones_like(theta), np.cos(2*theta), np.sin(2*theta)])
beta, *_ = np.linalg.lstsq(X, I, rcond=None)
I_fit = X @ beta
ss_res = float(np.sum((I - I_fit)**2))
ss_tot = float(np.sum((I - I.mean())**2))
r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
amp = float(np.hypot(beta[1], beta[2]))
phase = float(0.5 * math.atan2(beta[2], beta[1]))
summary['polarization_fit'] = {
    'offset': float(beta[0]),
    'cos2_amp': amp,
    'phase_rad': phase,
    'phase_deg': float(np.degrees(phase)),
    'r2': r2,
    'modulation_depth_percent_of_mean': float(100 * amp / beta[0]) if beta[0] else np.nan,
}

# Intensity metrics vs angle from raw maps near +1 replica energy/kx
rep_plus = replicas[replicas['order'] == 1].iloc[0]
ix = int(rep_plus['kx_idx'])
ie = int(rep_plus['energy_idx'])
window = 2
angle_rows = []
for a in angles:
    arr = pump_on[int(a)]
    local = arr[max(0, ie-window):ie+window+1, max(0, ix-window):ix+window+1]
    bg = pump_off[max(0, ie-window):ie+window+1, max(0, ix-window):ix+window+1]
    angle_rows.append({
        'angle': int(a),
        'local_mean_on': float(local.mean()),
        'local_mean_off': float(bg.mean()),
        'local_diff_mean': float((local-bg).mean()),
        'global_mean_on': float(arr.mean()),
    })
angle_df = pd.DataFrame(angle_rows)
summary['raw_local_replica_contrast'] = angle_df.to_dict(orient='records')

with open(OUT / 'analysis_summary.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
angle_df.to_csv(OUT / 'raw_local_replica_contrast.csv', index=False)
fit_df.to_csv(OUT / 'dirac_dispersion_fit_points.csv', index=False)

# Figure 1: data overview
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, arr, title in [
    (axes[0], pump_off, 'Pump off'),
    (axes[1], pump_on[0], 'Pump on, 0°'),
    (axes[2], pump_on[90], 'Pump on, 90°'),
]:
    im = ax.imshow(arr, origin='lower', aspect='auto',
                   extent=[kx.min(), kx.max(), energy.min(), energy.max()],
                   cmap='magma')
    ax.set_title(title)
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle('tr-ARPES intensity maps')
fig.savefig(IMG / 'data_overview_maps.png', dpi=220)
plt.close(fig)

# Figure 2: differential map with extracted bands
fig, ax = plt.subplots(figsize=(6, 5))
diff = pump_on[0] - pump_off
im = ax.imshow(diff, origin='lower', aspect='auto',
               extent=[kx.min(), kx.max(), energy.min(), energy.max()],
               cmap='coolwarm')
ax.scatter(main_disp['kx'], main_disp['energy'], s=8, c='k', label='Main cone')
for order, g in replicas.groupby('order'):
    ax.scatter(g['kx'], g['energy'], s=45, label=f'Replica n={order}')
ax.axhline(dirac_energy, ls='--', c='k', lw=1)
ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Pump-induced spectral redistribution and extracted bands')
ax.legend(fontsize=8)
fig.colorbar(im, ax=ax, label='Pump on - pump off')
fig.savefig(IMG / 'floquet_replica_map.png', dpi=220)
plt.close(fig)

# Figure 3: Dirac dispersion and Floquet offsets
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(main_disp['kx'], main_disp['energy'], s=10, label='Main dispersion')
ks = np.linspace(0, max(np.abs(main_disp['kx'])), 200)
ax.plot(ks, dirac_energy + velocity_eVA*ks, 'r--', lw=2, label=f'|E-E_D|≈v|k|, v={velocity_eVA:.2f} eVÅ')
ax.plot(-ks, dirac_energy + velocity_eVA*ks, 'r--', lw=2)
for _, row in replicas.iterrows():
    ax.scatter(row['kx'], row['energy'], s=60, marker='x', label=None, c='C2' if row['order']==1 else 'C3')
ax.axhline(dirac_energy + pump_energy, color='C2', ls=':', lw=1.5, label=r'$E_D + \hbar\omega$')
ax.axhline(dirac_energy - pump_energy, color='C3', ls=':', lw=1.5, label=r'$E_D - \hbar\omega$')
ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Replica energies track ± pump-photon shifts')
ax.legend(fontsize=8, loc='upper left')
fig.savefig(IMG / 'dispersion_and_replica_offsets.png', dpi=220)
plt.close(fig)

# Figure 4: polarization dependence
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(pol['angle_degrees'], pol['intensity'], 'o-', label='Processed intensity')
ang_dense = np.linspace(0, np.pi, 361)
fit_dense = beta[0] + beta[1]*np.cos(2*ang_dense) + beta[2]*np.sin(2*ang_dense)
ax.plot(np.degrees(ang_dense), fit_dense, '--', label=f'cos(2θ) fit, R²={r2:.3f}')
ax.plot(angle_df['angle'], angle_df['local_diff_mean']/angle_df['local_diff_mean'].max()*pol['intensity'].max(), 's--', label='Raw local contrast (scaled)')
ax.set_xlabel('Pump polarization angle (deg)')
ax.set_ylabel('Replica intensity / scaled contrast')
ax.set_title('Polarization dependence of replica-band signal')
ax.legend(fontsize=8)
fig.savefig(IMG / 'polarization_dependence.png', dpi=220)
plt.close(fig)

# Figure 5: line cuts through replica momentum
fig, ax = plt.subplots(figsize=(6,4.5))
for a in [0,30,60,90,120,150,180]:
    arr = pump_on[a]
    ax.plot(energy, arr[:, ix], label=f'{a}°', alpha=0.85)
ax.axvline(rep_plus['energy'], color='k', ls='--', lw=1, label='n=+1 energy')
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Intensity at fixed replica momentum')
ax.set_title(f'Energy cuts at kx≈{kx[ix]:.3f} Å$^{{-1}}$')
ax.legend(ncol=2, fontsize=8)
fig.savefig(IMG / 'energy_cuts_replica_momentum.png', dpi=220)
plt.close(fig)

print(json.dumps(summary, indent=2))
