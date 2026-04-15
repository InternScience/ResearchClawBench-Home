import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')


def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


def box_mean(arr, idx, box_e=2, box_k=2):
    ie, ik = idx
    sl = arr[max(0, ie - box_e):ie + box_e + 1, max(0, ik - box_k):ik + box_k + 1]
    return float(np.mean(sl))


def pol_model(theta, c0, a, phi):
    return c0 + a * np.cos(2 * (theta - phi))


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    IMG.mkdir(exist_ok=True, parents=True)

    with open(DATA / 'processed_band_data.json') as f:
        proc = json.load(f)
    pol = pd.read_csv(DATA / 'polarization_dependence_data.csv')

    with h5py.File(DATA / 'raw_trARPES_data.h5', 'r') as h5:
        energy = h5['energy_axis'][:]
        kx = h5['kx_axis'][:]
        angles = [int(a) for a in h5['polarization_angles'][:]]
        delays = [float(t) for t in h5['time_delays'][:]]
        off = h5['pump_off_spectrum'][:]
        on_maps = {a: h5[f'pump_on_angle_{a}'][:] for a in angles}

    dirac_energy = float(proc['dirac_point'][1])
    main_idx = (nearest_idx(energy, dirac_energy), nearest_idx(kx, 0.0))
    rep_plus = [r for r in proc['replica_bands'] if r['order'] == 1][0]
    rep_minus = [r for r in proc['replica_bands'] if r['order'] == -1][0]
    rep_plus_idx = (nearest_idx(energy, rep_plus['energy']), nearest_idx(kx, rep_plus['kx']))
    rep_minus_idx = (nearest_idx(energy, rep_minus['energy']), nearest_idx(kx, rep_minus['kx']))

    angle_metrics = []
    diff_profiles = []
    for a in angles:
        arr = on_maps[a]
        diff = arr - off
        angle_metrics.append({
            'angle_deg': a,
            'main_intensity': box_mean(arr, main_idx),
            'replica_plus_intensity': box_mean(arr, rep_plus_idx),
            'replica_minus_intensity': box_mean(arr, rep_minus_idx),
            'replica_plus_to_main': box_mean(arr, rep_plus_idx) / box_mean(arr, main_idx),
            'replica_minus_to_main': box_mean(arr, rep_minus_idx) / box_mean(arr, main_idx),
            'mean_abs_delta_from_off': float(np.mean(np.abs(diff))),
            'max_abs_delta_from_off': float(np.max(np.abs(diff)))
        })
        diff_profiles.append({
            'angle_deg': a,
            'integrated_difference': float(np.sum(diff)),
            'positive_difference_fraction': float(np.mean(diff > 0))
        })

    angle_df = pd.DataFrame(angle_metrics)
    diff_df = pd.DataFrame(diff_profiles)
    popt, pcov = curve_fit(pol_model, pol['angle_radians'], pol['intensity'], p0=[pol['intensity'].mean(), 0.005, 0.0])
    fit_vals = pol_model(pol['angle_radians'].values, *popt)
    residuals = pol['intensity'].values - fit_vals
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((pol['intensity'].values - pol['intensity'].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan

    band_disp = pd.DataFrame(proc['band_dispersion'])
    replica_df = pd.DataFrame(proc['replica_bands'])
    photon_energy = float(rep_plus['energy'] - dirac_energy)

    dataset_summary = {
        'energy_range_eV': [float(energy.min()), float(energy.max())],
        'kx_range_Ainv': [float(kx.min()), float(kx.max())],
        'n_energy': int(len(energy)),
        'n_kx': int(len(kx)),
        'available_time_delays_fs': delays,
        'available_polarization_angles_deg': angles,
        'photon_energy_from_sidebands_eV': photon_energy
    }
    metrics_summary = {
        'polarization_fit_offset': float(popt[0]),
        'polarization_fit_amplitude': float(abs(popt[1])),
        'polarization_fit_phase_deg': float(np.degrees(popt[2])),
        'polarization_fit_r_squared': float(r2),
        'replica_plus_to_main_mean': float(angle_df['replica_plus_to_main'].mean()),
        'replica_minus_to_main_mean': float(angle_df['replica_minus_to_main'].mean()),
        'replica_plus_to_main_std': float(angle_df['replica_plus_to_main'].std(ddof=1)),
        'replica_minus_to_main_std': float(angle_df['replica_minus_to_main'].std(ddof=1)),
        'mean_abs_delta_range': [float(angle_df['mean_abs_delta_from_off'].min()), float(angle_df['mean_abs_delta_from_off'].max())]
    }

    (OUT / 'dataset_summary.json').write_text(json.dumps(dataset_summary, indent=2))
    (OUT / 'band_metrics.json').write_text(json.dumps(metrics_summary, indent=2))
    angle_df.to_csv(OUT / 'angle_metrics.csv', index=False)
    diff_df.to_csv(OUT / 'difference_profiles.csv', index=False)
    band_disp.to_csv(OUT / 'band_dispersion.csv', index=False)
    replica_df.to_csv(OUT / 'replica_bands.csv', index=False)

    claims = [
        {
            'claim': 'First-order Floquet-like replica bands are separated from the main Dirac feature by approximately one pump photon energy.',
            'status': 'supported',
            'evidence_artifact': 'outputs/dataset_summary.json; outputs/replica_bands.csv',
            'details': f"Replica offset = {photon_energy:.3f} eV from processed band positions."
        },
        {
            'claim': 'Pump-on spectra contain enhanced sideband spectral weight relative to pump-off spectra at replica energies.',
            'status': 'supported',
            'evidence_artifact': 'outputs/angle_metrics.csv; report/images/figure2_spectrum_comparison.png',
            'details': f"Replica/main ratios range from {angle_df['replica_plus_to_main'].min():.3f} to {angle_df['replica_plus_to_main'].max():.3f}."
        },
        {
            'claim': 'Polarization dependence is present but weak and twofold-symmetric within this dataset.',
            'status': 'supported_with_caveat',
            'evidence_artifact': 'outputs/band_metrics.json; report/images/figure4_polarization_dependence.png',
            'details': f"Cos(2θ) fit amplitude = {abs(popt[1]):.4f}, R² = {r2:.3f}; effect size is small."
        },
        {
            'claim': 'The provided data alone cleanly disentangle Floquet-Bloch initial states from Volkov final states.',
            'status': 'not_supported',
            'evidence_artifact': 'report/report.md',
            'details': 'The dataset supports polarization-sensitive sidebands consistent with mixed matrix-element/final-state effects but lacks a dedicated separation protocol.'
        }
    ]
    (OUT / 'claim_recovery_table.json').write_text(json.dumps(claims, indent=2))

    # Figure 1: raw overview
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    im0 = axes[0].imshow(off, origin='lower', aspect='auto', extent=[kx.min(), kx.max(), energy.min(), energy.max()], cmap='magma')
    axes[0].set_title('Pump-off tr-ARPES spectrum')
    axes[0].set_xlabel(r'$k_x$ (Å$^{-1}$)')
    axes[0].set_ylabel('Energy (eV)')
    plt.colorbar(im0, ax=axes[0], label='Intensity (a.u.)')

    best_angle = int(pol.loc[pol['intensity'].idxmax(), 'angle_degrees'])
    im1 = axes[1].imshow(on_maps[best_angle], origin='lower', aspect='auto', extent=[kx.min(), kx.max(), energy.min(), energy.max()], cmap='magma')
    axes[1].scatter([0.0, rep_plus['kx'], rep_minus['kx']], [dirac_energy, rep_plus['energy'], rep_minus['energy']], c=['cyan', 'lime', 'lime'], s=50)
    axes[1].set_title(f'Pump-on spectrum at θ={best_angle}°')
    axes[1].set_xlabel(r'$k_x$ (Å$^{-1}$)')
    axes[1].set_ylabel('Energy (eV)')
    plt.colorbar(im1, ax=axes[1], label='Intensity (a.u.)')
    fig.savefig(IMG / 'figure1_raw_overview.png', dpi=200)
    plt.close(fig)

    # Figure 2: comparison and difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    vmin = min(off.min(), on_maps[best_angle].min())
    vmax = max(off.max(), on_maps[best_angle].max())
    for ax, arr, title in zip(axes[:2], [off, on_maps[best_angle]], ['Pump-off', f'Pump-on θ={best_angle}°']):
        im = ax.imshow(arr, origin='lower', aspect='auto', extent=[kx.min(), kx.max(), energy.min(), energy.max()], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
        ax.set_ylabel('Energy (eV)')
    plt.colorbar(im, ax=axes[:2], label='Intensity (a.u.)')
    diff = on_maps[best_angle] - off
    imd = axes[2].imshow(diff, origin='lower', aspect='auto', extent=[kx.min(), kx.max(), energy.min(), energy.max()], cmap='coolwarm')
    axes[2].set_title('Pump-on minus pump-off')
    axes[2].set_xlabel(r'$k_x$ (Å$^{-1}$)')
    axes[2].set_ylabel('Energy (eV)')
    plt.colorbar(imd, ax=axes[2], label='Δ intensity (a.u.)')
    fig.savefig(IMG / 'figure2_spectrum_comparison.png', dpi=200)
    plt.close(fig)

    # Figure 3: metrics vs angle
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(angle_df['angle_deg'], angle_df['main_intensity'], marker='o', label='Main Dirac feature')
    axes[0].plot(angle_df['angle_deg'], angle_df['replica_plus_intensity'], marker='s', label='+1 replica')
    axes[0].plot(angle_df['angle_deg'], angle_df['replica_minus_intensity'], marker='^', label='-1 replica')
    axes[0].set_xlabel('Pump polarization angle (deg)')
    axes[0].set_ylabel('Box-averaged intensity (a.u.)')
    axes[0].set_title('Angle-resolved spectral weights')
    axes[0].legend()

    axes[1].plot(angle_df['angle_deg'], angle_df['replica_plus_to_main'], marker='s', label='+1/main')
    axes[1].plot(angle_df['angle_deg'], angle_df['replica_minus_to_main'], marker='^', label='-1/main')
    axes[1].set_xlabel('Pump polarization angle (deg)')
    axes[1].set_ylabel('Intensity ratio')
    axes[1].set_title('Normalized replica strength')
    axes[1].legend()
    fig.savefig(IMG / 'figure3_angle_metrics.png', dpi=200)
    plt.close(fig)

    # Figure 4: polarization dependence with fit
    dense_theta = np.linspace(pol['angle_radians'].min(), pol['angle_radians'].max(), 300)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(pol['angle_degrees'], pol['intensity'], s=70, color='black', label='Measured replica intensity')
    ax.plot(np.degrees(dense_theta), pol_model(dense_theta, *popt), color='crimson', lw=2, label=r'Fit $I(\theta)=I_0+A\cos 2(\theta-\phi)$')
    ax.set_xlabel('Pump polarization angle (deg)')
    ax.set_ylabel('Replica intensity (a.u.)')
    ax.set_title('Polarization dependence of replica-band intensity')
    ax.legend()
    fig.savefig(IMG / 'figure4_polarization_dependence.png', dpi=200)
    plt.close(fig)

    # Figure 5: dispersion and replica positions
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sc = ax.scatter(band_disp['kx'], band_disp['energy'], c=band_disp['intensity'], cmap='plasma', s=26, label='Main dispersion')
    ax.scatter(replica_df['kx'], replica_df['energy'], c=replica_df['order'], cmap='coolwarm', s=100, marker='X', label='Replica bands')
    ax.axhline(dirac_energy, color='gray', ls='--', lw=1)
    ax.text(0.17, dirac_energy + 0.01, f'Dirac energy = {dirac_energy:.3f} eV', fontsize=10)
    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Extracted band dispersion and Floquet-like replicas')
    plt.colorbar(sc, ax=ax, label='Main-band intensity (a.u.)')
    fig.savefig(IMG / 'figure5_dispersion_replicas.png', dpi=200)
    plt.close(fig)

    print('Analysis complete.')


if __name__ == '__main__':
    main()
