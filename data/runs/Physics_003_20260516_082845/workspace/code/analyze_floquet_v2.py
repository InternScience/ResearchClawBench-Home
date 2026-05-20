#!/usr/bin/env python3
"""
Corrected analysis of Floquet-Bloch states in monolayer epitaxial graphene.
Key fix: dirac_point = [kx, energy], not [energy, kx].
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py, json, csv, os
from collections import defaultdict

WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_003_20260516_082845'
DATA = os.path.join(WORKSPACE, 'data')
OUT = os.path.join(WORKSPACE, 'outputs')
IMG = os.path.join(WORKSPACE, 'report', 'images')

plt.rcParams.update({'font.size':12, 'axes.titlesize':14, 'axes.labelsize':13,
    'xtick.labelsize':11, 'ytick.labelsize':11, 'legend.fontsize':10,
    'figure.dpi':150, 'savefig.dpi':150, 'savefig.bbox':'tight', 'font.family':'serif'})

# ---- LOAD ----
with h5py.File(os.path.join(DATA, 'raw_trARPES_data.h5'), 'r') as f:
    E = f['energy_axis'][:]
    K = f['kx_axis'][:]
    ang = f['polarization_angles'][:]
    td = f['time_delays'][:]
    off = f['pump_off_spectrum'][:]
    pon = {int(a): f[f'pump_on_angle_{a}'][:] for a in ang}

with open(os.path.join(DATA, 'processed_band_data.json')) as f:
    bd = json.load(f)

# Dirac point: [kx, energy] convention in the JSON
dp_kx, dp_e = bd['dirac_point']  # -0.3, -0.0427
dp_ei = int(bd['dirac_indices'][0])  # 91 -> energy_axis index
dp_ki = int(bd['dirac_indices'][1])  # 0 -> kx_axis index

replicas = bd['replica_bands']
dispersion = bd['band_dispersion']

with open(os.path.join(DATA, 'polarization_dependence_data.csv')) as f:
    pol = list(csv.DictReader(f))

pol_deg = [float(r['angle_degrees']) for r in pol]
pol_int = [float(r['intensity']) for r in pol]

HW = 0.248  # eV, 5μm pump

print(f"Dirac: E={dp_e:.4f} eV, kx={dp_kx:.4f} 1/A, indices=({dp_ei},{dp_ki})")
print(f"Photon energy: {HW} eV")
for r in replicas:
    de = r['energy'] - dp_e
    print(f"  n={r['order']:2d}: E={r['energy']:.4f}, dE={de:+.4f} (expect {r['order']*HW:+.4f})")

# ---- DIFFERENCE SPECTRA ----
diff = {}
for a in ang:
    diff[int(a)] = pon[int(a)] - off

# ---- FIGURE 1: Raw tr-ARPES spectra (pump off + selected pump on) ----
print("Generating Figure 1...")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
im = axes[0,0].pcolormesh(K, E, off, shading='auto', cmap='inferno')
axes[0,0].scatter(dp_kx, dp_e, c='cyan', s=100, marker='*', edgecolors='white', linewidths=1)
axes[0,0].set_title('(a) Pump Off (Equilibrium)')
axes[0,0].set_xlabel('$k_x$ ($\\AA^{-1}$)'); axes[0,0].set_ylabel('$E - E_F$ (eV)')
plt.colorbar(im, ax=axes[0,0], label='Intensity')

for idx, a in enumerate([0,30,60,90,120,150,180]):
    ax = axes[(idx+1)//4, (idx+1)%4]
    im = ax.pcolormesh(K, E, pon[a], shading='auto', cmap='inferno')
    ax.scatter(dp_kx, dp_e, c='cyan', s=60, marker='*', edgecolors='white', linewidths=0.8)
    ax.set_title(f'Pump On, θ={a}°')
    ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('$E - E_F$ (eV)')
    plt.colorbar(im, ax=ax, label='Intensity')
plt.tight_layout(); fig.savefig(os.path.join(IMG, 'figure1_raw_spectra.png'), dpi=150); plt.close()

# ---- FIGURE 2: Difference spectra ----
print("Generating Figure 2...")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for idx, a in enumerate([0,30,60,90,120,150,180]):
    ax = axes[idx//4, idx%4]
    vm = np.max(np.abs(diff[a])) * 0.5
    im = ax.pcolormesh(K, E, diff[a], shading='auto', cmap='RdBu_r', vmin=-vm, vmax=vm)
    ax.scatter(dp_kx, dp_e, c='black', s=60, marker='*')
    for r in replicas:
        ax.scatter(r['kx'], r['energy'], c='lime' if r['order']>0 else 'orange', s=50, marker='D', edgecolors='black', linewidths=0.5)
    ax.set_title(f'Diff: θ={a}°'); ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('$E - E_F$ (eV)')
    plt.colorbar(im, ax=ax, label='ΔI')
plt.tight_layout(); fig.savefig(os.path.join(IMG, 'figure2_difference_spectra.png'), dpi=150); plt.close()

# ---- FIGURE 3: Band dispersion with replicas ----
print("Generating Figure 3...")
fig, ax = plt.subplots(figsize=(10, 8))
disp_e = np.array([b['energy'] for b in dispersion])
disp_k = np.array([b['kx'] for b in dispersion])
disp_i = np.array([b['intensity'] for b in dispersion])
sc = ax.scatter(disp_k, disp_e, c=disp_i, cmap='inferno', s=15, alpha=0.8)
cbar = plt.colorbar(sc, ax=ax, label='Intensity')

for r in replicas:
    c = 'dodgerblue' if r['order']==-1 else 'crimson'
    m = 'v' if r['order']==-1 else '^'
    ax.scatter(r['kx'], r['energy'], c=c, s=120, marker=m, edgecolors='black', linewidths=1.2, zorder=10,
               label=f"n={r['order']} (I={r['intensity']:.3f})")
ax.scatter(dp_kx, dp_e, c='gold', s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=10, label='Dirac Point')
# Expected replica positions
for n in [-1, 1]:
    ax.axhline(y=dp_e + n*HW, color='green', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(0.25, dp_e + n*HW, f'$E_D{n:+d}\\hbar\\omega$', fontsize=9, color='green', va='center')
ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('$E - E_F$ (eV)')
ax.set_title('Dirac Cone and Floquet-Bloch Replica Bands'); ax.legend(loc='upper right', fontsize=9)
ax.axhline(y=0, color='gray', ls='--', alpha=0.3); ax.axvline(x=0, color='gray', ls='--', alpha=0.3)
plt.tight_layout(); fig.savefig(os.path.join(IMG, 'figure3_band_dispersion.png'), dpi=150); plt.close()

# ---- FIGURE 4: Polarization dependence ----
print("Generating Figure 4...")
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='polar')
ar = np.array([float(r['angle_radians']) for r in pol])
ar2 = np.append(ar, ar[0]+2*np.pi)
iv2 = np.append(pol_int, pol_int[0])
ax1.plot(ar2, iv2, 'o-', color='darkblue', lw=2, ms=8)
ax1.fill(ar2, iv2, alpha=0.15, color='blue')
ax1.set_title('(a) Polar: Replica Intensity'); ax1.set_theta_zero_location('E')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(pol_deg, pol_int, 'o-', color='darkred', lw=2, ms=8)
ax2.set_xlabel('Pump Polarization Angle θ (deg)'); ax2.set_ylabel('Replica Intensity (arb. u.)')
ax2.set_title('(b) Intensity vs. Polarization Angle')
ax2.axhline(y=np.mean(pol_int), color='gray', ls='--', alpha=0.5, label=f'Mean={np.mean(pol_int):.4f}')

# Group by parallel/perpendicular
I_par = np.mean([pol_int[0], pol_int[6]])  # 0° and 180°
I_perp = pol_int[3]  # 90°
I_obl = np.mean([pol_int[1], pol_int[5]])  # 30° and 150°
ax2.axhline(y=I_par, color='blue', ls=':', alpha=0.4, label=f'Parallel mean={I_par:.4f}')
ax2.axhline(y=I_perp, color='red', ls=':', alpha=0.4, label=f'Perp. (90°)={I_perp:.4f}')
ax2.legend(fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(IMG, 'figure4_polarization_dependence.png'), dpi=150); plt.close()

# ---- FIGURE 5: Floquet ladder and energy spacing ----
print("Generating Figure 5...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) Floquet ladder with measured replicas
ax = axes[0]
for n in range(-2, 3):
    ax.axhline(y=dp_e + n*HW, color='gray', ls='--', alpha=0.3, lw=1)
    ax.text(0.02, dp_e + n*HW + 0.008, f'n={n}', fontsize=9, color='gray')

# Plot measured replicas
for r in replicas:
    c = 'dodgerblue' if r['order']==-1 else 'crimson'
    m = 'v' if r['order']==-1 else '^'
    ax.scatter(r['kx'], r['energy'], c=c, s=120, marker=m, edgecolors='black', linewidths=1.2, zorder=10)
ax.scatter(dp_kx, dp_e, c='black', s=120, marker='*', zorder=10)
ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('$E - E_F$ (eV)')
ax.set_title('(a) Floquet Ladder with Measured Replicas')

# (b) Energy spacing comparison
ax = axes[1]
orders_abs = []
dE_measured = []
for r in replicas:
    o = abs(r['order'])
    if o not in orders_abs:
        orders_abs.append(o)
        dE_measured.append(abs(r['energy'] - dp_e))

for o, de in zip(orders_abs, dE_measured):
    ax.errorbar(o, de, fmt='o', capsize=8, ms=12, color='darkblue', lw=2, label='Measured' if o==1 else None)
    ax.scatter(o, o*HW, marker='s', s=100, color='darkred', zorder=5, label=f'Theory: n·ħω' if o==1 else None)

n_range = np.array([0.5, 1.5])
ax.plot(n_range, n_range*HW, '--', color='darkred', alpha=0.5, lw=1.5)
ax.set_xlabel('|Floquet Order n|'); ax.set_ylabel('Energy Spacing |ΔE| (eV)')
ax.set_title('(b) Energy Spacing: Measured vs. Theory'); ax.legend()
ax.set_xlim(0.5, 1.5)
plt.tight_layout(); fig.savefig(os.path.join(IMG, 'figure5_floquet_ladder.png'), dpi=150); plt.close()

# ---- FIGURE 6: EDC and MDC analysis ----
print("Generating Figure 6...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# EDC at Dirac kx
ax = axes[0]
ax.plot(E, off[:, dp_ki], 'k-', lw=2, label='Pump Off')
ax.plot(E, pon[0][:, dp_ki], 'b-', lw=1.5, label='Pump On (θ=0°)')
ax.plot(E, pon[90][:, dp_ki], 'r-', lw=1.5, label='Pump On (θ=90°)')
ax.axvline(x=dp_e, color='gray', ls='--', alpha=0.5, label='Dirac Point')
for r in replicas:
    ax.axvline(x=r['energy'], color='green', ls=':', alpha=0.3)
ax.set_xlabel('$E - E_F$ (eV)'); ax.set_ylabel('Intensity (arb. u.)')
ax.set_title('(a) EDC at Dirac $k_x$'); ax.legend(fontsize=9)

# MDC at Dirac energy
ax = axes[1]
ax.plot(K, off[dp_ei, :], 'k-', lw=2, label='Pump Off')
ax.plot(K, pon[0][dp_ei, :], 'b-', lw=1.5, label='Pump On (θ=0°)')
ax.plot(K, pon[90][dp_ei, :], 'r-', lw=1.5, label='Pump On (θ=90°)')
ax.axvline(x=dp_kx, color='gray', ls='--', alpha=0.5, label='Dirac Point')
ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('Intensity (arb. u.)')
ax.set_title('(b) MDC at Dirac Energy'); ax.legend(fontsize=9)
plt.tight_layout(); fig.savefig(os.path.join(IMG, 'figure6_edc_mdc.png'), dpi=150); plt.close()

# ---- FIGURE 7: Comprehensive summary ----
print("Generating Figure 7...")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig)

# (a) Pump off with annotations
ax = fig.add_subplot(gs[0,0])
ax.pcolormesh(K, E, off, shading='auto', cmap='inferno')
ax.scatter(dp_kx, dp_e, c='cyan', s=150, marker='*', edgecolors='white', linewidths=1.5)
ax.annotate('Dirac\nPoint', (dp_kx, dp_e), xytext=(25,15), textcoords='offset points', fontsize=9, color='white',
            arrowprops=dict(arrowstyle='->', color='white'))
ax.set_title('(a) Equilibrium (Pump Off)'); ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('$E - E_F$ (eV)')

# (b) Difference θ=0° 
ax = fig.add_subplot(gs[0,1])
vm = np.max(np.abs(diff[0])) * 0.5
ax.pcolormesh(K, E, diff[0], shading='auto', cmap='RdBu_r', vmin=-vm, vmax=vm)
ax.scatter(dp_kx, dp_e, c='black', s=80, marker='*')
for r in replicas:
    ax.scatter(r['kx'], r['energy'], c='lime', s=60, marker='D', edgecolors='black', linewidths=0.5)
ax.set_title('(b) Difference θ=0°'); ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('$E - E_F$ (eV)')

# (c) Difference θ=90°
ax = fig.add_subplot(gs[0,2])
vm = np.max(np.abs(diff[90])) * 0.5
ax.pcolormesh(K, E, diff[90], shading='auto', cmap='RdBu_r', vmin=-vm, vmax=vm)
ax.scatter(dp_kx, dp_e, c='black', s=80, marker='*')
for r in replicas:
    ax.scatter(r['kx'], r['energy'], c='lime', s=60, marker='D', edgecolors='black', linewidths=0.5)
ax.set_title('(c) Difference θ=90°'); ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('$E - E_F$ (eV)')

# (d) Replica intensity across polarizations
ax = fig.add_subplot(gs[1,0])
ang_s = sorted(diff.keys())
lo_vals, hi_vals = [], []
for a in ang_s:
    d = diff[a]
    lo_mask = (E > -0.33) & (E < -0.25)
    hi_mask = (E > 0.16) & (E < 0.25)
    lo_vals.append(np.max(np.abs(d[lo_mask,:])))
    hi_vals.append(np.max(np.abs(d[hi_mask,:])))
x = np.arange(len(ang_s)); w = 0.35
ax.bar(x-w/2, lo_vals, w, label='n=-1 Replica', color='dodgerblue', alpha=0.7)
ax.bar(x+w/2, hi_vals, w, label='n=+1 Replica', color='crimson', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([f'{a}°' for a in ang_s])
ax.set_xlabel('Polarization Angle'); ax.set_ylabel('Max |ΔI| in Replica Region')
ax.set_title('(d) Replica Intensity vs. Polarization'); ax.legend()

# (e) Momentum cuts near Dirac energy
ax = fig.add_subplot(gs[1,1])
for a, ls in [(0,'b-'), (90,'r-'), (180,'g-')]:
    ax.plot(K, pon[a][dp_ei,:], ls, lw=1.5, label=f'Pump On θ={a}°')
ax.plot(K, off[dp_ei,:], 'k-', lw=2, label='Pump Off')
ax.set_xlabel('$k_x$ ($\\AA^{-1}$)'); ax.set_ylabel('Intensity (arb. u.)')
ax.set_title('(e) MDC at Dirac Energy'); ax.legend(fontsize=9)

# (f) Polarization ratio I(0°)/I(90°) vs energy
ax = fig.add_subplot(gs[1,2])
ratios = []
for ei in range(len(E)):
    ws = 15
    k0 = max(0, dp_ki-ws); k1 = min(len(K), dp_ki+ws)
    i0 = np.max(pon[0][ei, k0:k1]) if np.max(pon[0][ei, k0:k1]) > 1e-3 else 1e-3
    i90 = np.max(pon[90][ei, k0:k1]) if np.max(pon[90][ei, k0:k1]) > 1e-3 else 1e-3
    ratios.append(i0/i90)
ax.plot(E, ratios, 'b-', lw=1.5)
ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
ax.axvline(x=dp_e, color='black', ls='--', alpha=0.5, label='Dirac')
for r in replicas[:2]:
    ax.axvline(x=r['energy'], color='green', ls=':', alpha=0.3)
ax.set_xlabel('$E - E_F$ (eV)'); ax.set_ylabel('I(θ=0°) / I(θ=90°)')
ax.set_title('(f) Polarization Anisotropy Ratio'); ax.legend(fontsize=8)
ax.set_ylim(0.7, 1.5)
plt.tight_layout(); fig.savefig(os.path.join(IMG, 'figure7_comprehensive.png'), dpi=150); plt.close()

# ---- SAVE INTERMEDIATES ----
print("Saving intermediate results...")

with open(os.path.join(OUT, 'data_overview.json'), 'w') as f:
    json.dump({
        'energy_axis': {'range': [float(E[0]), float(E[-1])], 'points': len(E)},
        'kx_axis': {'range': [float(K[0]), float(K[-1])], 'points': len(K)},
        'polarization_angles': [int(a) for a in ang],
        'time_delays': [float(t) for t in td],
        'pump_wavelength_um': 5.0, 'pump_photon_energy_ev': HW,
        'dirac_point_energy': float(dp_e), 'dirac_point_kx': float(dp_kx),
    }, f, indent=2)

with open(os.path.join(OUT, 'replica_band_analysis.json'), 'w') as f:
    json.dump({
        'pump_photon_energy_ev': HW,
        'replicas': [{
            'order': r['order'], 'energy': r['energy'], 'kx': r['kx'],
            'intensity': r['intensity'],
            'dE_from_dirac': r['energy'] - dp_e,
            'expected_dE': r['order'] * HW,
            'deviation': (r['energy'] - dp_e) - r['order'] * HW,
        } for r in replicas]
    }, f, indent=2)

with open(os.path.join(OUT, 'polarization_analysis.json'), 'w') as f:
    json.dump({
        'angles_deg': pol_deg, 'intensities': pol_int,
        'mean': float(np.mean(pol_int)), 'std': float(np.std(pol_int)),
        'I_parallel_mean': float(I_par), 'I_perpendicular': float(I_perp),
        'ratio_par_perp': float(I_par/I_perp),
        'modulation_depth': float((max(pol_int)-min(pol_int))/np.mean(pol_int)),
    }, f, indent=2)

with open(os.path.join(OUT, 'difference_spectra_summary.json'), 'w') as f:
    json.dump({f'angle_{a}': {
        'mean': float(np.mean(diff[a])), 'std': float(np.std(diff[a])),
        'max_abs': float(np.max(np.abs(diff[a]))),
    } for a in ang_s}, f, indent=2)

# Save key quantitative results for the report
results = {
    'dirac_point': {'energy_ev': float(dp_e), 'kx_invA': float(dp_kx)},
    'photon_energy_ev': HW,
    'replica_energy_spacing_ev': [abs(r['energy']-dp_e) for r in replicas],
    'replica_orders': [r['order'] for r in replicas],
    'polarization_ratio_par_perp': float(I_par/I_perp),
    'polarization_modulation': float((max(pol_int)-min(pol_int))/np.mean(pol_int)),
}
with open(os.path.join(OUT, 'key_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("Done! All figures and outputs saved.")
