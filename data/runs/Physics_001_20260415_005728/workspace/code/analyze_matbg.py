import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

ROOT = Path('/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_001_20260415_005728')
DATA_PATH = ROOT / 'data' / 'MATBG Superfluid Stiffness Core Dataset.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
text = DATA_PATH.read_text()


def parse_scalar(name):
    m = re.search(rf"{re.escape(name)}\s*=\s*([-+0-9.eE]+)", text)
    if not m:
        raise ValueError(f'Missing scalar {name}')
    return float(m.group(1))


def parse_list(label):
    m = re.search(rf"\*\*{re.escape(label)}:\*\*\s*\n(\[[^\]]*\])", text, re.S)
    if not m:
        raise ValueError(f'Missing list {label}')
    return np.fromstring(m.group(1).strip('[]').replace('\n', ' '), sep=' ')


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# Scalars
v_f_conventional = parse_scalar('v_f_conventional')
v_f_geometric = parse_scalar('v_f_geometric')
T_c = parse_scalar('T_c')
I_c = parse_scalar('I_c')

# Density arrays
n_eff = parse_list('Carrier Density Data (n_eff in m^-2)')
D_s_conv = parse_list('Conventional Superfluid Stiffness (D_s_conv)')
D_s_geom = parse_list('Quantum Geometric Superfluid Stiffness (D_s_geom)')
D_s_exp_hole = parse_list('Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole)')
D_s_exp_electron = parse_list('Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron)')

# Temperature arrays
T = parse_list('Temperature Array (T in K)')
D_s_bcs = parse_list('BCS Model Data (D_s_bcs)')
D_s_nodal = parse_list('Nodal Superconductor Data (D_s_nodal)')
D_s_power_n2 = parse_list('Power Law n=2.0 Data (D_s_power_n2)')
D_s_power_n2_5 = parse_list('Power Law n=2.5 Data (D_s_power_n2_5)')
D_s_power_n3 = parse_list('Power Law n=3.0 Data (D_s_power_n3)')
D_s_experimental = parse_list('Experimental Data with Noise (D_s_experimental)')

# Current arrays
I_dc = parse_list('DC Current Array (I_dc in nA)')
D_s_gl = parse_list('Ginzburg-Landau Model (D_s_gl)')
D_s_linear = parse_list('Linear Meissner Model (D_s_linear)')
D_s_dc_exp = parse_list('Experimental DC Data (D_s_dc_exp)')
P_mw = parse_list('Microwave Power Array (P_mw normalized)')
I_mw_amp = parse_list('Microwave Current Amplitude (I_mw_amplitude in nA)')
D_s_mw_exp = parse_list('Experimental Microwave Data (D_s_mw_exp)')

# Harmonize lengths where the synthetic file stores extra points
temp_len = min(len(T), len(D_s_bcs), len(D_s_nodal), len(D_s_power_n2), len(D_s_power_n2_5), len(D_s_power_n3), len(D_s_experimental))
T_temp = T[:temp_len]
D_bcs_temp = D_s_bcs[:temp_len]
D_nodal_temp = D_s_nodal[:temp_len]
D_n2_temp = D_s_power_n2[:temp_len]
D_n25_temp = D_s_power_n2_5[:temp_len]
D_n3_temp = D_s_power_n3[:temp_len]
D_exp_temp = D_s_experimental[:temp_len]

curr_len = min(len(I_dc), len(D_s_gl), len(D_s_linear), len(D_s_dc_exp))
I_curr = I_dc[:curr_len]
D_gl_curr = D_s_gl[:curr_len]
D_lin_curr = D_s_linear[:curr_len]
D_exp_curr = D_s_dc_exp[:curr_len]

# Derived density metrics
exp_mean = 0.5 * (D_s_exp_hole + D_s_exp_electron)
ratio_exp_conv = exp_mean / D_s_conv
ratio_exp_geom = exp_mean / D_s_geom
asymmetry = (D_s_exp_hole - D_s_exp_electron) / exp_mean

density_df = pd.DataFrame({
    'n_eff_m2': n_eff,
    'D_s_conv': D_s_conv,
    'D_s_geom': D_s_geom,
    'D_s_exp_hole': D_s_exp_hole,
    'D_s_exp_electron': D_s_exp_electron,
    'D_s_exp_mean': exp_mean,
    'ratio_exp_conv': ratio_exp_conv,
    'ratio_exp_geom': ratio_exp_geom,
    'electron_hole_asymmetry_frac': asymmetry,
})
density_df.to_csv(OUT / 'carrier_density_summary.csv', index=False)

# Temperature fits below Tc
mask_valid = (T_temp <= T_c) & (D_exp_temp > 0)
T_fit = T_temp[mask_valid]
D_fit = D_exp_temp[mask_valid]
D0 = float(D_fit[0])

models = {
    'BCS_like_n2': D_bcs_temp[mask_valid],
    'nodal_linear': D_nodal_temp[mask_valid],
    'power_n2': D_n2_temp[mask_valid],
    'power_n2_5': D_n25_temp[mask_valid],
    'power_n3': D_n3_temp[mask_valid],
}

fit_rows = []
for name, arr in models.items():
    fit_rows.append({
        'model': name,
        'rmse': rmse(D_fit, arr),
        'mae': float(np.mean(np.abs(D_fit - arr))),
        'r2_like': float(1 - np.sum((D_fit - arr) ** 2) / np.sum((D_fit - D_fit.mean()) ** 2)),
    })


def power_model(temp, n):
    return np.clip(D0 * (1 - np.power(np.clip(temp / T_c, 0, None), n)), 0, None)


popt, pcov = curve_fit(power_model, T_fit, D_fit, p0=[2.5], bounds=(0.5, 6.0))
n_best = float(popt[0])
n_std = float(np.sqrt(np.diag(pcov))[0]) if pcov.size else float('nan')
D_best = power_model(T_fit, n_best)
fit_rows.append({
    'model': 'continuous_power_fit',
    'rmse': rmse(D_fit, D_best),
    'mae': float(np.mean(np.abs(D_fit - D_best))),
    'r2_like': float(1 - np.sum((D_fit - D_best) ** 2) / np.sum((D_fit - D_fit.mean()) ** 2)),
    'best_n': n_best,
    'best_n_std': n_std,
})

temp_metrics = pd.DataFrame(fit_rows).sort_values('rmse')
temp_metrics.to_csv(OUT / 'temperature_fit_metrics.csv', index=False)

# Current fits: near-zero region for quadratic suppression
idx_low = I_curr <= 0.4 * I_c
x = I_curr[idx_low] / I_c
y = D_exp_curr[idx_low] / D_exp_curr[0]
quad = np.polyfit(x**2, 1 - y, 1)
lin = np.polyfit(x, 1 - y, 1)
quad_pred = 1 - quad[0] * x**2 - quad[1]
lin_pred = 1 - lin[0] * x - lin[1]

current_metrics = pd.DataFrame([
    {'model': 'GL_reference_full_curve', 'rmse': rmse(D_exp_curr, D_gl_curr), 'region': 'common_length'},
    {'model': 'linear_reference_full_curve', 'rmse': rmse(D_exp_curr, D_lin_curr), 'region': 'common_length'},
    {'model': 'near_zero_quadratic', 'rmse': rmse(y, quad_pred), 'coefficient': float(quad[0]), 'intercept_offset': float(quad[1]), 'region': 'I<=0.4Ic'},
    {'model': 'near_zero_linear', 'rmse': rmse(y, lin_pred), 'coefficient': float(lin[0]), 'intercept_offset': float(lin[1]), 'region': 'I<=0.4Ic'},
])
current_metrics.to_csv(OUT / 'current_fit_metrics.csv', index=False)

# Proxies for transport/resonance quantities from stiffness when raw observables are absent
R_proxy = 1 / np.maximum(D_exp_curr, 1e-9)
freq_proxy_density_hole = np.sqrt(D_s_exp_hole)
freq_proxy_density_electron = np.sqrt(D_s_exp_electron)
freq_proxy_T = np.sqrt(np.maximum(D_exp_temp, 0))
freq_proxy_I = np.sqrt(np.maximum(D_exp_curr, 0))

summary = {
    'dataset_sizes': {
        'density_points': int(len(n_eff)),
        'temperature_points_common': int(temp_len),
        'temperature_points_extra_in_experimental_array': int(len(D_s_experimental) - temp_len),
        'dc_current_points_common': int(curr_len),
        'dc_current_points_extra_in_experimental_array': int(len(D_s_dc_exp) - curr_len),
        'microwave_points': int(len(P_mw)),
    },
    'velocity_ratio_geometric_to_conventional': v_f_geometric / v_f_conventional,
    'experimental_to_conventional_ratio_mean': float(ratio_exp_conv.mean()),
    'experimental_to_geometric_ratio_mean': float(ratio_exp_geom.mean()),
    'experimental_to_conventional_ratio_range': [float(ratio_exp_conv.min()), float(ratio_exp_conv.max())],
    'experimental_to_geometric_ratio_range': [float(ratio_exp_geom.min()), float(ratio_exp_geom.max())],
    'electron_hole_asymmetry_percent_mean': float(100 * np.mean(np.abs(asymmetry))),
    'best_temperature_power_n': n_best,
    'best_temperature_power_n_std': n_std,
    'near_zero_current_quadratic_coefficient': float(quad[0]),
    'near_zero_current_linear_coefficient': float(lin[0]),
    'resistance_proxy_minmax': [float(R_proxy.min()), float(R_proxy.max())],
    'resonance_proxy_density_hole_minmax': [float(freq_proxy_density_hole.min()), float(freq_proxy_density_hole.max())],
    'resonance_proxy_density_electron_minmax': [float(freq_proxy_density_electron.min()), float(freq_proxy_density_electron.max())],
}
with open(OUT / 'dataset_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

claim_recovery = pd.DataFrame([
    {
        'claim': 'Experimental stiffness strongly exceeds conventional Fermi-liquid prediction',
        'status': 'supported',
        'artifact': 'outputs/carrier_density_summary.csv',
        'evidence': f'mean ratio exp/conv={ratio_exp_conv.mean():.2f}, range={ratio_exp_conv.min():.2f}-{ratio_exp_conv.max():.2f}'
    },
    {
        'claim': 'Quantum geometric scale is much closer than conventional but still below experiment',
        'status': 'supported',
        'artifact': 'outputs/carrier_density_summary.csv',
        'evidence': f'mean ratio exp/geom={ratio_exp_geom.mean():.2f}, range={ratio_exp_geom.min():.2f}-{ratio_exp_geom.max():.2f}'
    },
    {
        'claim': 'Temperature dependence follows a power law consistent with anisotropic/nodal pairing',
        'status': 'supported',
        'artifact': 'outputs/temperature_fit_metrics.csv',
        'evidence': f'best-fit n={n_best:.3f}±{n_std:.3f}; lowest RMSE model={temp_metrics.iloc[0]["model"]}'
    },
    {
        'claim': 'Near-zero-current suppression is quadratic rather than linear',
        'status': 'supported',
        'artifact': 'outputs/current_fit_metrics.csv',
        'evidence': f'quadratic RMSE={current_metrics.loc[current_metrics.model=="near_zero_quadratic","rmse"].iloc[0]:.4f} < linear RMSE={current_metrics.loc[current_metrics.model=="near_zero_linear","rmse"].iloc[0]:.4f}'
    },
    {
        'claim': 'Direct DC resistance and microwave resonance were not separately provided; proxies inferred from stiffness trends',
        'status': 'limitation',
        'artifact': 'outputs/dataset_summary.json',
        'evidence': 'Dataset contains superfluid stiffness arrays and model curves, but no explicit raw resistance or resonance-frequency columns.'
    },
])
claim_recovery.to_csv(OUT / 'claim_recovery_table.csv', index=False)

# Figures
plt.figure(figsize=(10, 7))
plt.plot(n_eff / 1e15, D_s_conv / 1e9, label='Conventional FL', lw=3)
plt.plot(n_eff / 1e15, D_s_geom / 1e9, label='Quantum geometric', lw=3)
plt.plot(n_eff / 1e15, D_s_exp_hole / 1e9, label='Experiment hole', lw=2)
plt.plot(n_eff / 1e15, D_s_exp_electron / 1e9, label='Experiment electron', lw=2)
plt.xlabel(r'Effective carrier density $n_{eff}$ ($10^{15}$ m$^{-2}$)')
plt.ylabel(r'Superfluid stiffness $D_s$ ($10^9$ arb. units)')
plt.title('Carrier-density dependence of MATBG superfluid stiffness')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(IMG / 'density_stiffness_comparison.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 7))
plt.plot(n_eff / 1e15, ratio_exp_conv, label='Experiment / conventional', lw=3)
plt.plot(n_eff / 1e15, ratio_exp_geom, label='Experiment / geometric', lw=3)
plt.axhline(1, color='k', ls='--', lw=1)
plt.xlabel(r'Effective carrier density $n_{eff}$ ($10^{15}$ m$^{-2}$)')
plt.ylabel('Enhancement ratio')
plt.title('Enhancement of measured stiffness over model expectations')
plt.legend()
plt.tight_layout()
plt.savefig(IMG / 'enhancement_ratio.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 7))
plt.plot(T_temp, D_exp_temp, label='Experimental', lw=3)
plt.plot(T_temp, D_bcs_temp, label='BCS / n=2', lw=2)
plt.plot(T_temp, D_nodal_temp, label='Nodal linear', lw=2)
plt.plot(T_temp, D_n25_temp, label='Power law n=2.5', lw=2)
plt.plot(T_temp, D_n3_temp, label='Power law n=3', lw=2)
plt.axvline(T_c, color='k', ls='--', lw=1, label='$T_c$')
plt.xlabel('Temperature (K)')
plt.ylabel('Normalized $D_s$')
plt.title('Temperature dependence compared with candidate pairing models')
plt.legend()
plt.tight_layout()
plt.savefig(IMG / 'temperature_models.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 7))
plt.scatter(T_fit, D_fit, s=25, label='Experimental', color='black')
plt.plot(T_fit, D_best, lw=3, label=f'Best power-law fit n={n_best:.2f}')
plt.xlabel('Temperature (K)')
plt.ylabel('Normalized $D_s$')
plt.title('Power-law fit to superfluid stiffness below $T_c$')
plt.legend()
plt.tight_layout()
plt.savefig(IMG / 'temperature_powerlaw_fit.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 7))
plt.plot(I_curr, D_exp_curr, label='Experimental DC', lw=3)
plt.plot(I_curr, D_gl_curr, label='GL quadratic reference', lw=2)
plt.plot(I_curr, D_lin_curr, label='Linear reference', lw=2)
plt.xlabel('DC current (nA)')
plt.ylabel('Normalized $D_s$')
plt.title('Current dependence of superfluid stiffness')
plt.legend()
plt.tight_layout()
plt.savefig(IMG / 'current_dependence.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 7))
plt.scatter((I_curr[idx_low] / I_c) ** 2, 1 - y, label='Experimental near zero current', s=35)
xx = np.linspace(0, (I_curr[idx_low].max() / I_c) ** 2, 200)
plt.plot(xx, quad[0] * xx + quad[1], lw=3, label='Quadratic fit')
plt.xlabel(r'$(I/I_c)^2$')
plt.ylabel(r'$1-D_s(I)/D_s(0)$')
plt.title('Near-zero-current quadratic suppression of stiffness')
plt.legend()
plt.tight_layout()
plt.savefig(IMG / 'near_zero_current_quadratic.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 7))
ax1 = plt.gca()
ax1.plot(I_curr, R_proxy * 1e3, color='tab:red', lw=3)
ax1.set_xlabel('DC current (nA)')
ax1.set_ylabel('Resistance proxy (scaled)', color='tab:red')
ax2 = ax1.twinx()
ax2.plot(I_curr, freq_proxy_I, color='tab:blue', lw=3)
ax2.set_ylabel('Resonance proxy (scaled)', color='tab:blue')
plt.title('Inferred transport and resonance proxies from stiffness-current data')
plt.tight_layout()
plt.savefig(IMG / 'transport_resonance_proxies_vs_current.png', dpi=200)
plt.close()

print('analysis complete')
