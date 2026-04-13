import re
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'MATBG Superfluid Stiffness Core Dataset.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 200

text = DATA_FILE.read_text()


def extract_array(label):
    idx = text.find(label)
    if idx < 0:
        raise ValueError(f'Could not find array for {label}')
    start = text.find('[', idx)
    if start < 0:
        raise ValueError(f'Could not find array start for {label}')
    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f'Could not find array end for {label}')
    payload = text[start:end]
    nums = np.fromstring(payload.strip('[]').replace('\n', ' '), sep=' ')
    return nums


def extract_scalar(label):
    pat = re.escape(label) + r"\s*=\s*([-+0-9.eE]+)"
    m = re.search(pat, text)
    if not m:
        raise ValueError(f'Could not find scalar for {label}')
    return float(m.group(1))

# Parse arrays
n_eff = extract_array('Carrier Density Data (n_eff in m^-2):')
D_s_conv = extract_array('Conventional Superfluid Stiffness (D_s_conv):')
D_s_geom = extract_array('Quantum Geometric Superfluid Stiffness (D_s_geom):')
D_s_exp_hole = extract_array('Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole):')
D_s_exp_electron = extract_array('Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron):')

T = extract_array('Temperature Array (T in K):')
D_s_bcs = extract_array('BCS Model Data (D_s_bcs):')
D_s_nodal = extract_array('Nodal Superconductor Data (D_s_nodal):')
D_s_power_n2 = extract_array('Power Law n=2.0 Data (D_s_power_n2):')
D_s_power_n25 = extract_array('Power Law n=2.5 Data (D_s_power_n2_5):')
D_s_power_n3 = extract_array('Power Law n=3.0 Data (D_s_power_n3):')
D_s_temp_exp = extract_array('Experimental Data with Noise (D_s_experimental):')

I_dc = extract_array('DC Current Array (I_dc in nA):')
D_s_gl = extract_array('Ginzburg-Landau Model (D_s_gl):')
D_s_linear = extract_array('Linear Meissner Model (D_s_linear):')
D_s_dc_exp = extract_array('Experimental DC Data (D_s_dc_exp):')
P_mw = extract_array('Microwave Power Array (P_mw normalized):')
I_mw_amp = extract_array('Microwave Current Amplitude (I_mw_amplitude in nA):')
D_s_mw_exp = extract_array('Experimental Microwave Data (D_s_mw_exp):')

# Align arrays with possible truncation/extended sampling in the text export
D_s_temp_exp = D_s_temp_exp[:len(T)]

def pad_to(arr, n):
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n-len(arr)), mode='constant', constant_values=0)

D_s_bcs = pad_to(D_s_bcs, len(T))
D_s_nodal = pad_to(D_s_nodal, len(T))
D_s_power_n2 = pad_to(D_s_power_n2, len(T))
D_s_power_n25 = pad_to(D_s_power_n25, len(T))
D_s_power_n3 = pad_to(D_s_power_n3, len(T))

D_s_gl = D_s_gl[:len(I_dc)]
D_s_linear = D_s_linear[:len(I_dc)]
D_s_dc_exp = D_s_dc_exp[:len(I_dc)]

# Helpers

def r2_score(y, yhat):
    ss_res = np.sum((y-yhat)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - ss_res/ss_tot

# Carrier-density analysis
carrier_df = pd.DataFrame({
    'n_eff_m2': n_eff,
    'n_eff_cm2': n_eff/1e4,
    'D_s_conv': D_s_conv,
    'D_s_geom': D_s_geom,
    'D_s_exp_hole': D_s_exp_hole,
    'D_s_exp_electron': D_s_exp_electron,
})
carrier_df['enhancement_hole_vs_conv'] = carrier_df['D_s_exp_hole']/carrier_df['D_s_conv']
carrier_df['enhancement_hole_vs_geom'] = carrier_df['D_s_exp_hole']/carrier_df['D_s_geom']
carrier_df['enhancement_electron_vs_conv'] = carrier_df['D_s_exp_electron']/carrier_df['D_s_conv']
carrier_df['enhancement_electron_vs_geom'] = carrier_df['D_s_exp_electron']/carrier_df['D_s_geom']
carrier_df.to_csv(OUT/'carrier_density_analysis.csv', index=False)

# Temperature analysis
mask_fit = (T > 0) & (T < 1.0) & (D_s_temp_exp > 0)
Tfit, Yfit = T[mask_fit], D_s_temp_exp[mask_fit]


def power_model(T, D0, Tc, n):
    x = 1 - (T/Tc)**n
    return D0*np.clip(x, 0, None)

best = None
for p0 in [(100,1.0,2.0),(100,1.0,2.5),(100,1.0,3.0)]:
    try:
        popt, pcov = curve_fit(power_model, Tfit, Yfit, p0=p0, bounds=([50,0.8,1.0],[150,1.2,5.0]), maxfev=20000)
        pred = power_model(Tfit, *popt)
        rss = float(np.sum((Yfit-pred)**2))
        cand = (rss, popt, pcov)
        if best is None or rss < best[0]:
            best = cand
    except Exception:
        pass
rss, popt, pcov = best
D0_fit, Tc_fit, n_fit = [float(x) for x in popt]
T_pred_all = power_model(T, *popt)

models_temp = {
    'BCS_n2': D_s_bcs,
    'Nodal_linear': D_s_nodal,
    'Power_n2.0': D_s_power_n2,
    'Power_n2.5': D_s_power_n25,
    'Power_n3.0': D_s_power_n3,
    'Best_fit_continuous': T_pred_all,
}

temp_scores = []
for name, arr in models_temp.items():
    m = (D_s_temp_exp > 0) | (arr > 0)
    rss = float(np.sum((D_s_temp_exp[m]-arr[m])**2))
    temp_scores.append({'model':name,'rss':rss,'r2':float(r2_score(D_s_temp_exp[m], arr[m]))})
pd.DataFrame(temp_scores).sort_values('rss').to_csv(OUT/'temperature_model_comparison.csv', index=False)

# Current analysis
mask_dc = I_dc <= 50
x = I_dc[mask_dc]
y = D_s_dc_exp[mask_dc]

coeff2 = np.polyfit(x, y, 2)
quad_pred = np.polyval(coeff2, x)
coeff1 = np.polyfit(x, y, 1)
lin_pred = np.polyval(coeff1, x)

current_metrics = pd.DataFrame([
    {'model':'quadratic_fit', 'r2': float(r2_score(y, quad_pred)), 'a2': float(coeff2[0]), 'a1': float(coeff2[1]), 'a0': float(coeff2[2])},
    {'model':'linear_fit', 'r2': float(r2_score(y, lin_pred)), 'a2': 0.0, 'a1': float(coeff1[0]), 'a0': float(coeff1[1])},
    {'model':'GL_reference', 'r2': float(r2_score(y, D_s_gl[:len(y)])), 'a2': np.nan, 'a1': np.nan, 'a0': np.nan},
    {'model':'linear_meissner_reference', 'r2': float(r2_score(y, D_s_linear[:len(y)])), 'a2': np.nan, 'a1': np.nan, 'a0': np.nan},
])
current_metrics.to_csv(OUT/'current_model_comparison.csv', index=False)

# Microwave-current derived resonance shift proxy
mw_df = pd.DataFrame({'P_mw':P_mw, 'I_mw_amp_nA':I_mw_amp, 'D_s_mw_exp':D_s_mw_exp})
mw_df['sqrt_D_ratio'] = np.sqrt(mw_df['D_s_mw_exp']/mw_df['D_s_mw_exp'].iloc[0])
mw_df['freq_shift_proxy'] = 1 - mw_df['sqrt_D_ratio']
mw_df.to_csv(OUT/'microwave_analysis.csv', index=False)

# Summary metrics
summary = {
    'carrier_density': {
        'hole_vs_conv_enhancement_mean': float(carrier_df['enhancement_hole_vs_conv'].mean()),
        'hole_vs_conv_enhancement_range': [float(carrier_df['enhancement_hole_vs_conv'].min()), float(carrier_df['enhancement_hole_vs_conv'].max())],
        'hole_vs_geom_enhancement_mean': float(carrier_df['enhancement_hole_vs_geom'].mean()),
        'electron_vs_conv_enhancement_mean': float(carrier_df['enhancement_electron_vs_conv'].mean()),
        'electron_vs_geom_enhancement_mean': float(carrier_df['enhancement_electron_vs_geom'].mean()),
    },
    'temperature_fit': {
        'D0_fit': D0_fit,
        'Tc_fit_K': Tc_fit,
        'n_fit': n_fit,
        'best_model_r2': float(r2_score(D_s_temp_exp, T_pred_all)),
    },
    'current_fit': {
        'quadratic_r2': float(r2_score(y, quad_pred)),
        'linear_r2': float(r2_score(y, lin_pred)),
        'quadratic_coefficients': [float(c) for c in coeff2],
    },
    'microwave': {
        'max_freq_shift_proxy': float(mw_df['freq_shift_proxy'].max())
    }
}
(OUT/'summary_metrics.json').write_text(json.dumps(summary, indent=2))

# Plots
# 1 carrier density
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(carrier_df['n_eff_cm2']/1e11, carrier_df['D_s_conv']/1e9, label='Conventional FL theory', lw=2.5)
ax.plot(carrier_df['n_eff_cm2']/1e11, carrier_df['D_s_geom']/1e9, label='Quantum geometric contribution', lw=2.5)
ax.plot(carrier_df['n_eff_cm2']/1e11, carrier_df['D_s_exp_hole']/1e9, label='Experimental hole-doped', lw=2.5)
ax.plot(carrier_df['n_eff_cm2']/1e11, carrier_df['D_s_exp_electron']/1e9, label='Experimental electron-doped', lw=2.5, ls='--')
ax.set_xlabel(r'Effective carrier density $n_{eff}$ ($10^{11}$ cm$^{-2}$)')
ax.set_ylabel(r'Superfluid stiffness $D_s$ ($10^9$ in arb. units)')
ax.set_title('Carrier-density dependence of superfluid stiffness in MATBG')
ax.legend(frameon=True, fontsize=10)
fig.tight_layout(); fig.savefig(IMG/'carrier_density_stiffness.png'); plt.close(fig)

# 2 enhancement ratio
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(carrier_df['n_eff_cm2']/1e11, carrier_df['enhancement_hole_vs_conv'], label='Hole / conventional', lw=2.5)
ax.plot(carrier_df['n_eff_cm2']/1e11, carrier_df['enhancement_hole_vs_geom'], label='Hole / geometric', lw=2.5)
ax.plot(carrier_df['n_eff_cm2']/1e11, carrier_df['enhancement_electron_vs_conv'], label='Electron / conventional', lw=2.5, ls='--')
ax.axhline(1.0, color='k', lw=1, alpha=0.6)
ax.set_xlabel(r'Effective carrier density $n_{eff}$ ($10^{11}$ cm$^{-2}$)')
ax.set_ylabel('Enhancement factor')
ax.set_title('Experimental stiffness strongly exceeds conventional expectation')
ax.legend(frameon=True, fontsize=10)
fig.tight_layout(); fig.savefig(IMG/'carrier_density_enhancement.png'); plt.close(fig)

# 3 temperature comparison
fig, ax = plt.subplots(figsize=(9,6))
ax.scatter(T, D_s_temp_exp, s=20, color='black', label='Experimental data', zorder=5)
ax.plot(T, D_s_bcs, label='BCS / n=2', lw=2)
ax.plot(T, D_s_power_n25, label='Power law n=2.5', lw=2)
ax.plot(T, D_s_power_n3, label='Power law n=3', lw=2)
ax.plot(T, T_pred_all, label=f'Best continuous fit n={n_fit:.2f}', lw=3, color='red')
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 105)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel(r'Normalized $D_s(T)$')
ax.set_title('Temperature dependence favors a power-law suppression')
ax.legend(frameon=True, fontsize=10)
fig.tight_layout(); fig.savefig(IMG/'temperature_dependence_fit.png'); plt.close(fig)

# 4 temperature residuals
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(T, D_s_temp_exp-D_s_bcs, label='Residual vs BCS/n=2', lw=2)
ax.plot(T, D_s_temp_exp-D_s_power_n25, label='Residual vs n=2.5', lw=2)
ax.plot(T, D_s_temp_exp-D_s_power_n3, label='Residual vs n=3', lw=2)
ax.axhline(0, color='k', lw=1)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Residual')
ax.set_title('Residual comparison for candidate gap structures')
ax.legend(frameon=True, fontsize=10)
fig.tight_layout(); fig.savefig(IMG/'temperature_model_residuals.png'); plt.close(fig)

# 5 dc current
fig, ax = plt.subplots(figsize=(9,6))
ax.scatter(x, y, s=26, color='black', label='Experimental DC response')
ax.plot(x, D_s_gl[:len(x)], lw=2.5, label='GL reference')
ax.plot(x, D_s_linear[:len(x)], lw=2.5, label='Linear Meissner reference')
ax.plot(x, quad_pred, lw=3, color='red', label='Quadratic fit')
ax.set_xlabel(r'DC bias current $I_{dc}$ (nA)')
ax.set_ylabel(r'Normalized $D_s(I)$')
ax.set_title('DC current dependence is captured by a quadratic law below $I_c$')
ax.legend(frameon=True, fontsize=10)
fig.tight_layout(); fig.savefig(IMG/'current_dependence_dc.png'); plt.close(fig)

# 6 microwave
fig, ax1 = plt.subplots(figsize=(9,6))
ax1.plot(I_mw_amp, D_s_mw_exp, lw=2.5, label=r'$D_s$ from microwave probe')
ax1.set_xlabel(r'Microwave current amplitude $I_{mw}$ (nA)')
ax1.set_ylabel(r'Normalized $D_s$')
ax2 = ax1.twinx()
ax2.plot(I_mw_amp, mw_df['freq_shift_proxy'], color='red', lw=2.5, ls='--', label='Resonance-shift proxy')
ax2.set_ylabel('Relative resonance-frequency shift proxy')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, frameon=True, fontsize=10, loc='center right')
ax1.set_title('Microwave response tracks the reduction of superfluid stiffness')
fig.tight_layout(); fig.savefig(IMG/'microwave_response.png'); plt.close(fig)

print(json.dumps(summary, indent=2))
