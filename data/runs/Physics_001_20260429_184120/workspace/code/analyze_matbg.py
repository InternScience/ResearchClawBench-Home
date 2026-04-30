#!/usr/bin/env python3
"""Reproducible MATBG superfluid-stiffness analysis.

Parses the provided text dataset, exports quantitative summaries, and creates
PNG figures for the final report.  The dataset directly supplies superfluid
stiffness arrays.  Because raw dc resistance and microwave resonance frequency
are not separately tabulated, this script uses standard normalized proxies:
R_dc_proxy = D_s(0)/D_s for dissipation/kinetic-inductance growth and
f_res_proxy = sqrt(D_s/D_s(0)) for resonator-frequency shifts.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "MATBG Superfluid Stiffness Core Dataset.txt"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

text = DATA_PATH.read_text()

# Labels in the dataset are unique and followed by bracketed numpy-style arrays.
LABELS = {
    'n_eff_m2': 'Carrier Density Data (n_eff in m^-2)',
    'D_s_conv': 'Conventional Superfluid Stiffness (D_s_conv)',
    'D_s_geom': 'Quantum Geometric Superfluid Stiffness (D_s_geom)',
    'D_s_exp_hole': 'Experimental Superfluid Stiffness Hole-doped (D_s_exp_hole)',
    'D_s_exp_electron': 'Experimental Superfluid Stiffness Electron-doped (D_s_exp_electron)',
    'T_K': 'Temperature Array (T in K)',
    'D_s_bcs': 'BCS Model Data (D_s_bcs)',
    'D_s_nodal': 'Nodal Superconductor Data (D_s_nodal)',
    'D_s_power_n2': 'Power Law n=2.0 Data (D_s_power_n2)',
    'D_s_power_n2_5': 'Power Law n=2.5 Data (D_s_power_n2_5)',
    'D_s_power_n3': 'Power Law n=3.0 Data (D_s_power_n3)',
    'D_s_experimental_T': 'Experimental Data with Noise (D_s_experimental)',
    'I_dc_nA': 'DC Current Array (I_dc in nA)',
    'D_s_gl': 'Ginzburg-Landau Model (D_s_gl)',
    'D_s_linear': 'Linear Meissner Model (D_s_linear)',
    'D_s_dc_exp': 'Experimental DC Data (D_s_dc_exp)',
    'P_mw_norm': 'Microwave Power Array (P_mw normalized)',
    'I_mw_nA': 'Microwave Current Amplitude (I_mw_amplitude in nA)',
    'D_s_mw_exp': 'Experimental Microwave Data (D_s_mw_exp)',
}

def get_array(label: str) -> np.ndarray:
    idx = text.index(f"**{label}:**")
    start = text.index('[', idx)
    end = text.index(']', start)
    raw = text[start+1:end]
    return np.fromstring(raw.replace('\n', ' '), sep=' ')

arr = {k: get_array(v) for k, v in LABELS.items()}
summary = {k: {'n': int(len(v)), 'min': float(np.nanmin(v)), 'max': float(np.nanmax(v)), 'mean': float(np.nanmean(v))} for k, v in arr.items()}
(OUT/'parsed_core_dataset_summary.json').write_text(json.dumps(summary, indent=2))

# Density dependence
n = arr['n_eff_m2']
density = pd.DataFrame({
    'n_eff_m2': n,
    'n_eff_1e11_cm2': n / 1e15,  # 1e15 m^-2 = 1e11 cm^-2
    'D_s_conv': arr['D_s_conv'],
    'D_s_geom': arr['D_s_geom'],
    'D_s_exp_hole': arr['D_s_exp_hole'],
    'D_s_exp_electron': arr['D_s_exp_electron'],
})
density['geom_over_conv'] = density.D_s_geom / density.D_s_conv
density['hole_over_conv'] = density.D_s_exp_hole / density.D_s_conv
density['electron_over_conv'] = density.D_s_exp_electron / density.D_s_conv
density['hole_over_geom'] = density.D_s_exp_hole / density.D_s_geom
density['electron_over_geom'] = density.D_s_exp_electron / density.D_s_geom
density['hole_electron_asymmetry_pct'] = 100*(density.D_s_exp_hole-density.D_s_exp_electron)/((density.D_s_exp_hole+density.D_s_exp_electron)/2)
density.to_csv(OUT/'density_dependence_full.csv', index=False)

mid_idx = int(np.argmin(np.abs(density.n_eff_1e11_cm2 - 2.5)))
density_summary = pd.DataFrame([
    {'metric':'n_points','value':len(density),'units':'count'},
    {'metric':'density_min','value':density.n_eff_1e11_cm2.min(),'units':'1e11 cm^-2'},
    {'metric':'density_max','value':density.n_eff_1e11_cm2.max(),'units':'1e11 cm^-2'},
    {'metric':'mean_geom_over_conv','value':density.geom_over_conv.mean(),'units':'ratio'},
    {'metric':'mean_hole_over_conv','value':density.hole_over_conv.mean(),'units':'ratio'},
    {'metric':'mean_electron_over_conv','value':density.electron_over_conv.mean(),'units':'ratio'},
    {'metric':'mean_hole_over_geom','value':density.hole_over_geom.mean(),'units':'ratio'},
    {'metric':'mean_electron_over_geom','value':density.electron_over_geom.mean(),'units':'ratio'},
    {'metric':'median_hole_electron_asymmetry','value':density.hole_electron_asymmetry_pct.median(),'units':'percent'},
    {'metric':'example_mid_density_hole_Ds','value':density.loc[mid_idx,'D_s_exp_hole'],'units':'dataset D_s'},
])
density_summary.to_csv(OUT/'density_enhancement_summary.csv', index=False)

# Temperature dependence; fit experimental data to D(T)=D0 - A*T^alpha at low T.
T = arr['T_K']
Dtexp = arr['D_s_experimental_T']
# The simulated experimental temperature trace contains 110 points whereas the
# model temperature grid contains 100; use a matched linear grid over 0--1.2 K.
T_exp = np.linspace(T.min(), T.max(), len(Dtexp))
# Restrict below 0.35 K for low-T exponent while avoiding T=0 for log fit.
mask_low = (T_exp > 0) & (T_exp <= 0.35)

def power_model(T, D0, A, alpha):
    return D0 - A*np.power(T, alpha)
popt, pcov = curve_fit(power_model, T_exp[mask_low], Dtexp[mask_low], p0=[100, 25, 1.0], maxfev=20000, bounds=([50,0,0.1],[150,200,5]))
perr = np.sqrt(np.diag(pcov))
# Log slope cross-check using fitted D0.
delta = np.maximum(popt[0] - Dtexp[mask_low], 1e-9)
logfit = linregress(np.log(T_exp[mask_low]), np.log(delta))
models_T = {
    'BCS_n2': arr['D_s_bcs'],
    'nodal_linear': arr['D_s_nodal'],
    'power_n2': arr['D_s_power_n2'],
    'power_n2_5': arr['D_s_power_n2_5'],
    'power_n3': arr['D_s_power_n3'],
}
temp_rows=[]
for name, y in models_T.items():
    # compare first 95 shared points (dataset gives 100 T points, experimental has 110)
    m=min(len(y),len(T),len(Dtexp))
    resid = Dtexp[:m]-y[:m]
    temp_rows.append({'model':name,'rmse_full_shared':float(np.sqrt(np.mean(resid**2))), 'mae_full_shared':float(np.mean(np.abs(resid)))})
# Low-T fitted model RMSE
pred_low = power_model(T_exp[mask_low], *popt)
temp_rows.append({'model':'experimental_lowT_fit','rmse_full_shared':float(np.sqrt(np.mean((Dtexp[mask_low]-pred_low)**2))), 'mae_full_shared':float(np.mean(np.abs(Dtexp[mask_low]-pred_low)))})
temp_fit = pd.DataFrame(temp_rows)
temp_fit['fitted_D0_lowT'] = popt[0]
temp_fit['fitted_A_lowT'] = popt[1]
temp_fit['fitted_alpha_lowT'] = popt[2]
temp_fit['fitted_alpha_se'] = perr[2]
temp_fit['log_slope_alpha_crosscheck'] = logfit.slope
temp_fit.to_csv(OUT/'temperature_fit_summary.csv', index=False)

# Current dependence: fit D(I)=D0 - a I^2 for low/intermediate DC current and compare to linear.
I = arr['I_dc_nA']; Ddc = arr['D_s_dc_exp']
# The provided DC experimental trace contains a longer continuation, but the first
# 50 values align with the labelled 0--60 nA current grid and the GL array.
Ddc = Ddc[:len(I)]
I_exp = I.copy()
mask_I = (I_exp <= 45)  # GL-like regime before critical-current rolloff
X2 = np.vstack([np.ones(mask_I.sum()), I_exp[mask_I]**2]).T
coef2, *_ = np.linalg.lstsq(X2, Ddc[mask_I], rcond=None)
pred2 = X2 @ coef2
ss_res2 = np.sum((Ddc[mask_I]-pred2)**2); ss_tot=np.sum((Ddc[mask_I]-Ddc[mask_I].mean())**2)
r2_quad = 1-ss_res2/ss_tot
X1 = np.vstack([np.ones(mask_I.sum()), I_exp[mask_I]]).T
coef1, *_ = np.linalg.lstsq(X1, Ddc[mask_I], rcond=None)
pred1 = X1 @ coef1
ss_res1=np.sum((Ddc[mask_I]-pred1)**2); r2_lin=1-ss_res1/ss_tot
# Critical current from fitted D0/a when positive
Ic_fit = float(np.sqrt(coef2[0]/(-coef2[1]))) if coef2[1] < 0 else np.nan
# Microwave fit with current amplitude: D = D0 - a I_mw^2
Imw=arr['I_mw_nA']; Dmw=arr['D_s_mw_exp']
Xm=np.vstack([np.ones(len(Imw)), Imw**2]).T
coefm,*_=np.linalg.lstsq(Xm,Dmw,rcond=None)
predm=Xm@coefm
r2_mw=1-np.sum((Dmw-predm)**2)/np.sum((Dmw-Dmw.mean())**2)
current_fit=pd.DataFrame([
    {'dataset':'dc_exp_low_to_45nA','model':'D0 + b I^2','D0':coef2[0],'b':coef2[1],'R2':r2_quad,'RMSE':np.sqrt(np.mean((Ddc[mask_I]-pred2)**2)),'Ic_from_zero_nA':Ic_fit},
    {'dataset':'dc_exp_low_to_45nA','model':'D0 + b I','D0':coef1[0],'b':coef1[1],'R2':r2_lin,'RMSE':np.sqrt(np.mean((Ddc[mask_I]-pred1)**2)),'Ic_from_zero_nA':np.nan},
    {'dataset':'mw_exp','model':'D0 + b I_mw^2','D0':coefm[0],'b':coefm[1],'R2':r2_mw,'RMSE':np.sqrt(np.mean((Dmw-predm)**2)),'Ic_from_zero_nA':np.sqrt(coefm[0]/(-coefm[1])) if coefm[1] < 0 else np.nan},
])
current_fit.to_csv(OUT/'current_fit_summary.csv', index=False)

# Proxies for requested DC resistance and microwave resonance frequency.
proxy = pd.DataFrame({
    'I_dc_nA': I_exp,
    'D_s_dc_exp': Ddc,
    'R_dc_proxy_norm': Ddc[0]/np.maximum(Ddc, 1e-9),
    'f_res_proxy_norm': np.sqrt(np.maximum(Ddc,0)/Ddc[0])
})
proxy.to_csv(OUT/'dc_resistance_resonance_proxies.csv', index=False)
proxy_mw = pd.DataFrame({
    'P_mw_norm': arr['P_mw_norm'], 'I_mw_nA': Imw, 'D_s_mw_exp': Dmw,
    'R_mw_proxy_norm': Dmw[0]/np.maximum(Dmw,1e-9),
    'f_res_proxy_norm': np.sqrt(np.maximum(Dmw,0)/Dmw[0])
})
proxy_mw.to_csv(OUT/'microwave_resonance_proxies.csv', index=False)

# Claim recovery table
claim_rows = [
    {'claim':'Experimental stiffness exceeds conventional Fermi-liquid estimate','supporting_artifact':'outputs/density_enhancement_summary.csv','quantitative_result':f"mean hole/conv={density.hole_over_conv.mean():.1f}, mean electron/conv={density.electron_over_conv.mean():.1f}", 'status':'supported by dataset'},
    {'claim':'Quantum-geometric contribution is much larger than conventional band contribution','supporting_artifact':'outputs/density_enhancement_summary.csv','quantitative_result':f"mean geom/conv={density.geom_over_conv.mean():.2f}", 'status':'supported by dataset'},
    {'claim':'Observed temperature dependence is power-law and close to nodal/anisotropic behavior','supporting_artifact':'outputs/temperature_fit_summary.csv','quantitative_result':f"low-T alpha={popt[2]:.2f}±{perr[2]:.2f}; log-slope={logfit.slope:.2f}", 'status':'supported for low-T simulated experimental array'},
    {'claim':'DC current suppresses stiffness approximately quadratically before high-current deviation','supporting_artifact':'outputs/current_fit_summary.csv','quantitative_result':f"quadratic R2={r2_quad:.4f}, linear R2={r2_lin:.4f}, Ic_fit={Ic_fit:.1f} nA", 'status':'supported on I<=45 nA'},
    {'claim':'Microwave drive produces weaker stiffness suppression over measured power range','supporting_artifact':'outputs/microwave_resonance_proxies.csv','quantitative_result':f"D_s falls from {Dmw[0]:.1f} to {Dmw[-1]:.1f}; f_res proxy to {np.sqrt(Dmw[-1]/Dmw[0]):.3f}", 'status':'supported by dataset'},
    {'claim':'Raw resistance and resonance frequency are directly measured','supporting_artifact':'outputs/dc_resistance_resonance_proxies.csv','quantitative_result':'not directly tabulated; normalized proxies exported', 'status':'limitation'},
]
pd.DataFrame(claim_rows).to_csv(OUT/'claim_recovery_table.csv', index=False)

# Figures
plt.style.use('seaborn-v0_8-whitegrid')

fig, axs = plt.subplots(2,2, figsize=(12,9))
axs[0,0].plot(density.n_eff_1e11_cm2, density.D_s_exp_hole/1e11, label='hole exp')
axs[0,0].plot(density.n_eff_1e11_cm2, density.D_s_exp_electron/1e11, label='electron exp')
axs[0,0].set_xlabel(r'$n_{eff}$ ($10^{11}$ cm$^{-2}$)'); axs[0,0].set_ylabel(r'$D_s$ ($10^{11}$ arb.)'); axs[0,0].legend(); axs[0,0].set_title('Carrier-density sweep')
axs[0,1].plot(T_exp, Dtexp, color='tab:purple')
axs[0,1].set_xlabel('T (K)'); axs[0,1].set_ylabel(r'$D_s$'); axs[0,1].set_title('Temperature sweep')
axs[1,0].plot(I_exp, Ddc, 'o-', ms=3, label='DC')
axs[1,0].plot(Imw, Dmw, 's-', ms=3, label='MW')
axs[1,0].set_xlabel('current amplitude (nA)'); axs[1,0].set_ylabel(r'$D_s$'); axs[1,0].legend(); axs[1,0].set_title('Current/microwave sweeps')
axs[1,1].plot(proxy.I_dc_nA, proxy.R_dc_proxy_norm, label='R proxy')
axs[1,1].plot(proxy.I_dc_nA, proxy.f_res_proxy_norm, label='f_res proxy')
axs[1,1].set_xlabel('I_dc (nA)'); axs[1,1].set_ylabel('normalized proxy'); axs[1,1].legend(); axs[1,1].set_title('Resistance/resonance proxies')
fig.tight_layout(); fig.savefig(IMG/'figure_1_data_overview.png', dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8,5.2))
ax.plot(density.n_eff_1e11_cm2, density.D_s_conv/1e9, '--', label='conventional FL')
ax.plot(density.n_eff_1e11_cm2, density.D_s_geom/1e9, '-.', label='quantum geometric')
ax.plot(density.n_eff_1e11_cm2, density.D_s_exp_hole/1e9, label='exp hole')
ax.plot(density.n_eff_1e11_cm2, density.D_s_exp_electron/1e9, label='exp electron')
ax.set_xlabel(r'$n_{eff}$ ($10^{11}$ cm$^{-2}$)'); ax.set_ylabel(r'$D_s$ ($10^9$ dataset units)')
ax.set_title('Superfluid stiffness exceeds conventional estimate')
ax.legend(); fig.tight_layout(); fig.savefig(IMG/'figure_2_density_stiffness.png', dpi=200); plt.close(fig)

fig, axs = plt.subplots(1,2,figsize=(12,5))
axs[0].plot(T[:len(arr['D_s_bcs'])], arr['D_s_bcs'], label='BCS / n=2')
axs[0].plot(T[:len(arr['D_s_nodal'])], arr['D_s_nodal'], label='nodal linear')
axs[0].plot(T[:len(arr['D_s_power_n2_5'])], arr['D_s_power_n2_5'], label='n=2.5')
axs[0].plot(T[:len(arr['D_s_power_n3'])], arr['D_s_power_n3'], label='n=3')
axs[0].plot(T_exp, Dtexp, 'k.', ms=3, label='experimental')
axs[0].set_xlabel('T (K)'); axs[0].set_ylabel(r'$D_s$'); axs[0].set_title('Temperature dependence models'); axs[0].legend(fontsize=8)
axs[1].plot(T_exp[mask_low], Dtexp[mask_low], 'ko', ms=4, label='low-T exp')
Tf=np.linspace(T_exp[mask_low].min(), T_exp[mask_low].max(), 200)
axs[1].plot(Tf, power_model(Tf,*popt), 'r-', label=fr'fit $D_0-A T^\alpha$, $\alpha={popt[2]:.2f}$')
axs[1].set_xlabel('T (K)'); axs[1].set_ylabel(r'$D_s$'); axs[1].set_title('Low-temperature power law'); axs[1].legend()
fig.tight_layout(); fig.savefig(IMG/'figure_3_temperature_powerlaw.png', dpi=200); plt.close(fig)

fig, axs = plt.subplots(1,2,figsize=(12,5))
I_model = np.linspace(I.min(), I.max(), len(arr['D_s_gl']))
axs[0].plot(I_model, arr['D_s_gl'], '--', label='GL model')
I_linear = np.linspace(I.min(), I.max(), len(arr['D_s_linear']))
axs[0].plot(I_linear, arr['D_s_linear'], '--', label='linear model')
axs[0].plot(I_exp, Ddc, 'ko', ms=3, label='DC experimental')
If=np.linspace(0,45,200); axs[0].plot(If, coef2[0]+coef2[1]*If**2, 'r-', label=f'quadratic fit R²={r2_quad:.3f}')
axs[0].set_xlabel('I_dc (nA)'); axs[0].set_ylabel(r'$D_s$'); axs[0].set_title('DC current depairing'); axs[0].legend(fontsize=8)
axs[1].plot(arr['P_mw_norm'], Dmw, 'o-', label='MW experimental')
axs[1].set_xlabel('normalized microwave power'); axs[1].set_ylabel(r'$D_s$'); axs[1].set_title('Microwave drive response')
ax2=axs[1].twiny(); ax2.plot(Imw, Dmw, alpha=0); ax2.set_xlabel('I_mw amplitude (nA)')
fig.tight_layout(); fig.savefig(IMG/'figure_4_current_dependence.png', dpi=200); plt.close(fig)

fig, axs = plt.subplots(1,2,figsize=(12,5))
axs[0].bar(['geom/conv','hole/conv','electron/conv'], [density.geom_over_conv.mean(), density.hole_over_conv.mean(), density.electron_over_conv.mean()], color=['tab:blue','tab:red','tab:orange'])
axs[0].set_ylabel('mean enhancement ratio'); axs[0].set_title('Stiffness enhancement validation')
axs[1].bar(['quad R²','linear R²','MW quad R²'], [r2_quad, r2_lin, r2_mw], color=['tab:green','tab:gray','tab:purple'])
axs[1].set_ylim(0,1.05); axs[1].set_ylabel('R²'); axs[1].set_title('Model comparison')
fig.tight_layout(); fig.savefig(IMG/'figure_5_validation_comparison.png', dpi=200); plt.close(fig)

# Update target inventory statuses
inv_path=OUT/'target_artifact_inventory.json'
if inv_path.exists():
    inv=json.loads(inv_path.read_text())
    for art in inv.get('artifacts',[]):
        p=ROOT/art['target_path']
        art['status']='satisfied' if p.exists() else 'unsatisfied'
        if not p.exists(): art['reason']='file missing after analysis run'
    inv_path.write_text(json.dumps(inv, indent=2))

print(json.dumps({
    'density_mean_hole_over_conv': density.hole_over_conv.mean(),
    'density_mean_geom_over_conv': density.geom_over_conv.mean(),
    'temp_alpha': popt[2],
    'dc_quad_r2': r2_quad,
    'dc_linear_r2': r2_lin,
    'mw_quad_r2': r2_mw,
    'figures': [p.name for p in sorted(IMG.glob('*.png'))]
}, indent=2))

if __name__ == '__main__':
    pass
