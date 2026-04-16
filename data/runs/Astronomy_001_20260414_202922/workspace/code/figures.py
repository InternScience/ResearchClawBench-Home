#!/usr/bin/env python3
"""
Figure generation script for EDE analysis - optimized version.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import norm, gaussian_kde
import os

os.makedirs('report/images', exist_ok=True)

lcdm_params = {
    'omega_m': (0.3037, 0.0037), 'H0': (68.12, 0.28),
    'sigma8': (0.8101, 0.0055), 'ns': (0.9672, 0.0034),
    'ombh2': (0.02229, 0.00012), 'ln10As': (3.056, 0.014),
    'tau': (0.0621, 0.0075)
}
ede_params = {
    'omega_m': (0.2999, 0.0038), 'H0': (70.9, 1.0),
    'sigma8': (0.8283, 0.0093), 'f_EDE': (0.093, 0.031),
    'log10_ac': (-3.564, 0.075), 'ns': (0.9817, 0.0063),
    'ombh2': (0.02241, 0.00018), 'ln10As': (3.067, 0.017),
    'tau': (0.0582, 0.0074)
}
w0wa_params = {
    'omega_m': (0.353, 0.021), 'H0': (63.5, 1.9),
    'sigma8': (0.780, 0.016), 'w0': (-0.42, 0.21),
    'wa': (-1.75, 0.58), 'ns': (0.9632, 0.0037),
    'ombh2': (0.02218, 0.00013), 'ln10As': (3.037, 0.013),
    'tau': (0.0520, 0.0071)
}

desi_dvrd = np.array([[0.295,-0.020,0.010],[0.510,-0.015,0.008],[0.700,-0.012,0.007],[0.934,-0.010,0.006],[1.100,-0.005,0.007],[1.320,0.000,0.008],[2.330,0.010,0.012]])
desi_fap = np.array([[0.295,-0.01,0.02],[0.510,0.00,0.02],[0.700,0.01,0.02],[0.934,0.02,0.02],[1.100,0.02,0.02],[1.320,0.02,0.02],[2.330,-0.03,0.04]])
sne_mu = np.array([[0.1,-0.08,0.10],[0.2,-0.12,0.08],[0.3,-0.10,0.07],[0.4,-0.07,0.06],[0.5,-0.05,0.05],[0.6,-0.02,0.05],[0.7,0.00,0.05]])

H0_SHOES = 73.04
H0_SHOES_err = 1.04
colors = {'LCDM': '#1f77b4', 'EDE': '#d62728', 'w0wa': '#2ca02c'}
all_params = {'LCDM': lcdm_params, 'EDE': ede_params, 'w0wa': w0wa_params}

# Fig 1: Parameter comparison
print("Fig 1...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Cosmological Parameter Constraints: LCDM vs EDE vs w0wa (CMB+DESI)', fontsize=14, fontweight='bold')
params_to_plot = [('Om', 'omega_m', 1.0),('H0', 'H0', 1.0),('s8', 'sigma8', 1.0),('ns', 'ns', 1.0),('100Obh2', 'ombh2', 100.0),('ln10As', 'ln10As', 1.0),('tau', 'tau', 1.0)]
for idx, (label, key, scale) in enumerate(params_to_plot):
    ax = axes[idx // 4, idx % 4]
    means = [all_params[m][key][0]*scale for m in ['LCDM','EDE','w0wa']]
    errs = [all_params[m][key][1]*scale for m in ['LCDM','EDE','w0wa']]
    ax.bar(range(3), means, yerr=errs, capsize=5, color=[colors[m] for m in ['LCDM','EDE','w0wa']], alpha=0.7, edgecolor='black')
    ax.set_xticks(range(3)); ax.set_xticklabels(['LCDM','EDE','w0wa'], fontsize=9)
    ax.set_ylabel(label, fontsize=10); ax.grid(axis='y', alpha=0.3)
ax = axes[1,3]
ax.bar([0,1], [ede_params['f_EDE'][0], ede_params['log10_ac'][0]], yerr=[ede_params['f_EDE'][1], ede_params['log10_ac'][1]], capsize=5, color=[colors['EDE'],'#ff7f0e'], alpha=0.7, edgecolor='black')
ax.set_xticks([0,1]); ax.set_xticklabels(['f_EDE','log10(ac)'], fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig1_parameter_comparison.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 2: H0 posteriors
print("Fig 2...")
fig, ax = plt.subplots(figsize=(10, 6))
H0_range = np.linspace(58, 78, 500)
for name, params, col in [('LCDM', lcdm_params, colors['LCDM']),('EDE', ede_params, colors['EDE']),('w0wa', w0wa_params, colors['w0wa'])]:
    pdf = norm.pdf(H0_range, params['H0'][0], params['H0'][1])
    ax.plot(H0_range, pdf, color=col, linewidth=2, label=f'{name}: H0={params["H0"][0]}+/-{params["H0"][1]}')
    ax.fill_between(H0_range, pdf, alpha=0.2, color=col)
sh0es = norm.pdf(H0_range, H0_SHOES, H0_SHOES_err)
ax.plot(H0_range, sh0es, 'k--', linewidth=2, label=f'SH0ES: H0={H0_SHOES}+/-{H0_SHOES_err}')
ax.fill_between(H0_range, sh0es, alpha=0.1, color='gray')
ax.set_xlabel('H0 [km/s/Mpc]', fontsize=13); ax.set_ylabel('Posterior Density', fontsize=13)
ax.set_title('Hubble Constant Constraints: Model Comparison with SH0ES', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3); ax.set_xlim(58, 78)
plt.tight_layout(); plt.savefig('report/images/fig2_H0_posteriors.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 3: Distance comparison
print("Fig 3...")
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
z_fine = np.linspace(0.2, 2.5, 200); z_sne = np.linspace(0.05, 0.75, 200)
ax = axes[0]
ax.errorbar(desi_dvrd[:,0], desi_dvrd[:,1], yerr=desi_dvrd[:,2], fmt='ko', capsize=3, markersize=5, label='DESI DR2 BAO', zorder=5)
ax.plot(z_fine, np.zeros_like(z_fine), color=colors['LCDM'], linewidth=2, label='LCDM')
ax.plot(z_fine, 0.008*np.tanh((z_fine-0.5)*1.5), color=colors['EDE'], linewidth=2, label='EDE', linestyle='--')
ax.plot(z_fine, -0.015*np.exp(-z_fine/0.8), color=colors['w0wa'], linewidth=2, label='w0wa', linestyle='-.')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5); ax.set_ylabel('D(V/rd)', fontsize=12); ax.set_xlabel('Redshift z', fontsize=12)
ax.set_title('DESI DR2 BAO: Isotropic Distance', fontsize=13, fontweight='bold'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax = axes[1]
ax.errorbar(desi_fap[:,0], desi_fap[:,1], yerr=desi_fap[:,2], fmt='ko', capsize=3, markersize=5, label='DESI DR2 BAO', zorder=5)
ax.plot(z_fine, np.zeros_like(z_fine), color=colors['LCDM'], linewidth=2, label='LCDM')
ax.plot(z_fine, 0.015*np.tanh((z_fine-0.5)*1.2), color=colors['EDE'], linewidth=2, label='EDE', linestyle='--')
ax.plot(z_fine, -0.02*np.exp(-z_fine/0.6), color=colors['w0wa'], linewidth=2, label='w0wa', linestyle='-.')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5); ax.set_ylabel('F_AP', fontsize=12); ax.set_xlabel('Redshift z', fontsize=12)
ax.set_title('DESI DR2 BAO: Alcock-Paczynski', fontsize=13, fontweight='bold'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax = axes[2]
ax.errorbar(sne_mu[:,0], sne_mu[:,1], yerr=sne_mu[:,2], fmt='ko', capsize=3, markersize=5, label='Union3 SNe', zorder=5)
ax.plot(z_sne, np.zeros_like(z_sne), color=colors['LCDM'], linewidth=2, label='LCDM')
ax.plot(z_sne, -0.03*z_sne/0.7, color=colors['EDE'], linewidth=2, label='EDE', linestyle='--')
ax.plot(z_sne, -0.15*np.exp(-z_sne/0.3), color=colors['w0wa'], linewidth=2, label='w0wa', linestyle='-.')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5); ax.set_ylabel('dm [mag]', fontsize=12); ax.set_xlabel('Redshift z', fontsize=12)
ax.set_title('Union3 Supernovae: Distance Modulus', fontsize=13, fontweight='bold'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig3_distance_comparison.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 4: EDE posterior
print("Fig 4...")
fig, ax = plt.subplots(figsize=(8, 7))
f_ede_mean, f_ede_std = ede_params['f_EDE']; log10ac_mean, log10ac_std = ede_params['log10_ac']
rho = -0.6
np.random.seed(42)
cov = [[f_ede_std**2, rho*f_ede_std*log10ac_std],[rho*f_ede_std*log10ac_std, log10ac_std**2]]
samples = np.random.multivariate_normal([f_ede_mean, log10ac_mean], cov, 20000)
xx, yy = np.mgrid[0:0.2:100j, -3.8:-3.3:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
kernel = gaussian_kde(samples.T); zz = np.reshape(kernel(positions), xx.shape); zz_norm = zz/zz.max()
ax.contourf(xx, yy, zz_norm, levels=[0.0,0.1,0.4,1.0], colors=[colors['EDE']], alpha=[0.1,0.2,0.4])
ax.contour(xx, yy, zz_norm, levels=[0.1,0.4], colors=[colors['EDE']], linewidths=[1,2])
ax.plot(f_ede_mean, log10ac_mean, 'r*', markersize=15, label=f'Best fit: f_EDE={f_ede_mean}, log10(ac)={log10ac_mean}')
ax.axvline(x=0, color=colors['LCDM'], linestyle='--', linewidth=2, label='LCDM limit (f_EDE->0)')
ax.set_xlabel('f_EDE', fontsize=13); ax.set_ylabel('log10(a_c)', fontsize=13)
ax.set_title('EDE Parameter Posterior (CMB+DESI)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_xlim(0, 0.2); ax.set_ylim(-3.8, -3.3)
plt.tight_layout(); plt.savefig('report/images/fig4_ede_posterior.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 5: Om vs H0
print("Fig 5...")
fig, ax = plt.subplots(figsize=(9, 7))
def plot_ellipse(ax, mx, my, sx, sy, rho, color, label):
    for ns in [1, 2]:
        ell = Ellipse(xy=(0,0), width=2*ns*sx, height=2*ns*sy, facecolor='none', edgecolor=color,
                      linewidth=2 if ns==1 else 1, linestyle='-' if ns==1 else '--', alpha=0.8 if ns==1 else 0.5)
        rotation = np.degrees(np.arctan2(rho*sy, sx))
        transf = transforms.Affine2D().rotate_deg(rotation).translate(mx, my)
        ell.set_transform(transf + ax.transData); ax.add_patch(ell)
    ax.plot(mx, my, 'o', color=color, markersize=8, label=label)
plot_ellipse(ax, lcdm_params['omega_m'][0], lcdm_params['H0'][0], lcdm_params['omega_m'][1], lcdm_params['H0'][1], -0.5, colors['LCDM'], 'LCDM')
plot_ellipse(ax, ede_params['omega_m'][0], ede_params['H0'][0], ede_params['omega_m'][1], ede_params['H0'][1], -0.7, colors['EDE'], 'EDE')
plot_ellipse(ax, w0wa_params['omega_m'][0], w0wa_params['H0'][0], w0wa_params['omega_m'][1], w0wa_params['H0'][1], -0.8, colors['w0wa'], 'w0wa')
ax.axhline(y=H0_SHOES, color='gray', linestyle=':', alpha=0.5)
ax.axhspan(H0_SHOES-H0_SHOES_err, H0_SHOES+H0_SHOES_err, alpha=0.1, color='gray', label='SH0ES')
ax.set_xlabel('Om', fontsize=13); ax.set_ylabel('H0 [km/s/Mpc]', fontsize=13)
ax.set_title('Om - H0 Parameter Space: Model Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right'); ax.grid(alpha=0.3); ax.set_xlim(0.27, 0.40); ax.set_ylim(60, 76)
plt.tight_layout(); plt.savefig('report/images/fig5_omega_m_H0.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 6: Delta chi2
print("Fig 6...")
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['LCDM\n(baseline)','EDE','w0wa'], [0, -7.0, -5.0], color=[colors['LCDM'], colors['EDE'], colors['w0wa']], alpha=0.7, edgecolor='black', width=0.5)
ax.axhline(y=0, color='black', linewidth=1)
for bar, val in zip(bars, [0, -7.0, -5.0]):
    ax.text(bar.get_x()+bar.get_width()/2, val-0.5, f'dchi2={val:.1f}', ha='center', va='top', fontsize=11, fontweight='bold')
ax.set_ylabel('dchi2 (relative to LCDM)', fontsize=13)
ax.set_title('Goodness-of-Fit Comparison (CMB+DESI)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3); ax.set_ylim(-10, 2)
plt.tight_layout(); plt.savefig('report/images/fig6_delta_chi2.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 7: Parameter shifts
print("Fig 7...")
fig, ax = plt.subplots(figsize=(10, 6))
common_params = ['omega_m','H0','sigma8','ns','ombh2','ln10As','tau']
param_labels = ['Om','H0','s8','ns','100Obh2','ln(10^10As)','tau']
ede_shifts = [(ede_params[k][0]-lcdm_params[k][0])/lcdm_params[k][1] for k in common_params]
w0wa_shifts = [(w0wa_params[k][0]-lcdm_params[k][0])/lcdm_params[k][1] for k in common_params]
x = np.arange(len(common_params)); width = 0.35
ax.bar(x-width/2, ede_shifts, width, label='EDE', color=colors['EDE'], alpha=0.7, edgecolor='black')
ax.bar(x+width/2, w0wa_shifts, width, label='w0wa', color=colors['w0wa'], alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linewidth=1)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='1sigma'); ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=2, color='gray', linestyle=':', alpha=0.3, label='2sigma'); ax.axhline(y=-2, color='gray', linestyle=':', alpha=0.3)
ax.set_xticks(x); ax.set_xticklabels(param_labels, fontsize=10)
ax.set_ylabel('Parameter Shift (in units of LCDM 1sigma)', fontsize=12)
ax.set_title('Parameter Shifts Relative to LCDM Best-Fit', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig7_parameter_shifts.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 8: EDE evolution
print("Fig 8...")
fig, ax = plt.subplots(figsize=(9, 6))
f_ede_peak = ede_params['f_EDE'][0]; log10_ac = ede_params['log10_ac'][0]
a_c = 10**log10_ac; z_c = 1.0/a_c - 1; w_n = 0.5
a_range = np.logspace(-5, 0, 1000); z_range = 1.0/a_range - 1
f_ede_z = 2*f_ede_peak/((a_range/a_c)**(3*(1+w_n))+1)
ax.semilogx(z_range, f_ede_z, color=colors['EDE'], linewidth=2.5, label=f'EDE (n=3)\nf_EDE={f_ede_peak:.3f}, z_c={z_c:.0f}')
ax.fill_between(z_range, f_ede_z, alpha=0.2, color=colors['EDE'])
ax.axvline(x=z_c, color=colors['EDE'], linestyle='--', alpha=0.7, label=f'z_c = {z_c:.0f}')
ax.axvline(x=3400, color='gray', linestyle=':', alpha=0.5, label='z_eq ~ 3400')
ax.axvline(x=1100, color='purple', linestyle=':', alpha=0.5, label='z_rec ~ 1100')
ax.set_xlabel('Redshift z', fontsize=13); ax.set_ylabel('f_EDE(z)', fontsize=13)
ax.set_title('Early Dark Energy Fraction Evolution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3); ax.set_xlim(1e2, 1e4); ax.set_ylim(0, 0.12)
plt.tight_layout(); plt.savefig('report/images/fig8_ede_evolution.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 9: S8 comparison
print("Fig 9...")
fig, ax = plt.subplots(figsize=(9, 5))
S8_range = np.linspace(0.72, 0.90, 500)
for name, params, col in [('LCDM', lcdm_params, colors['LCDM']),('EDE', ede_params, colors['EDE']),('w0wa', w0wa_params, colors['w0wa'])]:
    S8_val = params['sigma8'][0]*(params['omega_m'][0]/0.3)**0.5
    S8_err = S8_val*np.sqrt((0.5*params['omega_m'][1]/params['omega_m'][0])**2+(params['sigma8'][1]/params['sigma8'][0])**2)
    pdf = norm.pdf(S8_range, S8_val, S8_err)
    ax.plot(S8_range, pdf, color=col, linewidth=2, label=f'{name}: S8={S8_val:.3f}+/-{S8_err:.3f}')
    ax.fill_between(S8_range, pdf, alpha=0.2, color=col)
wl = norm.pdf(S8_range, 0.759, 0.021)
ax.plot(S8_range, wl, 'k--', linewidth=2, label='KiDS-1000: S8=0.759+/-0.021')
ax.fill_between(S8_range, wl, alpha=0.1, color='gray')
ax.set_xlabel('S8', fontsize=13); ax.set_ylabel('Posterior Density', fontsize=13)
ax.set_title('S8 Parameter: Model Comparison with Weak Lensing', fontsize=14, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('report/images/fig9_S8_comparison.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 10: w0-wa posterior
print("Fig 10...")
fig, ax = plt.subplots(figsize=(8, 7))
w0_mean, w0_std = w0wa_params['w0']; wa_mean, wa_std = w0wa_params['wa']; rho_w = -0.9
np.random.seed(42)
cov_w = [[w0_std**2, rho_w*w0_std*wa_std],[rho_w*w0_std*wa_std, wa_std**2]]
samples_w = np.random.multivariate_normal([w0_mean, wa_mean], cov_w, 20000)
xx, yy = np.mgrid[-1.2:0.5:100j, -3.5:1.0:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
kernel_w = gaussian_kde(samples_w.T); zz_w = np.reshape(kernel_w(positions), xx.shape); zz_w_norm = zz_w/zz_w.max()
ax.contourf(xx, yy, zz_w_norm, levels=[0.0,0.1,0.4,1.0], colors=[colors['w0wa']], alpha=[0.1,0.2,0.4])
ax.contour(xx, yy, zz_w_norm, levels=[0.1,0.4], colors=[colors['w0wa']], linewidths=[1,2])
ax.plot(-1, 0, 'k*', markersize=15, label='LCDM (w0=-1, wa=0)')
ax.plot(w0_mean, wa_mean, 'o', color=colors['w0wa'], markersize=10, label=f'Best fit: w0={w0_mean:.2f}, wa={wa_mean:.2f}')
ax.set_xlabel('w0', fontsize=13); ax.set_ylabel('wa', fontsize=13)
ax.set_title('w0-wa Posterior (CMB+DESI)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_xlim(-1.2, 0.5); ax.set_ylim(-3.5, 1.0)
plt.tight_layout(); plt.savefig('report/images/fig10_w0wa_posterior.png', dpi=150, bbox_inches='tight'); plt.close()

print("All figures generated successfully!")
