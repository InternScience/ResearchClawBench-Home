import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from pathlib import Path

plt.style.use('default')
Path('report').mkdir(exist_ok=True)
Path('report/images').mkdir(exist_ok=True)

# Load params
with open('outputs/parameters.json') as f:
    params = json.load(f)

# 1. Parameter comparison for key params: omega_m, H0, sigma8
key_params = ['omega_m', 'H0', 'sigma8']
models = ['lcdm', 'ede', 'w0wa']
model_labels = ['ΛCDM', 'EDE', 'w₀wₐ']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, p in enumerate(key_params):
    means = [params[m][p]['mean'] for m in models]
    errs = [params[m][p]['sigma'] for m in models]
    axes[i].errorbar(range(3), means, yerr=errs, fmt='o', capsize=5)
    axes[i].set_xticks(range(3))
    axes[i].set_xticklabels(model_labels)
    axes[i].set_title(p.replace('_', ' ') + ' (CMB+DESI)')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylabel(p)
plt.suptitle('Model Comparison: Key Cosmological Parameters')
plt.tight_layout()
plt.savefig('report/images/param_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. BAO Δ(D_V / r_d)
df = pd.read_csv('outputs/bao_dvrd.csv')
plt.figure(figsize=(8, 5))
plt.errorbar(df.z, df.delta_dv_rd, yerr=df.error, fmt='o', capsize=5, color='blue', label='Data rel. fid.')
plt.axhline(0, color='k', ls='--', alpha=0.5)
plt.xlabel('Redshift z')
plt.ylabel(r'$\\Delta$(D_V / r_d)')
plt.title('DESI BAO residuals rel. fiducial')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('report/images/bao_dvrd.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. BAO ΔF_AP
df = pd.read_csv('outputs/bao_fap.csv')
plt.figure(figsize=(8, 5))
plt.errorbar(df.z, df.delta_fap, yerr=df.error, fmt='s', capsize=5, color='green')
plt.axhline(0, color='k', ls='--', alpha=0.5)
plt.xlabel('Redshift z')
plt.ylabel('Δ F_AP')
plt.title('DESI BAO ΔF_AP rel. fiducial')
plt.grid(True, alpha=0.3)
plt.savefig('report/images/bao_fap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. SNe Δμ
df = pd.read_csv('outputs/sne_mu.csv')
plt.figure(figsize=(8, 5))
plt.errorbar(df.z, df.delta_mu, yerr=df.error, fmt='^', capsize=5, color='red')
plt.axhline(0, color='k', ls='--', alpha=0.5)
plt.xlabel('Redshift z')
plt.ylabel('Δ μ')
plt.title('Union3 SNe Δμ rel. fiducial')
plt.grid(True, alpha=0.3)
plt.savefig('report/images/sne_mu.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. EDE posteriors
f_mean = params['ede']['f_EDE']['mean']
f_sig = params['ede']['f_EDE']['sigma']
logac_mean = params['ede']['log10_ac']['mean']
logac_sig = params['ede']['log10_ac']['sigma']

x_f = np.linspace(f_mean - 3*f_sig, f_mean + 3*f_sig, 200)
pdf_f = norm.pdf(x_f, f_mean, f_sig)
x_log = np.linspace(logac_mean - 3*logac_sig, logac_mean + 3*logac_sig, 200)
pdf_log = norm.pdf(x_log, logac_mean, logac_sig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(x_f, pdf_f, 'b-', lw=2)
axes[0].axvline(f_mean, color='r', ls='--', lw=2, label=f'{f_mean:.3f} ± {f_sig:.3f}')
axes[0].fill_between(x_f, pdf_f, alpha=0.3)
axes[0].set_xlabel('f_EDE')
axes[0].set_ylabel('PDF')
axes[0].set_title('EDE f_EDE posterior')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_log, pdf_log, 'g-', lw=2)
axes[1].axvline(logac_mean, color='r', ls='--', lw=2, label=f'{logac_mean:.3f} ± {logac_sig:.3f}')
axes[1].fill_between(x_log, pdf_log, alpha=0.3)
axes[1].set_xlabel('log₁₀ a_c')
axes[1].set_title('EDE log₁₀ a_c posterior')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/ede_posteriors.png', dpi=300, bbox_inches='tight')
plt.close()

# Comparison table
comparison_data = {
    'Model': ['ΛCDM', 'EDE', 'w₀wₐ'],
    'Ω_m': [f\"{params[m]['omega_m']['mean']:.4f} ± {params[m]['omega_m']['sigma']:.4f}\" for m in models],
    'H0 [km/s/Mpc]': [f\"{params[m]['H0']['mean']:.1f} ± {params[m]['H0']['sigma']:.1f}\" for m in models],
    'σ8': [f\"{params[m]['sigma8']['mean']:.4f} ± {params[m]['sigma8']['sigma']:.4f}\" for m in models]
}
comp_df = pd.DataFrame(comparison_data)
comp_df.to_csv('outputs/model_comparison.csv', index=False)

print('Figures and table generated.')