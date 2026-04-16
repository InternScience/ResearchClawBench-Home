import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Data from DESI_EDE_Repro_Data.txt
# Best-fit parameters and 1sigma errors
models = ['$\Lambda$CDM', 'EDE', '$w_0w_a$']

# Format: (mean, sigma)
omega_m = [0.3037, 0.2999, 0.353]
omega_m_err = [0.0037, 0.0038, 0.021]

H0 = [68.12, 70.9, 63.5]
H0_err = [0.28, 1.0, 1.9]

sigma8 = [0.8101, 0.8283, 0.780]
sigma8_err = [0.0055, 0.0093, 0.016]

# 1. Plot Parameter Constraints (H0, Omega_m, sigma8)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Omega_m
axes[0].errorbar(models, omega_m, yerr=omega_m_err, fmt='o', capsize=5, markersize=8, color='blue')
axes[0].set_ylabel('$\Omega_m$')
axes[0].set_title('Matter Density ($\Omega_m$)')
axes[0].grid(True, linestyle='--', alpha=0.7)

# H0
axes[1].errorbar(models, H0, yerr=H0_err, fmt='o', capsize=5, markersize=8, color='red')
axes[1].set_ylabel('$H_0$ [km/s/Mpc]')
axes[1].set_title('Hubble Constant ($H_0$)')
axes[1].axhline(y=73.04, color='gray', linestyle='--', label='SH0ES (local)')
axes[1].axhspan(73.04 - 1.04, 73.04 + 1.04, color='gray', alpha=0.2)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.7)

# sigma8
axes[2].errorbar(models, sigma8, yerr=sigma8_err, fmt='o', capsize=5, markersize=8, color='green')
axes[2].set_ylabel('$\sigma_8$')
axes[2].set_title('Structure Growth ($\sigma_8$)')
axes[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('report/images/parameter_constraints.png', dpi=300)
plt.close()

# 2. Plot DESI BAO data
desi_z = [0.295, 0.510, 0.700, 0.934, 1.100, 1.320, 2.330]
desi_dvrd = [-0.020, -0.015, -0.012, -0.010, -0.005, 0.000, 0.010]
desi_dvrd_err = [0.010, 0.008, 0.007, 0.006, 0.007, 0.008, 0.012]

desi_fap = [-0.01, 0.00, 0.01, 0.02, 0.02, 0.02, -0.03]
desi_fap_err = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].errorbar(desi_z, desi_dvrd, yerr=desi_dvrd_err, fmt='o', capsize=5, color='purple')
axes[0].axhline(y=0, color='k', linestyle='--')
axes[0].set_xlabel('Redshift $z$')
axes[0].set_ylabel('$\Delta(D_V/r_d)$')
axes[0].set_title('DESI BAO: Distance Scale')
axes[0].grid(True, linestyle='--', alpha=0.7)

axes[1].errorbar(desi_z, desi_fap, yerr=desi_fap_err, fmt='o', capsize=5, color='orange')
axes[1].axhline(y=0, color='k', linestyle='--')
axes[1].set_xlabel('Redshift $z$')
axes[1].set_ylabel('$\Delta F_{AP}$')
axes[1].set_title('DESI BAO: Alcock-Paczynski Effect')
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('report/images/desi_bao.png', dpi=300)
plt.close()

# 3. Plot Union3 SNe data
sne_z = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
sne_mu = [-0.08, -0.12, -0.10, -0.07, -0.05, -0.02, 0.00]
sne_mu_err = [0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05]

plt.figure(figsize=(8, 6))
plt.errorbar(sne_z, sne_mu, yerr=sne_mu_err, fmt='o', capsize=5, color='darkred')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Redshift $z$')
plt.ylabel('$\Delta\mu$ (Distance Modulus)')
plt.title('Union3 Supernovae')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('report/images/union3_sne.png', dpi=300)
plt.close()

print("Analysis complete. Figures saved.")
