import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import superradiance as sr
import os

os.makedirs('report/images', exist_ok=True)

def load_data(filepath):
    data = np.loadtxt(filepath, comments='#')
    return data[:, 0], data[:, 1] # Mass, Spin

def compute_exclusion_contour(M_sol_range, a_star_range, mu_eV, tau_age_s):
    M_grid, a_grid = np.meshgrid(M_sol_range, a_star_range)
    excluded = np.zeros_like(M_grid)
    for i in range(M_grid.shape[0]):
        for j in range(M_grid.shape[1]):
            Gamma = sr.compute_gamma(M_grid[i,j], a_grid[i,j], mu_eV)
            if Gamma > 0 and (1.0 / Gamma) < tau_age_s:
                excluded[i,j] = 1
    return M_grid, a_grid, excluded

# M33 X-7
m33_masses, m33_spins = load_data('data/M33_X-7_samples.dat')
plt.figure(figsize=(8, 6))
sns.kdeplot(x=m33_masses, y=m33_spins, cmap="Blues", fill=True, alpha=0.8)

M_range = np.linspace(10, 14, 50)
a_range = np.linspace(0.5, 0.99, 50)
mu_test = 2e-12
M_grid, a_grid, excluded = compute_exclusion_contour(M_range, a_range, mu_test, 1e6 * 3.154e7)
plt.contour(M_grid, a_grid, excluded, levels=[0.5], colors='red', linestyles='dashed')

plt.xlabel(r'Black Hole Mass $M$ [$M_\odot$]')
plt.ylabel(r'Dimensionless Spin $a^*$')
plt.title(f'M33 X-7 Posterior & Exclusion Contour ($\mu = {mu_test}$ eV)')
plt.tight_layout()
plt.savefig('report/images/m33_posterior.png')

# IRAS 09149-6206
iras_masses, iras_spins = load_data('data/IRAS_09149-6206_samples.dat')
plt.figure(figsize=(8, 6))
sns.kdeplot(x=iras_masses, y=iras_spins, cmap="Oranges", fill=True, alpha=0.8)

M_range_iras = np.linspace(1e7, 3e8, 50)
a_range_iras = np.linspace(0.8, 0.99, 50)
mu_test_iras = 1e-18
M_grid_iras, a_grid_iras, excluded_iras = compute_exclusion_contour(M_range_iras, a_range_iras, mu_test_iras, 1e7 * 3.154e7)
plt.contour(M_grid_iras, a_grid_iras, excluded_iras, levels=[0.5], colors='red', linestyles='dashed')

plt.xlabel(r'Black Hole Mass $M$ [$M_\odot$]')
plt.ylabel(r'Dimensionless Spin $a^*$')
plt.title(f'IRAS 09149-6206 Posterior & Exclusion Contour ($\mu = {mu_test_iras}$ eV)')
plt.tight_layout()
plt.savefig('report/images/iras_posterior.png')

