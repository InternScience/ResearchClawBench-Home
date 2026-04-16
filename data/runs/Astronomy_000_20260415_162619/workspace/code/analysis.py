import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import superradiance as sr
from tqdm import tqdm
import os

# Create outputs and images directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

def load_data(filepath):
    data = np.loadtxt(filepath, comments='#')
    return data[:, 0], data[:, 1] # Mass, Spin

def compute_exclusion_prob(masses, spins, mu_eV, tau_age_s):
    """
    Compute the exclusion probability for a given boson mass mu_eV.
    Exclusion condition: tau_SR < tau_age
    """
    excluded_count = 0
    N = len(masses)
    for M, a in zip(masses, spins):
        Gamma = sr.compute_gamma(M, a, mu_eV, l=1, m=1, n=0)
        if Gamma > 0:
            tau_SR = 1.0 / Gamma
            if tau_SR < tau_age_s:
                excluded_count += 1
    return excluded_count / N

# Analysis for M33 X-7 (Stellar-mass BH)
m33_masses, m33_spins = load_data('data/M33_X-7_samples.dat')
# Typical age for a high-mass X-ray binary like M33 X-7 is ~ 10^6 years
tau_age_m33_yr = 1e6
tau_age_m33_s = tau_age_m33_yr * 365.25 * 24 * 3600

mu_grid_m33 = np.logspace(-13, -10, 50)
prob_m33 = []
for mu in tqdm(mu_grid_m33, desc="M33 X-7"):
    p = compute_exclusion_prob(m33_masses, m33_spins, mu, tau_age_m33_s)
    prob_m33.append(p)

plt.figure(figsize=(8, 6))
plt.plot(mu_grid_m33, prob_m33, lw=2, color='blue')
plt.xscale('log')
plt.xlabel(r'Boson Mass $\mu$ [eV]')
plt.ylabel('Exclusion Probability')
plt.title('M33 X-7: Ultralight Boson Exclusion')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('report/images/m33_exclusion.png')

# Analysis for IRAS 09149-6206 (Supermassive BH)
iras_masses, iras_spins = load_data('data/IRAS_09149-6206_samples.dat')
# Typical age/accretion timescale for an AGN is ~ 10^7 - 10^8 years. Let's use 10^7.
tau_age_iras_yr = 1e7
tau_age_iras_s = tau_age_iras_yr * 365.25 * 24 * 3600

mu_grid_iras = np.logspace(-20, -17, 50)
prob_iras = []
for mu in tqdm(mu_grid_iras, desc="IRAS 09149"):
    p = compute_exclusion_prob(iras_masses, iras_spins, mu, tau_age_iras_s)
    prob_iras.append(p)

plt.figure(figsize=(8, 6))
plt.plot(mu_grid_iras, prob_iras, lw=2, color='red')
plt.xscale('log')
plt.xlabel(r'Boson Mass $\mu$ [eV]')
plt.ylabel('Exclusion Probability')
plt.title('IRAS 09149-6206: Ultralight Boson Exclusion')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('report/images/iras_exclusion.png')

# Save intermediate results
np.savez('outputs/exclusion_results.npz', 
         mu_grid_m33=mu_grid_m33, prob_m33=prob_m33,
         mu_grid_iras=mu_grid_iras, prob_iras=prob_iras)

