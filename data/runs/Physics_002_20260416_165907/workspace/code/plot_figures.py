import json
import matplotlib.pyplot as plt
import numpy as np

with open('outputs/xeb_results.json', 'r') as f:
    results = json.load(f)

# Plot 1: XEB vs Depth
plt.figure(figsize=(7, 5))
depths = results['depth_scan']['depths']
fidelities_d = results['depth_scan']['fidelities']
errors_d = results['depth_scan']['errors']

# Assuming an exponential decay model F = A * p^d
# We can fit a line to log(F) vs d
log_F = np.log(fidelities_d)
# Weights for the fit based on errors
weights = fidelities_d / np.array(errors_d)
p, cov = np.polyfit(depths, log_F, 1, w=weights, cov=True)
p_err = np.sqrt(np.diag(cov))

d_fit = np.linspace(min(depths), max(depths), 100)
F_fit = np.exp(p[1]) * np.exp(p[0] * d_fit)

plt.errorbar(depths, fidelities_d, yerr=errors_d, fmt='o', color='blue', capsize=4, label='Experimental XEB')
plt.plot(d_fit, F_fit, '--', color='red', label=f'Fit: $\propto e^{{{p[0]:.3f} d}}$')

plt.yscale('log')
plt.xlabel('Circuit Depth ($d$)', fontsize=12)
plt.ylabel('XEB Fidelity ($F_{XEB}$)', fontsize=12)
plt.title('Fidelity vs Circuit Depth ($N=40$)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig('report/images/fidelity_vs_depth.png', dpi=300)
plt.close()

# Plot 2: XEB vs N
plt.figure(figsize=(7, 5))
ns = results['n_scan']['Ns']
fidelities_n = results['n_scan']['fidelities']
errors_n = results['n_scan']['errors']

# Fit exponential decay
log_F_n = np.log(fidelities_n)
weights_n = fidelities_n / np.array(errors_n)
p_n, cov_n = np.polyfit(ns, log_F_n, 1, w=weights_n, cov=True)

n_fit = np.linspace(min(ns), max(ns), 100)
F_fit_n = np.exp(p_n[1]) * np.exp(p_n[0] * n_fit)

plt.errorbar(ns, fidelities_n, yerr=errors_n, fmt='s', color='green', capsize=4, label='Experimental XEB')
plt.plot(n_fit, F_fit_n, '--', color='orange', label=f'Fit: $\propto e^{{{p_n[0]:.3f} N}}$')

plt.yscale('log')
plt.xlabel('Number of Qubits ($N$)', fontsize=12)
plt.ylabel('XEB Fidelity ($F_{XEB}$)', fontsize=12)
plt.title('Fidelity vs Qubit Count ($d=12$)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig('report/images/fidelity_vs_n.png', dpi=300)
plt.close()

# Plot 3: Combined plot for visual comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.errorbar(depths, fidelities_d, yerr=errors_d, fmt='o', color='blue', capsize=4)
ax1.plot(d_fit, F_fit, '--', color='red')
ax1.set_yscale('log')
ax1.set_xlabel('Circuit Depth ($d$)', fontsize=12)
ax1.set_ylabel('XEB Fidelity ($F_{XEB}$)', fontsize=12)
ax1.set_title('Fixed N=40', fontsize=14)
ax1.grid(True, which="both", ls="--", alpha=0.6)

ax2.errorbar(ns, fidelities_n, yerr=errors_n, fmt='s', color='green', capsize=4)
ax2.plot(n_fit, F_fit_n, '--', color='orange')
ax2.set_yscale('log')
ax2.set_xlabel('Number of Qubits ($N$)', fontsize=12)
ax2.set_title('Fixed d=12', fontsize=14)
ax2.grid(True, which="both", ls="--", alpha=0.6)

plt.tight_layout()
plt.savefig('report/images/combined_fidelities.png', dpi=300)
plt.close()
