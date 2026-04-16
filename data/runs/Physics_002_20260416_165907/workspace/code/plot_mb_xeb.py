import json
import matplotlib.pyplot as plt
import numpy as np

with open('outputs/xeb_results.json', 'r') as f:
    results = json.load(f)

plt.figure(figsize=(8, 6))

# Plot XEB
depths_xeb = results['depth_scan']['depths']
f_xeb = results['depth_scan']['fidelities']
err_xeb = results['depth_scan']['errors']

log_F_xeb = np.log(f_xeb)
w_xeb = f_xeb / np.array(err_xeb)
p_xeb, cov_xeb = np.polyfit(depths_xeb, log_F_xeb, 1, w=w_xeb, cov=True)
d_fit = np.linspace(min(depths_xeb), max(depths_xeb), 100)
F_fit_xeb = np.exp(p_xeb[1]) * np.exp(p_xeb[0] * d_fit)

plt.errorbar(depths_xeb, f_xeb, yerr=err_xeb, fmt='o', color='blue', capsize=4, label='XEB (Experimental)')
plt.plot(d_fit, F_fit_xeb, '--', color='blue', alpha=0.5, label=f'XEB Fit: $\propto e^{{{p_xeb[0]:.3f} d}}$')

# Plot MB
if 'mb_scan' in results:
    depths_mb = results['mb_scan']['depths']
    f_mb = results['mb_scan']['fidelities']
    err_mb = results['mb_scan']['errors']
    
    log_F_mb = np.log(f_mb)
    w_mb = f_mb / np.array(err_mb)
    p_mb, cov_mb = np.polyfit(depths_mb, log_F_mb, 1, w=w_mb, cov=True)
    F_fit_mb = np.exp(p_mb[1]) * np.exp(p_mb[0] * d_fit)
    
    plt.errorbar(depths_mb, f_mb, yerr=err_mb, fmt='s', color='green', capsize=4, label='Mirror Benchmarking (MB)')
    plt.plot(d_fit, F_fit_mb, '-.', color='green', alpha=0.5, label=f'MB Fit: $\propto e^{{{p_mb[0]:.3f} d}}$')

plt.yscale('log')
plt.xlabel('Circuit Depth ($d$)', fontsize=12)
plt.ylabel('Fidelity', fontsize=12)
plt.title('Comparison of XEB and Mirror Benchmarking ($N=40$)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig('report/images/xeb_vs_mb.png', dpi=300)
plt.close()
