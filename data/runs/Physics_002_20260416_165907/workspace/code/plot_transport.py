import json
import matplotlib.pyplot as plt
import numpy as np

with open('outputs/xeb_results.json', 'r') as f:
    results = json.load(f)

plt.figure(figsize=(8, 6))

if 'transport_scan' in results:
    depths_trans = results['transport_scan']['depths']
    f_trans = results['transport_scan']['fidelities']
    err_trans = results['transport_scan']['errors']
    
    log_F_trans = np.log(f_trans)
    w_trans = f_trans / np.array(err_trans)
    p_trans, cov_trans = np.polyfit(depths_trans, log_F_trans, 1, w=w_trans, cov=True)
    
    d_fit = np.linspace(min(depths_trans), max(depths_trans), 100)
    F_fit_trans = np.exp(p_trans[1]) * np.exp(p_trans[0] * d_fit)
    
    plt.errorbar(depths_trans, f_trans, yerr=err_trans, fmt='D', color='purple', capsize=4, label='Transport 1QRB')
    plt.plot(d_fit, F_fit_trans, ':', color='purple', alpha=0.5, label=f'Transport Fit: $\propto e^{{{p_trans[0]:.3f} d}}$')

plt.yscale('log')
plt.xlabel('Circuit Depth ($d$)', fontsize=12)
plt.ylabel('Fidelity', fontsize=12)
plt.title('Transport 1QRB Fidelity vs Depth ($N=40$)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig('report/images/transport_fidelity.png', dpi=300)
plt.close()
