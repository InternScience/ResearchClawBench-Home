"""
Generate publication-quality figures for the VOS framework report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import os

# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.figsize': (8, 5),
})
sns.set_style("whitegrid")
sns.set_palette("deep")

os.makedirs('report/images', exist_ok=True)

# Load data
conv = np.load('outputs/convergence_histories.npz')
sweep = np.load('outputs/parameter_sweep.npz', allow_pickle=True)
solutions = np.load('outputs/solutions.npz')
with open('outputs/metrics.json') as f:
    metrics = json.load(f)

print("Data loaded successfully.")
print(f"Reference objective: {metrics['reference_obj']:.8f}")
f_ref = metrics['reference_obj']

# ============================================================
# FIGURE 1: Convergence Comparison - All Methods
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Objective vs Iterations (log-log)
ax = axes[0]
ista_iter = conv['ista_iter']
fista_iter = conv['fista_iter']
admm_iter = conv['admm_iter']

ax.loglog(ista_iter, np.array(conv['ista_obj']) - f_ref + 1e-16, 
          '-.', linewidth=1.8, label='ISTA (Proximal Gradient)', alpha=0.8)
ax.loglog(fista_iter, np.array(conv['fista_obj']) - f_ref + 1e-16, 
          '-', linewidth=2.2, label='FISTA (Nesterov Acceleration)', alpha=0.9)
ax.loglog(admm_iter, np.array(conv['admm_obj']) - f_ref + 1e-16, 
          '--', linewidth=2.0, label='ADMM', alpha=0.85)

# Theoretical O(1/k^2) reference line
k_ref = np.array([1, 10, 100, 1000, 3000])
ax.loglog(k_ref, 100/k_ref**2, ':', color='gray', linewidth=1.5, alpha=0.7, label=r'$O(1/k^2)$ reference')

ax.set_xlabel('Iteration k')
ax.set_ylabel(r'$f(x_k) - f^*$')
ax.set_title('Convergence: Objective Suboptimality')
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)

# Subplot 2: Objective vs Time (semilog)
ax = axes[1]
ista_time = conv['ista_iter']  # we don't have direct time, derive from iteration
fista_time = conv['fista_iter']
admm_time = conv['admm_iter']

ax.semilogy(ista_time, np.array(conv['ista_obj']) - f_ref + 1e-16, 
           '-.', linewidth=2.0, label='ISTA')
ax.semilogy(fista_time, np.array(conv['fista_obj']) - f_ref + 1e-16, 
           '-', linewidth=2.5, label='FISTA')
ax.semilogy(admm_time, np.array(conv['admm_obj']) - f_ref + 1e-16, 
           '--', linewidth=2.0, label='ADMM')

ax.set_xlabel('Iteration k')
ax.set_ylabel(r'$f(x_k) - f^*$')
ax.set_title('Convergence: Semilog Scale')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig1_convergence_comparison.png')
plt.close()
print("Figure 1 saved.")

# ============================================================
# FIGURE 2: Generalized VOS - Damping Parameter Sweep
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

# Load sweep data
sweep_data = {k: sweep[k] for k in sweep.files}
def parse_r(key):
    val = key.replace('r_', '')
    return float(val)

r_values = sorted([parse_r(k) for k in sweep_data.keys()])
print(f"r_values: {r_values}")

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(r_values)))

for r, c in zip(r_values, colors):
    obj_vals = sweep_data[f'r_{str(r).rstrip("0").rstrip(".")}']
    gap = np.array([float(v) for v in obj_vals]) - f_ref
    gap = np.maximum(gap, 1e-16)
    k_vals = np.arange(0, len(gap) * 20, 20)
    ax.loglog(k_vals, gap, color=c, linewidth=1.8, 
             label=f'r = {r:.1f}' + (' (Nesterov)' if abs(r-3.0) < 0.01 else ''),
             linestyle='-' if abs(r-3.0) < 0.01 else '--')

# Theoretical O(1/k^2)
k_theory = np.array([10, 100, 1000])
ax.loglog(k_theory, 100/k_theory**2, ':', color='black', linewidth=1.5, alpha=0.6, label=r'$O(1/k^2)$')

ax.set_xlabel('Iteration k')
ax.set_ylabel(r'$f(x_k) - f^*$')
ax.set_title(r'Generalized VOS: Effect of Damping Coefficient $r$ in $\ddot{X} + \frac{r}{t}\dot{X} + \nabla f(X) = 0$')
ax.legend(fontsize=9, ncol=2, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xlim([10, 2000])

plt.tight_layout()
plt.savefig('report/images/fig2_damping_sweep.png')
plt.close()
print("Figure 2 saved.")

# ============================================================
# FIGURE 3: Lyapunov Function Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# FISTA Lyapunov
ax = axes[0]
fista_k = conv['fista_lyap_k']
fista_E = conv['fista_lyap_E']
ax.semilogy(fista_k, fista_E, '-', linewidth=2.0, color='C0', label=r'$\mathcal{E}_k$ (FISTA)')
ax.axhline(y=np.mean(fista_E[-10:]), color='C0', linestyle=':', alpha=0.5, label='Asymptotic value')
ax.set_xlabel('Iteration k')
ax.set_ylabel('Lyapunov Function Value')
ax.set_title(r'Nesterov/FISTA: $\mathcal{E}_k = t_k^2(f(x_k)-f^*) + 2\|z_k-x^*\|^2$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ADMM Lyapunov
ax = axes[1]
admm_k = conv['admm_lyap_k']
admm_E = conv['admm_lyap_E']
ax.semilogy(admm_k, admm_E, '-', linewidth=2.0, color='C2', label=r'$\mathcal{E}_k$ (ADMM)')
ax.set_xlabel('Iteration k')
ax.set_ylabel('Lyapunov Function Value')
ax.set_title(r'ADMM: $\mathcal{E}_k = \|x_k-x^*\|^2 + \rho\|z_k-z^*\|^2$')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig3_lyapunov_analysis.png')
plt.close()
print("Figure 3 saved.")

# ============================================================
# FIGURE 4: Solution Recovery Comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x_true = solutions['x_true']
x_ista = solutions['x_ista']
x_fista = solutions['x_fista']
x_admm = solutions['x_admm']

# True coefficients
ax = axes[0, 0]
ax.stem(range(len(x_true)), x_true, markerfmt='C0.', basefmt='C0-', linefmt='C0-')
ax.set_xlabel('Coefficient Index')
ax.set_ylabel('Value')
ax.set_title(f'Ground Truth (Sparsity: {np.count_nonzero(x_true)}/{len(x_true)})')
ax.set_xlim([-5, len(x_true)+5])

# FISTA recovery
ax = axes[0, 1]
ax.stem(range(len(x_fista)), x_fista, markerfmt='C1.', basefmt='C1-', linefmt='C1-')
ax.set_xlabel('Coefficient Index')
ax.set_ylabel('Value')
ax.set_title(f'FISTA Recovery')
ax.set_xlim([-5, len(x_true)+5])

# ADMM recovery
ax = axes[1, 0]
ax.stem(range(len(x_admm)), x_admm, markerfmt='C2.', basefmt='C2-', linefmt='C2-')
ax.set_xlabel('Coefficient Index')
ax.set_ylabel('Value')
ax.set_title(f'ADMM Recovery')
ax.set_xlim([-5, len(x_true)+5])

# Scatter: True vs Recovered (FISTA)
ax = axes[1, 1]
ax.scatter(x_true, x_fista, s=3, alpha=0.5, color='C1', label='FISTA')
ax.scatter(x_true, x_admm, s=3, alpha=0.5, color='C2', label='ADMM')
ax.plot([-2.5, 2.5], [-2.5, 2.5], 'k--', linewidth=0.8, alpha=0.5, label='Ideal')
ax.set_xlabel('True Coefficient')
ax.set_ylabel('Recovered Coefficient')
ax.set_title('True vs Recovered Coefficients')
ax.legend(fontsize=9)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('report/images/fig4_solution_recovery.png')
plt.close()
print("Figure 4 saved.")

# ============================================================
# FIGURE 5: VOS Continuous-Time Trajectory
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

t_vals = np.array(conv['vos_t'])
obj_vals = np.array(conv['vos_obj'])
lyap_vals = np.array(conv['vos_lyap'])

# Objective evolution
ax = axes[0]
ax.semilogy(t_vals, obj_vals - f_ref + 1e-16, '-', linewidth=2.0, color='C3')
# Theoretical O(1/t^2) 
t_theory = np.array([0.1, 1, 10, 50])
ax.semilogy(t_theory, 10/t_theory**2, ':', color='gray', linewidth=1.5, alpha=0.7, label=r'$O(1/t^2)$')
ax.set_xlabel('Time t')
ax.set_ylabel(r'$f(X(t)) - f^*$')
ax.set_title('Continuous-Time VOS: Objective Convergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Lyapunov function
ax = axes[1]
ax.plot(t_vals, lyap_vals, '-', linewidth=2.0, color='C3')
ax.set_xlabel('Time t')
ax.set_ylabel(r'$\mathcal{E}(t)$')
ax.set_title(r'Continuous-Time Lyapunov: $\mathcal{E}(t) = t^2(f(X)-f^*) + 2\|X + \frac{t}{2}\dot{X} - x^*\|^2$')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig5_vos_continuous.png')
plt.close()
print("Figure 5 saved.")

# ============================================================
# FIGURE 6: ADMM Primal and Dual Residual Convergence
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

admm_k = conv['admm_iter']
primal_res = np.array(conv['admm_primal'])
dual_res = np.array(conv['admm_dual'])

ax.semilogy(admm_k, primal_res, '-', linewidth=2.0, label='Primal residual $\\|x^k - z^k\\|$', color='C2')
ax.semilogy(admm_k, dual_res, '--', linewidth=2.0, label='Dual residual $\\rho\\|z^k - z^{k-1}\\|$', color='C4')
ax.set_xlabel('Iteration k')
ax.set_ylabel('Residual')
ax.set_title('ADMM: Primal and Dual Residual Convergence')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig6_admm_residuals.png')
plt.close()
print("Figure 6 saved.")

# ============================================================
# FIGURE 7: Phase Transition at r=3
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

# For each r, compute the scaled error k^2 * (f(x_k) - f*)
# to see if it stays bounded (convergence) or grows (divergence)
r_vals_plot = [1, 2, 2.5, 3, 4, 5]
colors_plot = plt.cm.plasma(np.linspace(0.1, 0.9, len(r_vals_plot)))

for r, c in zip(r_vals_plot, colors_plot):
    key = f'r_{str(r).rstrip("0").rstrip(".")}'
    obj_vals = np.array([float(v) for v in sweep_data[key]])
    gap = obj_vals - f_ref
    gap = np.maximum(gap, 1e-16)
    k_vals = np.arange(20, (len(gap)) * 20 + 20, 20)
    scaled = k_vals**2 * gap
    
    ax.semilogy(k_vals, scaled, color=c, linewidth=1.8,
               label=f'r = {r:.1f}',
               linestyle='-' if r >= 3.0 else '--')

ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('Iteration k')
ax.set_ylabel(r'$k^2 \cdot (f(x_k) - f^*)$')
ax.set_title(r'Phase Transition: $k^2 \cdot$ Suboptimality for Generalized VOS ($r \geq 3$ ensures $O(1/k^2)$)')
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig7_phase_transition.png')
plt.close()
print("Figure 7 saved.")

# ============================================================
# Summary statistics table for report
# ============================================================
print("\n" + "="*60)
print("KEY METRICS SUMMARY")
print("="*60)
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6f}")
    else:
        print(f"  {k}: {v}")

print("\nAll figures saved to report/images/")
