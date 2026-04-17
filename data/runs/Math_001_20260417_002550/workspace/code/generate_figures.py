"""
Generate all figures for the VOS Framework report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os

# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
})

base_dir = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Math_001_20260417_002550'
output_dir = os.path.join(base_dir, 'outputs')
img_dir = os.path.join(base_dir, 'report', 'images')
os.makedirs(img_dir, exist_ok=True)

# Load data
histories = np.load(os.path.join(output_dir, 'convergence_histories.npy'), allow_pickle=True).item()
lyapunov_data = np.load(os.path.join(output_dir, 'lyapunov_data.npy'), allow_pickle=True).item()
ode_data = np.load(os.path.join(output_dir, 'ode_trajectories.npy'), allow_pickle=True).item()
lambda_data = np.load(os.path.join(output_dir, 'lambda_sweep.npy'), allow_pickle=True).item()

with open(os.path.join(output_dir, 'algorithm_comparison.json'), 'r') as f:
    results = json.load(f)
with open(os.path.join(output_dir, 'problem_params.json'), 'r') as f:
    params = json.load(f)

f_ref = params['reference_obj']

# ============================================================
# Figure 1: Convergence Comparison (Main Result)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

colors = {
    'GD': '#1f77b4',
    'NAG': '#ff7f0e',
    'NAG-Restart': '#2ca02c',
    'ADMM': '#d62728',
    'VOS-NAG': '#9467bd',
    'VOS-ADMM': '#8c564b'
}
linestyles = {
    'GD': '-',
    'NAG': '-',
    'NAG-Restart': '--',
    'ADMM': '-',
    'VOS-NAG': '--',
    'VOS-ADMM': '--'
}

# Panel (a): Objective value vs iterations
ax = axes[0]
for name in ['GD', 'NAG', 'NAG-Restart', 'ADMM', 'VOS-NAG']:
    obj = histories[name]['obj']
    ax.semilogy(range(len(obj)), obj, label=name, color=colors[name], 
                linestyle=linestyles[name], alpha=0.9)
ax.set_xlabel('Iterations')
ax.set_ylabel('Objective Value $f(x_k) + g(x_k)$')
ax.set_title('(a) Objective Value Convergence')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Panel (b): Optimality gap (log scale)
ax = axes[1]
for name in ['GD', 'NAG', 'NAG-Restart', 'ADMM', 'VOS-NAG']:
    obj = histories[name]['obj']
    gaps = [max(o - f_ref, 1e-16) for o in obj]
    ax.semilogy(range(len(gaps)), gaps, label=name, color=colors[name],
                linestyle=linestyles[name], alpha=0.9)
ax.set_xlabel('Iterations')
ax.set_ylabel('Optimality Gap $f(x_k) - f^*$')
ax.set_title('(b) Optimality Gap')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=1e-14)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig1_convergence_comparison.png'))
plt.close()
print("Figure 1 saved.")

# ============================================================
# Figure 2: Lyapunov Function Decay
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): Lyapunov for GD
ax = axes[0]
lyap_gd = lyapunov_data['lyap_gd']
obj_gap_gd = lyapunov_data['obj_gap_gd']
iters = range(len(lyap_gd))
ax.semilogy(iters, lyap_gd, 'b-', label='$V(x_k) = \\|x_k - x^*\\|^2$ (GD)', linewidth=2)
ax.semilogy(range(len(obj_gap_gd)), [max(g, 1e-30) for g in obj_gap_gd], 
            'b--', label='$f(x_k) - f^*$ (GD)', linewidth=1.5, alpha=0.7)

# Theoretical rate for GD
kappa_q = lyapunov_data['kappa']
rate_gd = (kappa_q - 1) / (kappa_q + 1)
theoretical_gd = [lyap_gd[0] * rate_gd**(2*k) for k in range(len(lyap_gd))]
ax.semilogy(iters, theoretical_gd, 'b:', label=f'Theoretical $(\\frac{{\\kappa-1}}{{\\kappa+1}})^{{2k}}$', 
            linewidth=1.5, alpha=0.6)

ax.set_xlabel('Iterations')
ax.set_ylabel('Lyapunov Value / Objective Gap')
ax.set_title('(a) Gradient Descent Lyapunov Decay')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Lyapunov for NAG
ax = axes[1]
lyap_nag = lyapunov_data['lyap_nag']
obj_gap_nag = lyapunov_data['obj_gap_nag']
iters_nag = range(len(lyap_nag))
ax.semilogy(iters_nag, lyap_nag, 'r-', label='$E(x_k, v_k)$ (NAG Lyapunov)', linewidth=2)
ax.semilogy(range(len(obj_gap_nag)), [max(g, 1e-30) for g in obj_gap_nag],
            'r--', label='$f(x_k) - f^*$ (NAG)', linewidth=1.5, alpha=0.7)

# Theoretical rate for NAG
rate_nag = (np.sqrt(kappa_q) - 1) / (np.sqrt(kappa_q) + 1)
theoretical_nag = [lyap_nag[0] * rate_nag**(2*k) for k in range(len(lyap_nag))]
ax.semilogy(iters_nag, theoretical_nag, 'r:', 
            label=f'Theoretical $(\\frac{{\\sqrt{{\\kappa}}-1}}{{\\sqrt{{\\kappa}}+1}})^{{2k}}$',
            linewidth=1.5, alpha=0.6)

ax.set_xlabel('Iterations')
ax.set_ylabel('Lyapunov Value / Objective Gap')
ax.set_title('(b) Nesterov Accelerated Lyapunov Decay')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig2_lyapunov_decay.png'))
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Solution Recovery Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Load actual solutions by re-running briefly
import sys
sys.path.insert(0, os.path.join(base_dir, 'code'))
from vos_framework import LassoProblem, load_data, gradient_descent, nesterov_accelerated, admm_lasso, vos_nesterov

A, b, x_true, meta = load_data(os.path.join(base_dir, 'data', 'complex_optimization_data.npy'))
lam = 0.1 * np.max(np.abs(A.T @ b))
prob = LassoProblem(A, b, lam)
x0 = np.zeros(prob.n)

x_gd, _ = gradient_descent(prob, x0, max_iter=2000)
x_nag, _ = nesterov_accelerated(prob, x0, max_iter=2000)
x_admm, _ = admm_lasso(prob, x0, rho=1.0, max_iter=2000)

# Panel (a): True vs recovered (NAG)
ax = axes[0]
idx = np.argsort(np.abs(x_true))[::-1][:200]
ax.scatter(x_true[idx], x_nag[idx], s=10, alpha=0.6, c='#ff7f0e', label='NAG')
ax.scatter(x_true[idx], x_gd[idx], s=10, alpha=0.4, c='#1f77b4', label='GD')
lims = [min(x_true[idx].min(), x_nag[idx].min()) - 0.2, 
        max(x_true[idx].max(), x_nag[idx].max()) + 0.2]
ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Perfect recovery')
ax.set_xlabel('True $x^*$')
ax.set_ylabel('Recovered $\\hat{x}$')
ax.set_title('(a) Solution Recovery (Top 200 components)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Sparsity pattern
ax = axes[1]
support_true = np.abs(x_true) > 1e-6
support_nag = np.abs(x_nag) > 1e-6
support_admm = np.abs(x_admm) > 1e-6

# Show first 200 components
n_show = 200
ax.bar(range(n_show), np.abs(x_true[:n_show]), alpha=0.4, label='True', color='gray', width=1.0)
ax.bar(range(n_show), np.abs(x_nag[:n_show]), alpha=0.5, label='NAG', color='#ff7f0e', width=0.6)
ax.set_xlabel('Component Index')
ax.set_ylabel('$|x_i|$')
ax.set_title('(b) Sparsity Pattern (First 200 components)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (c): Bar chart of recovery errors
ax = axes[2]
names = list(results.keys())
errors = [results[n]['recovery_error'] for n in names]
colors_bar = [colors.get(n, '#333333') for n in names]
bars = ax.bar(range(len(names)), errors, color=colors_bar, alpha=0.8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_ylabel('$\\|\\hat{x} - x_{true}\\|_2$')
ax.set_title('(c) Recovery Error Comparison')
ax.grid(True, alpha=0.3, axis='y')
for bar, err in zip(bars, errors):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{err:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig3_solution_recovery.png'))
plt.close()
print("Figure 3 saved.")

# ============================================================
# Figure 4: ODE Trajectory Visualization (2D)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): Nesterov ODE trajectory
ax = axes[0]
ode_x = np.array(ode_data['nesterov_ode']['x'])
ax.plot(ode_x[:, 0], ode_x[:, 1], 'b-', linewidth=1.5, alpha=0.8, label='Nesterov ODE ($r=3$)')

# Add contours
x1_range = np.linspace(-0.5, 1.2, 100)
x2_range = np.linspace(-0.5, 1.2, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = X1**2 + 0.25 * X2**2  # f = x1^2 + 0.25*x2^2
ax.contour(X1, X2, Z, levels=15, alpha=0.3, colors='gray')
ax.plot(0, 0, 'r*', markersize=15, label='$x^*$')
ax.plot(1, 1, 'go', markersize=10, label='$x_0$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('(a) Nesterov ODE Trajectory ($\\ddot{X} + \\frac{3}{t}\\dot{X} + \\nabla f = 0$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Strongly convex ODE trajectory
ax = axes[1]
ode_sc_x = np.array(ode_data['strongly_convex_ode']['x'])
ax.plot(ode_sc_x[:, 0], ode_sc_x[:, 1], 'r-', linewidth=1.5, alpha=0.8, 
        label='Strongly Convex ODE')
ax.contour(X1, X2, Z, levels=15, alpha=0.3, colors='gray')
ax.plot(0, 0, 'r*', markersize=15, label='$x^*$')
ax.plot(1, 1, 'go', markersize=10, label='$x_0$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('(b) Strongly Convex ODE ($\\ddot{X} + 2\\sqrt{\\mu}\\dot{X} + \\nabla f = 0$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig4_ode_trajectories.png'))
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: Convergence Rate Analysis (Log-Log)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): Early convergence (first 200 iterations)
ax = axes[0]
n_early = 200
for name in ['GD', 'NAG', 'NAG-Restart', 'ADMM', 'VOS-NAG']:
    obj = histories[name]['obj'][:n_early]
    gaps = [max(o - f_ref, 1e-16) for o in obj]
    ax.semilogy(range(len(gaps)), gaps, label=name, color=colors[name],
                linestyle=linestyles[name])

# Add theoretical rates
iters_th = np.arange(1, n_early)
# O(1/k) for GD
ax.semilogy(iters_th, gaps[1] * 1.0/iters_th, 'k:', alpha=0.3, linewidth=1, label='$O(1/k)$')
# O(1/k^2) for NAG
ax.semilogy(iters_th, gaps[1] * 1.0/iters_th**2, 'k--', alpha=0.3, linewidth=1, label='$O(1/k^2)$')

ax.set_xlabel('Iterations')
ax.set_ylabel('Optimality Gap')
ax.set_title('(a) Early Convergence (First 200 Iterations)')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel (b): Log-log convergence rate
ax = axes[1]
for name in ['GD', 'NAG', 'NAG-Restart', 'ADMM', 'VOS-NAG']:
    obj = histories[name]['obj']
    gaps = [max(o - f_ref, 1e-16) for o in obj[1:]]  # skip first
    if len(gaps) > 1:
        # Compute per-iteration convergence rate
        rates = []
        for i in range(1, min(len(gaps), 500)):
            if gaps[i] > 1e-15 and gaps[i-1] > 1e-15:
                rates.append(gaps[i] / gaps[i-1])
        if rates:
            # Smooth with moving average
            window = 20
            if len(rates) > window:
                smoothed = np.convolve(rates, np.ones(window)/window, mode='valid')
                ax.plot(range(len(smoothed)), smoothed, label=name, color=colors[name],
                        linestyle=linestyles[name])

ax.set_xlabel('Iterations')
ax.set_ylabel('Per-Iteration Rate $f_{k+1}/f_k$')
ax.set_title('(b) Per-Iteration Convergence Rate')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.8, 1.01)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig5_convergence_rates.png'))
plt.close()
print("Figure 5 saved.")

# ============================================================
# Figure 6: Regularization Effect
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

lambdas_frac = sorted([float(k) for k in lambda_data.keys()])
recovery_errors = [lambda_data[str(l)]['recovery_error'] for l in lambdas_frac]
sparsities = [lambda_data[str(l)]['sparsity'] for l in lambdas_frac]
final_objs = [lambda_data[str(l)]['final_obj'] for l in lambdas_frac]

# Panel (a): Recovery error vs lambda
ax = axes[0]
ax.semilogx(lambdas_frac, recovery_errors, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('$\\lambda / \\lambda_{max}$')
ax.set_ylabel('$\\|\\hat{x} - x_{true}\\|_2$')
ax.set_title('(a) Recovery Error vs. Regularization')
ax.grid(True, alpha=0.3)

# Panel (b): Sparsity vs lambda
ax = axes[1]
ax.semilogx(lambdas_frac, sparsities, 'rs-', linewidth=2, markersize=8)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='True sparsity (100)')
ax.set_xlabel('$\\lambda / \\lambda_{max}$')
ax.set_ylabel('Number of nonzero components')
ax.set_title('(b) Sparsity vs. Regularization')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel (c): Convergence curves for different lambdas
ax = axes[2]
cmap = plt.cm.viridis
for i, lam_frac in enumerate(lambdas_frac):
    obj_hist = lambda_data[str(lam_frac)]['obj_history']
    color = cmap(i / len(lambdas_frac))
    ax.semilogy(range(min(len(obj_hist), 500)), obj_hist[:500], 
                color=color, label=f'$\\lambda/\\lambda_{{max}}$={lam_frac}')
ax.set_xlabel('Iterations')
ax.set_ylabel('Objective Value')
ax.set_title('(c) Convergence for Different $\\lambda$')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig6_regularization_effect.png'))
plt.close()
print("Figure 6 saved.")

# ============================================================
# Figure 7: VOS Framework Diagram (Conceptual)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(5, 7.5, 'Unified VOS Framework', fontsize=18, fontweight='bold',
        ha='center', va='center')

# Central ODE box
rect_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2)
ax.text(5, 5.5, 'Continuous-Time ODE\n$\\ddot{X} + \\gamma(t)\\dot{X} + \\nabla f(X) = 0$',
        fontsize=13, ha='center', va='center', bbox=rect_props)

# Variable Splitting
rect_vs = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='darkorange', linewidth=2)
ax.text(2.5, 3.5, 'Variable Splitting\n$X\' = V$\n$V\' = -\\gamma V - \\nabla f$',
        fontsize=11, ha='center', va='center', bbox=rect_vs)

# Operator Splitting
rect_os = dict(boxstyle='round,pad=0.4', facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
ax.text(7.5, 3.5, 'Operator Splitting\n$0 \\in \\partial f(x) + \\partial g(z)$\n$x = z$',
        fontsize=11, ha='center', va='center', bbox=rect_os)

# Nesterov
rect_n = dict(boxstyle='round,pad=0.3', facecolor='#FFD0D0', edgecolor='red', linewidth=2)
ax.text(2.5, 1.5, "Nesterov's Accelerated\nGradient Method\n$\\gamma(t) = r/t,\\ r \\geq 3$",
        fontsize=10, ha='center', va='center', bbox=rect_n)

# ADMM
rect_a = dict(boxstyle='round,pad=0.3', facecolor='#D0D0FF', edgecolor='blue', linewidth=2)
ax.text(7.5, 1.5, 'ADMM\n(Alternating Direction\nMethod of Multipliers)',
        fontsize=10, ha='center', va='center', bbox=rect_a)

# Arrows
from matplotlib.patches import FancyArrowPatch
arrow_props = dict(arrowstyle='->', mutation_scale=20, linewidth=2, color='gray')
ax.annotate('', xy=(3.5, 4.2), xytext=(4.2, 5.0), arrowprops=arrow_props)
ax.annotate('', xy=(6.5, 4.2), xytext=(5.8, 5.0), arrowprops=arrow_props)
ax.annotate('', xy=(2.5, 2.3), xytext=(2.5, 2.9), arrowprops=arrow_props)
ax.annotate('', xy=(7.5, 2.3), xytext=(7.5, 2.9), arrowprops=arrow_props)

# Labels on arrows
ax.text(3.3, 4.8, 'Discretize', fontsize=9, rotation=40, color='gray')
ax.text(6.2, 4.8, 'Discretize', fontsize=9, rotation=-40, color='gray')

# Lyapunov box
rect_ly = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='purple', linewidth=2)
ax.text(5, 0.5, 'Strong Lyapunov Function → Linear Convergence\n'
        '$E(t) = f(X) - f^* + \\frac{1}{2}\\|V + \\sqrt{\\mu}(X-x^*)\\|^2 \\leq E(0)e^{-\\sqrt{\\mu}t}$',
        fontsize=10, ha='center', va='center', bbox=rect_ly)

plt.savefig(os.path.join(img_dir, 'fig7_vos_framework_diagram.png'))
plt.close()
print("Figure 7 saved.")

# ============================================================
# Figure 8: ODE vs Discrete Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): ODE objective convergence
ax = axes[0]
ode_obj = ode_data['nesterov_ode']['obj']
ode_t = ode_data['nesterov_ode']['t']
ode_sc_obj = ode_data['strongly_convex_ode']['obj']
ode_sc_t = ode_data['strongly_convex_ode']['t']

ax.semilogy(ode_t, [max(o, 1e-16) for o in ode_obj], 'b-', linewidth=2, 
            label='Nesterov ODE ($r/t$ damping)')
ax.semilogy(ode_sc_t, [max(o, 1e-16) for o in ode_sc_obj], 'r-', linewidth=2,
            label='Strongly Convex ODE ($2\\sqrt{\\mu}$ damping)')

# Theoretical rates
t_th = np.array(ode_t[1:])
ax.semilogy(t_th, ode_obj[1] / t_th**2, 'b:', alpha=0.5, label='$O(1/t^2)$')
ax.semilogy(t_th, ode_sc_obj[1] * np.exp(-0.5 * t_th), 'r:', alpha=0.5, 
            label='$O(e^{-\\sqrt{\\mu}t})$')

ax.set_xlabel('Time $t$')
ax.set_ylabel('$f(X(t))$')
ax.set_title('(a) Continuous-Time ODE Convergence')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Phase portrait
ax = axes[1]
ode_x = np.array(ode_data['nesterov_ode']['x'])
ode_v = np.array(ode_data['nesterov_ode']['v'])
# Plot x1 vs v1
ax.plot(ode_x[:, 0], ode_v[:, 0], 'b-', linewidth=1.5, alpha=0.7, label='Nesterov ODE')
ode_sc_x = np.array(ode_data['strongly_convex_ode']['x'])
ode_sc_v = np.array(ode_data['strongly_convex_ode']['v'])
ax.plot(ode_sc_x[:, 0], ode_sc_v[:, 0], 'r-', linewidth=1.5, alpha=0.7, label='SC ODE')
ax.plot(0, 0, 'k*', markersize=12, label='Equilibrium')
ax.set_xlabel('$X_1$')
ax.set_ylabel('$\\dot{X}_1$')
ax.set_title('(b) Phase Portrait ($X_1$ vs $\\dot{X}_1$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig8_ode_dynamics.png'))
plt.close()
print("Figure 8 saved.")

# ============================================================
# Figure 9: Data Overview
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel (a): Singular values of A
ax = axes[0, 0]
U, s, Vt = np.linalg.svd(A, full_matrices=False)
ax.semilogy(range(len(s)), s, 'b-', linewidth=1.5)
ax.set_xlabel('Index')
ax.set_ylabel('Singular Value')
ax.set_title('(a) Singular Values of Design Matrix $A$')
ax.grid(True, alpha=0.3)

# Panel (b): True coefficients
ax = axes[0, 1]
ax.stem(range(len(x_true)), x_true, linefmt='b-', markerfmt='b.', basefmt='k-', 
        label='$x_{true}$')
ax.set_xlabel('Component Index')
ax.set_ylabel('Value')
ax.set_title(f'(b) True Sparse Coefficients (sparsity={np.sum(x_true != 0)})')
ax.grid(True, alpha=0.3)

# Panel (c): Distribution of A entries
ax = axes[1, 0]
ax.hist(A.flatten(), bins=100, density=True, alpha=0.7, color='steelblue')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('(c) Distribution of Design Matrix Entries')
ax.grid(True, alpha=0.3)

# Panel (d): Residual distribution
ax = axes[1, 1]
residual = A @ x_true - b
ax.hist(residual, bins=50, density=True, alpha=0.7, color='coral')
ax.set_xlabel('Residual Value')
ax.set_ylabel('Density')
ax.set_title('(d) Residual Distribution $Ax_{true} - b$')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'fig9_data_overview.png'))
plt.close()
print("Figure 9 saved.")

print("\nAll figures generated successfully!")
