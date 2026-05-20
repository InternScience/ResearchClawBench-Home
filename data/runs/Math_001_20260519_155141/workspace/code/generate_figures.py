"""
Generate all figures for the VOS framework report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Ensure output dirs exist
os.makedirs('report/images', exist_ok=True)

# Load results
with open('outputs/experiment_results.json') as f:
    r = json.load(f)

f_star = r['f_star']

# ---------------------------------------------------------------------------
# Figure 1: Convergence comparison (objective error vs iterations)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, label, color in [
    ('ista', 'ISTA (GD)', 'C0'),
    ('fista', 'FISTA (NAG)', 'C1'),
    ('fista_restart', 'FISTA-Restart', 'C2'),
    ('admm', 'ADMM', 'C3'),
    ('vos', 'VOS Unified', 'C4'),
]:
    if name in r:
        obj = np.array(r[name]['obj'])
        err = np.maximum(obj - f_star, 1e-16)
        iters = np.arange(1, len(err) + 1)
        ax.semilogy(iters, err, label=label, color=color, linewidth=1.5)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Objective Error $f(x_k) - f^*$', fontsize=12)
ax.set_title('Convergence Comparison (Log Scale)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

ax = axes[1]
for name, label, color in [
    ('ista', 'ISTA (GD)', 'C0'),
    ('fista', 'FISTA (NAG)', 'C1'),
    ('fista_restart', 'FISTA-Restart', 'C2'),
    ('admm', 'ADMM', 'C3'),
    ('vos', 'VOS Unified', 'C4'),
]:
    if name in r:
        obj = np.array(r[name]['obj'])
        err = np.maximum(obj - f_star, 1e-16)
        time = np.array(r[name]['time'])
        ax.semilogy(time, err, label=label, color=color, linewidth=1.5)

ax.set_xlabel('CPU Time (s)', fontsize=12)
ax.set_ylabel('Objective Error $f(x_k) - f^*$', fontsize=12)
ax.set_title('Convergence vs Time', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/convergence_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved convergence_comparison.png")

# ---------------------------------------------------------------------------
# Figure 2: Linear convergence demonstration (log-log and semilog)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, label, color in [
    ('ista', 'ISTA', 'C0'),
    ('fista', 'FISTA', 'C1'),
    ('fista_restart', 'FISTA-Restart', 'C2'),
]:
    if name in r:
        obj = np.array(r[name]['obj'])
        err = np.maximum(obj - f_star, 1e-16)
        iters = np.arange(1, len(err) + 1)
        ax.semilogy(iters, err, label=label, color=color, linewidth=1.5)

# Add theoretical rate lines
iters_theory = np.arange(1, 200)
ax.semilogy(iters_theory, 500.0 / iters_theory, 'k--', linewidth=1, label=r'$O(1/k)$')
ax.semilogy(iters_theory, 2000.0 / iters_theory**2, 'k:', linewidth=1, label=r'$O(1/k^2)$')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Objective Error', fontsize=12)
ax.set_title('Sublinear Rates: ISTA vs FISTA', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

ax = axes[1]
for name, label, color in [
    ('fista_restart', 'FISTA-Restart', 'C2'),
    ('admm', 'ADMM', 'C3'),
]:
    if name in r:
        obj = np.array(r[name]['obj'])
        err = np.maximum(obj - f_star, 1e-16)
        iters = np.arange(1, len(err) + 1)
        ax.semilogy(iters, err, 'o-', label=label, color=color, linewidth=1.5, markersize=3)

# Fit linear convergence line
if 'fista_restart' in r:
    obj = np.array(r['fista_restart']['obj'])
    err = np.maximum(obj - f_star, 1e-16)
    iters = np.arange(1, len(err) + 1)
    mask = err < 1.0
    if mask.sum() > 5:
        log_err = np.log(err[mask])
        it_fit = iters[mask]
        coeffs = np.polyfit(it_fit, log_err, 1)
        rate = np.exp(coeffs[0])
        ax.semilogy(it_fit, np.exp(coeffs[1]) * (rate ** it_fit), 'g--',
                    linewidth=2, label=f'Linear fit ($\\rho={rate:.3f}$)')

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Objective Error', fontsize=12)
ax.set_title('Linear Convergence: Restarted FISTA & ADMM', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/linear_convergence.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved linear_convergence.png")

# ---------------------------------------------------------------------------
# Figure 3: Lyapunov decay for continuous-time system
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
t_ode = np.array(r['ode']['t'])
E_ode = np.array(r['ode']['lyapunov'])
ax.plot(t_ode, E_ode, linewidth=1.5, color='C0')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('Lyapunov Function $\\mathcal{E}(t)$', fontsize=12)
ax.set_title('Lyapunov Function Decay (Continuous-Time)', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)

ax = axes[1]
ax.semilogy(t_ode, np.maximum(E_ode, 1e-16), linewidth=1.5, color='C0')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('Lyapunov Function $\\mathcal{E}(t)$ (log)', fontsize=12)
ax.set_title('Lyapunov Decay (Log Scale)', fontsize=13)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('report/images/lyapunov_decay.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved lyapunov_decay.png")

# ---------------------------------------------------------------------------
# Figure 4: ADMM residuals
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
if 'admm' in r:
    iters = np.arange(1, len(r['admm']['r_norm']) + 1)
    ax.semilogy(iters, r['admm']['r_norm'], label='Primal residual', linewidth=1.5)
    ax.semilogy(iters, r['admm']['s_norm'], label='Dual residual', linewidth=1.5)
    if 'eps_pri' in r['admm']:
        ax.semilogy(iters, r['admm']['eps_pri'], 'k--', label='Primal tolerance', linewidth=1)
        ax.semilogy(iters, r['admm']['eps_dual'], 'k:', label='Dual tolerance', linewidth=1)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.set_title('ADMM Convergence Diagnostics', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('report/images/admm_residuals.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved admm_residuals.png")

if not os.path.exists('report/images/admm_residuals.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, 'ADMM residual data not available', ha='center', va='center', fontsize=14)
    plt.savefig('report/images/admm_residuals.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved placeholder admm_residuals.png")

# ---------------------------------------------------------------------------
# Figure 5: Phase space trajectory (first 2 coordinates)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
# Use sparse ODE data for phase plot
X_ode_sparse = []
V_ode_sparse = []
t_sparse = np.array(r['ode']['t_sparse'])
obj_sparse = np.array(r['ode']['obj'])
# Generate simple spiral approximation for visualization
omega = 1.0
t_vis = np.linspace(0.1, 15, 1000)
x1_vis = np.exp(-0.1 * t_vis) * np.cos(omega * t_vis)
x2_vis = np.exp(-0.1 * t_vis) * np.sin(omega * t_vis)
# Scale to match problem scale
scale = 0.5
ax.plot(scale * x1_vis, scale * x2_vis, linewidth=1.0, alpha=0.7, label='ODE trajectory (schematic)')
ax.scatter([0], [0], color='red', s=100, marker='*', zorder=5, label='Optimum $x^*$')
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Phase Space Trajectory (Schematic)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.axis('equal')
plt.tight_layout()
plt.savefig('report/images/phase_space.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved phase_space.png")

# ---------------------------------------------------------------------------
# Figure 6: Solution recovery comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Load ground truth
import numpy as np
data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
x_true = data['x_true']
n = len(x_true)
idx = np.arange(n)

ax = axes[0]
ax.stem(idx, x_true, linefmt='C0-', markerfmt='C0o', basefmt=' ')
ax.set_xlabel('Index', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Ground Truth $x_{true}$', fontsize=12)
ax.set_xlim([0, n])

# Need x_star - approximate from VOS which is same as FISTA
# Load from the problem via quick computation
import sys
sys.path.insert(0, 'code')
from vos_framework import load_data, LassoProblem
A, b, x_true = load_data()
prob = LassoProblem(A, b, r['lambda'])
x_star = prob.exact_solution_cvx(max_iter=5000)

ax = axes[1]
ax.stem(idx, x_star, linefmt='C1-', markerfmt='C1o', basefmt=' ')
ax.set_xlabel('Index', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Recovered Solution $x^*$ (FISTA)', fontsize=12)
ax.set_xlim([0, n])

ax = axes[2]
ax.stem(idx, x_star - x_true, linefmt='C2-', markerfmt='C2o', basefmt=' ')
ax.set_xlabel('Index', fontsize=11)
ax.set_ylabel('Error', fontsize=11)
ax.set_title('Recovery Error $x^* - x_{true}$', fontsize=12)
ax.set_xlim([0, n])

plt.tight_layout()
plt.savefig('report/images/solution_recovery.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved solution_recovery.png")

# ---------------------------------------------------------------------------
# Figure 7: Convergence rate comparison (objective vs iteration, linear scale)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for name, label, color in [
    ('ista', 'ISTA', 'C0'),
    ('fista', 'FISTA', 'C1'),
    ('fista_restart', 'FISTA-Restart', 'C2'),
    ('admm', 'ADMM', 'C3'),
]:
    if name in r:
        obj = np.array(r[name]['obj'])
        iters = np.arange(1, len(obj) + 1)
        ax.plot(iters, obj, label=label, color=color, linewidth=1.5)

ax.axhline(f_star, color='black', linestyle='--', linewidth=1, label='$f^*$')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Objective Value', fontsize=12)
ax.set_title('Objective Convergence (Linear Scale)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim([0, 200])
plt.tight_layout()
plt.savefig('report/images/objective_linear.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved objective_linear.png")

print("\nAll figures generated successfully!")
