"""
Generate all figures for the VOS Framework research report.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# ─── Load Results ────────────────────────────────────────────────────────────
results = np.load('outputs/convergence_results.npz', allow_pickle=True)
solutions = np.load('outputs/final_solutions.npz', allow_pickle=True)
data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()

with open('outputs/opt_info.json', 'r') as f:
    opt_info = json.load(f)

A = data['A']
b = data['b']
x_true = data['x_true']
x_opt_ref = solutions['x_opt']

iterations = np.arange(1, len(results['gd_obj_gap']) + 1)

# ─── Figure 1: Convergence Comparison (Objective Gap) ────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

# Clip negative gaps to small positive value for log plotting
def clip_for_log(arr, min_val=1e-15):
    return np.maximum(arr, min_val)

ax.semilogy(iterations, clip_for_log(results['gd_obj_gap']), 
            label='Gradient Descent (ISTA)', linewidth=2, color='#1f77b4')
ax.semilogy(iterations, clip_for_log(results['nesterov_obj_gap']), 
            label='Nesterov AGD (FISTA)', linewidth=2, color='#ff7f0e')
ax.semilogy(iterations, clip_for_log(results['hb_obj_gap']), 
            label='Heavy Ball (Polyak)', linewidth=2, color='#2ca02c')
ax.semilogy(iterations, clip_for_log(results['admm_obj_gap']), 
            label='ADMM', linewidth=2, color='#d62728')
ax.semilogy(iterations, clip_for_log(results['vos_obj_gap']), 
            label='VOS Unified', linewidth=2, color='#9467bd', linestyle='--')

ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Objective Gap $f(x_k) - f^*$')
ax.set_title('Convergence Comparison: Objective Function Gap')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([1, 500])

plt.savefig('report/images/fig1_convergence_comparison.png')
plt.close()
print("Figure 1 saved.")

# ─── Figure 2: Solution Error Norm ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

ax.semilogy(iterations, results['gd_x_err'], 
            label='Gradient Descent (ISTA)', linewidth=2, color='#1f77b4')
ax.semilogy(iterations, results['nesterov_x_err'], 
            label='Nesterov AGD (FISTA)', linewidth=2, color='#ff7f0e')
ax.semilogy(iterations, results['hb_x_err'], 
            label='Heavy Ball (Polyak)', linewidth=2, color='#2ca02c')
ax.semilogy(iterations, results['admm_x_err'], 
            label='ADMM', linewidth=2, color='#d62728')
ax.semilogy(iterations, results['vos_x_err'], 
            label='VOS Unified', linewidth=2, color='#9467bd', linestyle='--')

ax.set_xlabel('Iteration $k$')
ax.set_ylabel('$\\|x_k - x^*\\|$')
ax.set_title('Convergence Comparison: Solution Error Norm')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([1, 500])

plt.savefig('report/images/fig2_solution_error.png')
plt.close()
print("Figure 2 saved.")

# ─── Figure 3: Lyapunov Function Decay ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

methods = [
    ('gd_lyapunov', 'Gradient Descent', '#1f77b4'),
    ('nesterov_lyapunov', 'Nesterov AGD', '#ff7f0e'),
    ('hb_lyapunov', 'Heavy Ball', '#2ca02c'),
    ('admm_lyapunov', 'ADMM', '#d62728'),
    ('vos_lyapunov', 'VOS Unified', '#9467bd'),
]

for idx, (key, name, color) in enumerate(methods):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    lyap = results[key]
    # Normalize by initial value
    lyap_norm = lyap / lyap[0] if lyap[0] > 0 else lyap
    ax.semilogy(iterations, clip_for_log(lyap_norm), linewidth=2, color=color)
    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('$E_k / E_0$')
    ax.set_title(f'Lyapunov Decay: {name}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 500])

# Combined Lyapunov comparison in last subplot
ax = axes[1, 2]
for key, name, color in methods:
    lyap = results[key]
    lyap_norm = lyap / lyap[0] if lyap[0] > 0 else lyap
    ax.semilogy(iterations, clip_for_log(lyap_norm), linewidth=2, color=color, label=name)
ax.set_xlabel('Iteration $k$')
ax.set_ylabel('$E_k / E_0$')
ax.set_title('Lyapunov Decay: All Methods')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([1, 500])

plt.tight_layout()
plt.savefig('report/images/fig3_lyapunov_decay.png')
plt.close()
print("Figure 3 saved.")

# ─── Figure 4: Nesterov ODE vs Discrete Scheme ─────────────────────────────
# Simulate the continuous-time ODE: ddotX + (3/t) dotX + nabla f(X) = 0
from scipy.integrate import solve_ivp

def nesterov_ode_rhs(t, state):
    """ODE system: [X, dX/dt] -> [dX/dt, -(3/t)*dX/dt - nabla f(X)]"""
    X = state[:n_features]
    V = state[n_features:]
    # nabla f(X) for smooth part
    grad_f = A.T @ (A @ X - b)
    # Add proximal-like term for L1 (approximate as gradient of Huber)
    huber_grad = np.where(np.abs(X) > lambda_lasso, 
                          lambda_lasso * np.sign(X),
                          X)
    total_grad = grad_f + huber_grad
    
    if t < 1e-6:
        damping = 3.0 / 1e-6  # avoid singularity
    else:
        damping = 3.0 / t
    
    dX = V
    dV = -damping * V - total_grad
    return np.concatenate([dX, dV])

n_features = A.shape[1]
x0_ode = np.zeros(2 * n_features)
x0_ode[:n_features] = np.zeros(n_features)  # start from zero

# Solve ODE for a short time range (the ODE is stiff for large problems)
t_span = (1e-3, 50.0)
t_eval = np.linspace(1e-3, 50.0, 200)

try:
    sol = solve_ivp(nesterov_ode_rhs, t_span, x0_ode, t_eval=t_eval, 
                    method='RK45', max_step=0.1, rtol=1e-4)
    
    ode_obj_gaps = []
    for i in range(len(sol.t)):
        X_t = sol.y[:n_features, i]
        # Apply soft threshold for L1 objective evaluation
        f_t = obj_smooth(X_t) + lambda_lasso * np.sum(np.abs(X_t))
        ode_obj_gaps.append(f_t - opt_info['f_opt'])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    t_mapped = sol.t / np.sqrt(1.0 / opt_info['L_lip'])  # map continuous time to iterations
    
    ax.semilogy(t_mapped, clip_for_log(ode_obj_gaps), 
                label='Nesterov ODE (continuous)', linewidth=2, color='#ff7f0e')
    ax.semilogy(iterations, clip_for_log(results['nesterov_obj_gap']), 
                label='Nesterov AGD (discrete)', linewidth=2, color='#ff7f0e', linestyle='--')
    ax.set_xlabel('Equivalent Iteration / Time')
    ax.set_ylabel('Objective Gap $f(x_k) - f^*$')
    ax.set_title('Nesterov ODE vs Discrete Scheme')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.savefig('report/images/fig4_ode_vs_discrete.png')
    plt.close()
    print("Figure 4 saved.")
except Exception as e:
    print(f"ODE simulation failed: {e}")
    # Create alternative figure
    fig, ax = plt.subplots(figsize=(10, 7))
    # Plot theoretical rate vs empirical
    k_arr = iterations
    theo_rate = 4 * opt_info['L_lip'] * np.linalg.norm(np.zeros(n_features) - x_opt_ref)**2 / (k_arr + 2)**2
    ax.semilogy(k_arr, clip_for_log(results['nesterov_obj_gap']), 
                label='Nesterov AGD (empirical)', linewidth=2, color='#ff7f0e')
    ax.semilogy(k_arr, clip_for_log(theo_rate), 
                label='$O(1/k^2)$ theoretical bound', linewidth=2, color='#ff7f0e', linestyle='--')
    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('Objective Gap $f(x_k) - f^*$')
    ax.set_title('Nesterov AGD: Empirical vs Theoretical Rate')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.savefig('report/images/fig4_ode_vs_discrete.png')
    plt.close()
    print("Figure 4 (alternative) saved.")

# ─── Figure 5: ADMM Residuals ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
ax1.semilogy(iterations, clip_for_log(results['admm_primal_res']), 
             label='Primal Residual', linewidth=2, color='#d62728')
ax1.semilogy(iterations, clip_for_log(results['admm_dual_res']), 
             label='Dual Residual', linewidth=2, color='#1f77b4')
ax1.set_xlabel('Iteration $k$')
ax1.set_ylabel('Residual Norm')
ax1.set_title('ADMM: Primal and Dual Residuals')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.semilogy(iterations, clip_for_log(results['admm_obj_gap']), 
             label='ADMM Objective Gap', linewidth=2, color='#d62728')
ax2.set_xlabel('Iteration $k$')
ax2.set_ylabel('Objective Gap $f(x_k) - f^*$')
ax2.set_title('ADMM: Objective Convergence')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig5_admm_residuals.png')
plt.close()
print("Figure 5 saved.")

# ─── Figure 6: Sparsity Recovery ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

solution_list = [
    (solutions['x_true'], 'Ground Truth $x_{\\mathrm{true}}$'),
    (solutions['x_opt'], 'Reference Optimum $x^*$'),
    (solutions['x_gd'], 'GD (ISTA)'),
    (solutions['x_nesterov'], 'Nesterov AGD (FISTA)'),
    (solutions['x_admm'], 'ADMM'),
    (solutions['x_vos'], 'VOS Unified'),
]

for idx, (sol, name) in enumerate(solution_list):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    # Show first 100 coefficients for visualization
    ax.bar(range(100), sol[:100], width=1.0, alpha=0.7)
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('Value')
    ax.set_title(name)
    ax.set_xlim([0, 100])

plt.tight_layout()
plt.savefig('report/images/fig6_sparsity_recovery.png')
plt.close()
print("Figure 6 saved.")

# ─── Figure 7: VOS Framework Diagram & Phase Portrait ──────────────────────
# Phase portrait showing trajectory convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: 2D projection of trajectories (first two significant components)
ax = axes[0]
# Project onto first 2 PCA components of x_true
# PCA not needed; using manual coordinate projection below

# Manual PCA: use first 2 coordinates that have nonzero x_true
nz_idx = np.where(np.abs(x_true) > 0.01)[0]
if len(nz_idx) >= 2:
    idx1, idx2 = nz_idx[0], nz_idx[1]
else:
    idx1, idx2 = 0, 1

# Track trajectories (we need to re-run with trajectory tracking)
# Instead, show final solution scatter
methods_final = {
    'GD': solutions['x_gd'],
    'Nesterov': solutions['x_nesterov'],
    'ADMM': solutions['x_admm'],
    'VOS': solutions['x_vos'],
}

for name, sol in methods_final.items():
    ax.scatter(sol[idx1], sol[idx2], s=80, label=name, zorder=5)

ax.scatter(x_opt_ref[idx1], x_opt_ref[idx2], s=120, marker='*', 
           color='black', label='$x^*$ (ref)', zorder=10)
ax.scatter(x_true[idx1], x_true[idx2], s=120, marker='D', 
           color='green', label='$x_{\\mathrm{true}}$', zorder=10)

ax.set_xlabel(f'$x_{{{idx1}}}$')
ax.set_ylabel(f'$x_{{{idx2}}}$')
ax.set_title('Solution Comparison: 2D Projection')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Right: Accelerated rate verification - plot k^2 * gap for Nesterov
ax = axes[1]
k_arr = iterations
nesterov_scaled = k_arr**2 * results['nesterov_obj_gap']
gd_scaled = k_arr * results['gd_obj_gap']

ax.plot(k_arr, nesterov_scaled, linewidth=2, color='#ff7f0e', label='$k^2(f(x_k)-f^*)$ [Nesterov]')
ax.plot(k_arr, gd_scaled, linewidth=2, color='#1f77b4', label='$k(f(x_k)-f^*)$ [GD]')

ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Scaled Objective Gap')
ax.set_title('Rate Verification: Scaled Gaps')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig7_phase_portrait_rate.png')
plt.close()
print("Figure 7 saved.")

# ─── Figure 8: Sensitivity Analysis (rho for ADMM) ──────────────────────────
# We need to re-run ADMM with different rho values
# Load the algorithms module
import sys
sys.path.insert(0, 'code')

# Re-import necessary functions
data_raw = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
A_r = data_raw['A']
b_r = data_raw['b']
x_true_r = data_raw['x_true']

def grad_smooth_r(x):
    return A_r.T @ (A_r @ x - b_r)

def obj_smooth_r(x):
    r = A_r @ x - b_r
    return 0.5 * np.dot(r, r)

def soft_threshold_r(x, theta):
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0)

lambda_r = opt_info['lambda_lasso']
L_r = opt_info['L_lip']
f_opt_r = opt_info['f_opt']
x_opt_r = np.load('outputs/x_opt_ref.npy', allow_pickle=True)

rho_values = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
admm_rho_results = {}

n_iter_rho = 300
n_r = A_r.shape[1]

for rho in rho_values:
    x = np.zeros(n_r)
    z = np.zeros(n_r)
    u = np.zeros(n_r)
    
    M = A_r.T @ A_r + rho * np.eye(n_r)
    
    gaps = []
    for k in range(n_iter_rho):
        z = np.linalg.solve(M, A_r.T @ b_r + rho * (x - u))
        x = soft_threshold_r(z + u, lambda_r / rho)
        u = u + z - x
        
        f_k = obj_smooth_r(x) + lambda_r * np.sum(np.abs(x))
        gaps.append(f_k - f_opt_r)
    
    admm_rho_results[rho] = gaps

fig, ax = plt.subplots(figsize=(10, 7))
iters_rho = np.arange(1, n_iter_rho + 1)

for rho in rho_values:
    ax.semilogy(iters_rho, clip_for_log(admm_rho_results[rho]), 
                linewidth=2, label=f'$\\rho={rho}$')

ax.set_xlabel('Iteration $k$')
ax.set_ylabel('Objective Gap $f(x_k) - f^*$')
ax.set_title('ADMM Convergence: Effect of Penalty Parameter $\\rho$')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.savefig('report/images/fig8_admm_rho_sensitivity.png')
plt.close()
print("Figure 8 saved.")

# ─── Save Summary Statistics ────────────────────────────────────────────────
summary = {}
for method in ['gd', 'nesterov', 'hb', 'admm', 'vos']:
    final_gap = results[f'{method}_obj_gap'][-1]
    final_err = results[f'{method}_x_err'][-1]
    summary[method] = {
        'final_obj_gap': float(final_gap),
        'final_x_err': float(final_err),
        'convergence_iters_to_1e-4': int(np.argmax(results[f'{method}_obj_gap'] < 1e-4) + 1) if np.any(results[f'{method}_obj_gap'] < 1e-4) else -1,
    }

with open('outputs/convergence_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSummary statistics:")
for method, stats in summary.items():
    print(f"  {method}: gap={stats['final_obj_gap']:.6e}, err={stats['final_x_err']:.6e}")

print("\nAll figures generated successfully!")