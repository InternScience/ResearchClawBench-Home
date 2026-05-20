"""
Main experiment: VOS framework for Nesterov's method and ADMM.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vos_framework import (run_all_algorithms, compute_lyapunov_values,
                            lasso_objective, soft_threshold, _vos_core)
from scipy.linalg import norm, cho_factor

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.titlesize': 16, 'lines.linewidth': 2, 'figure.dpi': 150
})
np.random.seed(42)

# Load data
print("Loading data...")
data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
A, b, x_true = data['A'], data['b'], data['x_true']
m, n = A.shape
lam = 0.1 * np.linalg.norm(A.T @ b, ord=np.inf)
print(f"Shape: {A.shape}, lambda: {lam:.4f}")

# Run all algorithms
max_iter = 200
print(f"Running algorithms (max_iter={max_iter})...")
t0 = time.time()
R = run_all_algorithms(A, b, lam, max_iter)
t_total = time.time() - t0
print(f"Done in {t_total:.1f}s")

x_ref = R['x_ref']
f_ref = R['f_ref']
print(f"Reference objective: {f_ref:.6f}")

for name in ['obj_gd', 'obj_nag', 'obj_nag_r', 'obj_admm', 'obj_vos', 'obj_vos2']:
    obj = R[name]
    print(f"  {name}: final gap = {obj[-1] - f_ref:.6e}")
print(f"  ADMM restarts (NAG-R): {R['restarts']}")

# Compute Lyapunov values
print("Computing Lyapunov values...")
lyap = {}
for name, key in [('GD','hist_gd'),('NAG','hist_nag'),('NAG+R','hist_nag_r'),
                   ('ADMM','hist_admm'),('VOS','hist_vos'),('VOS(2nd)','hist_vos2')]:
    lyap[name] = compute_lyapunov_values(R[key], A, b, lam, x_ref)

# Save results
os.makedirs('outputs', exist_ok=True)
np.save('outputs/x_reference.npy', x_ref)
np.save('outputs/x_true.npy', x_true)

# ===========================
# Figure 1: Convergence comparison
# ===========================
print("Generating figures...")
fig, ax = plt.subplots(figsize=(10, 7))
for name, key, color, ls, lw in [
    ('Gradient Descent', 'obj_gd', 'b', '-', 2),
    ('Nesterov (NAG)', 'obj_nag', 'r', '-', 2),
    ('NAG + Restart', 'obj_nag_r', 'm', '--', 2),
    ('ADMM', 'obj_admm', 'g', '-', 2),
    ('VOS (Unified)', 'obj_vos', 'k', '-', 2.5),
    ('VOS (2nd-Order)', 'obj_vos2', 'c', '--', 2),
]:
    obj = R[key]
    gap = np.maximum(obj - f_ref, 1e-16)
    ax.semilogy(np.arange(len(obj)), gap, color=color, label=name, alpha=0.85, linewidth=lw, linestyle=ls)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$f(x_k) - f^*$')
ax.set_title('Convergence Comparison: Objective Value Gap')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure1_convergence_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure1")

# ===========================
# Figure 2: Lyapunov analysis
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for name, color, ls in [('GD','b','-'),('NAG','r','-'),('ADMM','g','-'),('VOS','k','-')]:
    ax.semilogy(lyap[name], color=color, label=name, linewidth=2, linestyle=ls)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$V(x_k) = f(x_k) - f^* + \frac{1}{2}\|x_k - x^*\|^2$')
ax.set_title('Lyapunov Function (Decreasing = Converging)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for name, color, ls in [('GD','b','-'),('NAG','r','-'),('ADMM','g','-'),('VOS','k','-')]:
    L = lyap[name]
    L_safe = np.maximum(L, 1e-30)
    ratios = L_safe[1:] / L_safe[:-1]
    valid = (ratios > 0) & (ratios < 1)
    if np.any(valid):
        ax.plot(np.where(valid)[0], ratios[valid], color=color, label=name, alpha=0.7, linestyle=ls, linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$V(x_{k+1}) / V(x_k)$')
ax.set_title('Lyapunov Decay Rate (< 1 = Monotonically Decreasing)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('report/images/figure2_lyapunov_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure2")

# ===========================
# Figure 3: Solution quality
# ===========================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
idx = np.arange(n)
ax.stem(idx[::10], x_true[::10], linefmt='b-', markerfmt='bo', basefmt='r-', label='True')
ax.stem(idx[::10]+0.3, R['x_vos'][::10], linefmt='r-', markerfmt='rs', basefmt='r-', label='VOS')
ax.set_xlabel('Coefficient Index'); ax.set_ylabel('Value')
ax.set_title('Recovered vs True Coefficients')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
pr, dr = R['primal_res'], R['dual_res']
ax.semilogy(pr, 'g-', label='Primal Residual', linewidth=2)
ax.semilogy(dr, 'orange', label='Dual Residual', linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel('Residual Norm')
ax.set_title('ADMM Primal & Dual Residuals')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
labels_map = {'x_gd':'GD','x_nag':'NAG','x_nag_r':'NAG+R','x_admm':'ADMM','x_vos':'VOS','x_vos2':'VOS(2nd)'}
colors_map = {'x_gd':'blue','x_nag':'red','x_nag_r':'magenta','x_admm':'green','x_vos':'black','x_vos2':'cyan'}
metrics = {labels_map[k]: norm(R[k] - x_ref) for k in labels_map}
bars = ax.bar(list(metrics.keys()), list(metrics.values()),
              color=[colors_map[k] for k in labels_map], alpha=0.7)
ax.set_ylabel(r'$\|x - x_{ref}\|$')
ax.set_title('Solution Error to Reference')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(metrics.values())*0.02,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('report/images/figure3_solution_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure3")

# ===========================
# Figure 4: Phase portrait + velocity
# ===========================
top2 = np.argsort(np.abs(x_ref))[-2:]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for name, key, color in [('GD','hist_gd','b'),('NAG','hist_nag','r'),('ADMM','hist_admm','g'),('VOS','hist_vos','k')]:
    traj = np.array([x[top2] for x in R[key][:min(150, len(R[key]))]])
    ax.plot(traj[:,0], traj[:,1], color=color, label=name, alpha=0.7, linewidth=1.5)
    ax.plot(traj[0,0], traj[0,1], 'o', color=color, markersize=8)
    ax.plot(traj[-1,0], traj[-1,1], 's', color=color, markersize=8)
ax.plot(x_ref[top2[0]], x_ref[top2[1]], '*', color='gold', markersize=15, label=r'$x^*$')
ax.set_xlabel(f'$x_{{{top2[0]}}}$'); ax.set_ylabel(f'$x_{{{top2[1]}}}$')
ax.set_title('Phase Portrait (Top-2 Components)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
vn = R['vel_norm']
ax.semilogy(vn, 'c-', linewidth=2, label='VOS (2nd-order)')
ax.set_xlabel('Iteration'); ax.set_ylabel(r'$\|\dot{x}\|$')
ax.set_title('Velocity Norm (ODE Damping Dynamics)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure4_phase_portrait.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure4")

# ===========================
# Figure 5: Detailed comparison
# ===========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: relative suboptimality
ax = axes[0,0]
f0 = R['obj_gd'][0]
for name, key, color, ls in [('GD','obj_gd','b','-'),('NAG','obj_nag','r','-'),
                               ('ADMM','obj_admm','g','-'),('VOS','obj_vos','k','-')]:
    rel = np.maximum(R[key] - f_ref, 1e-16) / max(f0 - f_ref, 1e-16)
    ax.semilogy(rel, color=color, label=name, linewidth=2, linestyle=ls)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$(f(x_k)-f^*) / (f(x_0)-f^*)$')
ax.set_title('Relative Suboptimality')
ax.legend(); ax.grid(True, alpha=0.3)

# Top-right: smoothed convergence
ax = axes[0,1]
for name, key, color, ls in [('NAG','obj_nag','r','-'),('ADMM','obj_admm','g','-'),('VOS','obj_vos','k','-')]:
    obj = np.maximum(R[key] - f_ref, 1e-16)
    if len(obj) > 5:
        sm = np.convolve(obj, np.ones(5)/5, mode='valid')
        ax.semilogy(sm, color=color, label=name, linewidth=2, linestyle=ls)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$f(x_k)-f^*$ (smoothed)')
ax.set_title('Smoothed Convergence (removing NAG oscillations)')
ax.legend(); ax.grid(True, alpha=0.3)

# Bottom-left: distance to ground truth
ax = axes[1,0]
for name, key, color, ls in [('GD','hist_gd','b','-'),('NAG','hist_nag','r','-'),
                               ('ADMM','hist_admm','g','-'),('VOS','hist_vos','k','-')]:
    errs = [norm(x - x_true) for x in R[key]]
    ax.semilogy(errs, color=color, label=name, linewidth=2, linestyle=ls)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\|x_k - x_{true}\|$')
ax.set_title('Distance to Ground Truth')
ax.legend(); ax.grid(True, alpha=0.3)

# Bottom-right: zoomed late-stage
ax = axes[1,1]
for name, key, color, ls in [('NAG','obj_nag','r','-'),('ADMM','obj_admm','g','-'),
                               ('VOS','obj_vos','k','-')]:
    obj = np.maximum(R[key] - f_ref, 1e-16)
    start = min(80, len(obj)-1)
    ax.semilogy(np.arange(start, len(obj)), obj[start:], color=color, label=name, linewidth=2, linestyle=ls)
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$f(x_k)-f^*$')
ax.set_title('Late-Stage Convergence (zoomed)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure5_detailed_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure5")

# ===========================
# Figure 6: Parameter sensitivity
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
step_size = R['step_size']
AtA = A.T @ A
Atb = A.T @ b

# Left: different r values for Nesterov ODE damping parameter
ax = axes[0]
for r_val, color in zip([1,2,3,4,5], ['blue','orange','red','green','purple']):
    x_prev = np.zeros(n); y = x_prev.copy()
    obj_r = [lasso_objective(A, b, x_prev, lam)]
    for k in range(1, 150):
        grad = AtA @ y - Atb
        x_curr = soft_threshold(y - step_size * grad, step_size * lam)
        beta = (k - 1) / (k + r_val - 1)
        y = x_curr + beta * (x_curr - x_prev)
        obj_r.append(lasso_objective(A, b, x_curr, lam))
        x_prev = x_curr.copy()
    label = f'r={r_val}' + (' (optimal)' if r_val == 3 else '')
    ax.semilogy(np.maximum(np.array(obj_r) - f_ref, 1e-16), color=color, label=label, linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel(r'$f(x_k)-f^*$')
ax.set_title('Nesterov ODE: Damping Parameter r\n($r=3$ is optimal for $O(1/k^2)$ rate)')
ax.legend(); ax.grid(True, alpha=0.3)

# Right: VOS mixing weight alpha
ax = axes[1]
rho_val = 2.0
M = AtA + rho_val * np.eye(n)
M_chol = cho_factor(M)
for alpha_val, color in zip([0.0, 0.25, 0.5, 0.75, 1.0], plt.cm.viridis(np.linspace(0.2,0.8,5))):
    x_v, obj_v, _ = _vos_core(AtA, Atb, A, b, np.zeros(n), lam, step_size, rho_val, 150, alpha_val)
    ax.semilogy(np.maximum(obj_v - f_ref, 1e-16), color=color, label=f'α={alpha_val:.2f}', linewidth=2)
ax.set_xlabel('Iteration'); ax.set_ylabel(r'$f(x_k)-f^*$')
ax.set_title('VOS Mixing Weight α\n(0=ADMM only, 1=Nesterov only)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure6_parameter_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure6")

# ===========================
# Figure 7: Data overview
# ===========================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
sv = np.linalg.svd(A, compute_uv=False)
ax.semilogy(sv, 'b-', linewidth=2)
ax.set_xlabel('Index'); ax.set_ylabel('Singular Value')
ax.set_title(f'Singular Values of A (κ={sv[0]/sv[-1]:.1f})')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.stem(np.arange(n), x_true, linefmt='b-', markerfmt='b.', basefmt='r-')
ax.set_xlabel('Coefficient Index'); ax.set_ylabel('Value')
ax.set_title(f'Ground Truth (sparsity: {np.sum(np.abs(x_true)>1e-4)}/{n})')
ax.grid(True, alpha=0.3)

ax = axes[2]
xv = R['x_vos']
ax.stem(np.arange(n), xv, linefmt='k-', markerfmt='k.', basefmt='r-')
ax.set_xlabel('Coefficient Index'); ax.set_ylabel('Value')
sp_vos = np.sum(np.abs(xv) > 1e-4)
ax.set_title(f'VOS Solution (sparsity: {sp_vos}/{n})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure7_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figure7")

# Save summary
summary = {'f_ref': float(f_ref), 'condition_number': float(sv[0]/sv[-1])}
for label, xkey in [('GD','x_gd'),('NAG','x_nag'),('NAG_Restart','x_nag_r'),
                      ('ADMM','x_admm'),('VOS','x_vos'),('VOS_2ndOrder','x_vos2')]:
    obj_key = 'obj_' + label.lower().replace('_restart','_r').replace('_2ndorder','2')
    summary[label] = {
        'error_to_ref': float(norm(R[xkey] - x_ref)),
        'sparsity': int(np.sum(np.abs(R[xkey]) > 1e-4)),
        'final_gap': float(R[obj_key][-1] - f_ref)
    }
with open('outputs/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*60)
for label in ['GD','NAG','NAG_Restart','ADMM','VOS','VOS_2ndOrder']:
    s = summary[label]
    print(f"  {label:15s}: gap={s['final_gap']:.6e}, err={s['error_to_ref']:.4f}, sparsity={s['sparsity']}")
