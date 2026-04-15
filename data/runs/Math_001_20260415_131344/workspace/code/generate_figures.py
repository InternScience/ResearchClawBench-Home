"""
Generate all figures for the VOS Framework research report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os

os.makedirs('report/images', exist_ok=True)

# Load results
with open('outputs/convergence_results.json', 'r') as f:
    summary = json.load(f)

# Load histories
histories = {}
for name in ['gd', 'fista', 'fista_restart', 'admm', 'vos_noderiv', 'vos_restart']:
    data = np.load(f'outputs/{name}_history.npy', allow_pickle=True).item()
    histories[name] = data

# Load Lyapunov data
with open('outputs/lyapunov_analysis.json', 'r') as f:
    lyap_data = json.load(f)

# Load sweep data
sweep_data = np.load('outputs/vos_sweep.npy', allow_pickle=True).item()

f_star = summary['f_star']

# ============================================================
# Figure 1: Convergence Rate Comparison (Objective vs Iteration)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

method_labels = {
    'gd': 'Proximal Gradient Descent (O(1/k))',
    'fista': 'FISTA / Nesterov AGD (O(1/k²))',
    'fista_restart': 'FISTA + Restarting (Linear)',
    'admm': 'ADMM (Operator Splitting)',
    'vos_noderiv': 'VOS-NODE (ODE-derived)',
    'vos_restart': 'VOS + Adaptive Restart'
}

colors = {
    'gd': '#1f77b4',
    'fista': '#ff7f0e',
    'fista_restart': '#2ca02c',
    'admm': '#d62728',
    'vos_noderiv': '#9467bd',
    'vos_restart': '#8c564b'
}

linestyles = {
    'gd': '-',
    'fista': '--',
    'fista_restart': '-.',
    'admm': ':',
    'vos_noderiv': '--',
    'vos_restart': '-.'
}

max_iters = min(500, max(len(histories[n]['obj_history']) for n in histories))

for name in ['gd', 'fista', 'fista_restart', 'admm', 'vos_noderiv', 'vos_restart']:
    obj = np.array(histories[name]['obj_history'])
    gap = obj - f_star
    iters = np.arange(1, len(gap) + 1)
    
    # Clip to max_iters
    mask = iters <= max_iters
    ax.semilogy(iters[mask], np.maximum(gap[mask], 1e-16), 
                label=method_labels[name], color=colors[name], 
                linestyle=linestyles[name], linewidth=2)

ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Optimality Gap $F(x_k) - F^*$', fontsize=13)
ax.set_title('Convergence Rate Comparison: VOS Framework Methods', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_iters)

plt.tight_layout()
plt.savefig('report/images/convergence_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/convergence_comparison.png")

# ============================================================
# Figure 2: Objective Value vs Iteration (linear scale, early iterations)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

early_iters = 100
for name in ['gd', 'fista', 'fista_restart', 'admm', 'vos_noderiv', 'vos_restart']:
    obj = np.array(histories[name]['obj_history'])
    iters = np.arange(1, len(obj) + 1)
    mask = iters <= early_iters
    ax.plot(iters[mask], obj[mask], 
            label=method_labels[name], color=colors[name], 
            linestyle=linestyles[name], linewidth=2)

ax.axhline(y=f_star, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Optimal $F^*$')
ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Objective Value $F(x_k)$', fontsize=13)
ax.set_title('Objective Function Decay (First 100 Iterations)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/objective_decay_early.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/objective_decay_early.png")

# ============================================================
# Figure 3: Lyapunov Function Decay
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

t_vals = np.array(lyap_data['t_values'])
lyap_vos = np.array(lyap_data['vos_lyapunov'])

# Normalize Lyapunov function
lyap_normalized = lyap_vos / max(lyap_vos[0], 1e-16)

ax.semilogy(t_vals, np.maximum(lyap_normalized, 1e-16), 
            color='#9467bd', linewidth=2, label='VOS Lyapunov Energy $E(t)$')

# Reference O(1/t^2) decay line
ref_decay = (t_vals[0] / t_vals)**2
ax.semilogy(t_vals, ref_decay, 'k--', linewidth=1.5, alpha=0.6, label='$O(1/t^2)$ reference')

ax.set_xlabel('Continuous Time $t$', fontsize=13)
ax.set_ylabel('Normalized Lyapunov Energy $E(t)/E(0)$', fontsize=13)
ax.set_title('Strong Lyapunov Function Decay in VOS Framework', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/lyapunov_decay.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/lyapunov_decay.png")

# ============================================================
# Figure 4: Generalized Damping Parameter Sweep (Phase Transition at r=3)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

r_colors = {
    'r=1.0': '#d62728',
    'r=2.0': '#ff7f0e',
    'r=3.0': '#2ca02c',
    'r=4.0': '#1f77b4',
    'r=5.0': '#9467bd'
}

for r_key, result in sweep_data.items():
    obj = np.array(result['obj_history'])
    gap = obj - f_star
    iters = np.arange(1, len(gap) + 1)
    ax.semilogy(iters, np.maximum(gap, 1e-16), 
                label=f'damping $r={r_key[2:]}$', 
                color=r_colors[r_key], linewidth=2)

ax.axvline(x=0, color='gray', linewidth=0.5)
ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('Optimality Gap $F(x_k) - F^*$', fontsize=13)
ax.set_title('VOS Phase Transition: Generalized Damping Parameter $r$', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2000)

plt.tight_layout()
plt.savefig('report/images/damping_phase_transition.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/damping_phase_transition.png")

# ============================================================
# Figure 5: Method Comparison Table Visualization
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.axis('off')

methods_display = ['gd', 'fista', 'fista_restart', 'admm', 'vos_noderiv', 'vos_restart']
col_labels = ['Method', 'Final Objective', 'Optimality Gap', 'Iterations', 'Time (s)']

rows = []
for name in methods_display:
    info = summary['methods'][name]
    rows.append([
        method_labels[name].split('(')[0].strip(),
        f"{info['final_obj']:.6f}",
        f"{info['optimality_gap']:.2e}",
        str(info['iterations']),
        f"{info['time']:.2f}"
    ])

table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Color header
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(rows) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#D9E2F3')

ax.set_title('Method Comparison Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('report/images/method_comparison_table.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/method_comparison_table.png")

# ============================================================
# Figure 6: Convergence Rate Analysis (log-log plot for rate verification)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Focus on GD and FISTA for rate analysis
for name in ['gd', 'fista']:
    obj = np.array(histories[name]['obj_history'])
    gap = obj - f_star
    
    # Only use iterations where gap > machine epsilon
    valid = gap > 1e-14
    iters = np.arange(1, len(gap) + 1)
    
    if np.sum(valid) > 10:
        ax.loglog(iters[valid], gap[valid], 
                  label=method_labels[name], color=colors[name], 
                  linewidth=2)

# Reference lines
iters_ref = np.array([10, 100, 500])
ax.loglog(iters_ref, 1.0 / iters_ref, 'k--', linewidth=1.5, alpha=0.5, label='$O(1/k)$ reference')
ax.loglog(iters_ref, 1.0 / iters_ref**2, 'k:', linewidth=1.5, alpha=0.5, label='$O(1/k^2)$ reference')

ax.set_xlabel('Iteration $k$', fontsize=13)
ax.set_ylabel('Optimality Gap $F(x_k) - F^*$', fontsize=13)
ax.set_title('Convergence Rate Verification (Log-Log Scale)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/rate_verification.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/rate_verification.png")

# ============================================================
# Figure 7: Sparsity Pattern of Solution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Load ground truth
data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
x_true = data['x_true']

# Use FISTA solution
x_fista = np.array(summary['methods']['fista'].get('final_x', []))
if len(x_fista) == 0:
    x_fista = np.load('outputs/fista_history.npy', allow_pickle=True).item()

# For sparsity, load from the actual result
results_gd = np.load('outputs/gd_history.npy', allow_pickle=True).item()
results_fista = np.load('outputs/fista_history.npy', allow_pickle=True).item()

# Load precomputed FISTA solution
x_fista_sol = np.load('outputs/x_fista_final.npy')
data = np.load('data/complex_optimization_data.npy', allow_pickle=True).item()
x_true = data['x_true']

# Plot sparsity pattern
axes[0].stem(np.arange(len(x_true)), x_true, linefmt='b-', markerfmt='bo', basefmt='k-', label='Ground Truth')
axes[0].set_xlabel('Coefficient Index', fontsize=11)
axes[0].set_ylabel('Value', fontsize=11)
axes[0].set_title('Ground Truth Sparse Coefficients', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].stem(np.arange(len(x_fista_sol)), x_fista_sol, linefmt='r-', markerfmt='ro', basefmt='k-', label='Recovered (FISTA)')
axes[1].set_xlabel('Coefficient Index', fontsize=11)
axes[1].set_ylabel('Value', fontsize=11)
axes[1].set_title('Recovered Sparse Coefficients', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/sparsity_pattern.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/sparsity_pattern.png")

print("\nAll figures generated successfully!")
