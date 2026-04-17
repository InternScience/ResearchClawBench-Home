#!/usr/bin/env python3
"""
Generate all figures for the RCS fidelity estimation report.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Physics_002_20260417_012839"
OUTPUTS = os.path.join(BASE, "outputs")
IMAGES = os.path.join(BASE, "report/images")
os.makedirs(IMAGES, exist_ok=True)

# Load results
with open(os.path.join(OUTPUTS, "xeb_fidelity_results.json")) as f:
    xeb_data = json.load(f)
with open(os.path.join(OUTPUTS, "mb_probability_results.json")) as f:
    mb_data = json.load(f)
with open(os.path.join(OUTPUTS, "transport_1qrb_results.json")) as f:
    transport_data = json.load(f)

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
})

# ============================================================
# Figure 1: XEB Fidelity vs Depth (N=40)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
depths_40 = []
means_40 = []
sems_40 = []
stds_40 = []
for key, val in sorted(xeb_data.items()):
    if val['dataset'] == 'N40_verification':
        depths_40.append(val['d'])
        means_40.append(val['mean_fxeb'])
        sems_40.append(val['sem_fxeb'])
        stds_40.append(val['std_fxeb'])

ax.errorbar(depths_40, means_40, yerr=sems_40, fmt='o-', color='#2196F3', 
            capsize=4, capthick=1.5, linewidth=2, markersize=8, label='XEB Fidelity (mean ± SEM)')
# Shade std region
ax.fill_between(depths_40, 
                [m-s for m,s in zip(means_40, stds_40)],
                [m+s for m,s in zip(means_40, stds_40)],
                alpha=0.15, color='#2196F3', label='±1 std across instances')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Uniform sampling (F=0)')
ax.set_xlabel('Circuit Depth $d$')
ax.set_ylabel('XEB Fidelity $F_{\\mathrm{XEB}}$')
ax.set_title('XEB Fidelity vs Circuit Depth ($N=40$ qubits)')
ax.legend(loc='upper right')
ax.set_xlim(6, 22)
ax.set_ylim(-0.2, 1.2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "xeb_vs_depth_N40.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xeb_vs_depth_N40.png")

# ============================================================
# Figure 2: XEB Fidelity vs N (d=12)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
Ns_12 = []
means_12 = []
sems_12 = []
stds_12 = []
for key, val in sorted(xeb_data.items()):
    if val['dataset'] == 'N_scan_depth12':
        Ns_12.append(val['N'])
        means_12.append(val['mean_fxeb'])
        sems_12.append(val['sem_fxeb'])
        stds_12.append(val['std_fxeb'])

ax.errorbar(Ns_12, means_12, yerr=sems_12, fmt='s-', color='#E91E63', 
            capsize=4, capthick=1.5, linewidth=2, markersize=8, label='XEB Fidelity (mean ± SEM)')
ax.fill_between(Ns_12, 
                [m-s for m,s in zip(means_12, stds_12)],
                [m+s for m,s in zip(means_12, stds_12)],
                alpha=0.15, color='#E91E63', label='±1 std across instances')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Uniform sampling (F=0)')
ax.set_xlabel('Number of Qubits $N$')
ax.set_ylabel('XEB Fidelity $F_{\\mathrm{XEB}}$')
ax.set_title('XEB Fidelity vs Qubit Count ($d=12$)')
ax.legend(loc='upper right')
ax.set_ylim(-0.2, 1.4)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "xeb_vs_N_d12.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xeb_vs_N_d12.png")

# ============================================================
# Figure 3: MB Probability vs Depth (N=40 and N=56)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# N=40
depths_mb40 = []
means_mb40 = []
sems_mb40 = []
for key, val in sorted(mb_data.items()):
    if val['dataset'] == 'N40_verification':
        depths_mb40.append(val['d'])
        means_mb40.append(val['mean_prob'])
        sems_mb40.append(val['sem_prob'])

# N=56
depths_mb56 = []
means_mb56 = []
sems_mb56 = []
for key, val in sorted(mb_data.items()):
    if val['dataset'] == 'N56_depths':
        depths_mb56.append(val['d'])
        means_mb56.append(val['mean_prob'])
        sems_mb56.append(val['sem_prob'])

ax.errorbar(depths_mb40, means_mb40, yerr=sems_mb40, fmt='o-', color='#2196F3', 
            capsize=4, linewidth=2, markersize=7, label='$N=40$')
ax.errorbar(depths_mb56, means_mb56, yerr=sems_mb56, fmt='s-', color='#FF5722', 
            capsize=4, linewidth=2, markersize=7, label='$N=56$')
ax.set_xlabel('Circuit Depth $d$')
ax.set_ylabel('MB Regression Probability')
ax.set_title('Matched Bitstring (MB) Probability vs Circuit Depth')
ax.legend()
ax.set_ylim(0, 0.8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "mb_vs_depth.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: mb_vs_depth.png")

# ============================================================
# Figure 4: MB Probability vs N (d=12)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
Ns_mb = []
means_mb_n = []
sems_mb_n = []
for key, val in sorted(mb_data.items()):
    if val['dataset'] == 'N_scan_depth12':
        Ns_mb.append(val['N'])
        means_mb_n.append(val['mean_prob'])
        sems_mb_n.append(val['sem_prob'])

ax.errorbar(Ns_mb, means_mb_n, yerr=sems_mb_n, fmt='D-', color='#4CAF50', 
            capsize=4, linewidth=2, markersize=7, label='MB Probability (mean ± SEM)')
ax.set_xlabel('Number of Qubits $N$')
ax.set_ylabel('MB Regression Probability')
ax.set_title('Matched Bitstring (MB) Probability vs Qubit Count ($d=12$)')
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "mb_vs_N_d12.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: mb_vs_N_d12.png")

# ============================================================
# Figure 5: Transport/1QRB Fidelity vs Depth
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

colors_N = {16: '#9C27B0', 24: '#3F51B5', 32: '#009688', 40: '#2196F3', 48: '#FF9800', 56: '#F44336'}

# N40_verification
for N_val in [40, 56]:
    ds = 'N40_verification' if N_val == 40 else 'N56_depths'
    depths_t = []
    means_t = []
    sems_t = []
    for key, val in sorted(transport_data.items()):
        if val['dataset'] == ds:
            depths_t.append(val['d'])
            means_t.append(val['mean_prob'])
            sems_t.append(val['std_prob'] / np.sqrt(val['n_instances']))
    if depths_t:
        ax.errorbar(depths_t, means_t, yerr=sems_t, fmt='o-', color=colors_N[N_val], 
                    capsize=3, linewidth=2, markersize=6, label=f'$N={N_val}$')

ax.set_xlabel('Circuit Depth $d$')
ax.set_ylabel('Transport/1QRB Exact Match Probability')
ax.set_title('Transport/1QRB Fidelity vs Circuit Depth')
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "transport_vs_depth.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: transport_vs_depth.png")

# ============================================================
# Figure 6: Transport/1QRB Fidelity vs N (at d=16)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

for depth_val in [4, 16, 32, 48, 64, 96]:
    Ns_t = []
    means_t = []
    sems_t = []
    for key, val in sorted(transport_data.items()):
        if val['dataset'] == 'N_scan_depth12' and val['d'] == depth_val:
            Ns_t.append(val['N'])
            means_t.append(val['mean_prob'])
            sems_t.append(val['std_prob'] / np.sqrt(val['n_instances']))
    if Ns_t:
        ax.errorbar(Ns_t, means_t, yerr=sems_t, fmt='o-', 
                    capsize=3, linewidth=1.5, markersize=5, label=f'$d={depth_val}$')

ax.set_xlabel('Number of Qubits $N$')
ax.set_ylabel('Transport/1QRB Exact Match Probability')
ax.set_title('Transport/1QRB Fidelity vs Qubit Count')
ax.legend(ncol=2)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "transport_vs_N.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: transport_vs_N.png")

# ============================================================
# Figure 7: Combined XEB + MB comparison (N=40 depth scan)
# ============================================================
fig, ax1 = plt.subplots(figsize=(9, 5.5))

# XEB on left axis
ax1.errorbar(depths_40, means_40, yerr=sems_40, fmt='o-', color='#2196F3', 
            capsize=4, linewidth=2, markersize=8, label='XEB Fidelity')
ax1.set_xlabel('Circuit Depth $d$')
ax1.set_ylabel('XEB Fidelity $F_{\\mathrm{XEB}}$', color='#2196F3')
ax1.tick_params(axis='y', labelcolor='#2196F3')
ax1.set_ylim(-0.1, 1.0)

# MB on right axis
ax2 = ax1.twinx()
ax2.errorbar(depths_mb40, means_mb40, yerr=sems_mb40, fmt='s--', color='#FF5722', 
            capsize=4, linewidth=2, markersize=8, label='MB Probability')
ax2.set_ylabel('MB Regression Probability', color='#FF5722')
ax2.tick_params(axis='y', labelcolor='#FF5722')
ax2.set_ylim(0, 0.8)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax1.set_title('XEB Fidelity and MB Probability vs Depth ($N=40$)')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "xeb_mb_comparison_N40.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xeb_mb_comparison_N40.png")

# ============================================================
# Figure 8: Per-instance XEB scatter (N=40, selected depths)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
selected_depths = [8, 12, 16, 20]
for idx, d_sel in enumerate(selected_depths):
    ax = axes[idx // 2][idx % 2]
    key = f"N40_verification_N40_d{d_sel}"
    if key in xeb_data:
        instances = xeb_data[key]['per_instance']
        rs = [inst['r'] for inst in instances]
        fxebs = [inst['fxeb'] for inst in instances]
        ax.scatter(rs, fxebs, c='#2196F3', alpha=0.6, s=30, edgecolors='navy', linewidth=0.5)
        ax.axhline(y=xeb_data[key]['mean_fxeb'], color='red', linestyle='-', linewidth=1.5, 
                   label=f"Mean = {xeb_data[key]['mean_fxeb']:.3f}")
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Instance Index $r$')
        ax.set_ylabel('$F_{\\mathrm{XEB}}$')
        ax.set_title(f'$N=40$, $d={d_sel}$')
        ax.legend(fontsize=10)
        ax.set_ylim(-0.5, 2.0)
        ax.grid(True, alpha=0.3)

plt.suptitle('Per-Instance XEB Fidelity Distribution ($N=40$)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "xeb_per_instance_N40.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xeb_per_instance_N40.png")

# ============================================================
# Figure 9: Gate Error Model - Exponential Decay Fit
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Fit exponential decay to XEB vs depth (N=40)
from scipy.optimize import curve_fit

def exp_decay(d, A, alpha):
    return A * np.exp(-alpha * d)

# XEB decay
depths_arr = np.array(depths_40, dtype=float)
means_arr = np.array(means_40, dtype=float)
try:
    popt_xeb, pcov_xeb = curve_fit(exp_decay, depths_arr, means_arr, p0=[1.0, 0.05], maxfev=5000)
    d_fit = np.linspace(6, 22, 100)
    f_fit_xeb = exp_decay(d_fit, *popt_xeb)
    ax1.plot(d_fit, f_fit_xeb, '--', color='#2196F3', linewidth=2, 
             label=f'Fit: $A e^{{-\\alpha d}}$, $\\alpha$={popt_xeb[1]:.4f}')
except:
    pass

ax1.errorbar(depths_40, means_40, yerr=sems_40, fmt='o', color='#2196F3', 
            capsize=4, markersize=8, label='XEB Data')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Circuit Depth $d$')
ax1.set_ylabel('XEB Fidelity $F_{\\mathrm{XEB}}$')
ax1.set_title('XEB Fidelity Decay with Depth ($N=40$)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.0)

# MB decay
depths_mb_arr = np.array(depths_mb40, dtype=float)
means_mb_arr = np.array(means_mb40, dtype=float)
try:
    popt_mb, pcov_mb = curve_fit(exp_decay, depths_mb_arr, means_mb_arr, p0=[1.0, 0.05], maxfev=5000)
    d_fit_mb = np.linspace(6, 22, 100)
    f_fit_mb = exp_decay(d_fit_mb, *popt_mb)
    ax2.plot(d_fit_mb, f_fit_mb, '--', color='#FF5722', linewidth=2,
             label=f'Fit: $A e^{{-\\alpha d}}$, $\\alpha$={popt_mb[1]:.4f}')
except:
    pass

ax2.errorbar(depths_mb40, means_mb40, yerr=sems_mb40, fmt='s', color='#FF5722', 
            capsize=4, markersize=8, label='MB Data')
ax2.set_xlabel('Circuit Depth $d$')
ax2.set_ylabel('MB Regression Probability')
ax2.set_title('MB Probability Decay with Depth ($N=40$)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "exponential_decay_fits.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: exponential_decay_fits.png")

# ============================================================
# Figure 10: Comprehensive Fidelity Landscape (Heatmap-style)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Combine all fidelity metrics for N_scan_depth12 at d=12
# XEB (where available), MB, and Transport (at d=16 as proxy)
Ns_all = [16, 24, 32, 40, 48, 56]
xeb_vals = []
mb_vals = []
transport_d16 = []
transport_d32 = []

for N in Ns_all:
    # XEB
    key = f"N_scan_depth12_N{N}_d12"
    if key in xeb_data:
        xeb_vals.append(xeb_data[key]['mean_fxeb'])
    else:
        xeb_vals.append(None)
    
    # MB
    if key in mb_data:
        mb_vals.append(mb_data[key]['mean_prob'])
    else:
        mb_vals.append(None)
    
    # Transport d=16
    key_t16 = f"N_scan_depth12_N{N}_d16"
    if key_t16 in transport_data:
        transport_d16.append(transport_data[key_t16]['mean_prob'])
    else:
        transport_d16.append(None)
    
    # Transport d=32
    key_t32 = f"N_scan_depth12_N{N}_d32"
    if key_t32 in transport_data:
        transport_d32.append(transport_data[key_t32]['mean_prob'])
    else:
        transport_d32.append(None)

# Plot
x_pos = np.arange(len(Ns_all))
width = 0.2

# XEB
xeb_plot = [v if v is not None else 0 for v in xeb_vals]
xeb_mask = [v is not None for v in xeb_vals]
bars1 = ax.bar(x_pos - 1.5*width, xeb_plot, width, color='#2196F3', alpha=0.8, label='XEB Fidelity (d=12)')
# Hatch missing ones
for i, m in enumerate(xeb_mask):
    if not m:
        bars1[i].set_hatch('///')
        bars1[i].set_alpha(0.3)

# MB
mb_plot = [v if v is not None else 0 for v in mb_vals]
bars2 = ax.bar(x_pos - 0.5*width, mb_plot, width, color='#4CAF50', alpha=0.8, label='MB Probability (d=12)')

# Transport d=16
t16_plot = [v if v is not None else 0 for v in transport_d16]
bars3 = ax.bar(x_pos + 0.5*width, t16_plot, width, color='#FF9800', alpha=0.8, label='Transport/1QRB (d=16)')

# Transport d=32
t32_plot = [v if v is not None else 0 for v in transport_d32]
bars4 = ax.bar(x_pos + 1.5*width, t32_plot, width, color='#F44336', alpha=0.8, label='Transport/1QRB (d=32)')

ax.set_xlabel('Number of Qubits $N$')
ax.set_ylabel('Fidelity / Probability')
ax.set_title('Fidelity Comparison Across Metrics and Qubit Counts')
ax.set_xticks(x_pos)
ax.set_xticklabels(Ns_all)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "fidelity_comparison_bar.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fidelity_comparison_bar.png")

# ============================================================
# Figure 11: XEB vs MB correlation plot
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))

# Collect paired XEB and MB per-instance data for N40_verification
for d in [8, 10, 12, 14, 16, 18, 20]:
    xeb_key = f"N40_verification_N40_d{d}"
    mb_key = f"N40_verification_N40_d{d}"
    
    if xeb_key in xeb_data and mb_key in mb_data:
        xeb_mean = xeb_data[xeb_key]['mean_fxeb']
        mb_mean = mb_data[mb_key]['mean_prob']
        xeb_sem = xeb_data[xeb_key]['sem_fxeb']
        mb_sem = mb_data[mb_key]['sem_prob']
        
        ax.errorbar(xeb_mean, mb_mean, xerr=xeb_sem, yerr=mb_sem, 
                    fmt='o', markersize=10, capsize=3, color='#673AB7')
        ax.annotate(f'd={d}', (xeb_mean, mb_mean), textcoords="offset points", 
                   xytext=(8, 5), fontsize=10)

# Add diagonal reference
ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='y=x reference')
ax.set_xlabel('XEB Fidelity $F_{\\mathrm{XEB}}$')
ax.set_ylabel('MB Regression Probability')
ax.set_title('XEB Fidelity vs MB Probability ($N=40$)')
ax.legend()
ax.set_xlim(0, 0.8)
ax.set_ylim(0, 0.8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "xeb_vs_mb_correlation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xeb_vs_mb_correlation.png")

# ============================================================
# Figure 12: Transport 1QRB decay with exponential fit
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

for N_val, ds_name, color, marker in [(40, 'N40_verification', '#2196F3', 'o'), 
                                        (56, 'N56_depths', '#F44336', 's')]:
    depths_t = []
    means_t = []
    sems_t = []
    for key, val in sorted(transport_data.items()):
        if val['dataset'] == ds_name:
            depths_t.append(val['d'])
            means_t.append(val['mean_prob'])
            sems_t.append(val['std_prob'] / np.sqrt(val['n_instances']))
    
    if depths_t:
        ax.errorbar(depths_t, means_t, yerr=sems_t, fmt=f'{marker}-', color=color, 
                    capsize=3, linewidth=2, markersize=7, label=f'$N={N_val}$ data')
        
        # Fit exponential decay
        try:
            d_arr = np.array(depths_t, dtype=float)
            m_arr = np.array(means_t, dtype=float)
            popt, _ = curve_fit(exp_decay, d_arr, m_arr, p0=[1.0, 0.01], maxfev=5000)
            d_fit = np.linspace(0, 100, 200)
            ax.plot(d_fit, exp_decay(d_fit, *popt), '--', color=color, alpha=0.6,
                   label=f'$N={N_val}$ fit: $\\alpha$={popt[1]:.4f}')
        except:
            pass

ax.set_xlabel('Circuit Depth $d$')
ax.set_ylabel('Transport/1QRB Exact Match Probability')
ax.set_title('Transport/1QRB Fidelity Decay with Depth')
ax.legend()
ax.set_ylim(0, 1.1)
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "transport_decay_fit.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: transport_decay_fit.png")

# ============================================================
# Figure 13: Per-qubit error rate estimation
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# From XEB: estimate per-cycle error rate
# F_XEB ≈ (1 - e_cycle)^d where e_cycle is per-cycle error rate
# So e_cycle ≈ 1 - exp(ln(F_XEB)/d)
# Or from exponential fit: F = A*exp(-alpha*d), alpha ≈ e_cycle

# From Transport: per-qubit error from Hamming distance
# avg_hd / N ≈ per-qubit error rate at that depth
for N_val, ds_name, color, marker in [(40, 'N40_verification', '#2196F3', 'o'), 
                                        (56, 'N56_depths', '#F44336', 's')]:
    depths_t = []
    per_qubit_err = []
    for key, val in sorted(transport_data.items()):
        if val['dataset'] == ds_name and val['avg_hamming_distance'] is not None:
            depths_t.append(val['d'])
            per_qubit_err.append(val['avg_hamming_distance'] / N_val)
    
    if depths_t:
        ax1.plot(depths_t, per_qubit_err, f'{marker}-', color=color, 
                linewidth=2, markersize=7, label=f'$N={N_val}$')

ax1.set_xlabel('Circuit Depth $d$')
ax1.set_ylabel('Per-Qubit Error Rate (avg Hamming / N)')
ax1.set_title('Per-Qubit Error Rate from Transport/1QRB')
ax1.legend()
ax1.grid(True, alpha=0.3)

# From XEB: per-cycle error rate at each depth
for d in depths_40:
    key = f"N40_verification_N40_d{d}"
    if key in xeb_data:
        fxeb = xeb_data[key]['mean_fxeb']
        if fxeb > 0:
            e_per_cycle = 1 - np.exp(np.log(fxeb) / d)
            ax2.scatter(d, e_per_cycle, c='#2196F3', s=80, zorder=5)

ax2.set_xlabel('Circuit Depth $d$')
ax2.set_ylabel('Estimated Per-Cycle Error Rate')
ax2.set_title('Per-Cycle Error Rate from XEB ($N=40$)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 0.1)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "error_rate_estimates.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: error_rate_estimates.png")

# ============================================================
# Figure 14: Fidelity vs N - all methods combined
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

# XEB at d=12
xeb_ns = []
xeb_means = []
xeb_sems = []
for key, val in sorted(xeb_data.items()):
    if val['dataset'] == 'N_scan_depth12':
        xeb_ns.append(val['N'])
        xeb_means.append(val['mean_fxeb'])
        xeb_sems.append(val['sem_fxeb'])

ax.errorbar(xeb_ns, xeb_means, yerr=xeb_sems, fmt='o-', color='#2196F3', 
            capsize=4, linewidth=2, markersize=8, label='XEB Fidelity ($d=12$)')

# MB at d=12
mb_ns = []
mb_means = []
mb_sems = []
for key, val in sorted(mb_data.items()):
    if val['dataset'] == 'N_scan_depth12':
        mb_ns.append(val['N'])
        mb_means.append(val['mean_prob'])
        mb_sems.append(val['sem_prob'])

ax.errorbar(mb_ns, mb_means, yerr=mb_sems, fmt='s-', color='#4CAF50', 
            capsize=4, linewidth=2, markersize=8, label='MB Probability ($d=12$)')

# Transport at d=16
t_ns = []
t_means = []
t_sems = []
for key, val in sorted(transport_data.items()):
    if val['dataset'] == 'N_scan_depth12' and val['d'] == 16:
        t_ns.append(val['N'])
        t_means.append(val['mean_prob'])
        t_sems.append(val['std_prob'] / np.sqrt(val['n_instances']))

ax.errorbar(t_ns, t_means, yerr=t_sems, fmt='D-', color='#FF9800', 
            capsize=4, linewidth=2, markersize=8, label='Transport/1QRB ($d=16$)')

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Qubits $N$')
ax.set_ylabel('Fidelity / Probability')
ax.set_title('Fidelity Estimates vs Qubit Count (Multiple Methods)')
ax.legend()
ax.set_ylim(-0.1, 1.2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "all_methods_vs_N.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: all_methods_vs_N.png")

# ============================================================
# Figure 15: Histogram of per-instance XEB fidelities
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

idx = 0
for d in [8, 10, 12, 14, 16, 18, 20]:
    key = f"N40_verification_N40_d{d}"
    if key in xeb_data and idx < 7:
        instances = xeb_data[key]['per_instance']
        fxebs = [inst['fxeb'] for inst in instances]
        ax = axes[idx]
        ax.hist(fxebs, bins=15, color='#2196F3', alpha=0.7, edgecolor='navy')
        ax.axvline(x=xeb_data[key]['mean_fxeb'], color='red', linestyle='-', linewidth=2,
                  label=f"Mean={xeb_data[key]['mean_fxeb']:.3f}")
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('$F_{\\mathrm{XEB}}$')
        ax.set_ylabel('Count')
        ax.set_title(f'$d={d}$')
        ax.legend(fontsize=9)
        idx += 1

# Use last subplot for legend/info
ax = axes[7]
ax.text(0.5, 0.5, 'XEB Fidelity\nDistributions\n$N=40$ qubits\n50 instances each', 
        transform=ax.transAxes, ha='center', va='center', fontsize=14,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.axis('off')

plt.suptitle('Distribution of Per-Instance XEB Fidelity ($N=40$)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, "xeb_histograms_N40.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xeb_histograms_N40.png")

print("\n=== All figures generated ===")
