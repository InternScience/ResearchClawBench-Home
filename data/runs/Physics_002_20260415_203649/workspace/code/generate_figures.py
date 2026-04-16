#!/usr/bin/env python3
"""
Generate all figures for the RCS Fidelity Estimation report.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load results
with open('outputs/fidelity_results.json', 'r') as f:
    results = json.load(f)

IMG_DIR = 'report/images'
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.figsize': (8, 6),
})

# ============================================================
# Figure 1: N40 Depth Scan - XEB Fidelity vs Depth
# ============================================================

fig1, ax1 = plt.subplots(figsize=(8, 6))

n40_xeb = results['n40_xeb']
depths_xeb = sorted([int(d) for d in n40_xeb.keys()])
f_xeb_means = [n40_xeb[str(d)]['mean_fidelity'] for d in depths_xeb]
f_xeb_se = [n40_xeb[str(d)]['se_fidelity'] for d in depths_xeb]

n40_model = results['n40_model']
model_depths = sorted([int(d) for d in n40_model.keys()])
f_pred = [n40_model[str(d)]['f_pred'] for d in model_depths]

ax1.errorbar(depths_xeb, f_xeb_means, yerr=f_xeb_se, fmt='o-', 
             color='#2196F3', markersize=8, capsize=4, linewidth=2,
             label='XEB Fidelity (experimental)')
ax1.plot(model_depths, f_pred, 's--', color='#FF5722', markersize=8, linewidth=2,
         label='Gate-count model prediction')

# Classical approximability threshold line
ax1.axhline(y=1/2**40, color='#9E9E9E', linestyle=':', linewidth=1.5,
            label=f'Uniform threshold (1/2^40 ≈ {1/2**40:.2e})')

ax1.set_xlabel('Circuit Depth (d)')
ax1.set_ylabel('Fidelity')
ax1.set_title('N=40: XEB Fidelity vs Circuit Depth')
ax1.legend(loc='upper right')
ax1.set_ylim(-0.05, 0.8)
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(os.path.join(IMG_DIR, 'fig1_n40_depth_xeb.png'))
plt.close(fig1)

# ============================================================
# Figure 2: N40 Depth Scan - MB Survival Probability vs Depth
# ============================================================

fig2, ax2 = plt.subplots(figsize=(8, 6))

n40_mb = results['n40_mb']
mb_depths = sorted([int(d) for d in n40_mb.keys()])
p_surv_means = [n40_mb[str(d)]['mean_survival'] for d in mb_depths]
p_surv_se = [n40_mb[str(d)]['se_survival'] for d in mb_depths]

ax2.errorbar(mb_depths, p_surv_means, yerr=p_surv_se, fmt='o-', 
             color='#4CAF50', markersize=8, capsize=4, linewidth=2,
             label='MB Survival Probability')

# Also plot gate-count model for comparison
ax2.plot(model_depths, f_pred, 's--', color='#FF5722', markersize=8, linewidth=2,
         label='Gate-count model prediction')

ax2.set_xlabel('Circuit Depth (d)')
ax2.set_ylabel('Survival Probability / Fidelity')
ax2.set_title('N=40: MB Survival Probability vs Circuit Depth')
ax2.legend(loc='upper right')
ax2.set_ylim(-0.05, 0.7)
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(IMG_DIR, 'fig2_n40_depth_mb.png'))
plt.close(fig2)

# ============================================================
# Figure 3: N40 Combined - XEB + MB + Model vs Depth
# ============================================================

fig3, ax3 = plt.subplots(figsize=(10, 7))

ax3.errorbar(depths_xeb, f_xeb_means, yerr=f_xeb_se, fmt='o-', 
             color='#2196F3', markersize=8, capsize=4, linewidth=2,
             label='XEB Fidelity')
ax3.errorbar(mb_depths, p_surv_means, yerr=p_surv_se, fmt='D-', 
             color='#4CAF50', markersize=8, capsize=4, linewidth=2,
             label='MB Survival Probability')
ax3.plot(model_depths, f_pred, 's--', color='#FF5722', markersize=8, linewidth=2,
         label='Gate-count model prediction')

ax3.axhline(y=1/2**40, color='#9E9E9E', linestyle=':', linewidth=1.5,
            label=f'Uniform threshold')

ax3.set_xlabel('Circuit Depth (d)')
ax3.set_ylabel('Fidelity / Survival Probability')
ax3.set_title('N=40: Fidelity Comparison Across Methods vs Depth')
ax3.legend(loc='upper right')
ax3.set_ylim(-0.05, 0.75)
ax3.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(os.path.join(IMG_DIR, 'fig3_n40_depth_combined.png'))
plt.close(fig3)

# ============================================================
# Figure 4: N-scan at d=12 - XEB Fidelity vs N
# ============================================================

fig4, ax4 = plt.subplots(figsize=(8, 6))

n_scan_xeb = results['n_scan_xeb']
n_vals_xeb = sorted([int(n) for n in n_scan_xeb.keys()])
f_xeb_n_means = [n_scan_xeb[str(n)]['mean_fidelity'] for n in n_vals_xeb]
f_xeb_n_se = [n_scan_xeb[str(n)]['se_fidelity'] for n in n_vals_xeb]

n_scan_model = results['n_scan_model']
n_vals_model = sorted([int(n) for n in n_scan_model.keys()])
f_pred_n = [n_scan_model[str(n)]['f_pred'] for n in n_vals_model]

ax4.errorbar(n_vals_xeb, f_xeb_n_means, yerr=f_xeb_n_se, fmt='o-', 
             color='#2196F3', markersize=8, capsize=4, linewidth=2,
             label='XEB Fidelity (experimental)')
ax4.plot(n_vals_model, f_pred_n, 's--', color='#FF5722', markersize=8, linewidth=2,
         label='Gate-count model prediction')

# Classical approximability thresholds
for n in n_vals_xeb:
    ax4.axhline(y=1/2**n, color='#9E9E9E', linestyle=':', linewidth=0.5, alpha=0.3)

ax4.set_xlabel('Number of Qubits (N)')
ax4.set_ylabel('Fidelity')
ax4.set_title('Depth d=12: XEB Fidelity vs Number of Qubits')
ax4.legend(loc='upper right')
ax4.set_ylim(-0.05, 0.9)
ax4.grid(True, alpha=0.3)

fig4.tight_layout()
fig4.savefig(os.path.join(IMG_DIR, 'fig4_nscan_xeb.png'))
plt.close(fig4)

# ============================================================
# Figure 5: N-scan at d=12 - MB Survival Probability vs N
# ============================================================

fig5, ax5 = plt.subplots(figsize=(8, 6))

n_scan_mb = results['n_scan_mb']
n_vals_mb = sorted([int(n) for n in n_scan_mb.keys()])
p_surv_n_means = [n_scan_mb[str(n)]['mean_survival'] for n in n_vals_mb]
p_surv_n_se = [n_scan_mb[str(n)]['se_survival'] for n in n_vals_mb]

ax5.errorbar(n_vals_mb, p_surv_n_means, yerr=p_surv_n_se, fmt='D-', 
             color='#4CAF50', markersize=8, capsize=4, linewidth=2,
             label='MB Survival Probability')
ax5.plot(n_vals_model, f_pred_n, 's--', color='#FF5722', markersize=8, linewidth=2,
         label='Gate-count model prediction')

ax5.set_xlabel('Number of Qubits (N)')
ax5.set_ylabel('Survival Probability / Fidelity')
ax5.set_title('Depth d=12: MB Survival Probability vs Number of Qubits')
ax5.legend(loc='upper right')
ax5.set_ylim(-0.05, 0.9)
ax5.grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig(os.path.join(IMG_DIR, 'fig5_nscan_mb.png'))
plt.close(fig5)

# ============================================================
# Figure 6: N-scan combined - XEB + MB + Model vs N
# ============================================================

fig6, ax6 = plt.subplots(figsize=(10, 7))

ax6.errorbar(n_vals_xeb, f_xeb_n_means, yerr=f_xeb_n_se, fmt='o-', 
             color='#2196F3', markersize=8, capsize=4, linewidth=2,
             label='XEB Fidelity')
ax6.errorbar(n_vals_mb, p_surv_n_means, yerr=p_surv_n_se, fmt='D-', 
             color='#4CAF50', markersize=8, capsize=4, linewidth=2,
             label='MB Survival Probability')
ax6.plot(n_vals_model, f_pred_n, 's--', color='#FF5722', markersize=8, linewidth=2,
         label='Gate-count model prediction')

ax6.set_xlabel('Number of Qubits (N)')
ax6.set_ylabel('Fidelity / Survival Probability')
ax6.set_title('Depth d=12: Fidelity Comparison Across Methods vs Number of Qubits')
ax6.legend(loc='upper right')
ax6.set_ylim(-0.05, 0.9)
ax6.grid(True, alpha=0.3)

fig6.tight_layout()
fig6.savefig(os.path.join(IMG_DIR, 'fig6_nscan_combined.png'))
plt.close(fig6)

# ============================================================
# Figure 7: Transport 1QRB - Survival Probability vs Depth
# ============================================================

fig7, ax7 = plt.subplots(figsize=(8, 6))

n40_transport = results['n40_transport']
t_depths = sorted([int(d) for d in n40_transport.keys()])
t_surv_means = [n40_transport[str(d)]['mean_survival'] for d in t_depths]
t_surv_se = [n40_transport[str(d)]['se_survival'] for d in t_depths]

ax7.errorbar(t_depths, t_surv_means, yerr=t_surv_se, fmt='o-', 
             color='#9C27B0', markersize=8, capsize=4, linewidth=2,
             label='Transport 1QRB (N=40)')

# Fit exponential decay to transport data
if len(t_depths) > 2:
    from scipy.optimize import curve_fit
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)
    try:
        popt, pcov = curve_fit(exp_decay, t_depths, t_surv_means, p0=[1.0, 0.01])
        d_fit = np.linspace(min(t_depths), max(t_depths), 100)
        ax7.plot(d_fit, exp_decay(d_fit, *popt), '--', color='#9C27B0', alpha=0.5,
                label=f'Exp. fit: {popt[0]:.3f}·exp(-{popt[1]:.4f}·d)')
    except:
        pass

ax7.set_xlabel('Circuit Depth (d)')
ax7.set_ylabel('Survival Probability')
ax7.set_title('N=40: Transport 1QRB Survival Probability vs Depth')
ax7.legend(loc='upper right')
ax7.set_ylim(-0.05, 1.05)
ax7.grid(True, alpha=0.3)

fig7.tight_layout()
fig7.savefig(os.path.join(IMG_DIR, 'fig7_transport_depth.png'))
plt.close(fig7)

# ============================================================
# Figure 8: Gap Analysis - Experimental vs Classical Approximability
# ============================================================

fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Depth scan gap for N=40
gap_depths = depths_xeb
gap_xeb = f_xeb_means
gap_model = [n40_model[str(d)]['f_pred'] for d in gap_depths]

ax8a.fill_between(gap_depths, gap_model, gap_xeb, alpha=0.3, color='#FFC107',
                  label='Gap (XEB - Model)')
ax8a.plot(gap_depths, gap_xeb, 'o-', color='#2196F3', markersize=8, linewidth=2,
          label='XEB Fidelity')
ax8a.plot(gap_depths, gap_model, 's--', color='#FF5722', markersize=8, linewidth=2,
          label='Gate-count model')

ax8a.set_xlabel('Circuit Depth (d)')
ax8a.set_ylabel('Fidelity')
ax8a.set_title('N=40: Gap Between Experimental\nFidelity and Error Propagation Model')
ax8a.legend(loc='upper right')
ax8a.set_ylim(-0.05, 0.75)
ax8a.grid(True, alpha=0.3)

# Right panel: N-scan gap at d=12
gap_n_vals = n_vals_xeb
gap_xeb_n = f_xeb_n_means
gap_model_n = [n_scan_model[str(n)]['f_pred'] for n in gap_n_vals]

ax8b.fill_between(gap_n_vals, gap_model_n, gap_xeb_n, alpha=0.3, color='#FFC107',
                  label='Gap (XEB - Model)')
ax8b.plot(gap_n_vals, gap_xeb_n, 'o-', color='#2196F3', markersize=8, linewidth=2,
          label='XEB Fidelity')
ax8b.plot(gap_n_vals, gap_model_n, 's--', color='#FF5722', markersize=8, linewidth=2,
          label='Gate-count model')

ax8b.set_xlabel('Number of Qubits (N)')
ax8b.set_ylabel('Fidelity')
ax8b.set_title('d=12: Gap Between Experimental\nFidelity and Error Propagation Model')
ax8b.legend(loc='upper right')
ax8b.set_ylim(-0.05, 0.9)
ax8b.grid(True, alpha=0.3)

fig8.tight_layout()
fig8.savefig(os.path.join(IMG_DIR, 'fig8_gap_analysis.png'))
plt.close(fig8)

# ============================================================
# Figure 9: Log-scale fidelity comparison
# ============================================================

fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(14, 6))

# Left: N=40 depth scan (log scale)
ax9a.semilogy(depths_xeb, f_xeb_means, 'o-', color='#2196F3', markersize=8, linewidth=2,
              label='XEB Fidelity')
ax9a.semilogy(mb_depths, p_surv_means, 'D-', color='#4CAF50', markersize=8, linewidth=2,
              label='MB Survival Probability')
ax9a.semilogy(model_depths, f_pred, 's--', color='#FF5722', markersize=8, linewidth=2,
              label='Gate-count model')

ax9a.set_xlabel('Circuit Depth (d)')
ax9a.set_ylabel('Fidelity / Survival Probability (log scale)')
ax9a.set_title('N=40: Fidelity Decay (Log Scale)')
ax9a.legend(loc='upper right')
ax9a.grid(True, alpha=0.3)

# Right: N-scan at d=12 (log scale)
ax9b.semilogy(n_vals_xeb, f_xeb_n_means, 'o-', color='#2196F3', markersize=8, linewidth=2,
              label='XEB Fidelity')
ax9b.semilogy(n_vals_mb, p_surv_n_means, 'D-', color='#4CAF50', markersize=8, linewidth=2,
              label='MB Survival Probability')
ax9b.semilogy(n_vals_model, f_pred_n, 's--', color='#FF5722', markersize=8, linewidth=2,
              label='Gate-count model')

ax9b.set_xlabel('Number of Qubits (N)')
ax9b.set_ylabel('Fidelity / Survival Probability (log scale)')
ax9b.set_title('d=12: Fidelity Scaling (Log Scale)')
ax9b.legend(loc='upper right')
ax9b.grid(True, alpha=0.3)

fig9.tight_layout()
fig9.savefig(os.path.join(IMG_DIR, 'fig9_logscale_comparison.png'))
plt.close(fig9)

print("All figures generated successfully!")
print(f"Saved to: {IMG_DIR}")
for f in sorted(os.listdir(IMG_DIR)):
    if f.endswith('.png'):
        print(f"  {f}")