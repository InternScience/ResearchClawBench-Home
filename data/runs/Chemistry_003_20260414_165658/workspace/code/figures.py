"""Generate all figures for the LES report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os
import sys
sys.path.insert(0, 'code')
from parse_data import parse_xyz
from scipy.optimize import curve_fit
from scipy.special import erfc

os.makedirs('report/images', exist_ok=True)

# Load analysis results
with open('outputs/dataset1_charge_recovery.json') as f:
    d1 = json.load(f)
with open('outputs/dataset2_binding_curves.json') as f:
    d2 = json.load(f)
with open('outputs/dataset3_charge_states.json') as f:
    d3 = json.load(f)

# Figure 1: Data Overview
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

frames_rc = parse_xyz('data/random_charges.xyz')
tc = frames_rc[0]['props']['true_charges']
axes[0].bar(['+1e', '-1e'], [tc.count(1.0), tc.count(-1.0)], color=['steelblue', 'coral'], edgecolor='black', alpha=0.8)
axes[0].set_xlabel('Charge Type', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title(f'(a) Random Charges (N={len(tc)})', fontsize=13)
axes[0].grid(True, alpha=0.3, axis='y')

cd_en = d2['energies']
axes[1].hist(cd_en, bins=15, color='mediumpurple', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('Energy (Hartree)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title(f'(b) Charged Dimer Energies (N={len(cd_en)})', fontsize=13)
axes[1].grid(True, alpha=0.3, axis='y')

pos_en = d3['1']['energies']
neg_en = d3['-1']['energies']
axes[2].hist(pos_en, bins=10, color='red', alpha=0.6, label='Charge +1', edgecolor='darkred')
axes[2].hist(neg_en, bins=10, color='blue', alpha=0.6, label='Charge -1', edgecolor='darkblue')
axes[2].set_xlabel('Energy (Hartree)', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('(c) Ag3 Energies by Charge State', fontsize=13)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# Figure 2: LES Concept
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

r = np.linspace(1.5, 10, 200)
sr_cutoff = 5.0
coulomb = 1/r
lj = 4*(1/r)**12 - 4*(1/r)**6
sr_mask = r <= sr_cutoff

axes[0].plot(r, coulomb, 'r-', linewidth=2, label='Coulomb (1/r)')
axes[0].plot(r, lj + 0.5, 'b-', linewidth=2, label='LJ + offset')
axes[0].axvline(x=sr_cutoff, color='gray', linestyle='--', linewidth=1.5, label=f'Cutoff ({sr_cutoff} A)')
axes[0].fill_between(r[sr_mask], -0.2, 1.5, alpha=0.1, color='green', label='Short-range')
axes[0].fill_between(r[~sr_mask], -0.2, 1.5, alpha=0.1, color='red', label='Long-range')
axes[0].set_xlabel('Distance (A)', fontsize=12)
axes[0].set_ylabel('Energy (arb.)', fontsize=12)
axes[0].set_title('(a) Short-range vs Long-range', fontsize=13)
axes[0].legend(fontsize=9, loc='upper right')
axes[0].set_ylim(-0.15, 1.2)
axes[0].grid(True, alpha=0.3)

r2 = np.linspace(1.5, 8, 200)
alpha = 0.5
real_part = erfc(alpha*r2)/r2
full = 1/r2
axes[1].plot(r2, full, 'k-', linewidth=2, label='Full Coulomb')
axes[1].plot(r2, real_part, 'b-', linewidth=2, label='Real-space (erfc)')
axes[1].plot(r2, full - real_part, 'r--', linewidth=2, label='Reciprocal-space')
axes[1].set_xlabel('Distance (A)', fontsize=12)
axes[1].set_ylabel('Interaction', fontsize=12)
axes[1].set_title('(b) Ewald Decomposition', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].text(0.5, 0.9, 'Atomic Config', ha='center', fontsize=12, fontweight='bold',
            transform=axes[2].transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
axes[2].annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.82), arrowprops=dict(arrowstyle='->', lw=2), xycoords='axes fraction')
axes[2].text(0.25, 0.65, 'Short-Range\nModel', ha='center', fontsize=11, fontweight='bold',
            transform=axes[2].transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
axes[2].text(0.75, 0.65, 'Latent Charge\nNetwork', ha='center', fontsize=11, fontweight='bold',
            transform=axes[2].transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[2].annotate('', xy=(0.5, 0.45), xytext=(0.25, 0.58), arrowprops=dict(arrowstyle='->', lw=2), xycoords='axes fraction')
axes[2].annotate('', xy=(0.5, 0.45), xytext=(0.75, 0.58), arrowprops=dict(arrowstyle='->', lw=2), xycoords='axes fraction')
axes[2].text(0.5, 0.35, 'Ewald\nSummation', ha='center', fontsize=11, fontweight='bold',
            transform=axes[2].transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
axes[2].annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.28), arrowprops=dict(arrowstyle='->', lw=2), xycoords='axes fraction')
axes[2].text(0.5, 0.05, 'E_total = E_SR + E_LR\nF, q_latent', ha='center', fontsize=12, fontweight='bold',
            transform=axes[2].transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[2].set_title('(c) LES Architecture', fontsize=13)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('report/images/fig2_les_concept.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# Figure 3: Charge Recovery
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].semilogy(d1['training_history']['loss'], 'b-', linewidth=1.5)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('(a) Training Convergence', fontsize=13)
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(d1['training_history']['charge_mae'], 'r-', linewidth=1.5)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Charge MAE (e)', fontsize=12)
axes[1].set_title('(b) Charge Recovery Error', fontsize=13)
axes[1].grid(True, alpha=0.3)

true_c = np.array(d1['true_charges'])
recovered_c = np.array(d1['recovered_charges'])
axes[2].scatter(true_c, recovered_c, alpha=0.6, s=20, c='steelblue', edgecolors='navy', linewidth=0.5)
lim = [min(true_c.min(), recovered_c.min()) - 0.1, max(true_c.max(), recovered_c.max()) + 0.1]
axes[2].plot(lim, lim, 'r--', linewidth=1.5, label='Perfect recovery')
axes[2].set_xlabel('True Charge (e)', fontsize=12)
axes[2].set_ylabel('Recovered Charge (e)', fontsize=12)
axes[2].set_title('(c) True vs Recovered', fontsize=13)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig3_charge_recovery.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# Figure 4: Binding Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

distances = np.array(d2['distances'])
energies = np.array(d2['energies'])
r_smooth = np.linspace(distances.min(), distances.max(), 200)

def exp_coulomb(r, A, B, C, D):
    return A*np.exp(-B*r) + C/r + D
popt, _ = curve_fit(exp_coulomb, distances, energies, p0=[1.0, 1.0, 0.5, 0.0], maxfev=5000)

axes[0].scatter(distances, energies, c='black', s=30, zorder=5, label='Reference data')
axes[0].plot(r_smooth, exp_coulomb(r_smooth, *popt), 'r-', linewidth=2, label='Exp+Coulomb fit')
axes[0].set_xlabel('Inter-dimer Distance (A)', fontsize=12)
axes[0].set_ylabel('Energy (Hartree)', fontsize=12)
axes[0].set_title('(a) Charged Dimer Binding Curve', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

pred = exp_coulomb(distances, *popt)
axes[1].scatter(distances, energies - pred, c='red', s=20, alpha=0.7, label='Exp+Coulomb residuals')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_xlabel('Inter-dimer Distance (A)', fontsize=12)
axes[1].set_ylabel('Residual (Hartree)', fontsize=12)
axes[1].set_title('(b) Fit Residuals', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig4_binding_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# Figure 5: Ag3 PES
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pos_bonds = np.array(d3['1']['bond_lengths'])
pos_en = np.array(d3['1']['energies'])
neg_bonds = np.array(d3['-1']['bond_lengths'])
neg_en = np.array(d3['-1']['energies'])

axes[0].scatter(pos_bonds, pos_en, c='red', s=30, label='Charge state +1', zorder=5)
axes[0].scatter(neg_bonds, neg_en, c='blue', s=30, marker='s', label='Charge state -1', zorder=5)

r_sm = np.linspace(min(pos_bonds.min(), neg_bonds.min()), max(pos_bonds.max(), neg_bonds.max()), 200)
def morse(r, De, a, re, E0):
    return De*(1-np.exp(-a*(r-re)))**2 + E0
popt_m, _ = curve_fit(morse, pos_bonds, pos_en, p0=[5.0, 1.0, 2.7, 0.0], maxfev=10000)
axes[0].plot(r_sm, morse(r_sm, *popt_m), 'r-', linewidth=2, alpha=0.7)
popt_m2, _ = curve_fit(morse, neg_bonds, neg_en, p0=[5.0, 1.0, 2.7, 0.0], maxfev=10000)
axes[0].plot(r_sm, morse(r_sm, *popt_m2), 'b--', linewidth=2, alpha=0.7)

axes[0].set_xlabel('Average Bond Length (A)', fontsize=12)
axes[0].set_ylabel('Energy (Hartree)', fontsize=12)
axes[0].set_title('(a) Ag3 Potential Energy Surface', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

common = np.linspace(max(pos_bonds.min(), neg_bonds.min()), min(pos_bonds.max(), neg_bonds.max()), 50)
pe_interp = np.interp(common, pos_bonds, pos_en)
ne_interp = np.interp(common, neg_bonds, neg_en)
diff = pe_interp - ne_interp
axes[1].plot(common, diff, 'g-', linewidth=2)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[1].fill_between(common, diff, alpha=0.3, color='green')
axes[1].set_xlabel('Average Bond Length (A)', fontsize=12)
axes[1].set_ylabel('Delta E = E(+1) - E(-1) (Hartree)', fontsize=12)
axes[1].set_title('(b) Energy Difference Between States', fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig5_ag3_pes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

print("All figures generated.")
