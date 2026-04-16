"""
Generate all figures for the report.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

os.makedirs('report/images', exist_ok=True)

# Load results
with open('outputs/all_results.json') as f:
    results = json.load(f)

data = np.load('outputs/plot_data.npz', allow_pickle=True)


# ============================================================
# Figure 1: Random Charges - Charge Recovery
# ============================================================
print("Generating Figure 1: Charge Recovery...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel a: True vs predicted charges scatter plot
true_charges = data['rc_true_charges']
pred_charges = data['rc_pred_charges']

# Subsample for clarity
np.random.seed(42)
idx = np.random.choice(len(true_charges), min(2000, len(true_charges)), replace=False)

axes[0].scatter(true_charges[idx], pred_charges[idx], alpha=0.3, s=10, c='steelblue')
axes[0].set_xlabel('True Charge (e)')
axes[0].set_ylabel('Latent Charge (e)')
axes[0].set_title('(a) Latent vs True Charges')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)

# Add correlation
corr = np.corrcoef(true_charges, pred_charges)[0, 1]
axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel b: Energy prediction comparison
rc_energies = data['rc_energies']
rc_test_idx = data['rc_test_idx']

# Compute per-structure MAE for test set
les_pred = data['rc_les_pred_e']
sr_pred = data['rc_sr_pred_e']

les_mae_per_struct = np.abs(les_pred - rc_energies[rc_test_idx])
sr_mae_per_struct = np.abs(sr_pred - rc_energies[rc_test_idx])

axes[1].hist(sr_mae_per_struct, bins=15, alpha=0.6, label='Short-Range Only', color='coral')
axes[1].hist(les_mae_per_struct, bins=15, alpha=0.6, label='LES', color='steelblue')
axes[1].set_xlabel('Energy MAE (eV)')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Energy Error Distribution')
axes[1].legend()

# Panel c: Charge correlation distribution
charge_corrs = data['rc_charge_correlations']
axes[2].hist(charge_corrs, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Charge Correlation (r)')
axes[2].set_ylabel('Count')
axes[2].set_title('(c) Charge Recovery Correlation')
axes[2].axvline(x=charge_corrs.mean(), color='red', linestyle='--', 
                label=f'Mean = {charge_corrs.mean():.3f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('report/images/fig1_charge_recovery.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# Figure 2: Charged Dimer - Binding Energy Curve
# ============================================================
print("Generating Figure 2: Binding Energy Curve...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cd_sep = data['cd_separations']
cd_e = data['cd_energies']
cd_les_e = data['cd_les_pred_e']
cd_sr_e = data['cd_sr_pred_e']
cd_test_idx = data['cd_test_idx']

# Sort by separation
sort_idx = np.argsort(cd_sep)

# Panel a: Energy vs separation
axes[0].scatter(cd_sep, cd_e, c='black', s=20, alpha=0.7, label='Reference', zorder=3)
axes[0].scatter(cd_sep, cd_les_e, c='steelblue', s=15, alpha=0.5, label='LES', marker='x')
axes[0].scatter(cd_sep, cd_sr_e, c='coral', s=15, alpha=0.5, label='Short-Range', marker='+')
axes[0].set_xlabel('Dimer Separation (Å)')
axes[0].set_ylabel('Total Energy (eV)')
axes[0].set_title('(a) Energy vs Separation')
axes[0].legend()

# Panel b: Energy error vs separation
les_error = np.abs(cd_les_e - cd_e)
sr_error = np.abs(cd_sr_e - cd_e)

axes[1].scatter(cd_sep, les_error, c='steelblue', s=15, alpha=0.5, label='LES', marker='x')
axes[1].scatter(cd_sep, sr_error, c='coral', s=15, alpha=0.5, label='Short-Range', marker='+')
axes[1].set_xlabel('Dimer Separation (Å)')
axes[1].set_ylabel('|Energy Error| (eV)')
axes[1].set_title('(b) Energy Error vs Separation')
axes[1].legend()

plt.tight_layout()
plt.savefig('report/images/fig2_dimer_binding.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# Figure 3: Ag3 Charge States - PES Discrimination
# ============================================================
print("Generating Figure 3: Ag3 Charge States...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ag_e = data['ag_energies']
ag_sr_e = data['ag_sr_pred_e']
ag_ce_e = data['ag_les_ce_pred_e']
ag_cs = data['ag_charge_states']
ag_bl = data['ag_bond_lengths']

# Use average bond length as x-axis
avg_bl = ag_bl.mean(axis=1)

pos_mask = ag_cs == 1
neg_mask = ag_cs == -1

# Panel a: Reference PES
axes[0].scatter(avg_bl[pos_mask], ag_e[pos_mask], c='red', s=30, alpha=0.7, label='q = +1', marker='o')
axes[0].scatter(avg_bl[neg_mask], ag_e[neg_mask], c='blue', s=30, alpha=0.7, label='q = -1', marker='s')
axes[0].set_xlabel('Average Bond Length (Å)')
axes[0].set_ylabel('Energy (eV)')
axes[0].set_title('(a) Reference PES')
axes[0].legend()

# Panel b: SR-only model (cannot distinguish)
axes[1].scatter(avg_bl[pos_mask], ag_sr_e[pos_mask], c='red', s=30, alpha=0.7, label='q = +1', marker='o')
axes[1].scatter(avg_bl[neg_mask], ag_sr_e[neg_mask], c='blue', s=30, alpha=0.7, label='q = -1', marker='s')
axes[1].set_xlabel('Average Bond Length (Å)')
axes[1].set_ylabel('Predicted Energy (eV)')
axes[1].set_title('(b) Short-Range Only')
axes[1].legend()

# Panel c: LES + Charge Embedding
axes[2].scatter(avg_bl[pos_mask], ag_ce_e[pos_mask], c='red', s=30, alpha=0.7, label='q = +1', marker='o')
axes[2].scatter(avg_bl[neg_mask], ag_ce_e[neg_mask], c='blue', s=30, alpha=0.7, label='q = -1', marker='s')
axes[2].set_xlabel('Average Bond Length (Å)')
axes[2].set_ylabel('Predicted Energy (eV)')
axes[2].set_title('(c) LES + Charge Embedding')
axes[2].legend()

plt.tight_layout()
plt.savefig('report/images/fig3_ag3_chargestates.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# Figure 4: Model Comparison Summary
# ============================================================
print("Generating Figure 4: Model Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel a: Energy MAE comparison across datasets
datasets = ['Random\nCharges', 'Charged\nDimer', 'Ag3\n(+SR)', 'Ag3\n(+LES-CE)']
les_maes = [results['exp1']['les_test_mae'], results['exp2']['les_test_mae'],
            results['exp3']['sr_test_mae'], results['exp3']['les_ce_test_mae']]
sr_maes = [results['exp1']['sr_test_mae'], results['exp2']['sr_test_mae'],
           results['exp3']['sr_test_mae'], results['exp3']['sr_test_mae']]

x = np.arange(len(datasets))
width = 0.35

bars1 = axes[0].bar(x - width/2, les_maes, width, label='LES', color='steelblue', alpha=0.8)
bars2 = axes[0].bar(x + width/2, sr_maes, width, label='Short-Range', color='coral', alpha=0.8)

axes[0].set_ylabel('Energy MAE (eV)')
axes[0].set_title('(a) Energy Prediction Accuracy')
axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets, fontsize=9)
axes[0].legend()

# Panel b: Charge discrimination
models = ['SR-Only', 'LES+CE', 'Reference']
discriminations = [results['exp3']['sr_discrimination'],
                   results['exp3']['les_ce_discrimination'],
                   results['exp3']['ref_discrimination']]

bars = axes[1].bar(models, discriminations, color=['coral', 'steelblue', 'gray'], alpha=0.8)
axes[1].set_ylabel('Mean |ΔE| between charge states (eV)')
axes[1].set_title('(b) Charge State Discrimination')

# Panel c: Charge recovery correlation
axes[2].hist(charge_corrs, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
axes[2].axvline(x=0, color='red', linestyle='--', label='Zero correlation')
axes[2].axvline(x=1, color='green', linestyle='--', label='Perfect correlation')
axes[2].set_xlabel('Charge Correlation (r)')
axes[2].set_ylabel('Count')
axes[2].set_title('(c) Latent Charge Recovery')
axes[2].legend()

plt.tight_layout()
plt.savefig('report/images/fig4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# Figure 5: Training Curves
# ============================================================
print("Generating Figure 5: Training Curves...")

with open('outputs/histories.json') as f:
    histories = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Charged Dimer
if 'les_cd' in histories and histories['les_cd']['loss']:
    h = histories['les_cd']
    axes[0, 0].plot(h['e_mae'], label='LES', color='steelblue')
    if h.get('test_e_mae'):
        n_train = len(h['e_mae'])
        log_interval = max(1, n_train // len(h['test_e_mae']))
        epochs = list(range(log_interval, n_train + 1, log_interval))
        axes[0, 0].plot(epochs[:len(h['test_e_mae'])], h['test_e_mae'], 
                       '--', label='LES (test)', color='steelblue')

if 'sr_cd' in histories and histories['sr_cd']['loss']:
    h = histories['sr_cd']
    axes[0, 0].plot(h['e_mae'], label='SR', color='coral')
    if h.get('test_e_mae'):
        n_train = len(h['e_mae'])
        log_interval = max(1, n_train // len(h['test_e_mae']))
        epochs = list(range(log_interval, n_train + 1, log_interval))
        axes[0, 0].plot(epochs[:len(h['test_e_mae'])], h['test_e_mae'],
                       '--', label='SR (test)', color='coral')

axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Energy MAE (eV)')
axes[0, 0].set_title('(a) Charged Dimer - Energy')
axes[0, 0].legend()

# Ag3
if 'sr_ag' in histories and histories['sr_ag']['loss']:
    h = histories['sr_ag']
    axes[0, 1].plot(h['e_mae'], label='SR', color='coral')

if 'les_ce_ag' in histories and histories['les_ce_ag']['loss']:
    h = histories['les_ce_ag']
    axes[0, 1].plot(h['e_mae'], label='LES+CE', color='steelblue')

axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Energy MAE (eV)')
axes[0, 1].set_title('(b) Ag3 - Energy')
axes[0, 1].legend()

# Force MAE
if 'les_cd' in histories and histories['les_cd']['f_mae']:
    h = histories['les_cd']
    axes[1, 0].plot(h['f_mae'], label='LES', color='steelblue')

if 'sr_cd' in histories and histories['sr_cd']['f_mae']:
    h = histories['sr_cd']
    axes[1, 0].plot(h['f_mae'], label='SR', color='coral')

axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Force MAE (eV/Å)')
axes[1, 0].set_title('(c) Charged Dimer - Forces')
axes[1, 0].legend()

# Random charges
if 'les_rc' in histories and histories['les_rc']['e_mae']:
    h = histories['les_rc']
    axes[1, 1].plot(h['e_mae'], label='LES', color='steelblue')

if 'sr_rc' in histories and histories['sr_rc']['e_mae']:
    h = histories['sr_rc']
    axes[1, 1].plot(h['e_mae'], label='SR', color='coral')

axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Energy MAE (eV)')
axes[1, 1].set_title('(d) Random Charges - Energy')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('report/images/fig5_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# Figure 6: Energy Parity Plots
# ============================================================
print("Generating Figure 6: Energy Parity Plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Random Charges
test_e = rc_energies[rc_test_idx]
axes[0].scatter(test_e, les_pred, c='steelblue', s=30, alpha=0.7, label='LES')
axes[0].scatter(test_e, sr_pred, c='coral', s=20, alpha=0.5, label='SR')
lims = [min(test_e.min(), min(les_pred.min(), sr_pred.min())),
        max(test_e.max(), max(les_pred.max(), sr_pred.max()))]
axes[0].plot(lims, lims, 'k--', alpha=0.5)
axes[0].set_xlabel('Reference Energy (eV)')
axes[0].set_ylabel('Predicted Energy (eV)')
axes[0].set_title('(a) Random Charges')
axes[0].legend()

# Charged Dimer
axes[1].scatter(cd_e, cd_les_e, c='steelblue', s=30, alpha=0.7, label='LES')
axes[1].scatter(cd_e, cd_sr_e, c='coral', s=20, alpha=0.5, label='SR')
lims = [min(cd_e.min(), cd_les_e.min(), cd_sr_e.min()),
        max(cd_e.max(), cd_les_e.max(), cd_sr_e.max())]
axes[1].plot(lims, lims, 'k--', alpha=0.5)
axes[1].set_xlabel('Reference Energy (eV)')
axes[1].set_ylabel('Predicted Energy (eV)')
axes[1].set_title('(b) Charged Dimer')
axes[1].legend()

# Ag3
axes[2].scatter(ag_e, ag_ce_e, c='steelblue', s=30, alpha=0.7, label='LES+CE')
axes[2].scatter(ag_e, ag_sr_e, c='coral', s=20, alpha=0.5, label='SR')
lims = [min(ag_e.min(), ag_ce_e.min(), ag_sr_e.min()),
        max(ag_e.max(), ag_ce_e.max(), ag_sr_e.max())]
axes[2].plot(lims, lims, 'k--', alpha=0.5)
axes[2].set_xlabel('Reference Energy (eV)')
axes[2].set_ylabel('Predicted Energy (eV)')
axes[2].set_title('(c) Ag3 Charge States')
axes[2].legend()

plt.tight_layout()
plt.savefig('report/images/fig6_parity_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("All figures generated!")
