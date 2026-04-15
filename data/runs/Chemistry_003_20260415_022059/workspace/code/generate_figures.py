"""Generate all figures for the research report."""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(base, 'report/images'), exist_ok=True)

# Load results
with open(os.path.join(base, 'outputs/exp1_random_charges_results.json')) as f:
    exp1 = json.load(f)
with open(os.path.join(base, 'outputs/exp2_charged_dimer_results.json')) as f:
    exp2 = json.load(f)
with open(os.path.join(base, 'outputs/exp3_ag3_results.json')) as f:
    exp3 = json.load(f)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# ============================================================
# Figure 1: Random Charges - Energy prediction comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ref_e = np.array(exp1['reference_energy_stats'])
sr_pred = np.array(exp1['sr_predictions_all'])
les_pred = np.array(exp1['les_predictions_all'])
# We need the actual reference energies - reconstruct from predictions + errors
# Use the stored stats to create a reasonable scatter
n = 100
np.random.seed(42)

# Load parsed data to get actual energies
import re
def parse_xyz(filepath):
    structures = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip().replace('\r', '')
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        comment = lines[i+1].strip().replace('\r', '')
        props = {}
        m = re.search(r'energy=([-\d.eE+]+)', comment)
        if m:
            props['energy'] = float(m.group(1))
        m = re.search(r'true_charges="([^"]*)"', comment)
        if m:
            props['true_charges'] = [float(x) for x in m.group(1).split()]
        has_forces = 'forces:R:3' in comment
        positions = []
        species = []
        forces = []
        for j in range(i+2, i+2+n_atoms):
            parts = lines[j].strip().replace('\r', '').split()
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if has_forces and len(parts) >= 7:
                forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
        struct = {
            'n_atoms': n_atoms, 'species': species,
            'positions': np.array(positions), 'props': props,
        }
        if has_forces and forces:
            struct['forces'] = np.array(forces)
        structures.append(struct)
        i = i + 2 + n_atoms
    return structures

def compute_coulomb_lj(positions, charges, sigma=1.0, epsilon=0.1, cutoff=None):
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if cutoff is not None and r > cutoff:
                continue
            e_coul = charges[i] * charges[j] / r
            sr6 = (sigma / r) ** 6
            e_lj = 4 * epsilon * sr6 * sr6
            energy += e_coul + e_lj
    return energy

rc_structures = parse_xyz(os.path.join(base, 'data/random_charges.xyz'))
ref_energies = []
for s in rc_structures:
    charges = np.array(s['props']['true_charges'])
    e = compute_coulomb_lj(s['positions'], charges)
    ref_energies.append(e)
ref_energies = np.array(ref_energies)

# Panel A: SR-only parity plot
ax = axes[0]
ax.scatter(ref_energies, sr_pred, alpha=0.7, s=20, color='steelblue', edgecolors='white', linewidth=0.5)
lims = [ref_energies.min()-2, ref_energies.max()+2]
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('Reference Energy (eV)')
ax.set_ylabel('Predicted Energy (eV)')
ax.set_title(f'SR-Only Model\nMAE={exp1["sr_only"]["mae"]:.2f}, R²={exp1["sr_only"]["r2"]:.2f}')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Panel B: LES-augmented parity plot
ax = axes[1]
ax.scatter(ref_energies, les_pred, alpha=0.7, s=20, color='darkorange', edgecolors='white', linewidth=0.5)
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('Reference Energy (eV)')
ax.set_ylabel('Predicted Energy (eV)')
ax.set_title(f'LES-Augmented Model\nMAE={exp1["les_augmented"]["mae"]:.2f}, R²={exp1["les_augmented"]["r2"]:.2f}')
ax.set_xlim(lims)
ax.set_ylim(lims)

# Panel C: Error distribution
ax = axes[2]
sr_errors = sr_pred - ref_energies
les_errors = les_pred - ref_energies
ax.hist(sr_errors, bins=20, alpha=0.6, color='steelblue', label=f'SR-Only (σ={np.std(sr_errors):.2f})')
ax.hist(les_errors, bins=20, alpha=0.6, color='darkorange', label=f'LES (σ={np.std(les_errors):.2f})')
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Prediction Error (eV)')
ax.set_ylabel('Count')
ax.set_title('Error Distribution')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(base, 'report/images/fig1_random_charges_energy.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_random_charges_energy.png")

# ============================================================
# Figure 2: Latent Charge Recovery
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: True vs latent charges for one structure
ax = axes[0]
true_q = np.array(exp1['true_charges_sample'][0])
latent_q = np.array(exp1['latent_charges_sample'][0])
n_atoms = len(true_q)
ax.scatter(range(n_atoms), true_q, c=true_q, cmap='RdBu_r', s=30, zorder=3, label='True charges', vmin=-1.5, vmax=1.5)
ax.scatter(range(n_atoms), latent_q, marker='x', s=40, color='black', linewidth=1.5, zorder=4, label='Latent charges')
ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
ax.set_xlabel('Atom Index')
ax.set_ylabel('Charge (e)')
ax.set_title(f'Charge Recovery (Structure 0)\nCorrelation = {exp1["latent_charge_recovery"]["per_structure_correlations"][0]:.3f}')
ax.legend()

# Panel B: Correlation histogram
ax = axes[1]
corrs = exp1['latent_charge_recovery']['per_structure_correlations']
ax.hist(corrs, bins=15, alpha=0.7, color='teal', edgecolor='white')
ax.axvline(np.mean(corrs), color='red', linestyle='--', linewidth=1.5, 
           label=f'Mean r = {np.mean(corrs):.3f}')
ax.set_xlabel('Correlation Coefficient (r)')
ax.set_ylabel('Count')
ax.set_title('Latent vs True Charge Correlation\nAcross Structures')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(base, 'report/images/fig2_latent_charge_recovery.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_latent_charge_recovery.png")

# ============================================================
# Figure 3: Charged Dimer - Binding Energy Curves
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

separations = np.array(exp2['separations'])
energies = np.array(exp2['energies'])
sr_pred_all = np.array(exp2['sr_predictions_all'])
les_pred_all = np.array(exp2['les_predictions_all'])

# Sort by separation
sort_idx = np.argsort(separations)
sep_sorted = separations[sort_idx]
eng_sorted = energies[sort_idx]
sr_sorted = sr_pred_all[sort_idx]
les_sorted = les_pred_all[sort_idx]

# Panel A: Reference binding curve
ax = axes[0]
ax.scatter(sep_sorted, eng_sorted, c=eng_sorted, cmap='viridis', s=30, zorder=3)
ax.set_xlabel('Dimer Separation (Å)')
ax.set_ylabel('Energy (eV)')
ax.set_title('Reference Binding Energy Curve')

# Panel B: SR-only predictions
ax = axes[1]
ax.scatter(sep_sorted, eng_sorted, alpha=0.5, s=20, color='gray', label='Reference')
ax.plot(sep_sorted, sr_sorted, 'o-', color='steelblue', markersize=4, linewidth=1, alpha=0.7, label='SR-Only')
ax.set_xlabel('Dimer Separation (Å)')
ax.set_ylabel('Energy (eV)')
ax.set_title(f'SR-Only Prediction\nMAE={exp2["sr_only"]["mae"]:.3f}, R²={exp2["sr_only"]["r2"]:.3f}')
ax.legend()

# Panel C: LES-augmented predictions
ax = axes[2]
ax.scatter(sep_sorted, eng_sorted, alpha=0.5, s=20, color='gray', label='Reference')
ax.plot(sep_sorted, les_sorted, 'o-', color='darkorange', markersize=4, linewidth=1, alpha=0.7, label='LES-Augmented')
ax.set_xlabel('Dimer Separation (Å)')
ax.set_ylabel('Energy (eV)')
ax.set_title(f'LES-Augmented Prediction\nMAE={exp2["les_augmented"]["mae"]:.3f}, R²={exp2["les_augmented"]["r2"]:.3f}')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(base, 'report/images/fig3_charged_dimer_binding.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_charged_dimer_binding.png")

# ============================================================
# Figure 4: Ag3 Charge States
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ag3_energies = np.array(exp3['energies'])
ag3_cs = np.array(exp3['charge_states'])
ag3_sr_pred = np.array(exp3['sr_predictions_all'])
ag3_les_pred = np.array(exp3['les_predictions_all'])

mask_pos = ag3_cs == 1
mask_neg = ag3_cs == -1

# Panel A: Energy distributions by charge state
ax = axes[0]
ax.hist(ag3_energies[mask_pos], bins=15, alpha=0.6, color='red', label='Charge +1', edgecolor='white')
ax.hist(ag3_energies[mask_neg], bins=15, alpha=0.6, color='blue', label='Charge -1', edgecolor='white')
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Count')
ax.set_title('Energy Distribution by Charge State')
ax.legend()

# Panel B: SR-only predictions
ax = axes[1]
ax.scatter(range(len(ag3_energies)), ag3_energies, c=['red' if c == 1 else 'blue' for c in ag3_cs], 
           s=30, alpha=0.7, label='Reference')
ax.plot(range(len(ag3_energies)), ag3_sr_pred, 'ko-', markersize=3, linewidth=0.5, alpha=0.5, label='SR-Only')
ax.set_xlabel('Configuration Index')
ax.set_ylabel('Energy (eV)')
ax.set_title(f'SR-Only (no charge info)\nMAE={exp3["sr_only"]["mae"]:.4f}')
ax.legend()

# Panel C: LES+global charge predictions
ax = axes[2]
ax.scatter(range(len(ag3_energies)), ag3_energies, c=['red' if c == 1 else 'blue' for c in ag3_cs], 
           s=30, alpha=0.7, label='Reference')
ax.plot(range(len(ag3_energies)), ag3_les_pred, 'ko-', markersize=3, linewidth=0.5, alpha=0.5, label='LES+Charge')
ax.set_xlabel('Configuration Index')
ax.set_ylabel('Energy (eV)')
ax.set_title(f'LES+Global Charge\nMAE={exp3["les_augmented"]["mae"]:.4f}')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(base, 'report/images/fig4_ag3_charge_states.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_ag3_charge_states.png")

# ============================================================
# Figure 5: Summary comparison table visualization
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

# Create summary table
models = ['SR-Only', 'LES-Augmented']
datasets = ['Random Charges\n(Energy MAE)', 'Charged Dimer\n(Energy MAE)', 'Ag3 Charge States\n(Energy MAE)']

data = [
    [f'{exp1["sr_only"]["mae"]:.2f}', f'{exp2["sr_only"]["mae"]:.3f}', f'{exp3["sr_only"]["mae"]:.4f}'],
    [f'{exp1["les_augmented"]["mae"]:.2f}', f'{exp2["les_augmented"]["mae"]:.3f}', f'{exp3["les_augmented"]["mae"]:.4f}'],
]

table = ax.table(cellText=data, rowLabels=models, colLabels=datasets,
                 cellLoc='center', loc='center',
                 colWidths=[0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color cells
for i in range(2):
    for j in range(3):
        cell = table[i+1, j]
        if i == 0:
            cell.set_facecolor('#e8f4fd')
        else:
            cell.set_facecolor('#fff3e0')

ax.set_title('Model Performance Comparison Across Datasets', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(base, 'report/images/fig5_summary_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_summary_comparison.png")

# ============================================================
# Figure 6: Long-range contribution analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

trunc_mean = exp1['truncation_error']['mean']
trunc_std = exp1['truncation_error']['std']

ax = axes[0]
ax.bar(['Full Coulomb+LJ', 'Truncated (5Å cutoff)'], 
       [exp1['reference_energy_stats']['mean'], 
        exp1['reference_energy_stats']['mean'] - trunc_mean],
       yerr=[exp1['reference_energy_stats']['std'], trunc_std],
       color=['steelblue', 'darkorange'], alpha=0.7, edgecolor='white')
ax.set_ylabel('Mean Energy (eV)')
ax.set_title('Effect of Distance Cutoff on Energy')

ax = axes[1]
# Show how much of the energy is from long-range (>5A) interactions
total_std = exp1['reference_energy_stats']['std']
lr_fraction = abs(trunc_mean) / (abs(exp1['reference_energy_stats']['mean']) + 1e-10) * 100
ax.bar(['Long-Range\nContribution'], [lr_fraction], color='crimson', alpha=0.7, width=0.4)
ax.set_ylabel('Fraction of Total Energy (%)')
ax.set_title(f'Long-Range Energy Contribution\n(mean={trunc_mean:.2f} eV, {lr_fraction:.1f}% of total)')

plt.tight_layout()
plt.savefig(os.path.join(base, 'report/images/fig6_longrange_contribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_longrange_contribution.png")

print("\nAll figures generated!")
