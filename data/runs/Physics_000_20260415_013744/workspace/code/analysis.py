#!/usr/bin/env python3
"""
Analysis of Multi-component Icosahedral Shell Stacking Theory
Reproduces key results from the reproduction data file.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

# Ensure output directories exist
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

###############################################################################
# 1. Parse data from the reproduction file
###############################################################################

# --- Core Theory ---
mackay_sequence = [1, 13, 55, 147, 309]
new_sequence_b5 = [1, 13, 45, 117, 239, 431]
chiral_labels = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
shell_colors = {'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c',
                'Ch2': '#d62728', 'Ch3': '#9467bd', 'Ch4': '#8c564b', 'Ch5': '#e377c2'}

# --- Experimental Verification ---
atomic_radii = [('Na', 1.86), ('K', 2.27), ('Rb', 2.48), ('Cs', 2.65),
                ('Ag', 1.44), ('Cu', 1.28), ('Ni', 1.24)]

atomic_pairs_compatibility = [('Na', 'Rb', 0.22), ('Ag', 'Cu', 0.12),
                               ('Ag', 'Ni', 0.15), ('Cu', 'Ni', 0.032)]

optimal_mismatch_ranges = [('MC', 'MC', 0.03, 0.05), ('MC', 'Ch1', 0.12, 0.16),
                            ('MC', 'Ch2', 0.19, 0.22), ('MC', 'BG', 0.08, 0.10)]

multicomponent_clusters = [('Na13@Rb32', 'Na', 'Rb', 'MC', 'Ch1'),
                            ('K13@Cs42', 'K', 'Cs', 'MC', 'Ch2'),
                            ('Ag13@Cu45', 'Ag', 'Cu', 'MC', 'Ch1')]

shell_energies = [(1, 'MC', 0.00), (2, 'MC', -2.35), (2, 'Ch1', -2.15),
                  (3, 'MC', -4.82), (3, 'Ch1', -4.61), (3, 'BG', -4.55)]

mismatch_params = [(1, 2, 'MC', 'MC', 0.04), (1, 2, 'MC', 'Ch1', 0.14),
                   (2, 3, 'MC', 'MC', 0.038), (2, 3, 'MC', 'Ch1', 0.136),
                   (2, 3, 'Ch1', 'Ch2', 0.21)]

experimental_points = [(1, 3, 0.048, 0.045), (3, 4, 0.042, 0.044),
                        (4, 7, 0.138, 0.142), (7, 12, 0.132, 0.139)]

# --- Growth Simulation ---
growth_parameters = [('temperature', 300.0), ('deposition_rate', 0.01),
                     ('simulation_steps', 1000), ('beta_factor', 1.0),
                     ('delta_opt', 0.04), ('random_seed', 42)]

path_probability_weights = [('conservative_step', 0.65), ('mismatch_driven_step', 0.25),
                             ('random_step', 0.10)]

growth_results = [(0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02),
                  (30, 'MC', 0.025), (40, 'MC', 0.03), (50, 'MC', 0.035),
                  (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14),
                  (30, 'Ch1', 0.138), (40, 'Ch1', 0.136), (50, 'Ch1', 0.135),
                  (0, 'MC', 0.00), (10, 'MC', 0.08), (20, 'Ch1', 0.14),
                  (30, 'Ch1', 0.15), (40, 'Ch1', 0.145), (50, 'Ch1', 0.142)]

path_selection_stats = [('Conservative path', 325), ('Mismatch-driven path', 125),
                         ('Random path', 50), ('Reverse step', 100)]

lj_parameters = [('Na-Na', 1.0, 3.72), ('Rb-Rb', 1.0, 4.96), ('Cs-Cs', 1.0, 5.30),
                  ('Ag-Ag', 1.0, 2.88), ('Cu-Cu', 1.0, 2.56), ('Na-Rb', 1.0, 4.34),
                  ('Ag-Cu', 1.0, 2.72)]

thermodynamic_params = [('kT', 0.02585), ('boltzmann', 8.617e-5),
                         ('pressure', 1.0), ('timestep', 0.001)]

###############################################################################
# 2. Save parsed data to outputs/
###############################################################################

parsed = {
    'mackay_sequence': mackay_sequence,
    'new_sequence_b5': new_sequence_b5,
    'chiral_labels': chiral_labels,
    'atomic_radii': atomic_radii,
    'atomic_pairs_compatibility': atomic_pairs_compatibility,
    'optimal_mismatch_ranges': optimal_mismatch_ranges,
    'multicomponent_clusters': multicomponent_clusters,
    'shell_energies': shell_energies,
    'mismatch_params': mismatch_params,
    'experimental_points': experimental_points,
    'growth_parameters': growth_parameters,
    'growth_results': growth_results,
    'path_selection_stats': path_selection_stats,
    'lj_parameters': lj_parameters,
}

with open('outputs/parsed_data.json', 'w') as f:
    json.dump(parsed, f, indent=2)
print("Saved parsed_data.json")

###############################################################################
# 3. Figure 1: Magic number sequences comparison
###############################################################################

fig, ax = plt.subplots(figsize=(8, 5))
shells = np.arange(len(mackay_sequence))
shells_b5 = np.arange(len(new_sequence_b5))
ax.plot(shells, mackay_sequence, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Mackay (b=6)')
ax.plot(shells_b5, new_sequence_b5, 's--', color='#d62728', linewidth=2, markersize=8, label='New sequence (b=5)')
ax.set_xlabel('Shell index', fontsize=13)
ax.set_ylabel('Number of atoms', fontsize=13)
ax.set_title('Icosahedral Magic Number Sequences', fontsize=14)
ax.legend(fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig1_magic_numbers.png', dpi=150)
plt.close()
print("Saved fig1_magic_numbers.png")

###############################################################################
# 4. Figure 2: Atomic radii and size mismatch
###############################################################################

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: atomic radii
elements = [a[0] for a in atomic_radii]
radii = [a[1] for a in atomic_radii]
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
axes[0].bar(elements, radii, color=colors_bar, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('Atomic radius (Å)', fontsize=12)
axes[0].set_title('Atomic Radii of Selected Elements', fontsize=13)
for i, r in enumerate(radii):
    axes[0].text(i, r + 0.03, f'{r:.2f}', ha='center', fontsize=10)

# Right: optimal mismatch ranges
pairs = [f"{r[0]}-{r[1]}" for r in optimal_mismatch_ranges]
low = [r[2] for r in optimal_mismatch_ranges]
high = [r[3] for r in optimal_mismatch_ranges]
mid = [(l + h) / 2 for l, h in zip(low, high)]
err_low = [m - l for m, l in zip(mid, low)]
err_high = [h - m for h, m in zip(high, mid)]
cat_colors = [shell_colors.get(r[0], '#333') for r in optimal_mismatch_ranges]
axes[1].barh(pairs, mid, xerr=[err_low, err_high], color=cat_colors,
             edgecolor='black', linewidth=0.5, capsize=4)
axes[1].set_xlabel('Size mismatch δ', fontsize=12)
axes[1].set_title('Optimal Size Mismatch Ranges', fontsize=13)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('report/images/fig2_atomic_radii_mismatch.png', dpi=150)
plt.close()
print("Saved fig2_atomic_radii_mismatch.png")

###############################################################################
# 5. Figure 3: Shell energies by chiral category
###############################################################################

fig, ax = plt.subplots(figsize=(8, 5))
# Group by shell and category
shell_set = sorted(set(s[0] for s in shell_energies))
cat_set = sorted(set(s[1] for s in shell_energies))
width = 0.25
for ci, cat in enumerate(cat_set):
    vals = []
    for sh in shell_set:
        found = [e[2] for e in shell_energies if e[0] == sh and e[1] == cat]
        vals.append(found[0] if found else None)
    x = np.arange(len(shell_set)) + ci * width
    valid = [(xi, vi) for xi, vi in zip(x, vals) if vi is not None]
    if valid:
        xv, vv = zip(*valid)
        ax.bar(xv, vv, width, label=cat, color=shell_colors.get(cat, '#333'),
               edgecolor='black', linewidth=0.5)
ax.set_xticks(np.arange(len(shell_set)) + width)
ax.set_xticklabels([f'Shell {s}' for s in shell_set])
ax.set_ylabel('Relative shell energy (normalized)', fontsize=12)
ax.set_title('Shell Energies by Chiral Category', fontsize=14)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig3_shell_energies.png', dpi=150)
plt.close()
print("Saved fig3_shell_energies.png")

###############################################################################
# 6. Figure 4: Experimental validation — measured vs theoretical mismatch
###############################################################################

fig, ax = plt.subplots(figsize=(7, 6))
measured = [p[2] for p in experimental_points]
theoretical = [p[3] for p in experimental_points]
labels_exp = [f"T={p[0]}→{p[1]}" for p in experimental_points]
ax.scatter(theoretical, measured, s=120, c='#d62728', edgecolors='black', zorder=5)
for i, lbl in enumerate(labels_exp):
    ax.annotate(lbl, (theoretical[i], measured[i]), textcoords="offset points",
                xytext=(8, 5), fontsize=10)
# Perfect agreement line
lims = [min(measured + theoretical) - 0.01, max(measured + theoretical) + 0.01]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
ax.set_xlabel('Theoretical size mismatch', fontsize=12)
ax.set_ylabel('Measured size mismatch', fontsize=12)
ax.set_title('Experimental Validation: Measured vs Theoretical Mismatch', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig4_validation.png', dpi=150)
plt.close()
print("Saved fig4_validation.png")

###############################################################################
# 7. Figure 5: Growth simulation trajectories
###############################################################################

fig, ax = plt.subplots(figsize=(9, 5))
# Parse growth_results into separate trajectories
trajectories = {}
for step, cat, mismatch in growth_results:
    key = cat
    if key not in trajectories:
        trajectories[key] = {'steps': [], 'mismatch': []}
    trajectories[key]['steps'].append(step)
    trajectories[key]['mismatch'].append(mismatch)

# We have two MC runs and one Ch1 run; label them
run_labels = ['MC run 1', 'Ch1 run', 'MC→Ch1 transition']
run_colors = ['#1f77b4', '#2ca02c', '#d62728']
# Split growth_results into 3 runs of 6 points each
runs = [growth_results[i:i+6] for i in range(0, len(growth_results), 6)]
for ri, run in enumerate(runs):
    steps = [r[0] for r in run]
    mismatches = [r[2] for r in run]
    cats = [r[1] for r in run]
    ax.plot(steps, mismatches, 'o-', color=run_colors[ri], linewidth=2,
            markersize=7, label=run_labels[ri])
    # Annotate category transitions
    for i in range(1, len(cats)):
        if cats[i] != cats[i-1]:
            ax.annotate(f'→{cats[i]}', (steps[i], mismatches[i]),
                        textcoords="offset points", xytext=(5, 10),
                        fontsize=9, color=run_colors[ri], fontweight='bold')

ax.set_xlabel('Simulation step', fontsize=12)
ax.set_ylabel('Average size mismatch δ', fontsize=12)
ax.set_title('Growth Simulation Trajectories', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig5_growth_trajectories.png', dpi=150)
plt.close()
print("Saved fig5_growth_trajectories.png")

###############################################################################
# 8. Figure 6: Path selection statistics
###############################################################################

fig, ax = plt.subplots(figsize=(7, 5))
paths = [p[0] for p in path_selection_stats]
counts = [p[1] for p in path_selection_stats]
total = sum(counts)
pcts = [c / total * 100 for c in counts]
colors_pie = ['#1f77b4', '#ff770e', '#2ca02c', '#d62728']
wedges, texts, autotexts = ax.pie(counts, labels=paths, autopct='%1.1f%%',
                                   colors=colors_pie, startangle=140,
                                   textprops={'fontsize': 10})
ax.set_title('Path Selection Statistics in Growth Simulations', fontsize=13)
plt.tight_layout()
plt.savefig('report/images/fig6_path_selection.png', dpi=150)
plt.close()
print("Saved fig6_path_selection.png")

###############################################################################
# 9. Figure 7: Multi-component cluster overview
###############################################################################

fig, ax = plt.subplots(figsize=(9, 5))
cluster_names = [c[0] for c in multicomponent_clusters]
inner_elements = [c[1] for c in multicomponent_clusters]
outer_elements = [c[2] for c in multicomponent_clusters]
inner_cats = [c[3] for c in multicomponent_clusters]
outer_cats = [c[4] for c in multicomponent_clusters]

# Create a grouped bar showing radii
x = np.arange(len(cluster_names))
inner_r = [dict(atomic_radii)[e] for e in inner_elements]
outer_r = [dict(atomic_radii)[e] for e in outer_elements]
width = 0.35
bars1 = ax.bar(x - width/2, inner_r, width, label='Inner shell', color='#1f77b4',
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, outer_r, width, label='Outer shell', color='#d62728',
               edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(cluster_names, fontsize=11)
ax.set_ylabel('Atomic radius (Å)', fontsize=12)
ax.set_title('Multi-component Icosahedral Clusters: Shell Radii', fontsize=13)
ax.legend(fontsize=11)
# Annotate mismatch
for i, name in enumerate(cluster_names):
    delta = (outer_r[i] - inner_r[i]) / inner_r[i]
    ax.text(i, max(inner_r[i], outer_r[i]) + 0.08, f'δ={delta:.3f}',
            ha='center', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig7_clusters.png', dpi=150)
plt.close()
print("Saved fig7_clusters.png")

###############################################################################
# 10. Save summary statistics
###############################################################################

summary = {
    'mackay_sequence': mackay_sequence,
    'new_sequence_b5': new_sequence_b5,
    'num_atomic_species': len(atomic_radii),
    'num_validated_clusters': len(multicomponent_clusters),
    'optimal_mismatch_ranges': {f"{r[0]}-{r[1]}": {'min': r[2], 'max': r[3]} for r in optimal_mismatch_ranges},
    'experimental_validation_rmse': float(np.sqrt(np.mean([(m - t)**2 for m, t in zip(measured, theoretical)]))),
    'path_selection_percentages': {p[0]: round(p[1]/total*100, 1) for p in path_selection_stats},
    'growth_simulation_runs': len(runs),
}

with open('outputs/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved summary_statistics.json")
print("\nAll analysis complete.")
