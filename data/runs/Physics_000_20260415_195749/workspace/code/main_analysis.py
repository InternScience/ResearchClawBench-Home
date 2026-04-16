"""
Multi-component Icosahedral Shell Stacking Theory: Complete Analysis
Reproduces theoretical calculations, experimental verification, and dynamic growth simulations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
import json
import math
import os

# Set up paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_000_20260415_195749'
IMAGES_DIR = os.path.join(WORKSPACE, 'report', 'images')
OUTPUTS_DIR = os.path.join(WORKSPACE, 'outputs')
os.makedirs(IMAGES_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
})

# ============================================================
# 1. DATA DEFINITIONS (from reproduction data)
# ============================================================

hexagonal_coords = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5),
                    (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),
                    (2,0), (2,1), (2,2), (2,3), (2,4), (2,5),
                    (3,0), (3,1), (3,2), (3,3), (3,4), (3,5),
                    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5),
                    (5,0), (5,1), (5,2), (5,3), (5,4), (5,5)]

mackay_sequence = [1, 13, 55, 147, 309]
new_sequence_b5 = [1, 13, 45, 117, 239, 431]

chiral_labels = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']

shell_colors = {'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c',
                'Ch2': '#d62728', 'Ch3': '#9467bd', 'Ch4': '#8c564b',
                'Ch5': '#e377c2'}

sin_2pi_5 = 0.9510565162951535
cos_2pi_5 = 0.3090169943749474

atomic_radii_dict = {'Na': 1.86, 'K': 2.27, 'Rb': 2.48, 'Cs': 2.65,
                     'Ag': 1.44, 'Cu': 1.28, 'Ni': 1.24}

atomic_pairs_compatibility = [('Na', 'Rb', 0.22), ('Ag', 'Cu', 0.12),
                               ('Ag', 'Ni', 0.15), ('Cu', 'Ni', 0.032)]

optimal_mismatch_ranges = [('MC', 'MC', 0.03, 0.05),
                           ('MC', 'Ch1', 0.12, 0.16),
                           ('MC', 'Ch2', 0.19, 0.22),
                           ('MC', 'BG', 0.08, 0.10)]

multicomponent_clusters = [('Na13@Rb32', 'Na', 'Rb', 'MC', 'Ch1'),
                            ('K13@Cs42', 'K', 'Cs', 'MC', 'Ch2'),
                            ('Ag13@Cu45', 'Ag', 'Cu', 'MC', 'Ch1')]

shell_energies = [(1, 'MC', 0.00), (2, 'MC', -2.35), (2, 'Ch1', -2.15),
                  (3, 'MC', -4.82), (3, 'Ch1', -4.61), (3, 'BG', -4.55)]

mismatch_params = [(1, 2, 'MC', 'MC', 0.04),
                   (1, 2, 'MC', 'Ch1', 0.14),
                   (2, 3, 'MC', 'MC', 0.038),
                   (2, 3, 'MC', 'Ch1', 0.136),
                   (2, 3, 'Ch1', 'Ch2', 0.21)]

experimental_points = [(1, 3, 0.048, 0.045),
                       (3, 4, 0.042, 0.044),
                       (4, 7, 0.138, 0.142),
                       (7, 12, 0.132, 0.139)]

growth_parameters = {'temperature': 300.0, 'deposition_rate': 0.01,
                     'simulation_steps': 1000, 'beta_factor': 1.0,
                     'delta_opt': 0.04, 'random_seed': 42}

path_probability_weights = {'conservative_step': 0.65,
                            'mismatch_driven_step': 0.25,
                            'random_step': 0.10}

lj_parameters = {'Na-Na': (1.0, 3.72), 'Rb-Rb': (1.0, 4.96),
                 'Cs-Cs': (1.0, 5.30), 'Ag-Ag': (1.0, 2.88),
                 'Cu-Cu': (1.0, 2.56), 'Na-Rb': (1.0, 4.34),
                 'Ag-Cu': (1.0, 2.72)}

thermodynamic_params = {'kT': 0.02585, 'boltzmann': 8.617e-5,
                        'pressure': 1.0, 'timestep': 0.001}

path_selection_stats = [('Conservative path', 325),
                        ('Mismatch-driven path', 125),
                        ('Random path', 50),
                        ('Reverse step', 100)]

growth_results = [(0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02),
                  (30, 'MC', 0.025), (40, 'MC', 0.03), (50, 'MC', 0.035),
                  (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14),
                  (30, 'Ch1', 0.138), (40, 'Ch1', 0.136), (50, 'Ch1', 0.135),
                  (0, 'MC', 0.00), (10, 'MC', 0.08), (20, 'Ch1', 0.14),
                  (30, 'Ch1', 0.15), (40, 'Ch1', 0.145), (50, 'Ch1', 0.142)]


# ============================================================
# 2. CORE COMPUTATIONS
# ============================================================

def compute_size_mismatch(r_inner, r_outer):
    """Compute size mismatch between adjacent shells"""
    return abs(r_outer - r_inner) / r_inner

def lj_potential(r, epsilon, sigma):
    """Compute Lennard-Jones potential"""
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

def compute_shell_atom_numbers(shell_number, sequence_type='mackay'):
    """Compute number of atoms in each shell"""
    if sequence_type == 'mackay':
        seq = mackay_sequence
    else:
        seq = new_sequence_b5
    if shell_number < len(seq):
        return seq[shell_number]
    return None

def predict_stable_clusters():
    """Predict stable multi-shell clusters from atomic radii and mismatch theory"""
    predictions = []
    atoms = list(atomic_radii_dict.keys())
    
    # For each pair of atoms, compute size mismatch and determine optimal shell category
    for i, atom_inner in enumerate(atoms):
        for j, atom_outer in enumerate(atoms):
            if i == j:
                continue
            r_in = atomic_radii_dict[atom_inner]
            r_out = atomic_radii_dict[atom_outer]
            sm = compute_size_mismatch(r_in, r_out)
            
            # Determine which chiral category this mismatch falls into
            best_cat = None
            for inner_cat, outer_cat, sm_min, sm_max in optimal_mismatch_ranges:
                if sm_min <= sm <= sm_max:
                    best_cat = outer_cat
                    break
            
            if best_cat is not None:
                # Determine shell sizes based on Mackay sequence
                n_inner = mackay_sequence[0]  # 1 atom core
                n_shell1 = mackay_sequence[1] - mackay_sequence[0]  # 12 atoms first shell
                
                predictions.append({
                    'inner_atom': atom_inner,
                    'outer_atom': atom_outer,
                    'inner_radius': r_in,
                    'outer_radius': r_out,
                    'size_mismatch': sm,
                    'shell_category': best_cat,
                    'n_inner': n_inner,
                    'n_shell': n_shell1
                })
    
    return predictions

# Run computations
predictions = predict_stable_clusters()

# Compute size mismatches for all mismatch params
mismatch_results = []
for shell_i, shell_j, cat_i, cat_j, sm_val in mismatch_params:
    mismatch_results.append({
        'shell_pair': f'Shell {shell_i}→{shell_j}',
        'inner_category': cat_i,
        'outer_category': cat_j,
        'size_mismatch': sm_val
    })

# ============================================================
# 3. FIGURE GENERATION
# ============================================================

# --- Figure 1: Hexagonal Lattice Shell Sequence Path ---
fig1, ax1 = plt.subplots(figsize=(10, 10))

# Draw hexagonal lattice points
for h, k in hexagonal_coords:
    # Convert hex coords to cartesian for visualization
    x = h + k * cos_2pi_5
    y = k * sin_2pi_5
    ax1.plot(x, y, 'o', color='gray', markersize=8, alpha=0.5)

# Highlight shell sequence path: (0,0) -> (0,1) -> (1,1) -> (1,2) -> ...
# This represents the progression through shells
path_coords = [(0,0), (0,1), (1,1), (1,2), (2,2), (2,3), (3,3), (3,4)]
path_x = [h + k * cos_2pi_5 for h, k in path_coords]
path_y = [k * sin_2pi_5 for h, k in path_coords]

# Draw path with colored segments by chiral category
categories_on_path = ['MC', 'MC', 'Ch1', 'Ch1', 'Ch2', 'Ch2', 'Ch3', 'Ch3']
for idx in range(len(path_coords)-1):
    ax1.plot([path_x[idx], path_x[idx+1]], [path_y[idx], path_y[idx+1]],
             color=shell_colors[categories_on_path[idx]], linewidth=3, alpha=0.8)

for idx, (h, k) in enumerate(path_coords):
    x = h + k * cos_2pi_5
    y = k * sin_2pi_5
    ax1.plot(x, y, 'o', color=shell_colors[categories_on_path[idx]],
             markersize=15, zorder=5)
    ax1.annotate(f'({h},{k})\n{categories_on_path[idx]}',
                 xy=(x, y), xytext=(x+0.15, y+0.15),
                 fontsize=9, fontweight='bold')

ax1.set_xlabel('x (hexagonal lattice units)')
ax1.set_ylabel('y (hexagonal lattice units)')
ax1.set_title('Hexagonal Lattice Shell Sequence Path\nMulti-component Icosahedral Shell Stacking')
ax1.legend(handles=[mpatches.Patch(color=shell_colors[cat], label=cat)
                    for cat in ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3']],
           loc='upper left', title='Chiral Category')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(IMAGES_DIR, 'fig1_hexagonal_lattice_path.png'))
plt.close(fig1)
print("Figure 1 saved")

# --- Figure 2: Size Mismatch vs Shell Pair Analysis ---
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

# Left panel: Size mismatch bar chart for shell pairs
labels = [f'{mr["inner_category"]}→{mr["outer_category"]}\n({mr["shell_pair"]})'
          for mr in mismatch_results]
values = [mr['size_mismatch'] for mr in mismatch_results]
colors_bar = [shell_colors[mr['inner_category']] for mr in mismatch_results]

axes2[0].bar(range(len(labels)), values, color=colors_bar, edgecolor='black', linewidth=0.5)
axes2[0].set_xticks(range(len(labels)))
axes2[0].set_xticklabels(labels, fontsize=9)
axes2[0].set_ylabel('Size Mismatch (δ)')
axes2[0].set_title('Size Mismatch Between Adjacent Shells')

# Add optimal range annotations
for inner_cat, outer_cat, sm_min, sm_max in optimal_mismatch_ranges:
    axes2[0].axhline(y=sm_min, linestyle='--', alpha=0.3, color='gray')
    axes2[0].axhline(y=sm_max, linestyle='--', alpha=0.3, color='gray')

# Right panel: Optimal mismatch ranges as horizontal bars
cats = [f'{ic}→{oc}' for ic, oc, _, _ in optimal_mismatch_ranges]
mins = [sm_min for _, _, sm_min, _ in optimal_mismatch_ranges]
maxs = [sm_max for _, _, _, sm_max in optimal_mismatch_ranges]
range_colors = [shell_colors[ic] for ic, _, _, _ in optimal_mismatch_ranges]

for idx, (cat, mn, mx) in enumerate(zip(cats, mins, maxs)):
    axes2[1].barh(idx, mx - mn, left=mn, height=0.6,
                  color=range_colors[idx], edgecolor='black', alpha=0.7)
    axes2[1].text(mx + 0.01, idx, f'{mn:.2f}-{mx:.2f}', va='center', fontsize=10)

axes2[1].set_yticks(range(len(cats)))
axes2[1].set_yticklabels(cats)
axes2[1].set_xlabel('Size Mismatch (δ)')
axes2[1].set_title('Optimal Size Mismatch Ranges')
axes2[1].set_xlim(0, 0.25)

fig2.suptitle('Size Mismatch Analysis for Multi-Shell Icosahedral Structures', fontsize=16)
fig2.tight_layout()
fig2.savefig(os.path.join(IMAGES_DIR, 'fig2_size_mismatch_analysis.png'))
plt.close(fig2)
print("Figure 2 saved")

# --- Figure 3: Shell Energy Comparison by Chiral Category ---
fig3, ax3 = plt.subplots(figsize=(10, 7))

shell_nums = sorted(set([se[0] for se in shell_energies]))
categories_in_data = sorted(set([se[1] for se in shell_energies]))

for cat in categories_in_data:
    energies_for_cat = [(se[0], se[2]) for se in shell_energies if se[1] == cat]
    shells_e = [e[0] for e in energies_for_cat]
    vals_e = [e[1] for e in energies_for_cat]
    ax3.plot(shells_e, vals_e, 'o-', color=shell_colors[cat],
             linewidth=2, markersize=10, label=cat)

ax3.set_xlabel('Shell Number')
ax3.set_ylabel('Relative Shell Energy (normalized units)')
ax3.set_title('Shell Energy Comparison Across Chiral Categories')
ax3.legend(title='Chiral Category')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(shell_nums)
fig3.tight_layout()
fig3.savefig(os.path.join(IMAGES_DIR, 'fig3_shell_energy_comparison.png'))
plt.close(fig3)
print("Figure 3 saved")

# --- Figure 4: Theoretical vs Experimental Validation Parity Plot ---
fig4, ax4 = plt.subplots(figsize=(8, 8))

measured = [ep[2] for ep in experimental_points]
predicted_vals = [ep[3] for ep in experimental_points]
shell_labels_exp = [f'T{ep[0]}→T{ep[1]}' for ep in experimental_points]

ax4.scatter(measured, predicted_vals, s=120, c=[shell_colors['MC']],
            edgecolors='black', zorder=5)

# Add parity line
max_val = max(max(measured), max(predicted_vals)) * 1.2
min_val = min(min(measured), min(predicted_vals)) * 0.8
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)

# Annotate points
for idx, (m, p, label) in enumerate(zip(measured, predicted_vals, shell_labels_exp)):
    ax4.annotate(label, xy=(m, p), xytext=(m+0.005, p+0.005), fontsize=10)

# Compute R²
ss_res = sum((m - p)**2 for m, p in zip(measured, predicted_vals))
ss_tot = sum((m - sum(measured)/len(measured))**2 for m in measured)
r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0

ax4.set_xlabel('Measured Size Mismatch')
ax4.set_ylabel('Theoretical Size Mismatch')
ax4.set_title(f'Theoretical vs Experimental Validation\nR² = {r_squared:.4f}')
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(IMAGES_DIR, 'fig4_validation_parity_plot.png'))
plt.close(fig4)
print("Figure 4 saved")

# --- Figure 5: Growth Dynamics - Mismatch Evolution ---
fig5, ax5 = plt.subplots(figsize=(12, 7))

# Group growth results by trajectory
trajectory_1 = [(gr[0], gr[2]) for gr in growth_results[:6]]  # Pure MC growth
trajectory_2 = [(gr[0], gr[2]) for gr in growth_results[6:12]]  # Pure Ch1 growth
trajectory_3 = [(gr[0], gr[2]) for gr in growth_results[12:18]]  # MC→Ch1 transition

steps_1 = [t[0] for t in trajectory_1]
vals_1 = [t[1] for t in trajectory_1]
steps_2 = [t[0] for t in trajectory_2]
vals_2 = [t[1] for t in trajectory_2]
steps_3 = [t[0] for t in trajectory_3]
vals_3 = [t[1] for t in trajectory_3]

ax5.plot(steps_1, vals_1, 'o-', color=shell_colors['MC'], linewidth=2,
         markersize=8, label='Trajectory 1: MC shell growth')
ax5.plot(steps_2, vals_2, 's-', color=shell_colors['Ch1'], linewidth=2,
         markersize=8, label='Trajectory 2: Ch1 shell growth')
ax5.plot(steps_3, vals_3, 'D-', color='#555555', linewidth=2,
         markersize=8, label='Trajectory 3: MC→Ch1 transition')

# Add optimal mismatch range bands
ax5.axhspan(0.03, 0.05, alpha=0.15, color=shell_colors['MC'], label='MC optimal range')
ax5.axhspan(0.12, 0.16, alpha=0.15, color=shell_colors['Ch1'], label='Ch1 optimal range')

ax5.set_xlabel('Simulation Steps')
ax5.set_ylabel('Average Size Mismatch (δ)')
ax5.set_title('Growth Dynamics: Size Mismatch Evolution During Self-Assembly')
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(IMAGES_DIR, 'fig5_growth_dynamics.png'))
plt.close(fig5)
print("Figure 5 saved")

# --- Figure 6: LJ Potential Energy Landscape ---
fig6, ax6 = plt.subplots(figsize=(10, 7))

r_range = np.linspace(2.0, 8.0, 500)

for pair_name, (eps, sig) in lj_parameters.items():
    potentials = lj_potential(r_range, eps, sig)
    ax6.plot(r_range, potentials, linewidth=2, label=pair_name)

ax6.set_xlabel('Interatomic Distance r (Å)')
ax6.set_ylabel('LJ Potential Energy V(r)')
ax6.set_title('Lennard-Jones Potential Energy Landscape\nfor Multi-Component Atomic Pairs')
ax6.legend(loc='best')
ax6.set_ylim(-2, 5)
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='k', linewidth=0.5)
fig6.tight_layout()
fig6.savefig(os.path.join(IMAGES_DIR, 'fig6_lj_potential_landscape.png'))
plt.close(fig6)
print("Figure 6 saved")

# --- Figure 7: Predicted Stable Cluster Structures ---
fig7, axes7 = plt.subplots(1, 3, figsize=(18, 6))

cluster_info = [
    ('Na₁₃@Rb₃₂', 'Na', 'Rb', 'MC', 'Ch1', 13, 32),
    ('K₁₃@Cs₄₂', 'K', 'Cs', 'MC', 'Ch2', 13, 42),
    ('Ag₁₃@Cu₄₅', 'Ag', 'Cu', 'MC', 'Ch1', 13, 45)
]

for idx, (name, atom_in, atom_out, cat_in, cat_out, n_in, n_out) in enumerate(cluster_info):
    ax = axes7[idx]
    
    # Draw concentric shells as circles
    r_in = atomic_radii_dict[atom_in]
    r_out = atomic_radii_dict[atom_out]
    
    # Scale for visualization
    scale = 3.0
    radius_inner = scale * r_in / atomic_radii_dict['Cs']  # Normalize to largest
    radius_outer = scale * r_out / atomic_radii_dict['Cs']
    
    # Inner shell (core)
    circle_inner = plt.Circle((0, 0), radius_inner, color=shell_colors[cat_in],
                               alpha=0.6, linewidth=2, edgecolor='black')
    ax.add_patch(circle_inner)
    
    # Outer shell
    circle_outer = plt.Circle((0, 0), radius_outer, color=shell_colors[cat_out],
                               alpha=0.4, linewidth=2, edgecolor='black')
    ax.add_patch(circle_outer)
    
    # Place atoms on inner shell (core + first shell)
    n_core = 1
    angles_core = np.linspace(0, 2*np.pi, n_core, endpoint=False)
    ax.plot(0, 0, 'o', color=shell_colors[cat_in], markersize=12,
            markeredgecolor='black', zorder=5)
    
    # Place atoms around inner shell perimeter
    angles_shell1 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    ax.plot(radius_inner * 0.8 * np.cos(angles_shell1),
            radius_inner * 0.8 * np.sin(angles_shell1),
            'o', color=shell_colors[cat_in], markersize=8,
            markeredgecolor='black', zorder=5)
    
    # Place atoms on outer shell
    n_outer_vis = min(n_out, 20)
    angles_shell2 = np.linspace(0, 2*np.pi, n_outer_vis, endpoint=False)
    ax.plot(radius_outer * 0.85 * np.cos(angles_shell2),
            radius_outer * 0.85 * np.sin(angles_shell2),
            'o', color=shell_colors[cat_out], markersize=7,
            markeredgecolor='black', zorder=5)
    
    sm = compute_size_mismatch(r_in, r_out)
    ax.set_title(f'{name}\nδ = {sm:.3f} ({cat_in}→{cat_out})', fontsize=13)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend(handles=[
        mpatches.Patch(color=shell_colors[cat_in], label=f'{atom_in} ({cat_in})'),
        mpatches.Patch(color=shell_colors[cat_out], label=f'{atom_out} ({cat_out})')
    ], loc='upper right', fontsize=9)
    ax.axis('off')

fig7.suptitle('Predicted Stable Multi-Shell Icosahedral Structures', fontsize=16)
fig7.tight_layout()
fig7.savefig(os.path.join(IMAGES_DIR, 'fig7_predicted_clusters.png'))
plt.close(fig7)
print("Figure 7 saved")

# --- Figure 8: Path Selection Statistics ---
fig8, ax8 = plt.subplots(figsize=(10, 7))

path_names = [ps[0] for ps in path_selection_stats]
path_counts = [ps[1] for ps in path_selection_stats]
total = sum(path_counts)
path_colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

wedges, texts, autotexts = ax8.pie(path_counts, labels=path_names,
                                    colors=path_colors_list,
                                    autopct=lambda pct: f'{pct:.1f}%\n({int(pct*total/100)})',
                                    startangle=90, pctdistance=0.75,
                                    wedgeprops=dict(edgecolor='black', linewidth=1))

for autotext in autotexts:
    autotext.set_fontsize(10)
for text in texts:
    text.set_fontsize(11)

ax8.set_title('Path Selection Statistics in Growth Simulations\n(Total steps: 500)')
fig8.tight_layout()
fig8.savefig(os.path.join(IMAGES_DIR, 'fig8_path_selection_stats.png'))
plt.close(fig8)
print("Figure 8 saved")


# ============================================================
# 4. ADDITIONAL ANALYSIS: Magic Number Comparison & Extended Predictions
# ============================================================

# --- Figure 9: Magic Number Sequences Comparison ---
fig9, ax9 = plt.subplots(figsize=(10, 7))

shell_indices_mackay = range(len(mackay_sequence))
shell_indices_b5 = range(len(new_sequence_b5))

ax9.plot(shell_indices_mackay, mackay_sequence, 'o-', color='#1f77b4',
         linewidth=2, markersize=10, label='Mackay (b=3)')
ax9.plot(shell_indices_b5, new_sequence_b5, 's-', color='#ff7f0e',
         linewidth=2, markersize=10, label='New (b=5)')

ax9.set_xlabel('Shell Number (n)')
ax9.set_ylabel('Total Atom Count')
ax9.set_title('Magic Number Sequences: Mackay vs New (b=5)\nfor Icosahedral Shell Packing')
ax9.legend()
ax9.grid(True, alpha=0.3)
fig9.tight_layout()
fig9.savefig(os.path.join(IMAGES_DIR, 'fig9_magic_number_comparison.png'))
plt.close(fig9)
print("Figure 9 saved")

# --- Figure 10: Atomic Radii & Compatibility Matrix ---
fig10, axes10 = plt.subplots(1, 2, figsize=(16, 7))

# Left: Atomic radii bar chart
atoms_list = list(atomic_radii_dict.keys())
radii_vals = list(atomic_radii_dict.values())
atom_type_colors = ['#1f77b4' if a in ['Na','K','Rb','Cs'] else '#ff7f0e' for a in atoms_list]

axes10[0].barh(range(len(atoms_list)), radii_vals, color=atom_type_colors,
               edgecolor='black', linewidth=0.5)
axes10[0].set_yticks(range(len(atoms_list)))
axes10[0].set_yticklabels(atoms_list)
axes10[0].set_xlabel('Atomic Radius (Å)')
axes10[0].set_title('Atomic Radii of Constituent Elements')
axes10[0].legend(handles=[
    mpatches.Patch(color='#1f77b4', label='Alkali Metals'),
    mpatches.Patch(color='#ff7f0e', label='Transition Metals')
])

# Right: Pair compatibility heatmap
all_atoms = ['Na', 'K', 'Rb', 'Cs', 'Ag', 'Cu', 'Ni']
compat_matrix = np.zeros((len(all_atoms), len(all_atoms)))

for i, a1 in enumerate(all_atoms):
    for j, a2 in enumerate(all_atoms):
        if i == j:
            compat_matrix[i][j] = 0
        else:
            sm = compute_size_mismatch(atomic_radii_dict[a1], atomic_radii_dict[a2])
            compat_matrix[i][j] = sm

sns.heatmap(compat_matrix, annot=True, fmt='.3f', xticklabels=all_atoms,
            yticklabels=all_atoms, cmap='YlOrRd', ax=axes10[1],
            linewidths=0.5, vmin=0, vmax=0.3)
axes10[1].set_title('Size Mismatch Matrix Between All Atom Pairs')
axes10[1].set_xlabel('Outer Shell Atom')
axes10[1].set_ylabel('Inner Shell Atom')

fig10.suptitle('Atomic Properties and Pair Compatibility Analysis', fontsize=16)
fig10.tight_layout()
fig10.savefig(os.path.join(IMAGES_DIR, 'fig10_atomic_radii_compatibility.png'))
plt.close(fig10)
print("Figure 10 saved")


# ============================================================
# 5. SAVE INTERMEDIATE RESULTS
# ============================================================

results = {
    'size_mismatch_analysis': mismatch_results,
    'predicted_clusters': predictions,
    'validation_r_squared': r_squared,
    'experimental_comparison': [
        {'shell_pair': f'T{ep[0]}→T{ep[1]}', 'measured': ep[2], 'predicted': ep[3]}
        for ep in experimental_points
    ],
    'magic_numbers_mackay': mackay_sequence,
    'magic_numbers_b5': new_sequence_b5,
    'shell_energies_by_category': {},
    'growth_dynamics_summary': {
        'trajectory_1_final_mismatch': vals_1[-1],
        'trajectory_2_final_mismatch': vals_2[-1],
        'trajectory_3_final_mismatch': vals_3[-1]
    }
}

# Organize shell energies by category
for se in shell_energies:
    cat = se[1]
    if cat not in results['shell_energies_by_category']:
        results['shell_energies_by_category'][cat] = []
    results['shell_energies_by_category'][cat].append({'shell': se[0], 'energy': se[2]})

with open(os.path.join(OUTPUTS_DIR, 'analysis_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("All intermediate results saved")
print(f"Total figures generated: 10")