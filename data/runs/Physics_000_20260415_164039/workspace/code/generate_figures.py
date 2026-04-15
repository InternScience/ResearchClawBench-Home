#!/usr/bin/env python3
"""
Figure Generation for Multi-Component Icosahedral Nanocluster Research Report
=============================================================================
Generates all publication-quality figures for the research report.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

os.makedirs("report/images", exist_ok=True)

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# ============================================================================
# DATA (same as analysis.py)
# ============================================================================

hexagonal_coords = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5),
                    (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),
                    (2,0), (2,1), (2,2), (2,3), (2,4), (2,5),
                    (3,0), (3,1), (3,2), (3,3), (3,4), (3,5),
                    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5),
                    (5,0), (5,1), (5,2), (5,3), (5,4), (5,5)]

mackay_sequence = [1, 13, 55, 147, 309]
new_sequence_b5 = [1, 13, 45, 117, 239, 431]

chiral_labels = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
shell_colors_map = {
    'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c',
    'Ch2': '#d62728', 'Ch3': '#9467bd', 'Ch4': '#8c564b', 'Ch5': '#e377c2'
}

atomic_radii = {'Na': 1.86, 'K': 2.27, 'Rb': 2.48, 'Cs': 2.65,
                'Ag': 1.44, 'Cu': 1.28, 'Ni': 1.24}

atomic_pairs = [
    ('Na', 'Rb', 0.22), ('Ag', 'Cu', 0.12),
    ('Ag', 'Ni', 0.15), ('Cu', 'Ni', 0.032)
]

mismatch_ranges = [
    ('MC', 'MC', 0.03, 0.05), ('MC', 'Ch1', 0.12, 0.16),
    ('MC', 'Ch2', 0.19, 0.22), ('MC', 'BG', 0.08, 0.10)
]

shell_energies = [
    (1, 'MC', 0.00), (2, 'MC', -2.35), (2, 'Ch1', -2.15),
    (3, 'MC', -4.82), (3, 'Ch1', -4.61), (3, 'BG', -4.55)
]

mismatch_params = [
    (1, 2, 'MC', 'MC', 0.04), (1, 2, 'MC', 'Ch1', 0.14),
    (2, 3, 'MC', 'MC', 0.038), (2, 3, 'MC', 'Ch1', 0.136),
    (2, 3, 'Ch1', 'Ch2', 0.21)
]

experimental_points = [
    (1, 3, 0.048, 0.045), (3, 4, 0.042, 0.044),
    (4, 7, 0.138, 0.142), (7, 12, 0.132, 0.139)
]

growth_results = [
    (0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02),
    (30, 'MC', 0.025), (40, 'MC', 0.03), (50, 'MC', 0.035),
    (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14),
    (30, 'Ch1', 0.138), (40, 'Ch1', 0.136), (50, 'Ch1', 0.135),
    (0, 'MC', 0.00), (10, 'MC', 0.08), (20, 'Ch1', 0.14),
    (30, 'Ch1', 0.15), (40, 'Ch1', 0.145), (50, 'Ch1', 0.142)
]

path_selection_stats = [
    ('Conservative path', 325), ('Mismatch-driven path', 125),
    ('Random path', 50), ('Reverse step', 100)
]

lj_parameters = [
    ('Na-Na', 1.0, 3.72), ('Rb-Rb', 1.0, 4.96), ('Cs-Cs', 1.0, 5.30),
    ('Ag-Ag', 1.0, 2.88), ('Cu-Cu', 1.0, 2.56),
    ('Na-Rb', 1.0, 4.34), ('Ag-Cu', 1.0, 2.72)
]


# ============================================================================
# FIGURE 1: Hexagonal Lattice Coordinate System
# ============================================================================
print("Generating Figure 1: Hexagonal lattice coordinate system...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Hexagonal lattice with coordinates
ax = axes[0]
for h, k in hexagonal_coords:
    x = h + 0.5 * k
    y = np.sqrt(3) / 2 * k
    ax.plot(x, y, 'o', color='#1f77b4', markersize=8, zorder=3)
    if h <= 3 and k <= 3:
        ax.annotate(f'({h},{k})', (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, ha='left')

# Draw hexagonal grid lines
for h in range(6):
    for k in range(6):
        x1 = h + 0.5 * k
        y1 = np.sqrt(3) / 2 * k
        # Horizontal connections
        if k < 5:
            x2 = h + 0.5 * (k + 1)
            y2 = np.sqrt(3) / 2 * (k + 1)
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.15, linewidth=0.5)
        # Diagonal connections
        if h < 5:
            x2 = (h + 1) + 0.5 * k
            y2 = np.sqrt(3) / 2 * k
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.15, linewidth=0.5)
            x2 = (h + 1) + 0.5 * (k - 1) if k > 0 else None
            y2 = np.sqrt(3) / 2 * (k - 1) if k > 0 else None
            if x2 is not None:
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.15, linewidth=0.5)

# Highlight shell paths
path_coords = [(0,0), (0,1), (1,1), (1,2), (2,2), (2,3)]
path_x = [h + 0.5*k for h, k in path_coords]
path_y = [np.sqrt(3)/2 * k for h, k in path_coords]
ax.plot(path_x, path_y, 'r-', linewidth=2.5, marker='s', markersize=6,
        color='#d62728', label='Shell growth path', zorder=4)

ax.set_xlabel('x (lattice units)', fontsize=11)
ax.set_ylabel('y (lattice units)', fontsize=11)
ax.set_title('(A) Hexagonal Lattice Coordinates\nand Shell Growth Path', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_aspect('equal')
ax.grid(True, alpha=0.1)

# Panel B: Magic number sequences comparison
ax = axes[1]
shells = np.arange(len(mackay_sequence))
ax.plot(shells, mackay_sequence, 'o-', color='#1f77b4', linewidth=2,
        markersize=8, label='Mackay sequence', zorder=3)
ax.plot(shells[:len(new_sequence_b5)], new_sequence_b5[:len(mackay_sequence)],
        's-', color='#d62728', linewidth=2, markersize=8,
        label='New sequence (b=5)', zorder=3)

for i, (m, n) in enumerate(zip(mackay_sequence, new_sequence_b5)):
    ax.annotate(f'{m}', (i, m), textcoords="offset points",
                xytext=(0, 10), fontsize=9, ha='center', color='#1f77b4')
    ax.annotate(f'{n}', (i, n), textcoords="offset points",
                xytext=(0, -15), fontsize=9, ha='center', color='#d62728')

ax.set_xticks(shells)
ax.set_xticklabels([f'Shell {i}' for i in shells])
ax.set_xlabel('Shell Index', fontsize=11)
ax.set_ylabel('Cumulative Atom Count', fontsize=11)
ax.set_title('(B) Magic Number Sequences\nComparison', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("report/images/fig1_hexagonal_lattice.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: report/images/fig1_hexagonal_lattice.png")


# ============================================================================
# FIGURE 2: Size Mismatch vs Stability by Chiral Category
# ============================================================================
print("Generating Figure 2: Size mismatch vs stability...")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot optimal ranges as shaded bands
range_data = [
    ('MC→MC', 0.03, 0.05, '#1f77b4'),
    ('MC→BG', 0.08, 0.10, '#ff7f0e'),
    ('MC→Ch1', 0.12, 0.16, '#2ca02c'),
    ('MC→Ch2', 0.19, 0.22, '#d62728')
]

for label, lo, hi, color in range_data:
    ax.axvspan(lo, hi, alpha=0.15, color=color)
    ax.axvline((lo + hi) / 2, color=color, linestyle='--', alpha=0.5, linewidth=1)
    ax.text((lo + hi) / 2, 0.95, label, transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=9, rotation=0,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))

# Plot computed atomic pair mismatches
pair_deltas = []
pair_labels_list = []
pair_colors_list = []
for e1, e2, reported in atomic_pairs:
    r1 = atomic_radii[e1]
    r2 = atomic_radii[e2]
    delta = abs(r2 - r1) / r1
    pair_deltas.append(delta)
    pair_labels_list.append(f'{e1}-{e2}')
    pair_colors_list.append('#8c564b')

for i, (delta, label) in enumerate(zip(pair_deltas, pair_labels_list)):
    ax.axvline(delta, color='#8c564b', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(delta, 0.85 + i*0.05, label, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8, color='#8c564b', fontweight='bold')

# Plot mismatch parameters from data
for si, sj, ci, cj, delta in mismatch_params:
    ax.plot(delta, 0.5, 'D', color=shell_colors_map.get(ci, 'gray'),
            markersize=10, zorder=5)
    ax.annotate(f'{ci}→{cj}\n(Shell {si}→{sj})', (delta, 0.5),
                textcoords="offset points", xytext=(0, 12),
                fontsize=7, ha='center')

ax.set_xlabel('Size Mismatch δ = |r₂ - r₁| / r₁', fontsize=12)
ax.set_ylabel('Stability Region', fontsize=12)
ax.set_title('Optimal Size Mismatch Ranges by Chiral Category Transition', fontsize=13)
ax.set_xlim(0, 0.25)
ax.set_ylim(0, 1.1)
ax.set_yticks([])
ax.grid(True, alpha=0.15, axis='x')

plt.tight_layout()
plt.savefig("report/images/fig2_size_mismatch_stability.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: report/images/fig2_size_mismatch_stability.png")


# ============================================================================
# FIGURE 3: Growth Simulation Dynamics
# ============================================================================
print("Generating Figure 3: Growth simulation dynamics...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Parse trajectories
trajectories = {}
traj_id = 0
current_traj = []
for steps, cat, mm in growth_results:
    if steps == 0 and current_traj:
        trajectories[f"traj_{traj_id}"] = current_traj
        traj_id += 1
        current_traj = []
    current_traj.append({'steps': steps, 'category': cat, 'mismatch': mm})
if current_traj:
    trajectories[f"traj_{traj_id}"] = current_traj

# Panel A: Mismatch evolution over growth steps
ax = axes[0]
traj_names = list(trajectories.keys())
for tid, traj in trajectories.items():
    steps = [t['steps'] for t in traj]
    mismatches = [t['mismatch'] for t in traj]
    final_cat = traj[-1]['category']
    color = shell_colors_map.get(final_cat, 'gray')
    ax.plot(steps, mismatches, 'o-', color=color, linewidth=2,
            markersize=6, label=f'{tid} ({final_cat})', zorder=3)

ax.set_xlabel('Growth Steps', fontsize=11)
ax.set_ylabel('Average Size Mismatch δ', fontsize=11)
ax.set_title('(A) Size Mismatch Evolution During Growth', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Panel B: Convergence behavior
ax = axes[1]
for tid, traj in trajectories.items():
    steps = [t['steps'] for t in traj]
    mismatches = [t['mismatch'] for t in traj]
    final_cat = traj[-1]['category']
    color = shell_colors_map.get(final_cat, 'gray')

    # Compute running average
    running_avg = np.cumsum(mismatches) / np.arange(1, len(mismatches) + 1)
    ax.plot(steps, running_avg, 's-', color=color, linewidth=1.5,
            markersize=5, alpha=0.7, label=f'{tid} ({final_cat})', zorder=3)

ax.set_xlabel('Growth Steps', fontsize=11)
ax.set_ylabel('Running Average Mismatch δ', fontsize=11)
ax.set_title('(B) Running Average Convergence', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("report/images/fig3_growth_dynamics.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: report/images/fig3_growth_dynamics.png")


# ============================================================================
# FIGURE 4: Path Selection Statistics
# ============================================================================
print("Generating Figure 4: Path selection statistics...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Bar chart
ax = axes[0]
names = [p[0] for p in path_selection_stats]
counts = [p[1] for p in path_selection_stats]
total = sum(counts)
colors_bar = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

bars = ax.bar(names, counts, color=colors_bar, edgecolor='black', linewidth=0.8, width=0.6)
for bar, count in zip(bars, counts):
    pct = count / total * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Number of Steps', fontsize=11)
ax.set_title('(A) Path Selection Distribution', fontsize=12)
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.2, axis='y')

# Panel B: Pie chart
ax = axes[1]
wedges, texts, autotexts = ax.pie(counts, labels=names, autopct='%1.1f%%',
                                    colors=colors_bar, startangle=90,
                                    textprops={'fontsize': 9},
                                    pctdistance=0.8)
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)
ax.set_title('(B) Path Selection Proportions', fontsize=12)

plt.tight_layout()
plt.savefig("report/images/fig4_path_selection.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: report/images/fig4_path_selection.png")


# ============================================================================
# FIGURE 5: Shell Energy Comparison
# ============================================================================
print("Generating Figure 5: Shell energy comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Energy by shell and category
ax = axes[0]
shell_nums = sorted(set(sn for sn, _, _ in shell_energies))
categories = ['MC', 'Ch1', 'BG']
x_positions = np.arange(len(shell_nums))
width = 0.25

for i, cat in enumerate(categories):
    energies = []
    for sn in shell_nums:
        e = next((val for s, c, val in shell_energies if s == sn and c == cat), None)
        energies.append(e)
    valid_mask = [e is not None for e in energies]
    valid_energies = [e if e is not None else 0 for e in energies]
    bars = ax.bar(x_positions + i*width, valid_energies, width,
                  label=cat, color=shell_colors_map[cat], edgecolor='black',
                  linewidth=0.5, alpha=0.85)
    for j, (bar, e) in enumerate(zip(bars, energies)):
        if e is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15,
                    f'{e:.2f}', ha='center', va='top', fontsize=8, fontweight='bold')

ax.set_xticks(x_positions + width)
ax.set_xticklabels([f'Shell {sn}' for sn in shell_nums])
ax.set_ylabel('Normalized Energy (arb. units)', fontsize=11)
ax.set_title('(A) Shell Energy by Category', fontsize=12)
ax.legend(fontsize=10)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.2, axis='y')

# Panel B: Energy difference between categories
ax = axes[1]
diffs = []
diff_labels = []
diff_colors = []
for sn in shell_nums:
    mc_e = next((e for s, c, e in shell_energies if s == sn and c == 'MC'), None)
    ch1_e = next((e for s, c, e in shell_energies if s == sn and c == 'Ch1'), None)
    bg_e = next((e for s, c, e in shell_energies if s == sn and c == 'BG'), None)
    if mc_e is not None and ch1_e is not None:
        diffs.append(mc_e - ch1_e)
        diff_labels.append(f'Shell {sn}\nMC−Ch1')
        diff_colors.append('#9467bd')
    if mc_e is not None and bg_e is not None:
        diffs.append(mc_e - bg_e)
        diff_labels.append(f'Shell {sn}\nMC−BG')
        diff_colors.append('#e377c2')

x_pos = np.arange(len(diffs))
bars = ax.barh(x_pos, diffs, color=diff_colors, edgecolor='black', linewidth=0.5, height=0.5)
for bar, val in zip(bars, diffs):
    ax.text(val + (0.02 if val > 0 else -0.02), bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', ha='left' if val > 0 else 'right', va='center',
            fontsize=9, fontweight='bold')

ax.set_yticks(x_pos)
ax.set_yticklabels(diff_labels, fontsize=9)
ax.set_xlabel('Energy Difference ΔE (arb. units)', fontsize=11)
ax.set_title('(B) Energy Differences Between Categories', fontsize=12)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
plt.savefig("report/images/fig5_shell_energy.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: report/images/fig5_shell_energy.png")


# ============================================================================
# FIGURE 6: Theory vs Experiment Parity Plot
# ============================================================================
print("Generating Figure 6: Theory vs experiment parity plot...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Parity plot
ax = axes[0]
measured = [ep[2] for ep in experimental_points]
theoretical = [ep[3] for ep in experimental_points]
shell_labels = [f'{ep[0]}-{ep[1]}' for ep in experimental_points]

# Identity line
min_val = min(min(measured), min(theoretical)) - 0.005
max_val = max(max(measured), max(theoretical)) + 0.005
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect agreement')

colors_exp = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
for i, (m, t, label) in enumerate(zip(measured, theoretical, shell_labels)):
    ax.errorbar(t, m, yerr=abs(m-t)*0.1, fmt='o', color=colors_exp[i],
                markersize=10, capsize=4, elinewidth=1.5, zorder=3)
    ax.annotate(label, (t, m), textcoords="offset points",
                xytext=(8, 8), fontsize=9, fontweight='bold', color=colors_exp[i])

ax.set_xlabel('Theoretical Size Mismatch δ', fontsize=11)
ax.set_ylabel('Measured Size Mismatch δ', fontsize=11)
ax.set_title('(A) Theory vs Experiment\nSize Mismatch Validation', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.grid(True, alpha=0.2)

# Compute errors
errors = [abs(m - t) for m, t in zip(measured, theoretical)]
rel_errors = [e/t*100 for e, t in zip(errors, theoretical)]

# Panel B: Error analysis
ax = axes[1]
x_pos = np.arange(len(shell_labels))
bars1 = ax.bar(x_pos - 0.15, errors, 0.25, label='Absolute Error',
               color='#1f77b4', edgecolor='black', linewidth=0.5)
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos + 0.15, rel_errors, 0.25, label='Relative Error (%)',
                color='#d62728', edgecolor='black', linewidth=0.5)

for bar, val in zip(bars1, errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, rel_errors):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold',
             color='#d62728')

ax.set_xticks(x_pos)
ax.set_xticklabels(shell_labels, fontsize=9)
ax.set_xlabel('Shell Transition', fontsize=11)
ax.set_ylabel('Absolute Error', fontsize=11, color='#1f77b4')
ax2.set_ylabel('Relative Error (%)', fontsize=11, color='#d62728')
ax.set_title('(B) Error Analysis', fontsize=12)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig("report/images/fig6_theory_vs_experiment.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: report/images/fig6_theory_vs_experiment.png")


# ============================================================================
# FIGURE 7: Atomic Pair Compatibility & LJ Potential
# ============================================================================
print("Generating Figure 7: Atomic pair compatibility and LJ potential...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Atomic radii and computed mismatches
ax = axes[0]
elements = list(atomic_radii.keys())
radii = [atomic_radii[e] for e in elements]
colors_elem = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

bars = ax.bar(elements, radii, color=colors_elem, edgecolor='black', linewidth=0.8, width=0.6)
for bar, r in zip(bars, radii):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{r:.2f} Å', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Atomic Radius (Å)', fontsize=11)
ax.set_title('(A) Atomic Radii of Elements', fontsize=12)
ax.grid(True, alpha=0.2, axis='y')

# Panel B: LJ potential curves
ax = axes[1]
r_range = np.linspace(2.0, 8.0, 500)

for pair_name, epsilon, sigma in lj_parameters:
    sr6 = (sigma / r_range) ** 6
    V = 4 * epsilon * (sr6**2 - sr6)
    ax.plot(r_range, V, '-', linewidth=1.5, label=pair_name, alpha=0.8)
    # Mark minimum
    r_min = sigma * 2**(1/6)
    V_min = -epsilon
    ax.plot(r_min, V_min, 'o', markersize=6)

ax.set_xlabel('Interatomic Distance r (Å)', fontsize=11)
ax.set_ylabel('Lennard-Jones Potential V(r)', fontsize=11)
ax.set_title('(B) Lennard-Jones Potential Curves', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("report/images/fig7_atomic_pair_lj.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: report/images/fig7_atomic_pair_lj.png")


# ============================================================================
# FIGURE 8: Multi-component Cluster Prediction Summary
# ============================================================================
print("Generating Figure 8: Cluster prediction summary...")

fig, ax = plt.subplots(figsize=(10, 6))

clusters = [
    ('Na₁₃@Rb₃₂', 'Na', 'Rb', 'MC', 'Ch1', 0.333),
    ('K₁₃@Cs₄₂', 'K', 'Cs', 'MC', 'Ch2', 0.167),
    ('Ag₁₃@Cu₄₅', 'Ag', 'Cu', 'MC', 'Ch1', 0.111),
    ('K₁₃@Rb₃₂', 'K', 'Rb', 'MC', 'BG', 0.093),
    ('Ag₁₃@Ni₄₅', 'Ag', 'Ni', 'MC', 'Ch1', 0.139),
    ('Cu₁₃@Ni₄₅', 'Cu', 'Ni', 'MC', 'MC', 0.031)
]

names = [c[0] for c in clusters]
mismatches = [c[5] for c in clusters]
categories = [f'{c[3]}→{c[4]}' for c in clusters]
colors_clust = [shell_colors_map.get(c[4], 'gray') for c in clusters]

# Horizontal bar chart sorted by mismatch
sorted_idx = np.argsort(mismatches)
sorted_names = [names[i] for i in sorted_idx]
sorted_mm = [mismatches[i] for i in sorted_idx]
sorted_cats = [categories[i] for i in sorted_idx]
sorted_colors = [colors_clust[i] for i in sorted_idx]

y_pos = np.arange(len(sorted_names))
bars = ax.barh(y_pos, sorted_mm, color=sorted_colors, edgecolor='black',
               linewidth=0.8, height=0.6)

for bar, name, mm, cat in zip(bars, sorted_names, sorted_mm, sorted_cats):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'δ={mm:.3f}  [{cat}]', ha='left', va='center', fontsize=9, fontweight='bold')

# Add optimal range bands
for cat_i, cat_o, lo, hi in mismatch_ranges:
    ax.axvspan(lo, hi, alpha=0.08, color=shell_colors_map.get(cat_o, 'gray'))

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=10)
ax.set_xlabel('Computed Size Mismatch δ', fontsize=11)
ax.set_title('Predicted Multi-Component Cluster Stability\n(Size Mismatch vs Optimal Ranges)', fontsize=12)
ax.grid(True, alpha=0.15, axis='x')

plt.tight_layout()
plt.savefig("report/images/fig8_cluster_predictions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: report/images/fig8_cluster_predictions.png")


print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print("Figures saved to report/images/:")
for fname in sorted(os.listdir("report/images")):
    if fname.endswith('.png'):
        fpath = os.path.join("report/images", fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname} ({size_kb:.1f} KB)")
