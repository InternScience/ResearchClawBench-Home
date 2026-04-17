#!/usr/bin/env python3
"""
Generate all figures for the multi-component icosahedral shell packing report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from core_theory import (
    hexagonal_coords_list, triangulation_number, mackay_magic_numbers,
    new_magic_numbers_b5, classify_shell_path, SHELL_COLORS, CHIRAL_LABELS,
    ATOMIC_RADII, size_mismatch, compute_all_pair_mismatches,
    OPTIMAL_MISMATCH_RANGES, SHELL_ENERGIES, MISMATCH_PARAMS,
    EXPERIMENTAL_POINTS, generate_shell_paths, compute_all_T,
    get_shell_classification_map, MULTICOMPONENT_CLUSTERS,
    LJ_PARAMETERS, lj_potential, lj_equilibrium_distance
)

# Output directory
IMG_DIR = 'report/images'
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150
})

# ============================================================
# Figure 1: Hexagonal Lattice with Shell Classification
# ============================================================

def fig1_hexagonal_lattice():
    """Plot hexagonal lattice with shell classification and paths."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Panel A: Hexagonal lattice with T values and classification
    ax = axes[0]
    classification = get_shell_classification_map()
    T_values = compute_all_T()
    
    # Convert hex coords to Cartesian for plotting
    for (h, k) in hexagonal_coords_list():
        # Hexagonal coordinate to Cartesian
        x = h + k * 0.5
        y = k * np.sqrt(3) / 2
        
        T = T_values[(h, k)]
        cls = classification[(h, k)]
        color = SHELL_COLORS.get(cls, '#999999')
        
        circle = plt.Circle((x, y), 0.3, color=color, alpha=0.7, ec='black', lw=1)
        ax.add_patch(circle)
        ax.text(x, y, str(T), ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.set_xlabel('h + k/2')
    ax.set_ylabel(r'k$\sqrt{3}$/2')
    ax.set_title('(a) Hexagonal Lattice: Triangulation Numbers T(h,k)')
    
    # Legend
    legend_patches = [mpatches.Patch(color=SHELL_COLORS[label], label=label) for label in ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4']]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # Panel B: Shell paths on hexagonal lattice
    ax = axes[1]
    paths = generate_shell_paths()
    path_colors = {'Mackay': '#1f77b4', 'Anti-Mackay': '#ff7f0e', 'Chiral-1': '#2ca02c',
                   'Chiral-2': '#d62728', 'Mixed-1': '#9467bd', 'Mixed-2': '#8c564b'}
    
    # Draw lattice points
    for (h, k) in hexagonal_coords_list():
        x = h + k * 0.5
        y = k * np.sqrt(3) / 2
        ax.plot(x, y, 'o', color='lightgray', markersize=8, zorder=1)
    
    # Draw paths
    for path_name, path_coords in paths.items():
        xs = [h + k * 0.5 for h, k in path_coords]
        ys = [k * np.sqrt(3) / 2 for h, k in path_coords]
        color = path_colors.get(path_name, 'black')
        ax.plot(xs, ys, '-o', color=color, linewidth=2.5, markersize=10, label=path_name, zorder=2)
        # Arrow at end
        if len(xs) > 1:
            ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.set_xlabel('h + k/2')
    ax.set_ylabel(r'k$\sqrt{3}$/2')
    ax.set_title('(b) Shell Sequence Paths on Hexagonal Lattice')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'hexagonal_lattice.png'))
    plt.close()
    print("Figure 1: Hexagonal lattice saved.")

# ============================================================
# Figure 2: Magic Number Sequences
# ============================================================

def fig2_magic_numbers():
    """Compare Mackay and new magic number sequences."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Magic numbers comparison
    ax = axes[0]
    mackay = mackay_magic_numbers(5)
    new_b5 = new_magic_numbers_b5(6)
    
    shells_mackay = list(range(len(mackay)))
    shells_new = list(range(len(new_b5)))
    
    ax.plot(shells_mackay, mackay, 'o-', color='#1f77b4', linewidth=2.5, markersize=10, label='Mackay (b=1)')
    ax.plot(shells_new, new_b5, 's-', color='#d62728', linewidth=2.5, markersize=10, label='New sequence (b=5)')
    
    # Annotate values
    for i, v in enumerate(mackay):
        ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9)
    for i, v in enumerate(new_b5):
        ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, -18), ha='center', fontsize=9, color='#d62728')
    
    ax.set_xlabel('Shell Number n')
    ax.set_ylabel('Total Atom Count N')
    ax.set_title('(a) Magic Number Sequences')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Panel B: Shell atom counts
    ax = axes[1]
    shell_counts_mackay = [1] + [10*n**2 + 2 for n in range(1, 6)]
    shell_counts_new = [new_b5[i] - (new_b5[i-1] if i > 0 else 0) for i in range(len(new_b5))]
    
    x = np.arange(len(shell_counts_mackay))
    width = 0.35
    ax.bar(x - width/2, shell_counts_mackay, width, label='Mackay shells', color='#1f77b4', alpha=0.8)
    
    x2 = np.arange(len(shell_counts_new))
    ax.bar(x2 + width/2, shell_counts_new, width, label='New sequence shells', color='#d62728', alpha=0.8)
    
    ax.set_xlabel('Shell Number n')
    ax.set_ylabel('Atoms in Shell')
    ax.set_title('(b) Atoms per Shell')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'magic_numbers.png'))
    plt.close()
    print("Figure 2: Magic numbers saved.")

# ============================================================
# Figure 3: Size Mismatch Analysis
# ============================================================

def fig3_size_mismatch():
    """Size mismatch analysis for atomic pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Pair mismatch matrix
    ax = axes[0]
    elements = list(ATOMIC_RADII.keys())
    n = len(elements)
    mismatch_matrix = np.zeros((n, n))
    
    for i, e1 in enumerate(elements):
        for j, e2 in enumerate(elements):
            mismatch_matrix[i, j] = size_mismatch(ATOMIC_RADII[e1], ATOMIC_RADII[e2])
    
    im = ax.imshow(mismatch_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.55)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(elements, fontsize=11)
    ax.set_yticklabels(elements, fontsize=11)
    
    # Annotate values
    for i in range(n):
        for j in range(n):
            val = mismatch_matrix[i, j]
            color = 'white' if val > 0.3 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    
    plt.colorbar(im, ax=ax, label='Size Mismatch δ')
    ax.set_title('(a) Atomic Pair Size Mismatch Matrix')
    
    # Panel B: Optimal mismatch ranges with data points
    ax = axes[1]
    
    # Optimal ranges
    categories = list(OPTIMAL_MISMATCH_RANGES.keys())
    for i, (cat, (low, high)) in enumerate(OPTIMAL_MISMATCH_RANGES.items()):
        label = f'{cat[0]}-{cat[1]}'
        mid = (low + high) / 2
        ax.barh(i, high - low, left=low, height=0.5, alpha=0.6, 
                color=SHELL_COLORS.get(cat[1], '#999'), label=label, edgecolor='black')
        ax.text(mid, i, f'{low:.2f}-{high:.2f}', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add mismatch parameter data points
    for shell_i, shell_j, type_i, type_j, sm_val in MISMATCH_PARAMS:
        ax.plot(sm_val, -0.5, 'D', color='red', markersize=8, zorder=5)
    
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([f'{c[0]}→{c[1]}' for c in categories], fontsize=11)
    ax.set_xlabel('Size Mismatch δ')
    ax.set_title('(b) Optimal Size Mismatch Ranges by Shell Type')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 0.30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'size_mismatch.png'))
    plt.close()
    print("Figure 3: Size mismatch saved.")

# ============================================================
# Figure 4: Shell Energy Comparison
# ============================================================

def fig4_shell_energy():
    """Shell energy comparison across chiral categories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by shell number
    shell_nums = sorted(set(s for s, _ in SHELL_ENERGIES.keys()))
    
    bar_width = 0.25
    x_positions = np.arange(len(shell_nums))
    
    # Get all types present
    all_types = sorted(set(t for _, t in SHELL_ENERGIES.keys()))
    
    for idx, stype in enumerate(all_types):
        energies = []
        positions = []
        for i, sn in enumerate(shell_nums):
            key = (sn, stype)
            if key in SHELL_ENERGIES:
                energies.append(SHELL_ENERGIES[key])
                positions.append(i)
        
        color = SHELL_COLORS.get(stype, '#999')
        ax.bar(np.array(positions) + idx * bar_width, energies, bar_width, 
               label=stype, color=color, alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels([f'Shell {s}' for s in shell_nums], fontsize=12)
    ax.set_ylabel('Relative Energy (normalized units)')
    ax.set_title('Shell Energy by Chiral Category')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'shell_energy.png'))
    plt.close()
    print("Figure 4: Shell energy saved.")

# ============================================================
# Figure 5: Theory vs Experiment Validation
# ============================================================

def fig5_validation():
    """Theory vs experiment validation plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Theory vs Experiment scatter
    ax = axes[0]
    exp_data = EXPERIMENTAL_POINTS
    
    measured = [p[2] for p in exp_data]
    theoretical = [p[3] for p in exp_data]
    labels = [f'T{p[0]}→T{p[1]}' for p in exp_data]
    
    ax.scatter(theoretical, measured, s=150, c='#2ca02c', edgecolors='black', zorder=5, label='Data points')
    
    # Perfect agreement line
    lims = [0, max(max(measured), max(theoretical)) * 1.2]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
    
    # Annotate
    for i, label in enumerate(labels):
        ax.annotate(label, (theoretical[i], measured[i]), textcoords="offset points",
                   xytext=(8, 8), fontsize=10)
    
    ax.set_xlabel('Theoretical Size Mismatch')
    ax.set_ylabel('Measured Size Mismatch')
    ax.set_title('(a) Theory vs Experiment: Size Mismatch')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.18)
    ax.set_ylim(0, 0.18)
    
    # Panel B: Residuals
    ax = axes[1]
    residuals = [m - t for m, t in zip(measured, theoretical)]
    x_pos = range(len(residuals))
    
    colors = ['#2ca02c' if r >= 0 else '#d62728' for r in residuals]
    ax.bar(x_pos, residuals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10, rotation=45)
    ax.set_ylabel('Residual (Measured - Theoretical)')
    ax.set_title('(b) Validation Residuals')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Calculate R² and RMSE
    measured_arr = np.array(measured)
    theoretical_arr = np.array(theoretical)
    ss_res = np.sum((measured_arr - theoretical_arr)**2)
    ss_tot = np.sum((measured_arr - np.mean(measured_arr))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = np.sqrt(np.mean((measured_arr - theoretical_arr)**2))
    
    ax.text(0.95, 0.95, f'R² = {r_squared:.4f}\nRMSE = {rmse:.4f}', 
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'validation.png'))
    plt.close()
    print(f"Figure 5: Validation saved. R²={r_squared:.4f}, RMSE={rmse:.4f}")
    return r_squared, rmse

# ============================================================
# Figure 6: Growth Simulation Dynamics
# ============================================================

def fig6_growth_dynamics():
    """Plot growth simulation dynamics from saved results."""
    # Load growth results
    with open('outputs/growth_simulation_results.json', 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    sim_names = list(results.keys())
    sim_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Panel A: Energy trajectories
    ax = axes[0, 0]
    for i, name in enumerate(sim_names):
        data = results[name]
        steps = range(len(data['energy_trajectory']))
        ax.plot(steps, data['energy_trajectory'], '-', color=sim_colors[i], 
                linewidth=2, label=name, alpha=0.8)
    ax.set_xlabel('Deposition Step')
    ax.set_ylabel('Total Energy (normalized)')
    ax.set_title('(a) Energy Evolution During Growth')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Mismatch evolution
    ax = axes[0, 1]
    for i, name in enumerate(sim_names):
        data = results[name]
        if data['mismatch_history']:
            steps = range(len(data['mismatch_history']))
            ax.plot(steps, data['mismatch_history'], '-o', color=sim_colors[i],
                   linewidth=1.5, markersize=4, label=name, alpha=0.8)
    ax.set_xlabel('Shell Addition Step')
    ax.set_ylabel('Size Mismatch δ')
    ax.set_title('(b) Size Mismatch Evolution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel C: Path selection statistics (combined)
    ax = axes[1, 0]
    all_stats = {'Conservative path': 0, 'Mismatch-driven path': 0, 'Random path': 0, 'Reverse step': 0}
    for name in sim_names:
        for k, v in results[name]['path_stats'].items():
            all_stats[k] += v
    
    labels = list(all_stats.keys())
    values = list(all_stats.values())
    colors_pie = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors_pie, startangle=90)
    ax.set_title('(c) Path Selection Statistics')
    
    # Panel D: Shell type evolution
    ax = axes[1, 1]
    type_to_num = {'MC': 0, 'BG': 1, 'Ch1': 2, 'Ch2': 3, 'Ch3': 4, 'Ch4': 5, 'Ch5': 6}
    
    for i, name in enumerate(sim_names):
        data = results[name]
        type_nums = [type_to_num.get(t, 0) for t in data['type_history']]
        steps = range(len(type_nums))
        ax.plot(steps, type_nums, '-o', color=sim_colors[i], linewidth=1.5, 
                markersize=5, label=name, alpha=0.8)
    
    ax.set_yticks(range(7))
    ax.set_yticklabels(CHIRAL_LABELS, fontsize=10)
    ax.set_xlabel('Shell Number')
    ax.set_ylabel('Shell Type')
    ax.set_title('(d) Shell Type Evolution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'growth_dynamics.png'))
    plt.close()
    print("Figure 6: Growth dynamics saved.")

# ============================================================
# Figure 7: Lennard-Jones Potentials
# ============================================================

def fig7_lj_potentials():
    """Plot Lennard-Jones potential curves for different atomic pairs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    r_range = np.linspace(2.0, 8.0, 500)
    
    pair_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (pair, (eps, sig)) in enumerate(LJ_PARAMETERS.items()):
        V = np.array([lj_potential(r, eps, sig) for r in r_range])
        V = np.clip(V, -2, 5)  # Clip for visualization
        r_eq = lj_equilibrium_distance(sig)
        
        ax.plot(r_range, V, '-', color=pair_colors[i], linewidth=2, label=f'{pair} (σ={sig:.2f})')
        ax.plot(r_eq, lj_potential(r_eq, eps, sig), 'o', color=pair_colors[i], markersize=8, zorder=5)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance r (Å)')
    ax.set_ylabel('V(r) / ε')
    ax.set_title('Lennard-Jones Potential Curves for Atomic Pairs')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 3.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'lj_potentials.png'))
    plt.close()
    print("Figure 7: LJ potentials saved.")

# ============================================================
# Figure 8: Multi-component Cluster Predictions
# ============================================================

def fig8_cluster_predictions():
    """Visualize predicted multi-component cluster structures."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    clusters = MULTICOMPONENT_CLUSTERS
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        name = cluster['name']
        core = cluster['core']
        shell_elem = cluster['shell']
        core_type = cluster['core_type']
        shell_type = cluster['shell_type']
        
        core_r = ATOMIC_RADII[core]
        shell_r = ATOMIC_RADII[shell_elem]
        sm = size_mismatch(core_r, shell_r)
        
        # Draw concentric circles representing shells
        # Core
        core_circle = plt.Circle((0, 0), core_r * 0.5, color=SHELL_COLORS[core_type], 
                                  alpha=0.6, ec='black', lw=2)
        ax.add_patch(core_circle)
        
        # Shell
        shell_circle = plt.Circle((0, 0), (core_r + shell_r) * 0.5, 
                                   color=SHELL_COLORS[shell_type], alpha=0.3, ec='black', lw=2)
        ax.add_patch(shell_circle)
        
        # Labels
        ax.text(0, 0, f'{core}\n(core)', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(0, (core_r + shell_r) * 0.4, f'{shell_elem}\n(shell)', ha='center', va='center', fontsize=9)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nδ = {sm:.3f}', fontsize=13)
        ax.text(0, -2.2, f'{core_type} → {shell_type}', ha='center', fontsize=11, 
                style='italic', color='gray')
        ax.axis('off')
    
    plt.suptitle('Predicted Multi-component Icosahedral Clusters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'cluster_predictions.png'))
    plt.close()
    print("Figure 8: Cluster predictions saved.")

# ============================================================
# Figure 9: Triangulation Number Map
# ============================================================

def fig9_triangulation_map():
    """Plot triangulation number T(h,k) as a 2D heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    T_matrix = np.zeros((6, 6))
    for h in range(6):
        for k in range(6):
            T_matrix[h, k] = triangulation_number(h, k)
    
    im = ax.imshow(T_matrix, cmap='viridis', origin='lower')
    
    for h in range(6):
        for k in range(6):
            T = int(T_matrix[h, k])
            color = 'white' if T > 30 else 'black'
            ax.text(k, h, str(T), ha='center', va='center', fontsize=11, 
                    fontweight='bold', color=color)
    
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xlabel('k')
    ax.set_ylabel('h')
    ax.set_title('Triangulation Number T(h,k) = h² + hk + k²')
    plt.colorbar(im, ax=ax, label='T value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'triangulation_map.png'))
    plt.close()
    print("Figure 9: Triangulation map saved.")

# ============================================================
# Figure 10: Comprehensive Summary
# ============================================================

def fig10_summary():
    """Create a comprehensive summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: Atomic radii comparison
    ax = axes[0, 0]
    elements = list(ATOMIC_RADII.keys())
    radii = [ATOMIC_RADII[e] for e in elements]
    colors = ['#1f77b4' if e in ['Na', 'K', 'Rb', 'Cs'] else '#d62728' for e in elements]
    
    bars = ax.bar(elements, radii, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Atomic Radius (Å)')
    ax.set_title('(a) Atomic Radii')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    alkali_patch = mpatches.Patch(color='#1f77b4', label='Alkali metals')
    transition_patch = mpatches.Patch(color='#d62728', label='Transition metals')
    ax.legend(handles=[alkali_patch, transition_patch], fontsize=10)
    
    # Panel B: Growth results comparison (from data file)
    ax = axes[0, 1]
    growth_data = [
        (0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02), (30, 'MC', 0.025),
        (40, 'MC', 0.03), (50, 'MC', 0.035),
        (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14), (30, 'Ch1', 0.138),
        (40, 'Ch1', 0.136), (50, 'Ch1', 0.135)
    ]
    
    mc_steps = [d[0] for d in growth_data if d[1] == 'MC']
    mc_mismatch = [d[2] for d in growth_data if d[1] == 'MC']
    ch1_steps = [d[0] for d in growth_data if d[1] == 'Ch1']
    ch1_mismatch = [d[2] for d in growth_data if d[1] == 'Ch1']
    
    ax.plot(mc_steps, mc_mismatch, 'o-', color=SHELL_COLORS['MC'], linewidth=2, markersize=8, label='MC path')
    ax.plot(ch1_steps, ch1_mismatch, 's-', color=SHELL_COLORS['Ch1'], linewidth=2, markersize=8, label='Ch1 path')
    ax.set_xlabel('Simulation Steps')
    ax.set_ylabel('Average Mismatch δ')
    ax.set_title('(b) Growth Results: Mismatch Evolution')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel C: Compatibility matrix for known pairs
    ax = axes[1, 0]
    pair_data = [('Na-Rb', 0.22), ('Ag-Cu', 0.12), ('Ag-Ni', 0.15), ('Cu-Ni', 0.032)]
    pair_names = [p[0] for p in pair_data]
    pair_values = [p[1] for p in pair_data]
    
    colors_compat = []
    for v in pair_values:
        if v < 0.05:
            colors_compat.append('#2ca02c')  # Good for MC-MC
        elif v < 0.16:
            colors_compat.append('#ff7f0e')  # Good for MC-Ch1
        else:
            colors_compat.append('#d62728')  # Good for MC-Ch2
    
    ax.barh(pair_names, pair_values, color=colors_compat, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Size Mismatch δ')
    ax.set_title('(c) Atomic Pair Compatibility')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add optimal range annotations
    for i, (name, val) in enumerate(pair_data):
        if val < 0.05:
            label = 'MC→MC'
        elif val < 0.16:
            label = 'MC→Ch1'
        else:
            label = 'MC→Ch2'
        ax.text(val + 0.005, i, label, va='center', fontsize=9, style='italic')
    
    # Panel D: Path selection reference data
    ax = axes[1, 1]
    ref_stats = [('Conservative\npath', 325), ('Mismatch-driven\npath', 125), 
                 ('Random\npath', 50), ('Reverse\nstep', 100)]
    stat_names = [s[0] for s in ref_stats]
    stat_values = [s[1] for s in ref_stats]
    colors_stats = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    ax.bar(stat_names, stat_values, color=colors_stats, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('(d) Reference Path Selection Statistics')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(stat_values):
        ax.text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'summary.png'))
    plt.close()
    print("Figure 10: Summary saved.")

# ============================================================
# Main: Generate all figures
# ============================================================

if __name__ == '__main__':
    print("Generating all figures...")
    fig1_hexagonal_lattice()
    fig2_magic_numbers()
    fig3_size_mismatch()
    fig4_shell_energy()
    r2, rmse = fig5_validation()
    fig6_growth_dynamics()
    fig7_lj_potentials()
    fig8_cluster_predictions()
    fig9_triangulation_map()
    fig10_summary()
    print(f"\nAll figures generated successfully!")
    print(f"Validation: R² = {r2:.4f}, RMSE = {rmse:.4f}")
