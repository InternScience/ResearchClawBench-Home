#!/usr/bin/env python3
"""
Analysis code for multi-component icosahedral nanocluster data.
This script processes the reproduction data and generates visualizations
for the research report.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("husl")

# Data from the reproduction file
# =================================

# Hexagonal coordinate sequence
hexagonal_coords = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), 
                    (2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), 
                    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5), (5,0), (5,1), (5,2), (5,3), (5,4), (5,5)]

# Mackay magic number sequence
mackay_sequence = [1, 13, 55, 147, 309]

# New magic number sequence (b=5)
new_sequence_b5 = [1, 13, 45, 117, 239, 431]

# Chiral category labels
chiral_labels = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']

# Geometric constants
sin_2pi_5 = 0.9510565162951535
cos_2pi_5 = 0.3090169943749474

# Shell color mapping
shell_colors = {'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c', 'Ch2': '#d62728', 
                'Ch3': '#9467bd', 'Ch4': '#8c564b', 'Ch5': '#e377c2'}

# Atomic physical parameters (atomic radius, Å)
atomic_radii = [('Na', 1.86), ('K', 2.27), ('Rb', 2.48), ('Cs', 2.65), ('Ag', 1.44), ('Cu', 1.28), ('Ni', 1.24)]

# Atomic pair compatibility data
atomic_pairs_compatibility = [('Na', 'Rb', 0.22), ('Ag', 'Cu', 0.12), ('Ag', 'Ni', 0.15), ('Cu', 'Ni', 0.032)]

# Optimal size mismatch range
optimal_mismatch_ranges = [('MC', 'MC', 0.03, 0.05), ('MC', 'Ch1', 0.12, 0.16), ('MC', 'Ch2', 0.19, 0.22), ('MC', 'BG', 0.08, 0.10)]

# Multi-component cluster validation data
multicomponent_clusters = [('Na13@Rb32', 'Na', 'Rb', 'MC', 'Ch1'), ('K13@Cs42', 'K', 'Cs', 'MC', 'Ch2'), ('Ag13@Cu45', 'Ag', 'Cu', 'MC', 'Ch1')]

# Shell energy relative values (normalized units)
shell_energies = [(1, 'MC', 0.00), (2, 'MC', -2.35), (2, 'Ch1', -2.15), (3, 'MC', -4.82), (3, 'Ch1', -4.61), (3, 'BG', -4.55)]

# Size mismatch calculation parameters
mismatch_params = [(1, 2, 'MC', 'MC', 0.04), (1, 2, 'MC', 'Ch1', 0.14), (2, 3, 'MC', 'MC', 0.038), (2, 3, 'MC', 'Ch1', 0.136), (2, 3, 'Ch1', 'Ch2', 0.21)]

# Experimental validation points (T_i, T_{i+1}, measured sm, theoretical sm)
experimental_points = [(1, 3, 0.048, 0.045), (3, 4, 0.042, 0.044), (4, 7, 0.138, 0.142), (7, 12, 0.132, 0.139)]

# Growth simulation parameters
growth_parameters = [('temperature', 300.0), ('deposition_rate', 0.01), ('simulation_steps', 1000), 
                     ('beta_factor', 1.0), ('delta_opt', 0.04), ('random_seed', 42)]

# Path probability weights
path_probability_weights = [('conservative_step', 0.65), ('mismatch_driven_step', 0.25), ('random_step', 0.10)]

# Initial seed structure
initial_seeds = [('Na13', 'MC', [(0,0)], [('Na', 1.86)]), ('Na13@Rb32', 'Ch1', [(0,0), (1,0)], [('Na', 1.86), ('Rb', 2.48)]), 
                 ('Ag13', 'MC', [(0,0)], [('Ag', 1.44)])]

# Growth experimental result data (steps, chiral category, average mismatch)
growth_results = [(0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02), (30, 'MC', 0.025), (40, 'MC', 0.03), (50, 'MC', 0.035), 
                  (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14), (30, 'Ch1', 0.138), (40, 'Ch1', 0.136), (50, 'Ch1', 0.135), 
                  (0, 'MC', 0.00), (10, 'MC', 0.08), (20, 'Ch1', 0.14), (30, 'Ch1', 0.15), (40, 'Ch1', 0.145), (50, 'Ch1', 0.142)]

# Path selection statistics
path_selection_stats = [('Conservative path', 325), ('Mismatch-driven path', 125), ('Random path', 50), ('Reverse step', 100)]

# Potential function parameters (Lennard-Jones parameters: ε, σ)
lj_parameters = [('Na-Na', 1.0, 3.72), ('Rb-Rb', 1.0, 4.96), ('Cs-Cs', 1.0, 5.30), ('Ag-Ag', 1.0, 2.88), 
                 ('Cu-Cu', 1.0, 2.56), ('Na-Rb', 1.0, 4.34), ('Ag-Cu', 1.0, 2.72)]

# Thermodynamic parameters
thermodynamic_params = [('kT', 0.02585), ('boltzmann', 8.617e-5), ('pressure', 1.0), ('timestep', 0.001)]

print("Data loaded successfully!")
print(f"Total hexagonal coordinates: {len(hexagonal_coords)}")
print(f"Mackay sequence: {mackay_sequence}")
print(f"New sequence (b=5): {new_sequence_b5}")

# ============================================
# Figure 1: Magic Number Sequences Comparison
# ============================================
def plot_magic_numbers():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shells = np.arange(1, 6)
    mackay_extended = mackay_sequence + [None]  # Pad to match length
    new_extended = new_sequence_b5[:5]
    
    x = np.arange(len(shells))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mackay_sequence, width, label='Mackay Sequence', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, new_extended, width, label='New Sequence (b=5)', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Shell Number', fontsize=12)
    ax.set_ylabel('Number of Atoms', fontsize=12)
    ax.set_title('Comparison of Magic Number Sequences for Icosahedral Shells', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Shell {i}' for i in shells])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig1_magic_numbers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved: Magic Number Sequences")

# ============================================
# Figure 2: Atomic Radii and Size Mismatch
# ============================================
def plot_atomic_radii():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Atomic radii
    elements = [a[0] for a in atomic_radii]
    radii = [a[1] for a in atomic_radii]
    colors = plt.cm.viridis(np.linspace(0, 1, len(elements)))
    
    bars = ax1.barh(elements, radii, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Atomic Radius (Å)', fontsize=12)
    ax1.set_title('Atomic Radii of Candidate Elements', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (bar, radius) in enumerate(zip(bars, radii)):
        ax1.text(radius + 0.05, bar.get_y() + bar.get_height()/2, f'{radius:.2f} Å', 
                va='center', fontsize=10)
    
    # Right panel: Size mismatch ranges
    categories = [f'{r[0]}-{r[1]}' for r in optimal_mismatch_ranges]
    min_mismatch = [r[2] for r in optimal_mismatch_ranges]
    max_mismatch = [r[3] for r in optimal_mismatch_ranges]
    
    y_pos = np.arange(len(categories))
    
    for i, (cat, min_val, max_val) in enumerate(zip(categories, min_mismatch, max_mismatch)):
        ax2.barh(i, max_val - min_val, left=min_val, height=0.6, color=plt.cm.RdYlBu(0.3 + i*0.15), 
                alpha=0.8, edgecolor='black')
        ax2.text((min_val + max_val)/2, i, f'{min_val:.2f}-{max_val:.2f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories)
    ax2.set_xlabel('Optimal Size Mismatch Range', fontsize=12)
    ax2.set_title('Optimal Size Mismatch Between Shell Categories', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 0.25)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig2_atomic_radii.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved: Atomic Radii and Size Mismatch")

# ============================================
# Figure 3: Shell Energy Landscape
# ============================================
def plot_shell_energies():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Organize data by shell number
    shell_data = {}
    for shell_num, category, energy in shell_energies:
        if shell_num not in shell_data:
            shell_data[shell_num] = {}
        shell_data[shell_num][category] = energy
    
    shell_nums = sorted(shell_data.keys())
    categories = ['MC', 'Ch1', 'BG']
    x_offset = {'MC': -0.25, 'Ch1': 0, 'BG': 0.25}
    markers = {'MC': 'o', 'Ch1': 's', 'BG': '^'}
    
    for cat in categories:
        x_vals = []
        y_vals = []
        for shell_num in shell_nums:
            if cat in shell_data[shell_num]:
                x_vals.append(shell_num + x_offset[cat])
                y_vals.append(shell_data[shell_num][cat])
        ax.scatter(x_vals, y_vals, label=cat, marker=markers[cat], s=150, 
                  color=shell_colors[cat], alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.plot(x_vals, y_vals, '--', color=shell_colors[cat], alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Shell Number', fontsize=12)
    ax.set_ylabel('Relative Energy (normalized units)', fontsize=12)
    ax.set_title('Shell Energy Landscape by Chiral Category', fontsize=14, fontweight='bold')
    ax.set_xticks(shell_nums)
    ax.legend(fontsize=11, title='Category')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig3_shell_energies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved: Shell Energy Landscape")

# ============================================
# Figure 4: Experimental Validation
# ============================================
def plot_experimental_validation():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ti_vals = [p[0] for p in experimental_points]
    ti1_vals = [p[1] for p in experimental_points]
    measured = [p[2] for p in experimental_points]
    theoretical = [p[3] for p in experimental_points]
    
    # Create scatter plot
    x_labels = [f'T{ti}→T{ti1}' for ti, ti1 in zip(ti_vals, ti1_vals)]
    x_pos = np.arange(len(x_labels))
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, measured, width, label='Measured', color='#2ca02c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, theoretical, width, label='Theoretical', color='#d62728', alpha=0.8, edgecolor='black')
    
    # Add error bars showing difference
    for i, (m, t) in enumerate(zip(measured, theoretical)):
        diff = abs(m - t)
        ax.annotate(f'Δ={diff:.3f}', xy=(i, max(m, t) + 0.005), ha='center', fontsize=9, color='blue')
    
    ax.set_xlabel('Shell Transition', fontsize=12)
    ax.set_ylabel('Size Mismatch', fontsize=12)
    ax.set_title('Experimental Validation of Size Mismatch Predictions', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig4_experimental_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved: Experimental Validation")

# ============================================
# Figure 5: Growth Simulation Results
# ============================================
def plot_growth_simulation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parse growth results
    mc_data = [(r[0], r[2]) for r in growth_results if r[1] == 'MC']
    ch1_data = [(r[0], r[2]) for r in growth_results if r[1] == 'Ch1']
    
    # Left panel: Mismatch evolution
    if mc_data:
        steps_mc, mismatch_mc = zip(*mc_data[:6])
        ax1.plot(steps_mc, mismatch_mc, 'o-', label='MC Category', color=shell_colors['MC'], linewidth=2, markersize=8)
    if ch1_data:
        steps_ch1, mismatch_ch1 = zip(*ch1_data[:6])
        ax1.plot(steps_ch1, mismatch_ch1, 's-', label='Ch1 Category', color=shell_colors['Ch1'], linewidth=2, markersize=8)
    
    ax1.set_xlabel('Simulation Steps', fontsize=12)
    ax1.set_ylabel('Average Size Mismatch', fontsize=12)
    ax1.set_title('Size Mismatch Evolution During Growth', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Right panel: Path selection statistics
    paths = [p[0] for p in path_selection_stats]
    counts = [p[1] for p in path_selection_stats]
    colors = plt.cm.Set3(np.linspace(0, 1, len(paths)))
    
    wedges, texts, autotexts = ax2.pie(counts, labels=paths, autopct='%1.1f%%', colors=colors,
                                        startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Path Selection Statistics\n(Growth Simulation)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig5_growth_simulation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 5 saved: Growth Simulation Results")

# ============================================
# Figure 6: Lennard-Jones Parameters
# ============================================
def plot_lj_parameters():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pairs = [p[0] for p in lj_parameters]
    epsilon = [p[1] for p in lj_parameters]
    sigma = [p[2] for p in lj_parameters]
    
    x = np.arange(len(pairs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, epsilon, width, label='ε (well depth)', color='#1f77b4', alpha=0.8, edgecolor='black')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, sigma, width, label='σ (collision diameter)', color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Atomic Pair', fontsize=12)
    ax.set_ylabel('ε (eV)', fontsize=12, color='#1f77b4')
    ax2.set_ylabel('σ (Å)', fontsize=12, color='#ff7f0e')
    ax.set_title('Lennard-Jones Potential Parameters', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig6_lj_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 6 saved: Lennard-Jones Parameters")

# ============================================
# Figure 7: Hexagonal Lattice Visualization
# ============================================
def plot_hexagonal_lattice():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract coordinates
    coords = np.array(hexagonal_coords)
    
    # Plot hexagonal grid points
    ax.scatter(coords[:, 0], coords[:, 1], s=200, c='#1f77b4', alpha=0.8, edgecolor='black', linewidth=2, zorder=3)
    
    # Draw hexagonal lattice connections
    for i, (x, y) in enumerate(hexagonal_coords):
        # Connect to neighbors in hexagonal lattice
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x+1, y-1), (x-1, y+1)]
        for nx, ny in neighbors:
            if (nx, ny) in hexagonal_coords:
                ax.plot([x, nx], [y, ny], 'b-', alpha=0.3, linewidth=1, zorder=1)
        
        # Add coordinate labels
        ax.annotate(f'({x},{y})', (x, y), textcoords="offset points", xytext=(0, -15), 
                   ha='center', fontsize=7, alpha=0.7)
    
    # Highlight specific path
    path_coords = [(0,0), (0,1), (1,1), (1,2), (2,2)]
    path_x = [c[0] for c in path_coords]
    path_y = [c[1] for c in path_coords]
    ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7, zorder=2, label='Example Shell Path')
    ax.scatter(path_x, path_y, s=300, c='red', alpha=0.6, edgecolor='darkred', linewidth=2, zorder=4)
    
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.set_xlabel('h coordinate', fontsize=12)
    ax.set_ylabel('k coordinate', fontsize=12)
    ax.set_title('Hexagonal Lattice Coordinate System\nfor Icosahedral Shell Stacking', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig7_hexagonal_lattice.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 7 saved: Hexagonal Lattice")

# ============================================
# Figure 8: Multi-component Clusters Summary
# ============================================
def plot_multicomponent_clusters():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    clusters = [c[0] for c in multicomponent_clusters]
    core_elements = [c[1] for c in multicomponent_clusters]
    shell_elements = [c[2] for c in multicomponent_clusters]
    core_cat = [c[3] for c in multicomponent_clusters]
    shell_cat = [c[4] for c in multicomponent_clusters]
    
    y_pos = np.arange(len(clusters))
    
    # Create horizontal bar chart
    colors = [shell_colors[shell_cat[i]] for i in range(len(clusters))]
    bars = ax.barh(y_pos, [1]*len(clusters), color=colors, alpha=0.7, edgecolor='black')
    
    # Add labels
    for i, (cluster, core, shell, cc, sc) in enumerate(zip(clusters, core_elements, shell_elements, core_cat, shell_cat)):
        ax.text(0.5, i, f'{cluster}: {core} (core, {cc}) + {shell} (shell, {sc})', 
               ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Validated Multi-component Clusters', fontsize=12)
    ax.set_title('Experimentally Validated Multi-shell Icosahedral Clusters', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    
    # Add legend for categories
    legend_elements = [mpatches.Patch(color=shell_colors[cat], label=cat) for cat in set(shell_cat)]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, title='Shell Category')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig8_multicomponent_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 8 saved: Multi-component Clusters")

# ============================================
# Figure 9: Size Mismatch Parameter Analysis
# ============================================
def plot_mismatch_analysis():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Parse mismatch params
    shell_transitions = [f'{p[0]}→{p[1]}' for p in mismatch_params]
    cat_pairs = [f'{p[2]}-{p[3]}' for p in mismatch_params]
    mismatch_values = [p[4] for p in mismatch_params]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(mismatch_values)))
    
    bars = ax.barh(shell_transitions, mismatch_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add category labels
    for i, (bar, cat_pair, val) in enumerate(zip(bars, cat_pairs, mismatch_values)):
        ax.text(val + 0.005, i, f'{cat_pair}: {val:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Size Mismatch Value', fontsize=12)
    ax.set_ylabel('Shell Transition', fontsize=12)
    ax.set_title('Optimal Size Mismatch Parameters by Shell Transition', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 0.25)
    
    plt.tight_layout()
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig9_mismatch_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 9 saved: Mismatch Analysis")

# ============================================
# Figure 10: Theory Framework Summary
# ============================================
def plot_framework_summary():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main central concept
    ax_center = fig.add_subplot(gs[1, 1])
    ax_center.text(0.5, 0.5, 'Multi-shell\nIcosahedral\nTheory', ha='center', va='center', 
                  fontsize=20, fontweight='bold', 
                  bbox=dict(boxstyle='circle', facecolor='lightblue', edgecolor='navy', linewidth=3))
    ax_center.set_xlim(0, 1)
    ax_center.set_ylim(0, 1)
    ax_center.axis('off')
    
    # Surrounding concepts
    concepts = [
        ('Hexagonal\nLattice', 0.5, 0.85, gs[0, 1]),
        ('Magic Number\nSequences', 0.15, 0.5, gs[1, 0]),
        ('Chiral\nCategories', 0.85, 0.5, gs[1, 2]),
        ('Size Mismatch\nOptimization', 0.5, 0.15, gs[2, 1]),
        ('Growth\nSimulations', 0.15, 0.15, gs[2, 0]),
        ('Lennard-Jones\nPotentials', 0.85, 0.15, gs[2, 2]),
    ]
    
    for text, x, y, sub_gs in concepts:
        ax = fig.add_subplot(sub_gs)
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', linewidth=2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    fig.suptitle('Universal Theoretical Framework for Multi-component Nanoclusters', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/report/images/fig10_framework_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 10 saved: Framework Summary")

# ============================================
# Generate all figures
# ============================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating figures for research report...")
    print("="*60 + "\n")
    
    plot_magic_numbers()
    plot_atomic_radii()
    plot_shell_energies()
    plot_experimental_validation()
    plot_growth_simulation()
    plot_lj_parameters()
    plot_hexagonal_lattice()
    plot_multicomponent_clusters()
    plot_mismatch_analysis()
    plot_framework_summary()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
