#!/usr/bin/env python3
"""
Multi-Component Icosahedral Shell Stacking Analysis
Reproduction and analysis of simulation data from 
"General theory for packing icosahedral shells into multi-component aggregates"
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import json
import os
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Create output directories
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = Path("report/images")
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

###############################################################################
# 1. Core Theory Data
###############################################################################

# Hexagonal coordinate sequence
hexagonal_coords = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), 
                    (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),
                    (2,0), (2,1), (2,2), (2,3), (2,4), (2,5),
                    (3,0), (3,1), (3,2), (3,3), (3,4), (3,5),
                    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5),
                    (5,0), (5,1), (5,2), (5,3), (5,4), (5,5)]

# Mackay magic number sequence (traditional icosahedral)
mackay_sequence = [1, 13, 55, 147, 309]

# New magic number sequence (b=5 chiral variant)
new_sequence_b5 = [1, 13, 45, 117, 239, 431]

# Chiral category labels
chiral_labels = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']

# Geometric constants for icosahedral geometry
sin_2pi_5 = 0.9510565162951535
cos_2pi_5 = 0.3090169943749474

# Shell color mapping for visualization
shell_colors = {
    'MC': '#1f77b4',   # blue
    'BG': '#ff7f0e',   # orange
    'Ch1': '#2ca02c',  # green
    'Ch2': '#d62728',  # red
    'Ch3': '#9467bd',  # purple
    'Ch4': '#8c564b',  # brown
    'Ch5': '#e377c2'   # pink
}

###############################################################################
# 2. Experimental Verification Data
###############################################################################

# Atomic physical parameters (atomic radius in Angstroms)
atomic_radii = {
    'Na': 1.86, 'K': 2.27, 'Rb': 2.48, 'Cs': 2.65,
    'Ag': 1.44, 'Cu': 1.28, 'Ni': 1.24
}

# Atomic pair compatibility data (size mismatch tolerance)
atomic_pairs_compatibility = [
    ('Na', 'Rb', 0.22),
    ('Ag', 'Cu', 0.12),
    ('Ag', 'Ni', 0.15),
    ('Cu', 'Ni', 0.032)
]

# Optimal size mismatch ranges for different chiral combinations
optimal_mismatch_ranges = [
    ('MC', 'MC', 0.03, 0.05),
    ('MC', 'Ch1', 0.12, 0.16),
    ('MC', 'Ch2', 0.19, 0.22),
    ('MC', 'BG', 0.08, 0.10)
]

# Multi-component cluster validation data
multicomponent_clusters = [
    ('Na13@Rb32', 'Na', 'Rb', 'MC', 'Ch1'),
    ('K13@Cs42', 'K', 'Cs', 'MC', 'Ch2'),
    ('Ag13@Cu45', 'Ag', 'Cu', 'MC', 'Ch1')
]

# Shell energy relative values (normalized units, lower is more stable)
shell_energies = [
    (1, 'MC', 0.00),
    (2, 'MC', -2.35),
    (2, 'Ch1', -2.15),
    (3, 'MC', -4.82),
    (3, 'Ch1', -4.61),
    (3, 'BG', -4.55)
]

# Size mismatch calculation parameters
mismatch_params = [
    (1, 2, 'MC', 'MC', 0.04),
    (1, 2, 'MC', 'Ch1', 0.14),
    (2, 3, 'MC', 'MC', 0.038),
    (2, 3, 'MC', 'Ch1', 0.136),
    (2, 3, 'Ch1', 'Ch2', 0.21)
]

# Experimental validation points (T_i, T_{i+1}, measured sm, theoretical sm)
experimental_points = [
    (1, 3, 0.048, 0.045),
    (3, 4, 0.042, 0.044),
    (4, 7, 0.138, 0.142),
    (7, 12, 0.132, 0.139)
]

###############################################################################
# 3. Dynamic Growth Simulation Data
###############################################################################

# Growth simulation parameters
growth_parameters = {
    'temperature': 300.0,
    'deposition_rate': 0.01,
    'simulation_steps': 1000,
    'beta_factor': 1.0,
    'delta_opt': 0.04,
    'random_seed': 42
}

# Growth path probability weights
path_probability_weights = [
    ('conservative_step', 0.65),
    ('mismatch_driven_step', 0.25),
    ('random_step', 0.10)
]

# Initial seed structures
initial_seeds = [
    ('Na13', 'MC', [(0,0)], [('Na', 1.86)]),
    ('Na13@Rb32', 'Ch1', [(0,0), (1,0)], [('Na', 1.86), ('Rb', 2.48)]),
    ('Ag13', 'MC', [(0,0)], [('Ag', 1.44)])
]

# Deposition atom sequences
deposition_sequences = [
    ('Na13 + Na', ['Na']*50),
    ('Na13@Rb32 + Rb', ['Rb']*30),
    ('Ag13 + Cu', ['Cu']*20 + ['Ag']*10 + ['Cu']*20),
    ('Rb72 + Cs', ['Cs']*40)
]

# Growth experimental results (steps, chiral category, average mismatch)
growth_results = [
    (0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02),
    (30, 'MC', 0.025), (40, 'MC', 0.03), (50, 'MC', 0.035),
    (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14),
    (30, 'Ch1', 0.138), (40, 'Ch1', 0.136), (50, 'Ch1', 0.135),
    (0, 'MC', 0.00), (10, 'MC', 0.08), (20, 'Ch1', 0.14),
    (30, 'Ch1', 0.15), (40, 'Ch1', 0.145), (50, 'Ch1', 0.142)
]

# Path selection statistics from simulations
path_selection_stats = [
    ('Conservative path', 325),
    ('Mismatch-driven path', 125),
    ('Random path', 50),
    ('Reverse step', 100)
]

# Lennard-Jones potential parameters (epsilon, sigma)
lj_parameters = {
    ('Na', 'Na'): (1.0, 3.72),
    ('Rb', 'Rb'): (1.0, 4.96),
    ('Cs', 'Cs'): (1.0, 5.30),
    ('Ag', 'Ag'): (1.0, 2.88),
    ('Cu', 'Cu'): (1.0, 2.56),
    ('Na', 'Rb'): (1.0, 4.34),
    ('Ag', 'Cu'): (1.0, 2.72)
}

# Thermodynamic parameters
thermodynamic_params = {
    'kT': 0.02585,
    'boltzmann': 8.617e-5,
    'pressure': 1.0,
    'timestep': 0.001
}

###############################################################################
# Data Export Functions
###############################################################################

def export_method_contract():
    """Export the method contract summary."""
    contract = {
        "task_type": "multi_component_icosahedral_shell_stacking",
        "input_requirements": [
            "particle_types_and_sizes",
            "shell_sequence_paths_hexagonal_lattice",
            "interaction_parameters"
        ],
        "output_deliverables": [
            "predicted_stable_multi_shell_structures",
            "optimal_size_mismatch_values",
            "shell_sequences_from_growth_simulations"
        ],
        "scientific_objective": "universal_theoretical_framework_for_rational_design_of_nanoclusters",
        "key_methods": [
            "hexagonal_coordinate_mapping",
            "magic_number_sequence_analysis",
            "chiral_category_classification",
            "size_mismatch_optimization",
            "dynamic_growth_simulation"
        ]
    }
    with open(OUTPUT_DIR / "method_contract.json", 'w') as f:
        json.dump(contract, f, indent=2)
    return contract

def export_target_artifact_inventory():
    """Export target artifact inventory."""
    inventory = {
        "required_artifacts": [
            {"type": "figure", "name": "magic_number_sequences_comparison", "status": "pending"},
            {"type": "figure", "name": "atomic_radii_comparison", "status": "pending"},
            {"type": "figure", "name": "optimal_mismatch_ranges", "status": "pending"},
            {"type": "figure", "name": "shell_energies_by_chiral_category", "status": "pending"},
            {"type": "figure", "name": "growth_simulation_results", "status": "pending"},
            {"type": "figure", "name": "path_selection_statistics", "status": "pending"},
            {"type": "figure", "name": "experimental_vs_theoretical_mismatch", "status": "pending"},
            {"type": "table", "name": "multicomponent_cluster_predictions", "status": "pending"},
            {"type": "table", "name": "mismatch_parameters_summary", "status": "pending"}
        ],
        "data_sources": [
            "Multi-component Icosahedral Reproduction Data.txt"
        ]
    }
    with open(OUTPUT_DIR / "target_artifact_inventory.json", 'w') as f:
        json.dump(inventory, f, indent=2)
    return inventory

###############################################################################
# Visualization Functions
###############################################################################

def plot_magic_number_sequences():
    """Plot comparison of Mackay vs new magic number sequences."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shell_indices = range(len(mackay_sequence))
    ax.plot(shell_indices, mackay_sequence, 'o-', label='Mackay Sequence', 
            color=shell_colors['MC'], linewidth=2, markersize=10)
    
    # Extend indices for new sequence
    new_indices = range(len(new_sequence_b5))
    ax.plot(new_indices, new_sequence_b5, 's-', label='New Sequence (b=5)', 
            color=shell_colors['Ch1'], linewidth=2, markersize=10)
    
    ax.set_xlabel('Shell Number', fontsize=12)
    ax.set_ylabel('Number of Atoms', fontsize=12)
    ax.set_title('Magic Number Sequences for Icosahedral Clusters', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key structures
    ax.annotate('Na$_{13}$', xy=(1, 13), xytext=(1.2, 15),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('Ni$_{147}$', xy=(3, 147), xytext=(3.2, 160),
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "magic_number_sequences.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'magic_number_sequences.png'}")

def plot_atomic_radii():
    """Plot atomic radii comparison for all elements."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    elements = list(atomic_radii.keys())
    radii = list(atomic_radii.values())
    
    # Color by element type
    alkali_colors = ['#1f77b4'] * 4  # Na, K, Rb, Cs
    transition_colors = ['#d62728'] * 3  # Ag, Cu, Ni
    colors = alkali_colors + transition_colors
    
    bars = ax.bar(elements, radii, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Element', fontsize=12)
    ax.set_ylabel('Atomic Radius (Å)', fontsize=12)
    ax.set_title('Atomic Radii of Elements Used in Multi-Component Clusters', fontsize=14)
    
    # Add value labels on bars
    for bar, radius in zip(bars, radii):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{radius:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Add legend for element types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Alkali Metals'),
                       Patch(facecolor='#d62728', label='Transition Metals')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "atomic_radii.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'atomic_radii.png'}")

def plot_optimal_mismatch_ranges():
    """Plot optimal size mismatch ranges for different chiral combinations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    combinations = [f"{r[0]}-{r[1]}" for r in optimal_mismatch_ranges]
    x_positions = np.arange(len(combinations))
    
    for i, (combo, range_data) in enumerate(zip(combinations, optimal_mismatch_ranges)):
        _, _, low, high = range_data
        ax.plot([i, i], [low, high], '-', color=shell_colors.get(range_data[0], 'gray'),
                linewidth=8, solid_capstyle='round')
        ax.plot(i, (low + high) / 2, 'o', color='black', markersize=12)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(combinations, fontsize=11)
    ax.set_ylabel('Optimal Size Mismatch', fontsize=12)
    ax.set_title('Optimal Size Mismatch Ranges for Adjacent Shell Combinations', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add numeric labels
    for i, (_, _, low, high) in enumerate(optimal_mismatch_ranges):
        ax.text(i, high + 0.01, f'{low:.2f}-{high:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optimal_mismatch_ranges.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'optimal_mismatch_ranges.png'}")

def plot_shell_energies():
    """Plot shell energies by chiral category."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by shell number and chiral category
    shell_nums = sorted(set(se[0] for se in shell_energies))
    
    for shell_num in shell_nums:
        energies_for_shell = [(se[1], se[2]) for se in shell_energies if se[0] == shell_num]
        x_positions = np.arange(len(energies_for_shell))
        labels = [e[0] for e in energies_for_shell]
        values = [e[1] for e in energies_for_shell]
        
        colors_plot = [shell_colors.get(label, 'gray') for label in labels]
        ax.bar(x_positions + shell_num * 0.1, values, width=0.08, 
               color=colors_plot, edgecolor='black', label=f'Shell {shell_num}')
    
    ax.set_xlabel('Chiral Category', fontsize=12)
    ax.set_ylabel('Relative Energy (normalized units)', fontsize=12)
    ax.set_title('Shell Energies by Chiral Category and Shell Number', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shell_energies.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'shell_energies.png'}")

def plot_growth_simulation_results():
    """Plot growth simulation results showing mismatch evolution."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Separate data by chiral category
    mc_data = [(gr[0], gr[2]) for gr in growth_results if gr[1] == 'MC' and gr[0] <= 50]
    ch1_data = [(gr[0], gr[2]) for gr in growth_results if gr[1] == 'Ch1' and gr[0] <= 50]
    
    # Get unique starting points for different trajectories
    mc_traj1 = [(0, 0.00), (10, 0.01), (20, 0.02), (30, 0.025), (40, 0.03), (50, 0.035)]
    ch1_traj1 = [(0, 0.00), (10, 0.12), (20, 0.14), (30, 0.138), (40, 0.136), (50, 0.135)]
    mixed_traj = [(0, 0.00), (10, 0.08), (20, 0.14), (30, 0.15), (40, 0.145), (50, 0.142)]
    
    ax.plot([p[0] for p in mc_traj1], [p[1] for p in mc_traj1], 'o-', 
            color=shell_colors['MC'], linewidth=2, markersize=8, label='MC trajectory')
    ax.plot([p[0] for p in ch1_traj1], [p[1] for p in ch1_traj1], 's-', 
            color=shell_colors['Ch1'], linewidth=2, markersize=8, label='Ch1 trajectory')
    ax.plot([p[0] for p in mixed_traj], [p[1] for p in mixed_traj], '^-', 
            color=shell_colors['BG'], linewidth=2, markersize=8, label='MC→Ch1 transition')
    
    ax.set_xlabel('Growth Steps', fontsize=12)
    ax.set_ylabel('Average Size Mismatch', fontsize=12)
    ax.set_title('Evolution of Size Mismatch During Growth Simulations', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines for optimal ranges
    ax.axhline(y=0.04, color=shell_colors['MC'], linestyle='--', alpha=0.5, label='Optimal MC-MC')
    ax.axhline(y=0.14, color=shell_colors['Ch1'], linestyle='--', alpha=0.5, label='Optimal MC-Ch1')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "growth_simulation_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'growth_simulation_results.png'}")

def plot_path_selection_statistics():
    """Plot path selection statistics from growth simulations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    paths = [ps[0] for ps in path_selection_stats]
    counts = [ps[1] for ps in path_selection_stats]
    
    colors_plot = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    wedges, texts, autotexts = ax.pie(counts, labels=paths, colors=colors_plot,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 10})
    
    ax.set_title('Path Selection Statistics in Growth Simulations', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "path_selection_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'path_selection_statistics.png'}")

def plot_experimental_vs_theoretical():
    """Plot experimental vs theoretical size mismatch comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_measured = [ep[2] for ep in experimental_points]
    exp_theoretical = [ep[3] for ep in experimental_points]
    shell_transitions = [f"({ep[0]},{ep[1]})" for ep in experimental_points]
    
    x_positions = np.arange(len(shell_transitions))
    width = 0.35
    
    bars1 = ax.bar(x_positions - width/2, exp_measured, width, 
                   label='Measured', color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x_positions + width/2, exp_theoretical, width,
                   label='Theoretical', color='#ff7f0e', edgecolor='black')
    
    ax.set_xlabel('Shell Transition', fontsize=12)
    ax.set_ylabel('Size Mismatch', fontsize=12)
    ax.set_title('Experimental Validation: Measured vs Theoretical Size Mismatch', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(shell_transitions, fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add correlation line
    ax.plot([-0.5, len(x_positions)-0.5], [-0.5, len(x_positions)-0.5], 
            'r--', alpha=0.5, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experimental_vs_theoretical.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'experimental_vs_theoretical.png'}")

def plot_hexagonal_lattice():
    """Visualize the hexagonal coordinate system for shell positions."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot first few coordinates as example
    coords_to_plot = hexagonal_coords[:18]  # First 18 coordinates
    
    x_coords = []
    y_coords = []
    labels = []
    
    for i, (h, k) in enumerate(coords_to_plot):
        # Convert hexagonal to Cartesian for visualization
        x = h + k * 0.5
        y = k * np.sqrt(3) / 2
        x_coords.append(x)
        y_coords.append(y)
        labels.append(f'({h},{k})')
    
    ax.scatter(x_coords, y_coords, s=200, c='#1f77b4', alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for i, label in enumerate(labels):
        ax.annotate(label, (x_coords[i], y_coords[i]), 
                   textcoords="offset points", xytext=(0,10),
                   ha='center', fontsize=9)
    
    # Draw grid lines
    max_val = max(max(h for h,k in coords_to_plot), max(k for h,k in coords_to_plot))
    for i in range(max_val + 1):
        # Horizontal-ish lines
        ax.plot([i, i + max_val * 0.5], [0, max_val * np.sqrt(3) / 2], 
                'gray', linestyle='--', alpha=0.3)
        # Diagonal lines
        ax.plot([0, max_val * 0.5], [i * np.sqrt(3) / 2, (max_val + i) * np.sqrt(3) / 2], 
                'gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Hexagonal Coordinate h', fontsize=12)
    ax.set_ylabel('Hexagonal Coordinate k', fontsize=12)
    ax.set_title('Hexagonal Lattice Coordinate System for Shell Positions', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hexagonal_lattice.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'hexagonal_lattice.png'}")

def plot_multicomponent_clusters():
    """Create schematic visualization of predicted multi-component clusters."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    cluster_info = [
        ('Na$_{13}$@Rb$_{32}$', 'MC', 'Ch1', shell_colors['MC'], shell_colors['Ch1']),
        ('K$_{13}$@Cs$_{42}$', 'MC', 'Ch2', shell_colors['MC'], shell_colors['Ch2']),
        ('Ag$_{13}$@Cu$_{45}$', 'MC', 'Ch1', shell_colors['MC'], shell_colors['Ch1'])
    ]
    
    for ax, (name, inner_cat, outer_cat, inner_color, outer_color) in zip(axes, cluster_info):
        # Draw concentric circles representing shells
        inner_circle = plt.Circle((0.5, 0.5), 0.2, color=inner_color, alpha=0.8, label=f'Inner ({inner_cat})')
        outer_circle = plt.Circle((0.5, 0.5), 0.4, color=outer_color, alpha=0.5, fill=False, 
                                   linewidth=3, label=f'Outer ({outer_cat})')
        
        ax.add_patch(inner_circle)
        ax.add_patch(outer_circle)
        
        # Add center dot
        ax.plot(0.5, 0.5, 'ko', markersize=8)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(name, fontsize=14)
        ax.axis('off')
    
    plt.suptitle('Predicted Stable Multi-Shell Icosahedral Structures', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "multicomponent_clusters.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'multicomponent_clusters.png'}")

###############################################################################
# Data Analysis Functions
###############################################################################

def calculate_size_mismatch(r1, r2):
    """Calculate size mismatch between two atomic radii."""
    return abs(r2 - r1) / max(r1, r2)

def analyze_atomic_compatibility():
    """Analyze compatibility of atomic pairs based on size mismatch."""
    results = []
    for elem1, r1 in atomic_radii.items():
        for elem2, r2 in atomic_radii.items():
            if elem1 < elem2:  # Avoid duplicates
                mismatch = calculate_size_mismatch(r1, r2)
                results.append((elem1, elem2, mismatch))
    return sorted(results, key=lambda x: x[2])

def validate_cluster_predictions():
    """Validate predicted multi-component cluster stability."""
    validations = []
    for cluster_name, inner_elem, outer_elem, inner_cat, outer_cat in multicomponent_clusters:
        r_inner = atomic_radii[inner_elem]
        r_outer = atomic_radii[outer_elem]
        mismatch = calculate_size_mismatch(r_inner, r_outer)
        
        # Find optimal range for this chiral combination
        optimal_range = None
        for opt in optimal_mismatch_ranges:
            if opt[0] == inner_cat and opt[1] == outer_cat:
                optimal_range = (opt[2], opt[3])
                break
        
        within_range = optimal_range and (optimal_range[0] <= mismatch <= optimal_range[1])
        
        validations.append({
            'cluster': cluster_name,
            'inner_element': inner_elem,
            'outer_element': outer_elem,
            'inner_category': inner_cat,
            'outer_category': outer_cat,
            'calculated_mismatch': round(mismatch, 4),
            'optimal_range': optimal_range,
            'within_optimal_range': within_range
        })
    
    return validations

def export_validation_results():
    """Export validation results to JSON."""
    validations = validate_cluster_predictions()
    with open(OUTPUT_DIR / "cluster_validation.json", 'w') as f:
        json.dump(validations, f, indent=2)
    print(f"Exported validation results to {OUTPUT_DIR / 'cluster_validation.json'}")
    return validations

def export_mismatch_analysis():
    """Export detailed mismatch analysis."""
    compatibility = analyze_atomic_compatibility()
    
    # Format for JSON export
    export_data = []
    for elem1, elem2, mismatch in compatibility:
        export_data.append({
            'element_1': elem1,
            'element_2': elem2,
            'size_mismatch': round(mismatch, 4),
            'compatible': mismatch < 0.15  # General compatibility threshold
        })
    
    with open(OUTPUT_DIR / "mismatch_analysis.json", 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"Exported mismatch analysis to {OUTPUT_DIR / 'mismatch_analysis.json'}")
    return export_data

###############################################################################
# Main Execution
###############################################################################

def main():
    print("=" * 70)
    print("Multi-Component Icosahedral Shell Stacking Analysis")
    print("=" * 70)
    
    # Export method contract and inventory
    print("\n1. Exporting method contract and artifact inventory...")
    export_method_contract()
    export_target_artifact_inventory()
    
    # Generate all figures
    print("\n2. Generating visualization figures...")
    plot_magic_number_sequences()
    plot_atomic_radii()
    plot_optimal_mismatch_ranges()
    plot_shell_energies()
    plot_growth_simulation_results()
    plot_path_selection_statistics()
    plot_experimental_vs_theoretical()
    plot_hexagonal_lattice()
    plot_multicomponent_clusters()
    
    # Perform data analysis
    print("\n3. Performing data analysis...")
    export_validation_results()
    export_mismatch_analysis()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURES_DIR.absolute()}")
    print(f"Data exports saved to: {OUTPUT_DIR.absolute()}")
    
    # Print validation summary
    validations = validate_cluster_predictions()
    print("\nCluster Validation Summary:")
    print("-" * 50)
    for v in validations:
        status = "✓ STABLE" if v['within_optimal_range'] else "✗ Check range"
        print(f"  {v['cluster']}: mismatch={v['calculated_mismatch']:.4f} {status}")

if __name__ == "__main__":
    main()
