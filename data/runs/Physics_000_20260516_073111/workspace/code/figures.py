#!/usr/bin/env python3
"""
Figure Generation: Publication-quality figures for multi-component icosahedral shell research.

Figures generated:
1. Magic number sequences (Mackay vs new)
2. Size mismatch heatmap for element pairs
3. Shell energy landscape across chiral categories
4. Growth trajectory evolution (mismatch vs steps)
5. Path selection statistics
6. Experimental vs theoretical mismatch comparison
7. Lennard-Jones potential curves
8. Predicted stable multi-component clusters
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker
import json
import os
import sys

# Add code directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report', 'images')
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

# Color schemes
COLORS = {
    'MC': '#1f77b4',
    'BG': '#ff7f0e', 
    'Ch1': '#2ca02c',
    'Ch2': '#d62728',
    'Ch3': '#9467bd',
    'Ch4': '#8c564b',
    'Ch5': '#e377c2',
    'Na': '#e6194b',
    'K': '#3cb44b',
    'Rb': '#ffe119',
    'Cs': '#4363d8',
    'Ag': '#f58231',
    'Cu': '#911eb4',
    'Ni': '#42d4f4'
}

os.makedirs(IMAGES_DIR, exist_ok=True)


def load_json(filename):
    """Load JSON from outputs directory."""
    path = os.path.join(OUTPUTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 1: Magic Number Sequences
# ============================================================
def figure_magic_numbers():
    """Compare Mackay and new magic number sequences."""
    core = load_json('core_theory_output.json')
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_mackay = list(range(1, len(core['mackay_sequence']) + 1))
    k_new = list(range(1, len(core['new_sequence_b5']) + 1))
    
    ax.plot(k_mackay, core['mackay_sequence'], 'o-', color=COLORS['MC'], 
            linewidth=2, markersize=8, label='Mackay Sequence (Classical)')
    ax.plot(k_new, core['new_sequence_b5'], 's--', color=COLORS['Ch1'], 
            linewidth=2, markersize=8, label='New Sequence (b=5, Chiral-extended)')
    
    # Annotate values
    for k, v in zip(k_mackay, core['mackay_sequence']):
        ax.annotate(str(v), (k, v), textcoords="offset points", xytext=(0, 10), 
                   ha='center', fontsize=8, color=COLORS['MC'])
    for k, v in zip(k_new, core['new_sequence_b5']):
        ax.annotate(str(v), (k, v), textcoords="offset points", xytext=(0, -15), 
                   ha='center', fontsize=8, color=COLORS['Ch1'])
    
    ax.set_xlabel('Shell Index $k$')
    ax.set_ylabel('Cumulative Atom Count $N_k$')
    ax.set_title('Icosahedral Magic Number Sequences')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max(len(k_mackay), len(k_new)) + 0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure1_magic_numbers.png'))
    plt.close()
    print("  Figure 1: Magic numbers saved.")


# ============================================================
# Figure 2: Size Mismatch Heatmap
# ============================================================
def figure_mismatch_heatmap():
    """Generate heatmap of size mismatches between all element pairs."""
    mismatch_data = load_json('size_mismatch_output.json')
    elements = mismatch_data['elements']
    matrix = np.array(mismatch_data['mismatch_matrix'])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Mask diagonal
    mask = np.zeros_like(matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    matrix_masked = np.ma.masked_where(mask, matrix)
    
    im = ax.imshow(matrix_masked, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.5)
    
    # Annotate cells
    for i in range(len(elements)):
        for j in range(len(elements)):
            if i != j:
                text_color = 'white' if matrix[i, j] > 0.25 else 'black'
                ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', 
                       fontsize=9, color=text_color, fontweight='bold')
    
    ax.set_xticks(range(len(elements)))
    ax.set_yticks(range(len(elements)))
    ax.set_xticklabels(elements)
    ax.set_yticklabels(elements)
    ax.set_xlabel('Element $j$')
    ax.set_ylabel('Element $i$')
    ax.set_title('Size Mismatch Matrix $|r_j - r_i| / r_i$')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Size Mismatch')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure2_mismatch_heatmap.png'))
    plt.close()
    print("  Figure 2: Mismatch heatmap saved.")


# ============================================================
# Figure 3: Shell Energy Landscape
# ============================================================
def figure_shell_energies():
    """Plot relative shell energies across chiral categories."""
    core = load_json('core_theory_output.json')
    energies = core['shell_energies']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Organize data
    shell_idx = [e['shell'] for e in energies]
    energy_vals = [e['energy'] for e in energies]
    chiral_vals = [e['chiral'] for e in energies]
    
    # Plot by chiral category
    for cat in ['MC', 'Ch1', 'BG']:
        cat_data = [(s, e) for s, c, e in [(e['shell'], e['chiral'], e['energy']) for e in energies] if c == cat]
        if cat_data:
            shells_cat = [s for s, _ in cat_data]
            energies_cat = [e for _, e in cat_data]
            ax.plot(shells_cat, energies_cat, 'o-', color=COLORS.get(cat, 'gray'),
                   linewidth=2, markersize=10, label=f'{cat}')
            for s, e in zip(shells_cat, energies_cat):
                ax.annotate(f'{e:.2f}', (s, e), textcoords="offset points",
                          xytext=(0, 8), ha='center', fontsize=9, color=COLORS.get(cat))
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Shell Index')
    ax.set_ylabel('Relative Energy (normalized units)')
    ax.set_title('Shell Energy Landscape by Chiral Category')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure3_shell_energies.png'))
    plt.close()
    print("  Figure 3: Shell energies saved.")


# ============================================================
# Figure 4: Growth Trajectory Evolution
# ============================================================
def figure_growth_trajectories():
    """Plot growth trajectories showing mismatch evolution over steps."""
    growth_data = load_json('growth_simulation_output.json')
    trajectories = growth_data['growth_trajectories']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    for idx, (traj_name, traj_data) in enumerate(trajectories.items()):
        ax = axes[idx]
        
        for chiral_type in ['MC', 'Ch1']:
            if chiral_type in traj_data and traj_data[chiral_type]['steps']:
                steps = traj_data[chiral_type]['steps']
                mismatches = traj_data[chiral_type]['mismatches']
                color = COLORS.get(chiral_type, 'gray')
                ax.plot(steps, mismatches, 'o-', color=color, linewidth=2, 
                       markersize=6, label=chiral_type)
        
        ax.set_xlabel('Growth Steps')
        ax.set_ylabel('Average Mismatch')
        ax.set_title(f'Trajectory {idx+1}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Growth Trajectories: Mismatch Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure4_growth_trajectories.png'))
    plt.close()
    print("  Figure 4: Growth trajectories saved.")


# ============================================================
# Figure 5: Path Selection Statistics
# ============================================================
def figure_path_statistics():
    """Bar chart of path selection statistics."""
    growth_data = load_json('growth_simulation_output.json')
    path_analysis = growth_data['path_probability_analysis']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    paths = [p['path'] for p in path_analysis]
    counts = [p['count'] for p in path_analysis]
    percentages = [p['percentage'] for p in path_analysis]
    
    colors_bar = [COLORS['MC'], COLORS['Ch1'], COLORS['BG'], COLORS['Ch2']]
    bars = ax1.bar(paths, counts, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Count')
    ax1.set_title('Path Selection Counts')
    ax1.tick_params(axis='x', rotation=15)
    for bar, count, pct in zip(bars, counts, percentages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # Pie chart
    colors_pie = [COLORS['MC'], COLORS['Ch1'], COLORS['BG'], COLORS['Ch2']]
    wedges, texts, autotexts = ax2.pie(counts, labels=paths, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    ax2.set_title('Path Selection Distribution')
    for autotext in autotexts:
        autotext.set_fontsize(9)
    
    fig.suptitle('Growth Path Selection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure5_path_statistics.png'))
    plt.close()
    print("  Figure 5: Path statistics saved.")


# ============================================================
# Figure 6: Experimental vs Theoretical Mismatch
# ============================================================
def figure_experimental_validation():
    """Compare experimental and theoretical size mismatch values."""
    mismatch_data = load_json('size_mismatch_output.json')
    exp_valid = mismatch_data['experimental_validation']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    measured = [e['measured_sm'] for e in exp_valid]
    theoretical = [e['theoretical_sm'] for e in exp_valid]
    t_labels = [f"T({e['T_i']},{e['T_j']})" for e in exp_valid]
    
    # Parity plot
    ax.scatter(theoretical, measured, c=COLORS['Ch1'], s=120, zorder=5, edgecolors='black', linewidth=0.5)
    
    # Perfect agreement line
    max_val = max(max(measured), max(theoretical)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
    
    # Annotate points
    for i, (m, t, label) in enumerate(zip(measured, theoretical, t_labels)):
        ax.annotate(label, (t, m), textcoords="offset points", xytext=(10, 5),
                   ha='left', fontsize=9, 
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
    
    ax.set_xlabel('Theoretical Mismatch')
    ax.set_ylabel('Measured Mismatch')
    ax.set_title('Experimental Validation: Measured vs Theoretical Size Mismatch')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    
    # Add MAE annotation
    mae = mismatch_data['experimental_summary']['MAE']
    rmse = mismatch_data['experimental_summary']['RMSE']
    ax.text(0.05, 0.95, f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure6_experimental_validation.png'))
    plt.close()
    print("  Figure 6: Experimental validation saved.")


# ============================================================
# Figure 7: Lennard-Jones Potential Curves
# ============================================================
def figure_lj_potentials():
    """Plot Lennard-Jones potential curves for element pairs."""
    growth_data = load_json('growth_simulation_output.json')
    lj_analysis = growth_data['lj_analysis']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    r = np.linspace(2.0, 8.0, 500)
    colors_lj = [COLORS['Na'], COLORS['Rb'], COLORS['Cs'], 
                 COLORS['Ag'], COLORS['Cu'], COLORS['Ch1'], COLORS['Ch2']]
    
    for idx, (pair, data) in enumerate(lj_analysis.items()):
        eps = data['epsilon']
        sigma = data['sigma']
        V = 4 * eps * ((sigma / r)**12 - (sigma / r)**6)
        color = colors_lj[idx % len(colors_lj)]
        ax.plot(r, V, '-', color=color, linewidth=1.5, alpha=0.8, label=f'{pair}')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Interatomic Distance $r$ (Å)')
    ax.set_ylabel('Potential Energy $V(r)$ (ε)')
    ax.set_title('Lennard-Jones Interaction Potentials')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_ylim(-1.5, 2.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure7_lj_potentials.png'))
    plt.close()
    print("  Figure 7: LJ potentials saved.")


# ============================================================
# Figure 8: Predicted Stable Multi-Component Clusters
# ============================================================
def figure_predicted_clusters():
    """Visualize predicted stable multi-component clusters."""
    mismatch_data = load_json('size_mismatch_output.json')
    predicted = mismatch_data.get('predicted_optimal_clusters', [])
    validated = mismatch_data.get('validated_clusters', [])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if predicted:
        # Sort by size mismatch
        sorted_clusters = sorted(predicted, key=lambda x: x['size_mismatch'])
        
        names = [c['name'] for c in sorted_clusters]
        mismatches = [c['size_mismatch'] for c in sorted_clusters]
        transitions = [c['chiral_transition'] for c in sorted_clusters]
        
        # Color by transition type
        colors_list = []
        for t in transitions:
            if 'Ch1' in t:
                colors_list.append(COLORS['Ch1'])
            elif 'Ch2' in t:
                colors_list.append(COLORS['Ch2'])
            elif 'BG' in t:
                colors_list.append(COLORS['BG'])
            else:
                colors_list.append(COLORS['MC'])
        
        y_pos = range(len(names))
        bars = ax.barh(y_pos, mismatches, color=colors_list, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Size Mismatch')
        ax.set_title('Predicted Stable Multi-Component Clusters')
        
        # Add optimal range bands
        for inner_type, outer_type, smin, smax in [
            ('MC', 'MC', 0.03, 0.05),
            ('MC', 'Ch1', 0.12, 0.16),
            ('MC', 'Ch2', 0.19, 0.22),
            ('MC', 'BG', 0.08, 0.10)
        ]:
            label = f'{inner_type}→{outer_type}'
            color = COLORS.get(outer_type, 'gray')
            ax.axvspan(smin, smax, alpha=0.1, color=color)
            ax.annotate(label, (smin, ax.get_ylim()[1]-0.3), fontsize=7, 
                       color=color, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure8_predicted_clusters.png'))
    plt.close()
    print("  Figure 8: Predicted clusters saved.")


# ============================================================
# Figure 9: Atomic Radii Comparison
# ============================================================
def figure_atomic_radii():
    """Bar chart comparing atomic radii of available elements."""
    mismatch_data = load_json('size_mismatch_output.json')
    elements = mismatch_data['elements']
    
    # Atomic radii data
    radii_dict = {'Na': 1.86, 'K': 2.27, 'Rb': 2.48, 'Cs': 2.65,
                  'Ag': 1.44, 'Cu': 1.28, 'Ni': 1.24}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sorted_elements = sorted(elements, key=lambda e: radii_dict.get(e, 0))
    radii = [radii_dict[e] for e in sorted_elements]
    colors_radii = [COLORS.get(e, 'gray') for e in sorted_elements]
    
    bars = ax.bar(sorted_elements, radii, color=colors_radii, edgecolor='black', linewidth=0.5)
    
    # Group labels
    ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.5)
    ax.text(0.8, max(radii) * 0.95, 'Transition\nMetals', ha='center', fontsize=9, style='italic')
    ax.text(4.8, max(radii) * 0.95, 'Alkali\nMetals', ha='center', fontsize=9, style='italic')
    
    for bar, r in zip(bars, radii):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{r:.2f} Å', ha='center', fontsize=9)
    
    ax.set_ylabel('Atomic Radius (Å)')
    ax.set_title('Atomic Radii of Available Elements')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure9_atomic_radii.png'))
    plt.close()
    print("  Figure 9: Atomic radii saved.")


# ============================================================
# Figure 10: Energy Stability Analysis
# ============================================================
def figure_energy_stability():
    """Plot energy differences between chiral and MC configurations."""
    mismatch_data = load_json('size_mismatch_output.json')
    stability = mismatch_data.get('energy_stability', {})
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Extract data
    labels = []
    deltas = []
    colors_stab = []
    
    for key, data in stability.items():
        labels.append(key.replace('_vs_MC', '').replace('shell', 'Shell '))
        deltas.append(data['delta_E'])
        chiral_type = key.split('_')[1] if '_' in key else 'MC'
        colors_stab.append(COLORS.get(chiral_type, 'gray'))
    
    bars = ax.barh(labels, deltas, color=colors_stab, edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('ΔE (normalized units)')
    ax.set_title('Energy Stability: Chiral vs Mackay Configurations')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add annotations
    for bar, d in zip(bars, deltas):
        label = f'{d:.2f}'
        x_pos = d + 0.02 if d >= 0 else d - 0.12
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label, 
               va='center', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, 'figure10_energy_stability.png'))
    plt.close()
    print("  Figure 10: Energy stability saved.")


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    print("Generating all figures...")
    
    figure_magic_numbers()
    figure_mismatch_heatmap()
    figure_shell_energies()
    figure_growth_trajectories()
    figure_path_statistics()
    figure_experimental_validation()
    figure_lj_potentials()
    figure_predicted_clusters()
    figure_atomic_radii()
    figure_energy_stability()
    
    print(f"\nAll figures saved to {IMAGES_DIR}")
    
    # Verify
    png_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
    print(f"Total PNG files: {len(png_files)}")
    for f in sorted(png_files):
        size_kb = os.path.getsize(os.path.join(IMAGES_DIR, f)) / 1024
        print(f"  {f}: {size_kb:.1f} KB")


if __name__ == '__main__':
    main()
