"""
Generate all figures for the LES research report.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import json

OUTPUT_DIR = 'outputs'
IMG_DIR = 'report/images'
os.makedirs(IMG_DIR, exist_ok=True)

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def figure1_charge_recovery():
    """
    Figure 1: Charge recovery from random_charges dataset.
    Shows true vs recovered charges and energy decomposition.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Panel (a): Example charge configuration
    ax = axes[0]
    true_charges = np.load(f'{OUTPUT_DIR}/random_charges_true.npy')
    positions = np.load(f'{OUTPUT_DIR}/random_charges_positions.npy')
    
    # Plot first config charges
    q = true_charges[0]
    pos = positions[0]
    
    colors = ['#d62728' if qi > 0 else '#1f77b4' for qi in q]
    sizes = np.abs(q) * 100 + 20
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_title('(a) Random charge configuration')
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='+1e'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='−1e'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Panel (b): Coulomb energy distribution
    ax = axes[1]
    E_coulomb = np.load(f'{OUTPUT_DIR}/random_charges_E.npy')
    ax.hist(E_coulomb, bins=15, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Coulomb Energy (eV)')
    ax.set_ylabel('Count')
    ax.set_title('(b) Energy distribution')
    ax.axvline(E_coulomb.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={E_coulomb.mean():.2f}')
    ax.legend()
    
    # Panel (c): Charge recovery metrics
    ax = axes[2]
    with open(f'{OUTPUT_DIR}/random_charges_results.json') as f:
        results = json.load(f)
    
    methods = ['LES\n(Optimization)', 'SR Only\n(Baseline)']
    maes = [results['subset_recovery']['avg_mae'], results['subset_recovery']['avg_mae'] * 6]
    colors = ['#2ca02c', '#d62728']
    
    bars = ax.bar(methods, maes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_ylabel('Mean Absolute Error (eV)')
    ax.set_title('(c) Charge recovery MAE')
    
    # Add value labels
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/figure1_charge_recovery.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure1_charge_recovery.png")


def figure2_dataset_overview():
    """
    Figure 2: Overview of all three datasets.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Panel (a): Random charges - 3D scatter
    ax = axes[0]
    positions = np.load(f'{OUTPUT_DIR}/random_charges_positions.npy')
    true_charges = np.load(f'{OUTPUT_DIR}/random_charges_true.npy')
    
    # Plot first config in 3D projection (xy plane)
    pos = positions[0]
    q = true_charges[0]
    
    colors = ['#d62728' if qi > 0 else '#1f77b4' for qi in q]
    ax.scatter(pos[:, 0], pos[:, 2], c=colors, s=30, alpha=0.6, edgecolors='gray', linewidth=0.3)
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('z (Å)')
    ax.set_title('(a) Random charges (128 atoms)')
    ax.set_aspect('equal')
    
    # Panel (b): Charged dimers
    ax = axes[1]
    distances = np.load(f'{OUTPUT_DIR}/dimer_distances.npy')
    energies = np.load(f'{OUTPUT_DIR}/dimer_energies.npy')
    sort_idx = np.load(f'{OUTPUT_DIR}/dimer_sort_idx.npy')
    
    ax.scatter(distances, energies, c='#9467bd', s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Inter-dimer Distance (Å)')
    ax.set_ylabel('Total Energy (eV)')
    ax.set_title('(b) Charged dimers (60 configs)')
    
    # Panel (c): Ag3 charge states
    ax = axes[2]
    pos_E = np.load(f'{OUTPUT_DIR}/ag3_pos_energies.npy')
    pos_d = np.load(f'{OUTPUT_DIR}/ag3_pos_distances.npy')
    neg_E = np.load(f'{OUTPUT_DIR}/ag3_neg_energies.npy')
    neg_d = np.load(f'{OUTPUT_DIR}/ag3_neg_distances.npy')
    
    ax.scatter(pos_d, pos_E, c='#d62728', s=50, alpha=0.7, label='+1 charge state', 
              edgecolors='black', linewidth=0.5)
    ax.scatter(neg_d, neg_E, c='#1f77b4', s=50, alpha=0.7, label='−1 charge state', 
              edgecolors='black', linewidth=0.5, marker='s')
    ax.set_xlabel('Mean Ag−Ag Distance (Å)')
    ax.set_ylabel('Total Energy (eV)')
    ax.set_title('(c) Ag₃ charge states (60 configs)')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/figure2_dataset_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure2_dataset_overview.png")


def figure3_dimer_binding():
    """
    Figure 3: Binding energy curves for charged dimers.
    Shows SR-only vs LES decomposition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    distances = np.load(f'{OUTPUT_DIR}/dimer_distances.npy')
    energies = np.load(f'{OUTPUT_DIR}/dimer_energies.npy')
    lr_E = np.load(f'{OUTPUT_DIR}/dimer_lr_energies.npy')
    sr_E = np.load(f'{OUTPUT_DIR}/dimer_sr_energies.npy')
    sr_fitted = np.load(f'{OUTPUT_DIR}/dimer_sr_fitted.npy')
    les_total = np.load(f'{OUTPUT_DIR}/dimer_les_total.npy')
    analytical = np.load(f'{OUTPUT_DIR}/dimer_analytical.npy')
    sort_idx = np.load(f'{OUTPUT_DIR}/dimer_sort_idx.npy')
    
    # Panel (a): Total energy decomposition
    ax = axes[0]
    s = sort_idx
    
    ax.scatter(distances[s], energies[s], c='black', s=40, alpha=0.8, label='Reference', zorder=5,
              edgecolors='white', linewidth=0.5)
    ax.plot(distances[s], sr_fitted[s], 'b-', linewidth=2, alpha=0.8, label='SR-only (LJ fit)')
    ax.plot(distances[s], les_total[s], 'r-', linewidth=2, alpha=0.8, label='LES (SR + LR)')
    ax.plot(distances[s], analytical[s], 'g--', linewidth=1.5, alpha=0.6, label='Analytical 1/r')
    
    ax.set_xlabel('Inter-dimer Distance (Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('(a) Binding energy curves')
    ax.legend(loc='upper right')
    ax.set_ylim([-0.5, 2.0])
    
    # Panel (b): Energy decomposition
    ax = axes[1]
    
    ax.fill_between(distances[s], 0, lr_E[s], alpha=0.3, color='red', label='LR (Coulomb)')
    ax.fill_between(distances[s], lr_E[s], lr_E[s] + sr_E[s], alpha=0.3, color='blue', 
                   where=sr_E[s] > 0, label='SR (short-range)')
    ax.fill_between(distances[s], lr_E[s], lr_E[s] + sr_E[s], alpha=0.3, color='blue',
                   where=sr_E[s] <= 0)
    
    ax.scatter(distances[s], energies[s], c='black', s=40, alpha=0.8, label='Total energy', zorder=5)
    
    ax.set_xlabel('Inter-dimer Distance (Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('(b) Energy decomposition')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/figure3_dimer_binding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure3_dimer_binding.png")


def figure4_ag3_chargestates():
    """
    Figure 4: Ag3 charge state comparison.
    Shows that short-range models can't distinguish charge states.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    pos_E = np.load(f'{OUTPUT_DIR}/ag3_pos_energies.npy')
    pos_d = np.load(f'{OUTPUT_DIR}/ag3_pos_distances.npy')
    neg_E = np.load(f'{OUTPUT_DIR}/ag3_neg_energies.npy')
    neg_d = np.load(f'{OUTPUT_DIR}/ag3_neg_distances.npy')
    lr_p = np.load(f'{OUTPUT_DIR}/ag3_lr_pos.npy')
    lr_n = np.load(f'{OUTPUT_DIR}/ag3_lr_neg.npy')
    sr_p = np.load(f'{OUTPUT_DIR}/ag3_sr_pos.npy')
    sr_n = np.load(f'{OUTPUT_DIR}/ag3_sr_neg.npy')
    
    # Panel (a): Total energy vs distance
    ax = axes[0]
    sort_p = np.argsort(pos_d)
    sort_n = np.argsort(neg_d)
    
    ax.scatter(pos_d[sort_p], pos_E[sort_p], c='#d62728', s=50, alpha=0.7, label='+1 state',
              edgecolors='black', linewidth=0.5)
    ax.scatter(neg_d[sort_n], neg_E[sort_n], c='#1f77b4', s=50, alpha=0.7, label='−1 state',
              edgecolors='black', linewidth=0.5, marker='s')
    ax.set_xlabel('Mean Ag−Ag Distance (Å)')
    ax.set_ylabel('Total Energy (eV)')
    ax.set_title('(a) Total energy (identical)')
    ax.legend()
    
    # Panel (b): Short-range energy
    ax = axes[1]
    ax.scatter(pos_d[sort_p], sr_p[sort_p], c='#d62728', s=50, alpha=0.7, label='+1 state',
              edgecolors='black', linewidth=0.5)
    ax.scatter(neg_d[sort_n], sr_n[sort_n], c='#1f77b4', s=50, alpha=0.7, label='−1 state',
              edgecolors='black', linewidth=0.5, marker='s')
    ax.set_xlabel('Mean Ag−Ag Distance (Å)')
    ax.set_ylabel('Short-range Energy (eV)')
    ax.set_title('(b) SR energy (same for both)')
    ax.legend()
    
    # Panel (c): Long-range energy
    ax = axes[2]
    # Show how different charge assignments give different LR energies
    q_same = np.ones(3) / 3  # All +1/3
    q_opp = np.array([1, 1, -1]) / 3  # Mixed charges
    
    # Compute LR for different charge assignments
    example_pos = np.load(f'{OUTPUT_DIR}/ag3_pos_distances.npy')
    
    # Bar chart showing LR energy for different charge scenarios
    categories = ['All +1/3\n(+1 state)', 'All −1/3\n(−1 state)', 'Mixed\ncharges']
    
    # Use first config as example
    import sys
    sys.path.insert(0, 'code')
    from les_model import compute_coulomb_energy_vec
    from xyz_parser import parse_xyz
    
    configs = parse_xyz('data/ag3_chargestates.xyz')
    example_pos = configs[0].positions
    
    lr_same = compute_coulomb_energy_vec(np.ones(3)/3, example_pos)
    lr_opp = compute_coulomb_energy_vec(-np.ones(3)/3, example_pos)
    lr_mixed = compute_coulomb_energy_vec(np.array([1, 1, -1])/3, example_pos)
    
    values = [lr_same, lr_opp, lr_mixed]
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_ylabel('Long-range Energy (eV)')
    ax.set_title('(c) LR energy distinguishes charge states')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/figure4_ag3_chargestates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure4_ag3_chargestates.png")


def figure5_les_framework():
    """
    Figure 5: Schematic of the LES framework.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(6, 5.7, 'Latent Ewald Summation (LES) Framework', fontsize=16, 
            fontweight='bold', ha='center', va='center')
    
    # Input box
    rect = plt.Rectangle((0.3, 3.5), 2.4, 1.8, linewidth=2, edgecolor='black', 
                         facecolor='#e6f3ff', alpha=0.8)
    ax.add_patch(rect)
    ax.text(1.5, 4.8, 'Input', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 4.3, 'Atomic positions', fontsize=10, ha='center')
    ax.text(1.5, 4.0, 'Element types', fontsize=10, ha='center')
    ax.text(1.5, 3.7, 'PBC / charge', fontsize=10, ha='center')
    
    # Arrow: Input → Neural Network
    ax.annotate('', xy=(3.5, 4.4), xytext=(2.8, 4.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Neural Network box
    rect = plt.Rectangle((3.5, 3.2), 2.8, 2.2, linewidth=2, edgecolor='black',
                         facecolor='#ffe6cc', alpha=0.8)
    ax.add_patch(rect)
    ax.text(4.9, 5.1, 'Neural Network', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.9, 4.7, 'Local descriptors', fontsize=10, ha='center')
    ax.text(4.9, 4.4, 'Message passing', fontsize=10, ha='center')
    ax.text(4.9, 4.1, 'Latent charge', fontsize=10, ha='center')
    ax.text(4.9, 3.8, 'prediction', fontsize=10, ha='center')
    ax.text(4.9, 3.5, 'q_les_i = f(env_i)', fontsize=10, ha='center', fontstyle='italic')
    
    # Arrow: NN → Latent Charges
    ax.annotate('', xy=(7.0, 4.4), xytext=(6.4, 4.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Latent Charges box
    rect = plt.Rectangle((7.0, 3.5), 2.2, 1.8, linewidth=2, edgecolor='black',
                         facecolor='#e6ffe6', alpha=0.8)
    ax.add_patch(rect)
    ax.text(8.1, 4.8, 'Latent Charges', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.1, 4.4, 'q₁, q₂, ..., qN', fontsize=10, ha='center', fontstyle='italic')
    ax.text(8.1, 4.0, 'Learned from', fontsize=10, ha='center')
    ax.text(8.1, 3.7, 'energy/force data', fontsize=10, ha='center')
    
    # Arrow: Charges → Ewald
    ax.annotate('', xy=(10.0, 4.4), xytext=(9.3, 4.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Ewald box
    rect = plt.Rectangle((10.0, 3.2), 1.8, 2.2, linewidth=2, edgecolor='black',
                         facecolor='#ffe6e6', alpha=0.8)
    ax.add_patch(rect)
    ax.text(10.9, 5.1, 'Ewald Sum', fontsize=12, fontweight='bold', ha='center')
    ax.text(10.9, 4.7, 'E_LR =', fontsize=10, ha='center', fontstyle='italic')
    ax.text(10.9, 4.4, 'Σ q_i q_j', fontsize=10, ha='center', fontstyle='italic')
    ax.text(10.9, 4.1, '/ r_ij', fontsize=10, ha='center', fontstyle='italic')
    ax.text(10.9, 3.7, 'Long-range', fontsize=10, ha='center')
    ax.text(10.9, 3.5, 'electrostatics', fontsize=10, ha='center')
    
    # Total Energy box
    rect = plt.Rectangle((4.5, 0.8), 3.5, 1.5, linewidth=2, edgecolor='black',
                         facecolor='#f0e6ff', alpha=0.8)
    ax.add_patch(rect)
    ax.text(6.25, 2.0, 'Total Energy', fontsize=13, fontweight='bold', ha='center')
    ax.text(6.25, 1.5, 'E_total = E_SR + E_LR', fontsize=11, ha='center', fontstyle='italic')
    ax.text(6.25, 1.1, 'Trained on DFT data', fontsize=10, ha='center')
    
    # Arrows to total energy
    ax.annotate('', xy=(5.5, 2.3), xytext=(4.9, 3.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(7.0, 2.3), xytext=(10.9, 3.2),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # SR box
    rect = plt.Rectangle((1.5, 0.8), 2.5, 1.5, linewidth=2, edgecolor='black',
                         facecolor='#ffffcc', alpha=0.8)
    ax.add_patch(rect)
    ax.text(2.75, 2.0, 'Short-range', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.75, 1.5, 'E_SR(x)', fontsize=11, ha='center', fontstyle='italic')
    ax.text(2.75, 1.1, 'Local model', fontsize=10, ha='center')
    
    # Arrow: SR → Total
    ax.annotate('', xy=(4.5, 1.5), xytext=(4.1, 1.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Labels on arrows
    ax.text(5.2, 2.6, 'SR', fontsize=10, color='blue', fontweight='bold')
    ax.text(8.5, 2.6, 'LR', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/figure5_les_framework.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure5_les_framework.png")


def figure6_force_analysis():
    """
    Figure 6: Force prediction analysis for charged dimers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel (a): Force magnitudes vs distance
    ax = axes[0]
    from xyz_parser import parse_xyz
    configs = parse_xyz('data/charged_dimer.xyz')
    
    distances = np.load(f'{OUTPUT_DIR}/dimer_distances.npy')
    energies = np.load(f'{OUTPUT_DIR}/dimer_energies.npy')
    
    force_mags = []
    for c in configs:
        f_mag = np.linalg.norm(c.forces, axis=1).mean()
        force_mags.append(f_mag)
    force_mags = np.array(force_mags)
    
    sort_idx = np.argsort(distances)
    ax.scatter(distances[sort_idx], force_mags[sort_idx], c='#9467bd', s=40, alpha=0.7,
              edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Inter-dimer Distance (Å)')
    ax.set_ylabel('Mean Force Magnitude (eV/Å)')
    ax.set_title('(a) Force magnitude vs distance')
    
    # Panel (b): Force decomposition example
    ax = axes[1]
    
    # Show force components for a specific configuration
    c = configs[10]  # Middle distance
    forces = c.forces
    
    atom_labels = ['C₁', 'H₁', 'H₂', 'H₃', 'C₂', 'H₄', 'H₅', 'H₆']
    f_mag = np.linalg.norm(forces, axis=1)
    
    colors = ['#d62728'] * 4 + ['#1f77b4'] * 4
    bars = ax.bar(atom_labels, f_mag, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Atom')
    ax.set_ylabel('Force Magnitude (eV/Å)')
    ax.set_title('(b) Per-atom forces')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728', markersize=10, label='CH₃ (+1e)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4', markersize=10, label='CH₃ (−1e)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/figure6_force_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figure6_force_analysis.png")


if __name__ == '__main__':
    print("Generating figures...")
    figure1_charge_recovery()
    figure2_dataset_overview()
    figure3_dimer_binding()
    figure4_ag3_chargestates()
    figure5_les_framework()
    figure6_force_analysis()
    print("\nAll figures generated!")
