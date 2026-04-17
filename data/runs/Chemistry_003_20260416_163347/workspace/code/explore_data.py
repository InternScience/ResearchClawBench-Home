#!/usr/bin/env python3
"""
Comprehensive data exploration and visualization for Chemistry_003 task.
Generates overview plots and statistics for all datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from analyze_data import parse_xyz_file, compute_pairwise_distances, compute_coulomb_energy


def plot_random_charges_analysis(configs, output_dir):
    """Analyze random_charges dataset - charge recovery benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_configs = len(configs)
    n_atoms = configs[0]['n_atoms']
    
    # Extract charges and positions
    true_charges_all = np.array([c['true_charges'] for c in configs])
    positions_all = np.array([c['positions'] for c in configs])
    energies = np.array([c.get('energy', 0) for c in configs])
    
    # Charge distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Charge histogram
    axes[0, 0].hist(true_charges_all.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Charge (e)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Charge Distribution\n({n_configs} configs × {n_atoms} atoms)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Net charge per configuration
    net_charges = true_charges_all.sum(axis=1)
    axes[0, 1].hist(net_charges, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Net Charge (e)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Net Charge per Configuration\nMean: {net_charges.mean():.4f}, Std: {net_charges.std():.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Positive vs negative charge count
    pos_counts = [(c == 1.0).sum() for c in true_charges_all]
    neg_counts = [(c == -1.0).sum() for c in true_charges_all]
    
    x = np.arange(n_configs)
    axes[1, 0].bar(x, pos_counts, label='+1e', alpha=0.7)
    axes[1, 0].bar(x, neg_counts, bottom=pos_counts, label='-1e', alpha=0.7)
    axes[1, 0].set_xlabel('Configuration Index')
    axes[1, 0].set_ylabel('Atom Count')
    axes[1, 0].set_title('Charge Composition per Configuration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Coulomb energy distribution (computed from true charges)
    coulomb_energies = []
    for c in configs:
        e_coul = compute_coulomb_energy(c['true_charges'], c['positions'], c.get('pbc', 'F F F'))
        coulomb_energies.append(e_coul)
    
    axes[1, 1].hist(coulomb_energies, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Coulomb Energy (a.u.)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Computed Coulomb Energy Distribution\nMean: {np.mean(coulomb_energies):.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'random_charges_overview.png', dpi=150)
    plt.close()
    
    # Compute pairwise distance statistics
    distances_all = []
    for c in configs[:10]:  # Sample first 10 configs
        dists = compute_pairwise_distances(c['positions'], c.get('pbc', 'F F F'))
        # Get upper triangle (excluding diagonal)
        upper_tri = dists[np.triu_indices(n_atoms, k=1)]
        distances_all.extend(upper_tri.flatten())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(distances_all, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Interatomic Distance (Å)')
    ax.set_ylabel('Count')
    ax.set_title(f'Pairwise Distance Distribution (sampled from 10 configs)\nTotal pairs: {len(distances_all)}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'random_charges_distances.png', dpi=150)
    plt.close()
    
    # Save summary statistics
    stats = {
        'n_configurations': n_configs,
        'n_atoms_per_config': n_atoms,
        'charge_values': sorted(list(np.unique(true_charges_all))),
        'mean_pos_count': float(np.mean(pos_counts)),
        'mean_neg_count': float(np.mean(neg_counts)),
        'net_charge_mean': float(net_charges.mean()),
        'net_charge_std': float(net_charges.std()),
        'coulomb_energy_mean': float(np.mean(coulomb_energies)),
        'coulomb_energy_std': float(np.std(coulomb_energies)),
        'distance_mean': float(np.mean(distances_all)),
        'distance_std': float(np.std(distances_all)),
        'distance_min': float(np.min(distances_all)),
        'distance_max': float(np.max(distances_all))
    }
    
    with open(output_dir / 'random_charges_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Random charges analysis saved to {output_dir}")
    print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    return stats


def plot_charged_dimer_analysis(configs, output_dir):
    """Analyze charged_dimer dataset - binding energy curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_configs = len(configs)
    
    # Extract data
    energies = np.array([c.get('energy', 0) for c in configs])
    
    # Compute dimer separation distances
    separations = []
    for c in configs:
        positions = c['positions']
        # First dimer: atoms 0-3, Second dimer: atoms 4-7
        center1 = positions[:4].mean(axis=0)
        center2 = positions[4:].mean(axis=0)
        sep = np.linalg.norm(center1 - center2)
        separations.append(sep)
    
    separations = np.array(separations)
    
    # Sort by separation for plotting
    sort_idx = np.argsort(separations)
    separations_sorted = separations[sort_idx]
    energies_sorted = energies[sort_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Energy vs separation
    axes[0, 0].scatter(separations_sorted, energies_sorted, alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Dimer Separation (Å)')
    axes[0, 0].set_ylabel('Energy (a.u.)')
    axes[0, 0].set_title('Binding Energy Curve\n(Charged Dimer System)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Energy distribution
    axes[0, 1].hist(energies, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Energy (a.u.)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Energy Distribution\nMean: {energies.mean():.4f}, Std: {energies.std():.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Separation distribution
    axes[1, 0].hist(separations, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Separation (Å)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Separation Distribution\nRange: [{separations.min():.2f}, {separations.max():.2f}]')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Force magnitude distribution
    force_mags = []
    for c in configs:
        if 'forces' in c:
            force_mag = np.linalg.norm(c['forces'], axis=1)
            force_mags.extend(force_mag)
    
    if force_mags:
        axes[1, 1].hist(force_mags, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Force Magnitude (a.u.)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Force Magnitude Distribution\nMax: {np.max(force_mags):.4f}')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'charged_dimer_overview.png', dpi=150)
    plt.close()
    
    # Save summary statistics
    stats = {
        'n_configurations': n_configs,
        'n_atoms_per_config': 8,
        'separation_min': float(separations.min()),
        'separation_max': float(separations.max()),
        'separation_mean': float(separations.mean()),
        'energy_min': float(energies.min()),
        'energy_max': float(energies.max()),
        'energy_mean': float(energies.mean()),
        'energy_std': float(energies.std())
    }
    
    with open(output_dir / 'charged_dimer_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Charged dimer analysis saved to {output_dir}")
    
    return stats, separations, energies


def plot_ag3_chargestates_analysis(configs, output_dir):
    """Analyze ag3_chargestates dataset - charge state discrimination."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate by charge state
    configs_plus = [c for c in configs if c.get('charge_state') == 1]
    configs_minus = [c for c in configs if c.get('charge_state') == -1]
    
    print(f"Ag₃ dataset: {len(configs_plus)} configs with +1 charge, {len(configs_minus)} configs with -1 charge")
    
    # Extract energies
    energies_plus = np.array([c.get('energy', 0) for c in configs_plus])
    energies_minus = np.array([c.get('energy', 0) for c in configs_minus])
    
    # Compute bond lengths
    def get_bond_lengths(positions):
        """Get all three bond lengths for Ag₃ trimer."""
        d12 = np.linalg.norm(positions[0] - positions[1])
        d13 = np.linalg.norm(positions[0] - positions[2])
        d23 = np.linalg.norm(positions[1] - positions[2])
        return [d12, d13, d23]
    
    bond_lengths_plus = []
    bond_lengths_minus = []
    
    for c in configs_plus:
        bond_lengths_plus.extend(get_bond_lengths(c['positions']))
    for c in configs_minus:
        bond_lengths_minus.extend(get_bond_lengths(c['positions']))
    
    bond_lengths_plus = np.array(bond_lengths_plus)
    bond_lengths_minus = np.array(bond_lengths_minus)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Energy comparison
    axes[0, 0].hist(energies_plus, bins=20, alpha=0.7, label='+1 charge', edgecolor='black')
    axes[0, 0].hist(energies_minus, bins=20, alpha=0.7, label='-1 charge', edgecolor='black')
    axes[0, 0].set_xlabel('Energy (a.u.)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Energy Distribution by Charge State')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bond length comparison
    axes[0, 1].hist(bond_lengths_plus, bins=30, alpha=0.7, label='+1 charge', edgecolor='black')
    axes[0, 1].hist(bond_lengths_minus, bins=30, alpha=0.7, label='-1 charge', edgecolor='black')
    axes[0, 1].set_xlabel('Bond Length (Å)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Bond Length Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Energy vs average bond length
    avg_bond_plus = [np.mean(get_bond_lengths(c['positions'])) for c in configs_plus]
    avg_bond_minus = [np.mean(get_bond_lengths(c['positions'])) for c in configs_minus]
    
    axes[0, 2].scatter(avg_bond_plus, energies_plus, alpha=0.6, label='+1 charge', s=50)
    axes[0, 2].scatter(avg_bond_minus, energies_minus, alpha=0.6, label='-1 charge', s=50)
    axes[0, 2].set_xlabel('Average Bond Length (Å)')
    axes[0, 2].set_ylabel('Energy (a.u.)')
    axes[0, 2].set_title('Potential Energy Surface')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Force magnitude distribution
    force_mags_plus = []
    force_mags_minus = []
    
    for c in configs_plus:
        if 'forces' in c:
            force_mags_plus.extend(np.linalg.norm(c['forces'], axis=1))
    for c in configs_minus:
        if 'forces' in c:
            force_mags_minus.extend(np.linalg.norm(c['forces'], axis=1))
    
    axes[1, 0].hist(force_mags_plus, bins=30, alpha=0.7, label='+1 charge', edgecolor='black')
    axes[1, 0].hist(force_mags_minus, bins=30, alpha=0.7, label='-1 charge', edgecolor='black')
    axes[1, 0].set_xlabel('Force Magnitude (a.u.)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Force Magnitude Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Box plot of energies
    axes[1, 1].boxplot([energies_plus, energies_minus], labels=['+1', '-1'])
    axes[1, 1].set_ylabel('Energy (a.u.)')
    axes[1, 1].set_title('Energy Comparison by Charge State')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. PES difference visualization
    # Create scatter showing how same geometry might have different energies
    axes[1, 2].scatter(range(len(energies_plus)), energies_plus, alpha=0.6, label='+1', s=30)
    axes[1, 2].scatter(range(len(energies_minus)), energies_minus, alpha=0.6, label='-1', s=30)
    axes[1, 2].set_xlabel('Configuration Index')
    axes[1, 2].set_ylabel('Energy (a.u.)')
    axes[1, 2].set_title('Energy vs Configuration Index')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ag3_chargestates_overview.png', dpi=150)
    plt.close()
    
    # Save summary statistics
    stats = {
        'n_configs_plus': len(configs_plus),
        'n_configs_minus': len(configs_minus),
        'energy_plus_mean': float(energies_plus.mean()),
        'energy_plus_std': float(energies.std()),
        'energy_minus_mean': float(energies_minus.mean()),
        'energy_minus_std': float(energies_minus.std()),
        'bond_length_plus_mean': float(bond_lengths_plus.mean()),
        'bond_length_minus_mean': float(bond_lengths_minus.mean()),
        'bond_length_range': [float(bond_lengths_plus.min()), float(bond_lengths_plus.max())]
    }
    
    with open(output_dir / 'ag3_chargestates_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Ag₃ charge states analysis saved to {output_dir}")
    
    return stats


if __name__ == "__main__":
    base_dir = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260416_163347')
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'outputs'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Data Exploration and Visualization")
    print("=" * 60)
    
    # Parse all datasets
    print("\nParsing datasets...")
    random_charges = parse_xyz_file(str(data_dir / 'random_charges.xyz'))
    charged_dimer = parse_xyz_file(str(data_dir / 'charged_dimer.xyz'))
    ag3_chargestates = parse_xyz_file(str(data_dir / 'ag3_chargestates.xyz'))
    
    print(f"  random_charges: {len(random_charges)} configurations")
    print(f"  charged_dimer: {len(charged_dimer)} configurations")
    print(f"  ag3_chargestates: {len(ag3_chargestates)} configurations")
    
    # Analyze each dataset
    print("\nAnalyzing random_charges dataset...")
    rc_stats = plot_random_charges_analysis(random_charges, output_dir / 'random_charges')
    
    print("\nAnalyzing charged_dimer dataset...")
    cd_stats, separations, energies = plot_charged_dimer_analysis(charged_dimer, output_dir / 'charged_dimer')
    
    print("\nAnalyzing ag3_chargestates dataset...")
    ag3_stats = plot_ag3_chargestates_analysis(ag3_chargestates, output_dir / 'ag3_chargestates')
    
    print("\n" + "=" * 60)
    print("Data exploration complete!")
    print("=" * 60)
