"""
Data analysis for Machine Learning Interatomic Potentials with Long-Range Electrostatics
This script analyzes the three benchmark datasets and implements models for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read
from ase import Atoms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Output directories
OUTPUT_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/outputs'
FIGURE_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def parse_xyz_extended(filename):
    """Parse extended XYZ format with forces and charges."""
    frames = []
    with open(filename, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    i = 0
    while i < len(lines):
        if lines[i].strip() == '':
            i += 1
            continue
        try:
            natoms = int(lines[i].strip())
            i += 1
            # Parse properties line
            props_line = lines[i]
            i += 1
            
            # Extract energy if present
            energy = None
            if 'energy=' in props_line:
                energy_str = props_line.split('energy=')[1].split()[0]
                energy = float(energy_str)
            
            # Extract true charges if present
            true_charges = None
            if 'true_charges=' in props_line:
                charges_str = props_line.split('true_charges="')[1].split('"')[0]
                true_charges = np.array([float(x) for x in charges_str.split()])
            
            # Extract charge state if present
            charge_state = None
            if 'charge_state=' in props_line:
                charge_state_str = props_line.split('charge_state=')[1].split()[0]
                charge_state = int(charge_state_str)
            
            # Extract total charge if present
            total_charge = None
            if 'total_charge=' in props_line:
                total_charge_str = props_line.split('total_charge=')[1].split()[0]
                total_charge = int(total_charge_str)
            
            # Parse atoms
            symbols = []
            positions = []
            forces = []
            for j in range(natoms):
                parts = lines[i].split()
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
                if len(parts) >= 7:
                    forces.append([float(x) for x in parts[4:7]])
                i += 1
            
            frame = {
                'natoms': natoms,
                'symbols': symbols,
                'positions': np.array(positions),
                'forces': np.array(forces) if forces else None,
                'energy': energy,
                'true_charges': true_charges,
                'charge_state': charge_state,
                'total_charge': total_charge
            }
            frames.append(frame)
        except Exception as e:
            print(f"Error parsing frame at line {i}: {e}")
            i += 1
    
    return frames


def analyze_random_charges():
    """Analyze the random_charges dataset."""
    print("=" * 60)
    print("Analyzing random_charges.xyz")
    print("=" * 60)
    
    frames = parse_xyz_extended('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/data/random_charges.xyz')
    print(f"Number of frames: {len(frames)}")
    print(f"Atoms per frame: {frames[0]['natoms']}")
    print(f"Has true charges: {frames[0]['true_charges'] is not None}")
    
    # Analyze charge distribution
    charges = frames[0]['true_charges']
    print(f"Charge distribution: {np.sum(charges > 0)} positive, {np.sum(charges < 0)} negative")
    print(f"Charge values: +{np.max(charges)}, {np.min(charges)}")
    
    # Calculate distances between opposite charges for a few frames
    distances = []
    for frame in frames[:10]:
        pos = frame['positions']
        charges = frame['true_charges']
        pos_pos = pos[charges > 0]
        pos_neg = pos[charges < 0]
        for pp in pos_pos[:5]:
            for pn in pos_neg[:5]:
                dist = np.linalg.norm(pp - pn)
                distances.append(dist)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Charge distribution visualization (first frame)
    ax = axes[0, 0]
    frame = frames[0]
    pos = frame['positions']
    charges = frame['true_charges']
    pos_mask = charges > 0
    neg_mask = charges < 0
    ax.scatter(pos[pos_mask, 0], pos[pos_mask, 1], c='red', s=100, alpha=0.6, label='+1e charges', marker='+')
    ax.scatter(pos[neg_mask, 0], pos[neg_mask, 1], c='blue', s=100, alpha=0.6, label='-1e charges', marker='_')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_title('Random Charge Distribution (Frame 1, XY projection)')
    ax.legend()
    ax.set_aspect('equal')
    
    # 2. Distance distribution
    ax = axes[0, 1]
    ax.hist(distances, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Distance (Å)')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution Between Opposite Charges')
    
    # 3. 3D visualization of first frame
    ax = axes[1, 0]
    from mpl_toolkits.mplot3d import Axes3D
    ax.remove()
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(pos[pos_mask, 0], pos[pos_mask, 1], pos[pos_mask, 2], 
               c='red', s=50, alpha=0.6, label='+1e', marker='o')
    ax.scatter(pos[neg_mask, 0], pos[neg_mask, 1], pos[neg_mask, 2], 
               c='blue', s=50, alpha=0.6, label='-1e', marker='s')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('3D Charge Distribution')
    ax.legend()
    
    # 4. Charge balance verification
    ax = axes[1, 1]
    total_charges = [np.sum(f['true_charges']) for f in frames]
    ax.plot(total_charges, 'o-', markersize=3, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--', label='Neutral')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Total Charge (e)')
    ax.set_title('System Charge Neutrality')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/fig1_random_charges_analysis.png', bbox_inches='tight')
    plt.close()
    
    # Save summary
    summary = {
        'n_frames': len(frames),
        'n_atoms': frames[0]['natoms'],
        'n_positive': int(np.sum(charges > 0)),
        'n_negative': int(np.sum(charges < 0)),
        'charge_neutral': all(abs(tc) < 1e-10 for tc in total_charges)
    }
    
    print(f"Summary: {summary}")
    return frames, summary


def analyze_charged_dimer():
    """Analyze the charged_dimer dataset."""
    print("\n" + "=" * 60)
    print("Analyzing charged_dimer.xyz")
    print("=" * 60)
    
    frames = parse_xyz_extended('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/data/charged_dimer.xyz')
    print(f"Number of frames: {len(frames)}")
    print(f"Atoms per frame: {frames[0]['natoms']}")
    
    # Extract energies
    energies = np.array([f['energy'] for f in frames if f['energy'] is not None])
    print(f"Energy range: [{np.min(energies):.4f}, {np.max(energies):.4f}] eV")
    
    # Calculate inter-molecular distances
    distances = []
    for frame in frames:
        pos = frame['positions']
        # CH3 groups: C at index 0 and 4
        c1_pos = pos[0]
        c2_pos = pos[4]
        dist = np.linalg.norm(c1_pos - c2_pos)
        distances.append(dist)
    distances = np.array(distances)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Energy vs Distance (Binding curve)
    ax = axes[0, 0]
    scatter = ax.scatter(distances, energies, c=range(len(energies)), cmap='viridis', s=50, alpha=0.7)
    ax.set_xlabel('Inter-molecular Distance (Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Binding Energy Curve: Charged Dimer')
    plt.colorbar(scatter, ax=ax, label='Frame Index')
    
    # Sort by distance for smooth curve
    sort_idx = np.argsort(distances)
    ax.plot(distances[sort_idx], energies[sort_idx], 'k--', alpha=0.3, linewidth=1)
    
    # 2. Force magnitudes
    ax = axes[0, 1]
    force_magnitudes = []
    for frame in frames:
        if frame['forces'] is not None:
            f_mag = np.linalg.norm(frame['forces'], axis=1)
            force_magnitudes.extend(f_mag)
    ax.hist(force_magnitudes, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Force Magnitude (eV/Å)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Force Magnitudes')
    ax.set_yscale('log')
    
    # 3. Sample configuration
    ax = axes[1, 0]
    from mpl_toolkits.mplot3d import Axes3D
    ax.remove()
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    frame = frames[len(frames)//2]
    pos = frame['positions']
    symbols = frame['symbols']
    colors = {'C': 'gray', 'H': 'white'}
    sizes = {'C': 100, 'H': 50}
    for i, (s, p) in enumerate(zip(symbols, pos)):
        ax.scatter(p[0], p[1], p[2], c=colors.get(s, 'blue'), s=sizes.get(s, 50), 
                  edgecolors='black', linewidth=0.5, label=s if i < 2 else "")
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Sample Dimer Configuration')
    
    # 4. Energy histogram
    ax = axes[1, 1]
    ax.hist(energies, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Distribution')
    ax.axvline(x=np.mean(energies), color='r', linestyle='--', label=f'Mean: {np.mean(energies):.3f} eV')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/fig2_charged_dimer_analysis.png', bbox_inches='tight')
    plt.close()
    
    summary = {
        'n_frames': len(frames),
        'n_atoms': frames[0]['natoms'],
        'energy_min': float(np.min(energies)),
        'energy_max': float(np.max(energies)),
        'energy_mean': float(np.mean(energies)),
        'distance_min': float(np.min(distances)),
        'distance_max': float(np.max(distances))
    }
    
    print(f"Summary: {summary}")
    return frames, summary, energies, distances


def analyze_ag3_chargestates():
    """Analyze the Ag3 chargestates dataset."""
    print("\n" + "=" * 60)
    print("Analyzing ag3_chargestates.xyz")
    print("=" * 60)
    
    frames = parse_xyz_extended('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/data/ag3_chargestates.xyz')
    print(f"Number of frames: {len(frames)}")
    print(f"Atoms per frame: {frames[0]['natoms']}")
    
    # Separate by charge state
    charge_state_1 = [f for f in frames if f['charge_state'] == 1]
    charge_state_neg1 = [f for f in frames if f['charge_state'] == -1]
    
    print(f"Frames with charge state +1: {len(charge_state_1)}")
    print(f"Frames with charge state -1: {len(charge_state_neg1)}")
    
    energies_1 = np.array([f['energy'] for f in charge_state_1])
    energies_neg1 = np.array([f['energy'] for f in charge_state_neg1])
    
    print(f"Energy range (+1): [{np.min(energies_1):.4f}, {np.max(energies_1):.4f}] eV")
    print(f"Energy range (-1): [{np.min(energies_neg1):.4f}, {np.max(energies_neg1):.4f}] eV")
    
    # Calculate bond lengths
    def calculate_bond_lengths(frame):
        pos = frame['positions']
        bonds = []
        for i in range(3):
            for j in range(i+1, 3):
                dist = np.linalg.norm(pos[i] - pos[j])
                bonds.append(dist)
        return bonds
    
    bonds_1 = [calculate_bond_lengths(f) for f in charge_state_1]
    bonds_neg1 = [calculate_bond_lengths(f) for f in charge_state_neg1]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Energy vs bond length for +1 state
    ax = axes[0, 0]
    avg_bonds_1 = [np.mean(b) for b in bonds_1]
    ax.scatter(avg_bonds_1, energies_1, c='red', s=50, alpha=0.6, label='Charge +1')
    ax.set_xlabel('Average Bond Length (Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Ag₃ PES: Charge State +1')
    ax.legend()
    
    # 2. Energy vs bond length for -1 state
    ax = axes[0, 1]
    avg_bonds_neg1 = [np.mean(b) for b in bonds_neg1]
    ax.scatter(avg_bonds_neg1, energies_neg1, c='blue', s=50, alpha=0.6, label='Charge -1')
    ax.set_xlabel('Average Bond Length (Å)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Ag₃ PES: Charge State -1')
    ax.legend()
    
    # 3. Combined energy comparison
    ax = axes[1, 0]
    ax.hist(energies_1, bins=20, alpha=0.5, label='Charge +1', color='red', edgecolor='black')
    ax.hist(energies_neg1, bins=20, alpha=0.5, label='Charge -1', color='blue', edgecolor='black')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Count')
    ax.set_title('Energy Distribution by Charge State')
    ax.legend()
    
    # 4. Sample Ag3 configurations
    ax = axes[1, 1]
    from mpl_toolkits.mplot3d import Axes3D
    ax.remove()
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Plot +1 state
    frame_1 = charge_state_1[0]
    pos_1 = frame_1['positions']
    ax.scatter(pos_1[:, 0], pos_1[:, 1], pos_1[:, 2], 
              c='red', s=200, alpha=0.8, label='+1 state', marker='o')
    
    # Plot -1 state (offset)
    frame_neg1 = charge_state_neg1[0]
    pos_neg1 = frame_neg1['positions'] + np.array([3, 0, 0])
    ax.scatter(pos_neg1[:, 0], pos_neg1[:, 1], pos_neg1[:, 2], 
              c='blue', s=200, alpha=0.8, label='-1 state', marker='s')
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Sample Ag₃ Configurations')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/fig3_ag3_chargestates_analysis.png', bbox_inches='tight')
    plt.close()
    
    summary = {
        'n_frames': len(frames),
        'n_atoms': frames[0]['natoms'],
        'n_charge_1': len(charge_state_1),
        'n_charge_neg1': len(charge_state_neg1),
        'energy_1_min': float(np.min(energies_1)),
        'energy_1_max': float(np.max(energies_1)),
        'energy_neg1_min': float(np.min(energies_neg1)),
        'energy_neg1_max': float(np.max(energies_neg1))
    }
    
    print(f"Summary: {summary}")
    return frames, summary, energies_1, energies_neg1


def implement_les_model():
    """
    Implement a simplified Latent Ewald Summation (LES) model.
    This is a conceptual implementation for demonstration.
    """
    print("\n" + "=" * 60)
    print("Implementing LES-style Model")
    print("=" * 60)
    
    # Load random charges data
    frames = parse_xyz_extended('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/data/random_charges.xyz')
    
    # For demonstration, we'll implement a simple charge prediction model
    # based on local environment descriptors
    
    def compute_rdf_descriptor(positions, charges, n_bins=20, r_max=10.0):
        """Compute a simple radial distribution function descriptor."""
        n_atoms = len(positions)
        descriptor = np.zeros(n_bins)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                r = np.linalg.norm(positions[i] - positions[j])
                if r < r_max:
                    bin_idx = int(r / r_max * n_bins)
                    if bin_idx < n_bins:
                        descriptor[bin_idx] += charges[i] * charges[j] / r
        return descriptor
    
    # Compute descriptors for each frame
    descriptors = []
    target_charges_list = []
    
    for frame in frames[:20]:  # Use subset for speed
        pos = frame['positions']
        charges = frame['true_charges']
        
        # Compute simple distance-based features for each atom
        for i in range(len(pos)):
            # Distance to all other atoms
            dists = np.linalg.norm(pos - pos[i], axis=1)
            dists = np.sort(dists)[1:11]  # 10 nearest neighbors
            
            # Simple feature: mean and std of distances
            feat = [np.mean(dists), np.std(dists), len(dists)]
            descriptors.append(feat)
            target_charges_list.append(charges[i])
    
    X = np.array(descriptors)
    y = np.array(target_charges_list)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Charge prediction MAE: {mae:.4f}")
    print(f"Charge prediction RMSE: {rmse:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
    ax.plot([-1.5, 1.5], [-1.5, 1.5], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Charge (e)')
    ax.set_ylabel('Predicted Charge (e)')
    ax.set_title(f'Charge Prediction (MAE: {mae:.4f})')
    ax.legend()
    ax.set_aspect('equal')
    
    ax = axes[1]
    residuals = y_test - y_pred
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Residual (e)')
    ax.set_ylabel('Count')
    ax.set_title('Charge Prediction Residuals')
    ax.axvline(x=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/fig4_les_charge_prediction.png', bbox_inches='tight')
    plt.close()
    
    return {'mae': mae, 'rmse': rmse}


def compare_model_generations():
    """
    Compare different generations of ML potentials conceptually.
    """
    print("\n" + "=" * 60)
    print("Comparing ML Potential Generations")
    print("=" * 60)
    
    # Conceptual comparison based on capabilities
    models = ['2G (Local)', '3G (Charge)', '4G (Charge Eq.)', 'LES (Latent)']
    
    capabilities = {
        'Local Interactions': [1, 1, 1, 1],
        'Long-range Electrostatics': [0, 1, 1, 1],
        'Non-local Charge Transfer': [0, 0, 1, 1],
        'Multiple Charge States': [0, 0, 1, 1],
        'No Explicit Charges': [0, 0, 0, 1]
    }
    
    # Create comparison table
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = np.array(list(capabilities.values()))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(capabilities)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(capabilities.keys())
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(capabilities)):
        for j in range(len(models)):
            text = ax.text(j, i, '✓' if data[i, j] == 1 else '✗',
                          ha="center", va="center", color="black", fontsize=16)
    
    ax.set_title('ML Potential Generations: Capabilities Comparison')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/fig5_model_comparison.png', bbox_inches='tight')
    plt.close()
    
    print("Model comparison figure saved")


def generate_summary_statistics():
    """Generate summary statistics for all datasets."""
    print("\n" + "=" * 60)
    print("Generating Summary Statistics")
    print("=" * 60)
    
    # Parse all datasets
    frames_rc = parse_xyz_extended('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/data/random_charges.xyz')
    frames_cd = parse_xyz_extended('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/data/charged_dimer.xyz')
    frames_ag = parse_xyz_extended('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Chemistry_003_20260415_124720/data/ag3_chargestates.xyz')
    
    summary = {
        'random_charges': {
            'n_frames': len(frames_rc),
            'n_atoms_per_frame': frames_rc[0]['natoms'],
            'total_atoms': len(frames_rc) * frames_rc[0]['natoms'],
            'has_forces': frames_rc[0]['forces'] is not None,
            'has_charges': frames_rc[0]['true_charges'] is not None
        },
        'charged_dimer': {
            'n_frames': len(frames_cd),
            'n_atoms_per_frame': frames_cd[0]['natoms'],
            'total_atoms': len(frames_cd) * frames_cd[0]['natoms'],
            'energy_mean': float(np.mean([f['energy'] for f in frames_cd])),
            'energy_std': float(np.std([f['energy'] for f in frames_cd])),
            'has_forces': frames_cd[0]['forces'] is not None
        },
        'ag3_chargestates': {
            'n_frames': len(frames_ag),
            'n_atoms_per_frame': frames_ag[0]['natoms'],
            'total_atoms': len(frames_ag) * frames_ag[0]['natoms'],
            'n_charge_states': len(set(f['charge_state'] for f in frames_ag)),
            'has_forces': frames_ag[0]['forces'] is not None
        }
    }
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dataset sizes
    ax = axes[0, 0]
    datasets = ['random_charges', 'charged_dimer', 'ag3_chargestates']
    frame_counts = [summary[d]['n_frames'] for d in datasets]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(datasets, frame_counts, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Frames')
    ax.set_title('Dataset Sizes (Frames)')
    for bar, count in zip(bars, frame_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Atoms per frame
    ax = axes[0, 1]
    atom_counts = [summary[d]['n_atoms_per_frame'] for d in datasets]
    bars = ax.bar(datasets, atom_counts, color=colors, edgecolor='black')
    ax.set_ylabel('Atoms per Frame')
    ax.set_title('System Sizes')
    for bar, count in zip(bars, atom_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Feature availability
    ax = axes[1, 0]
    features = ['Energy', 'Forces', 'Charges']
    random_charges_feats = [0, 0, 1]  # Has charges but no energy/forces in data
    dimer_feats = [1, 1, 0]
    ag3_feats = [1, 1, 1]  # Has charge state info
    
    x = np.arange(len(features))
    width = 0.25
    ax.bar(x - width, random_charges_feats, width, label='random_charges', color=colors[0], edgecolor='black')
    ax.bar(x, dimer_feats, width, label='charged_dimer', color=colors[1], edgecolor='black')
    ax.bar(x + width, ag3_feats, width, label='ag3_chargestates', color=colors[2], edgecolor='black')
    ax.set_ylabel('Available (1=Yes, 0=No)')
    ax.set_title('Available Features by Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    ax.set_ylim(0, 1.5)
    
    # Energy ranges for datasets with energy
    ax = axes[1, 1]
    cd_energies = [f['energy'] for f in frames_cd]
    ag_energies = [f['energy'] for f in frames_ag]
    
    bp = ax.boxplot([cd_energies, ag_energies], labels=['charged_dimer', 'ag3_chargestates'],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[1:]):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Energy Distributions')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/fig6_summary_statistics.png', bbox_inches='tight')
    plt.close()
    
    return summary


def main():
    """Run all analyses."""
    print("=" * 60)
    print("Machine Learning Interatomic Potentials Analysis")
    print("Long-Range Electrostatic Interactions Study")
    print("=" * 60)
    
    # Analyze each dataset
    frames_rc, summary_rc = analyze_random_charges()
    frames_cd, summary_cd, energies_cd, distances_cd = analyze_charged_dimer()
    frames_ag, summary_ag, energies_1, energies_neg1 = analyze_ag3_chargestates()
    
    # Implement LES-style model
    les_results = implement_les_model()
    
    # Compare model generations
    compare_model_generations()
    
    # Generate summary statistics
    summary = generate_summary_statistics()
    
    # Save all results
    results = {
        'random_charges': summary_rc,
        'charged_dimer': summary_cd,
        'ag3_chargestates': summary_ag,
        'les_results': les_results,
        'overall_summary': summary
    }
    
    import json
    with open(f'{OUTPUT_DIR}/analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {OUTPUT_DIR}/analysis_results.json")
    print(f"Figures saved to: {FIGURE_DIR}/")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
