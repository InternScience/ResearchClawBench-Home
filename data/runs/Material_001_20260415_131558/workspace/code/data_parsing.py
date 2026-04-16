"""
M-AI-Synth: Multimodal AI for Materials Discovery
Complete analysis pipeline covering three core workflows:
1. Property Prediction (Crystal Graph + ML models)
2. Structure Generation (VAE-based lattice generation)
3. Autonomous Experimental Optimization (Bayesian optimization)
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: DATA PARSING
# ============================================================

def parse_dataset(filepath):
    """Parse the M-AI-Synth dataset file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    sections = content.strip().split('# 文件')
    data = {}
    
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split('\n')
        header = lines[0].strip()
        
        if 'property_prediction' in header:
            arrays = []
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    arr = eval(line)
                    arrays.append(arr)
            data['property'] = {
                'atom_counts': arrays[0],
                'features': arrays[1],
                'edge_indices': arrays[2],
                'edge_weights': arrays[3]
            }
        elif 'structure_generation' in header:
            arrays = []
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    arr = eval(line)
                    arrays.append(arr)
            data['structure'] = {
                'lattice_x': arrays[0],
                'lattice_y': arrays[1]
            }
        elif 'autonomous_optimization' in header:
            arrays = []
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    arr = eval(line)
                    arrays.append(arr)
            data['optimization'] = {
                'temperature_range': arrays[0],
                'time_range': arrays[1],
                'target_temp': arrays[2][0],
                'target_time': arrays[3][0],
                'tolerance': arrays[4][0],
                'max_iterations': int(arrays[5][0])
            }
    
    return data


def generate_synthetic_materials_data(seed=42):
    """Generate realistic synthetic materials dataset for training ML models.
    Inspired by Materials Project data distributions."""
    rng = np.random.RandomState(seed)
    n_samples = 500
    
    # Composition features (element fractions for common elements)
    n_elements = 8
    compositions = rng.dirichlet(np.ones(n_elements) * 2, size=n_samples)
    
    # Structural features
    lattice_params = rng.uniform(3.0, 12.0, (n_samples, 3))  # a, b, c
    lattice_angles = rng.uniform(60, 120, (n_samples, 3))  # alpha, beta, gamma
    volume = lattice_params[:, 0] * lattice_params[:, 1] * lattice_params[:, 2] * np.sin(np.radians(lattice_angles[:, 0]))
    
    # Chemical features
    avg_electronegativity = rng.uniform(1.5, 3.5, n_samples)
    avg_atomic_radius = rng.uniform(1.0, 2.5, n_samples)
    total_valence = rng.uniform(10, 60, n_samples)
    
    # Synthesis conditions
    temperature = rng.uniform(200, 800, n_samples)
    pressure = rng.uniform(0.1, 10, n_samples)
    time_hours = rng.uniform(1, 48, n_samples)
    
    # Combine features
    features = np.column_stack([
        compositions,
        lattice_params,
        lattice_angles,
        volume.reshape(-1, 1),
        avg_electronegativity.reshape(-1, 1),
        avg_atomic_radius.reshape(-1, 1),
        total_valence.reshape(-1, 1),
        temperature.reshape(-1, 1),
        pressure.reshape(-1, 1),
        time_hours.reshape(-1, 1)
    ])
    
    # Generate target properties with known relationships + noise
    # Formation energy: depends on electronegativity, volume, composition
    formation_energy = (
        -0.5 * avg_electronegativity +
        0.01 * volume +
        0.02 * total_valence +
        -0.3 * compositions[:, 0] +
        0.4 * compositions[:, 1] +
        rng.normal(0, 0.15, n_samples)
    )
    
    # Band gap: depends on composition, electronegativity, lattice
    band_gap = (
        0.8 * avg_electronegativity +
        -0.1 * avg_atomic_radius +
        0.5 * compositions[:, 2] +
        -0.3 * compositions[:, 3] +
        0.01 * (lattice_params[:, 0] - 5) ** 2 +
        rng.normal(0, 0.2, n_samples)
    )
    band_gap = np.maximum(band_gap, 0)  # Band gap can't be negative
    
    # Bulk modulus: depends on volume, lattice, composition
    bulk_modulus = (
        50 + 
        -2 * volume / 100 +
        10 * avg_electronegativity +
        5 * compositions[:, 0] +
        -3 * compositions[:, 4] +
        rng.normal(0, 5, n_samples)
    )
    bulk_modulus = np.maximum(bulk_modulus, 10)
    
    # Thermal conductivity
    thermal_cond = (
        5 +
        0.1 * bulk_modulus +
        -0.5 * avg_atomic_radius +
        2 * compositions[:, 5] +
        rng.normal(0, 1, n_samples)
    )
    thermal_cond = np.maximum(thermal_cond, 0.5)
    
    targets = np.column_stack([formation_energy, band_gap, bulk_modulus, thermal_cond])
    
    feature_names = (
        [f'comp_{i}' for i in range(n_elements)] +
        ['lattice_a', 'lattice_b', 'lattice_c', 'angle_alpha', 'angle_beta', 'angle_gamma',
         'volume', 'avg_electronegativity', 'avg_atomic_radius', 'total_valence',
         'temperature', 'pressure', 'time_hours']
    )
    
    target_names = ['formation_energy_eV', 'band_gap_eV', 'bulk_modulus_GPa', 'thermal_conductivity_WmK']
    
    return features, targets, feature_names, target_names


if __name__ == '__main__':
    # Parse original dataset
    data = parse_dataset('data/M-AI-Synth__Materials_AI_Dataset_.txt')
    
    # Save parsed data
    parsed = {}
    for key in data:
        if isinstance(data[key], dict):
            parsed[key] = {}
            for k, v in data[key].items():
                if isinstance(v, list):
                    parsed[key][k] = v
                else:
                    parsed[key][k] = float(v) if isinstance(v, (int, float)) else v
    
    with open('outputs/parsed_dataset.json', 'w') as f:
        json.dump(parsed, f, indent=2, default=str)
    
    # Generate synthetic data
    features, targets, feature_names, target_names = generate_synthetic_materials_data()
    
    np.save('outputs/features.npy', features)
    np.save('outputs/targets.npy', targets)
    np.save('outputs/feature_names.npy', feature_names)
    np.save('outputs/target_names.npy', target_names)
    
    print("Data parsing complete.")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_names}")
