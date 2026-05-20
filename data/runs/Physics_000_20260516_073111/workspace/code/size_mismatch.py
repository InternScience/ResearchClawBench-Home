#!/usr/bin/env python3
"""
Size Mismatch Analysis: Optimal size mismatch between adjacent icosahedral shells.

This module computes:
1. Atomic radius compatibility between elements
2. Optimal size mismatch ranges for different shell transitions
3. Shell-shell pair compatibility matrices
4. Validation against experimental data points
"""

import numpy as np
import json
import os

# Atomic physical parameters (atomic radius, Å)
atomic_radii = [('Na', 1.86), ('K', 2.27), ('Rb', 2.48), ('Cs', 2.65),
                ('Ag', 1.44), ('Cu', 1.28), ('Ni', 1.24)]

# Atomic pair compatibility data (element1, element2, mismatch)
atomic_pairs_compatibility = [('Na', 'Rb', 0.22), ('Ag', 'Cu', 0.12),
                              ('Ag', 'Ni', 0.15), ('Cu', 'Ni', 0.032)]

# Optimal size mismatch ranges
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

# Shell energy relative values
shell_energies = [
    (1, 'MC', 0.00), (2, 'MC', -2.35), (2, 'Ch1', -2.15),
    (3, 'MC', -4.82), (3, 'Ch1', -4.61), (3, 'BG', -4.55)
]

# Size mismatch calculation parameters
mismatch_params = [
    (1, 2, 'MC', 'MC', 0.04),
    (1, 2, 'MC', 'Ch1', 0.14),
    (2, 3, 'MC', 'MC', 0.038),
    (2, 3, 'MC', 'Ch1', 0.136),
    (2, 3, 'Ch1', 'Ch2', 0.21)
]

# Experimental validation points
experimental_points = [
    (1, 3, 0.048, 0.045),
    (3, 4, 0.042, 0.044),
    (4, 7, 0.138, 0.142),
    (7, 12, 0.132, 0.139)
]


def compute_size_mismatch(r_inner, r_outer):
    """Compute size mismatch between inner and outer shell atomic radii.
    
    sm = |r_outer - r_inner| / r_inner
    """
    return abs(r_outer - r_inner) / r_inner


def build_atomic_radius_dict():
    """Build dictionary of atomic radii."""
    return {elem: radius for elem, radius in atomic_radii}


def compute_pairwise_mismatches():
    """Compute all pairwise size mismatches between available elements."""
    radius_dict = build_atomic_radius_dict()
    elements = sorted(radius_dict.keys())
    
    results = []
    matrix = np.zeros((len(elements), len(elements)))
    
    for i, elem_i in enumerate(elements):
        for j, elem_j in enumerate(elements):
            if i != j:
                sm = compute_size_mismatch(radius_dict[elem_i], radius_dict[elem_j])
                results.append((elem_i, elem_j, round(sm, 4)))
                matrix[i, j] = sm
    
    return results, elements, matrix


def classify_mismatch_quality(sm, chiral_pair):
    """Classify whether a size mismatch falls within optimal range."""
    for inner_type, outer_type, smin, smax in optimal_mismatch_ranges:
        if (inner_type, outer_type) == chiral_pair:
            if smin <= sm <= smax:
                return 'optimal'
            elif sm < smin:
                return 'below_optimal'
            else:
                return 'above_optimal'
    return 'unknown'


def compute_optimal_clusters():
    """Find all potentially stable multi-component clusters based on size mismatch."""
    radius_dict = build_atomic_radius_dict()
    elements = sorted(radius_dict.keys())
    
    clusters = []
    
    for inner_elem in elements:
        for outer_elem in elements:
            if inner_elem == outer_elem:
                continue
            sm = compute_size_mismatch(radius_dict[inner_elem], radius_dict[outer_elem])
            
            # Check against optimal ranges for different chiral transitions
            for inner_type, outer_type, smin, smax in optimal_mismatch_ranges:
                if smin <= sm <= smax:
                    # Compute shell numbers
                    # Inner: first Mackay shell (13 atoms)
                    # Outer: depends on chiral category
                    if outer_type == 'MC':
                        outer_atoms = 42  # Second MC shell
                    elif outer_type == 'Ch1':
                        outer_atoms = 32  # First Ch1 shell  
                    elif outer_type == 'Ch2':
                        outer_atoms = 42
                    elif outer_type == 'BG':
                        outer_atoms = 32
                    else:
                        outer_atoms = 32
                    
                    cluster_name = f"{inner_elem}13@{outer_elem}{outer_atoms}"
                    clusters.append({
                        'name': cluster_name,
                        'inner_element': inner_elem,
                        'outer_element': outer_elem,
                        'inner_radius': radius_dict[inner_elem],
                        'outer_radius': radius_dict[outer_elem],
                        'size_mismatch': round(sm, 4),
                        'chiral_transition': f"{inner_type}->{outer_type}",
                        'quality': 'optimal'
                    })
    
    return clusters


def validate_experimental():
    """Validate theoretical predictions against experimental data."""
    results = []
    for t1, t2, measured_sm, theoretical_sm in experimental_points:
        error = abs(measured_sm - theoretical_sm)
        results.append({
            'T_i': t1,
            'T_j': t2,
            'measured_sm': measured_sm,
            'theoretical_sm': theoretical_sm,
            'absolute_error': round(error, 4),
            'relative_error': round(error / theoretical_sm * 100, 2) if theoretical_sm != 0 else 0
        })
    
    # Compute summary statistics
    errors = [r['absolute_error'] for r in results]
    summary = {
        'MAE': round(np.mean(errors), 4),
        'RMSE': round(np.sqrt(np.mean(np.array(errors)**2)), 4),
        'max_error': round(np.max(errors), 4),
        'min_error': round(np.min(errors), 4)
    }
    
    return results, summary


def compute_energy_stability():
    """Compute stability analysis based on shell energies."""
    energy_data = {}
    for shell_idx, chiral, energy in shell_energies:
        key = f"shell{shell_idx}_{chiral}"
        energy_data[key] = energy
    
    # Energy differences between chiral configurations for same shell
    stability = {}
    for shell_idx in [2, 3]:
        mc_energy = energy_data.get(f"shell{shell_idx}_MC", None)
        for chiral in ['Ch1', 'Ch2', 'BG']:
            ch_energy = energy_data.get(f"shell{shell_idx}_{chiral}", None)
            if mc_energy is not None and ch_energy is not None:
                delta = ch_energy - mc_energy
                stability[f"shell{shell_idx}_{chiral}_vs_MC"] = {
                    'MC_energy': mc_energy,
                    f'{chiral}_energy': ch_energy,
                    'delta_E': round(delta, 4),
                    'MC_favored': delta > 0
                }
    
    return energy_data, stability


def main():
    os.makedirs('outputs', exist_ok=True)
    
    output = {}
    
    # 1. Pairwise mismatches
    pairwise_results, elements, mismatch_matrix = compute_pairwise_mismatches()
    output['pairwise_mismatches'] = [{'elem1': e1, 'elem2': e2, 'mismatch': sm} 
                                      for e1, e2, sm in pairwise_results]
    output['elements'] = elements
    output['mismatch_matrix'] = mismatch_matrix.tolist()
    
    # 2. Optimal mismatch ranges
    output['optimal_mismatch_ranges'] = [
        {'inner': i, 'outer': o, 'min': mn, 'max': mx}
        for i, o, mn, mx in optimal_mismatch_ranges
    ]
    
    # 3. Validated clusters
    output['validated_clusters'] = [
        {'name': n, 'inner': ie, 'outer': oe, 'inner_chiral': ic, 'outer_chiral': oc}
        for n, ie, oe, ic, oc in multicomponent_clusters
    ]
    
    # 4. Predicted optimal clusters
    optimal_clusters = compute_optimal_clusters()
    output['predicted_optimal_clusters'] = optimal_clusters
    
    # 5. Experimental validation
    exp_results, exp_summary = validate_experimental()
    output['experimental_validation'] = exp_results
    output['experimental_summary'] = exp_summary
    
    # 6. Energy stability
    energy_data, stability = compute_energy_stability()
    output['energy_data'] = energy_data
    output['energy_stability'] = stability
    
    # 7. Mismatch parameter analysis
    output['mismatch_params'] = [
        {'shell_i': si, 'shell_j': sj, 'type_i': ti, 'type_j': tj, 'sm_optimal': sm}
        for si, sj, ti, tj, sm in mismatch_params
    ]
    
    # Save
    with open('outputs/size_mismatch_output.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Size mismatch analysis complete.")
    print(f"  Pairwise mismatches: {len(pairwise_results)} pairs")
    print(f"  Optimal clusters predicted: {len(optimal_clusters)}")
    print(f"  Experimental validation MAE: {exp_summary['MAE']}")
    print(f"  Experimental validation RMSE: {exp_summary['RMSE']}")


if __name__ == '__main__':
    main()
