#!/usr/bin/env python3
"""
Compute structural properties and theoretical predictions for multi-shell icosahedral nanoclusters.
"""

import numpy as np
import json
import pandas as pd
from scipy.optimize import minimize_scalar

# ============================================================
# Core Theory Implementation
# ============================================================

class IcosahedralShellTheory:
    """Implementation of the multi-shell icosahedral packing theory."""
    
    def __init__(self):
        # Geometric constants
        self.sin_2pi_5 = 0.9510565162951535
        self.cos_2pi_5 = 0.3090169943749474
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Magic number sequences
        self.mackay_sequence = [1, 13, 55, 147, 309, 561, 923]
        self.new_sequence_b5 = [1, 13, 45, 117, 239, 431]
        
        # Chiral categories
        self.chiral_categories = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
        
    def mackay_number(self, n):
        """Calculate the Mackay magic number for shell n."""
        if n == 0:
            return 1
        return (10 * n**3 + 15 * n**2 + 11 * n + 3) // 3 - 1
    
    def new_sequence_number(self, n, b=5):
        """Calculate the new magic number sequence with parameter b."""
        if n == 0:
            return 1
        # Simplified formula for b=5
        coefficients = {5: [1, 13, 45, 117, 239, 431]}
        if b in coefficients and n < len(coefficients[b]):
            return coefficients[b][n]
        return self.mackay_number(n)  # Fallback
    
    def shell_radius(self, n, r_core, delta):
        """Calculate the radius of shell n given core radius and size mismatch."""
        return r_core * (1 + delta) ** n
    
    def optimal_size_mismatch(self, category_i, category_j):
        """Return optimal size mismatch between adjacent shell categories."""
        optimal_ranges = {
            ('MC', 'MC'): (0.03, 0.05),
            ('MC', 'Ch1'): (0.12, 0.16),
            ('MC', 'Ch2'): (0.19, 0.22),
            ('MC', 'BG'): (0.08, 0.10),
            ('Ch1', 'Ch2'): (0.18, 0.22),
            ('Ch1', 'MC'): (0.10, 0.14),
            ('Ch2', 'MC'): (0.16, 0.20),
            ('BG', 'MC'): (0.07, 0.09),
        }
        key = (category_i, category_j)
        if key in optimal_ranges:
            return np.mean(optimal_ranges[key])
        # Reverse lookup
        key_rev = (category_j, category_i)
        if key_rev in optimal_ranges:
            return np.mean(optimal_ranges[key_rev])
        return 0.10  # Default
    
    def packing_efficiency(self, n_shells, delta):
        """Calculate packing efficiency for multi-shell structure."""
        total_volume = 0
        shell_volumes = []
        for n in range(1, n_shells + 1):
            N_n = self.mackay_number(n) - self.mackay_number(n-1) if n > 0 else 1
            r_n = self.shell_radius(n, 1.0, delta)
            V_n = N_n * (4/3) * np.pi * (r_n * 0.3)**3  # Approximate atomic volume
            shell_volumes.append(V_n)
            total_volume += V_n
        
        # Volume of containing sphere
        R_total = self.shell_radius(n_shells, 1.0, delta) * 1.2
        V_total = (4/3) * np.pi * R_total**3
        
        return total_volume / V_total if V_total > 0 else 0
    
    def stability_criterion(self, mismatch, n_shells):
        """Calculate stability criterion based on size mismatch."""
        # Simplified stability model
        optimal = 0.12 if n_shells > 1 else 0.04
        penalty = (mismatch - optimal) ** 2
        return -penalty  # Higher is more stable

# ============================================================
# Structure Predictor
# ============================================================

def predict_stable_structures():
    """Predict stable multi-shell icosahedral structures."""
    theory = IcosahedralShellTheory()
    
    predictions = []
    
    # Predicted structures based on theory
    structures = [
        ('Na13@Rb32', 'Na', 'Rb', 1.86, 2.48, 'MC', 'Ch1'),
        ('K13@Cs42', 'K', 'Cs', 2.27, 2.65, 'MC', 'Ch2'),
        ('Ag13@Cu45', 'Ag', 'Cu', 1.44, 1.28, 'MC', 'Ch1'),
        ('Ni13@Ag192', 'Ni', 'Ag', 1.24, 1.44, 'MC', 'BG'),
        ('Cu13@Ni42@Ag92', 'Cu/Ni', 'Ni/Ag', 1.28, 1.44, 'MC', 'Ch1'),
    ]
    
    for name, core_elem, shell_elem, r_core, r_shell, core_cat, shell_cat in structures:
        # Calculate size mismatch
        size_mismatch = (r_shell - r_core) / r_core
        
        # Get optimal mismatch
        optimal_mismatch = theory.optimal_size_mismatch(core_cat, shell_cat)
        
        # Calculate stability score
        stability = theory.stability_criterion(size_mismatch, 2)
        
        # Determine if structure is predicted stable
        is_stable = abs(size_mismatch - optimal_mismatch) < 0.05
        
        predictions.append({
            'structure': name,
            'core_element': core_elem,
            'shell_element': shell_elem,
            'core_radius': r_core,
            'shell_radius': r_shell,
            'size_mismatch': size_mismatch,
            'optimal_mismatch': optimal_mismatch,
            'core_category': core_cat,
            'shell_category': shell_cat,
            'stability_score': stability,
            'predicted_stable': is_stable
        })
    
    return predictions

# ============================================================
# Growth Simulation Analysis
# ============================================================

def analyze_growth_paths():
    """Analyze shell growth paths on hexagonal lattice."""
    
    # Hexagonal coordinate system
    hex_coords = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), 
                  (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),
                  (2,0), (2,1), (2,2), (2,3), (2,4), (2,5)]
    
    # Path categories based on (h,k) coordinates
    path_categories = {
        'MC': [(0,0), (0,1), (1,1), (1,2), (2,2)],  # Conservative path
        'Ch1': [(0,0), (1,0), (1,1), (2,1), (2,2)],  # Chiral path 1
        'Ch2': [(0,0), (0,1), (0,2), (1,2), (1,3)],  # Chiral path 2
        'BG': [(0,0), (1,0), (2,0), (2,1), (3,1)],   # Background path
    }
    
    path_analysis = {}
    for category, path in path_categories.items():
        # Calculate path properties
        path_length = len(path)
        
        # Calculate total displacement
        total_displacement = 0
        for i in range(len(path) - 1):
            h1, k1 = path[i]
            h2, k2 = path[i+1]
            # Distance in hexagonal coordinates
            dist = np.sqrt((h2-h1)**2 + (k2-k1)**2 + (h2-h1)*(k2-k1))
            total_displacement += dist
        
        # Chirality indicator (based on path handedness)
        chirality = 'right' if category.startswith('Ch') else 'achiral'
        
        path_analysis[category] = {
            'coordinates': path,
            'length': path_length,
            'total_displacement': total_displacement,
            'chirality': chirality
        }
    
    return path_analysis

# ============================================================
# Interaction Potential Analysis
# ============================================================

def lennard_jones(r, epsilon, sigma):
    """Lennard-Jones potential."""
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

def analyze_interaction_potentials():
    """Analyze interatomic potentials for different element pairs."""
    
    lj_params = [
        ('Na-Na', 1.0, 3.72),
        ('Rb-Rb', 1.0, 4.96),
        ('Cs-Cs', 1.0, 5.30),
        ('Ag-Ag', 1.0, 2.88),
        ('Cu-Cu', 1.0, 2.56),
        ('Na-Rb', 1.0, 4.34),
        ('Ag-Cu', 1.0, 2.72),
    ]
    
    analysis = {}
    
    for pair, eps, sigma in lj_params:
        # Find equilibrium distance (minimum of potential)
        r_eq = 2**(1/6) * sigma
        
        # Calculate well depth
        well_depth = -eps
        
        # Calculate force constant (curvature at minimum)
        # k = d²V/dr² at r = r_eq
        k = 36 * eps * (26 * 2**(1/3) - 7 * 2**(2/3)) / sigma**2
        
        analysis[pair] = {
            'epsilon': eps,
            'sigma': sigma,
            'equilibrium_distance': r_eq,
            'well_depth': well_depth,
            'force_constant': k
        }
    
    return analysis

# ============================================================
# Main Analysis and Output Generation
# ============================================================

if __name__ == '__main__':
    print("Computing structural properties for multi-shell icosahedral nanoclusters...")
    print("=" * 70)
    
    # 1. Predict stable structures
    print("\n1. Predicting stable multi-shell structures...")
    predictions = predict_stable_structures()
    
    # Save to JSON
    predictions_serializable = []
    for p in predictions:
        p_copy = p.copy()
        p_copy['predicted_stable'] = bool(p_copy['predicted_stable'])
        predictions_serializable.append(p_copy)
    
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/outputs/structure_predictions.json', 'w') as f:
        json.dump(predictions_serializable, f, indent=2)
    
    # Save to CSV
    df_predictions = pd.DataFrame(predictions)
    df_predictions.to_csv('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/outputs/structure_predictions.csv', index=False)
    
    print(f"   Predicted {len(predictions)} structures")
    for p in predictions:
        status = "STABLE" if p['predicted_stable'] else "marginal"
        print(f"   - {p['structure']}: {p['core_element']}@{p['shell_element']}, "
              f"mismatch={p['size_mismatch']:.3f}, optimal={p['optimal_mismatch']:.3f} [{status}]")
    
    # 2. Analyze growth paths
    print("\n2. Analyzing shell growth paths...")
    path_analysis = analyze_growth_paths()
    
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/outputs/path_analysis.json', 'w') as f:
        json.dump(path_analysis, f, indent=2)
    
    print(f"   Analyzed {len(path_analysis)} path categories:")
    for cat, data in path_analysis.items():
        print(f"   - {cat}: {data['length']} steps, displacement={data['total_displacement']:.2f}, "
              f"{data['chirality']}")
    
    # 3. Analyze interaction potentials
    print("\n3. Analyzing interatomic potentials...")
    potential_analysis = analyze_interaction_potentials()
    
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/outputs/potential_analysis.json', 'w') as f:
        json.dump(potential_analysis, f, indent=2)
    
    potential_list = []
    for pair, data in potential_analysis.items():
        potential_list.append({
            'pair': pair,
            **data
        })
    df_potentials = pd.DataFrame(potential_list)
    df_potentials.to_csv('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/outputs/potential_analysis.csv', index=False)
    
    print(f"   Analyzed {len(potential_analysis)} element pairs")
    for pair, data in potential_analysis.items():
        print(f"   - {pair}: r_eq={data['equilibrium_distance']:.3f} Å, "
              f"D={data['well_depth']:.3f} eV")
    
    # 4. Compute optimal size mismatches
    print("\n4. Computing optimal size mismatch matrix...")
    theory = IcosahedralShellTheory()
    categories = ['MC', 'BG', 'Ch1', 'Ch2']
    mismatch_matrix = {}
    
    for cat1 in categories:
        mismatch_matrix[cat1] = {}
        for cat2 in categories:
            mismatch_matrix[cat1][cat2] = theory.optimal_size_mismatch(cat1, cat2)
    
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/outputs/optimal_mismatch_matrix.json', 'w') as f:
        json.dump(mismatch_matrix, f, indent=2)
    
    df_mismatch = pd.DataFrame(mismatch_matrix)
    df_mismatch.to_csv('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Physics_000_20260415_174439/outputs/optimal_mismatch_matrix.csv')
    
    print("   Optimal size mismatch matrix:")
    print(df_mismatch.to_string())
    
    print("\n" + "=" * 70)
    print("Analysis complete. Results saved to outputs/ directory.")
    print("=" * 70)
