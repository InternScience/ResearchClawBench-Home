#!/usr/bin/env python3
"""
Multi-Component Icosahedral Nanocluster Analysis
================================================
Comprehensive analysis of multi-component icosahedral shell stacking theory,
including size mismatch optimization, shell energy calculations, growth
simulation dynamics, and validation against experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Ensure output directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("report/images", exist_ok=True)

# ============================================================================
# 1. DATA DEFINITIONS (from reproduction data file)
# ============================================================================

# Hexagonal coordinate sequence
hexagonal_coords = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5),
                    (1,0), (1,1), (1,2), (1,3), (1,4), (1,5),
                    (2,0), (2,1), (2,2), (2,3), (2,4), (2,5),
                    (3,0), (3,1), (3,2), (3,3), (3,4), (3,5),
                    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5),
                    (5,0), (5,1), (5,2), (5,3), (5,4), (5,5)]

# Magic number sequences
mackay_sequence = [1, 13, 55, 147, 309]
new_sequence_b5 = [1, 13, 45, 117, 239, 431]

# Chiral categories
chiral_labels = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
shell_colors = {
    'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c',
    'Ch2': '#d62728', 'Ch3': '#9467bd', 'Ch4': '#8c564b', 'Ch5': '#e377c2'
}

# Geometric constants
sin_2pi_5 = 0.9510565162951535
cos_2pi_5 = 0.3090169943749474

# Atomic radii (Å)
atomic_radii = {'Na': 1.86, 'K': 2.27, 'Rb': 2.48, 'Cs': 2.65,
                'Ag': 1.44, 'Cu': 1.28, 'Ni': 1.24}

# Atomic pair compatibility (element1, element2, size_mismatch)
atomic_pairs = [
    ('Na', 'Rb', 0.22), ('Ag', 'Cu', 0.12),
    ('Ag', 'Ni', 0.15), ('Cu', 'Ni', 0.032)
]

# Optimal size mismatch ranges by chiral category pair
mismatch_ranges = [
    ('MC', 'MC', 0.03, 0.05), ('MC', 'Ch1', 0.12, 0.16),
    ('MC', 'Ch2', 0.19, 0.22), ('MC', 'BG', 0.08, 0.10)
]

# Multi-component cluster validation
multicomponent_clusters = [
    ('Na13@Rb32', 'Na', 'Rb', 'MC', 'Ch1'),
    ('K13@Cs42', 'K', 'Cs', 'MC', 'Ch2'),
    ('Ag13@Cu45', 'Ag', 'Cu', 'MC', 'Ch1')
]

# Shell energies (shell_number, chiral_category, energy_normalized)
shell_energies = [
    (1, 'MC', 0.00), (2, 'MC', -2.35), (2, 'Ch1', -2.15),
    (3, 'MC', -4.82), (3, 'Ch1', -4.61), (3, 'BG', -4.55)
]

# Size mismatch parameters (shell_i, shell_j, cat_i, cat_j, delta)
mismatch_params = [
    (1, 2, 'MC', 'MC', 0.04), (1, 2, 'MC', 'Ch1', 0.14),
    (2, 3, 'MC', 'MC', 0.038), (2, 3, 'MC', 'Ch1', 0.136),
    (2, 3, 'Ch1', 'Ch2', 0.21)
]

# Experimental validation points
experimental_points = [
    (1, 3, 0.048, 0.045), (3, 4, 0.042, 0.044),
    (4, 7, 0.138, 0.142), (7, 12, 0.132, 0.139)
]

# Growth simulation parameters
growth_params = {
    'temperature': 300.0, 'deposition_rate': 0.01,
    'simulation_steps': 1000, 'beta_factor': 1.0,
    'delta_opt': 0.04, 'random_seed': 42
}

# Path probability weights
path_weights = [
    ('conservative_step', 0.65), ('mismatch_driven_step', 0.25),
    ('random_step', 0.10)
]

# Growth results (steps, chiral_category, avg_mismatch)
growth_results = [
    (0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02),
    (30, 'MC', 0.025), (40, 'MC', 0.03), (50, 'MC', 0.035),
    (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14),
    (30, 'Ch1', 0.138), (40, 'Ch1', 0.136), (50, 'Ch1', 0.135),
    (0, 'MC', 0.00), (10, 'MC', 0.08), (20, 'Ch1', 0.14),
    (30, 'Ch1', 0.15), (40, 'Ch1', 0.145), (50, 'Ch1', 0.142)
]

# Path selection statistics
path_selection_stats = [
    ('Conservative path', 325), ('Mismatch-driven path', 125),
    ('Random path', 50), ('Reverse step', 100)
]

# LJ parameters (pair, epsilon, sigma)
lj_parameters = [
    ('Na-Na', 1.0, 3.72), ('Rb-Rb', 1.0, 4.96), ('Cs-Cs', 1.0, 5.30),
    ('Ag-Ag', 1.0, 2.88), ('Cu-Cu', 1.0, 2.56),
    ('Na-Rb', 1.0, 4.34), ('Ag-Cu', 1.0, 2.72)
]

# Thermodynamic parameters
thermo_params = {
    'kT': 0.02585, 'boltzmann': 8.617e-5,
    'pressure': 1.0, 'timestep': 0.001
}


# ============================================================================
# 2. COMPUTATIONAL FUNCTIONS
# ============================================================================

def compute_size_mismatch(r_inner, r_outer):
    """Compute size mismatch δ = |r_outer - r_inner| / r_inner."""
    return abs(r_outer - r_inner) / r_inner


def lennard_jones_potential(r, epsilon, sigma):
    """Compute Lennard-Jones potential: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]."""
    sr6 = (sigma / r) ** 6
    return 4 * epsilon * (sr6 ** 2 - sr6)


def shell_atom_count_mackay(n_shell):
    """Return cumulative atom count for Mackay icosahedron at shell n."""
    # N = (10n^3 + 15n^2 + 11n + 3)/3 for n shells
    if n_shell == 0:
        return 1
    n = n_shell
    return int((10 * n**3 + 15 * n**2 + 11 * n + 3) // 3)


def shell_atom_count_new_b5(n_shell):
    """Return cumulative atom count for new sequence (b=5)."""
    seq = [1, 13, 45, 117, 239, 431]
    if n_shell < len(seq):
        return seq[n_shell]
    # Extrapolate using polynomial fit
    x = np.arange(len(seq))
    coeffs = np.polyfit(x, seq, 3)
    return int(np.round(np.polyval(coeffs, n_shell)))


def hexagonal_to_cartesian(h, k, a=1.0):
    """Convert hexagonal lattice coordinates to Cartesian (2D)."""
    x = a * (h + 0.5 * k)
    y = a * (np.sqrt(3) / 2) * k
    return x, y


def compute_shell_energy_theoretical(shell_num, chiral_cat, ref_energies):
    """Interpolate or predict shell energy based on reference data."""
    for sn, cc, e in ref_energies:
        if sn == shell_num and cc == chiral_cat:
            return e
    # If not found, estimate from nearest reference
    mc_energies = [(sn, e) for sn, cc, e in ref_energies if cc == 'MC']
    if mc_energies:
        # Linear extrapolation from MC trend
        shells = [s for s, _ in mc_energies]
        energies = [e for _, e in mc_energies]
        if shell_num in shells:
            idx = shells.index(shell_num)
            base_e = energies[idx]
            # Apply chiral correction
            corrections = {'MC': 0.0, 'Ch1': 0.20, 'Ch2': 0.27, 'BG': 0.27}
            return base_e + corrections.get(chiral_cat, 0.25)
    return None


def stability_score(delta, optimal_delta, tolerance=0.02):
    """Compute stability score based on proximity to optimal mismatch."""
    deviation = abs(delta - optimal_delta)
    if deviation <= tolerance:
        return 1.0 - deviation / tolerance
    else:
        return max(0.0, 1.0 - deviation / (2 * tolerance))


# ============================================================================
# 3. ANALYSIS & COMPUTATION
# ============================================================================

print("=" * 60)
print("MULTI-COMPONENT ICOSAHEDRAL NANACLUSTER ANALYSIS")
print("=" * 60)

# --- 3a. Size mismatch analysis for atomic pairs ---
print("\n[1] Computing size mismatches for atomic pairs...")
pair_results = []
for elem1, elem2, reported_mm in atomic_pairs:
    r1 = atomic_radii[elem1]
    r2 = atomic_radii[elem2]
    computed_mm = compute_size_mismatch(r1, r2)
    pair_results.append({
        'pair': f'{elem1}-{elem2}',
        'r1_A': r1, 'r2_A': r2,
        'computed_mismatch': round(computed_mm, 4),
        'reported_mismatch': reported_mm,
        'deviation': round(abs(computed_mm - reported_mm), 4)
    })
    print(f"  {elem1}-{elem2}: r={r1}/{r2} Å, δ_computed={computed_mm:.4f}, "
          f"δ_reported={reported_mm:.3f}")

# Save pair results
with open("outputs/atomic_pair_mismatches.json", "w") as f:
    json.dump(pair_results, f, indent=2)

# --- 3b. Shell energy analysis ---
print("\n[2] Analyzing shell energies...")
energy_data = {}
for sn, cc, e in shell_energies:
    key = f"shell{sn}_{cc}"
    energy_data[key] = {'shell': sn, 'category': cc, 'energy': e}

# Compute per-shell energy differences between categories
shell_comparison = {}
for sn in [2, 3]:
    mc_e = next((e for s, c, e in shell_energies if s == sn and c == 'MC'), None)
    ch1_e = next((e for s, c, e in shell_energies if s == sn and c == 'Ch1'), None)
    bg_e = next((e for s, c, e in shell_energies if s == sn and c == 'BG'), None)
    shell_comparison[f"shell{sn}"] = {
        'MC': mc_e, 'Ch1': ch1_e, 'BG': bg_e,
        'MC_vs_Ch1_diff': round(mc_e - ch1_e, 3) if mc_e and ch1_e else None,
        'MC_vs_BG_diff': round(mc_e - bg_e, 3) if mc_e and bg_e else None
    }
    print(f"  Shell {sn}: MC={mc_e}, Ch1={ch1_e}, BG={bg_e}")

with open("outputs/shell_energy_analysis.json", "w") as f:
    json.dump(shell_comparison, f, indent=2)

# --- 3c. Theoretical vs experimental mismatch validation ---
print("\n[3] Validating theoretical vs experimental mismatch...")
validation_results = []
for ti, tj, measured, theoretical in experimental_points:
    error = abs(measured - theoretical)
    rel_error = error / theoretical * 100 if theoretical != 0 else float('inf')
    validation_results.append({
        'shells': f'{ti}-{tj}',
        'measured': measured,
        'theoretical': theoretical,
        'absolute_error': round(error, 4),
        'relative_error_pct': round(rel_error, 2)
    })
    print(f"  Shells {ti}-{tj}: measured={measured:.3f}, "
          f"theoretical={theoretical:.3f}, rel_error={rel_error:.2f}%")

with open("outputs/theory_experiment_validation.json", "w") as f:
    json.dump(validation_results, f, indent=2)

# --- 3d. Predicted stable multi-component clusters ---
print("\n[4] Predicting stable multi-component clusters...")
cluster_predictions = []
for cluster_name, inner_elem, outer_elem, inner_cat, outer_cat in multicomponent_clusters:
    r_inner = atomic_radii[inner_elem]
    r_outer = atomic_radii[outer_elem]
    delta = compute_size_mismatch(r_inner, r_outer)

    # Find optimal range for this category pair
    opt_range = None
    for cat_i, cat_o, lo, hi in mismatch_ranges:
        if cat_i == inner_cat and cat_o == outer_cat:
            opt_range = (lo, hi)
            break

    in_range = opt_range and (opt_range[0] <= delta <= opt_range[1])

    # Compute LJ interaction energy at equilibrium distance
    lj_key = f"{inner_elem}-{outer_elem}"
    lj_key_rev = f"{outer_elem}-{inner_elem}"
    lj_sigma = None
    for pair, eps, sig in lj_parameters:
        if pair == lj_key or pair == lj_key_rev:
            lj_sigma = sig
            break
    if lj_sigma is None:
        lj_sigma = (atomic_radii[inner_elem] * 2 + atomic_radii[outer_elem] * 2) / 2

    eq_distance = lj_sigma * 2**(1/6)
    lj_energy = lennard_jones_potential(eq_distance, 1.0, lj_sigma)

    pred = {
        'cluster': cluster_name,
        'inner_element': inner_elem,
        'outer_element': outer_elem,
        'inner_category': inner_cat,
        'outer_category': outer_cat,
        'r_inner_A': r_inner,
        'r_outer_A': r_outer,
        'size_mismatch': round(delta, 4),
        'optimal_range': list(opt_range) if opt_range else None,
        'within_optimal_range': bool(in_range),
        'lj_sigma': round(lj_sigma, 3),
        'lj_equilibrium_distance': round(eq_distance, 3),
        'lj_min_energy': round(lj_energy, 4)
    }
    cluster_predictions.append(pred)
    print(f"  {cluster_name}: δ={delta:.4f}, opt_range={opt_range}, "
          f"in_range={in_range}, E_LJ={lj_energy:.4f}")

# Additional predicted clusters based on atomic radius analysis
print("\n  Additional predicted clusters:")
additional_predictions = []
element_list = list(atomic_radii.keys())
for i, elem_inner in enumerate(element_list):
    for elem_outer in element_list[i+1:]:
        r_in = atomic_radii[elem_inner]
        r_out = atomic_radii[elem_outer]
        delta = compute_size_mismatch(r_in, r_out)

        # Check against all mismatch ranges
        best_match = None
        for cat_i, cat_o, lo, hi in mismatch_ranges:
            if lo <= delta <= hi:
                best_match = f"{cat_i}->{cat_o}"
                break

        if best_match:
            inner_n = mackay_sequence[0]  # 13 atoms for first shell
            # Estimate outer shell atom count from magic numbers
            outer_n = mackay_sequence[1]  # 32 atoms for second shell
            cluster_name = f"{elem_inner}{inner_n}@{elem_outer}{outer_n}"
            add_pred = {
                'cluster': cluster_name,
                'inner_element': elem_inner,
                'outer_element': elem_outer,
                'size_mismatch': round(delta, 4),
                'predicted_category': best_match
            }
            additional_predictions.append(add_pred)
            cluster_predictions.append(add_pred)
            print(f"    {cluster_name}: δ={delta:.4f}, category={best_match}")

with open("outputs/cluster_predictions.json", "w") as f:
    json.dump(cluster_predictions, f, indent=2)

# --- 3e. Growth simulation analysis ---
print("\n[5] Analyzing growth simulation dynamics...")
# Separate growth results by trajectory
trajectories = {}
traj_id = 0
current_traj = []
for steps, cat, mm in growth_results:
    if steps == 0 and current_traj:
        trajectories[f"traj_{traj_id}"] = current_traj
        traj_id += 1
        current_traj = []
    current_traj.append({'steps': steps, 'category': cat, 'mismatch': mm})
if current_traj:
    trajectories[f"traj_{traj_id}"] = current_traj

for tid, traj in trajectories.items():
    final_mm = traj[-1]['mismatch'] if traj else 0
    final_cat = traj[-1]['category'] if traj else 'N/A'
    print(f"  {tid}: {len(traj)} steps, final_cat={final_cat}, "
          f"final_mismatch={final_mm:.4f}")

# Compute convergence metrics
convergence_data = {}
for tid, traj in trajectories.items():
    mismatches = [t['mismatch'] for t in traj]
    if len(mismatches) > 1:
        convergence_data[tid] = {
            'initial_mismatch': mismatches[0],
            'final_mismatch': mismatches[-1],
            'mean_mismatch': round(np.mean(mismatches), 4),
            'std_mismatch': round(np.std(mismatches), 4),
            'convergence_rate': round(
                abs(mismatches[-1] - mismatches[0]) / (len(mismatches) - 1), 4
            ) if len(mismatches) > 1 else 0
        }

with open("outputs/growth_dynamics.json", "w") as f:
    json.dump({
        'trajectories': {k: v for k, v in trajectories.items()},
        'convergence': convergence_data
    }, f, indent=2)

# --- 3f. Path selection analysis ---
print("\n[6] Analyzing path selection statistics...")
total_paths = sum(count for _, count in path_selection_stats)
path_analysis = []
for name, count in path_selection_stats:
    pct = count / total_paths * 100
    path_analysis.append({
        'path_type': name,
        'count': count,
        'percentage': round(pct, 1)
    })
    print(f"  {name}: {count} ({pct:.1f}%)")

with open("outputs/path_selection_analysis.json", "w") as f:
    json.dump(path_analysis, f, indent=2)

# --- 3g. Magic number sequence comparison ---
print("\n[7] Comparing magic number sequences...")
seq_comparison = []
max_shells = min(len(mackay_sequence), len(new_sequence_b5))
for i in range(max_shells):
    seq_comparison.append({
        'shell_index': i,
        'mackay': mackay_sequence[i],
        'new_b5': new_sequence_b5[i],
        'difference': mackay_sequence[i] - new_sequence_b5[i]
    })
    print(f"  Shell {i}: Mackay={mackay_sequence[i]}, New(b=5)={new_sequence_b5[i]}, "
          f"diff={mackay_sequence[i] - new_sequence_b5[i]}")

with open("outputs/magic_number_comparison.json", "w") as f:
    json.dump(seq_comparison, f, indent=2)

# --- 3h. Hexagonal lattice geometry ---
print("\n[8] Computing hexagonal lattice geometry...")
lattice_geometry = []
for h, k in hexagonal_coords[:12]:  # First 12 coords
    x, y = hexagonal_to_cartesian(h, k)
    dist = np.sqrt(x**2 + y**2)
    lattice_geometry.append({
        'h': h, 'k': k, 'x': round(x, 4), 'y': round(y, 4),
        'distance_from_origin': round(dist, 4)
    })

with open("outputs/hexagonal_lattice_geometry.json", "w") as f:
    json.dump(lattice_geometry, f, indent=2)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE - All intermediate results saved to outputs/")
print("=" * 60)
