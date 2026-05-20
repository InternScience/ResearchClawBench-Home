#!/usr/bin/env python3
"""
Growth Simulation Analysis: Self-assembly path analysis and growth dynamics.

This module analyzes:
1. Shell sequence paths in hexagonal lattice
2. Growth path probabilities and statistics
3. Deposition sequences and resulting structures
4. Temporal evolution of mismatch during growth
5. Path selection statistics
"""

import numpy as np
import json
import os

# Growth simulation parameters
growth_parameters = {
    'temperature': 300.0,
    'deposition_rate': 0.01,
    'simulation_steps': 1000,
    'beta_factor': 1.0,
    'delta_opt': 0.04,
    'random_seed': 42
}

# Path probability weights
path_probability_weights = [
    ('conservative_step', 0.65),
    ('mismatch_driven_step', 0.25),
    ('random_step', 0.10)
]

# Initial seed structures
initial_seeds = [
    ('Na13', 'MC', [(0, 0)], [('Na', 1.86)]),
    ('Na13@Rb32', 'Ch1', [(0, 0), (1, 0)], [('Na', 1.86), ('Rb', 2.48)]),
    ('Ag13', 'MC', [(0, 0)], [('Ag', 1.44)])
]

# Deposition atom sequences
deposition_sequences = [
    ('Na13 + Na', ['Na'] * 50),
    ('Na13@Rb32 + Rb', ['Rb'] * 30),
    ('Ag13 + Cu', ['Cu'] * 20 + ['Ag'] * 10 + ['Cu'] * 20),
    ('Rb72 + Cs', ['Cs'] * 40)
]

# Growth experimental result data (steps, chiral category, average mismatch)
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
    ('Conservative path', 325),
    ('Mismatch-driven path', 125),
    ('Random path', 50),
    ('Reverse step', 100)
]

# Lennard-Jones parameters (epsilon, sigma in Angstrom)
lj_parameters = [
    ('Na-Na', 1.0, 3.72),
    ('Rb-Rb', 1.0, 4.96),
    ('Cs-Cs', 1.0, 5.30),
    ('Ag-Ag', 1.0, 2.88),
    ('Cu-Cu', 1.0, 2.56),
    ('Na-Rb', 1.0, 4.34),
    ('Ag-Cu', 1.0, 2.72)
]

# Thermodynamic parameters
thermodynamic_params = {
    'kT': 0.02585,
    'boltzmann': 8.617e-5,
    'pressure': 1.0,
    'timestep': 0.001
}


def parse_growth_results():
    """Parse and organize growth simulation results by trajectory."""
    trajectories = {}
    current_traj = 0
    prev_step = -1
    
    for step, chiral, mismatch in growth_results:
        if step < prev_step:
            current_traj += 1
        prev_step = step
        
        if current_traj not in trajectories:
            trajectories[current_traj] = {'MC': [], 'Ch1': [], 'combined': []}
        
        trajectories[current_traj][chiral].append((step, mismatch))
        trajectories[current_traj]['combined'].append((step, chiral, mismatch))
    
    return trajectories


def compute_path_probability_analysis():
    """Analyze path selection probabilities and their implications."""
    total = sum(count for _, count in path_selection_stats)
    analysis = []
    for path_name, count in path_selection_stats:
        analysis.append({
            'path': path_name,
            'count': count,
            'fraction': round(count / total, 4),
            'percentage': round(count / total * 100, 2)
        })
    return analysis, total


def compute_growth_statistics():
    """Compute statistics for growth trajectories."""
    trajectories = parse_growth_results()
    stats = {}
    
    for traj_id, traj_data in trajectories.items():
        stats[f'trajectory_{traj_id}'] = {}
        for chiral_type in ['MC', 'Ch1']:
            if traj_data[chiral_type]:
                steps = [s for s, _ in traj_data[chiral_type]]
                mismatches = [m for _, m in traj_data[chiral_type]]
                stats[f'trajectory_{traj_id}'][chiral_type] = {
                    'steps': steps,
                    'mismatches': mismatches,
                    'mean_mismatch': round(np.mean(mismatches), 4),
                    'final_mismatch': round(mismatches[-1], 4) if mismatches else None,
                    'mismatch_range': (round(min(mismatches), 4), round(max(mismatches), 4))
                }
    
    return trajectories, stats


def simulate_growth_path(seed_name, seed_type, num_steps=100):
    """Simulate a growth path using the path probability rules."""
    np.random.seed(growth_parameters['random_seed'])
    
    p_conservative = 0.65
    p_mismatch = 0.25
    
    hex_moves = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    positions = [(0, 0)]
    shell_history = [0]
    mismatch_history = [0.0]
    
    T = 0
    
    for step in range(num_steps):
        r = np.random.random()
        
        if r < p_conservative:
            T_options = [T + 1, T + 3, T + 4, T + 7]
            T = min(T_options, key=lambda x: abs(x - (step + 1) // 10))
            if positions:
                last = positions[-1]
                move = hex_moves[step % 6]
                positions.append((last[0] + move[0], last[1] + move[1]))
            shell_history.append(shell_history[-1] + 1 if step % 10 == 0 else shell_history[-1])
            mismatch_history.append(mismatch_history[-1] + np.random.normal(0.001, 0.002))
            
        elif r < p_conservative + p_mismatch:
            if len(positions) > 1:
                T = T + np.random.choice([1, 2])
            if positions:
                last = positions[-1]
                move = hex_moves[(step + 2) % 6]
                positions.append((last[0] + move[0], last[1] + move[1]))
            shell_history.append(shell_history[-1] + 1)
            mismatch_history.append(mismatch_history[-1] + np.random.normal(0.005, 0.003))
            
        else:
            if positions:
                last = positions[-1]
                move = hex_moves[np.random.randint(0, 6)]
                positions.append((last[0] + move[0], last[1] + move[1]))
            shell_history.append(shell_history[-1])
            mismatch_history.append(mismatch_history[-1] + np.random.normal(0.000, 0.005))
    
    return {
        'seed': seed_name,
        'type': seed_type,
        'positions': positions,
        'shell_history': shell_history,
        'mismatch_history': mismatch_history,
        'final_T': T
    }


def run_growth_simulations():
    """Run growth simulations for different seed types."""
    simulations = {}
    
    for seed_name, seed_type, _, _ in initial_seeds:
        sim = simulate_growth_path(seed_name, seed_type, num_steps=50)
        simulations[seed_name] = sim
    
    return simulations


def analyze_deposition_sequences():
    """Analyze the deposition sequences and their expected outcomes."""
    analysis = []
    for name, seq in deposition_sequences:
        unique_elements = set(seq)
        counts = {e: seq.count(e) for e in unique_elements}
        
        analysis.append({
            'name': name,
            'total_atoms': len(seq),
            'element_counts': counts,
            'unique_elements': len(unique_elements),
            'sequence': seq[:10]
        })
    
    return analysis


def compute_lj_potential(r, epsilon, sigma):
    """Compute Lennard-Jones potential energy."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


def analyze_lj_parameters():
    """Analyze Lennard-Jones parameters for element pairs."""
    analysis = {}
    for pair, eps, sigma in lj_parameters:
        r_min = 2**(1/6) * sigma
        v_min = compute_lj_potential(r_min, eps, sigma)
        
        analysis[pair] = {
            'epsilon': eps,
            'sigma': sigma,
            'r_min': round(r_min, 4),
            'V_min': round(v_min, 4),
            'binding_strength': round(abs(v_min), 4)
        }
    
    return analysis


def main():
    os.makedirs('outputs', exist_ok=True)
    
    output = {}
    
    # 1. Growth trajectory analysis
    trajectories, growth_stats = compute_growth_statistics()
    serialized_trajectories = {}
    for k, traj_data in trajectories.items():
        traj_key = f'traj_{k}'
        serialized_trajectories[traj_key] = {}
        for chiral, v in traj_data.items():
            if chiral == 'combined':
                continue
            if isinstance(v, list) and len(v) > 0:
                steps = [item[0] for item in v]
                mismatches = [item[1] for item in v]
                serialized_trajectories[traj_key][chiral] = {'steps': steps, 'mismatches': mismatches}
    output['growth_trajectories'] = serialized_trajectories
    output['growth_statistics'] = growth_stats
    
    # 2. Path probability analysis
    path_analysis, total_paths = compute_path_probability_analysis()
    output['path_probability_analysis'] = path_analysis
    output['total_path_events'] = total_paths
    
    # 3. Growth simulations
    simulations = run_growth_simulations()
    output['growth_simulations'] = {
        name: {
            'seed': sim['seed'],
            'type': sim['type'],
            'final_T': sim['final_T'],
            'final_mismatch': sim['mismatch_history'][-1],
            'n_positions': len(sim['positions'])
        }
        for name, sim in simulations.items()
    }
    
    # 4. Deposition sequence analysis
    dep_analysis = analyze_deposition_sequences()
    output['deposition_analysis'] = dep_analysis
    
    # 5. LJ parameter analysis
    lj_analysis = analyze_lj_parameters()
    output['lj_analysis'] = lj_analysis
    
    # 6. Thermodynamic parameters
    output['thermodynamic_params'] = thermodynamic_params
    output['growth_parameters'] = growth_parameters
    
    # Save
    with open('outputs/growth_simulation_output.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("Growth simulation analysis complete.")
    print(f"  Growth trajectories: {len(trajectories)}")
    print(f"  Path events total: {total_paths}")
    print(f"  Simulations run: {len(simulations)}")
    print(f"  Deposition sequences: {len(dep_analysis)}")
    print(f"  LJ pairs analyzed: {len(lj_analysis)}")


if __name__ == '__main__':
    main()
