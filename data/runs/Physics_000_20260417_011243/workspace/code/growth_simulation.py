#!/usr/bin/env python3
"""
Dynamic Growth Simulation for Multi-component Icosahedral Shells
Implements Monte Carlo-style growth simulation with path selection.
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from core_theory import (
    ATOMIC_RADII, LJ_PARAMETERS, size_mismatch, lj_potential,
    lj_equilibrium_distance, classify_shell_path, SHELL_COLORS,
    CHIRAL_LABELS, OPTIMAL_MISMATCH_RANGES
)

# ============================================================
# Growth Simulation Parameters
# ============================================================

GROWTH_PARAMS = {
    'temperature': 300.0,
    'deposition_rate': 0.01,
    'simulation_steps': 1000,
    'beta_factor': 1.0,
    'delta_opt': 0.04,
    'random_seed': 42,
    'kT': 0.02585,  # eV at 300K
    'boltzmann': 8.617e-5,  # eV/K
    'pressure': 1.0,
    'timestep': 0.001
}

PATH_PROBABILITY_WEIGHTS = {
    'conservative_step': 0.65,
    'mismatch_driven_step': 0.25,
    'random_step': 0.10
}

# ============================================================
# Shell Growth State
# ============================================================

class ShellGrowthState:
    """Represents the state of a growing multi-shell icosahedral cluster."""
    
    def __init__(self, seed_name, seed_type, seed_path, seed_elements):
        self.name = seed_name
        self.shell_type = seed_type
        self.path = list(seed_path)  # list of (h,k) tuples
        self.elements = list(seed_elements)  # list of (element, radius) tuples
        self.energy = 0.0
        self.mismatch_history = []
        self.type_history = [seed_type]
        self.step_history = []
        
    def compute_current_mismatch(self):
        """Compute current size mismatch between last two shells."""
        if len(self.elements) < 2:
            return 0.0
        r1 = self.elements[-2][1]
        r2 = self.elements[-1][1]
        return abs(r1 - r2) / max(r1, r2)
    
    def add_shell(self, element, radius, h, k):
        """Add a new shell to the cluster."""
        self.elements.append((element, radius))
        self.path.append((h, k))
        shell_type = classify_shell_path(h, k)
        self.shell_type = shell_type
        self.type_history.append(shell_type)
        sm = self.compute_current_mismatch()
        self.mismatch_history.append(sm)

# ============================================================
# Growth Simulation Engine
# ============================================================

class GrowthSimulation:
    """Monte Carlo growth simulation for multi-shell icosahedral clusters."""
    
    def __init__(self, seed, deposition_sequence, params=None):
        self.seed = seed
        self.deposition_sequence = deposition_sequence
        self.params = params or GROWTH_PARAMS
        self.rng = np.random.RandomState(self.params['random_seed'])
        
        # Initialize state
        self.state = ShellGrowthState(
            seed_name=seed['name'],
            seed_type=seed['type'],
            seed_path=seed['path'],
            seed_elements=seed['elements']
        )
        
        # Statistics
        self.path_stats = {
            'Conservative path': 0,
            'Mismatch-driven path': 0,
            'Random path': 0,
            'Reverse step': 0
        }
        
        self.growth_trajectory = []
        self.energy_trajectory = []
    
    def select_path_step(self, current_h, current_k, deposited_element):
        """Select next step on hexagonal lattice based on probability weights."""
        r = self.rng.random()
        
        deposited_radius = ATOMIC_RADII.get(deposited_element, 1.5)
        current_radius = self.state.elements[-1][1] if self.state.elements else 1.5
        current_sm = abs(current_radius - deposited_radius) / max(current_radius, deposited_radius)
        
        # Possible next positions on hexagonal lattice
        neighbors = [
            (current_h + 1, current_k),     # Forward in h
            (current_h, current_k + 1),     # Forward in k
            (current_h + 1, current_k + 1), # Diagonal
        ]
        if current_h > 0:
            neighbors.append((current_h - 1, current_k))
        if current_k > 0:
            neighbors.append((current_h, current_k - 1))
        neighbors = [(h, k) for h, k in neighbors if 0 <= h <= 5 and 0 <= k <= 5]
        
        if not neighbors:
            return current_h, current_k, 'Conservative path'
        
        if r < PATH_PROBABILITY_WEIGHTS['conservative_step']:
            # Conservative: prefer MC path (h+1, k) or (h, k+1)
            mc_neighbors = [(h, k) for h, k in neighbors if h == 0 or k == 0 or h == k]
            if mc_neighbors:
                choice = mc_neighbors[self.rng.randint(len(mc_neighbors))]
            else:
                choice = neighbors[0]
            path_type = 'Conservative path'
            
        elif r < PATH_PROBABILITY_WEIGHTS['conservative_step'] + PATH_PROBABILITY_WEIGHTS['mismatch_driven_step']:
            # Mismatch-driven: choose neighbor that best matches optimal mismatch
            best_score = float('inf')
            best_neighbor = neighbors[0]
            for nh, nk in neighbors:
                shell_type = classify_shell_path(nh, nk)
                current_type = self.state.shell_type
                key = (current_type, shell_type)
                if key in OPTIMAL_MISMATCH_RANGES:
                    opt_low, opt_high = OPTIMAL_MISMATCH_RANGES[key]
                    opt_mid = (opt_low + opt_high) / 2
                    score = abs(current_sm - opt_mid)
                else:
                    score = abs(current_sm - 0.04)  # Default optimal
                if score < best_score:
                    best_score = score
                    best_neighbor = (nh, nk)
            choice = best_neighbor
            path_type = 'Mismatch-driven path'
            
        else:
            # Random step
            choice = neighbors[self.rng.randint(len(neighbors))]
            path_type = 'Random path'
        
        # Check for reverse step
        if len(self.state.path) >= 2:
            prev = self.state.path[-2]
            if choice == prev:
                path_type = 'Reverse step'
        
        return choice[0], choice[1], path_type
    
    def compute_step_energy(self, element, h, k):
        """Compute energy change for adding a shell."""
        shell_type = classify_shell_path(h, k)
        
        # Base energy from shell type
        type_energies = {'MC': -2.35, 'BG': -2.15, 'Ch1': -2.20, 'Ch2': -2.00, 'Ch3': -1.90, 'Ch4': -1.80, 'Ch5': -1.70}
        base_energy = type_energies.get(shell_type, -1.5)
        
        # LJ contribution
        if self.state.elements:
            prev_element = self.state.elements[-1][0]
            pair_key = f"{prev_element}-{element}"
            alt_key = f"{element}-{prev_element}"
            if pair_key in LJ_PARAMETERS:
                eps, sig = LJ_PARAMETERS[pair_key]
            elif alt_key in LJ_PARAMETERS:
                eps, sig = LJ_PARAMETERS[alt_key]
            else:
                eps, sig = 1.0, (ATOMIC_RADII.get(prev_element, 1.5) + ATOMIC_RADII.get(element, 1.5))
            r_eq = lj_equilibrium_distance(sig)
            lj_energy = lj_potential(r_eq, eps, sig)
            base_energy += lj_energy * 0.1  # Scale factor
        
        # Mismatch penalty
        if self.state.elements:
            current_radius = self.state.elements[-1][1]
            new_radius = ATOMIC_RADII.get(element, 1.5)
            sm = abs(current_radius - new_radius) / max(current_radius, new_radius)
            # Penalty for being far from optimal
            mismatch_penalty = (sm - self.params['delta_opt'])**2 * 10
            base_energy += mismatch_penalty
        
        return base_energy
    
    def run(self, max_steps=None):
        """Run the growth simulation."""
        if max_steps is None:
            max_steps = len(self.deposition_sequence)
        
        current_h = self.state.path[-1][0] if self.state.path else 0
        current_k = self.state.path[-1][1] if self.state.path else 0
        
        for step_idx in range(min(max_steps, len(self.deposition_sequence))):
            element = self.deposition_sequence[step_idx]
            radius = ATOMIC_RADII.get(element, 1.5)
            
            # Select path step
            new_h, new_k, path_type = self.select_path_step(current_h, current_k, element)
            
            # Compute energy
            energy_change = self.compute_step_energy(element, new_h, new_k)
            
            # Accept/reject based on Metropolis criterion
            if energy_change < 0:
                accept = True
            else:
                beta = 1.0 / self.params['kT']
                accept = self.rng.random() < np.exp(-beta * energy_change)
            
            if accept:
                self.state.add_shell(element, radius, new_h, new_k)
                self.state.energy += energy_change
                self.path_stats[path_type] += 1
                current_h, current_k = new_h, new_k
            
            # Record trajectory
            sm = self.state.compute_current_mismatch()
            self.growth_trajectory.append({
                'step': step_idx,
                'element': element,
                'h': current_h,
                'k': current_k,
                'shell_type': self.state.shell_type,
                'mismatch': sm,
                'energy': self.state.energy,
                'accepted': accept,
                'path_type': path_type
            })
            self.energy_trajectory.append(self.state.energy)
        
        return self.state, self.growth_trajectory, self.path_stats

# ============================================================
# Run all simulations from data
# ============================================================

def run_all_simulations():
    """Run all growth simulations defined in the data."""
    
    # Define initial seeds
    seeds = [
        {'name': 'Na13', 'type': 'MC', 'path': [(0,0)], 'elements': [('Na', 1.86)]},
        {'name': 'Na13@Rb32', 'type': 'Ch1', 'path': [(0,0), (1,0)], 'elements': [('Na', 1.86), ('Rb', 2.48)]},
        {'name': 'Ag13', 'type': 'MC', 'path': [(0,0)], 'elements': [('Ag', 1.44)]}
    ]
    
    # Define deposition sequences
    deposition_sequences = [
        ('Na13 + Na', ['Na']*50),
        ('Na13@Rb32 + Rb', ['Rb']*30),
        ('Ag13 + Cu', ['Cu']*20 + ['Ag']*10 + ['Cu']*20),
        ('Rb72 + Cs', ['Cs']*40)
    ]
    
    # Map seeds to deposition sequences
    sim_configs = [
        (seeds[0], deposition_sequences[0]),
        (seeds[1], deposition_sequences[1]),
        (seeds[2], deposition_sequences[2]),
    ]
    
    all_results = {}
    
    for seed, (seq_name, dep_seq) in sim_configs:
        print(f"\nRunning simulation: {seq_name}")
        sim = GrowthSimulation(seed, dep_seq)
        final_state, trajectory, path_stats = sim.run()
        
        all_results[seq_name] = {
            'final_name': final_state.name,
            'final_type': final_state.shell_type,
            'final_energy': final_state.energy,
            'n_shells': len(final_state.elements),
            'type_history': final_state.type_history,
            'mismatch_history': final_state.mismatch_history,
            'path_stats': path_stats,
            'trajectory': trajectory,
            'energy_trajectory': sim.energy_trajectory
        }
        
        print(f"  Final state: {final_state.name}")
        print(f"  Shell type: {final_state.shell_type}")
        print(f"  Energy: {final_state.energy:.4f}")
        print(f"  Shells: {len(final_state.elements)}")
        print(f"  Path stats: {path_stats}")
    
    return all_results

if __name__ == '__main__':
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_all_simulations()
    
    # Save results (convert trajectory to summary for JSON serialization)
    save_results = {}
    for name, data in results.items():
        save_results[name] = {
            'final_name': data['final_name'],
            'final_type': data['final_type'],
            'final_energy': data['final_energy'],
            'n_shells': data['n_shells'],
            'type_history': data['type_history'],
            'mismatch_history': [round(m, 4) for m in data['mismatch_history']],
            'path_stats': data['path_stats'],
            'energy_trajectory': [round(e, 4) for e in data['energy_trajectory'][:50]]  # First 50 steps
        }
    
    with open(os.path.join(output_dir, 'growth_simulation_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print("\nGrowth simulation results saved.")
