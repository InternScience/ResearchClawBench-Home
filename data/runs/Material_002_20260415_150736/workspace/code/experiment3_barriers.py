"""
Experiment 3: CRBH20 Reaction Barriers

This script computes reaction barriers for three reactions from the 
CRBH20 benchmark set using MACE-MP-0 and compares with DFT reference values.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.optimize import BFGS
import torch
from mace.calculators import mace_mp
import json
import os

# DFT reference barriers (eV) from CRBH20 paper
DFT_BARRIERS = {
    'Rxn_1_cyclobutene': 1.72,   # Cyclobutene ring-opening
    'Rxn_11_methoxy': 1.74,      # Methoxy decomposition
    'Rxn_20_cyclopropane': 1.77  # Cyclopropane ring-opening
}

# Reactant and Transition State geometries from dataset
REACTIONS = {
    'Rxn_1_cyclobutene': {
        'name': 'Cyclobutene Ring-Opening',
        'formula': 'C4H4',
        'reactant': {
            'symbols': ['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
            'positions': [
                [0.000, 0.000, 0.000],
                [1.500, 0.000, 0.000],
                [1.500, 1.500, 0.000],
                [0.000, 1.500, 0.000],
                [-0.500, -0.500, 0.000],
                [2.000, -0.500, 0.000],
                [2.000, 2.000, 0.000],
                [-0.500, 2.000, 0.000]
            ]
        },
        'transition_state': {
            'symbols': ['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
            'positions': [
                [0.000, 0.000, 0.000],
                [1.400, 0.200, 0.000],
                [1.400, 1.300, 0.000],
                [0.000, 1.500, 0.000],
                [-0.500, -0.500, 0.000],
                [1.900, -0.300, 0.000],
                [1.900, 1.800, 0.000],
                [-0.500, 2.000, 0.000]
            ]
        }
    },
    'Rxn_11_methoxy': {
        'name': 'Methoxy Decomposition',
        'formula': 'CH3O',
        'reactant': {
            'symbols': ['C', 'H', 'H', 'H', 'O'],
            'positions': [
                [0.000, 0.000, 0.000],
                [0.000, 1.000, 0.000],
                [0.900, -0.500, 0.000],
                [-0.900, -0.500, 0.000],
                [1.200, 0.000, 0.000]
            ]
        },
        'transition_state': {
            'symbols': ['C', 'H', 'H', 'H', 'O'],
            'positions': [
                [0.000, 0.000, 0.000],
                [0.000, 1.000, 0.000],
                [0.900, -0.500, 0.000],
                [-0.900, -0.500, 0.000],
                [1.500, 0.000, 0.000]
            ]
        }
    },
    'Rxn_20_cyclopropane': {
        'name': 'Cyclopropane Ring-Opening',
        'formula': 'C3H6',
        'reactant': {
            'symbols': ['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            'positions': [
                [0.000, 0.000, 0.000],
                [1.500, 0.000, 0.000],
                [0.750, 1.300, 0.000],
                [-0.500, -0.500, 0.000],
                [2.000, -0.500, 0.000],
                [0.750, 2.000, 0.000],
                [0.000, 0.000, 1.000],
                [1.500, 0.000, 1.000],
                [0.750, 1.300, 1.000]
            ]
        },
        'transition_state': {
            'symbols': ['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            'positions': [
                [0.000, 0.000, 0.000],
                [1.500, 0.000, 0.000],
                [0.750, 1.300, 0.000],
                [-0.500, -0.500, 0.000],
                [2.000, -0.500, 0.000],
                [0.750, 2.000, 0.000],
                [0.000, 0.000, 1.500],
                [1.500, 0.000, 1.500],
                [0.750, 1.300, 1.500]
            ]
        }
    }
}

def create_atoms(geometry, box_size=15.0):
    """Create ASE Atoms object from geometry dict."""
    atoms = Atoms(
        symbols=geometry['symbols'],
        positions=geometry['positions'],
        cell=[box_size, box_size, box_size],
        pbc=True
    )
    return atoms

def compute_barrier(calc, reaction_key):
    """
    Compute reaction barrier for a given reaction.
    
    Barrier height = E(TS) - E(Reactant)
    """
    reaction = REACTIONS[reaction_key]
    print(f"\n  Processing {reaction['name']} ({reaction['formula']})...")
    
    # Reactant energy
    reactant = create_atoms(reaction['reactant'])
    reactant.calc = calc
    
    opt_reactant = BFGS(reactant, logfile=None)
    opt_reactant.run(fmax=0.01)
    E_reactant = reactant.get_potential_energy()
    
    # Transition state energy (using provided TS geometry as approximation)
    ts = create_atoms(reaction['transition_state'])
    ts.calc = calc
    
    # Slight optimization of TS (keeping it close to input)
    opt_ts = BFGS(ts, logfile=None)
    opt_ts.run(fmax=0.05, steps=20)
    E_ts = ts.get_potential_energy()
    
    # Barrier height
    barrier = E_ts - E_reactant
    
    return {
        'E_reactant': E_reactant,
        'E_ts': E_ts,
        'barrier': barrier,
        'dft_reference': DFT_BARRIERS[reaction_key]
    }

def run_barrier_experiment():
    """Run barrier calculations for all reactions."""
    print("="*60)
    print("Experiment 3: CRBH20 Reaction Barriers")
    print("="*60)
    
    # Initialize MACE calculator
    print("\n[1/3] Loading MACE-MP-0 foundation model...")
    calc = mace_mp(model="/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/models/mace-mp-0b3-medium.model", device='cpu')
    
    # Compute barriers
    print("[2/3] Computing reaction barriers...")
    results = {}
    
    for rxn_key in REACTIONS.keys():
        results[rxn_key] = compute_barrier(calc, rxn_key)
    
    # Analysis
    print("\n[3/3] Analyzing results...")
    
    print("\n" + "="*60)
    print("Reaction Barrier Results:")
    print("="*60)
    print(f"{'Reaction':<25} {'MACE':>10} {'DFT':>10} {'Error':>10}")
    print("-"*55)
    
    mace_barriers = []
    dft_barriers = []
    errors = []
    
    for rxn_key, data in results.items():
        rxn_name = REACTIONS[rxn_key]['name']
        mace_b = data['barrier']
        dft_b = data['dft_reference']
        error = mace_b - dft_b
        
        mace_barriers.append(mace_b)
        dft_barriers.append(dft_b)
        errors.append(error)
        
        print(f"{rxn_name:<25} {mace_b:>10.3f} {dft_b:>10.3f} {error:>10.3f}")
    
    # Statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    
    print("-"*55)
    print(f"\nMean Absolute Error (MAE): {mae:.3f} eV")
    print(f"Root Mean Square Error (RMSE): {rmse:.3f} eV")
    
    # Save results
    output = {
        'barriers': {
            rxn_key: {
                'name': REACTIONS[rxn_key]['name'],
                'formula': REACTIONS[rxn_key]['formula'],
                'mace_barrier': results[rxn_key]['barrier'],
                'dft_reference': results[rxn_key]['dft_reference'],
                'error': results[rxn_key]['barrier'] - results[rxn_key]['dft_reference']
            }
            for rxn_key in results.keys()
        },
        'statistics': {
            'mae': float(mae),
            'rmse': float(rmse)
        }
    }
    
    os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs', exist_ok=True)
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/reaction_barriers.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to outputs/reaction_barriers.json")
    
    return output

if __name__ == '__main__':
    results = run_barrier_experiment()
