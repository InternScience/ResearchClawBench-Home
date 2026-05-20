#!/usr/bin/env python3
"""
Experiment 3: Reaction Barrier Comparison (CRBH20)
Uses MACE-MP-0 to compute energy barriers for three reactions:
1. Cyclobutene ring-opening (Rxn 1)
2. Methoxy decomposition (Rxn 11)
3. Cyclopropane ring-opening (Rxn 20)
"""

import os
import json
import time
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import mace_mp

# ── Reference DFT barriers (eV) ──────────────────────────────────
DFT_BARRIERS = {
    'Rxn_1': 1.72,
    'Rxn_11': 1.74,
    'Rxn_20': 1.77,
}

# ── Reaction coordinates ─────────────────────────────────────────

# Reaction 1: Cyclobutene ring-opening
RXN1_REACTANT = Atoms(
    symbols=['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 0.000, 0.000],
        [1.500, 0.000, 0.000],
        [1.500, 1.500, 0.000],
        [0.000, 1.500, 0.000],
        [-0.500, -0.500, 0.000],
        [2.000, -0.500, 0.000],
        [2.000, 2.000, 0.000],
        [-0.500, 2.000, 0.000],
    ],
    cell=[10, 10, 10],
    pbc=[True, True, True],
)

RXN1_TS = Atoms(
    symbols=['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 0.000, 0.000],
        [1.400, 0.200, 0.000],
        [1.400, 1.300, 0.000],
        [0.000, 1.500, 0.000],
        [-0.500, -0.500, 0.000],
        [1.900, -0.300, 0.000],
        [1.900, 1.800, 0.000],
        [-0.500, 2.000, 0.000],
    ],
    cell=[10, 10, 10],
    pbc=[True, True, True],
)

# Reaction 2: Methoxy decomposition
RXN11_REACTANT = Atoms(
    symbols=['C', 'H', 'H', 'H', 'O'],
    positions=[
        [0.000, 0.000, 0.000],
        [0.000, 1.000, 0.000],
        [0.900, -0.500, 0.000],
        [-0.900, -0.500, 0.000],
        [1.200, 0.000, 0.000],
    ],
    cell=[10, 10, 10],
    pbc=[True, True, True],
)

RXN11_TS = Atoms(
    symbols=['C', 'H', 'H', 'H', 'O'],
    positions=[
        [0.000, 0.000, 0.000],
        [0.000, 1.000, 0.000],
        [0.900, -0.500, 0.000],
        [-0.900, -0.500, 0.000],
        [1.500, 0.000, 0.000],
    ],
    cell=[10, 10, 10],
    pbc=[True, True, True],
)

# Reaction 3: Cyclopropane ring-opening
RXN20_REACTANT = Atoms(
    symbols=['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 0.000, 0.000],
        [1.500, 0.000, 0.000],
        [0.750, 1.300, 0.000],
        [-0.500, -0.500, 0.000],
        [2.000, -0.500, 0.000],
        [0.750, 2.000, 0.000],
        [0.000, 0.000, 1.000],
        [1.500, 0.000, 1.000],
        [0.750, 1.300, 1.000],
    ],
    cell=[10, 10, 10],
    pbc=[True, True, True],
)

RXN20_TS = Atoms(
    symbols=['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 0.000, 0.000],
        [1.500, 0.000, 0.000],
        [0.750, 1.300, 0.000],
        [-0.500, -0.500, 0.000],
        [2.000, -0.500, 0.000],
        [0.750, 2.000, 0.000],
        [0.000, 0.000, 1.500],
        [1.500, 0.000, 1.500],
        [0.750, 1.300, 1.500],
    ],
    cell=[10, 10, 10],
    pbc=[True, True, True],
)

REACTIONS = {
    'Rxn_1': {'name': 'Cyclobutene ring-opening', 'reactant': RXN1_REACTANT, 'ts': RXN1_TS},
    'Rxn_11': {'name': 'Methoxy decomposition', 'reactant': RXN11_REACTANT, 'ts': RXN11_TS},
    'Rxn_20': {'name': 'Cyclopropane ring-opening', 'reactant': RXN20_REACTANT, 'ts': RXN20_TS},
}


def relax_structure(atoms, calc, fmax=0.05, steps=200):
    """Relax structure with BFGS."""
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    opt = BFGS(atoms_copy)
    opt.run(fmax=fmax, steps=steps)
    return atoms_copy.get_potential_energy()


def main():
    print("=" * 60)
    print("Experiment 3: Reaction Barrier Comparison (CRBH20)")
    print("=" * 60)
    
    t_start = time.time()
    
    # Load MACE model
    print("Loading MACE-MP-0 model...")
    calc = mace_mp(model='medium', device='cpu', default_dtype='float64')
    print(f"Model loaded in {time.time()-t_start:.1f}s")
    
    results = {'reactions': {}, 'dft_barriers': DFT_BARRIERS}
    
    for rxn_id, rxn_data in REACTIONS.items():
        print(f"\n--- {rxn_id}: {rxn_data['name']} ---")
        t_rxn = time.time()
        
        # Single-point energies (no relaxation for TS - it's already at saddle point)
        reactant = rxn_data['reactant'].copy()
        ts = rxn_data['ts'].copy()
        
        reactant.calc = calc
        ts.calc = calc
        
        e_reactant = reactant.get_potential_energy()
        e_ts = ts.get_potential_energy()
        
        barrier = e_ts - e_reactant
        
        dft_barrier = DFT_BARRIERS[rxn_id]
        error = barrier - dft_barrier
        
        print(f"  E_reactant = {e_reactant:.3f} eV")
        print(f"  E_TS = {e_ts:.3f} eV")
        print(f"  MACE barrier = {barrier:.3f} eV")
        print(f"  DFT barrier = {dft_barrier:.3f} eV")
        print(f"  Error = {error:.3f} eV")
        
        results['reactions'][rxn_id] = {
            'name': rxn_data['name'],
            'E_reactant': float(e_reactant),
            'E_TS': float(e_ts),
            'barrier_MACE': float(barrier),
            'barrier_DFT': float(dft_barrier),
            'error': float(error),
            'abs_error': float(abs(error)),
        }
        
        print(f"  Done in {time.time()-t_rxn:.1f}s")
    
    # Summary statistics
    errors = [abs(results['reactions'][r]['error']) for r in REACTIONS]
    results['summary'] = {
        'MAE': float(np.mean(errors)),
        'RMSE': float(np.sqrt(np.mean(np.array(errors)**2))),
        'Max_error': float(np.max(errors)),
    }
    
    print(f"\nSummary: MAE = {results['summary']['MAE']:.3f} eV, "
          f"RMSE = {results['summary']['RMSE']:.3f} eV")
    
    os.makedirs('../outputs', exist_ok=True)
    with open('../outputs/reaction_barriers_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to outputs/reaction_barriers_results.json")
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
