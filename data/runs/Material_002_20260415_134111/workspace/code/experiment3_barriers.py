"""
MACE-MP-0 Foundation Model Reproduction
Experiment 3: CRBH20 Reaction Barriers
"""
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from ase import Atoms
from mace.calculators import MACECalculator

# ---- Parameters from dataset ----
# DFT reference barriers (eV)
dft_barriers = {
    'Rxn 1': 1.72,   # cyclobutene ring-opening
    'Rxn 11': 1.74,  # methoxy decomposition
    'Rxn 20': 1.77,  # cyclopropane ring-opening
}

# ---- Build geometries ----
# Reaction 1: cyclobutene ring-opening (C4H4)
rxn1_reactant = Atoms(symbols=['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
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
    cell=[15, 15, 15], pbc=True)

rxn1_ts = Atoms(symbols=['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
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
    cell=[15, 15, 15], pbc=True)

# Reaction 2: methoxy decomposition (CH3O)
rxn2_reactant = Atoms(symbols=['C', 'H', 'H', 'H', 'O'],
    positions=[
        [0.000, 0.000, 0.000],
        [0.000, 1.000, 0.000],
        [0.900, -0.500, 0.000],
        [-0.900, -0.500, 0.000],
        [1.200, 0.000, 0.000],
    ],
    cell=[15, 15, 15], pbc=True)

rxn2_ts = Atoms(symbols=['C', 'H', 'H', 'H', 'O'],
    positions=[
        [0.000, 0.000, 0.000],
        [0.000, 1.000, 0.000],
        [0.900, -0.500, 0.000],
        [-0.900, -0.500, 0.000],
        [1.500, 0.000, 0.000],
    ],
    cell=[15, 15, 15], pbc=True)

# Reaction 3: cyclopropane ring-opening (C3H6)
rxn3_reactant = Atoms(symbols=['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
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
    cell=[15, 15, 15], pbc=True)

rxn3_ts = Atoms(symbols=['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
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
    cell=[15, 15, 15], pbc=True)

# ---- Load MACE calculator ----
calc = MACECalculator(model_paths='mace_mp_0.model', device='cpu')

# ---- Compute energies ----
reactions = {
    'Rxn 1': {'reactant': rxn1_reactant, 'ts': rxn1_ts, 'name': 'Cyclobutene ring-opening'},
    'Rxn 11': {'reactant': rxn2_reactant, 'ts': rxn2_ts, 'name': 'Methoxy decomposition'},
    'Rxn 20': {'reactant': rxn3_reactant, 'ts': rxn3_ts, 'name': 'Cyclopropane ring-opening'},
}

results = {}
for rxn_id, rxn_data in reactions.items():
    print(f"\n{rxn_id}: {rxn_data['name']}")
    
    # Reactant energy
    reactant = rxn_data['reactant']
    reactant.calc = calc
    E_reactant = reactant.get_potential_energy()
    print(f"  Reactant energy: {E_reactant:.4f} eV")
    
    # TS energy
    ts = rxn_data['ts']
    ts.calc = calc
    E_ts = ts.get_potential_energy()
    print(f"  TS energy: {E_ts:.4f} eV")
    
    # Barrier
    barrier = E_ts - E_reactant
    dft_ref = dft_barriers[rxn_id]
    error = barrier - dft_ref
    
    print(f"  MACE barrier: {barrier:.4f} eV")
    print(f"  DFT reference: {dft_ref:.2f} eV")
    print(f"  Error: {error:.4f} eV")
    
    results[rxn_id] = {
        'name': rxn_data['name'],
        'E_reactant': E_reactant,
        'E_ts': E_ts,
        'barrier_mace': barrier,
        'barrier_dft': dft_ref,
        'error': error,
    }

# Save results
with open('outputs/reaction_barriers.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nReaction barrier results saved to outputs/reaction_barriers.json")
print("\nSummary:")
print(f"{'Reaction':>8} {'MACE (eV)':>12} {'DFT (eV)':>12} {'Error (eV)':>12}")
for rxn_id, r in results.items():
    print(f"{rxn_id:>8} {r['barrier_mace']:>12.4f} {r['barrier_dft']:>12.2f} {r['error']:>12.4f}")

mae = np.mean([abs(r['error']) for r in results.values()])
print(f"\nMAE: {mae:.4f} eV")
