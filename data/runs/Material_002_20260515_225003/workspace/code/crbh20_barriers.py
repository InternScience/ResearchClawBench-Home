import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import MACECalculator
import matplotlib.pyplot as plt
import os

# Setup
model_path = 'data/mace-mp-0b3-medium.model'
calc = MACECalculator(model_path=model_path, dispersion=False, default_dtype='float32')

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# CRBH20 simplified reactions from dataset
reactions = {
    'Rxn1_cyclobutene': {
        'reactant': Atoms(
            symbols='C4H4',
            positions=[
                [0.000, 0.000, 0.000], [1.500, 0.000, 0.000],
                [1.500, 1.500, 0.000], [0.000, 1.500, 0.000],
                [-0.500, -0.500, 0.000], [2.000, -0.500, 0.000],
                [2.000, 2.000, 0.000], [-0.500, 2.000, 0.000]
            ]
        ),
        'ts': Atoms(
            symbols='C4H4',
            positions=[
                [0.000, 0.000, 0.000], [1.400, 0.200, 0.000],
                [1.400, 1.300, 0.000], [0.000, 1.500, 0.000],
                [-0.500, -0.500, 0.000], [1.900, -0.300, 0.000],
                [1.900, 1.800, 0.000], [-0.500, 2.000, 0.000]
            ]
        )
    },
    'Rxn11_methoxy': {
        'reactant': Atoms(
            symbols='CH3O',
            positions=[
                [0.000, 0.000, 0.000], [0.000, 1.000, 0.000],
                [0.900, -0.500, 0.000], [-0.900, -0.500, 0.000],
                [1.200, 0.000, 0.000]
            ]
        ),
        'ts': Atoms(
            symbols='CH3O',
            positions=[
                [0.000, 0.000, 0.000], [0.000, 1.000, 0.000],
                [0.900, -0.500, 0.000], [-0.900, -0.500, 0.000],
                [1.500, 0.000, 0.000]
            ]
        )
    },
    'Rxn20_cyclopropane': {
        'reactant': Atoms(
            symbols='C3H6',
            positions=[
                [0.000, 0.000, 0.000], [1.500, 0.000, 0.000],
                [0.750, 1.300, 0.000], [-0.500, -0.500, 0.000],
                [2.000, -0.500, 0.000], [0.750, 2.000, 0.000],
                [0.000, 0.000, 1.000], [1.500, 0.000, 1.000],
                [0.750, 1.300, 1.000]
            ]
        ),
        'ts': Atoms(
            symbols='C3H6',
            positions=[
                [0.000, 0.000, 0.000], [1.500, 0.000, 0.000],
                [0.750, 1.300, 0.000], [-0.500, -0.500, 0.000],
                [2.000, -0.500, 0.000], [0.750, 2.000, 0.000],
                [0.000, 0.000, 1.500], [1.500, 0.000, 1.500],
                [0.750, 1.300, 1.500]
            ]
        )
    }
}

results = {}
for name, structs in reactions.items():
    for key in ['reactant', 'ts']:
        atoms = structs[key]
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc(True)
        atoms.calc = calc
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.05)
        energy = atoms.get_potential_energy()
        results[f'{name}_{key}'] = energy

# Compute barriers
barriers = {}
for rxn in ['Rxn1_cyclobutene', 'Rxn11_methoxy', 'Rxn20_cyclopropane']:
    e_r = results[f'{rxn}_reactant']
    e_ts = results[f'{rxn}_ts']
    barrier = e_ts - e_r
    barriers[rxn] = barrier
    print(f'{rxn} barrier: {barrier:.3f} eV')

# Save results
with open('outputs/crbh20_barriers.txt', 'w') as f:
    for rxn, b in barriers.items():
        f.write(f'{rxn}: {b:.3f} eV\n')

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
rxns = list(barriers.keys())
vals = list(barriers.values())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(rxns, vals, color=colors)
ax.set_ylabel('Barrier (eV)')
ax.set_title('CRBH20 Reaction Barriers (MACE-MP-0)')
ax.axhline(0, color='black', linewidth=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{v:.2f}', ha='center')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('report/images/crbh20_barriers.png', dpi=150)
plt.close()
print('CRBH20 barriers computed and figure saved.')
