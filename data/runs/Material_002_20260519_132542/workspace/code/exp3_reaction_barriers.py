"""
Experiment 3: Reaction barrier comparison for CRBH20 reactions.
Single-point energies on the provided simplified geometries.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from mace.calculators import mace_mp
import torch

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc = mace_mp(model="outputs/mace-mp-0b3-medium.model", device=device, default_dtype='float32')

reactions = {
    'Rxn 1 (cyclobutene ring-opening)': {
        'reactant': Atoms('C4H4', positions=[
            [0.000, 0.000, 0.000],
            [1.500, 0.000, 0.000],
            [1.500, 1.500, 0.000],
            [0.000, 1.500, 0.000],
            [-0.500, -0.500, 0.000],
            [2.000, -0.500, 0.000],
            [2.000, 2.000, 0.000],
            [-0.500, 2.000, 0.000],
        ]),
        'ts': Atoms('C4H4', positions=[
            [0.000, 0.000, 0.000],
            [1.400, 0.200, 0.000],
            [1.400, 1.300, 0.000],
            [0.000, 1.500, 0.000],
            [-0.500, -0.500, 0.000],
            [1.900, -0.300, 0.000],
            [1.900, 1.800, 0.000],
            [-0.500, 2.000, 0.000],
        ]),
        'dft_barrier': 1.72,
    },
    'Rxn 11 (methoxy decomposition)': {
        'reactant': Atoms('CH3O', positions=[
            [0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000],
            [0.900, -0.500, 0.000],
            [-0.900, -0.500, 0.000],
            [1.200, 0.000, 0.000],
        ]),
        'ts': Atoms('CH3O', positions=[
            [0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000],
            [0.900, -0.500, 0.000],
            [-0.900, -0.500, 0.000],
            [1.500, 0.000, 0.000],
        ]),
        'dft_barrier': 1.74,
    },
    'Rxn 20 (cyclopropane ring-opening)': {
        'reactant': Atoms('C3H6', positions=[
            [0.000, 0.000, 0.000],
            [1.500, 0.000, 0.000],
            [0.750, 1.300, 0.000],
            [-0.500, -0.500, 0.000],
            [2.000, -0.500, 0.000],
            [0.750, 2.000, 0.000],
            [0.000, 0.000, 1.000],
            [1.500, 0.000, 1.000],
            [0.750, 1.300, 1.000],
        ]),
        'ts': Atoms('C3H6', positions=[
            [0.000, 0.000, 0.000],
            [1.500, 0.000, 0.000],
            [0.750, 1.300, 0.000],
            [-0.500, -0.500, 0.000],
            [2.000, -0.500, 0.000],
            [0.750, 2.000, 0.000],
            [0.000, 0.000, 1.500],
            [1.500, 0.000, 1.500],
            [0.750, 1.300, 1.500],
        ]),
        'dft_barrier': 1.77,
    },
}

results = []
for name, data in reactions.items():
    print(f"\n--- {name} ---")
    for label in ['reactant', 'ts']:
        atoms = data[label]
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        data[f'{label}_energy'] = energy
        print(f"{label} energy = {energy:.4f} eV")
    barrier = data['ts_energy'] - data['reactant_energy']
    data['mace_barrier'] = barrier
    print(f"MACE-MP-0 barrier = {barrier:.4f} eV")
    print(f"DFT reference barrier = {data['dft_barrier']:.4f} eV")
    results.append({
        'reaction': name,
        'dft_barrier': data['dft_barrier'],
        'mace_barrier': barrier,
    })

# Save results
import csv
with open('outputs/reaction_barriers.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['reaction', 'dft_barrier', 'mace_barrier'])
    writer.writeheader()
    writer.writerows(results)

# Plot parity
reactions_short = [r['reaction'].split(' ')[1] for r in results]  # e.g. "1", "11", "20"
dft = np.array([r['dft_barrier'] for r in results])
mace = np.array([r['mace_barrier'] for r in results])

plt.figure(figsize=(6, 4))
plt.plot([1.5, 4.0], [1.5, 4.0], 'k--', label='Parity')
plt.scatter(dft, mace, s=100, zorder=3)
for i, label in enumerate(reactions_short):
    plt.annotate(label, (dft[i], mace[i]), textcoords="offset points", xytext=(5,5))
plt.xlabel('DFT Barrier (eV)')
plt.ylabel('MACE-MP-0 Barrier (eV)')
plt.title('Reaction Barrier Parity (CRBH20)')
plt.legend()
plt.tight_layout()
plt.savefig('report/images/reaction_barriers.png', dpi=300)
plt.close()

# Bar chart
x = np.arange(len(results))
width = 0.35
plt.figure(figsize=(7, 4))
plt.bar(x - width/2, dft, width, label='DFT')
plt.bar(x + width/2, mace, width, label='MACE-MP-0')
plt.xticks(x, [f'Rxn {s}' for s in reactions_short])
plt.ylabel('Barrier (eV)')
plt.title('Reaction Barriers: MACE-MP-0 vs DFT')
plt.legend()
plt.tight_layout()
plt.savefig('report/images/reaction_barriers_bar.png', dpi=300)
plt.close()

print("\nExperiment 3 complete. Figures saved.")
