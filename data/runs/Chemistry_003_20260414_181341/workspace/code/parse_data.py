import ase.io
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

datasets = {
    'random_charges': 'data/random_charges.xyz',
    'charged_dimer': 'data/charged_dimer.xyz',
    'ag3_chargestates': 'data/ag3_chargestates.xyz'
}

stats = {}

for name, path in datasets.items():
    atoms_list = ase.io.read(path, index=':')  # read all frames
    stats[name] = {
        'num_frames': len(atoms_list),
        'natoms': len(atoms_list[0]) if atoms_list else 0,
        'energies': [a.get_potential_energy() for a in atoms_list],
        'forces': [a.get_forces() for a in atoms_list] if atoms_list[0].has('forces') else None,
        'charge_state': np.unique([a.info.get('charge_state', None) for a in atoms_list]) if name=='ag3_chargestates' else None,
        'true_charges': [a.arrays.get('true_charges') for a in atoms_list] if 'true_charges' in atoms_list[0].arrays else None,
        'pbc': atoms_list[0].pbc.tolist() if atoms_list else None
    }
    if stats[name]['true_charges'] is not None:
        all_charges = np.concatenate(stats[name]['true_charges'])
        stats[name]['charge_stats'] = {'mean': float(np.mean(all_charges)), 'std': float(np.std(all_charges)), 'unique': np.unique(all_charges).tolist()}

# Save stats
with open('outputs/data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

# For random_charges: box size
if atoms_list and name=='random_charges':
    pos = atoms_list[0].positions
    box = np.max(pos,0) - np.min(pos,0) + 2  # rough
    print(f'Random box approx: {box}')

print('Stats saved')

# Plots
fig, axs = plt.subplots(2,2, figsize=(12,10))
for i, name in enumerate(datasets):
    row,col = i//2, i%2
    axs[row,col].hist(stats[name]['energies'], bins=20)
    axs[row,col].set_title(f'{name} energies')

plt.tight_layout()
plt.savefig('report/images/energy_hist.png')
plt.close()

if stats['random_charges']['true_charges']:
    plt.figure()
    charges = np.concatenate(stats['random_charges']['true_charges'])
    plt.hist(charges, bins=20)
    plt.title('Random charges dist')
    plt.savefig('report/images/random_charges_hist.png')
    plt.close()

print('Plots saved')
