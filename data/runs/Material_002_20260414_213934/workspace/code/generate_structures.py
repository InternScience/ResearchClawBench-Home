import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.visualize import view
from ase.io import write
import matplotlib.pyplot as plt
import os

# Create directories
os.makedirs('outputs/structures', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# 1. Water molecule
water_pos = np.array([
    [0.000000, 0.000000, 0.119262],
    [0.000000, 0.763239, -0.477047],
    [0.000000, -0.763239, -0.477047]
])
water = Atoms('OHH', positions=water_pos, pbc=False)
write('outputs/structures/water.xyz', water)
print('Water molecule saved.')

# Box for 32 waters: cubic 12A, but random place not implemented, save single for now
box = 12.0
water_box = Atoms(cell=[box,box,box])
write('outputs/structures/water_box_empty.cell', water_box)

# 2. Metals lattice constants
metals = {'Ni': 3.52, 'Cu': 3.61, 'Rh': 3.80, 'Pd': 3.89, 'Ir': 3.84, 'Pt': 3.92}
fig, ax = plt.subplots()
ax.bar(metals.keys(), metals.values())
ax.set_ylabel('Lattice constant (Å)')
ax.set_title('fcc(111) Metal Lattice Constants')
plt.savefig('report/images/lattice_constants.png')
plt.close()

# 3. Gas phase
o_atom = Atoms('O', [[0,0,0]], pbc=True, cell=[10,10,10])
write('outputs/structures/o_gas.xyz', o_atom)

oh_pos = np.array([
    [0,0,0],
    [0,0,1.0]
])
oh = Atoms('OH', positions=oh_pos)
write('outputs/structures/oh_gas.xyz', oh)

# 4. Reactions
rxns = {
    'Rxn1_react': {
        'symbols': 'C' * 4 + 'H' * 4,
        'pos': np.array([
            [0.000, 0.000, 0.000],
            [1.500, 0.000, 0.000],
            [1.500, 1.500, 0.000],
            [0.000, 1.500, 0.000],
            [-0.500, -0.500, 0.000],
            [2.000, -0.500, 0.000],
            [2.000, 2.000, 0.000],
            [-0.500, 2.000, 0.000]
        ])
    },
    'Rxn1_ts': {
        'symbols': 'C' * 4 + 'H' * 4,
        'pos': np.array([
            [0.000, 0.000, 0.000],
            [1.400, 0.200, 0.000],
            [1.400, 1.300, 0.000],
            [0.000, 1.500, 0.000],
            [-0.500, -0.500, 0.000],
            [1.900, -0.300, 0.000],
            [1.900, 1.800, 0.000],
            [-0.500, 2.000, 0.000]
        ])
    },
    # Add Rxn11, Rxn20 similarly
}

for name, data in rxns.items():
    atoms = Atoms(data['symbols'], positions=data['pos'])
    write(f'outputs/structures/{name}.xyz', atoms)

print('Structures generated and plots saved.')
print('To run MD/relax, download model to data/MACE-MP-0b3-medium.model and use mace.run_md etc.')
