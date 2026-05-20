import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os

os.makedirs('report/images', exist_ok=True)

# Load datasets
random = read('data/random_charges.xyz', index=':')
dimer = read('data/charged_dimer.xyz', index=':')
ag3 = read('data/ag3_chargestates.xyz', index=':')

print(f"random_charges: {len(random)} frames, {len(random[0])} atoms")
print(f"charged_dimer: {len(dimer)} frames, {len(dimer[0])} atoms")
print(f"ag3_chargestates: {len(ag3)} frames, {len(ag3[0])} atoms")

# Basic statistics
def get_stats(frames):
    natoms = [len(f) for f in frames]
    forces = []
    for f in frames:
        if 'forces' in f.arrays:
            forces.append(f.arrays['forces'])
    return {
        'n_frames': len(frames),
        'n_atoms_mean': np.mean(natoms),
        'has_forces': len(forces) > 0,
        'force_magnitude_mean': np.mean([np.linalg.norm(f) for f in forces]) if forces else None
    }

print("random:", get_stats(random))
print("dimer:", get_stats(dimer))
print("ag3:", get_stats(ag3))

# Plot 1: Random charges overview (dummy since no labels)
fig, ax = plt.subplots()
ax.text(0.5, 0.5, 'Random Charges Dataset\n128 atoms per frame\n100 frames', ha='center', va='center', fontsize=14)
ax.axis('off')
plt.savefig('report/images/figure1_random_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Charged dimer binding curve proxy (distances)
dists = []
for f in dimer:
    pos = f.get_positions()
    dists.append(np.linalg.norm(pos[0] - pos[4]))  # approx separation
plt.figure()
plt.plot(dists, label='Dimer separation')
plt.xlabel('Frame')
plt.ylabel('Distance (Å)')
plt.title('Charged Dimer Configurations')
plt.savefig('report/images/figure2_dimer_distances.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Ag3 charge states
charges = [f.info.get('charge', 0) for f in ag3]
plt.figure()
plt.hist(charges, bins=2)
plt.title('Ag3 Charge State Distribution')
plt.savefig('report/images/figure3_ag3_charges.png', dpi=150, bbox_inches='tight')
plt.close()

print("Basic figures generated.")
