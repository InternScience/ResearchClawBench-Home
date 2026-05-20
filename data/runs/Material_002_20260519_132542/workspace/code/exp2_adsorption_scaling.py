"""
Experiment 2: Adsorption energy scaling relations on transition metal fcc(111) surfaces.
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.build import fcc111, add_adsorbate, molecule
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from mace.calculators import mace_mp
import torch

# Parameters from dataset
METALS = {
    'Ni': 3.52,
    'Cu': 3.61,
    'Rh': 3.80,
    'Pd': 3.89,
    'Ir': 3.84,
    'Pt': 3.92,
}
SLAB_SIZE = (2, 2, 3)
VACUUM = 10.0
MILLER = (1, 1, 1)
ADS_HEIGHT = 1.5
FMAX = 0.05

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc = mace_mp(model="outputs/mace-mp-0b3-medium.model", device=device, default_dtype='float32')

# Gas phase references
# O atom
o_atom = Atoms('O', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
o_atom.calc = calc
# Relax gas phase (trivial)
dyn_o = BFGS(o_atom)
dyn_o.run(fmax=FMAX)
E_O_gas = o_atom.get_potential_energy()

# OH molecule
oh_mol = Atoms('OH', positions=[[0, 0, 0], [0, 0, 1.0]], cell=[10, 10, 10], pbc=True)
oh_mol.calc = calc
dyn_oh = BFGS(oh_mol)
dyn_oh.run(fmax=FMAX)
E_OH_gas = oh_mol.get_potential_energy()

print(f"E_O_gas = {E_O_gas:.4f} eV")
print(f"E_OH_gas = {E_OH_gas:.4f} eV")

results = []

for metal, a in METALS.items():
    print(f"\n--- {metal} (a={a}) ---")
    # Clean slab
    slab = fcc111(metal, size=SLAB_SIZE, a=a, vacuum=VACUUM)
    slab.calc = calc
    c = FixAtoms(mask=slab.get_tags() >= 2)
    slab.set_constraint(c)
    dyn = BFGS(slab)
    dyn.run(fmax=FMAX)
    E_slab = slab.get_potential_energy()
    print(f"E_slab = {E_slab:.4f} eV")

    # O adsorption
    slab_o = fcc111(metal, size=SLAB_SIZE, a=a, vacuum=VACUUM)
    o_ads = Atoms('O', positions=[[0, 0, 0]])
    add_adsorbate(slab_o, o_ads, height=ADS_HEIGHT, position='fcc')
    slab_o.calc = calc
    c_o = FixAtoms(mask=slab_o.get_tags() >= 2)
    slab_o.set_constraint(c_o)
    dyn_o = BFGS(slab_o)
    dyn_o.run(fmax=FMAX)
    E_O_slab = slab_o.get_potential_energy()
    E_ads_O = E_O_slab - E_slab - E_O_gas
    print(f"E_ads_O = {E_ads_O:.4f} eV")

    # OH adsorption
    slab_oh = fcc111(metal, size=SLAB_SIZE, a=a, vacuum=VACUUM)
    oh_ads = Atoms('OH', positions=[[0, 0, 0], [0, 0, 1.0]])
    add_adsorbate(slab_oh, oh_ads, height=ADS_HEIGHT, position='fcc')
    slab_oh.calc = calc
    c_oh = FixAtoms(mask=slab_oh.get_tags() >= 2)
    slab_oh.set_constraint(c_oh)
    dyn_oh = BFGS(slab_oh)
    dyn_oh.run(fmax=FMAX)
    E_OH_slab = slab_oh.get_potential_energy()
    E_ads_OH = E_OH_slab - E_slab - E_OH_gas
    print(f"E_ads_OH = {E_ads_OH:.4f} eV")

    results.append({
        'metal': metal,
        'E_ads_O': E_ads_O,
        'E_ads_OH': E_ads_OH,
    })

# Save results
import csv
with open('outputs/adsorption_scaling.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['metal', 'E_ads_O', 'E_ads_OH'])
    writer.writeheader()
    writer.writerows(results)

# Fit linear scaling
E_O = np.array([r['E_ads_O'] for r in results])
E_OH = np.array([r['E_ads_OH'] for r in results])
coeffs = np.polyfit(E_O, E_OH, 1)
fit_line = np.poly1d(coeffs)
E_O_fit = np.linspace(E_O.min() - 0.2, E_O.max() + 0.2, 100)

plt.figure(figsize=(6, 4))
plt.plot(E_O_fit, fit_line(E_O_fit), 'k--', label=f'Fit: $E_{{OH}} = {coeffs[0]:.2f}E_O + {coeffs[1]:.2f}$')
for r in results:
    plt.plot(r['E_ads_O'], r['E_ads_OH'], 'o', label=r['metal'], markersize=8)
plt.xlabel(r'$E_{ads}^O$ (eV)')
plt.ylabel(r'$E_{ads}^{OH}$ (eV)')
plt.title('Adsorption Energy Scaling Relation (fcc(111))')
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.savefig('report/images/adsorption_scaling.png', dpi=300)
plt.close()

print("\nExperiment 2 complete. Figure saved to report/images/adsorption_scaling.png")
