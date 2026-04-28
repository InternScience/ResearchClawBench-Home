"""Experiment 2: O* and OH* adsorption energy scaling on fcc(111) transition-metal surfaces.

Using MACE-MP-0b3 to compute E_ads for O and OH at fcc-hollow site of
Ni, Cu, Rh, Pd, Ir, Pt(111). Compare scaling line E_OH = a * E_O + b
(Abild-Pedersen-style). Reference DFT-PBE gives slope ~ 0.5.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import make_calc, OUTPUTS

from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS

METALS = {
    "Ni": 3.52, "Cu": 3.61, "Rh": 3.80,
    "Pd": 3.89, "Ir": 3.84, "Pt": 3.92,
}


def make_slab(metal, a):
    slab = fcc111(metal, size=(2, 2, 3), a=a, vacuum=10.0)
    # tag: ASE's fcc111 tags layers from top: 1 (top)..3 (bottom); spec says
    # "fixed layers: bottom 2 layers (tags >= 2)" -> fix tag 2,3
    fix = FixAtoms(mask=[atom.tag >= 2 for atom in slab])
    slab.set_constraint(fix)
    return slab


def make_O_adsorbate():
    return Atoms("O", positions=[[0.0, 0.0, 0.0]])


def make_OH_adsorbate():
    return Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )


def relax(atoms, fmax=0.05, steps=200, label=""):
    atoms.calc = make_calc()
    e0 = atoms.get_potential_energy()
    opt = BFGS(atoms, logfile=os.path.join(OUTPUTS, f"exp2_{label}_bfgs.log"))
    opt.run(fmax=fmax, steps=steps)
    e1 = atoms.get_potential_energy()
    return e1, e0


def gas_O():
    a = Atoms("O", positions=[[0.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=False)
    a.calc = make_calc()
    return a.get_potential_energy()


def gas_OH():
    a = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1.0]],
             cell=[10, 10, 10], pbc=False)
    a.calc = make_calc()
    e0 = a.get_potential_energy()
    opt = BFGS(a, logfile=os.path.join(OUTPUTS, "exp2_gasOH_bfgs.log"))
    opt.run(fmax=0.05, steps=100)
    return a.get_potential_energy()


def main():
    t0 = time.time()
    results = {}

    # gas-phase references
    print("Gas-phase O/OH...", flush=True)
    E_O_gas = gas_O()
    E_OH_gas = gas_OH()
    results["E_O_gas_eV"] = float(E_O_gas)
    results["E_OH_gas_eV"] = float(E_OH_gas)
    print(f"  E[O] = {E_O_gas:.3f} eV   E[OH] = {E_OH_gas:.3f} eV "
          f"(t={time.time()-t0:.1f}s)", flush=True)

    rows = []
    for metal, a_lat in METALS.items():
        print(f"--- {metal} (a={a_lat}) ---", flush=True)
        # clean slab
        slab = make_slab(metal, a_lat)
        E_slab, _ = relax(slab.copy(), label=f"{metal}_slab")

        # O*
        slab_O = make_slab(metal, a_lat)
        add_adsorbate(slab_O, make_O_adsorbate(), height=1.5, position="fcc")
        E_slab_O, _ = relax(slab_O, label=f"{metal}_O")

        # OH*
        slab_OH = make_slab(metal, a_lat)
        add_adsorbate(slab_OH, make_OH_adsorbate(), height=1.5, position="fcc")
        E_slab_OH, _ = relax(slab_OH, label=f"{metal}_OH")

        Eads_O = E_slab_O - E_slab - E_O_gas
        Eads_OH = E_slab_OH - E_slab - E_OH_gas
        print(f"  E_slab={E_slab:.3f}  E_slab_O={E_slab_O:.3f}  "
              f"E_slab_OH={E_slab_OH:.3f}", flush=True)
        print(f"  Eads(O*)={Eads_O:.3f} eV  Eads(OH*)={Eads_OH:.3f} eV "
              f"(t={time.time()-t0:.1f}s)", flush=True)

        rows.append({
            "metal": metal, "a_lat": a_lat,
            "E_slab_eV": float(E_slab),
            "E_slab_O_eV": float(E_slab_O),
            "E_slab_OH_eV": float(E_slab_OH),
            "Eads_O_eV": float(Eads_O),
            "Eads_OH_eV": float(Eads_OH),
        })

    # linear regression
    Eo = np.array([r["Eads_O_eV"] for r in rows])
    Eoh = np.array([r["Eads_OH_eV"] for r in rows])
    slope, intercept = np.polyfit(Eo, Eoh, 1)
    pred = slope * Eo + intercept
    r2 = 1.0 - np.sum((Eoh - pred) ** 2) / np.sum((Eoh - Eoh.mean()) ** 2)

    summary = {
        "rows": rows,
        "slope": float(slope),
        "intercept_eV": float(intercept),
        "r2": float(r2),
        "E_O_gas_eV": float(E_O_gas),
        "E_OH_gas_eV": float(E_OH_gas),
        "model": "MACE-MP-0b3-medium",
    }
    with open(os.path.join(OUTPUTS, "exp2_adsorption.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    import csv
    with open(os.path.join(OUTPUTS, "exp2_adsorption.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nScaling: E_OH = {slope:.3f} * E_O + {intercept:.3f} eV  R^2={r2:.3f}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
