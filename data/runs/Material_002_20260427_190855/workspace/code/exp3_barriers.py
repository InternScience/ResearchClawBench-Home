"""Experiment 3: Reaction barriers for three CRBH20-like model reactions.

Single-point MACE-MP-0 energies on the reactant and transition-state geometries
provided in the dataset. Compute barriers E_TS - E_R and compare to DFT
references (Rxn 1: 1.72, Rxn 11: 1.74, Rxn 20: 1.77 eV).

Note: the supplied geometries are simplified placeholders (not optimised
DFT TS structures), so the absolute barriers can deviate from CRBH20
references. The comparison still illustrates the workflow and the relative
ordering across reactions.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import make_calc, OUTPUTS

from ase import Atoms

DATA = {
    "Rxn1_cyclobutene_ringopen": {
        "dft_eV": 1.72,
        "reactant": ([
            ("C", [0.000, 0.000, 0.000]),
            ("C", [1.500, 0.000, 0.000]),
            ("C", [1.500, 1.500, 0.000]),
            ("C", [0.000, 1.500, 0.000]),
            ("H", [-0.500, -0.500, 0.000]),
            ("H", [2.000, -0.500, 0.000]),
            ("H", [2.000, 2.000, 0.000]),
            ("H", [-0.500, 2.000, 0.000]),
        ]),
        "ts": ([
            ("C", [0.000, 0.000, 0.000]),
            ("C", [1.400, 0.200, 0.000]),
            ("C", [1.400, 1.300, 0.000]),
            ("C", [0.000, 1.500, 0.000]),
            ("H", [-0.500, -0.500, 0.000]),
            ("H", [1.900, -0.300, 0.000]),
            ("H", [1.900, 1.800, 0.000]),
            ("H", [-0.500, 2.000, 0.000]),
        ]),
    },
    "Rxn11_methoxy_decomp": {
        "dft_eV": 1.74,
        "reactant": ([
            ("C", [0.000, 0.000, 0.000]),
            ("H", [0.000, 1.000, 0.000]),
            ("H", [0.900, -0.500, 0.000]),
            ("H", [-0.900, -0.500, 0.000]),
            ("O", [1.200, 0.000, 0.000]),
        ]),
        "ts": ([
            ("C", [0.000, 0.000, 0.000]),
            ("H", [0.000, 1.000, 0.000]),
            ("H", [0.900, -0.500, 0.000]),
            ("H", [-0.900, -0.500, 0.000]),
            ("O", [1.500, 0.000, 0.000]),
        ]),
    },
    "Rxn20_cyclopropane_ringopen": {
        "dft_eV": 1.77,
        "reactant": ([
            ("C", [0.000, 0.000, 0.000]),
            ("C", [1.500, 0.000, 0.000]),
            ("C", [0.750, 1.300, 0.000]),
            ("H", [-0.500, -0.500, 0.000]),
            ("H", [2.000, -0.500, 0.000]),
            ("H", [0.750, 2.000, 0.000]),
            ("H", [0.000, 0.000, 1.000]),
            ("H", [1.500, 0.000, 1.000]),
            ("H", [0.750, 1.300, 1.000]),
        ]),
        "ts": ([
            ("C", [0.000, 0.000, 0.000]),
            ("C", [1.500, 0.000, 0.000]),
            ("C", [0.750, 1.300, 0.000]),
            ("H", [-0.500, -0.500, 0.000]),
            ("H", [2.000, -0.500, 0.000]),
            ("H", [0.750, 2.000, 0.000]),
            ("H", [0.000, 0.000, 1.500]),
            ("H", [1.500, 0.000, 1.500]),
            ("H", [0.750, 1.300, 1.500]),
        ]),
    },
}


def to_atoms(spec):
    syms = [s for s, _ in spec]
    pos = np.array([p for _, p in spec])
    a = Atoms(syms, positions=pos, cell=[15, 15, 15], pbc=False)
    a.center()
    return a


def main():
    calc = make_calc()
    rows = []
    for name, info in DATA.items():
        R = to_atoms(info["reactant"])
        TS = to_atoms(info["ts"])
        R.calc = calc; TS.calc = calc
        Er = R.get_potential_energy()
        Ets = TS.get_potential_energy()
        barr = Ets - Er
        rows.append({
            "reaction": name,
            "E_reactant_eV": float(Er),
            "E_TS_eV": float(Ets),
            "barrier_MACE_eV": float(barr),
            "barrier_DFT_eV": float(info["dft_eV"]),
            "delta_eV": float(barr - info["dft_eV"]),
        })
        print(name, "MACE", round(barr, 3), "DFT", info["dft_eV"], flush=True)

    mae = float(np.mean([abs(r["delta_eV"]) for r in rows]))
    rmse = float(np.sqrt(np.mean([r["delta_eV"]**2 for r in rows])))
    summary = {"rows": rows, "MAE_eV": mae, "RMSE_eV": rmse,
               "model": "MACE-MP-0b3-medium",
               "note": "Single-point energies on simplified placeholder "
                       "geometries from the supplied dataset; the exact "
                       "DFT-optimised TS structures of CRBH20 are not "
                       "provided, so absolute deviations can be large."}
    with open(os.path.join(OUTPUTS, "exp3_barriers.json"), "w") as f:
        json.dump(summary, f, indent=2)
    import csv
    with open(os.path.join(OUTPUTS, "exp3_barriers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"MAE={mae:.3f} eV  RMSE={rmse:.3f} eV")


if __name__ == "__main__":
    main()
