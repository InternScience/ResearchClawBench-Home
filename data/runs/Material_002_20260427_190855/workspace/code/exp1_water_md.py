"""Experiment 1: 32 water molecules NVT MD at 330 K with MACE-MP-0b3.

Reproduces Fig. 1 of the MACE-MP-0 paper: liquid water RDFs (O-O, O-H, H-H).
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import make_calc, OUTPUTS, IMAGES

import ase
from ase import Atoms
from ase.build import molecule
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units


# ---- Build the 32-water box --------------------------------------------------
def build_water_box(n=32, box=12.0, seed=42):
    """Place 32 water molecules randomly inside a cubic box.

    Single H2O coords from the dataset (ASE molecule('H2O') after centering).
    """
    rng = np.random.default_rng(seed)

    template = Atoms(
        "OHH",
        positions=[
            [0.000000, 0.000000, 0.119262],
            [0.000000, 0.763239, -0.477047],
            [0.000000, -0.763239, -0.477047],
        ],
    )

    cell = np.eye(3) * box
    atoms = Atoms(cell=cell, pbc=True)

    placed = 0
    min_dist = 2.4  # min O-O distance to avoid overlap
    O_positions = []
    tries = 0
    while placed < n and tries < 50000:
        tries += 1
        center = rng.uniform(0.5, box - 0.5, size=3)
        if any(np.linalg.norm(center - p) < min_dist for p in O_positions):
            continue
        # random rotation
        from scipy.spatial.transform import Rotation as R
        rot = R.random(random_state=rng).as_matrix()
        new = template.copy()
        new.set_positions(template.get_positions() @ rot.T + center)
        atoms += new
        O_positions.append(center)
        placed += 1
    print(f"Placed {placed} water molecules in {tries} tries.")
    return atoms


def run_md(steps=2000, T=330.0, dt_fs=0.5, friction_fs=0.01,
           log_every=20, checkpoint=None):
    atoms = build_water_box()
    atoms.calc = make_calc()

    # initial velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=T,
        friction=friction_fs / units.fs,  # ASE expects 1/time in internal
    )

    log = []
    traj_positions = []  # list of (n_atoms,3) positions arrays
    traj_symbols = atoms.get_chemical_symbols()
    cell = np.array(atoms.get_cell())

    t0 = time.time()

    def record():
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        T_inst = atoms.get_temperature()
        log.append({
            "step": dyn.nsteps, "time_fs": dyn.nsteps * dt_fs,
            "Epot_eV": float(e_pot), "Ekin_eV": float(e_kin),
            "T_K": float(T_inst), "wall_s": time.time() - t0,
        })
        # save full positions every log_every steps for RDF after equilibration
        traj_positions.append(atoms.get_positions().copy())
        if dyn.nsteps % 100 == 0:
            print(f"  step {dyn.nsteps:4d}  T={T_inst:6.1f} K  "
                  f"E={e_pot:9.2f} eV  wall={time.time() - t0:.1f}s",
                  flush=True)

    record()
    for k in range(steps):
        dyn.run(1)
        if (k + 1) % log_every == 0:
            record()
            if checkpoint:
                np.savez(checkpoint, positions=np.array(traj_positions),
                         cell=cell, symbols=np.array(traj_symbols))
                with open(checkpoint.replace(".npz", "_log.json"), "w") as f:
                    json.dump(log, f, indent=2)

    np.savez(checkpoint, positions=np.array(traj_positions),
             cell=cell, symbols=np.array(traj_symbols))
    with open(checkpoint.replace(".npz", "_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    return atoms, log, traj_positions, cell, traj_symbols


def compute_rdf(positions_list, symbols, cell, pair, r_max=6.0, n_bins=120,
                skip_frames=0):
    """Compute g(r) for an atomic pair under PBC, averaged over frames."""
    a, b = pair
    sym = np.array(symbols)
    idx_a = np.where(sym == a)[0]
    idx_b = np.where(sym == b)[0]
    L = np.diag(cell)
    bins = np.linspace(0, r_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist = np.zeros(n_bins)
    used = positions_list[skip_frames:]
    for pos in used:
        ra = pos[idx_a]
        rb = pos[idx_b]
        # all pairs (a in A, b in B) with PBC
        d = ra[:, None, :] - rb[None, :, :]
        d -= np.round(d / L) * L
        r = np.linalg.norm(d, axis=-1)
        if a == b:
            r = r[~np.eye(r.shape[0], dtype=bool)]
        else:
            r = r.ravel()
        h, _ = np.histogram(r, bins=bins)
        hist += h
    n_frames = len(used)
    n_a, n_b = len(idx_a), len(idx_b)
    V = float(np.linalg.det(cell))
    rho_b = n_b / V
    shell_vol = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    if a == b:
        norm = rho_b * shell_vol * (n_a - 1) * n_frames
    else:
        norm = rho_b * shell_vol * n_a * n_frames
    g = hist / norm
    return centers, g


if __name__ == "__main__":
    n_steps = int(os.environ.get("MD_STEPS", "2000"))
    log_every = int(os.environ.get("MD_LOG_EVERY", "20"))
    skip_frames = int(os.environ.get("RDF_SKIP", "25"))  # discard ~ first 25 frames

    ckpt = os.path.join(OUTPUTS, "exp1_water_md.npz")
    atoms, log, traj, cell, symbols = run_md(
        steps=n_steps, log_every=log_every, checkpoint=ckpt)

    rdf = {}
    for pair, label in [(("O", "O"), "OO"), (("O", "H"), "OH"), (("H", "H"), "HH")]:
        r, g = compute_rdf(traj, symbols, cell, pair, r_max=6.0,
                           n_bins=120, skip_frames=skip_frames)
        rdf[label] = {"r": r.tolist(), "g": g.tolist()}

    rdf["meta"] = {"n_steps": n_steps, "log_every": log_every,
                   "skip_frames": skip_frames,
                   "n_frames_used": len(traj) - skip_frames,
                   "T_target_K": 330.0, "dt_fs": 0.5,
                   "n_water": 32, "box_A": 12.0,
                   "model": "MACE-MP-0b3-medium"}
    with open(os.path.join(OUTPUTS, "exp1_rdf.json"), "w") as f:
        json.dump(rdf, f)

    print("Done. Wrote", os.path.join(OUTPUTS, "exp1_rdf.json"))
