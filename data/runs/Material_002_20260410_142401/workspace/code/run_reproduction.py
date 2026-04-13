import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import add_adsorbate, fcc111, molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase import units
from mace.calculators import mace_mp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)


def get_calc():
    return mace_mp(model='medium', device='cpu', default_dtype='float32')


def random_rotation_matrix(rng):
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)],
    ])


def make_water_box(n_mol=32, box=12.0, seed=0):
    rng = np.random.default_rng(seed)
    base = molecule('H2O')
    base.set_positions(np.array([
        [0.000000, 0.000000, 0.119262],
        [0.000000, 0.763239, -0.477047],
        [0.000000, -0.763239, -0.477047],
    ]))
    numbers = []
    positions = []
    oxygen_positions = []
    min_oo = 2.3
    attempts = 0
    while len(oxygen_positions) < n_mol and attempts < 200000:
        attempts += 1
        center = rng.uniform(1.5, box - 1.5, size=3)
        if oxygen_positions:
            d = np.linalg.norm(np.array(oxygen_positions) - center, axis=1)
            if np.min(d) < min_oo:
                continue
        R = random_rotation_matrix(rng)
        pos = base.get_positions() @ R.T + center
        oxygen_positions.append(center)
        numbers.extend(base.numbers)
        positions.extend(pos)
    if len(oxygen_positions) < n_mol:
        raise RuntimeError('Failed to place all water molecules without overlap')
    atoms = Atoms(numbers=numbers, positions=positions, cell=[box, box, box], pbc=True)
    return atoms


def minimum_image(vecs, cell_len):
    return vecs - np.round(vecs / cell_len) * cell_len


def oxygen_rdf(frames, box_len, r_max=6.0, bins=120):
    dr = r_max / bins
    hist = np.zeros(bins)
    n_frames = len(frames)
    if n_frames == 0:
        raise RuntimeError('No frames collected for RDF')
    for at in frames:
        pos = at.positions[np.array(at.numbers) == 8]
        N = len(pos)
        for i in range(N - 1):
            rij = pos[i+1:] - pos[i]
            rij = minimum_image(rij, box_len)
            dist = np.linalg.norm(rij, axis=1)
            valid = dist < r_max
            idx = (dist[valid] / dr).astype(int)
            idx = idx[idx < bins]
            np.add.at(hist, idx, 2)
    r = (np.arange(bins) + 0.5) * dr
    N = np.sum(np.array(frames[0].numbers) == 8)
    volume = box_len**3
    rho = N / volume
    shell = 4 * np.pi * r**2 * dr
    norm = n_frames * N * rho * shell
    g = hist / norm
    return r, g


def run_water(calc):
    atoms = make_water_box()
    atoms.calc = calc
    dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=330, friction=0.01 / units.fs)
    temps = []
    energies = []
    frames = []
    start = time.time()
    for step in range(800):
        dyn.run(1)
        if step % 10 == 0:
            temps.append(atoms.get_temperature())
            energies.append(atoms.get_potential_energy())
        if step >= 300 and step % 20 == 0:
            frames.append(atoms.copy())
    runtime = time.time() - start
    r, g = oxygen_rdf(frames, box_len=12.0)
    rdf_df = pd.DataFrame({'r_angstrom': r, 'g_oo': g})
    rdf_df.to_csv(OUT / 'water_rdf.csv', index=False)
    trace_df = pd.DataFrame({'sample_index': range(len(temps)), 'temperature_K': temps, 'potential_energy_eV': energies})
    trace_df.to_csv(OUT / 'water_trace.csv', index=False)

    plt.figure(figsize=(6,4))
    plt.plot(r, g, lw=2)
    plt.axvline(r[np.argmax(g)], color='tab:red', ls='--', lw=1, label=f'first peak = {r[np.argmax(g)]:.2f} Å')
    plt.xlabel('r (Å)')
    plt.ylabel('g$_{OO}$(r)')
    plt.title('Liquid water O–O radial distribution from MACE-MP-0 MD')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(IMG / 'water_rdf.png', dpi=200)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(6,4))
    x = np.arange(len(temps)) * 10 * 0.5
    ax1.plot(x, temps, color='tab:blue', label='Temperature')
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Temperature (K)', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(x, energies, color='tab:orange', label='Potential energy')
    ax2.set_ylabel('Potential energy (eV)', color='tab:orange')
    plt.title('Water MD stability trace')
    fig.tight_layout()
    plt.savefig(IMG / 'water_trace.png', dpi=200)
    plt.close()
    return {
        'n_frames_rdf': len(frames),
        'runtime_s': runtime,
        'mean_temperature_K': float(np.mean(temps)),
        'std_temperature_K': float(np.std(temps)),
        'first_peak_r_angstrom': float(r[np.argmax(g)]),
        'first_peak_height': float(np.max(g)),
    }


def relax_energy(atoms, calc, fmax=0.05, steps=80):
    atoms = atoms.copy()
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    e = atoms.get_potential_energy()
    return atoms, float(e)


def run_adsorption(calc):
    metals = {'Ni': 3.52, 'Cu': 3.61, 'Rh': 3.80, 'Pd': 3.89, 'Ir': 3.84, 'Pt': 3.92}
    rows = []
    gas_O = Atoms('O', positions=[[0,0,0]], cell=[10,10,10], pbc=False)
    gas_OH = Atoms('OH', positions=[[0,0,0],[0,0,1.0]], cell=[10,10,10], pbc=False)
    _, E_O = relax_energy(gas_O, calc, steps=1)
    _, E_OH = relax_energy(gas_OH, calc, steps=80)
    for metal, a in metals.items():
        slab = fcc111(metal, size=(2,2,3), a=a, vacuum=10.0)
        mask = slab.get_tags() >= 2
        slab.set_constraint(FixAtoms(mask=mask))
        slab_relaxed, E_slab = relax_energy(slab, calc, steps=80)
        for ads, gasE in [('O', E_O), ('OH', E_OH)]:
            ads_slab = slab_relaxed.copy()
            if ads == 'O':
                add_adsorbate(ads_slab, Atoms('O', positions=[[0,0,0]]), 1.5, 'fcc')
            else:
                add_adsorbate(ads_slab, Atoms('OH', positions=[[0,0,0],[0,0,1.0]]), 1.5, 'fcc')
            ads_slab.set_constraint(FixAtoms(mask=ads_slab.get_tags() >= 2))
            ads_relaxed, E_tot = relax_energy(ads_slab, calc, steps=100)
            E_ads = E_tot - E_slab - gasE
            rows.append({'metal': metal, 'adsorbate': ads, 'slab_energy_eV': E_slab, 'gas_energy_eV': gasE, 'total_energy_eV': E_tot, 'adsorption_energy_eV': E_ads})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'adsorption_energies.csv', index=False)
    piv = df.pivot(index='metal', columns='adsorbate', values='adsorption_energy_eV').reset_index()
    x = piv['O'].values
    y = piv['OH'].values
    coef = np.polyfit(x, y, 1)
    fitx = np.linspace(x.min()-0.2, x.max()+0.2, 100)
    fity = coef[0] * fitx + coef[1]
    r2 = 1 - np.sum((y - (coef[0]*x + coef[1]))**2) / np.sum((y - y.mean())**2)
    plt.figure(figsize=(5.5,4.5))
    plt.scatter(x, y, s=50)
    for _, row in piv.iterrows():
        plt.text(row['O']+0.02, row['OH']+0.02, row['metal'], fontsize=9)
    plt.plot(fitx, fity, color='tab:red', lw=2, label=f'y = {coef[0]:.2f}x + {coef[1]:.2f}\n$R^2$ = {r2:.2f}')
    plt.xlabel('O adsorption energy (eV)')
    plt.ylabel('OH adsorption energy (eV)')
    plt.title('Adsorption-energy scaling on fcc(111) surfaces')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(IMG / 'adsorption_scaling.png', dpi=200)
    plt.close()
    return {'slope': float(coef[0]), 'intercept': float(coef[1]), 'r2': float(r2), 'gas_E_O': E_O, 'gas_E_OH': E_OH}


def build_atoms(symbols, coords):
    return Atoms(symbols=symbols, positions=np.array(coords, dtype=float), cell=[20,20,20], pbc=False)


def run_reactions(calc):
    reactions = {
        'Rxn 1': {
            'reactant': (['C','C','C','C','H','H','H','H'], [[0.000,0.000,0.000],[1.500,0.000,0.000],[1.500,1.500,0.000],[0.000,1.500,0.000],[-0.500,-0.500,0.000],[2.000,-0.500,0.000],[2.000,2.000,0.000],[-0.500,2.000,0.000]]),
            'ts': (['C','C','C','C','H','H','H','H'], [[0.000,0.000,0.000],[1.400,0.200,0.000],[1.400,1.300,0.000],[0.000,1.500,0.000],[-0.500,-0.500,0.000],[1.900,-0.300,0.000],[1.900,1.800,0.000],[-0.500,2.000,0.000]]),
            'dft_barrier_eV': 1.72,
        },
        'Rxn 11': {
            'reactant': (['C','H','H','H','O'], [[0.000,0.000,0.000],[0.000,1.000,0.000],[0.900,-0.500,0.000],[-0.900,-0.500,0.000],[1.200,0.000,0.000]]),
            'ts': (['C','H','H','H','O'], [[0.000,0.000,0.000],[0.000,1.000,0.000],[0.900,-0.500,0.000],[-0.900,-0.500,0.000],[1.500,0.000,0.000]]),
            'dft_barrier_eV': 1.74,
        },
        'Rxn 20': {
            'reactant': (['C','C','C','H','H','H','H','H','H'], [[0.000,0.000,0.000],[1.500,0.000,0.000],[0.750,1.300,0.000],[-0.500,-0.500,0.000],[2.000,-0.500,0.000],[0.750,2.000,0.000],[0.000,0.000,1.000],[1.500,0.000,1.000],[0.750,1.300,1.000]]),
            'ts': (['C','C','C','H','H','H','H','H','H'], [[0.000,0.000,0.000],[1.500,0.000,0.000],[0.750,1.300,0.000],[-0.500,-0.500,0.000],[2.000,-0.500,0.000],[0.750,2.000,0.000],[0.000,0.000,1.500],[1.500,0.000,1.500],[0.750,1.300,1.500]]),
            'dft_barrier_eV': 1.77,
        },
    }
    rows = []
    for name, spec in reactions.items():
        react = build_atoms(*spec['reactant'])
        ts = build_atoms(*spec['ts'])
        react.calc = calc
        ts.calc = calc
        E_r = float(react.get_potential_energy())
        E_ts = float(ts.get_potential_energy())
        barrier = E_ts - E_r
        rows.append({'reaction': name, 'E_reactant_eV': E_r, 'E_ts_eV': E_ts, 'barrier_eV': barrier, 'dft_barrier_eV': spec['dft_barrier_eV'], 'abs_error_eV': abs(barrier-spec['dft_barrier_eV'])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'reaction_barriers.csv', index=False)

    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, df['dft_barrier_eV'], width, label='DFT reference')
    plt.bar(x + width/2, df['barrier_eV'], width, label='MACE-MP-0')
    plt.xticks(x, df['reaction'])
    plt.ylabel('Barrier (eV)')
    plt.title('Reaction barrier comparison')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(IMG / 'reaction_barriers.png', dpi=200)
    plt.close()

    plt.figure(figsize=(4.5,4.5))
    plt.scatter(df['dft_barrier_eV'], df['barrier_eV'], s=60)
    lims = [min(df['dft_barrier_eV'].min(), df['barrier_eV'].min())-0.2, max(df['dft_barrier_eV'].max(), df['barrier_eV'].max())+0.2]
    plt.plot(lims, lims, 'k--', lw=1)
    for _, row in df.iterrows():
        plt.text(row['dft_barrier_eV']+0.01, row['barrier_eV']+0.01, row['reaction'], fontsize=9)
    plt.xlabel('DFT barrier (eV)')
    plt.ylabel('MACE-MP-0 barrier (eV)')
    plt.title('Barrier correlation')
    plt.tight_layout()
    plt.savefig(IMG / 'reaction_barrier_correlation.png', dpi=200)
    plt.close()

    mae = float(df['abs_error_eV'].mean())
    return {'mae_eV': mae}


def main():
    calc = get_calc()
    summary = {}
    summary['water'] = run_water(calc)
    summary['adsorption'] = run_adsorption(calc)
    summary['reactions'] = run_reactions(calc)
    with open(OUT / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
