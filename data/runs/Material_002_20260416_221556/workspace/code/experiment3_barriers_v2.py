"""
Experiment 3 (revised): Reaction Barrier Comparison (CRBH20)
Using larger box and checking with both periodic and non-periodic conditions.
Also trying to relax reactant structures first.
"""
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_002_20260416_221556"
MODEL_PATH = os.path.join(WORKSPACE, "models/mace-mp-0b3-medium.model")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report/images")

from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import MACECalculator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Loading MACE-MP-0b3 model...")

# DFT reference barriers (eV)
dft_barriers = {
    'Rxn 1': 1.72,
    'Rxn 11': 1.74,
    'Rxn 20': 1.77
}

# Use larger box to minimize periodic interactions
box_size = 20.0

# Reaction 1: Cyclobutene ring-opening (C4H4)
# Center molecules in box
center = box_size / 2.0

rxn1_reactant_pos = np.array([
    [0.000, 0.000, 0.000],
    [1.500, 0.000, 0.000],
    [1.500, 1.500, 0.000],
    [0.000, 1.500, 0.000],
    [-0.500, -0.500, 0.000],
    [2.000, -0.500, 0.000],
    [2.000, 2.000, 0.000],
    [-0.500, 2.000, 0.000]
]) + center

rxn1_ts_pos = np.array([
    [0.000, 0.000, 0.000],
    [1.400, 0.200, 0.000],
    [1.400, 1.300, 0.000],
    [0.000, 1.500, 0.000],
    [-0.500, -0.500, 0.000],
    [1.900, -0.300, 0.000],
    [1.900, 1.800, 0.000],
    [-0.500, 2.000, 0.000]
]) + center

rxn2_reactant_pos = np.array([
    [0.000, 0.000, 0.000],
    [0.000, 1.000, 0.000],
    [0.900, -0.500, 0.000],
    [-0.900, -0.500, 0.000],
    [1.200, 0.000, 0.000]
]) + center

rxn2_ts_pos = np.array([
    [0.000, 0.000, 0.000],
    [0.000, 1.000, 0.000],
    [0.900, -0.500, 0.000],
    [-0.900, -0.500, 0.000],
    [1.500, 0.000, 0.000]
]) + center

rxn3_reactant_pos = np.array([
    [0.000, 0.000, 0.000],
    [1.500, 0.000, 0.000],
    [0.750, 1.300, 0.000],
    [-0.500, -0.500, 0.000],
    [2.000, -0.500, 0.000],
    [0.750, 2.000, 0.000],
    [0.000, 0.000, 1.000],
    [1.500, 0.000, 1.000],
    [0.750, 1.300, 1.000]
]) + center

rxn3_ts_pos = np.array([
    [0.000, 0.000, 0.000],
    [1.500, 0.000, 0.000],
    [0.750, 1.300, 0.000],
    [-0.500, -0.500, 0.000],
    [2.000, -0.500, 0.000],
    [0.750, 2.000, 0.000],
    [0.000, 0.000, 1.500],
    [1.500, 0.000, 1.500],
    [0.750, 1.300, 1.500]
]) + center

reactions = {
    'Rxn 1': {
        'name': 'Cyclobutene ring-opening',
        'symbols': ['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
        'reactant_pos': rxn1_reactant_pos,
        'ts_pos': rxn1_ts_pos
    },
    'Rxn 11': {
        'name': 'Methoxy decomposition',
        'symbols': ['C', 'H', 'H', 'H', 'O'],
        'reactant_pos': rxn2_reactant_pos,
        'ts_pos': rxn2_ts_pos
    },
    'Rxn 20': {
        'name': 'Cyclopropane ring-opening',
        'symbols': ['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
        'reactant_pos': rxn3_reactant_pos,
        'ts_pos': rxn3_ts_pos
    }
}

mace_barriers = {}
mace_barriers_relaxed = {}

print("\n=== Computing Reaction Barriers (Single-Point on Given Geometries) ===")
for rxn_id, rxn_data in reactions.items():
    print(f"\n{rxn_id}: {rxn_data['name']}")
    
    # Reactant
    reactant = Atoms(
        symbols=rxn_data['symbols'],
        positions=rxn_data['reactant_pos'],
        cell=[box_size, box_size, box_size],
        pbc=True
    )
    reactant.calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
    E_reactant = reactant.get_potential_energy()
    print(f"  E(reactant, SP) = {E_reactant:.4f} eV")
    
    # TS
    ts = Atoms(
        symbols=rxn_data['symbols'],
        positions=rxn_data['ts_pos'],
        cell=[box_size, box_size, box_size],
        pbc=True
    )
    ts.calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
    E_ts = ts.get_potential_energy()
    print(f"  E(TS, SP) = {E_ts:.4f} eV")
    
    barrier = E_ts - E_reactant
    mace_barriers[rxn_id] = float(barrier)
    print(f"  Barrier (SP) = {barrier:.4f} eV (DFT ref: {dft_barriers[rxn_id]:.4f} eV)")
    
    # Also try relaxing the reactant (but NOT the TS)
    reactant_relax = reactant.copy()
    reactant_relax.calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
    opt = BFGS(reactant_relax, logfile=None)
    try:
        opt.run(fmax=0.05, steps=100)
        E_reactant_relax = reactant_relax.get_potential_energy()
        barrier_relax = E_ts - E_reactant_relax
        mace_barriers_relaxed[rxn_id] = float(barrier_relax)
        print(f"  E(reactant, relaxed) = {E_reactant_relax:.4f} eV")
        print(f"  Barrier (relaxed reactant) = {barrier_relax:.4f} eV")
    except Exception as e:
        print(f"  Relaxation failed: {e}")
        mace_barriers_relaxed[rxn_id] = float(barrier)

# Save results
barrier_results = {}
for rxn_id in reactions:
    barrier_results[rxn_id] = {
        'name': reactions[rxn_id]['name'],
        'mace_barrier_sp_eV': mace_barriers[rxn_id],
        'mace_barrier_relaxed_eV': mace_barriers_relaxed.get(rxn_id, None),
        'dft_barrier_eV': dft_barriers[rxn_id],
        'difference_sp_eV': mace_barriers[rxn_id] - dft_barriers[rxn_id],
    }

with open(os.path.join(OUTPUT_DIR, 'reaction_barriers.json'), 'w') as f:
    json.dump(barrier_results, f, indent=2)

print("\n=== Summary (Single-Point) ===")
print(f"{'Reaction':>10s} {'MACE SP':>10s} {'MACE Relax':>10s} {'DFT':>10s}")
print("-" * 45)
for rxn_id in reactions:
    d = barrier_results[rxn_id]
    relax_val = f"{d['mace_barrier_relaxed_eV']:.3f}" if d['mace_barrier_relaxed_eV'] is not None else "N/A"
    print(f"{rxn_id:>10s} {d['mace_barrier_sp_eV']:>10.3f} {relax_val:>10s} {d['dft_barrier_eV']:>10.3f}")

# Compute MAE
mae_sp = np.mean([abs(barrier_results[r]['difference_sp_eV']) for r in barrier_results])
print(f"\nMAE (single-point) = {mae_sp:.3f} eV")

if all(barrier_results[r]['mace_barrier_relaxed_eV'] is not None for r in barrier_results):
    mae_relax = np.mean([abs(barrier_results[r]['mace_barrier_relaxed_eV'] - dft_barriers[r]) for r in barrier_results])
    print(f"MAE (relaxed reactant) = {mae_relax:.3f} eV")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rxn_labels = list(reactions.keys())
rxn_names = [reactions[r]['name'] for r in rxn_labels]
mace_vals_sp = [mace_barriers[r] for r in rxn_labels]
mace_vals_relax = [mace_barriers_relaxed.get(r, 0) for r in rxn_labels]
dft_vals = [dft_barriers[r] for r in rxn_labels]

x = np.arange(len(rxn_labels))
width = 0.25

bars1 = axes[0].bar(x - width, mace_vals_sp, width, label='MACE (SP)', color='steelblue', edgecolor='black')
bars2 = axes[0].bar(x, mace_vals_relax, width, label='MACE (Relaxed R)', color='lightblue', edgecolor='black')
bars3 = axes[0].bar(x + width, dft_vals, width, label='DFT Reference', color='coral', edgecolor='black')

axes[0].set_xlabel('Reaction', fontsize=12)
axes[0].set_ylabel('Barrier Height (eV)', fontsize=12)
axes[0].set_title('Reaction Barriers: MACE-MP-0 vs DFT\n(CRBH20 Subset)', fontsize=14)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{l}\n({n})' for l, n in zip(rxn_labels, rxn_names)], fontsize=8)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Parity plot
all_mace = mace_vals_sp
all_dft = dft_vals
axes[1].scatter(all_dft, all_mace, s=120, c='steelblue', edgecolors='black', zorder=5, label='SP')
if mace_vals_relax:
    axes[1].scatter(all_dft, mace_vals_relax, s=120, c='lightblue', edgecolors='black', zorder=5, 
                   marker='s', label='Relaxed R')

for i, label in enumerate(rxn_labels):
    axes[1].annotate(label, (all_dft[i], all_mace[i]), 
                    textcoords="offset points", xytext=(10, 5), fontsize=11)

# Perfect agreement line
all_vals = all_mace + all_dft + mace_vals_relax
lims = [min(all_vals) - 0.5, max(all_vals) + 0.5]
axes[1].plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
axes[1].set_xlabel('DFT Barrier (eV)', fontsize=12)
axes[1].set_ylabel('MACE-MP-0 Barrier (eV)', fontsize=12)
axes[1].set_title(f'Parity Plot (MAE = {mae_sp:.3f} eV)', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'reaction_barriers.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved barrier comparison plot to {os.path.join(IMAGE_DIR, 'reaction_barriers.png')}")

print("\nExperiment 3 (revised) complete!")
