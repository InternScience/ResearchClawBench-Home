"""
Experiment 3: Reaction Barrier Comparison (CRBH20)
Compare MACE-MP-0 predicted reaction barriers to DFT reference values.
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

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

from ase import Atoms
from mace.calculators import MACECalculator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Loading MACE-MP-0b3 model...")
calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")

# DFT reference barriers (eV)
dft_barriers = {
    'Rxn 1': 1.72,
    'Rxn 11': 1.74,
    'Rxn 20': 1.77
}

# Reaction 1: Cyclobutene ring-opening (C4H4)
rxn1_reactant = Atoms(
    symbols=['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 0.000, 0.000],
        [1.500, 0.000, 0.000],
        [1.500, 1.500, 0.000],
        [0.000, 1.500, 0.000],
        [-0.500, -0.500, 0.000],
        [2.000, -0.500, 0.000],
        [2.000, 2.000, 0.000],
        [-0.500, 2.000, 0.000]
    ],
    cell=[15, 15, 15],
    pbc=True
)

rxn1_ts = Atoms(
    symbols=['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 0.000, 0.000],
        [1.400, 0.200, 0.000],
        [1.400, 1.300, 0.000],
        [0.000, 1.500, 0.000],
        [-0.500, -0.500, 0.000],
        [1.900, -0.300, 0.000],
        [1.900, 1.800, 0.000],
        [-0.500, 2.000, 0.000]
    ],
    cell=[15, 15, 15],
    pbc=True
)

# Reaction 2: Methoxy decomposition (CH3O)
rxn2_reactant = Atoms(
    symbols=['C', 'H', 'H', 'H', 'O'],
    positions=[
        [0.000, 0.000, 0.000],
        [0.000, 1.000, 0.000],
        [0.900, -0.500, 0.000],
        [-0.900, -0.500, 0.000],
        [1.200, 0.000, 0.000]
    ],
    cell=[15, 15, 15],
    pbc=True
)

rxn2_ts = Atoms(
    symbols=['C', 'H', 'H', 'H', 'O'],
    positions=[
        [0.000, 0.000, 0.000],
        [0.000, 1.000, 0.000],
        [0.900, -0.500, 0.000],
        [-0.900, -0.500, 0.000],
        [1.500, 0.000, 0.000]
    ],
    cell=[15, 15, 15],
    pbc=True
)

# Reaction 3: Cyclopropane ring-opening (C3H6)
rxn3_reactant = Atoms(
    symbols=['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 0.000, 0.000],
        [1.500, 0.000, 0.000],
        [0.750, 1.300, 0.000],
        [-0.500, -0.500, 0.000],
        [2.000, -0.500, 0.000],
        [0.750, 2.000, 0.000],
        [0.000, 0.000, 1.000],
        [1.500, 0.000, 1.000],
        [0.750, 1.300, 1.000]
    ],
    cell=[15, 15, 15],
    pbc=True
)

rxn3_ts = Atoms(
    symbols=['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 0.000, 0.000],
        [1.500, 0.000, 0.000],
        [0.750, 1.300, 0.000],
        [-0.500, -0.500, 0.000],
        [2.000, -0.500, 0.000],
        [0.750, 2.000, 0.000],
        [0.000, 0.000, 1.500],
        [1.500, 0.000, 1.500],
        [0.750, 1.300, 1.500]
    ],
    cell=[15, 15, 15],
    pbc=True
)

reactions = {
    'Rxn 1': {'name': 'Cyclobutene ring-opening', 'reactant': rxn1_reactant, 'ts': rxn1_ts},
    'Rxn 11': {'name': 'Methoxy decomposition', 'reactant': rxn2_reactant, 'ts': rxn2_ts},
    'Rxn 20': {'name': 'Cyclopropane ring-opening', 'reactant': rxn3_reactant, 'ts': rxn3_ts}
}

mace_barriers = {}

print("\n=== Computing Reaction Barriers ===")
for rxn_id, rxn_data in reactions.items():
    print(f"\n{rxn_id}: {rxn_data['name']}")
    
    # Reactant energy
    rxn_data['reactant'].calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
    E_reactant = rxn_data['reactant'].get_potential_energy()
    print(f"  E(reactant) = {E_reactant:.4f} eV")
    
    # TS energy
    rxn_data['ts'].calc = MACECalculator(model_paths=MODEL_PATH, device="cpu", default_dtype="float64")
    E_ts = rxn_data['ts'].get_potential_energy()
    print(f"  E(TS) = {E_ts:.4f} eV")
    
    # Barrier
    barrier = E_ts - E_reactant
    mace_barriers[rxn_id] = float(barrier)
    print(f"  Barrier (MACE) = {barrier:.4f} eV")
    print(f"  Barrier (DFT)  = {dft_barriers[rxn_id]:.4f} eV")
    print(f"  Difference     = {barrier - dft_barriers[rxn_id]:.4f} eV")

# Save results
barrier_results = {}
for rxn_id in reactions:
    barrier_results[rxn_id] = {
        'name': reactions[rxn_id]['name'],
        'mace_barrier_eV': mace_barriers[rxn_id],
        'dft_barrier_eV': dft_barriers[rxn_id],
        'difference_eV': mace_barriers[rxn_id] - dft_barriers[rxn_id],
        'relative_error_pct': abs(mace_barriers[rxn_id] - dft_barriers[rxn_id]) / dft_barriers[rxn_id] * 100
    }

with open(os.path.join(OUTPUT_DIR, 'reaction_barriers.json'), 'w') as f:
    json.dump(barrier_results, f, indent=2)

print("\n=== Summary ===")
print(f"{'Reaction':>10s} {'MACE (eV)':>10s} {'DFT (eV)':>10s} {'Diff (eV)':>10s} {'Rel. Err.':>10s}")
print("-" * 55)
for rxn_id in reactions:
    d = barrier_results[rxn_id]
    print(f"{rxn_id:>10s} {d['mace_barrier_eV']:>10.3f} {d['dft_barrier_eV']:>10.3f} "
          f"{d['difference_eV']:>10.3f} {d['relative_error_pct']:>9.1f}%")

# MAE
mae = np.mean([abs(barrier_results[r]['difference_eV']) for r in barrier_results])
print(f"\nMAE = {mae:.3f} eV")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart comparison
rxn_labels = list(reactions.keys())
rxn_names = [reactions[r]['name'] for r in rxn_labels]
mace_vals = [mace_barriers[r] for r in rxn_labels]
dft_vals = [dft_barriers[r] for r in rxn_labels]

x = np.arange(len(rxn_labels))
width = 0.35

bars1 = axes[0].bar(x - width/2, mace_vals, width, label='MACE-MP-0', color='steelblue', edgecolor='black')
bars2 = axes[0].bar(x + width/2, dft_vals, width, label='DFT Reference', color='coral', edgecolor='black')

axes[0].set_xlabel('Reaction', fontsize=12)
axes[0].set_ylabel('Barrier Height (eV)', fontsize=12)
axes[0].set_title('Reaction Barriers: MACE-MP-0 vs DFT', fontsize=14)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{l}\n({n})' for l, n in zip(rxn_labels, rxn_names)], fontsize=9)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# Parity plot
axes[1].scatter(dft_vals, mace_vals, s=120, c='steelblue', edgecolors='black', zorder=5)
for i, label in enumerate(rxn_labels):
    axes[1].annotate(label, (dft_vals[i], mace_vals[i]), 
                    textcoords="offset points", xytext=(10, 5), fontsize=11)

# Perfect agreement line
lims = [min(min(dft_vals), min(mace_vals)) - 0.3, max(max(dft_vals), max(mace_vals)) + 0.3]
axes[1].plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel('DFT Barrier (eV)', fontsize=12)
axes[1].set_ylabel('MACE-MP-0 Barrier (eV)', fontsize=12)
axes[1].set_title(f'Parity Plot (MAE = {mae:.3f} eV)', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'reaction_barriers.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved barrier comparison plot to {os.path.join(IMAGE_DIR, 'reaction_barriers.png')}")

print("\nExperiment 3 complete!")
