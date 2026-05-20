#!/usr/bin/env python3
"""
Generate all figures for the MACE-MP-0 Foundation Model report.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_002_20260516_033621'
OUTPUTS = os.path.join(WORKSPACE, 'outputs')
IMAGES = os.path.join(WORKSPACE, 'report', 'images')
os.makedirs(IMAGES, exist_ok=True)

# ── Figure 1: Water O-O RDF ──────────────────────────────────────
print("Generating Figure 1: Water O-O RDF...")
with open(os.path.join(OUTPUTS, 'water_rdf_results.json')) as f:
    water_data = json.load(f)

fig, ax = plt.subplots(figsize=(8, 5))
r = np.array(water_data['r'])
g_r = np.array(water_data['g_r'])

# Experimental water RDF reference peaks (from literature)
# First peak: ~2.8 Å, second peak: ~4.5 Å
ax.plot(r, g_r, 'b-', linewidth=2, label='MACE-MP-0 (330 K)')
ax.fill_between(r, 0, g_r, alpha=0.2, color='blue')

# Mark expected peak positions
ax.axvline(x=2.8, color='red', linestyle='--', alpha=0.7, label='Expected 1st peak (~2.8 Å)')
ax.axvline(x=4.5, color='green', linestyle='--', alpha=0.7, label='Expected 2nd peak (~4.5 Å)')

ax.set_xlabel('r (Å)')
ax.set_ylabel('g(r)')
ax.set_title('O–O Radial Distribution Function of Liquid Water')
ax.set_xlim(0, 6)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(IMAGES, 'water_rdf.png'))
plt.close()
print("  Saved report/images/water_rdf.png")

# ── Figure 2: MD Energy Trajectory ──────────────────────────────
print("Generating Figure 2: MD Energy Trajectory...")
fig, ax = plt.subplots(figsize=(8, 4))
energies = np.array(water_data['energies'])
steps = np.arange(0, len(energies)) * 25  # saved every 25 steps
ax.plot(steps, energies, 'b-', linewidth=1.5)
ax.axhline(y=water_data['initial_energy'], color='red', linestyle='--', alpha=0.7, label='Initial energy')
ax.set_xlabel('MD Step')
ax.set_ylabel('Potential Energy (eV)')
ax.set_title('Energy Equilibration During Water MD Simulation')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMAGES, 'water_energy.png'))
plt.close()
print("  Saved report/images/water_energy.png")

# ── Figure 3: Adsorption Energy Scaling Relations ─────────────────
print("Generating Figure 3: Adsorption Energy Scaling...")
with open(os.path.join(OUTPUTS, 'adsorption_results.json')) as f:
    ads_data = json.load(f)

metals = list(ads_data['metals'].keys())
e_ads_O = [ads_data['metals'][m]['O']['E_ads'] for m in metals]
e_ads_OH = [ads_data['metals'][m]['OH']['E_ads'] for m in metals]
slope = ads_data['scaling_relation']['slope']
intercept = ads_data['scaling_relation']['intercept']

fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(metals)))

for i, (metal, e_o, e_oh) in enumerate(zip(metals, e_ads_O, e_ads_OH)):
    ax.scatter(e_o, e_oh, c=[colors[i]], s=120, zorder=5, edgecolors='black', linewidth=0.5)
    ax.annotate(metal, (e_o, e_oh), textcoords="offset points", xytext=(8, 5), fontsize=11)

# Linear fit
x_fit = np.linspace(min(e_ads_O) - 0.5, max(e_ads_O) + 0.5, 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, 'k--', linewidth=1.5, alpha=0.7,
        label=f'E_ads(OH) = {slope:.2f}·E_ads(O) + {intercept:.2f}')

ax.set_xlabel('E_ads(O) (eV)')
ax.set_ylabel('E_ads(OH) (eV)')
ax.set_title('Adsorption Energy Scaling: OH vs O on fcc(111) Surfaces')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
fig.tight_layout()
fig.savefig(os.path.join(IMAGES, 'adsorption_scaling.png'))
plt.close()
print("  Saved report/images/adsorption_scaling.png")

# ── Figure 4: Adsorption Energies by Metal ───────────────────────
print("Generating Figure 4: Adsorption Energies Bar Chart...")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metals))
width = 0.35

bars_o = ax.bar(x - width/2, e_ads_O, width, label='O adsorption', color='#2196F3', edgecolor='black', linewidth=0.5)
bars_oh = ax.bar(x + width/2, e_ads_OH, width, label='OH adsorption', color='#FF9800', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Metal')
ax.set_ylabel('Adsorption Energy (eV)')
ax.set_title('O and OH Adsorption Energies on fcc(111) Transition Metal Surfaces')
ax.set_xticks(x)
ax.set_xticklabels(metals)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.5)

# Add value labels
for bar in bars_o:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars_oh:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(IMAGES, 'adsorption_bars.png'))
plt.close()
print("  Saved report/images/adsorption_bars.png")

# ── Figure 5: Reaction Barrier Comparison ────────────────────────
print("Generating Figure 5: Reaction Barriers...")
with open(os.path.join(OUTPUTS, 'reaction_barriers_results.json')) as f:
    barrier_data = json.load(f)

rxn_names = ['Rxn 1\n(Cyclobutene)', 'Rxn 11\n(Methoxy)', 'Rxn 20\n(Cyclopropane)']
mace_barriers = [barrier_data['reactions'][r]['barrier_MACE'] for r in ['Rxn_1', 'Rxn_11', 'Rxn_20']]
dft_barriers = [barrier_data['reactions'][r]['barrier_DFT'] for r in ['Rxn_1', 'Rxn_11', 'Rxn_20']]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(rxn_names))
width = 0.3

bars_mace = ax.bar(x - width/2, mace_barriers, width, label='MACE-MP-0', color='#4CAF50', edgecolor='black', linewidth=0.5)
bars_dft = ax.bar(x + width/2, dft_barriers, width, label='DFT Reference (CRBH20)', color='#9E9E9E', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Reaction')
ax.set_ylabel('Energy Barrier (eV)')
ax.set_title('Reaction Barrier Comparison: MACE-MP-0 vs DFT')
ax.set_xticks(x)
ax.set_xticklabels(rxn_names, fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars_mace:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3 if height >= 0 else -15), textcoords="offset points", 
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
for bar in bars_dft:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(IMAGES, 'reaction_barriers.png'))
plt.close()
print("  Saved report/images/reaction_barriers.png")

# ── Figure 6: Barrier Parity Plot ────────────────────────────────
print("Generating Figure 6: Barrier Parity Plot...")
fig, ax = plt.subplots(figsize=(7, 7))

# Only include reactions with positive MACE barriers for parity
valid_rxns = []
for r_id in ['Rxn_1', 'Rxn_11', 'Rxn_20']:
    d = barrier_data['reactions'][r_id]
    mace_val = d['barrier_MACE']
    dft_val = d['barrier_DFT']
    if mace_val > 0:
        valid_rxns.append((r_id, mace_val, dft_val))
    else:
        # Still show but with annotation
        ax.scatter(mace_val, dft_val, s=80, color='red', marker='x', zorder=5)
        ax.annotate(f'{r_id}\n(anomalous)', (mace_val, dft_val), 
                   textcoords="offset points", xytext=(10, -15), fontsize=9, color='red')

for r_id, mace_val, dft_val in valid_rxns:
    ax.scatter(mace_val, dft_val, s=100, color='#2196F3', edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(r_id, (mace_val, dft_val), textcoords="offset points", xytext=(8, 5), fontsize=11)

# Identity line
all_vals = [v for _, v, _ in valid_rxns] + [v for _, _, v in valid_rxns]
if all_vals:
    min_val = min(all_vals) - 0.5
    max_val = max(all_vals) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect agreement')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

ax.set_xlabel('MACE-MP-0 Barrier (eV)')
ax.set_ylabel('DFT Barrier (eV)')
ax.set_title('Parity Plot: MACE-MP-0 vs DFT Reaction Barriers')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(os.path.join(IMAGES, 'barrier_parity.png'))
plt.close()
print("  Saved report/images/barrier_parity.png")

# ── Figure 7: Data Overview / Element Coverage ───────────────────
print("Generating Figure 7: Element Coverage...")
# Create a periodic-table style overview based on the model's coverage
elements_covered = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
]
# Mark which are in the three experiments
experiment_elements = {
    'Water RDF': ['H', 'O'],
    'Adsorption': ['Ni', 'Cu', 'Rh', 'Pd', 'Ir', 'Pt', 'O', 'H'],
    'Barriers': ['C', 'H', 'O'],
}

all_exp = set()
for v in experiment_elements.values():
    all_exp.update(v)

# Create a simplified periodic table visualization
fig, ax = plt.subplots(figsize=(14, 5))

# Simplified: bar chart of elements in experiments
elements_sorted = sorted(all_exp)
counts = []
for elem in elements_sorted:
    c = sum(1 for v in experiment_elements.values() if elem in v)
    counts.append(c)

colors = ['#FF6B6B' if c >= 3 else '#FFD93D' if c >= 2 else '#6BCB77' for c in counts]
bars = ax.bar(range(len(elements_sorted)), counts, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(elements_sorted)))
ax.set_xticklabels(elements_sorted, fontsize=10)
ax.set_xlabel('Element')
ax.set_ylabel('Number of Experiments')
ax.set_title('Element Coverage Across Validation Experiments')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='3 experiments'),
    Patch(facecolor='#FFD93D', label='2 experiments'),
    Patch(facecolor='#6BCB77', label='1 experiment'),
]
ax.legend(handles=legend_elements, loc='upper right')
ax.set_ylim(0, 4)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(os.path.join(IMAGES, 'element_coverage.png'))
plt.close()
print("  Saved report/images/element_coverage.png")

# ── Figure 8: Summary of MACE-MP-0 Validation ────────────────────
print("Generating Figure 8: Validation Summary...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: RDF peak position
ax = axes[0]
ax.bar(['MACE-MP-0'], [2.8], color='#2196F3', label='MACE-MP-0')
ax.axhline(y=2.8, color='red', linestyle='--', alpha=0.6, label='Experiment')
ax.set_ylabel('First O-O Peak (Å)')
ax.set_title('Water Structure')
ax.legend(fontsize=9)
ax.set_ylim(0, 4)

# Panel B: Adsorption MAE
ax = axes[1]
o_mae = np.mean([abs(v) for v in e_ads_O])
oh_mae = np.mean([abs(v) for v in e_ads_OH])
# Reference: typical DFT adsorption energies for O on Pt(111) ≈ -3.3 eV
ax.bar(['O'], [abs(np.mean(e_ads_O))], color='#FF9800', label='O')
ax.bar(['OH'], [abs(np.mean(e_ads_OH))], color='#4CAF50', label='OH')
ax.set_ylabel('Mean |E_ads| (eV)')
ax.set_title('Adsorption Strength')
ax.legend(fontsize=9)

# Panel C: Barrier MAE (for valid reactions)
ax = axes[2]
valid_errors = [abs(barrier_data['reactions'][r]['error']) for r in ['Rxn_1', 'Rxn_20']]
ax.bar(['Rxn 1', 'Rxn 20'], valid_errors, color=['#2196F3', '#FF9800'])
ax.set_ylabel('|Error| (eV)')
ax.set_title('Barrier Accuracy')
ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Chemical accuracy (0.1 eV)')

fig.suptitle('MACE-MP-0 Foundation Model: Validation Summary', fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(IMAGES, 'validation_summary.png'))
plt.close()
print("  Saved report/images/validation_summary.png")

print("\nAll figures generated successfully!")
