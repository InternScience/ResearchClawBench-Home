#!/usr/bin/env python3
"""Simple figure generation without heavy dependencies."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images', exist_ok=True)

# Load data
with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/water_rdf_results.json') as f:
    water_data = json.load(f)
with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/adsorption_energies.json') as f:
    ads_data = json.load(f)
with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/outputs/reaction_barriers.json') as f:
    barrier_data = json.load(f)

# Figure 1: Water RDF
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(water_data['r_bins_oo'], water_data['rdf_oo'], 'b-', linewidth=2)
axes[0].axvline(x=2.75, color='r', linestyle='--', alpha=0.7, label='Exp. peak')
axes[0].set_xlabel('r (Angstrom)')
axes[0].set_ylabel('g(r)')
axes[0].set_title('O-O Radial Distribution')
axes[0].set_xlim(0, 6)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(water_data['r_bins_oh'], water_data['rdf_oh'], 'b-', linewidth=2)
axes[1].axvline(x=1.85, color='r', linestyle='--', alpha=0.7, label='Exp. peak')
axes[1].set_xlabel('r (Angstrom)')
axes[1].set_ylabel('g(r)')
axes[1].set_title('O-H Radial Distribution')
axes[1].set_xlim(0, 6)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(water_data['r_bins_hh'], water_data['rdf_hh'], 'b-', linewidth=2)
axes[2].axvline(x=2.25, color='r', linestyle='--', alpha=0.7, label='Exp. peak')
axes[2].set_xlabel('r (Angstrom)')
axes[2].set_ylabel('g(r)')
axes[2].set_title('H-H Radial Distribution')
axes[2].set_xlim(0, 6)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images/figure_water_rdf.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: figure_water_rdf.png")

# Figure 2: Adsorption
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metals = ads_data['metals']
E_O = np.array(ads_data['E_O'])
E_OH = np.array(ads_data['E_OH'])

ax = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, len(metals)))
for i, metal in enumerate(metals):
    ax.scatter(E_O[i], E_OH[i], s=150, c=[colors[i]], label=metal, edgecolors='black', linewidths=1.5, zorder=3)
x_fit = np.linspace(min(E_O)-0.2, max(E_O)+0.2, 100)
slope = ads_data['scaling_relation']['slope']
intercept = ads_data['scaling_relation']['intercept']
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, 'k--', linewidth=2, label='Fit: E_OH = %.2f*E_O + %.2f' % (slope, intercept))
ax.set_xlabel('E_O (eV)', fontsize=12)
ax.set_ylabel('E_OH (eV)', fontsize=12)
ax.set_title('OH vs O Adsorption Energy Scaling (R^2 = %.4f)' % ads_data['scaling_relation']['r_squared'], fontsize=12)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
x = np.arange(len(metals))
width = 0.35
ax.bar(x - width/2, E_O, width, label='O', color='steelblue', edgecolor='black')
ax.bar(x + width/2, E_OH, width, label='OH', color='coral', edgecolor='black')
ax.set_xlabel('Metal', fontsize=12)
ax.set_ylabel('Adsorption Energy (eV)', fontsize=12)
ax.set_title('Adsorption Energies on fcc(111) Surfaces', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metals)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images/figure_adsorption_scaling.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: figure_adsorption_scaling.png")

# Figure 3: Barriers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
reactions = list(barrier_data['barriers'].keys())
names = [barrier_data['barriers'][r]['name'] for r in reactions]
mace = [barrier_data['barriers'][r]['mace_barrier'] for r in reactions]
dft = [barrier_data['barriers'][r]['dft_reference'] for r in reactions]

ax = axes[0]
x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, mace, width, label='MACE-MP-0', color='steelblue', edgecolor='black')
ax.bar(x + width/2, dft, width, label='DFT', color='coral', edgecolor='black')
ax.set_ylabel('Barrier Height (eV)', fontsize=12)
ax.set_title('Reaction Barrier Comparison', fontsize=12)
ax.set_xticks(x)
short_names = [n[:15] + '...' if len(n) > 15 else n for n in names]
ax.set_xticklabels(short_names, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.scatter(dft, mace, s=200, c='steelblue', edgecolors='black', linewidths=2, zorder=3)
min_val = min(min(dft), min(mace)) - 0.1
max_val = max(max(dft), max(mace)) + 0.1
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Parity')
for i, name in enumerate(names):
    ax.annotate(name[:10], (dft[i], mace[i]), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.set_xlabel('DFT Barrier (eV)', fontsize=12)
ax.set_ylabel('MACE-MP-0 Barrier (eV)', fontsize=12)
ax.set_title('Barrier Parity Plot (MAE = %.3f eV)' % barrier_data['statistics']['mae'], fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

plt.tight_layout()
plt.savefig('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_002_20260415_150736/report/images/figure_reaction_barriers.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: figure_reaction_barriers.png")
print("All figures generated!")
