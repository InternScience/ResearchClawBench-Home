"""
Create improved figures for the report.
"""
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_002_20260416_221556"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report/images")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load data
with open(os.path.join(OUTPUT_DIR, 'adsorption_energies.json')) as f:
    ads_data = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'reaction_barriers.json')) as f:
    barrier_data = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'water_rdf_data.json')) as f:
    rdf_data = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'water_md_trajectory.json')) as f:
    md_data = json.load(f)

# ============================================================
# Figure 1: Improved Reaction Barriers (SP only, cleaner)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

rxn_labels = ['Rxn 1', 'Rxn 11', 'Rxn 20']
rxn_names = ['Cyclobutene\nring-opening', 'Methoxy\ndecomposition', 'Cyclopropane\nring-opening']
mace_sp = [barrier_data[r]['mace_barrier_sp_eV'] for r in rxn_labels]
dft_ref = [barrier_data[r]['dft_barrier_eV'] for r in rxn_labels]

x = np.arange(len(rxn_labels))
width = 0.35

bars1 = axes[0].bar(x - width/2, mace_sp, width, label='MACE-MP-0 (SP)', 
                     color='steelblue', edgecolor='black', linewidth=1.2)
bars2 = axes[0].bar(x + width/2, dft_ref, width, label='DFT Reference', 
                     color='coral', edgecolor='black', linewidth=1.2)

axes[0].set_xlabel('Reaction', fontsize=12)
axes[0].set_ylabel('Barrier Height (eV)', fontsize=12)
axes[0].set_title('Reaction Barriers: MACE-MP-0 vs DFT\n(Simplified CRBH20 Geometries)', fontsize=13)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{l}\n{n}' for l, n in zip(rxn_labels, rxn_names)], fontsize=9)
axes[0].legend(fontsize=11, loc='upper left')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=0, color='black', linewidth=0.5)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    if height >= 0:
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        axes[0].text(bar.get_x() + bar.get_width()/2., height - 0.3,
                    f'{height:.2f}', ha='center', va='top', fontsize=9, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Parity plot (SP only)
colors_rxn = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (rxn, color) in enumerate(zip(rxn_labels, colors_rxn)):
    axes[1].scatter(dft_ref[i], mace_sp[i], s=150, c=color, edgecolors='black', 
                   zorder=5, linewidth=1.5, label=rxn)

all_vals = mace_sp + dft_ref
lims = [min(all_vals) - 1, max(all_vals) + 1]
axes[1].plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Perfect agreement')
axes[1].fill_between(lims, [l-0.5 for l in lims], [l+0.5 for l in lims], 
                     alpha=0.1, color='gray', label='±0.5 eV')
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel('DFT Barrier (eV)', fontsize=12)
axes[1].set_ylabel('MACE-MP-0 Barrier (eV)', fontsize=12)
mae = np.mean([abs(m - d) for m, d in zip(mace_sp, dft_ref)])
axes[1].set_title(f'Parity Plot (MAE = {mae:.2f} eV)\nNote: Simplified geometries', fontsize=13)
axes[1].legend(fontsize=10, loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'reaction_barriers.png'), dpi=150, bbox_inches='tight')
print("Saved improved reaction barriers plot")

# ============================================================
# Figure 2: Summary overview figure
# ============================================================
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: O-O RDF
ax1 = fig.add_subplot(gs[0, 0])
r = np.array(rdf_data['r'])
g_OO = np.array(rdf_data['g_OO'])
ax1.plot(r, g_OO, 'b-', linewidth=2, label='MACE-MP-0')
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=2.8, color='r', linestyle=':', alpha=0.5, label='Exp. (~2.8 Å)')
ax1.set_xlabel('r (Å)', fontsize=11)
ax1.set_ylabel('g(r)', fontsize=11)
ax1.set_title('(a) O-O RDF (330 K)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_xlim(0, 6)

# Panel B: MD Temperature
ax2 = fig.add_subplot(gs[0, 1])
steps = [i*100 for i in range(len(md_data['temperatures_K']))]
ax2.plot(steps, md_data['temperatures_K'], 'r-', linewidth=1.5)
ax2.axhline(y=330, color='k', linestyle='--', linewidth=1.5, label='Target (330 K)')
ax2.axvline(x=500, color='blue', linestyle=':', alpha=0.5, label='Equilibration')
ax2.set_xlabel('MD Step', fontsize=11)
ax2.set_ylabel('Temperature (K)', fontsize=11)
ax2.set_title('(b) MD Temperature', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

# Panel C: Adsorption Scaling
ax3 = fig.add_subplot(gs[0, 2])
metals = ['Ni', 'Cu', 'Rh', 'Pd', 'Ir', 'Pt']
colors_m = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
E_O_list = [ads_data[m]['E_ads_O'] for m in metals]
E_OH_list = [ads_data[m]['E_ads_OH'] for m in metals]

for i, metal in enumerate(metals):
    ax3.scatter(ads_data[metal]['E_ads_O'], ads_data[metal]['E_ads_OH'], 
               s=100, zorder=5, c=colors_m[i], edgecolors='black', linewidth=1)
    ax3.annotate(metal, (ads_data[metal]['E_ads_O'], ads_data[metal]['E_ads_OH']),
                textcoords="offset points", xytext=(8, 4), fontsize=10, fontweight='bold')

coeffs = np.polyfit(E_O_list, E_OH_list, 1)
R2 = np.corrcoef(E_O_list, E_OH_list)[0,1]**2
x_fit = np.linspace(min(E_O_list) - 0.5, max(E_O_list) + 0.5, 100)
y_fit = np.polyval(coeffs, x_fit)
ax3.plot(x_fit, y_fit, 'k--', alpha=0.7, linewidth=1.5)
ax3.set_xlabel('E$_{ads}$(O) (eV)', fontsize=11)
ax3.set_ylabel('E$_{ads}$(OH) (eV)', fontsize=11)
ax3.set_title(f'(c) Scaling (slope={coeffs[0]:.2f}, R²={R2:.2f})', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel D: O-H and H-H RDF
ax4 = fig.add_subplot(gs[1, 0])
g_OH = np.array(rdf_data['g_OH'])
g_HH = np.array(rdf_data['g_HH'])
# Plot only intermolecular part (r > 1.5 Å for O-H, r > 2.0 Å for H-H)
mask_oh = r > 1.3
mask_hh = r > 1.8
ax4.plot(r[mask_oh], g_OH[mask_oh], 'g-', linewidth=2, label='O-H')
ax4.plot(r[mask_hh], g_HH[mask_hh], 'r-', linewidth=2, label='H-H')
ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('r (Å)', fontsize=11)
ax4.set_ylabel('g(r)', fontsize=11)
ax4.set_title('(d) Intermolecular RDFs', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.set_xlim(1.3, 6)

# Panel E: Adsorption energies bar chart
ax5 = fig.add_subplot(gs[1, 1])
x_m = np.arange(len(metals))
w = 0.35
bars_O = ax5.bar(x_m - w/2, [ads_data[m]['E_ads_O'] for m in metals], w, 
                 label='E$_{ads}$(O)', color='steelblue', edgecolor='black')
bars_OH = ax5.bar(x_m + w/2, [ads_data[m]['E_ads_OH'] for m in metals], w, 
                  label='E$_{ads}$(OH)', color='coral', edgecolor='black')
ax5.set_xlabel('Metal', fontsize=11)
ax5.set_ylabel('Adsorption Energy (eV)', fontsize=11)
ax5.set_title('(e) Adsorption Energies', fontsize=12, fontweight='bold')
ax5.set_xticks(x_m)
ax5.set_xticklabels(metals, fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Panel F: Reaction barriers comparison
ax6 = fig.add_subplot(gs[1, 2])
rxn_short = ['Rxn 1', 'Rxn 11', 'Rxn 20']
x_r = np.arange(len(rxn_short))
w = 0.35
mace_vals = [barrier_data[r]['mace_barrier_sp_eV'] for r in rxn_short]
dft_vals = [barrier_data[r]['dft_barrier_eV'] for r in rxn_short]
ax6.bar(x_r - w/2, mace_vals, w, label='MACE-MP-0', color='steelblue', edgecolor='black')
ax6.bar(x_r + w/2, dft_vals, w, label='DFT Ref.', color='coral', edgecolor='black')
ax6.set_xlabel('Reaction', fontsize=11)
ax6.set_ylabel('Barrier (eV)', fontsize=11)
ax6.set_title('(f) Reaction Barriers', fontsize=12, fontweight='bold')
ax6.set_xticks(x_r)
ax6.set_xticklabels(rxn_short, fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')
ax6.axhline(y=0, color='black', linewidth=0.5)

plt.savefig(os.path.join(IMAGE_DIR, 'summary_overview.png'), dpi=150, bbox_inches='tight')
print("Saved summary overview figure")

# ============================================================
# Figure 3: Model architecture schematic (text-based)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')

# Draw boxes
boxes = [
    (0.5, 1.5, 2.0, 1.5, 'MPtrj Dataset\n~1.5M structures\n(DFT: PBE/PBE+U)', '#E8F5E9'),
    (3.5, 1.5, 2.0, 1.5, 'MACE Architecture\nHigher-order\nequivariant MPNN', '#E3F2FD'),
    (6.5, 1.5, 2.0, 1.5, 'MACE-MP-0\nFoundation Model\n(Universal MLIP)', '#FFF3E0'),
    (9.5, 1.5, 2.0, 1.5, 'Applications\n• Water MD\n• Catalysis\n• Reactions', '#FCE4EC'),
]

for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
for i in range(3):
    x_start = boxes[i][0] + boxes[i][2]
    x_end = boxes[i+1][0]
    y_mid = boxes[i][1] + boxes[i][3]/2
    ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_title('MACE-MP-0 Foundation Model Pipeline', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'model_pipeline.png'), dpi=150, bbox_inches='tight')
print("Saved model pipeline figure")

print("\nAll figures generated successfully!")
