"""
Plot reaction barrier comparison: MACE-MP-0 vs DFT
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('outputs/reaction_barriers.json', 'r') as f:
    results = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- Panel 1: Bar chart comparison ----
rxn_ids = list(results.keys())
rxn_names = [results[r]['name'] for r in rxn_ids]
mace_barriers = [results[r]['barrier_mace'] for r in rxn_ids]
dft_barriers = [results[r]['barrier_dft'] for r in rxn_ids]

x = np.arange(len(rxn_ids))
width = 0.35

bars1 = ax1.bar(x - width/2, dft_barriers, width, label='DFT Reference', color='#2196F3', edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, mace_barriers, width, label='MACE-MP-0', color='#FF5722', edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Reaction', fontsize=12)
ax1.set_ylabel('Barrier Height (eV)', fontsize=12)
ax1.set_title('Reaction Barrier Comparison', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(rxn_ids, fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='k', linewidth=0.5)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    else:
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -12), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# ---- Panel 2: Parity plot ----
dft_vals = np.array(dft_barriers)
mace_vals = np.array(mace_barriers)

# Only plot positive barriers for parity
mask = mace_vals > 0
if mask.any():
    ax2.scatter(dft_vals[mask], mace_vals[mask], s=120, zorder=5, edgecolors='black', linewidth=0.5, color='#FF5722')
    for i, rxn_id in enumerate(np.array(rxn_ids)[mask]):
        ax2.annotate(rxn_id, (dft_vals[i], mace_vals[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight='bold')

# Ideal line
lim_min = min(min(dft_vals) - 0.5, 0)
lim_max = max(max(dft_vals) + 0.5, max(mace_vals) + 0.5)
ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, alpha=0.5, label='Ideal (y=x)')

ax2.set_xlabel('DFT Barrier (eV)', fontsize=12)
ax2.set_ylabel('MACE-MP-0 Barrier (eV)', fontsize=12)
ax2.set_title('Parity Plot: Reaction Barriers', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Note about simplified geometries
ax2.text(0.05, 0.05, 'Note: Simplified geometries\n(not fully relaxed TS)',
         transform=ax2.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('CRBH20 Reaction Barriers — MACE-MP-0 vs DFT', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/reaction_barriers.png', dpi=200, bbox_inches='tight')
print("Saved report/images/reaction_barriers.png")
