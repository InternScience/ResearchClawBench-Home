"""
Plot water RDF results from MACE-MP-0 simulation
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('outputs/water_rdf_data.json', 'r') as f:
    data = json.load(f)

r_oo = np.array(data['r_oo'])
g_oo = np.array(data['g_oo'])
r_oh = np.array(data['r_oh'])
g_oh = np.array(data['g_oh'])
r_hh = np.array(data['r_hh'])
g_hh = np.array(data['g_hh'])

# Reference experimental water RDF features (approximate peak positions)
# O-O: first peak ~2.8 Å, second peak ~4.5 Å
# O-H: first peak ~1.8 Å (H-bond), second peak ~3.3 Å
# H-H: first peak ~2.4 Å, second peak ~3.8 Å

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# O-O RDF
ax = axes[0]
ax.plot(r_oo, g_oo, 'b-', linewidth=1.5, label='MACE-MP-0')
ax.axvline(x=2.8, color='r', linestyle='--', alpha=0.5, label='Exp. ~2.8 Å')
ax.set_xlabel('r (Å)', fontsize=12)
ax.set_ylabel('g_OO(r)', fontsize=12)
ax.set_title('O-O RDF', fontsize=13)
ax.set_xlim(1.0, 6.0)
ax.set_ylim(0, None)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# O-H RDF
ax = axes[1]
ax.plot(r_oh, g_oh, 'g-', linewidth=1.5, label='MACE-MP-0')
ax.axvline(x=1.8, color='r', linestyle='--', alpha=0.5, label='Exp. ~1.8 Å')
ax.set_xlabel('r (Å)', fontsize=12)
ax.set_ylabel('g_OH(r)', fontsize=12)
ax.set_title('O-H RDF', fontsize=13)
ax.set_xlim(0.5, 6.0)
ax.set_ylim(0, None)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# H-H RDF
ax = axes[2]
ax.plot(r_hh, g_hh, 'r-', linewidth=1.5, label='MACE-MP-0')
ax.axvline(x=2.4, color='b', linestyle='--', alpha=0.5, label='Exp. ~2.4 Å')
ax.set_xlabel('r (Å)', fontsize=12)
ax.set_ylabel('g_HH(r)', fontsize=12)
ax.set_title('H-H RDF', fontsize=13)
ax.set_xlim(1.0, 6.0)
ax.set_ylim(0, None)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(f'Liquid Water RDF — MACE-MP-0 MD ({data["n_water"]} H₂O, T={data["temperature"]} K)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/water_rdf.png', dpi=200, bbox_inches='tight')
print("Saved report/images/water_rdf.png")
