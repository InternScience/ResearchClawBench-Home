"""
Plot adsorption energy scaling relations from MACE-MP-0
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

with open('outputs/adsorption_results.json', 'r') as f:
    results = json.load(f)

# Extract data
metals_list = []
e_ads_O = []
e_ads_OH = []

for metal in ['Ni', 'Cu', 'Rh', 'Pd', 'Ir', 'Pt']:
    if metal in results and results[metal]['E_ads_O'] is not None and results[metal]['E_ads_OH'] is not None:
        metals_list.append(metal)
        e_ads_O.append(results[metal]['E_ads_O'])
        e_ads_OH.append(results[metal]['E_ads_OH'])

e_ads_O = np.array(e_ads_O)
e_ads_OH = np.array(e_ads_OH)

# Fit linear scaling relation: E_ads(OH) = a * E_ads(O) + b
# Exclude Ni outlier for the fit (or include all)
coeffs = np.polyfit(e_ads_O, e_ads_OH, 1)
slope = coeffs[0]
intercept = coeffs[1]

# DFT reference scaling relation (Abild-Pedersen et al., typical: slope ~0.5)
dft_slope = 0.50
dft_intercept = 0.10  # approximate

fig, ax = plt.subplots(figsize=(7, 6))

# Plot MACE-MP-0 data points
for i, metal in enumerate(metals_list):
    ax.scatter(e_ads_O[i], e_ads_OH[i], s=100, zorder=5, edgecolors='black', linewidth=0.5)
    ax.annotate(metal, (e_ads_O[i], e_ads_OH[i]), 
                textcoords="offset points", xytext=(8, 5), fontsize=11, fontweight='bold')

# Plot MACE-MP-0 scaling line
x_fit = np.linspace(min(e_ads_O) - 0.3, max(e_ads_O) + 0.3, 100)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, 'b-', linewidth=2, label=f'MACE-MP-0 fit (slope={slope:.2f})')

# Plot DFT reference scaling line
y_dft = dft_slope * x_fit + dft_intercept
ax.plot(x_fit, y_dft, 'r--', linewidth=2, label=f'DFT typical (slope≈{dft_slope:.2f})')

ax.set_xlabel('E$_{ads}$(O*) (eV)', fontsize=13)
ax.set_ylabel('E$_{ads}$(OH*) (eV)', fontsize=13)
ax.set_title('Adsorption Energy Scaling Relations\nO* vs OH* on fcc(111) Surfaces', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

# Add R² value
r_squared = 1 - np.sum((e_ads_OH - (slope * e_ads_O + intercept))**2) / np.sum((e_ads_OH - np.mean(e_ads_OH))**2)
ax.text(0.05, 0.05, f'R² = {r_squared:.3f}', transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('report/images/adsorption_scaling.png', dpi=200, bbox_inches='tight')
print("Saved report/images/adsorption_scaling.png")

# Save summary
summary = {
    'metals': metals_list,
    'E_ads_O': e_ads_O.tolist(),
    'E_ads_OH': e_ads_OH.tolist(),
    'scaling_slope': float(slope),
    'scaling_intercept': float(intercept),
    'R_squared': float(r_squared),
}
with open('outputs/adsorption_scaling_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Scaling relation: E_ads(OH) = {slope:.3f} * E_ads(O) + {intercept:.3f}")
print(f"R² = {r_squared:.4f}")
