"""
Phase 7: Design Rules & De Novo Formulation Proposal
Generate concrete formulation proposals targeting >1 MPa adhesion.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load training data
df = pd.read_csv('outputs/training_data_184.csv')
monomer_cols = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
target_col = 'Glass (kPa)_10s'
short_names = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']

X_train = df[monomer_cols].values
y_train = df[target_col].values

# Train models
rfr = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rfr.fit(X_train, y_train)

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, normalize_y=True)
gp.fit(X_train, y_train)

# ============================================================
# Analyze top-performing formulations from initial data
# ============================================================
top_10_pct = df.nlargest(int(len(df) * 0.1), target_col)
print("=== Top 10% Formulations (Initial Data) ===")
print(f"  Count: {len(top_10_pct)}")
print(f"  Strength range: {top_10_pct[target_col].min():.1f} - {top_10_pct[target_col].max():.1f} kPa")
print(f"  Mean composition:")
for col, sname in zip(monomer_cols, short_names):
    print(f"    {sname}: {top_10_pct[col].mean():.3f} ± {top_10_pct[col].std():.3f}")

# ============================================================
# Design rules from analysis
# ============================================================
design_rules = {
    'high_adhesion_composition': {
        'HEA': 'Low (<0.15) - negatively correlated with adhesion',
        'BA': 'High (0.45-0.65) - primary driver of adhesion',
        'CBEA': 'Low (<0.10) - weakly negative correlation',
        'ATAC': 'Moderate (0.05-0.20) - important for charge interactions',
        'PEA': 'High (0.20-0.40) - aromatic interactions enhance adhesion',
        'AAm': 'Low (<0.10) - minimal contribution'
    },
    'key_principles': [
        'Maximize hydrophobic (BA) and aromatic (PEA) content for strong interfacial adhesion',
        'Minimize nucleophilic (HEA) content which reduces adhesion',
        'Include moderate cationic (ATAC) content for electrostatic interactions',
        'BA + PEA should constitute >60% of total composition',
        'Low CBEA and AAm fractions minimize dilution of adhesive functionality'
    ]
}

with open('outputs/design_rules.json', 'w') as f:
    json.dump(design_rules, f, indent=2)

# ============================================================
# Generate targeted formulations using constrained optimization
# ============================================================
np.random.seed(42)
n_proposals = 100000

# Strategy: Focus on high BA and PEA, low HEA
# Use Dirichlet with concentration parameters favoring BA and PEA
alpha = np.array([0.5, 3.0, 0.3, 1.0, 2.0, 0.2])  # HEA, BA, CBEA, ATAC, PEA, AAm
proposals = np.random.dirichlet(alpha, size=n_proposals)

# Predict
rfr_preds = rfr.predict(proposals)
gp_preds, gp_stds = gp.predict(proposals, return_std=True)

# Select top candidates
top_idx = np.argsort(rfr_preds)[::-1][:20]

print("\n=== Top 20 Proposed Formulations ===")
proposed_formulations = []
for rank, idx in enumerate(top_idx):
    comp = proposals[idx]
    form = {
        'rank': rank + 1,
        'HEA': round(float(comp[0]), 4),
        'BA': round(float(comp[1]), 4),
        'CBEA': round(float(comp[2]), 4),
        'ATAC': round(float(comp[3]), 4),
        'PEA': round(float(comp[4]), 4),
        'AAm': round(float(comp[5]), 4),
        'RFR_predicted_kPa': round(float(rfr_preds[idx]), 1),
        'GP_predicted_kPa': round(float(gp_preds[idx]), 1),
        'GP_std_kPa': round(float(gp_stds[idx]), 1)
    }
    proposed_formulations.append(form)
    if rank < 5:
        print(f"  #{rank+1}: RFR={rfr_preds[idx]:.1f} kPa, GP={gp_preds[idx]:.1f}±{gp_stds[idx]:.1f} kPa")
        print(f"        HEA={comp[0]:.3f}, BA={comp[1]:.3f}, CBEA={comp[2]:.3f}, ATAC={comp[3]:.3f}, PEA={comp[4]:.3f}, AAm={comp[5]:.3f}")

with open('outputs/proposed_formulations.json', 'w') as f:
    json.dump(proposed_formulations, f, indent=2)

# ============================================================
# Figure 18: Proposed formulation composition profile
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top 5 formulations as stacked bar
top5 = proposed_formulations[:5]
x = np.arange(5)
bottom = np.zeros(5)
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
for i, (sname, color) in enumerate(zip(short_names, colors)):
    vals = [f[sname] for f in top5]
    axes[0].bar(x, vals, bottom=bottom, label=sname, color=color, alpha=0.8)
    bottom += np.array(vals)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'#{i+1}' for i in range(5)], fontsize=11)
axes[0].set_ylabel('Fraction', fontsize=12)
axes[0].set_title('Top 5 Proposed Formulations', fontsize=13)
axes[0].legend(loc='upper right', fontsize=9)

# Comparison: initial mean vs optimized mean
init_mean = df[monomer_cols].mean().values
opt_mean = np.mean([list(f.values())[1:7] for f in top5], axis=0)
# Convert to float
opt_mean = np.array([float(v) for v in opt_mean])

x2 = np.arange(len(short_names))
width = 0.35
axes[1].bar(x2 - width/2, init_mean, width, label='Initial Mean', color='#2196F3', alpha=0.8)
axes[1].bar(x2 + width/2, opt_mean, width, label='Optimized Mean', color='#F44336', alpha=0.8)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(short_names, fontsize=11)
axes[1].set_ylabel('Mean Fraction', fontsize=12)
axes[1].set_title('Composition Shift: Initial → Optimized', fontsize=13)
axes[1].legend()

plt.tight_layout()
plt.savefig('report/images/fig18_proposed_formulations.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 18 saved.")

# ============================================================
# Figure 19: Predicted strength landscape (BA vs PEA)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Create a grid over BA and PEA, fixing other monomers at optimal values
n_grid = 50
ba_range = np.linspace(0.1, 0.7, n_grid)
pea_range = np.linspace(0.05, 0.45, n_grid)
BA, PEA = np.meshgrid(ba_range, pea_range)

# Fix other monomers at low values
grid_comps = np.zeros((n_grid * n_grid, 6))
grid_comps[:, 1] = BA.ravel()  # BA
grid_comps[:, 4] = PEA.ravel()  # PEA
grid_comps[:, 3] = 0.10  # ATAC
grid_comps[:, 2] = 0.05  # CBEA
grid_comps[:, 5] = 0.02  # AAm
# HEA gets the remainder
grid_comps[:, 0] = 1.0 - grid_comps.sum(axis=1)
grid_comps[:, 0] = np.maximum(grid_comps[:, 0], 0)

# Normalize
grid_comps = grid_comps / grid_comps.sum(axis=1, keepdims=True)

# Predict
grid_preds = rfr.predict(grid_comps)
Z = grid_preds.reshape(n_grid, n_grid)

# Plot
contour = ax.contourf(BA, PEA, Z, levels=20, cmap='YlOrRd')
plt.colorbar(contour, ax=ax, label='Predicted Adhesive Strength (kPa)')

# Overlay top formulations
for f in top5[:5]:
    ax.scatter(f['BA'], f['PEA'], s=100, c='blue', edgecolors='white', zorder=5, marker='*')

# Overlay initial data
ax.scatter(df['Hydrophobic-BA'], df['Aromatic-PEA'], c='black', alpha=0.3, s=15, label='Initial Data')

ax.set_xlabel('Hydrophobic-BA Fraction', fontsize=12)
ax.set_ylabel('Aromatic-PEA Fraction', fontsize=12)
ax.set_title('Predicted Adhesive Strength Landscape (BA vs PEA)', fontsize=13)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('report/images/fig19_strength_landscape.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 19 saved.")

# ============================================================
# Figure 20: Pathway to >1 MPa - extrapolation analysis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Sort training data by strength
sorted_y = np.sort(y_train)[::-1]
cummax = np.maximum.accumulate(sorted_y[::-1])[::-1]

# Plot
ax.plot(range(len(sorted_y)), sorted_y, 'o-', markersize=3, color='#2196F3', alpha=0.7)
ax.axhline(y=1000, color='red', linestyle='--', lw=2, label='1 MPa Target')
ax.axhline(y=y_train.max(), color='gray', linestyle=':', lw=1, label=f'Max observed ({y_train.max():.0f} kPa)')

# Add annotation for gap
ax.annotate(f'Gap to 1 MPa: {1000-y_train.max():.0f} kPa\n({(1000/y_train.max()-1)*100:.0f}% improvement needed)',
            xy=(0, y_train.max()), xytext=(50, 600),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=11, color='red')

ax.set_xlabel('Sample Rank', fontsize=12)
ax.set_ylabel('Adhesive Strength (kPa)', fontsize=12)
ax.set_title('Gap Analysis: Current Performance vs 1 MPa Target', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('report/images/fig20_gap_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 20 saved.")

print("\nPhase 7 complete.")
