#!/usr/bin/env python3
"""
Extrapolation analysis: use trained GP to explore composition space densely
and identify formulations predicted to approach/exceed 1 MPa.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from itertools import product

np.random.seed(42)
sns.set_style("whitegrid")

# Load data
df = pd.read_csv("outputs/df184_clean.csv")
monomers = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
X = df[monomers].values
y = df['Glass (kPa)_10s'].values

# Train GP on full data
gp = GaussianProcessRegressor(
    kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
    n_restarts_optimizer=10, random_state=42, normalize_y=True
)
gp.fit(X, y)
print("GP trained on full 184 data.")

# Dense grid search with step 0.05 for 4 key monomers, keeping others fixed
# We'll do a smarter sampling: sample 200,000 random compositions that sum to 1
n_samples = 200000
rand_comp = np.random.dirichlet(np.ones(6), size=n_samples)
# Ensure no negative values (dirichlet always positive)
rand_df = pd.DataFrame(rand_comp, columns=monomers)

# Predict with GP
y_pred, y_std = gp.predict(rand_df.values, return_std=True)
rand_df['GP_pred'] = y_pred
rand_df['GP_std'] = y_std
rand_df['GP_ucb'] = y_pred + 1.96 * y_std  # Upper confidence bound

# Sort by prediction
best_pred = rand_df.nlargest(20, 'GP_pred')
best_ucb = rand_df.nlargest(20, 'GP_ucb')

print("\n=== Top 20 predicted by GP mean ===")
print(best_pred[monomers + ['GP_pred', 'GP_std']].to_string())

print("\n=== Top 20 predicted by GP UCB ===")
print(best_ucb[monomers + ['GP_ucb', 'GP_pred', 'GP_std']].to_string())

# Save
best_pred.to_csv("outputs/top20_gp_prediction.csv", index=False)
best_ucb.to_csv("outputs/top20_gp_ucb.csv", index=False)

# Overall max
max_pred = rand_df['GP_pred'].max()
max_ucb = rand_df['GP_ucb'].max()
print(f"\nMaximum GP prediction: {max_pred:.2f} kPa")
print(f"Maximum GP UCB: {max_ucb:.2f} kPa")

# Check how many are predicted > 1000 kPa
n_above_1MPa = (rand_df['GP_pred'] > 1000).sum()
n_above_500 = (rand_df['GP_pred'] > 500).sum()
print(f"Predictions > 1000 kPa: {n_above_1MPa} / {n_samples}")
print(f"Predictions > 500 kPa: {n_above_500} / {n_samples}")

# Figure 15: GP predicted landscape - 2D slices
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# For each monomer, show predicted strength vs that monomer (marginal)
for i, mon in enumerate(monomers):
    ax = axes[i]
    # Bin the monomer and compute mean prediction per bin
    bins = np.linspace(0, 1, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    for j in range(len(bins)-1):
        mask = (rand_df[mon] >= bins[j]) & (rand_df[mon] < bins[j+1])
        if mask.sum() > 0:
            bin_means.append(rand_df.loc[mask, 'GP_pred'].mean())
            bin_stds.append(rand_df.loc[mask, 'GP_pred'].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    ax.plot(bin_centers, bin_means, color='steelblue', linewidth=2)
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, alpha=0.3, color='steelblue')
    ax.axhline(1000, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel(f'{mon} fraction')
    ax.set_ylabel('Mean GP Predicted Strength (kPa)')
    ax.set_title(f'Marginal Landscape: {mon}')
    ax.set_ylim(0, max(1200, max_pred*1.1))

plt.tight_layout()
plt.savefig("report/images/fig15_gp_landscape.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig15_gp_landscape.png")

# Figure 16: Heatmap of predicted strength vs 2 most important monomers
# Based on feature importance: Hydrophobic-BA and Aromatic-PEA
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap 1: Hydrophobic-BA vs Aromatic-PEA
ax = axes[0]
# Fix other monomers at their median values
fixed_vals = rand_df[monomers].median().to_dict()
# Create 2D grid
ba_vals = np.linspace(0, 0.8, 41)
pea_vals = np.linspace(0, 0.6, 31)
Z = np.zeros((len(pea_vals), len(ba_vals)))
for ii, ba in enumerate(ba_vals):
    for jj, pea in enumerate(pea_vals):
        comp = fixed_vals.copy()
        comp['Hydrophobic-BA'] = ba
        comp['Aromatic-PEA'] = pea
        # Normalize remaining 4 monomers to sum to 1 - ba - pea
        others = ['Nucleophilic-HEA', 'Acidic-CBEA', 'Cationic-ATAC', 'Amide-AAm']
        rem = 1 - ba - pea
        if rem < 0:
            Z[jj, ii] = np.nan
            continue
        # Scale others proportionally
        orig_sum = sum(comp[o] for o in others)
        for o in others:
            comp[o] = comp[o] / orig_sum * rem if orig_sum > 0 else rem / len(others)
        comp_arr = np.array([[comp[m] for m in monomers]])
        Z[jj, ii] = gp.predict(comp_arr)[0]

im = ax.imshow(Z, aspect='auto', origin='lower', cmap='viridis',
               extent=[ba_vals.min(), ba_vals.max(), pea_vals.min(), pea_vals.max()])
ax.set_xlabel('Hydrophobic-BA fraction')
ax.set_ylabel('Aromatic-PEA fraction')
ax.set_title('GP Predicted Strength Landscape')
plt.colorbar(im, ax=ax, label='Predicted Strength (kPa)')

# Heatmap 2: Nucleophilic-HEA vs Hydrophobic-BA
ax = axes[1]
hea_vals = np.linspace(0, 0.8, 41)
ba_vals2 = np.linspace(0, 0.8, 41)
Z2 = np.zeros((len(ba_vals2), len(hea_vals)))
for ii, hea in enumerate(hea_vals):
    for jj, ba in enumerate(ba_vals2):
        comp = fixed_vals.copy()
        comp['Nucleophilic-HEA'] = hea
        comp['Hydrophobic-BA'] = ba
        others = ['Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
        rem = 1 - hea - ba
        if rem < 0:
            Z2[jj, ii] = np.nan
            continue
        orig_sum = sum(comp[o] for o in others)
        for o in others:
            comp[o] = comp[o] / orig_sum * rem if orig_sum > 0 else rem / len(others)
        comp_arr = np.array([[comp[m] for m in monomers]])
        Z2[jj, ii] = gp.predict(comp_arr)[0]

im2 = ax.imshow(Z2, aspect='auto', origin='lower', cmap='viridis',
                extent=[hea_vals.min(), hea_vals.max(), ba_vals2.min(), ba_vals2.max()])
ax.set_xlabel('Nucleophilic-HEA fraction')
ax.set_ylabel('Hydrophobic-BA fraction')
ax.set_title('GP Predicted Strength Landscape')
plt.colorbar(im2, ax=ax, label='Predicted Strength (kPa)')

plt.tight_layout()
plt.savefig("report/images/fig16_heatmaps.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig16_heatmaps.png")

# Save summary
extrap_summary = {
    "max_gp_prediction_kPa": float(max_pred),
    "max_gp_ucb_kPa": float(max_ucb),
    "n_above_1MPa": int(n_above_1MPa),
    "n_above_500kPa": int(n_above_500),
    "total_samples": n_samples,
    "top_mean_composition": {m: float(best_pred.iloc[0][m]) for m in monomers},
    "top_ucb_composition": {m: float(best_ucb.iloc[0][m]) for m in monomers},
}
with open("outputs/extrapolation_summary.json", "w") as f:
    json.dump(extrap_summary, f, indent=2)
print("Saved outputs/extrapolation_summary.json")
