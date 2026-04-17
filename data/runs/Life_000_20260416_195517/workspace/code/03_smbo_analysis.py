#!/usr/bin/env python3
"""
Phase 3: Bayesian Optimization (SMBO) Analysis
Sequential Model-Based Optimization for hydrogel formulation discovery
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

BASE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_000_20260416_195517"

# Load data
df_train = pd.read_excel(f"{BASE}/data/184_verified_Original Data_ML_20230926.xlsx", sheet_name='Data_to_HU')
features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
target = 'Glass (kPa)_10s'

# Load optimization data (3 rounds)
df_opt_ei = pd.read_excel(f"{BASE}/data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='EI')
df_opt_pred = pd.read_excel(f"{BASE}/data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='PRED')

# Forward fill ML method names
df_opt_ei['ML'] = df_opt_ei['ML'].ffill()
df_opt_pred['ML'] = df_opt_pred['ML'].ffill()

# Convert Glass (kPa)_max to numeric
df_opt_ei['Glass (kPa)_max'] = pd.to_numeric(df_opt_ei['Glass (kPa)_max'], errors='coerce')
df_opt_pred['Glass (kPa)_max'] = pd.to_numeric(df_opt_pred['Glass (kPa)_max'], errors='coerce')

X_train = df_train[features].values
y_train = df_train[target].values

# ========== Implement SMBO with Expected Improvement ==========
print("=== Implementing RFR-GP SMBO ===")

def expected_improvement(X_new, gp_model, y_best, xi=0.01):
    """Calculate Expected Improvement for GP model."""
    mu, sigma = gp_model.predict(X_new, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei, mu, sigma

# Train RFR model
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Train GP model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=True)
gp.fit(X_scaled, y_train)

# Generate candidate formulations (compositional constraint: sum = 1)
print("Generating candidate formulations...")
np.random.seed(42)
n_candidates = 50000

# Use Dirichlet distribution to generate valid compositions
candidates = np.random.dirichlet(np.ones(6), size=n_candidates)

# Also add focused candidates near top performers
top_idx = np.argsort(y_train)[-20:]
for idx in top_idx:
    # Perturb top formulations
    for _ in range(100):
        perturbation = np.random.normal(0, 0.03, 6)
        new_comp = X_train[idx] + perturbation
        new_comp = np.maximum(new_comp, 0)
        new_comp = new_comp / new_comp.sum()
        candidates = np.vstack([candidates, new_comp])

print(f"Total candidates: {candidates.shape[0]}")

# RFR predictions for all candidates
rf_preds = rf.predict(candidates)

# GP predictions with uncertainty
candidates_scaled = scaler.transform(candidates)
gp_mu, gp_sigma = gp.predict(candidates_scaled, return_std=True)

# Expected Improvement
y_best = y_train.max()
ei_values, _, _ = expected_improvement(candidates_scaled, gp, y_best)

# Top by EI
top_ei_idx = np.argsort(ei_values)[::-1][:20]
# Top by prediction
top_pred_idx = np.argsort(rf_preds)[::-1][:20]

print(f"\n=== Top 10 by Expected Improvement ===")
print(f"{'Rank':>4} {'HEA':>6} {'BA':>6} {'CBEA':>6} {'ATAC':>6} {'PEA':>6} {'AAm':>6} {'EI':>10} {'RF_Pred':>10} {'GP_Pred':>10} {'GP_Std':>8}")
for i, idx in enumerate(top_ei_idx[:10]):
    comp = candidates[idx]
    print(f"{i+1:4d} {comp[0]:6.3f} {comp[1]:6.3f} {comp[2]:6.3f} {comp[3]:6.3f} {comp[4]:6.3f} {comp[5]:6.3f} {ei_values[idx]:10.2f} {rf_preds[idx]:10.2f} {gp_mu[idx]:10.2f} {gp_sigma[idx]:8.2f}")

print(f"\n=== Top 10 by RF Prediction ===")
for i, idx in enumerate(top_pred_idx[:10]):
    comp = candidates[idx]
    print(f"{i+1:4d} {comp[0]:6.3f} {comp[1]:6.3f} {comp[2]:6.3f} {comp[3]:6.3f} {comp[4]:6.3f} {comp[5]:6.3f} {ei_values[idx]:10.2f} {rf_preds[idx]:10.2f} {gp_mu[idx]:10.2f} {gp_sigma[idx]:8.2f}")

# Save SMBO results
smbo_results = {
    'top_ei': [],
    'top_pred': []
}
for idx in top_ei_idx[:20]:
    smbo_results['top_ei'].append({
        'composition': candidates[idx].tolist(),
        'EI': float(ei_values[idx]),
        'RF_pred': float(rf_preds[idx]),
        'GP_pred': float(gp_mu[idx]),
        'GP_std': float(gp_sigma[idx])
    })
for idx in top_pred_idx[:20]:
    smbo_results['top_pred'].append({
        'composition': candidates[idx].tolist(),
        'EI': float(ei_values[idx]),
        'RF_pred': float(rf_preds[idx]),
        'GP_pred': float(gp_mu[idx]),
        'GP_std': float(gp_sigma[idx])
    })

with open(f"{BASE}/outputs/smbo_results.json", 'w') as f:
    json.dump(smbo_results, f, indent=2)

# ========== Figure 10: SMBO Optimization Landscape ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# EI vs GP prediction
ax = axes[0, 0]
scatter = ax.scatter(gp_mu[:5000], ei_values[:5000], c=rf_preds[:5000], cmap='viridis', 
                    alpha=0.3, s=5)
ax.scatter(gp_mu[top_ei_idx[:10]], ei_values[top_ei_idx[:10]], c='red', s=100, marker='*', 
          zorder=5, label='Top EI')
plt.colorbar(scatter, ax=ax, label='RF Prediction (kPa)')
ax.set_xlabel('GP Mean Prediction (kPa)')
ax.set_ylabel('Expected Improvement')
ax.set_title('EI vs GP Prediction')
ax.legend()

# GP uncertainty vs prediction
ax = axes[0, 1]
scatter = ax.scatter(gp_mu[:5000], gp_sigma[:5000], c=ei_values[:5000], cmap='hot', 
                    alpha=0.3, s=5)
ax.scatter(gp_mu[top_ei_idx[:10]], gp_sigma[top_ei_idx[:10]], c='blue', s=100, marker='*',
          zorder=5, label='Top EI')
plt.colorbar(scatter, ax=ax, label='Expected Improvement')
ax.set_xlabel('GP Mean Prediction (kPa)')
ax.set_ylabel('GP Uncertainty (σ)')
ax.set_title('Exploration-Exploitation Trade-off')
ax.legend()

# RF vs GP predictions
ax = axes[1, 0]
ax.scatter(rf_preds[:5000], gp_mu[:5000], alpha=0.2, s=5, color='gray')
ax.scatter(rf_preds[top_ei_idx[:10]], gp_mu[top_ei_idx[:10]], c='red', s=100, marker='*',
          zorder=5, label='Top EI')
ax.scatter(rf_preds[top_pred_idx[:10]], gp_mu[top_pred_idx[:10]], c='blue', s=100, marker='^',
          zorder=5, label='Top RF Pred')
lims = [0, max(rf_preds.max(), gp_mu.max()) + 10]
ax.plot(lims, lims, 'k--', alpha=0.5)
ax.set_xlabel('RF Prediction (kPa)')
ax.set_ylabel('GP Prediction (kPa)')
ax.set_title('RF vs GP Model Agreement')
ax.legend()

# Distribution of top candidates
ax = axes[1, 1]
top_comps = candidates[top_ei_idx[:20]]
mean_top = top_comps.mean(axis=0)
std_top = top_comps.std(axis=0)
x_pos = np.arange(len(features))
bars = ax.bar(x_pos, mean_top, yerr=std_top, capsize=5, 
              color=['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#795548'],
              alpha=0.8, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm'])
ax.set_ylabel('Mean Fraction')
ax.set_title('Composition of Top 20 EI Candidates')

plt.suptitle('Sequential Model-Based Optimization (SMBO) Analysis', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig10_smbo_landscape.png")
plt.close()
print("\nFigure 10 saved: SMBO landscape")

# ========== Figure 11: Optimization Trajectory (from experimental data) ==========
# Analyze the actual experimental optimization rounds
print("\n=== Analyzing Optimization Rounds ===")

# Categorize methods by round
round1_ei_methods = ['RFR-RFR', 'RFR-GP', 'GP-GP', 'GP-RFR', 'ENU-RFR', 'ENU-GP', 'CLMax', 'CLMin', 'LP_df', 'old-SM-GP']
round2_ei_methods = ['RFR-GP-2rd-ei', 'GP-GP-2rd-ei']
round3_ei_methods = ['RFR-GP-3rd-ei', 'GP-GP-3rd-ei']

round1_pred_methods = ['RFR-GP', 'GP-GP', 'old-SM-GP', 'SM-ETR', 'SM-GBM']
round2_pred_methods = ['RFR-GP-2rd-pred', 'GP-GP-2rd-pred']
round3_pred_methods = ['RFR-GP-3rd-pred', 'GP-GP-3rd-pred']

# Compute stats per round
rounds_data = []

# Training data baseline
rounds_data.append({
    'Round': 'Initial (n=184)',
    'Mean': y_train.mean(),
    'Max': y_train.max(),
    'Std': y_train.std(),
    'N': len(y_train),
    'Type': 'Training'
})

# Round 1 EI
r1_ei = df_opt_ei[df_opt_ei['ML'].isin(round1_ei_methods)]
r1_vals = r1_ei['Glass (kPa)_max'].dropna()
rounds_data.append({
    'Round': 'Round 1 (EI)',
    'Mean': r1_vals.mean(),
    'Max': r1_vals.max(),
    'Std': r1_vals.std(),
    'N': len(r1_vals),
    'Type': 'EI'
})

# Round 1 PRED
r1_pred = df_opt_pred[df_opt_pred['ML'].isin(round1_pred_methods)]
r1p_vals = r1_pred['Glass (kPa)_max'].dropna()
rounds_data.append({
    'Round': 'Round 1 (PRED)',
    'Mean': r1p_vals.mean(),
    'Max': r1p_vals.max(),
    'Std': r1p_vals.std(),
    'N': len(r1p_vals),
    'Type': 'PRED'
})

# Round 2 EI
r2_ei = df_opt_ei[df_opt_ei['ML'].isin(round2_ei_methods)]
r2_vals = r2_ei['Glass (kPa)_max'].dropna()
rounds_data.append({
    'Round': 'Round 2 (EI)',
    'Mean': r2_vals.mean(),
    'Max': r2_vals.max(),
    'Std': r2_vals.std(),
    'N': len(r2_vals),
    'Type': 'EI'
})

# Round 2 PRED
r2_pred = df_opt_pred[df_opt_pred['ML'].isin(round2_pred_methods)]
r2p_vals = r2_pred['Glass (kPa)_max'].dropna()
rounds_data.append({
    'Round': 'Round 2 (PRED)',
    'Mean': r2p_vals.mean(),
    'Max': r2p_vals.max(),
    'Std': r2p_vals.std(),
    'N': len(r2p_vals),
    'Type': 'PRED'
})

# Round 3 EI
r3_ei = df_opt_ei[df_opt_ei['ML'].isin(round3_ei_methods)]
r3_vals = r3_ei['Glass (kPa)_max'].dropna()
rounds_data.append({
    'Round': 'Round 3 (EI)',
    'Mean': r3_vals.mean(),
    'Max': r3_vals.max(),
    'Std': r3_vals.std(),
    'N': len(r3_vals),
    'Type': 'EI'
})

# Round 3 PRED
r3_pred = df_opt_pred[df_opt_pred['ML'].isin(round3_pred_methods)]
r3p_vals = r3_pred['Glass (kPa)_max'].dropna()
rounds_data.append({
    'Round': 'Round 3 (PRED)',
    'Mean': r3p_vals.mean(),
    'Max': r3p_vals.max(),
    'Std': r3p_vals.std(),
    'N': len(r3p_vals),
    'Type': 'PRED'
})

rounds_df = pd.DataFrame(rounds_data)
print(rounds_df.to_string(index=False))
rounds_df.to_csv(f"{BASE}/outputs/optimization_rounds.csv", index=False)

# Figure 11: Optimization trajectory
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mean adhesive strength across rounds
ax = axes[0]
x_labels = rounds_df['Round'].values
x_pos = np.arange(len(x_labels))
colors_round = ['gray'] + ['#2196F3', '#FF9800'] * 3 + ['#4CAF50']
bar_colors = ['gray', '#2196F3', '#FF9800', '#2196F3', '#FF9800', '#2196F3', '#FF9800']
bars = ax.bar(x_pos, rounds_df['Mean'], yerr=rounds_df['Std'], capsize=4,
              color=bar_colors[:len(x_pos)], alpha=0.8, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean Adhesive Strength (kPa)')
ax.set_title('Mean Adhesive Strength Across Optimization Rounds')
ax.axhline(1000, color='red', linestyle='--', linewidth=1.5, label='1 MPa Target')
ax.legend()

# Max adhesive strength across rounds
ax = axes[1]
bars = ax.bar(x_pos, rounds_df['Max'], color=bar_colors[:len(x_pos)], alpha=0.8, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Max Adhesive Strength (kPa)')
ax.set_title('Maximum Adhesive Strength Across Optimization Rounds')
ax.axhline(1000, color='red', linestyle='--', linewidth=1.5, label='1 MPa Target')
for bar, val in zip(bars, rounds_df['Max']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.0f}', 
            ha='center', fontsize=8)
ax.legend()

plt.suptitle('Optimization Trajectory: Bio-Inspired Hydrogel Adhesive Strength', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig11_optimization_trajectory.png")
plt.close()
print("\nFigure 11 saved: optimization trajectory")

# ========== Figure 12: ML Method Comparison ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# EI methods comparison
ax = axes[0]
ei_methods = df_opt_ei.groupby('ML')['Glass (kPa)_max'].agg(['mean', 'max', 'count']).reset_index()
ei_methods = ei_methods.sort_values('max', ascending=True)
y_pos = np.arange(len(ei_methods))
ax.barh(y_pos, ei_methods['max'], color='#2196F3', alpha=0.7, label='Max', height=0.4)
ax.barh(y_pos + 0.4, ei_methods['mean'], color='#FF9800', alpha=0.7, label='Mean', height=0.4)
ax.set_yticks(y_pos + 0.2)
ax.set_yticklabels(ei_methods['ML'], fontsize=8)
ax.set_xlabel('Adhesive Strength (kPa)')
ax.set_title('EI-based Optimization Methods')
ax.legend()
ax.axvline(1000, color='red', linestyle='--', alpha=0.5)

# PRED methods comparison
ax = axes[1]
pred_methods = df_opt_pred.groupby('ML')['Glass (kPa)_max'].agg(['mean', 'max', 'count']).reset_index()
pred_methods = pred_methods.sort_values('max', ascending=True)
y_pos = np.arange(len(pred_methods))
ax.barh(y_pos, pred_methods['max'], color='#4CAF50', alpha=0.7, label='Max', height=0.4)
ax.barh(y_pos + 0.4, pred_methods['mean'], color='#E91E63', alpha=0.7, label='Mean', height=0.4)
ax.set_yticks(y_pos + 0.2)
ax.set_yticklabels(pred_methods['ML'], fontsize=8)
ax.set_xlabel('Adhesive Strength (kPa)')
ax.set_title('Prediction-based Optimization Methods')
ax.legend()
ax.axvline(1000, color='red', linestyle='--', alpha=0.5)

plt.suptitle('Comparison of ML-Guided Optimization Methods', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig12_ml_method_comparison.png")
plt.close()
print("Figure 12 saved: ML method comparison")

# Save method comparison
ei_methods.to_csv(f"{BASE}/outputs/ei_methods_comparison.csv", index=False)
pred_methods.to_csv(f"{BASE}/outputs/pred_methods_comparison.csv", index=False)

# ========== Figure 13: Composition Evolution Across Rounds ==========
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
monomer_names = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']

# Collect compositions per round
all_rounds = {
    'Training': df_train[features].values,
}

# Round 1 (all EI methods)
r1_data = df_opt_ei[df_opt_ei['ML'].isin(round1_ei_methods)][features].values
all_rounds['Round 1'] = r1_data

# Round 2
r2_data = df_opt_ei[df_opt_ei['ML'].isin(round2_ei_methods)][features].values
all_rounds['Round 2'] = r2_data

# Round 3
r3_data = df_opt_ei[df_opt_ei['ML'].isin(round3_ei_methods)][features].values
all_rounds['Round 3'] = r3_data

round_colors = {'Training': 'gray', 'Round 1': '#2196F3', 'Round 2': '#FF9800', 'Round 3': '#4CAF50'}

for i, (feat, name) in enumerate(zip(features, monomer_names)):
    ax = axes[i//3, i%3]
    for round_name, data in all_rounds.items():
        if data.shape[0] > 0:
            ax.hist(data[:, i], bins=15, alpha=0.5, label=round_name, color=round_colors[round_name])
    ax.set_xlabel(f'{name} Fraction')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}')
    ax.legend(fontsize=8)

plt.suptitle('Monomer Composition Evolution Across Optimization Rounds', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig13_composition_evolution.png")
plt.close()
print("Figure 13 saved: composition evolution")

print("\n=== Phase 3 Complete ===")
