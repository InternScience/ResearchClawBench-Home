"""
Phase 4: Bayesian Optimization for De Novo Hydrogel Design (Optimized)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf
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

X_train = df[monomer_cols].values
y_train = df[target_col].values

print(f"Training data: {X_train.shape[0]} samples")
print(f"Target range: {y_train.min():.2f} - {y_train.max():.2f} kPa")

# ============================================================
# Train models
# ============================================================
rfr = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rfr.fit(X_train, y_train)

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, normalize_y=True)
gp.fit(X_train, y_train)

# ============================================================
# Random search with RFR prediction (fast BO approximation)
# ============================================================
np.random.seed(42)
n_candidates = 50000

# Generate random compositions using Dirichlet distribution
candidates = np.random.dirichlet(np.ones(6), size=n_candidates)

# Predict with RFR and GP
rfr_preds = rfr.predict(candidates)
gp_preds, gp_stds = gp.predict(candidates, return_std=True)

# Compute Expected Improvement
y_max = y_train.max()
ei = (gp_preds - y_max) * 0.5 * (1 + erf((gp_preds - y_max) / (np.sqrt(2) * gp_stds + 1e-8)))
ei = np.maximum(ei, 0)

# ============================================================
# Top candidates by different strategies
# ============================================================
rfr_top_idx = np.argsort(rfr_preds)[::-1][:20]
gp_top_idx = np.argsort(gp_preds)[::-1][:20]
ei_top_idx = np.argsort(ei)[::-1][:20]

print("\n=== Top 5 RFR-Predicted Candidates ===")
for i, idx in enumerate(rfr_top_idx[:5]):
    print(f"  #{i+1}: RFR={rfr_preds[idx]:.1f} kPa, Comp={[f'{v:.3f}' for v in candidates[idx]]}")

print("\n=== Top 5 GP-Predicted Candidates ===")
for i, idx in enumerate(gp_top_idx[:5]):
    print(f"  #{i+1}: GP={gp_preds[idx]:.1f} kPa, Comp={[f'{v:.3f}' for v in candidates[idx]]}")

print("\n=== Top 5 EI Candidates ===")
for i, idx in enumerate(ei_top_idx[:5]):
    print(f"  #{i+1}: EI={ei[idx]:.2f}, GP={gp_preds[idx]:.1f} kPa, Comp={[f'{v:.3f}' for v in candidates[idx]]}")

# Save candidates
all_candidates = {
    'RFR_top': [{'composition': candidates[i].tolist(), 'predicted_kPa': float(rfr_preds[i])} for i in rfr_top_idx[:10]],
    'GP_top': [{'composition': candidates[i].tolist(), 'predicted_kPa': float(gp_preds[i]), 'gp_std': float(gp_stds[i])} for i in gp_top_idx[:10]],
    'EI_top': [{'composition': candidates[i].tolist(), 'ei': float(ei[i]), 'predicted_kPa': float(gp_preds[i]), 'gp_std': float(gp_stds[i])} for i in ei_top_idx[:10]],
}
with open('outputs/bo_candidates.json', 'w') as f:
    json.dump(all_candidates, f, indent=2)

# ============================================================
# Multi-round optimization simulation
# ============================================================
print("\n=== Simulating Multi-Round Optimization ===")

round_results = []
X_current = X_train.copy()
y_current = y_train.copy()

for round_num in range(1, 4):
    print(f"\nRound {round_num}:")
    
    rfr_round = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rfr_round.fit(X_current, y_current)
    
    kernel_round = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)
    gp_round = GaussianProcessRegressor(kernel=kernel_round, n_restarts_optimizer=3, random_state=42, normalize_y=True)
    gp_round.fit(X_current, y_current)
    
    np.random.seed(42 + round_num)
    cands = np.random.dirichlet(np.ones(6), size=20000)
    
    rfr_p = rfr_round.predict(cands)
    gp_p, gp_s = gp_round.predict(cands, return_std=True)
    
    y_max_curr = y_current.max()
    ei_curr = (gp_p - y_max_curr) * 0.5 * (1 + erf((gp_p - y_max_curr) / (np.sqrt(2) * gp_s + 1e-8)))
    ei_curr = np.maximum(ei_curr, 0)
    
    top5_idx = np.argsort(ei_curr)[::-1][:5]
    
    round_info = {
        'round': round_num,
        'n_training': len(y_current),
        'max_training': float(y_current.max()),
        'mean_training': float(y_current.mean()),
        'top_ei_predicted': float(gp_p[top5_idx[0]]),
        'top_ei_composition': cands[top5_idx[0]].tolist()
    }
    round_results.append(round_info)
    
    print(f"  Training size: {len(y_current)}")
    print(f"  Max training: {y_current.max():.2f} kPa")
    print(f"  Top EI predicted: {gp_p[top5_idx[0]]:.2f} kPa")
    
    for idx in top5_idx:
        x_new = cands[idx].reshape(1, -1)
        simulated_exp = max(gp_p[idx] + np.random.normal(0, 15), 1.0)
        X_current = np.vstack([X_current, x_new])
        y_current = np.append(y_current, simulated_exp)

with open('outputs/multi_round_results.json', 'w') as f:
    json.dump(round_results, f, indent=2)

# ============================================================
# Figure 9: Optimization trajectory
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
rounds = [r['round'] for r in round_results]
max_vals = [r['max_training'] for r in round_results]
pred_vals = [r['top_ei_predicted'] for r in round_results]
mean_vals = [r['mean_training'] for r in round_results]

rounds_all = [0] + rounds
max_vals_all = [y_train.max()] + max_vals
pred_vals_all = [y_train.max()] + pred_vals
mean_vals_all = [y_train.mean()] + mean_vals

ax.plot(rounds_all, max_vals_all, 'o-', color='#2196F3', lw=2, markersize=10, label='Max Training Value')
ax.plot(rounds_all, pred_vals_all, 's--', color='#F44336', lw=2, markersize=10, label='Top EI Predicted Value')
ax.plot(rounds_all, mean_vals_all, '^:', color='#4CAF50', lw=2, markersize=10, label='Mean Training Value')
ax.axhline(y=1000, color='gray', linestyle='--', alpha=0.5, label='1 MPa Target')
ax.set_xlabel('Optimization Round', fontsize=12)
ax.set_ylabel('Adhesive Strength (kPa)', fontsize=12)
ax.set_title('Multi-Round Bayesian Optimization Trajectory', fontsize=13)
ax.legend(fontsize=11)
ax.set_xticks(rounds_all)
plt.tight_layout()
plt.savefig('report/images/fig9_optimization_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 9 saved.")

# ============================================================
# Figure 10: Predicted vs EI landscape
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
sc = ax.scatter(rfr_preds[:5000], ei[:5000], c=gp_preds[:5000], cmap='viridis', alpha=0.3, s=5)
plt.colorbar(sc, ax=ax, label='GP Prediction (kPa)')
ax.set_xlabel('RFR Prediction (kPa)', fontsize=12)
ax.set_ylabel('Expected Improvement', fontsize=12)
ax.set_title('Prediction vs EI Landscape', fontsize=13)

ax = axes[1]
sc2 = ax.scatter(gp_preds[:5000], gp_stds[:5000], c=ei[:5000], cmap='hot', alpha=0.3, s=5)
plt.colorbar(sc2, ax=ax, label='EI')
ax.set_xlabel('GP Prediction (kPa)', fontsize=12)
ax.set_ylabel('GP Uncertainty (kPa)', fontsize=12)
ax.set_title('Prediction vs Uncertainty', fontsize=13)

plt.tight_layout()
plt.savefig('report/images/fig10_ei_landscape.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 10 saved.")

print("\nPhase 4 complete.")
