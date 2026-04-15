"""
Phase 2 & 3: Model Training, Bayesian Optimization, and De Novo Design
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
from pathlib import Path
from math import erf, pi, sqrt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import differential_evolution
from sklearn.inspection import permutation_importance

# Paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_000_20260415_094527')
DATA_DIR = WORKSPACE / 'data'
OUTPUTS_DIR = WORKSPACE / 'outputs'
REPORT_IMAGES_DIR = WORKSPACE / 'report' / 'images'

FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET_COL = 'Glass (kPa)_10s'

print("=" * 80)
print("PHASE 2: Model Training & Validation")
print("=" * 80)

# Load data
df_primary = pd.read_excel(DATA_DIR / '184_verified_Original Data_ML_20230926.xlsx')
X = df_primary[FEATURE_COLS].values.astype(np.float64)
y = df_primary[TARGET_COL].values.astype(np.float64)

valid_mask = ~np.isnan(y)
X = X[valid_mask]
y = y[valid_mask]

n_samples, n_features = X.shape
print(f"Training data: {n_samples} samples, {n_features} features")

# ============================================================
# Train Random Forest Regressor
# ============================================================
print("\n--- Training Random Forest Regressor ---")

rfr = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rfr_cv_scores = cross_val_score(rfr, X, y, cv=kf, scoring='r2')
print(f"RFR CV R2 scores: {rfr_cv_scores}")
print(f"RFR CV R2 mean +/- std: {rfr_cv_scores.mean():.4f} +/- {rfr_cv_scores.std():.4f}")

rfr.fit(X, y)
y_pred_rfr = rfr.predict(X)
r2_rfr = r2_score(y, y_pred_rfr)
mae_rfr = mean_absolute_error(y, y_pred_rfr)
rmse_rfr = np.sqrt(mean_squared_error(y, y_pred_rfr))
print(f"RFR Full-fit R2: {r2_rfr:.4f}, MAE: {mae_rfr:.2f}, RMSE: {rmse_rfr:.2f}")

# ============================================================
# Train Gaussian Process Regressor
# ============================================================
print("\n--- Training Gaussian Process Regressor ---")

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42,
    normalize_y=True
)

gp_cv_scores = cross_val_score(gp, X, y, cv=kf, scoring='r2')
print(f"GP CV R2 scores: {gp_cv_scores}")
print(f"GP CV R2 mean +/- std: {gp_cv_scores.mean():.4f} +/- {gp_cv_scores.std():.4f}")

gp.fit(X, y)
y_pred_gp = gp.predict(X)
r2_gp = r2_score(y, y_pred_gp)
mae_gp = mean_absolute_error(y, y_pred_gp)
rmse_gp = np.sqrt(mean_squared_error(y, y_pred_gp))
print(f"GP Full-fit R2: {r2_gp:.4f}, MAE: {mae_gp:.2f}, RMSE: {rmse_gp:.2f}")

model_perf = {
    'RFR': {
        'cv_r2_mean': float(rfr_cv_scores.mean()),
        'cv_r2_std': float(rfr_cv_scores.std()),
        'full_r2': float(r2_rfr),
        'mae_kPa': float(mae_rfr),
        'rmse_kPa': float(rmse_rfr)
    },
    'GP': {
        'cv_r2_mean': float(gp_cv_scores.mean()),
        'cv_r2_std': float(gp_cv_scores.std()),
        'full_r2': float(r2_gp),
        'mae_kPa': float(mae_gp),
        'rmse_kPa': float(rmse_gp)
    }
}
with open(OUTPUTS_DIR / 'model_performance.json', 'w') as f:
    json.dump(model_perf, f, indent=2)

# ============================================================
# Feature Importance
# ============================================================
print("\n--- Feature Importance Analysis ---")

rfr_importance = rfr.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance_RFR': rfr_importance
})
feature_imp_df = feature_imp_df.sort_values('Importance_RFR', ascending=False)
print(feature_imp_df.to_string(index=False))

with open(OUTPUTS_DIR / 'feature_importance.json', 'w') as f:
    json.dump({
        'features': list(feature_imp_df['Feature']),
        'importance_rfr': [float(x) for x in feature_imp_df['Importance_RFR']]
    }, f, indent=2)

# ============================================================
# Figure 4: Model Performance
# ============================================================
print("\n--- Generating Figure 4: Model Performance ---")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y, y_pred_rfr, alpha=0.6, s=40, c='steelblue', edgecolors='none')
lims = [min(y.min(), y_pred_rfr.min()) - 10, max(y.max(), y_pred_rfr.max()) + 10]
ax1.plot(lims, lims, 'r--', linewidth=2, alpha=0.7)
ax1.set_xlabel('Actual Adhesion (kPa)', fontsize=11)
ax1.set_ylabel('Predicted Adhesion (kPa)', fontsize=11)
ax1.set_title(f'Random Forest\nR2={r2_rfr:.3f}, CV R2={rfr_cv_scores.mean():.3f}', 
              fontsize=12, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y, y_pred_gp, alpha=0.6, s=40, c='darkorange', edgecolors='none')
ax2.plot(lims, lims, 'r--', linewidth=2, alpha=0.7)
ax2.set_xlabel('Actual Adhesion (kPa)', fontsize=11)
ax2.set_ylabel('Predicted Adhesion (kPa)', fontsize=11)
ax2.set_title(f'Gaussian Process\nR2={r2_gp:.3f}, CV R2={gp_cv_scores.mean():.3f}',
              fontsize=12, fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
resid_rfr = y - y_pred_rfr
resid_gp = y - y_pred_gp
ax3.hist(resid_rfr, bins=20, alpha=0.6, color='steelblue', label='RFR', density=True)
ax3.hist(resid_gp, bins=20, alpha=0.6, color='darkorange', label='GP', density=True)
ax3.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax3.set_xlabel('Residual (kPa)', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)

ax4 = fig.add_subplot(gs[1, 0])
models = ['RFR', 'GP']
cv_means = [rfr_cv_scores.mean(), gp_cv_scores.mean()]
cv_stds = [rfr_cv_scores.std(), gp_cv_scores.std()]
bars = ax4.bar(models, cv_means, yerr=cv_stds, capsize=5, 
               color=['steelblue', 'darkorange'], alpha=0.8, edgecolor='black')
ax4.set_ylabel('CV R2 Score', fontsize=11)
ax4.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 1)
for bar, val in zip(bars, cv_means):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax5 = fig.add_subplot(gs[1, 1:])
sorted_idx = np.argsort(rfr_importance)
y_pos = np.arange(len(FEATURE_COLS))
ax5.barh(y_pos, rfr_importance[sorted_idx], color='steelblue', alpha=0.8, edgecolor='black')
ax5.set_yticks(y_pos)
ax5.set_yticklabels([FEATURE_COLS[i] for i in sorted_idx], fontsize=10)
ax5.set_xlabel('Feature Importance (Gini)', fontsize=11)
ax5.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')

plt.savefig(REPORT_IMAGES_DIR / 'figure4_model_performance.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure4_model_performance.png")

# ============================================================
# PHASE 3: Bayesian Optimization for De Novo Design
# ============================================================
print("\n" + "=" * 80)
print("PHASE 3: Bayesian Optimization for De Novo Design")
print("=" * 80)

bounds = [(0.0, 0.7) for _ in range(n_features)]

def predict_with_uncertainty(model, x_arr):
    """Get prediction mean and uncertainty from any model."""
    x = x_arr.reshape(1, -1)
    if hasattr(model, 'estimators_'):
        tree_preds = np.array([tree.predict(x)[0] for tree in model.estimators_])
        return float(tree_preds.mean()), float(max(tree_preds.std(), 1e-8))
    else:
        mu_arr, sigma_arr = model.predict(x, return_std=True)
        return float(mu_arr[0]), float(max(sigma_arr[0], 1e-8))

def expected_improvement(mu, sigma, y_max, xi=0.1):
    """Compute Expected Improvement."""
    improvement = mu - y_max - xi
    if sigma < 1e-10:
        return 0.0
    Z = improvement / sigma
    ei = improvement * (0.5 * (1.0 + erf(Z / sqrt(2.0)))) + \
         sigma * (1.0 / sqrt(2.0 * pi)) * np.exp(-0.5 * Z**2)
    return float(ei)

def objective_for_optimization(x, model, X_train, y_train):
    """Negative of (prediction + EI bonus) with sum-to-one penalty."""
    mu, sigma = predict_with_uncertainty(model, x)
    y_max = float(y_train.max())
    ei = expected_improvement(mu, sigma, y_max, xi=0.1)
    sum_penalty = 1000.0 * (sum(x) - 1.0)**2
    return float(-(mu + 5.0 * ei) + sum_penalty)

# Run optimization
n_opt_runs = 10
best_designs_rfr = []
best_designs_gp = []

print("\nRunning optimization with RFR model...")
for run in range(n_opt_runs):
    result = differential_evolution(
        lambda x: objective_for_optimization(x, rfr, X, y),
        bounds=bounds,
        seed=run,
        maxiter=100,
        tol=1e-6,
        polish=True
    )
    composition = result.x / result.x.sum()
    pred_strength = float(rfr.predict(composition.reshape(1, -1))[0])
    best_designs_rfr.append({
        'composition': composition.tolist(),
        'predicted_strength_kPa': pred_strength,
        'run': run
    })

print(f"Generated {len(best_designs_rfr)} RFR designs")

print("\nRunning optimization with GP model...")
for run in range(n_opt_runs):
    result = differential_evolution(
        lambda x: objective_for_optimization(x, gp, X, y),
        bounds=bounds,
        seed=run + 100,
        maxiter=100,
        tol=1e-6,
        polish=True
    )
    composition = result.x / result.x.sum()
    pred_strength = float(gp.predict(composition.reshape(1, -1))[0])
    best_designs_gp.append({
        'composition': composition.tolist(),
        'predicted_strength_kPa': pred_strength,
        'run': run
    })

print(f"Generated {len(best_designs_gp)} GP designs")

# Combine and rank
all_designs = []
for d in best_designs_rfr:
    d['model'] = 'RFR'
    all_designs.append(d)
for d in best_designs_gp:
    d['model'] = 'GP'
    all_designs.append(d)

all_designs.sort(key=lambda x: x['predicted_strength_kPa'], reverse=True)

# Get unique top designs
unique_designs = []
seen_compositions = []
for d in all_designs:
    comp = np.array(d['composition'])
    is_duplicate = False
    for seen in seen_compositions:
        if np.allclose(comp, seen, atol=0.01):
            is_duplicate = True
            break
    if not is_duplicate:
        unique_designs.append(d)
        seen_compositions.append(comp)

designs_above_1mpa = [d for d in unique_designs if d['predicted_strength_kPa'] > 1000]
print(f"\nUnique top designs: {len(unique_designs)}")
print(f"Designs predicted above 1 MPa (1000 kPa): {len(designs_above_1mpa)}")

# Save top designs
top_designs = unique_designs[:20]
with open(OUTPUTS_DIR / 'top_designs.json', 'w') as f:
    json.dump(top_designs, f, indent=2)

top_designs_df = pd.DataFrame([
    {
        **{f: d['composition'][i] for i, f in enumerate(FEATURE_COLS)},
        'predicted_strength_kPa': d['predicted_strength_kPa'],
        'model': d['model'],
        'exceeds_1MPa': d['predicted_strength_kPa'] > 1000
    }
    for d in top_designs
])
top_designs_df.to_csv(OUTPUTS_DIR / 'top_designs.csv', index=False)
print("Top designs saved to outputs/")

# ============================================================
# Figure 5: Optimization Results
# ============================================================
print("\n--- Generating Figure 5: Optimization Results ---")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
strengths_rfr = [d['predicted_strength_kPa'] for d in best_designs_rfr]
strengths_gp = [d['predicted_strength_kPa'] for d in best_designs_gp]
ax1.hist(strengths_rfr, bins=30, alpha=0.6, color='steelblue', label='RFR-optimized', density=True)
ax1.hist(strengths_gp, bins=30, alpha=0.6, color='darkorange', label='GP-optimized', density=True)
ax1.axvline(1000, color='green', linestyle='--', linewidth=2.5, label='1 MPa target')
ax1.axvline(float(y.max()), color='red', linestyle=':', linewidth=2, label=f'Max experimental ({y.max():.0f} kPa)')
ax1.set_xlabel('Predicted Adhesive Strength (kPa)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Distribution of ML-Optimized Designs', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)

ax2 = fig.add_subplot(gs[1, :2])
n_top = min(10, len(top_designs))
x_pos = np.arange(n_top)
width = 0.12
colors_bar = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
for j, col in enumerate(FEATURE_COLS):
    vals = [top_designs[i]['composition'][j] for i in range(n_top)]
    ax2.bar(x_pos + j*width, vals, width, label=col.replace('-', '\n'), color=colors_bar[j])
ax2.set_xticks(x_pos + width*2.5)
labels = []
for i in range(n_top):
    strength = top_designs[i]['predicted_strength_kPa']
    marker = '*' if strength > 1000 else ''
    labels.append(f'D{i+1}\n{strength:.0f}kPa{marker}')
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('Monomer Fraction', fontsize=11)
ax2.set_title('Top-10 ML-Designed Hydrogel Compositions (* = >1 MPa)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8, loc='upper right', ncol=3)
ax2.set_ylim(0, 1)

ax3 = fig.add_subplot(gs[1, 2])
ax3.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, alpha=0.5, label='Experimental', edgecolors='none')
des_comp = np.array([d['composition'] for d in top_designs[:15]])
des_pred = np.array([d['predicted_strength_kPa'] for d in top_designs[:15]])
sc = ax3.scatter(des_comp[:, 0], des_comp[:, 1], c=des_pred, cmap='Reds', 
                 s=80, alpha=0.9, marker='*', label='ML-designed', edgecolors='black', linewidth=1)
ax3.set_xlabel('Nucleophilic-HEA', fontsize=11)
ax3.set_ylabel('Hydrophobic-BA', fontsize=11)
ax3.set_title('Designed vs Experimental\nComposition Space', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
plt.colorbar(sc, ax=ax3, label='Predicted (kPa)')

ax4 = fig.add_subplot(gs[2, :])
n_features_plot = len(FEATURE_COLS)
x_feat = np.arange(n_features_plot)
width_feat = 0.25

exp_mean = X.mean(axis=0)
exp_std = X.std(axis=0)
des_mean = des_comp.mean(axis=0)
des_std = des_comp.std(axis=0)

ax4.bar(x_feat - width_feat/2, exp_mean, width_feat, yerr=exp_std, 
        label='Experimental (mean+/-std)', color='steelblue', alpha=0.7, capsize=3)
ax4.bar(x_feat + width_feat/2, des_mean, width_feat, yerr=des_std,
        label='ML-designed (mean+/-std)', color='darkorange', alpha=0.7, capsize=3)
ax4.set_xticks(x_feat)
ax4.set_xticklabels([f.replace('-', '\n') for f in FEATURE_COLS], fontsize=9)
ax4.set_ylabel('Mean Monomer Fraction', fontsize=11)
ax4.set_title('Feature Comparison: Experimental vs ML-Designed Compositions', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_ylim(0, max(exp_mean.max(), des_mean.max()) * 1.3)

plt.savefig(REPORT_IMAGES_DIR / 'figure5_optimization_results.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure5_optimization_results.png")

# ============================================================
# Figure 6: Interpretability Analysis
# ============================================================
print("\n--- Generating Figure 6: Interpretability Analysis ---")

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

perm_imp_rfr = permutation_importance(rfr, X, y, n_repeats=30, random_state=42, n_jobs=-1)
sorted_idx_rfr = perm_imp_rfr.importances_mean.argsort()

ax1 = fig.add_subplot(gs[0, 0])
ax1.boxplot(perm_imp_rfr.importances[sorted_idx_rfr].T, vert=False, 
            labels=[FEATURE_COLS[i] for i in sorted_idx_rfr])
ax1.set_xlabel('Decrease in R2 Score', fontsize=11)
ax1.set_title('RFR Permutation Importance', fontsize=12, fontweight='bold')

perm_imp_gp = permutation_importance(gp, X, y, n_repeats=30, random_state=42)
sorted_idx_gp = perm_imp_gp.importances_mean.argsort()

ax2 = fig.add_subplot(gs[0, 1])
ax2.boxplot(perm_imp_gp.importances[sorted_idx_gp].T, vert=False,
            labels=[FEATURE_COLS[i] for i in sorted_idx_gp])
ax2.set_xlabel('Decrease in R2 Score', fontsize=11)
ax2.set_title('GP Permutation Importance', fontsize=12, fontweight='bold')

ax3 = fig.add_subplot(gs[1, :])
n_points = 50
for i, col in enumerate(FEATURE_COLS):
    x_vals = np.linspace(X[:, i].min(), X[:, i].max(), n_points)
    y_effects = []
    for xv in x_vals:
        X_synth = X.copy()
        X_synth[:, i] = xv
        for j in range(n_features):
            if j != i:
                X_synth[:, j] = X[:, j].mean()
        preds = rfr.predict(X_synth)
        y_effects.append(float(preds.mean()))
    
    ax3.plot(x_vals, y_effects, linewidth=2.5, label=col.replace('-', '\n'), alpha=0.8)

ax3.axhline(float(y.mean()), color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Mean adhesion')
ax3.axhline(1000, color='green', linestyle=':', linewidth=2, alpha=0.7, label='1 MPa target')
ax3.set_xlabel('Monomer Fraction', fontsize=12)
ax3.set_ylabel('Predicted Adhesion (kPa)', fontsize=12)
ax3.set_title('Marginal Effect of Each Monomer on Predicted Adhesion', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9, ncol=2, loc='best')
ax3.grid(True, alpha=0.3)

plt.savefig(REPORT_IMAGES_DIR / 'figure6_interpretability.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: report/images/figure6_interpretability.png")

# ============================================================
# Save results summary
# ============================================================
print("\n--- Saving Results Summary ---")

results_summary = {
    'model_performance': model_perf,
    'feature_importance': {
        'features': list(feature_imp_df['Feature']),
        'importance_rfr': [float(x) for x in feature_imp_df['Importance_RFR']]
    },
    'optimization_summary': {
        'n_total_designs': len(all_designs),
        'n_unique_designs': len(unique_designs),
        'n_above_1mpa': len(designs_above_1mpa),
        'max_predicted_strength_kPa': float(unique_designs[0]['predicted_strength_kPa']) if unique_designs else None,
        'best_composition': unique_designs[0]['composition'] if unique_designs else None,
        'experimental_max_kPa': float(y.max()),
        'experimental_mean_kPa': float(y.mean()),
        'improvement_factor': float(unique_designs[0]['predicted_strength_kPa'] / y.max()) if unique_designs else None
    },
    'top_5_designs': [
        {
            'composition': {FEATURE_COLS[j]: d['composition'][j] for j in range(n_features)},
            'predicted_strength_kPa': d['predicted_strength_kPa'],
            'model': d['model']
        }
        for d in unique_designs[:5]
    ]
}

with open(OUTPUTS_DIR / 'results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print("Results summary saved to outputs/results_summary.json")

print("\n" + "=" * 80)
print("PHASES 2 & 3 COMPLETE!")
print("=" * 80)
print(f"\nKey Findings:")
print(f"  Best experimental adhesion: {y.max():.1f} kPa")
print(f"  Best predicted design: {unique_designs[0]['predicted_strength_kPa']:.1f} kPa")
if unique_designs:
    print(f"  Improvement factor: {unique_designs[0]['predicted_strength_kPa']/y.max():.1f}x")
print(f"  Designs exceeding 1 MPa: {len(designs_above_1mpa)}")
