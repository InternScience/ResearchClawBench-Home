"""
Phase 3: ML Model Training & Comparison
Train RFR, GP, and other models; cross-validate; extract feature importance.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
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

X = df[monomer_cols].values
y = df[target_col].values

print(f"Training data: X={X.shape}, y={y.shape}")
print(f"Target range: {y.min():.2f} - {y.max():.2f} kPa")

# ============================================================
# Define models
# ============================================================
models = {
    'RFR': RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1),
    'ETR': ExtraTreesRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1),
    'GBR': GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42),
    'GP_Matern': GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0),
        n_restarts_optimizer=10, random_state=42, normalize_y=True
    ),
}

# ============================================================
# Cross-validation
# ============================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation scores
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    neg_mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    neg_rmse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    
    # Cross-val predictions
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    results[name] = {
        'r2_mean': float(r2_scores.mean()),
        'r2_std': float(r2_scores.std()),
        'mae_mean': float(-neg_mae_scores.mean()),
        'mae_std': float(neg_mae_scores.std()),
        'rmse_mean': float(-neg_rmse_scores.mean()),
        'rmse_std': float(neg_rmse_scores.std()),
        'y_pred': y_pred.tolist()
    }
    
    print(f"  R² = {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"  MAE = {-neg_mae_scores.mean():.2f} ± {neg_mae_scores.std():.2f}")
    print(f"  RMSE = {-neg_rmse_scores.mean():.2f} ± {neg_rmse_scores.std():.2f}")

# Save results (without y_pred for JSON)
results_json = {}
for name in results:
    results_json[name] = {k: v for k, v in results[name].items() if k != 'y_pred'}
with open('outputs/model_comparison.json', 'w') as f:
    json.dump(results_json, f, indent=2)

# ============================================================
# Figure 6: Model comparison bar chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_names = list(results.keys())
metrics = ['r2_mean', 'mae_mean', 'rmse_mean']
metric_labels = ['R²', 'MAE (kPa)', 'RMSE (kPa)']
stds = ['r2_std', 'mae_std', 'rmse_std']

for i, (metric, label, std_key) in enumerate(zip(metrics, metric_labels, stds)):
    vals = [results[m][metric] for m in model_names]
    errs = [results[m][std_key] for m in model_names]
    axes[i].bar(model_names, vals, yerr=errs, capsize=5, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'], alpha=0.8)
    axes[i].set_ylabel(label, fontsize=12)
    axes[i].set_title(f'Model Comparison: {label}', fontsize=13)

plt.tight_layout()
plt.savefig('report/images/fig6_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 6 saved.")

# ============================================================
# Figure 7: Parity plots (predicted vs actual)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for i, name in enumerate(model_names):
    ax = axes[i//2, i%2]
    y_pred = np.array(results[name]['y_pred'])
    ax.scatter(y, y_pred, alpha=0.5, s=30, c=f'C{i}')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    r2 = results[name]['r2_mean']
    ax.set_xlabel('Actual Glass (kPa)', fontsize=11)
    ax.set_ylabel('Predicted Glass (kPa)', fontsize=11)
    ax.set_title(f'{name} (R²={r2:.3f})', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
plt.suptitle('Parity Plots: Predicted vs Actual Adhesive Strength', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/images/fig7_parity_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7 saved.")

# ============================================================
# Feature importance from RFR
# ============================================================
rfr = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rfr.fit(X, y)
importances = rfr.feature_importances_
short_names = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']

# Save feature importance
fi_dict = {name: float(imp) for name, imp in zip(short_names, importances)}
with open('outputs/feature_importance_rfr.json', 'w') as f:
    json.dump(fi_dict, f, indent=2)

# ============================================================
# Figure 8: Feature importance
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(importances)[::-1]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
ax.bar(range(len(importances)), importances[sorted_idx], color=[colors[i] for i in range(len(sorted_idx))], alpha=0.8)
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([short_names[i] for i in sorted_idx], fontsize=12)
ax.set_ylabel('Feature Importance', fontsize=12)
ax.set_title('RFR Feature Importance for Adhesive Strength Prediction', fontsize=13)
plt.tight_layout()
plt.savefig('report/images/fig8_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8 saved.")

# Also get GBR feature importance
gbr = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=42)
gbr.fit(X, y)
gbr_importances = gbr.feature_importances_
fi_dict_gbr = {name: float(imp) for name, imp in zip(short_names, gbr_importances)}
with open('outputs/feature_importance_gbr.json', 'w') as f:
    json.dump(fi_dict_gbr, f, indent=2)

print("\nPhase 3 complete.")
