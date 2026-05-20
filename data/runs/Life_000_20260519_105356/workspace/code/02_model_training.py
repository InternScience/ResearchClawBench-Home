#!/usr/bin/env python3
"""
Train baseline ML models (RFR, GP, XGBoost) on initial 184 data.
Evaluate with cross-validation and save results.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")

# Load data
df = pd.read_csv("outputs/df184_clean.csv")
monomers = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
X = df[monomers].values
y = df['Glass (kPa)_10s'].values

# Define models
models = {
    'RFR': RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1),
    'GP': GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1)),
        n_restarts_optimizer=10, random_state=42, normalize_y=True
    ),
    'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=42, n_jobs=-1),
}

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    y_true_all = []
    y_pred_all = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    results[name] = {
        'r2_mean': float(np.mean(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
        'rmse_mean': float(np.mean(rmse_scores)),
        'rmse_std': float(np.std(rmse_scores)),
        'mae_mean': float(np.mean(mae_scores)),
        'mae_std': float(np.std(mae_scores)),
        'y_true': np.array(y_true_all),
        'y_pred': np.array(y_pred_all),
    }
    print(f"{name}: R²={results[name]['r2_mean']:.3f}±{results[name]['r2_std']:.3f}, "
          f"RMSE={results[name]['rmse_mean']:.2f}±{results[name]['rmse_std']:.2f}, "
          f"MAE={results[name]['mae_mean']:.2f}±{results[name]['mae_std']:.2f}")

# Save results
with open("outputs/model_cv_results.json", "w") as f:
    json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else 
                   [float(x) for x in vv] if isinstance(vv, np.ndarray) and vv.ndim==1 else vv
                   for kk, vv in v.items() if kk not in ('y_true','y_pred')} 
               for k, v in results.items()}, f, indent=2)

# Figure 4: Model performance comparison (bar chart)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
metrics = ['r2_mean', 'rmse_mean', 'mae_mean']
metric_labels = ['R²', 'RMSE (kPa)', 'MAE (kPa)']
stds = ['r2_std', 'rmse_std', 'mae_std']

for ax, metric, label, std in zip(axes, metrics, metric_labels, stds):
    names = list(results.keys())
    vals = [results[n][metric] for n in names]
    errs = [results[n][std] for n in names]
    bars = ax.bar(names, vals, yerr=errs, capsize=5, color=['steelblue', 'coral', 'seagreen'], edgecolor='black')
    ax.set_ylabel(label)
    ax.set_title(f'Cross-Validation {label}')
    ax.set_ylim(bottom=0)
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + errs[i] + 0.01*max(vals) if max(vals)>0 else bar.get_height() + errs[i] + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("report/images/fig4_model_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig4_model_comparison.png")

# Figure 5: Parity plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, name in zip(axes, results.keys()):
    yt = results[name]['y_true']
    yp = results[name]['y_pred']
    ax.scatter(yt, yp, alpha=0.6, edgecolors='white', s=60, c='steelblue')
    min_val = min(yt.min(), yp.min())
    max_val = max(yt.max(), yp.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax.set_xlabel('Observed Adhesive Strength (kPa)')
    ax.set_ylabel('Predicted Adhesive Strength (kPa)')
    ax.set_title(f'{name}: R²={results[name]["r2_mean"]:.3f}')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

plt.tight_layout()
plt.savefig("report/images/fig5_parity_plots.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig5_parity_plots.png")

# Train final RFR model for feature importance and save
rfr_final = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rfr_final.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    'Feature': monomers,
    'Importance': rfr_final.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance['Feature'], importance['Importance'], color='darkgreen', edgecolor='white')
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig("report/images/fig6_feature_importance.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved report/images/fig6_feature_importance.png")

# Save importance
importance.to_csv("outputs/feature_importance.csv", index=False)
print("Saved outputs/feature_importance.csv")
