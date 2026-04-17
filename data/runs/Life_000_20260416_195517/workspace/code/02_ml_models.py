#!/usr/bin/env python3
"""
Phase 2: ML Model Training & Evaluation
Random Forest Regressor and Gaussian Process for adhesive strength prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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

X = df_train[features].values
y = df_train[target].values

print(f"Training data: X={X.shape}, y={y.shape}")
print(f"y range: [{y.min():.1f}, {y.max():.1f}] kPa")

# ========== Model Definitions ==========
models = {
    'Random Forest': RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=5, 
                                            min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                    min_samples_split=5, random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_split=5,
                                        min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gaussian Process': GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1.0),
        n_restarts_optimizer=10, random_state=42, normalize_y=True
    ),
}

# ========== Cross-Validation ==========
cv = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}

print("\n=== 10-Fold Cross-Validation Results ===")
for name, model in models.items():
    if name == 'Gaussian Process':
        # GP needs scaled features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv)
    else:
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    mae = mean_absolute_error(y, y_pred_cv)
    r2 = r2_score(y, y_pred_cv)
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'y_pred': y_pred_cv
    }
    print(f"  {name:25s}: RMSE={rmse:.2f} kPa, MAE={mae:.2f} kPa, R²={r2:.4f}")

# Save results
results_summary = {name: {k: v for k, v in vals.items() if k != 'y_pred'} for name, vals in results.items()}
with open(f"{BASE}/outputs/cv_results.json", 'w') as f:
    json.dump(results_summary, f, indent=2)

# ========== Figure 6: Predicted vs Actual (all models) ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
model_colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx//2, idx%2]
    y_pred = res['y_pred']
    
    ax.scatter(y, y_pred, alpha=0.5, color=model_colors[idx], edgecolors='black', linewidth=0.3, s=30)
    
    # Perfect prediction line
    lims = [min(y.min(), y_pred.min())-5, max(y.max(), y_pred.max())+5]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5, label='Perfect prediction')
    
    ax.set_xlabel('Actual Adhesive Strength (kPa)')
    ax.set_ylabel('Predicted Adhesive Strength (kPa)')
    ax.set_title(f'{name}\nR²={res["R2"]:.3f}, RMSE={res["RMSE"]:.1f} kPa')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')

plt.suptitle('10-Fold Cross-Validation: Predicted vs Actual Adhesive Strength', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig6_cv_predicted_vs_actual.png")
plt.close()
print("\nFigure 6 saved: CV predicted vs actual")

# ========== Figure 7: Model Comparison Bar Chart ==========
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_names = list(results.keys())
rmses = [results[m]['RMSE'] for m in model_names]
maes = [results[m]['MAE'] for m in model_names]
r2s = [results[m]['R2'] for m in model_names]

ax = axes[0]
bars = ax.bar(range(len(model_names)), rmses, color=model_colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('RMSE (kPa)')
ax.set_title('Root Mean Squared Error')
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=9)

ax = axes[1]
bars = ax.bar(range(len(model_names)), maes, color=model_colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('MAE (kPa)')
ax.set_title('Mean Absolute Error')
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=9)

ax = axes[2]
bars = ax.bar(range(len(model_names)), r2s, color=model_colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('R² Score')
ax.set_title('Coefficient of Determination')
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=9)

plt.suptitle('Model Performance Comparison (10-Fold CV)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig7_model_comparison.png")
plt.close()
print("Figure 7 saved: model comparison")

# ========== Feature Importance (Random Forest) ==========
rf = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=5,
                           min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importance (Random Forest) ===")
print(importance_df.to_string(index=False))
importance_df.to_csv(f"{BASE}/outputs/feature_importance_rf.csv", index=False)

# ========== SHAP Analysis ==========
print("\nComputing SHAP values...")
try:
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    # Figure 8: SHAP Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.title('SHAP Feature Importance for Adhesive Strength Prediction')
    plt.tight_layout()
    plt.savefig(f"{BASE}/report/images/fig8_shap_summary.png")
    plt.close()
    print("Figure 8 saved: SHAP summary")
    
    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df.to_csv(f"{BASE}/outputs/shap_values.csv", index=False)
    
    # Mean absolute SHAP
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'Feature': features,
        'Mean_Abs_SHAP': mean_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    shap_importance.to_csv(f"{BASE}/outputs/shap_importance.csv", index=False)
    print("SHAP importance:")
    print(shap_importance.to_string(index=False))
    
except Exception as e:
    print(f"SHAP analysis error: {e}")

# ========== Figure 9: Feature Importance Comparison ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# RF importance
ax = axes[0]
imp_sorted = importance_df.sort_values('Importance', ascending=True)
ax.barh(range(len(imp_sorted)), imp_sorted['Importance'], color='#2196F3', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(imp_sorted)))
ax.set_yticklabels(imp_sorted['Feature'])
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance')

# SHAP importance
try:
    ax = axes[1]
    shap_sorted = shap_importance.sort_values('Mean_Abs_SHAP', ascending=True)
    ax.barh(range(len(shap_sorted)), shap_sorted['Mean_Abs_SHAP'], color='#FF9800', alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(shap_sorted)))
    ax.set_yticklabels(shap_sorted['Feature'])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('SHAP Feature Importance')
except:
    pass

plt.suptitle('Feature Importance Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig9_feature_importance.png")
plt.close()
print("Figure 9 saved: feature importance comparison")

# ========== Leave-One-Out for best model ==========
print("\n=== Leave-One-Out Cross-Validation (Random Forest) ===")
loo = LeaveOneOut()
y_pred_loo = cross_val_predict(rf, X, y, cv=loo, n_jobs=-1)
loo_rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
loo_r2 = r2_score(y, y_pred_loo)
loo_mae = mean_absolute_error(y, y_pred_loo)
print(f"  LOO RMSE: {loo_rmse:.2f} kPa")
print(f"  LOO MAE: {loo_mae:.2f} kPa")
print(f"  LOO R²: {loo_r2:.4f}")

loo_results = {'LOO_RMSE': loo_rmse, 'LOO_MAE': loo_mae, 'LOO_R2': loo_r2}
with open(f"{BASE}/outputs/loo_results.json", 'w') as f:
    json.dump(loo_results, f, indent=2)

print("\n=== Phase 2 Complete ===")
