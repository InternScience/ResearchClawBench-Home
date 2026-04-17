#!/usr/bin/env python3
"""
Phase 1: Data Exploration and Overview
Phase 2: Gaussian Process Calibration Model
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.figsize': (10, 8),
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_003_20260416_230923'
IMG_DIR = os.path.join(BASE, 'report', 'images')
OUT_DIR = os.path.join(BASE, 'outputs')
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# PHASE 1: Data Exploration
# ============================================================
print("=" * 60)
print("PHASE 1: Data Exploration")
print("=" * 60)

# Load calibration data
df_cal = pd.read_csv(os.path.join(BASE, 'data', 'tg_calibration.csv'))
print(f"\nCalibration dataset shape: {df_cal.shape}")
print(f"Columns: {list(df_cal.columns)}")
print(f"\nSummary statistics:")
print(df_cal[['tg_exp', 'tg_md', 'std']].describe())

# Load vitrimer data
df_vit = pd.read_csv(os.path.join(BASE, 'data', 'tg_vitrimer_MD.csv'))
print(f"\nVitrimer dataset shape: {df_vit.shape}")
print(f"Columns: {list(df_vit.columns)}")
print(f"\nSummary statistics:")
print(df_vit[['tg', 'std']].describe())

# Save summary stats
summary = {
    'calibration': {
        'n_samples': int(df_cal.shape[0]),
        'tg_exp_mean': float(df_cal['tg_exp'].mean()),
        'tg_exp_std': float(df_cal['tg_exp'].std()),
        'tg_exp_min': float(df_cal['tg_exp'].min()),
        'tg_exp_max': float(df_cal['tg_exp'].max()),
        'tg_md_mean': float(df_cal['tg_md'].mean()),
        'tg_md_std': float(df_cal['tg_md'].std()),
        'tg_md_min': float(df_cal['tg_md'].min()),
        'tg_md_max': float(df_cal['tg_md'].max()),
    },
    'vitrimer': {
        'n_samples': int(df_vit.shape[0]),
        'tg_md_mean': float(df_vit['tg'].mean()),
        'tg_md_std': float(df_vit['tg'].std()),
        'tg_md_min': float(df_vit['tg'].min()),
        'tg_md_max': float(df_vit['tg'].max()),
    }
}
with open(os.path.join(OUT_DIR, 'data_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# --- Figure 1: Data Overview ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1a: Distribution of experimental Tg
axes[0, 0].hist(df_cal['tg_exp'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Experimental Tg (K)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('(a) Distribution of Experimental Tg\n(Calibration Dataset)')
axes[0, 0].axvline(df_cal['tg_exp'].mean(), color='red', linestyle='--', label=f"Mean = {df_cal['tg_exp'].mean():.1f} K")
axes[0, 0].legend()

# 1b: Distribution of MD Tg (calibration)
axes[0, 1].hist(df_cal['tg_md'], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('MD Simulated Tg (K)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('(b) Distribution of MD Tg\n(Calibration Dataset)')
axes[0, 1].axvline(df_cal['tg_md'].mean(), color='red', linestyle='--', label=f"Mean = {df_cal['tg_md'].mean():.1f} K")
axes[0, 1].legend()

# 1c: MD vs Exp Tg scatter
axes[1, 0].scatter(df_cal['tg_exp'], df_cal['tg_md'], c='steelblue', alpha=0.6, edgecolor='black', linewidth=0.5)
min_val = min(df_cal['tg_exp'].min(), df_cal['tg_md'].min()) - 20
max_val = max(df_cal['tg_exp'].max(), df_cal['tg_md'].max()) + 20
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
axes[1, 0].set_xlabel('Experimental Tg (K)')
axes[1, 0].set_ylabel('MD Simulated Tg (K)')
axes[1, 0].set_title('(c) MD vs Experimental Tg')
axes[1, 0].legend()
axes[1, 0].set_xlim(min_val, max_val)
axes[1, 0].set_ylim(min_val, max_val)

# 1d: Distribution of vitrimer MD Tg
axes[1, 1].hist(df_vit['tg'], bins=50, color='seagreen', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('MD Simulated Tg (K)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'(d) Distribution of Vitrimer MD Tg\n(n = {len(df_vit)})')
axes[1, 1].axvline(df_vit['tg'].mean(), color='red', linestyle='--', label=f"Mean = {df_vit['tg'].mean():.1f} K")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1_data_overview.png'))
plt.close()
print("\nSaved fig1_data_overview.png")

# ============================================================
# PHASE 2: Gaussian Process Calibration
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Gaussian Process Calibration")
print("=" * 60)

# Prepare data
X_cal = df_cal['tg_md'].values.reshape(-1, 1)
y_cal = df_cal['tg_exp'].values
noise_cal = df_cal['std'].values

# Define GP kernel
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=50.0, length_scale_bounds=(1.0, 500.0), nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))

# Cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_predictions = np.zeros_like(y_cal)
cv_uncertainties = np.zeros_like(y_cal)
cv_metrics = []

print(f"\nRunning {n_folds}-fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_cal)):
    X_train, X_test = X_cal[train_idx], X_cal[test_idx]
    y_train, y_test = y_cal[train_idx], y_cal[test_idx]
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=noise_cal[train_idx]**2, normalize_y=True)
    gp.fit(X_train, y_train)
    
    y_pred, y_std = gp.predict(X_test, return_std=True)
    cv_predictions[test_idx] = y_pred
    cv_uncertainties[test_idx] = y_std
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    cv_metrics.append({'fold': fold+1, 'mae': mae, 'rmse': rmse, 'r2': r2})
    print(f"  Fold {fold+1}: MAE={mae:.2f} K, RMSE={rmse:.2f} K, R²={r2:.4f}")

# Overall CV metrics
overall_mae = mean_absolute_error(y_cal, cv_predictions)
overall_rmse = np.sqrt(mean_squared_error(y_cal, cv_predictions))
overall_r2 = r2_score(y_cal, cv_predictions)

print(f"\nOverall CV: MAE={overall_mae:.2f} K, RMSE={overall_rmse:.2f} K, R²={overall_r2:.4f}")

# Train final model on all data
print("\nTraining final GP model on all calibration data...")
gp_final = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=noise_cal**2, normalize_y=True)
gp_final.fit(X_cal, y_cal)
print(f"Optimized kernel: {gp_final.kernel_}")

# Save GP metrics
gp_results = {
    'cv_metrics': cv_metrics,
    'overall': {
        'mae': float(overall_mae),
        'rmse': float(overall_rmse),
        'r2': float(overall_r2)
    },
    'final_kernel': str(gp_final.kernel_),
    'n_training_samples': int(len(X_cal))
}
with open(os.path.join(OUT_DIR, 'gp_calibration_results.json'), 'w') as f:
    json.dump(gp_results, f, indent=2)

# --- Figure 2: GP Calibration Parity Plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 2a: CV Parity plot
ax = axes[0]
scatter = ax.scatter(y_cal, cv_predictions, c=cv_uncertainties, cmap='viridis', 
                     alpha=0.7, edgecolor='black', linewidth=0.5, s=50)
ax.errorbar(y_cal, cv_predictions, yerr=cv_uncertainties, fmt='none', ecolor='gray', alpha=0.3)
min_v = min(y_cal.min(), cv_predictions.min()) - 20
max_v = max(y_cal.max(), cv_predictions.max()) + 20
ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect calibration')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('GP-Predicted Tg (K)')
ax.set_title(f'(a) GP Calibration: 5-Fold CV\nMAE={overall_mae:.1f} K, RMSE={overall_rmse:.1f} K, R²={overall_r2:.3f}')
ax.legend(fontsize=12)
ax.set_xlim(min_v, max_v)
ax.set_ylim(min_v, max_v)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Prediction Uncertainty (K)')

# 2b: Residuals
ax = axes[1]
residuals = cv_predictions - y_cal
ax.scatter(y_cal, residuals, c='steelblue', alpha=0.6, edgecolor='black', linewidth=0.5, s=50)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.axhline(y=residuals.mean() + 2*residuals.std(), color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=residuals.mean() - 2*residuals.std(), color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('Residual (Predicted - Experimental) (K)')
ax.set_title(f'(b) Residual Analysis\nMean={residuals.mean():.1f} K, Std={residuals.std():.1f} K')
ax.fill_between([min_v, max_v], 
                [residuals.mean() - 2*residuals.std()]*2, 
                [residuals.mean() + 2*residuals.std()]*2, 
                alpha=0.1, color='gray', label='±2σ')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_gp_calibration.png'))
plt.close()
print("Saved fig2_gp_calibration.png")

# ============================================================
# Apply GP Calibration to Vitrimer Data
# ============================================================
print("\nApplying GP calibration to vitrimer MD Tg values...")
X_vit = df_vit['tg'].values.reshape(-1, 1)
tg_calibrated, tg_calibrated_std = gp_final.predict(X_vit, return_std=True)

df_vit['tg_calibrated'] = tg_calibrated
df_vit['tg_calibrated_std'] = tg_calibrated_std

# Save calibrated predictions
df_vit.to_csv(os.path.join(OUT_DIR, 'vitrimer_calibrated_tg.csv'), index=False)
print(f"Saved calibrated Tg predictions for {len(df_vit)} vitrimers")
print(f"Calibrated Tg range: {tg_calibrated.min():.1f} - {tg_calibrated.max():.1f} K")
print(f"Calibrated Tg mean: {tg_calibrated.mean():.1f} ± {tg_calibrated.std():.1f} K")

# --- Figure 3: Calibrated Tg Distribution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 3a: Before vs After calibration
axes[0].hist(df_vit['tg'], bins=50, alpha=0.6, color='coral', edgecolor='black', label='MD Tg (raw)')
axes[0].hist(tg_calibrated, bins=50, alpha=0.6, color='steelblue', edgecolor='black', label='Calibrated Tg')
axes[0].set_xlabel('Tg (K)')
axes[0].set_ylabel('Count')
axes[0].set_title('(a) MD vs Calibrated Tg Distribution')
axes[0].legend()

# 3b: Calibration function
x_range = np.linspace(X_cal.min() - 50, X_cal.max() + 50, 200).reshape(-1, 1)
y_pred_range, y_std_range = gp_final.predict(x_range, return_std=True)
axes[1].fill_between(x_range.ravel(), y_pred_range - 2*y_std_range, y_pred_range + 2*y_std_range, 
                     alpha=0.2, color='steelblue', label='95% CI')
axes[1].plot(x_range, y_pred_range, 'b-', linewidth=2, label='GP mean')
axes[1].scatter(X_cal.ravel(), y_cal, c='red', s=30, alpha=0.5, zorder=5, label='Training data')
axes[1].plot([150, 650], [150, 650], 'k--', alpha=0.5, label='y = x')
axes[1].set_xlabel('MD Simulated Tg (K)')
axes[1].set_ylabel('Calibrated (Experimental) Tg (K)')
axes[1].set_title('(b) GP Calibration Function')
axes[1].legend()

# 3c: Uncertainty distribution
axes[2].hist(tg_calibrated_std, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Prediction Uncertainty (K)')
axes[2].set_ylabel('Count')
axes[2].set_title(f'(c) Calibration Uncertainty Distribution\nMean={tg_calibrated_std.mean():.1f} K')
axes[2].axvline(tg_calibrated_std.mean(), color='red', linestyle='--', label=f'Mean = {tg_calibrated_std.mean():.1f} K')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_calibrated_tg.png'))
plt.close()
print("Saved fig3_calibrated_tg.png")

print("\nPhase 1 & 2 complete!")
