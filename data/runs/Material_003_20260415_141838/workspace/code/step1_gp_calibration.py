"""
Step 1: Gaussian Process Calibration
- Train GP model mapping MD Tg -> Experimental Tg
- Evaluate calibration performance
- Apply to vitrimer MD data
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

os.makedirs('../outputs', exist_ok=True)
os.makedirs('../report/images', exist_ok=True)

# Load data
df_cal = pd.read_csv('../data/tg_calibration.csv')
df_vitrimer = pd.read_csv('../data/tg_vitrimer_MD.csv')

print(f"Calibration data: {df_cal.shape}")
print(f"Vitrimer MD data: {df_vitrimer.shape}")

# ==============================
# GP Calibration: MD Tg -> Exp Tg
# ==============================
X_md = df_cal['tg_md'].values.reshape(-1, 1)
y_exp = df_cal['tg_exp'].values

# Define kernel
kernel = ConstantKernel(1.0, constant_value_bounds=(0.1, 100.0)) * RBF(length_scale=100.0, length_scale_bounds=(1.0, 1000.0)) + WhiteKernel(noise_level=50.0, noise_level_bounds=(1.0, 500.0))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1.0, normalize_y=True)

# Cross-validation evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(gp, X_md, y_exp, cv=kf)

r2_cv = r2_score(y_exp, y_pred_cv)
mae_cv = mean_absolute_error(y_exp, y_pred_cv)
rmse_cv = np.sqrt(mean_squared_error(y_exp, y_pred_cv))

print(f"\nGP Calibration CV Performance:")
print(f"  R² = {r2_cv:.4f}")
print(f"  MAE = {mae_cv:.2f} K")
print(f"  RMSE = {rmse_cv:.2f} K")

# Fit on full calibration data
gp.fit(X_md, y_exp)
print(f"\nOptimized kernel: {gp.kernel_}")

# Apply calibration to vitrimer MD data
X_vitrimer_md = df_vitrimer['tg'].values.reshape(-1, 1)
tg_calibrated, tg_std = gp.predict(X_vitrimer_md, return_std=True)

df_vitrimer['tg_calibrated'] = tg_calibrated
df_vitrimer['tg_calibrated_std'] = tg_std

print(f"\nCalibrated Vitrimer Tg Statistics:")
print(f"  Mean: {tg_calibrated.mean():.2f} K")
print(f"  Std: {tg_calibrated.std():.2f} K")
print(f"  Min: {tg_calibrated.min():.2f} K")
print(f"  Max: {tg_calibrated.max():.2f} K")

# Save calibrated data
df_vitrimer.to_csv('../outputs/vitrimer_calibrated.csv', index=False)

# Save metrics
metrics = {
    'gp_r2_cv': float(r2_cv),
    'gp_mae_cv': float(mae_cv),
    'gp_rmse_cv': float(rmse_cv),
    'gp_kernel': str(gp.kernel_),
    'n_calibration': int(len(df_cal)),
    'n_vitrimer': int(len(df_vitrimer)),
    'calibrated_tg_mean': float(tg_calibrated.mean()),
    'calibrated_tg_std': float(tg_calibrated.std()),
    'calibrated_tg_min': float(tg_calibrated.min()),
    'calibrated_tg_max': float(tg_calibrated.max()),
}
with open('../outputs/gp_calibration_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ==============================
# Figure 1: GP Calibration Parity Plot
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1a: Parity plot
ax = axes[0]
ax.scatter(y_exp, y_pred_cv, alpha=0.6, s=30, c='steelblue', edgecolors='navy', linewidth=0.3)
lims = [min(y_exp.min(), y_pred_cv.min()) - 20, max(y_exp.max(), y_pred_cv.max()) + 20]
ax.plot(lims, lims, 'r--', lw=1.5, label='y = x')
ax.set_xlabel('Experimental Tg (K)', fontsize=12)
ax.set_ylabel('GP-Calibrated Tg (K)', fontsize=12)
ax.set_title(f'GP Calibration Parity Plot\nR²={r2_cv:.3f}, MAE={mae_cv:.1f} K', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')

# 1b: Residual distribution
residuals = y_exp - y_pred_cv
ax = axes[1]
ax.hist(residuals, bins=30, color='steelblue', edgecolor='navy', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', lw=1.5)
ax.set_xlabel('Residual (Exp - Calibrated) Tg (K)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Residual Distribution\nMean={residuals.mean():.1f} K, Std={residuals.std():.1f} K', fontsize=12)

# 1c: MD Tg vs Exp Tg with GP correction
ax = axes[2]
ax.scatter(df_cal['tg_md'], df_cal['tg_exp'], alpha=0.5, s=25, c='gray', label='Raw MD vs Exp')
# GP fit line
x_grid = np.linspace(df_cal['tg_md'].min()-20, df_cal['tg_md'].max()+20, 200).reshape(-1,1)
y_grid, y_grid_std = gp.predict(x_grid, return_std=True)
ax.plot(x_grid.ravel(), y_grid, 'r-', lw=2, label='GP calibration curve')
ax.fill_between(x_grid.ravel(), y_grid - 2*y_grid_std, y_grid + 2*y_grid_std, alpha=0.2, color='red', label='95% CI')
ax.set_xlabel('MD Simulated Tg (K)', fontsize=12)
ax.set_ylabel('Experimental Tg (K)', fontsize=12)
ax.set_title('GP Calibration: MD → Experimental Tg', fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('../report/images/fig1_gp_calibration.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig1_gp_calibration.png")

# ==============================
# Figure 2: Calibrated Vitrimer Tg Distribution
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist(df_vitrimer['tg'], bins=60, alpha=0.6, color='steelblue', edgecolor='navy', label='MD Tg (raw)')
ax.hist(tg_calibrated, bins=60, alpha=0.6, color='coral', edgecolor='darkred', label='Calibrated Tg')
ax.set_xlabel('Tg (K)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Vitrimer Tg Distribution: MD vs Calibrated', fontsize=12)
ax.legend(fontsize=10)

ax = axes[1]
ax.scatter(df_vitrimer['tg'], tg_calibrated, alpha=0.3, s=5, c='steelblue')
ax.plot([df_vitrimer['tg'].min()-10, df_vitrimer['tg'].max()+10],
        [df_vitrimer['tg'].min()-10, df_vitrimer['tg'].max()+10], 'r--', lw=1)
ax.set_xlabel('MD Tg (K)', fontsize=12)
ax.set_ylabel('Calibrated Tg (K)', fontsize=12)
ax.set_title('MD Tg vs Calibrated Tg for Vitrimers', fontsize=12)

plt.tight_layout()
plt.savefig('../report/images/fig2_vitrimer_tg_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig2_vitrimer_tg_distribution.png")

print("\nStep 1 complete.")
