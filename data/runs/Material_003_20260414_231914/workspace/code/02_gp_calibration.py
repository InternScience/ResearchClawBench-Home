"""
Step 2: Gaussian Process Calibration Model.
Train GP to map MD-simulated Tg -> Experimental Tg.
Apply to vitrimer data for calibrated predictions.
"""
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import pickle

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
cal = pd.read_csv('outputs/calibration_data.csv')
cal_desc = pd.read_csv('outputs/calibration_descriptors.csv')
vit = pd.read_csv('outputs/vitrimer_data.csv')
vit_desc = pd.read_csv('outputs/vitrimer_descriptors.csv')

# Features: MD Tg + molecular descriptors
X_cal = np.column_stack([cal['tg_md'].values, cal_desc.values])
y_cal = cal['tg_exp'].values
X_vit = np.column_stack([vit['tg'].values, vit_desc.values])

print(f"Calibration features shape: {X_cal.shape}")
print(f"Vitrimer features shape: {X_vit.shape}")

# Standardize features
scaler_X = StandardScaler()
X_cal_scaled = scaler_X.fit_transform(X_cal)
X_vit_scaled = scaler_X.transform(X_vit)

# Cross-validation on calibration data
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_preds = np.zeros(len(y_cal))
cv_stds = np.zeros(len(y_cal))

for train_idx, test_idx in kf.split(X_cal_scaled):
    fold_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
    gp = GaussianProcessRegressor(kernel=fold_kernel, n_restarts_optimizer=10, random_state=42)
    gp.fit(X_cal_scaled[train_idx], y_cal[train_idx])
    pred, std = gp.predict(X_cal_scaled[test_idx], return_std=True)
    cv_preds[test_idx] = pred
    cv_stds[test_idx] = std

# Metrics
mae = mean_absolute_error(y_cal, cv_preds)
rmse = np.sqrt(mean_squared_error(y_cal, cv_preds))
r2 = r2_score(y_cal, cv_preds)
print(f"\nCross-validation Results:")
print(f"  MAE: {mae:.2f} K")
print(f"  RMSE: {rmse:.2f} K")
print(f"  R2: {r2:.4f}")

# Train final GP on all calibration data
final_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
gp_final = GaussianProcessRegressor(kernel=final_kernel, n_restarts_optimizer=10, random_state=42)
gp_final.fit(X_cal_scaled, y_cal)
print(f"\nOptimized kernel: {gp_final.kernel_}")

# Predict on vitrimer data
vit_calibrated, vit_std = gp_final.predict(X_vit_scaled, return_std=True)

# Save calibrated predictions
vit['tg_calibrated'] = vit_calibrated
vit['tg_calibrated_std'] = vit_std
vit.to_csv('outputs/vitrimer_calibrated.csv', index=False)

# Save calibration results
cal_results = {
    'cv_mae': float(mae),
    'cv_rmse': float(rmse),
    'cv_r2': float(r2),
    'optimized_kernel': str(gp_final.kernel_),
}
with open('outputs/gp_calibration_results.json', 'w') as f:
    json.dump(cal_results, f, indent=2)

# Save model
with open('outputs/gp_model.pkl', 'wb') as f:
    pickle.dump({'model': gp_final, 'scaler': scaler_X}, f)

# --- Plots ---
fig, axes = plt.subplots(2, 3, figsize=(16, 11))

ax = axes[0, 0]
ax.scatter(y_cal, cv_preds, alpha=0.6, s=30, c='steelblue')
lims = [min(y_cal.min(), cv_preds.min()) - 20, max(y_cal.max(), cv_preds.max()) + 20]
ax.plot(lims, lims, 'k--', alpha=0.5)
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('GP Predicted Tg (K)')
ax.set_title(f'GP Calibration CV (R2={r2:.3f}, MAE={mae:.1f} K)')

ax = axes[0, 1]
residuals = cv_preds - y_cal
ax.scatter(y_cal, residuals, alpha=0.6, s=30, c='coral')
ax.axhline(0, color='k', linestyle='--')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('Residual (K)')
ax.set_title('GP Calibration Residuals')

ax = axes[0, 2]
abs_residuals = np.abs(residuals)
ax.scatter(cv_stds, abs_residuals, alpha=0.6, s=30, c='seagreen')
ax.set_xlabel('GP Predicted Std (K)')
ax.set_ylabel('|Residual| (K)')
ax.set_title('Uncertainty Calibration')

ax = axes[1, 0]
ax.hist(vit['tg'], bins=50, alpha=0.5, label='MD Tg', color='steelblue')
ax.hist(vit['tg_calibrated'], bins=50, alpha=0.5, label='Calibrated Tg', color='coral')
ax.set_xlabel('Tg (K)')
ax.set_ylabel('Count')
ax.set_title('Vitrimer Tg: MD vs Calibrated')
ax.legend()

ax = axes[1, 1]
sample_idx = np.random.choice(len(vit), min(2000, len(vit)), replace=False)
ax.scatter(vit['tg'].values[sample_idx], vit['tg_calibrated'].values[sample_idx], alpha=0.3, s=10, c='steelblue')
lims2 = [vit['tg'].min() - 10, vit['tg'].max() + 10]
ax.plot(lims2, lims2, 'k--', alpha=0.5)
ax.set_xlabel('MD Simulated Tg (K)')
ax.set_ylabel('GP Calibrated Tg (K)')
ax.set_title('Vitrimer: MD vs Calibrated Tg')

ax = axes[1, 2]
shift = vit['tg_calibrated'] - vit['tg']
ax.hist(shift, bins=50, alpha=0.7, color='seagreen')
ax.axvline(0, color='k', linestyle='--')
ax.set_xlabel('Calibration Shift (K)')
ax.set_ylabel('Count')
ax.set_title(f'Tg Calibration Shift (Mean: {shift.mean():.1f} K)')

plt.tight_layout()
plt.savefig('report/images/gp_calibration.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nGP calibration complete.")
print(f"Vitrimer calibrated Tg range: [{vit['tg_calibrated'].min():.1f}, {vit['tg_calibrated'].max():.1f}] K")
print("Plots saved to report/images/gp_calibration.png")
