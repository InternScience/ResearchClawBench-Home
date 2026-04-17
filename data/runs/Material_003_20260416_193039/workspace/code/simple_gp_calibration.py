#!/usr/bin/env python3
"""Simple GP calibration script."""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
import os

np.random.seed(42)

# Load data
calib_df = pd.read_csv('data/tg_calibration.csv')
vitrimer_df = pd.read_csv('data/tg_vitrimer_MD.csv')

print('Calibration data:', len(calib_df), 'samples')
print('Vitrimer data:', len(vitrimer_df), 'samples')

# Create simple features from MD Tg and basic statistics
X_calib = calib_df[['tg_md', 'std']].values
y_calib = calib_df['tg_exp'].values

valid = np.isfinite(X_calib).all(axis=1) & np.isfinite(y_calib)
X_calib = X_calib[valid]
y_calib = y_calib[valid]

print('Training GP on', len(y_calib), 'samples...')

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_calib, y_calib, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# GP
kernel = C(1.0) * RBF(1.0) + WhiteKernel(0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
gp.fit(X_train_s, y_train)

y_pred, y_std = gp.predict(X_val_s, return_std=True)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f'GP Results: RMSE={rmse:.2f}K, MAE={mae:.2f}K, R²={r2:.3f}')

# Save results
results = {
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2),
    'mean_uncertainty': float(np.mean(y_std)),
    'n_train': int(len(y_train)),
    'n_val': int(len(y_val))
}
with open('outputs/gp_calibration_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.scatter(y_val, y_pred, alpha=0.6)
min_v, max_v = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
ax.plot([min_v, max_v], [min_v, max_v], 'r--')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('GP Predicted Tg (K)')
ax.set_title(f'Parity Plot\nR² = {r2:.3f}, RMSE = {rmse:.1f}K')
ax.grid(True, alpha=0.3)

ax = axes[1]
residuals = y_pred - y_val
ax.scatter(y_val, residuals, alpha=0.6)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('Residual (K)')
ax.set_title(f'Residuals\nMAE = {mae:.1f}K')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
ax.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: {residuals.mean():.1f}K')
ax.set_xlabel('Residual (K)')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/gp_calibration_results.png', dpi=150, bbox_inches='tight')
plt.close()

print('GP calibration complete!')
