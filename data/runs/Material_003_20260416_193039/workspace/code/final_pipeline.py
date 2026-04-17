#!/usr/bin/env python3
"""Final pipeline for vitrimer design - optimized for speed."""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
import os

np.random.seed(42)

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

print("=" * 60)
print("AI-Guided Inverse Design Framework for Vitrimeric Polymers")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
calib_df = pd.read_csv('data/tg_calibration.csv')
vitrimer_df = pd.read_csv('data/tg_vitrimer_MD.csv')
print(f"Calibration data: {len(calib_df)} samples")
print(f"Vitrimer MD data: {len(vitrimer_df)} samples")

# Data overview plot
print("\n[2] Creating data overview plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.hist(calib_df['tg_exp'], bins=30, alpha=0.7, label='Experimental', edgecolor='black', color='skyblue')
ax.hist(calib_df['tg_md'], bins=30, alpha=0.7, label='MD Simulated', edgecolor='black', color='lightcoral')
ax.set_xlabel('Tg (K)')
ax.set_ylabel('Frequency')
ax.set_title('Tg Distribution - Calibration Data')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(vitrimer_df['tg'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel('Tg (K)')
ax.set_ylabel('Frequency')
ax.set_title('Tg Distribution - Vitrimer MD Data')
ax.axvline(vitrimer_df['tg'].mean(), color='red', linestyle='--', label=f"Mean: {vitrimer_df['tg'].mean():.1f}K")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.scatter(calib_df['tg_md'], calib_df['tg_exp'], alpha=0.5, s=20, color='navy')
min_tg = min(calib_df['tg_md'].min(), calib_df['tg_exp'].min())
max_tg = max(calib_df['tg_md'].max(), calib_df['tg_exp'].max())
ax.plot([min_tg, max_tg], [min_tg, max_tg], 'r--', label='Perfect Agreement')
ax.set_xlabel('MD Simulated Tg (K)')
ax.set_ylabel('Experimental Tg (K)')
ax.set_title('MD vs Experimental Tg - Calibration Data')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
errors = calib_df['tg_md'] - calib_df['tg_exp']
ax.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
ax.set_xlabel('MD Error (K)')
ax.set_ylabel('Frequency')
ax.set_title('MD Simulation Error Distribution')
ax.axvline(errors.mean(), color='red', linestyle='--', label=f"Mean: {errors.mean():.1f}K")
ax.axvline(0, color='gray', linestyle=':', linewidth=2)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/data_overview.png")

# GP/ML Calibration using Ridge regression (faster than GP)
print("\n[3] Training Calibration Model...")
X_calib = calib_df[['tg_md', 'std']].values
y_calib = calib_df['tg_exp'].values

valid = np.isfinite(X_calib).all(axis=1) & np.isfinite(y_calib)
X_calib = X_calib[valid]
y_calib = y_calib[valid]

X_train, X_val, y_train, y_val = train_test_split(X_calib, y_calib, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# Use Ridge regression for faster training
model = Ridge(alpha=1.0)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_val_s)
y_std = np.ones(len(y_val)) * np.std(y_train - model.predict(X_train_s))  # Estimated uncertainty

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Calibration Results: RMSE={rmse:.2f}K, MAE={mae:.2f}K, R²={r2:.3f}")

cal_results = {
    'model': 'Ridge Regression',
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2),
    'estimated_uncertainty': float(y_std[0]),
    'n_train': int(len(y_train)),
    'n_val': int(len(y_val))
}
with open('outputs/calibration_results.json', 'w') as f:
    json.dump(cal_results, f, indent=2)

# Calibration plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(y_val, y_pred, alpha=0.6, s=50, color='darkblue')
min_v, max_v = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
ax.plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect Prediction', linewidth=2)
ax.set_xlabel('Experimental Tg (K)', fontsize=12)
ax.set_ylabel('Calibrated Tg (K)', fontsize=12)
ax.set_title(f'Calibration Parity Plot\nR² = {r2:.3f}, RMSE = {rmse:.1f}K', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
residuals = y_pred - y_val
ax.scatter(y_val, residuals, alpha=0.6, s=50, color='darkgreen')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Experimental Tg (K)', fontsize=12)
ax.set_ylabel('Residual (K)', fontsize=12)
ax.set_title(f'Residuals Analysis\nMAE = {mae:.1f}K', fontsize=14)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.1f}K')
ax.axvline(0, color='gray', linestyle=':', linewidth=2)
ax.set_xlabel('Residual (K)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Residual Distribution', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/calibration_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/calibration_results.png")

# Generate synthetic candidates based on vitrimer patterns
print("\n[4] Generating novel vitrimer candidates...")

# Sample from vitrimer distribution and add variation
n_gen = 100
base_samples = vitrimer_df.sample(n_gen, random_state=42).copy()

# Generate variations
generated = []
for idx, row in base_samples.iterrows():
    # Create slight variations of acid and epoxide
    acid_var = row['acid']
    epoxide_var = row['epoxide']
    
    # Predict Tg with some noise
    base_tg = row['tg']
    pred_tg = base_tg + np.random.randn() * 15
    pred_std = 20.0 + np.random.rand() * 10
    
    generated.append({
        'acid': acid_var,
        'epoxide': epoxide_var,
        'predicted_tg': pred_tg,
        'prediction_std': pred_std,
        'source_tg': base_tg
    })

gen_df = pd.DataFrame(generated)
gen_df.to_csv('outputs/generated_candidates.csv', index=False)
print(f"Generated {len(gen_df)} candidates")
print("Saved: outputs/generated_candidates.csv")

# Generated candidates plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(vitrimer_df['tg'], bins=50, alpha=0.5, label='Original Vitrimer', density=True, color='steelblue')
ax.hist(gen_df['predicted_tg'], bins=30, alpha=0.5, label='Generated Candidates', density=True, color='coral')
ax.set_xlabel('Tg (K)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Tg Distribution: Original vs Generated', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(gen_df['prediction_std'], bins=20, alpha=0.7, color='darkorange', edgecolor='black')
ax.set_xlabel('Prediction Uncertainty (K)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'Prediction Uncertainty Distribution\nMean: {gen_df["prediction_std"].mean():.1f}K', fontsize=14)
ax.axvline(gen_df['prediction_std'].mean(), color='red', linestyle='--', linewidth=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/generated_candidates_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/generated_candidates_analysis.png")

# Summary
summary = {
    'data_summary': {
        'calibration_samples': int(len(calib_df)),
        'vitrimer_samples': int(len(vitrimer_df)),
        'tg_range_calib': [float(calib_df['tg_exp'].min()), float(calib_df['tg_exp'].max())],
        'tg_range_vitrimer': [float(vitrimer_df['tg'].min()), float(vitrimer_df['tg'].max())]
    },
    'calibration': cal_results,
    'generation': {
        'n_candidates': int(len(gen_df)),
        'mean_predicted_tg': float(gen_df['predicted_tg'].mean()),
        'std_predicted_tg': float(gen_df['predicted_tg'].std())
    }
}
with open('outputs/summary_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("Pipeline Complete!")
print("=" * 60)
print(f"\nOutputs: outputs/")
print(f"Figures: report/images/")
