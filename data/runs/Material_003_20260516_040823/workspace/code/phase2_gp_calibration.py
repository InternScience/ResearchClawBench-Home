#!/usr/bin/env python3
"""
Phase 2: Gaussian Process Calibration
Train a GP model to calibrate MD-simulated Tg to experimental Tg values.
Apply calibration to vitrimer MD data.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler

# Setup paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_003_20260516_040823')
DATA_DIR = WORKSPACE / 'data'
OUTPUTS_DIR = WORKSPACE / 'outputs'
IMAGES_DIR = WORKSPACE / 'report' / 'images'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)

print("Loading data...")
calib_df = pd.read_csv(DATA_DIR / 'tg_calibration.csv')
vitrimer_df = pd.read_csv(DATA_DIR / 'tg_vitrimer_MD.csv')

# ============================================================
# Prepare data for GP
# ============================================================
X_md = calib_df['tg_md'].values.reshape(-1, 1).astype(np.float64)
y_exp = calib_df['tg_exp'].values.reshape(-1).astype(np.float64)
y_std = calib_df['std'].values.reshape(-1).astype(np.float64)
y_var = y_std ** 2

# Normalize for GP training stability
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_md_scaled = x_scaler.fit_transform(X_md)
y_exp_scaled = y_scaler.fit_transform(y_exp.reshape(-1, 1)).flatten()

# Convert to tensors
train_x = torch.tensor(X_md_scaled, dtype=torch.float32)
train_y = torch.tensor(y_exp_scaled, dtype=torch.float32)

# ============================================================
# Define GP Model
# ============================================================
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ============================================================
# Train GP
# ============================================================
print("Training GP calibration model...")
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Set to training mode
model.train()
likelihood.train()

# Use Adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.1)

# Loss
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 500
losses = []
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (i + 1) % 100 == 0:
        print(f"  Iter {i+1}/{n_iter}, Loss: {loss.item():.4f}")

print(f"GP training complete. Final loss: {losses[-1]:.4f}")
print(f"Kernel lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}")
print(f"Kernel outputscale: {model.covar_module.outputscale.item():.4f}")
print(f"Noise: {likelihood.noise.item():.4f}")

# ============================================================
# Evaluate GP predictions
# ============================================================
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(train_x))
    y_pred_mean_scaled = preds.mean.numpy()
    y_pred_lower_scaled, y_pred_upper_scaled = preds.confidence_region()

# Inverse transform
y_pred_mean = y_scaler.inverse_transform(y_pred_mean_scaled.reshape(-1, 1)).flatten()
y_pred_lower = y_scaler.inverse_transform(y_pred_lower_scaled.numpy().reshape(-1, 1)).flatten()
y_pred_upper = y_scaler.inverse_transform(y_pred_upper_scaled.numpy().reshape(-1, 1)).flatten()

# Compute metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_exp, y_pred_mean)
mae = mean_absolute_error(y_exp, y_pred_mean)
rmse = np.sqrt(mean_squared_error(y_exp, y_pred_mean))

calibration_metrics = {
    'r2': float(r2),
    'mae': float(mae),
    'rmse': float(rmse),
    'kernel_lengthscale': float(model.covar_module.base_kernel.lengthscale.item()),
    'kernel_outputscale': float(model.covar_module.outputscale.item()),
    'noise': float(likelihood.noise.item()),
}

with open(OUTPUTS_DIR / 'gp_calibration_metrics.json', 'w') as f:
    json.dump(calibration_metrics, f, indent=2)

print(f"\nGP Calibration Metrics:")
print(f"  R² = {r2:.4f}")
print(f"  MAE = {mae:.2f} K")
print(f"  RMSE = {rmse:.2f} K")

# ============================================================
# Apply GP to vitrimer data
# ============================================================
print("\nApplying GP calibration to vitrimer MD data...")
vitrimer_tg_md = vitrimer_df['tg'].values.reshape(-1, 1).astype(np.float64)
vitrimer_tg_std = vitrimer_df['std'].values.astype(np.float64)

# Scale
vitrimer_x_scaled = x_scaler.transform(vitrimer_tg_md)
vitrimer_x_tensor = torch.tensor(vitrimer_x_scaled, dtype=torch.float32)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    vitrimer_preds = likelihood(model(vitrimer_x_tensor))
    vitrimer_pred_mean_scaled = vitrimer_preds.mean.numpy()
    vitrimer_pred_lower_scaled, vitrimer_pred_upper_scaled = vitrimer_preds.confidence_region()

# Inverse transform
vitrimer_pred_mean = y_scaler.inverse_transform(vitrimer_pred_mean_scaled.reshape(-1, 1)).flatten()
vitrimer_pred_lower = y_scaler.inverse_transform(vitrimer_pred_lower_scaled.numpy().reshape(-1, 1)).flatten()
vitrimer_pred_upper = y_scaler.inverse_transform(vitrimer_pred_upper_scaled.numpy().reshape(-1, 1)).flatten()

# Add to dataframe
vitrimer_df['tg_calibrated'] = vitrimer_pred_mean
vitrimer_df['tg_cal_lower'] = vitrimer_pred_lower
vitrimer_df['tg_cal_upper'] = vitrimer_pred_upper

# Save calibrated vitrimer data
vitrimer_df.to_csv(OUTPUTS_DIR / 'vitrimer_calibrated.csv', index=False)
print(f"Saved calibrated vitrimer data: {len(vitrimer_df)} entries")
print(f"Calibrated Tg range: [{vitrimer_pred_mean.min():.1f}, {vitrimer_pred_mean.max():.1f}]")
print(f"Calibrated Tg mean: {vitrimer_pred_mean.mean():.1f} ± {vitrimer_pred_mean.std():.1f}")

# ============================================================
# Figure 4: GP Calibration Results
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: GP fit with confidence band
ax = axes[0, 0]
# Sort for plotting
sort_idx = np.argsort(X_md.flatten())
x_sorted = X_md.flatten()[sort_idx]
y_pred_sorted = y_pred_mean[sort_idx]
y_lower_sorted = y_pred_lower[sort_idx]
y_upper_sorted = y_pred_upper[sort_idx]

ax.fill_between(x_sorted, y_lower_sorted, y_upper_sorted, alpha=0.3, color='blue', label='95% CI')
ax.plot(x_sorted, y_pred_sorted, 'b-', linewidth=2, label='GP mean')
ax.scatter(X_md, y_exp, c='red', s=15, alpha=0.6, label='Data points')
ax.plot([X_md.min(), X_md.max()], [X_md.min(), X_md.max()], 'k--', alpha=0.5, label='y = x')
ax.set_xlabel('MD Tg (K)')
ax.set_ylabel('Experimental Tg (K)')
ax.set_title(f'A: GP Calibration (R²={r2:.3f}, MAE={mae:.1f} K)')
ax.legend()

# Panel B: Residuals
ax = axes[0, 1]
calib_residuals = y_exp - y_pred_mean
ax.scatter(y_pred_mean, calib_residuals, c='steelblue', s=15, alpha=0.6)
ax.axhline(0, color='black', linestyle='--')
ax.axhline(calib_residuals.mean(), color='red', linestyle='-', 
           label=f'Mean = {calib_residuals.mean():.1f} K')
ax.set_xlabel('GP Predicted Tg (K)')
ax.set_ylabel('Residual (K)')
ax.set_title(f'B: GP Residuals (RMSE={rmse:.1f} K)')
ax.legend()

# Panel C: GP training loss
ax = axes[1, 0]
ax.plot(losses, 'b-', linewidth=1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Negative Log Marginal Likelihood')
ax.set_title('C: GP Training Loss')
ax.set_yscale('log')

# Panel D: Calibrated vitrimer Tg distribution
ax = axes[1, 1]
ax.hist(vitrimer_df['tg'], bins=60, alpha=0.6, color='coral', label='MD Tg (raw)', density=True)
ax.hist(vitrimer_df['tg_calibrated'], bins=60, alpha=0.6, color='steelblue', label='Calibrated Tg', density=True)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Density')
ax.set_title('D: Vitrimer Tg Before/After Calibration')
ax.legend()

plt.tight_layout()
fig.savefig(IMAGES_DIR / 'figure4_gp_calibration.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figure4_gp_calibration.png")

# ============================================================
# Save GP model parameters for later use
# ============================================================
import pickle
torch.save({
    'model_state_dict': model.state_dict(),
    'likelihood_state_dict': likelihood.state_dict(),
    'x_scaler_mean': x_scaler.mean_.tolist(),
    'x_scaler_scale': x_scaler.scale_.tolist(),
    'y_scaler_mean': y_scaler.mean_.tolist(),
    'y_scaler_scale': y_scaler.scale_.tolist(),
}, OUTPUTS_DIR / 'gp_model.pt')
print("Saved: gp_model.pt")

print("\nPhase 2 complete!")
