"""
Phase 2: Gaussian Process Calibration Model
Trains GP to map MD-simulated Tg + molecular fingerprints -> experimental Tg.
Evaluates on held-out test set and applies to vitrimer dataset.
"""
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
TMP = '/tmp/vitrimer_work'

# Load data
cal_df = pd.read_csv('data/tg_calibration.csv')
vit_df = pd.read_csv('data/tg_vitrimer_MD.csv')

# Load fingerprints
cal_fps = np.load(os.path.join(TMP, 'cal_fps.npy')).astype(np.float64)
vit_fps = np.load(os.path.join(TMP, 'vit_fps.npy')).astype(np.float64)

print(f"Calibration FP shape: {cal_fps.shape}")
print(f"Vitrimer FP shape: {vit_fps.shape}")

y_cal = cal_df['tg_exp'].values
print(f"y_cal range: [{y_cal.min():.1f}, {y_cal.max():.1f}]")

# PCA on calibration fingerprints (1024-dim)
pca = PCA(n_components=30, random_state=42)
fp_cal_pca = pca.fit_transform(cal_fps)
print(f"Calibration PCA total variance: {pca.explained_variance_ratio_.sum():.4f}")

X_cal_pca = np.hstack([cal_df['tg_md'].values.reshape(-1, 1), fp_cal_pca])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_cal_pca, y_cal, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Build GP kernel
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

gp = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10, normalize_y=True, alpha=1e-6, random_state=42
)
print("Training GP model...")
gp.fit(X_train, y_train)

# Predictions
y_pred, y_std = gp.predict(X_test, return_std=True)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nGP Test Results:")
print(f"  MAE: {mae:.2f} K")
print(f"  RMSE: {rmse:.2f} K")
print(f"  R2: {r2:.4f}")

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae, cv_rmse, cv_r2 = [], [], []
for fold, (tr_idx, te_idx) in enumerate(kf.split(X_cal_pca)):
    X_tr, X_te = X_cal_pca[tr_idx], X_cal_pca[te_idx]
    y_tr, y_te = y_cal[tr_idx], y_cal[te_idx]
    gp_cv = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42
    )
    gp_cv.fit(X_tr, y_tr)
    y_te_pred, _ = gp_cv.predict(X_te, return_std=True)
    cv_mae.append(mean_absolute_error(y_te, y_te_pred))
    cv_rmse.append(np.sqrt(mean_squared_error(y_te, y_te_pred)))
    cv_r2.append(r2_score(y_te, y_te_pred))

print(f"\n5-Fold CV Results:")
print(f"  MAE: {np.mean(cv_mae):.2f} +/- {np.std(cv_mae):.2f} K")
print(f"  RMSE: {np.mean(cv_rmse):.2f} +/- {np.std(cv_rmse):.2f} K")
print(f"  R2: {np.mean(cv_r2):.4f} +/- {np.std(cv_r2):.4f}")

# Train final GP on all calibration data
print("\nTraining final GP on full calibration set...")
gp_full = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42
)
gp_full.fit(X_cal_pca, y_cal)
y_cal_pred, y_cal_std = gp_full.predict(X_cal_pca, return_std=True)

# Apply to vitrimer dataset
# Vitrimer FP is [acid_fp(1024), epox_fp(1024)] - average them to match calibration format
vit_acid_fp = vit_fps[:, :1024]
vit_epox_fp = vit_fps[:, 1024:]
vit_avg_fp = (vit_acid_fp + vit_epox_fp) / 2.0
vit_avg_pca = pca.transform(vit_avg_fp)
X_vit = np.hstack([vit_df['tg'].values.reshape(-1, 1), vit_avg_pca])

print("Predicting calibrated Tg for vitrimer dataset...")
y_vit_pred, y_vit_std = gp_full.predict(X_vit, return_std=True)

print(f"\nVitrimer calibrated Tg stats:")
print(f"  Mean: {y_vit_pred.mean():.2f} K")
print(f"  Std: {y_vit_pred.std():.2f} K")
print(f"  Range: [{y_vit_pred.min():.2f}, {y_vit_pred.max():.2f}] K")
print(f"  Mean uncertainty: {y_vit_std.mean():.2f} K")

# Save results
cal_results = pd.DataFrame({
    'name': cal_df['name'],
    'smiles': cal_df['smiles'],
    'tg_exp': cal_df['tg_exp'],
    'tg_md': cal_df['tg_md'],
    'tg_calibrated': y_cal_pred,
    'calibration_std': y_cal_std,
})
cal_results.to_csv('outputs/gp_calibration_results.csv', index=False)

vit_results = pd.DataFrame({
    'acid': vit_df['acid'],
    'epoxide': vit_df['epoxide'],
    'tg_md': vit_df['tg'],
    'tg_calibrated': y_vit_pred,
    'calibration_std': y_vit_std,
})
vit_results.to_csv('outputs/gp_vitrimer_predictions.csv', index=False)

test_results = pd.DataFrame({
    'tg_exp': y_test,
    'tg_pred': y_pred,
    'pred_std': y_std,
})
test_results.to_csv('outputs/gp_test_predictions.csv', index=False)

cv_results = pd.DataFrame({
    'fold': range(5),
    'mae': cv_mae,
    'rmse': cv_rmse,
    'r2': cv_r2,
})
cv_results.to_csv('outputs/gp_cv_results.csv', index=False)

metrics = {
    'test_mae': float(mae),
    'test_rmse': float(rmse),
    'test_r2': float(r2),
    'cv_mae_mean': float(np.mean(cv_mae)),
    'cv_mae_std': float(np.std(cv_mae)),
    'cv_rmse_mean': float(np.mean(cv_rmse)),
    'cv_rmse_std': float(np.std(cv_rmse)),
    'cv_r2_mean': float(np.mean(cv_r2)),
    'cv_r2_std': float(np.std(cv_r2)),
    'gp_kernel': str(gp.kernel_),
    'pca_n_components': 30,
    'pca_total_variance': float(pca.explained_variance_ratio_.sum()),
    'n_calibration': len(cal_df),
    'n_vitrimer': len(vit_df),
    'vit_tg_cal_mean': float(y_vit_pred.mean()),
    'vit_tg_cal_std': float(y_vit_pred.std()),
    'vit_tg_cal_min': float(y_vit_pred.min()),
    'vit_tg_cal_max': float(y_vit_pred.max()),
}
with open('outputs/gp_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

np.save(os.path.join(TMP, 'pca_components.npy'), pca.components_)
np.save(os.path.join(TMP, 'pca_mean.npy'), pca.mean_)

print("\nGP calibration complete. Results saved to outputs/")
