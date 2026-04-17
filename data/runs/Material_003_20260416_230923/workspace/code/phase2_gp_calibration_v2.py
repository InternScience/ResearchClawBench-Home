#!/usr/bin/env python3
"""
Phase 2 (Revised): GP Calibration with molecular fingerprints
Uses Morgan fingerprints + MD Tg as features to predict experimental Tg
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern, DotProduct
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import json
import os
import warnings
warnings.filterwarnings('ignore')

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

# Load data
df_cal = pd.read_csv(os.path.join(BASE, 'data', 'tg_calibration.csv'))
df_vit = pd.read_csv(os.path.join(BASE, 'data', 'tg_vitrimer_MD.csv'))

print("=" * 60)
print("PHASE 2 (Revised): GP Calibration with Molecular Features")
print("=" * 60)

# ============================================================
# Feature Engineering: Morgan Fingerprints + MD Tg
# ============================================================
def smiles_to_fingerprint(smiles, radius=2, nbits=256):
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)

def get_descriptors(smiles):
    """Get basic molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 6
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
    ]

# Build features for calibration data
print("\nBuilding features for calibration data...")
FP_BITS = 128
fps_cal = np.array([smiles_to_fingerprint(s, nbits=FP_BITS) for s in df_cal['smiles']])
desc_cal = np.array([get_descriptors(s) for s in df_cal['smiles']])

# Combine: [MD Tg, MD std, fingerprints, descriptors]
X_cal_full = np.column_stack([
    df_cal['tg_md'].values,
    df_cal['std'].values,
    fps_cal,
    desc_cal
])
y_cal = df_cal['tg_exp'].values

print(f"Feature matrix shape: {X_cal_full.shape}")
print(f"  - MD features: 2 (tg_md, std)")
print(f"  - Morgan FP bits: {FP_BITS}")
print(f"  - Descriptors: 6")

# Scale features
scaler = StandardScaler()
X_cal_scaled = scaler.fit_transform(X_cal_full)

# PCA for dimensionality reduction (GP works better in lower dims)
n_components = 30
pca = PCA(n_components=n_components)
X_cal_pca = pca.fit_transform(X_cal_scaled)
print(f"PCA: {n_components} components explain {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

# ============================================================
# GP Calibration with enhanced features
# ============================================================
kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_components), nu=2.5) + WhiteKernel(noise_level=10.0)

# Cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_predictions = np.zeros_like(y_cal)
cv_uncertainties = np.zeros_like(y_cal)
cv_metrics = []

print(f"\nRunning {n_folds}-fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_cal_pca)):
    X_train, X_test = X_cal_pca[train_idx], X_cal_pca[test_idx]
    y_train, y_test = y_cal[train_idx], y_cal[test_idx]
    
    gp = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=5, 
        alpha=df_cal['std'].values[train_idx]**2 / (y_train.std()**2 + 1e-8),
        normalize_y=True
    )
    gp.fit(X_train, y_train)
    
    y_pred, y_std = gp.predict(X_test, return_std=True)
    cv_predictions[test_idx] = y_pred
    cv_uncertainties[test_idx] = y_std
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    cv_metrics.append({'fold': fold+1, 'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)})
    print(f"  Fold {fold+1}: MAE={mae:.2f} K, RMSE={rmse:.2f} K, R²={r2:.4f}")

# Overall CV metrics
overall_mae = mean_absolute_error(y_cal, cv_predictions)
overall_rmse = np.sqrt(mean_squared_error(y_cal, cv_predictions))
overall_r2 = r2_score(y_cal, cv_predictions)

print(f"\nOverall CV: MAE={overall_mae:.2f} K, RMSE={overall_rmse:.2f} K, R²={overall_r2:.4f}")

# Also compute baseline (just using MD Tg directly)
baseline_mae = mean_absolute_error(y_cal, df_cal['tg_md'].values)
baseline_rmse = np.sqrt(mean_squared_error(y_cal, df_cal['tg_md'].values))
baseline_r2 = r2_score(y_cal, df_cal['tg_md'].values)
print(f"Baseline (raw MD): MAE={baseline_mae:.2f} K, RMSE={baseline_rmse:.2f} K, R²={baseline_r2:.4f}")

# Train final model on all data
print("\nTraining final GP model on all calibration data...")
gp_final = GaussianProcessRegressor(
    kernel=kernel, 
    n_restarts_optimizer=10, 
    alpha=df_cal['std'].values**2 / (y_cal.std()**2 + 1e-8),
    normalize_y=True
)
gp_final.fit(X_cal_pca, y_cal)
print(f"Optimized kernel: {gp_final.kernel_}")

# Full model predictions on training data (for visualization)
y_train_pred, y_train_std = gp_final.predict(X_cal_pca, return_std=True)

# Save GP metrics
gp_results = {
    'cv_metrics': cv_metrics,
    'overall': {
        'mae': float(overall_mae),
        'rmse': float(overall_rmse),
        'r2': float(overall_r2)
    },
    'baseline': {
        'mae': float(baseline_mae),
        'rmse': float(baseline_rmse),
        'r2': float(baseline_r2)
    },
    'final_kernel': str(gp_final.kernel_),
    'n_training_samples': int(len(X_cal_pca)),
    'n_features': int(X_cal_full.shape[1]),
    'n_pca_components': n_components,
    'pca_variance_explained': float(pca.explained_variance_ratio_.sum())
}
with open(os.path.join(OUT_DIR, 'gp_calibration_results.json'), 'w') as f:
    json.dump(gp_results, f, indent=2)

# --- Figure 2: GP Calibration Parity Plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 2a: CV Parity plot
ax = axes[0]
scatter = ax.scatter(y_cal, cv_predictions, c=cv_uncertainties, cmap='viridis', 
                     alpha=0.7, edgecolor='black', linewidth=0.5, s=50)
min_v = min(y_cal.min(), cv_predictions.min()) - 20
max_v = max(y_cal.max(), cv_predictions.max()) + 20
ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect calibration')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('GP-Predicted Tg (K)')
ax.set_title(f'(a) GP Calibration: 5-Fold CV\nMAE={overall_mae:.1f} K, RMSE={overall_rmse:.1f} K, R²={overall_r2:.3f}')
ax.legend(fontsize=12)
ax.set_xlim(min_v, max_v)
ax.set_ylim(min_v, max_v)
ax.set_aspect('equal')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Prediction Uncertainty (K)')

# 2b: Residuals
ax = axes[1]
residuals = cv_predictions - y_cal
ax.scatter(y_cal, residuals, c='steelblue', alpha=0.6, edgecolor='black', linewidth=0.5, s=50)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.fill_between([min_v, max_v], 
                [residuals.mean() - 2*residuals.std()]*2, 
                [residuals.mean() + 2*residuals.std()]*2, 
                alpha=0.1, color='gray', label='±2σ')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('Residual (Predicted - Experimental) (K)')
ax.set_title(f'(b) Residual Analysis\nMean={residuals.mean():.1f} K, Std={residuals.std():.1f} K')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_gp_calibration.png'))
plt.close()
print("Saved fig2_gp_calibration.png")

# ============================================================
# Apply GP Calibration to Vitrimer Data
# ============================================================
print("\nBuilding features for vitrimer data...")

# For vitrimers, we have acid + epoxide SMILES
# Combine their fingerprints
def vitrimer_features(acid_smiles, epoxide_smiles, tg_md, std_md, fp_bits=FP_BITS):
    """Build features for a vitrimer system (acid + epoxide)."""
    fp_acid = smiles_to_fingerprint(acid_smiles, nbits=fp_bits)
    fp_epoxide = smiles_to_fingerprint(epoxide_smiles, nbits=fp_bits)
    # Combine fingerprints (bitwise OR or concatenate)
    fp_combined = np.maximum(fp_acid, fp_epoxide)  # Union of bits
    
    desc_acid = get_descriptors(acid_smiles)
    desc_epoxide = get_descriptors(epoxide_smiles)
    # Average descriptors
    desc_combined = [(a + b) / 2 for a, b in zip(desc_acid, desc_epoxide)]
    
    return np.concatenate([[tg_md, std_md], fp_combined, desc_combined])

print("Computing vitrimer features (this may take a moment)...")
vit_features = []
valid_mask = []
for idx, row in df_vit.iterrows():
    try:
        feat = vitrimer_features(row['acid'], row['epoxide'], row['tg'], row['std'])
        vit_features.append(feat)
        valid_mask.append(True)
    except:
        vit_features.append(np.zeros(2 + FP_BITS + 6))
        valid_mask.append(False)

X_vit_full = np.array(vit_features)
valid_mask = np.array(valid_mask)
print(f"Valid vitrimers: {valid_mask.sum()} / {len(valid_mask)}")

# Scale and PCA transform
X_vit_scaled = scaler.transform(X_vit_full)
X_vit_pca = pca.transform(X_vit_scaled)

# Predict calibrated Tg
print("Predicting calibrated Tg for vitrimers...")
tg_calibrated, tg_calibrated_std = gp_final.predict(X_vit_pca, return_std=True)

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

# 3b: MD Tg vs Calibrated Tg scatter
axes[1].scatter(df_vit['tg'], tg_calibrated, c=tg_calibrated_std, cmap='viridis', 
               alpha=0.3, s=10)
axes[1].plot([300, 570], [300, 570], 'r--', label='y = x')
axes[1].set_xlabel('MD Simulated Tg (K)')
axes[1].set_ylabel('Calibrated Tg (K)')
axes[1].set_title('(b) MD vs Calibrated Tg')
cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('Uncertainty (K)')
axes[1].legend()

# 3c: Uncertainty distribution
axes[2].hist(tg_calibrated_std, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Prediction Uncertainty (K)')
axes[2].set_ylabel('Count')
axes[2].set_title(f'(c) Calibration Uncertainty Distribution\nMean={tg_calibrated_std.mean():.1f} K')
axes[2].axvline(tg_calibrated_std.mean(), color='red', linestyle='--', 
               label=f'Mean = {tg_calibrated_std.mean():.1f} K')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_calibrated_tg.png'))
plt.close()
print("Saved fig3_calibrated_tg.png")

# --- Figure: Comparison of baseline vs GP calibration ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(y_cal, df_cal['tg_md'].values, c='coral', alpha=0.5, label=f'Raw MD (MAE={baseline_mae:.1f} K)', s=40)
ax.scatter(y_cal, cv_predictions, c='steelblue', alpha=0.5, label=f'GP Calibrated (MAE={overall_mae:.1f} K)', s=40)
ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Experimental Tg (K)')
ax.set_ylabel('Predicted Tg (K)')
ax.set_title('Comparison: Raw MD vs GP-Calibrated Tg Predictions')
ax.legend(fontsize=12)
ax.set_xlim(min_v, max_v)
ax.set_ylim(min_v, max_v)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4_calibration_comparison.png'))
plt.close()
print("Saved fig4_calibration_comparison.png")

print("\nPhase 2 (Revised) complete!")
