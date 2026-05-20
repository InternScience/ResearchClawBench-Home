import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, RationalQuadratic, Matern
import os
import json
import joblib

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
cal = pd.read_csv('data/tg_calibration.csv')

# Compute Morgan fingerprints for calibration data
def morgan_fp(smiles, radius=2, n_bits=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))

X_fp = np.array([morgan_fp(s) for s in cal['smiles']])
X_md = cal['tg_md'].values.reshape(-1, 1)
X = np.hstack([X_fp, X_md])
y = cal['tg_exp'].values

# Train-test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.2, random_state=42
)

# Standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# GP Model
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False, alpha=1e-5)

print("Training GP model...")
gp.fit(X_train_s, y_train_s)
print(f"Optimized kernel: {gp.kernel_}")

# Predictions
y_train_pred_s = gp.predict(X_train_s, return_std=True)
y_test_pred_s = gp.predict(X_test_s, return_std=True)

y_train_pred = scaler_y.inverse_transform(y_train_pred_s[0].reshape(-1, 1)).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_s[0].reshape(-1, 1)).flatten()
y_train_std = y_train_pred_s[1] * scaler_y.scale_[0]
y_test_std = y_test_pred_s[1] * scaler_y.scale_[0]

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nTrain RMSE: {train_rmse:.2f} K, R²: {train_r2:.3f}, MAE: {train_mae:.2f} K")
print(f"Test RMSE: {test_rmse:.2f} K, R²: {test_r2:.3f}, MAE: {test_mae:.2f} K")

# Save metrics
metrics = {
    'train_rmse': float(train_rmse),
    'test_rmse': float(test_rmse),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'train_mae': float(train_mae),
    'test_mae': float(test_mae),
    'kernel': str(gp.kernel_),
    'log_marginal_likelihood': float(gp.log_marginal_likelihood_value_),
}
with open('outputs/gp_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save predictions
cal['gp_pred'] = np.nan
cal.loc[idx_train, 'gp_pred'] = y_train_pred
cal.loc[idx_test, 'gp_pred'] = y_test_pred
cal.loc[idx_train, 'gp_std'] = y_train_std
cal.loc[idx_test, 'gp_std'] = y_test_std
cal.to_csv('outputs/calibration_with_gp.csv', index=False)

# Save model
joblib.dump({'gp': gp, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, 'outputs/gp_model.pkl')

# Figure 2: GP Calibration Performance
def create_figures():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Parity plot
    ax = axes[0]
    ax.scatter(y_train, y_train_pred, alpha=0.6, c='steelblue', edgecolors='k', linewidths=0.3, label=f'Train (R²={train_r2:.3f})')
    ax.scatter(y_test, y_test_pred, alpha=0.6, c='coral', edgecolors='k', linewidths=0.3, label=f'Test (R²={test_r2:.3f})')
    ax.plot([150, 650], [150, 650], 'r--', lw=1.5)
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('GP Predicted Tg (K)')
    ax.set_title('GP Calibration: Parity Plot')
    ax.legend()
    
    # Residuals
    ax = axes[1]
    train_res = y_train_pred - y_train
    test_res = y_test_pred - y_test
    ax.hist(train_res, bins=20, alpha=0.6, color='steelblue', edgecolor='k', label='Train')
    ax.hist(test_res, bins=20, alpha=0.6, color='coral', edgecolor='k', label='Test')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Predicted - Experimental Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('GP Calibration Residuals')
    ax.legend()
    
    # Uncertainty
    ax = axes[2]
    all_y = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    all_std = np.concatenate([y_train_std, y_test_std])
    all_res = all_pred - all_y
    ax.errorbar(all_y, all_res, yerr=all_std, fmt='o', alpha=0.4, c='gray', markersize=3)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('Predicted - Experimental Tg (K)')
    ax.set_title('GP Calibration: Residuals with Uncertainty')
    
    plt.tight_layout()
    plt.savefig('report/images/fig02_gp_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")

create_figures()
print("GP calibration complete.")
