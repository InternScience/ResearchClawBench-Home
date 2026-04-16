import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_gp_calibration():
    # Load calibration data
    df_calib = pd.read_csv('data/tg_calibration.csv')
    print("Calibration data shape:", df_calib.shape)
    
    # Features: MD simulated Tg
    # Target: Experimental Tg
    X = df_calib['tg_md'].values.reshape(-1, 1)
    y = df_calib['tg_exp'].values
    std = df_calib['std'].values if 'std' in df_calib.columns else np.ones_like(y)
    
    # Define GP kernel
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=10.0, noise_level_bounds=(1e-5, 1e3))
    
    # Initialize and fit GP
    # We can use alpha to incorporate MD simulation uncertainty if desired, but WhiteKernel also handles noise
    alpha = (std / y.mean())**2 # simple heuristic if we want to use alpha
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1.0)
    
    gp.fit(X, y)
    
    print("Optimized Kernel:", gp.kernel_)
    
    # Predict on training data
    y_pred, y_std = gp.predict(X, return_std=True)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Calibration MSE: {mse:.2f}")
    print(f"Calibration R2: {r2:.4f}")
    
    # Plot parity plot
    plt.figure(figsize=(6, 6))
    plt.errorbar(y, y_pred, yerr=y_std, fmt='o', ecolor='lightgray', elinewidth=3, capsize=0, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Experimental $T_g$ (K)')
    plt.ylabel('Calibrated $T_g$ (K)')
    plt.title(f'GP Calibration (R$^2$ = {r2:.3f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('report/images/gp_calibration_parity.png')
    plt.close()
    
    return gp

if __name__ == '__main__':
    gp_model = train_gp_calibration()
    
    # Load vitrimer MD data
    df_vitrimer = pd.read_csv('data/tg_vitrimer_MD.csv')
    print("Vitrimer MD data shape:", df_vitrimer.shape)
    
    X_vitrimer = df_vitrimer['tg'].values.reshape(-1, 1)
    
    # Predict calibrated Tg
    y_calib, y_calib_std = gp_model.predict(X_vitrimer, return_std=True)
    
    df_vitrimer['tg_calibrated'] = y_calib
    df_vitrimer['tg_calib_std'] = y_calib_std
    
    df_vitrimer.to_csv('outputs/tg_vitrimer_calibrated.csv', index=False)
    print("Saved calibrated vitrimer data to outputs/tg_vitrimer_calibrated.csv")
    
    # Plot distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_vitrimer['tg'], color='blue', alpha=0.5, label='MD Simulated $T_g$', kde=True)
    sns.histplot(df_vitrimer['tg_calibrated'], color='red', alpha=0.5, label='GP Calibrated $T_g$', kde=True)
    plt.xlabel('$T_g$ (K)')
    plt.ylabel('Count')
    plt.title('Distribution of $T_g$ before and after GP Calibration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/tg_distribution.png')
    plt.close()
