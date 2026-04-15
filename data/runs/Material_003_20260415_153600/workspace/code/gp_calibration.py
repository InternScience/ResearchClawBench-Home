"""
Gaussian Process Calibration Model for MD Tg Corrections.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

def create_features_for_calibration(cal_df):
    """Create features for GP calibration model."""
    # Features: MD prediction and its uncertainty
    X = np.column_stack([
        cal_df['tg_md'].values,
        cal_df['std'].values
    ])
    # Target: experimental Tg
    y = cal_df['tg_exp'].values
    return X, y

def train_gp_calibration(cal_df):
    """Train Gaussian Process calibration model."""
    X, y = create_features_for_calibration(cal_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define kernel: RBF for smooth interpolation + noise
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=[50.0, 10.0], 
        length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=20.0, noise_level_bounds=(1e-5, 1e2))
    
    # Train GP
    print("Training GP calibration model...")
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42
    )
    gp.fit(X_train, y_train)
    
    # Predictions
    y_pred_train, sigma_train = gp.predict(X_train, return_std=True)
    y_pred_test, sigma_test = gp.predict(X_test, return_std=True)
    
    # Metrics
    print("\n=== GP Calibration Model Performance ===")
    print(f"Training MAE: {mean_absolute_error(y_train, y_pred_train):.2f} K")
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.2f} K")
    print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
    
    # Compare with raw MD
    md_mae = mean_absolute_error(y_test, X_test[:, 0])
    print(f"\nRaw MD MAE: {md_mae:.2f} K")
    print(f"Improvement: {(md_mae - mean_absolute_error(y_test, y_pred_test)):.2f} K")
    
    # Save model
    with open('outputs/gp_calibration_model.pkl', 'wb') as f:
        pickle.dump(gp, f)
    
    return gp, X_test, y_test, y_pred_test, sigma_test

def visualize_calibration(cal_df, gp, X_test, y_test, y_pred_test, sigma_test):
    """Visualize calibration results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Full data predictions
    X_full, y_full = create_features_for_calibration(cal_df)
    y_pred_full, sigma_full = gp.predict(X_full, return_std=True)
    
    # 1. Calibration curve: Raw MD vs Calibrated
    ax = axes[0, 0]
    ax.scatter(cal_df['tg_md'], y_pred_full, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
    ax.plot([150, 550], [150, 550], 'r--', label='1:1 Line')
    ax.set_xlabel('Raw MD Tg (K)')
    ax.set_ylabel('GP Calibrated Tg (K)')
    ax.set_title('Calibration Effect on Tg Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Calibrated vs Experimental
    ax = axes[0, 1]
    ax.scatter(cal_df['tg_exp'], y_pred_full, alpha=0.6, c='green', edgecolors='black', linewidth=0.5)
    ax.plot([150, 550], [150, 550], 'r--', label='Perfect Agreement')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('GP Calibrated Tg (K)')
    ax.set_title('Calibrated Predictions vs Experimental')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Residuals before and after
    ax = axes[1, 0]
    raw_error = cal_df['tg_md'] - cal_df['tg_exp']
    cal_error = y_pred_full - cal_df['tg_exp']
    ax.hist(raw_error, bins=25, alpha=0.5, label=f'Raw MD (MAE: {np.abs(raw_error).mean():.1f} K)', color='red')
    ax.hist(cal_error, bins=25, alpha=0.5, label=f'Calibrated (MAE: {np.abs(cal_error).mean():.1f} K)', color='green')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel('Prediction Error (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution: Before vs After Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Uncertainty analysis
    ax = axes[1, 1]
    ax.errorbar(y_test, y_pred_test, yerr=2*sigma_test, fmt='o', alpha=0.5, capsize=3)
    ax.plot([150, 550], [150, 550], 'r--', label='Perfect Agreement')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('GP Calibrated Tg (K)')
    ax.set_title('Test Set Predictions with 2σ Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/gp_calibration.png', dpi=150, bbox_inches='tight')
    print("\nSaved: report/images/gp_calibration.png")
    plt.close()

def apply_calibration_to_vitrimer(vit_df, gp):
    """Apply GP calibration to vitrimer MD predictions."""
    X_vit = np.column_stack([
        vit_df['tg'].values,
        vit_df['std'].values
    ])
    
    y_cal, sigma_cal = gp.predict(X_vit, return_std=True)
    
    vit_df['tg_calibrated'] = y_cal
    vit_df['tg_cal_uncertainty'] = sigma_cal
    
    print(f"\n=== Calibrated Vitrimer Tg Statistics ===")
    print(f"Raw MD Tg: mean={vit_df['tg'].mean():.2f}, std={vit_df['tg'].std():.2f}")
    print(f"Calibrated Tg: mean={vit_df['tg_calibrated'].mean():.2f}, std={vit_df['tg_calibrated'].std():.2f}")
    
    return vit_df

def visualize_vitrimer_calibration(vit_df):
    """Visualize calibrated vitrimer Tg distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Raw vs Calibrated
    ax = axes[0]
    ax.scatter(vit_df['tg'], vit_df['tg_calibrated'], alpha=0.3, s=5)
    ax.plot([300, 570], [300, 570], 'r--', label='1:1 Line')
    ax.set_xlabel('Raw MD Tg (K)')
    ax.set_ylabel('Calibrated Tg (K)')
    ax.set_title('Vitrimer: Raw vs Calibrated Tg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution comparison
    ax = axes[1]
    ax.hist(vit_df['tg'], bins=50, alpha=0.5, label='Raw MD', edgecolor='black')
    ax.hist(vit_df['tg_calibrated'], bins=50, alpha=0.5, label='Calibrated', edgecolor='black')
    ax.set_xlabel('Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Vitrimer Tg Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Calibration shift vs raw Tg
    ax = axes[2]
    shift = vit_df['tg_calibrated'] - vit_df['tg']
    ax.scatter(vit_df['tg'], shift, alpha=0.3, s=5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Raw MD Tg (K)')
    ax.set_ylabel('Calibration Shift (K)')
    ax.set_title('Calibration Adjustment vs Raw Tg')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/vitrimer_calibration.png', dpi=150, bbox_inches='tight')
    print("Saved: report/images/vitrimer_calibration.png")
    plt.close()

def main():
    # Load processed calibration data
    cal_df = pd.read_csv('outputs/calibration_processed.csv')
    
    # Train GP calibration
    gp, X_test, y_test, y_pred_test, sigma_test = train_gp_calibration(cal_df)
    
    # Visualize
    visualize_calibration(cal_df, gp, X_test, y_test, y_pred_test, sigma_test)
    
    # Load and calibrate vitrimer data
    vit_df = pd.read_csv('outputs/vitrimer_processed.csv')
    vit_df = apply_calibration_to_vitrimer(vit_df, gp)
    
    # Visualize vitrimer calibration
    visualize_vitrimer_calibration(vit_df)
    
    # Save calibrated data
    vit_df.to_csv('outputs/vitrimer_calibrated.csv', index=False)
    print("\nSaved: outputs/vitrimer_calibrated.csv")
    
    return gp, vit_df

if __name__ == "__main__":
    main()
