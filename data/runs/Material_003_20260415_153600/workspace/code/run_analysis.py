"""
Main analysis script for vitrimer inverse design framework.
Runs all components in sequence.
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
plt.rcParams['figure.dpi'] = 150

def load_and_explore():
    """Load and explore data."""
    print("=" * 60)
    print("STEP 1: Data Loading and Exploration")
    print("=" * 60)
    
    cal_df = pd.read_csv('data/tg_calibration.csv')
    vit_df = pd.read_csv('data/tg_vitrimer_MD.csv')
    
    print(f"\nCalibration dataset: {len(cal_df)} polymers")
    print(f"Vitrimer dataset: {len(vit_df)} systems")
    
    # Calculate errors
    cal_df['error'] = cal_df['tg_md'] - cal_df['tg_exp']
    cal_df['abs_error'] = np.abs(cal_df['error'])
    
    print(f"\nMD Simulation Performance:")
    print(f"  MAE: {cal_df['abs_error'].mean():.2f} K")
    print(f"  RMSE: {np.sqrt((cal_df['error']**2).mean()):.2f} K")
    
    # Basic visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.scatter(cal_df['tg_exp'], cal_df['tg_md'], alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.plot([150, 600], [150, 600], 'r--', label='Perfect Agreement')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('MD Simulated Tg (K)')
    ax.set_title('Calibration: MD vs Experimental Tg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(cal_df['error'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Prediction Error (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('MD Error Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.hist(vit_df['tg'], bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
    ax.axvline(vit_df['tg'].mean(), color='r', linestyle='--', label=f'Mean: {vit_df["tg"].mean():.1f} K')
    ax.set_xlabel('MD Simulated Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Vitrimer Tg Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.scatter(vit_df['tg'], vit_df['std'], alpha=0.3, s=5)
    ax.set_xlabel('Tg (K)')
    ax.set_ylabel('Standard Deviation (K)')
    ax.set_title('Vitrimer Tg vs Uncertainty')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
    print("\nSaved: report/images/data_overview.png")
    plt.close()
    
    cal_df.to_csv('outputs/calibration_processed.csv', index=False)
    vit_df.to_csv('outputs/vitrimer_processed.csv', index=False)
    
    return cal_df, vit_df

def train_gp_calibration(cal_df):
    """Train GP calibration model."""
    print("\n" + "=" * 60)
    print("STEP 2: Gaussian Process Calibration")
    print("=" * 60)
    
    X = np.column_stack([cal_df['tg_md'].values, cal_df['std'].values])
    y = cal_df['tg_exp'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=[50.0, 10.0], 
        length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=20.0, noise_level_bounds=(1e-5, 1e2))
    
    print("\nTraining GP model...")
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)
    gp.fit(X_train, y_train)
    
    y_pred_train, _ = gp.predict(X_train, return_std=True)
    y_pred_test, sigma_test = gp.predict(X_test, return_std=True)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nGP Model Performance:")
    print(f"  Training MAE: {train_mae:.2f} K")
    print(f"  Test MAE: {test_mae:.2f} K")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Raw MD MAE: {mean_absolute_error(y_test, X_test[:, 0]):.2f} K")
    print(f"  Improvement: {mean_absolute_error(y_test, X_test[:, 0]) - test_mae:.2f} K")
    
    # Visualizations
    y_pred_full, sigma_full = gp.predict(X, return_std=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.scatter(cal_df['tg_md'], y_pred_full, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
    ax.plot([150, 600], [150, 600], 'r--')
    ax.set_xlabel('Raw MD Tg (K)')
    ax.set_ylabel('Calibrated Tg (K)')
    ax.set_title('Calibration Effect')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.scatter(cal_df['tg_exp'], y_pred_full, alpha=0.6, c='green', edgecolors='black', linewidth=0.5)
    ax.plot([150, 600], [150, 600], 'r--')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('Calibrated Tg (K)')
    ax.set_title('Calibrated vs Experimental')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    raw_err = cal_df['tg_md'] - cal_df['tg_exp']
    cal_err = y_pred_full - cal_df['tg_exp']
    ax.hist(raw_err, bins=25, alpha=0.5, label=f'Raw MAE: {np.abs(raw_err).mean():.1f} K', color='red')
    ax.hist(cal_err, bins=25, alpha=0.5, label=f'Cal MAE: {np.abs(cal_err).mean():.1f} K', color='green')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel('Error (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.errorbar(y_test, y_pred_test, yerr=2*sigma_test, fmt='o', alpha=0.5, capsize=3)
    ax.plot([150, 600], [150, 600], 'r--')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('Calibrated Tg (K)')
    ax.set_title('Test Set with 2σ Uncertainty')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/gp_calibration.png', dpi=150, bbox_inches='tight')
    print("\nSaved: report/images/gp_calibration.png")
    plt.close()
    
    with open('outputs/gp_model.pkl', 'wb') as f:
        pickle.dump(gp, f)
    
    return gp

def apply_calibration(vit_df, gp):
    """Apply calibration to vitrimer data."""
    print("\n" + "=" * 60)
    print("STEP 3: Apply Calibration to Vitrimers")
    print("=" * 60)
    
    X_vit = np.column_stack([vit_df['tg'].values, vit_df['std'].values])
    y_cal, sigma_cal = gp.predict(X_vit, return_std=True)
    
    vit_df['tg_calibrated'] = y_cal
    vit_df['tg_cal_uncertainty'] = sigma_cal
    
    print(f"\nTg Statistics:")
    print(f"  Raw MD: {vit_df['tg'].mean():.2f} ± {vit_df['tg'].std():.2f} K")
    print(f"  Calibrated: {vit_df['tg_calibrated'].mean():.2f} ± {vit_df['tg_calibrated'].std():.2f} K")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.scatter(vit_df['tg'], vit_df['tg_calibrated'], alpha=0.3, s=5)
    ax.plot([300, 570], [300, 570], 'r--')
    ax.set_xlabel('Raw MD Tg (K)')
    ax.set_ylabel('Calibrated Tg (K)')
    ax.set_title('Raw vs Calibrated')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.hist(vit_df['tg'], bins=50, alpha=0.5, label='Raw MD')
    ax.hist(vit_df['tg_calibrated'], bins=50, alpha=0.5, label='Calibrated')
    ax.set_xlabel('Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    shift = vit_df['tg_calibrated'] - vit_df['tg']
    ax.scatter(vit_df['tg'], shift, alpha=0.3, s=5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Raw MD Tg (K)')
    ax.set_ylabel('Shift (K)')
    ax.set_title('Calibration Shift')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/vitrimer_calibration.png', dpi=150, bbox_inches='tight')
    print("\nSaved: report/images/vitrimer_calibration.png")
    plt.close()
    
    vit_df.to_csv('outputs/vitrimer_calibrated.csv', index=False)
    return vit_df

def simple_gvae_model(vit_df):
    """Simplified GVAE-inspired property prediction."""
    print("\n" + "=" * 60)
    print("STEP 4: Molecular Representation Learning")
    print("=" * 60)
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Create simple molecular descriptors from SMILES
    print("\nGenerating molecular descriptors...")
    
    def smiles_to_simple_features(smiles):
        """Extract simple features from SMILES string."""
        features = {}
        features['length'] = len(smiles)
        features['num_c'] = smiles.count('C')
        features['num_o'] = smiles.count('O')
        features['num_n'] = smiles.count('N')
        features['num_ring'] = smiles.count('1') + smiles.count('2')
        features['num_aromatic'] = smiles.count('c')
        features['num_double'] = smiles.count('=')
        features['num_triple'] = smiles.count('#')
        features['num_paren'] = smiles.count('(')
        features['has_epoxide'] = 1 if 'CO1' in smiles or 'CO2' in smiles or 'CO3' in smiles else 0
        features['num_co'] = smiles.count('CO')
        features['num_cc'] = smiles.count('CC')
        return features
    
    # Extract features for acids and epoxides
    acid_features = pd.DataFrame([smiles_to_simple_features(s) for s in vit_df['acid']])
    epoxide_features = pd.DataFrame([smiles_to_simple_features(s) for s in vit_df['epoxide']])
    
    # Combine features
    acid_features.columns = ['acid_' + c for c in acid_features.columns]
    epoxide_features.columns = ['epoxide_' + c for c in epoxide_features.columns]
    
    combined = pd.concat([acid_features.reset_index(drop=True), 
                          epoxide_features.reset_index(drop=True)], axis=1)
    
    print(f"Generated {combined.shape[1]} molecular features")
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined)
    
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Store latent representations
    for i in range(10):
        vit_df[f'latent_{i}'] = X_pca[:, i]
    
    # Visualize latent space
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=vit_df['tg_calibrated'], cmap='viridis', alpha=0.5, s=5)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Latent Space (colored by Tg)')
    plt.colorbar(scatter, ax=ax)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.bar(range(10), pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.scatter(vit_df['tg_calibrated'], X_pca[:, 0], alpha=0.3, s=5)
    ax.set_xlabel('Calibrated Tg (K)')
    ax.set_ylabel('Latent Dimension 1')
    ax.set_title('Tg vs Latent Dimension 1')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    feature_importance = np.abs(pca.components_[0])
    top_indices = np.argsort(feature_importance)[-10:]
    ax.barh(range(10), feature_importance[top_indices])
    ax.set_yticks(range(10))
    ax.set_yticklabels([combined.columns[i] for i in top_indices], fontsize=8)
    ax.set_xlabel('Absolute Component Value')
    ax.set_title('Top Features in PC1')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/latent_space.png', dpi=150, bbox_inches='tight')
    print("\nSaved: report/images/latent_space.png")
    plt.close()
    
    return vit_df, X_pca

def generate_candidates(vit_df, X_latent):
    """Generate candidate molecules with target Tg."""
    print("\n" + "=" * 60)
    print("STEP 5: Inverse Design - Candidate Generation")
    print("=" * 60)
    
    from sklearn.neighbors import NearestNeighbors
    
    # Target Tg values for different applications
    targets = {
        'Low Tg (flexible)': 350,
        'Medium Tg (general)': 400,
        'High Tg (rigid)': 450
    }
    
    candidates = []
    
    for target_name, target_tg in targets.items():
        # Find molecules closest to target Tg
        vit_df['tg_distance'] = np.abs(vit_df['tg_calibrated'] - target_tg)
        
        # Get candidates with good confidence
        good_candidates = vit_df[vit_df['tg_cal_uncertainty'] < 30].copy()
        top_candidates = good_candidates.nsmallest(5, 'tg_distance')
        
        print(f"\n{target_name} (target: {target_tg} K):")
        for _, row in top_candidates.iterrows():
            print(f"  Tg: {row['tg_calibrated']:.1f} ± {row['tg_cal_uncertainty']:.1f} K")
            candidates.append({
                'target_category': target_name,
                'target_tg': target_tg,
                'acid': row['acid'],
                'epoxide': row['epoxide'],
                'tg_raw': row['tg'],
                'tg_calibrated': row['tg_calibrated'],
                'uncertainty': row['tg_cal_uncertainty']
            })
    
    # Visualize candidates
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    scatter = ax.scatter(vit_df['tg_calibrated'], vit_df['tg_cal_uncertainty'], 
                         c='lightgray', alpha=0.3, s=5, label='All systems')
    
    cand_df = pd.DataFrame(candidates)
    for target_name in targets.keys():
        subset = cand_df[cand_df['target_category'] == target_name]
        ax.scatter(subset['tg_calibrated'], subset['uncertainty'], 
                  s=100, label=target_name, edgecolors='black', linewidth=1.5)
    
    for target_name, target_tg in targets.items():
        ax.axvline(target_tg, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Calibrated Tg (K)')
    ax.set_ylabel('Uncertainty (K)')
    ax.set_title('Candidate Selection in Property Space')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    colors = {'Low Tg (flexible)': 'blue', 'Medium Tg (general)': 'green', 'High Tg (rigid)': 'red'}
    for target_name in targets.keys():
        subset = cand_df[cand_df['target_category'] == target_name]
        ax.hist(subset['tg_calibrated'], bins=5, alpha=0.6, label=target_name, color=colors[target_name])
    
    ax.set_xlabel('Calibrated Tg (K)')
    ax.set_ylabel('Count')
    ax.set_title('Selected Candidates by Category')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/candidate_generation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: report/images/candidate_generation.png")
    plt.close()
    
    cand_df.to_csv('outputs/candidates.csv', index=False)
    return cand_df

def main():
    import os
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    
    cal_df, vit_df = load_and_explore()
    gp = train_gp_calibration(cal_df)
    vit_df = apply_calibration(vit_df, gp)
    vit_df, X_latent = simple_gvae_model(vit_df)
    candidates = generate_candidates(vit_df, X_latent)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nGenerated outputs:")
    print("  - outputs/calibration_processed.csv")
    print("  - outputs/vitrimer_calibrated.csv")
    print("  - outputs/gp_model.pkl")
    print("  - outputs/candidates.csv")
    print("\nGenerated figures:")
    print("  - report/images/data_overview.png")
    print("  - report/images/gp_calibration.png")
    print("  - report/images/vitrimer_calibration.png")
    print("  - report/images/latent_space.png")
    print("  - report/images/candidate_generation.png")

if __name__ == "__main__":
    main()
