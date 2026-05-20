#!/usr/bin/env python3
"""
rfr_gp.py - Random Forest Regressor (RFR) as EI maximizer + Gaussian Process (GP) as value provider
for de novo hydrogel adhesive strength optimization (>1 MPa = 1000 kPa target).

Implements SMBO/EI-based optimization on 184 verified hydrogel formulations.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Constants
MONOMER_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA',
                'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET_COL = 'Glass (kPa)_10s'
N_NEW_CANDIDATES = 20
EI_THRESHOLD = 0.01  # minimum EI improvement

def load_and_clean_data(path='data/184_verified_Original Data_ML_20230926.xlsx'):
    """Load verified 184 hydrogel dataset and clean."""
    df = pd.read_excel(path)
    # Drop rows with missing target or monomer data
    df = df.dropna(subset=[TARGET_COL] + MONOMER_COLS)
    # Ensure monomer fractions sum to ~1 (tolerance 0.05)
    df['sum_frac'] = df[MONOMER_COLS].sum(axis=1)
    df = df[(df['sum_frac'] > 0.95) & (df['sum_frac'] < 1.05)]
    df = df.drop(columns=['sum_frac'])
    # Clip extreme outliers in target (keep 99th percentile)
    q99 = df[TARGET_COL].quantile(0.99)
    df[TARGET_COL] = df[TARGET_COL].clip(upper=q99)
    print(f"Cleaned dataset: {len(df)} samples")
    return df

def train_models(X, y):
    """Train RFR (EI maximizer) and GP (value provider)."""
    # RFR
    rfr = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rfr.fit(X, y)
    rfr_cv = cross_val_score(rfr, X, y, cv=KFold(5, shuffle=True, random_state=42),
                             scoring='neg_mean_absolute_error')
    print(f"RFR 5-fold MAE: {-np.mean(rfr_cv):.2f} ± {np.std(rfr_cv):.2f} kPa")

    # GP with RBF + WhiteKernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
    gp.fit(X, y)
    gp_cv = cross_val_score(gp, X, y, cv=KFold(5, shuffle=True, random_state=42),
                            scoring='neg_mean_absolute_error')
    print(f"GP  5-fold MAE: {-np.mean(gp_cv):.2f} ± {np.std(gp_cv):.2f} kPa")

    return rfr, gp

def expected_improvement(X_new, rfr, gp, y_max, xi=0.01):
    """Compute Expected Improvement using GP posterior."""
    mu, sigma = gp.predict(X_new, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    Z = (mu - y_max - xi) / sigma
    from scipy.stats import norm
    ei = (mu - y_max - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei

def propose_new_formulations(rfr, gp, X_train, y_train, n_candidates=N_NEW_CANDIDATES):
    """Propose new monomer compositions via EI maximization (RFR-guided sampling)."""
    # Use RFR to guide candidate sampling around high-performing regions
    y_max = y_train.max()
    # Generate random compositions (Dirichlet-like) and refine with RFR top predictions
    rng = np.random.default_rng(42)
    candidates = []
    for _ in range(5000):
        frac = rng.dirichlet(np.ones(len(MONOMER_COLS)))
        candidates.append(frac)
    candidates = np.array(candidates)

    # Predict with RFR and keep top 200 for EI evaluation
    rfr_pred = rfr.predict(candidates)
    top_idx = np.argsort(rfr_pred)[-200:]
    candidates = candidates[top_idx]

    # Compute EI with GP
    ei = expected_improvement(candidates, rfr, gp, y_max)
    best_idx = np.argsort(ei)[-n_candidates:]
    new_formulations = candidates[best_idx]
    ei_values = ei[best_idx]

    # Convert to DataFrame
    df_new = pd.DataFrame(new_formulations, columns=MONOMER_COLS)
    df_new['EI'] = ei_values
    df_new['RFR_pred_kPa'] = rfr.predict(new_formulations)
    mu_gp, sigma_gp = gp.predict(new_formulations, return_std=True)
    df_new['GP_mean_kPa'] = mu_gp
    df_new['GP_std_kPa'] = sigma_gp
    df_new['Formulation_ID'] = [f'New-{i+1:03d}' for i in range(len(df_new))]
    return df_new.sort_values('EI', ascending=False)

def generate_figures(df, rfr, gp, X, y, new_df):
    """Generate publication-quality figures."""
    sns.set(style='whitegrid', font_scale=1.1)
    plt.rcParams['figure.dpi'] = 150

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[TARGET_COL], bins=30, kde=True, ax=ax, color='#2E86AB')
    ax.axvline(1000, color='red', linestyle='--', label='1 MPa target')
    ax.set_xlabel('Adhesive Strength (kPa)')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig('report/images/figure1_target_distribution.png', bbox_inches='tight')
    plt.close()

    # 2. Correlation heatmap of monomers
    fig, ax = plt.subplots(figsize=(7,6))
    corr = df[MONOMER_COLS].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Monomer Fraction Correlations')
    plt.tight_layout()
    plt.savefig('report/images/figure2_monomer_correlation.png', bbox_inches='tight')
    plt.close()

    # 3. RFR vs GP parity plot
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    for ax, model, name in zip(axes, [rfr, gp], ['RFR', 'GP']):
        pred = model.predict(X)
        r2 = r2_score(y, pred)
        mae = mean_absolute_error(y, pred)
        ax.scatter(y, pred, alpha=0.6, s=30, edgecolor='k')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel('Measured (kPa)')
        ax.set_ylabel(f'{name} Predicted (kPa)')
        ax.set_title(f'{name} (R²={r2:.3f}, MAE={mae:.1f})')
    plt.tight_layout()
    plt.savefig('report/images/figure3_parity.png', bbox_inches='tight')
    plt.close()

    # 4. EI optimization trajectory (simulated)
    fig, ax = plt.subplots(figsize=(7,4))
    sorted_ei = new_df['EI'].values
    ax.plot(range(1, len(sorted_ei)+1), sorted_ei, marker='o', color='#E94F37')
    ax.set_xlabel('Ranked New Formulation')
    ax.set_ylabel('Expected Improvement')
    ax.set_title('EI-Guided Candidate Ranking')
    plt.tight_layout()
    plt.savefig('report/images/figure4_ei_ranking.png', bbox_inches='tight')
    plt.close()

    # 5. New formulation monomer composition heatmap
    fig, ax = plt.subplots(figsize=(8,5))
    comp = new_df[MONOMER_COLS].values
    sns.heatmap(comp, cmap='viridis', yticklabels=new_df['Formulation_ID'], ax=ax)
    ax.set_xlabel('Monomer')
    ax.set_ylabel('New Formulation')
    ax.set_xticklabels(MONOMER_COLS, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('report/images/figure5_new_formulations.png', bbox_inches='tight')
    plt.close()

    print("Figures saved to report/images/")

def main():
    print("=== Hydrogel Adhesive Strength Optimization (RFR+GP EI) ===")
    df = load_and_clean_data()
    X = df[MONOMER_COLS].values
    y = df[TARGET_COL].values

    # Scale features (important for GP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'outputs/scaler.pkl')

    rfr, gp = train_models(X_scaled, y)

    # Save models
    joblib.dump(rfr, 'outputs/rfr_model.pkl')
    joblib.dump(gp, 'outputs/gp_model.pkl')

    # Propose new formulations
    new_df = propose_new_formulations(rfr, gp, X_scaled, y)
    new_df.to_csv('outputs/new_formulations_ei.csv', index=False)
    print(f"Proposed {len(new_df)} new formulations saved.")

    # Generate figures
    generate_figures(df, rfr, gp, X_scaled, y, new_df)

    # Summary
    print("\n=== Optimization Summary ===")
    print(f"Best training adhesive strength: {y.max():.1f} kPa")
    print(f"Top new formulation predicted (RFR): {new_df.iloc[0]['RFR_pred_kPa']:.1f} kPa")
    print(f"Top new formulation EI: {new_df.iloc[0]['EI']:.4f}")
    print("Task complete. Ready for experimental validation.")

if __name__ == '__main__':
    main()