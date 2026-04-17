#!/usr/bin/env python3
"""
Phase 4: Validation, Protein-Inspired Design Analysis, and Pathway to >1 MPa
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import norm
import json
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

BASE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_000_20260416_195517"

# Load data
df_train = pd.read_excel(f"{BASE}/data/184_verified_Original Data_ML_20230926.xlsx", sheet_name='Data_to_HU')
features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
target = 'Glass (kPa)_10s'

X = df_train[features].values
y = df_train[target].values

# Load optimization data
df_opt_ei = pd.read_excel(f"{BASE}/data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='EI')
df_opt_pred = pd.read_excel(f"{BASE}/data/ML_ei&pred (1&2&3rounds)_20240408.xlsx", sheet_name='PRED')
df_opt_ei['ML'] = df_opt_ei['ML'].ffill()
df_opt_pred['ML'] = df_opt_pred['ML'].ffill()
df_opt_ei['Glass (kPa)_max'] = pd.to_numeric(df_opt_ei['Glass (kPa)_max'], errors='coerce')
df_opt_pred['Glass (kPa)_max'] = pd.to_numeric(df_opt_pred['Glass (kPa)_max'], errors='coerce')
for f in features:
    df_opt_ei[f] = pd.to_numeric(df_opt_ei[f], errors='coerce')
    df_opt_pred[f] = pd.to_numeric(df_opt_pred[f], errors='coerce')

# ========== Train final models ==========
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, normalize_y=True)
gp.fit(X_scaled, y)

# ========== Figure 14: Learning Curve ==========
print("Computing learning curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, model, name, color in [(axes[0], RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), 
                                  'Random Forest', '#2196F3'),
                                 (axes[1], GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, 
                                  random_state=42, normalize_y=True), 'Gaussian Process', '#FF9800')]:
    if name == 'Gaussian Process':
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_scaled, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 8),
            scoring='neg_root_mean_squared_error'
        )
    else:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 8),
            scoring='neg_root_mean_squared_error'
        )
    
    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=color)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    ax.plot(train_sizes, train_mean, 'o-', color=color, label='Training RMSE')
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation RMSE')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('RMSE (kPa)')
    ax.set_title(f'{name} Learning Curve')
    ax.legend()

plt.suptitle('Learning Curves: Model Performance vs Training Data Size', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig14_learning_curves.png")
plt.close()
print("Figure 14 saved: learning curves")

# ========== Figure 15: Residual Analysis ==========
cv = KFold(n_splits=10, shuffle=True, random_state=42)
y_pred_rf = cross_val_predict(rf, X, y, cv=cv)
y_pred_gp = cross_val_predict(gp, X_scaled, y, cv=cv)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# RF residuals
residuals_rf = y - y_pred_rf
ax = axes[0, 0]
ax.scatter(y_pred_rf, residuals_rf, alpha=0.5, color='#2196F3', edgecolors='black', linewidth=0.3, s=30)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Predicted (kPa)')
ax.set_ylabel('Residual (kPa)')
ax.set_title('Random Forest: Residual Plot')

ax = axes[0, 1]
ax.hist(residuals_rf, bins=25, color='#2196F3', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Residual (kPa)')
ax.set_ylabel('Count')
ax.set_title(f'RF Residual Distribution (μ={residuals_rf.mean():.1f}, σ={residuals_rf.std():.1f})')

# GP residuals
residuals_gp = y - y_pred_gp
ax = axes[1, 0]
ax.scatter(y_pred_gp, residuals_gp, alpha=0.5, color='#FF9800', edgecolors='black', linewidth=0.3, s=30)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Predicted (kPa)')
ax.set_ylabel('Residual (kPa)')
ax.set_title('Gaussian Process: Residual Plot')

ax = axes[1, 1]
ax.hist(residuals_gp, bins=25, color='#FF9800', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel('Residual (kPa)')
ax.set_ylabel('Count')
ax.set_title(f'GP Residual Distribution (μ={residuals_gp.mean():.1f}, σ={residuals_gp.std():.1f})')

plt.suptitle('Residual Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig15_residual_analysis.png")
plt.close()
print("Figure 15 saved: residual analysis")

# ========== Figure 16: Protein-Inspired Design Space Analysis ==========
# Analyze the relationship between monomer types and protein amino acid categories
# Monomers map to amino acid properties:
# HEA → Nucleophilic (Ser, Thr, Cys)
# BA → Hydrophobic (Ala, Val, Leu, Ile, Pro, Met)
# CBEA → Acidic (Asp, Glu)
# ATAC → Cationic (Lys, Arg, His)
# PEA → Aromatic (Phe, Tyr, Trp)
# AAm → Amide (Asn, Gln)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# For each monomer, show the relationship with adhesion strength using 2D density
for i, (feat, name) in enumerate(zip(features, ['Nucleophilic\n(HEA)', 'Hydrophobic\n(BA)', 
                                                  'Acidic\n(CBEA)', 'Cationic\n(ATAC)',
                                                  'Aromatic\n(PEA)', 'Amide\n(AAm)'])):
    ax = axes[i//3, i%3]
    
    # Training data
    ax.scatter(df_train[feat], df_train[target], alpha=0.4, s=20, color='gray', label='Training')
    
    # Optimization data (EI)
    opt_data = df_opt_ei.dropna(subset=[feat, 'Glass (kPa)_max'])
    if len(opt_data) > 0:
        ax.scatter(opt_data[feat], opt_data['Glass (kPa)_max'], alpha=0.6, s=40, 
                  color='#E91E63', marker='^', label='Optimized (EI)')
    
    # Optimization data (PRED)
    opt_pred = df_opt_pred.dropna(subset=[feat, 'Glass (kPa)_max'])
    if len(opt_pred) > 0:
        ax.scatter(opt_pred[feat], opt_pred['Glass (kPa)_max'], alpha=0.6, s=40,
                  color='#4CAF50', marker='s', label='Optimized (PRED)')
    
    ax.set_xlabel(f'{name} Fraction')
    ax.set_ylabel('Adhesive Strength (kPa)')
    ax.set_title(f'{name}')
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Training vs Optimized Formulations: Monomer-Adhesion Relationships', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig16_design_space.png")
plt.close()
print("Figure 16 saved: design space analysis")

# ========== Figure 17: Pathway to >1 MPa Analysis ==========
# Extrapolation analysis: what compositions might achieve >1 MPa?
print("\n=== Pathway to >1 MPa (1000 kPa) Analysis ===")

# Generate focused candidates
np.random.seed(42)
n_focused = 100000

# Strategy 1: Dirichlet with concentration on high-performing monomers
# Based on top performers: high BA, moderate ATAC, some PEA
alpha_focused = np.array([0.5, 3.0, 0.3, 1.0, 2.0, 0.5])  # Focus on BA and PEA
candidates_focused = np.random.dirichlet(alpha_focused, size=n_focused)

# Predict with RF
rf_preds = rf.predict(candidates_focused)

# Predict with GP (with uncertainty)
candidates_scaled = scaler.transform(candidates_focused)
gp_mu, gp_sigma = gp.predict(candidates_scaled, return_std=True)

# Upper confidence bound
ucb = gp_mu + 2 * gp_sigma

# Find candidates with highest predicted adhesion
top_rf_idx = np.argsort(rf_preds)[::-1][:100]
top_gp_idx = np.argsort(gp_mu)[::-1][:100]
top_ucb_idx = np.argsort(ucb)[::-1][:100]

print(f"Max RF prediction: {rf_preds.max():.1f} kPa")
print(f"Max GP prediction: {gp_mu.max():.1f} kPa")
print(f"Max UCB (μ+2σ): {ucb.max():.1f} kPa")
print(f"Max GP σ: {gp_sigma.max():.1f} kPa")

# Analyze what compositions are needed for high adhesion
print("\nTop 5 by RF prediction:")
for i in range(5):
    idx = top_rf_idx[i]
    comp = candidates_focused[idx]
    print(f"  HEA={comp[0]:.3f} BA={comp[1]:.3f} CBEA={comp[2]:.3f} ATAC={comp[3]:.3f} PEA={comp[4]:.3f} AAm={comp[5]:.3f} → RF={rf_preds[idx]:.1f} kPa")

print("\nTop 5 by GP prediction:")
for i in range(5):
    idx = top_gp_idx[i]
    comp = candidates_focused[idx]
    print(f"  HEA={comp[0]:.3f} BA={comp[1]:.3f} CBEA={comp[2]:.3f} ATAC={comp[3]:.3f} PEA={comp[4]:.3f} AAm={comp[5]:.3f} → GP={gp_mu[idx]:.1f}±{gp_sigma[idx]:.1f} kPa")

# Figure 17: Prediction landscape
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# BA vs PEA colored by prediction
ax = axes[0, 0]
scatter = ax.scatter(candidates_focused[:5000, 1], candidates_focused[:5000, 4], 
                    c=rf_preds[:5000], cmap='viridis', alpha=0.3, s=5)
plt.colorbar(scatter, ax=ax, label='RF Predicted Adhesion (kPa)')
ax.set_xlabel('Hydrophobic-BA Fraction')
ax.set_ylabel('Aromatic-PEA Fraction')
ax.set_title('RF Prediction Landscape')

# BA vs ATAC colored by prediction
ax = axes[0, 1]
scatter = ax.scatter(candidates_focused[:5000, 1], candidates_focused[:5000, 3],
                    c=rf_preds[:5000], cmap='viridis', alpha=0.3, s=5)
plt.colorbar(scatter, ax=ax, label='RF Predicted Adhesion (kPa)')
ax.set_xlabel('Hydrophobic-BA Fraction')
ax.set_ylabel('Cationic-ATAC Fraction')
ax.set_title('RF Prediction Landscape')

# GP prediction vs uncertainty
ax = axes[1, 0]
scatter = ax.scatter(gp_mu[:5000], gp_sigma[:5000], c=rf_preds[:5000], cmap='viridis', alpha=0.3, s=5)
plt.colorbar(scatter, ax=ax, label='RF Predicted Adhesion (kPa)')
ax.set_xlabel('GP Mean Prediction (kPa)')
ax.set_ylabel('GP Uncertainty σ (kPa)')
ax.set_title('GP Prediction vs Uncertainty')

# Composition of top candidates
ax = axes[1, 1]
# Compare top RF, top GP, and top UCB compositions
top_rf_comp = candidates_focused[top_rf_idx[:50]].mean(axis=0)
top_gp_comp = candidates_focused[top_gp_idx[:50]].mean(axis=0)
top_ucb_comp = candidates_focused[top_ucb_idx[:50]].mean(axis=0)
train_comp = X.mean(axis=0)

x_pos = np.arange(len(features))
width = 0.2
ax.bar(x_pos - 1.5*width, train_comp, width, label='Training Mean', color='gray', alpha=0.8)
ax.bar(x_pos - 0.5*width, top_rf_comp, width, label='Top RF', color='#2196F3', alpha=0.8)
ax.bar(x_pos + 0.5*width, top_gp_comp, width, label='Top GP', color='#FF9800', alpha=0.8)
ax.bar(x_pos + 1.5*width, top_ucb_comp, width, label='Top UCB', color='#4CAF50', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm'])
ax.set_ylabel('Mean Fraction')
ax.set_title('Optimal Compositions by Strategy')
ax.legend(fontsize=8)

plt.suptitle('Pathway to High-Adhesion Hydrogels: Prediction Landscape', fontsize=14)
plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig17_pathway_analysis.png")
plt.close()
print("\nFigure 17 saved: pathway analysis")

# ========== Figure 18: Combined Optimization Summary ==========
fig, ax = plt.subplots(figsize=(12, 7))

# Combine all experimental data
all_data = []

# Training
for _, row in df_train.iterrows():
    all_data.append({
        'Round': 'Initial Training',
        'Adhesion': row[target],
        'Source': 'Experiment'
    })

# Optimization rounds
for _, row in df_opt_ei.dropna(subset=['Glass (kPa)_max']).iterrows():
    ml = row['ML']
    if 'rd' not in ml.lower():
        round_name = 'Round 1'
    elif '2rd' in ml.lower():
        round_name = 'Round 2'
    elif '3rd' in ml.lower():
        round_name = 'Round 3'
    else:
        round_name = 'Round 1'
    all_data.append({
        'Round': round_name,
        'Adhesion': row['Glass (kPa)_max'],
        'Source': 'ML-Optimized (EI)'
    })

for _, row in df_opt_pred.dropna(subset=['Glass (kPa)_max']).iterrows():
    ml = row['ML']
    if 'rd' not in ml.lower():
        round_name = 'Round 1'
    elif '2rd' in ml.lower():
        round_name = 'Round 2'
    elif '3rd' in ml.lower():
        round_name = 'Round 3'
    else:
        round_name = 'Round 1'
    all_data.append({
        'Round': round_name,
        'Adhesion': row['Glass (kPa)_max'],
        'Source': 'ML-Optimized (PRED)'
    })

all_df = pd.DataFrame(all_data)

# Violin plot
order = ['Initial Training', 'Round 1', 'Round 2', 'Round 3']
palette = {'Initial Training': 'gray', 'Round 1': '#2196F3', 'Round 2': '#FF9800', 'Round 3': '#4CAF50'}
sns.violinplot(data=all_df, x='Round', y='Adhesion', order=order, palette=palette, 
               inner='box', cut=0, ax=ax)
ax.axhline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa Target')
ax.set_ylabel('Adhesive Strength (kPa)')
ax.set_xlabel('Optimization Stage')
ax.set_title('Adhesive Strength Distribution Across Optimization Stages')
ax.legend(fontsize=12)

# Add annotations
for i, round_name in enumerate(order):
    subset = all_df[all_df['Round'] == round_name]['Adhesion']
    ax.text(i, subset.max() + 10, f'Max: {subset.max():.0f} kPa\nn={len(subset)}', 
            ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{BASE}/report/images/fig18_optimization_summary.png")
plt.close()
print("Figure 18 saved: optimization summary")

# ========== Save comprehensive results ==========
# Compute improvement metrics
initial_max = y.max()
initial_mean = y.mean()

opt_all = pd.concat([df_opt_ei[['Glass (kPa)_max']].dropna(), df_opt_pred[['Glass (kPa)_max']].dropna()])
opt_max = opt_all['Glass (kPa)_max'].max()
opt_mean = opt_all['Glass (kPa)_max'].mean()

improvement_metrics = {
    'initial_training': {
        'n_samples': int(len(y)),
        'mean_kPa': float(initial_mean),
        'max_kPa': float(initial_max),
        'std_kPa': float(y.std()),
        'above_100kPa': int((y > 100).sum()),
        'above_200kPa': int((y > 200).sum()),
        'above_300kPa': int((y > 300).sum()),
    },
    'optimized': {
        'n_samples': int(len(opt_all)),
        'mean_kPa': float(opt_mean),
        'max_kPa': float(opt_max),
        'above_100kPa': int((opt_all['Glass (kPa)_max'] > 100).sum()),
        'above_200kPa': int((opt_all['Glass (kPa)_max'] > 200).sum()),
        'above_300kPa': int((opt_all['Glass (kPa)_max'] > 300).sum()),
    },
    'improvement': {
        'max_improvement_pct': float((opt_max - initial_max) / initial_max * 100),
        'mean_improvement_pct': float((opt_mean - initial_mean) / initial_mean * 100),
        'max_predicted_rf_kPa': float(rf_preds.max()),
        'max_predicted_gp_kPa': float(gp_mu.max()),
        'max_ucb_kPa': float(ucb.max()),
    },
    'pathway_to_1MPa': {
        'current_max_kPa': float(max(opt_max, initial_max)),
        'target_kPa': 1000,
        'gap_kPa': float(1000 - max(opt_max, initial_max)),
        'gap_factor': float(1000 / max(opt_max, initial_max)),
        'key_findings': [
            'Hydrophobic (BA) and Aromatic (PEA) monomers are critical for high adhesion',
            'Cationic (ATAC) monomer shows strong positive correlation with adhesion',
            'Optimal compositions tend to have BA > 0.4 and low HEA',
            'ML-guided optimization improved max adhesion from ~305 to ~353 kPa',
            'Further optimization rounds show diminishing returns in current formulation space',
            'Achieving >1 MPa may require expanding the monomer library or crosslinking chemistry'
        ]
    }
}

with open(f"{BASE}/outputs/improvement_metrics.json", 'w') as f:
    json.dump(improvement_metrics, f, indent=2)

print("\n=== Improvement Summary ===")
print(f"Initial training max: {initial_max:.1f} kPa")
print(f"Optimized max: {opt_max:.1f} kPa")
print(f"Improvement: {(opt_max - initial_max)/initial_max*100:.1f}%")
print(f"Gap to 1 MPa: {1000 - opt_max:.1f} kPa ({1000/opt_max:.1f}x needed)")
print(f"Max RF prediction: {rf_preds.max():.1f} kPa")
print(f"Max GP prediction: {gp_mu.max():.1f} kPa")

print("\n=== Phase 4 Complete ===")
