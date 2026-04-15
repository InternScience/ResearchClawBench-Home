"""
Comprehensive analysis of hydrogel adhesive strength prediction
from monomer composition features.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')
import json
import os

np.random.seed(42)

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================
print("=" * 60)
print("PHASE 1: Data Loading & Preprocessing")
print("=" * 60)

FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET_COL = 'Glass (kPa)_10s'

# Load initial training data (verified 184)
df_init = pd.read_excel('data/184_verified_Original Data_ML_20230926.xlsx')
print(f"Initial training data: {df_init.shape[0]} samples, {df_init.shape[1]} features")

# Load optimization data (EI sheet)
df_opt_ei = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='EI')
df_opt_pred = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='PRED')

# Clean optimization data
df_opt_ei['Glass (kPa)_max'] = pd.to_numeric(df_opt_ei['Glass (kPa)_max'], errors='coerce')
df_opt_pred['Glass (kPa)_max'] = pd.to_numeric(df_opt_pred['Glass (kPa)_max'], errors='coerce')

# Combine all experimental data
# First convert all feature columns to numeric in optimization data
for col in FEATURE_COLS:
    df_opt_ei[col] = pd.to_numeric(df_opt_ei[col], errors='coerce')
    df_opt_pred[col] = pd.to_numeric(df_opt_pred[col], errors='coerce')

df_all = pd.DataFrame()
for col in FEATURE_COLS:
    df_all[col] = pd.concat([df_init[col], df_opt_ei[col], df_opt_pred[col]], ignore_index=True)
df_all['adhesive_strength_kPa'] = pd.concat([
    df_init[TARGET_COL], 
    df_opt_ei['Glass (kPa)_max'], 
    df_opt_pred['Glass (kPa)_max']
], ignore_index=True)
df_all['source'] = (['Initial'] * len(df_init) + 
                    ['Opt_EI'] * len(df_opt_ei) + 
                    ['Opt_PRED'] * len(df_opt_pred))

# Remove rows with missing target
df_clean = df_all.dropna(subset=['adhesive_strength_kPa']).copy()
df_clean = df_clean[df_clean['adhesive_strength_kPa'] > 0].copy()
print(f"Clean dataset: {len(df_clean)} samples")

# Check monomer composition sums
# Drop rows with any NaN in feature columns for analysis
df_clean = df_clean.dropna(subset=FEATURE_COLS).copy()
print(f"After dropping NaN features: {len(df_clean)} samples")
df_clean['monomer_sum'] = df_clean[FEATURE_COLS].sum(axis=1)
print(f"Monomer sum range: {df_clean['monomer_sum'].min():.3f} - {df_clean['monomer_sum'].max():.3f}")
print(f"Mean monomer sum: {df_clean['monomer_sum'].mean():.3f}")

# Target statistics
fa = df_clean['adhesive_strength_kPa']
print(f"\nAdhesive Strength Statistics (kPa):")
print(f"  Mean: {fa.mean():.2f}")
print(f"  Median: {fa.median():.2f}")
print(f"  Std: {fa.std():.2f}")
print(f"  Min: {fa.min():.2f}")
print(f"  Max: {fa.max():.2f}")
print(f"  > 100 kPa: {(fa > 100).sum()} ({(fa > 100).mean()*100:.1f}%)")
print(f"  > 200 kPa: {(fa > 200).sum()} ({(fa > 200).mean()*100:.1f}%)")
print(f"  > 300 kPa: {(fa > 300).sum()} ({(fa > 300).mean()*100:.1f}%)")

# Save summary
summary = {
    'n_samples': int(len(df_clean)),
    'n_initial': int(len(df_init)),
    'n_opt_ei': int(df_opt_ei['Glass (kPa)_max'].notna().sum()),
    'n_opt_pred': int(df_opt_pred['Glass (kPa)_max'].notna().sum()),
    'target_mean_kPa': round(float(fa.mean()), 2),
    'target_median_kPa': round(float(fa.median()), 2),
    'target_max_kPa': round(float(fa.max()), 2),
    'target_max_MPa': round(float(fa.max() / 1000), 4),
    'n_above_200kPa': int((fa > 200).sum()),
    'feature_cols': FEATURE_COLS,
}
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ============================================================
# 2. FIGURE 1: Data Overview
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 1: Data Overview")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1a: Distribution of adhesive strength
ax = axes[0, 0]
ax.hist(fa, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(fa.mean(), color='red', linestyle='--', label=f'Mean: {fa.mean():.1f} kPa')
ax.axvline(200, color='orange', linestyle='--', label='200 kPa threshold')
ax.set_xlabel('Adhesive Strength (kPa)')
ax.set_ylabel('Count')
ax.set_title('A. Distribution of Adhesive Strength')
ax.legend()

# 1b: Box plot by source
ax = axes[0, 1]
source_data = [df_clean[df_clean['source'] == s]['adhesive_strength_kPa'].values 
               for s in ['Initial', 'Opt_EI', 'Opt_PRED']]
bp = ax.boxplot(source_data, labels=['Initial\n(184)', 'Optimization\nEI', 'Optimization\nPRED'],
                patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightsalmon']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('Adhesive Strength (kPa)')
ax.set_title('B. Adhesive Strength by Dataset Source')

# 1c: Monomer composition distributions
ax = axes[0, 2]
monomer_means = df_clean[FEATURE_COLS].mean()
monomer_stds = df_clean[FEATURE_COLS].std()
bars = ax.bar(range(len(FEATURE_COLS)), monomer_means, yerr=monomer_stds, 
              color='steelblue', alpha=0.7, capsize=3)
ax.set_xticks(range(len(FEATURE_COLS)))
ax.set_xticklabels([c.split('-')[1] for c in FEATURE_COLS], rotation=45)
ax.set_ylabel('Mole Fraction')
ax.set_title('C. Average Monomer Composition')

# 1d: Correlation heatmap
ax = axes[1, 0]
corr_data = df_clean[FEATURE_COLS + ['adhesive_strength_kPa']].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, cbar_kws={'shrink': 0.8})
ax.set_title('D. Feature-Target Correlations')

# 1e: Scatter: each monomer vs adhesive strength
ax = axes[1, 1]
for i, col in enumerate(FEATURE_COLS):
    ax.scatter(df_clean[col], df_clean['adhesive_strength_kPa'], 
               alpha=0.3, s=10, label=col.split('-')[1])
ax.set_xlabel('Monomer Fraction')
ax.set_ylabel('Adhesive Strength (kPa)')
ax.set_title('E. Monomer Fraction vs Adhesive Strength')
ax.legend(fontsize=7, ncol=2)

# 1f: Composition ternary-like visualization (top 3 monomers)
ax = axes[1, 2]
top3 = ['Hydrophobic-BA', 'Nucleophilic-HEA', 'Aromatic-PEA']
scatter = ax.scatter(df_clean[top3[0]], df_clean[top3[1]], 
                     c=df_clean['adhesive_strength_kPa'], cmap='hot', s=20, alpha=0.6)
plt.colorbar(scatter, ax=ax, label='Adhesive Strength (kPa)')
ax.set_xlabel(top3[0].split('-')[1] + ' Fraction')
ax.set_ylabel(top3[1].split('-')[1] + ' Fraction')
ax.set_title('F. BA vs HEA (colored by strength)')

plt.tight_layout()
plt.savefig('report/images/fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig1_data_overview.png")

# ============================================================
# 3. ML MODEL TRAINING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: ML Model Training & Evaluation")
print("=" * 60)

# Use initial 184 samples for training (cleaner, verified data)
df_train = df_init.dropna(subset=[TARGET_COL]).copy()
df_train = df_train[df_train[TARGET_COL] > 0].copy()
X_train = df_train[FEATURE_COLS].values
y_train = df_train[TARGET_COL].values

print(f"Training set: {len(X_train)} samples")

# Also prepare full dataset for final model
X_all = df_clean[FEATURE_COLS].values
y_all = df_clean['adhesive_strength_kPa'].values

# Models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, 
                                            min_samples_leaf=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, 
                                                     learning_rate=0.05, random_state=42),
}

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation on initial data
    r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    rmse_scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'))
    
    print(f"  CV R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"  CV MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f} kPa")
    print(f"  CV RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f} kPa")
    
    # Fit on all training data
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    results[name] = {
        'cv_r2_mean': float(r2_scores.mean()),
        'cv_r2_std': float(r2_scores.std()),
        'cv_mae_mean': float(mae_scores.mean()),
        'cv_mae_std': float(mae_scores.std()),
        'cv_rmse_mean': float(rmse_scores.mean()),
        'cv_rmse_std': float(rmse_scores.std()),
        'train_r2': float(train_r2),
        'model': model,
    }

# Save model results
model_summary = {k: {kk: vv for kk, vv in v.items() if kk != 'model'} 
                 for k, v in results.items()}
with open('outputs/model_results.json', 'w') as f:
    json.dump(model_summary, f, indent=2)

# ============================================================
# 4. FIGURE 2: Model Performance
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 2: Model Performance")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 2a: CV R² comparison
ax = axes[0, 0]
model_names = list(results.keys())
r2_means = [results[m]['cv_r2_mean'] for m in model_names]
r2_stds = [results[m]['cv_r2_std'] for m in model_names]
bars = ax.bar(model_names, r2_means, yerr=r2_stds, color=['steelblue', 'coral'], capsize=5)
ax.set_ylabel('R² Score')
ax.set_title('A. Cross-Validation R² Comparison')
ax.set_ylim(0, 1)
for bar, val in zip(bars, r2_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', fontsize=10)

# 2b: Predicted vs Actual (best model)
best_model_name = max(results, key=lambda k: results[k]['cv_r2_mean'])
best_model = results[best_model_name]['model']
ax = axes[0, 1]
y_pred_best = best_model.predict(X_train)
ax.scatter(y_train, y_pred_best, alpha=0.5, s=30, color='steelblue')
lims = [0, max(y_train.max(), y_pred_best.max()) * 1.1]
ax.plot(lims, lims, 'r--', linewidth=2)
ax.set_xlabel('Actual Adhesive Strength (kPa)')
ax.set_ylabel('Predicted Adhesive Strength (kPa)')
ax.set_title(f'B. {best_model_name}: Predicted vs Actual (R²={results[best_model_name]["cv_r2_mean"]:.3f})')

# 2c: Residuals
ax = axes[1, 0]
residuals = y_train - y_pred_best
ax.scatter(y_pred_best, residuals, alpha=0.5, s=30, color='steelblue')
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Predicted Adhesive Strength (kPa)')
ax.set_ylabel('Residual (kPa)')
ax.set_title('C. Residual Plot')

# 2d: Feature importance (Random Forest)
ax = axes[1, 1]
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
ax.barh(range(len(FEATURE_COLS)), importances[indices], color='steelblue')
ax.set_yticks(range(len(FEATURE_COLS)))
ax.set_yticklabels([FEATURE_COLS[i].split('-')[1] for i in indices])
ax.set_xlabel('Feature Importance')
ax.set_title('D. Random Forest Feature Importance')

plt.tight_layout()
plt.savefig('report/images/fig2_model_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig2_model_performance.png")

# ============================================================
# 5. DESIGN SPACE EXPLORATION
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Design Space Exploration")
print("=" * 60)

# Use best model trained on all available data
final_model = RandomForestRegressor(n_estimators=500, max_depth=12, 
                                     min_samples_leaf=3, random_state=42)
final_model.fit(X_all, y_all)
print(f"Final model trained on {len(X_all)} samples")

# Monte Carlo sampling of composition space
n_samples = 100000
print(f"Sampling {n_samples} random compositions...")

compositions = np.random.dirichlet(np.ones(len(FEATURE_COLS)), size=n_samples)
predictions = final_model.predict(compositions)

# Find high-performance compositions
high_perf_mask = predictions > 200  # > 200 kPa
very_high_mask = predictions > 300  # > 300 kPa
print(f"Predicted > 200 kPa: {high_perf_mask.sum()} ({high_perf_mask.mean()*100:.2f}%)")
print(f"Predicted > 300 kPa: {very_high_mask.sum()} ({very_high_mask.mean()*100:.2f}%)")

if very_high_mask.sum() > 0:
    best_idx = predictions.argmax()
    best_comp = compositions[best_idx]
    best_pred = predictions[best_idx]
    print(f"\nBest predicted composition:")
    for col, val in zip(FEATURE_COLS, best_comp):
        print(f"  {col}: {val:.4f}")
    print(f"  Predicted strength: {best_pred:.2f} kPa = {best_pred/1000:.4f} MPa")

# Sensitivity analysis
print("\nSensitivity analysis...")
n_sens = 50
sensitivity = {}
for feat_idx, feat_name in enumerate(FEATURE_COLS):
    strengths = []
    for frac in np.linspace(0.01, 0.7, n_sens):
        comp = np.ones(len(FEATURE_COLS)) * (1 - frac) / (len(FEATURE_COLS) - 1)
        comp[feat_idx] = frac
        pred = final_model.predict(comp.reshape(1, -1))[0]
        strengths.append(pred)
    sensitivity[feat_name] = strengths
    print(f"  {feat_name}: max pred = {max(strengths):.1f} kPa at frac = {np.argmax(strengths)/n_sens*0.69+0.01:.2f}")

# Save design space results
design_results = {
    'n_samples': n_samples,
    'n_above_200kPa': int(high_perf_mask.sum()),
    'n_above_300kPa': int(very_high_mask.sum()),
    'best_predicted_kPa': float(predictions.max()),
    'best_predicted_MPa': float(predictions.max() / 1000),
    'best_composition': {col: float(val) for col, val in zip(FEATURE_COLS, compositions[predictions.argmax()])},
    'sensitivity_max': {k: float(max(v)) for k, v in sensitivity.items()},
}
with open('outputs/design_space_results.json', 'w') as f:
    json.dump(design_results, f, indent=2)

# ============================================================
# 6. FIGURE 3: Design Space
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 3: Design Space Exploration")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 3a: Distribution of predicted strengths
ax = axes[0, 0]
ax.hist(predictions, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
ax.axvline(200, color='orange', linestyle='--', label='200 kPa')
ax.axvline(300, color='red', linestyle='--', label='300 kPa')
ax.set_xlabel('Predicted Adhesive Strength (kPa)')
ax.set_ylabel('Count')
ax.set_title('A. Distribution of Predicted Strengths (100k MC)')
ax.legend()

# 3b-f: Top 5 sensitivity curves
top5_feats = sorted(sensitivity.items(), key=lambda x: max(x[1]), reverse=True)[:5]
for i, (feat_name, strengths) in enumerate(top5_feats):
    row = (i + 1) // 3
    col = (i + 1) % 3
    ax = axes[row, col]
    fracs = np.linspace(0.01, 0.7, n_sens)
    ax.plot(fracs, strengths, linewidth=2, color='steelblue')
    ax.fill_between(fracs, strengths, alpha=0.15, color='steelblue')
    ax.set_xlabel(f'{feat_name.split("-")[1]} Fraction')
    ax.set_ylabel('Predicted Strength (kPa)')
    ax.set_title(f'{chr(66+i)}. Sensitivity: {feat_name.split("-")[1]}')
    ax.axhline(200, color='orange', linestyle=':', alpha=0.5)

# Hide unused subplot
axes[1, 2].set_visible(False)

plt.tight_layout()
plt.savefig('report/images/fig3_design_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig3_design_space.png")

# ============================================================
# 7. FIGURE 4: Optimization Trajectory
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 4: Optimization Trajectory")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 4a: Adhesive strength over optimization rounds
ax = axes[0]
round_data = {
    'Initial (184)': df_init[TARGET_COL].dropna(),
    'Round 1 (EI)': df_opt_ei['Glass (kPa)_max'].dropna(),
    'Round 1 (PRED)': df_opt_pred['Glass (kPa)_max'].dropna(),
}
positions = range(len(round_data))
means = [d.mean() for d in round_data.values()]
maxes = [d.max() for d in round_data.values()]
ax.bar(positions, means, color='steelblue', alpha=0.7, label='Mean')
ax.plot(positions, maxes, 'ro-', markersize=8, label='Max')
ax.set_xticks(positions)
ax.set_xticklabels(round_data.keys(), rotation=30, ha='right')
ax.set_ylabel('Adhesive Strength (kPa)')
ax.set_title('A. Adhesive Strength Across Rounds')
ax.legend()

# 4b: Composition shift visualization
ax = axes[1]
init_means = df_init[FEATURE_COLS].mean()
opt_means = df_opt_ei[FEATURE_COLS].mean()
x = np.arange(len(FEATURE_COLS))
width = 0.35
ax.bar(x - width/2, init_means, width, label='Initial', color='steelblue', alpha=0.7)
ax.bar(x + width/2, opt_means, width, label='Optimized', color='coral', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([c.split('-')[1] for c in FEATURE_COLS], rotation=45)
ax.set_ylabel('Mean Mole Fraction')
ax.set_title('B. Composition Shift: Initial vs Optimized')
ax.legend()

# 4c: Cumulative best
ax = axes[2]
all_fa_sorted = np.sort(df_clean['adhesive_strength_kPa'].values)[::-1]
cummax = np.maximum.accumulate(all_fa_sorted)
ax.plot(range(len(cummax)), cummax, linewidth=2, color='steelblue')
ax.set_xlabel('Rank')
ax.set_ylabel('Cumulative Best Adhesive Strength (kPa)')
ax.set_title('C. Cumulative Best Performance')
ax.axhline(200, color='orange', linestyle='--', alpha=0.5, label='200 kPa')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/fig4_optimization_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig4_optimization_trajectory.png")

# ============================================================
# 8. FIGURE 5: Top Formulations Analysis
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 5: Top Formulations")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 5a: Top 20 formulations composition heatmap
ax = axes[0]
top20 = df_clean.nlargest(20, 'adhesive_strength_kPa')
top20_comp = top20[FEATURE_COLS].values
sns.heatmap(top20_comp, ax=ax, cmap='YlOrRd', 
            xticklabels=[c.split('-')[1] for c in FEATURE_COLS],
            yticklabels=[f'#{i+1} ({v:.0f}kPa)' for i, v in enumerate(top20['adhesive_strength_kPa'].values)],
            cbar_kws={'label': 'Mole Fraction'})
ax.set_title('A. Top 20 Formulations by Composition')

# 5b: Composition statistics of top performers vs all
ax = axes[1]
top10pct = df_clean.nlargest(int(len(df_clean)*0.1), 'adhesive_strength_kPa')
all_means = df_clean[FEATURE_COLS].mean()
top_means = top10pct[FEATURE_COLS].mean()
x = np.arange(len(FEATURE_COLS))
width = 0.35
ax.bar(x - width/2, all_means, width, label='All samples', color='steelblue', alpha=0.7)
ax.bar(x + width/2, top_means, width, label='Top 10%', color='coral', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([c.split('-')[1] for c in FEATURE_COLS], rotation=45)
ax.set_ylabel('Mean Mole Fraction')
ax.set_title('B. Composition: Top 10% vs All Samples')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/fig5_top_formulations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: report/images/fig5_top_formulations.png")

print("\n" + "=" * 60)
print("ALL ANALYSIS COMPLETE")
print("=" * 60)
print(f"Best observed adhesive strength: {fa.max():.2f} kPa = {fa.max()/1000:.4f} MPa")
print(f"Best predicted adhesive strength: {predictions.max():.2f} kPa = {predictions.max()/1000:.4f} MPa")
print(f"Target >1 MPa (1000 kPa): NOT ACHIEVED in current data")
print(f"Max observed is {fa.max()/1000:.4f} MPa, which is {1000/fa.max():.1f}x below 1 MPa target")
