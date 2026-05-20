import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, norm
import warnings, os, json
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_excel('data/184_verified_Original Data_ML_20230926.xlsx')
opt2 = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx')
opt2['ML_group'] = opt2['ML'].ffill()
opt2['Glass (kPa)_max'] = pd.to_numeric(opt2['Glass (kPa)_max'], errors='coerce')

features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
# Convert all feature columns to numeric, coercing errors to NaN
for f in features:
    opt2[f] = pd.to_numeric(opt2[f], errors='coerce')
feature_labels = ['HEA\n(Nucleophilic)', 'BA\n(Hydrophobic)', 'CBEA\n(Acidic)', 'ATAC\n(Cationic)', 'PEA\n(Aromatic)', 'AAm\n(Amide)']

# ============================================================
# FIG 1: Feature distributions (initial vs optimized)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, (feat, lbl) in enumerate(zip(features, feature_labels)):
    ax = axes[i//3, i%3]
    ax.hist(df[feat], bins=25, color='steelblue', edgecolor='black', alpha=0.6, label='Initial (n=184)')
    opt_vals = opt2[feat].dropna()
    ax.hist(opt_vals, bins=15, color='coral', edgecolor='black', alpha=0.6, label='Optimized (n=119)')
    ax.set_xlabel('Mole Fraction', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(lbl, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
plt.suptitle('Monomer Composition Distributions: Initial vs Optimized Hydrogels', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('report/images/fig1_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 2: Adhesion distributions
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].hist(df['Glass (kPa)_10s'].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=1000, color='red', linestyle='--', linewidth=2, label='1 MPa target')
axes[0].set_xlabel('Glass Adhesion (kPa)')
axes[0].set_ylabel('Count')
axes[0].set_title('Initial Dataset (n=184)\nMean=51, Max=305 kPa')
axes[0].legend()

axes[1].hist(opt2['Glass (kPa)_max'].dropna(), bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1].axvline(x=1000, color='red', linestyle='--', linewidth=2, label='1 MPa target')
axes[1].set_xlabel('Glass Adhesion (kPa)')
axes[1].set_ylabel('Count')
axes[1].set_title('Optimized Formulations (n=119)\nMean=144, Max=321 kPa')
axes[1].legend()

axes[2].hist(df['Glass (kPa)_10s'].dropna(), bins=25, color='steelblue', alpha=0.5, label='Initial', edgecolor='black')
axes[2].hist(opt2['Glass (kPa)_max'].dropna(), bins=25, color='coral', alpha=0.5, label='Optimized', edgecolor='black')
axes[2].axvline(x=1000, color='red', linestyle='--', linewidth=2, label='1 MPa target')
axes[2].set_xlabel('Glass Adhesion (kPa)')
axes[2].set_ylabel('Count')
axes[2].set_title('Comparison: Initial vs Optimized')
axes[2].legend()
plt.suptitle('Adhesion Strength Distributions on Glass Substrate', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('report/images/fig2_adhesion_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 3: Correlation heatmap
# ============================================================
corr_data = df[features + ['Glass (kPa)_10s', 'Steel (kPa)_10s', 'Modulus (kPa)', 'Q']].dropna()
corr = corr_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=corr.columns, yticklabels=corr.columns, square=True)
plt.title('Feature-Feature and Feature-Target Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 4: Feature-adhesion scatter
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, (feat, lbl) in enumerate(zip(features, feature_labels)):
    ax = axes[i//3, i%3]
    valid = df[[feat, 'Glass (kPa)_10s']].dropna()
    ax.scatter(valid[feat], valid['Glass (kPa)_10s'], alpha=0.5, c='steelblue', edgecolors='black', s=40)
    r, p = pearsonr(valid[feat], valid['Glass (kPa)_10s'])
    ax.set_xlabel('Mole Fraction')
    ax.set_ylabel('Glass Adhesion (kPa)')
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.set_title(f'{lbl}: r={r:.3f} ({sig})', fontsize=10)
plt.suptitle('Monomer Composition vs Glass Adhesion Strength', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('report/images/fig4_feature_adhesion_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# MODEL TRAINING
# ============================================================
X = df[features].values
y_glass = df['Glass (kPa)_10s'].values
y_steel = df['Steel (kPa)_10s'].values
mask_g = ~np.isnan(y_glass)
mask_s = ~np.isnan(y_steel)
X_g, y_g = X[mask_g], y_glass[mask_g]
X_s, y_s = X[mask_s], y_steel[mask_s]
cv = KFold(n_splits=5, shuffle=True, random_state=42)

rfr_g = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=3, random_state=42)
rfr_s = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=3, random_state=42)
r2_g_rfr = cross_val_score(rfr_g, X_g, y_g, cv=cv, scoring='r2')
mae_g_rfr = -cross_val_score(rfr_g, X_g, y_g, cv=cv, scoring='neg_mean_absolute_error')

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
scaler_g = StandardScaler().fit(X_g)
gpr_g = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5)
r2_g_gpr = cross_val_score(gpr_g, scaler_g.transform(X_g), y_g, cv=cv, scoring='r2')
mae_g_gpr = -cross_val_score(gpr_g, scaler_g.transform(X_g), y_g, cv=cv, scoring='neg_mean_absolute_error')

gbr_g = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
r2_g_gbr = cross_val_score(gbr_g, X_g, y_g, cv=cv, scoring='r2')
mae_g_gbr = -cross_val_score(gbr_g, X_g, y_g, cv=cv, scoring='neg_mean_absolute_error')

r2_s_rfr = cross_val_score(rfr_s, X_s, y_s, cv=cv, scoring='r2')
mae_s_rfr = -cross_val_score(rfr_s, X_s, y_s, cv=cv, scoring='neg_mean_absolute_error')

rfr_g.fit(X_g, y_g)
rfr_s.fit(X_s, y_s)
gbr_g.fit(X_g, y_g)

print(f"RFR Glass: R2={r2_g_rfr.mean():.3f}+/-{r2_g_rfr.std():.3f}, MAE={mae_g_rfr.mean():.1f}+/-{mae_g_rfr.std():.1f}")
print(f"GPR Glass: R2={r2_g_gpr.mean():.3f}+/-{r2_g_gpr.std():.3f}, MAE={mae_g_gpr.mean():.1f}+/-{mae_g_gpr.std():.1f}")
print(f"GBR Glass: R2={r2_g_gbr.mean():.3f}+/-{r2_g_gbr.std():.3f}, MAE={mae_g_gbr.mean():.1f}+/-{mae_g_gbr.std():.1f}")
print(f"RFR Steel: R2={r2_s_rfr.mean():.3f}+/-{r2_s_rfr.std():.3f}, MAE={mae_s_rfr.mean():.1f}+/-{mae_s_rfr.std():.1f}")

# ============================================================
# FIG 5: Model comparison
# ============================================================
models = ['RFR', 'GPR', 'GBR']
glass_r2 = [r2_g_rfr.mean(), r2_g_gpr.mean(), r2_g_gbr.mean()]
glass_r2_err = [r2_g_rfr.std(), r2_g_gpr.std(), r2_g_gbr.std()]
glass_mae = [mae_g_rfr.mean(), mae_g_gpr.mean(), mae_g_gbr.mean()]
glass_mae_err = [mae_g_rfr.std(), mae_g_gpr.std(), mae_g_gbr.std()]
colors = ['#2196F3', '#FF9800', '#4CAF50']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(models, glass_r2, yerr=glass_r2_err, color=colors, edgecolor='black', capsize=5)
axes[0].set_ylabel('Cross-validated R2')
axes[0].set_title('Prediction Accuracy (R2)')
axes[0].set_ylim(0, 1)
for i, v in enumerate(glass_r2):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[1].bar(models, glass_mae, yerr=glass_mae_err, color=colors, edgecolor='black', capsize=5)
axes[1].set_ylabel('MAE (kPa)')
axes[1].set_title('Prediction Error (MAE)')
for i, v in enumerate(glass_mae):
    axes[1].text(i, v + 0.3, f'{v:.1f}', ha='center', fontweight='bold')
plt.suptitle('Model Performance Comparison for Glass Adhesion Prediction (5-Fold CV)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('report/images/fig5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 6: Feature importance
# ============================================================
fi_rfr = rfr_g.feature_importances_
fi_gbr = gbr_g.feature_importances_
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(features))
ax.barh(y_pos - 0.2, fi_rfr, height=0.35, color='steelblue', edgecolor='black', label='Random Forest')
ax.barh(y_pos + 0.2, fi_gbr, height=0.35, color='coral', edgecolor='black', label='Gradient Boosting')
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_labels, fontsize=10)
ax.set_xlabel('Feature Importance (MDI)')
ax.set_title('Monomer Feature Importance for Glass Adhesion Prediction', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('report/images/fig6_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 7: Predicted vs Actual
# ============================================================
y_pred_g = rfr_g.predict(X_g)
r2_fit = r2_score(y_g, y_pred_g)
rmse_fit = np.sqrt(mean_squared_error(y_g, y_pred_g))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
lims = [0, max(y_g.max(), y_pred_g.max()) * 1.1]
axes[0].scatter(y_g, y_pred_g, alpha=0.5, c='steelblue', edgecolors='black', s=50)
axes[0].plot(lims, lims, 'r--', linewidth=2)
axes[0].set_xlabel('Actual Glass Adhesion (kPa)')
axes[0].set_ylabel('Predicted Glass Adhesion (kPa)')
axes[0].set_title(f'RFR: R2={r2_fit:.3f}, RMSE={rmse_fit:.1f} kPa')
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
residuals = y_g - y_pred_g
axes[1].scatter(y_pred_g, residuals, alpha=0.5, c='steelblue', edgecolors='black', s=50)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Glass Adhesion (kPa)')
axes[1].set_ylabel('Residual (kPa)')
axes[1].set_title('Residual Distribution')
plt.suptitle('RFR Model Diagnostic: Predicted vs Actual', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('report/images/fig7_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 8: Optimization strategy comparison
# ============================================================
strat_data = opt2.dropna(subset=['Glass (kPa)_max']).groupby('ML_group')['Glass (kPa)_max'].agg(['mean', 'max', 'std', 'count'])
strat_data = strat_data.sort_values('max', ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
names = strat_data.index.tolist()
maxvals = strat_data['max'].values
colors_bar = ['#d32f2f' if v > 250 else '#ff9800' if v > 200 else '#4CAF50' if v > 150 else '#2196F3' for v in maxvals]
ax.barh(range(len(names)), maxvals, color=colors_bar, edgecolor='black')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Maximum Glass Adhesion (kPa)')
ax.set_title('Best Adhesion by ML Strategy')
ax.axvline(x=305, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Initial max (305 kPa)')
ax.legend()
for i, v in enumerate(maxvals):
    ax.text(v + 3, i, f'{v:.0f}', va='center', fontsize=9, fontweight='bold')

groups_for_box = ['RFR-RFR', 'RFR-GP', 'GP-GP', 'GP-RFR', 'LP_df', 'old-SM-GP', 'RFR-GP-2rd-ei', 'GP-GP-2rd-ei', 'RFR-GP-3rd-ei', 'GP-GP-3rd-ei']
box_data = [opt2[opt2['ML_group'] == g]['Glass (kPa)_max'].dropna().values for g in groups_for_box]
bp = axes[1].boxplot(box_data, labels=groups_for_box, patch_artist=True, vert=True)
for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(groups_for_box)))):
    patch.set_facecolor(color)
axes[1].set_xticklabels(groups_for_box, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('Glass Adhesion (kPa)')
axes[1].set_title('Adhesion Distribution by Strategy')
axes[1].axhline(y=305, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Initial max')
axes[1].legend()
plt.suptitle('ML-Guided Optimization Performance Across Strategies', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('report/images/fig8_optimization_results.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 9: PCA design space
# ============================================================
X_opt = opt2[features].dropna().values
X_all = np.vstack([X_g, X_opt])
labels_all = np.array(['Initial'] * len(X_g) + ['Optimized'] * len(X_opt))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_all))
fig, ax = plt.subplots(figsize=(10, 7))
m_init = labels_all == 'Initial'
m_opt = labels_all == 'Optimized'
ax.scatter(X_pca[m_init, 0], X_pca[m_init, 1], alpha=0.3, c='gray', edgecolors='black', s=25, label='Initial (n=184)')
ax.scatter(X_pca[m_opt, 0], X_pca[m_opt, 1], alpha=0.7, c='red', edgecolors='black', s=50, label='Optimized (n=119)', marker='^')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('Chemical Design Space: Initial vs Optimized Formulations (PCA)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('report/images/fig9_pca_design_space.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 10: Monomer role analysis
# ============================================================
bio_roles = [
    ('Nucleophilic-HEA', 'Nucleophilic (Dopa mimic)\nCatechol-like surface binding'),
    ('Hydrophobic-BA', 'Hydrophobic\nWater exclusion barrier'),
    ('Acidic-CBEA', 'Acidic\nH-bonding & surface contact'),
    ('Cationic-ATAC', 'Cationic\nElectrostatic interactions'),
    ('Aromatic-PEA', 'Aromatic\nPi-stacking & metal coord.'),
    ('Amide-AAm', 'Amide\nH-bond network & cohesion')
]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, (feat, role) in enumerate(bio_roles):
    ax = axes[i//3, i%3]
    init = df[feat].dropna().values
    opt_vals = opt2[feat].dropna().values
    ax.hist(init, bins=20, alpha=0.5, color='gray', label='Initial', edgecolor='black', density=True)
    ax.hist(opt_vals, bins=15, alpha=0.6, color='red', label='Optimized', edgecolor='black', density=True)
    init_mean, opt_mean = np.mean(init), np.mean(opt_vals)
    ax.axvline(init_mean, color='gray', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(opt_mean, color='red', linestyle='--', linewidth=2, alpha=0.8)
    direction = 'UP' if opt_mean > init_mean else 'DOWN'
    ax.set_xlabel('Mole Fraction')
    ax.set_ylabel('Density')
    ax.set_title(f'{role}\nMean: {init_mean:.3f} -> {opt_mean:.3f} ({direction})', fontsize=9)
    ax.legend(fontsize=8)
plt.suptitle('Biological Role Mapping: How Optimization Shifts Monomer Compositions\n(Mimicking Natural Adhesive Protein Features)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('report/images/fig10_monomer_roles.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 11: Learning curve
# ============================================================
train_sizes, train_scores, val_scores = learning_curve(
    rfr_g, X_g, y_g, train_sizes=np.linspace(0.2, 1.0, 8), cv=5, scoring='r2', random_state=42
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='steelblue', label='Training R2', linewidth=2)
ax.plot(train_sizes, val_scores.mean(axis=1), 's-', color='coral', label='Validation R2', linewidth=2)
ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2, color='coral')
ax.set_xlabel('Training Set Size', fontsize=12)
ax.set_ylabel('R2 Score', fontsize=12)
ax.set_title('Learning Curve: Random Forest for Glass Adhesion', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig11_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 12: Compositional shift heatmap
# ============================================================
init_means = df[features].mean()
opt_means = opt2[features].dropna().mean()
diff_pct = ((opt_means - init_means) / init_means * 100).values.reshape(1, -1)
fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(diff_pct, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0,
            xticklabels=[f.replace('-', '\n') for f in features],
            yticklabels=['Change (%)'], ax=ax, cbar_kws={'label': '% Change'})
ax.set_title('Compositional Shift: Optimized vs Initial Mean Mole Fractions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig12_compositional_shift.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 13: Adhesion improvement trajectory
# ============================================================
round_map = {
    'RFR-RFR': 1, 'RFR-GP': 1, 'GP-GP': 1, 'GP-RFR': 1,
    'ENU-RFR': 1, 'ENU-GP': 1, 'CLMax': 1, 'CLMin': 1, 'LP_df': 1, 'old-SM-GP': 1,
    'RFR-GP-2rd-ei': 2, 'GP-GP-2rd-ei': 2,
    'RFR-GP-3rd-ei': 3, 'GP-GP-3rd-ei': 3
}
opt2['Round'] = opt2['ML_group'].map(round_map)
fig, ax = plt.subplots(figsize=(10, 6))
for rnd in [1, 2, 3]:
    subset = opt2[opt2['Round'] == rnd].dropna(subset=['Glass (kPa)_max'])
    vals = subset['Glass (kPa)_max'].values
    jitter = np.random.normal(0, 0.05, len(vals))
    ax.scatter([rnd]*len(vals)+jitter, vals, alpha=0.5, s=50, label=f'Round {rnd} (n={len(vals)})', edgecolors='black')
for rnd in [1, 2, 3]:
    subset = opt2[opt2['Round'] == rnd].dropna(subset=['Glass (kPa)_max'])
    if len(subset) > 0:
        ax.hlines(subset['Glass (kPa)_max'].mean(), rnd-0.2, rnd+0.2, colors='black', linewidth=3)
ax.axhline(y=305, color='red', linestyle='--', linewidth=2, label='Initial max (305 kPa)')
ax.set_xlabel('Optimization Round', fontsize=12)
ax.set_ylabel('Glass Adhesion (kPa)', fontsize=12)
ax.set_title('Adhesion Improvement Across Optimization Rounds', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Round 1\n(10 strategies)', 'Round 2\n(EI-guided)', 'Round 3\n(EI-guided)'])
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig('report/images/fig13_optimization_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FIG 14: Top candidates composition
# ============================================================
top10 = opt2.dropna(subset=['Glass (kPa)_max']).nlargest(10, 'Glass (kPa)_max')
fig, ax = plt.subplots(figsize=(10, 6))
for i, (_, row) in enumerate(top10.iterrows()):
    vals = row[features].values
    ax.plot(range(len(features)), vals, 'o-', alpha=0.5, label=f'{row["Glass (kPa)_max"]:.0f} kPa')
ax.set_xticks(range(len(features)))
ax.set_xticklabels([f.replace('-', '\n') for f in features], fontsize=10)
ax.set_ylabel('Mole Fraction')
ax.set_title('Top 10 Optimized Formulations: Monomer Composition Profiles', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)
plt.tight_layout()
plt.savefig('report/images/fig14_top_candidates.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# SAVE SUMMARY
# ============================================================
summary = {
    'n_initial': int(len(df)),
    'glass_10s_mean_kPa': float(df['Glass (kPa)_10s'].mean()),
    'glass_10s_max_kPa': float(df['Glass (kPa)_10s'].max()),
    'steel_10s_mean_kPa': float(df['Steel (kPa)_10s'].mean()),
    'steel_10s_max_kPa': float(df['Steel (kPa)_10s'].max()),
    'n_optimized': int(opt2.dropna(subset=['Glass (kPa)_max']).shape[0]),
    'opt_glass_mean_kPa': float(opt2['Glass (kPa)_max'].dropna().mean()),
    'opt_glass_max_kPa': float(opt2['Glass (kPa)_max'].dropna().max()),
    'n_above_200_kPa': int((opt2['Glass (kPa)_max'].dropna() > 200).sum()),
    'rfr_glass_r2': f"{r2_g_rfr.mean():.3f}+/-{r2_g_rfr.std():.3f}",
    'rfr_glass_mae': f"{mae_g_rfr.mean():.1f}+/-{mae_g_rfr.std():.1f}",
    'gpr_glass_r2': f"{r2_g_gpr.mean():.3f}+/-{r2_g_gpr.std():.3f}",
    'gpr_glass_mae': f"{mae_g_gpr.mean():.1f}+/-{mae_g_gpr.std():.1f}",
    'gbr_glass_r2': f"{r2_g_gbr.mean():.3f}+/-{r2_g_gbr.std():.3f}",
    'gbr_glass_mae': f"{mae_g_gbr.mean():.1f}+/-{mae_g_gbr.std():.1f}",
    'rfr_steel_r2': f"{r2_s_rfr.mean():.3f}+/-{r2_s_rfr.std():.3f}",
}
with open('outputs/experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print("\nAll figures and outputs saved!")
