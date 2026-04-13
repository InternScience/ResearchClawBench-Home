
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

ROOT = Path('.')
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150

feature_cols = [
    'Nucleophilic-HEA','Hydrophobic-BA','Acidic-CBEA',
    'Cationic-ATAC','Aromatic-PEA','Amide-AAm',
    'Q','Phase Seperation','Modulus (kPa)','Tanδ','Slope','XlogP3'
]
response = 'Glass (kPa)_10s'

train = pd.read_excel(DATA / '184_verified_Original Data_ML_20230926.xlsx', sheet_name='Data_to_HU')
for c in ['Tanδ','Slope']:
    train[c] = pd.to_numeric(train[c], errors='coerce')
train['Phase Seperation'] = pd.to_numeric(train['Phase Seperation'], errors='coerce')
train['is_superadhesive'] = (train[response] >= 1000).astype(int)
X = train[feature_cols].copy()
y = pd.to_numeric(train[response], errors='coerce')

rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
oof_cv = KFold(n_splits=5, shuffle=True, random_state=42)

rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

gp = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel(noise_level=1.0),
        normalize_y=True,
        random_state=42,
        n_restarts_optimizer=0
    ))
])

models = {'RF': rf, 'GP': gp}
metrics = {}
preds = {}
for name, model in models.items():
    cv = cross_validate(model, X, y, cv=rkf, scoring=['r2','neg_mean_absolute_error','neg_root_mean_squared_error'], n_jobs=1)
    # cross_val_predict requires non-overlapping partitions; use a single shuffled 5-fold split for OOF plots
    from sklearn.model_selection import cross_val_predict
    yhat = cross_val_predict(model, X, y, cv=oof_cv, n_jobs=1)
    metrics[name] = {
        'cv_r2_mean': float(np.mean(cv['test_r2'])),
        'cv_r2_std': float(np.std(cv['test_r2'])),
        'cv_mae_mean': float(-np.mean(cv['test_neg_mean_absolute_error'])),
        'cv_rmse_mean': float(-np.mean(cv['test_neg_root_mean_squared_error'])),
        'overall_r2_from_oof': float(r2_score(y, yhat)),
        'overall_mae_from_oof': float(mean_absolute_error(y, yhat)),
        'overall_rmse_from_oof': float(np.sqrt(mean_squared_error(y, yhat))),
    }
    preds[name] = yhat

rf.fit(X, y)
imp = permutation_importance(rf, X, y, n_repeats=20, random_state=42, scoring='neg_mean_absolute_error')
imp_df = pd.DataFrame({'feature': feature_cols, 'importance_mean': imp.importances_mean, 'importance_std': imp.importances_std}).sort_values('importance_mean', ascending=False)
imp_df.to_csv(OUT / 'rf_permutation_importance.csv', index=False)

# Optimization dataset
opt_ei = pd.read_excel(DATA / 'ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='EI')
opt_pred = pd.read_excel(DATA / 'ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='PRED')
for df in [opt_ei, opt_pred]:
    df.rename(columns={'NO.':'NO'}, inplace=True)
    df['strategy'] = df['ML'].ffill()
    df['dataset'] = 'EI' if df is opt_ei else 'PRED'
    for c in ['Nucleophilic-HEA','Hydrophobic-BA','Acidic-CBEA','Cationic-ATAC','Aromatic-PEA','Amide-AAm','Glass (kPa)_max']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['predicted_strength_kPa'] = pd.to_numeric(df['Glass (kPa)_max'], errors='coerce')
all_opt = pd.concat([opt_ei, opt_pred], ignore_index=True)
all_opt = all_opt.dropna(subset=['Nucleophilic-HEA','Hydrophobic-BA','Acidic-CBEA','Cationic-ATAC','Aromatic-PEA','Amide-AAm']).reset_index(drop=True)

# Predictions on optimization formulas from trained RF
opt_X = all_opt[[c for c in feature_cols if c in all_opt.columns]].copy()
for missing in [c for c in feature_cols if c not in opt_X.columns]:
    opt_X[missing] = np.nan
opt_X = opt_X[feature_cols]
all_opt['rf_pred_from_initial'] = rf.predict(opt_X)

# Rank candidate formulations by model prediction
candidate_grid = all_opt[['strategy','dataset','NO','predicted_strength_kPa','rf_pred_from_initial'] + [c for c in feature_cols if c in all_opt.columns]].copy()
candidate_grid.to_csv(OUT / 'optimization_candidates_scored.csv', index=False)

top_candidates = candidate_grid.sort_values('rf_pred_from_initial', ascending=False).head(20)
top_candidates.to_csv(OUT / 'top20_candidates.csv', index=False)

# Dominant design motif among top candidates
comp_cols = ['Nucleophilic-HEA','Hydrophobic-BA','Acidic-CBEA','Cationic-ATAC','Aromatic-PEA','Amide-AAm']
km = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = km.fit_predict(top_candidates[comp_cols])
top_candidates = top_candidates.copy()
top_candidates['cluster'] = clusters
cluster_centers = pd.DataFrame(km.cluster_centers_, columns=comp_cols)
cluster_centers.to_csv(OUT / 'top20_cluster_centers.csv', index=False)
top_candidates.to_csv(OUT / 'top20_candidates_clustered.csv', index=False)

summary = {
    'n_train': int(len(train)),
    'n_superadhesive_ge_1000_kpa': int(train['is_superadhesive'].sum()),
    'superadhesive_fraction': float(train['is_superadhesive'].mean()),
    'train_response_summary': train[response].describe().to_dict(),
    'model_metrics': metrics,
    'feature_importance_top5': imp_df.head(5).to_dict(orient='records'),
    'top_candidate_examples': top_candidates.head(10).to_dict(orient='records')
}
with open(OUT / 'summary_metrics.json', 'w') as f:
    json.dump(summary, f, indent=2)

train.to_csv(OUT / 'training_data_processed.csv', index=False)

# Figures
# 1 data overview distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(train[response], bins=30, ax=axes[0,0], color='#4C78A8')
axes[0,0].axvline(1000, color='red', linestyle='--', label='1 MPa target')
axes[0,0].set_title('Distribution of glass adhesion strength')
axes[0,0].legend()

comp_long = train[comp_cols].melt(var_name='Monomer', value_name='Fraction')
sns.boxplot(data=comp_long, x='Monomer', y='Fraction', ax=axes[0,1], color='#72B7B2')
axes[0,1].tick_params(axis='x', rotation=30)
axes[0,1].set_title('Monomer composition ranges')

sns.scatterplot(data=train, x='Hydrophobic-BA', y=response, hue='Aromatic-PEA', palette='viridis', ax=axes[1,0])
axes[1,0].axhline(1000, color='red', linestyle='--')
axes[1,0].set_title('Hydrophobicity-aromaticity interaction')

sns.scatterplot(data=train, x='Modulus (kPa)', y=response, hue='Q', palette='magma', ax=axes[1,1])
axes[1,1].axhline(1000, color='red', linestyle='--')
axes[1,1].set_title('Mechanics-swelling interaction')
plt.tight_layout()
fig.savefig(IMG / 'data_overview.png', bbox_inches='tight')
plt.close(fig)

# 2 observed vs predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, name in zip(axes, ['RF','GP']):
    ax.scatter(y, preds[name], alpha=0.65, s=45)
    lims = [min(y.min(), preds[name].min()), max(y.max(), preds[name].max())]
    ax.plot(lims, lims, 'r--')
    ax.axhline(1000, color='grey', linestyle=':')
    ax.axvline(1000, color='grey', linestyle=':')
    ax.set_xlabel('Observed glass adhesion (kPa)')
    ax.set_ylabel('OOF predicted (kPa)')
    ax.set_title(f'{name}: R²={metrics[name]["overall_r2_from_oof"]:.2f}, RMSE={metrics[name]["overall_rmse_from_oof"]:.1f}')
plt.tight_layout()
fig.savefig(IMG / 'model_validation.png', bbox_inches='tight')
plt.close(fig)

# 3 importance
fig, ax = plt.subplots(figsize=(10, 7))
plot_df = imp_df.sort_values('importance_mean')
ax.barh(plot_df['feature'], plot_df['importance_mean'], xerr=plot_df['importance_std'], color='#F58518')
ax.set_xlabel('Permutation importance (decrease in neg-MAE)')
ax.set_title('Feature importance from random forest')
plt.tight_layout()
fig.savefig(IMG / 'feature_importance.png', bbox_inches='tight')
plt.close(fig)

# 4 candidate space comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(data=train, x='Hydrophobic-BA', y='Aromatic-PEA', color='lightgray', alpha=0.6, ax=axes[0], label='Initial 184')
sns.scatterplot(data=top_candidates, x='Hydrophobic-BA', y='Aromatic-PEA', hue='strategy', s=90, ax=axes[0])
axes[0].set_title('Top candidates shift toward hydrophobic/aromatic corner')

sns.scatterplot(data=train, x='Nucleophilic-HEA', y='Cationic-ATAC', color='lightgray', alpha=0.6, ax=axes[1], label='Initial 184')
sns.scatterplot(data=top_candidates, x='Nucleophilic-HEA', y='Cationic-ATAC', hue='strategy', s=90, ax=axes[1])
axes[1].set_title('Top candidates reduce HEA while retaining cationic content')
plt.tight_layout()
fig.savefig(IMG / 'candidate_comparison.png', bbox_inches='tight')
plt.close(fig)

# 5 top candidates bars
fig, ax = plt.subplots(figsize=(13, 7))
show = top_candidates.head(12).copy()
show['label'] = show['strategy'] + '-' + show['dataset'] + '-' + show['NO'].astype(str)
sns.barplot(data=show, x='label', y='rf_pred_from_initial', color='#54A24B', ax=ax)
ax.axhline(1000, color='red', linestyle='--', label='1 MPa target')
ax.tick_params(axis='x', rotation=60)
ax.set_ylabel('RF-predicted adhesion from initial model (kPa)')
ax.set_xlabel('Candidate')
ax.set_title('Highest-scoring de novo candidate formulations')
ax.legend()
plt.tight_layout()
fig.savefig(IMG / 'top_candidates.png', bbox_inches='tight')
plt.close(fig)

print('Analysis complete.')
print(json.dumps(summary, indent=2)[:3000])
