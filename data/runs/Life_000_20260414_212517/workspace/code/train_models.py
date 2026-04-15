import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json

# Load data
df = pd.read_csv('outputs/initial_data_processed.csv')
features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
X = df[features].values
y = df['Glass_max'].values

print('Data for training:', X.shape, y.shape)
print('y mean/std/max:', y.mean(), y.std(), y.max())

# Models
models = {}
cv_results = {}

# RFR
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
cv_rfr = cross_validate(rfr, X, y, cv=5, scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'])
models['rfr'] = rfr.fit(X, y)
cv_results['rfr'] = {
    'r2_mean': cv_rfr['test_r2'].mean(),
    'r2_std': cv_rfr['test_r2'].std(),
    'mae_mean': -cv_rfr['test_neg_mean_absolute_error'].mean(),
    'mae_std': cv_rfr['test_neg_mean_absolute_error'].std(),
    'rmse_mean': np.sqrt(-cv_rfr['test_neg_root_mean_squared_error'].mean())
}
print('RFR CV:', cv_results['rfr'])

# GP
kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e1)) + WhiteKernel(noise_level=1e-5)
gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=10)
cv_gp = cross_validate(gp, X, y, cv=5, scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'])
models['gp'] = gp.fit(X, y)
cv_results['gp'] = {
    'r2_mean': cv_gp['test_r2'].mean(),
    'r2_std': cv_gp['test_r2'].std(),
    'mae_mean': -cv_gp['test_neg_mean_absolute_error'].mean(),
    'mae_std': cv_gp['test_neg_mean_absolute_error'].std(),
    'rmse_mean': np.sqrt(-cv_gp['test_neg_root_mean_squared_error'].mean())
}
print('GP CV:', cv_results['gp'])

# Save
Path('outputs/models').mkdir(exist_ok=True)
joblib.dump(models, 'outputs/models/trained_models.joblib')
with open('outputs/model_metrics.json', 'w') as f:
    json.dump(cv_results, f, indent=2)

# Predictions for plots (use full fit)
y_pred_rfr = models['rfr'].predict(X)
y_pred_gp = models['gp'].predict(X)

# Plots
Path('report/images').mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(12,10))

# RFR pred vs true
axes[0,0].scatter(y, y_pred_rfr, alpha=0.6)
axes[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axes[0,0].set_xlabel('True')
axes[0,0].set_ylabel('Pred RFR')
axes[0,0].set_title('RFR Pred vs True')

# GP
axes[0,1].scatter(y, y_pred_gp, alpha=0.6)
axes[0,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axes[0,1].set_xlabel('True')
axes[0,1].set_ylabel('Pred GP')
axes[0,1].set_title('GP Pred vs True')

# Feature importance RFR
importances = models['rfr'].feature_importances_
axes[1,0].bar(features, importances)
axes[1,0].set_title('RFR Feature Importance')
axes[1,0].tick_params(axis='x', rotation=45)

# Residuals RFR
residuals = y - y_pred_rfr
axes[1,1].scatter(y_pred_rfr, residuals, alpha=0.6)
axes[1,1].axhline(0, color='r', ls='--')
axes[1,1].set_xlabel('Pred RFR')
axes[1,1].set_ylabel('Residuals')
axes[1,1].set_title('RFR Residuals')

plt.tight_layout()
plt.savefig('report/images/model_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print('Models trained, plots saved')