import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
df = pd.read_excel('data/184_verified_Original Data_ML_20230926.xlsx')

# Features are the monomer compositions
features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
# Target is adhesive strength
target = 'Glass (kPa)_10s'

X = df[features].values
y = df[target].values

# 1. Model Training & Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_r2_scores = []
gp_r2_scores = []
rf_preds = np.zeros_like(y)
gp_preds = np.zeros_like(y)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2_scores.append(r2_score(y_test, rf_pred))
    rf_preds[test_idx] = rf_pred
    
    # Gaussian Process
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
    gp.fit(X_train, y_train)
    gp_pred = gp.predict(X_test)
    gp_r2_scores.append(r2_score(y_test, gp_pred))
    gp_preds[test_idx] = gp_pred

print(f"Random Forest CV R2: {np.mean(rf_r2_scores):.3f} +/- {np.std(rf_r2_scores):.3f}")
print(f"Gaussian Process CV R2: {np.mean(gp_r2_scores):.3f} +/- {np.std(gp_r2_scores):.3f}")

# Plot Correlation
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y, rf_preds, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Experimental Adhesive Strength (kPa)')
plt.ylabel('Predicted Adhesive Strength (kPa)')
plt.title(f'Random Forest (R2 = {r2_score(y, rf_preds):.3f})')

plt.subplot(1, 2, 2)
plt.scatter(y, gp_preds, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Experimental Adhesive Strength (kPa)')
plt.ylabel('Predicted Adhesive Strength (kPa)')
plt.title(f'Gaussian Process (R2 = {r2_score(y, gp_preds):.3f})')

plt.tight_layout()
plt.savefig('report/images/model_correlation.png', dpi=300)
plt.close()

# 2. Feature Importance
rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
rf_full.fit(X, y)

importances = rf_full.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig('report/images/feature_importance.png', dpi=300)
plt.close()

# Save full model predictions
df['RF_Pred'] = rf_preds
df['GP_Pred'] = gp_preds
df.to_csv('outputs/initial_predictions.csv', index=False)
