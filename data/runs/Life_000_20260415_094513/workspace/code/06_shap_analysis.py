"""
Phase 6: SHAP Analysis for Interpretability
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
import shap
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load training data
df = pd.read_csv('outputs/training_data_184.csv')
monomer_cols = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
target_col = 'Glass (kPa)_10s'
short_names = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']

X = df[monomer_cols].values
y = df[target_col].values

# Train RFR
rfr = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rfr.fit(X, y)

# ============================================================
# SHAP Analysis
# ============================================================
print("Computing SHAP values...")
explainer = shap.TreeExplainer(rfr)
shap_values = explainer.shap_values(X)

# ============================================================
# Figure 15: SHAP summary plot
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=short_names, show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig('report/images/fig15_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 15 saved.")

# ============================================================
# Figure 16: SHAP bar plot (mean absolute SHAP values)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=short_names, plot_type='bar', show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig('report/images/fig16_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 16 saved.")

# ============================================================
# Figure 17: SHAP dependence plots for top features
# ============================================================
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features = np.argsort(mean_abs_shap)[::-1][:3]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feat_idx in enumerate(top_features):
    shap.dependence_plot(feat_idx, shap_values, X, feature_names=short_names, ax=axes[i], show=False)
plt.tight_layout()
plt.savefig('report/images/fig17_shap_dependence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 17 saved.")

# Save SHAP importance
shap_importance = {name: float(val) for name, val in zip(short_names, mean_abs_shap)}
with open('outputs/shap_importance.json', 'w') as f:
    json.dump(shap_importance, f, indent=2)

print("\nSHAP importance:", shap_importance)
print("\nPhase 6 complete.")
