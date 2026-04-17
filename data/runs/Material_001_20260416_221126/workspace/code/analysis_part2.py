"""Part 2: Property Prediction with Multiple ML Models"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_001_20260416_221126"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report", "images")

# Load processed data
data = np.load(os.path.join(OUTPUT_DIR, "processed_data.npz"))
X = data['sample_features']
y = data['target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")

# Import sklearn
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'Support Vector Regression': SVR(kernel='rbf', C=10, epsilon=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1),
    'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "=" * 70)
print("Property Prediction: Model Comparison (5-Fold CV)")
print("=" * 70)
print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print("-" * 70)

predictions = {}
for name, model in models.items():
    y_pred = cross_val_predict(model, X_scaled, y, cv=kf)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    predictions[name] = y_pred
    print(f"{name:<30} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

# Save results
with open(os.path.join(OUTPUT_DIR, "property_prediction_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

# Find best model
best_model_name = max(results, key=lambda k: results[k]['R2'])
print(f"\nBest model: {best_model_name} (R2 = {results[best_model_name]['R2']:.4f})")

# ============================================================
# FIGURE 3: Model Comparison Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Property Prediction: Model Comparison', fontsize=14, fontweight='bold')

model_names = list(results.keys())
short_names = ['LR', 'Ridge', 'Lasso', 'KNN', 'SVR', 'RF', 'GBR', 'MLP']
colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

# MAE
maes = [results[m]['MAE'] for m in model_names]
bars = axes[0].bar(short_names, maes, color=colors, edgecolor='black')
axes[0].set_ylabel('MAE (eV)')
axes[0].set_title('(a) Mean Absolute Error')
axes[0].tick_params(axis='x', rotation=45)
for bar, val in zip(bars, maes):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# RMSE
rmses = [results[m]['RMSE'] for m in model_names]
bars = axes[1].bar(short_names, rmses, color=colors, edgecolor='black')
axes[1].set_ylabel('RMSE (eV)')
axes[1].set_title('(b) Root Mean Squared Error')
axes[1].tick_params(axis='x', rotation=45)
for bar, val in zip(bars, rmses):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# R2
r2s = [results[m]['R2'] for m in model_names]
bars = axes[2].bar(short_names, r2s, color=colors, edgecolor='black')
axes[2].set_ylabel('R2 Score')
axes[2].set_title('(c) Coefficient of Determination')
axes[2].tick_params(axis='x', rotation=45)
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, r2s):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 3: Model comparison saved")

# ============================================================
# FIGURE 4: Predicted vs Actual for Best Models
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Predicted vs Actual Band Gap (5-Fold CV)', fontsize=14, fontweight='bold')

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx // 4, idx % 4]
    ax.scatter(y, y_pred, alpha=0.6, s=30, c=colors[idx], edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect')
    
    ax.set_xlabel('Actual (eV)')
    ax.set_ylabel('Predicted (eV)')
    r2 = results[name]['R2']
    ax.set_title(f'{short_names[idx]}\nR2={r2:.3f}', fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "predicted_vs_actual.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4: Predicted vs actual saved")

# ============================================================
# FIGURE 5: Feature Importance (Random Forest)
# ============================================================
# Train full RF model for feature importance
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
rf.fit(X_scaled, y)
importances = rf.feature_importances_
feature_names = ['Crystal Feature', 'Neighbor Feature', 'Periodic Feature', 'Edge Attribute', 'Atomic Number']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')

# Bar chart
sorted_idx = np.argsort(importances)[::-1]
axes[0].barh(range(len(importances)), importances[sorted_idx], color='steelblue', edgecolor='black')
axes[0].set_yticks(range(len(importances)))
axes[0].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[0].set_xlabel('Importance')
axes[0].set_title('(a) Random Forest Feature Importance')
axes[0].invert_yaxis()

# Permutation importance
from sklearn.inspection import permutation_importance
perm_result = permutation_importance(rf, X_scaled, y, n_repeats=30, random_state=42)
perm_imp = perm_result.importances_mean

sorted_idx2 = np.argsort(perm_imp)[::-1]
axes[1].barh(range(len(perm_imp)), perm_imp[sorted_idx2], color='coral', edgecolor='black')
axes[1].set_yticks(range(len(perm_imp)))
axes[1].set_yticklabels([feature_names[i] for i in sorted_idx2])
axes[1].set_xlabel('Mean Decrease in R2')
axes[1].set_title('(b) Permutation Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "feature_importance.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5: Feature importance saved")

# Save feature importance
fi_data = {
    "rf_importance": {feature_names[i]: float(importances[i]) for i in range(len(feature_names))},
    "permutation_importance": {feature_names[i]: float(perm_imp[i]) for i in range(len(feature_names))}
}
with open(os.path.join(OUTPUT_DIR, "feature_importance.json"), 'w') as f:
    json.dump(fi_data, f, indent=2)

print("\nPart 2 complete!")
