"""
Property Prediction Workflow: Train multiple ML models to predict material properties
from crystal structure features.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load processed data
data = np.load('outputs/processed_data.npz', allow_pickle=True)
lattice_dim = data['lattice_dim']
x_coords = data['x_coords']
atom_types = data['atom_types']
targets = data['targets']

# Feature engineering
# Create a richer feature set from the raw data
X = pd.DataFrame({
    'lattice_dim': lattice_dim,
    'x_coord': x_coords,
    'atom_type': atom_types,
})

# Add derived features
X['x_coord_sq'] = X['x_coord'] ** 2
X['x_coord_abs'] = np.abs(X['x_coord'])
X['sin_x'] = np.sin(X['x_coord'])
X['cos_x'] = np.cos(X['x_coord'])
X['atom_type_sq'] = X['atom_type'] ** 2

# One-hot encode atom type
for at in sorted(X['atom_type'].unique()):
    X[f'atom_{at}'] = (X['atom_type'] == at).astype(int)

# Interaction features
X['lattice_x'] = X['lattice_dim'] * X['x_coord']
X['lattice_atom'] = X['lattice_dim'] * X['atom_type']

y = targets

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
    'MLP (NN)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, early_stopping=True),
}

results = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ['Ridge', 'Lasso', 'MLP (NN)']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Cross-validation on scaled data
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='neg_mean_squared_error')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_RMSE': cv_rmse,
        'y_pred': y_pred,
    }
    print(f"  Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, CV_RMSE: {cv_rmse:.4f}")

# Save results
with open('outputs/property_prediction_results.json', 'w') as f:
    json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else 
                   (vv.tolist() if isinstance(vv, np.ndarray) else vv) 
                   for kk, vv in v.items()} 
               for k, v in results.items()}, f, indent=2)

# Figure 1: Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

model_names = list(results.keys())
rmse_vals = [results[m]['RMSE'] for m in model_names]
mae_vals = [results[m]['MAE'] for m in model_names]
r2_vals = [results[m]['R2'] for m in model_names]

axes[0].bar(model_names, rmse_vals, color='steelblue', edgecolor='black')
axes[0].set_ylabel('RMSE')
axes[0].set_title('Test RMSE by Model')
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(model_names, mae_vals, color='coral', edgecolor='black')
axes[1].set_ylabel('MAE')
axes[1].set_title('Test MAE by Model')
axes[1].tick_params(axis='x', rotation=30)

axes[2].bar(model_names, r2_vals, color='seagreen', edgecolor='black')
axes[2].set_ylabel('R² Score')
axes[2].set_title('Test R² by Model')
axes[2].tick_params(axis='x', rotation=30)
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('report/images/figure_property_prediction_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# Figure 2: Parity plots for best model
best_model = min(results, key=lambda x: results[x]['RMSE'])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Parity plot for best model
y_pred_best = results[best_model]['y_pred']
axes[0].scatter(y_test, y_pred_best, c='steelblue', edgecolors='k', alpha=0.7)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Target')
axes[0].set_ylabel('Predicted Target')
axes[0].set_title(f'Parity Plot: {best_model}\nRMSE={results[best_model]["RMSE"]:.4f}, R²={results[best_model]["R2"]:.4f}')
axes[0].legend()

# Residual plot
residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, c='coral', edgecolors='k', alpha=0.7)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Target')
axes[1].set_ylabel('Residuals')
axes[1].set_title(f'Residual Plot: {best_model}')

plt.tight_layout()
plt.savefig('report/images/figure_property_prediction_parity.png', dpi=200, bbox_inches='tight')
plt.close()

# Figure 3: Feature importance from Random Forest
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df['feature'], imp_df['importance'], color='darkgreen', edgecolor='black')
ax.set_xlabel('Feature Importance')
ax.set_title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('report/images/figure_feature_importance.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nProperty prediction complete.")
print(f"Best model: {best_model} with RMSE={results[best_model]['RMSE']:.4f}")
