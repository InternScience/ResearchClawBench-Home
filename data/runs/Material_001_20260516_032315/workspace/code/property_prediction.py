#!/usr/bin/env python3
"""
Property Prediction Workflow
Train ML models to predict material properties from structural/compositional features.
Uses data from File 1 of the M-AI-Synth dataset.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Parse data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'M-AI-Synth__Materials_AI_Dataset_.txt')
with open(data_path, 'r') as f:
    content = f.read()

# Extract arrays from File 1
sections = content.split('# 文件')
file1_content = sections[1]  # property_prediction.py

# Parse arrays
lines = [l.strip() for l in file1_content.split('\n') if l.strip().startswith('[')]
atomic_nums = np.array(json.loads(lines[0]))  # 100 values, all 5 - atomic number proxy
features_raw = np.array(json.loads(lines[1]))  # 117 values - structural features
pairs = np.array(json.loads(lines[2]))          # 20 pairs
targets = np.array(json.loads(lines[3]))        # 97 values - target property

# Align data: take the first 97 from features and atomic_nums to match targets
n_samples = len(targets)  # 97
atomic_nums = atomic_nums[:n_samples]
features = features_raw[:n_samples]

print(f"Atomic nums: shape={atomic_nums.shape}, unique={np.unique(atomic_nums)}")
print(f"Features: shape={features.shape}, range=[{features.min():.4f}, {features.max():.4f}]")
print(f"Targets: shape={targets.shape}, range=[{targets.min():.4f}, {targets.max():.4f}]")

# Create feature matrix
X = features.reshape(-1, 1)
y = targets

# Add polynomial and interaction features
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
poly = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Split data
from sklearn.model_selection import train_test_split, cross_val_score, KFold
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

# 1. Linear Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
results['Linear Regression'] = {
    'MAE': float(mean_absolute_error(y_test, y_pred_lr)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_lr))),
    'R2': float(r2_score(y_test, y_pred_lr))
}

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
results['Ridge Regression'] = {
    'MAE': float(mean_absolute_error(y_test, y_pred_ridge)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_ridge))),
    'R2': float(r2_score(y_test, y_pred_ridge))
}

# Lasso Regression
lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
results['Lasso Regression'] = {
    'MAE': float(mean_absolute_error(y_test, y_pred_lasso)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_lasso))),
    'R2': float(r2_score(y_test, y_pred_lasso))
}

# 2. Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'MAE': float(mean_absolute_error(y_test, y_pred_rf)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
    'R2': float(r2_score(y_test, y_pred_rf))
}

# 3. Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
results['Gradient Boosting'] = {
    'MAE': float(mean_absolute_error(y_test, y_pred_gbr)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_gbr))),
    'R2': float(r2_score(y_test, y_pred_gbr))
}

# 4. SVR
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=10.0, gamma='scale')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
results['SVR (RBF)'] = {
    'MAE': float(mean_absolute_error(y_test, y_pred_svr)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_svr))),
    'R2': float(r2_score(y_test, y_pred_svr))
}

# 5. Simple Neural Network
try:
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', 
                       max_iter=2000, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    results['Neural Network (MLP)'] = {
        'MAE': float(mean_absolute_error(y_test, y_pred_mlp)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_mlp))),
        'R2': float(r2_score(y_test, y_pred_mlp))
    }
except Exception as e:
    print(f"MLP failed: {e}")

# Cross-validation for top models
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, model in [('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                      ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
                      ('Ridge', Ridge(alpha=1.0)),
                      ('SVR', SVR(kernel='rbf', C=10.0, gamma='scale'))]:
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')
    cv_results[name] = {
        'CV_MAE_mean': float(-cv_scores.mean()),
        'CV_MAE_std': float(cv_scores.std())
    }

# Save all results
output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Save results
all_results = {
    'test_set_results': results,
    'cross_validation': cv_results,
    'data_summary': {
        'n_samples': int(len(X)),
        'feature_range': [float(X.min()), float(X.max())],
        'target_range': [float(y.min()), float(y.max())],
        'target_mean': float(y.mean()),
        'target_std': float(y.std())
    },
    'predictions': {
        'y_test': y_test.tolist(),
        'y_pred_best': y_pred_gbr.tolist(),
        'model': 'Gradient Boosting'
    }
}

with open(os.path.join(output_dir, 'property_prediction_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n=== Property Prediction Results ===")
for model, metrics in results.items():
    print(f"\n{model}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

print("\n=== Cross-Validation Results ===")
for model, metrics in cv_results.items():
    print(f"\n{model}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

print("\nResults saved to outputs/property_prediction_results.json")
