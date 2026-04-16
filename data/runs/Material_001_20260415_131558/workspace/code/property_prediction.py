"""
Workflow 1: Property Prediction
- Implements multiple ML models for materials property prediction
- Includes: Random Forest, Gradient Boosting, Neural Network, and a simplified CGCNN-inspired model
- Generates comparison plots and feature importance analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

OUTPUT_DIR = 'report/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
features = np.load('outputs/features.npy')
targets = np.load('outputs/targets.npy')
feature_names = list(np.load('outputs/feature_names.npy', allow_pickle=True))
target_names = list(np.load('outputs/target_names.npy', allow_pickle=True))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(y_true, y_pred, target_name):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ============================================================
# Model 1: Random Forest
# ============================================================
print("Training Random Forest...")
rf_model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ============================================================
# Model 2: Gradient Boosting
# ============================================================
print("Training Gradient Boosting...")
gb_model = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# ============================================================
# Model 3: Neural Network (MLP)
# ============================================================
print("Training Neural Network (MLP)...")
mlp_model = MultiOutputRegressor(
    MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, 
                 learning_rate='adaptive', random_state=42, early_stopping=True)
)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)

# ============================================================
# Model 4: SVR
# ============================================================
print("Training SVR...")
svr_model = MultiOutputRegressor(
    SVR(kernel='rbf', C=10, epsilon=0.1)
)
svr_model.fit(X_train_scaled, y_train)
svr_pred = svr_model.predict(X_test_scaled)

# ============================================================
# Model 5: Simplified CGCNN-inspired Graph Neural Network
# ============================================================
print("Training CGCNN-inspired model...")

# Build a graph-based representation from the dataset
# Use the edge indices and weights from the original dataset
import ast

with open('outputs/parsed_dataset.json', 'r') as f:
    parsed = json.load(f)

edge_indices = parsed['property']['edge_indices']
edge_weights = parsed['property']['edge_weights']

# Create adjacency-weighted features (simulate graph convolution)
n_graph_layers = 3
adj_dim = len(edge_indices) // 2

# Build adjacency matrix from edges
n_nodes = 5  # atom count from dataset
adj_matrix = np.zeros((n_nodes, n_nodes))
for idx in range(0, len(edge_indices), 2):
    i, j = edge_indices[idx], edge_indices[idx+1]
    w = edge_weights[idx // 2] if idx // 2 < len(edge_weights) else 1.0
    adj_matrix[i][j] = w
    adj_matrix[j][i] = w

# Normalize adjacency
row_sum = adj_matrix.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1
adj_norm = adj_matrix / row_sum

# Apply graph convolution to composition features
def graph_conv_features(X, adj, layers=3):
    """Apply simplified graph convolution to create graph-enhanced features."""
    comp_features = X[:, :8]  # Composition features
    n_samples = X.shape[0]
    node_features = comp_features.reshape(n_samples, -1, 1)  # Treat as single-feature nodes
    
    # Reshape for convolution: distribute across 5 nodes
    node_feats = np.zeros((n_samples, n_nodes, comp_features.shape[1] // n_nodes + 1))
    for i in range(n_nodes):
        start = i * (comp_features.shape[1] // n_nodes)
        end = min(start + comp_features.shape[1] // n_nodes, comp_features.shape[1])
        if start < comp_features.shape[1]:
            node_feats[:, i, :end-start] = comp_features[:, start:end]
    
    # Graph convolution layers
    for _ in range(layers):
        # Message passing: aggregate neighbor features
        new_feats = np.zeros_like(node_feats)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj[i, j] > 0:
                    new_feats[:, i, :] += adj[i, j] * node_feats[:, j, :]
        # Update: combine with self
        node_feats = 0.5 * node_feats + 0.5 * np.tanh(new_feats)
    
    # Pool: mean over nodes
    pooled = node_feats.mean(axis=1)
    return pooled

gconv_train = graph_conv_features(X_train, adj_norm)
gconv_test = graph_conv_features(X_test, adj_norm)

# Combine graph features with original features
X_train_gconv = np.hstack([X_train, gconv_train])
X_test_gconv = np.hstack([X_test, gconv_test])

scaler_gconv = StandardScaler()
X_train_gconv_scaled = scaler_gconv.fit_transform(X_train_gconv)
X_test_gconv_scaled = scaler_gconv.transform(X_test_gconv)

gconv_model = MultiOutputRegressor(
    MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000,
                 learning_rate='adaptive', random_state=42, early_stopping=True)
)
gconv_model.fit(X_train_gconv_scaled, y_train)
gconv_pred = gconv_model.predict(X_test_gconv_scaled)

# ============================================================
# Collect all results
# ============================================================
models = {
    'Random Forest': rf_pred,
    'Gradient Boosting': gb_pred,
    'MLP Neural Net': mlp_pred,
    'SVR': svr_pred,
    'CGCNN-inspired': gconv_pred
}

all_metrics = {}
for model_name, pred in models.items():
    all_metrics[model_name] = {}
    for i, tname in enumerate(target_names):
        metrics = evaluate_model(y_test[:, i], pred[:, i], tname)
        all_metrics[model_name][tname] = metrics

# Save metrics
with open('outputs/property_prediction_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

print("\n=== Property Prediction Results ===")
for model_name in all_metrics:
    print(f"\n{model_name}:")
    for tname in target_names:
        m = all_metrics[model_name][tname]
        print(f"  {tname}: MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}")

# ============================================================
# FIGURE 1: Parity plots for all models and properties
# ============================================================
fig, axes = plt.subplots(5, 4, figsize=(20, 20))
fig.suptitle('Parity Plots: Predicted vs. Actual Material Properties', fontsize=16, fontweight='bold')

model_list = list(models.keys())
for row, model_name in enumerate(model_list):
    pred = models[model_name]
    for col, tname in enumerate(target_names):
        ax = axes[row, col]
        ax.scatter(y_test[:, col], pred[:, col], alpha=0.5, s=20, c=f'C{row}')
        lims = [min(y_test[:, col].min(), pred[:, col].min()),
                max(y_test[:, col].max(), pred[:, col].max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Actual', fontsize=9)
        ax.set_ylabel('Predicted', fontsize=9)
        r2 = all_metrics[model_name][tname]['R2']
        ax.set_title(f'{model_name}\n{tname} (R²={r2:.3f})', fontsize=9)
        ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_parity_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_parity_plots.png")

# ============================================================
# FIGURE 2: Model comparison bar chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Comparison Across Material Properties', fontsize=14, fontweight='bold')

metric_names = ['MAE', 'RMSE', 'R2']
for idx, metric in enumerate(metric_names):
    ax = axes[idx]
    x = np.arange(len(target_names))
    width = 0.15
    for i, model_name in enumerate(model_list):
        values = [all_metrics[model_name][tname][metric] for tname in target_names]
        ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)
    ax.set_xlabel('Property')
    ax.set_ylabel(metric)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([t.replace('_', '\n') for t in target_names], fontsize=8)
    ax.legend(fontsize=7, loc='best')
    ax.set_title(metric)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_model_comparison.png")

# ============================================================
# FIGURE 3: Feature importance (Random Forest)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Importance from Random Forest Model', fontsize=14, fontweight='bold')

for idx, tname in enumerate(target_names):
    ax = axes[idx // 2, idx % 2]
    importances = rf_model.estimators_[idx].feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    ax.barh(range(len(indices)), importances[indices], align='center', color=f'C{idx}')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title(tname.replace('_', ' ').title())
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_feature_importance.png")

# ============================================================
# FIGURE 4: Error distribution
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prediction Error Distributions (Best Model per Property)', fontsize=14, fontweight='bold')

# Find best model per property
best_models = {}
for tname in target_names:
    best_r2 = -999
    best_model = None
    for model_name in model_list:
        r2 = all_metrics[model_name][tname]['R2']
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name
    best_models[tname] = best_model

for idx, tname in enumerate(target_names):
    ax = axes[idx // 2, idx % 2]
    model_name = best_models[tname]
    pred = models[model_name]
    errors = pred[:, idx] - y_test[:, idx]
    ax.hist(errors, bins=30, alpha=0.7, color=f'C{idx}', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Count')
    ax.set_title(f'{tname.replace("_", " ").title()}\n(Best: {model_name}, R²={all_metrics[model_name][tname]["R2"]:.3f})')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_error_distribution.png")

# ============================================================
# FIGURE 5: Cross-validation stability
# ============================================================
print("\nRunning cross-validation...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for model_name, model_class, params, use_scaled in [
    ('RF', RandomForestRegressor, {'n_estimators': 200, 'max_depth': 15, 'random_state': 42}, False),
    ('GB', GradientBoostingRegressor, {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}, False),
    ('MLP', MLPRegressor, {'hidden_layer_sizes': (128, 64, 32), 'max_iter': 1000, 'random_state': 42}, True),
]:
    cv_results[model_name] = {}
    for tidx, tname in enumerate(target_names):
        X_use = X_train_scaled if use_scaled else X_train
        model = model_class(**params)
        scores = cross_val_score(model, X_use, y_train[:, tidx], cv=cv, scoring='r2')
        cv_results[model_name][tname] = {'mean': float(scores.mean()), 'std': float(scores.std())}

with open('outputs/cv_results.json', 'w') as f:
    json.dump(cv_results, f, indent=2)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(target_names))
width = 0.25
for i, model_name in enumerate(cv_results):
    means = [cv_results[model_name][tname]['mean'] for tname in target_names]
    stds = [cv_results[model_name][tname]['std'] for tname in target_names]
    ax.bar(x + i * width, means, width, yerr=stds, label=model_name, alpha=0.8, capsize=3)

ax.set_xlabel('Property')
ax.set_ylabel('R² Score')
ax.set_xticks(x + width)
ax.set_xticklabels([t.replace('_', '\n') for t in target_names], fontsize=9)
ax.legend()
ax.set_title('5-Fold Cross-Validation R² Scores', fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_cross_validation.png")

print("\nProperty prediction workflow complete!")
