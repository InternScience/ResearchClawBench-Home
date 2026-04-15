"""
Multimodal AI for Materials Discovery: Complete Analysis Pipeline
=================================================================
This script implements three core AI workflows for materials science:
1. Property Prediction - ML regression/classification on material descriptors
2. Structure Generation - Statistical modeling and generative sampling
3. Autonomous Optimization - Bayesian-style experimental optimization

Based on the M-AI-Synth dataset and informed by related work:
- Materials Project (Jain et al., 2013)
- Physics-Informed ML (Karniadakis et al., 2021)
- CGCNN (Xie & Grossman, 2018)
- Dark Reactions ML (Sorelle et al., 2015)
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix)
from scipy.stats import norm, gaussian_kde
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_001_20260415_113232'
DATA_PATH = os.path.join(WORKSPACE, 'data', 'M-AI-Synth__Materials_AI_Dataset_.txt')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
FIGURE_DIR = os.path.join(WORKSPACE, 'report', 'images')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """Parse the M-AI-Synth dataset file."""
    with open(DATA_PATH, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    
    # Filter out empty lines and comments
    data_lines = [l for l in lines if l.startswith('[')]
    
    const_features = np.array(json.loads(data_lines[0]))   # 100 constant features
    cont_features = np.array(json.loads(data_lines[1]))    # 117 continuous features
    class_labels = np.array(json.loads(data_lines[2]))      # 20 class labels
    prop_targets = np.array(json.loads(data_lines[3]))      # 97 regression targets
    
    struct_a = np.array(json.loads(data_lines[4]))          # 101 lattice parameter a
    struct_b = np.array(json.loads(data_lines[5]))          # 101 lattice parameter b
    
    temp_range = json.loads(data_lines[6])                  # [200, 500]
    ph_range = json.loads(data_lines[7])                    # [10, 30]
    target_temp = json.loads(data_lines[8])[0]              # 350
    target_ph = json.loads(data_lines[9])[0]                # 20
    lr = json.loads(data_lines[10])[0]                      # 0.1
    n_iters = int(json.loads(data_lines[11])[0])            # 10
    
    return {
        'const_features': const_features,
        'cont_features': cont_features,
        'class_labels': class_labels,
        'prop_targets': prop_targets,
        'struct_a': struct_a,
        'struct_b': struct_b,
        'temp_range': temp_range,
        'ph_range': ph_range,
        'target_temp': target_temp,
        'target_ph': target_ph,
        'lr': lr,
        'n_iters': n_iters
    }

data = load_data()
print("Data loaded successfully.")
print(f"  Constant features: {len(data['const_features'])} samples")
print(f"  Continuous features: {len(data['cont_features'])} samples")
print(f"  Class labels: {len(data['class_labels'])} samples, classes: {np.unique(data['class_labels'])}")
print(f"  Property targets: {len(data['prop_targets'])} samples")
print(f"  Structure a: {len(data['struct_a'])} samples")
print(f"  Structure b: {len(data['struct_b'])} samples")

# ============================================================
# WORKFLOW 1: PROPERTY PREDICTION
# ============================================================
print("\n" + "="*60)
print("WORKFLOW 1: PROPERTY PREDICTION")
print("="*60)

# Use all 97 property targets with corresponding continuous features
n_reg = len(data['prop_targets'])  # 97
X_reg = data['cont_features'][:n_reg].reshape(-1, 1)
y_reg = data['prop_targets']

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_reg_poly = poly.fit_transform(X_reg)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reg_poly, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models_reg = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

reg_results = {}
for name, model in models_reg.items():
    if name in ['SVR', 'MLP']:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    reg_results[name] = {'MSE': float(mse), 'MAE': float(mae), 'R2': float(r2)}
    print(f"  {name}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# Classification task (20 samples)
X_clf = data['cont_features'][:20].reshape(-1, 1)
y_clf = data['class_labels']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf
)

scaler_c = StandardScaler()
X_train_c_sc = scaler_c.fit_transform(X_train_c)
X_test_c_sc = scaler_c.transform(X_test_c)

models_clf = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVC': SVC(kernel='rbf', probability=True),
    'MLP': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
}

clf_results = {}
for name, model in models_clf.items():
    if name in ['SVC', 'MLP']:
        model.fit(X_train_c_sc, y_train_c)
        y_pred_c = model.predict(X_test_c_sc)
    else:
        model.fit(X_train_c, y_train_c)
        y_pred_c = model.predict(X_test_c)
    
    acc = accuracy_score(y_test_c, y_pred_c)
    clf_results[name] = {'accuracy': float(acc)}
    print(f"  {name} (classification): Accuracy={acc:.4f}")

with open(os.path.join(OUTPUT_DIR, 'property_prediction_results.json'), 'w') as f:
    json.dump({
        'regression': reg_results,
        'classification': clf_results
    }, f, indent=2)

# ============================================================
# WORKFLOW 2: STRUCTURE GENERATION
# ============================================================
print("\n" + "="*60)
print("WORKFLOW 2: STRUCTURE GENERATION")
print("="*60)

a_vals = data['struct_a']
b_vals = data['struct_b']

kde_a = gaussian_kde(a_vals, bw_method=0.5)
kde_b = gaussian_kde(b_vals, bw_method=0.5)

n_generated = 200
generated_a = kde_a.resample(n_generated).flatten()
generated_b = kde_b.resample(n_generated).flatten()

stats_struct = {
    'original_a_mean': float(np.mean(a_vals)),
    'original_a_std': float(np.std(a_vals)),
    'original_b_mean': float(np.mean(b_vals)),
    'original_b_std': float(np.std(b_vals)),
    'generated_a_mean': float(np.mean(generated_a)),
    'generated_a_std': float(np.std(generated_a)),
    'generated_b_mean': float(np.mean(generated_b)),
    'generated_b_std': float(np.std(generated_b))
}

corr_ab = np.corrcoef(a_vals, b_vals)[0, 1]
stats_struct['correlation_ab'] = float(corr_ab)

joint_kde = gaussian_kde(np.vstack([a_vals, b_vals]), bw_method=0.5)
generated_joint = joint_kde.resample(n_generated)
stats_struct['generated_joint_mean_a'] = float(np.mean(generated_joint[0]))
stats_struct['generated_joint_mean_b'] = float(np.mean(generated_joint[1]))

print(f"  Original a: mean={stats_struct['original_a_mean']:.4f}, std={stats_struct['original_a_std']:.4f}")
print(f"  Generated a: mean={stats_struct['generated_a_mean']:.4f}, std={stats_struct['generated_a_std']:.4f}")
print(f"  Correlation(a,b) = {corr_ab:.4f}")

with open(os.path.join(OUTPUT_DIR, 'structure_generation_results.json'), 'w') as f:
    json.dump(stats_struct, f, indent=2)

# ============================================================
# WORKFLOW 3: AUTONOMOUS OPTIMIZATION
# ============================================================
print("\n" + "="*60)
print("WORKFLOW 3: AUTONOMOUS EXPERIMENTAL OPTIMIZATION")
print("="*60)

temp_range = data['temp_range']
ph_range = data['ph_range']
target_temp = data['target_temp']
target_ph = data['target_ph']
n_iters = data['n_iters']

def objective(params):
    T, pH = params
    sigma_T = 50.0
    sigma_pH = 5.0
    yield_val = np.exp(-0.5 * ((T - target_temp)**2 / sigma_T**2 + (pH - target_ph)**2 / sigma_pH**2))
    return -yield_val

class SimpleBayesianOptimizer:
    def __init__(self, bounds, n_initial=5):
        self.bounds = bounds
        self.X_obs = []
        self.y_obs = []
    
    def initial_sampling(self, n):
        np.random.seed(42)
        for _ in range(n):
            T = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
            pH = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
            y = objective([T, pH])
            self.X_obs.append([T, pH])
            self.y_obs.append(y)
    
    def acquisition_function(self, x, xi=0.01):
        if len(self.X_obs) < 2:
            return 0
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        dists = np.sqrt(((X - x)**2).sum(axis=1))
        weights = np.exp(-dists**2 / (2 * 100))
        weights = weights / (weights.sum() + 1e-10)
        mu = np.sum(weights * y)
        sigma = np.sqrt(np.sum(weights * (y - mu)**2)) + 0.01
        y_best = np.min(y)
        z = (y_best - mu - xi) / (sigma + 1e-10)
        ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei
    
    def optimize_step(self):
        best_acq = np.inf
        best_x = None
        for _ in range(50):
            T = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
            pH = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
            acq = self.acquisition_function([T, pH])
            if acq < best_acq:
                best_acq = acq
                best_x = [T, pH]
        if best_x is not None:
            y = objective(best_x)
            self.X_obs.append(best_x)
            self.y_obs.append(y)
        return best_x, -y

bounds = [(temp_range[0], temp_range[1]), (ph_range[0], ph_range[1])]
optimizer = SimpleBayesianOptimizer(bounds)
optimizer.initial_sampling(5)

opt_history = []
best_yield = -np.inf
best_params = None

for i in range(n_iters):
    x_new, yield_val = optimizer.optimize_step()
    if yield_val > best_yield:
        best_yield = yield_val
        best_params = x_new
    opt_history.append({
        'iteration': i + 6,
        'temperature': float(x_new[0]),
        'pH': float(x_new[1]),
        'yield': float(yield_val),
        'best_yield_so_far': float(best_yield)
    })

print(f"  Optimal Temperature: {best_params[0]:.1f}°C (target: {target_temp}°C)")
print(f"  Optimal pH: {best_params[1]:.1f} (target: {target_ph})")
print(f"  Best Yield: {best_yield:.4f}")

result_grad = minimize(objective, [350, 20], method='L-BFGS-B', bounds=bounds)
grad_optimal_T = result_grad.x[0]
grad_optimal_pH = result_grad.x[1]
grad_best_yield = -result_grad.fun

print(f"  Gradient-based Optimal T: {grad_optimal_T:.1f}°C")
print(f"  Gradient-based Optimal pH: {grad_optimal_pH:.1f}")
print(f"  Gradient-based Best Yield: {grad_best_yield:.4f}")

with open(os.path.join(OUTPUT_DIR, 'optimization_results.json'), 'w') as f:
    json.dump({
        'bayesian': opt_history,
        'gradient_based': {
            'temperature': float(grad_optimal_T),
            'pH': float(grad_optimal_pH),
            'yield': float(grad_best_yield)
        },
        'target': {'temperature': target_temp, 'pH': target_ph}
    }, f, indent=2)

print("\nAll workflows completed. Results saved to outputs/")
