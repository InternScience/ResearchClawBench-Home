"""
Train more models one by one
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                              accuracy_score, average_precision_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import json, os
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

feature_cols = [str(i) for i in range(20)]
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
X_test = test_df[feature_cols].values
y_test = test_df['label'].values
deg_test = test_df['degradation'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

all_results = {}

# Load LR results
with open('outputs/lr_results.json') as f:
    lr_res = json.load(f)
all_results['Logistic Regression'] = {k: v for k, v in lr_res.items() if k not in ['y_prob', 'y_pred']}

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, 
                           class_weight='balanced', n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
y_pred_rf = rf.predict(X_test_scaled)
all_results['Random Forest'] = {
    'auc': float(roc_auc_score(y_test, y_prob_rf)),
    'f1': float(f1_score(y_test, y_pred_rf)),
    'precision': float(precision_score(y_test, y_pred_rf)),
    'recall': float(recall_score(y_test, y_pred_rf)),
    'ap': float(average_precision_score(y_test, y_prob_rf)),
    'accuracy': float(accuracy_score(y_test, y_pred_rf))
}
np.save('outputs/y_prob_rf.npy', y_prob_rf)
print(f"  AUC: {all_results['Random Forest']['auc']:.4f}, F1: {all_results['Random Forest']['f1']:.4f}")

# Gradient Boosting
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
y_prob_gb = gb.predict_proba(X_test_scaled)[:, 1]
y_pred_gb = gb.predict(X_test_scaled)
all_results['Gradient Boosting'] = {
    'auc': float(roc_auc_score(y_test, y_prob_gb)),
    'f1': float(f1_score(y_test, y_pred_gb)),
    'precision': float(precision_score(y_test, y_pred_gb)),
    'recall': float(recall_score(y_test, y_pred_gb)),
    'ap': float(average_precision_score(y_test, y_prob_gb)),
    'accuracy': float(accuracy_score(y_test, y_pred_gb))
}
np.save('outputs/y_prob_gb.npy', y_prob_gb)
print(f"  AUC: {all_results['Gradient Boosting']['auc']:.4f}, F1: {all_results['Gradient Boosting']['f1']:.4f}")

# AdaBoost
print("Training AdaBoost...")
ab = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
ab.fit(X_train_scaled, y_train)
y_prob_ab = ab.predict_proba(X_test_scaled)[:, 1]
y_pred_ab = ab.predict(X_test_scaled)
all_results['AdaBoost'] = {
    'auc': float(roc_auc_score(y_test, y_prob_ab)),
    'f1': float(f1_score(y_test, y_pred_ab)),
    'precision': float(precision_score(y_test, y_pred_ab)),
    'recall': float(recall_score(y_test, y_pred_ab)),
    'ap': float(average_precision_score(y_test, y_prob_ab)),
    'accuracy': float(accuracy_score(y_test, y_pred_ab))
}
np.save('outputs/y_prob_ab.npy', y_prob_ab)
print(f"  AUC: {all_results['AdaBoost']['auc']:.4f}, F1: {all_results['AdaBoost']['f1']:.4f}")

# MLP
print("Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
mlp.fit(X_train_scaled, y_train)
y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
y_pred_mlp = mlp.predict(X_test_scaled)
all_results['MLP'] = {
    'auc': float(roc_auc_score(y_test, y_prob_mlp)),
    'f1': float(f1_score(y_test, y_pred_mlp)),
    'precision': float(precision_score(y_test, y_pred_mlp)),
    'recall': float(recall_score(y_test, y_pred_mlp)),
    'ap': float(average_precision_score(y_test, y_prob_mlp)),
    'accuracy': float(accuracy_score(y_test, y_pred_mlp))
}
np.save('outputs/y_prob_mlp.npy', y_prob_mlp)
print(f"  AUC: {all_results['MLP']['auc']:.4f}, F1: {all_results['MLP']['f1']:.4f}")

# Save all results
best_name = max(all_results.keys(), key=lambda x: all_results[x]['auc'])
all_results['best_model'] = best_name

with open('outputs/all_model_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Degradation-specific for best model
print(f"\nBest model: {best_name}")
degradations = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']

# Use RF probability for best
y_prob_best = y_prob_rf if best_name == 'Random Forest' else y_prob_gb if best_name == 'Gradient Boosting' else y_prob_mlp

deg_results = {}
for deg in degradations:
    mask = deg_test == deg
    y_prob_d = y_prob_best[mask]
    y_d = y_test[mask]
    y_pred_d = (y_prob_d > 0.5).astype(int)
    deg_results[deg] = {
        'auc': float(roc_auc_score(y_d, y_prob_d)),
        'f1': float(f1_score(y_d, y_pred_d)),
        'ap': float(average_precision_score(y_d, y_prob_d)),
        'precision': float(precision_score(y_d, y_pred_d)),
        'recall': float(recall_score(y_d, y_pred_d)),
    }
    print(f"  {deg}: AUC={deg_results[deg]['auc']:.4f}")

with open('outputs/deg_results.json', 'w') as f:
    json.dump(deg_results, f, indent=2)

# Save feature importances
rf_imp = rf.feature_importances_.tolist()
gb_imp = gb.feature_importances_.tolist()
with open('outputs/feature_importances.json', 'w') as f:
    json.dump({'rf': rf_imp, 'gb': gb_imp}, f, indent=2)

print("\n✓ All models trained and saved")
