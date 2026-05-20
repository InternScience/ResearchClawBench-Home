"""
Train models on subsampled data for speed
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                              accuracy_score, average_precision_score, confusion_matrix,
                              roc_curve, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
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

# Subsample for faster training
np.random.seed(42)
n_sub = 30000
idx = np.random.choice(len(X_train), n_sub, replace=False)
X_sub = X_train[idx]
y_sub = y_train[idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_sub)
X_test_scaled = scaler.transform(X_test)

all_results = {}

# 1. Logistic Regression
print("1. Logistic Regression...")
lr = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_sub)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = lr.predict(X_test_scaled)
all_results['Logistic Regression'] = {
    'auc': float(roc_auc_score(y_test, y_prob_lr)),
    'f1': float(f1_score(y_test, y_pred_lr)),
    'precision': float(precision_score(y_test, y_pred_lr)),
    'recall': float(recall_score(y_test, y_pred_lr)),
    'ap': float(average_precision_score(y_test, y_prob_lr)),
    'accuracy': float(accuracy_score(y_test, y_pred_lr))
}
np.save('outputs/y_prob_lr.npy', y_prob_lr)
print(f"  AUC: {all_results['Logistic Regression']['auc']:.4f}")

# 2. Random Forest
print("2. Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, 
                           class_weight='balanced', n_jobs=-1)
rf.fit(X_train_scaled, y_sub)
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
np.save('outputs/y_pred_rf.npy', y_pred_rf)
print(f"  AUC: {all_results['Random Forest']['auc']:.4f}")

# 3. Gradient Boosting
print("3. Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_sub)
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
np.save('outputs/y_pred_gb.npy', y_pred_gb)
print(f"  AUC: {all_results['Gradient Boosting']['auc']:.4f}")

# 4. AdaBoost
print("4. AdaBoost...")
ab = AdaBoostClassifier(n_estimators=80, learning_rate=0.1, random_state=42)
ab.fit(X_train_scaled, y_sub)
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
np.save('outputs/y_pred_ab.npy', y_pred_ab)
print(f"  AUC: {all_results['AdaBoost']['auc']:.4f}")

# 5. MLP
print("5. MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
mlp.fit(X_train_scaled, y_sub)
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
np.save('outputs/y_pred_mlp.npy', y_pred_mlp)
print(f"  AUC: {all_results['MLP']['auc']:.4f}")

# Find best
best_name = max(all_results.keys(), key=lambda x: all_results[x]['auc'])
print(f"\nBest model: {best_name} (AUC={all_results[best_name]['auc']:.4f})")

# Degradation-specific for all models
degradations = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']
y_prob_map = {'Logistic Regression': y_prob_lr, 'Random Forest': y_prob_rf, 
              'Gradient Boosting': y_prob_gb, 'AdaBoost': y_prob_ab, 'MLP': y_prob_mlp}

deg_results = {}
for deg in degradations:
    mask = deg_test == deg
    deg_results[deg] = {}
    for name in all_results:
        y_prob_d = y_prob_map[name][mask]
        y_d = y_test[mask]
        y_pred_d = (y_prob_d > 0.5).astype(int)
        deg_results[deg][name] = {
            'auc': float(roc_auc_score(y_d, y_prob_d)),
            'f1': float(f1_score(y_d, y_pred_d))
        }
    print(f"  {deg}: {', '.join(f'{n}={deg_results[deg][n]['auc']:.3f}' for n in all_results)}")

# Feature importance
rf_imp = rf.feature_importances_.tolist()
gb_imp = gb.feature_importances_.tolist()

# Cross validation on smaller data
print("\nCross-validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, model in [('Logistic Regression', lr), ('Random Forest', rf), ('Gradient Boosting', gb)]:
    fold_aucs = []
    for train_idx, val_idx in cv.split(X_train_scaled, y_sub):
        m = type(model)(**model.get_params())
        m.fit(X_train_scaled[train_idx], y_sub[train_idx])
        y_prob = m.predict_proba(X_train_scaled[val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_sub[val_idx], y_prob))
    cv_results[name] = {'auc_mean': float(np.mean(fold_aucs)), 'auc_std': float(np.std(fold_aucs))}
    print(f"  {name}: CV AUC={np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f}")

# Save everything
all_data = {
    'model_results': all_results,
    'best_model': best_name,
    'degradation_results': deg_results,
    'feature_importances': {'rf': rf_imp, 'gb': gb_imp},
    'cross_validation': cv_results
}

with open('outputs/all_results.json', 'w') as f:
    json.dump(all_data, f, indent=2)

print("\n✓ All models trained and saved")
