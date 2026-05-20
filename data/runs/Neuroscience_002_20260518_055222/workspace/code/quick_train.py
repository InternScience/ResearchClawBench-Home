"""
Quick model training and evaluation
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                              accuracy_score, average_precision_score, confusion_matrix,
                              roc_curve, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
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

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = lr.predict(X_test_scaled)

print(f"  AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"  F1: {f1_score(y_test, y_pred_lr):.4f}")

# Save results
results = {
    'y_prob': y_prob_lr.tolist(),
    'y_pred': y_pred_lr.tolist(),
    'auc': float(roc_auc_score(y_test, y_prob_lr)),
    'f1': float(f1_score(y_test, y_pred_lr)),
    'precision': float(precision_score(y_test, y_pred_lr)),
    'recall': float(recall_score(y_test, y_pred_lr)),
    'ap': float(average_precision_score(y_test, y_prob_lr)),
}

with open('outputs/lr_results.json', 'w') as f:
    json.dump(results, f)

# Degradation-specific
degradations = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']
deg_results = {}
for deg in degradations:
    mask = deg_test == deg
    y_prob_d = y_prob_lr[mask]
    y_d = y_test[mask]
    y_pred_d = y_pred_lr[mask]
    deg_results[deg] = {
        'auc': float(roc_auc_score(y_d, y_prob_d)),
        'f1': float(f1_score(y_d, y_pred_d)),
        'ap': float(average_precision_score(y_d, y_prob_d)),
        'precision': float(precision_score(y_d, y_pred_d)),
        'recall': float(recall_score(y_d, y_pred_d)),
    }
    print(f"  {deg}: AUC={deg_results[deg]['auc']:.4f}")

with open('outputs/lr_deg_results.json', 'w') as f:
    json.dump(deg_results, f, indent=2)

print("\n✓ Done")
