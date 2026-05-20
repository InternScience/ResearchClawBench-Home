#!/usr/bin/env python3
"""
Neuron Segment Merge Prediction Model
Trains a classifier on simulated EM neuron segment features to predict merge (same neuron).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Ensure output dirs
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

feature_cols = [str(i) for i in range(20)]
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
X_test = test_df[feature_cols].values
y_test = test_df['label'].values

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model (Random Forest for interpretability)
print("Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # handle imbalance
)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'outputs/model_rf.joblib')
joblib.dump(scaler, 'outputs/scaler.joblib')

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1: {f1:.4f}")
print(f"Test ROC-AUC: {auc:.4f}")

# Save metrics
metrics = {
    'accuracy': acc, 'precision': prec, 'recall': rec,
    'f1': f1, 'roc_auc': auc
}
pd.DataFrame([metrics]).to_csv('outputs/metrics.csv', index=False)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Merge', 'Merge'], yticklabels=['No Merge', 'Merge'])
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('report/images/confusion_matrix.png', dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.3f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('report/images/roc_curve.png', dpi=150)
plt.close()

# Feature Importance
importances = model.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('report/images/feature_importance.png', dpi=150)
plt.close()

# Performance by Degradation Type
test_df['pred'] = y_pred
test_df['correct'] = (test_df['pred'] == test_df['label']).astype(int)
degrad_perf = test_df.groupby('degradation')['correct'].mean().reset_index()
degrad_perf.columns = ['degradation', 'accuracy']
plt.figure(figsize=(7,4))
sns.barplot(data=degrad_perf, x='degradation', y='accuracy', palette='Set2')
plt.title('Accuracy by Degradation Type')
plt.ylim(0,1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('report/images/degradation_accuracy.png', dpi=150)
plt.close()

print("All figures and metrics saved successfully.")