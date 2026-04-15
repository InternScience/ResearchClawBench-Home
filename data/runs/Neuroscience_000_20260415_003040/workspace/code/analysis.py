#!/usr/bin/env python3
"""
SimBA-style behavior classification pipeline.
Trains supervised classifiers on pose-derived features to predict Attack and Sniffing.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, confusion_matrix,
    classification_report, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

feat = pd.read_csv('data/Together_1_features_extracted.csv', index_col=0)
targ = pd.read_csv('data/Together_1_targets_inserted.csv', index_col=0)
ref = pd.read_csv('data/Together_1_machine_results_reference.csv', index_col=0)

print(f"Features shape: {feat.shape}")
print(f"Targets shape: {targ.shape}")
print(f"Reference shape: {ref.shape}")

# Use features from targets file (same data, aligned)
X = targ.drop(columns=['Attack', 'Sniffing']).values
feature_names = list(targ.drop(columns=['Attack', 'Sniffing']).columns)
y_attack = targ['Attack'].values
y_sniffing = targ['Sniffing'].values

print(f"\nFeature matrix X: {X.shape}")
print(f"Attack label distribution: {dict(zip(*np.unique(y_attack, return_counts=True)))}")
print(f"Sniffing label distribution: {dict(zip(*np.unique(y_sniffing, return_counts=True)))}")

# Handle NaN
nan_count = np.isnan(X).sum()
print(f"NaN values in features: {nan_count}")
X = np.nan_to_num(X, nan=0.0)

# ============================================================
# 2. TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("2. TRAIN/TEST SPLIT (80/20, stratified)")
print("=" * 60)

X_train, X_test, y_attack_train, y_attack_test, y_sniff_train, y_sniff_test = train_test_split(
    X, y_attack, y_sniffing, test_size=0.2, random_state=42, stratify=y_attack
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# 3. TRAIN CLASSIFIERS
# ============================================================
print("\n" + "=" * 60)
print("3. TRAINING RANDOM FOREST CLASSIFIERS")
print("=" * 60)

# Random Forest (SimBA default)
rf_attack = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_sniffing = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')

rf_attack.fit(X_train, y_attack_train)
rf_sniffing.fit(X_train, y_sniff_train)

print("Attack classifier trained.")
print("Sniffing classifier trained.")

# Also train Gradient Boosting for comparison
gb_attack = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
gb_sniffing = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)

gb_attack.fit(X_train, y_attack_train)
gb_sniffing.fit(X_train, y_sniff_train)

print("Gradient Boosting classifiers trained.")

# ============================================================
# 4. EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("4. EVALUATION")
print("=" * 60)

results = {}

for name, clf, y_test in [
    ('Attack_RF', rf_attack, y_attack_test),
    ('Attack_GB', gb_attack, y_attack_test),
    ('Sniffing_RF', rf_sniffing, y_sniff_test),
    ('Sniffing_GB', gb_sniffing, y_sniff_test),
]:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ap = average_precision_score(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'average_precision': ap, 'auc_roc': auc,
        'confusion_matrix': cm.tolist(), 'y_pred': y_pred, 'y_prob': y_prob
    }
    
    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AP:        {ap:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Confusion Matrix:\n    {cm}")

# Cross-validation
print("\n" + "=" * 60)
print("5. CROSS-VALIDATION (5-fold)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf_class, y_all in [
    ('Attack_RF', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'), y_attack),
    ('Sniffing_RF', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'), y_sniffing),
]:
    scores_f1 = cross_val_score(clf_class, X, y_all, cv=cv, scoring='f1')
    scores_prec = cross_val_score(clf_class, X, y_all, cv=cv, scoring='precision')
    scores_rec = cross_val_score(clf_class, X, y_all, cv=cv, scoring='recall')
    scores_ap = cross_val_score(clf_class, X, y_all, cv=cv, scoring='average_precision')
    
    print(f"\n--- {name} CV ---")
    print(f"  F1:   {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")
    print(f"  Prec: {scores_prec.mean():.4f} ± {scores_prec.std():.4f}")
    print(f"  Rec:  {scores_rec.mean():.4f} ± {scores_rec.std():.4f}")
    print(f"  AP:   {scores_ap.mean():.4f} ± {scores_ap.std():.4f}")
    
    results[f'{name}_cv'] = {
        'f1_mean': scores_f1.mean(), 'f1_std': scores_f1.std(),
        'prec_mean': scores_prec.mean(), 'prec_std': scores_prec.std(),
        'rec_mean': scores_rec.mean(), 'rec_std': scores_rec.std(),
        'ap_mean': scores_ap.mean(), 'ap_std': scores_ap.std(),
    }

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE")
print("=" * 60)

os.makedirs('outputs', exist_ok=True)

for name, clf in [('Attack_RF', rf_attack), ('Sniffing_RF', rf_sniffing)]:
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    print(f"\n--- {name} Top 15 Features ---")
    fi_data = []
    for i, j in enumerate(idx[:15]):
        print(f"  {i+1}. {feature_names[j]}: {importances[j]:.4f}")
        fi_data.append({'rank': i+1, 'feature': feature_names[j], 'importance': importances[j]})
    
    pd.DataFrame(fi_data).to_csv(f'outputs/feature_importance_{name}.csv', index=False)

# Save all importances
all_fi = pd.DataFrame({
    'feature': feature_names,
    'importance_attack': rf_attack.feature_importances_,
    'importance_sniffing': rf_sniffing.feature_importances_,
})
all_fi.to_csv('outputs/feature_importance_all.csv', index=False)

# ============================================================
# 7. SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("7. SAVING RESULTS")
print("=" * 60)

# Save metrics
metrics_out = {}
for k, v in results.items():
    metrics_out[k] = {kk: vv for kk, vv in v.items() if kk not in ['y_pred', 'y_prob']}
    if 'confusion_matrix' in metrics_out[k]:
        metrics_out[k]['confusion_matrix'] = v['confusion_matrix']

with open('outputs/classification_metrics.json', 'w') as f:
    json.dump(metrics_out, f, indent=2)

# Save predictions
pred_df = pd.DataFrame({
    'y_true_attack': y_attack_test,
    'pred_attack_rf': results['Attack_RF']['y_pred'],
    'prob_attack_rf': results['Attack_RF']['y_prob'],
    'pred_attack_gb': results['Attack_GB']['y_pred'],
    'prob_attack_gb': results['Attack_GB']['y_prob'],
    'y_true_sniffing': y_sniff_test,
    'pred_sniffing_rf': results['Sniffing_RF']['y_pred'],
    'prob_sniffing_rf': results['Sniffing_RF']['y_prob'],
    'pred_sniffing_gb': results['Sniffing_GB']['y_pred'],
    'prob_sniffing_gb': results['Sniffing_GB']['y_prob'],
})
pred_df.to_csv('outputs/test_predictions.csv', index=False)

# ============================================================
# 8. COMPARE WITH REFERENCE
# ============================================================
print("\n" + "=" * 60)
print("8. COMPARISON WITH REFERENCE OUTPUT")
print("=" * 60)

# Reference has 300 rows with Probability_Attack, Probability_Sniffing, Attack, Sniffing
ref_attack_dist = ref['Attack'].value_counts().to_dict() if 'Attack' in ref.columns else {}
ref_sniff_dist = ref['Sniffing'].value_counts().to_dict() if 'Sniffing' in ref.columns else {}
print(f"Reference Attack distribution: {ref_attack_dist}")
print(f"Reference Sniffing distribution: {ref_sniff_dist}")

if 'Probability_Attack' in ref.columns:
    print(f"Reference Attack prob stats: mean={ref['Probability_Attack'].mean():.4f}, std={ref['Probability_Attack'].std():.4f}")
if 'Probability_Sniffing' in ref.columns:
    print(f"Reference Sniffing prob stats: mean={ref['Probability_Sniffing'].mean():.4f}, std={ref['Probability_Sniffing'].std():.4f}")

print("\nDone with analysis. Proceeding to generate figures...")
