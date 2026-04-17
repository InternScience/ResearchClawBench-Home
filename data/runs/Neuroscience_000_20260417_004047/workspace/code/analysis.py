#!/usr/bin/env python3
"""
SimBA-style Behavior Classification Reproducibility Study
=========================================================
Train supervised classifiers on pose-derived features to classify
Attack and Sniffing behaviors from the official SimBA sample project.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# Paths
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'outputs')
IMG = os.path.join(BASE, 'report', 'images')
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df_feat = pd.read_csv(os.path.join(DATA, 'Together_1_features_extracted.csv'), index_col=0)
df_tgt = pd.read_csv(os.path.join(DATA, 'Together_1_targets_inserted.csv'), index_col=0)
df_ref = pd.read_csv(os.path.join(DATA, 'Together_1_machine_results_reference.csv'), index_col=0)

print(f"Features shape: {df_feat.shape}")
print(f"Targets shape:  {df_tgt.shape}")
print(f"Reference shape: {df_ref.shape}")

# Extract targets
y_attack = df_tgt['Attack'].values
y_sniffing = df_tgt['Sniffing'].values

print(f"\nAttack distribution:  0={np.sum(y_attack==0)}, 1={np.sum(y_attack==1)}")
print(f"Sniffing distribution: 0={np.sum(y_sniffing==0)}, 1={np.sum(y_sniffing==1)}")

# ============================================================
# 2. Feature Engineering (SimBA-style)
# ============================================================
print("\n" + "=" * 60)
print("2. FEATURE ENGINEERING")
print("=" * 60)

# Extract raw pose coordinates
pose_cols = [c for c in df_feat.columns if c not in ['Feature_1', 'Feature_2']]
raw_features = df_feat[pose_cols].copy()

# Body part names for each animal
bp_names_1 = ['Nose_1', 'Ear_left_1', 'Ear_right_1', 'Center_1', 'Lat_left_1', 'Lat_right_1', 'Tail_base_1', 'Tail_end_1']
bp_names_2 = ['Nose_2', 'Ear_left_2', 'Ear_right_2', 'Center_2', 'Lat_left_2', 'Lat_right_2', 'Tail_base_2', 'Tail_end_2']

def get_xy(df, bp):
    return df[f'{bp}_x'].values, df[f'{bp}_y'].values

def euclidean(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Build engineered features
eng = pd.DataFrame(index=df_feat.index)

# --- Distances between body parts (within animal) ---
for animal, bps in [('M1', bp_names_1), ('M2', bp_names_2)]:
    for i in range(len(bps)):
        for j in range(i+1, len(bps)):
            x1, y1 = get_xy(raw_features, bps[i])
            x2, y2 = get_xy(raw_features, bps[j])
            eng[f'{animal}_{bps[i].split("_")[0]}_{bps[j].split("_")[0]}_dist'] = euclidean(x1, y1, x2, y2)

# --- Inter-animal distances ---
for bp1 in bp_names_1:
    for bp2 in bp_names_2:
        x1, y1 = get_xy(raw_features, bp1)
        x2, y2 = get_xy(raw_features, bp2)
        name1 = bp1.replace('_1', '')
        name2 = bp2.replace('_2', '')
        eng[f'M1{name1}_M2{name2}_dist'] = euclidean(x1, y1, x2, y2)

# --- Movement features (frame-to-frame displacement) ---
for bp in bp_names_1 + bp_names_2:
    x, y = get_xy(raw_features, bp)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    eng[f'{bp}_movement'] = np.sqrt(dx**2 + dy**2)

# --- Body area approximation (polygon area using Shoelace) ---
def polygon_area(xs, ys):
    """Compute polygon area using the Shoelace formula."""
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0

for animal, bps in [('M1', bp_names_1), ('M2', bp_names_2)]:
    xs_arr = np.column_stack([raw_features[f'{bp}_x'].values for bp in bps])
    ys_arr = np.column_stack([raw_features[f'{bp}_y'].values for bp in bps])
    n_pts = xs_arr.shape[1]
    area = np.zeros(len(raw_features))
    for i in range(n_pts):
        j = (i + 1) % n_pts
        area += xs_arr[:, i] * ys_arr[:, j] - xs_arr[:, j] * ys_arr[:, i]
    eng[f'{animal}_body_area'] = np.abs(area) / 2.0

# --- Centroid distance ---
cx1 = np.mean([raw_features[f'{bp}_x'].values for bp in bp_names_1], axis=0)
cy1 = np.mean([raw_features[f'{bp}_y'].values for bp in bp_names_1], axis=0)
cx2 = np.mean([raw_features[f'{bp}_x'].values for bp in bp_names_2], axis=0)
cy2 = np.mean([raw_features[f'{bp}_y'].values for bp in bp_names_2], axis=0)
eng['centroid_distance'] = euclidean(cx1, cy1, cx2, cy2)

# --- Nose-to-nose distance ---
nx1, ny1 = get_xy(raw_features, 'Nose_1')
nx2, ny2 = get_xy(raw_features, 'Nose_2')
eng['nose_to_nose_dist'] = euclidean(nx1, ny1, nx2, ny2)

# --- Angles ---
def angle_3pt(ax, ay, bx, by, cx, cy):
    """Angle at point B formed by A-B-C."""
    ba_x, ba_y = ax - bx, ay - by
    bc_x, bc_y = cx - bx, cy - by
    dot = ba_x * bc_x + ba_y * bc_y
    mag_ba = np.sqrt(ba_x**2 + ba_y**2) + 1e-10
    mag_bc = np.sqrt(bc_x**2 + bc_y**2) + 1e-10
    cos_angle = np.clip(dot / (mag_ba * mag_bc), -1, 1)
    return np.degrees(np.arccos(cos_angle))

# Head angle (ear-nose-ear)
for animal, bps in [('M1', bp_names_1), ('M2', bp_names_2)]:
    nx, ny = get_xy(raw_features, bps[0])  # Nose
    elx, ely = get_xy(raw_features, bps[1])  # Ear_left
    erx, ery = get_xy(raw_features, bps[2])  # Ear_right
    eng[f'{animal}_head_angle'] = angle_3pt(elx, ely, nx, ny, erx, ery)

# Body bend angle (nose-center-tail_base)
for animal, bps in [('M1', bp_names_1), ('M2', bp_names_2)]:
    nx, ny = get_xy(raw_features, bps[0])
    cx, cy = get_xy(raw_features, bps[3])  # Center
    tx, ty = get_xy(raw_features, bps[6])  # Tail_base
    eng[f'{animal}_body_bend'] = angle_3pt(nx, ny, cx, cy, tx, ty)

# --- Rolling window features (mean, std over 5 and 10 frames) ---
movement_cols = [c for c in eng.columns if 'movement' in c]
distance_cols = [c for c in eng.columns if 'dist' in c]

for window in [5, 10]:
    for col in movement_cols[:8]:  # Subset to avoid explosion
        eng[f'{col}_roll_mean_{window}'] = eng[col].rolling(window, min_periods=1).mean()
        eng[f'{col}_roll_std_{window}'] = eng[col].rolling(window, min_periods=1).std().fillna(0)
    for col in ['centroid_distance', 'nose_to_nose_dist']:
        eng[f'{col}_roll_mean_{window}'] = eng[col].rolling(window, min_periods=1).mean()
        eng[f'{col}_roll_std_{window}'] = eng[col].rolling(window, min_periods=1).std().fillna(0)

# Also include original Feature_1 and Feature_2
eng['Feature_1'] = df_feat['Feature_1'].values
eng['Feature_2'] = df_feat['Feature_2'].values

# Handle any NaN/inf
eng = eng.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Engineered features shape: {eng.shape}")
print(f"Feature categories: {len([c for c in eng.columns if 'dist' in c])} distance, "
      f"{len([c for c in eng.columns if 'movement' in c])} movement, "
      f"{len([c for c in eng.columns if 'angle' in c or 'bend' in c])} angle, "
      f"{len([c for c in eng.columns if 'area' in c])} area, "
      f"{len([c for c in eng.columns if 'roll' in c])} rolling")

# Save feature summary
feat_summary = eng.describe().T
feat_summary.to_csv(os.path.join(OUT, 'feature_summary.csv'))

X = eng.values
feature_names = list(eng.columns)

# ============================================================
# 3. Train/Test Split
# ============================================================
print("\n" + "=" * 60)
print("3. TRAIN/TEST SPLIT")
print("=" * 60)

X_train, X_test, y_atk_train, y_atk_test, y_snf_train, y_snf_test = \
    train_test_split(X, y_attack, y_sniffing, test_size=0.2, random_state=42,
                     stratify=y_attack)  # stratify on attack (majority task)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")
print(f"Train Attack: 0={np.sum(y_atk_train==0)}, 1={np.sum(y_atk_train==1)}")
print(f"Test  Attack: 0={np.sum(y_atk_test==0)}, 1={np.sum(y_atk_test==1)}")
print(f"Train Sniffing: 0={np.sum(y_snf_train==0)}, 1={np.sum(y_snf_train==1)}")
print(f"Test  Sniffing: 0={np.sum(y_snf_test==0)}, 1={np.sum(y_snf_test==1)}")

# ============================================================
# 4. Model Training
# ============================================================
print("\n" + "=" * 60)
print("4. MODEL TRAINING")
print("=" * 60)

results = {}

behaviors = {
    'Attack': (y_atk_train, y_atk_test),
    'Sniffing': (y_snf_train, y_snf_test)
}

models_config = {
    'RandomForest': lambda: RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, random_state=42, class_weight='balanced'
    ),
    'GradientBoosting': lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42
    )
}

trained_models = {}

for beh_name, (y_tr, y_te) in behaviors.items():
    print(f"\n--- {beh_name} ---")
    results[beh_name] = {}
    trained_models[beh_name] = {}
    
    for model_name, model_fn in models_config.items():
        print(f"  Training {model_name}...")
        clf = model_fn()
        clf.fit(X_train, y_tr)
        
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_prob)
        ap = average_precision_score(y_te, y_prob)
        
        results[beh_name][model_name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'auc_roc': round(auc, 4),
            'avg_precision': round(ap, 4),
            'y_pred': y_pred,
            'y_prob': y_prob,
            'y_true': y_te
        }
        trained_models[beh_name][model_name] = clf
        
        print(f"    Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# ============================================================
# 5. Cross-Validation
# ============================================================
print("\n" + "=" * 60)
print("5. CROSS-VALIDATION (5-fold)")
print("=" * 60)

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for beh_name, y_all in [('Attack', y_attack), ('Sniffing', y_sniffing)]:
    print(f"\n--- {beh_name} ---")
    cv_results[beh_name] = {}
    
    for model_name, model_fn in models_config.items():
        fold_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
        
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y_all)):
            clf = model_fn()
            clf.fit(X[tr_idx], y_all[tr_idx])
            y_pred = clf.predict(X[te_idx])
            y_prob = clf.predict_proba(X[te_idx])[:, 1]
            
            fold_metrics['acc'].append(accuracy_score(y_all[te_idx], y_pred))
            fold_metrics['prec'].append(precision_score(y_all[te_idx], y_pred, zero_division=0))
            fold_metrics['rec'].append(recall_score(y_all[te_idx], y_pred, zero_division=0))
            fold_metrics['f1'].append(f1_score(y_all[te_idx], y_pred, zero_division=0))
            fold_metrics['auc'].append(roc_auc_score(y_all[te_idx], y_prob))
        
        cv_results[beh_name][model_name] = {
            k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in fold_metrics.items()
        }
        print(f"  {model_name}: F1={cv_results[beh_name][model_name]['f1']}, AUC={cv_results[beh_name][model_name]['auc']}")

# Save CV results
with open(os.path.join(OUT, 'cv_results.json'), 'w') as f:
    json.dump(cv_results, f, indent=2)

# ============================================================
# 6. Figures
# ============================================================
print("\n" + "=" * 60)
print("6. GENERATING FIGURES")
print("=" * 60)

# --- Figure 1: Class Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, y) in zip(axes, [('Attack', y_attack), ('Sniffing', y_sniffing)]):
    counts = [np.sum(y == 0), np.sum(y == 1)]
    bars = ax.bar(['Absent (0)', 'Present (1)'], counts, color=['#4C72B0', '#DD8452'])
    ax.set_title(f'{name} Label Distribution', fontsize=13)
    ax.set_ylabel('Frame Count')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved class_distribution.png")

# --- Figure 2: Confusion Matrices ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, beh_name in enumerate(['Attack', 'Sniffing']):
    for j, model_name in enumerate(['RandomForest', 'GradientBoosting']):
        r = results[beh_name][model_name]
        cm = confusion_matrix(r['y_true'], r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i][j],
                    xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'])
        axes[i][j].set_title(f'{beh_name} - {model_name}', fontsize=12)
        axes[i][j].set_ylabel('True Label')
        axes[i][j].set_xlabel('Predicted Label')
fig.suptitle('Confusion Matrices', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved confusion_matrices.png")

# --- Figure 3: Precision-Recall Curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, beh_name in zip(axes, ['Attack', 'Sniffing']):
    for model_name in ['RandomForest', 'GradientBoosting']:
        r = results[beh_name][model_name]
        prec_arr, rec_arr, _ = precision_recall_curve(r['y_true'], r['y_prob'])
        ap = r['avg_precision']
        ax.plot(rec_arr, prec_arr, label=f'{model_name} (AP={ap:.3f})', linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{beh_name} - Precision-Recall Curve', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'precision_recall_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved precision_recall_curves.png")

# --- Figure 4: ROC Curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, beh_name in zip(axes, ['Attack', 'Sniffing']):
    for model_name in ['RandomForest', 'GradientBoosting']:
        r = results[beh_name][model_name]
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
        auc_val = r['auc_roc']
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc_val:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{beh_name} - ROC Curve', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved roc_curves.png")

# --- Figure 5: Feature Importance (Top 20) ---
fig, axes = plt.subplots(2, 1, figsize=(12, 14))
for ax, beh_name in zip(axes, ['Attack', 'Sniffing']):
    clf = trained_models[beh_name]['RandomForest']
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-20:]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]
    
    ax.barh(range(20), top_vals, color='#4C72B0')
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
    ax.set_title(f'{beh_name} - Top 20 Feature Importances (Random Forest)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved feature_importance.png")

# --- Figure 6: Probability Distribution ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, beh_name in enumerate(['Attack', 'Sniffing']):
    for j, model_name in enumerate(['RandomForest', 'GradientBoosting']):
        r = results[beh_name][model_name]
        prob_0 = r['y_prob'][r['y_true'] == 0]
        prob_1 = r['y_prob'][r['y_true'] == 1]
        axes[i][j].hist(prob_0, bins=30, alpha=0.6, label='Absent', color='#4C72B0', density=True)
        axes[i][j].hist(prob_1, bins=30, alpha=0.6, label='Present', color='#DD8452', density=True)
        axes[i][j].set_xlabel('Predicted Probability', fontsize=11)
        axes[i][j].set_ylabel('Density', fontsize=11)
        axes[i][j].set_title(f'{beh_name} - {model_name}', fontsize=12)
        axes[i][j].legend(fontsize=10)
fig.suptitle('Predicted Probability Distributions by True Class', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'probability_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved probability_distributions.png")

# --- Figure 7: Feature Correlation Heatmap (top features) ---
top_feats_attack = np.argsort(trained_models['Attack']['RandomForest'].feature_importances_)[-15:]
top_feats_sniff = np.argsort(trained_models['Sniffing']['RandomForest'].feature_importances_)[-15:]
top_feats_combined = list(set(list(top_feats_attack) + list(top_feats_sniff)))
top_feat_names = [feature_names[i] for i in top_feats_combined]

corr_df = eng[top_feat_names].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_df, annot=False, cmap='RdBu_r', center=0, ax=ax,
            xticklabels=True, yticklabels=True)
ax.set_title('Feature Correlation Matrix (Top Important Features)', fontsize=14)
plt.xticks(fontsize=8, rotation=45, ha='right')
plt.yticks(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'feature_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved feature_correlation.png")

# --- Figure 8: Temporal behavior predictions ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
# Use RF model to predict on all data for temporal view
for ax, beh_name, y_all in zip(axes, ['Attack', 'Sniffing'], [y_attack, y_sniffing]):
    clf = trained_models[beh_name]['RandomForest']
    y_prob_all = clf.predict_proba(X)[:, 1]
    
    ax.fill_between(range(len(y_all)), y_all, alpha=0.3, color='#DD8452', label='True Label')
    ax.plot(y_prob_all, color='#4C72B0', alpha=0.7, linewidth=0.5, label='Predicted Probability')
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Probability / Label', fontsize=11)
    ax.set_title(f'{beh_name} - Temporal Prediction Profile', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, len(y_all)])
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'temporal_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved temporal_predictions.png")

# --- Figure 9: Permutation Importance ---
fig, axes = plt.subplots(2, 1, figsize=(12, 14))
for ax, beh_name in zip(axes, ['Attack', 'Sniffing']):
    clf = trained_models[beh_name]['RandomForest']
    y_te = results[beh_name]['RandomForest']['y_true']
    perm_imp = permutation_importance(clf, X_test, y_te, n_repeats=10, random_state=42, n_jobs=-1)
    
    top_idx = np.argsort(perm_imp.importances_mean)[-20:]
    top_names_perm = [feature_names[i] for i in top_idx]
    top_vals_perm = perm_imp.importances_mean[top_idx]
    top_stds_perm = perm_imp.importances_std[top_idx]
    
    ax.barh(range(20), top_vals_perm, xerr=top_stds_perm, color='#55A868', capsize=3)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names_perm, fontsize=9)
    ax.set_xlabel('Mean Accuracy Decrease', fontsize=11)
    ax.set_title(f'{beh_name} - Top 20 Permutation Importances', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'permutation_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved permutation_importance.png")

# --- Figure 10: Model Comparison Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
x_pos = np.arange(len(metrics_to_plot))
width = 0.35

for ax, beh_name in zip(axes, ['Attack', 'Sniffing']):
    rf_vals = [results[beh_name]['RandomForest'][m] for m in metrics_to_plot]
    gb_vals = [results[beh_name]['GradientBoosting'][m] for m in metrics_to_plot]
    
    ax.bar(x_pos - width/2, rf_vals, width, label='Random Forest', color='#4C72B0')
    ax.bar(x_pos + width/2, gb_vals, width, label='Gradient Boosting', color='#DD8452')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC'], fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.set_title(f'{beh_name} - Model Comparison', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved model_comparison.png")

# ============================================================
# 7. Save Results
# ============================================================
print("\n" + "=" * 60)
print("7. SAVING RESULTS")
print("=" * 60)

# Clean results for JSON (remove numpy arrays)
clean_results = {}
for beh_name in results:
    clean_results[beh_name] = {}
    for model_name in results[beh_name]:
        r = results[beh_name][model_name]
        clean_results[beh_name][model_name] = {
            k: v for k, v in r.items() if k not in ['y_pred', 'y_prob', 'y_true']
        }

with open(os.path.join(OUT, 'classification_results.json'), 'w') as f:
    json.dump(clean_results, f, indent=2)

# Feature importance tables
for beh_name in ['Attack', 'Sniffing']:
    clf = trained_models[beh_name]['RandomForest']
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'gini_importance': clf.feature_importances_
    }).sort_values('gini_importance', ascending=False)
    imp_df.to_csv(os.path.join(OUT, f'feature_importance_{beh_name.lower()}.csv'), index=False)

# Classification reports
for beh_name in ['Attack', 'Sniffing']:
    for model_name in ['RandomForest', 'GradientBoosting']:
        r = results[beh_name][model_name]
        report = classification_report(r['y_true'], r['y_pred'],
                                       target_names=['Absent', 'Present'])
        with open(os.path.join(OUT, f'report_{beh_name.lower()}_{model_name.lower()}.txt'), 'w') as f:
            f.write(report)

# Confusion matrices as CSV
for beh_name in ['Attack', 'Sniffing']:
    for model_name in ['RandomForest', 'GradientBoosting']:
        r = results[beh_name][model_name]
        cm = confusion_matrix(r['y_true'], r['y_pred'])
        cm_df = pd.DataFrame(cm, index=['True_Absent', 'True_Present'],
                             columns=['Pred_Absent', 'Pred_Present'])
        cm_df.to_csv(os.path.join(OUT, f'confusion_matrix_{beh_name.lower()}_{model_name.lower()}.csv'))

print("All results saved to outputs/")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
