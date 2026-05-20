"""
SimBA Behavior Classification - Optimized Feature Engineering & Training
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

print("=" * 70)
print("SIMBA BEHAVIOR CLASSIFICATION REPRODUCIBILITY ANALYSIS")
print("=" * 70)

# 1. LOAD DATA
print("\n[1] Loading data...")
df_feat = pd.read_csv('data/Together_1_features_extracted.csv')
df_tgt = pd.read_csv('data/Together_1_targets_inserted.csv')
df_ref = pd.read_csv('data/Together_1_machine_results_reference.csv')

labels_attack = df_tgt['Attack'].values
labels_sniffing = df_tgt['Sniffing'].values
print(f"  Features: {df_feat.shape}, Targets: {df_tgt.shape}, Reference: {df_ref.shape}")
print(f"  Attack dist: {np.bincount(labels_attack.astype(int))}, Sniffing dist: {np.bincount(labels_sniffing.astype(int))}")

# 2. OPTIMIZED FEATURE ENGINEERING
print("\n[2] Engineering features...")

def engineer_features(df):
    feat = {}
    
    m1_parts = {
        'nose': ('Nose_1_x', 'Nose_1_y'),
        'ear_l': ('Ear_left_1_x', 'Ear_left_1_y'),
        'ear_r': ('Ear_right_1_x', 'Ear_right_1_y'),
        'center': ('Center_1_x', 'Center_1_y'),
        'lat_l': ('Lat_left_1_x', 'Lat_left_1_y'),
        'lat_r': ('Lat_right_1_x', 'Lat_right_1_y'),
        'tail_b': ('Tail_base_1_x', 'Tail_base_1_y'),
        'tail_e': ('Tail_end_1_x', 'Tail_end_1_y'),
    }
    m2_parts = {
        'nose': ('Nose_2_x', 'Nose_2_y'),
        'ear_l': ('Ear_left_2_x', 'Ear_left_2_y'),
        'ear_r': ('Ear_right_2_x', 'Ear_right_2_y'),
        'center': ('Center_2_x', 'Center_2_y'),
        'lat_l': ('Lat_left_2_x', 'Lat_left_2_y'),
        'lat_r': ('Lat_right_2_x', 'Lat_right_2_y'),
        'tail_b': ('Tail_base_2_x', 'Tail_base_2_y'),
        'tail_e': ('Tail_end_2_x', 'Tail_end_2_y'),
    }
    
    def euc(c1, c2):
        return np.sqrt((df[c1[0]] - df[c2[0]])**2 + (df[c1[1]] - df[c2[1]])**2)
    
    # Within-mouse distances
    for mname, parts in [('M1', m1_parts), ('M2', m2_parts)]:
        feat[f'{mname}_nose_to_tail'] = euc(parts['nose'], parts['tail_e'])
        feat[f'{mname}_ear_distance'] = euc(parts['ear_l'], parts['ear_r'])
        feat[f'{mname}_nose_to_centroid'] = euc(parts['nose'], parts['center'])
        feat[f'{mname}_nose_to_lat_left'] = euc(parts['nose'], parts['lat_l'])
        feat[f'{mname}_nose_to_lat_right'] = euc(parts['nose'], parts['lat_r'])
        feat[f'{mname}_centroid_to_lat_left'] = euc(parts['center'], parts['lat_l'])
        feat[f'{mname}_centroid_to_lat_right'] = euc(parts['center'], parts['lat_r'])
    
    # Between-mouse distances
    feat['centroid_distance'] = euc(m1_parts['center'], m2_parts['center'])
    feat['nose_to_nose_distance'] = euc(m1_parts['nose'], m2_parts['nose'])
    feat['M1_nose_to_M2_lat_left'] = euc(m1_parts['nose'], m2_parts['lat_l'])
    feat['M1_nose_to_M2_lat_right'] = euc(m1_parts['nose'], m2_parts['lat_r'])
    feat['M2_nose_to_M1_lat_left'] = euc(m2_parts['nose'], m1_parts['lat_l'])
    feat['M2_nose_to_M1_lat_right'] = euc(m2_parts['nose'], m1_parts['lat_r'])
    feat['M1_nose_to_M2_tail_base'] = euc(m1_parts['nose'], m2_parts['tail_b'])
    feat['M2_nose_to_M1_tail_base'] = euc(m2_parts['nose'], m1_parts['tail_b'])
    
    # Movement features
    all_parts = [
        ('c1', m1_parts['center']), ('c2', m2_parts['center']),
        ('n1', m1_parts['nose']), ('n2', m2_parts['nose']),
        ('tb1', m1_parts['tail_b']), ('tb2', m2_parts['tail_b']),
        ('te1', m1_parts['tail_e']), ('te2', m2_parts['tail_e']),
        ('el1', m1_parts['ear_l']), ('el2', m2_parts['ear_l']),
        ('er1', m1_parts['ear_r']), ('er2', m2_parts['ear_r']),
        ('ll1', m1_parts['lat_l']), ('ll2', m2_parts['lat_l']),
        ('lr1', m1_parts['lat_r']), ('lr2', m2_parts['lat_r']),
    ]
    for pn, (px, py) in all_parts:
        dx = df[px].diff().fillna(0).values
        dy = df[py].diff().fillna(0).values
        feat[f'mov_{pn}'] = np.sqrt(dx**2 + dy**2)
    
    feat['total_mov_centroid'] = feat['mov_c1'] + feat['mov_c2']
    feat['total_mov_nose'] = feat['mov_n1'] + feat['mov_n2']
    feat['total_mov_tail'] = feat['mov_te1'] + feat['mov_te2']
    
    # Rolling statistics
    key_feats = ['centroid_distance', 'nose_to_nose_distance', 'total_mov_centroid', 'total_mov_nose',
                 'M1_nose_to_tail', 'M2_nose_to_tail', 'M1_ear_distance', 'M2_ear_distance']
    windows = [2, 5, 15]
    arr = pd.DataFrame(feat)
    for col in key_feats:
        for w in windows:
            roll = arr[col].rolling(window=w, min_periods=1)
            feat[f'{col}_med{w}'] = roll.median().values
            feat[f'{col}_mean{w}'] = roll.mean().values
    
    # Deviation features
    for col in ['total_mov_centroid', 'centroid_distance', 'total_mov_nose']:
        feat[f'{col}_dev'] = feat[col] - np.mean(feat[col])
    
    # Angle features
    feat['M1_angle'] = np.arctan2(df['Nose_1_y'].values - df['Tail_base_1_y'].values,
                                   df['Nose_1_x'].values - df['Tail_base_1_x'].values)
    feat['M2_angle'] = np.arctan2(df['Nose_2_y'].values - df['Tail_base_2_y'].values,
                                   df['Nose_2_x'].values - df['Tail_base_2_x'].values)
    feat['total_angle'] = feat['M1_angle'] + feat['M2_angle']
    
    # Confidence
    for p in ['Nose_1_p', 'Nose_2_p', 'Center_1_p', 'Center_2_p']:
        if p in df.columns:
            feat[f'{p}_mean5'] = pd.Series(df[p].values).rolling(5, min_periods=1).mean().values
    
    result = pd.DataFrame(feat)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result

features_df = engineer_features(df_feat)
print(f"  Engineered {features_df.shape[1]} features for {features_df.shape[0]} frames")
features_df.to_csv('outputs/engineered_features.csv', index=False)

# 3. TRAIN/TEST SPLIT
print("\n[3] Train/test split...")
X = features_df.values
X_train, X_test, ya_train, ya_test, ys_train, ys_test, idx_tr, idx_te = \
    train_test_split(X, labels_attack, labels_sniffing, np.arange(len(X)),
                     test_size=0.2, random_state=42, stratify=labels_attack)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)
print(f"  Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")

# 4. TRAIN CLASSIFIERS
print("\n[4] Training classifiers...")
models = {}

for beh_name, y_tr in [('Attack', ya_train), ('Sniffing', ys_train)]:
    print(f"\n  --- {beh_name} ---")
    for clf_name, clf in [('RF', RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                                        random_state=42, class_weight='balanced', n_jobs=-1)),
                           ('GB', GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                             learning_rate=0.1, random_state=42, subsample=0.8))]:
        clf.fit(X_tr, y_tr)
        key = f'{beh_name}_{clf_name}'
        models[key] = clf
        print(f"    {key} trained")

# 5. CROSS-VALIDATION
print("\n[5] Cross-validation (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for beh_name, y_tr in [('Attack', ya_train), ('Sniffing', ys_train)]:
    for clf_name in ['RF', 'GB']:
        key = f'{beh_name}_{clf_name}'
        scores = cross_val_score(models[key], X_tr, y_tr, cv=cv, scoring='f1')
        cv_results[key] = {'mean_f1': float(scores.mean()), 'std_f1': float(scores.std()), 'fold_scores': scores.tolist()}
        print(f"  {key} F1: {scores.mean():.4f} +/- {scores.std():.4f}")

with open('outputs/cross_validation_results.json', 'w') as f:
    json.dump(cv_results, f, indent=2)

# 6. EVALUATION
print("\n[6] Evaluation metrics...")
eval_results = {}
all_predictions = {}

for beh_name, y_te in [('Attack', ya_test), ('Sniffing', ys_test)]:
    for clf_name in ['RF', 'GB']:
        key = f'{beh_name}_{clf_name}'
        y_pred = models[key].predict(X_te)
        y_prob = models[key].predict_proba(X_te)[:, 1]
        
        all_predictions[key] = {'y_true': y_te, 'y_pred': y_pred, 'y_prob': y_prob}
        
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        ap = average_precision_score(y_te, y_prob)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc_val = auc(fpr, tpr)
        cm = confusion_matrix(y_te, y_pred)
        prec_arr, rec_arr, _ = precision_recall_curve(y_te, y_prob)
        
        eval_results[key] = {
            'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec),
            'f1': float(f1), 'average_precision': float(ap), 'roc_auc': float(roc_auc_val),
            'confusion_matrix': cm.tolist(),
            'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
        }
        print(f"  {key}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={roc_auc_val:.4f}, AP={ap:.4f}")

with open('outputs/evaluation_results.json', 'w') as f:
    json.dump(eval_results, f, indent=2, default=str)

# 7. FEATURE IMPORTANCE
print("\n[7] Feature importance...")
feature_names = list(features_df.columns)
importance_data = {}

for beh_name in ['Attack', 'Sniffing']:
    rf_key = f'{beh_name}_RF'
    gb_key = f'{beh_name}_GB'
    
    perm_imp = permutation_importance(models[rf_key], X_te, 
                                       ya_test if beh_name == 'Attack' else ys_test,
                                       n_repeats=10, random_state=42, n_jobs=-1)
    
    importance_data[beh_name] = pd.DataFrame({
        'feature': feature_names,
        'rf_gini': models[rf_key].feature_importances_,
        'gb_importance': models[gb_key].feature_importances_,
        'rf_permutation_mean': perm_imp.importances_mean,
        'rf_permutation_std': perm_imp.importances_std,
    })
    importance_data[beh_name].to_csv(f'outputs/feature_importance_{beh_name.lower()}.csv', index=False)
    
    print(f"\n  Top 10 features for {beh_name} (RF Gini):")
    top = importance_data[beh_name].nlargest(10, 'rf_gini')
    for _, row in top.iterrows():
        print(f"    {row['feature']:50s} {row['rf_gini']:.6f}")

imp_combined = importance_data['Attack'].merge(importance_data['Sniffing'], 
    on='feature', suffixes=('_attack', '_sniff'))
imp_combined.to_csv('outputs/feature_importance.csv', index=False)

# 8. COMPARISON WITH REFERENCE
print("\n[8] Comparison with reference...")
ref_indices = df_ref['Unnamed: 0'].values.astype(int)
X_ref = features_df.iloc[ref_indices].values
X_ref_scaled = scaler.transform(X_ref)

ref_compare = pd.DataFrame({
    'frame_idx': ref_indices,
    'ref_attack_prob': df_ref['Probability_Attack'].values,
    'ref_sniff_prob': df_ref['Probability_Sniffing'].values,
    'ref_attack_pred': df_ref['Attack'].values.astype(int),
    'ref_sniff_pred': df_ref['Sniffing'].values.astype(int),
    'true_attack': labels_attack[ref_indices],
    'true_sniffing': labels_sniffing[ref_indices],
    'our_attack_prob': models['Attack_RF'].predict_proba(X_ref_scaled)[:, 1],
    'our_sniff_prob': models['Sniffing_RF'].predict_proba(X_ref_scaled)[:, 1],
})
ref_compare['our_attack_pred'] = (ref_compare['our_attack_prob'] >= 0.5).astype(int)
ref_compare['our_sniff_pred'] = (ref_compare['our_sniff_prob'] >= 0.5).astype(int)
ref_compare.to_csv('outputs/reference_comparison.csv', index=False)

attack_agree = (ref_compare['ref_attack_pred'] == ref_compare['our_attack_pred']).mean()
sniff_agree = (ref_compare['ref_sniff_pred'] == ref_compare['our_sniff_pred']).mean()
att_spear, att_p = spearmanr(ref_compare['ref_attack_prob'], ref_compare['our_attack_prob'])
snf_spear, snf_p = spearmanr(ref_compare['ref_sniff_prob'], ref_compare['our_sniff_prob'])
att_pear, att_pp = pearsonr(ref_compare['ref_attack_prob'], ref_compare['our_attack_prob'])
snf_pear, snf_pp = pearsonr(ref_compare['ref_sniff_prob'], ref_compare['our_sniff_prob'])

print(f"  Attack agreement: {attack_agree:.4f}, Sniffing agreement: {sniff_agree:.4f}")
print(f"  Attack prob Spearman: {att_spear:.4f} (p={att_p:.2e})")
print(f"  Sniffing prob Spearman: {snf_spear:.4f} (p={snf_p:.2e})")

ref_summary = {
    'attack_prediction_agreement': float(attack_agree),
    'sniff_prediction_agreement': float(sniff_agree),
    'attack_prob_spearman': float(att_spear),
    'attack_prob_spearman_pval': float(att_p),
    'sniff_prob_spearman': float(snf_spear),
    'sniff_prob_spearman_pval': float(snf_p),
    'attack_prob_pearson': float(att_pear),
    'sniff_prob_pearson': float(snf_pear),
}
with open('outputs/reference_comparison_summary.json', 'w') as f:
    json.dump(ref_summary, f, indent=2)

# Save full predictions
print("\n  Generating full dataset predictions...")
X_full_scaled = scaler.transform(X)
full_preds = pd.DataFrame({
    'frame_idx': np.arange(len(X)),
    'attack_rf_prob': models['Attack_RF'].predict_proba(X_full_scaled)[:, 1],
    'attack_gb_prob': models['Attack_GB'].predict_proba(X_full_scaled)[:, 1],
    'sniff_rf_prob': models['Sniffing_RF'].predict_proba(X_full_scaled)[:, 1],
    'sniff_gb_prob': models['Sniffing_GB'].predict_proba(X_full_scaled)[:, 1],
    'attack_rf_pred': models['Attack_RF'].predict(X_full_scaled),
    'attack_gb_pred': models['Attack_GB'].predict(X_full_scaled),
    'sniff_rf_pred': models['Sniffing_RF'].predict(X_full_scaled),
    'sniff_gb_pred': models['Sniffing_GB'].predict(X_full_scaled),
    'true_attack': labels_attack,
    'true_sniffing': labels_sniffing,
})
full_preds.to_csv('outputs/full_predictions.csv', index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
