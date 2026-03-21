"""
SimBA-style behavior classification pipeline.
Reproduces the SimBA workflow: feature engineering from pose data → supervised classification.
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'outputs')
IMG = os.path.join(BASE, 'report', 'images')
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ─── 1. Load data ───────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

feat_df = pd.read_csv(os.path.join(DATA, 'Together_1_features_extracted.csv'), index_col=0)
tgt_df = pd.read_csv(os.path.join(DATA, 'Together_1_targets_inserted.csv'), index_col=0)
ref_df = pd.read_csv(os.path.join(DATA, 'Together_1_machine_results_reference.csv'), index_col=0)

print(f"Features: {feat_df.shape}")
print(f"Targets:  {tgt_df.shape}")
print(f"Reference: {ref_df.shape}")

# Extract targets
y_attack = tgt_df['Attack'].values
y_sniff = tgt_df['Sniffing'].values
print(f"\nAttack: {y_attack.sum()}/{len(y_attack)} positive ({100*y_attack.mean():.1f}%)")
print(f"Sniffing: {y_sniff.sum()}/{len(y_sniff)} positive ({100*y_sniff.mean():.1f}%)")

# Body part definitions
BODY_PARTS_1 = ['Nose_1', 'Ear_left_1', 'Ear_right_1', 'Center_1',
                'Lat_left_1', 'Lat_right_1', 'Tail_base_1', 'Tail_end_1']
BODY_PARTS_2 = ['Nose_2', 'Ear_left_2', 'Ear_right_2', 'Center_2',
                'Lat_left_2', 'Lat_right_2', 'Tail_base_2', 'Tail_end_2']

def get_xy(df, bp):
    return df[f'{bp}_x'].values, df[f'{bp}_y'].values

def euclidean(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# ─── 2. Feature Engineering (SimBA-style) ───────────────────────────────────
print("\n" + "=" * 60)
print("2. FEATURE ENGINEERING")
print("=" * 60)

pose_df = feat_df.drop(columns=['Feature_1', 'Feature_2'], errors='ignore')
n_frames = len(pose_df)
features = {}

# 2a. Intra-animal distances
for animal, parts in [('M1', BODY_PARTS_1), ('M2', BODY_PARTS_2)]:
    for i in range(len(parts)):
        for j in range(i+1, len(parts)):
            x1, y1 = get_xy(pose_df, parts[i])
            x2, y2 = get_xy(pose_df, parts[j])
            name = f'{animal}_{parts[i].split("_")[0]}_{parts[j].split("_")[0]}_dist'
            features[name] = euclidean(x1, y1, x2, y2)

# 2b. Inter-animal distances (key body parts)
for bp1 in BODY_PARTS_1:
    for bp2 in BODY_PARTS_2:
        x1, y1 = get_xy(pose_df, bp1)
        x2, y2 = get_xy(pose_df, bp2)
        name = f'{bp1.replace("_1","")}_to_{bp2.replace("_2","")}_dist'
        features[name] = euclidean(x1, y1, x2, y2)

# 2c. Body lengths (nose to tail)
for animal, parts in [('M1', BODY_PARTS_1), ('M2', BODY_PARTS_2)]:
    nx, ny = get_xy(pose_df, parts[0])  # Nose
    tx, ty = get_xy(pose_df, parts[-1])  # Tail_end
    features[f'{animal}_body_length'] = euclidean(nx, ny, tx, ty)
    # Width: Lat_left to Lat_right
    lx, ly = get_xy(pose_df, parts[4])  # Lat_left
    rx, ry = get_xy(pose_df, parts[5])  # Lat_right
    features[f'{animal}_width'] = euclidean(lx, ly, rx, ry)
    # Ear distance
    elx, ely = get_xy(pose_df, parts[1])
    erx, ery = get_xy(pose_df, parts[2])
    features[f'{animal}_ear_dist'] = euclidean(elx, ely, erx, ery)

# 2d. Centroid distance
cx1, cy1 = get_xy(pose_df, 'Center_1')
cx2, cy2 = get_xy(pose_df, 'Center_2')
features['centroid_distance'] = euclidean(cx1, cy1, cx2, cy2)

# 2e. Movement features (velocity, acceleration)
for animal, parts in [('M1', BODY_PARTS_1), ('M2', BODY_PARTS_2)]:
    for bp in parts:
        x, y = get_xy(pose_df, bp)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        vel = np.sqrt(dx**2 + dy**2)
        features[f'{animal}_{bp.split("_")[0]}_velocity'] = vel
        acc = np.diff(vel, prepend=vel[0])
        features[f'{animal}_{bp.split("_")[0]}_acceleration'] = acc

# 2f. Angles
def angle_3pts(ax, ay, bx, by, cx, cy):
    """Angle at B formed by A-B-C."""
    ba = np.stack([ax - bx, ay - by], axis=-1)
    bc = np.stack([cx - bx, cy - by], axis=-1)
    cos_angle = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-10)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

for animal, parts in [('M1', BODY_PARTS_1), ('M2', BODY_PARTS_2)]:
    # Nose-Center-Tail angle (body curvature)
    nx, ny = get_xy(pose_df, parts[0])
    cx, cy = get_xy(pose_df, parts[3])
    tx, ty = get_xy(pose_df, parts[6])  # Tail_base
    features[f'{animal}_body_angle'] = angle_3pts(nx, ny, cx, cy, tx, ty)
    # Head angle: Ear_left - Nose - Ear_right
    elx, ely = get_xy(pose_df, parts[1])
    erx, ery = get_xy(pose_df, parts[2])
    features[f'{animal}_head_angle'] = angle_3pts(elx, ely, nx, ny, erx, ery)

# 2g. Facing angle (is M1 facing M2?)
for a, ap, b, bp_name in [('M1', BODY_PARTS_1, 'M2', BODY_PARTS_2), ('M2', BODY_PARTS_2, 'M1', BODY_PARTS_1)]:
    nx, ny = get_xy(pose_df, ap[0])  # Nose
    cx, cy = get_xy(pose_df, ap[3])  # Center
    ox, oy = get_xy(pose_df, bp_name[3])  # Other center
    # Direction vector: Center→Nose; Target vector: Center→Other
    features[f'{a}_facing_{b}_angle'] = angle_3pts(ox, oy, cx, cy, nx, ny)

# 2h. Rolling window statistics (mean, std over 5, 10 frame windows)
key_dist_features = ['centroid_distance', 'M1_body_length', 'M2_body_length',
                     'M1_Nose_velocity', 'M2_Nose_velocity',
                     'M1_Center_velocity', 'M2_Center_velocity']

for fname in key_dist_features:
    if fname in features:
        s = pd.Series(features[fname])
        for w in [5, 10]:
            features[f'{fname}_mean{w}'] = s.rolling(w, min_periods=1, center=True).mean().values
            features[f'{fname}_std{w}'] = s.rolling(w, min_periods=1, center=True).std().fillna(0).values

# 2i. Probability features from pose confidence
for animal, parts in [('M1', BODY_PARTS_1), ('M2', BODY_PARTS_2)]:
    probs = []
    for bp in parts:
        probs.append(pose_df[f'{bp}_p'].values)
    features[f'{animal}_mean_confidence'] = np.mean(probs, axis=0)
    features[f'{animal}_min_confidence'] = np.min(probs, axis=0)

# Build feature matrix
X = pd.DataFrame(features)
print(f"Engineered features: {X.shape[1]}")
print(f"Frames: {X.shape[0]}")

# Handle NaN/inf
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

feature_names = X.columns.tolist()
X_arr = X.values

# Save engineered features
X.to_csv(os.path.join(OUT, 'engineered_features.csv'), index=False)
print(f"Saved engineered features to outputs/")

# ─── 3. Train/Test Split ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. TRAIN/TEST SPLIT")
print("=" * 60)

# Use 80/20 stratified split
X_train, X_test, y_attack_train, y_attack_test = train_test_split(
    X_arr, y_attack, test_size=0.2, random_state=SEED, stratify=y_attack)
_, _, y_sniff_train, y_sniff_test = train_test_split(
    X_arr, y_sniff, test_size=0.2, random_state=SEED, stratify=y_sniff)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Attack train pos: {y_attack_train.sum()} ({100*y_attack_train.mean():.1f}%)")
print(f"Sniffing train pos: {y_sniff_train.sum()} ({100*y_sniff_train.mean():.1f}%)")

# ─── 4. Model Training & Evaluation ─────────────────────────────────────────
print("\n" + "=" * 60)
print("4. MODEL TRAINING & EVALUATION")
print("=" * 60)

classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=None,
                                            min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                     learning_rate=0.1, random_state=SEED),
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=SEED, C=1.0),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=SEED, C=1.0),
}

behaviors = {'Attack': (y_attack_train, y_attack_test),
             'Sniffing': (y_sniff_train, y_sniff_test)}

all_results = {}
all_models = {}

for behavior, (y_tr, y_te) in behaviors.items():
    print(f"\n--- {behavior} ---")
    all_results[behavior] = {}
    all_models[behavior] = {}

    for clf_name, clf in classifiers.items():
        print(f"  Training {clf_name}...")
        # Use scaled features for LR and SVM, unscaled for tree methods
        if clf_name in ['Logistic Regression', 'SVM (RBF)']:
            clf.fit(X_train_sc, y_tr)
            y_pred = clf.predict(X_test_sc)
            y_prob = clf.predict_proba(X_test_sc)[:, 1]
        else:
            clf.fit(X_train, y_tr)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_prob)
        ap = average_precision_score(y_te, y_prob)

        results = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc_roc': auc, 'avg_precision': ap
        }
        all_results[behavior][clf_name] = results
        all_models[behavior][clf_name] = {
            'model': clf, 'y_pred': y_pred, 'y_prob': y_prob,
            'y_test': y_te
        }
        print(f"    Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  AP={ap:.3f}")

# ─── 5. Cross-Validation for Random Forest ──────────────────────────────────
print("\n" + "=" * 60)
print("5. CROSS-VALIDATION (Random Forest)")
print("=" * 60)

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for behavior, y_all in [('Attack', y_attack), ('Sniffing', y_sniff)]:
    fold_scores = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_arr, y_all)):
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                    random_state=SEED, n_jobs=-1)
        rf.fit(X_arr[tr_idx], y_all[tr_idx])
        y_prob_cv = rf.predict_proba(X_arr[te_idx])[:, 1]
        y_pred_cv = rf.predict(X_arr[te_idx])
        fold_scores.append({
            'f1': f1_score(y_all[te_idx], y_pred_cv),
            'auc': roc_auc_score(y_all[te_idx], y_prob_cv),
            'precision': precision_score(y_all[te_idx], y_pred_cv),
            'recall': recall_score(y_all[te_idx], y_pred_cv),
        })
    cv_results[behavior] = fold_scores
    means = {k: np.mean([s[k] for s in fold_scores]) for k in fold_scores[0]}
    stds = {k: np.std([s[k] for s in fold_scores]) for k in fold_scores[0]}
    print(f"{behavior}: F1={means['f1']:.3f}±{stds['f1']:.3f}  "
          f"AUC={means['auc']:.3f}±{stds['auc']:.3f}")

# ─── 6. Feature Importance ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. FEATURE IMPORTANCE")
print("=" * 60)

importance_tables = {}
for behavior in ['Attack', 'Sniffing']:
    rf_model = all_models[behavior]['Random Forest']['model']
    imp = rf_model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': imp})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    importance_tables[behavior] = imp_df
    imp_df.to_csv(os.path.join(OUT, f'feature_importance_{behavior}.csv'), index=False)
    print(f"\nTop 15 features for {behavior}:")
    print(imp_df.head(15).to_string(index=False))

# ─── 7. Reference Comparison ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. REFERENCE COMPARISON")
print("=" * 60)

ref_attack = ref_df['Attack'].values
ref_sniff = ref_df['Sniffing'].values
print(f"Reference Attack positives: {ref_attack.sum()}/{len(ref_attack)} ({100*ref_attack.mean():.1f}%)")
print(f"Reference Sniffing positives: {ref_sniff.sum()}/{len(ref_sniff)} ({100*ref_sniff.mean():.1f}%)")

# Compare label distributions
for behavior, y_full in [('Attack', y_attack), ('Sniffing', y_sniff)]:
    ref_y = ref_df[behavior].values
    print(f"\n{behavior}:")
    print(f"  Full dataset prevalence: {100*y_full.mean():.1f}%")
    print(f"  Reference prevalence:    {100*ref_y.mean():.1f}%")

# ─── 8. Save Results ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. SAVING RESULTS")
print("=" * 60)

# Save all metrics
results_json = {}
for behavior in all_results:
    results_json[behavior] = {}
    for clf_name in all_results[behavior]:
        results_json[behavior][clf_name] = {
            k: round(v, 4) for k, v in all_results[behavior][clf_name].items()
        }

with open(os.path.join(OUT, 'classification_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

# Save CV results
cv_json = {}
for behavior in cv_results:
    cv_json[behavior] = {
        'folds': cv_results[behavior],
        'mean': {k: round(np.mean([s[k] for s in cv_results[behavior]]), 4)
                 for k in cv_results[behavior][0]},
        'std': {k: round(np.std([s[k] for s in cv_results[behavior]]), 4)
                for k in cv_results[behavior][0]}
    }

with open(os.path.join(OUT, 'cv_results.json'), 'w') as f:
    json.dump(cv_json, f, indent=2)

print("Results saved to outputs/")

# ─── 9. Generate Figures ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. GENERATING FIGURES")
print("=" * 60)

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# --- Fig 1: Data overview - label distribution ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (behavior, y_all) in zip(axes, [('Attack', y_attack), ('Sniffing', y_sniff)]):
    counts = [np.sum(y_all == 0), np.sum(y_all == 1)]
    bars = ax.bar(['Absent (0)', 'Present (1)'], counts,
                  color=['#4C72B0', '#DD8452'], edgecolor='black', linewidth=0.5)
    ax.set_title(f'{behavior} Label Distribution')
    ax.set_ylabel('Frame Count')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{c}\n({100*c/len(y_all):.1f}%)', ha='center', va='bottom', fontsize=10)
fig.suptitle('Behavior Label Distribution (N=1737 frames)', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig1_label_distribution.png'), bbox_inches='tight')
plt.close()
print("  Fig 1: Label distribution")

# --- Fig 2: Temporal behavior trace ---
fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
for ax, (behavior, y_all) in zip(axes, [('Attack', y_attack), ('Sniffing', y_sniff)]):
    ax.fill_between(range(len(y_all)), y_all, alpha=0.6, color='#DD8452' if behavior == 'Attack' else '#4C72B0')
    ax.set_ylabel(behavior)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Absent', 'Present'])
axes[-1].set_xlabel('Frame')
fig.suptitle('Temporal Behavior Annotations', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig2_temporal_trace.png'), bbox_inches='tight')
plt.close()
print("  Fig 2: Temporal trace")

# --- Fig 3: Model comparison bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metrics_to_plot = ['f1', 'auc_roc', 'avg_precision']
metric_labels = ['F1 Score', 'AUC-ROC', 'Avg Precision']
colors = ['#4C72B0', '#55A868', '#C44E52']

for ax, behavior in zip(axes, ['Attack', 'Sniffing']):
    clf_names = list(all_results[behavior].keys())
    x = np.arange(len(clf_names))
    width = 0.25
    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        vals = [all_results[behavior][c][metric] for c in clf_names]
        ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Classifier')
    ax.set_ylabel('Score')
    ax.set_title(f'{behavior} Classification')
    ax.set_xticks(x + width)
    ax.set_xticklabels(clf_names, rotation=20, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)

fig.suptitle('Classifier Performance Comparison', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig3_model_comparison.png'), bbox_inches='tight')
plt.close()
print("  Fig 3: Model comparison")

# --- Fig 4: Confusion matrices (Random Forest) ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, behavior in zip(axes, ['Attack', 'Sniffing']):
    m = all_models[behavior]['Random Forest']
    cm = confusion_matrix(m['y_test'], m['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{behavior} — Random Forest')
fig.suptitle('Confusion Matrices', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig4_confusion_matrices.png'), bbox_inches='tight')
plt.close()
print("  Fig 4: Confusion matrices")

# --- Fig 5: Precision-Recall curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, behavior in zip(axes, ['Attack', 'Sniffing']):
    for clf_name, color in zip(classifiers.keys(), ['#4C72B0', '#55A868', '#C44E52', '#8172B3']):
        m = all_models[behavior][clf_name]
        prec_vals, rec_vals, _ = precision_recall_curve(m['y_test'], m['y_prob'])
        ap = all_results[behavior][clf_name]['avg_precision']
        ax.plot(rec_vals, prec_vals, label=f'{clf_name} (AP={ap:.3f})', color=color, linewidth=1.5)
    prevalence = m['y_test'].mean()
    ax.axhline(y=prevalence, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({prevalence:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{behavior}')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
fig.suptitle('Precision-Recall Curves', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig5_precision_recall.png'), bbox_inches='tight')
plt.close()
print("  Fig 5: Precision-recall curves")

# --- Fig 6: Top 20 feature importances ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, behavior in zip(axes, ['Attack', 'Sniffing']):
    top = importance_tables[behavior].head(20)
    ax.barh(range(len(top)-1, -1, -1), top['importance'].values,
            color='#4C72B0', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top)-1, -1, -1))
    ax.set_yticklabels(top['feature'].values, fontsize=8)
    ax.set_xlabel('Importance (Gini)')
    ax.set_title(f'{behavior}')
fig.suptitle('Top 20 Feature Importances (Random Forest)', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig6_feature_importance.png'), bbox_inches='tight')
plt.close()
print("  Fig 6: Feature importance")

# --- Fig 7: Cross-validation box plots ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, behavior in zip(axes, ['Attack', 'Sniffing']):
    data = cv_results[behavior]
    metrics = ['f1', 'auc', 'precision', 'recall']
    vals = [[s[m] for s in data] for m in metrics]
    bp = ax.boxplot(vals, labels=['F1', 'AUC', 'Precision', 'Recall'],
                    patch_artist=True, widths=0.5)
    colors_box = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{behavior}')
    ax.set_ylabel('Score')
fig.suptitle('5-Fold Cross-Validation (Random Forest)', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig7_cv_boxplots.png'), bbox_inches='tight')
plt.close()
print("  Fig 7: CV box plots")

# --- Fig 8: Feature correlation with targets ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, behavior, y_all in zip(axes, ['Attack', 'Sniffing'], [y_attack, y_sniff]):
    corrs = X.corrwith(pd.Series(y_all)).abs().sort_values(ascending=False)
    top = corrs.head(20)
    ax.barh(range(len(top)-1, -1, -1), top.values,
            color='#DD8452', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top)-1, -1, -1))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel('|Correlation|')
    ax.set_title(f'{behavior}')
fig.suptitle('Top 20 Features by Correlation with Target', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig8_feature_correlations.png'), bbox_inches='tight')
plt.close()
print("  Fig 8: Feature correlations")

# --- Fig 9: Confusion matrices for all models ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for row, behavior in enumerate(['Attack', 'Sniffing']):
    for col, clf_name in enumerate(classifiers.keys()):
        m = all_models[behavior][clf_name]
        cm = confusion_matrix(m['y_test'], m['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                    xticklabels=['Abs', 'Pres'], yticklabels=['Abs', 'Pres'])
        axes[row, col].set_title(f'{behavior}\n{clf_name}', fontsize=9)
        if col == 0:
            axes[row, col].set_ylabel('Actual')
        if row == 1:
            axes[row, col].set_xlabel('Predicted')
fig.suptitle('Confusion Matrices — All Models', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig9_all_confusion_matrices.png'), bbox_inches='tight')
plt.close()
print("  Fig 9: All confusion matrices")

# --- Fig 10: Key feature distributions by behavior ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
key_feats = ['centroid_distance', 'M1_Nose_velocity', 'M2_Nose_velocity',
             'M1_body_angle', 'M1_facing_M2_angle', 'M1_body_length']
for idx, (ax, feat_name) in enumerate(zip(axes.flat, key_feats)):
    for behavior, y_all, color in [('Attack', y_attack, '#DD8452'), ('Sniffing', y_sniff, '#4C72B0')]:
        present = X[feat_name][y_all == 1]
        absent = X[feat_name][y_all == 0]
        ax.hist(present, bins=50, alpha=0.5, color=color, label=f'{behavior}=1', density=True)
    ax.hist(X[feat_name][y_attack == 0], bins=50, alpha=0.3, color='gray', label='Neither', density=True)
    ax.set_title(feat_name.replace('_', ' '), fontsize=10)
    ax.legend(fontsize=7)
fig.suptitle('Key Feature Distributions by Behavior', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig10_feature_distributions.png'), bbox_inches='tight')
plt.close()
print("  Fig 10: Feature distributions")

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
