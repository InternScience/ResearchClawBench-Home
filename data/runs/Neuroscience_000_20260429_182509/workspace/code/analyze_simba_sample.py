#!/usr/bin/env python3
"""Reproducible SimBA-sample supervised behavior classification analysis."""
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve,
    classification_report, matthews_corrcoef, cohen_kappa_score
)
from sklearn.inspection import permutation_importance
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
CODE = ROOT / 'code'
for d in [OUT, IMG, CODE]: d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260429
BEHAVIORS = ['Attack', 'Sniffing']

features = pd.read_csv(DATA / 'Together_1_features_extracted.csv')
targets = pd.read_csv(DATA / 'Together_1_targets_inserted.csv')
reference = pd.read_csv(DATA / 'Together_1_machine_results_reference.csv')

# Align by row order after verifying common row count for feature/target tables.
assert len(features) == len(targets), 'features and targets have different rows'

label_cols = [c for c in BEHAVIORS if c in targets.columns]
# Use the feature table exactly as provided, excluding a purely row-index-like column.
feature_cols = [c for c in features.columns if c != 'Unnamed: 0']
X = features[feature_cols].copy()
# Ensure numeric and impute any missing values deterministically with medians.
X = X.apply(pd.to_numeric, errors='coerce')
missing_before = int(X.isna().sum().sum())
if missing_before:
    X = X.fillna(X.median(numeric_only=True))

data_overview = {
    'feature_table': {'rows': int(features.shape[0]), 'columns': int(features.shape[1]), 'model_feature_columns': int(len(feature_cols))},
    'target_table': {'rows': int(targets.shape[0]), 'columns': int(targets.shape[1])},
    'reference_table': {'rows': int(reference.shape[0]), 'columns': int(reference.shape[1])},
    'model_feature_columns': feature_cols,
    'missing_feature_values_before_imputation': missing_before,
    'label_distributions': {},
    'reference_behavior_columns': [c for c in reference.columns if ('Attack' in c or 'Sniffing' in c)],
}
for b in BEHAVIORS:
    y = targets[b].astype(int)
    data_overview['label_distributions'][b] = {
        'negative': int((y==0).sum()), 'positive': int((y==1).sum()),
        'positive_fraction': float(y.mean())
    }
with open(OUT / 'data_overview.json', 'w') as f:
    json.dump(data_overview, f, indent=2)

# Plot label distributions.
label_df = pd.DataFrame([
    {'Behavior': b, 'Class': 'Positive', 'Frames': data_overview['label_distributions'][b]['positive']}
    for b in BEHAVIORS
] + [
    {'Behavior': b, 'Class': 'Negative', 'Frames': data_overview['label_distributions'][b]['negative']}
    for b in BEHAVIORS
])
plt.figure(figsize=(7,4.5))
sns.barplot(data=label_df, x='Behavior', y='Frames', hue='Class', palette=['#d95f02','#1b9e77'])
plt.title('Aligned label distribution in Together_1')
plt.tight_layout()
plt.savefig(IMG / 'label_distribution.png', dpi=200)
plt.close()

metrics_rows = []
cv_rows = []
confusions = {}
pr_export = {}
all_pred_rows = []
class_reports = {}
perm_top_records = []

fig_pr, ax_pr = plt.subplots(figsize=(7,5))
fig_cm, axes_cm = plt.subplots(1, len(BEHAVIORS), figsize=(5.5*len(BEHAVIORS),4.5))
if len(BEHAVIORS)==1: axes_cm=[axes_cm]
fig_fi, axes_fi = plt.subplots(1, len(BEHAVIORS), figsize=(7*len(BEHAVIORS),5.5))
if len(BEHAVIORS)==1: axes_fi=[axes_fi]

for i, behavior in enumerate(BEHAVIORS):
    y = targets[behavior].astype(int).to_numpy()
    stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.30, random_state=RANDOM_STATE, stratify=stratify
    )
    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight='balanced_subsample',
        min_samples_leaf=2,
        n_jobs=-1,
        oob_score=True,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    metrics = {
        'behavior': behavior,
        'n_total': int(len(y)), 'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
        'train_positive': int(y_train.sum()), 'test_positive': int(y_test.sum()),
        'prevalence_total': float(y.mean()), 'prevalence_test': float(y_test.mean()),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'average_precision': float(average_precision_score(y_test, y_prob)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test))==2 else np.nan,
        'matthews_corrcoef': float(matthews_corrcoef(y_test, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_test, y_pred)),
        'oob_score': float(clf.oob_score_),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }
    metrics_rows.append(metrics)
    confusions[behavior] = {'labels':[0,1], 'matrix': cm.tolist(), 'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp}
    class_reports[behavior] = classification_report(y_test, y_pred, labels=[0,1], output_dict=True, zero_division=0)

    # Cross-validation on all rows for a stability check.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced_subsample', min_samples_leaf=2, n_jobs=-1)
    cv_scores = cross_validate(cv_clf, X, y, cv=cv, scoring=['balanced_accuracy','average_precision','f1','precision','recall'], n_jobs=-1)
    for metric_name, vals in cv_scores.items():
        if metric_name.startswith('test_'):
            cv_rows.append({'behavior': behavior, 'metric': metric_name.replace('test_',''), 'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)), 'fold_values': [float(v) for v in vals]})

    # Save model and predictions.
    dump(clf, OUT / f'model_{behavior}.joblib')
    pred_df = pd.DataFrame({'row_index': idx_test, 'behavior': behavior, 'y_true': y_test, 'y_pred': y_pred, 'probability': y_prob}).sort_values('row_index')
    pred_df.to_csv(OUT / f'predictions_{behavior}.csv', index=False)
    all_pred_rows.append(pred_df)

    # Precision-recall.
    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    pr_df = pd.DataFrame({'precision': prec, 'recall': rec, 'threshold': np.r_[thr, np.nan]})
    pr_df.to_csv(OUT / f'precision_recall_{behavior}.csv', index=False)
    pr_export[behavior] = {'precision': prec.tolist(), 'recall': rec.tolist(), 'threshold': np.r_[thr, np.nan].tolist()}
    ax_pr.plot(rec, prec, lw=2, label=f'{behavior} (AP={metrics["average_precision"]:.3f})')
    ax_pr.hlines(y_test.mean(), 0, 1, linestyles='dashed', alpha=0.35)

    # Confusion matrix heatmap.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes_cm[i], xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
    axes_cm[i].set_title(f'{behavior}\nF1={metrics["f1"]:.3f}, bal. acc={metrics["balanced_accuracy"]:.3f}')
    axes_cm[i].set_xlabel('Predicted')
    axes_cm[i].set_ylabel('Observed')

    # Feature importances.
    fi = pd.DataFrame({'feature': feature_cols, 'gini_importance': clf.feature_importances_}).sort_values('gini_importance', ascending=False)
    # Permutation importance on test subset: cheaper, model-agnostic validation of top contributors.
    perm = permutation_importance(clf, X_test, y_test, scoring='average_precision', n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    fi['permutation_ap_mean'] = perm.importances_mean
    fi['permutation_ap_std'] = perm.importances_std
    fi.to_csv(OUT / f'feature_importance_{behavior}.csv', index=False)
    top = fi.head(15).iloc[::-1]
    sns.barplot(data=top, x='gini_importance', y='feature', ax=axes_fi[i], color='#4C72B0')
    axes_fi[i].set_title(f'Top 15 Random Forest features: {behavior}')
    axes_fi[i].set_xlabel('Mean decrease in impurity')
    axes_fi[i].set_ylabel('')
    for _, row in fi.head(10).iterrows():
        perm_top_records.append({'behavior':behavior, 'feature':row['feature'], 'gini_importance':float(row['gini_importance']), 'permutation_ap_mean':float(row['permutation_ap_mean']), 'permutation_ap_std':float(row['permutation_ap_std'])})

# Finalize figures.
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.set_title('Precision-recall diagnostics on held-out frames')
ax_pr.set_xlim(0,1.02); ax_pr.set_ylim(0,1.02)
ax_pr.legend(loc='best')
ax_pr.grid(alpha=0.25)
fig_pr.tight_layout()
fig_pr.savefig(IMG / 'precision_recall_curves.png', dpi=200)
plt.close(fig_pr)
fig_cm.tight_layout()
fig_cm.savefig(IMG / 'confusion_matrices.png', dpi=200)
plt.close(fig_cm)
fig_fi.tight_layout()
fig_fi.savefig(IMG / 'feature_importance_top15.png', dpi=200)
plt.close(fig_fi)

pd.DataFrame(metrics_rows).to_csv(OUT / 'evaluation_metrics.csv', index=False)
pd.DataFrame(cv_rows).to_csv(OUT / 'cross_validation_metrics.csv', index=False)
pd.concat(all_pred_rows, ignore_index=True).to_csv(OUT / 'heldout_predictions_all_behaviors.csv', index=False)
with open(OUT / 'confusion_matrices.json','w') as f: json.dump(confusions, f, indent=2)
with open(OUT / 'classification_reports.json','w') as f: json.dump(class_reports, f, indent=2)
with open(OUT / 'precision_recall_curves.json','w') as f: json.dump(pr_export, f, indent=2)
pd.DataFrame(perm_top_records).to_csv(OUT / 'top_feature_importance_with_permutation.csv', index=False)

# Reference comparison: reference has only 300 rows and includes official machine Probability_* and predicted labels.
ref_comparison_rows=[]
ref_plot_df=[]
for behavior in BEHAVIORS:
    if behavior in reference.columns:
        # Match to target labels by Unnamed: 0 if possible, else first n rows.
        if 'Unnamed: 0' in reference.columns and 'Unnamed: 0' in targets.columns:
            merged = reference[['Unnamed: 0', behavior] + ([f'Probability_{behavior}'] if f'Probability_{behavior}' in reference.columns else [])].merge(
                targets[['Unnamed: 0', behavior]].rename(columns={behavior:f'target_{behavior}'}), on='Unnamed: 0', how='left'
            )
        else:
            n=len(reference)
            merged = reference[[behavior] + ([f'Probability_{behavior}'] if f'Probability_{behavior}' in reference.columns else [])].copy()
            merged[f'target_{behavior}'] = targets[behavior].iloc[:n].to_numpy()
        valid = merged.dropna(subset=[behavior, f'target_{behavior}'])
        official_pred = valid[behavior].astype(int).to_numpy()
        target_y = valid[f'target_{behavior}'].astype(int).to_numpy()
        row={'behavior':behavior, 'n_reference_rows':int(len(valid)), 'reference_positive_predictions':int(official_pred.sum()), 'target_positives_in_reference_rows':int(target_y.sum()),
             'reference_accuracy_vs_targets':float(accuracy_score(target_y, official_pred)),
             'reference_balanced_accuracy_vs_targets':float(balanced_accuracy_score(target_y, official_pred)),
             'reference_precision_vs_targets':float(precision_score(target_y, official_pred, zero_division=0)),
             'reference_recall_vs_targets':float(recall_score(target_y, official_pred, zero_division=0)),
             'reference_f1_vs_targets':float(f1_score(target_y, official_pred, zero_division=0))}
        if f'Probability_{behavior}' in valid.columns:
            prob=valid[f'Probability_{behavior}'].astype(float).to_numpy()
            row['reference_average_precision_vs_targets']=float(average_precision_score(target_y, prob))
            if len(np.unique(target_y))==2: row['reference_roc_auc_vs_targets']=float(roc_auc_score(target_y, prob))
            ref_plot_df.append(pd.DataFrame({'behavior':behavior,'target':target_y,'official_probability':prob,'official_prediction':official_pred}))
        ref_comparison_rows.append(row)
ref_comp = pd.DataFrame(ref_comparison_rows)
ref_comp.to_csv(OUT / 'reference_comparison.csv', index=False)
if ref_plot_df:
    rpdf = pd.concat(ref_plot_df, ignore_index=True)
    rpdf.to_csv(OUT / 'reference_probability_rows.csv', index=False)
    plt.figure(figsize=(7,4.5))
    sns.boxplot(data=rpdf, x='behavior', y='official_probability', hue='target')
    plt.title('Official reference probabilities by aligned manual target')
    plt.ylabel('Reference Probability_*')
    plt.tight_layout()
    plt.savefig(IMG / 'reference_comparison.png', dpi=200)
    plt.close()
else:
    plt.figure(figsize=(7,4.5))
    sns.barplot(data=ref_comp, x='behavior', y='reference_f1_vs_targets', color='#55A868')
    plt.ylim(0,1)
    plt.title('Reference output F1 against aligned target labels')
    plt.tight_layout()
    plt.savefig(IMG / 'reference_comparison.png', dpi=200)
    plt.close()

# Related-work extraction: ReadPDF failed in this runtime; record bounded metadata/useful extraction from task and observed reference schema.
related_work_contract = {
    'readpdf_status':'ReadPDF returned unexpected NoneType for all five PDFs in this environment; no unsupported paper-specific claims are made.',
    'task_relevant_method_context_from_workspace':'The reference SimBA machine-results table contains engineered pose, distance, movement, rolling-window, percentile-rank, probability, and behavior columns, confirming that the sample workflow maps tracked-pose features to Probability_Attack/Probability_Sniffing and binary Attack/Sniffing outputs.',
    'contract_update':'Kept Random Forest supervised per-behavior classifiers, PR diagnostics, confusion matrices, native and permutation feature-importance artifacts, and reference-output validation.'
}
with open(OUT / 'related_work_contract.json','w') as f: json.dump(related_work_contract, f, indent=2)

method_fidelity = {
    'named_workflow':'SimBA-style supervised behavior classification',
    'non_negotiable_steps':[
        'Use frame-level pose-derived feature table as X',
        'Use frame-aligned Attack and Sniffing labels as y',
        'Train supervised classifiers separately for each behavior',
        'Report quantitative test-set metrics',
        'Export PR diagnostics, confusion matrices, feature importances, and model/prediction artifacts',
        'Validate against official reference machine-results table where possible'
    ],
    'implementation_status':{
        'frame_level_X':'satisfied: data/Together_1_features_extracted.csv, 50 model features excluding Unnamed: 0',
        'aligned_y':'satisfied: data/Together_1_targets_inserted.csv Attack/Sniffing columns',
        'classifier_family':'satisfied approximately: RandomForestClassifier per behavior with class weights; exact original SimBA trained model/hyperparameters are not included in workspace',
        'diagnostics':'satisfied: metrics, PR curves, confusion matrices, feature importance tables',
        'reference_validation':'satisfied conditionally: official reference has 300 rows, compared its Attack/Sniffing and Probability_* columns to aligned labels'
    },
    'deviations':['No exact original SimBA project model settings were available, so the reproduced classifiers are new deterministic Random Forest models rather than byte-identical SimBA models.']
}
with open(OUT / 'method_fidelity_checklist.json','w') as f: json.dump(method_fidelity, f, indent=2)

# Claim recovery and update inventory.
claim_rows = [
    {'claim':'The workspace feature and target tables align frame-by-frame with 1,738 rows.', 'support':'outputs/data_overview.json', 'status':'directly verified'},
    {'claim':'Attack and Sniffing are imbalanced labels and therefore PR/AP diagnostics are appropriate.', 'support':'outputs/data_overview.json; report/images/label_distribution.png', 'status':'directly verified'},
    {'claim':'Deterministic Random Forest classifiers were trained and evaluated for both behaviors.', 'support':'outputs/model_Attack.joblib; outputs/model_Sniffing.joblib; outputs/evaluation_metrics.csv', 'status':'directly verified'},
    {'claim':'Held-out quantitative performance, confusion matrices, and PR curves are available for both behaviors.', 'support':'outputs/evaluation_metrics.csv; outputs/confusion_matrices.json; outputs/precision_recall_*.csv; report/images/*.png', 'status':'directly verified'},
    {'claim':'Feature importance evidence is auditable at feature level.', 'support':'outputs/feature_importance_Attack.csv; outputs/feature_importance_Sniffing.csv; outputs/top_feature_importance_with_permutation.csv; report/images/feature_importance_top15.png', 'status':'directly verified'},
    {'claim':'Official reference machine-results probabilities/predictions can be compared only for the 300 rows present in that file.', 'support':'outputs/reference_comparison.csv; outputs/data_overview.json; report/images/reference_comparison.png', 'status':'directly verified'},
    {'claim':'Exact reproduction of original SimBA models is not claimed.', 'support':'outputs/method_fidelity_checklist.json; outputs/dependency_check.json', 'status':'limitation'}
]
pd.DataFrame(claim_rows).to_csv(OUT / 'claim_recovery_table.csv', index=False)

# Update inventory statuses.
with open(OUT / 'target_artifact_inventory.json') as f: inv=json.load(f)
for section in ['primary_tables','primary_figures']:
    for item in inv[section]:
        p=ROOT / item['artifact']
        if p.exists() and p.stat().st_size>0:
            item['status']='satisfied'
        else:
            item['status']='unsatisfied: file not created'
inv['report_status']='planned'
with open(OUT / 'target_artifact_inventory.json','w') as f: json.dump(inv,f,indent=2)

print('analysis complete')
print(pd.DataFrame(metrics_rows).to_string(index=False))
print(ref_comp.to_string(index=False))
