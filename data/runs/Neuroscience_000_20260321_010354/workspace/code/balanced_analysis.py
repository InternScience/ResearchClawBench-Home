"""
Supplementary analysis: class-balanced classifiers for Sniffing behavior,
and additional diagnostics.
"""

import numpy as np
import pandas as pd
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs')
IMG = os.path.join(BASE, 'report', 'images')
SEED = 42
np.random.seed(SEED)

# Load engineered features and targets
X = pd.read_csv(os.path.join(OUT, 'engineered_features.csv'))
tgt_df = pd.read_csv(os.path.join(BASE, 'data', 'Together_1_targets_inserted.csv'), index_col=0)
y_attack = tgt_df['Attack'].values
y_sniff = tgt_df['Sniffing'].values
feature_names = X.columns.tolist()
X_arr = X.values

# Split
X_train, X_test, y_attack_train, y_attack_test = train_test_split(
    X_arr, y_attack, test_size=0.2, random_state=SEED, stratify=y_attack)
_, _, y_sniff_train, y_sniff_test = train_test_split(
    X_arr, y_sniff, test_size=0.2, random_state=SEED, stratify=y_sniff)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ─── Balanced classifiers ────────────────────────────────────────────────────
print("=" * 60)
print("BALANCED CLASSIFIERS")
print("=" * 60)

classifiers_balanced = {
    'RF (balanced)': RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                            class_weight='balanced', random_state=SEED, n_jobs=-1),
    'GB (balanced)': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                 learning_rate=0.1, random_state=SEED),
    'LR (balanced)': LogisticRegression(max_iter=2000, class_weight='balanced',
                                        random_state=SEED, C=1.0),
    'SVM (balanced)': SVC(kernel='rbf', probability=True, class_weight='balanced',
                          random_state=SEED, C=1.0),
}

behaviors = {'Attack': (y_attack_train, y_attack_test),
             'Sniffing': (y_sniff_train, y_sniff_test)}

balanced_results = {}
balanced_models = {}

for behavior, (y_tr, y_te) in behaviors.items():
    print(f"\n--- {behavior} ---")
    balanced_results[behavior] = {}
    balanced_models[behavior] = {}

    for clf_name, clf in classifiers_balanced.items():
        if 'LR' in clf_name or 'SVM' in clf_name:
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

        balanced_results[behavior][clf_name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc_roc': auc, 'avg_precision': ap
        }
        balanced_models[behavior][clf_name] = {
            'y_pred': y_pred, 'y_prob': y_prob, 'y_test': y_te, 'model': clf
        }
        print(f"  {clf_name}: Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

# Save balanced results
with open(os.path.join(OUT, 'balanced_classification_results.json'), 'w') as f:
    json.dump({b: {c: {k: round(v, 4) for k, v in m.items()}
                   for c, m in r.items()} for b, r in balanced_results.items()}, f, indent=2)

# ─── CV for balanced RF ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BALANCED RF CROSS-VALIDATION")
print("=" * 60)

cv_balanced = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for behavior, y_all in [('Attack', y_attack), ('Sniffing', y_sniff)]:
    fold_scores = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_arr, y_all)):
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                    class_weight='balanced', random_state=SEED, n_jobs=-1)
        rf.fit(X_arr[tr_idx], y_all[tr_idx])
        y_prob_cv = rf.predict_proba(X_arr[te_idx])[:, 1]
        y_pred_cv = rf.predict(X_arr[te_idx])
        fold_scores.append({
            'f1': f1_score(y_all[te_idx], y_pred_cv),
            'auc': roc_auc_score(y_all[te_idx], y_prob_cv),
            'precision': precision_score(y_all[te_idx], y_pred_cv),
            'recall': recall_score(y_all[te_idx], y_pred_cv),
        })
    cv_balanced[behavior] = fold_scores
    means = {k: np.mean([s[k] for s in fold_scores]) for k in fold_scores[0]}
    stds = {k: np.std([s[k] for s in fold_scores]) for k in fold_scores[0]}
    print(f"{behavior}: F1={means['f1']:.3f}±{stds['f1']:.3f}  AUC={means['auc']:.3f}±{stds['auc']:.3f}")

with open(os.path.join(OUT, 'cv_balanced_results.json'), 'w') as f:
    json.dump({b: {'folds': s, 'mean': {k: round(np.mean([f[k] for f in s]), 4) for k in s[0]},
                   'std': {k: round(np.std([f[k] for f in s]), 4) for k in s[0]}}
               for b, s in cv_balanced.items()}, f, indent=2)

# ─── Feature importance for balanced RF (per behavior) ────────────────────
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Balanced RF, per behavior)")
print("=" * 60)

importance_balanced = {}
for behavior, y_all in [('Attack', y_attack), ('Sniffing', y_sniff)]:
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                class_weight='balanced', random_state=SEED, n_jobs=-1)
    rf.fit(X_arr, y_all)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    importance_balanced[behavior] = imp_df
    imp_df.to_csv(os.path.join(OUT, f'feature_importance_balanced_{behavior}.csv'), index=False)
    print(f"\nTop 15 for {behavior}:")
    print(imp_df.head(15).to_string(index=False))

# ─── Figures ──────────────────────────────────────────────────────────────────
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# Fig 11: Balanced vs unbalanced comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Load unbalanced results
with open(os.path.join(OUT, 'classification_results.json')) as f:
    unbal_results = json.load(f)

for ax, behavior in zip(axes, ['Attack', 'Sniffing']):
    models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'SVM (RBF)']
    bal_models = ['RF (balanced)', 'GB (balanced)', 'LR (balanced)', 'SVM (balanced)']
    x = np.arange(len(models))
    width = 0.35
    f1_unbal = [unbal_results[behavior][m]['f1'] for m in models]
    f1_bal = [balanced_results[behavior][m]['f1'] for m in bal_models]
    ax.bar(x - width/2, f1_unbal, width, label='Default', color='#4C72B0', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, f1_bal, width, label='Balanced', color='#DD8452', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Classifier')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'{behavior}')
    ax.set_xticks(x)
    ax.set_xticklabels(['RF', 'GB', 'LR', 'SVM'], rotation=0)
    ax.legend()
    ax.set_ylim(0, 1.05)
fig.suptitle('Default vs Class-Balanced Classifiers (F1)', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig11_balanced_comparison.png'), bbox_inches='tight')
plt.close()
print("\nFig 11: Balanced comparison")

# Fig 12: Confusion matrices for balanced RF
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, behavior in zip(axes, ['Attack', 'Sniffing']):
    m = balanced_models[behavior]['RF (balanced)']
    cm = confusion_matrix(m['y_test'], m['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{behavior} — RF (balanced)')
fig.suptitle('Confusion Matrices — Balanced Random Forest', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig12_balanced_confusion.png'), bbox_inches='tight')
plt.close()
print("Fig 12: Balanced confusion matrices")

# Fig 13: PR curves for balanced models (Sniffing focus)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
for (clf_name, m), color in zip(balanced_models['Sniffing'].items(), colors):
    prec_vals, rec_vals, _ = precision_recall_curve(m['y_test'], m['y_prob'])
    ap = balanced_results['Sniffing'][clf_name]['avg_precision']
    ax.plot(rec_vals, prec_vals, label=f'{clf_name} (AP={ap:.3f})', color=color, linewidth=1.5)
prevalence = y_sniff_test.mean()
ax.axhline(y=prevalence, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({prevalence:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Sniffing — Precision-Recall (Balanced Classifiers)')
ax.legend(fontsize=9)
ax.set_xlim([0, 1.02])
ax.set_ylim([0, 1.02])
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig13_sniffing_balanced_pr.png'), bbox_inches='tight')
plt.close()
print("Fig 13: Sniffing balanced PR curves")

# Fig 14: Feature importance comparison (Attack vs Sniffing)
fig, ax = plt.subplots(figsize=(10, 6))
top_attack = importance_balanced['Attack'].head(20)['feature'].tolist()
top_sniff = importance_balanced['Sniffing'].head(20)['feature'].tolist()
all_top = list(dict.fromkeys(top_attack[:15] + top_sniff[:15]))[:20]

att_imp = importance_balanced['Attack'].set_index('feature').loc[all_top, 'importance'].values
snf_imp = importance_balanced['Sniffing'].set_index('feature').loc[all_top, 'importance'].values

x = np.arange(len(all_top))
width = 0.35
ax.barh(x - width/2, att_imp, width, label='Attack', color='#DD8452', edgecolor='black', linewidth=0.5)
ax.barh(x + width/2, snf_imp, width, label='Sniffing', color='#4C72B0', edgecolor='black', linewidth=0.5)
ax.set_yticks(x)
ax.set_yticklabels(all_top, fontsize=8)
ax.set_xlabel('Importance (Gini)')
ax.set_title('Feature Importance Comparison: Attack vs Sniffing')
ax.legend()
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(IMG, 'fig14_importance_comparison.png'), bbox_inches='tight')
plt.close()
print("Fig 14: Importance comparison")

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

summary_rows = []
for behavior in ['Attack', 'Sniffing']:
    for clf_name in balanced_results[behavior]:
        r = balanced_results[behavior][clf_name]
        summary_rows.append({
            'Behavior': behavior, 'Classifier': clf_name,
            'Accuracy': f"{r['accuracy']:.3f}",
            'Precision': f"{r['precision']:.3f}",
            'Recall': f"{r['recall']:.3f}",
            'F1': f"{r['f1']:.3f}",
            'AUC-ROC': f"{r['auc_roc']:.3f}",
        })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUT, 'summary_table.csv'), index=False)

print("\nDone!")
