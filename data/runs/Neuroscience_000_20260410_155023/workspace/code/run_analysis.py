
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path('.').resolve()
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150

features = pd.read_csv(DATA / 'Together_1_features_extracted.csv')
targets = pd.read_csv(DATA / 'Together_1_targets_inserted.csv')
ref = pd.read_csv(DATA / 'Together_1_machine_results_reference.csv')

if 'Unnamed: 0' in features.columns and 'Unnamed: 0' in targets.columns:
    merged = features.merge(targets[['Unnamed: 0', 'Attack', 'Sniffing']], on='Unnamed: 0', how='inner')
else:
    merged = features.copy()
    merged[['Attack', 'Sniffing']] = targets[['Attack', 'Sniffing']]

drop_cols = [c for c in ['Unnamed: 0'] if c in merged.columns]
X = merged.drop(columns=drop_cols + ['Attack', 'Sniffing'])
y_df = merged[['Attack', 'Sniffing']].copy()
X = X.replace([np.inf, -np.inf], np.nan)

summary = {
    'n_frames': int(len(merged)),
    'n_features': int(X.shape[1]),
    'targets': {col: y_df[col].value_counts().to_dict() for col in y_df.columns},
    'missing_values_total': int(X.isna().sum().sum()),
    'reference_shape': list(ref.shape),
}
(OUT / 'data_summary.json').write_text(json.dumps(summary, indent=2))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
counts = pd.DataFrame({
    'Attack': y_df['Attack'].value_counts().sort_index(),
    'Sniffing': y_df['Sniffing'].value_counts().sort_index(),
}).T
counts.columns = ['Negative', 'Positive']
counts.plot(kind='bar', stacked=True, ax=axes[0], color=['#9aa0a6','#d55e00'])
axes[0].set_title('Class balance')
axes[0].set_ylabel('Frames')
axes[0].legend(frameon=False)

sample_feats = [c for c in ['Nose_1_x', 'Nose_1_y', 'Feature_1', 'Feature_2'] if c in X.columns]
for i, col in enumerate(sample_feats[:2], start=1):
    sns.histplot(X[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution: {col}')
plt.tight_layout()
fig.savefig(IMG / 'data_overview.png', bbox_inches='tight')
plt.close(fig)

results_rows = []
all_pred_rows = []
strat = y_df['Attack'].astype(str) + '_' + y_df['Sniffing'].astype(str)
train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=42, shuffle=True, stratify=strat)
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

for label in ['Attack', 'Sniffing']:
    y_train = y_df.iloc[train_idx][label]
    y_test = y_df.iloc[test_idx][label]
    models = {
        'logistic_regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
        ]),
        'random_forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=-1, random_state=42))
        ]),
        'extra_trees': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42))
        ])
    }
    scores = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        ap = average_precision_score(y_test, prob)
        roc = roc_auc_score(y_test, prob)
        scores.append((name, ap, roc))
        fitted[name] = (model, prob)
    best_name, _, _ = sorted(scores, key=lambda x: (x[1], x[2]), reverse=True)[0]
    best_model, y_prob = fitted[best_name]

    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best_thr = float(thr[np.nanargmax(f1s)]) if len(thr) else 0.5
    y_pred = (y_prob >= best_thr).astype(int)

    metrics = {
        'label': label, 'best_model': best_name, 'threshold': best_thr,
        'average_precision': float(average_precision_score(y_test, y_prob)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_test, y_pred)),
        'support_positive': int(y_test.sum()), 'support_total': int(len(y_test))
    }
    results_rows.append(metrics)
    all_pred_rows.append(pd.DataFrame({'frame_index': X_test.index, 'label': label, 'y_true': y_test.values, 'y_prob': y_prob, 'y_pred': y_pred, 'best_model': best_name, 'threshold': best_thr}))

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, label=f'AP={metrics["average_precision"]:.3f}')
    plt.axhline(y_test.mean(), color='gray', ls='--', label=f'Prevalence={y_test.mean():.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision–Recall: {label}')
    plt.legend(frameon=False); plt.tight_layout(); plt.savefig(IMG / f'pr_curve_{label.lower()}.png', bbox_inches='tight'); plt.close()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.title(f'Confusion matrix: {label}'); plt.tight_layout(); plt.savefig(IMG / f'confusion_matrix_{label.lower()}.png', bbox_inches='tight'); plt.close()

    score_df = pd.DataFrame({'y_true': y_test.values.astype(str), 'y_prob': y_prob})
    plt.figure(figsize=(6, 5))
    sns.kdeplot(data=score_df, x='y_prob', hue='y_true', common_norm=False, fill=True, alpha=0.35)
    plt.axvline(best_thr, color='black', ls='--', label=f'Threshold={best_thr:.3f}')
    plt.title(f'Predicted probability separation: {label}'); plt.xlabel('Predicted probability'); plt.legend(frameon=False, title='True label')
    plt.tight_layout(); plt.savefig(IMG / f'probability_density_{label.lower()}.png', bbox_inches='tight'); plt.close()

    clf = best_model.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        importance = pd.Series(clf.feature_importances_, index=X.columns)
    else:
        importance = pd.Series(np.abs(np.ravel(clf.coef_)), index=X.columns)
    imp_df = importance.sort_values(ascending=False).reset_index()
    imp_df.columns = ['feature', 'importance']
    imp_df.to_csv(OUT / f'feature_importance_{label.lower()}.csv', index=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df.head(15), y='feature', x='importance', color='#4c72b0')
    plt.title(f'Top feature importance: {label} ({best_name})'); plt.xlabel('Importance'); plt.ylabel('Feature')
    plt.tight_layout(); plt.savefig(IMG / f'feature_importance_{label.lower()}.png', bbox_inches='tight'); plt.close()
    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T.to_csv(OUT / f'classification_report_{label.lower()}.csv')

results_df = pd.DataFrame(results_rows)
results_df.to_csv(OUT / 'model_metrics_summary.csv', index=False)
pd.concat(all_pred_rows, ignore_index=True).to_csv(OUT / 'predictions_all_labels.csv', index=False)
comparison = {'reference_available': True, 'n_reference_rows': int(len(ref)), 'reference_probability_columns': [c for c in ref.columns if 'Probability_' in c]}
(OUT / 'reference_comparison.json').write_text(json.dumps(comparison, indent=2))
lines = ['# Analysis summary', '', f"Frames: {summary['n_frames']}", f"Features used: {summary['n_features']}", '']
for _, row in results_df.iterrows():
    lines += [f"## {row['label']}", f"- Best model: {row['best_model']}", f"- AP: {row['average_precision']:.3f}", f"- ROC AUC: {row['roc_auc']:.3f}", f"- Balanced accuracy: {row['balanced_accuracy']:.3f}", f"- F1: {row['f1']:.3f}", f"- Precision: {row['precision']:.3f}", f"- Recall: {row['recall']:.3f}", '']
(OUT / 'analysis_summary.md').write_text('\n'.join(lines))
print(results_df.to_string(index=False))
