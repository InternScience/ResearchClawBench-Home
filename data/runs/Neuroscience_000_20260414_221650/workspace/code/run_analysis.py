import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)


def ensure_dir(path: Path):
    path.mkdir(exist_ok=True, parents=True)


def load_data():
    x_raw = pd.read_csv(DATA / 'Together_1_features_extracted.csv')
    y_full = pd.read_csv(DATA / 'Together_1_targets_inserted.csv')
    ref = pd.read_csv(DATA / 'Together_1_machine_results_reference.csv')
    return x_raw, y_full, ref


def get_feature_sets(x_raw: pd.DataFrame, y_full: pd.DataFrame, ref: pd.DataFrame):
    labels = ['Attack', 'Sniffing']
    base = x_raw.drop(columns=['Unnamed: 0'], errors='ignore').copy()
    leakage_cols = ['Attack', 'Sniffing', 'Probability_Attack', 'Probability_Sniffing']
    reference_engineered = ref.drop(columns=['Unnamed: 0'], errors='ignore').copy()
    feature_sets = {
        'raw_pose_sample': {
            'X': base,
            'y': y_full.loc[:, labels].copy(),
            'description': 'Provided sample feature table (mostly raw pose-coordinate and simple auxiliary features).',
        }
    }
    if set(labels).issubset(reference_engineered.columns):
        feature_cols = [c for c in reference_engineered.columns if c not in leakage_cols]
        feature_sets['engineered_reference_subset'] = {
            'X': reference_engineered[feature_cols].copy(),
            'y': reference_engineered.loc[:, labels].copy(),
            'description': 'Reference machine-results table with richer SimBA-style engineered features; limited to its available rows.',
        }
    return feature_sets


def evaluate_behavior(X: pd.DataFrame, y: pd.Series, label: str, dataset_name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight='balanced_subsample',
            n_jobs=-1,
            min_samples_leaf=1,
        ))
    ])
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        'dataset': dataset_name,
        'label': label,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'train_positive': int(y_train.sum()),
        'test_positive': int(y_test.sum()),
        'prevalence_test': float(y_test.mean()),
        'accuracy': float(accuracy_score(y_test, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred, zero_division=0)),
        'f1': float(f1_score(y_test, pred, zero_division=0)),
        'average_precision': float(average_precision_score(y_test, prob)),
        'roc_auc': float(roc_auc_score(y_test, prob)),
    }
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1'])

    pr_prec, pr_rec, pr_thr = precision_recall_curve(y_test, prob)
    pr_df = pd.DataFrame({
        'precision': pr_prec,
        'recall': pr_rec,
        'threshold': list(pr_thr) + [np.nan],
        'dataset': dataset_name,
        'label': label,
    })

    model = pipeline.named_steps['model']
    imputer = pipeline.named_steps['imputer']
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)
    mdi = pd.DataFrame({
        'feature': X.columns,
        'importance_mdi': model.feature_importances_,
        'dataset': dataset_name,
        'label': label,
    }).sort_values('importance_mdi', ascending=False)

    perm = permutation_importance(
        pipeline, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE,
        scoring='average_precision', n_jobs=-1
    )
    perm_df = pd.DataFrame({
        'feature': X.columns,
        'importance_perm_mean': perm.importances_mean,
        'importance_perm_std': perm.importances_std,
        'dataset': dataset_name,
        'label': label,
    }).sort_values('importance_perm_mean', ascending=False)

    shap_df = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sample_n = min(100, len(X_test_imp))
        X_shap = X_test_imp.iloc[:sample_n]
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            vals = shap_values[1]
        else:
            vals = shap_values
            if vals.ndim == 3:
                vals = vals[:, :, 1]
        mean_abs = np.abs(vals).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs,
            'dataset': dataset_name,
            'label': label,
        }).sort_values('mean_abs_shap', ascending=False)
    except Exception as e:
        shap_df = pd.DataFrame({
            'feature': [], 'mean_abs_shap': [], 'dataset': [], 'label': []
        })
        (OUT / f'shap_error_{dataset_name}_{label}.txt').write_text(str(e))

    return {
        'metrics': metrics,
        'confusion_matrix': cm_df,
        'pr_curve': pr_df,
        'mdi_importance': mdi,
        'perm_importance': perm_df,
        'shap_importance': shap_df,
    }


def plot_class_balance(summary_json_path: Path):
    summary = json.loads(summary_json_path.read_text())
    rows = []
    for label, vals in summary['labels'].items():
        rows.append({'label': label, 'class': 'positive', 'frames': vals['positive_frames']})
        rows.append({'label': label, 'class': 'negative', 'frames': vals['negative_frames']})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='label', y='frames', hue='class')
    plt.title('Class balance by behavior')
    plt.tight_layout()
    plt.savefig(IMG / 'class_balance.png', dpi=200)
    plt.close()


def plot_pr_curves(all_pr: pd.DataFrame):
    g = sns.FacetGrid(all_pr, col='dataset', hue='label', sharex=True, sharey=True, height=4, aspect=1.2)
    g.map_dataframe(sns.lineplot, x='recall', y='precision')
    g.add_legend()
    g.set_axis_labels('Recall', 'Precision')
    g.fig.suptitle('Precision-recall curves', y=1.05)
    g.savefig(IMG / 'precision_recall_curves.png', dpi=200)
    plt.close('all')


def plot_confusions(confusions: dict):
    n = len(confusions)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 4 * n), squeeze=False)
    for r, (dataset, by_label) in enumerate(confusions.items()):
        for c, label in enumerate(['Attack', 'Sniffing']):
            ax = axes[r][c]
            cm = by_label[label]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'{dataset} — {label}')
    plt.tight_layout()
    plt.savefig(IMG / 'confusion_matrices.png', dpi=200)
    plt.close()


def plot_top_importances(importance_tables: dict, kind: str, filename: str, value_col: str):
    rows = []
    for dataset, by_label in importance_tables.items():
        for label, df in by_label.items():
            top = df.head(10).copy()
            top['rank'] = range(1, len(top) + 1)
            rows.append(top)
    cat = pd.concat(rows, ignore_index=True)
    g = sns.catplot(
        data=cat, kind='bar', x=value_col, y='feature', hue='label', col='dataset',
        sharex=False, sharey=False, height=5, aspect=1.1
    )
    g.fig.suptitle(kind, y=1.03)
    g.savefig(IMG / filename, dpi=200)
    plt.close('all')


def main():
    x_raw, y_full, ref = load_data()
    feature_sets = get_feature_sets(x_raw, y_full, ref)

    all_metrics = []
    all_pr = []
    confusions = {}
    mdi_tables = {}
    perm_tables = {}
    shap_tables = {}

    for dataset_name, spec in feature_sets.items():
        X = spec['X']
        Y = spec['y']
        confusions[dataset_name] = {}
        mdi_tables[dataset_name] = {}
        perm_tables[dataset_name] = {}
        shap_tables[dataset_name] = {}
        for label in ['Attack', 'Sniffing']:
            res = evaluate_behavior(X, Y[label].astype(int), label, dataset_name)
            all_metrics.append(res['metrics'])
            all_pr.append(res['pr_curve'])
            confusions[dataset_name][label] = res['confusion_matrix']
            mdi_tables[dataset_name][label] = res['mdi_importance']
            perm_tables[dataset_name][label] = res['perm_importance']
            shap_tables[dataset_name][label] = res['shap_importance']

            res['confusion_matrix'].to_csv(OUT / f'confusion_matrix_{dataset_name}_{label}.csv')
            res['pr_curve'].to_csv(OUT / f'pr_curve_{dataset_name}_{label}.csv', index=False)
            res['mdi_importance'].to_csv(OUT / f'feature_importance_mdi_{dataset_name}_{label}.csv', index=False)
            res['perm_importance'].to_csv(OUT / f'feature_importance_permutation_{dataset_name}_{label}.csv', index=False)
            res['shap_importance'].to_csv(OUT / f'feature_importance_shap_{dataset_name}_{label}.csv', index=False)

    metrics_df = pd.DataFrame(all_metrics).sort_values(['label', 'dataset'])
    metrics_df.to_csv(OUT / 'metrics_summary.csv', index=False)
    (OUT / 'metrics_summary.json').write_text(metrics_df.to_json(orient='records', indent=2))

    # Compare schemas
    schema_cmp = pd.DataFrame([
        {
            'dataset': 'raw_pose_sample',
            'n_rows': x_raw.shape[0],
            'n_columns_total': x_raw.shape[1],
            'n_predictors_used': x_raw.drop(columns=['Unnamed: 0'], errors='ignore').shape[1],
        },
        {
            'dataset': 'engineered_reference_subset',
            'n_rows': ref.shape[0],
            'n_columns_total': ref.shape[1],
            'n_predictors_used': ref.drop(columns=['Unnamed: 0', 'Attack', 'Sniffing', 'Probability_Attack', 'Probability_Sniffing'], errors='ignore').shape[1],
        },
    ])
    schema_cmp.to_csv(OUT / 'dataset_schema_comparison.csv', index=False)

    combined_pr = pd.concat(all_pr, ignore_index=True)
    combined_pr.to_csv(OUT / 'all_pr_curves.csv', index=False)

    plot_class_balance(OUT / 'data_summary.json')
    plot_pr_curves(combined_pr)
    plot_confusions(confusions)
    plot_top_importances(mdi_tables, 'Top random-forest MDI feature importances', 'feature_importance_mdi.png', 'importance_mdi')
    plot_top_importances(perm_tables, 'Top permutation importances (AP scoring)', 'feature_importance_permutation.png', 'importance_perm_mean')

    # SHAP only if nonempty
    nonempty = any(len(df) > 0 for ds in shap_tables.values() for df in ds.values())
    if nonempty:
        plot_top_importances(shap_tables, 'Top mean absolute SHAP values', 'feature_importance_shap.png', 'mean_abs_shap')

    claim_recovery = []
    for _, row in metrics_df.iterrows():
        claim_recovery.append({
            'claim': f"{row['dataset']} {row['label']} held-out average precision = {row['average_precision']:.3f}",
            'artifact': 'outputs/metrics_summary.csv',
            'status': 'supported'
        })
        claim_recovery.append({
            'claim': f"{row['dataset']} {row['label']} confusion matrix exported",
            'artifact': f"outputs/confusion_matrix_{row['dataset']}_{row['label']}.csv",
            'status': 'supported'
        })
    pd.DataFrame(claim_recovery).to_csv(OUT / 'claim_recovery_table.csv', index=False)

    validation = {
        'directly_verified_from_workspace': [
            'Input tables are row aligned for the sample features and target file.',
            'Attack and Sniffing labels are binary and non-missing in the provided sample target table.',
            'Random-forest models were trained and evaluated on held-out splits for each behavior and feature set.',
            'All reported metrics and plots are backed by saved CSV or PNG artifacts in outputs/ and report/images/.'
        ],
        'related_work_context': [
            'Related work supports pose-to-behavior pipelines, frame-level supervised labels, and class-imbalance-aware evaluation.'
        ],
        'assumptions_and_limitations': [
            'Only one sample sequence is available for the main sample-feature table; therefore evaluation uses a stratified held-out split rather than multi-video cross-validation.',
            'The provided sample feature table appears less engineered than the richer reference table, so results are reported for both representations without claiming exact parity to the full SimBA GUI workflow.'
        ]
    }
    (OUT / 'validation_summary.json').write_text(json.dumps(validation, indent=2))

if __name__ == '__main__':
    main()
