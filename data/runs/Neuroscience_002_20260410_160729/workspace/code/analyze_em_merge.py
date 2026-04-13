import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / 'data' / 'train_simulated.csv'
TEST_PATH = ROOT / 'data' / 'test_simulated.csv'
OUTPUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'
FEATURES = [str(i) for i in range(20)]
TARGET = 'label'
GROUP = 'degradation'
RANDOM_STATE = 42


sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def make_preprocessor(include_degradation=True):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    transformers = [('num', numeric_transformer, FEATURES)]
    if include_degradation:
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])
        transformers.append(('cat', cat_transformer, [GROUP]))
    return ColumnTransformer(transformers=transformers)


def make_models():
    models = {
        'logistic_regression': Pipeline([
            ('prep', make_preprocessor(True)),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'random_forest': Pipeline([
            ('prep', make_preprocessor(True)),
            ('clf', RandomForestClassifier(
                n_estimators=120,
                max_depth=14,
                min_samples_leaf=4,
                n_jobs=-1,
                class_weight='balanced_subsample',
                random_state=RANDOM_STATE,
            ))
        ]),
        'hist_gradient_boosting': Pipeline([
            ('prep', make_preprocessor(True)),
            ('clf', HistGradientBoostingClassifier(
                max_depth=5,
                learning_rate=0.08,
                max_iter=160,
                random_state=RANDOM_STATE,
            ))
        ]),
    }
    return models


def optimal_threshold(y_true, prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        scores.append(f1_score(y_true, pred))
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def compute_metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        'roc_auc': float(roc_auc_score(y_true, prob)),
        'pr_auc': float(average_precision_score(y_true, prob)),
        'accuracy': float(accuracy_score(y_true, pred)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'brier': float(brier_score_loss(y_true, prob)),
        'threshold': float(thr),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def evaluate_models(train):
    X = train[FEATURES + [GROUP]]
    y = train[TARGET].astype(int)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    models = make_models()
    rows = []
    cv_predictions = {}
    for name, model in models.items():
        prob = cross_val_predict(model, X, y, cv=skf, method='predict_proba', n_jobs=1)[:, 1]
        thr, _ = optimal_threshold(y, prob)
        metrics_05 = compute_metrics(y, prob, 0.5)
        metrics_opt = compute_metrics(y, prob, thr)
        rows.append({
            'model': name,
            **{f'cv05_{k}': v for k, v in metrics_05.items() if k not in ['tn','fp','fn','tp','threshold']},
            **{f'cvopt_{k}': v for k, v in metrics_opt.items() if k not in ['tn','fp','fn','tp']},
        })
        cv_predictions[name] = {'prob': prob, 'optimal_threshold': thr}
    results = pd.DataFrame(rows).sort_values(['cv05_pr_auc', 'cv05_roc_auc'], ascending=False)
    return results, cv_predictions


def fit_and_test(train, test, model_name):
    models = make_models()
    model = models[model_name]
    X_train = train[FEATURES + [GROUP]]
    y_train = train[TARGET].astype(int)
    X_test = test[FEATURES + [GROUP]]
    y_test = test[TARGET].astype(int)
    model.fit(X_train, y_train)
    prob_test = model.predict_proba(X_test)[:, 1]
    prob_train = model.predict_proba(X_train)[:, 1]
    thr, _ = optimal_threshold(y_train, prob_train)
    metrics_05 = compute_metrics(y_test, prob_test, 0.5)
    metrics_opt = compute_metrics(y_test, prob_test, thr)
    return model, prob_test, metrics_05, metrics_opt, thr


def plot_data_overview(train):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    label_counts = train[TARGET].astype(int).value_counts().sort_index()
    axes[0].bar(['Different neuron (0)', 'Same neuron (1)'], label_counts.values, color=['#4C72B0', '#DD8452'])
    axes[0].set_title('Class distribution in training set')
    axes[0].set_ylabel('Count')

    degr = train.groupby(GROUP)[TARGET].mean().sort_index()
    axes[1].bar(degr.index, degr.values, color='#55A868')
    axes[1].set_title('Positive rate by degradation type')
    axes[1].set_ylabel('P(label=1)')
    axes[1].tick_params(axis='x', rotation=25)

    corr = train[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    top = pd.concat([corr.head(5), corr.tail(5)])
    axes[2].barh(top.index.astype(str), top.values, color=['#C44E52' if v < 0 else '#8172B3' for v in top.values])
    axes[2].set_title('Most label-associated features')
    axes[2].set_xlabel('Pearson correlation with label')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'data_overview.png', bbox_inches='tight')
    plt.close(fig)


def plot_model_comparison(results):
    df = results[['model', 'cv05_roc_auc', 'cv05_pr_auc', 'cv05_f1']].melt(id_vars='model', var_name='metric', value_name='value')
    metric_map = {'cv05_roc_auc': 'ROC AUC', 'cv05_pr_auc': 'PR AUC', 'cv05_f1': 'F1 @ 0.5'}
    df['metric'] = df['metric'].map(metric_map)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='metric', y='value', hue='model', ax=ax)
    ax.set_title('Cross-validated model comparison on training set')
    ax.set_ylabel('Score')
    ax.set_xlabel('')
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'model_comparison.png', bbox_inches='tight')
    plt.close(fig)


def plot_roc_pr(y_true, prob):
    fpr, tpr, _ = roc_curve(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(fpr, tpr, lw=2, color='#4C72B0')
    axes[0].plot([0,1], [0,1], '--', color='gray')
    axes[0].set_title('Receiver operating characteristic')
    axes[0].set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate')
    axes[0].text(0.6, 0.1, f'AUC = {roc_auc_score(y_true, prob):.3f}', transform=axes[0].transAxes)

    axes[1].plot(recall, precision, lw=2, color='#DD8452')
    baseline = y_true.mean()
    axes[1].hlines(baseline, 0, 1, linestyles='--', color='gray', label='Class prevalence')
    axes[1].set_title('Precision-recall curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].text(0.05, 0.1, f'AP = {average_precision_score(y_true, prob):.3f}', transform=axes[1].transAxes)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'roc_pr_curves.png', bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrices(y_true, prob, thr_default, thr_opt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, thr, title in zip(axes, [thr_default, thr_opt], [f'Threshold = {thr_default:.2f}', f'Threshold = {thr_opt:.2f}']):
        pred = (prob >= thr).astype(int)
        cm = confusion_matrix(y_true, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
    fig.suptitle('Test-set confusion matrices for the selected model', y=1.03)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'confusion_matrices.png', bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(model, test):
    X_test = test[FEATURES + [GROUP]]
    y_test = test[TARGET].astype(int)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring='average_precision', n_jobs=1)
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, ax=ax, color='#55A868')
    ax.set_title('Permutation importance (AP drop) on test set')
    ax.set_xlabel('Mean importance')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'feature_importance.png', bbox_inches='tight')
    plt.close(fig)
    return importances


def plot_calibration(y_true, prob):
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(mean_pred, frac_pos, marker='o', lw=2, color='#8172B3')
    ax.plot([0,1], [0,1], '--', color='gray')
    ax.set_title('Probability calibration on test set')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed positive fraction')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'calibration.png', bbox_inches='tight')
    plt.close(fig)


def subgroup_metrics(test, prob, thr):
    rows = []
    for degr, df in test.groupby(GROUP):
        idx = df.index
        m = compute_metrics(df[TARGET].astype(int), prob[idx], thr)
        rows.append({'degradation': degr, **m, 'n': len(df)})
    return pd.DataFrame(rows).sort_values('pr_auc', ascending=False)


def main():
    ensure_dirs()
    train, test = load_data()
    plot_data_overview(train)
    cv_results, cv_predictions = evaluate_models(train)
    cv_results.to_csv(OUTPUT_DIR / 'cv_results.csv', index=False)
    plot_model_comparison(cv_results)

    best_model_name = cv_results.iloc[0]['model']
    model, prob_test, test_metrics_05, test_metrics_opt, train_opt_thr = fit_and_test(train, test, best_model_name)
    y_test = test[TARGET].astype(int).to_numpy()
    plot_roc_pr(y_test, prob_test)
    plot_confusion_matrices(y_test, prob_test, 0.5, train_opt_thr)
    importances = plot_feature_importance(model, test)
    plot_calibration(y_test, prob_test)

    subgroup = subgroup_metrics(test, pd.Series(prob_test, index=test.index), train_opt_thr)
    subgroup.to_csv(OUTPUT_DIR / 'subgroup_metrics.csv', index=False)

    summary = {
        'selected_model': best_model_name,
        'cv_results': cv_results.to_dict(orient='records'),
        'test_metrics_default_0.5': test_metrics_05,
        'test_metrics_optimal_threshold': test_metrics_opt,
        'optimal_threshold_from_train': train_opt_thr,
        'test_positive_rate': float(test[TARGET].mean()),
        'feature_importance_top12': importances.to_dict(),
        'subgroup_metrics': subgroup.to_dict(orient='records'),
    }
    with open(OUTPUT_DIR / 'summary_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
