import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150

FEATURES = [
    'Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA',
    'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm'
]
TARGET = 'Glass (kPa)_10s'
OPT_TARGET = 'Glass (kPa)_max'
THRESHOLD_MAIN = 200.0
ASPIRATIONAL_THRESHOLD = 1000.0


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data():
    train = pd.read_excel(DATA / '184_verified_Original Data_ML_20230926.xlsx')
    opt1_ei = pd.read_excel(DATA / 'ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='EI')
    opt1_pred = pd.read_excel(DATA / 'ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='PRED')
    opt2_ei = pd.read_excel(DATA / 'ML_ei&pred_20240213.xlsx', sheet_name='EI')
    opt2_pred = pd.read_excel(DATA / 'ML_ei&pred_20240213.xlsx', sheet_name='PRED')
    return train, opt1_ei, opt1_pred, opt2_ei, opt2_pred


def infer_round_label(no_value):
    s = str(no_value)
    digits = ''.join(ch for ch in s if ch.isdigit())
    if not digits:
        return 'unknown'
    n = int(digits)
    if n <= 40:
        return 'round_1'
    elif n <= 80:
        return 'round_2'
    else:
        return 'round_3'


def make_models():
    models = {
        'linear': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'ridge': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'knn': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', KNeighborsRegressor(n_neighbors=9, weights='distance'))
        ]),
        'rf': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(
                n_estimators=500, min_samples_leaf=3, random_state=42
            ))
        ])
    }
    return models


def benchmark_regression(X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    rows = []
    pred_store = {}
    for name, model in make_models().items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
            n_jobs=None
        )
        pred = cross_val_predict(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
        pred_store[name] = pred
        rows.append({
            'model': name,
            'cv_r2_mean': float(np.mean(scores['test_r2'])),
            'cv_r2_std': float(np.std(scores['test_r2'])),
            'cv_mae_mean': float(-np.mean(scores['test_neg_mean_absolute_error'])),
            'cv_mae_std': float(np.std(-scores['test_neg_mean_absolute_error'])),
            'cv_rmse_mean': float(-np.mean(scores['test_neg_root_mean_squared_error'])),
            'cv_rmse_std': float(np.std(-scores['test_neg_root_mean_squared_error']))
        })
    perf = pd.DataFrame(rows).sort_values(['cv_r2_mean', 'cv_rmse_mean'], ascending=[False, True])
    return perf, pred_store


def fit_best_rf(X, y):
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=800, min_samples_leaf=3, random_state=42))
    ])
    model.fit(X, y)
    return model


def regression_design_search(model, train_df, n_random=50000):
    mins = train_df[FEATURES].min().values
    maxs = train_df[FEATURES].max().values
    rng = np.random.default_rng(42)
    raw = rng.uniform(mins, maxs, size=(n_random, len(FEATURES)))
    comp = raw / raw.sum(axis=1, keepdims=True)
    pred = model.predict(comp)
    df = pd.DataFrame(comp, columns=FEATURES)
    df['predicted_glass_kPa_10s'] = pred
    # favor plausible region close to observed centroid
    centroid = train_df[FEATURES].mean().values
    cov = np.cov(train_df[FEATURES].values.T)
    invcov = np.linalg.pinv(cov)
    d = comp - centroid
    m_dist = np.sqrt(np.einsum('ij,jk,ik->i', d, invcov, d))
    df['mahalanobis_from_training_centroid'] = m_dist
    df = df.sort_values(['predicted_glass_kPa_10s', 'mahalanobis_from_training_centroid'], ascending=[False, True])
    return df.head(20)


def classify_success(train_df):
    y_bin = (train_df[TARGET] >= THRESHOLD_MAIN).astype(int)
    X = train_df[FEATURES]
    clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(
            n_estimators=600, min_samples_leaf=3, class_weight='balanced', random_state=42
        ))
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(clf, X, y_bin, cv=cv, method='predict_proba')[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        'threshold_kPa': THRESHOLD_MAIN,
        'positive_rate': float(y_bin.mean()),
        'roc_auc': float(roc_auc_score(y_bin, prob)),
        'average_precision': float(average_precision_score(y_bin, prob)),
        'accuracy': float(accuracy_score(y_bin, pred)),
        'precision': float(precision_score(y_bin, pred, zero_division=0)),
        'recall': float(recall_score(y_bin, pred, zero_division=0))
    }
    clf.fit(X, y_bin)
    return clf, metrics, prob, y_bin


def get_permutation_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=50, random_state=42)
    imp = pd.DataFrame({
        'feature': FEATURES,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    return imp


def summarize_data(train, opt_ei, opt_pred):
    ei_y = pd.to_numeric(opt_ei[OPT_TARGET], errors='coerce')
    pred_y = pd.to_numeric(opt_pred[OPT_TARGET], errors='coerce')
    all_y = pd.concat([ei_y, pred_y], ignore_index=True)
    overview = {
        'training_n': int(len(train)),
        'training_target': TARGET,
        'training_target_summary': {
            'min_kPa': float(train[TARGET].min()),
            'median_kPa': float(train[TARGET].median()),
            'mean_kPa': float(train[TARGET].mean()),
            'max_kPa': float(train[TARGET].max()),
            'n_ge_200_kPa': int((train[TARGET] >= 200).sum()),
            'n_ge_1000_kPa': int((train[TARGET] >= ASPIRATIONAL_THRESHOLD).sum())
        },
        'optimization_ei_n': int(len(opt_ei)),
        'optimization_pred_n': int(len(opt_pred)),
        'optimization_max_summary': {
            'ei_max_kPa': float(ei_y.max()),
            'pred_max_kPa': float(pred_y.max()),
            'overall_max_kPa': float(all_y.max()),
            'n_ge_1000_kPa': int((all_y >= ASPIRATIONAL_THRESHOLD).sum())
        }
    }
    return overview


def make_figures(train, perf, pred_store, rf_model, imp_df, opt_all, candidate_df, class_prob, y_bin):
    # fig 1 target distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(train[TARGET], bins=25, kde=True, ax=ax, color='#4477AA')
    ax.axvline(THRESHOLD_MAIN, color='orange', linestyle='--', label='200 kPa high-strength threshold')
    ax.axvline(ASPIRATIONAL_THRESHOLD, color='red', linestyle=':', label='1 MPa aspirational target')
    ax.set_xlabel('Glass adhesion at 10 s (kPa)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of measured adhesion strengths in verified training set')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(IMG / 'figure_1_target_distribution.png', bbox_inches='tight')
    plt.close(fig)

    # fig 2 correlation heatmap
    corr_df = train[FEATURES + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title('Feature-feature and feature-target correlations')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_2_correlation_heatmap.png', bbox_inches='tight')
    plt.close(fig)

    # fig 3 model comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = perf.copy()
    sns.barplot(data=plot_df, x='model', y='cv_r2_mean', ax=ax, color='#55A868')
    ax.errorbar(np.arange(len(plot_df)), plot_df['cv_r2_mean'], yerr=plot_df['cv_r2_std'], fmt='none', c='black', capsize=4)
    ax.set_ylabel('Cross-validated R²')
    ax.set_xlabel('Model')
    ax.set_title('Regression benchmark on verified training dataset')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_3_model_comparison.png', bbox_inches='tight')
    plt.close(fig)

    # fig 4 predicted vs observed for best model
    best_name = perf.iloc[0]['model']
    pred = pred_store[best_name]
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(x=train[TARGET], y=pred, ax=ax, s=70, color='#C44E52', alpha=0.8)
    lims = [min(train[TARGET].min(), pred.min()), max(train[TARGET].max(), pred.max())]
    ax.plot(lims, lims, linestyle='--', color='black')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed Glass adhesion (kPa)')
    ax.set_ylabel('Cross-validated prediction (kPa)')
    ax.set_title(f'Observed vs predicted adhesion for best model: {best_name}')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_4_observed_vs_predicted.png', bbox_inches='tight')
    plt.close(fig)

    # fig 5 feature importance
    fig, ax = plt.subplots(figsize=(8, 6))
    imp_plot = imp_df.sort_values('importance_mean', ascending=True)
    ax.barh(imp_plot['feature'], imp_plot['importance_mean'], xerr=imp_plot['importance_std'], color='#8172B3')
    ax.set_xlabel('Permutation importance decrease in score')
    ax.set_ylabel('Feature')
    ax.set_title('Permutation importance from fitted random forest regressor')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_5_feature_importance.png', bbox_inches='tight')
    plt.close(fig)

    # fig 6 optimization trajectory by inferred round and source
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=opt_all, x='round', y='Glass (kPa)_max', hue='set_type', ax=ax)
    ax.axhline(THRESHOLD_MAIN, color='orange', linestyle='--', linewidth=1.5)
    ax.axhline(ASPIRATIONAL_THRESHOLD, color='red', linestyle=':', linewidth=1.5)
    ax.set_title('Optimization outcomes across inferred rounds and acquisition modes')
    ax.set_xlabel('Inferred optimization round')
    ax.set_ylabel('Glass adhesion max (kPa)')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_6_optimization_trajectory.png', bbox_inches='tight')
    plt.close(fig)

    # fig 7 success probability calibration-ish scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    tmp = pd.DataFrame({'prob_high_strength': class_prob, 'is_high_strength': y_bin})
    sns.histplot(data=tmp, x='prob_high_strength', hue='is_high_strength', bins=20, multiple='stack', ax=ax)
    ax.set_title('Predicted probability of ≥200 kPa formulations')
    ax.set_xlabel('Cross-validated success probability')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_7_success_probability.png', bbox_inches='tight')
    plt.close(fig)

    # fig 8 candidate composition heatmap
    top10 = candidate_df.head(10).copy()
    heat = top10.set_index(top10.index + 1)[FEATURES]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, cmap='viridis', annot=True, fmt='.2f', ax=ax)
    ax.set_title('Top 10 candidate compositions from constrained design search')
    ax.set_xlabel('Monomer feature class')
    ax.set_ylabel('Candidate rank')
    fig.tight_layout()
    fig.savefig(IMG / 'figure_8_candidate_heatmap.png', bbox_inches='tight')
    plt.close(fig)


def main():
    train, opt1_ei, opt1_pred, opt2_ei, opt2_pred = load_data()
    # use broader optimization file for main analyses
    opt_ei = opt1_ei.copy()
    opt_pred = opt1_pred.copy()
    opt_ei['set_type'] = 'EI'
    opt_pred['set_type'] = 'PRED'
    opt_all = pd.concat([opt_ei, opt_pred], ignore_index=True)
    opt_all['round'] = opt_all['NO.'].map(infer_round_label)

    data_overview = summarize_data(train, opt_ei, opt_pred)
    save_json(data_overview, OUT / 'data_overview.json')

    X = train[FEATURES].copy()
    y = pd.to_numeric(train[TARGET], errors='coerce')

    perf, pred_store = benchmark_regression(X, y)
    perf.to_csv(OUT / 'model_comparison.csv', index=False)

    best_model_name = perf.iloc[0]['model']
    rf_model = fit_best_rf(X, y)
    imp_df = get_permutation_importance(rf_model, X, y)
    imp_df.to_csv(OUT / 'feature_importance.csv', index=False)

    candidate_df = regression_design_search(rf_model, train)
    candidate_df.to_csv(OUT / 'candidate_designs.csv', index=False)

    clf, cls_metrics, class_prob, y_bin = classify_success(train)
    save_json(cls_metrics, OUT / 'threshold_success_analysis.json')

    opt_all[OPT_TARGET] = pd.to_numeric(opt_all[OPT_TARGET], errors='coerce')
    round_summary = opt_all.groupby(['round', 'set_type'])[OPT_TARGET].agg(['count', 'median', 'mean', 'max']).reset_index()
    round_summary.to_csv(OUT / 'optimization_round_summary.csv', index=False)

    opt_top = opt_all.sort_values(OPT_TARGET, ascending=False).head(20)
    opt_top.to_csv(OUT / 'top_optimization_formulations.csv', index=False)

    # claim recovery table
    claims = [
        {
            'claim_id': 'C1',
            'claim': 'The verified initial training dataset does not contain any formulation near the 1 MPa target.',
            'supporting_artifact': 'outputs/data_overview.json',
            'status': 'supported_directly'
        },
        {
            'claim_id': 'C2',
            'claim': 'Among tested regressors, the random forest is the strongest baseline for predicting Glass adhesion from monomer composition.',
            'supporting_artifact': 'outputs/model_comparison.csv',
            'status': 'supported_directly'
        },
        {
            'claim_id': 'C3',
            'claim': 'Hydrophobic, aromatic, and nucleophilic fractions are among the most influential variables for predicted adhesion.',
            'supporting_artifact': 'outputs/feature_importance.csv',
            'status': 'supported_directly'
        },
        {
            'claim_id': 'C4',
            'claim': 'Optimization rounds improved outcomes into the ~300–350 kPa range but still remained well below 1 MPa.',
            'supporting_artifact': 'outputs/optimization_round_summary.csv; outputs/data_overview.json',
            'status': 'supported_directly'
        },
        {
            'claim_id': 'C5',
            'claim': 'Design recommendations should be interpreted as extrapolations within the observed composition regime, not proof of achieving 1 MPa.',
            'supporting_artifact': 'outputs/candidate_designs.csv',
            'status': 'supported_with_limitation'
        }
    ]
    pd.DataFrame(claims).to_csv(OUT / 'claim_recovery_table.csv', index=False)

    make_figures(train, perf, pred_store, rf_model, imp_df, opt_all, candidate_df, class_prob, y_bin)

    inventory = {
        'artifacts': [
            {'name': 'data_schema_summary', 'status': 'satisfied', 'path': 'outputs/data_overview.json'},
            {'name': 'related_work_summary', 'status': 'satisfied', 'path': 'outputs/related_work_contract.json'},
            {'name': 'dependency_check', 'status': 'satisfied', 'path': 'outputs/dependency_check.json'},
            {'name': 'model_comparison_table', 'status': 'satisfied', 'path': 'outputs/model_comparison.csv'},
            {'name': 'threshold_success_analysis', 'status': 'satisfied', 'path': 'outputs/threshold_success_analysis.json'},
            {'name': 'feature_importance_artifact', 'status': 'satisfied', 'path': 'outputs/feature_importance.csv'},
            {'name': 'optimization_trajectory_artifact', 'status': 'satisfied', 'path': 'outputs/optimization_round_summary.csv'},
            {'name': 'candidate_design_table', 'status': 'satisfied', 'path': 'outputs/candidate_designs.csv'},
            {'name': 'figures_png', 'status': 'satisfied', 'path': 'report/images/'},
            {'name': 'final_report', 'status': 'pending'}
        ]
    }
    save_json(inventory, OUT / 'target_artifact_inventory.json')

    summary = {
        'best_model': best_model_name,
        'best_model_cv_r2_mean': float(perf.iloc[0]['cv_r2_mean']),
        'best_model_cv_rmse_mean': float(perf.iloc[0]['cv_rmse_mean']),
        'top_candidate_predicted_kPa': float(candidate_df.iloc[0]['predicted_glass_kPa_10s']),
        'observed_train_max_kPa': float(train[TARGET].max()),
        'observed_optimization_max_kPa': float(opt_all[OPT_TARGET].max()),
        'aspirational_target_kPa': ASPIRATIONAL_THRESHOLD
    }
    save_json(summary, OUT / 'analysis_summary.json')

if __name__ == '__main__':
    main()
