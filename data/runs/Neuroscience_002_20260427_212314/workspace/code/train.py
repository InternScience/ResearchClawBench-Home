"""Train multiple classifiers and save predictions + metrics on test set."""
import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import xgboost as xgb

os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

tr = pd.read_csv('data/train_simulated.csv')
te = pd.read_csv('data/test_simulated.csv')
feat_cols = [str(i) for i in range(20)]
Xtr, ytr = tr[feat_cols].values, tr['label'].astype(int).values
Xte, yte = te[feat_cols].values, te['label'].astype(int).values
deg_te = te['degradation'].values

# scale_pos_weight for class imbalance
pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
print(f'class imbalance scale_pos_weight={pos_w:.2f}')

models = {
    'logreg': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                   C=1.0, n_jobs=-1, solver='lbfgs'))
    ]),
    'rf': RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5,
        class_weight='balanced_subsample', n_jobs=-1, random_state=0),
    'gbm': GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0),
    'xgb': xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.07,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=pos_w, eval_metric='logloss',
        tree_method='hist', n_jobs=-1, random_state=0),
    'mlp': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                              max_iter=60, random_state=0, early_stopping=True,
                              validation_fraction=0.1, n_iter_no_change=8))
    ]),
}

results = {}
preds_proba = {}
threshold_default = 0.5
for name, m in models.items():
    print(f'Training {name} ...')
    t0 = time.time()
    m.fit(Xtr, ytr)
    train_t = time.time() - t0
    p = m.predict_proba(Xte)[:, 1]
    yhat = (p >= threshold_default).astype(int)
    res = {
        'accuracy':  accuracy_score(yte, yhat),
        'precision': precision_score(yte, yhat, zero_division=0),
        'recall':    recall_score(yte, yhat, zero_division=0),
        'f1':        f1_score(yte, yhat, zero_division=0),
        'roc_auc':   roc_auc_score(yte, p),
        'pr_auc':    average_precision_score(yte, p),
        'train_time_s': round(train_t, 2),
    }
    # per-degradation metrics
    per_deg = {}
    for d in np.unique(deg_te):
        mask = deg_te == d
        per_deg[d] = {
            'accuracy':  accuracy_score(yte[mask], yhat[mask]),
            'precision': precision_score(yte[mask], yhat[mask], zero_division=0),
            'recall':    recall_score(yte[mask], yhat[mask], zero_division=0),
            'f1':        f1_score(yte[mask], yhat[mask], zero_division=0),
            'roc_auc':   roc_auc_score(yte[mask], p[mask]) if len(np.unique(yte[mask])) > 1 else float('nan'),
            'pr_auc':    average_precision_score(yte[mask], p[mask]),
            'n':         int(mask.sum()),
        }
    res['per_degradation'] = per_deg
    results[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != 'per_degradation'}
    results[name]['per_degradation'] = {d: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in vals.items()} for d, vals in per_deg.items()}
    preds_proba[name] = p
    joblib.dump(m, f'outputs/models/{name}.joblib')
    print(f'  done in {train_t:.1f}s, ROC-AUC={res["roc_auc"]:.4f} F1={res["f1"]:.4f}')

# Save predictions and metrics
with open('outputs/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

pred_df = pd.DataFrame(preds_proba)
pred_df['label'] = yte
pred_df['degradation'] = deg_te
pred_df.to_csv('outputs/test_predictions.csv', index=False)
print('Saved metrics.json + test_predictions.csv')
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'per_degradation'} for k, v in results.items()}, indent=2))
