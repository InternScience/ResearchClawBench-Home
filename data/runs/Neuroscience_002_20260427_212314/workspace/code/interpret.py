"""Interpretability: SHAP on XGBoost, plus permutation importance and a
modality ablation study."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

tr = pd.read_csv('data/train_simulated.csv')
te = pd.read_csv('data/test_simulated.csv')
feat_cols = [str(i) for i in range(20)]
Xtr, ytr = tr[feat_cols].values, tr['label'].astype(int).values
Xte, yte = te[feat_cols].values, te['label'].astype(int).values
deg_te = te['degradation'].values

modalities = {
    'morphology': [0, 1, 2, 3, 4],
    'intensity':  [5, 6, 7, 8, 9],
    'embedding':  list(range(10, 20)),
}

# 1) SHAP on XGBoost
xgb_model = joblib.load('outputs/models/xgb.joblib')
print('Computing SHAP values on XGBoost ...')
rng = np.random.RandomState(0)
sub = rng.choice(len(Xte), size=4000, replace=False)
Xs = Xte[sub]
# Newer XGBoost stores base_score as JSON list which breaks shap.TreeExplainer.
# Use XGBoost built-in TreeSHAP via Booster.predict(pred_contribs=True).
import xgboost as xgb_lib
booster = xgb_model.get_booster()
dmat = xgb_lib.DMatrix(Xs)
shap_full = booster.predict(dmat, pred_contribs=True)
shap_vals = shap_full[:, :-1]
print('shap_vals shape:', np.array(shap_vals).shape)

# Summary (beeswarm)
plt.figure()
shap.summary_plot(shap_vals, Xs, feature_names=feat_cols, show=False, max_display=20)
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.title('SHAP feature importance (XGBoost, 4k test samples)')
plt.tight_layout()
plt.savefig('report/images/shap_summary_xgb.png', dpi=150, bbox_inches='tight')
plt.close()

# Mean |SHAP| per feature, grouped by modality
mean_abs = np.abs(shap_vals).mean(axis=0)
mod_color = {'morphology': '#4C72B0', 'intensity': '#DD8452', 'embedding': '#55A868'}
fcolors = []
for i in range(20):
    for mod, idx in modalities.items():
        if i in idx:
            fcolors.append(mod_color[mod])
            break
plt.figure(figsize=(11, 4))
plt.bar(feat_cols, mean_abs, color=fcolors)
plt.ylabel('mean |SHAP|')
plt.xlabel('feature index')
plt.title('Mean |SHAP| per feature — XGBoost')
import matplotlib.patches as mp
handles = [mp.Patch(color=c, label=mod) for mod, c in mod_color.items()]
plt.legend(handles=handles, loc='upper left')
plt.tight_layout()
plt.savefig('report/images/shap_mean_abs.png', dpi=150)
plt.close()
pd.DataFrame({'feature': feat_cols, 'mean_abs_shap': mean_abs}).to_csv(
    'outputs/shap_mean_abs.csv', index=False)

# Per-degradation mean |SHAP| (modality contributions)
deg_sub = deg_te[sub]
mod_shap = {}
for mod, idx in modalities.items():
    mod_shap[mod] = np.abs(shap_vals[:, idx]).sum(axis=1)
deg_levels = sorted(np.unique(deg_sub).tolist())
rows = []
for d in deg_levels:
    mask = deg_sub == d
    for mod in modalities:
        rows.append({'degradation': d, 'modality': mod,
                     'mean_total_abs_shap': float(mod_shap[mod][mask].mean())})
mdf = pd.DataFrame(rows)
mdf.to_csv('outputs/shap_modality_per_degradation.csv', index=False)

pivot = mdf.pivot(index='degradation', columns='modality', values='mean_total_abs_shap')
pivot = pivot[['morphology', 'intensity', 'embedding']]
plt.figure(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='magma_r')
plt.title('Mean total |SHAP| per modality × degradation\n(XGBoost, 4k test samples)')
plt.tight_layout()
plt.savefig('report/images/shap_modality_per_degradation.png', dpi=150)
plt.close()

# 2) Permutation importance on MLP (post-hoc, model-agnostic)
print('Permutation importance on MLP ...')
mlp = joblib.load('outputs/models/mlp.joblib')
sub2 = rng.choice(len(Xte), size=8000, replace=False)
pi = permutation_importance(mlp, Xte[sub2], yte[sub2], n_repeats=5,
                            random_state=0, n_jobs=-1, scoring='roc_auc')
pi_df = pd.DataFrame({'feature': feat_cols,
                      'mean_drop_auc': pi.importances_mean,
                      'std_drop_auc': pi.importances_std}).sort_values(
    'mean_drop_auc', ascending=False)
pi_df.to_csv('outputs/permutation_importance_mlp.csv', index=False)

plt.figure(figsize=(11, 4))
order = [str(i) for i in range(20)]
vals = [float(pi_df.set_index('feature').loc[c, 'mean_drop_auc']) for c in order]
errs = [float(pi_df.set_index('feature').loc[c, 'std_drop_auc']) for c in order]
plt.bar(order, vals, yerr=errs, color=fcolors, capsize=2)
plt.ylabel('Δ ROC-AUC when feature is permuted')
plt.xlabel('feature index')
plt.title('Permutation importance — MLP (5 repeats, 8k test samples)')
plt.legend(handles=handles, loc='upper left')
plt.tight_layout()
plt.savefig('report/images/permutation_importance_mlp.png', dpi=150)
plt.close()

# 3) Modality ablation study using a quick MLP retrain on each subset
print('Modality ablation using retrained MLPs ...')
combos = {
    'morph_only':       modalities['morphology'],
    'intensity_only':   modalities['intensity'],
    'embedding_only':   modalities['embedding'],
    'morph+intensity':  modalities['morphology'] + modalities['intensity'],
    'morph+embedding':  modalities['morphology'] + modalities['embedding'],
    'intensity+embed':  modalities['intensity'] + modalities['embedding'],
    'all':              list(range(20)),
}
ablation_rows = []
for name, idx in combos.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                              max_iter=40, random_state=0, early_stopping=True,
                              validation_fraction=0.1, n_iter_no_change=6))
    ])
    pipe.fit(Xtr[:, idx], ytr)
    p = pipe.predict_proba(Xte[:, idx])[:, 1]
    yhat = (p >= 0.5).astype(int)
    ablation_rows.append({
        'features':  name,
        'n_features': len(idx),
        'roc_auc':   roc_auc_score(yte, p),
        'pr_auc':    average_precision_score(yte, p),
        'f1':        f1_score(yte, yhat, zero_division=0),
    })
abl = pd.DataFrame(ablation_rows)
abl.to_csv('outputs/modality_ablation.csv', index=False)

plt.figure(figsize=(10, 4.5))
xpos = np.arange(len(abl))
plt.bar(xpos - 0.25, abl['roc_auc'], 0.25, label='ROC-AUC')
plt.bar(xpos, abl['pr_auc'], 0.25, label='PR-AUC')
plt.bar(xpos + 0.25, abl['f1'], 0.25, label='F1')
plt.xticks(xpos, abl['features'], rotation=20)
plt.ylim(0.5, 1.01)
plt.title('Modality ablation — retrained MLP (64,32)')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/modality_ablation.png', dpi=150)
plt.close()
print(abl.round(4).to_string(index=False))

# 4) Calibration of best model (reliability diagram)
preds = pd.read_csv('outputs/test_predictions.csv')
from sklearn.calibration import calibration_curve
plt.figure(figsize=(6, 5.5))
for name, color in [('xgb', '#DD8452'), ('mlp', '#55A868'), ('rf', '#4C72B0')]:
    p = preds[name].values
    frac, mean = calibration_curve(yte, p, n_bins=15, strategy='quantile')
    plt.plot(mean, frac, marker='o', label=name, color=color)
plt.plot([0, 1], [0, 1], 'k--', lw=0.7, label='perfect')
plt.xlabel('mean predicted probability')
plt.ylabel('observed positive rate')
plt.title('Reliability diagram (test set)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/calibration.png', dpi=150)
plt.close()

print('interpret done')
