import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import shap

plt.style.use('seaborn-v0_8')
out_dir = Path('report/images')
outputs_dir = Path('outputs')
out_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(exist_ok=True)

# Load data
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

feature_cols = [str(i) for i in range(20)]
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
degr_train = train_df['degradation']

X_test = test_df[feature_cols].values
y_test = test_df['label'].values
degr_test = test_df['degradation']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, outputs_dir / 'scaler.pkl')

# Models with class_weight='balanced' for imbalance
models = {
    'LR': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'RF': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
    'XGB': XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42, eval_metric='logloss'),
    'MLP': MLPClassifier(random_state=42, max_iter=500)
}

# With SMOTE pipeline for comparison
smote_pipe = {
    'LR-SMOTE': Pipeline([('smote', SMOTE(random_state=42)), ('clf', LogisticRegression(random_state=42, max_iter=1000))]),
    'RF-SMOTE': Pipeline([('smote', SMOTE(random_state=42)), ('clf', RandomForestClassifier(random_state=42, n_estimators=100))])
}

all_models = {**models, **smote_pipe}

results = {}
cv_scores = {}

for name, model in all_models.items():
    print(f'Training {name}...')
    # CV
    if 'SMOTE' in name:
        cv_scores[name] = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    else:
        cv_scores[name] = cross_val_score(model, X_train_scaled, y_train, cv=StratifiedKFold(5), scoring='roc_auc')
    
    # Train
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, outputs_dir / f'{name.lower()}_model.pkl')
    
    # Test predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test_scaled)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {'CV_AUC_mean': cv_scores[name].mean(), 'CV_AUC_std': cv_scores[name].std(),
                     'Test_AUC': auc, 'Test_AUPRC': auprc, 'Test_F1': f1}
    
    # Per degradation
    degr_results = {}
    for degr in degr_test.unique():
        mask = degr_test == degr
        if mask.sum() > 0:
            auc_degr = roc_auc_score(y_test[mask], y_pred_proba[mask])
            auprc_degr = average_precision_score(y_test[mask], y_pred_proba[mask])
            f1_degr = f1_score(y_test[mask], y_pred[mask])
            degr_results[degr] = {'AUC': auc_degr, 'AUPRC': auprc_degr, 'F1': f1_degr}
    results[name]['per_degr'] = degr_results

# Save results
with open(outputs_dir / 'model_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
pd.DataFrame(cv_scores, index=['mean', 'std']).T.to_json(outputs_dir / 'cv_scores.json')

# Metrics table plot
metrics_df = pd.DataFrame({k: v for k, v in results.items() if 'Test_' in str(v) or isinstance(v, dict)}).T
fig, ax = plt.subplots(figsize=(10,6))
metrics_df[['Test_AUC', 'Test_AUPRC', 'Test_F1']].plot(kind='bar', ax=ax)
ax.set_title('Test Metrics Comparison')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(out_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')

# ROC/PR for best model (say XGB)
best_model = all_models['XGB']
y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:,1]
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
RocCurveDisplay.from_predictions(y_test, y_pred_proba_best, ax=ax1)
ax1.set_title('ROC Curve (XGB)')
PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba_best, ax=ax2)
ax2.set_title('PR Curve (XGB)')
plt.tight_layout()
plt.savefig(out_dir / 'roc_pr_xgb.png', dpi=300, bbox_inches='tight')

# Confusion matrix
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (XGB)')
plt.ylabel('True')
plt.xlabel('Pred')
plt.savefig(out_dir / 'confusion_xgb.png', dpi=300, bbox_inches='tight')

# Feature importance for tree models
if hasattr(best_model, 'feature_importances_'):
    imp = pd.DataFrame({'feature': feature_cols, 'imp': best_model.feature_importances_}).sort_values('imp', ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(data=imp.head(10), x='imp', y='feature')
    plt.title('Top 10 Feature Importances (XGB)')
    plt.savefig(out_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')

# SHAP (for XGB)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled[:1000])  # subsample
shap.summary_plot(shap_values[1], X_test_scaled[:1000], feature_names=feature_cols, show=False)
plt.savefig(out_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')

print('Modeling complete.')