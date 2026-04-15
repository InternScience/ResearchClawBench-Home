import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

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

# Models
models = {
    'LR': LogisticRegression(class_weight='balanced', random_state=42, max_iter=500),
    'RF': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=50, n_jobs=-1),
    'XGB': XGBClassifier(scale_pos_weight=9, random_state=42, n_estimators=100, n_jobs=-1),
    'MLP': MLPClassifier(random_state=42, max_iter=200, hidden_layer_sizes=(64,32))
}

results = {}
cv_scores = {}

for name, model in models.items():
    print(f'Training {name}...')
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores[name] = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, outputs_dir / f'{name.lower()}_model.pkl')
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {'CV_AUC_mean': cv_scores[name].mean(), 'CV_AUC_std': cv_scores[name].std(),
                     'Test_AUC': auc, 'Test_AUPRC': auprc, 'Test_F1': f1}
    
    # Per degr
    degr_results = {}
    for degr in sorted(degr_test.unique()):
        mask = degr_test == degr
        if mask.sum() > 10:
            auc_degr = roc_auc_score(y_test[mask], y_pred_proba[mask])
            auprc_degr = average_precision_score(y_test[mask], y_pred_proba[mask])
            f1_degr = f1_score(y_test[mask], y_pred[mask])
            degr_results[degr] = {'AUC': auc_degr, 'AUPRC': auprc_degr, 'F1': f1_degr}
    results[name]['per_degr'] = degr_results

# Save
with open(outputs_dir / 'model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
pd.DataFrame({k: [v['CV_AUC_mean'], v['CV_AUC_std']] for k,v in results.items() if 'CV' in str(v)}).to_json(outputs_dir / 'cv_summary.json')

# Plots
results_df = pd.DataFrame([(k, v['Test_AUC'], v['Test_AUPRC'], v['Test_F1']) for k,v in results.items()], columns=['Model', 'AUC', 'AUPRC', 'F1'])
results_df.plot(x='Model', kind='bar', figsize=(10,6))
plt.title('Test Metrics')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig(out_dir / 'test_metrics_bar.png', dpi=300, bbox_inches='tight')

# Per degr heatmap for best model (XGB)
xgb_results = pd.DataFrame(results['XGB']['per_degr']).T
sns.heatmap(xgb_results, annot=True, cmap='RdYlGn')
plt.title('XGB Per Degradation Metrics')
plt.savefig(out_dir / 'per_degr_heatmap.png', dpi=300, bbox_inches='tight')

# Feature imp RF
imp_df = pd.DataFrame({'feat': range(20), 'imp': models['RF'].feature_importances_}).sort_values('imp', ascending=False)
sns.barplot(data=imp_df.head(10), y='feat', x='imp')
plt.title('RF Feature Importance Top 10')
plt.savefig(out_dir / 'feat_imp_rf.png', dpi=300, bbox_inches='tight')

print('Light modeling complete.')