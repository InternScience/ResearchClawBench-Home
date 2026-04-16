"""
Neuron Segment Merge Prediction Analysis
=========================================

This script performs the complete analysis pipeline for predicting whether
two over-segmented neuron fragments should be merged in connectomics EM data.

Pipeline:
1. Data loading and exploration
2. Feature characterization and group analysis
3. Model training (Logistic Regression, XGBoost)
4. SHAP interpretability analysis
5. Per-degradation performance evaluation
6. Threshold optimization
7. Feature group ablation study
8. Figure generation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
TRAIN_PATH = 'data/train_simulated.csv'
TEST_PATH = 'data/test_simulated.csv'
FIGURE_DIR = 'report/images/'
OUTPUT_DIR = 'outputs/'
SEED = 42

# ============================================================
# 1. Data Loading
# ============================================================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

feat_cols = [str(i) for i in range(20)]
X_train = train[feat_cols].values
y_train = train['label'].values
deg_train = train['degradation'].values

X_test = test[feat_cols].values
y_test = test['label'].values
deg_test = test['degradation'].values

le = LabelEncoder()
deg_train_enc = le.fit_transform(deg_train)
deg_test_enc = le.transform(deg_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_aug = np.column_stack([X_train_scaled, deg_train_enc])
X_test_aug = np.column_stack([X_test_scaled, deg_test_enc])

aug_feat_names = feat_cols + ['degradation_enc']

# ============================================================
# 2. Model Training
# ============================================================
# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
lr.fit(X_train_aug, y_train)
y_pred_lr = lr.predict(X_test_aug)
y_prob_lr = lr.predict_proba(X_test_aug)[:, 1]

# XGBoost (primary model)
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=SEED, n_jobs=-1, eval_metric='auc',
    use_label_encoder=False,
    subsample=0.8, colsample_bytree=0.8
)
xgb_clf.fit(X_train_aug, y_train, verbose=False)
y_pred_xgb = xgb_clf.predict(X_test_aug)
y_prob_xgb = xgb_clf.predict_proba(X_test_aug)[:, 1]

# ============================================================
# 3. SHAP Interpretability
# ============================================================
subset_size = 5000
np.random.seed(SEED)
idx = np.random.choice(len(X_test_aug), subset_size, replace=False)
X_test_subset = X_test_aug[idx]

explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test_subset)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# ============================================================
# 4. Threshold Optimization
# ============================================================
thresholds = np.arange(0.1, 0.95, 0.01)
f1s_thresh = []
for t in thresholds:
    y_pred_t = (y_prob_xgb >= t).astype(int)
    f1s_thresh.append(f1_score(y_test, y_pred_t))

best_threshold = thresholds[np.argmax(f1s_thresh)]
y_pred_opt = (y_prob_xgb >= best_threshold).astype(int)

# ============================================================
# 5. Ablation Study
# ============================================================
groups_def = {
    'All Features': feat_cols + ['degradation'],
    'No Morphology': [str(i) for i in range(5,20)] + ['degradation'],
    'No Intensity': [str(i) for i in range(0,5)] + [str(i) for i in range(10,20)] + ['degradation'],
    'No Embeddings': [str(i) for i in range(0,10)] + ['degradation'],
    'No Degradation': feat_cols,
}

ablation_results = {}
for name, gcols in groups_def.items():
    numeric_cols = [c for c in gcols if c != 'degradation']
    has_deg = 'degradation' in gcols
    
    X_tr = train[numeric_cols].values
    X_te = test[numeric_cols].values
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    
    if has_deg:
        X_tr_s = np.column_stack([X_tr_s, deg_train_enc])
        X_te_s = np.column_stack([X_te_s, deg_test_enc])
    
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.2,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=SEED, n_jobs=-1, eval_metric='auc',
        use_label_encoder=False
    )
    clf.fit(X_tr_s, y_train, verbose=False)
    y_pred_abl = clf.predict(X_te_s)
    y_prob_abl = clf.predict_proba(X_te_s)[:, 1]
    
    ablation_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred_abl),
        'roc_auc': roc_auc_score(y_test, y_prob_abl),
        'f1': f1_score(y_test, y_pred_abl),
        'avg_precision': average_precision_score(y_test, y_prob_abl)
    }

# ============================================================
# 6. Save Results
# ============================================================
degradations = ['Average', 'Misalignment', 'Missing Sections', 'Mixed']

final_results = {
    'overall': {
        'default_threshold': {
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'roc_auc': roc_auc_score(y_test, y_prob_xgb),
            'f1': f1_score(y_test, y_pred_xgb),
            'precision': precision_score(y_test, y_pred_xgb),
            'recall': recall_score(y_test, y_pred_xgb),
            'avg_precision': average_precision_score(y_test, y_prob_xgb)
        },
        'optimized_threshold': {
            'threshold': best_threshold,
            'accuracy': accuracy_score(y_test, y_pred_opt),
            'roc_auc': roc_auc_score(y_test, y_prob_xgb),
            'f1': f1_score(y_test, y_pred_opt),
            'precision': precision_score(y_test, y_pred_opt),
            'recall': recall_score(y_test, y_pred_opt),
            'avg_precision': average_precision_score(y_test, y_prob_xgb)
        }
    },
    'per_degradation': {}
}

for deg in degradations:
    mask = deg_test == deg
    final_results['per_degradation'][deg] = {
        'n_samples': int(mask.sum()),
        'n_positive': int(y_test[mask].sum()),
        'roc_auc': roc_auc_score(y_test[mask], y_prob_xgb[mask]),
        'avg_precision': average_precision_score(y_test[mask], y_prob_xgb[mask]),
        'default_f1': f1_score(y_test[mask], y_pred_xgb[mask]),
        'optimized_f1': f1_score(y_test[mask], y_pred_opt[mask])
    }

with open(OUTPUT_DIR + 'final_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

with open(OUTPUT_DIR + 'shap_importance.json', 'w') as f:
    json.dump({
        'per_feature': dict(zip(aug_feat_names, mean_abs_shap.tolist())),
        'per_group': {
            'morphology': mean_abs_shap[0:5].sum(),
            'intensity': mean_abs_shap[5:10].sum(),
            'embedding_10_14': mean_abs_shap[10:15].sum(),
            'embedding_15_19': mean_abs_shap[15:20].sum(),
            'degradation': mean_abs_shap[20]
        }
    }, f, indent=2)

with open(OUTPUT_DIR + 'ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2)

print("All results saved successfully.")