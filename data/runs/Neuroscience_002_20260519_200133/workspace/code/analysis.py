"""
Analysis script for neuron segment merge prediction.
Binary classification: predict whether two segments belong to the same neuron.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
REPORT_IMG_DIR = 'report/images'
for d in [OUTPUT_DIR, REPORT_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, 'train_simulated.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test_simulated.csv'))

feature_cols = [str(i) for i in range(20)]
label_col = 'label'
degradation_col = 'degradation'

X_train = train[feature_cols].values.astype(np.float32)
y_train = train[label_col].values.astype(np.int32)
deg_train = train[degradation_col].values

X_test = test[feature_cols].values.astype(np.float32)
y_test = test[label_col].values.astype(np.int32)
deg_test = test[degradation_col].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train label balance: {np.bincount(y_train)}")
print(f"Test label balance: {np.bincount(y_test)}")

# Save basic stats
stats = {
    'train_n': int(X_train.shape[0]),
    'test_n': int(X_test.shape[0]),
    'n_features': int(X_train.shape[1]),
    'train_pos_ratio': float(y_train.mean()),
    'test_pos_ratio': float(y_test.mean()),
    'degradation_types': list(np.unique(deg_train)),
}
with open(os.path.join(OUTPUT_DIR, 'data_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

# ============================================================
# 2. EDA Figures
# ============================================================
print("Generating EDA figures...")

# Figure 1: Label distribution by degradation type
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Train
deg_label = pd.crosstab(train[degradation_col], train[label_col], normalize='index')
deg_label.plot(kind='bar', ax=ax[0], color=['#e74c3c', '#2ecc71'])
ax[0].set_title('Train: Label Distribution by Degradation Type')
ax[0].set_ylabel('Proportion')
ax[0].set_xlabel('Degradation Type')
ax[0].legend(['Negative (0)', 'Positive (1)'])
ax[0].tick_params(axis='x', rotation=45)

# Test
deg_label_test = pd.crosstab(test[degradation_col], test[label_col], normalize='index')
deg_label_test.plot(kind='bar', ax=ax[1], color=['#e74c3c', '#2ecc71'])
ax[1].set_title('Test: Label Distribution by Degradation Type')
ax[1].set_ylabel('Proportion')
ax[1].set_xlabel('Degradation Type')
ax[1].legend(['Negative (0)', 'Positive (1)'])
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig1_label_distribution.png'), dpi=200)
plt.close()

# Figure 2: Feature distributions by label (violin plot of first 10 features)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i in range(10):
    data_plot = [train[train[label_col] == 0][str(i)].values,
                 train[train[label_col] == 1][str(i)].values]
    axes[i].violinplot(data_plot, positions=[0, 1], showmeans=True, showmedians=True)
    axes[i].set_title(f'Feature {i}')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['Neg', 'Pos'])
plt.suptitle('Feature Distributions by Label (First 10 Features)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig2_feature_distributions.png'), dpi=200)
plt.close()

# Figure 3: Correlation heatmap of features
corr = train[feature_cols].astype(float).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, ax=ax,
            cbar_kws={'label': 'Pearson r'})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig3_correlation_heatmap.png'), dpi=200)
plt.close()

# ============================================================
# 3. Model Training & Evaluation
# ============================================================
print("Training models...")

# Scale features for LR and MLP
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

scale_pos_weight = float((y_train == 0).sum()) / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

models = {
    'LogisticRegression': LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED, n_jobs=4),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=SEED, n_jobs=4),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=SEED),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=SEED),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED, n_jobs=4, eval_metric='logloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced',
        random_state=SEED, n_jobs=4, verbose=-1
    ),
}

results = {}
trained_models = {}
probas = {}

for name, model in models.items():
    print(f"  Training {name}...")
    if name in ['LogisticRegression', 'MLP']:
        model.fit(X_train_s, y_train)
        y_proba = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

    probas[name] = y_proba
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    # Optimal threshold for F1
    thresholds = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    best_f1 = np.max(f1s)
    y_pred_best = (y_proba >= best_t).astype(int)
    prec_best = precision_score(y_test, y_pred_best, zero_division=0)
    rec_best = recall_score(y_test, y_pred_best, zero_division=0)

    results[name] = {
        'accuracy': float(acc),
        'precision_05': float(prec),
        'recall_05': float(rec),
        'f1_05': float(f1),
        'roc_auc': float(roc),
        'pr_auc': float(pr),
        'best_threshold': float(best_t),
        'precision_best': float(prec_best),
        'recall_best': float(rec_best),
        'f1_best': float(best_f1),
    }
    trained_models[name] = model
    print(f"    ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}, F1@0.5={f1:.4f}, Best F1={best_f1:.4f} @ t={best_t:.3f}")

with open(os.path.join(OUTPUT_DIR, 'model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ============================================================
# 4. Comparison Figures
# ============================================================
print("Generating comparison figures...")

# Figure 4: Model comparison bar chart
metrics_df = pd.DataFrame(results).T
metrics_df = metrics_df.sort_values('roc_auc', ascending=False)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics_to_plot = ['roc_auc', 'pr_auc', 'f1_best']
titles = ['ROC-AUC', 'PR-AUC', 'Best F1 Score']
for ax, metric, title in zip(axes, metrics_to_plot, titles):
    metrics_df[metric].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(metrics_df[metric]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig4_model_comparison.png'), dpi=200)
plt.close()

# Figure 5: ROC curves
fig, ax = plt.subplots(figsize=(8, 8))
for name in models:
    y_proba = probas[name]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig5_roc_curves.png'), dpi=200)
plt.close()

# Figure 6: PR curves
fig, ax = plt.subplots(figsize=(8, 8))
for name in models:
    y_proba = probas[name]
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    ax.plot(rec_curve, prec_curve, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
ax.axhline(y_test.mean(), color='k', linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig6_pr_curves.png'), dpi=200)
plt.close()

# ============================================================
# 5. Best Model Deep Dive
# ============================================================
best_model_name = metrics_df.index[0]
print(f"Best model: {best_model_name}")
best_model = trained_models[best_model_name]

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
else:
    importances = np.zeros(20)

feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False)

# Figure 7: Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_imp, x='importance', y='feature', palette='viridis', ax=ax)
ax.set_title(f'Feature Importance ({best_model_name})')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig7_feature_importance.png'), dpi=200)
plt.close()

feat_imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

# Confusion matrix for best model
y_proba_best = probas[best_model_name]
best_t = results[best_model_name]['best_threshold']
y_pred_best = (y_proba_best >= best_t).astype(int)
cm = confusion_matrix(y_test, y_pred_best)

# Figure 8: Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred Neg', 'Pred Pos'],
            yticklabels=['True Neg', 'True Pos'])
ax.set_title(f'Confusion Matrix ({best_model_name}, threshold={best_t:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig8_confusion_matrix.png'), dpi=200)
plt.close()

# ============================================================
# 6. Per-Degradation Analysis
# ============================================================
print("Analyzing per-degradation performance...")

degradation_results = {}
for deg_type in np.unique(deg_test):
    mask = deg_test == deg_type
    y_deg = y_test[mask]
    y_proba_deg = y_proba_best[mask]

    roc_deg = roc_auc_score(y_deg, y_proba_deg)
    pr_deg = average_precision_score(y_deg, y_proba_deg)

    thresholds = np.linspace(0.01, 0.99, 99)
    f1s_deg = [f1_score(y_deg, (y_proba_deg >= t).astype(int), zero_division=0) for t in thresholds]
    best_t_deg = thresholds[np.argmax(f1s_deg)]
    best_f1_deg = np.max(f1s_deg)

    degradation_results[deg_type] = {
        'n': int(mask.sum()),
        'pos_ratio': float(y_deg.mean()),
        'roc_auc': float(roc_deg),
        'pr_auc': float(pr_deg),
        'best_threshold': float(best_t_deg),
        'best_f1': float(best_f1_deg),
    }

with open(os.path.join(OUTPUT_DIR, 'degradation_results.json'), 'w') as f:
    json.dump(degradation_results, f, indent=2)

# Figure 9: Per-degradation performance
deg_df = pd.DataFrame(degradation_results).T
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(deg_df.index, deg_df['roc_auc'], color='steelblue')
axes[0].set_title('ROC-AUC by Degradation Type')
axes[0].set_ylabel('ROC-AUC')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(deg_df['roc_auc']):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

axes[1].bar(deg_df.index, deg_df['pr_auc'], color='coral')
axes[1].set_title('PR-AUC by Degradation Type')
axes[1].set_ylabel('PR-AUC')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(deg_df['pr_auc']):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

axes[2].bar(deg_df.index, deg_df['best_f1'], color='mediumseagreen')
axes[2].set_title('Best F1 by Degradation Type')
axes[2].set_ylabel('F1 Score')
axes[2].tick_params(axis='x', rotation=45)
for i, v in enumerate(deg_df['best_f1']):
    axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig9_per_degradation.png'), dpi=200)
plt.close()

# ============================================================
# 7. Calibration analysis for best model
# ============================================================
print("Generating calibration plot...")
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(8, 6))
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_proba_best, n_bins=10)
ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=best_model_name)
ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Plot (Reliability Diagram)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_IMG_DIR, 'fig11_calibration.png'), dpi=200)
plt.close()

print("Analysis complete. All outputs saved.")
