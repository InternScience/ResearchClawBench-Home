import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Create directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

# Features and target
target = 'label'
features = [str(i) for i in range(20)] + ['degradation']

# Encode categorical feature
le = LabelEncoder()
train_df['degradation_encoded'] = le.fit_transform(train_df['degradation'])
test_df['degradation_encoded'] = le.transform(test_df['degradation'])

X_train = train_df[[str(i) for i in range(20)] + ['degradation_encoded']]
y_train = train_df[target]

X_test = test_df[[str(i) for i in range(20)] + ['degradation_encoded']]
y_test = test_df[target]

# Train XGBoost
print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC'],
    'Value': [acc, prec, rec, f1, roc_auc]
})
metrics_df.to_csv('outputs/overall_metrics.csv', index=False)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('report/images/roc_curve.png')
plt.close()

# Feature Importance
importance = model.feature_importances_
features_names = [str(i) for i in range(20)] + ['degradation_encoded']
feat_imp = pd.DataFrame({'Feature': features_names, 'Importance': importance})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('report/images/feature_importance.png')
plt.close()

# Performance by degradation type
test_df['pred'] = y_pred
test_df['prob'] = y_prob

deg_results = []
for deg in test_df['degradation'].unique():
    subset = test_df[test_df['degradation'] == deg]
    y_true_sub = subset['label']
    y_pred_sub = subset['pred']
    y_prob_sub = subset['prob']
    
    acc_sub = accuracy_score(y_true_sub, y_pred_sub)
    f1_sub = f1_score(y_true_sub, y_pred_sub)
    roc_auc_sub = roc_auc_score(y_true_sub, y_prob_sub)
    
    deg_results.append({
        'Degradation': deg,
        'Accuracy': acc_sub,
        'F1-score': f1_sub,
        'ROC AUC': roc_auc_sub,
        'Count': len(subset)
    })

deg_df = pd.DataFrame(deg_results)
deg_df.to_csv('outputs/degradation_metrics.csv', index=False)

plt.figure(figsize=(10, 6))
deg_melted = pd.melt(deg_df, id_vars=['Degradation'], value_vars=['Accuracy', 'F1-score', 'ROC AUC'])
sns.barplot(x='Degradation', y='value', hue='variable', data=deg_melted)
plt.title('Performance by Degradation Type')
plt.ylim([0.0, 1.0])
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('report/images/degradation_performance.png')
plt.close()

print("Done.")
