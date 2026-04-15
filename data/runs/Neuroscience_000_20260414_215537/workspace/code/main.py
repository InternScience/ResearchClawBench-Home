import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, average_precision_score, 
                             confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

# Directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
print('Loading data...')
df = pd.read_csv('data/Together_1_targets_inserted.csv')
X_cols = [col for col in df.columns if col not in ['Unnamed: 0', 'Feature_1', 'Feature_2', 'Attack', 'Sniffing']]
X = df[X_cols].fillna(0)  # safe fill
y_attack = df['Attack']
y_sniffing = df['Sniffing']

print(f'X shape: {X.shape}')
print('Attack positives:', y_attack.sum())
print('Sniffing positives:', y_sniffing.sum())

# Splits
random_state = 42
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X, y_attack, test_size=0.2, stratify=y_attack, random_state=random_state)
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X, y_sniffing, test_size=0.2, stratify=y_sniffing, random_state=random_state)

def train_and_eval(X_tr, y_tr, X_te, y_te, name):
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=random_state, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    
    y_pred = rf.predict(X_te)
    y_proba = rf.predict_proba(X_te)[:, 1]
    
    ap = average_precision_score(y_te, y_proba)
    f1 = f1_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    
    print(f'{name} AP: {ap:.3f}, F1: {f1:.3f}')
    print(classification_report(y_te, y_pred))
    
    # PR curve
    prec_curve, rec_curve, _ = precision_recall_curve(y_te, y_proba)
    plt.figure()
    plt.plot(rec_curve, prec_curve, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {name}')
    plt.legend()
    plt.savefig(f'report/images/pr_curve_{name.lower().replace(\" \", \"_\")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'report/images/confusion_matrix_{name.lower().replace(\" \", \"_\")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature importance top 20
    imp_df = pd.DataFrame({
        'feature': X_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    imp_df.to_csv(f'outputs/feature_importance_{name.lower().replace(\" \", \"_\")}.csv', index=False)
    plt.figure(figsize=(10,6))
    sns.barplot(data=imp_df, x='importance', y='feature')
    plt.title(f'Top 20 Feature Importances - {name}')
    plt.savefig(f'report/images/feature_importance_{name.lower().replace(\" \", \"_\")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save model
    with open(f'outputs/rf_{name.lower().replace(\" \", \"_\")}.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    return {'AP': ap, 'F1': f1, 'Precision': prec, 'Recall': rec}

metrics_a = train_and_eval(X_tr_a, y_tr_a, X_te_a, y_te_a, 'Attack')
metrics_s = train_and_eval(X_tr_s, y_tr_s, X_te_s, y_te_s, 'Sniffing')

# Save metrics
metrics = {'Attack': metrics_a, 'Sniffing': metrics_s}
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2, default=float)

# Predictions
pred_df = pd.DataFrame({
    'frame': df['Unnamed: 0'][:len(X_te_a)],  # approx
    'attack_pred': rf_a.predict_proba(X_te_a)[:,1],
    'attack_true': y_te_a,
    'sniff_pred': rf_s.predict_proba(X_te_s)[:,1],
    'sniff_true': y_te_s
})
pred_df.to_csv('outputs/test_predictions.csv', index=False)

print('Training and evaluation complete. Check outputs/ and report/images/.')
print(metrics)","path">code/main.py