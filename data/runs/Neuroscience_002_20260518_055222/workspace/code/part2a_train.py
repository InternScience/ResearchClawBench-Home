"""
Part 2a: Train models
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                              accuracy_score, average_precision_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import json
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess
train_df = pd.read_csv('data/train_simulated.csv')
test_df = pd.read_csv('data/test_simulated.csv')

feature_cols = [str(i) for i in range(20)]
X_train = train_df[feature_cols].values
y_train = train_df['label'].values

X_test = test_df[feature_cols].values
y_test = test_df['label'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, 
                                           class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, 
                                                    learning_rate=0.1, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                         random_state=42, early_stopping=True),
}

all_results = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    all_results[name] = {
        'y_pred': y_pred.tolist(),
        'y_prob': y_prob.tolist(),
        'auc': float(roc_auc_score(y_test, y_prob)),
        'ap': float(average_precision_score(y_test, y_prob)),
        'f1': float(f1_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'accuracy': float(accuracy_score(y_test, y_pred))
    }
    
    print(f"    AUC: {all_results[name]['auc']:.4f}, F1: {all_results[name]['f1']:.4f}")

# Degradation-specific
degradations = ['Misalignment', 'Missing Sections', 'Mixed', 'Average']
best_name = max(all_results.keys(), key=lambda x: all_results[x]['auc'])
best_model = models[best_name]

print(f"\nBest: {best_name}")

deg_results = {}
for deg in degradations:
    mask = test_df['degradation'].values == deg
    y_prob_d = best_model.predict_proba(X_test_scaled[mask])[:, 1]
    y_pred_d = best_model.predict(X_test_scaled[mask])
    y_d = y_test[mask]
    
    deg_results[deg] = {
        'auc': float(roc_auc_score(y_d, y_prob_d)),
        'f1': float(f1_score(y_d, y_pred_d)),
        'ap': float(average_precision_score(y_d, y_prob_d)),
        'precision': float(precision_score(y_d, y_pred_d)),
        'recall': float(recall_score(y_d, y_pred_d)),
        'n': int(len(y_d))
    }
    print(f"  {deg}: AUC={deg_results[deg]['auc']:.4f}")

# Save results (without y_pred/y_prob arrays to keep file small)
save_results = {k: {kk: vv for kk, vv in v.items() if kk not in ['y_pred', 'y_prob']} 
                for k, v in all_results.items()}
save_results['best_model'] = best_name
save_results['degradation'] = deg_results

with open('outputs/model_results.json', 'w') as f:
    json.dump(save_results, f, indent=2)

# Save y_prob for plotting
np.save('outputs/y_probs.npy', {name: all_results[name]['y_prob'] for name in all_results})

# Save predictions for each model
for name in all_results:
    np.save(f'outputs/y_prob_{name.replace(" ", "_").lower()}.npy', np.array(all_results[name]['y_prob']))

print("\n✓ Models trained and results saved")
