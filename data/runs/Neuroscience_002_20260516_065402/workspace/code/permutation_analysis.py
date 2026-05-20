#!/usr/bin/env python3
"""Permutation importance and additional interpretability analysis."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

OUT = 'report/images/'
DATA = 'data/'
OUTPUTS = 'outputs/'

def main():
    train = pd.read_csv(DATA + 'train_simulated.csv')
    test = pd.read_csv(DATA + 'test_simulated.csv')
    feat_cols = [str(i) for i in range(20)]
    feat_names = [f'F{i}' for i in range(20)]
    
    X_train = train[feat_cols].values
    y_train = train['label'].values
    X_test = test[feat_cols].values
    y_test = test['label'].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Load or retrain best model (MLP)
    print("Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        alpha=0.001, batch_size=256, max_iter=200,
        early_stopping=True, random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    # Permutation importance for MLP
    print("Computing permutation importance for MLP...")
    perm_result = permutation_importance(
        mlp, X_test_scaled, y_test, n_repeats=10, random_state=42,
        scoring='roc_auc', n_jobs=-1
    )
    
    # Also for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    
    print("Computing permutation importance for XGBoost...")
    perm_result_xgb = permutation_importance(
        xgb, X_test, y_test, n_repeats=10, random_state=42,
        scoring='roc_auc', n_jobs=-1
    )
    
    # Save permutation importance
    perm_df = pd.DataFrame({
        'feature': feat_names,
        'mlp_importance_mean': perm_result.importances_mean,
        'mlp_importance_std': perm_result.importances_std,
        'xgb_importance_mean': perm_result_xgb.importances_mean,
        'xgb_importance_std': perm_result_xgb.importances_std,
    }).sort_values('mlp_importance_mean', ascending=False)
    perm_df.to_csv(OUTPUTS + 'permutation_importance.csv', index=False)
    print(perm_df.to_string())
    
    # ============================================================
    # Figure: Permutation importance comparison
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # MLP
    idx = np.argsort(perm_result.importances_mean)
    axes[0].barh(range(20), perm_result.importances_mean[idx], 
                xerr=perm_result.importances_std[idx],
                color='#9b59b6', edgecolor='black', linewidth=0.5, 
                capsize=3)
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels([feat_names[i] for i in idx])
    axes[0].set_xlabel('Mean Decrease in AUC-ROC', fontsize=11)
    axes[0].set_title('Permutation Importance (MLP)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # XGBoost
    idx_xgb = np.argsort(perm_result_xgb.importances_mean)
    axes[1].barh(range(20), perm_result_xgb.importances_mean[idx_xgb],
                xerr=perm_result_xgb.importances_std[idx_xgb],
                color='#2ecc71', edgecolor='black', linewidth=0.5,
                capsize=3)
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([feat_names[i] for i in idx_xgb])
    axes[1].set_xlabel('Mean Decrease in AUC-ROC', fontsize=11)
    axes[1].set_title('Permutation Importance (XGBoost)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Permutation Feature Importance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'figB1_permutation_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] figB1_permutation_importance.png')
    
    # ============================================================
    # Figure: Threshold analysis
    # ============================================================
    y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 50)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precisions.append(precision_score(y_test, y_pred_t))
        recalls.append(recall_score(y_test, y_pred_t))
        f1s.append(f1_score(y_test, y_pred_t))
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
    ax.plot(thresholds, f1s, 'g-', linewidth=2.5, label='F1 Score')
    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(x=best_t, color='green', linestyle='--', alpha=0.5, 
              label=f'Best F1 threshold = {best_t:.2f}')
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision-Recall-F1 vs. Decision Threshold (MLP)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT + 'figB2_threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] figB2_threshold_analysis.png')
    
    # ============================================================
    # Figure: Score distribution by label
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_prob[y_test == 0], bins=50, alpha=0.6, color='#3498db', 
           label='Different Neuron (0)', density=True)
    ax.hist(y_prob[y_test == 1], bins=50, alpha=0.6, color='#e74c3c',
           label='Same Neuron (1)', density=True)
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prediction Score Distribution (MLP)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT + 'figB3_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] figB3_score_distribution.png')
    
if __name__ == '__main__':
    main()
