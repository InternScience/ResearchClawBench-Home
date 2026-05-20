#!/usr/bin/env python3
"""Model training and evaluation for connectomics segment merging."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                              accuracy_score, precision_score, recall_score, 
                              f1_score, roc_curve, precision_recall_curve,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

OUT = 'report/images/'
DATA = 'data/'
OUTPUTS = 'outputs/'

def evaluate_model(model, X_test, y_test, model_name):
    """Compute comprehensive evaluation metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    return {
        'model': model_name,
        'auc_roc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_prob': y_prob,
        'y_pred': y_pred,
    }

def evaluate_by_degradation(model, X_test, y_test, test_df, model_name):
    """Evaluate model performance per degradation type."""
    results = []
    for deg in test_df['degradation'].unique():
        mask = test_df['degradation'] == deg
        X_d = X_test[mask]
        y_d = y_test[mask]
        y_prob = model.predict_proba(X_d)[:, 1]
        y_pred = model.predict(X_d)
        
        results.append({
            'degradation': deg,
            'n_samples': len(y_d),
            'n_positive': int(y_d.sum()),
            'auc_roc': roc_auc_score(y_d, y_prob),
            'avg_precision': average_precision_score(y_d, y_prob),
            'accuracy': accuracy_score(y_d, y_pred),
            'precision': precision_score(y_d, y_pred),
            'recall': recall_score(y_d, y_pred),
            'f1': f1_score(y_d, y_pred),
        })
    return results

def main():
    # Load data
    train = pd.read_csv(DATA + 'train_simulated.csv')
    test = pd.read_csv(DATA + 'test_simulated.csv')
    
    feat_cols = [str(i) for i in range(20)]
    X_train = train[feat_cols].values
    y_train = train['label'].values
    X_test = test[feat_cols].values
    y_test = test['label'].values
    
    print(f"Train: {X_train.shape}, Positive ratio: {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape}, Positive ratio: {y_test.mean():.4f}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute scale_pos_weight for XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # ============================================================
    # Define models
    # ============================================================
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=2000, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        ),
        'MLP (Neural Net)': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            alpha=0.001, batch_size=256, max_iter=200, 
            early_stopping=True, random_state=42
        ),
    }
    
    # ============================================================
    # Train and evaluate all models
    # ============================================================
    all_results = []
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        result = evaluate_model(model, X_test_scaled, y_test, name)
        all_results.append(result)
        
        print(f"  AUC-ROC: {result['auc_roc']:.4f}")
        print(f"  Avg Precision: {result['avg_precision']:.4f}")
        print(f"  F1 Score: {result['f1']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
    
    # Save overall results
    overall_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('y_prob', 'y_pred')} 
                                for r in all_results])
    overall_df.to_csv(OUTPUTS + 'overall_results.csv', index=False)
    print(f"\nOverall results saved to {OUTPUTS}overall_results.csv")
    print(overall_df.to_string())
    
    # ============================================================
    # Per-degradation evaluation for best model (XGBoost)
    # ============================================================
    best_model = models['XGBoost']
    deg_results = evaluate_by_degradation(best_model, X_test_scaled, y_test, test, 'XGBoost')
    
    # Also get degradation results for all models
    all_deg_results = {}
    for name, model in models.items():
        deg_res = evaluate_by_degradation(model, X_test_scaled, y_test, test, name)
        all_deg_results[name] = deg_res
    
    # Save degradation results
    deg_flat = []
    for name, results in all_deg_results.items():
        for r in results:
            r['model'] = name
            deg_flat.append(r)
    deg_df = pd.DataFrame(deg_flat)
    deg_df.to_csv(OUTPUTS + 'degradation_results.csv', index=False)
    
    # ============================================================
    # Figure 6: ROC curves for all models
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Overall ROC
    for result, color in zip(all_results, colors):
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        axes[0].plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{result['model']} (AUC={result['auc_roc']:.4f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title('ROC Curves - All Models', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall curves
    for result, color in zip(all_results, colors):
        precision, recall, _ = precision_recall_curve(y_test, result['y_prob'])
        axes[1].plot(recall, precision, color=color, linewidth=2,
                    label=f"{result['model']} (AP={result['avg_precision']:.4f})")
    axes[1].axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.5, 
                   label=f'Random (AP={y_test.mean():.4f})')
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title('Precision-Recall Curves - All Models', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Model Performance Comparison on Connectomics Segment Merging Task', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig6_model_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig6_model_curves.png')
    
    # ============================================================
    # Figure 7: Performance by degradation type (bar chart)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metrics = ['auc_roc', 'avg_precision', 'f1', 'recall']
    metric_names = ['AUC-ROC', 'Average Precision', 'F1 Score', 'Recall']
    
    for ax, metric, mname in zip(axes, metrics, metric_names):
        model_names = list(all_deg_results.keys())
        degradations_list = test['degradation'].unique()
        x = np.arange(len(degradations_list))
        width = 0.15
        
        for i, (mname_model, color) in enumerate(zip(model_names, colors)):
            values = [r[metric] for r in all_deg_results[mname_model]]
            ax.bar(x + i*width, values, width, label=mname_model, color=color, 
                  edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Degradation Type', fontsize=11)
        ax.set_ylabel(mname, fontsize=11)
        ax.set_title(f'{mname} by Degradation Type', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(degradations_list, fontsize=9)
        if metric == 'auc_roc':
            ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Model Performance Across Degradation Types', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig7_degradation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig7_degradation_comparison.png')
    
    # ============================================================
    # Figure 8: Confusion matrices for best model
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, deg in zip(axes, test['degradation'].unique()):
        mask = test['degradation'] == deg
        y_prob_best = best_model.predict_proba(X_test_scaled[mask])[:, 1]
        y_pred_best = best_model.predict(X_test_scaled[mask])
        cm = confusion_matrix(y_test[mask], y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Pred 0', 'Pred 1'], 
                   yticklabels=['True 0', 'True 1'])
        ax.set_title(f'{deg}\nAcc={accuracy_score(y_test[mask], y_pred_best):.3f}, '
                    f'F1={f1_score(y_test[mask], y_pred_best):.3f}', 
                    fontsize=11, fontweight='bold')
    
    fig.suptitle('Confusion Matrices by Degradation Type (XGBoost)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig8_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig8_confusion_matrices.png')
    
    # ============================================================
    # Figure 9: Feature importance (XGBoost)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # XGBoost feature importance
    xgb_importance = best_model.feature_importances_
    idx = np.argsort(xgb_importance)
    axes[0].barh(range(20), xgb_importance[idx], color='#2ecc71', edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels([feat_cols[i] for i in idx])
    axes[0].set_xlabel('Importance (Gain)', fontsize=11)
    axes[0].set_title('XGBoost Feature Importance', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Random Forest feature importance
    rf = models['Random Forest']
    rf_importance = rf.feature_importances_
    idx_rf = np.argsort(rf_importance)
    axes[1].barh(range(20), rf_importance[idx_rf], color='#3498db', edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([feat_cols[i] for i in idx_rf])
    axes[1].set_xlabel('Importance (MDI)', fontsize=11)
    axes[1].set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig9_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig9_feature_importance.png')
    
    # ============================================================
    # Figure 10: Calibration curves
    # ============================================================
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for result, color in zip(all_results, colors):
        prob_true, prob_pred = calibration_curve(y_test, result['y_prob'], n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', color=color, linewidth=2, 
               markersize=6, label=result['model'])
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('Calibration Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUT + 'fig10_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig10_calibration.png')
    
    # ============================================================
    # Save detailed results to JSON
    # ============================================================
    summary = {
        'overall': overall_df.to_dict(orient='records'),
        'by_degradation': deg_df.to_dict(orient='records'),
    }
    with open(OUTPUTS + 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n[OK] evaluation_summary.json saved')
    
    # ============================================================
    # Save best model predictions
    # ============================================================
    y_prob_final = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_final = best_model.predict(X_test_scaled)
    
    pred_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred_final,
        'predicted_probability': y_prob_final,
        'degradation': test['degradation'].values
    })
    pred_df.to_csv(OUTPUTS + 'predictions.csv', index=False)
    print('[OK] predictions.csv saved')
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    best_idx = np.argmax([r['auc_roc'] for r in all_results])
    best = all_results[best_idx]
    print(f"Best model: {best['model']}")
    print(f"  AUC-ROC: {best['auc_roc']:.4f}")
    print(f"  Average Precision: {best['avg_precision']:.4f}")
    print(f"  F1 Score: {best['f1']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Accuracy: {best['accuracy']:.4f}")

if __name__ == '__main__':
    main()
