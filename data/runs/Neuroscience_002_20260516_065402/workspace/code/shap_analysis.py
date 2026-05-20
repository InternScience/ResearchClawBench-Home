#!/usr/bin/env python3
"""SHAP interpretability analysis for connectomics segment merging."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import json
from sklearn.preprocessing import StandardScaler
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
    
    # Train XGBoost again (already trained but let's do it here)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # SHAP analysis on a subset of test data
    sample_size = 2000
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # ============================================================
    # Figure SHAP 1: Summary plot (bar)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names, 
                      plot_type='bar', show=False, max_display=20)
    plt.title('SHAP Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'figA1_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] figA1_shap_bar.png')
    
    # ============================================================
    # Figure SHAP 2: Summary plot (beeswarm)
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names, 
                      show=False, max_display=20)
    plt.title('SHAP Beeswarm Plot (XGBoost)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'figA2_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] figA2_shap_beeswarm.png')
    
    # ============================================================
    # Figure SHAP 3: SHAP dependence for top features
    # ============================================================
    # Get top 4 features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top4_idx = np.argsort(mean_abs_shap)[-4:][::-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, feat_idx in zip(axes, top4_idx):
        shap.dependence_plot(feat_idx, shap_values, X_sample, 
                           feature_names=feat_names, show=False, ax=ax)
        ax.set_title(f'SHAP Dependence: {feat_names[feat_idx]}', 
                    fontsize=12, fontweight='bold')
    
    fig.suptitle('SHAP Dependence Plots for Top-4 Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'figA3_shap_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] figA3_shap_dependence.png')
    
    # ============================================================
    # Figure SHAP 4: SHAP by degradation type
    # ============================================================
    degradations = test['degradation'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (ax, deg) in enumerate(zip(axes, degradations)):
        mask = test['degradation'].values[idx] == deg
        if mask.sum() > 10:
            plt.sca(ax)
            shap.summary_plot(shap_values[mask], X_sample[mask], 
                            feature_names=feat_names, show=False, 
                            max_display=20, plot_type='bar')
            ax.set_title(f'{deg} (n={mask.sum()})', fontsize=11, fontweight='bold')
    
    fig.suptitle('SHAP Feature Importance by Degradation Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'figA4_shap_by_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] figA4_shap_by_degradation.png')
    
    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feat_names)
    shap_df['true_label'] = y_sample
    shap_df.to_csv(OUTPUTS + 'shap_values_sample.csv', index=False)
    print('[OK] shap_values_sample.csv saved')
    
    # Save SHAP importance summary
    shap_importance = pd.DataFrame({
        'feature': feat_names,
        'mean_abs_shap': mean_abs_shap,
        'feature_index': list(range(20))
    }).sort_values('mean_abs_shap', ascending=False)
    shap_importance.to_csv(OUTPUTS + 'shap_importance.csv', index=False)
    print('[OK] shap_importance.csv saved')
    print(shap_importance.to_string())

if __name__ == '__main__':
    main()
