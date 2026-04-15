"""
Neuron Segment Connectivity Prediction Analysis
==============================================

This script performs comprehensive analysis of the connectomics data
to predict whether two neuron segments belong to the same neuron.

Author: Research Analysis
Date: 2026-04-15
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
np.random.seed(42)

# Paths
DATA_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260415_170839/data"
OUTPUT_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260415_170839/outputs"
REPORT_IMG_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_002_20260415_170839/report/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

def load_data():
    """Load training and test data."""
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_simulated.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_simulated.csv"))
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

def analyze_data_distribution(train_df, test_df):
    """Analyze and visualize data distribution."""
    print("\n=== Data Distribution Analysis ===")
    
    # Label distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    train_counts = train_df['label'].value_counts()
    test_counts = test_df['label'].value_counts()
    
    axes[0].bar(['Different (0)', 'Same (1)'], train_counts.values, color=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Training Set Label Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(train_counts.values):
        axes[0].text(i, v + 1000, f'{v}\n({v/len(train_df)*100:.1f}%)', ha='center')
    
    axes[1].bar(['Different (0)', 'Same (1)'], test_counts.values, color=['#e74c3c', '#2ecc71'])
    axes[1].set_title('Test Set Label Distribution')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(test_counts.values):
        axes[1].text(i, v + 500, f'{v}\n({v/len(test_df)*100:.1f}%)', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'label_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Degradation type distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    train_deg = train_df['degradation'].value_counts()
    test_deg = test_df['degradation'].value_counts()
    
    colors_deg = {'Misalignment': '#3498db', 'Missing Sections': '#9b59b6', 
                  'Mixed': '#e67e22', 'Average': '#1abc9c'}
    train_colors = [colors_deg.get(d, '#95a5a6') for d in train_deg.index]
    test_colors = [colors_deg.get(d, '#95a5a6') for d in test_deg.index]
    
    axes[0].bar(train_deg.index, train_deg.values, color=train_colors)
    axes[0].set_title('Training Set: Degradation Type Distribution')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(test_deg.index, test_deg.values, color=test_colors)
    axes[1].set_title('Test Set: Degradation Type Distribution')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'degradation_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Train label distribution: {dict(train_counts)}")
    print(f"Test label distribution: {dict(test_counts)}")
    
    return train_counts, test_counts

def analyze_features(df, dataset_name="Train"):
    """Analyze feature statistics and distributions."""
    print(f"\n=== Feature Analysis: {dataset_name} ===")
    
    feature_cols = [str(i) for i in range(20)]
    
    # Feature statistics
    stats = df[feature_cols].describe()
    print(f"\nFeature statistics:\n{stats}")
    stats.to_csv(os.path.join(OUTPUT_DIR, f'feature_stats_{dataset_name.lower()}.csv'))
    
    # Sample for visualization (to speed up)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Feature distributions by label
    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        df_sample[df_sample['label'] == 0][col].hist(ax=ax, alpha=0.6, bins=30, label='Different (0)', color='#e74c3c')
        df_sample[df_sample['label'] == 1][col].hist(ax=ax, alpha=0.6, bins=30, label='Same (1)', color='#2ecc71')
        ax.set_title(f'Feature {col}')
        ax.legend(fontsize=7)
    
    plt.suptitle(f'Feature Distributions by Label ({dataset_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, f'feature_distributions_{dataset_name.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return stats

def correlation_analysis(df):
    """Analyze feature correlations."""
    print("\n=== Correlation Analysis ===")
    
    feature_cols = [str(i) for i in range(20)]
    corr_matrix = df[feature_cols].corr()
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Find highly correlated pairs
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    print(f"Highly correlated feature pairs (|r| > 0.5): {len(high_corr)}")
    for pair in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
    
    return corr_matrix

def prepare_data(train_df, test_df):
    """Prepare data for modeling."""
    print("\n=== Data Preparation ===")
    
    feature_cols = [str(i) for i in range(20)]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Train and evaluate multiple models."""
    print("\n=== Model Training and Evaluation ===")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
        'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, 
                                          random_state=42, early_stopping=True, validation_fraction=0.1)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'pr_auc': average_precision_score(y_test, y_prob)
        }
        
        results[name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))
    print(f"\nModel Comparison:\n{results_df}")
    
    return trained_models, results_df

def plot_model_comparison(results_df):
    """Create visualization of model comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(results_df.index, results_df[metric], color=colors)
        ax.set_ylim(0, 1)
        ax.set_title(f'{metric.upper().replace("_", "-")} Score')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_pr_curves(trained_models, X_test, y_test):
    """Plot ROC and Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curves
    ax1 = axes[0]
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    ax2 = axes[1]
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax2.plot(recall, precision, label=f'{name} (AP={pr_auc:.3f})', linewidth=2)
    
    baseline = np.sum(y_test) / len(y_test)
    ax2.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'roc_pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(trained_models, X_test, y_test):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()

def analyze_by_degradation_type(test_df, best_model, scaler):
    """Analyze model performance across degradation types."""
    print("\n=== Degradation Type Analysis ===")
    
    feature_cols = [str(i) for i in range(20)]
    degradation_results = {}
    
    for deg_type in test_df['degradation'].unique():
        subset = test_df[test_df['degradation'] == deg_type]
        X_sub = scaler.transform(subset[feature_cols].values)
        y_sub = subset['label'].values
        
        y_pred = best_model.predict(X_sub)
        y_prob = best_model.predict_proba(X_sub)[:, 1]
        
        degradation_results[deg_type] = {
            'n_samples': len(subset),
            'accuracy': accuracy_score(y_sub, y_pred),
            'f1': f1_score(y_sub, y_pred),
            'roc_auc': roc_auc_score(y_sub, y_prob)
        }
    
    deg_df = pd.DataFrame(degradation_results).T
    print(f"\nPerformance by Degradation Type:\n{deg_df}")
    deg_df.to_csv(os.path.join(OUTPUT_DIR, 'degradation_analysis.csv'))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['accuracy', 'f1', 'roc_auc']
    colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(deg_df.index, deg_df[metric], color=colors[:len(deg_df)])
        ax.set_ylim(0, 1)
        ax.set_title(f'{metric.upper()} by Degradation Type')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Model Performance Across Degradation Types\n({best_model.__class__.__name__})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'degradation_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return degradation_results

def feature_importance_analysis(model, feature_names):
    """Analyze feature importance."""
    print("\n=== Feature Importance Analysis ===")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not provide feature importance.")
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:\n{importance_df.head(10)}")
    importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
    bars = ax.barh(importance_df['feature'][::-1], importance_df['importance'][::-1], color=colors[::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance Ranking')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return importance_df

def cross_validation_analysis(X_train, y_train):
    """Perform cross-validation analysis on a subset for speed."""
    print("\n=== Cross-Validation Analysis ===")
    
    # Sample for CV speed
    sample_size = min(10000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train[indices]
    y_sample = y_train[indices]
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42)
    }
    
    cv_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Cross-validating {name}...")
        scores = cross_val_score(model, X_sample, y_sample, cv=cv, scoring='f1', n_jobs=-1)
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(cv_results.keys())
    means = [cv_results[n]['mean'] for n in names]
    stds = [cv_results[n]['std'] for n in names]
    
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylabel('F1 Score')
    ax.set_title('5-Fold Cross-Validation F1 Scores (10K sample)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMG_DIR, 'cross_validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return cv_results

def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Neuron Segment Connectivity Prediction Analysis")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_data()
    
    # Data overview
    train_counts, test_counts = analyze_data_distribution(train_df, test_df)
    train_stats = analyze_features(train_df, "Train")
    test_stats = analyze_features(test_df, "Test")
    
    # Correlation analysis
    corr_matrix = correlation_analysis(train_df)
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_data(train_df, test_df)
    
    # Cross-validation
    cv_results = cross_validation_analysis(X_train, y_train)
    
    # Train and evaluate models
    trained_models, results_df = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    plot_model_comparison(results_df)
    plot_roc_pr_curves(trained_models, X_test, y_test)
    plot_confusion_matrices(trained_models, X_test, y_test)
    
    # Identify best model (by F1 score)
    best_model_name = results_df['f1'].idxmax()
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")
    
    # Feature importance
    feature_names = [f'Feature_{i}' for i in range(20)]
    importance_df = feature_importance_analysis(best_model, feature_names)
    
    # Degradation type analysis
    degradation_results = analyze_by_degradation_type(test_df, best_model, scaler)
    
    # Save summary
    summary = {
        'dataset_info': {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': 20,
            'train_positive_ratio': float(train_counts.get(1, 0) / len(train_df)),
            'test_positive_ratio': float(test_counts.get(1, 0) / len(test_df))
        },
        'best_model': best_model_name,
        'best_model_metrics': results_df.loc[best_model_name].to_dict(),
        'degradation_results': degradation_results,
        'cv_results': cv_results
    }
    
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {REPORT_IMG_DIR}")
    print("=" * 60)
    
    return summary

if __name__ == "__main__":
    summary = main()
