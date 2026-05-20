"""
SimBA-style behavior classifier training and evaluation.
Trains Random Forest, Gradient Boosting, and Logistic Regression classifiers
for Attack and Sniffing behavior classification from pose-derived features.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_auc_score,
                              average_precision_score, precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.figsize': (8, 6)
})
sns.set_style("whitegrid")

def load_data():
    """Load engineered features and labels."""
    X = pd.read_csv('outputs/engineered_features.csv')
    y = pd.read_csv('outputs/behavior_labels.csv')
    return X, y

def train_evaluate_classifier(clf, clf_name, X_train, X_test, y_train, y_test, behavior_name):
    """Train and evaluate a single classifier for one behavior."""
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
    
    # Metrics
    results = {
        'classifier': clf_name,
        'behavior': behavior_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    if hasattr(clf, 'predict_proba'):
        results['roc_auc'] = roc_auc_score(y_test, y_prob)
        results['pr_auc'] = average_precision_score(y_test, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return clf, results, cm, y_pred, y_prob

def run_full_analysis():
    """Run the complete analysis pipeline."""
    print("="*70)
    print("SimBA-Style Behavior Classification: Full Analysis Pipeline")
    print("="*70)
    
    # Load data
    X, y = load_data()
    print(f"\nLoaded data: {X.shape[0]} frames, {X.shape[1]} features")
    print(f"Attack positive: {y['Attack'].sum()} ({y['Attack'].mean()*100:.1f}%)")
    print(f"Sniffing positive: {y['Sniffing'].sum()} ({y['Sniffing'].mean()*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # === Train/Test Split ===
    # Use stratified split to preserve class balance
    # For multi-label, we'll split based on both labels
    combined_label = y['Attack'] * 2 + y['Sniffing']  # 0: none, 1: sniffing only, 2: attack only, 3: both
    # But if there's overlap, handle carefully
    # Actually, let's just do separate splits per behavior
    
    behaviors = ['Attack', 'Sniffing']
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                                  min_samples_split=5, min_samples_leaf=2,
                                                  class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                          learning_rate=0.05, subsample=0.8,
                                                          random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', 
                                                    random_state=42, C=0.5),
    }
    
    all_results = []
    trained_models = {}
    
    for behavior in behaviors:
        print(f"\n{'='*70}")
        print(f"Behavior: {behavior}")
        print(f"{'='*70}")
        
        y_b = y[behavior].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_b, test_size=0.25, random_state=42, stratify=y_b
        )
        
        print(f"Train: {len(X_train)} frames ({y_train.mean()*100:.1f}% positive)")
        print(f"Test: {len(X_test)} frames ({y_test.mean()*100:.1f}% positive)")
        
        for clf_name, clf_template in classifiers.items():
            print(f"\n  Training {clf_name}...")
            clf = clf_template.__class__(**clf_template.get_params())
            
            clf, results, cm, y_pred, y_prob = train_evaluate_classifier(
                clf, clf_name, X_train, X_test, y_train, y_test, behavior
            )
            
            trained_models[f"{clf_name}_{behavior}"] = {
                'model': clf,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
            }
            
            all_results.append(results)
            
            # Print results
            print(f"    Accuracy:  {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall:    {results['recall']:.4f}")
            print(f"    F1:        {results['f1']:.4f}")
            if 'roc_auc' in results:
                print(f"    ROC AUC:   {results['roc_auc']:.4f}")
                print(f"    PR AUC:    {results['pr_auc']:.4f}")
    
    # === Save Results ===
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('outputs/classification_results.csv', index=False)
    print(f"\nResults saved to outputs/classification_results.csv")
    
    # Save detailed results as JSON
    with open('outputs/classification_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # === Generate Figures ===
    print("\nGenerating figures...")
    generate_figures(trained_models, results_df, X, y, classifiers)
    
    # === Feature Importance ===
    print("\nComputing feature importance...")
    compute_feature_importance(trained_models, X_scaled, y, behaviors)
    
    return trained_models, results_df

def generate_figures(trained_models, results_df, X, y, classifiers):
    """Generate all required figures."""
    behaviors = ['Attack', 'Sniffing']
    
    # Figure 1: Results comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for b_idx, behavior in enumerate(behaviors):
        ax = axes[b_idx]
        behav_results = results_df[results_df['behavior'] == behavior]
        
        x_pos = np.arange(len(metrics))
        width = 0.25
        
        for i, (_, row) in enumerate(behav_results.iterrows()):
            values = [row[m] for m in metrics]
            bars = ax.bar(x_pos + i*width, values, width, 
                         label=row['classifier'], color=colors[i % len(colors)])
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title(f'{behavior} Classification Performance')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('report/images/classification_performance.png', bbox_inches='tight')
    plt.close()
    print("  Saved: classification_performance.png")
    
    # Figure 2: Confusion Matrices for best classifier per behavior
    # Determine best classifier (by F1)
    best_clf_per_behavior = {}
    for behavior in behaviors:
        behav_results = results_df[results_df['behavior'] == behavior]
        best_row = behav_results.loc[behav_results['f1'].idxmax()]
        best_clf_per_behavior[behavior] = best_row['classifier']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for b_idx, behavior in enumerate(behaviors):
        ax = axes[b_idx]
        best_name = best_clf_per_behavior[behavior]
        key = f"{best_name}_{behavior}"
        data = trained_models[key]
        
        cm = confusion_matrix(data['y_test'], data['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Not ' + behavior, behavior],
                   yticklabels=['Not ' + behavior, behavior])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{behavior} - {best_name}\nConfusion Matrix')
    
    plt.tight_layout()
    plt.savefig('report/images/confusion_matrices.png', bbox_inches='tight')
    plt.close()
    print("  Saved: confusion_matrices.png")
    
    # Figure 3: Precision-Recall Curves for best classifier
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for b_idx, behavior in enumerate(behaviors):
        ax = axes[b_idx]
        best_name = best_clf_per_behavior[behavior]
        key = f"{best_name}_{behavior}"
        data = trained_models[key]
        
        precision, recall, thresholds = precision_recall_curve(
            data['y_test'], data['y_prob']
        )
        pr_auc = average_precision_score(data['y_test'], data['y_prob'])
        
        ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR AUC = {pr_auc:.3f}')
        ax.fill_between(recall, precision, alpha=0.2)
        
        # Baseline (random)
        baseline = data['y_test'].mean()
        ax.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
                  label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{behavior} - {best_name}\nPrecision-Recall Curve')
        ax.legend(loc='lower left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('report/images/precision_recall_curves.png', bbox_inches='tight')
    plt.close()
    print("  Saved: precision_recall_curves.png")
    
    # Figure 4: ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for b_idx, behavior in enumerate(behaviors):
        ax = axes[b_idx]
        best_name = best_clf_per_behavior[behavior]
        key = f"{best_name}_{behavior}"
        data = trained_models[key]
        
        fpr, tpr, _ = roc_curve(data['y_test'], data['y_prob'])
        roc_auc = roc_auc_score(data['y_test'], data['y_prob'])
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.1)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{behavior} - {best_name}\nROC Curve')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('report/images/roc_curves.png', bbox_inches='tight')
    plt.close()
    print("  Saved: roc_curves.png")
    
    # Figure 5: Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for b_idx, behavior in enumerate(behaviors):
        ax = axes[b_idx]
        counts = [y[behavior].value_counts().get(0, 0), y[behavior].value_counts().get(1, 0)]
        colors_pie = ['#95a5a6', '#e74c3c']
        ax.pie(counts, labels=['Absent', 'Present'], autopct='%1.1f%%',
              colors=colors_pie, explode=(0, 0.05), startangle=90)
        ax.set_title(f'{behavior} Class Distribution\n(n={len(y)})')
    
    plt.tight_layout()
    plt.savefig('report/images/class_distribution.png', bbox_inches='tight')
    plt.close()
    print("  Saved: class_distribution.png")
    
    # Figure 6: Cross-validation stability
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cv_results = []
    for behavior in behaviors:
        y_b = y[behavior].values
        for clf_name, clf_template in classifiers.items():
            clf = clf_template.__class__(**clf_template.get_params())
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y_b, cv=cv, scoring='f1')
            for score in scores:
                cv_results.append({
                    'Behavior': behavior,
                    'Classifier': clf_name,
                    'F1 Score': score
                })
    
    cv_df = pd.DataFrame(cv_results)
    
    # Violin plot or box plot
    sns.boxplot(data=cv_df, x='Classifier', y='F1 Score', hue='Behavior', palette=['#e74c3c', '#3498db'])
    ax.set_title('5-Fold Cross-Validation F1 Scores')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('report/images/cross_validation.png', bbox_inches='tight')
    plt.close()
    print("  Saved: cross_validation.png")
    
    # Figure 7: Feature correlation heatmap (top features)
    top_features = get_top_features(trained_models, X, behaviors, n=15)
    
    if len(top_features) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = X[top_features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, annot=False,
                   ax=ax, vmin=-1, vmax=1, square=True)
        ax.set_title('Top Feature Correlations')
        plt.tight_layout()
        plt.savefig('report/images/feature_correlations.png', bbox_inches='tight')
        plt.close()
        print("  Saved: feature_correlations.png")
    
    # Figure 8: Classifier comparison across behaviors
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = results_df.pivot_table(values='f1', index='classifier', columns='behavior', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'], rot=0)
    ax.set_xlabel('Classifier')
    ax.set_ylabel('F1 Score')
    ax.set_title('Classifier F1 Score Comparison Across Behaviors')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(title='Behavior')
    
    plt.tight_layout()
    plt.savefig('report/images/classifier_comparison.png', bbox_inches='tight')
    plt.close()
    print("  Saved: classifier_comparison.png")

def get_top_features(trained_models, X, behaviors, n=10):
    """Get top features across behaviors from RF models."""
    all_top = set()
    for behavior in behaviors:
        key = f"Random Forest_{behavior}"
        if key in trained_models:
            model = trained_models[key]['model']
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-n:]
            all_top.update([X.columns[i] for i in top_idx])
    return list(all_top)

def compute_feature_importance(trained_models, X, y, behaviors):
    """Compute and save feature importance tables."""
    for behavior in behaviors:
        print(f"\n  Feature Importance for {behavior}:")
        
        for clf_name in ['Random Forest', 'Gradient Boosting']:
            key = f"{clf_name}_{behavior}"
            if key in trained_models:
                model = trained_models[key]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    top_features = []
                    for i in indices[:20]:
                        top_features.append({
                            'rank': len(top_features) + 1,
                            'feature': X.columns[i],
                            'importance': float(importances[i])
                        })
                    
                    feat_df = pd.DataFrame(top_features)
                    feat_df.to_csv(f'outputs/feature_importance_{clf_name.replace(" ", "_")}_{behavior}.csv', 
                                  index=False)
                    print(f"    {clf_name}: Top 5 features:")
                    for _, row in feat_df.head(5).iterrows():
                        print(f"      {row['rank']}. {row['feature']}: {row['importance']:.4f}")
        
        # Also permutation importance for Logistic Regression
        key = f"Logistic Regression_{behavior}"
        if key in trained_models:
            data = trained_models[key]
            perm_imp = permutation_importance(
                data['model'], data['X_test'], data['y_test'],
                n_repeats=10, random_state=42, n_jobs=-1
            )
            indices = np.argsort(perm_imp.importances_mean)[::-1]
            
            top_perm = []
            for i in indices[:20]:
                top_perm.append({
                    'rank': len(top_perm) + 1,
                    'feature': X.columns[i],
                    'importance_mean': float(perm_imp.importances_mean[i]),
                    'importance_std': float(perm_imp.importances_std[i])
                })
            
            perm_df = pd.DataFrame(top_perm)
            perm_df.to_csv(f'outputs/feature_importance_Logistic_Regression_{behavior}.csv', 
                          index=False)
    
    # === Generate Feature Importance Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for b_idx, behavior in enumerate(behaviors):
        ax = axes[b_idx]
        key = f"Random Forest_{behavior}"
        if key in trained_models:
            model = trained_models[key]['model']
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:]
            
            ax.barh(range(len(indices)), importances[indices], color='#2c3e50')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([X.columns[i] for i in indices], fontsize=7)
            ax.set_xlabel('Importance')
            ax.set_title(f'{behavior} - Random Forest\nFeature Importance (Top 15)')
    
    plt.tight_layout()
    plt.savefig('report/images/feature_importance.png', bbox_inches='tight')
    plt.close()
    print("  Saved: feature_importance.png")
    
    # === Comparison feature importance ===
    # Unified feature importance across behaviors
    fig, ax = plt.subplots(figsize=(12, 8))
    
    combined_importance = {}
    for behavior in behaviors:
        for clf_name in ['Random Forest', 'Gradient Boosting']:
            key = f"{clf_name}_{behavior}"
            if key in trained_models:
                model = trained_models[key]['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, col in enumerate(X.columns):
                        if col not in combined_importance:
                            combined_importance[col] = {}
                        combined_importance[col][f"{clf_name}_{behavior}"] = float(importances[i])
    
    if combined_importance:
        imp_df = pd.DataFrame(combined_importance).T
        # Average across classifiers for each behavior
        for behavior in behaviors:
            cols = [c for c in imp_df.columns if behavior in c]
            if cols:
                imp_df[f'{behavior}_avg'] = imp_df[cols].mean(axis=1)
        
        # Get top features by average importance
        avg_cols = [c for c in imp_df.columns if c.endswith('_avg')]
        if avg_cols:
            imp_df['total_avg'] = imp_df[avg_cols].mean(axis=1)
            top_20 = imp_df.nlargest(20, 'total_avg')
            top_20[avg_cols].plot(kind='barh', ax=ax)
            ax.set_xlabel('Average Feature Importance')
            ax.set_title('Top 20 Features: Average Importance Across Classifiers')
            ax.legend(loc='lower right')
            
            plt.tight_layout()
            plt.savefig('report/images/feature_importance_combined.png', bbox_inches='tight')
            plt.close()
            print("  Saved: feature_importance_combined.png")

def main():
    trained_models, results_df = run_full_analysis()
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for behavior in ['Attack', 'Sniffing']:
        behav_results = results_df[results_df['behavior'] == behavior]
        best_row = behav_results.loc[behav_results['f1'].idxmax()]
        print(f"\n{behavior} - Best Classifier: {best_row['classifier']}")
        print(f"  Accuracy:  {best_row['accuracy']:.4f}")
        print(f"  Precision: {best_row['precision']:.4f}")
        print(f"  Recall:    {best_row['recall']:.4f}")
        print(f"  F1 Score:  {best_row['f1']:.4f}")
        if 'roc_auc' in best_row:
            print(f"  ROC AUC:   {best_row['roc_auc']:.4f}")
    
    return trained_models, results_df

if __name__ == '__main__':
    trained_models, results_df = main()
