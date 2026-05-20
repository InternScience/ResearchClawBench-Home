"""
Generate remaining figures and feature importance tables.
Separated from main script to avoid timeouts.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
})
sns.set_style("whitegrid")

def main():
    X = pd.read_csv('outputs/engineered_features.csv')
    y = pd.read_csv('outputs/behavior_labels.csv')
    results_df = pd.read_csv('outputs/classification_results.csv')
    
    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
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
    
    behaviors = ['Attack', 'Sniffing']
    
    # Train models on full data for feature importance
    print("Training models on full data for feature importance...")
    models = {}
    for behavior in behaviors:
        y_b = y[behavior].values
        for clf_name, clf_template in classifiers.items():
            clf = clf_template.__class__(**clf_template.get_params())
            clf.fit(X_scaled, y_b)
            models[f"{clf_name}_{behavior}"] = clf
            print(f"  {clf_name} - {behavior}: trained")
    
    # Figure: Feature Importance (RF top 15 per behavior)
    print("\nGenerating feature importance plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for b_idx, behavior in enumerate(behaviors):
        ax = axes[b_idx]
        clf = models[f"Random Forest_{behavior}"]
        importances = clf.feature_importances_
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
    
    # Figure: Combined feature importance
    print("Generating combined feature importance...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    combined_importance = {}
    for behavior in behaviors:
        for clf_name in ['Random Forest', 'Gradient Boosting']:
            clf = models[f"{clf_name}_{behavior}"]
            importances = clf.feature_importances_
            for i, col in enumerate(X.columns):
                if col not in combined_importance:
                    combined_importance[col] = {}
                combined_importance[col][f"{clf_name}_{behavior}"] = float(importances[i])
    
    imp_df = pd.DataFrame(combined_importance).T
    for behavior in behaviors:
        cols = [c for c in imp_df.columns if behavior in c]
        if cols:
            imp_df[f'{behavior}_avg'] = imp_df[cols].mean(axis=1)
    
    avg_cols = [c for c in imp_df.columns if c.endswith('_avg')]
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
    
    # Figure: Feature correlations
    print("Generating feature correlations...")
    all_top = set()
    for behavior in behaviors:
        clf = models[f"Random Forest_{behavior}"]
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[-15:]
        all_top.update([X.columns[i] for i in top_idx])
    
    top_features = list(all_top)
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
    
    # Figure: Classifier comparison
    print("Generating classifier comparison...")
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
    
    # Save feature importance tables
    print("\nSaving feature importance tables...")
    for behavior in behaviors:
        for clf_name in ['Random Forest', 'Gradient Boosting']:
            clf = models[f"{clf_name}_{behavior}"]
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
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
                print(f"  Saved: feature_importance_{clf_name.replace(' ', '_')}_{behavior}.csv")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
