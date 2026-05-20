#!/usr/bin/env python3
"""Exploratory Data Analysis for connectomics segment merging task."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUT = 'report/images/'
DATA = 'data/'

def main():
    train = pd.read_csv(DATA + 'train_simulated.csv')
    test = pd.read_csv(DATA + 'test_simulated.csv')
    feat_cols = [str(i) for i in range(20)]
    
    # ============================================================
    # Figure 1: Label distribution
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, df, title in zip(axes, [train, test], ['Training Set', 'Test Set']):
        counts = df['label'].value_counts()
        bars = ax.bar(['Different Neuron\n(0)', 'Same Neuron\n(1)'], 
                       [counts.get(0,0), counts.get(1,0)], 
                       color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.2)
        ax.set_title(f'{title}\nn = {len(df):,}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12)
        for bar, count in zip(bars, [counts.get(0,0), counts.get(1,0)]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                    f'{count:,}\n({count/len(df)*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
    
    fig.suptitle('Class Distribution in Connectomics Segment Pair Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig1_label_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig1_label_distribution.png')
    
    # ============================================================
    # Figure 2: Feature distributions by label (first 8 features)
    # ============================================================
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, col in enumerate(feat_cols):
        ax = axes[i]
        pos = train[train['label']==1][col]
        neg = train[train['label']==0][col].sample(n=len(pos), random_state=42)
        ax.hist(neg, bins=50, alpha=0.6, label='Different (0)', color='#3498db', density=True)
        ax.hist(pos, bins=50, alpha=0.6, label='Same (1)', color='#e74c3c', density=True)
        ax.set_title(f'Feature {col}', fontsize=10)
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')
        ax.tick_params(labelsize=8)
    
    fig.suptitle('Feature Distributions by Label (Training Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig2_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig2_feature_distributions.png')
    
    # ============================================================
    # Figure 3: Feature-label correlations by degradation type
    # ============================================================
    degradations = train['degradation'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for ax, deg, color in zip(axes, degradations, colors):
        sub = train[train['degradation'] == deg]
        corrs = [sub[col].corr(sub['label']) for col in feat_cols]
        ax.bar(range(20), corrs, color=color, edgecolor='black', linewidth=0.8)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_title(f'{deg} (n={len(sub):,})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index', fontsize=11)
        ax.set_ylabel('Correlation with Label', fontsize=11)
        ax.set_xticks(range(20))
        ax.set_xticklabels(range(20), fontsize=8)
    
    fig.suptitle('Feature-Label Correlations by Degradation Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig3_correlations_by_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig3_correlations_by_degradation.png')
    
    # ============================================================
    # Figure 4: Feature box plots for top features by degradation
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, deg in zip(axes, degradations):
        sub = train[train['degradation'] == deg]
        corrs = {col: sub[col].corr(sub['label']) for col in feat_cols}
        top4 = sorted(corrs, key=lambda x: abs(corrs[x]), reverse=True)[:4]
        data_to_plot = []
        labels_pos = []
        for col in top4:
            data_to_plot.append(sub[sub['label']==0][col].values)
            data_to_plot.append(sub[sub['label']==1][col].values)
            labels_pos.extend([f'F{col}\nNeg', f'F{col}\nPos'])
        
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels_pos, 
                        showfliers=False, widths=0.7)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor('#e74c3c' if i % 2 == 1 else '#3498db')
        ax.set_title(f'{deg}', fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.set_ylabel('Feature Value', fontsize=11)
    
    fig.suptitle('Top-4 Predictive Features by Degradation Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig4_top_features_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig4_top_features_boxplot.png')
    
    # ============================================================
    # Figure 5: PCA / t-SNE visualization of the feature space
    # ============================================================
    # Use stratified sample for visualization
    sample_size = 5000
    train_sample = train.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), sample_size//2), random_state=42)
    )
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(train_sample[feat_cols].values)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for label, color, marker, name in [(0, '#3498db', 'o', 'Different'), (1, '#e74c3c', '^', 'Same')]:
        mask = train_sample['label'] == label
        axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=color, marker=marker, alpha=0.5, s=10, label=name)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    axes[0].set_title('PCA of Feature Space', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    
    # t-SNE on smaller sample
    tsne_sample = train_sample.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 1000), random_state=42)
    )
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_result = tsne.fit_transform(tsne_sample[feat_cols].values)
    
    for label, color, marker, name in [(0, '#3498db', 'o', 'Different'), (1, '#e74c3c', '^', 'Same')]:
        mask = tsne_sample['label'] == label
        axes[1].scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       c=color, marker=marker, alpha=0.6, s=15, label=name)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[1].set_title('t-SNE of Feature Space', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    
    fig.suptitle('Dimensionality Reduction Visualization of Connectomics Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT + 'fig5_dimensionality_reduction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] fig5_dimensionality_reduction.png')
    
    print('\nAll EDA figures generated successfully!')

if __name__ == '__main__':
    main()
