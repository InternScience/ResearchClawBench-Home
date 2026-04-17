#!/usr/bin/env python3
"""
Analysis script for hydrogel adhesive strength optimization.
This script processes monomer composition data and adhesive strength measurements
to understand the relationship between protein sequence features and hydrogel performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Paths
DATA_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Life_000_20260416_180742/data'
OUTPUT_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Life_000_20260416_180742/outputs'
IMAGES_DIR = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Life_000_20260416_180742/report/images'

def load_data():
    """Load all datasets and return processed dataframes."""
    
    # Primary verified dataset (184 samples)
    df_primary = pd.read_excel(f'{DATA_DIR}/184_verified_Original Data_ML_20230926.xlsx')
    
    # Final optimization dataset (all rounds)
    df_opt = pd.read_excel(f'{DATA_DIR}/ML_ei&pred (1&2&3rounds)_20240408.xlsx')
    
    # Convert numeric columns
    df_opt['Nucleophilic-HEA'] = pd.to_numeric(df_opt['Nucleophilic-HEA'], errors='coerce')
    df_opt['Glass (kPa)_max'] = pd.to_numeric(df_opt['Glass (kPa)_max'], errors='coerce')
    
    # Batch datasets
    df_batch1 = pd.read_excel(f'{DATA_DIR}/Original Data_ML_20220829.xlsx')
    df_batch2 = pd.read_excel(f'{DATA_DIR}/Original Data_ML_20221031.xlsx')
    df_batch3 = pd.read_excel(f'{DATA_DIR}/Original Data_ML_20221129.xlsx')
    
    return df_primary, df_opt, df_batch1, df_batch2, df_batch3

def prepare_features(df, target_col='Glass (kPa)_max'):
    """Prepare feature matrix and target vector."""
    feature_cols = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                    'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
    
    # Check if target column exists
    if target_col not in df.columns:
        # Try alternative column names
        if 'Glass (kPa)' in df.columns:
            target_col = 'Glass (kPa)'
        elif 'Steel (kPa)_max' in df.columns:
            target_col = 'Steel (kPa)_max'
        else:
            raise ValueError(f"Target column {target_col} not found")
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]
    
    # Remove rows where target is NaN
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y, feature_cols

def train_models(X, y):
    """Train and evaluate multiple ML models."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        if 'RandomForest' in name or 'GradientBoosting' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'y_test': y_test.values,
            'y_pred': y_pred
        }
    
    # Cross-validation scores
    cv_results = {}
    for name, model in models.items():
        if 'RandomForest' in name or 'GradientBoosting' in name:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_results[name] = cv_scores
    
    return results, cv_results, scaler

def analyze_composition_performance(df_opt):
    """Analyze relationship between monomer composition and adhesive strength."""
    
    feature_cols = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                    'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
    
    # Convert composition columns to numeric
    for col in feature_cols:
        df_opt[col] = pd.to_numeric(df_opt[col], errors='coerce')
    
    df_opt['Glass (kPa)_max'] = pd.to_numeric(df_opt['Glass (kPa)_max'], errors='coerce')
    
    # Calculate correlations
    corr_data = df_opt[feature_cols + ['Glass (kPa)_max']].dropna()
    correlation_matrix = corr_data.corr()
    
    # Find high-performing formulations (>1 MPa = 1000 kPa)
    high_perf = df_opt[df_opt['Glass (kPa)_max'] > 1000].copy()
    
    # Calculate mean composition for different performance tiers
    tiers = {
        'Low (<100 kPa)': df_opt[df_opt['Glass (kPa)_max'] < 100],
        'Medium (100-200 kPa)': df_opt[(df_opt['Glass (kPa)_max'] >= 100) & (df_opt['Glass (kPa)_max'] < 200)],
        'High (>200 kPa)': df_opt[df_opt['Glass (kPa)_max'] >= 200]
    }
    
    tier_compositions = {}
    for tier_name, tier_df in tiers.items():
        if len(tier_df) > 0:
            tier_compositions[tier_name] = tier_df[feature_cols].mean()
    
    return correlation_matrix, high_perf, tier_compositions

def generate_figures(df_primary, df_opt, results, correlation_matrix, tier_compositions):
    """Generate all figures for the report."""
    
    figures = {}
    
    # Figure 1: Data Overview - Distribution of adhesive strengths
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Primary dataset distribution
    if 'Glass (kPa)_max' in df_primary.columns:
        target_primary = df_primary['Glass (kPa)_max']
    elif 'Glass (kPa)' in df_primary.columns:
        target_primary = df_primary['Glass (kPa)']
    else:
        target_primary = pd.Series([0])
    
    axes[0].hist(target_primary.dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Adhesive Strength (kPa)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Initial Training Data (184 samples)')
    axes[0].axvline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa target')
    axes[0].legend()
    
    # Optimization dataset distribution
    opt_target = pd.to_numeric(df_opt['Glass (kPa)_max'], errors='coerce')
    axes[1].hist(opt_target.dropna(), bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Adhesive Strength (kPa)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Optimization Rounds Dataset (120 samples)')
    axes[1].axvline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa target')
    axes[1].legend()
    
    plt.tight_layout()
    fig1_path = f'{IMAGES_DIR}/figure1_data_overview.png'
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    figures['data_overview'] = fig1_path
    
    # Figure 2: Correlation Heatmap
    fig2, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', ax=ax, square=True, linewidths=0.5)
    ax.set_title('Monomer Composition vs Adhesive Strength Correlation')
    plt.tight_layout()
    fig2_path = f'{IMAGES_DIR}/figure2_correlation_heatmap.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    figures['correlation'] = fig2_path
    
    # Figure 3: Composition by Performance Tier
    fig3, ax = plt.subplots(figsize=(12, 6))
    tier_df = pd.DataFrame(tier_compositions)
    tier_df.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_xlabel('Monomer Type')
    ax.set_ylabel('Mean Composition Fraction')
    ax.set_title('Average Monomer Composition by Performance Tier')
    ax.legend(title='Performance Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fig3_path = f'{IMAGES_DIR}/figure3_composition_by_tier.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    figures['composition_tier'] = fig3_path
    
    # Figure 4: Model Performance Comparison
    fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² comparison
    model_names = list(results.keys())
    r2_scores = [results[m]['r2'] for m in model_names]
    colors = ['steelblue', 'coral']
    
    axes[0].bar(model_names, r2_scores, color=colors, edgecolor='black')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Model Performance Comparison (Test Set)')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12)
    
    # Parity plot for best model
    best_model = model_names[np.argmax(r2_scores)]
    axes[1].scatter(results[best_model]['y_test'], results[best_model]['y_pred'], 
                   alpha=0.6, s=50, color='steelblue', edgecolor='black')
    min_val = min(results[best_model]['y_test'].min(), results[best_model]['y_pred'].min())
    max_val = max(results[best_model]['y_test'].max(), results[best_model]['y_pred'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('Experimental Adhesive Strength (kPa)')
    axes[1].set_ylabel('Predicted Adhesive Strength (kPa)')
    axes[1].set_title(f'{best_model} - Parity Plot (R²={r2_scores[np.argmax(r2_scores)]:.3f})')
    axes[1].legend()
    axes[1].set_aspect('equal', 'box')
    
    plt.tight_layout()
    fig4_path = f'{IMAGES_DIR}/figure4_model_performance.png'
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    figures['model_performance'] = fig4_path
    
    # Figure 5: Feature Importance
    fig5, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    feature_cols = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
                    'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
    
    # Random Forest feature importance
    if 'RandomForest' in results:
        rf_importance = results['RandomForest']['model'].feature_importances_
        axes[0].barh(feature_cols, rf_importance, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Feature Importance')
        axes[0].set_title('Random Forest Feature Importance')
        axes[0].invert_yaxis()
    
    # Gradient Boosting feature importance
    if 'GradientBoosting' in results:
        gb_importance = results['GradientBoosting']['model'].feature_importances_
        axes[1].barh(feature_cols, gb_importance, color='coral', edgecolor='black')
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('Gradient Boosting Feature Importance')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    fig5_path = f'{IMAGES_DIR}/figure5_feature_importance.png'
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close(fig5)
    figures['feature_importance'] = fig5_path
    
    # Figure 6: Optimization Trajectory
    fig6, ax = plt.subplots(figsize=(12, 6))
    
    # Group by ML method and calculate statistics
    df_opt_clean = df_opt.copy()
    df_opt_clean['Glass (kPa)_max'] = pd.to_numeric(df_opt_clean['Glass (kPa)_max'], errors='coerce')
    
    # Fill forward ML labels
    df_opt_clean['ML'] = df_opt_clean['ML'].fillna(method='ffill')
    
    # Calculate mean and std for each ML method
    ml_stats = df_opt_clean.groupby('ML')['Glass (kPa)_max'].agg(['mean', 'std', 'count']).reset_index()
    ml_stats = ml_stats[ml_stats['count'] >= 3]  # Only methods with enough samples
    
    # Sort by mean performance
    ml_stats = ml_stats.sort_values('mean', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(ml_stats)))
    
    bars = ax.barh(ml_stats['ML'], ml_stats['mean'], xerr=ml_stats['std'], 
                   color=colors, edgecolor='black', capsize=3)
    ax.set_xlabel('Mean Adhesive Strength (kPa)')
    ax.set_title('Optimization Method Comparison')
    ax.axvline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa target')
    ax.legend()
    
    plt.tight_layout()
    fig6_path = f'{IMAGES_DIR}/figure6_optimization_trajectory.png'
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close(fig6)
    figures['optimization'] = fig6_path
    
    # Figure 7: High Performers Analysis
    fig7, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Find samples with adhesive strength > 200 kPa (highest achieved)
    high_perf_threshold = 200
    high_perf = df_opt_clean[df_opt_clean['Glass (kPa)_max'] > high_perf_threshold].copy()
    
    if len(high_perf) > 0:
        # Radar chart for top 5 performers
        top5 = high_perf.nlargest(5, 'Glass (kPa)_max')
        
        angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
        angles += angles[:1]
        
        ax_radar = axes[0]
        ax_radar = fig6.add_subplot(121, projection='polar') if False else axes[0]
        
        colors_top5 = plt.cm.plasma(np.linspace(0, 1, 5))
        for i, (_, row) in enumerate(top5.iterrows()):
            values = row[feature_cols].tolist()
            values += values[:1]
            axes[0].plot(angles, values, 'o-', linewidth=2, label=f"#{row['NO.']}", color=colors_top5[i])
            axes[0].fill(angles, values, alpha=0.1, color=colors_top5[i])
        
        axes[0].set_xticks(angles[:-1])
        axes[0].set_xticklabels(feature_cols, fontsize=9)
        axes[0].set_ylim(0, 0.7)
        axes[0].set_title(f'Top 5 High Performers (>{high_perf_threshold} kPa)', pad=20)
        axes[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        
        # Box plot of compositions
        high_perf_melted = high_perf[feature_cols].melt(var_name='Monomer', value_name='Composition')
        sns.boxplot(data=high_perf_melted, x='Monomer', y='Composition', ax=axes[1], palette='viridis')
        axes[1].set_title(f'Composition Distribution (High Performers >{high_perf_threshold} kPa)')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig7_path = f'{IMAGES_DIR}/figure7_high_performers.png'
    plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
    plt.close(fig7)
    figures['high_performers'] = fig7_path
    
    return figures

def save_results(results, cv_results, correlation_matrix, tier_compositions):
    """Save intermediate results to outputs directory."""
    
    # Model performance summary
    perf_summary = {}
    for name, res in results.items():
        perf_summary[name] = {
            'r2_test': float(res['r2']),
            'rmse_test': float(res['rmse']),
            'mae_test': float(res['mae'])
        }
    
    # CV results
    cv_summary = {}
    for name, scores in cv_results.items():
        cv_summary[name] = {
            'cv_r2_mean': float(scores.mean()),
            'cv_r2_std': float(scores.std())
        }
    
    # Correlation matrix
    corr_dict = correlation_matrix.to_dict()
    
    # Save as JSON-compatible format
    import json
    output_data = {
        'model_performance': perf_summary,
        'cross_validation': cv_summary,
        'tier_compositions': {k: v.to_dict() for k, v in tier_compositions.items()}
    }
    
    with open(f'{OUTPUT_DIR}/model_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save correlation matrix
    correlation_matrix.to_csv(f'{OUTPUT_DIR}/correlation_matrix.csv')
    
    return output_data

def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("Hydrogel Adhesive Strength Analysis Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading datasets...")
    df_primary, df_opt, df_batch1, df_batch2, df_batch3 = load_data()
    print(f"  - Primary dataset: {df_primary.shape}")
    print(f"  - Optimization dataset: {df_opt.shape}")
    print(f"  - Batch 1: {df_batch1.shape}")
    print(f"  - Batch 2: {df_batch2.shape}")
    print(f"  - Batch 3: {df_batch3.shape}")
    
    # Prepare features
    print("\n[2/5] Preparing features...")
    X, y, feature_cols = prepare_features(df_opt)
    print(f"  - Features: {feature_cols}")
    print(f"  - Samples after cleaning: {len(X)}")
    
    # Train models
    print("\n[3/5] Training ML models...")
    results, cv_results, scaler = train_models(X, y)
    for name, res in results.items():
        print(f"  - {name}: R²={res['r2']:.3f}, RMSE={res['rmse']:.1f} kPa")
    
    # Analyze composition-performance relationships
    print("\n[4/5] Analyzing composition-performance relationships...")
    correlation_matrix, high_perf, tier_compositions = analyze_composition_performance(df_opt)
    print(f"  - High performers (>{1000} kPa): {len(high_perf)} samples")
    print(f"  - Performance tiers analyzed: {list(tier_compositions.keys())}")
    
    # Generate figures
    print("\n[5/5] Generating figures...")
    figures = generate_figures(df_primary, df_opt, results, correlation_matrix, tier_compositions)
    for name, path in figures.items():
        print(f"  - Saved: {path}")
    
    # Save results
    print("\nSaving intermediate results...")
    output_data = save_results(results, cv_results, correlation_matrix, tier_compositions)
    print(f"  - Saved: {OUTPUT_DIR}/model_results.json")
    print(f"  - Saved: {OUTPUT_DIR}/correlation_matrix.csv")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return results, figures, output_data

if __name__ == '__main__':
    results, figures, output_data = main()
