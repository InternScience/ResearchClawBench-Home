"""
Data-Driven De Novo Design of Super-Adhesive Hydrogels
Analysis Script

This script performs comprehensive analysis of hydrogel adhesive strength data
using machine learning approaches including Random Forest and Gaussian Process.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define monomer features
MONOMER_FEATURES = [
    'Nucleophilic-HEA',
    'Hydrophobic-BA', 
    'Acidic-CBEA',
    'Cationic-ATAC',
    'Aromatic-PEA',
    'Amide-AAm'
]

OUTPUT_COLUMNS = ['Glass (kPa)_10s', 'Glass (kPa)_60s', 
                  'Steel (kPa)_10s', 'Steel (kPa)_60s']

def load_and_clean_data(filepath):
    """Load and clean the training data."""
    df = pd.read_excel(filepath)
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ['No.', 'ML', 'NO.']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def analyze_data_distribution(df, output_dir='outputs'):
    """Analyze and visualize the distribution of data."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Monomer composition distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(MONOMER_FEATURES):
        ax = axes[i]
        ax.hist(df[feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'{feature} (mole fraction)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/monomer_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Adhesive strength distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(OUTPUT_COLUMNS):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f} kPa')
        ax.set_xlabel(f'{col} (kPa)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/adhesive_strength_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def correlation_analysis(df, output_dir='outputs'):
    """Perform correlation analysis between features and outputs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select numeric columns for correlation
    numeric_cols = MONOMER_FEATURES + OUTPUT_COLUMNS
    corr_data = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: Monomer Composition vs Adhesive Strength')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance correlation with each output
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, output in enumerate(OUTPUT_COLUMNS):
        ax = axes[i]
        correlations = df[MONOMER_FEATURES + [output]].corr()[output][:-1].sort_values()
        colors = ['red' if x < 0 else 'green' for x in correlations.values]
        correlations.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title(f'Feature Correlations with {output}')
        ax.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_data

def train_random_forest(df, output_col='Glass (kPa)_60s', output_dir='outputs'):
    """Train Random Forest model and evaluate performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    X = df[MONOMER_FEATURES].copy()
    y = df[output_col].copy()
    
    # Remove NaN values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n=== Random Forest Results for {output_col} ===")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.2f} kPa")
    print(f"Testing RMSE: {test_rmse:.2f} kPa")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    print(f"5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': MONOMER_FEATURES,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    # Plot feature importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Feature importance
    ax1 = axes[0]
    importance_df.plot(kind='barh', x='Feature', y='Importance', ax=ax1, 
                       color='steelblue', alpha=0.8, legend=False)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title(f'Random Forest Feature Importance\nfor {output_col}')
    
    # Parity plot
    ax2 = axes[1]
    ax2.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='black', s=50)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel(f'Actual {output_col} (kPa)')
    ax2.set_ylabel(f'Predicted {output_col} (kPa)')
    ax2.set_title(f'Parity Plot: R² = {test_r2:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rf_analysis_{output_col.replace(" ", "_").replace("(", "").replace(")", "")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'model': rf,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'feature_importance': importance_df,
        'test_data': (y_test, y_test_pred)
    }

def analyze_optimization_datasets(filepath1, filepath2, output_dir='outputs'):
    """Analyze the optimization round datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load optimization data
    df_opt1 = pd.read_excel(filepath1)
    df_opt2 = pd.read_excel(filepath2)
    
    print(f"\nOptimization Dataset 1: {df_opt1.shape[0]} samples")
    print(f"Optimization Dataset 2: {df_opt2.shape[0]} samples")
    
    # Analyze ML model types
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (df, title) in enumerate([(df_opt1, 'Dataset 1'), (df_opt2, 'Dataset 2')]):
        ax = axes[idx]
        
        # Get ML types and their performance
        ml_types = df['ML'].dropna().unique()
        performance = []
        
        for ml_type in ml_types:
            mask = df['ML'] == ml_type
            # Fill forward ML type for subsequent rows
            df_filled = df.copy()
            df_filled['ML'] = df_filled['ML'].fillna(method='ffill')
            mask = df_filled['ML'] == ml_type
            perf = df_filled[mask]['Glass (kPa)_max'].values
            performance.append(perf.mean() if len(perf) > 0 else 0)
        
        ax.bar(range(len(ml_types)), performance, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(ml_types)))
        ax.set_xticklabels(ml_types, rotation=45, ha='right')
        ax.set_ylabel('Mean Predicted Glass Adhesion (kPa)')
        ax.set_title(f'ML Model Performance Comparison\n{title}')
        ax.axhline(y=1000, color='red', linestyle='--', label='1 MPa Target')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Composition analysis of top performers
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (df, title) in enumerate([(df_opt1, 'Dataset 1'), (df_opt2, 'Dataset 2')]):
        ax = axes[idx]
        
        # Get top 10 performers
        top10 = df.nlargest(10, 'Glass (kPa)_max')
        
        # Stacked bar chart of compositions
        compositions = top10[MONOMER_FEATURES].values
        x_pos = np.arange(len(top10))
        
        bottom = np.zeros(len(top10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(MONOMER_FEATURES)))
        
        for i, feature in enumerate(MONOMER_FEATURES):
            ax.bar(x_pos, compositions[:, i], bottom=bottom, label=feature, 
                   color=colors[i], edgecolor='white', linewidth=0.5)
            bottom += compositions[:, i]
        
        ax.set_xlabel('Top 10 Formulations (Ranked by Adhesion)')
        ax.set_ylabel('Mole Fraction')
        ax.set_title(f'Compositions of Top Performers\n{title}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'#{i+1}' for i in range(len(top10))])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_performer_compositions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_opt1, df_opt2

def create_summary_statistics(df, output_dir='outputs'):
    """Create summary statistics table."""
    os.makedirs(output_dir, exist_ok=True)
    
    stats = []
    
    for col in MONOMER_FEATURES + OUTPUT_COLUMNS:
        data = df[col].dropna()
        stats.append({
            'Feature': col,
            'Count': len(data),
            'Mean': data.mean(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Median': data.median()
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f'{output_dir}/summary_statistics.csv', index=False)
    
    print("\n=== Summary Statistics ===")
    print(stats_df.to_string(index=False))
    
    return stats_df

def main():
    """Main analysis function."""
    print("="*60)
    print("Hydrogel Adhesive Strength Analysis")
    print("="*60)
    
    # Load main training data
    df = load_and_clean_data('data/184_verified_Original Data_ML_20230926.xlsx')
    print(f"\nLoaded training data: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Create summary statistics
    create_summary_statistics(df)
    
    # Analyze data distribution
    print("\nGenerating distribution plots...")
    analyze_data_distribution(df)
    
    # Correlation analysis
    print("\nPerforming correlation analysis...")
    correlation_analysis(df)
    
    # Train models for each output
    results = {}
    for output_col in OUTPUT_COLUMNS:
        print(f"\nTraining model for {output_col}...")
        results[output_col] = train_random_forest(df, output_col)
    
    # Analyze optimization datasets
    print("\nAnalyzing optimization datasets...")
    analyze_optimization_datasets(
        'data/ML_ei&pred (1&2&3rounds)_20240408.xlsx',
        'data/ML_ei&pred_20240213.xlsx'
    )
    
    # Save model performance summary
    performance_summary = pd.DataFrame([
        {
            'Output': output,
            'Train_R2': results[output]['train_r2'],
            'Test_R2': results[output]['test_r2'],
            'CV_R2_Mean': results[output]['cv_r2_mean'],
            'CV_R2_Std': results[output]['cv_r2_std']
        }
        for output in OUTPUT_COLUMNS
    ])
    performance_summary.to_csv('outputs/model_performance_summary.csv', index=False)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("Results saved to outputs/ directory")
    print("="*60)
    
    return results

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    main()
