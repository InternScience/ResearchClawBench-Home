"""
Comprehensive Analysis for Hydrogel Adhesive Strength Research
Generates all figures and outputs for the research report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Constants
MONOMERS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 
            'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET = 'Glass (kPa)_10s'

def load_data():
    """Load and clean the primary dataset."""
    df = pd.read_excel('data/184_verified_Original Data_ML_20230926.xlsx')
    
    # Convert to numeric
    for col in df.columns:
        if col not in ['No.', 'Tanδ', 'Log_Slope']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def create_figure_1_data_overview(df):
    """Figure 1: Data Overview - Monomer Composition Distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, monomer in enumerate(MONOMERS):
        ax = axes[i]
        data = df[monomer].dropna()
        ax.hist(data, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {data.mean():.3f}')
        ax.set_xlabel(f'{monomer} (mole fraction)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{monomer}', fontsize=12, fontweight='bold')
        ax.legend()
    
    plt.suptitle('Distribution of Monomer Compositions in Training Dataset (n=184)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('report/images/fig1_monomer_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/fig1_monomer_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig1_monomer_distribution.png")

def create_figure_2_adhesive_strength(df):
    """Figure 2: Adhesive Strength Distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Glass adhesion
    ax1 = axes[0]
    data = df[TARGET].dropna()
    ax1.hist(data, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {data.mean():.1f} kPa')
    ax1.axvline(data.median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {data.median():.1f} kPa')
    ax1.axvline(1000, color='orange', linestyle='-', linewidth=2, 
                label='Target: 1000 kPa (1 MPa)')
    ax1.set_xlabel('Glass Adhesion Strength (kPa)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Glass Adhesion Strength\n(n=184)', fontsize=13, fontweight='bold')
    ax1.legend()
    
    # Steel adhesion
    ax2 = axes[1]
    steel_col = 'Steel (kPa)_10s'
    data_steel = df[steel_col].dropna()
    ax2.hist(data_steel, bins=15, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(data_steel.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {data_steel.mean():.1f} kPa')
    ax2.set_xlabel('Steel Adhesion Strength (kPa)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Steel Adhesion Strength\n(n=28)', fontsize=13, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('report/images/fig2_adhesive_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/fig2_adhesive_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig2_adhesive_distribution.png")

def create_figure_3_correlation_heatmap(df):
    """Figure 3: Correlation Analysis"""
    # Prepare correlation data
    corr_cols = MONOMERS + [TARGET, 'Steel (kPa)_10s', 'Q', 'Modulus (kPa)', "G''", 'XlogP3']
    corr_data = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, 
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: Monomer Composition vs Properties', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report/images/fig3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/fig3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig3_correlation_heatmap.png")

def create_figure_4_feature_importance(df):
    """Figure 4: Feature Importance from Random Forest"""
    # Prepare data
    X = df[MONOMERS].copy()
    y = df[TARGET].copy()
    
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Feature importance
    ax1 = axes[0]
    importance = pd.DataFrame({
        'Feature': MONOMERS,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(MONOMERS)))
    ax1.barh(importance['Feature'], importance['Importance'], color=colors, edgecolor='black')
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.set_title('Random Forest Feature Importance\nfor Glass Adhesion Prediction', 
                  fontsize=13, fontweight='bold')
    
    # Parity plot
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred, alpha=0.7, edgecolor='black', s=80, color='steelblue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Glass Adhesion (kPa)', fontsize=12)
    ax2.set_ylabel('Predicted Glass Adhesion (kPa)', fontsize=12)
    ax2.set_title(f'Parity Plot: R² = {r2:.3f}, RMSE = {rmse:.1f} kPa', 
                  fontsize=13, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('report/images/fig4_rf_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/fig4_rf_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig4_rf_analysis.png")
    
    return rf, r2, rmse

def create_figure_5_ternary_compositions(df):
    """Figure 5: Ternary composition plots for top performers"""
    # Get top 20 formulations by adhesion strength
    top_n = 20
    top_df = df.nlargest(top_n, TARGET)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Group monomers for ternary plot
    # Hydrophilic: Nucleophilic-HEA, Amide-AAm
    # Hydrophobic: Hydrophobic-BA, Aromatic-PEA  
    # Charged: Acidic-CBEA, Cationic-ATAC
    
    df_plot = df.copy()
    df_plot['Hydrophilic'] = df['Nucleophilic-HEA'] + df['Amide-AAm']
    df_plot['Hydrophobic'] = df['Hydrophobic-BA'] + df['Aromatic-PEA']
    df_plot['Charged'] = df['Acidic-CBEA'] + df['Cationic-ATAC']
    
    # All data
    ax1 = axes[0]
    scatter = ax1.scatter(df_plot['Hydrophilic'], df_plot['Hydrophobic'], 
                         c=df_plot[TARGET], cmap='viridis', s=50, alpha=0.6, edgecolor='black')
    plt.colorbar(scatter, ax=ax1, label='Glass Adhesion (kPa)')
    ax1.set_xlabel('Hydrophilic Fraction (HEA + AAm)', fontsize=12)
    ax1.set_ylabel('Hydrophobic Fraction (BA + PEA)', fontsize=12)
    ax1.set_title('Composition Space: All Formulations\n(n=184)', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Top performers
    ax2 = axes[1]
    top_plot = df_plot.nlargest(top_n, TARGET)
    scatter2 = ax2.scatter(top_plot['Hydrophilic'], top_plot['Hydrophobic'], 
                          c=top_plot[TARGET], cmap='plasma', s=100, alpha=0.8, edgecolor='black')
    plt.colorbar(scatter2, ax=ax2, label='Glass Adhesion (kPa)')
    ax2.set_xlabel('Hydrophilic Fraction (HEA + AAm)', fontsize=12)
    ax2.set_ylabel('Hydrophobic Fraction (BA + PEA)', fontsize=12)
    ax2.set_title(f'Composition Space: Top {top_n} Performers', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('report/images/fig5_composition_space.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/fig5_composition_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig5_composition_space.png")

def create_figure_6_optimization_results():
    """Figure 6: ML-Guided Optimization Results"""
    # Load optimization data
    df_opt = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx')
    
    # Forward fill ML model types
    df_opt['ML'] = df_opt['ML'].fillna(method='ffill')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ML model comparison
    ax1 = axes[0]
    ml_performance = df_opt.groupby('ML')['Glass (kPa)_max'].agg(['mean', 'std', 'count'])
    ml_performance = ml_performance.sort_values('mean', ascending=True)
    
    y_pos = np.arange(len(ml_performance))
    ax1.barh(y_pos, ml_performance['mean'], xerr=ml_performance['std'], 
             color='steelblue', alpha=0.7, edgecolor='black', capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ml_performance.index)
    ax1.set_xlabel('Mean Predicted Glass Adhesion (kPa)', fontsize=12)
    ax1.set_title('ML Model Performance Comparison\n(Optimization Dataset)', fontsize=13, fontweight='bold')
    ax1.axvline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa Target')
    ax1.legend()
    
    # Distribution of predictions
    ax2 = axes[1]
    for ml_type in df_opt['ML'].unique():
        data = df_opt[df_opt['ML'] == ml_type]['Glass (kPa)_max']
        ax2.hist(data, bins=15, alpha=0.5, label=ml_type, edgecolor='black')
    ax2.set_xlabel('Predicted Glass Adhesion (kPa)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Predicted Adhesion\nby ML Model', fontsize=13, fontweight='bold')
    ax2.axvline(1000, color='red', linestyle='--', linewidth=2, label='1 MPa Target')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('report/images/fig6_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/fig6_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig6_optimization_results.png")

def generate_summary_stats(df):
    """Generate summary statistics table"""
    stats = []
    
    for col in MONOMERS + [TARGET, 'Steel (kPa)_10s']:
        data = df[col].dropna()
        stats.append({
            'Feature': col,
            'N': len(data),
            'Mean': f"{data.mean():.3f}",
            'Std': f"{data.std():.3f}",
            'Min': f"{data.min():.3f}",
            'Max': f"{data.max():.3f}",
            'Median': f"{data.median():.3f}"
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('outputs/summary_statistics.csv', index=False)
    print("\n=== Summary Statistics ===")
    print(stats_df.to_string(index=False))
    return stats_df

def main():
    print("="*60)
    print("Hydrogel Adhesive Strength Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nLoaded training data: {df.shape[0]} samples")
    
    # Generate summary statistics
    generate_summary_stats(df)
    
    # Create all figures
    print("\nGenerating figures...")
    create_figure_1_data_overview(df)
    create_figure_2_adhesive_strength(df)
    create_figure_3_correlation_heatmap(df)
    rf, r2, rmse = create_figure_4_feature_importance(df)
    create_figure_5_ternary_compositions(df)
    create_figure_6_optimization_results()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Random Forest Test R²: {r2:.4f}")
    print(f"Random Forest Test RMSE: {rmse:.2f} kPa")
    print("="*60)

if __name__ == "__main__":
    main()
