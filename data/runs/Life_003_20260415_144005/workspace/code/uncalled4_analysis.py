#!/usr/bin/env python3
"""
Uncalled4 Analysis Pipeline

This script performs comprehensive analysis of nanopore sequencing data
comparing Uncalled4 performance with other tools (f5c, Nanopolish, Tombo)
in terms of speed, accuracy, and modification detection capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
DATA_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Life_003_20260415_144005/data"
OUTPUT_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Life_003_20260415_144005/outputs"
REPORT_IMG_DIR = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Life_003_20260415_144005/report/images"

def load_pore_models():
    """Load all pore model data files."""
    print("Loading pore model data...")
    
    models = {
        'dna_r9.4.1_6mer': pd.read_csv(f"{DATA_DIR}/dna_r9.4.1_400bps_6mer_uncalled4.csv"),
        'dna_r10.4.1_9mer': pd.read_csv(f"{DATA_DIR}/dna_r10.4.1_400bps_9mer_uncalled4.csv"),
        'rna_r9.4.1_5mer': pd.read_csv(f"{DATA_DIR}/rna_r9.4.1_70bps_5mer_uncalled4.csv"),
        'rna004_9mer': pd.read_csv(f"{DATA_DIR}/rna004_130bps_9mer_uncalled4.csv")
    }
    
    return models

def load_performance_data():
    """Load performance benchmark data."""
    print("Loading performance data...")
    return pd.read_csv(f"{DATA_DIR}/performance_summary.csv")

def load_m6a_data():
    """Load m6A modification prediction data."""
    print("Loading m6A prediction data...")
    
    uncalled4_preds = pd.read_csv(f"{DATA_DIR}/m6a_predictions_uncalled4.csv")
    nanopolish_preds = pd.read_csv(f"{DATA_DIR}/m6a_predictions_nanopolish.csv")
    labels = pd.read_csv(f"{DATA_DIR}/m6a_labels.csv")
    
    return uncalled4_preds, nanopolish_preds, labels

def analyze_pore_models(models):
    """Analyze pore model characteristics."""
    print("\n=== Pore Model Analysis ===")
    
    results = {}
    
    for name, df in models.items():
        print(f"\n{name}:")
        print(f"  - Number of k-mers: {len(df)}")
        print(f"  - Mean current: {df['current_mean'].mean():.3f} ± {df['current_mean'].std():.3f}")
        print(f"  - Mean std: {df['current_std'].mean():.3f} ± {df['current_std'].std():.3f}")
        print(f"  - Mean dwell time: {df['dwell_time'].mean():.1f} ± {df['dwell_time'].std():.1f}")
        print(f"  - Current range: [{df['current_mean'].min():.3f}, {df['current_mean'].max():.3f}]")
        
        results[name] = {
            'n_kmers': len(df),
            'mean_current': df['current_mean'].mean(),
            'std_current': df['current_mean'].std(),
            'mean_std': df['current_std'].mean(),
            'mean_dwell': df['dwell_time'].mean(),
            'current_range': (df['current_mean'].min(), df['current_mean'].max())
        }
    
    return results

def plot_pore_model_comparison(models, save_path):
    """Create comprehensive pore model comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Current mean distribution
    ax = axes[0, 0]
    for i, (name, df) in enumerate(models.items()):
        label = name.replace('_', ' ').upper()
        ax.hist(df['current_mean'], bins=50, alpha=0.6, label=label, color=colors[i], density=True)
    ax.set_xlabel('Normalized Current Mean', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Current Mean Distribution by Pore Model', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # Current std distribution
    ax = axes[0, 1]
    for i, (name, df) in enumerate(models.items()):
        label = name.replace('_', ' ').upper()
        ax.hist(df['current_std'], bins=50, alpha=0.6, label=label, color=colors[i], density=True)
    ax.set_xlabel('Current Standard Deviation', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Current Std Distribution by Pore Model', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # Dwell time distribution
    ax = axes[1, 0]
    for i, (name, df) in enumerate(models.items()):
        label = name.replace('_', ' ').upper()
        ax.hist(df['dwell_time'], bins=50, alpha=0.6, label=label, color=colors[i], density=True, range=(0, 50))
    ax.set_xlabel('Dwell Time', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Dwell Time Distribution by Pore Model', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    # Mean current vs std scatter
    ax = axes[1, 1]
    for i, (name, df) in enumerate(models.items()):
        label = name.replace('_', ' ').upper()
        # Sample for visualization
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        ax.scatter(sample_df['current_mean'], sample_df['current_std'], 
                  alpha=0.3, label=label, color=colors[i], s=5)
    ax.set_xlabel('Current Mean', fontsize=11)
    ax.set_ylabel('Current Std', fontsize=11)
    ax.set_title('Current Mean vs Standard Deviation', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pore model comparison to {save_path}")

def analyze_performance(perf_df, save_path):
    """Analyze and plot performance benchmarks."""
    print("\n=== Performance Benchmark Analysis ===")
    
    # Print summary statistics
    for chem in perf_df['Chemistry'].unique():
        chem_data = perf_df[perf_df['Chemistry'] == chem]
        print(f"\n{chem}:")
        for _, row in chem_data.iterrows():
            tool = row['Tool']
            time_val = row['Time_min']
            size_val = row['FileSize_MB']
            time_str = f"{time_val:.1f} min" if pd.notna(time_val) else "N/A"
            size_str = f"{size_val:.1f} MB" if pd.notna(size_val) else "N/A"
            print(f"  {tool}: Time={time_str}, Size={size_str}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter out NaN values for plotting
    perf_clean = perf_df.dropna(subset=['Time_min', 'FileSize_MB'])
    
    # Time comparison
    ax = axes[0]
    chemistries = perf_clean['Chemistry'].unique()
    tools = perf_clean['Tool'].unique()
    x = np.arange(len(chemistries))
    width = 0.2
    
    for i, tool in enumerate(tools):
        tool_data = perf_clean[perf_clean['Tool'] == tool]
        times = [tool_data[tool_data['Chemistry'] == chem]['Time_min'].values[0] 
                if len(tool_data[tool_data['Chemistry'] == chem]) > 0 else 0
                for chem in chemistries]
        ax.bar(x + i*width, times, width, label=tool)
    
    ax.set_xlabel('Chemistry', fontsize=11)
    ax.set_ylabel('Alignment Time (minutes)', fontsize=11)
    ax.set_title('Alignment Time Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(chemistries, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    
    # File size comparison
    ax = axes[1]
    for i, tool in enumerate(tools):
        tool_data = perf_clean[perf_clean['Tool'] == tool]
        sizes = [tool_data[tool_data['Chemistry'] == chem]['FileSize_MB'].values[0] 
                if len(tool_data[tool_data['Chemistry'] == chem]) > 0 else 0
                for chem in chemistries]
        ax.bar(x + i*width, sizes, width, label=tool)
    
    ax.set_xlabel('Chemistry', fontsize=11)
    ax.set_ylabel('File Size (MB)', fontsize=11)
    ax.set_title('Output File Size Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(chemistries, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance comparison to {save_path}")
    
    return perf_clean

def analyze_m6a_predictions(uncalled4_preds, nanopolish_preds, labels, save_path):
    """Analyze m6A modification prediction performance."""
    print("\n=== m6A Prediction Analysis ===")
    
    # Merge predictions with labels
    merged = pd.merge(labels, uncalled4_preds, on='site_id', suffixes=('', '_uncalled4'))
    merged = pd.merge(merged, nanopolish_preds, on='site_id', suffixes=('', '_nanopolish'))
    
    y_true = merged['label'].values
    y_uncalled4 = merged['probability'].values
    y_nanopolish = merged['probability_nanopolish'].values
    
    print(f"Total sites analyzed: {len(y_true)}")
    print(f"Positive sites (m6A): {y_true.sum()} ({100*y_true.mean():.1f}%)")
    print(f"Negative sites: {(~y_true.astype(bool)).sum()} ({100*(1-y_true.mean()):.1f}%)")
    
    # Calculate ROC curves
    fpr_uc, tpr_uc, _ = roc_curve(y_true, y_uncalled4)
    fpr_np, tpr_np, _ = roc_curve(y_true, y_nanopolish)
    
    auc_uc = auc(fpr_uc, tpr_uc)
    auc_np = auc(fpr_np, tpr_np)
    
    print(f"\nUncalled4 AUC-ROC: {auc_uc:.4f}")
    print(f"Nanopolish AUC-ROC: {auc_np:.4f}")
    
    # Calculate PR curves
    precision_uc, recall_uc, _ = precision_recall_curve(y_true, y_uncalled4)
    precision_np, recall_np, _ = precision_recall_curve(y_true, y_nanopolish)
    
    ap_uc = average_precision_score(y_true, y_uncalled4)
    ap_np = average_precision_score(y_true, y_nanopolish)
    
    print(f"Uncalled4 Average Precision: {ap_uc:.4f}")
    print(f"Nanopolish Average Precision: {ap_np:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ROC curves
    ax = axes[0, 0]
    ax.plot(fpr_uc, tpr_uc, label=f'Uncalled4 (AUC = {auc_uc:.3f})', linewidth=2, color='#1f77b4')
    ax.plot(fpr_np, tpr_np, label=f'Nanopolish (AUC = {auc_np:.3f})', linewidth=2, color='#ff7f0e')
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve - m6A Detection', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # PR curves
    ax = axes[0, 1]
    ax.plot(recall_uc, precision_uc, label=f'Uncalled4 (AP = {ap_uc:.3f})', linewidth=2, color='#1f77b4')
    ax.plot(recall_np, precision_np, label=f'Nanopolish (AP = {ap_np:.3f})', linewidth=2, color='#ff7f0e')
    ax.axhline(y=y_true.mean(), color='k', linestyle='--', label=f'Baseline ({y_true.mean():.3f})', linewidth=1)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve - m6A Detection', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Prediction distributions
    ax = axes[1, 0]
    ax.hist(y_uncalled4[y_true==0], bins=50, alpha=0.6, label='Uncalled4 - Negative', color='#1f77b4', density=True)
    ax.hist(y_uncalled4[y_true==1], bins=50, alpha=0.6, label='Uncalled4 - Positive', color='#d62728', density=True)
    ax.set_xlabel('Prediction Probability', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Uncalled4 Prediction Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    
    ax = axes[1, 1]
    ax.hist(y_nanopolish[y_true==0], bins=50, alpha=0.6, label='Nanopolish - Negative', color='#ff7f0e', density=True)
    ax.hist(y_nanopolish[y_true==1], bins=50, alpha=0.6, label='Nanopolish - Positive', color='#d62728', density=True)
    ax.set_xlabel('Prediction Probability', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Nanopolish Prediction Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved m6A prediction analysis to {save_path}")
    
    return {
        'auc_uncalled4': auc_uc,
        'auc_nanopolish': auc_np,
        'ap_uncalled4': ap_uc,
        'ap_nanopolish': ap_np,
        'n_sites': len(y_true),
        'n_positive': int(y_true.sum())
    }

def plot_speedup_analysis(perf_df, save_path):
    """Create speedup analysis visualization."""
    print("\n=== Speedup Analysis ===")
    
    # Calculate speedups relative to Uncalled4
    chemistries = perf_df['Chemistry'].unique()
    tools = [t for t in perf_df['Tool'].unique() if t != 'Uncalled4']
    
    speedup_data = []
    
    for chem in chemistries:
        chem_data = perf_df[perf_df['Chemistry'] == chem]
        uncalled4_time = chem_data[chem_data['Tool'] == 'Uncalled4']['Time_min'].values
        
        if len(uncalled4_time) == 0 or pd.isna(uncalled4_time[0]):
            continue
        
        uncalled4_time = uncalled4_time[0]
        
        for tool in tools:
            tool_time = chem_data[chem_data['Tool'] == tool]['Time_min'].values
            if len(tool_time) > 0 and pd.notna(tool_time[0]):
                speedup = tool_time[0] / uncalled4_time
                speedup_data.append({
                    'Chemistry': chem,
                    'Tool': tool,
                    'Speedup': speedup,
                    'Uncalled4_Time': uncalled4_time,
                    'Tool_Time': tool_time[0]
                })
                print(f"{chem} - {tool}: {speedup:.1f}x slower than Uncalled4")
    
    speedup_df = pd.DataFrame(speedup_data)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    chem_short = [c.replace('DNA ', '').replace('RNA', 'RNA') for c in speedup_df['Chemistry'].unique()]
    tools_unique = speedup_df['Tool'].unique()
    x = np.arange(len(chem_short))
    width = 0.25
    
    colors = {'f5c': '#ff7f0e', 'Nanopolish': '#2ca02c', 'Tombo': '#d62728'}
    
    for i, tool in enumerate(tools_unique):
        tool_speedups = speedup_df[speedup_df['Tool'] == tool]['Speedup'].values
        ax.bar(x + i*width, tool_speedups, width, label=tool, color=colors.get(tool, '#9467bd'))
    
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Uncalled4 baseline')
    ax.set_xlabel('Chemistry', fontsize=12)
    ax.set_ylabel('Relative Time (vs Uncalled4)', fontsize=12)
    ax.set_title('Speedup Factor: Uncalled4 vs Other Tools', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(chem_short)
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved speedup analysis to {save_path}")
    
    return speedup_df

def plot_current_by_base_composition(models, save_path):
    """Analyze current signal by nucleotide composition."""
    print("\n=== Base Composition Analysis ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (name, df) in enumerate(models.items()):
        ax = axes[idx]
        
        # Count GC content in each k-mer
        df_copy = df.copy()
        df_copy['gc_count'] = df_copy['kmer'].apply(lambda x: x.count('G') + x.count('C'))
        df_copy['gc_content'] = df_copy['gc_count'] / df_copy['kmer'].str.len()
        
        # Group by GC content and calculate mean current
        gc_groups = df_copy.groupby('gc_count')['current_mean'].agg(['mean', 'std', 'count']).reset_index()
        
        ax.errorbar(gc_groups['gc_count'], gc_groups['mean'], yerr=gc_groups['std'],
                   marker='o', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel('GC Count in k-mer', fontsize=11)
        ax.set_ylabel('Mean Current', fontsize=11)
        ax.set_title(f'{name.replace("_", " ").upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = df_copy['gc_content'].corr(df_copy['current_mean'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved base composition analysis to {save_path}")

def generate_summary_table(perf_df, m6a_results, pore_stats, save_path):
    """Generate summary statistics table."""
    print("\n=== Generating Summary Table ===")
    
    summary = {
        'Metric': [],
        'Value': []
    }
    
    # Performance metrics
    summary['Metric'].append('Uncalled4 AUC-ROC (m6A detection)')
    summary['Value'].append(f"{m6a_results['auc_uncalled4']:.4f}")
    
    summary['Metric'].append('Nanopolish AUC-ROC (m6A detection)')
    summary['Value'].append(f"{m6a_results['auc_nanopolish']:.4f}")
    
    summary['Metric'].append('Uncalled4 Average Precision (m6A)')
    summary['Value'].append(f"{m6a_results['ap_uncalled4']:.4f}")
    
    summary['Metric'].append('Nanopolish Average Precision (m6A)')
    summary['Value'].append(f"{m6a_results['ap_nanopolish']:.4f}")
    
    summary['Metric'].append('Total m6A sites analyzed')
    summary['Value'].append(f"{m6a_results['n_sites']:,}")
    
    summary['Metric'].append('Positive m6A sites')
    summary['Value'].append(f"{m6a_results['n_positive']:,}")
    
    # Speed metrics
    for chem in perf_df['Chemistry'].unique():
        chem_data = perf_df[perf_df['Chemistry'] == chem]
        uncalled4_time = chem_data[chem_data['Tool'] == 'Uncalled4']['Time_min'].values
        if len(uncalled4_time) > 0 and pd.notna(uncalled4_time[0]):
            summary['Metric'].append(f'Uncalled4 time ({chem})')
            summary['Value'].append(f"{uncalled4_time[0]:.1f} min")
    
    # Pore model metrics
    for name, stats in pore_stats.items():
        summary['Metric'].append(f'{name} k-mer count')
        summary['Value'].append(f"{stats['n_kmers']:,}")
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(save_path, index=False)
    print(f"Saved summary table to {save_path}")
    
    return summary_df

def main():
    """Main analysis pipeline."""
    print("="*60)
    print("Uncalled4 Comprehensive Analysis Pipeline")
    print("="*60)
    
    # Load data
    models = load_pore_models()
    perf_df = load_performance_data()
    uncalled4_preds, nanopolish_preds, labels = load_m6a_data()
    
    # Run analyses
    pore_stats = analyze_pore_models(models)
    
    # Generate figures
    plot_pore_model_comparison(models, f"{REPORT_IMG_DIR}/fig1_pore_model_comparison.png")
    perf_clean = analyze_performance(perf_df, f"{REPORT_IMG_DIR}/fig2_performance_benchmark.png")
    m6a_results = analyze_m6a_predictions(uncalled4_preds, nanopolish_preds, labels, 
                                          f"{REPORT_IMG_DIR}/fig3_m6a_prediction_performance.png")
    speedup_df = plot_speedup_analysis(perf_df, f"{REPORT_IMG_DIR}/fig4_speedup_analysis.png")
    plot_current_by_base_composition(models, f"{REPORT_IMG_DIR}/fig5_gc_content_analysis.png")
    
    # Generate summary table
    summary_df = generate_summary_table(perf_df, m6a_results, pore_stats, 
                                       f"{OUTPUT_DIR}/summary_statistics.csv")
    
    # Save processed data
    perf_clean.to_csv(f"{OUTPUT_DIR}/performance_clean.csv", index=False)
    speedup_df.to_csv(f"{OUTPUT_DIR}/speedup_analysis.csv", index=False)
    
    print("\n" + "="*60)
    print("Analysis complete! Output files saved to:")
    print(f"  - Figures: {REPORT_IMG_DIR}/")
    print(f"  - Data: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
