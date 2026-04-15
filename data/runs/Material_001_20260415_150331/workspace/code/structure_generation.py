"""
Structure Generation Workflow Analysis
Analyzes the quality of AI-generated crystal structures by comparing
generated lattice constants against target values.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats


def load_data():
    """Load parsed structure generation data."""
    with open('../outputs/parsed_data.json', 'r') as f:
        data = json.load(f)
    return data['structure_generation']


def analyze_structures(sg_data):
    """Analyze generated vs target structures."""
    generated = np.array(sg_data['generated_lattice'])
    target = np.array(sg_data['target_lattice'])
    
    # Ensure same length for comparison
    min_len = min(len(generated), len(target))
    generated = generated[:min_len]
    target = target[:min_len]
    
    # Calculate error metrics
    errors = generated - target
    abs_errors = np.abs(errors)
    
    metrics = {
        'n_samples': min_len,
        'generated_mean': float(np.mean(generated)),
        'generated_std': float(np.std(generated)),
        'target_mean': float(np.mean(target)),
        'target_std': float(np.std(target)),
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'max_error': float(np.max(abs_errors)),
        'min_error': float(np.min(abs_errors)),
        'mean_error': float(np.mean(errors)),
        'correlation': float(np.corrcoef(generated, target)[0, 1]),
        'r2': float(1 - np.sum(errors**2) / np.sum((target - np.mean(target))**2))
    }
    
    # Statistical significance test
    t_stat, p_value = stats.ttest_rel(generated, target)
    metrics['t_statistic'] = float(t_stat)
    metrics['p_value'] = float(p_value)
    
    return metrics, generated, target, errors


def plot_analysis(metrics, generated, target, errors):
    """Generate comprehensive visualization plots."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Generated vs Target Scatter Plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(target, generated, alpha=0.6, c='steelblue', edgecolors='black', s=80)
    ax1.plot([target.min(), target.max()], [target.min(), target.max()], 
             'r--', lw=2, label='Perfect Match')
    ax1.set_xlabel('Target Lattice Constant (Å)', fontsize=12)
    ax1.set_ylabel('Generated Lattice Constant (Å)', fontsize=12)
    ax1.set_title('Generated vs Target Structures', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add metrics text
    textstr = f'R² = {metrics["r2"]:.4f}\nCorrelation = {metrics["correlation"]:.4f}\nMAE = {metrics["mae"]:.4f} Å\nRMSE = {metrics["rmse"]:.4f} Å'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Distribution Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(target, bins=20, alpha=0.5, label='Target', color='coral', edgecolor='black')
    ax2.hist(generated, bins=20, alpha=0.5, label='Generated', color='steelblue', edgecolor='black')
    ax2.set_xlabel('Lattice Constant (Å)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # 3. Error Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(errors, bins=20, color='green', alpha=0.6, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=metrics['mean_error'], color='orange', linestyle='-', linewidth=2, label=f'Mean: {metrics["mean_error"]:.4f}')
    ax3.set_xlabel('Error (Generated - Target) (Å)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # 4. Absolute Error Trend
    ax4 = fig.add_subplot(gs[1, 1])
    sample_idx = np.arange(len(errors))
    ax4.bar(sample_idx[:30], np.abs(errors[:30]), color='coral', alpha=0.7, edgecolor='black')
    ax4.axhline(y=metrics['mae'], color='blue', linestyle='--', linewidth=2, label=f'MAE = {metrics["mae"]:.4f}')
    ax4.set_xlabel('Sample Index', fontsize=11)
    ax4.set_ylabel('Absolute Error (Å)', fontsize=11)
    ax4.set_title('Absolute Errors (First 30 Samples)', fontsize=12, fontweight='bold')
    ax4.legend()
    
    # 5. Q-Q Plot for Normality Check
    ax5 = fig.add_subplot(gs[1, 2])
    stats.probplot(errors, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Time series of predictions
    ax6 = fig.add_subplot(gs[2, :])
    sample_range = slice(0, min(50, len(generated)))
    x = np.arange(len(generated[sample_range]))
    ax6.plot(x, target[sample_range], 'o-', label='Target', color='coral', markersize=6, linewidth=2)
    ax6.plot(x, generated[sample_range], 's--', label='Generated', color='steelblue', markersize=6, linewidth=2)
    ax6.fill_between(x, target[sample_range], generated[sample_range], alpha=0.2, color='gray')
    ax6.set_xlabel('Sample Index', fontsize=12)
    ax6.set_ylabel('Lattice Constant (Å)', fontsize=12)
    ax6.set_title('Target vs Generated Comparison (First 50 Samples)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.savefig('../report/images/structure_generation.png', dpi=300, bbox_inches='tight')
    plt.savefig('../outputs/structure_generation.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_analysis(metrics):
    """Print detailed analysis results."""
    print("\n" + "=" * 60)
    print("STRUCTURE GENERATION ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nSample Size: {metrics['n_samples']}")
    
    print("\n1. DESCRIPTIVE STATISTICS")
    print("-" * 40)
    print(f"Generated Structures:")
    print(f"  Mean: {metrics['generated_mean']:.4f} ± {metrics['generated_std']:.4f} Å")
    print(f"Target Structures:")
    print(f"  Mean: {metrics['target_mean']:.4f} ± {metrics['target_std']:.4f} Å")
    
    print("\n2. ERROR METRICS")
    print("-" * 40)
    print(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f} Å")
    print(f"  Root Mean Square Error (RMSE): {metrics['rmse']:.4f} Å")
    print(f"  Max Absolute Error: {metrics['max_error']:.4f} Å")
    print(f"  Min Absolute Error: {metrics['min_error']:.4f} Å")
    print(f"  Mean Error (Bias): {metrics['mean_error']:.4f} Å")
    
    print("\n3. CORRELATION ANALYSIS")
    print("-" * 40)
    print(f"  Pearson Correlation: {metrics['correlation']:.4f}")
    print(f"  R² Score: {metrics['r2']:.4f}")
    
    print("\n4. STATISTICAL TESTS")
    print("-" * 40)
    print(f"  Paired t-test statistic: {metrics['t_statistic']:.4f}")
    print(f"  p-value: {metrics['p_value']:.4f}")
    if metrics['p_value'] < 0.05:
        print("  Result: Statistically significant difference (p < 0.05)")
    else:
        print("  Result: No statistically significant difference (p >= 0.05)")
    
    print("\n5. QUALITY ASSESSMENT")
    print("-" * 40)
    if metrics['r2'] > 0.9:
        quality = "Excellent"
    elif metrics['r2'] > 0.8:
        quality = "Good"
    elif metrics['r2'] > 0.6:
        quality = "Moderate"
    else:
        quality = "Needs Improvement"
    print(f"  Generation Quality: {quality}")
    print(f"  Mean Deviation: {metrics['mae']:.4f} Å ({metrics['mae']/metrics['target_mean']*100:.2f}% of mean)")
    
    print("\n" + "=" * 60)


def save_metrics(metrics):
    """Save metrics to JSON file."""
    with open('../outputs/structure_generation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to outputs/structure_generation_metrics.json")


def main():
    print("=" * 60)
    print("STRUCTURE GENERATION WORKFLOW ANALYSIS")
    print("=" * 60)
    
    # Load data
    sg_data = load_data()
    print(f"\nLoaded {len(sg_data['generated_lattice'])} generated structures")
    print(f"Loaded {len(sg_data['target_lattice'])} target structures")
    
    # Analyze
    metrics, generated, target, errors = analyze_structures(sg_data)
    
    # Print results
    print_analysis(metrics)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_analysis(metrics, generated, target, errors)
    
    # Save metrics
    save_metrics(metrics)
    
    print("\nStructure generation analysis complete!")
    print("Plots saved to: report/images/structure_generation.png")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    main()
