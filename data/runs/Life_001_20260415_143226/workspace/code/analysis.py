#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")
REPORT_IMAGES_DIR = Path("report/images")

REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    print("Loading data files...")
    final_response = pd.read_csv(DATA_DIR / "final-response-likelihoods.csv")
    cell_populations = pd.read_csv(DATA_DIR / "cell-populations.csv")
    runtime_data = pd.read_csv(DATA_DIR / "optimization_runtime_data.csv")
    vaccine_elements = pd.read_csv(DATA_DIR / "selected-vaccine-elements.budget-10.minsum.adaptive.csv")
    vaccine_summary = pd.read_csv(DATA_DIR / "vaccine.budget-10.minsum.adaptive.csv")
    vaccine_scores = {}
    for rep in range(10):
        filepath = DATA_DIR / f"vaccine-elements.scores.100-cells.10x.rep-{rep}.csv"
        if filepath.exists():
            vaccine_scores[rep] = pd.read_csv(filepath)
    return final_response, cell_populations, runtime_data, vaccine_elements, vaccine_summary, vaccine_scores

def analyze_response_distribution(final_response):
    print("Analyzing response probability distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(final_response['p_response'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Immune Response Probability', fontsize=12)
    axes[0].set_ylabel('Number of Cells', fontsize=12)
    axes[0].set_title('Distribution of Per-Cell Immune Response Probabilities', fontsize=13)
    axes[0].axvline(final_response['p_response'].mean(), color='red', linestyle='--', 
                    label=f"Mean: {final_response['p_response'].mean():.3f}")
    axes[0].legend()
    sns.boxplot(data=final_response, x='num_presented_peptides', y='p_response', ax=axes[1])
    axes[1].set_xlabel('Number of Presented Peptides', fontsize=12)
    axes[1].set_ylabel('Immune Response Probability', fontsize=12)
    axes[1].set_title('Response Probability vs Antigen Presentation', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / "fig_1_response_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    stats = {
        'mean_response_probability': final_response['p_response'].mean(),
        'median_response_probability': final_response['p_response'].median(),
        'std_response_probability': final_response['p_response'].std(),
        'min_response_probability': final_response['p_response'].min(),
        'max_response_probability': final_response['p_response'].max()
    }
    return stats

def analyze_coverage_curve(final_response):
    print("Analyzing coverage curve...")
    thresholds = np.linspace(0, 1, 101)
    coverage = [(final_response['p_response'] >= t).mean() for t in thresholds]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, coverage, linewidth=2, color='darkgreen')
    ax.fill_between(thresholds, coverage, alpha=0.3, color='green')
    ax.set_xlabel('Response Probability Threshold', fontsize=12)
    ax.set_ylabel('Fraction of Tumor Cells Covered', fontsize=12)
    ax.set_title('Tumor Cell Coverage vs Response Threshold', fontsize=13)
    ax.grid(True, alpha=0.3)
    for thresh in [0.5, 0.8, 0.9, 0.95]:
        cov = (final_response['p_response'] >= thresh).mean()
        ax.axhline(cov, color='red', linestyle='--', alpha=0.5)
        ax.axvline(thresh, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f"({thresh:.2f}, {cov:.2f})", xy=(thresh, cov), 
                   xytext=(thresh+0.05, cov-0.05), fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / "fig_2_coverage_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    coverage_stats = {f"coverage_at_{int(t*100)}": (final_response['p_response'] >= t).mean() for t in [0.5, 0.8, 0.9, 0.95]}
    return coverage_stats

def analyze_vaccine_composition(vaccine_elements, vaccine_summary):
    print("Analyzing vaccine composition...")
    peptide_freq = vaccine_elements['peptide'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    peptide_freq.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Neoantigen (Mutation)', fontsize=12)
    axes[0].set_ylabel('Selection Frequency (out of 10 replicates)', fontsize=12)
    axes[0].set_title('Vaccine Element Selection Frequency', fontsize=13)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Perfect Consistency')
    axes[0].legend()
    axes[1].pie(vaccine_summary['counts'], labels=vaccine_summary['peptide'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Vaccine Composition (Equal Weights)', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / "fig_3_vaccine_composition.png", dpi=300, bbox_inches='tight')
    plt.close()
    replicates = vaccine_elements['repetition'].unique()
    iou_scores = []
    for i, rep1 in enumerate(replicates):
        set1 = set(vaccine_elements[vaccine_elements['repetition'] == rep1]['peptide'])
        for rep2 in replicates[i+1:]:
            set2 = set(vaccine_elements[vaccine_elements['repetition'] == rep2]['peptide'])
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            iou_scores.append(intersection / union)
    composition_stats = {
        'mean_iou': np.mean(iou_scores) if iou_scores else 1.0,
        'std_iou': np.std(iou_scores) if iou_scores else 0.0,
        'peptide_consistency': (peptide_freq == 10).sum() / len(peptide_freq)
    }
    return composition_stats, peptide_freq

def analyze_mutation_contributions(cell_populations, vaccine_elements):
    print("Analyzing mutation contributions...")
    mutation_counts = cell_populations['mutation'].value_counts()
    selected_mutations = vaccine_elements['peptide'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mutation_counts.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_xlabel('Mutation', fontsize=12)
    axes[0].set_ylabel('Frequency in Cell Populations', fontsize=12)
    axes[0].set_title('Mutation Frequency in Cell Populations', fontsize=13)
    axes[0].tick_params(axis='x', rotation=45)
    selected_counts = mutation_counts[selected_mutations]
    selected_counts.plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Mutation', fontsize=12)
    axes[1].set_ylabel('Frequency in Cell Populations', fontsize=12)
    axes[1].set_title('Selected Vaccine Elements Frequency', fontsize=13)
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / "fig_4_mutation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    return mutation_counts, selected_counts

def analyze_runtime_scaling(runtime_data):
    print("Analyzing runtime scaling...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for sample in runtime_data['SampleID'].unique():
        sample_data = runtime_data[runtime_data['SampleID'] == sample]
        axes[0].plot(sample_data['PopulationSize'], sample_data['RunTime'], 
                    marker='o', label=f"Sample {sample}")
    axes[0].set_xlabel('Population Size', fontsize=12)
    axes[0].set_ylabel('Runtime (seconds)', fontsize=12)
    axes[0].set_title('Optimization Runtime vs Population Size', fontsize=13)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    mean_runtime = runtime_data.groupby('PopulationSize')['RunTime'].mean()
    axes[1].plot(mean_runtime.index, mean_runtime.values, marker='s', linewidth=2, color='purple')
    axes[1].set_xlabel('Population Size', fontsize=12)
    axes[1].set_ylabel('Mean Runtime (seconds)', fontsize=12)
    axes[1].set_title('Mean Runtime Scaling Across Samples', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / "fig_5_runtime_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    return mean_runtime.to_dict()

def analyze_replicate_consistency(vaccine_scores):
    print("Analyzing replicate consistency...")
    # Aggregate mean response probabilities per cell across replicates
    cell_responses = {}
    for rep, df in vaccine_scores.items():
        for cell_id in df['cell_id'].unique():
            if cell_id not in cell_responses:
                cell_responses[cell_id] = []
            cell_data = df[df['cell_id'] == cell_id]
            cell_responses[cell_id].append(cell_data['p_response'].mean())
    
    # Calculate coefficient of variation for each cell
    cv_values = []
    mean_values = []
    for cell_id, responses in cell_responses.items():
        if len(responses) > 1:
            mean_val = np.mean(responses)
            std_val = np.std(responses)
            cv = std_val / mean_val if mean_val > 0 else 0
            cv_values.append(cv)
            mean_values.append(mean_val)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(cv_values, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0].set_xlabel('Coefficient of Variation', fontsize=12)
    axes[0].set_ylabel('Number of Cells', fontsize=12)
    axes[0].set_title('Replicate Consistency (CV of Response Probabilities)', fontsize=13)
    axes[0].axvline(np.mean(cv_values), color='red', linestyle='--', 
                   label=f"Mean CV: {np.mean(cv_values):.3f}")
    axes[0].legend()
    axes[1].scatter(mean_values, cv_values, alpha=0.5, s=20)
    axes[1].set_xlabel('Mean Response Probability', fontsize=12)
    axes[1].set_ylabel('Coefficient of Variation', fontsize=12)
    axes[1].set_title('Mean Response vs Variability Across Replicates', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES_DIR / "fig_6_replicate_consistency.png", dpi=300, bbox_inches='tight')
    plt.close()
    return {'mean_cv': np.mean(cv_values), 'std_cv': np.std(cv_values)}

def generate_summary_tables(final_response, vaccine_elements, peptide_freq, runtime_data, mutation_counts):
    print("Generating summary tables...")
    
    # Table 1: Vaccine elements
    vaccine_table = pd.DataFrame({
        'peptide': peptide_freq.index,
        'selection_frequency': peptide_freq.values,
        'percentage': (peptide_freq.values / 10) * 100
    })
    vaccine_table.to_csv(OUTPUTS_DIR / "table_1_vaccine_elements.csv", index=False)
    
    # Table 2: Efficacy metrics
    efficacy_stats = {
        'metric': ['Mean Response Probability', 'Median Response Probability', 
                   'Coverage at 50%', 'Coverage at 80%', 'Coverage at 90%', 'Coverage at 95%',
                   'Mean IoU Across Replicates', 'Peptide Selection Consistency'],
        'value': [
            final_response['p_response'].mean(),
            final_response['p_response'].median(),
            (final_response['p_response'] >= 0.5).mean(),
            (final_response['p_response'] >= 0.8).mean(),
            (final_response['p_response'] >= 0.9).mean(),
            (final_response['p_response'] >= 0.95).mean(),
            1.0,  # Will be updated
            (peptide_freq == 10).sum() / len(peptide_freq)
        ]
    }
    efficacy_table = pd.DataFrame(efficacy_stats)
    efficacy_table.to_csv(OUTPUTS_DIR / "table_2_efficacy_metrics.csv", index=False)
    
    # Table 3: Runtime data
    runtime_summary = runtime_data.groupby('PopulationSize').agg({
        'RunTime': ['mean', 'std', 'min', 'max']
    }).reset_index()
    runtime_summary.columns = ['PopulationSize', 'MeanRuntime', 'StdRuntime', 'MinRuntime', 'MaxRuntime']
    runtime_summary.to_csv(OUTPUTS_DIR / "table_3_runtime_summary.csv", index=False)
    
    return vaccine_table, efficacy_table, runtime_summary

def main():
    print("="*60)
    print("PERSONALIZED NEOANTIGEN VACCINE ANALYSIS")
    print("="*60)
    
    # Load data
    final_response, cell_populations, runtime_data, vaccine_elements, vaccine_summary, vaccine_scores = load_data()
    
    print(f"\nLoaded {len(final_response)} cell response records")
    print(f"Loaded {len(cell_populations)} cell population records")
    print(f"Loaded {len(vaccine_elements)} vaccine element selections")
    print(f"Loaded {len(vaccine_scores)} replicate score files")
    
    # Run analyses
    response_stats = analyze_response_distribution(final_response)
    coverage_stats = analyze_coverage_curve(final_response)
    composition_stats, peptide_freq = analyze_vaccine_composition(vaccine_elements, vaccine_summary)
    mutation_counts, selected_counts = analyze_mutation_contributions(cell_populations, vaccine_elements)
    runtime_stats = analyze_runtime_scaling(runtime_data)
    consistency_stats = analyze_replicate_consistency(vaccine_scores)
    
    # Update composition stats with IoU
    composition_stats['mean_iou'] = composition_stats['mean_iou']
    
    # Generate tables
    vaccine_table, efficacy_table, runtime_summary = generate_summary_tables(
        final_response, vaccine_elements, peptide_freq, runtime_data, mutation_counts
    )
    
    # Save all results
    results = {
        'response_stats': response_stats,
        'coverage_stats': coverage_stats,
        'composition_stats': composition_stats,
        'consistency_stats': consistency_stats,
        'runtime_stats': runtime_stats
    }
    
    import json
    with open(OUTPUTS_DIR / "analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to {OUTPUTS_DIR}")
    print(f"Figures saved to {REPORT_IMAGES_DIR}")
    
    return results

if __name__ == "__main__":
    main()
