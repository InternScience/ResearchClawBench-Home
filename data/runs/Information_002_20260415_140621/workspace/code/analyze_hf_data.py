#!/usr/bin/env python3
"""
Analysis script for Hartree-Fock Hamiltonian construction task evaluation.
This script processes the YAML data containing LLM responses and human evaluator scores
to generate comprehensive analysis and visualizations.
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_yaml_data(filepath):
    """Load and parse the YAML data file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def extract_task_scores(data):
    """Extract scores for each task and evaluator."""
    tasks = []
    
    for item in data:
        if 'task' not in item:
            continue
            
        task_name = item['task']
        task_data = {'task': task_name}
        
        # Extract overall scores
        if 'score' in item:
            scores = item['score']
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    task_data[f'score_{metric}'] = value
        
        # Extract placeholder-level scores from evaluators
        if 'placeholder' in item:
            for placeholder_name, placeholder_data in item['placeholder'].items():
                if isinstance(placeholder_data, dict) and 'score' in placeholder_data:
                    scores = placeholder_data['score']
                    for evaluator, score in scores.items():
                        if isinstance(score, (int, float)):
                            key = f'{placeholder_name}_{evaluator}'
                            task_data[key] = score
        
        tasks.append(task_data)
    
    return pd.DataFrame(tasks)

def calculate_evaluator_statistics(data):
    """Calculate statistics for each evaluator across all placeholders."""
    evaluator_scores = {'Haining': [], 'Will': [], 'Yasaman': []}
    
    for item in data:
        if 'placeholder' not in item:
            continue
            
        for placeholder_name, placeholder_data in item['placeholder'].items():
            if isinstance(placeholder_data, dict) and 'score' in placeholder_data:
                scores = placeholder_data['score']
                for evaluator in evaluator_scores.keys():
                    if evaluator in scores and isinstance(scores[evaluator], (int, float)):
                        evaluator_scores[evaluator].append(scores[evaluator])
    
    stats = {}
    for evaluator, scores in evaluator_scores.items():
        if scores:
            stats[evaluator] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores),
                'total_possible': len(scores) * 2,
                'accuracy': np.mean(scores) / 2 * 100,
                'raw_scores': scores
            }
    
    return stats, evaluator_scores

def calculate_dimension_scores(data):
    """Calculate scores by evaluation dimension."""
    dimension_scores = {}
    
    for item in data:
        if 'score' in item:
            scores = item['score']
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    if metric not in dimension_scores:
                        dimension_scores[metric] = []
                    dimension_scores[metric].append(value)
    
    stats = {}
    for metric, scores in dimension_scores.items():
        stats[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'count': len(scores)
        }
    
    return stats, dimension_scores

def plot_evaluator_comparison(stats, output_dir):
    """Create comparison plots for evaluators."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    evaluators = list(stats.keys())
    means = [stats[e]['mean'] for e in evaluators]
    stds = [stats[e]['std'] for e in evaluators]
    accuracies = [stats[e]['accuracy'] for e in evaluators]
    
    # Plot 1: Mean score comparison
    ax = axes[0, 0]
    bars = ax.bar(evaluators, means, yerr=stds, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Mean Score (out of 2)', fontsize=12)
    ax.set_title('Evaluator Mean Scores with Std Dev', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 2.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Score accuracy percentage
    ax = axes[0, 1]
    bars = ax.bar(evaluators, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Evaluator Accuracy (% of max possible)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Score distribution histogram
    ax = axes[1, 0]
    raw_scores_list = [stats[e]['raw_scores'] for e in evaluators]
    ax.hist(raw_scores_list, bins=np.arange(-0.5, 3, 1), label=evaluators, alpha=0.7, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Score Distribution by Evaluator', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.legend()
    
    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for evaluator in evaluators:
        s = stats[evaluator]
        table_data.append([
            evaluator,
            f"{s['mean']:.2f}",
            f"{s['std']:.2f}",
            f"{s['median']:.1f}",
            f"{s['count']}",
            f"{s['accuracy']:.1f}%"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Evaluator', 'Mean', 'Std', 'Median', 'Count', 'Accuracy'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Evaluator Statistics Summary', fontsize=14, fontweight='bold', y=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluator_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_dimension_analysis(dim_stats, output_dir):
    """Create plots for evaluation dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    dimensions = list(dim_stats.keys())
    means = [dim_stats[d]['mean'] for d in dimensions]
    stds = [dim_stats[d]['std'] for d in dimensions]
    
    # Plot 1: Bar chart of dimension scores
    ax = axes[0]
    bars = ax.barh(dimensions, means, xerr=stds, capsize=5, color='#9b59b6')
    ax.set_xlabel('Mean Score', fontsize=12)
    ax.set_title('Scores by Evaluation Dimension', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2.5)
    for bar, mean in zip(bars, means):
        ax.text(mean + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{mean:.2f}', ha='left', va='center', fontsize=10)
    
    # Plot 2: Dimension score distribution
    ax = axes[1]
    dim_names = [d.replace('score_', '') for d in dimensions]
    ax.bar(dim_names, means, yerr=stds, capsize=5, color='#f39c12')
    ax.set_ylabel('Mean Score (out of 2)', fontsize=12)
    ax.set_title('Dimension Scores with Error Bars', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 2.5)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dimension_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_task_progression(data, output_dir):
    """Plot scores across task progression."""
    tasks = []
    in_paper_scores = []
    physics_logic_scores = []
    math_derivation_scores = []
    final_answer_scores = []
    
    for item in data:
        if 'task' in item and 'score' in item:
            tasks.append(item['task'][:40] + '...' if len(item['task']) > 40 else item['task'])
            scores = item['score']
            in_paper_scores.append(scores.get('in_paper', 0))
            physics_logic_scores.append(scores.get('physics_logic', 0))
            math_derivation_scores.append(scores.get('math_derivation', 0))
            final_answer_scores.append(scores.get('final_answer_accuracy', 0))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(tasks))
    width = 0.2
    
    ax.bar(x - 1.5*width, in_paper_scores, width, label='In Paper', color='#3498db')
    ax.bar(x - 0.5*width, physics_logic_scores, width, label='Physics Logic', color='#e74c3c')
    ax.bar(x + 0.5*width, math_derivation_scores, width, label='Math Derivation', color='#2ecc71')
    ax.bar(x + 1.5*width, final_answer_scores, width, label='Final Answer', color='#f39c12')
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Score (out of 2)', fontsize=12)
    ax.set_title('Score Progression Across Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i+1}' for i in range(len(tasks))], rotation=0)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 2.5)
    
    # Add task mapping as text
    task_mapping = '\n'.join([f'T{i+1}: {t[:30]}...' if len(t) > 30 else f'T{i+1}: {t}' 
                               for i, t in enumerate(tasks)])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/task_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save task mapping
    with open(f'{output_dir}/task_mapping.txt', 'w') as f:
        f.write(task_mapping)

def generate_summary_report(stats, dim_stats, output_dir):
    """Generate a JSON summary report."""
    report = {
        'evaluator_statistics': {},
        'dimension_statistics': dim_stats,
        'overall_summary': {}
    }
    
    for evaluator, stat in stats.items():
        report['evaluator_statistics'][evaluator] = {
            'mean_score': round(stat['mean'], 3),
            'std_score': round(stat['std'], 3),
            'median_score': round(stat['median'], 3),
            'total_evaluations': stat['count'],
            'accuracy_percentage': round(stat['accuracy'], 2)
        }
    
    # Calculate overall averages
    all_accuracies = [stat['accuracy'] for stat in stats.values()]
    report['overall_summary']['average_accuracy'] = round(np.mean(all_accuracies), 2)
    report['overall_summary']['consistency'] = round(100 - np.std(all_accuracies), 2)
    
    with open(f'{output_dir}/summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    # Paths
    data_file = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Information_002_20260415_140621/data/2111.01152/2111.01152.yaml'
    output_dir = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Information_002_20260415_140621/outputs'
    report_dir = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Information_002_20260415_140621/report/images'
    
    # Load data
    print("Loading YAML data...")
    data = load_yaml_data(data_file)
    
    # Calculate statistics
    print("Calculating evaluator statistics...")
    evaluator_stats, evaluator_scores = calculate_evaluator_statistics(data)
    
    print("Calculating dimension statistics...")
    dim_stats, dim_scores = calculate_dimension_scores(data)
    
    # Generate plots
    print("Generating evaluator comparison plots...")
    plot_evaluator_comparison(evaluator_stats, output_dir)
    plot_evaluator_comparison(evaluator_stats, report_dir)
    
    print("Generating dimension analysis plots...")
    plot_dimension_analysis(dim_stats, output_dir)
    plot_dimension_analysis(dim_stats, report_dir)
    
    print("Generating task progression plots...")
    plot_task_progression(data, output_dir)
    plot_task_progression(data, report_dir)
    
    # Generate summary report
    print("Generating summary report...")
    report = generate_summary_report(evaluator_stats, dim_stats, output_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: {report_dir}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for evaluator, stat in evaluator_stats.items():
        print(f"{evaluator}: Mean={stat['mean']:.2f}, Accuracy={stat['accuracy']:.1f}%")
    
    return report

if __name__ == '__main__':
    main()
