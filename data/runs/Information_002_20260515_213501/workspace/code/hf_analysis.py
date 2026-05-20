#!/usr/bin/env python3
"""
Hartree-Fock calculation analysis for AB-stacked MoTe2/WSe2 moiré system
Paper: 2111.01152
"""

import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_yaml_data(yaml_path):
    """Load and parse the structured HF calculation data."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def extract_tasks(data):
    """Extract individual tasks from the YAML structure."""
    tasks = []
    for item in data:
        if isinstance(item, dict) and 'task' in item:
            task_info = {
                'task_name': item['task'],
                'source': item.get('source', {}),
                'placeholder': item.get('placeholder', {}),
                'answer': item.get('answer', ''),
                'score': item.get('score', {})
            }
            tasks.append(task_info)
    return tasks

def compute_step_scores(tasks):
    """Compute aggregated step scores from expert evaluations."""
    score_data = []
    
    for task in tasks:
        task_name = task['task_name']
        scores = task['score']
        
        # Extract expert scores
        expert_scores = {}
        for expert in ['Haining', 'Will', 'Yasaman']:
            if expert in scores:
                expert_scores[expert] = scores[expert]
        
        # Calculate mean score per expert
        mean_score = np.mean(list(expert_scores.values())) if expert_scores else np.nan
        
        score_data.append({
            'task': task_name,
            'mean_score': mean_score,
            'Haining': expert_scores.get('Haining', np.nan),
            'Will': expert_scores.get('Will', np.nan),
            'Yasaman': expert_scores.get('Yasaman', np.nan),
            'num_experts': len(expert_scores)
        })
    
    return pd.DataFrame(score_data)

def extract_hamiltonians(tasks):
    """Extract Hamiltonian expressions from tasks."""
    hamiltonians = []
    
    for task in tasks:
        task_name = task['task_name']
        answer = task['answer']
        
        if answer and isinstance(answer, str):
            hamiltonians.append({
                'task': task_name,
                'hamiltonian': answer[:200] + '...' if len(answer) > 200 else answer
            })
    
    return pd.DataFrame(hamiltonians)

def generate_figures(score_df, ham_df, output_dir):
    """Generate publication-quality figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Score distribution by expert
    fig, ax = plt.subplots(figsize=(10, 6))
    
    expert_cols = ['Haining', 'Will', 'Yasaman']
    score_melted = score_df[expert_cols].melt(var_name='Expert', value_name='Score')
    score_melted = score_melted.dropna()
    
    sns.boxplot(data=score_melted, x='Expert', y='Score', ax=ax)
    sns.swarmplot(data=score_melted, x='Expert', y='Score', ax=ax, color='black', alpha=0.6)
    
    ax.set_ylabel('Step Score (0-2)', fontsize=12)
    ax.set_xlabel('Expert Evaluator', fontsize=12)
    ax.set_title('Distribution of Hartree-Fock Calculation Step Scores\nby Expert Evaluator', fontsize=14, pad=20)
    ax.set_ylim(-0.1, 2.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Mean scores per task
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Truncate long task names
    score_df['task_short'] = score_df['task'].str[:50] + '...'
    
    colors = ['#2ecc71' if s >= 1.5 else '#f39c12' if s >= 1.0 else '#e74c3c' 
              for s in score_df['mean_score']]
    
    bars = ax.barh(range(len(score_df)), score_df['mean_score'], color=colors)
    ax.set_yticks(range(len(score_df)))
    ax.set_yticklabels(score_df['task_short'], fontsize=9)
    ax.set_xlabel('Mean Step Score (0-2)', fontsize=12)
    ax.set_title('Hartree-Fock Calculation Performance by Task\n(Color: Green≥1.5, Orange≥1.0, Red<1.0)', 
                 fontsize=14, pad=20)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 2.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_task_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Score correlation between experts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pairs = [('Haining', 'Will'), ('Haining', 'Yasaman'), ('Will', 'Yasaman')]
    
    for idx, (exp1, exp2) in enumerate(pairs):
        ax = axes[idx]
        valid = score_df[[exp1, exp2]].dropna()
        
        ax.scatter(valid[exp1], valid[exp2], alpha=0.7, s=100)
        ax.plot([0, 2], [0, 2], 'k--', alpha=0.3)
        ax.set_xlabel(f'{exp1} Score', fontsize=11)
        ax.set_ylabel(f'{exp2} Score', fontsize=11)
        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(-0.1, 2.1)
        ax.set_title(f'{exp1} vs {exp2}', fontsize=12)
        
        # Add correlation coefficient
        if len(valid) > 1:
            corr = valid[exp1].corr(valid[exp2])
            ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top')
    
    plt.suptitle('Inter-Expert Score Correlation for Hartree-Fock Calculations', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_expert_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Task category performance summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Categorize tasks
    categories = {
        'Kinetic': [],
        'Potential': [],
        'Second-quantization': [],
        'Transformation': []
    }
    
    for _, row in score_df.iterrows():
        task = row['task'].lower()
        if 'kinetic' in task:
            categories['Kinetic'].append(row['mean_score'])
        elif 'potential' in task:
            categories['Potential'].append(row['mean_score'])
        elif 'second' in task or 'quantized' in task:
            categories['Second-quantization'].append(row['mean_score'])
        elif 'particle' in task or 'hole' in task or 'momentum' in task:
            categories['Transformation'].append(row['mean_score'])
    
    cat_means = {k: np.mean(v) if v else 0 for k, v in categories.items()}
    cat_stds = {k: np.std(v) if v else 0 for k, v in categories.items()}
    
    cats = list(cat_means.keys())
    means = [cat_means[c] for c in cats]
    stds = [cat_stds[c] for c in cats]
    
    x = np.arange(len(cats))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel('Mean Score ± Std Dev', fontsize=12)
    ax.set_title('Hartree-Fock Calculation Performance by Task Category', fontsize=14, pad=20)
    ax.set_ylim(0, 2.2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Acceptable threshold')
    
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_category_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated 4 figures in {output_dir}")
    return ['figure1_score_distribution.png', 'figure2_task_scores.png', 
            'figure3_expert_correlation.png', 'figure4_category_performance.png']

def main():
    # Paths
    yaml_path = Path('data/2111.01152/2111.01152.yaml')
    output_dir = Path('outputs')
    images_dir = Path('report/images')
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading YAML data...")
    data = load_yaml_data(yaml_path)
    tasks = extract_tasks(data)
    
    print(f"Extracted {len(tasks)} tasks")
    
    # Compute scores
    score_df = compute_step_scores(tasks)
    score_df.to_csv(output_dir / 'step_scores.csv', index=False)
    
    # Extract Hamiltonians
    ham_df = extract_hamiltonians(tasks)
    ham_df.to_csv(output_dir / 'hamiltonians.csv', index=False)
    
    # Generate figures
    figures = generate_figures(score_df, ham_df, images_dir)
    
    # Save summary statistics
    summary = {
        'num_tasks': len(tasks),
        'mean_score_overall': float(score_df['mean_score'].mean()),
        'std_score_overall': float(score_df['mean_score'].std()),
        'tasks_above_threshold': int((score_df['mean_score'] >= 1.0).sum()),
        'expert_agreement': float(score_df[['Haining', 'Will', 'Yasaman']].std(axis=1).mean())
    }
    
    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Analysis Summary ===")
    print(f"Total tasks: {summary['num_tasks']}")
    print(f"Overall mean score: {summary['mean_score_overall']:.2f} ± {summary['std_score_overall']:.2f}")
    print(f"Tasks above threshold (≥1.0): {summary['tasks_above_threshold']}/{summary['num_tasks']}")
    print(f"Mean expert disagreement (std): {summary['expert_agreement']:.2f}")
    print("\nAnalysis complete. Results saved to outputs/ and report/images/")

if __name__ == '__main__':
    main()