#!/usr/bin/env python3
"""
Main analysis and visualization script for the Hartree-Fock LLM scoring benchmark.
Analyzes the scoring data for paper 2111.01152 (AB-stacked MoTe2/WSe2).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from collections import defaultdict

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'report', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Scoring dimensions
SCORE_DIMS = ['in_paper', 'prompt_quality', 'follow_instructions', 
              'physics_logic', 'math_derivation', 'final_answer_accuracy']

SCORE_DIM_LABELS = {
    'in_paper': 'In-Paper Content',
    'prompt_quality': 'Prompt Quality',
    'follow_instructions': 'Follow Instructions',
    'physics_logic': 'Physics Logic',
    'math_derivation': 'Math Derivation',
    'final_answer_accuracy': 'Final Answer Accuracy'
}

# Task category mapping
TASK_CATEGORIES = {
    'Construct Kinetic Hamiltonian': 'Hamiltonian Construction',
    'Define each term in Kinetic Hamiltonian': 'Hamiltonian Definition',
    'Construct Potential Hamiltonian': 'Hamiltonian Construction',
    'Define each term in Potential Hamiltonian': 'Hamiltonian Definition', 
    'Convert from single-particle to second-quantized': 'Quantization',
    'Convert noninteracting Hamiltonian': 'Fourier Transform',
    'Particle-hole transformation': 'Symmetry Transformation',
    'Simplify the Hamiltonian': 'Algebraic Simplification',
    'Construct interaction Hamiltonian': 'Interaction Construction',
    "Wick's theorem": 'Mean-Field Theory',
    'Extract quadratic term': 'Mean-Field Theory',
    'Swap the index': 'Algebraic Simplification',
    'Reduce momentum in Hartree term': 'Momentum Reduction',
    'Reduce momentum in Fock term': 'Momentum Reduction',
    'Combine the Hartree and Fock term': 'Mean-Field Theory',
}

def load_data():
    """Load all parsed data."""
    with open(os.path.join(OUTPUTS_DIR, 'task_scores.json')) as f:
        task_scores = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, 'annotator_agreements.json')) as f:
        agreements = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, 'parsed_tasks.json')) as f:
        parsed = json.load(f)
    return task_scores, agreements, parsed


def figure1_task_score_summary(task_scores):
    """Figure 1: Heatmap of all scores across tasks."""
    # Filter to tasks that have scores
    scored = [t for t in task_scores if any(d in t for d in SCORE_DIMS)]
    
    n_tasks = len(scored)
    n_dims = len(SCORE_DIMS)
    
    matrix = np.zeros((n_tasks, n_dims))
    task_names = []
    for i, t in enumerate(scored):
        task_names.append(t.get('task_name', f'Task {t["task_id"]}')[:60])
        for j, dim in enumerate(SCORE_DIMS):
            matrix[i, j] = t.get(dim, 0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=2)
    
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([SCORE_DIM_LABELS[d] for d in SCORE_DIMS], rotation=45, ha='right')
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_names, fontsize=8)
    
    # Add text annotations
    for i in range(n_tasks):
        for j in range(n_dims):
            val = matrix[i, j]
            color = 'white' if val <= 1 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', 
                   color=color, fontsize=8, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1, 2])
    cbar.set_label('Score', fontsize=11)
    
    ax.set_title('Task-Level Score Summary Across All Hartree-Fock Calculation Steps')
    plt.tight_layout()
    
    path = os.path.join(IMAGES_DIR, 'figure1_task_score_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def figure2_score_distribution(task_scores):
    """Figure 2: Distribution of scores per dimension."""
    scored = [t for t in task_scores if any(d in t for d in SCORE_DIMS)]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
    
    for j, dim in enumerate(SCORE_DIMS):
        ax = axes[j]
        values = [t.get(dim, 0) for t in scored]
        
        counts = {0: 0, 1: 0, 2: 0}
        for v in values:
            counts[int(v)] = counts.get(int(v), 0) + 1
        
        bars = ax.bar([0, 1, 2], [counts[0], counts[1], counts[2]], 
                     color=[colors[j]] * 3, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add count labels
        for bar, count in zip(bars, [counts[0], counts[1], counts[2]]):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', fontsize=10, fontweight='bold')
        
        mean_val = np.mean(values)
        ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=1.5, 
                  label=f'Mean={mean_val:.2f}')
        
        ax.set_title(SCORE_DIM_LABELS[dim])
        ax.set_xlabel('Score (0=Low, 2=High)')
        ax.set_ylabel('Count')
        ax.set_xticks([0, 1, 2])
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(counts.values()) + 2)
    
    fig.suptitle('Distribution of Hartree-Fock Calculation Scores by Dimension', 
                fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    path = os.path.join(IMAGES_DIR, 'figure2_score_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def figure3_annotator_agreement(agreements):
    """Figure 3: Inter-annotator agreement analysis."""
    annotators = ['Haining', 'Will', 'Yasaman']
    
    # Collect pairwise agreements
    pairs = [('Haining', 'Will'), ('Haining', 'Yasaman'), ('Will', 'Yasaman')]
    pair_data = {p: {'same': 0, 'diff1': 0, 'diff2': 0, 'total': 0} for p in pairs}
    
    # Per-annotator score distributions
    ann_scores = {a: [] for a in annotators}
    
    for agreement in agreements:
        scores = agreement['scores']
        for a in annotators:
            if a in scores:
                ann_scores[a].append(scores[a])
        
        for a1, a2 in pairs:
            if a1 in scores and a2 in scores:
                diff = abs(scores[a1] - scores[a2])
                pair_data[(a1, a2)]['total'] += 1
                if diff == 0:
                    pair_data[(a1, a2)]['same'] += 1
                elif diff == 1:
                    pair_data[(a1, a2)]['diff1'] += 1
                else:
                    pair_data[(a1, a2)]['diff2'] += 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Pairwise agreement heatmap-style
    ax = axes[0]
    pair_labels = ['Haining-Will', 'Haining-Yasaman', 'Will-Yasaman']
    
    x = np.arange(len(pair_labels))
    width = 0.25
    
    for idx, (pair, label) in enumerate(zip(pairs, pair_labels)):
        data = pair_data[pair]
        total = data['total']
        if total > 0:
            ax.bar(idx - width, data['same']/total*100, width, color='#4CAF50', 
                  alpha=0.8, label='Exact Agreement' if idx == 0 else '')
            ax.bar(idx, data['diff1']/total*100, width, color='#FF9800', 
                  alpha=0.8, label='1-pt Difference' if idx == 0 else '')
            ax.bar(idx + width, data['diff2']/total*100, width, color='#F44336', 
                  alpha=0.8, label='2-pt Difference' if idx == 0 else '')
    
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel('Percentage of Placeholder Scores (%)')
    ax.set_title('Pairwise Inter-Annotator Agreement')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 100)
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']
    # Right: Per-annotator score distribution
    ax = axes[1]
    for a, color in zip(annotators, colors):
        if ann_scores[a]:
            counts = {0: ann_scores[a].count(0), 1: ann_scores[a].count(1), 
                     2: ann_scores[a].count(2)}
            mean_s = np.mean(ann_scores[a])
            n = len(ann_scores[a])
            ax.bar(annotators.index(a) * 3 + np.array([0, 1, 2]), 
                  [counts[0], counts[1], counts[2]],
                  width=0.8, color=color, alpha=0.7, edgecolor='black', linewidth=0.5,
                  label=f'{a} (n={n}, μ={mean_s:.2f})')
    
    ax.set_xticks([1, 4, 7])
    ax.set_xticklabels(annotators)
    ax.set_ylabel('Number of Placeholder Scores')
    ax.set_title('Per-Annotator Score Distribution')
    ax.legend(fontsize=9)
    
    fig.suptitle('Inter-Annotator Agreement Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(IMAGES_DIR, 'figure3_annotator_agreement.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def figure4_category_analysis(task_scores):
    """Figure 4: Score breakdown by task category."""
    scored = [t for t in task_scores if any(d in t for d in SCORE_DIMS)]
    
    # Group by category
    categories = defaultdict(list)
    for t in scored:
        cat = 'Other'
        for prefix, cat_name in TASK_CATEGORIES.items():
            if t['task_name'].startswith(prefix):
                cat = cat_name
                break
        categories[cat].append(t)
    
    cat_order = ['Hamiltonian Construction', 'Hamiltonian Definition', 
                 'Quantization', 'Fourier Transform', 'Symmetry Transformation',
                 'Interaction Construction', 'Mean-Field Theory', 
                 'Algebraic Simplification', 'Momentum Reduction']
    
    cat_data = {}
    for cat in cat_order:
        if cat in categories:
            tasks_in_cat = categories[cat]
            means = {}
            for dim in SCORE_DIMS:
                vals = [t.get(dim, 0) for t in tasks_in_cat]
                means[dim] = np.mean(vals)
            cat_data[cat] = {'n': len(tasks_in_cat), 'means': means}
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    cats = list(cat_data.keys())
    n_cats = len(cats)
    n_dims = len(SCORE_DIMS)
    
    x = np.arange(n_cats)
    width = 0.12
    colors_bar = plt.cm.tab10(np.linspace(0, 1, n_dims))
    
    for j, dim in enumerate(SCORE_DIMS):
        values = [cat_data[cat]['means'].get(dim, 0) for cat in cats]
        offset = (j - n_dims/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=SCORE_DIM_LABELS[dim], 
              color=colors_bar[j], alpha=0.8, edgecolor='white', linewidth=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{cat}\n(n={cat_data[cat]["n"]})' for cat in cats], 
                       fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('Mean Score')
    ax.set_title('Hartree-Fock Calculation Scores by Task Category')
    ax.legend(fontsize=8, ncol=3, loc='lower left')
    ax.set_ylim(0, 2.5)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    path = os.path.join(IMAGES_DIR, 'figure4_category_analysis.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def figure5_task_progression(task_scores):
    """Figure 5: Score progression across tasks (showing difficulty progression)."""
    scored = [t for t in task_scores if any(d in t for d in SCORE_DIMS)]
    
    # Only use tasks that have scores
    valid = []
    for t in scored:
        has_scores = all(d in t for d in SCORE_DIMS)
        if has_scores:
            total = sum(t[d] for d in SCORE_DIMS)
            valid.append((t['task_id'], t['task_name'], total, t))
    
    valid.sort(key=lambda x: x[0])
    
    task_ids = [v[0] for v in valid]
    task_short = [v[1][:50] + '...' if len(v[1]) > 50 else v[1] for v in valid]
    totals = [v[2] for v in valid]
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Bar chart for total score
    bars = ax1.bar(range(len(task_ids)), totals, color='steelblue', alpha=0.8, 
                  edgecolor='navy', linewidth=0.5)
    ax1.set_ylabel('Total Score (max=12)', color='steelblue')
    ax1.set_ylim(0, 13)
    ax1.axhline(y=6, color='gray', linestyle='--', alpha=0.5, label='Midpoint (6/12)')
    
    # Overlay individual dimensions as stacked
    dims_data = defaultdict(list)
    for v in valid:
        t = v[3]
        for dim in SCORE_DIMS:
            dims_data[dim].append(t.get(dim, 0))
    
    n = len(task_ids)
    bottom = np.zeros(n)
    colors_stack = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
    
    for j, dim in enumerate(SCORE_DIMS):
        vals = np.array(dims_data[dim])
        # Just show cumulative
        pass
    
    ax1.set_xticks(range(len(task_ids)))
    ax1.set_xticklabels([f'T{i}' for i in task_ids], fontsize=8)
    ax1.set_title('Total Score per Hartree-Fock Calculation Step')
    ax1.legend(fontsize=9)
    
    plt.tight_layout()
    
    path = os.path.join(IMAGES_DIR, 'figure5_task_progression.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def figure6_radar_chart(task_scores):
    """Figure 6: Radar chart of average scores across dimensions."""
    scored = [t for t in task_scores if any(d in t for d in SCORE_DIMS)]
    
    means = {}
    for dim in SCORE_DIMS:
        vals = [t.get(dim, 0) for t in scored]
        means[dim] = np.mean(vals)
    
    # Radar chart
    n = len(SCORE_DIMS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    
    values = [means[d] for d in SCORE_DIMS]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue', markersize=8)
    
    # Add reference circles
    for r in [0.5, 1.0, 1.5, 2.0]:
        ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, 'k-', alpha=0.1, linewidth=0.5)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([SCORE_DIM_LABELS[d] for d in SCORE_DIMS], fontsize=10)
    ax.set_ylim(0, 2.2)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(['0.5', '1.0', '1.5', '2.0'], fontsize=8)
    
    ax.set_title('Average Hartree-Fock LLM Performance\nAcross Scoring Dimensions', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    path = os.path.join(IMAGES_DIR, 'figure6_radar_chart.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def figure7_placeholder_level_detail(task_scores, parsed):
    """Figure 7: Placeholder-level scoring detail - LLM accuracy per placeholder type."""
    # Extract placeholder-level LLM vs Human answers and scores
    # We need to re-parse the YAML for this detail
    import yaml
    yaml_path = os.path.join(SCRIPT_DIR, '..', 'data', '2111.01152', '2111.01152.yaml')
    with open(yaml_path, 'r') as f:
        tasks = yaml.safe_load(f)
    
    # Collect per-placeholder scores
    ph_scores = defaultdict(list)
    for task in tasks:
        if 'placeholder' in task:
            for ph_key, ph_val in task['placeholder'].items():
                if ph_val and 'score' in ph_val:
                    for annotator, score in ph_val['score'].items():
                        if isinstance(score, (int, float)):
                            ph_scores[ph_key].append(score)
    
    # Get top/middle/bottom performing placeholders
    ph_means = []
    for ph_key, scores in ph_scores.items():
        if scores:
            ph_means.append({
                'placeholder': ph_key[:40],
                'mean': np.mean(scores),
                'std': np.std(scores),
                'n': len(scores),
                'all_scores': scores
            })
    
    ph_means.sort(key=lambda x: x['mean'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_n = min(20, len(ph_means))
    top_items = ph_means[-top_n:] if len(ph_means) > 0 else ph_means
    
    y_pos = range(len(top_items))
    means = [item['mean'] for item in top_items]
    labels = [item['placeholder'] for item in top_items]
    
    colors_bar = plt.cm.RdYlGn(np.array(means) / 2.0)
    bars = ax.barh(y_pos, means, color=colors_bar, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean Score (0-2)')
    ax.set_title('Placeholder-Level LLM Performance\n(Top Performing Placeholders)', fontsize=14)
    ax.set_xlim(0, 2.2)
    
    plt.tight_layout()
    
    path = os.path.join(IMAGES_DIR, 'figure7_placeholder_detail.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def generate_summary_statistics(task_scores, agreements, parsed):
    """Generate comprehensive summary statistics."""
    stats = {}
    
    # Task-level statistics
    scored = [t for t in task_scores if any(d in t for d in SCORE_DIMS)]
    
    # Per-dimension stats
    dim_stats = {}
    for dim in SCORE_DIMS:
        vals = [t.get(dim, 0) for t in scored]
        dim_stats[dim] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': int(np.min(vals)),
            'max': int(np.max(vals)),
            'median': float(np.median(vals)),
            'count_zero': sum(1 for v in vals if v == 0),
            'count_one': sum(1 for v in vals if v == 1),
            'count_two': sum(1 for v in vals if v == 2),
        }
    stats['per_dimension'] = dim_stats
    
    # Overall statistics
    all_scores = []
    for t in scored:
        for dim in SCORE_DIMS:
            all_scores.append(t.get(dim, 0))
    stats['overall_mean'] = float(np.mean(all_scores))
    stats['overall_std'] = float(np.std(all_scores))
    
    # Total score per task
    totals = []
    for t in scored:
        total = sum(t.get(d, 0) for d in SCORE_DIMS)
        totals.append(total)
    stats['total_per_task_mean'] = float(np.mean(totals))
    stats['total_per_task_std'] = float(np.std(totals))
    stats['total_per_task_max'] = float(max(totals))
    stats['total_per_task_min'] = float(min(totals))
    
    # Inter-annotator agreement
    annotators = ['Haining', 'Will', 'Yasaman']
    ann_means = {}
    for a in annotators:
        scores = []
        for ag in agreements:
            if a in ag['scores']:
                scores.append(ag['scores'][a])
        if scores:
            ann_means[a] = {'mean': float(np.mean(scores)), 'count': len(scores)}
    stats['annotator_means'] = ann_means
    
    # Pairwise exact agreement rates
    pairs = [('Haining', 'Will'), ('Haining', 'Yasaman'), ('Will', 'Yasaman')]
    pair_rates = {}
    for a1, a2 in pairs:
        same, total = 0, 0
        for ag in agreements:
            if a1 in ag['scores'] and a2 in ag['scores']:
                total += 1
                if ag['scores'][a1] == ag['scores'][a2]:
                    same += 1
        if total > 0:
            pair_rates[f'{a1}-{a2}'] = float(same / total)
    stats['pairwise_exact_agreement'] = pair_rates
    
    return stats


def main():
    print("Loading data...")
    task_scores, agreements, parsed = load_data()
    
    print("Generating figures...")
    figure1_task_score_summary(task_scores)
    figure2_score_distribution(task_scores)
    figure3_annotator_agreement(agreements)
    figure4_category_analysis(task_scores)
    figure5_task_progression(task_scores)
    figure6_radar_chart(task_scores)
    figure7_placeholder_level_detail(task_scores, parsed)
    
    print("Computing summary statistics...")
    stats = generate_summary_statistics(task_scores, agreements, parsed)
    
    with open(os.path.join(OUTPUTS_DIR, 'summary_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Summary statistics saved.")
    
    # Print key findings
    print(f"\n=== KEY FINDINGS ===")
    print(f"Overall mean score across all dimensions: {stats['overall_mean']:.2f} ± {stats['overall_std']:.2f}")
    print(f"Average total score per task: {stats['total_per_task_mean']:.1f} / 12")
    print(f"\nPer-dimension means:")
    for dim in SCORE_DIMS:
        d = stats['per_dimension'][dim]
        print(f"  {SCORE_DIM_LABELS[dim]}: {d['mean']:.2f} ± {d['std']:.2f}")
    
    print(f"\nInter-annotator exact agreement rates:")
    for pair, rate in stats['pairwise_exact_agreement'].items():
        print(f"  {pair}: {rate:.1%}")
    
    print(f"\nAnnotator mean scores:")
    for a, d in stats['annotator_means'].items():
        print(f"  {a}: {d['mean']:.2f} (n={d['count']})")
    
    return stats


if __name__ == '__main__':
    stats = main()
