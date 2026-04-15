#!/usr/bin/env python3
"""
Analyze Hartree-Fock multi-step calculation scoring data from paper 2111.01152.
Parse YAML scoring data, compute aggregate statistics, extract paper information,
and generate visualization figures.
"""

import yaml
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = 'data/2111.01152'
OUTPUTS_DIR = 'outputs'
REPORT_IMAGES_DIR = 'report/images'

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)

# ─── Step 1: Parse YAML ──────────────────────────────────────────────────────
def parse_yaml():
    """Parse the YAML scoring file and return structured data."""
    with open(os.path.join(DATA_DIR, '2111.01152.yaml'), 'r') as f:
        raw = yaml.safe_load(f)
    
    tasks = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        
        task_name = item.get('task', 'Unknown')
        branch = item.get('branch', '')
        
        # Extract placeholder scores
        placeholders = item.get('placeholder', {})
        llm_values = {}
        human_values = {}
        scorer_scores = {}  # {placeholder_key: {scorer: score}}
        
        for pk, pv in placeholders.items():
            if not isinstance(pv, dict):
                continue
            llm_val = pv.get('LLM', None)
            human_val = pv.get('human', None)
            score_dict = pv.get('score', {})
            
            llm_values[pk] = llm_val
            human_values[pk] = human_val
            
            if score_dict:
                scorer_scores[pk] = {}
                for scorer, val in score_dict.items():
                    try:
                        scorer_scores[pk][scorer] = float(val)
                    except (ValueError, TypeError):
                        scorer_scores[pk][scorer] = None  # e.g., "(?)"
        
        # Extract step-level scores
        step_score = item.get('score', {})
        step_score_parsed = {}
        for k, v in step_score.items():
            try:
                step_score_parsed[k] = float(v)
            except (ValueError, TypeError):
                step_score_parsed[k] = None
        
        answer = item.get('answer', '')
        source = item.get('source', {})
        
        tasks.append({
            'task': task_name,
            'branch': branch,
            'llm_values': llm_values,
            'human_values': human_values,
            'scorer_scores': scorer_scores,
            'step_score': step_score_parsed,
            'answer': answer,
            'source': source
        })
    
    return tasks

# ─── Step 2: Compute Statistics ──────────────────────────────────────────────
def compute_statistics(tasks):
    """Compute aggregate statistics across all tasks."""
    scorers = ['Haining', 'Will', 'Yasaman']
    dimensions = ['in_paper', 'prompt_quality', 'follow_instructions', 
                  'physics_logic', 'math_derivation', 'final_answer_accuracy']
    
    # Per-task statistics
    task_stats = []
    for t in tasks:
        name = t['task']
        ss = t['step_score']
        
        # Average score per evaluator across dimensions
        eval_avgs = {}
        for scorer in scorers:
            vals = []
            for dim in dimensions:
                v = ss.get(dim)
                if v is not None:
                    vals.append(v)
            eval_avgs[scorer] = np.mean(vals) if vals else None
        
        # Placeholder scores: average per evaluator
        ph_scores = {}
        for scorer in scorers:
            all_vals = []
            for pk, sc_dict in t['scorer_scores'].items():
                v = sc_dict.get(scorer)
                if v is not None:
                    all_vals.append(v)
            ph_scores[scorer] = np.mean(all_vals) if all_vals else None
        
        task_stats.append({
            'task': name,
            'step_score': ss,
            'evaluator_avg': eval_avgs,
            'placeholder_avg': ph_scores,
            'n_placeholders': len(t['scorer_scores']),
            'n_scored_placeholders': sum(
                1 for sc_dict in t['scorer_scores'].values()
                for v in sc_dict.values() if v is not None
            )
        })
    
    # Overall statistics
    overall = {}
    for scorer in scorers:
        all_step_vals = []
        all_ph_vals = []
        for ts in task_stats:
            if ts['evaluator_avg'][scorer] is not None:
                all_step_vals.append(ts['evaluator_avg'][scorer])
            if ts['placeholder_avg'][scorer] is not None:
                all_ph_vals.append(ts['placeholder_avg'][scorer])
        
        overall[scorer] = {
            'avg_step_score': np.mean(all_step_vals) if all_step_vals else None,
            'std_step_score': np.std(all_step_vals) if all_step_vals else None,
            'avg_placeholder_score': np.mean(all_ph_vals) if all_ph_vals else None,
            'std_placeholder_score': np.std(all_ph_vals) if all_ph_vals else None,
            'n_tasks_evaluated': len(all_step_vals)
        }
    
    # Dimension-level statistics
    dim_stats = {}
    for dim in dimensions:
        dim_vals = {}
        for scorer in scorers:
            vals = []
            for ts in task_stats:
                v = ts['step_score'].get(dim)
                if v is not None:
                    vals.append(v)
            dim_vals[scorer] = {
                'mean': np.mean(vals) if vals else None,
                'std': np.std(vals) if vals else None,
                'min': min(vals) if vals else None,
                'max': max(vals) if vals else None,
                'count': len(vals)
            }
        dim_stats[dim] = dim_vals
    
    return {
        'task_stats': task_stats,
        'overall': overall,
        'dimension_stats': dim_stats,
        'scorers': scorers,
        'dimensions': dimensions
    }

# ─── Step 3: Extract Paper Information ───────────────────────────────────────
def extract_paper_info():
    """Extract key parameters and information from the paper."""
    paper_info = {
        "title": "Topological Phases in AB-Stacked MoTe2/WSe2",
        "authors": ["Haining Pan", "Ming Xie", "Fengcheng Wu", "Sankar Das Sarma"],
        "arxiv_id": "2111.01152",
        "system": "AB-stacked MoTe2/WSe2 moiré heterobilayer",
        "lattice_constants": {
            "a_b_MoTe2_A": 3.575,
            "a_t_WSe2_A": 3.32
        },
        "moire_period_nm": 4.7,
        "effective_masses": {
            "m_bottom_me": 0.65,
            "m_top_me": 0.35
        },
        "key_parameters": {
            "w_tunneling_meV": 12,
            "Vb_meV": 7,
            "Vzt_meV": -20,
            "psi_b_deg": -14,
            "gate_distance_d_nm": 5,
            "dielectric_constant_epsilon": [10, 15, 20],
            "bandwidth_meV": 47,
            "coulomb_scale_meV": 31
        },
        "filling_factors_studied": [1, 2, "2/3"],
        "phases": {
            "nu_2": "Z2 topological insulator",
            "nu_1": ["spin density wave", "in-plane ferromagnetic", "valley polarized (Chern insulator)"],
            "nu_2_3": "topological charge density wave"
        },
        "interaction_model": "dual-gate screened Coulomb: V(q) = 2*pi*e^2*tanh(q*d)/(epsilon*q)",
        "method": "self-consistent Hartree-Fock approximation in plane-wave basis",
        "symmetry_group": "C3v point group",
        "valleys": "+K and -K (tau = +/-1)",
        "kappa_vector": "4*pi/(3*a_M) * (1, 0)"
    }
    return paper_info

# ─── Step 4: Generate Figures ────────────────────────────────────────────────
def plot_score_distribution(stats, tasks):
    """Figure 1: Score distribution by task for each evaluator."""
    scorers = stats['scorers']
    task_names = [ts['task'] for ts in stats['task_stats']]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(task_names))
    width = 0.25
    
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    for i, scorer in enumerate(scorers):
        vals = []
        for ts in stats['task_stats']:
            v = ts['evaluator_avg'].get(scorer)
            vals.append(v if v is not None else 0)
        
        bars = ax.bar(x + i*width, vals, width, label=scorer, color=colors[i], alpha=0.85)
        
        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)
    
    ax.set_xlabel('Calculation Step', fontsize=11)
    ax.set_ylabel('Average Score (0-2)', fontsize=11)
    ax.set_title('LLM Performance: Average Step Scores by Evaluator\n(Hartree-Fock Multi-Step Derivation)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([t.split('(')[0].strip()[:20] for t in task_names], 
                       rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 2.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'score_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: score_distribution.png")

def plot_score_heatmap(stats):
    """Figure 2: Heatmap of scores by evaluator and scoring dimension."""
    scorers = stats['scorers']
    dimensions = stats['dimensions']
    
    # Build matrix: rows=dimensions, cols=scorers
    matrix = np.zeros((len(dimensions), len(scorers)))
    
    for j, scorer in enumerate(scorers):
        for i, dim in enumerate(dimensions):
            ds = stats['dimension_stats'][dim][scorer]
            matrix[i, j] = ds['mean'] if ds['mean'] is not None else 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')
    
    ax.set_xticks(np.arange(len(scorers)))
    ax.set_xticklabels(scorers, fontsize=12)
    ax.set_yticks(np.arange(len(dimensions)))
    ax.set_yticklabels([d.replace('_', ' ').title() for d in dimensions], fontsize=9)
    
    # Add text annotations
    for i in range(len(dimensions)):
        for j in range(len(scorers)):
            text = ax.text(j, i, f'{matrix[i,j]:.2f}',
                          ha='center', va='center', fontsize=11,
                          color='black' if matrix[i,j] > 1 else 'white')
    
    ax.set_xlabel('Evaluator', fontsize=12)
    ax.set_ylabel('Scoring Dimension', fontsize=12)
    ax.set_title('Mean Scores by Evaluator and Dimension\n(Across All 16 HF Calculation Steps)', fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Score (0-2 scale)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'score_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: score_heatmap.png")

def plot_placeholder_analysis(tasks):
    """Figure 3: Placeholder quality vs LLM answer accuracy correlation."""
    scorers = ['Haining', 'Will', 'Yasaman']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for idx, scorer in enumerate(scorers):
        ax = axes[idx]
        
        ph_scores = []
        step_scores = []
        task_labels = []
        
        for t in tasks:
            ss = t['step_score']
            avg_step = np.mean([v for v in ss.values() if v is not None]) if ss else 0
            
            ph_vals = []
            for pk, sc_dict in t['scorer_scores'].items():
                v = sc_dict.get(scorer)
                if v is not None:
                    ph_vals.append(v)
            avg_ph = np.mean(ph_vals) if ph_vals else 0
            
            ph_scores.append(avg_ph)
            step_scores.append(avg_step)
            task_labels.append(t['task'][:30])
        
        ax.scatter(ph_scores, step_scores, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Linear fit
        if len(ph_scores) > 1:
            z = np.polyfit(ph_scores, step_scores, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(ph_scores), max(ph_scores), 50)
            ax.plot(x_line, p(x_line), '--', alpha=0.5, color='red')
            corr = np.corrcoef(ph_scores, step_scores)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Avg Placeholder Score', fontsize=10)
        ax.set_xlim(-0.1, 2.1)
        ax.set_title(f'{scorer}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
    
    axes[0].set_ylabel('Avg Step Score', fontsize=10)
    fig.suptitle('Placeholder Quality vs. Final Step Accuracy\n(Correlation Analysis)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'placeholder_accuracy.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: placeholder_accuracy.png")

def plot_performance_summary(stats):
    """Figure 4: Overall performance summary with radar/spider chart."""
    scorers = stats['scorers']
    dimensions = stats['dimensions']
    
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(projection='polar'))
    
    num_vars = len(dimensions)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop
    
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    for i, scorer in enumerate(scorers):
        values = []
        for dim in dimensions:
            ds = stats['dimension_stats'][dim][scorer]
            values.append(ds['mean'] if ds['mean'] is not None else 0)
        values += values[:1]  # close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=scorer, color=colors[i], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace('_', '\n').title() for d in dimensions], fontsize=9)
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(['0', '0.5', '1', '1.5', '2'], fontsize=8)
    ax.set_title('Overall LLM Performance Profile\n(Hartree-Fock Multi-Step Derivation)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'performance_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: performance_summary.png")

def plot_step_detail_comparison(stats):
    """Figure 5: Detailed step-by-step comparison across all dimensions."""
    scorers = stats['scorers']
    task_stats = stats['task_stats']
    task_names = [ts['task'] for ts in task_stats]
    short_names = [t.split('(')[0].strip()[:18] for t in task_names]
    
    dimensions = stats['dimensions']
    n_steps = len(task_names)
    n_dims = len(dimensions)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    
    for d_idx, dim in enumerate(dimensions):
        ax = axes[d_idx]
        x = np.arange(n_steps)
        width = 0.25
        
        for s_idx, scorer in enumerate(scorers):
            vals = []
            for ts in task_stats:
                v = ts['step_score'].get(dim)
                vals.append(v if v is not None else 0)
            
            ax.bar(x + s_idx*width, vals, width, label=scorer, color=colors[s_idx], alpha=0.8)
        
        ax.set_title(dim.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)
        ax.set_ylim(0, 2.3)
        ax.grid(axis='y', alpha=0.3)
        if d_idx == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplot
    axes[-1].axis('off')
    
    fig.suptitle('Step-by-Step Scores by Scoring Dimension', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'step_detail_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: step_detail_comparison.png")

def plot_placeholder_score_distribution(tasks):
    """Figure 6: Distribution of placeholder scores across all tasks."""
    scorers = ['Haining', 'Will', 'Yasaman']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_scores_by_scorer = {s: [] for s in scorers}
    
    for t in tasks:
        for pk, sc_dict in t['scorer_scores'].items():
            for scorer in scorers:
                v = sc_dict.get(scorer)
                if v is not None:
                    all_scores_by_scorer[scorer].append(v)
    
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bins = np.arange(-0.25, 2.25, 0.5)
    
    for i, scorer in enumerate(scorers):
        vals = all_scores_by_scorer[scorer]
        ax.hist(vals, bins=bins, alpha=0.6, label=f'{scorer} (n={len(vals)})', 
               color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Placeholder Score (0-2)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Placeholder Scores Across All Tasks', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_IMAGES_DIR, 'placeholder_score_dist.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: placeholder_score_dist.png")

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Hartree-Fock Multi-Step Calculation Analysis")
    print("Paper: 2111.01152 (Pan et al.)")
    print("=" * 60)
    
    # Parse data
    print("\n[1] Parsing YAML data...")
    tasks = parse_yaml()
    print(f"    Found {len(tasks)} calculation steps")
    
    # Compute statistics
    print("\n[2] Computing statistics...")
    stats = compute_statistics(tasks)
    
    # Save parsed data
    parsed_output = {
        'n_tasks': len(tasks),
        'tasks': [
            {
                'task': t['task'],
                'step_score': t['step_score'],
                'n_placeholders': len(t['scorer_scores'])
            }
            for t in tasks
        ],
        'overall': stats['overall'],
        'dimension_stats': {
            dim: {
                scorer: {k: v for k, v in sd.items() if k != 'count'}
                for scorer, sd in sdd.items()
            }
            for dim, sdd in stats['dimension_stats'].items()
        }
    }
    
    with open(os.path.join(OUTPUTS_DIR, 'scores_parsed.json'), 'w') as f:
        json.dump(parsed_output, f, indent=2)
    print("    Saved: outputs/scores_parsed.json")
    
    with open(os.path.join(OUTPUTS_DIR, 'aggregate_stats.json'), 'w') as f:
        json.dump({
            'overall': stats['overall'],
            'dimension_stats': stats['dimension_stats'],
            'task_stats': [
                {
                    'task': ts['task'],
                    'evaluator_avg': ts['evaluator_avg'],
                    'placeholder_avg': ts['placeholder_avg']
                }
                for ts in stats['task_stats']
            ]
        }, f, indent=2, default=str)
    print("    Saved: outputs/aggregate_stats.json")
    
    # Extract paper info
    print("\n[3] Extracting paper information...")
    paper_info = extract_paper_info()
    with open(os.path.join(OUTPUTS_DIR, 'paper_info.json'), 'w') as f:
        json.dump(paper_info, f, indent=2)
    print("    Saved: outputs/paper_info.json")
    
    # Generate figures
    print("\n[4] Generating figures...")
    plot_score_distribution(stats, tasks)
    plot_score_heatmap(stats)
    plot_placeholder_analysis(tasks)
    plot_performance_summary(stats)
    plot_step_detail_comparison(stats)
    plot_placeholder_score_distribution(tasks)
    
    # Print summary
    print("\n[5] Summary Statistics:")
    print("-" * 50)
    for scorer in stats['scorers']:
        o = stats['overall'][scorer]
        print(f"  {scorer}:")
        print(f"    Avg step score:     {o['avg_step_score']:.3f} ± {o['std_step_score']:.3f}")
        print(f"    Avg placeholder:    {o['avg_placeholder_score']:.3f} ± {o['std_placeholder_score']:.3f}")
        print(f"    Tasks evaluated:    {o['n_tasks_evaluated']}")
    
    print("\n  Dimension-level means:")
    for dim in stats['dimensions']:
        vals = []
        for scorer in stats['scorers']:
            ds = stats['dimension_stats'][dim][scorer]
            if ds['mean'] is not None:
                vals.append(ds['mean'])
        if vals:
            print(f"    {dim:25s}: {np.mean(vals):.3f} (range: {min(vals):.2f}-{max(vals):.2f})")
    
    print("\n" + "=" * 60)
    print("Analysis complete. All artifacts saved.")
    print("=" * 60)

if __name__ == '__main__':
    main()
