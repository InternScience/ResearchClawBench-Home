#!/usr/bin/env python3
"""
Analysis of LLM Performance on Hartree-Fock Calculations
for AB-stacked MoTe2/WSe2 moiré system (paper 2111.01152)
"""

import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from collections import defaultdict

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)
os.makedirs('code', exist_ok=True)

# Load data
with open('data/2111.01152/2111.01152.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Extract tasks
tasks = [item for item in data if isinstance(item, dict) and 'task' in item]
print(f"Loaded {len(tasks)} tasks")

# Parse scores
def extract_scores(tasks):
    """Extract all score information from tasks."""
    results = []
    for i, task in enumerate(tasks):
        task_name = task['task']
        scores = task.get('score', {})
        answer = task.get('answer', '')
        placeholders = task.get('placeholder', {})
        placeholder_scores = {}
        for key, value in placeholders.items():
            if isinstance(value, dict) and 'score' in value:
                placeholder_scores[key] = value['score']
        results.append({
            'task_id': i + 1,
            'task_name': task_name,
            'answer': str(answer) if answer else '',
            'scores': scores,
            'placeholder_scores': placeholder_scores,
            'placeholders': placeholders
        })
    return results

results = extract_scores(tasks)
with open('outputs/extracted_data.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Extracted data for {len(results)} tasks")

# Calculate overall scores
def calculate_overall_scores(results):
    metrics = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 
               'math_derivation', 'final_answer_accuracy']
    task_scores = {}
    for r in results:
        scores = r['scores']
        vals = [float(scores.get(m, 0)) for m in metrics]
        avg_score = np.mean(vals)
        task_scores[r['task_id']] = {'avg': float(avg_score), 'metrics': scores}
    return task_scores, metrics

task_scores, metrics = calculate_overall_scores(results)
with open('outputs/task_scores.json', 'w') as f:
    json.dump(task_scores, f, indent=2, default=str)
print("Task scores calculated")

# Calculate evaluator scores from placeholders
def calculate_evaluator_scores(results):
    evaluators = ['Haining', 'Will', 'Yasaman']
    evaluator_totals = {e: [] for e in evaluators}
    for r in results:
        for key, scores in r['placeholder_scores'].items():
            for e in evaluators:
                if e in scores:
                    val = scores[e]
                    if val not in ['(?)', '?', None, '']:
                        try:
                            evaluator_totals[e].append(float(val))
                        except (ValueError, TypeError):
                            pass
    return {e: {
        'mean': float(np.mean(scores)) if scores else 0,
        'std': float(np.std(scores)) if scores else 0,
        'count': len(scores)
    } for e, scores in evaluator_totals.items()}

evaluator_stats = calculate_evaluator_scores(results)
print("Evaluator scores calculated")

# Calculate placeholder accuracy
def calculate_placeholder_accuracy(results):
    accuracy_data = []
    for r in results:
        for key, value in r['placeholders'].items():
            if isinstance(value, dict):
                llm_answer = str(value.get('LLM', ''))
                human_answer = str(value.get('human', ''))
                scores = value.get('score', {})
                if llm_answer and llm_answer != 'None' and human_answer and human_answer != 'None':
                    match = 'Match' if llm_answer.lower().strip() == human_answer.lower().strip() else 'Different'
                elif llm_answer and llm_answer != 'None' and (not human_answer or human_answer == 'None'):
                    match = 'LLM only'
                elif (not llm_answer or llm_answer == 'None') and human_answer and human_answer != 'None':
                    match = 'Human only'
                else:
                    match = 'Empty'
                valid_scores = []
                for e in ['Haining', 'Will', 'Yasaman']:
                    if e in scores and scores[e] not in ['(?)', '?', None, '']:
                        try:
                            valid_scores.append(float(scores[e]))
                        except (ValueError, TypeError):
                            pass
                avg_evaluator = float(np.mean(valid_scores)) if valid_scores else 0
                accuracy_data.append({
                    'task_id': r['task_id'],
                    'placeholder': key,
                    'llm_answer': llm_answer[:100] if llm_answer else '',
                    'human_answer': human_answer[:100] if human_answer else '',
                    'match_status': match,
                    'avg_evaluator_score': avg_evaluator,
                    'scores': scores
                })
    return accuracy_data

placeholder_accuracy = calculate_placeholder_accuracy(results)
print(f"Placeholder accuracy calculated for {len(placeholder_accuracy)} items")

with open('outputs/placeholder_accuracy.json', 'w') as f:
    json.dump(placeholder_accuracy, f, indent=2, default=str)

# =============================================================================
# FIGURES
# =============================================================================
task_ids = [r['task_id'] for r in results]
avg_scores_list = [task_scores[tid]['avg'] for tid in task_ids]

# FIGURE 1: Overall Task Scores
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(task_ids, avg_scores_list, color='steelblue', alpha=0.8)
ax.set_xlabel('Task ID', fontsize=12)
ax.set_ylabel('Average Score (0-2)', fontsize=12)
ax.set_title('Overall Task Scores for Hartree-Fock Calculation Steps', fontsize=14)
ax.set_xticks(task_ids)
ax.set_ylim(0, 2.5)
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
ax.legend()
for bar, score in zip(bars, avg_scores_list):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{score:.2f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('report/images/figure1_task_scores.png', dpi=150)
plt.close()
print("Figure 1 saved")

# FIGURE 2: Score Breakdown by Metric
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(task_ids))
width = 0.15
multiplier = 0
for i, metric in enumerate(metrics):
    scores = [task_scores[tid]['metrics'].get(metric, 0) for tid in task_ids]
    offset = width * multiplier
    ax.bar(x + offset, scores, width, label=metric.replace('_', ' ').title())
    multiplier += 1
ax.set_xlabel('Task ID', fontsize=12)
ax.set_ylabel('Score (0-2)', fontsize=12)
ax.set_title('Score Breakdown by Metric for Each Task', fontsize=14)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(task_ids)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
ax.set_ylim(0, 2.5)
plt.tight_layout()
plt.savefig('report/images/figure2_score_breakdown.png', dpi=150)
plt.close()
print("Figure 2 saved")

# FIGURE 3: Evaluator Comparison
fig, ax = plt.subplots(figsize=(10, 6))
evaluators = ['Haining', 'Will', 'Yasaman']
means = [evaluator_stats[e]['mean'] for e in evaluators]
stds = [evaluator_stats[e]['std'] for e in evaluators]
bars = ax.bar(evaluators, means, yerr=stds, capsize=5, 
              color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
ax.set_xlabel('Evaluator', fontsize=12)
ax.set_ylabel('Average Score (0-2)', fontsize=12)
ax.set_title('Evaluator Comparison: Average Scores on Placeholders', fontsize=14)
ax.set_ylim(0, 2.5)
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.05,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/figure3_evaluator_comparison.png', dpi=150)
plt.close()
print("Figure 3 saved")

# FIGURE 4: Placeholder Accuracy Distribution
fig, ax = plt.subplots(figsize=(10, 6))
status_counts = defaultdict(int)
for item in placeholder_accuracy:
    status_counts[item['match_status']] += 1
statuses = list(status_counts.keys())
counts = [status_counts[s] for s in statuses]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
ax.pie(counts, labels=statuses, autopct='%1.1f%%', colors=colors[:len(statuses)], startangle=90)
ax.set_title('Distribution of LLM vs Human Answer Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('report/images/figure4_placeholder_accuracy.png', dpi=150)
plt.close()
print("Figure 4 saved")

# FIGURE 5: Task Difficulty Analysis
fig, ax = plt.subplots(figsize=(10, 6))
difficulty_categories = {'Easy (>=1.5)': [], 'Medium (1.0-1.5)': [], 'Hard (<1.0)': []}
for r in results:
    avg = task_scores[r['task_id']]['avg']
    if avg >= 1.5:
        difficulty_categories['Easy (>=1.5)'].append(r['task_id'])
    elif avg >= 1.0:
        difficulty_categories['Medium (1.0-1.5)'].append(r['task_id'])
    else:
        difficulty_categories['Hard (<1.0)'].append(r['task_id'])
categories = list(difficulty_categories.keys())
counts = [len(difficulty_categories[c]) for c in categories]
colors = ['#4CAF50', '#FFC107', '#F44336']
bars = ax.bar(categories, counts, color=colors, alpha=0.8)
ax.set_xlabel('Difficulty Category', fontsize=12)
ax.set_ylabel('Number of Tasks', fontsize=12)
ax.set_title('Task Difficulty Distribution', fontsize=14)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            str(count), ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig('report/images/figure5_difficulty_analysis.png', dpi=150)
plt.close()
print("Figure 5 saved")

# FIGURE 6: Physics vs Math Performance
fig, ax = plt.subplots(figsize=(10, 6))
physics_scores = [task_scores[tid]['metrics'].get('physics_logic', 0) for tid in task_ids]
math_scores = [task_scores[tid]['metrics'].get('math_derivation', 0) for tid in task_ids]
ax.scatter(physics_scores, math_scores, s=100, c='steelblue', alpha=0.7, edgecolors='navy')
ax.set_xlabel('Physics Logic Score (0-2)', fontsize=12)
ax.set_ylabel('Math Derivation Score (0-2)', fontsize=12)
ax.set_title('Physics Logic vs Math Derivation Performance', fontsize=14)
for i, tid in enumerate(task_ids):
    ax.annotate(f'T{tid}', (physics_scores[i], math_scores[i]), 
                textcoords="offset points", xytext=(5,5), fontsize=8)
ax.set_xlim(-0.2, 2.5)
ax.set_ylim(-0.2, 2.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure6_physics_vs_math.png', dpi=150)
plt.close()
print("Figure 6 saved")

# FIGURE 7: Step-by-Step Flow Performance
fig, ax = plt.subplots(figsize=(14, 6))
cumulative_scores = np.cumsum(avg_scores_list)
ax.plot(task_ids, avg_scores_list, 'o-', color='steelblue', linewidth=2, markersize=8, label='Individual Score')
ax.plot(task_ids, cumulative_scores, 's--', color='coral', linewidth=2, markersize=8, label='Cumulative Score')
ax.set_xlabel('Task ID (Step in Derivation)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Step-by-Step Performance in Hartree-Fock Derivation', fontsize=14)
ax.set_xticks(task_ids)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure7_step_flow.png', dpi=150)
plt.close()
print("Figure 7 saved")

# FIGURE 8: Final Answer Accuracy vs Follow Instructions
fig, ax = plt.subplots(figsize=(10, 6))
follow_scores = [task_scores[tid]['metrics'].get('follow_instructions', 0) for tid in task_ids]
final_scores = [task_scores[tid]['metrics'].get('final_answer_accuracy', 0) for tid in task_ids]
ax.scatter(follow_scores, final_scores, s=100, c='darkgreen', alpha=0.7, edgecolors='black')
ax.set_xlabel('Follow Instructions Score (0-2)', fontsize=12)
ax.set_ylabel('Final Answer Accuracy Score (0-2)', fontsize=12)
ax.set_title('Follow Instructions vs Final Answer Accuracy', fontsize=14)
for i, tid in enumerate(task_ids):
    ax.annotate(f'T{tid}', (follow_scores[i], final_scores[i]), 
                textcoords="offset points", xytext=(5,5), fontsize=8)
ax.set_xlim(-0.2, 2.5)
ax.set_ylim(-0.2, 2.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure8_instructions_vs_accuracy.png', dpi=150)
plt.close()
print("Figure 8 saved")

# =============================================================================
# Summary Statistics
# =============================================================================
summary = {
    'paper': '2111.01152',
    'system': 'AB-stacked MoTe2/WSe2 moiré system',
    'total_tasks': len(tasks),
    'total_placeholders': len(placeholder_accuracy),
    'overall_avg_score': float(np.mean(avg_scores_list)),
    'evaluator_stats': evaluator_stats,
    'difficulty_distribution': {k: len(v) for k, v in difficulty_categories.items()},
    'task_details': []
}
for r in results:
    summary['task_details'].append({
        'task_id': r['task_id'],
        'task_name': r['task_name'],
        'avg_score': task_scores[r['task_id']]['avg'],
        'scores': r['scores']
    })
with open('outputs/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nSummary Statistics:")
print(f"  Paper: {summary['paper']}")
print(f"  System: {summary['system']}")
print(f"  Total Tasks: {summary['total_tasks']}")
print(f"  Total Placeholders: {summary['total_placeholders']}")
print(f"  Overall Average Score: {summary['overall_avg_score']:.2f}")
print(f"  Difficulty Distribution: {summary['difficulty_distribution']}")
print(f"\nEvaluator Stats:")
for e, stats in evaluator_stats.items():
    print(f"  {e}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, count={stats['count']}")
print("\nAll analysis files and figures generated successfully!")
