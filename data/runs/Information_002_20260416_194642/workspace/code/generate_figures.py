#!/usr/bin/env python3
"""
Generate figures for the Hartree-Fock LLM Evaluation report
"""

import yaml
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Parse YAML
yaml_path = 'data/2111.01152/2111.01152.yaml'
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

tasks = [entry for entry in data if 'task' in entry]

# Create output directories
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ============================================================
# Figure 1: Task-level scores heatmap
# ============================================================
score_categories = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']
cat_labels = ['In Paper', 'Prompt\nQuality', 'Follow\nInstructions', 'Physics\nLogic', 'Math\nDerivation', 'Final Answer\nAccuracy']

task_names_short = [
    '1. Kinetic Ham.',
    '2. Define Kinetic',
    '3. Potential Ham.',
    '4. Define Potential',
    '5. 2nd Quant. (matrix)',
    '6. 2nd Quant. (expand)',
    '7. Real→Momentum',
    '8. Particle-hole',
    '9. Simplify PH basis',
    '10. Interaction Ham.',
    '11. Wick\'s theorem',
    '12. Extract quadratic',
    '13. Swap indices',
    '14. Reduce Hartree',
    '15. Reduce Fock',
    '16. Combine H+F'
]

score_matrix = np.full((len(tasks), len(score_categories)), np.nan)
for i, task in enumerate(tasks):
    if 'score' in task:
        for j, cat in enumerate(score_categories):
            if cat in task['score']:
                score_matrix[i, j] = task['score'][cat]

fig, ax = plt.subplots(figsize=(12, 10))
cmap = plt.cm.RdYlGn
im = ax.imshow(score_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)

ax.set_xticks(range(len(cat_labels)))
ax.set_xticklabels(cat_labels, fontsize=10, ha='center')
ax.set_yticks(range(len(task_names_short)))
ax.set_yticklabels(task_names_short, fontsize=9)

# Add text annotations
for i in range(score_matrix.shape[0]):
    for j in range(score_matrix.shape[1]):
        val = score_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if val < 1 else 'black'
            ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, label='Score (0=Incorrect, 1=Partial, 2=Correct)', shrink=0.8)
ax.set_title('Task-Level Scores for LLM Hartree-Fock Calculation Steps\n(Paper 2111.01152: AB-stacked MoTe₂/WSe₂)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/task_scores_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: task_scores_heatmap.png")

# ============================================================
# Figure 2: Placeholder extraction scores by grader
# ============================================================
placeholder_data = []
for i, task in enumerate(tasks):
    if 'placeholder' in task:
        for ph_name, ph_val in task['placeholder'].items():
            if isinstance(ph_val, dict) and 'score' in ph_val:
                scores = ph_val['score']
                grader_scores = {}
                for grader in ['Haining', 'Will', 'Yasaman']:
                    if grader in scores:
                        val = scores[grader]
                        if isinstance(val, (int, float)):
                            grader_scores[grader] = val
                placeholder_data.append({
                    'task_index': i+1,
                    'task_name': task['task'],
                    'placeholder': ph_name,
                    'llm_value': ph_val.get('LLM', None),
                    'human_value': ph_val.get('human', None),
                    'grader_scores': grader_scores,
                })

# Score distribution by grader
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
graders = ['Haining', 'Will', 'Yasaman']
colors_bar = ['#e74c3c', '#f39c12', '#27ae60']

for idx, grader in enumerate(graders):
    vals = [pd['grader_scores'][grader] for pd in placeholder_data if grader in pd['grader_scores']]
    counts = [vals.count(0), vals.count(1), vals.count(2)]
    total = len(vals)
    pcts = [100*c/total for c in counts]
    
    bars = axes[idx].bar([0, 1, 2], counts, color=colors_bar, edgecolor='black', linewidth=0.5)
    axes[idx].set_xlabel('Score', fontsize=12)
    axes[idx].set_ylabel('Count', fontsize=12)
    axes[idx].set_title(f'Grader: {grader}\n(n={total}, mean={np.mean(vals):.2f})', fontsize=11)
    axes[idx].set_xticks([0, 1, 2])
    axes[idx].set_xticklabels(['0\n(Wrong)', '1\n(Partial)', '2\n(Correct)'])
    
    for bar, count, pct in zip(bars, counts, pcts):
        axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                      f'{count}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=10)

plt.suptitle('LLM Information Extraction Scores by Grader\n(Placeholder-Level Evaluation)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/grader_score_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: grader_score_distribution.png")

# ============================================================
# Figure 3: Per-task average placeholder scores
# ============================================================
task_ph_scores = defaultdict(list)
for pd in placeholder_data:
    grader_vals = list(pd['grader_scores'].values())
    if grader_vals:
        task_ph_scores[pd['task_index']].append(np.mean(grader_vals))

fig, ax = plt.subplots(figsize=(14, 6))
task_indices = sorted(task_ph_scores.keys())
means = [np.mean(task_ph_scores[ti]) for ti in task_indices]
stds = [np.std(task_ph_scores[ti]) for ti in task_indices]
labels = [task_names_short[ti-1] for ti in task_indices]

bars = ax.bar(range(len(task_indices)), means, yerr=stds, capsize=3, 
              color=['#3498db' if m >= 1.5 else '#e74c3c' if m < 1.0 else '#f39c12' for m in means],
              edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(range(len(task_indices)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean Placeholder Score (0-2)', fontsize=11)
ax.set_title('Average LLM Extraction Score per Calculation Step\n(Mean ± Std across placeholders and graders)', fontsize=12, fontweight='bold')
ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Perfect (2.0)')
ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Partial (1.0)')
ax.set_ylim(0, 2.5)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('report/images/per_task_placeholder_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: per_task_placeholder_scores.png")

# ============================================================
# Figure 4: Score category averages (radar/bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
cat_means = []
cat_stds = []
for cat in score_categories:
    vals = [score_matrix[i, j] for i in range(score_matrix.shape[0]) 
            for j, c in enumerate(score_categories) if c == cat and not np.isnan(score_matrix[i, j])]
    cat_means.append(np.mean(vals))
    cat_stds.append(np.std(vals))

colors_cat = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#2ecc71']
bars = ax.bar(range(len(score_categories)), cat_means, yerr=cat_stds, capsize=5,
              color=colors_cat, edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(range(len(score_categories)))
ax.set_xticklabels(['In Paper', 'Prompt\nQuality', 'Follow\nInstructions', 'Physics\nLogic', 'Math\nDerivation', 'Final Answer\nAccuracy'], fontsize=10)
ax.set_ylabel('Mean Score (0-2)', fontsize=11)
ax.set_title('Average Scores by Evaluation Category\n(Across 16 HF Calculation Steps)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 2.5)

for bar, mean, std in zip(bars, cat_means, cat_stds):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.05,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/score_categories_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: score_categories_bar.png")

# ============================================================
# Figure 5: Workflow diagram - HF calculation steps pipeline
# ============================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis('off')

# Define step groups
groups = [
    ("Hamiltonian Construction", ['1. Kinetic Ham.', '2. Define Kinetic', '3. Potential Ham.', '4. Define Potential'], '#3498db'),
    ("Second Quantization", ['5. 2nd Quant. (matrix)', '6. 2nd Quant. (expand)', '7. Real→Momentum'], '#2ecc71'),
    ("Particle-Hole Basis", ['8. Particle-hole', '9. Simplify PH basis'], '#9b59b6'),
    ("Interaction & HF", ['10. Interaction Ham.', '11. Wick\'s theorem', '12. Extract quadratic'], '#e74c3c'),
    ("Momentum Reduction", ['13. Swap indices', '14. Reduce Hartree', '15. Reduce Fock', '16. Combine H+F'], '#f39c12'),
]

# Get final answer scores for coloring
final_scores = {}
for i, task in enumerate(tasks):
    if 'score' in task and 'final_answer_accuracy' in task['score']:
        final_scores[i+1] = task['score']['final_answer_accuracy']

y_pos = 17
for group_name, steps, color in groups:
    # Group header
    ax.text(5, y_pos, group_name, fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3, edgecolor=color, linewidth=2))
    y_pos -= 0.8
    
    for step in steps:
        step_num = int(step.split('.')[0])
        score = final_scores.get(step_num, None)
        
        if score == 2:
            face_color = '#27ae60'
            score_text = '✓'
        elif score == 1:
            face_color = '#f39c12'
            score_text = '~'
        else:
            face_color = '#e74c3c'
            score_text = '✗'
        
        ax.text(5, y_pos, f'{step}  [{score_text}]', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=face_color, alpha=0.2, edgecolor='gray'))
        y_pos -= 0.7
    
    y_pos -= 0.3
    if group_name != groups[-1][0]:
        ax.annotate('', xy=(5, y_pos+0.2), xytext=(5, y_pos+0.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2))

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#27ae60', alpha=0.3, label='Score 2 (Correct)'),
    mpatches.Patch(facecolor='#f39c12', alpha=0.3, label='Score 1 (Partial)'),
    mpatches.Patch(facecolor='#e74c3c', alpha=0.3, label='Score 0 (Incorrect)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_title('Hartree-Fock Calculation Pipeline: LLM Performance on Each Step\n(Final Answer Accuracy)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/hf_pipeline_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: hf_pipeline_performance.png")

# ============================================================
# Figure 6: Inter-grader agreement analysis
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Compute agreement matrix
agreement_data = []
for pd in placeholder_data:
    gs = pd['grader_scores']
    if len(gs) >= 2:
        vals = list(gs.values())
        agreement_data.append({
            'range': max(vals) - min(vals),
            'all_agree': len(set(vals)) == 1,
            'scores': vals
        })

ranges = [a['range'] for a in agreement_data]
agree_count = sum(1 for a in agreement_data if a['all_agree'])
total = len(agreement_data)

# Plot range distribution
range_counts = {0: ranges.count(0), 1: ranges.count(1), 2: ranges.count(2)}
bars = ax.bar(range_counts.keys(), range_counts.values(), 
              color=['#27ae60', '#f39c12', '#e74c3c'], edgecolor='black', linewidth=0.5, width=0.6)

for bar, (r, c) in zip(bars, range_counts.items()):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{c}\n({100*c/total:.0f}%)', ha='center', va='bottom', fontsize=11)

ax.set_xlabel('Score Range Among Graders', fontsize=12)
ax.set_ylabel('Number of Placeholders', fontsize=12)
ax.set_title(f'Inter-Grader Agreement Analysis\n(Full agreement: {agree_count}/{total} = {100*agree_count/total:.0f}%)', 
             fontsize=12, fontweight='bold')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['0 (Full Agreement)', '1 (Minor Disagreement)', '2 (Major Disagreement)'])
plt.tight_layout()
plt.savefig('report/images/intergrader_agreement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: intergrader_agreement.png")

# ============================================================
# Figure 7: Comparison of LLM vs Human answers - error analysis
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Categorize errors
error_types = defaultdict(int)
for pd in placeholder_data:
    mean_score = np.mean(list(pd['grader_scores'].values())) if pd['grader_scores'] else None
    if mean_score is not None:
        if mean_score == 2.0:
            error_types['Correct (2.0)'] += 1
        elif mean_score >= 1.5:
            error_types['Mostly Correct (1.5-2.0)'] += 1
        elif mean_score >= 1.0:
            error_types['Partial (1.0-1.5)'] += 1
        elif mean_score > 0:
            error_types['Mostly Wrong (0-1.0)'] += 1
        else:
            error_types['Completely Wrong (0)'] += 1

labels = list(error_types.keys())
sizes = list(error_types.values())
colors_pie = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
                                   startangle=90, textprops={'fontsize': 10})
for autotext in autotexts:
    autotext.set_fontweight('bold')

ax.set_title('Distribution of LLM Extraction Quality\n(Across All Placeholders)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/extraction_quality_pie.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: extraction_quality_pie.png")

# ============================================================
# Figure 8: Cumulative score progression through HF steps
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Get per-task mean placeholder score and final answer score
task_mean_ph = []
task_final_ans = []
for i, task in enumerate(tasks):
    # Placeholder scores
    ph_scores_for_task = [np.mean(list(pd['grader_scores'].values())) 
                          for pd in placeholder_data 
                          if pd['task_index'] == i+1 and pd['grader_scores']]
    task_mean_ph.append(np.mean(ph_scores_for_task) if ph_scores_for_task else np.nan)
    
    # Final answer score
    if 'score' in task and 'final_answer_accuracy' in task['score']:
        task_final_ans.append(task['score']['final_answer_accuracy'])
    else:
        task_final_ans.append(np.nan)

x = range(1, len(tasks)+1)
ax.plot(x, task_mean_ph, 'o-', color='#3498db', linewidth=2, markersize=8, label='Mean Placeholder Score')
ax.plot(x, task_final_ans, 's-', color='#e74c3c', linewidth=2, markersize=8, label='Final Answer Accuracy')

ax.set_xticks(list(x))
ax.set_xticklabels([f'S{i}' for i in x], fontsize=9)
ax.set_xlabel('Calculation Step', fontsize=11)
ax.set_ylabel('Score (0-2)', fontsize=11)
ax.set_title('LLM Performance Progression Through Hartree-Fock Calculation Steps', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(-0.1, 2.3)
ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.3)
ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/score_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: score_progression.png")

print("\nAll figures generated successfully!")
