#!/usr/bin/env python3
"""
Generate figures for the Hartree-Fock LLM evaluation report.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("outputs/task_scores.json", "r") as f:
    task_scores = json.load(f)

with open("outputs/summary_statistics.json", "r") as f:
    summary = json.load(f)

score_categories = ["in_paper", "prompt_quality", "follow_instructions", 
                    "physics_logic", "math_derivation", "final_answer_accuracy"]

cat_labels = ["In Paper", "Prompt Quality", "Follow Instructions", 
              "Physics Logic", "Math Derivation", "Final Answer Accuracy"]

# Figure 1: Bar chart of mean scores by category
fig, ax = plt.subplots(figsize=(10, 6))
means = [summary[cat]["mean"] for cat in score_categories]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
bars = ax.bar(cat_labels, means, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylim(0, 2.2)
ax.set_ylabel('Mean Score (0-2)', fontsize=12)
ax.set_title('LLM Performance Across Score Categories\n(15 Hartree-Fock Calculation Steps)', fontsize=14)
ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Perfect Score')
ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='75% Threshold')
plt.xticks(rotation=30, ha='right', fontsize=10)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('report/images/fig1_category_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Heatmap of task scores
fig, ax = plt.subplots(figsize=(14, 10))
task_names = [t["task"][:40] for t in task_scores]
score_matrix = np.array([[t[cat] for cat in score_categories] for t in task_scores])

im = ax.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
ax.set_xticks(range(len(cat_labels)))
ax.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(task_names)))
ax.set_yticklabels(task_names, fontsize=8)
ax.set_title('Score Heatmap: Task × Category', fontsize=14)

# Add text annotations
for i in range(len(task_names)):
    for j in range(len(cat_labels)):
        val = score_matrix[i, j]
        color = 'white' if val < 1 else 'black'
        ax.text(j, i, f'{int(val)}', ha='center', va='center', 
                fontsize=9, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='Score (0=incorrect, 1=partial, 2=correct)')
plt.tight_layout()
plt.savefig('report/images/fig2_score_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Task-by-task total scores
fig, ax = plt.subplots(figsize=(14, 6))
totals = [sum(t[cat] for cat in score_categories) for t in task_scores]
max_possible = len(score_categories) * 2
percentages = [t/max_possible*100 for t in totals]

colors_task = ['#2ecc71' if p >= 90 else '#f39c12' if p >= 75 else '#e74c3c' for p in percentages]
bars = ax.bar(range(1, len(totals)+1), percentages, color=colors_task, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Task Number', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Total Score per Calculation Step (12 points max)', fontsize=14)
ax.set_ylim(0, 110)
ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75%')
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')

for bar, pct, total in zip(bars, percentages, totals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{total}/{max_possible}', ha='center', va='bottom', fontsize=8)

ax.legend(loc='lower right')
ax.set_xticks(range(1, len(totals)+1))
plt.tight_layout()
plt.savefig('report/images/fig3_task_totals.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Reviewer agreement comparison
fig, ax = plt.subplots(figsize=(10, 6))
reviewer_means = summary["reviewer_means"]
x = np.arange(len(cat_labels))
width = 0.25

for idx, reviewer in enumerate(["Haining", "Will", "Yasaman"]):
    vals = [reviewer_means[cat][reviewer] for cat in score_categories]
    ax.bar(x + idx*width, vals, width, label=reviewer, alpha=0.85)

ax.set_ylabel('Mean Score', fontsize=12)
ax.set_title('Inter-Reviewer Agreement Across Categories', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(cat_labels, rotation=30, ha='right', fontsize=9)
ax.legend()
ax.set_ylim(0, 2.3)
plt.tight_layout()
plt.savefig('report/images/fig4_reviewer_agreement.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 5: Score distribution pie chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distribution of all individual scores
all_scores = []
for t in task_scores:
    for cat in score_categories:
        all_scores.append(t[cat])

score_counts = {0: all_scores.count(0), 1: all_scores.count(1), 2: all_scores.count(2)}
labels_pie = ['Incorrect (0)', 'Partial (1)', 'Correct (2)']
colors_pie = ['#e74c3c', '#f39c12', '#2ecc71']
axes[0].pie(score_counts.values(), labels=labels_pie, colors=colors_pie, 
            autopct='%1.1f%%', startangle=90)
axes[0].set_title('Distribution of All Scores\n(90 individual ratings)', fontsize=12)

# Task completion status
perfect = sum(1 for t in task_scores if sum(t[cat] for cat in score_categories) == max_possible)
partial = sum(1 for t in task_scores if 8 <= sum(t[cat] for cat in score_categories) < max_possible)
incomplete = len(task_scores) - perfect - partial
axes[1].pie([perfect, partial, incomplete], 
            labels=['Perfect (12/12)', 'Near-perfect (8-11)', 'Partial (<8)'],
            colors=['#2ecc71', '#3498db', '#e74c3c'],
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Task Completion Status\n(15 calculation steps)', fontsize=12)

plt.tight_layout()
plt.savefig('report/images/fig5_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

print("All figures saved to report/images/")
print("Generated: fig1_category_scores.png, fig2_score_heatmap.png,")
print("           fig3_task_totals.png, fig4_reviewer_agreement.png,")
print("           fig5_score_distributions.png")
