"""
Generate visualizations for the IMO geometry analysis.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

# Load analysis results
with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/outputs/analysis_results.json') as f:
    results = json.load(f)

output_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/report/images'

# Figure 1: Goal type distribution
fig, ax = plt.subplots(figsize=(10, 6))
goal_dist = results['goal_distribution']
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4', '#795548']
labels = list(goal_dist.keys())
values = list(goal_dist.values())
bars = ax.bar(labels, values, color=colors[:len(labels)], edgecolor='white', linewidth=1.5)
ax.set_ylabel('Number of Problems', fontsize=14)
ax.set_xlabel('Goal Type', fontsize=14)
ax.set_title('Distribution of Problem Goal Types in IMO AG-30 Benchmark', fontsize=16, fontweight='bold')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(val), 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=11)
plt.tight_layout()
plt.savefig(f'{output_dir}/goal_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Top predicates frequency
fig, ax = plt.subplots(figsize=(12, 6))
pred_freq = results['predicate_frequency']
top_preds = list(pred_freq.keys())[:15]
top_vals = [pred_freq[p] for p in top_preds]
bars = ax.barh(range(len(top_preds)), top_vals, color='#3F51B5', edgecolor='white')
ax.set_yticks(range(len(top_preds)))
ax.set_yticklabels(top_preds, fontsize=11)
ax.set_xlabel('Frequency (across 30 problems)', fontsize=14)
ax.set_title('Most Common Geometric Predicates in IMO AG-30', fontsize=16, fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, top_vals):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, str(val),
            ha='left', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/predicate_frequency.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Problem complexity scatter
fig, ax = plt.subplots(figsize=(10, 7))
classifications = results['classifications']
goal_types = list(set(c['goal_type'] for c in classifications))
color_map = {gt: plt.cm.Set2(i/len(goal_types)) for i, gt in enumerate(goal_types)}

for c in classifications:
    x = c['num_steps']
    y = c['num_predicates']
    d = c['difficulty']
    ax.scatter(x, y, s=d*8+30, c=[color_map[c['goal_type']]], alpha=0.7, edgecolors='black', linewidth=0.5)

# Legend
legend_patches = [mpatches.Patch(color=color_map[gt], label=gt) for gt in sorted(goal_types)]
ax.legend(handles=legend_patches, loc='upper left', fontsize=10)
ax.set_xlabel('Number of Construction Steps', fontsize=14)
ax.set_ylabel('Number of Distinct Predicates', fontsize=14)
ax.set_title('Problem Complexity: Steps vs Predicates (size = difficulty score)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/complexity_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Construction type usage by goal type
fig, ax = plt.subplots(figsize=(12, 7))
construction_types = ['circle', 'line', 'transformation', 'center', 'primitive', 'relation']
goal_types_sorted = sorted(set(c['goal_type'] for c in classifications))
matrix = np.zeros((len(goal_types_sorted), len(construction_types)))

for c in classifications:
    gi = goal_types_sorted.index(c['goal_type'])
    for ct in c['construction_types']:
        if ct in construction_types:
            ci = construction_types.index(ct)
            matrix[gi][ci] += 1

im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(construction_types)))
ax.set_xticklabels(construction_types, fontsize=11, rotation=30, ha='right')
ax.set_yticks(range(len(goal_types_sorted)))
ax.set_yticklabels(goal_types_sorted, fontsize=11)
for i in range(len(goal_types_sorted)):
    for j in range(len(construction_types)):
        ax.text(j, i, int(matrix[i][j]), ha='center', va='center', fontsize=12, fontweight='bold')
ax.set_title('Construction Types Used by Goal Type', fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax, label='Number of Problems')
plt.tight_layout()
plt.savefig(f'{output_dir}/construction_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 5: Difficulty distribution
fig, ax = plt.subplots(figsize=(10, 6))
difficulties = [c['difficulty'] for c in classifications]
names = [c['name'].replace('translated_', '').replace('_', '-') for c in classifications]
sorted_idx = np.argsort(difficulties)[::-1]
difficulties = [difficulties[i] for i in sorted_idx]
names = [names[i] for i in sorted_idx]

colors_diff = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(difficulties)))
bars = ax.barh(range(len(difficulties)), difficulties, color=colors_diff, edgecolor='white')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Difficulty Score', fontsize=14)
ax.set_title('Problem Difficulty Ranking (heuristic score)', fontsize=16, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_dir}/difficulty_ranking.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 6: Rule categories
fig, ax = plt.subplots(figsize=(8, 8))
rule_cats = results['rule_categories']
labels = list(rule_cats.keys())
values = list(rule_cats.values())
colors_pie = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.0f%%', 
                                   colors=colors_pie, startangle=90, textprops={'fontsize': 11})
ax.set_title(f'Inference Rule Categories (Total: {results["total_rules"]} rules)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/rule_categories.png', dpi=150, bbox_inches='tight')
plt.close()

print("All figures generated successfully.")
