"""
Generate all figures for the research report.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('report/images', exist_ok=True)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# Load data
with open('outputs/problem_analysis.json', 'r') as f:
    analysis = json.load(f)
with open('outputs/training_curves.json', 'r') as f:
    training = json.load(f)
with open('outputs/combined_results.json', 'r') as f:
    combined = json.load(f)

# ============================================================
# Figure 1: Problem Complexity Overview
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

names = [a['name'].replace('translated_', '') for a in analysis]
num_points = [a['num_points'] for a in analysis]
num_facts = [a['num_facts'] for a in analysis]
num_constructions = [a['num_constructions'] for a in analysis]

axes[0].bar(range(len(names)), num_points, color='steelblue', alpha=0.8)
axes[0].set_title('Number of Points per Problem')
axes[0].set_xlabel('Problem Index')
axes[0].set_ylabel('Points')
axes[0].set_xticks(range(0, len(names), 5))

axes[1].bar(range(len(names)), num_facts, color='coral', alpha=0.8)
axes[1].set_title('Number of Facts per Problem')
axes[1].set_xlabel('Problem Index')
axes[1].set_ylabel('Facts')
axes[1].set_xticks(range(0, len(names), 5))

axes[2].bar(range(len(names)), num_constructions, color='seagreen', alpha=0.8)
axes[2].set_title('Number of Constructions per Problem')
axes[2].set_xlabel('Problem Index')
axes[2].set_ylabel('Constructions')
axes[2].set_xticks(range(0, len(names), 5))

plt.tight_layout()
plt.savefig('report/images/fig1_problem_complexity.png')
plt.close()
print("Saved fig1_problem_complexity.png")

# ============================================================
# Figure 2: Predicate Distribution
# ============================================================
all_preds = {}
for a in analysis:
    for pred, count in a['predicate_distribution'].items():
        all_preds[pred] = all_preds.get(pred, 0) + count

preds = sorted(all_preds.items(), key=lambda x: -x[1])[:15]
pred_names = [p[0] for p in preds]
pred_counts = [p[1] for p in preds]

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.tab20(np.linspace(0, 1, len(pred_names)))
ax.barh(range(len(pred_names)), pred_counts, color=colors)
ax.set_yticks(range(len(pred_names)))
ax.set_yticklabels(pred_names)
ax.set_xlabel('Total Count Across All Problems')
ax.set_title('Top-15 Geometric Predicate Distribution in IMO-AG-30')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('report/images/fig2_predicate_distribution.png')
plt.close()
print("Saved fig2_predicate_distribution.png")

# ============================================================
# Figure 3: Training Curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(training['train_loss'], label='Train Loss', color='steelblue', linewidth=2)
ax.plot(training['val_loss'], label='Validation Loss', color='coral', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('GNN Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fig3_training_curves.png')
plt.close()
print("Saved fig3_training_curves.png")

# ============================================================
# Figure 4: Strategy Comparison
# ============================================================
strategies = list(combined.keys())
solve_rates = [combined[s]['solve_rate'] * 100 for s in strategies]
avg_nodes = [combined[s]['avg_nodes'] for s in strategies]
avg_times = [combined[s]['avg_time'] for s in strategies]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

x = np.arange(len(strategies))
width = 0.6

axes[0].bar(x, solve_rates, width, color='steelblue', alpha=0.8)
axes[0].set_ylabel('Solve Rate (%)')
axes[0].set_title('Solve Rate by Strategy')
axes[0].set_xticks(x)
axes[0].set_xticklabels(strategies, rotation=45, ha='right')
axes[0].set_ylim(0, max(solve_rates + [5]))

axes[1].bar(x, avg_nodes, width, color='coral', alpha=0.8)
axes[1].set_ylabel('Average Nodes Expanded')
axes[1].set_title('Search Efficiency by Strategy')
axes[1].set_xticks(x)
axes[1].set_xticklabels(strategies, rotation=45, ha='right')

axes[2].bar(x, avg_times, width, color='seagreen', alpha=0.8)
axes[2].set_ylabel('Average Time (s)')
axes[2].set_title('Runtime by Strategy')
axes[2].set_xticks(x)
axes[2].set_xticklabels(strategies, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('report/images/fig4_strategy_comparison.png')
plt.close()
print("Saved fig4_strategy_comparison.png")

# ============================================================
# Figure 5: Problem Difficulty Scatter (Facts vs Constructions)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(num_constructions, num_facts, c=num_points, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
ax.set_xlabel('Number of Constructions')
ax.set_ylabel('Number of Initial Facts')
ax.set_title('Problem Complexity Space (color = number of points)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Points')

# Annotate some outliers
for i, name in enumerate(names):
    if num_constructions[i] > 14 or num_facts[i] > 35:
        ax.annotate(name, (num_constructions[i], num_facts[i]), fontsize=7, alpha=0.7)

plt.tight_layout()
plt.savefig('report/images/fig5_complexity_scatter.png')
plt.close()
print("Saved fig5_complexity_scatter.png")

# ============================================================
# Figure 6: Goal Predicate Distribution
# ============================================================
goal_preds = {}
for a in analysis:
    gp = a['goal_predicate']
    goal_preds[gp] = goal_preds.get(gp, 0) + 1

fig, ax = plt.subplots(figsize=(8, 5))
wedges, texts, autotexts = ax.pie(goal_preds.values(), labels=goal_preds.keys(), autopct='%1.0f%%', startangle=90)
ax.set_title('Distribution of Goal Predicate Types')
plt.tight_layout()
plt.savefig('report/images/fig6_goal_distribution.png')
plt.close()
print("Saved fig6_goal_distribution.png")

# ============================================================
# Figure 7: Nodes Expanded per Problem (Heatmap)
# ============================================================
problem_names = [a['name'].replace('translated_', '') for a in analysis]
strat_names = list(combined.keys())

node_matrix = []
for s in strat_names:
    row = []
    for p in analysis:
        pname = p['name']
        nodes = combined[s]['per_problem'].get(pname, {}).get('nodes', 0)
        row.append(nodes)
    node_matrix.append(row)

node_matrix = np.array(node_matrix)

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(node_matrix, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(problem_names)))
ax.set_xticklabels(problem_names, rotation=90, fontsize=7)
ax.set_yticks(range(len(strat_names)))
ax.set_yticklabels(strat_names)
ax.set_title('Search Nodes Expanded per Problem (Heatmap)')
plt.colorbar(im, ax=ax, label='Nodes Expanded')
plt.tight_layout()
plt.savefig('report/images/fig7_nodes_heatmap.png')
plt.close()
print("Saved fig7_nodes_heatmap.png")

print("\nAll figures generated!")
