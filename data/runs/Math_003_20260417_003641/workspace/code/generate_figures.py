"""
generate_figures.py - Generate all figures for the research report
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# Load data
with open('outputs/problem_analysis.json') as f:
    analysis = json.load(f)

with open('outputs/comprehensive_results.json') as f:
    comprehensive = json.load(f)

problems = comprehensive['problems']
strategies = comprehensive['strategies']
summary = comprehensive['summary_statistics']

# ============================================================
# Figure 1: Goal Type Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
goal_types = analysis['goal_types']
labels = list(goal_types.keys())
sizes = list(goal_types.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
explode = [0.05] * len(labels)

wedges, texts, autotexts = axes[0].pie(sizes, explode=explode, labels=labels, 
                                         colors=colors, autopct='%1.0f%%',
                                         shadow=True, startangle=90,
                                         textprops={'fontsize': 11})
axes[0].set_title('Goal Type Distribution\n(IMO AG 30 Benchmark)', fontweight='bold')

# Bar chart
bars = axes[1].bar(labels, sizes, color=colors, edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('Goal Predicate Type')
axes[1].set_ylabel('Number of Problems')
axes[1].set_title('Problem Count by Goal Type', fontweight='bold')
for bar, count in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/goal_type_distribution.png')
plt.close()
print("Saved: goal_type_distribution.png")

# ============================================================
# Figure 2: Construction Function Frequency
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

funcs = analysis['construction_funcs']
func_names = list(funcs.keys())[:15]
func_counts = [funcs[k] for k in func_names]

colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(func_names)))
bars = ax.barh(range(len(func_names)), func_counts, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(func_names)))
ax.set_yticklabels(func_names)
ax.set_xlabel('Frequency')
ax.set_title('Top 15 Geometric Construction Functions\nin IMO AG 30 Benchmark', fontweight='bold')
ax.invert_yaxis()

for bar, count in zip(bars, func_counts):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
            str(count), ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('report/images/construction_frequency.png')
plt.close()
print("Saved: construction_frequency.png")

# ============================================================
# Figure 3: Problem Complexity Analysis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scatter: points vs definitions
points = [p['num_points'] for p in problems]
defs = [p['num_definitions'] for p in problems]
complexity = [p['complexity_score'] for p in problems]
goal_types_list = [p['goal_type'] for p in problems]

# Color by goal type
unique_goals = list(set(goal_types_list))
goal_colors = {g: plt.cm.Set1(i/len(unique_goals)) for i, g in enumerate(unique_goals)}

for g in unique_goals:
    mask = [gt == g for gt in goal_types_list]
    x = [p for p, m in zip(points, mask) if m]
    y = [d for d, m in zip(defs, mask) if m]
    c = [comp for comp, m in zip(complexity, mask) if m]
    axes[0, 0].scatter(x, y, c=goal_colors[g], s=80, label=g, edgecolors='black', linewidth=0.5, alpha=0.8)

axes[0, 0].set_xlabel('Number of Points')
axes[0, 0].set_ylabel('Number of Definitions')
axes[0, 0].set_title('Problem Structure: Points vs Definitions', fontweight='bold')
axes[0, 0].legend(title='Goal Type', fontsize=8)

# Histogram: complexity scores
axes[0, 1].hist(complexity, bins=12, color='steelblue', edgecolor='black', alpha=0.8)
axes[0, 1].set_xlabel('Complexity Score')
axes[0, 1].set_ylabel('Number of Problems')
axes[0, 1].set_title('Distribution of Problem Complexity', fontweight='bold')
axes[0, 1].axvline(np.mean(complexity), color='red', linestyle='--', label=f'Mean={np.mean(complexity):.1f}')
axes[0, 1].legend()

# Bar: complexity by year
year_summary = comprehensive['year_summary']
years = sorted(year_summary.keys())
avg_complexities = [year_summary[y]['avg_complexity'] for y in years]
problem_counts = [year_summary[y]['count'] for y in years]

ax2 = axes[1, 0]
x_pos = range(len(years))
bars1 = ax2.bar(x_pos, avg_complexities, color='coral', edgecolor='black', linewidth=0.5, alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([y[-4:] for y in years], rotation=45, ha='right')
ax2.set_xlabel('IMO Year')
ax2.set_ylabel('Average Complexity Score')
ax2.set_title('Problem Complexity by IMO Year', fontweight='bold')

# Scatter: search space vs complexity
search_spaces = [p['search_space_estimate'] for p in problems]
axes[1, 1].scatter(complexity, [max(1, s) for s in search_spaces], 
                    c='darkorange', s=80, edgecolors='black', linewidth=0.5, alpha=0.8)
axes[1, 1].set_xlabel('Complexity Score')
axes[1, 1].set_ylabel('Estimated Search Space (log scale)')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Complexity vs Search Space', fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/problem_complexity.png')
plt.close()
print("Saved: problem_complexity.png")

# ============================================================
# Figure 4: Search Strategy Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

strategy_names = list(strategies.keys())
strategy_labels = [
    'Pure BFS', 'Pure DFS', 'Random', 
    'Neural Beam', 'Neural MCTS', 'AlphaGeometry'
]
solve_rates = [summary['strategy_comparison'][s]['solve_rate'] * 100 for s in strategy_names]
avg_nodes = [summary['strategy_comparison'][s]['avg_nodes'] for s in strategy_names]

colors_strat = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCB77', '#4D96FF', '#9B59B6']

# Solve rate
bars = axes[0].bar(range(len(strategy_labels)), solve_rates, color=colors_strat, 
                    edgecolor='black', linewidth=0.5)
axes[0].set_xticks(range(len(strategy_labels)))
axes[0].set_xticklabels(strategy_labels, rotation=30, ha='right')
axes[0].set_ylabel('Solve Rate (%)')
axes[0].set_title('Problem Solve Rate by Strategy', fontweight='bold')
axes[0].set_ylim(0, 70)
for bar, rate in zip(bars, solve_rates):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

# Nodes explored (log scale)
bars2 = axes[1].bar(range(len(strategy_labels)), [max(1, n) for n in avg_nodes], 
                     color=colors_strat, edgecolor='black', linewidth=0.5)
axes[1].set_xticks(range(len(strategy_labels)))
axes[1].set_xticklabels(strategy_labels, rotation=30, ha='right')
axes[1].set_ylabel('Average Nodes Explored (log scale)')
axes[1].set_yscale('log')
axes[1].set_title('Search Efficiency by Strategy', fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/strategy_comparison.png')
plt.close()
print("Saved: strategy_comparison.png")

# ============================================================
# Figure 5: Rule Dependency Graph Visualization
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))

rule_graph = comprehensive['rule_graph']
predicates = rule_graph['predicates']
in_deg = rule_graph['in_degree']
out_deg = rule_graph['out_degree']

# Create a simple layout
n_preds = len(predicates)
angles = np.linspace(0, 2*np.pi, n_preds, endpoint=False)
radius = 3.5
positions = {p: (radius * np.cos(a), radius * np.sin(a)) for p, a in zip(predicates, angles)}

# Draw edges
for src, dst in rule_graph['edges']:
    if src in positions and dst in positions:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))

# Draw nodes
for pred in predicates:
    x, y = positions[pred]
    total_deg = in_deg.get(pred, 0) + out_deg.get(pred, 0)
    node_size = max(300, total_deg * 80)
    
    # Color by type
    if pred in ('cong', 'eqangle', 'eqratio', 'eqangle6', 'eqratio6'):
        color = '#4D96FF'
    elif pred in ('para', 'perp', 'coll'):
        color = '#6BCB77'
    elif pred in ('cyclic', 'circle'):
        color = '#FF6B6B'
    elif pred in ('midp',):
        color = '#FFD93D'
    else:
        color = '#C0C0C0'
    
    ax.scatter(x, y, s=node_size, c=color, edgecolors='black', linewidth=1, zorder=5)
    ax.annotate(pred, (x, y), ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.set_title('Deduction Rule Dependency Graph\n(Node size ∝ connectivity)', fontweight='bold')
ax.axis('off')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#4D96FF', edgecolor='black', label='Equality predicates'),
    mpatches.Patch(facecolor='#6BCB77', edgecolor='black', label='Geometric relations'),
    mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Circle predicates'),
    mpatches.Patch(facecolor='#FFD93D', edgecolor='black', label='Midpoint'),
    mpatches.Patch(facecolor='#C0C0C0', edgecolor='black', label='Other'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/rule_graph.png')
plt.close()
print("Saved: rule_graph.png")

# ============================================================
# Figure 6: Architecture Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Draw the neuro-symbolic architecture
def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, text=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.15, text, ha='center', va='bottom', fontsize=8, style='italic')

# Input
draw_box(ax, 0.5, 3.5, 2.5, 0.8, 'Problem Statement\n(Formal Language)', '#FFE0B2')

# Parser
draw_box(ax, 4, 3.5, 2, 0.8, 'Construction\nParser', '#BBDEFB')

# Symbolic Engine
draw_box(ax, 7, 3.5, 2.5, 0.8, 'Symbolic\nDeduction Engine', '#C8E6C9')

# Neural Network
draw_box(ax, 4, 1.5, 2.5, 0.8, 'Transformer\nLanguage Model', '#E1BEE7')

# Search
draw_box(ax, 7.5, 1.5, 2, 0.8, 'Proof Search\n(MCTS/Beam)', '#FFCDD2')

# Auxiliary Construction
draw_box(ax, 1, 1.5, 2.5, 0.8, 'Auxiliary\nConstruction Gen.', '#B2EBF2')

# Output
draw_box(ax, 10.5, 2.5, 2, 0.8, 'Verified\nProof', '#DCEDC8')

# Arrows
draw_arrow(ax, 3, 3.9, 4, 3.9, 'parse')
draw_arrow(ax, 6, 3.9, 7, 3.9, 'facts')
draw_arrow(ax, 9.5, 3.9, 10.5, 3.3)
draw_arrow(ax, 5.25, 2.3, 5.25, 3.5, 'suggest')
draw_arrow(ax, 7.5, 3.5, 8.5, 2.3, 'guide')
draw_arrow(ax, 6.5, 1.9, 7.5, 1.9, 'policy')
draw_arrow(ax, 9.5, 1.9, 10.5, 2.5)
draw_arrow(ax, 3.5, 1.9, 4, 1.9, 'propose')
draw_arrow(ax, 2.25, 2.3, 2.25, 3.5, 'new points')

# Feedback loop
ax.annotate('', xy=(1, 2.3), xytext=(7.5, 1.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red', 
                          connectionstyle='arc3,rad=0.3', linestyle='--'))
ax.text(4, 0.8, 'feedback: unsolved → generate auxiliary constructions', 
        ha='center', fontsize=9, color='red', style='italic')

# Title
ax.set_title('Neuro-Symbolic Geometry Theorem Prover Architecture', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(0.3, 5)
ax.axis('off')

plt.tight_layout()
plt.savefig('report/images/architecture.png')
plt.close()
print("Saved: architecture.png")

# ============================================================
# Figure 7: Per-Problem Solve Analysis (Heatmap-style)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 8))

# Create a matrix: problems x strategies
prob_names = [p['name'].replace('translated_imo_', '') for p in problems]
strat_names = ['Pure BFS', 'Pure DFS', 'Random', 'Neural Beam', 'Neural MCTS', 'AlphaGeometry']
strat_keys = list(strategies.keys())

matrix = np.zeros((len(problems), len(strat_keys)))
for j, sk in enumerate(strat_keys):
    for i, result in enumerate(strategies[sk]):
        # Match by order (they should be in same order)
        matrix[i, j] = 1 if result['solved'] else 0

# Sort by total solvability
row_sums = matrix.sum(axis=1)
sort_idx = np.argsort(-row_sums)
matrix = matrix[sort_idx]
prob_names_sorted = [prob_names[i] for i in sort_idx]

im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(strat_names)))
ax.set_xticklabels(strat_names, rotation=30, ha='right')
ax.set_yticks(range(len(prob_names_sorted)))
ax.set_yticklabels(prob_names_sorted, fontsize=8)
ax.set_title('Problem Solvability by Strategy\n(Green=Solved, Red=Unsolved)', fontweight='bold')

# Add text annotations
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        text = '✓' if matrix[i, j] > 0.5 else '✗'
        color = 'white' if matrix[i, j] < 0.5 else 'black'
        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

plt.colorbar(im, ax=ax, label='Solved (1) / Unsolved (0)', shrink=0.8)
plt.tight_layout()
plt.savefig('report/images/solvability_heatmap.png')
plt.close()
print("Saved: solvability_heatmap.png")

# ============================================================
# Figure 8: Proof Search Space Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Branching factor vs depth for different strategies
depths = np.arange(1, 12)
bfs_nodes = [43**d for d in depths]  # Pure symbolic: 43 rules
neural_nodes = [6**d for d in depths]  # Neural-guided: ~6 effective branching
ag_nodes = [3**d * 5 for d in depths]  # AlphaGeometry: 3 branching + aux constructions

axes[0].semilogy(depths, bfs_nodes, 'r-o', label='Pure Symbolic (b=43)', linewidth=2)
axes[0].semilogy(depths, neural_nodes, 'b-s', label='Neural-Guided (b≈6)', linewidth=2)
axes[0].semilogy(depths, ag_nodes, 'g-^', label='AlphaGeometry-style (b≈3)', linewidth=2)
axes[0].axhline(y=1e6, color='gray', linestyle='--', alpha=0.5, label='Practical limit (10⁶)')
axes[0].set_xlabel('Proof Depth')
axes[0].set_ylabel('Search Nodes (log scale)')
axes[0].set_title('Search Space Growth by Strategy', fontweight='bold')
axes[0].legend()
axes[0].set_ylim(1, 1e15)
axes[0].grid(True, alpha=0.3)

# Right: Solve probability vs complexity
complexities_sorted = sorted([p['complexity_score'] for p in problems])
for sk, label, color in [
    ('neural_guided_beam', 'Neural Beam', 'blue'),
    ('neural_guided_mcts', 'Neural MCTS', 'green'),
    ('alphageometry_style', 'AlphaGeometry', 'purple')
]:
    if 'solve_probability' in strategies[sk][0]:
        probs_by_complexity = sorted(
            [(p['complexity_score'], strategies[sk][i].get('solve_probability', 0)) 
             for i, p in enumerate(problems)],
            key=lambda x: x[0]
        )
        x = [p[0] for p in probs_by_complexity]
        y = [p[1] * 100 for p in probs_by_complexity]
        axes[1].plot(x, y, '-o', label=label, color=color, linewidth=2, markersize=4)

axes[1].set_xlabel('Problem Complexity Score')
axes[1].set_ylabel('Estimated Solve Probability (%)')
axes[1].set_title('Solve Probability vs Problem Complexity', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/search_space_analysis.png')
plt.close()
print("Saved: search_space_analysis.png")

# ============================================================
# Figure 9: IMO Year Trend Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

years = sorted(comprehensive['year_summary'].keys())
year_ints = [int(y) for y in years]
avg_comp = [comprehensive['year_summary'][y]['avg_complexity'] for y in years]
counts = [comprehensive['year_summary'][y]['count'] for y in years]
avg_pts = [comprehensive['year_summary'][y]['avg_points'] for y in years]

# Complexity trend
axes[0].bar(year_ints, avg_comp, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.8)
z = np.polyfit(year_ints, avg_comp, 1)
p = np.poly1d(z)
axes[0].plot(year_ints, p(year_ints), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.2f})')
axes[0].set_xlabel('IMO Year')
axes[0].set_ylabel('Average Complexity Score')
axes[0].set_title('Problem Complexity Trend Over Time', fontweight='bold')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Points and constructions trend
axes[1].scatter(year_ints, avg_pts, c='coral', s=100, edgecolors='black', linewidth=0.5, zorder=5, label='Avg Points')
avg_constr = [comprehensive['year_summary'][y]['avg_constructions'] for y in years]
axes[1].scatter(year_ints, avg_constr, c='mediumseagreen', s=100, edgecolors='black', linewidth=0.5, zorder=5, marker='s', label='Avg Constructions')
axes[1].set_xlabel('IMO Year')
axes[1].set_ylabel('Count')
axes[1].set_title('Problem Size Metrics Over Time', fontweight='bold')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('report/images/year_trends.png')
plt.close()
print("Saved: year_trends.png")

# ============================================================
# Figure 10: Synthetic Data Generation Pipeline
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

# Draw pipeline stages
stages = [
    ('Random\nConstruction\nSampling', '#FFE0B2'),
    ('Symbolic\nDeduction\n(Forward)', '#BBDEFB'),
    ('Theorem\nExtraction', '#C8E6C9'),
    ('Proof\nTree\nRecording', '#E1BEE7'),
    ('Training\nData\nFormatting', '#FFCDD2'),
    ('Transformer\nTraining', '#B2EBF2'),
]

for i, (text, color) in enumerate(stages):
    x = i * 2.2 + 0.5
    rect = mpatches.FancyBboxPatch((x, 1), 1.8, 1.5, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.9, 1.75, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    if i < len(stages) - 1:
        ax.annotate('', xy=(x + 2.2, 1.75), xytext=(x + 1.8, 1.75),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Stats
ax.text(1.4, 0.5, '100M+\ndiagrams', ha='center', fontsize=9, color='gray')
ax.text(3.6, 0.5, '~10⁹\nfacts', ha='center', fontsize=9, color='gray')
ax.text(5.8, 0.5, '~10⁸\ntheorems', ha='center', fontsize=9, color='gray')
ax.text(8.0, 0.5, '~10⁷\nproof trees', ha='center', fontsize=9, color='gray')
ax.text(10.2, 0.5, '~10⁸\nstep pairs', ha='center', fontsize=9, color='gray')
ax.text(12.4, 0.5, '~10⁶\niterations', ha='center', fontsize=9, color='gray')

ax.set_title('Synthetic Training Data Generation Pipeline', fontsize=13, fontweight='bold')
ax.set_xlim(0, 14)
ax.set_ylim(0, 3.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('report/images/data_pipeline.png')
plt.close()
print("Saved: data_pipeline.png")

print("\nAll figures generated successfully!")
