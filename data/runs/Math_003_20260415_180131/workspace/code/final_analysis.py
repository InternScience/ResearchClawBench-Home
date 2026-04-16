#!/usr/bin/env python3
"""
Step 5: Comprehensive analysis combining all results and generating
final visualizations for the report. Also produces a detailed
problem-by-problem analysis table.
"""

import json
import os
import matplotlib
import re
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_180131"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report/images")

# Load all data
with open(os.path.join(OUTPUT_DIR, 'parsed_problems.json'), 'r') as f:
    problems = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'parsed_rules.json'), 'r') as f:
    rules = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'parsed_definitions.json'), 'r') as f:
    definitions = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'proof_results_enhanced.json'), 'r') as f:
    proof_results = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'coord_verification_results.json'), 'r') as f:
    coord_results = json.load(f)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 150

# ─── Build comprehensive problem table ────────────────────────────────
problem_table = []
for i, p in enumerate(problems):
    pr = proof_results[i]
    cr = coord_results[i]
    
    # Extract year from name
    name = p['name']
    year_match = re.search(r'(\d{4})', name)
    year = int(year_match.group(1)) if year_match else 0
    
    # Difficulty score: weighted combination
    difficulty = p['num_points'] * 0.3 + p['num_construction_steps'] * 0.5 + \
                 (1 if p['goal_predicate'] in ['eqangle', 'eqratio'] else 0.5) * 2
    
    problem_table.append({
        'name': name,
        'year': year,
        'goal_type': p['goal_predicate'],
        'num_points': p['num_points'],
        'num_steps': p['num_construction_steps'],
        'initial_facts': pr['initial_facts_count'],
        'derived_facts': pr['total_facts_count'] - pr['initial_facts_count'],
        'proof_steps': pr['proof_steps'],
        'iterations': pr['iterations'],
        'fc_solved': pr['solved'],
        'coord_coverage': cr['valid_trials'],
        'coord_confidence': cr['confidence'],
        'difficulty_score': difficulty
    })

import re

# Save comprehensive table
with open(os.path.join(OUTPUT_DIR, 'comprehensive_problem_table.json'), 'w') as f:
    json.dump(problem_table, f, indent=2)

# ─── Figure 7: Comprehensive Problem Analysis Heatmap ────────────────
fig, ax = plt.subplots(figsize=(16, 8))

# Create heatmap data: problems x metrics
metrics = ['num_points', 'num_steps', 'initial_facts', 'proof_steps', 'difficulty_score']
metric_labels = ['Points', 'Constructions', 'Initial Facts', 'Derived Steps', 'Difficulty']

data_matrix = np.array([[pt[m] for m in metrics] for pt in problem_table])
# Normalize each column
for j in range(data_matrix.shape[1]):
    col = data_matrix[:, j]
    col_range = col.max() - col.min()
    if col_range > 0:
        data_matrix[:, j] = (col - col.min()) / col_range
    else:
        data_matrix[:, j] = 0

im = ax.imshow(data_matrix.T, cmap='YlOrRd', aspect='auto')
ax.set_yticks(range(len(metric_labels)))
ax.set_yticklabels(metric_labels)
ax.set_xticks(range(len(problem_table)))
ax.set_xticklabels([f"IMO {pt['year']}" for pt in problem_table], rotation=45, ha='right', fontsize=7)
ax.set_title('IMO-AG-30 Problem Characteristics Heatmap')

plt.colorbar(im, ax=ax, label='Normalized Value')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig7_problem_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_problem_heatmap.png")

# ─── Figure 8: Year-by-Year Difficulty Trend ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

years = sorted(set(pt['year'] for pt in problem_table))
year_data = defaultdict(list)
for pt in problem_table:
    year_data[pt['year']].append(pt)

ax = axes[0]
avg_diff_by_year = {y: np.mean([pt['difficulty_score'] for pt in year_data[y]]) for y in years}
avg_pts_by_year = {y: np.mean([pt['num_points'] for pt in year_data[y]]) for y in years}

ax.plot(years, [avg_diff_by_year[y] for y in years], 'o-', color='#E74C3C', linewidth=2, label='Difficulty Score')
ax.plot(years, [avg_pts_by_year[y] for y in years], 's--', color='#3498DB', linewidth=2, label='Avg Points')
ax.set_xlabel('IMO Year')
ax.set_ylabel('Score / Count')
ax.set_title('(a) Difficulty Trend Over Years')
ax.legend()

# Goal type by year
ax = axes[1]
goal_types_all = list(set(pt['goal_type'] for pt in problem_table))
goal_type_by_year = {}
for y in years:
    gt_counts = defaultdict(int)
    for pt in year_data[y]:
        gt_counts[pt['goal_type']] += 1
    goal_type_by_year[y] = dict(gt_counts)

# Stacked bar chart
bottom = np.zeros(len(years))
colors = sns.color_palette("Set2", len(goal_types_all))
for i, gt in enumerate(goal_types_all):
    vals = [goal_type_by_year[y].get(gt, 0) for y in years]
    ax.bar(years, vals, bottom=bottom, color=colors[i], label=gt, edgecolor='white')
    bottom += np.array(vals)

ax.set_xlabel('IMO Year')
ax.set_ylabel('Number of Problems')
ax.set_title('(b) Goal Type Distribution by Year')
ax.legend(title='Goal Type')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig8_year_trends.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_year_trends.png")

# ─── Figure 9: Proof Engine Capability Analysis ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Fact derivation efficiency
ax = axes[0, 0]
derived_facts = [pt['derived_facts'] for pt in problem_table]
initial_facts = [pt['initial_facts'] for pt in problem_table]
growth_ratio = [(d/i if i > 0 else 0) for d, i in zip(derived_facts, initial_facts)]

scatter_colors = [sns.color_palette("Set2")[list(set(pt['goal_type'] for pt in problem_table)).index(pt['goal_type'])] 
                  for pt in problem_table]
ax.scatter(initial_facts, derived_facts, c=scatter_colors, s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Initial Facts')
ax.set_ylabel('New Facts Derived')
ax.set_title('(a) Fact Derivation Efficiency')

# (b) Rule coverage analysis
ax = axes[0, 1]
rule_preds = defaultdict(int)
for r in rules:
    for p in r['premises']:
        rule_preds[p['predicate']] += 1
    rule_preds[r['conclusion']['predicate']] += 1

pred_names = sorted(rule_preds.keys(), key=lambda x: rule_preds[x], reverse=True)[:15]
pred_counts = [rule_preds[p] for p in pred_names]
ax.barh(pred_names, pred_counts, color=sns.color_palette("viridis", len(pred_names)))
ax.set_xlabel('Total Occurrences in Rules')
ax.set_title('(b) Predicate Frequency in Rules')

# (c) Construction type distribution across problems
ax = axes[1, 0]
constr_types = defaultdict(int)
for p in problems:
    for c in p['constructions']:
        for con in c['constraints']:
            constr_types[con['predicate']] += 1

top_constrs = sorted(constr_types.items(), key=lambda x: x[1], reverse=True)[:12]
c_names = [x[0] for x in top_constrs]
c_counts = [x[1] for x in top_constrs]
ax.barh(c_names, c_counts, color=sns.color_palette("magma", len(c_names)))
ax.set_xlabel('Total Occurrences')
ax.set_title('(c) Construction Type Frequency')

# (d) Proof engine vs coordinate verification comparison
ax = axes[1, 1]
methods = ['Forward\nChaining', 'Coordinate\nVerification', 'DDAR*', 'AlphaGeometry*']
solve_rates = [0, 0, 66.7, 83.3]
bar_colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']
bars = ax.bar(methods, solve_rates, color=bar_colors, edgecolor='black', linewidth=0.5)
for bar, rate in zip(bars, solve_rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Solve Rate (%)')
ax.set_title('(d) Method Comparison on IMO-AG-30')
ax.set_ylim(0, 100)
ax.annotate('*Literature benchmarks', xy=(0.5, -0.12), xycoords='axes fraction',
            fontsize=8, ha='center', color='gray')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig9_capability_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_capability_analysis.png")

# ─── Figure 10: System Architecture Diagram ──────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Draw boxes for system components
boxes = [
    {'x': 1, 'y': 6, 'w': 3, 'h': 1.5, 'label': 'Problem\nParser', 'color': '#3498DB'},
    {'x': 5, 'y': 6, 'w': 3, 'h': 1.5, 'label': 'Fact\nExtractor', 'color': '#2ECC71'},
    {'x': 9, 'y': 6, 'w': 3, 'h': 1.5, 'label': 'Rule\nEngine', 'color': '#E74C3C'},
    {'x': 1, 'y': 3, 'w': 3, 'h': 1.5, 'label': 'Coordinate\nVerifier', 'color': '#F39C12'},
    {'x': 5, 'y': 3, 'w': 3, 'h': 1.5, 'label': 'Search\nStrategy', 'color': '#9B59B6'},
    {'x': 9, 'y': 3, 'w': 3, 'h': 1.5, 'label': 'Proof\nChecker', 'color': '#1ABC9C'},
    {'x': 5, 'y': 0.5, 'w': 4, 'h': 1.5, 'label': 'Result\nAggregator', 'color': '#34495E'},
]

for box in boxes:
    rect = plt.Rectangle((box['x'], box['y']), box['w'], box['h'],
                          facecolor=box['color'], edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(rect)
    ax.text(box['x']+box['w']/2, box['y']+box['h']/2, box['label'],
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Draw arrows
arrows = [
    (4, 6.75, 5, 6.75),   # Parser -> Fact Extractor
    (8, 6.75, 9, 6.75),   # Fact Extractor -> Rule Engine
    (2.5, 6, 2.5, 4.5),   # Parser -> Coordinate Verifier
    (6.5, 6, 6.5, 4.5),   # Fact Extractor -> Search Strategy
    (10.5, 6, 10.5, 4.5), # Rule Engine -> Proof Checker
    (4, 3.75, 5, 3.75),   # Coord Verifier -> Search Strategy
    (8, 3.75, 9, 3.75),   # Search Strategy -> Proof Checker
    (2.5, 3, 5.5, 2),     # Coord Verifier -> Result Aggregator
    (6.5, 3, 7, 2),       # Search Strategy -> Result Aggregator
    (10.5, 3, 8.5, 2),    # Proof Checker -> Result Aggregator
]

for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_title('Neuro-Symbolic Geometry Proving System Architecture', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig10_system_architecture.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_system_architecture.png")

# ─── Save method contract and artifact inventory ──────────────────────
method_contract = {
    "task": "Autonomous geometry theorem proving for IMO-AG-30",
    "primary_methods": [
        "Forward-chaining symbolic deduction with 43 inference rules",
        "Coordinate geometry numerical verification",
        "Construction-based fact extraction from definitions",
        "Heuristic search strategies for proof exploration"
    ],
    "named_approaches": {
        "symbolic_deduction": "Forward chaining using the 43 rules from rules.txt",
        "algebraic_verification": "Randomized coordinate assignment + numerical checking",
        "neuro_symbolic_combination": "Combined symbolic + algebraic approach"
    },
    "target_artifacts": [
        "parsed_problems.json",
        "parsed_rules.json", 
        "parsed_definitions.json",
        "proof_results_enhanced.json",
        "coord_verification_results.json",
        "comprehensive_problem_table.json",
        "analysis_stats.json",
        "fig1-fig10 PNG figures",
        "report.md"
    ]
}

with open(os.path.join(OUTPUT_DIR, 'method_contract.json'), 'w') as f:
    json.dump(method_contract, f, indent=2)

artifact_inventory = {
    "parsed_problems.json": {"status": "satisfied", "path": "outputs/parsed_problems.json"},
    "parsed_rules.json": {"status": "satisfied", "path": "outputs/parsed_rules.json"},
    "parsed_definitions.json": {"status": "satisfied", "path": "outputs/parsed_definitions.json"},
    "proof_results_enhanced.json": {"status": "satisfied", "path": "outputs/proof_results_enhanced.json"},
    "coord_verification_results.json": {"status": "satisfied", "path": "outputs/coord_verification_results.json"},
    "comprehensive_problem_table.json": {"status": "satisfied", "path": "outputs/comprehensive_problem_table.json"},
    "analysis_stats.json": {"status": "satisfied", "path": "outputs/analysis_stats.json"},
    "fig1_problem_complexity.png": {"status": "satisfied", "path": "report/images/fig1_problem_complexity.png"},
    "fig2_proof_engine_performance.png": {"status": "satisfied", "path": "report/images/fig2_proof_engine_performance.png"},
    "fig3_rule_analysis.png": {"status": "satisfied", "path": "report/images/fig3_rule_analysis.png"},
    "fig4_difficulty_landscape.png": {"status": "satisfied", "path": "report/images/fig4_difficulty_landscape.png"},
    "fig5_fact_growth.png": {"status": "satisfied", "path": "report/images/fig5_fact_growth.png"},
    "fig6_method_comparison.png": {"status": "satisfied", "path": "report/images/fig6_method_comparison.png"},
    "fig7_problem_heatmap.png": {"status": "satisfied", "path": "report/images/fig7_problem_heatmap.png"},
    "fig8_year_trends.png": {"status": "satisfied", "path": "report/images/fig8_year_trends.png"},
    "fig9_capability_analysis.png": {"status": "satisfied", "path": "report/images/fig9_capability_analysis.png"},
    "fig10_system_architecture.png": {"status": "satisfied", "path": "report/images/fig10_system_architecture.png"},
    "report.md": {"status": "pending", "path": "report/report.md"}
}

with open(os.path.join(OUTPUT_DIR, 'target_artifact_inventory.json'), 'w') as f:
    json.dump(artifact_inventory, f, indent=2)

print("\nAll figures and intermediate results saved.")
print(f"Problem table entries: {len(problem_table)}")
print(f"Figures generated: 10")