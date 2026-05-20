#!/usr/bin/env python3
"""
Generate analysis figures for the research report.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("report/images", exist_ok=True)

# Load benchmark analysis
with open("outputs/benchmark_analysis.json", "r") as f:
    analysis = json.load(f)

# Load solver evaluation
with open("outputs/solver_evaluation.json", "r") as f:
    eval_results = json.load(f)

# Figure 1: Problem Complexity Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Number of points distribution
points_data = [p["num_points"] for p in analysis["problems"]]
axes[0].hist(points_data, bins=range(min(points_data), max(points_data)+2), 
             edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Number of Points', fontsize=12)
axes[0].set_ylabel('Number of Problems', fontsize=12)
axes[0].set_title('Distribution of Problem Complexity\n(Number of Points)', fontsize=14)
axes[0].grid(axis='y', alpha=0.3)

# Number of statements distribution
stmts_data = [p["num_statements"] for p in analysis["problems"]]
axes[1].hist(stmts_data, bins=range(min(stmts_data), max(stmts_data)+2), 
             edgecolor='black', alpha=0.7, color='coral')
axes[1].set_xlabel('Number of Statements', fontsize=12)
axes[1].set_ylabel('Number of Problems', fontsize=12)
axes[1].set_title('Distribution of Problem Complexity\n(Number of Statements)', fontsize=14)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure_1_complexity_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved")

# Figure 2: Conclusion Types
fig, ax = plt.subplots(figsize=(10, 6))

conclusion_types = analysis["statistics"]["conclusion_types"]
labels = list(conclusion_types.keys())
sizes = list(conclusion_types.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                   colors=colors, startangle=90, textprops={'fontsize': 11})
ax.set_title('Distribution of Conclusion Types\nin IMO Geometry Problems', fontsize=14)

plt.tight_layout()
plt.savefig('report/images/figure_2_conclusion_types.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved")

# Figure 3: Statement Types Frequency
fig, ax = plt.subplots(figsize=(12, 6))

stmt_types = analysis["statistics"]["statement_types"]
sorted_stmts = sorted(stmt_types.items(), key=lambda x: x[1], reverse=True)[:12]
labels = [s[0] for s in sorted_stmts]
values = [s[1] for s in sorted_stmts]

bars = ax.barh(range(len(labels)), values, color='teal', alpha=0.8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('Most Common Geometric Relations\nin IMO Problems', fontsize=14)
ax.grid(axis='x', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
            str(val), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('report/images/figure_3_statement_types.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved")

# Figure 4: Solver Performance Analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Derived facts per problem
derived_facts = [d.get("derived_facts", 0) for d in eval_results["details"]]
problem_names = [d["name"].replace("translated_imo_", "") for d in eval_results["details"]]

# Sort by derived facts
sorted_indices = np.argsort(derived_facts)[::-1][:15]  # Top 15
sorted_names = [problem_names[i] for i in sorted_indices]
sorted_facts = [derived_facts[i] for i in sorted_indices]

bars = axes[0].barh(range(len(sorted_names)), sorted_facts, color='mediumpurple', alpha=0.8)
axes[0].set_yticks(range(len(sorted_names)))
axes[0].set_yticklabels(sorted_names, fontsize=9)
axes[0].set_xlabel('Derived Facts', fontsize=12)
axes[0].set_title('Facts Derived per Problem\n(Top 15)', fontsize=14)
axes[0].grid(axis='x', alpha=0.3)

# Time per problem
times = [d.get("time", 0) for d in eval_results["details"]]
sorted_times = [times[i] for i in sorted_indices]

axes[1].barh(range(len(sorted_names)), sorted_times, color='indianred', alpha=0.8)
axes[1].set_yticks(range(len(sorted_names)))
axes[1].set_yticklabels(sorted_names, fontsize=9)
axes[1].set_xlabel('Time (seconds)', fontsize=12)
axes[1].set_title('Computation Time per Problem\n(Top 15 by Facts)', fontsize=14)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure_4_solver_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved")

# Figure 5: Year Distribution
fig, ax = plt.subplots(figsize=(10, 5))

year_dist = analysis["statistics"]["year_distribution"]
years = sorted(year_dist.keys())
counts = [year_dist[y] for y in years]

ax.bar(range(len(years)), counts, color='darkorange', alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(years)))
ax.set_xticklabels([f"p{y}" for y in years], fontsize=9, rotation=45)
ax.set_ylabel('Number of Problems', fontsize=12)
ax.set_title('Problem Distribution by Position\nin IMO', fontsize=14)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure_5_position_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved")

# Figure 6: Problem Solving Pipeline Diagram
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

# Draw pipeline stages
stages = [
    (1.5, 1.5, 'Problem\nParser'),
    (4, 1.5, 'Fact\nExtraction'),
    (6.5, 1.5, 'Forward\nChaining'),
    (9, 1.5, 'Conclusion\nVerification')
]

for x, y, text in stages:
    rect = plt.Rectangle((x-0.7, y-0.4), 1.4, 0.8, 
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')

# Draw arrows
for i in range(len(stages)-1):
    ax.annotate('', xy=(stages[i+1][0]-0.8, stages[i+1][1]), 
                xytext=(stages[i][0]+0.8, stages[i][1]),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax.set_title('Geometry Problem Solving Pipeline', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('report/images/figure_6_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved")

print("\nAll figures generated successfully!")