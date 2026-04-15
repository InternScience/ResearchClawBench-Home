"""
Generate additional figures from prover results and combined analysis.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/outputs/enhanced_prover_results.json') as f:
    prover_results = json.load(f)

with open('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/outputs/analysis_results.json') as f:
    analysis = json.load(f)

output_dir = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260415_002019/report/images'

# Figure: Prover results overview
fig, ax = plt.subplots(figsize=(10, 6))
proved = sum(1 for r in prover_results if r['goal_satisfied'])
not_proved = len(prover_results) - proved
bars = ax.bar(['Proved (Forward Chaining)', 'Not Proved'], [proved, not_proved], 
              color=['#4CAF50', '#F44336'], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, [proved, not_proved]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(val),
            ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Problems', fontsize=14)
ax.set_title('Forward Chaining Prover Results on IMO AG-30', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/prover_results.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure: Facts derived per problem
fig, ax = plt.subplots(figsize=(14, 6))
names = [r['name'].replace('translated_', '').replace('_', '-') for r in prover_results]
derived = [r['derived_facts'] for r in prover_results]
proved_flags = [r['goal_satisfied'] for r in prover_results]
colors = ['#4CAF50' if p else '#2196F3' for p in proved_flags]
bars = ax.bar(range(len(names)), derived, color=colors, edgecolor='white')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=90, fontsize=7)
ax.set_ylabel('Derived Facts', fontsize=14)
ax.set_title('Facts Derived by Forward Chaining per Problem (green=proved)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/derived_facts.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure: Fact type distribution per problem
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
names_short = [r['name'].replace('translated_', '').replace('_', '-') for r in prover_results]
x = range(len(names_short))

ax1 = axes[0]
ax1.bar(x, [r['cong_count'] for r in prover_results], label='Congruence', color='#2196F3')
ax1.bar(x, [r['coll_count'] for r in prover_results], bottom=[r['cong_count'] for r in prover_results], 
        label='Collinear', color='#4CAF50')
bottoms2 = [r['cong_count'] + r['coll_count'] for r in prover_results]
ax1.bar(x, [r['perp_count'] for r in prover_results], bottom=bottoms2, 
        label='Perpendicular', color='#FF9800')
bottoms3 = [b + r['perp_count'] for b, r in zip(bottoms2, prover_results)]
ax1.bar(x, [r['para_count'] for r in prover_results], bottom=bottoms3, 
        label='Parallel', color='#E91E63')
bottoms4 = [b + r['para_count'] for b, r in zip(bottoms3, prover_results)]
ax1.bar(x, [r['cyclic_count'] for r in prover_results], bottom=bottoms4, 
        label='Cyclic', color='#9C27B0')
ax1.set_xticks(x)
ax1.set_xticklabels(names_short, rotation=90, fontsize=7)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Geometric Fact Types per Problem', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')

ax2 = axes[1]
ax2.bar(x, [r['total_facts'] for r in prover_results], color='#607D8B', edgecolor='white')
ax2.set_xticks(x)
ax2.set_xticklabels(names_short, rotation=90, fontsize=7)
ax2.set_ylabel('Total Facts', fontsize=12)
ax2.set_title('Total Facts in Knowledge Base per Problem', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/fact_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure: Points per problem
fig, ax = plt.subplots(figsize=(12, 6))
num_points = [r['num_points'] for r in prover_results]
sorted_idx = np.argsort(num_points)[::-1]
ax.barh(range(len(num_points)), [num_points[i] for i in sorted_idx], color='#00BCD4', edgecolor='white')
ax.set_yticks(range(len(names_short)))
ax.set_yticklabels([names_short[i] for i in sorted_idx], fontsize=8)
ax.set_xlabel('Number of Geometric Points', fontsize=14)
ax.set_title('Number of Points per Problem', fontsize=16, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_dir}/points_per_problem.png', dpi=150, bbox_inches='tight')
plt.close()

print("All additional figures generated.")
