#!/usr/bin/env python3
"""Performance benchmark analysis: Table 1 reproduction and visualization."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load performance data
perf = pd.read_csv('data/performance_summary.csv')
print("Raw performance data:")
print(perf)
print()

# Clean data
perf['Tool'] = perf['Tool'].str.strip()
perf['Chemistry'] = perf['Chemistry'].str.strip()

# Save cleaned data
perf.to_csv('outputs/performance_cleaned.csv', index=False)

# Create a summary table
summary = perf.pivot(index='Chemistry', columns='Tool', values='Time_min')
file_sizes = perf.pivot(index='Chemistry', columns='Tool', values='FileSize_MB')
print("Time (min):")
print(summary)
print()
print("File Size (MB):")
print(file_sizes)
print()

# Save pivoted tables
summary.to_csv('outputs/performance_time_pivot.csv')
file_sizes.to_csv('outputs/performance_filesize_pivot.csv')

# Compute speedup of Uncalled4 vs others
tools = ['f5c', 'Nanopolish', 'Tombo']
speedups = {}
for tool in tools:
    speedup_vals = []
    for chem in perf['Chemistry'].unique():
        u4_time = perf[(perf['Chemistry'] == chem) & (perf['Tool'] == 'Uncalled4')]['Time_min'].values
        t_time = perf[(perf['Chemistry'] == chem) & (perf['Tool'] == tool)]['Time_min'].values
        if len(u4_time) > 0 and len(t_time) > 0 and not np.isnan(t_time[0]) and t_time[0] > 0:
            speedup_vals.append({'Chemistry': chem, 'Speedup': float(t_time[0] / u4_time[0])})
    if speedup_vals:
        speedups[tool] = speedup_vals

# Print speedups
for tool, vals in speedups.items():
    for v in vals:
        print(f"Uncalled4 vs {tool} on {v['Chemistry']}: {v['Speedup']:.1f}x faster")

# Save speedup data
all_speedup_rows = []
for tool, vals in speedups.items():
    for v in vals:
        all_speedup_rows.append({'Tool': tool, 'Chemistry': v['Chemistry'], 'Speedup': v['Speedup']})
speedup_df = pd.DataFrame(all_speedup_rows)
speedup_df.to_csv('outputs/performance_speedups.csv', index=False)

# === Figure 1: Performance Comparison Bar Chart ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color palette
palette = {'Uncalled4': '#2196F3', 'f5c': '#FF9800', 'Nanopolish': '#F44336', 'Tombo': '#9C27B0'}

chem_order = ['DNA r9.4', 'DNA r10.4', 'RNA001', 'RNA004']
tool_order = ['Uncalled4', 'f5c', 'Nanopolish', 'Tombo']

time_data = perf.pivot(index='Chemistry', columns='Tool', values='Time_min')
time_data = time_data.reindex(chem_order)[tool_order]
fsize_data = perf.pivot(index='Chemistry', columns='Tool', values='FileSize_MB')
fsize_data = fsize_data.reindex(chem_order)[tool_order]

x = np.arange(len(chem_order))
width = 0.2

# Time plot
ax = axes[0]
for i, tool in enumerate(tool_order):
    vals = time_data[tool].values
    bars = ax.bar(x + i * width, vals, width, label=tool, color=palette[tool])
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{val:.0f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xlabel('Sequencing Chemistry')
ax.set_ylabel('Run Time (minutes)')
ax.set_title('Alignment Run Time by Chemistry')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(chem_order)
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

# File size plot
ax = axes[1]
for i, tool in enumerate(tool_order):
    vals = fsize_data[tool].values
    bars = ax.bar(x + i * width, vals, width, label=tool, color=palette[tool])
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                    f'{val:.0f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xlabel('Sequencing Chemistry')
ax.set_ylabel('Output File Size (MB)')
ax.set_title('Output File Size by Chemistry')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(chem_order)
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig1_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: fig1_performance_comparison.png")

# === Figure 2: Speedup Heatmap ===
fig, ax = plt.subplots(figsize=(8, 4))

su_pivot = speedup_df.pivot(index='Chemistry', columns='Tool', values='Speedup')
su_pivot = su_pivot.reindex(chem_order)

sns.heatmap(su_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Speedup Factor'})
ax.set_title('Uncalled4 Speedup vs. Other Tools')
ax.set_ylabel('Sequencing Chemistry')
ax.set_xlabel('Comparison Tool')

plt.tight_layout()
plt.savefig('report/images/fig2_speedup_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: fig2_speedup_heatmap.png")

# Save performance summary JSON for report
perf_summary = {
    'time_min': {k: {kk: vv for kk, vv in v.items() if not (isinstance(vv, float) and np.isnan(vv))} 
                 for k, v in summary.to_dict().items()},
    'file_size_mb': {k: {kk: vv for kk, vv in v.items() if not (isinstance(vv, float) and np.isnan(vv))} 
                     for k, v in file_sizes.to_dict().items()},
    'speedups': {tool: {v['Chemistry']: v['Speedup'] for v in vals} for tool, vals in speedups.items()}
}
with open('outputs/performance_summary.json', 'w') as f:
    json.dump(perf_summary, f, indent=2, default=str)

print("\nPerformance analysis complete.")
