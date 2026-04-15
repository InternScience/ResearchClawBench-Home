"""
Scenario Comparison Figure
"""
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = "outputs"
FIG_DIR = "report/images"

with open(f"{OUT_DIR}/scenario_comparison.json") as f:
    scenarios = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

labels = [s['scenario'] for s in scenarios]
x = np.arange(len(labels))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

# Total cost
ax = axes[0, 0]
costs = [s['total_cost_billion_gbp'] for s in scenarios]
bars = ax.bar(x, costs, color=colors, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Billion £')
ax.set_title('Total System Cost')
ax.grid(True, alpha=0.3, axis='y')

# Shedding percentage
ax = axes[0, 1]
shed = [s['shedding_pct'] for s in scenarios]
bars = ax.bar(x, shed, color=colors, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('%')
ax.set_title('Load Shedding (% of Demand)')
ax.grid(True, alpha=0.3, axis='y')

# Generation mix
ax = axes[1, 0]
width = 0.2
carriers = ['wind_generation_gwh', 'gas_generation_gwh', 'nuclear_generation_gwh']
carrier_labels = ['Wind', 'Gas', 'Nuclear']
carrier_colors = ['#2ecc71', '#e74c3c', '#9b59b6']
for i, (c, cl, cc) in enumerate(zip(carriers, carrier_labels, carrier_colors)):
    vals = [s[c] for s in scenarios]
    ax.bar(x + i*width - width, vals, width, label=cl, color=cc, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('GWh')
ax.set_title('Generation by Carrier')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Curtailment and storage
ax = axes[1, 1]
curt = [s['curtailment_gwh'] for s in scenarios]
sto = [s['storage_discharge_gwh'] for s in scenarios]
ax.bar(x - 0.15, curt, 0.3, label='Curtailment', color='#f39c12', alpha=0.8)
ax.bar(x + 0.15, sto, 0.3, label='Storage discharge', color='#1abc9c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('GWh')
ax.set_title('Curtailment and Storage Discharge')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/scenario_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Scenario comparison figure saved.")
