import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load data
with open('outputs/data.json', 'r') as f:
    data = json.load(f)

# Ensure dirs
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Atomic radii bar plot
atomic_radii = data['atomic_radii']
df_radii = pd.DataFrame(atomic_radii, columns=['Element', 'Radius (Å)'])
plt.figure(figsize=(8,5))
sns.barplot(data=df_radii, x='Element', y='Radius (Å)')
plt.title('Atomic Radii of Elements')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/images/atomic_radii.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Optimal mismatch ranges heatmap
ranges = data['optimal_mismatch_ranges']
df_ranges = pd.DataFrame(ranges, columns=['Shell1', 'Shell2', 'Min', 'Max'])
pivot_min = df_ranges.pivot(index='Shell1', columns='Shell2', values='Min')
pivot_max = df_ranges.pivot(index='Shell1', columns='Shell2', values='Max')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
sns.heatmap(pivot_min, annot=True, ax=ax1, cmap='YlOrRd')
ax1.set_title('Optimal Size Mismatch Min')
sns.heatmap(pivot_max, annot=True, ax=ax2, cmap='YlOrRd')
ax2.set_title('Optimal Size Mismatch Max')
plt.tight_layout()
plt.savefig('report/images/mismatch_ranges.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Shell energies
energies = data['shell_energies']
df_energies = pd.DataFrame(energies, columns=['Shell', 'Category', 'Energy'])
plt.figure(figsize=(10,6))
sns.barplot(data=df_energies, x='Shell', y='Energy', hue='Category')
plt.title('Relative Shell Energies')
plt.tight_layout()
plt.savefig('report/images/shell_energies.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Growth results: mismatch evolution
growth = data['growth_results']
df_growth = pd.DataFrame(growth, columns=['Steps', 'Category', 'Mismatch'])
plt.figure(figsize=(10,6))
sns.lineplot(data=df_growth, x='Steps', y='Mismatch', hue='Category', marker='o')
plt.title('Growth Simulation: Average Mismatch vs Steps')
plt.xlabel('Simulation Steps')
plt.ylabel('Average Size Mismatch')
plt.tight_layout()
plt.savefig('report/images/growth_results.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Path selection stats pie
paths = data['path_selection_stats']
df_paths = pd.DataFrame(paths, columns=['Path', 'Count'])
plt.figure(figsize=(8,8))
plt.pie(df_paths['Count'], labels=df_paths['Path'], autopct='%1.1f%%')
plt.title('Path Selection Statistics')
plt.savefig('report/images/path_stats.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Mackay vs New sequences
mackay = data['mackay_sequence']
new_b5 = data['new_sequence_b5']
shells = list(range(1, len(mackay)+1))
plt.figure(figsize=(8,5))
plt.plot(shells, mackay, 'o-', label='Mackay (MC)')
plt.plot(shells, new_b5[:len(shells)], 's-', label='New b=5')
plt.xlabel('Shell Number')
plt.ylabel('Atoms per Shell')
plt.title('Icosahedral Shell Sequences')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('report/images/shell_sequences.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Multicomponent clusters table as image
clusters = data['multicomponent_clusters']
df_clusters = pd.DataFrame(clusters, columns=['Cluster', 'Inner Atom', 'Outer Atom', 'Inner Shell', 'Outer Shell'])
fig, ax = plt.subplots(figsize=(10,3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_clusters.values, colLabels=df_clusters.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.title('Predicted Stable Multi-Component Clusters')
plt.savefig('report/images/clusters_table.png', dpi=300, bbox_inches='tight')
plt.close()

# Save tables
df_radii.to_csv('outputs/atomic_radii.csv', index=False)
df_clusters.to_csv('outputs/multicomponent_clusters.csv', index=False)
df_ranges.to_csv('outputs/optimal_mismatch_ranges.csv', index=False)
df_growth.to_csv('outputs/growth_results.csv', index=False)

print("All plots and tables generated successfully!")