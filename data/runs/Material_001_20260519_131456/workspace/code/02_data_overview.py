"""
Generate data overview visualizations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load data
data = np.load('outputs/parsed_data.npz', allow_pickle=True)
lattice_dim = data['lattice_dim']
x_coords = data['x_coords']
atom_types = data['atom_types']
targets = data['targets']
a_vals = data['a_vals']
b_vals = data['b_vals']

# Ensure consistent length for property prediction
n_pp = min(len(lattice_dim), len(x_coords), len(targets))
lattice_dim = lattice_dim[:n_pp]
x_coords = x_coords[:n_pp]
targets = targets[:n_pp]

# Tile atom_types to match n_pp
atom_types_tiled = np.tile(atom_types, (n_pp // len(atom_types) + 1))[:n_pp]

# Ensure consistent length for structure generation
n_sg = min(len(a_vals), len(b_vals))
a_vals = a_vals[:n_sg]
b_vals = b_vals[:n_sg]

print(f"Property prediction samples: {n_pp}")
print(f"Structure generation samples: {n_sg}")

# Create comprehensive data overview figure
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 1. Target distribution
axes[0, 0].hist(targets, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Property Prediction: Target Distribution', fontsize=11)
axes[0, 0].set_xlabel('Target Energy/Property')
axes[0, 0].set_ylabel('Frequency')

# 2. X-coordinates vs Targets scatter
scatter = axes[0, 1].scatter(x_coords, targets, c=atom_types_tiled, cmap='tab10', alpha=0.7, edgecolors='k', linewidths=0.3)
axes[0, 1].set_title('X-coords vs Target (colored by atom type)', fontsize=11)
axes[0, 1].set_xlabel('X Coordinate')
axes[0, 1].set_ylabel('Target Energy/Property')
cbar = plt.colorbar(scatter, ax=axes[0, 1])
cbar.set_label('Atom Type')

# 3. Atom type distribution
atom_counts = pd.Series(atom_types_tiled).value_counts().sort_index()
axes[0, 2].bar(atom_counts.index, atom_counts.values, color='coral', edgecolor='black')
axes[0, 2].set_title('Atom Type Distribution', fontsize=11)
axes[0, 2].set_xlabel('Atom Type')
axes[0, 2].set_ylabel('Count')

# 4. Lattice constants scatter
axes[1, 0].scatter(a_vals, b_vals, alpha=0.6, color='green', edgecolors='k', linewidths=0.3)
axes[1, 0].set_title('Structure Generation: Lattice Constants', fontsize=11)
axes[1, 0].set_xlabel('Lattice Constant a (Å)')
axes[1, 0].set_ylabel('Lattice Constant b (Å)')

# 5. Lattice constant distributions
axes[1, 1].hist(a_vals, bins=15, alpha=0.6, label='a', color='green', edgecolor='black')
axes[1, 1].hist(b_vals, bins=15, alpha=0.6, label='b', color='orange', edgecolor='black')
axes[1, 1].set_title('Lattice Constant Distributions', fontsize=11)
axes[1, 1].set_xlabel('Lattice Constant (Å)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

# 6. Correlation heatmap for property prediction
pp_df = pd.DataFrame({
    'lattice_dim': lattice_dim,
    'x_coord': x_coords,
    'target': targets
})
corr = pp_df.corr()
im = axes[1, 2].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1, 2].set_xticks(range(len(corr.columns)))
axes[1, 2].set_yticks(range(len(corr.columns)))
axes[1, 2].set_xticklabels(corr.columns, rotation=45, ha='right')
axes[1, 2].set_yticklabels(corr.columns)
axes[1, 2].set_title('Feature Correlation Matrix', fontsize=11)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        axes[1, 2].text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('report/images/figure_data_overview.png', dpi=200, bbox_inches='tight')
plt.close()

print("Saved report/images/figure_data_overview.png")

# Save processed data for downstream
np.savez('outputs/processed_data.npz',
         lattice_dim=lattice_dim,
         x_coords=x_coords,
         atom_types=atom_types_tiled,
         targets=targets,
         a_vals=a_vals,
         b_vals=b_vals)

print("Saved outputs/processed_data.npz")
