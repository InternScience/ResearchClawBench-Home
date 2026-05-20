"""
Generate validation and summary figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load all results
with open('outputs/property_prediction_results.json', 'r') as f:
    pp_results = json.load(f)

with open('outputs/structure_generation_stats.json', 'r') as f:
    sg_stats = json.load(f)

with open('outputs/optimization_results.json', 'r') as f:
    opt_results = json.load(f)

# Figure 1: Workflow comparison summary
fig = plt.figure(figsize=(14, 10))

# Property prediction comparison
ax1 = plt.subplot(2, 3, 1)
models = list(pp_results.keys())
rmse_vals = [pp_results[m]['RMSE'] for m in models]
r2_vals = [pp_results[m]['R2'] for m in models]
x = np.arange(len(models))
width = 0.35
bars1 = ax1.bar(x - width/2, rmse_vals, width, label='RMSE', color='steelblue', edgecolor='black')
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, r2_vals, width, label='R²', color='coral', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_ylabel('RMSE', color='steelblue')
ax1_twin.set_ylabel('R²', color='coral')
ax1.set_title('Property Prediction: Model Comparison')
ax1.set_ylim(0, max(rmse_vals)*1.2)
ax1_twin.set_ylim(0, 1)

# Feature importance (top 5)
ax2 = plt.subplot(2, 3, 2)
# Recalculate feature importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = np.load('outputs/processed_data.npz', allow_pickle=True)
lattice_dim = data['lattice_dim'][:97]
x_coords = data['x_coords'][:97]
atom_types = data['atom_types'][:97]
targets = data['targets']

X = pd.DataFrame({
    'lattice_dim': lattice_dim,
    'x_coord': x_coords,
    'atom_type': atom_types,
})
X['x_coord_sq'] = X['x_coord'] ** 2
X['x_coord_abs'] = np.abs(X['x_coord'])
X['sin_x'] = np.sin(X['x_coord'])
X['cos_x'] = np.cos(X['x_coord'])
X['atom_type_sq'] = X['atom_type'] ** 2
for at in sorted(X['atom_type'].unique()):
    X[f'atom_{at}'] = (X['atom_type'] == at).astype(int)
X['lattice_x'] = X['lattice_dim'] * X['x_coord']
X['lattice_atom'] = X['lattice_dim'] * X['atom_type']

rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X, targets)
importances = rf.feature_importances_
imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=True).tail(8)
ax2.barh(imp_df['feature'], imp_df['importance'], color='darkgreen', edgecolor='black')
ax2.set_xlabel('Importance')
ax2.set_title('Top Features (Property Prediction)')

# Structure generation: distribution stats comparison
ax3 = plt.subplot(2, 3, 3)
methods = ['Real', 'PCA Gen', 'KDE Gen']
mean_a = [sg_stats['real']['mean_a'], sg_stats['gen_pca']['mean_a'], sg_stats['gen_kde']['mean_a']]
std_a = [sg_stats['real']['std_a'], sg_stats['gen_pca']['std_a'], sg_stats['gen_kde']['std_a']]
x = np.arange(len(methods))
ax3.bar(x, mean_a, yerr=std_a, capsize=5, color=['blue', 'green', 'purple'], edgecolor='black', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(methods)
ax3.set_ylabel('Lattice a (Å)')
ax3.set_title('Structure Gen: Mean ± Std (a)')

# Optimization convergence
ax4 = plt.subplot(2, 3, 4)
history = opt_results['history']
ax4.plot(history['iteration'], history['best_obj'], 'o-', color='steelblue', lw=2, markersize=5)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Best Objective')
ax4.set_title('Bayesian Optimization Convergence')
ax4.grid(True, alpha=0.3)

# Optimization: target vs achieved
ax5 = plt.subplot(2, 3, 5)
categories = ['Yield', 'Time']
targets_plot = [opt_results['target_yield'], opt_results['target_time']]
achieved = [opt_results['best_yield'], opt_results['best_time']]
x = np.arange(len(categories))
width = 0.35
ax5.bar(x - width/2, targets_plot, width, label='Target', color='coral', edgecolor='black')
ax5.bar(x + width/2, achieved, width, label='Achieved', color='seagreen', edgecolor='black')
ax5.set_xticks(x)
ax5.set_xticklabels(categories)
ax5.set_ylabel('Value')
ax5.set_title('Optimization: Target vs Achieved')
ax5.legend()

# Summary metrics table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
Workflow Summary

Property Prediction:
• Best model: Random Forest
• Test RMSE: {pp_results['Random Forest']['RMSE']:.4f}
• Test R²: {pp_results['Random Forest']['R2']:.4f}

Structure Generation:
• Real mean a: {sg_stats['real']['mean_a']:.4f} Å
• Generated mean a: {sg_stats['gen_kde']['mean_a']:.4f} Å
• Correlation preserved: {abs(sg_stats['real']['corr'] - sg_stats['gen_kde']['corr']) < 0.1}

Synthesis Optimization:
• Best T: {opt_results['best_T']:.1f}°C
• Best P: {opt_results['best_P']:.1f} MPa
• Achieved yield: {opt_results['best_yield']:.4f}
• Achieved time: {opt_results['best_time']:.1f} h
"""
ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('report/images/figure_validation_summary.png', dpi=200, bbox_inches='tight')
plt.close()

print("Saved validation summary figure.")
