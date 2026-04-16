"""
Phase 5: Analysis of Optimization Results from Experimental Data
Analyze the actual optimization trajectory from the provided datasets.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load training data
df = pd.read_csv('outputs/training_data_184.csv')
monomer_cols = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
target_col = 'Glass (kPa)_10s'
short_names = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']

# ============================================================
# Load optimization data
# ============================================================
opt1_ei = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='EI')
opt1_pred = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name='PRED')

# Forward fill ML column
opt1_ei['ML'] = opt1_ei['ML'].ffill()
opt1_pred['ML'] = opt1_pred['ML'].ffill()

# Convert numeric columns
for col in monomer_cols + ['Glass (kPa)_max']:
    opt1_ei[col] = pd.to_numeric(opt1_ei[col], errors='coerce')
    opt1_pred[col] = pd.to_numeric(opt1_pred[col], errors='coerce')

print("EI methods:", opt1_ei['ML'].unique())
print("PRED methods:", opt1_pred['ML'].unique())
print(f"EI data shape: {opt1_ei.shape}")
print(f"PRED data shape: {opt1_pred.shape}")

# ============================================================
# Analyze by method and round
# ============================================================
# Identify round from method name
def get_round(method_name):
    if '2rd' in method_name or '2nd' in method_name:
        return 2
    elif '3rd' in method_name:
        return 3
    else:
        return 1

opt1_ei['round'] = opt1_ei['ML'].apply(get_round)
opt1_pred['round'] = opt1_pred['ML'].apply(get_round)

# Extract base method name
def get_base_method(method_name):
    for m in ['RFR-GP', 'RFR-RFR', 'GP-GP', 'GP-RFR', 'ENU-RFR', 'ENU-GP', 'CLMax', 'CLMin', 'LP_df', 'old-SM-GP', 'SM-ETR', 'SM-GBM']:
        if m in method_name:
            return m
    return method_name

opt1_ei['base_method'] = opt1_ei['ML'].apply(get_base_method)
opt1_pred['base_method'] = opt1_pred['ML'].apply(get_base_method)

# ============================================================
# Summary statistics by method
# ============================================================
print("\n=== EI-based Optimization Results ===")
ei_summary = opt1_ei.groupby(['base_method', 'round']).agg(
    n_samples=('Glass (kPa)_max', 'count'),
    mean_pred=('Glass (kPa)_max', 'mean'),
    max_pred=('Glass (kPa)_max', 'max'),
    std_pred=('Glass (kPa)_max', 'std')
).reset_index()
print(ei_summary.to_string())

print("\n=== PRED-based Optimization Results ===")
pred_summary = opt1_pred.groupby(['base_method', 'round']).agg(
    n_samples=('Glass (kPa)_max', 'count'),
    mean_pred=('Glass (kPa)_max', 'mean'),
    max_pred=('Glass (kPa)_max', 'max'),
    std_pred=('Glass (kPa)_max', 'std')
).reset_index()
print(pred_summary.to_string())

# Save summaries
ei_summary.to_csv('outputs/ei_summary_by_method.csv', index=False)
pred_summary.to_csv('outputs/pred_summary_by_method.csv', index=False)

# ============================================================
# Figure 11: Optimization results by method (EI)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# EI methods - Round 1
r1_ei = opt1_ei[opt1_ei['round'] == 1]
methods_r1 = r1_ei['base_method'].unique()
data_r1 = [r1_ei[r1_ei['base_method'] == m]['Glass (kPa)_max'].values for m in methods_r1]

bp1 = axes[0].boxplot(data_r1, labels=methods_r1, patch_artist=True, showmeans=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(methods_r1)))
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
axes[0].set_ylabel('Predicted Adhesive Strength (kPa)', fontsize=11)
axes[0].set_title('Round 1: EI-based Optimization', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# EI methods - all rounds for key methods
key_methods = ['RFR-GP', 'GP-GP', 'RFR-RFR']
for km in key_methods:
    if km not in opt1_ei['base_method'].values:
        print(f"  Warning: {km} not found in EI data")

# Plot by round for key methods
r1_data = {}
r2_data = {}
r3_data = {}
for km in key_methods:
    r1_data[km] = opt1_ei[(opt1_ei['base_method'] == km) & (opt1_ei['round'] == 1)]['Glass (kPa)_max'].values
    r2_data[km] = opt1_ei[(opt1_ei['base_method'] == km) & (opt1_ei['round'] == 2)]['Glass (kPa)_max'].values
    r3_data[km] = opt1_ei[(opt1_ei['base_method'] == km) & (opt1_ei['round'] == 3)]['Glass (kPa)_max'].values

x_positions = np.arange(len(key_methods))
width = 0.25

r1_means = [r1_data[km].mean() if len(r1_data[km]) > 0 else 0 for km in key_methods]
r2_means = [r2_data[km].mean() if len(r2_data[km]) > 0 else 0 for km in key_methods]
r3_means = [r3_data[km].mean() if len(r3_data[km]) > 0 else 0 for km in key_methods]

r1_maxs = [r1_data[km].max() if len(r1_data[km]) > 0 else 0 for km in key_methods]
r2_maxs = [r2_data[km].max() if len(r2_data[km]) > 0 else 0 for km in key_methods]
r3_maxs = [r3_data[km].max() if len(r3_data[km]) > 0 else 0 for km in key_methods]

axes[1].bar(x_positions - width, r1_maxs, width, label='Round 1', color='#2196F3', alpha=0.8)
axes[1].bar(x_positions, r2_maxs, width, label='Round 2', color='#4CAF50', alpha=0.8)
axes[1].bar(x_positions + width, r3_maxs, width, label='Round 3', color='#FF9800', alpha=0.8)
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(key_methods, fontsize=11)
axes[1].set_ylabel('Max Predicted Adhesive Strength (kPa)', fontsize=11)
axes[1].set_title('Max Predictions by Method & Round', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig('report/images/fig11_optimization_by_method.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 11 saved.")

# ============================================================
# Figure 12: Composition comparison - Initial vs Optimized
# ============================================================
# Initial data compositions
init_comps = df[monomer_cols].mean()
init_comps_std = df[monomer_cols].std()

# Top optimized compositions (from RFR-GP round 1)
rfr_gp_r1 = opt1_ei[(opt1_ei['base_method'] == 'RFR-GP') & (opt1_ei['round'] == 1)]
if len(rfr_gp_r1) > 0:
    top10_idx = rfr_gp_r1['Glass (kPa)_max'].nlargest(10).index
    opt_comps = rfr_gp_r1.loc[top10_idx, monomer_cols].mean()
    opt_comps_std = rfr_gp_r1.loc[top10_idx, monomer_cols].std()
else:
    # Use GP-GP
    gp_gp_r1 = opt1_ei[(opt1_ei['base_method'] == 'GP-GP') & (opt1_ei['round'] == 1)]
    top10_idx = gp_gp_r1['Glass (kPa)_max'].nlargest(10).index
    opt_comps = gp_gp_r1.loc[top10_idx, monomer_cols].mean()
    opt_comps_std = gp_gp_r1.loc[top10_idx, monomer_cols].std()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart comparison
x = np.arange(len(short_names))
width = 0.35
axes[0].bar(x - width/2, init_comps.values, width, yerr=init_comps_std.values, 
            label='Initial (n=184)', color='#2196F3', alpha=0.8, capsize=3)
axes[0].bar(x + width/2, opt_comps.values, width, yerr=opt_comps_std.values,
            label='Optimized (Top 10)', color='#F44336', alpha=0.8, capsize=3)
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names, fontsize=11)
axes[0].set_ylabel('Mean Fraction', fontsize=12)
axes[0].set_title('Composition: Initial vs Optimized', fontsize=13)
axes[0].legend()

# Radar chart
angles = np.linspace(0, 2*np.pi, len(short_names), endpoint=False).tolist()
angles += angles[:1]

init_vals = init_comps.values.tolist() + [init_comps.values[0]]
opt_vals = opt_comps.values.tolist() + [opt_comps.values[0]]

ax2 = fig.add_subplot(122, polar=True)
ax2.fill(angles, init_vals, alpha=0.2, color='#2196F3')
ax2.plot(angles, init_vals, 'o-', color='#2196F3', lw=2, label='Initial')
ax2.fill(angles, opt_vals, alpha=0.2, color='#F44336')
ax2.plot(angles, opt_vals, 'o-', color='#F44336', lw=2, label='Optimized')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(short_names, fontsize=10)
ax2.set_title('Composition Radar Chart', fontsize=13, pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Remove the empty second subplot
axes[1].remove()

plt.tight_layout()
plt.savefig('report/images/fig12_composition_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 12 saved.")

# ============================================================
# Figure 13: Predicted strength distribution by round
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Collect data by round for RFR-GP
for round_num in [1, 2, 3]:
    round_data = opt1_ei[(opt1_ei['base_method'] == 'RFR-GP') & (opt1_ei['round'] == round_num)]
    if len(round_data) > 0:
        ax.hist(round_data['Glass (kPa)_max'], bins=15, alpha=0.5, 
                label=f'Round {round_num} (n={len(round_data)})')

# Also plot initial training data
ax.hist(df[target_col], bins=15, alpha=0.3, color='gray', label=f'Initial (n={len(df)})')
ax.set_xlabel('Predicted Adhesive Strength (kPa)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('RFR-GP: Predicted Strength Distribution by Round', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('report/images/fig13_strength_by_round.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 13 saved.")

# ============================================================
# Figure 14: Monomer contribution heatmap across optimization
# ============================================================
# Compare top formulations across methods
fig, ax = plt.subplots(figsize=(12, 8))

methods_to_compare = ['RFR-GP', 'GP-GP', 'RFR-RFR']
heatmap_data = []

for method in methods_to_compare:
    for round_num in [1, 2, 3]:
        subset = opt1_ei[(opt1_ei['base_method'] == method) & (opt1_ei['round'] == round_num)]
        if len(subset) > 0:
            top5 = subset.nlargest(5, 'Glass (kPa)_max')
            mean_comp = top5[monomer_cols].mean()
            heatmap_data.append([method, f'R{round_num}'] + mean_comp.values.tolist())

heatmap_df = pd.DataFrame(heatmap_data, columns=['Method', 'Round'] + short_names)
heatmap_matrix = heatmap_df[short_names].values
row_labels = [f"{r['Method']}_{r['Round']}" for _, r in heatmap_df.iterrows()]

sns.heatmap(heatmap_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=short_names, yticklabels=row_labels, ax=ax)
ax.set_title('Mean Composition of Top-5 Formulations by Method & Round', fontsize=13)
ax.set_ylabel('Method_Round', fontsize=11)
plt.tight_layout()
plt.savefig('report/images/fig14_composition_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 14 saved.")

# ============================================================
# Save key findings
# ============================================================
findings = {
    'initial_max_kPa': float(df[target_col].max()),
    'initial_mean_kPa': float(df[target_col].mean()),
    'initial_std_kPa': float(df[target_col].std()),
    'top_optimized_prediction_kPa': float(opt1_ei['Glass (kPa)_max'].max()),
    'key_monomers_for_adhesion': ['Hydrophobic-BA', 'Aromatic-PEA', 'Cationic-ATAC'],
    'optimal_composition_trend': 'High BA (0.5-0.6), High PEA (0.25-0.35), Low HEA (<0.1)',
    'correlation_HEA': -0.494,
    'correlation_BA': 0.443,
    'correlation_PEA': 0.276,
}
with open('outputs/key_findings.json', 'w') as f:
    json.dump(findings, f, indent=2)

print("\nPhase 5 complete.")
