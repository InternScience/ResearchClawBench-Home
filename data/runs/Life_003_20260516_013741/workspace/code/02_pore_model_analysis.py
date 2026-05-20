#!/usr/bin/env python3
"""Pore model analysis: compare k-mer models across chemistries."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import Counter

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load all pore model data
dna_r9 = pd.read_csv('data/dna_r9.4.1_400bps_6mer_uncalled4.csv')
dna_r10 = pd.read_csv('data/dna_r10.4.1_400bps_9mer_uncalled4.csv')
rna_r9 = pd.read_csv('data/rna_r9.4.1_70bps_5mer_uncalled4.csv')
rna004 = pd.read_csv('data/rna004_130bps_9mer_uncalled4.csv')

print(f"DNA R9.4.1 (6-mer): {len(dna_r9)} kmers")
print(f"DNA R10.4.1 (9-mer): {len(dna_r10)} kmers")
print(f"RNA R9.4.1 (5-mer): {len(rna_r9)} kmers")
print(f"RNA004 (9-mer): {len(rna004)} kmers")

# === Basic Statistics ===
datasets = {
    'DNA R9.4.1 (6-mer)': dna_r9,
    'DNA R10.4.1 (9-mer)': dna_r10,
    'RNA R9.4.1 (5-mer)': rna_r9,
    'RNA004 (9-mer)': rna004
}

stats_rows = []
for name, df in datasets.items():
    stats_rows.append({
        'Model': name,
        'N_kmers': len(df),
        'Current_Mean_Mean': df['current_mean'].mean(),
        'Current_Mean_Std': df['current_mean'].std(),
        'Current_Std_Mean': df['current_std'].mean(),
        'Current_Std_Std': df['current_std'].std(),
        'Dwell_Time_Mean': df['dwell_time'].mean(),
        'Dwell_Time_Std': df['dwell_time'].std(),
        'Current_Range_Min': df['current_mean'].min(),
        'Current_Range_Max': df['current_mean'].max(),
    })

stats_df = pd.DataFrame(stats_rows)
print("\nPore Model Statistics:")
print(stats_df.to_string(index=False))
stats_df.to_csv('outputs/pore_model_statistics.csv', index=False)

# === Figure 3: Current Distribution Comparison ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, df) in zip(axes.flat, datasets.items()):
    ax.hist(df['current_mean'], bins=80, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(df['current_mean'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f"Mean: {df['current_mean'].mean():.2f}")
    ax.set_xlabel('Mean Current (pA)')
    ax.set_ylabel('Frequency')
    ax.set_title(name)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('report/images/fig3_current_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: fig3_current_distributions.png")

# === Figure 4: Base-position effect on current ===
# For each k-mer, analyze the effect of each position's nucleotide on current
def analyze_position_effects(df, k, name):
    nucleotides = ['A', 'C', 'G', 'T']
    pos_effects = {}
    
    for pos in range(k):
        means_by_base = {}
        for base in nucleotides:
            mask = df['kmer'].str[pos] == base
            means_by_base[base] = df[mask]['current_mean'].mean()
        pos_effects[f'pos_{pos+1}'] = means_by_base
    
    return pd.DataFrame(pos_effects)

# Analyze position effects for all models
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, df) in enumerate(datasets.items()):
    k = len(df['kmer'].iloc[0])
    pos_df = analyze_position_effects(df, k, name)
    
    x = np.arange(k)
    width = 0.2
    colors = {'A': '#E41A1C', 'C': '#377EB8', 'G': '#4DAF4A', 'T': '#FFD92F'}
    
    ax = axes[idx]
    for i, base in enumerate(['A', 'C', 'G', 'T']):
        ax.bar(x + i*width, pos_df.loc[base], width, label=base, color=colors[base], alpha=0.8)
    
    ax.set_xlabel('Position in k-mer')
    ax.set_ylabel('Mean Current (pA)')
    ax.set_title(f'{name} - Base-Position Effect')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([f'{i+1}' for i in range(k)])
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig4_position_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: fig4_position_effects.png")

# === Figure 5: DNA R9 vs R10 current comparison (common 6-mers via central match) ===
# For R10.4.1 9-mers, extract the central 6-mer for comparison
dna_r10['central_6mer'] = dna_r10['kmer'].str[1:7]  # positions 2-7 (0-indexed)
dna_r9_kmers = set(dna_r9['kmer'])
dna_r10['is_common'] = dna_r10['central_6mer'].isin(dna_r9_kmers)
common_r10 = dna_r10[dna_r10['is_common']].copy()
print(f"\nDNA R10.4.1 9-mers with common central 6-mer in R9: {len(common_r10)}/{len(dna_r10)}")

# Compare currents for common central 6-mers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot
ax = axes[0]
r9_lookup = dna_r9.set_index('kmer')['current_mean'].to_dict()
common_r10['r9_current'] = common_r10['central_6mer'].map(r9_lookup)

ax.scatter(common_r10['r9_current'], common_r10['current_mean'], alpha=0.3, s=1, c='steelblue')
min_val = min(common_r10['r9_current'].min(), common_r10['current_mean'].min())
max_val = max(common_r10['r9_current'].max(), common_r10['current_mean'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='y=x')
ax.set_xlabel('R9.4.1 6-mer Mean Current (pA)')
ax.set_ylabel('R10.4.1 9-mer Mean Current (pA)')
ax.set_title(f'DNA Pore Model Correlation\n(r={common_r10["r9_current"].corr(common_r10["current_mean"]):.3f})')
ax.legend()
ax.grid(alpha=0.3)

# Flanking base effect on current shift
ax = axes[1]
common_r10['current_shift'] = common_r10['current_mean'] - common_r10['r9_current']
flank_5p = common_r10['kmer'].str[0]
flank_3p = common_r10['kmer'].str[-1]

# Group by flanking bases
flank_data = []
for base5 in ['A','C','G','T']:
    for base3 in ['A','C','G','T']:
        mask = (flank_5p == base5) & (flank_3p == base3)
        if mask.sum() > 0:
            flank_data.append({
                '5\' Flank': base5,
                '3\' Flank': base3,
                'Mean Current Shift': common_r10[mask]['current_shift'].mean(),
                'Count': mask.sum()
            })

flank_df = pd.DataFrame(flank_data)
flank_pivot = flank_df.pivot(index='5\' Flank', columns='3\' Flank', values='Mean Current Shift')

sns.heatmap(flank_pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            cbar_kws={'label': 'Mean Current Shift (pA)'})
ax.set_title('R10.4.1 Flanking Base Effect on Current Shift')

plt.tight_layout()
plt.savefig('report/images/fig5_dna_pore_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: fig5_dna_pore_comparison.png")

# === Figure 6: RNA R9 vs RNA004 comparison ===
rna_r9_kmers = set(rna_r9['kmer'])
rna004['central_5mer'] = rna004['kmer'].str[2:7]  # positions 3-7 (0-indexed)
rna004['is_common'] = rna004['central_5mer'].isin(rna_r9_kmers)
common_rna = rna004[rna004['is_common']].copy()
print(f"RNA004 9-mers with common central 5-mer in RNA001: {len(common_rna)}/{len(rna004)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

rna_lookup = rna_r9.set_index('kmer')['current_mean'].to_dict()
common_rna['r9_current'] = common_rna['central_5mer'].map(rna_lookup)

ax = axes[0]
ax.scatter(common_rna['r9_current'], common_rna['current_mean'], alpha=0.3, s=1, c='darkorange')
min_val = min(common_rna['r9_current'].min(), common_rna['current_mean'].min())
max_val = max(common_rna['r9_current'].max(), common_rna['current_mean'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='y=x')
ax.set_xlabel('RNA001 5-mer Mean Current (pA)')
ax.set_ylabel('RNA004 9-mer Mean Current (pA)')
ax.set_title(f'RNA Pore Model Correlation\n(r={common_rna["r9_current"].corr(common_rna["current_mean"]):.3f})')
ax.legend()
ax.grid(alpha=0.3)

# Dwell time comparison
ax = axes[1]
datasets_dwell = {
    'DNA R9.4.1\n(6-mer)': dna_r9['dwell_time'],
    'DNA R10.4.1\n(9-mer)': dna_r10['dwell_time'],
    'RNA R9.4.1\n(5-mer)': rna_r9['dwell_time'],
    'RNA004\n(9-mer)': rna004['dwell_time'],
}
box_data = [d for d in datasets_dwell.values()]
labels = list(datasets_dwell.keys())
bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
colors_box = ['#2196F3', '#1565C0', '#FF9800', '#E65100']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Dwell Time (arbitrary units)')
ax.set_title('Dwell Time Distribution by Chemistry')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig6_rna_pore_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: fig6_rna_pore_comparison.png")

# === Figure 7: GC Content vs Current ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, df) in enumerate(datasets.items()):
    k = len(df['kmer'].iloc[0])
    df_temp = df.copy()
    df_temp['gc_count'] = df_temp['kmer'].apply(lambda x: x.count('G') + x.count('C'))
    df_temp['gc_fraction'] = df_temp['gc_count'] / k
    
    ax = axes[idx]
    sc = ax.scatter(df_temp['gc_fraction'], df_temp['current_mean'], 
                    c=df_temp['dwell_time'], cmap='viridis', alpha=0.5, s=2)
    ax.set_xlabel('GC Fraction')
    ax.set_ylabel('Mean Current (pA)')
    ax.set_title(name)
    plt.colorbar(sc, ax=ax, label='Dwell Time')

plt.tight_layout()
plt.savefig('report/images/fig7_gc_vs_current.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 7 saved: fig7_gc_vs_current.png")

# Save summary JSON
pore_summary = {
    'dna_r9_n_kmers': len(dna_r9),
    'dna_r10_n_kmers': len(dna_r10),
    'rna_r9_n_kmers': len(rna_r9),
    'rna004_n_kmers': len(rna004),
    'dna_correlation_r9_r10': float(common_r10['r9_current'].corr(common_r10['current_mean'])),
    'rna_correlation_r9_r004': float(common_rna['r9_current'].corr(common_rna['current_mean'])),
}
with open('outputs/pore_model_summary.json', 'w') as f:
    json.dump(pore_summary, f, indent=2, default=str)

print("\nPore model analysis complete.")
