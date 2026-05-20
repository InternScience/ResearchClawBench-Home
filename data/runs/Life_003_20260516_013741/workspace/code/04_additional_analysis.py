#!/usr/bin/env python3
"""Additional analyses: substitution profiles, nucleotide composition,
and deeper signal characterization."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
dna_r9 = pd.read_csv('data/dna_r9.4.1_400bps_6mer_uncalled4.csv')
dna_r10 = pd.read_csv('data/dna_r10.4.1_400bps_9mer_uncalled4.csv')
rna_r9 = pd.read_csv('data/rna_r9.4.1_70bps_5mer_uncalled4.csv')
rna004 = pd.read_csv('data/rna004_130bps_9mer_uncalled4.csv')

datasets = [
    ('DNA R9.4.1 (6-mer)', dna_r9),
    ('DNA R10.4.1 (9-mer)', dna_r10),
    ('RNA R9.4.1 (5-mer)', rna_r9),
    ('RNA004 (9-mer)', rna004),
]

# === Figure 12: Signal variability analysis ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, df) in enumerate(datasets):
    ax = axes[idx]
    sc = ax.scatter(df['current_mean'], df['current_std'], 
                    c=df['dwell_time'], cmap='plasma', alpha=0.3, s=1, rasterized=True)
    ax.set_xlabel('Mean Current')
    ax.set_ylabel('Current Std')
    ax.set_title(name + '\nSignal Mean vs. Variability')
    plt.colorbar(sc, ax=ax, label='Dwell Time')

plt.tight_layout()
plt.savefig('report/images/fig12_signal_variability.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 12 saved: fig12_signal_variability.png")

# === Figure 13: Nucleotide Composition Analysis ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
base_colors = {'A': '#E41A1C', 'C': '#377EB8', 'G': '#4DAF4A', 'T': '#FFD92F'}

for idx, (name, df) in enumerate(datasets):
    ax = axes[idx]
    df_temp = df.copy()
    
    for base in ['A', 'C', 'G', 'T']:
        col_name = 'count_' + base
        df_temp[col_name] = df_temp['kmer'].apply(lambda x, b=base: x.count(b))
    
    # Average current by base count
    for base in ['A', 'C', 'G', 'T']:
        col_name = 'count_' + base
        counts = df_temp.groupby(col_name)['current_mean'].agg(['mean', 'std']).reset_index()
        ax.errorbar(counts[col_name], counts['mean'], yerr=counts['std'], 
                    fmt='o-', color=base_colors[base], capsize=3, label=base, markersize=5, 
                    linewidth=1.5)
    
    ax.set_xlabel('Base Count in k-mer')
    ax.set_ylabel('Mean Current')
    ax.set_title(name + ': Current vs. Base Count')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig13_nucleotide_composition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 13 saved: fig13_nucleotide_composition.png")

# === Figure 14: Substitution Profile Heatmaps ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, df) in enumerate(datasets):
    ax = axes[idx]
    k = len(df['kmer'].iloc[0])
    
    # For each position, compute mean current per base
    pos_currents = []
    bases = ['A', 'C', 'G', 'T']
    
    for pos in range(k):
        row = []
        for base in bases:
            mask = df['kmer'].str[pos] == base
            row.append(df[mask]['current_mean'].mean())
        pos_currents.append(row)
    
    pos_currents = np.array(pos_currents)
    
    sns.heatmap(pos_currents.T, annot=False, cmap='RdBu_r', center=0,
                xticklabels=['Pos ' + str(i+1) for i in range(k)],
                yticklabels=bases, ax=ax, cbar_kws={'label': 'Mean Current'})
    ax.set_title(name)
    ax.set_ylabel('Nucleotide')

plt.tight_layout()
plt.savefig('report/images/fig14_substitution_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 14 saved: fig14_substitution_heatmaps.png")

# === Figure 15: k-mer size effect on signal resolution ===
fig, ax = plt.subplots(figsize=(10, 6))

data_pairs = [
    ('DNA R9.4.1\n6-mer', dna_r9['current_mean'], 'steelblue'),
    ('DNA R10.4.1\n9-mer', dna_r10['current_mean'], 'darkblue'),
    ('RNA R9.4.1\n5-mer', rna_r9['current_mean'], 'coral'),
    ('RNA004\n9-mer', rna004['current_mean'], 'darkred'),
]

for label, data, color in data_pairs:
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)
    ax.fill_between(x_range, kde(x_range), alpha=0.1, color=color)

ax.set_xlabel('Mean Current')
ax.set_ylabel('Density')
ax.set_title('Current Distribution by Pore Chemistry')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/fig15_kmer_size_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 15 saved: fig15_kmer_size_effect.png")

# === Summary Statistics for report ===
summary_stats = {}
for name, df in datasets:
    k = len(df['kmer'].iloc[0])
    summary_stats[name] = {
        'kmer_length': k,
        'n_unique_kmers': len(df),
        'current_mean_range': [float(df['current_mean'].min()), float(df['current_mean'].max())],
        'current_mean_mean': float(df['current_mean'].mean()),
        'current_mean_std': float(df['current_mean'].std()),
        'current_std_mean': float(df['current_std'].mean()),
        'dwell_time_mean': float(df['dwell_time'].mean()),
    }
    
    for base in ['A', 'C', 'G', 'T']:
        avg_count = np.mean([kmer.count(base) for kmer in df['kmer']])
        summary_stats[name]['avg_' + base + '_count'] = float(avg_count)

with open('outputs/comprehensive_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\nAdditional analysis complete.")
