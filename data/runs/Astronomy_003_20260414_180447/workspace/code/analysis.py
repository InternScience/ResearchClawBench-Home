import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
np.random.seed(42)

data_dir = 'data'
images_dir = 'report/images'
os.makedirs(images_dir, exist_ok=True)

# Load data
df6 = pd.read_csv(f'{data_dir}/fig6_data.csv')
df7 = pd.read_csv(f'{data_dir}/fig7_data.csv')
df8 = pd.read_csv(f'{data_dir}/fig8_data.csv')

# Log10 transforms
df6['log10_diff'] = np.log10(df6['waveform_difference'])
df7_log = df7.apply(np.log10)
df8_log = df8.apply(np.log10)

# Lognormal fits
fits = {}
fits['fig6'] = dict(zip(['s', 'loc', 'scale'], stats.lognorm.fit(df6['waveform_difference'])))
for col in df7.columns:
    fits[f'fig7_{col}'] = dict(zip(['s', 'loc', 'scale'], stats.lognorm.fit(df7[col])))
for col in df8.columns:
    fits[f'fig8_{col}'] = dict(zip(['s', 'loc', 'scale'], stats.lognorm.fit(df8[col])))

with open('outputs/lognormal_fits.json', 'w') as f:
    json.dump(fits, f, indent=2)

print('Saved lognormal fits')

# Fig6: Histogram
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(data=df6, x='log10_diff', bins=50, kde=True, ax=ax)
ax.axvline(np.log10(np.median(df6['waveform_difference'])), color='red', ls='--', label='Median')
ax.set_xlabel('log10(Waveform Difference)')
ax.set_ylabel('Count')
ax.legend()
plt.savefig(f'{images_dir}/fig6_hist.png', bbox_inches='tight')
plt.close()
print('Saved fig6_hist.png')

# Fig7: Violin per mode
fig, ax = plt.subplots(figsize=(10,6))
sns.violinplot(data=df7_log, ax=ax)
ax.set_xlabel('Spherical Harmonic Mode ℓ')
ax.set_ylabel('log10(Difference)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{images_dir}/fig7_modes.png', bbox_inches='tight')
plt.close()
print('Saved fig7_modes.png')

# Alternative boxplot
fig, ax = plt.subplots(figsize=(10,6))
sns.boxplot(data=df7_log, ax=ax)
ax.set_xlabel('ℓ')
ax.set_ylabel('log10(Diff)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{images_dir}/fig7_box.png', bbox_inches='tight')
plt.close()

# Fig8: Scatter N2vsN3 vs N2vsN4
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df8_log, x='N2vsN3', y='N2vsN4', alpha=0.5, ax=ax)
ax.axline((0,0), slope=1, color='r', ls='--', label='1:1')
med_n3 = np.median(df8_log['N2vsN3'])
med_n4 = np.median(df8_log['N2vsN4'])
ax.axvline(med_n3, color='g', ls=':', label=f'Med N2-N3: {med_n3:.2e}')
ax.axhline(med_n4, color='orange', ls=':', label=f'Med N2-N4: {med_n4:.2e}')
ax.set_xlabel('log10(N2 vs N3)')
ax.set_ylabel('log10(N2 vs N4)')
ax.legend()
plt.savefig(f'{images_dir}/fig8_extrap.png', bbox_inches='tight')
plt.close()
print('Saved fig8_extrap.png')

# Overview: all log10 diffs histograms
fig, axes = plt.subplots(1,3, figsize=(15,5))
sns.histplot(df6['log10_diff'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Fig6: Resolution Mismatch')
sns.histplot(df7_log, bins=50, ax=axes[1])
axes[1].set_title('Fig7: Per-Mode')
sns.histplot(df8_log, bins=50, ax=axes[2])
axes[2].set_title('Fig8: Extrapolation')
plt.tight_layout()
plt.savefig(f'{images_dir}/overview_hists.png', bbox_inches='tight')
plt.close()
print('Saved overview_hists.png')

print('All figures generated!')