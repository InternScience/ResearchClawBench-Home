import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")

# Figure 6: Histogram of waveform differences
df6 = pd.read_csv('data/fig6_data.csv')
plt.figure(figsize=(8, 6))
sns.histplot(df6['waveform_difference'], bins=np.logspace(-6, 0, 50), log_scale=True, color='blue', alpha=0.7)
plt.axvline(df6['waveform_difference'].median(), color='red', linestyle='dashed', linewidth=2, label=f"Median: {df6['waveform_difference'].median():.1e}")
plt.xlabel('Waveform Difference (Mismatch)')
plt.ylabel('Count')
plt.title('Distribution of Waveform Differences (Highest Resolutions)')
plt.legend()
plt.tight_layout()
plt.savefig('report/images/fig6.png', dpi=300)
plt.close()

# Save stats
with open('outputs/fig6_stats.txt', 'w') as f:
    f.write(f"Median: {df6['waveform_difference'].median():.2e}\n")
    f.write(f"Mean: {df6['waveform_difference'].mean():.2e}\n")
    f.write(f"95th percentile: {df6['waveform_difference'].quantile(0.95):.2e}\n")

# Figure 7: Modal error distributions (ell=2 to ell=8)
df7 = pd.read_csv('data/fig7_data.csv')
plt.figure(figsize=(10, 6))
df7_melted = df7.melt(var_name='Mode', value_name='Difference')
sns.boxplot(x='Mode', y='Difference', data=df7_melted, color='lightblue', showfliers=False)
plt.yscale('log')
plt.xlabel('Spherical Harmonic Mode (ℓ)')
plt.ylabel('Waveform Difference')
plt.title('Modal Error Distributions (ℓ=2 to 8)')
plt.tight_layout()
plt.savefig('report/images/fig7.png', dpi=300)
plt.close()

# Save stats
df7.median().to_csv('outputs/fig7_medians.csv')

# Figure 8: Extrapolation order comparisons
df8 = pd.read_csv('data/fig8_data.csv')
plt.figure(figsize=(8, 6))
sns.kdeplot(df8['N2vsN3'], log_scale=True, label='N=2 vs N=3', color='green', fill=True, alpha=0.3)
sns.kdeplot(df8['N2vsN4'], log_scale=True, label='N=2 vs N=4', color='orange', fill=True, alpha=0.3)
plt.axvline(df8['N2vsN3'].median(), color='green', linestyle='dashed', linewidth=1.5)
plt.axvline(df8['N2vsN4'].median(), color='orange', linestyle='dashed', linewidth=1.5)
plt.xlabel('Waveform Difference')
plt.ylabel('Density')
plt.title('Extrapolation Order Comparisons')
plt.legend()
plt.tight_layout()
plt.savefig('report/images/fig8.png', dpi=300)
plt.close()

# Save stats
with open('outputs/fig8_stats.txt', 'w') as f:
    f.write(f"N2vsN3 Median: {df8['N2vsN3'].median():.2e}\n")
    f.write(f"N2vsN4 Median: {df8['N2vsN4'].median():.2e}\n")

print("Plots and stats generated successfully.")
