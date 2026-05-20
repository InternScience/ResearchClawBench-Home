import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory
os.makedirs('report/images', exist_ok=True)

# Load data
fig6 = pd.read_csv('data/fig6_data.csv')
fig7 = pd.read_csv('data/fig7_data.csv')
fig8 = pd.read_csv('data/fig8_data.csv')

# Summary statistics
print("=== Fig6 (Overall Waveform Differences) ===")
print(fig6.describe())
print("Median:", fig6.median().values[0])

print("\n=== Fig7 (Modal Differences ℓ=2..8) ===")
print(fig7.describe())

print("\n=== Fig8 (Extrapolation Orders) ===")
print(fig8.describe())

# Plot 1: Fig6 histogram (log scale)
plt.figure(figsize=(8,5))
sns.histplot(np.log10(fig6.iloc[:,0]), bins=50, kde=True, color='steelblue')
plt.xlabel('log10(Waveform Difference)')
plt.ylabel('Count')
plt.title('Fig6: Distribution of Waveform Differences (log scale)')
plt.tight_layout()
plt.savefig('report/images/fig6_histogram.png', dpi=150)
plt.close()

# Plot 2: Fig7 boxplot per mode
plt.figure(figsize=(10,6))
sns.boxplot(data=fig7, palette='viridis')
plt.xlabel('Spherical Harmonic Mode ℓ')
plt.ylabel('Waveform Difference')
plt.title('Fig7: Modal Waveform Differences by ℓ')
plt.xticks(range(7), [f'ℓ={i}' for i in range(2,9)])
plt.yscale('log')
plt.tight_layout()
plt.savefig('report/images/fig7_boxplot.png', dpi=150)
plt.close()

# Plot 3: Fig8 comparison histograms
plt.figure(figsize=(8,5))
sns.histplot(np.log10(fig8['N2vsN3']), bins=40, label='N2 vs N3', alpha=0.6, color='coral')
sns.histplot(np.log10(fig8['N2vsN4']), bins=40, label='N2 vs N4', alpha=0.6, color='teal')
plt.xlabel('log10(Waveform Difference)')
plt.ylabel('Count')
plt.title('Fig8: Extrapolation Order Differences')
plt.legend()
plt.tight_layout()
plt.savefig('report/images/fig8_comparison.png', dpi=150)
plt.close()

# Plot 4: Fig6 vs Fig8 tail comparison (CDF)
plt.figure(figsize=(8,5))
sns.ecdfplot(np.log10(fig6.iloc[:,0]), label='Overall (Fig6)', color='navy')
sns.ecdfplot(np.log10(fig8['N2vsN3']), label='N2vsN3 (Fig8)', color='coral')
sns.ecdfplot(np.log10(fig8['N2vsN4']), label='N2vsN4 (Fig8)', color='teal')
plt.xlabel('log10(Waveform Difference)')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('report/images/cdf_comparison.png', dpi=150)
plt.close()

print("\nFigures saved to report/images/")
print("Analysis complete.")