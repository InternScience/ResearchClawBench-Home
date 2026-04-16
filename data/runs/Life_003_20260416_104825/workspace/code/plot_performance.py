import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('data/performance_summary.csv')

# Plot Alignment Time
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Chemistry', y='Time_min', hue='Tool')
plt.title('Alignment Time by Tool and Chemistry')
plt.ylabel('Time (minutes)')
plt.yscale('log')
plt.tight_layout()
plt.savefig('report/images/alignment_time.png', dpi=300)
plt.close()

# Plot File Size
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Chemistry', y='FileSize_MB', hue='Tool')
plt.title('File Size by Tool and Chemistry')
plt.ylabel('File Size (MB)')
plt.yscale('log')
plt.tight_layout()
plt.savefig('report/images/file_size.png', dpi=300)
plt.close()

print("Performance plots generated.")
