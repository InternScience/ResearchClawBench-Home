import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df = pd.read_csv('data/train_simulated.csv')

# Plot distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=train_df)
plt.title('Target Distribution (Train)')
plt.savefig('report/images/target_distribution.png')
plt.close()

# Plot distribution of degradation types
plt.figure(figsize=(8, 5))
sns.countplot(x='degradation', data=train_df)
plt.title('Degradation Type Distribution (Train)')
plt.savefig('report/images/degradation_distribution.png')
plt.close()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
corr = train_df[[str(i) for i in range(20)] + ['label']].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('report/images/correlation_matrix.png')
plt.close()

print("Plots generated.")
