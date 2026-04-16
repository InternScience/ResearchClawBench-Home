import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

df_opt = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx')
df_opt['ML'] = df_opt['ML'].ffill()

# Convert 'Glass (kPa)_max' to numeric, coercing errors like "NO GELATION" to NaN
df_opt['Glass (kPa)_max'] = pd.to_numeric(df_opt['Glass (kPa)_max'], errors='coerce')

# Plot the distribution of Glass (kPa)_max for different ML methods
plt.figure(figsize=(12, 6))
sns.boxplot(x='ML', y='Glass (kPa)_max', data=df_opt)
plt.xticks(rotation=45, ha='right')
plt.title('Adhesive Strength by Optimization Strategy')
plt.ylabel('Adhesive Strength (kPa)')
plt.tight_layout()
plt.savefig('report/images/optimization_strategies.png', dpi=300)
plt.close()

# Group by round and strategy
# RFR-GP, RFR-GP-2rd-ei, RFR-GP-3rd-ei
rfr_gp_rounds = ['RFR-GP', 'RFR-GP-2rd-ei', 'RFR-GP-3rd-ei']
gp_gp_rounds = ['GP-GP', 'GP-GP-2rd-ei', 'GP-GP-3rd-ei']

df_rfr_gp = df_opt[df_opt['ML'].isin(rfr_gp_rounds)].copy()
df_rfr_gp['Round'] = df_rfr_gp['ML'].map({'RFR-GP': 'Round 1', 'RFR-GP-2rd-ei': 'Round 2', 'RFR-GP-3rd-ei': 'Round 3'})

df_gp_gp = df_opt[df_opt['ML'].isin(gp_gp_rounds)].copy()
df_gp_gp['Round'] = df_gp_gp['ML'].map({'GP-GP': 'Round 1', 'GP-GP-2rd-ei': 'Round 2', 'GP-GP-3rd-ei': 'Round 3'})

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Round', y='Glass (kPa)_max', data=df_rfr_gp, order=['Round 1', 'Round 2', 'Round 3'])
plt.title('RFR-GP Optimization Trajectory')
plt.ylabel('Adhesive Strength (kPa)')

plt.subplot(1, 2, 2)
sns.boxplot(x='Round', y='Glass (kPa)_max', data=df_gp_gp, order=['Round 1', 'Round 2', 'Round 3'])
plt.title('GP-GP Optimization Trajectory')
plt.ylabel('Adhesive Strength (kPa)')

plt.tight_layout()
plt.savefig('report/images/optimization_trajectory.png', dpi=300)
plt.close()

# Print max values
print("Max Adhesive Strength by Round (RFR-GP):")
print(df_rfr_gp.groupby('Round')['Glass (kPa)_max'].max())

print("\nMax Adhesive Strength by Round (GP-GP):")
print(df_gp_gp.groupby('Round')['Glass (kPa)_max'].max())
