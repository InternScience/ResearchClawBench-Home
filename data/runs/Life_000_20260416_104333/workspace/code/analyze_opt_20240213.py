import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

df_opt = pd.read_excel('data/ML_ei&pred_20240213.xlsx')
df_opt['ML'] = df_opt['ML'].ffill()
df_opt['Glass (kPa)_max'] = pd.to_numeric(df_opt['Glass (kPa)_max'], errors='coerce')

plt.figure(figsize=(12, 6))
sns.boxplot(x='ML', y='Glass (kPa)_max', data=df_opt)
plt.xticks(rotation=45, ha='right')
plt.title('Adhesive Strength by Optimization Strategy (20240213 Dataset)')
plt.ylabel('Adhesive Strength (kPa)')
plt.tight_layout()
plt.savefig('report/images/optimization_strategies_20240213.png', dpi=300)
plt.close()

print(df_opt.groupby('ML')['Glass (kPa)_max'].max())
