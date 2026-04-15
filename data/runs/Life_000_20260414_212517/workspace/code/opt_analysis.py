import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json

# Load models
models = joblib.load('outputs/models/trained_models.joblib')
rfr = models['rfr']
features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']

# Load opt data
df_opt1 = pd.read_excel('data/ML_ei&pred (1&2&3rounds)_20240408.xlsx')
df_opt1['Glass_max'] = pd.to_numeric(df_opt1['Glass (kPa)_max'], errors='coerce')
df_opt1_clean = df_opt1[features + ['Glass_max', 'ML', 'NO.']].dropna()
X_opt1 = df_opt1_clean[features].values
y_opt1 = df_opt1_clean['Glass_max'].values
y_pred_rfr1 = rfr.predict(X_opt1)

df_opt2 = pd.read_excel('data/ML_ei&pred_20240213.xlsx')
df_opt2['Glass_max'] = pd.to_numeric(df_opt2['Glass (kPa)_max'], errors='coerce')
df_opt2_clean = df_opt2[features + ['Glass_max', 'ML', 'NO.']].dropna()
X_opt2 = df_opt2_clean[features].values
y_opt2 = df_opt2_clean['Glass_max'].values
y_pred_rfr2 = rfr.predict(X_opt2)

# Trajectory: assume NO. groups rounds
df_opt1_clean['round'] = pd.cut(df_opt1_clean['NO.'], bins=[0,10,20,40,100], labels=['R1','R2','R3','R4'])
df_opt2_clean['round'] = pd.cut(df_opt2_clean['NO.'], bins=[0,10,20,40,100], labels=['R1','R2','R3','R4'])

# Cum max per round for opt1
cum_max1 = df_opt1_clean.groupby('round')['Glass_max'].max().cumsum()
cum_max1_pred = df_opt1_clean.groupby('round')['pred_rfr'].max().cumsum()  # wait, add pred
df_opt1_clean['pred_rfr'] = y_pred_rfr1
df_opt2_clean['pred_rfr'] = y_pred_rfr2
cum_max1 = df_opt1_clean.groupby('round', observed=True)['Glass_max'].max()
cum_max1_pred = df_opt1_clean.groupby('round', observed=True)['pred_rfr'].max()

# Plots
plt.style.use('default')
fig, axes = plt.subplots(2,2, figsize=(12,10))

# Pred vs obs opt1
axes[0,0].scatter(y_opt1, y_pred_rfr1, alpha=0.6)
axes[0,0].plot([0,350], [0,350], 'r--')
axes[0,0].set_xlabel('Obs Opt1')
axes[0,0].set_ylabel('Pred RFR')
axes[0,0].set_title('Opt1 Pred vs Obs')

# Opt2
axes[0,1].scatter(y_opt2, y_pred_rfr2, alpha=0.6)
axes[0,1].plot([0,350], [0,350], 'r--')
axes[0,1].set_xlabel('Obs Opt2')
axes[0,1].set_ylabel('Pred RFR')
axes[0,1].set_title('Opt2 Pred vs Obs')

# Trajectory opt1
ax = axes[1,0]
cum_max1.plot(ax=ax, marker='o')
cum_max1_pred.plot(ax=ax, marker='s', ls='--')
ax.set_title('Opt1 Cumulative Max')
ax.set_xlabel('Round')
ax.set_ylabel('Max kPa')
ax.legend(['Observed', 'Predicted'])

# All opt max hist
all_y = np.concatenate([y_opt1, y_opt2])
axes[1,1].hist(all_y, bins=20)
axes[1,1].axvline(all_y.max(), color='r')
axes[1,1].set_title('Opt Data Max Hist')
axes[1,1].set_xlabel('Glass_max kPa')

plt.tight_layout()
plt.savefig('report/images/opt_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save trajectory json
traj = {
    'opt1_max_per_round': cum_max1.to_dict(),
    'opt1_pred_max_per_round': cum_max1_pred.to_dict(),
    'opt1_rmse': np.sqrt(np.mean((y_opt1 - y_pred_rfr1)**2)),
    'opt2_rmse': np.sqrt(np.mean((y_opt2 - y_pred_rfr2)**2)),
    'overall_opt_max': float(np.max([y_opt1.max(), y_opt2.max()]))
}
with open('outputs/opt_trajectory.json', 'w') as f:
    json.dump(traj, f)
print('Opt analysis complete')