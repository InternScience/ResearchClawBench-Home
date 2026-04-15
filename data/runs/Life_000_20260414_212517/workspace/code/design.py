import pandas as pd
import numpy as np
import joblib
from scipy.spatial.distance import cdist
from pathlib import Path
import json

# Load model and data
models = joblib.load('outputs/models/trained_models.joblib')
rfr = models['rfr']
features = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']

df_train = pd.read_csv('outputs/initial_data_processed.csv')
X_train = df_train[features].values

# Sample 100k compositions on simplex
np.random.seed(42)
n_samples = 100000
alpha = np.ones(6)
samples = np.random.dirichlet(alpha, n_samples)
preds = rfr.predict(samples)

# Top 10 highest predicted
top_idx = np.argsort(preds)[::-1][:10]
top_designs = samples[top_idx]
top_preds = preds[top_idx]

# Distance to training data (euclidean in comp space)
dists = cdist(top_designs, X_train).min(axis=1)

designs_df = pd.DataFrame(top_designs, columns=features)
designs_df['pred_kpa'] = top_preds
designs_df['min_dist_to_train'] = dists
designs_df['new'] = dists > 0.05  # threshold for novel

print('Top pred:', top_preds[0])
print('Designs shape:', designs_df.shape)
designs_df.to_csv('outputs/proposed_designs.csv', index=False)

# JSON
designs_json = designs_df.round(4).to_dict('records')
result = {
  'top_pred_kpa': float(top_preds[0]),
  'designs': designs_json,
  'n_novel': int(designs_df['new'].sum()),
  'max_novel_pred': float(designs_df[designs_df['new']]['pred_kpa'].max())
}
with open('outputs/proposed_designs.json', 'w') as f:
  json.dump(result, f, indent=2)

# Plot top design vs train high
top_comp = top_designs[0]
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(features, top_comp, alpha=0.7, label='Top Design')
high_train = df_train.loc[df_train['Glass_max'].idxmax(), features]
ax.bar(features, high_train.values, alpha=0.7, label='Best Train Obs')
ax.legend()
ax.set_ylabel('Composition')
ax.set_title(f'Top Predicted {top_preds[0]:.1f} kPa vs Best Obs {high_train.name}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/images/top_design.png', dpi=300, bbox_inches='tight')
plt.close()

print('Designs proposed')