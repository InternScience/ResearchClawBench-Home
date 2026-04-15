import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('outputs/dataset.json', 'r') as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12,10))

# Property pred
feat = np.array(data['property_prediction'][1])
targ = np.array(data['property_prediction'][3])
cat = np.array(data['property_prediction'][2])

axes[0,0].hist(feat, bins=20, alpha=0.7, label='Features')
axes[0,0].set_title('Feature Distribution')
axes[0,1].hist(targ, bins=20, alpha=0.7, label='Targets', color='orange')
axes[0,1].set_title('Target Distribution')
axes[0,1].hist(cat, bins=np.unique(cat).size, alpha=0.7, label='Categories')
axes[1,0].scatter(feat, targ, alpha=0.6)
axes[1,0].set_xlabel('Feature')
axes[1,0].set_ylabel('Target')
axes[1,0].set_title('Feature vs Target')

# Structure gen
a = np.array(data['structure_generation'][0])
b = np.array(data['structure_generation'][1])
axes[0,0].hist(a, bins=20, alpha=0.5, label='a')
axes[0,0].hist(b, bins=20, alpha=0.5, label='b')
axes[0,0].legend()
axes[1,1].scatter(a, b, alpha=0.6)
axes[1,1].set_xlabel('a param')
axes[1,1].set_ylabel('b param')
axes[1,1].set_title('Lattice Params')

plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print('Data overview saved')