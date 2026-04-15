import prody
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import os

# Load structures
query = prody.parsePDB('data/7xg4.pdb')
target = prody.parsePDB('data/6n40.pdb')

print(f"Query 7XG4: {query.numAtoms()} atoms")
print(f"Target 6N40: {target.numAtoms()} atoms")

# Chain stats using CA
def chain_stats(struct):
    chains = {}
    for chain in struct.iterChains():
        chid = chain.getChid()
        ca = struct.select(f"chain {chid} and calpha")
        if ca is not None and ca.numAtoms() > 0:
            chains[chid] = {
                'ca_atoms': int(ca.numAtoms()),
                'residues': int(ca.numResidues())
            }
        else:
            chains[chid] = {'ca_atoms': 0, 'residues': 0}
    return chains

q_stats = chain_stats(query)
t_stats = chain_stats(target)

stats = {
    '7xg4': q_stats,
    '6n40': t_stats,
    'total_atoms_7xg4': int(query.numAtoms()),
    'total_atoms_6n40': int(target.numAtoms())
}

os.makedirs('outputs', exist_ok=True)
with open('outputs/data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

# Plot 1: Chain CA atoms bar
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(q_stats.keys(), [v['ca_atoms'] for v in q_stats.values()], color='skyblue')
axes[0].set_title('7XG4 Chains: Cα Atoms')
axes[0].set_ylabel('Number of Cα Atoms')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(t_stats.keys(), [v['ca_atoms'] for v in t_stats.values()], color='lightgreen')
axes[1].set_title('6N40 Chains: Cα Atoms')
axes[1].set_ylabel('Number of Cα Atoms')

plt.tight_layout()
os.makedirs('report/images', exist_ok=True)
plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
plt.close()

print('Data overview plot saved to report/images/data_overview.png')