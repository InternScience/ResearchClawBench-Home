import prody
from prody.proteins import matchAlign
from prody.measure.transform import superpose, calcRMSD
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load
query = prody.parsePDB('data/7xg4.pdb')
target = prody.parsePDB('data/6n40.pdb')

t_chain_ca = target.select('chain A and calpha')

chain_aligns = {}
correspondences = {}

for q_chain in query.iterChains():
    chid = q_chain.getChid()
    q_chain_ca = query.select(f'chain {chid} and calpha')
    if q_chain_ca is None or q_chain_ca.numAtoms() == 0: 
        continue
    
    print(f'Aligning chain {chid} ({q_chain_ca.numResidues()} res)')
    
    aligned_q, aligned_t = matchAlign(q_chain_ca, t_chain_ca)
    
    if aligned_q is None:
        continue
    
    rmsd = calcRMSD(aligned_q, aligned_t)
    
    sup_q, rot, tran = superpose(aligned_q, aligned_t)
    
    q_res = [r for r in aligned_q.getResindices()]
    t_res = [r for r in aligned_t.getResindices()]
    corr = list(zip(q_res, t_res))
    
    chain_aligns[chid] = {
        'residues_q': int(q_chain_ca.numResidues()),
        'rmsd': float(rmsd),
        'rot_matrix': rot.tolist(),
        'translation': tran.tolist(),
        'num_aligned': len(q_res)
    }
    
    correspondences[chid] = corr[:20]  # sample
    
    prody.writePDB(f'outputs/{chid}_aligned.pdb', sup_q)

results = {
    'chain_alignments': chain_aligns,
    'sample_correspondences': correspondences,
    'best_chain': min(chain_aligns, key=lambda k: chain_aligns[k]['rmsd'])
}

with open('outputs/alignment_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot RMSD bar
fig, ax = plt.subplots(figsize=(10,6))
chains = list(chain_aligns.keys())
rmsds = [chain_aligns[c]['rmsd'] for c in chains]
ax.bar(chains, rmsds, color='orange')
ax.axhline(y=3.0, color='r', linestyle='--', label='RMSD < 3Å similar')
ax.set_title('Post-Alignment RMSD: 7XG4 Chains vs 6N40 A')
ax.set_ylabel('RMSD (Å)')
ax.tick_params(axis='x', rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig('report/images/rmsd_per_chain.png', dpi=150, bbox_inches='tight')
plt.close()

# 3D plot for best chain
best = results['best_chain']
best_q = query.select(f'chain {best} and calpha')
best_aligned_file = f'outputs/{best}_aligned.pdb'
best_aligned = prody.parsePDB(best_aligned_file).select('calpha')
t_ca = target.select('calpha')

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(best_q.getCoords()[:,0], best_q.getCoords()[:,1], best_q.getCoords()[:,2], 'b-', label='7XG4 '+best)
ax1.plot(t_ca.getCoords()[:,0], t_ca.getCoords()[:,1], t_ca.getCoords()[:,2], 'r-', label='6N40 A')
ax1.set_title('Before Alignment')
ax1.legend()

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(best_aligned.getCoords()[:,0], best_aligned.getCoords()[:,1], best_aligned.getCoords()[:,2], 'b-', label='Aligned 7XG4 '+best)
ax2.plot(t_ca.getCoords()[:,0], t_ca.getCoords()[:,1], t_ca.getCoords()[:,2], 'r-', label='6N40 A')
ax2.set_title('After Alignment')
ax2.legend()

ax3 = fig.add_subplot(133)
ax3.bar(['All chains avg RMSD'], [np.mean(rmsds)])
ax3.set_title('Average RMSD')
plt.tight_layout()
plt.savefig('report/images/alignment_visual.png', dpi=150, bbox_inches='tight')
plt.close()

print('Alignment analysis complete. Best match:', best, 'RMSD:', chain_aligns[best]['rmsd'])