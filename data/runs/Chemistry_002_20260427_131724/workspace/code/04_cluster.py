"""Cluster top docking poses by ligand-RMSD on Cα atoms; report top clusters."""
import numpy as np, pandas as pd, json, os

D = np.load('outputs/structure.npz', allow_pickle=True)
chains = D['chains']; coords=D['coords'].astype(np.float64); name=D['name']
iD = chains=='D'
cD0 = coords[iD]
nameD = name[iD]
ca_mask = nameD=='CA'
cD0_ca = cD0[ca_mask]
comD0 = cD0.mean(0)
cD0_c_ca = cD0_ca - comD0

P = np.load('outputs/poses_full.npz')
R_all = P['R']; t_all=P['t']; scores=P['scores']

# transform Cα coords for every pose
cD0_c = cD0 - comD0
poses_ca = np.einsum('nij,kj->nki', R_all, cD0_c_ca) + t_all[:,None,:]  # (Npose, Nca, 3)

# ligand RMSD between poses
N = poses_ca.shape[0]
print('Npose:', N)
# Sort by score, take top 50% (or all if small)
order = np.argsort(scores)
top_n = max(20, int(N*0.5))
top = order[:top_n]
# compute pairwise lrmsd
def lrmsd(a,b):
    return np.sqrt(((a-b)**2).sum(-1).mean(-1))
M = np.zeros((len(top),len(top)))
for i in range(len(top)):
    M[i] = lrmsd(poses_ca[top[i]][None,:,:], poses_ca[top])

# Single-linkage clustering at 4 Å
visited = np.zeros(len(top), dtype=bool)
clusters=[]
THR=7.5  # HADDOCK default cluster cutoff is FCC 0.6; here we use 7.5 Å L-RMSD
import collections
for i in range(len(top)):
    if visited[i]: continue
    # BFS
    queue=collections.deque([i])
    members=set()
    while queue:
        j=queue.popleft()
        if visited[j]: continue
        visited[j]=True
        members.add(j)
        for k in range(len(top)):
            if not visited[k] and M[j,k]<THR:
                queue.append(k)
    clusters.append(sorted(members))

print(f'{len(clusters)} clusters from top {len(top)} poses')

# Build cluster summary
recs=[]
for cid,c in enumerate(clusters):
    pose_idx = [int(top[k]) for k in c]
    s = scores[pose_idx]
    P_lrmsd = P['lrmsd'][pose_idx]
    P_irmsd = P['irmsd'][pose_idx]
    best = pose_idx[int(np.argmin(s))]
    recs.append(dict(cluster=cid+1,
                     size=len(c),
                     best_pose=best,
                     best_score=float(s.min()),
                     mean_score=float(s.mean()),
                     mean_lrmsd=float(np.mean(P_lrmsd)),
                     mean_irmsd=float(np.mean(P_irmsd)),
                     best_lrmsd=float(P_lrmsd[np.argmin(s)]),
                     best_irmsd=float(P_irmsd[np.argmin(s)]),
                     poses=','.join(map(str,pose_idx))))
cdf = pd.DataFrame(recs).sort_values('best_score').reset_index(drop=True)
cdf.to_csv('outputs/clusters.csv', index=False)
print(cdf.head(10))

# Save top1 pose coordinates (full ligand atoms) for use in validation
top1_idx = int(order[0])
R = R_all[top1_idx]; t = t_all[top1_idx]
cD_top1 = cD0_c@R.T + t
np.savez('outputs/top1_pose.npz', cD=cD_top1, idx=top1_idx, R=R, t=t)
print('top1 pose idx:', top1_idx, 'score=', float(scores[top1_idx]),
      'lrmsd=', float(P['lrmsd'][top1_idx]), 'irmsd=', float(P['irmsd'][top1_idx]))
