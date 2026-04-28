"""HADDOCK-style rigid-body sampling driven by AIRs - vectorized, faster.

Final scoring follows HADDOCK water-stage weights:
    HS = 1.0*E_vdw + 0.2*E_elec + 0.1*E_AIR + 1.0*E_desolv
"""
import numpy as np, json, os, time, pandas as pd

rng = np.random.default_rng(20260427)
D = np.load('outputs/structure.npz', allow_pickle=True)
coords = D['coords']; chains = D['chains']; resi = D['resi']
res = D['res']; name = D['name']; elem = D['elem']
chg = D['chg']; vdw = D['vdw']

iA = chains=='A'; iD = chains=='D'
cA  = coords[iA].astype(np.float64)
cD0 = coords[iD].astype(np.float64)
chgA, chgD = chg[iA].astype(np.float64), chg[iD].astype(np.float64)
vdwA, vdwD = vdw[iA].astype(np.float64), vdw[iD].astype(np.float64)
resiA, resiD = resi[iA], resi[iD]
resA,  resD  = res[iA],  res[iD]
nameA, nameD = name[iA], name[iD]

HP = {'ALA':0.31,'ARG':-1.01,'ASN':-0.60,'ASP':-0.77,'CYS':1.54,'GLN':-0.22,'GLU':-0.64,
      'GLY':0.00,'HIS':0.13,'ILE':1.80,'LEU':1.70,'LYS':-0.99,'MET':1.23,'PHE':1.79,
      'PRO':0.72,'SER':-0.04,'THR':0.26,'TRP':2.25,'TYR':0.96,'VAL':1.22}
hpA = np.array([HP.get(r,0.0) for r in resA])
hpD = np.array([HP.get(r,0.0) for r in resD])

AIRS = json.load(open('outputs/airs.json'))
active_A = set(AIRS['active_chainA_barnase'])
active_D = set(AIRS['active_chainD_barstar'])
air_A_mask = np.array([r in active_A for r in resiA])
air_D_mask = np.array([r in active_D for r in resiD])
# Pre-build groupings
unique_actA = sorted(active_A); unique_actD = sorted(active_D)
A_res_to_atomidx = {r: np.where(resiA==r)[0] for r in unique_actA}
D_res_to_atomidx = {r: np.where(resiD==r)[0] for r in unique_actD}
sigAD = (vdwA[:,None]+vdwD[None,:])
qqAD  = chgA[:,None]*chgD[None,:]
hpAD  = hpA[:,None]*hpD[None,:]
print('atoms A:', cA.shape, 'D:', cD0.shape)
print('AIR A residues:', len(unique_actA), 'D residues:', len(unique_actD))

comA = cA.mean(0)
comD0 = cD0.mean(0)
cD0_c = cD0 - comD0

def rotmat_axis_angle(ax, th):
    ax = ax/np.linalg.norm(ax)
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K

def random_rotation():
    u1,u2,u3 = rng.uniform(0,1,3)
    q = np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2),
                  np.sqrt(1-u1)*np.cos(2*np.pi*u2),
                  np.sqrt(u1)*np.sin(2*np.pi*u3),
                  np.sqrt(u1)*np.cos(2*np.pi*u3)])
    w,x,y,z=q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

CUT2 = 8.0**2
def energies(cD):
    diff = cA[:,None,:] - cD[None,:,:]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    d2 = np.maximum(d2, 1.0)
    d  = np.sqrt(d2)
    inv2 = (sigAD*sigAD)/d2
    inv6 = inv2**3
    inv12 = inv6*inv6
    e_lj = 4*0.10*(inv12 - inv6)
    np.clip(e_lj, -2.0, 5.0, out=e_lj)
    E_vdw = e_lj.sum()
    E_elec = (332.0*qqAD/(10.0*d2)).sum()
    # desolvation
    close = d2<CUT2
    weight = np.exp(-((d-3.5)/2.0)**2) * close
    E_des = -(weight*hpAD).sum()*0.05
    # AIR: per active residue r on A, min distance to ANY active atom on D
    e_air = 0.0
    target=2.0
    # Build (Na_air x Nd_air) sub matrix once
    sub = d[air_A_mask][:, air_D_mask]
    # group rows by residue id
    A_resids_air = resiA[air_A_mask]
    D_resids_air = resiD[air_D_mask]
    for r,idxs in A_res_to_atomidx.items():
        # rows in sub corresponding to active atoms of residue r
        rows = np.where(A_resids_air==r)[0]
        if len(rows)==0: continue
        dmin = sub[rows].min()
        if dmin>target:
            e_air += (dmin-target)**2
    for r,idxs in D_res_to_atomidx.items():
        cols = np.where(D_resids_air==r)[0]
        if len(cols)==0: continue
        dmin = sub[:,cols].min()
        if dmin>target:
            e_air += (dmin-target)**2
    return E_vdw, E_elec, e_air, E_des

def hscore(e):
    return 1.0*e[0]+0.2*e[1]+0.1*e[2]+1.0*e[3]

def transform(R, t):
    return cD0_c@R.T + t

def gstep(R, t, lr_t=0.4, lr_R=0.04):
    cD = transform(R,t); base = energies(cD); s_b = hscore(base)
    h=0.5
    grad_t = np.zeros(3)
    for i in range(3):
        dt=t.copy(); dt[i]+=h
        grad_t[i] = (hscore(energies(transform(R,dt)))-s_b)/h
    eps=0.05
    grad_w=np.zeros(3); axes=np.eye(3)
    for i in range(3):
        Rp=rotmat_axis_angle(axes[i],eps)@R
        grad_w[i]=(hscore(energies(transform(Rp,t)))-s_b)/eps
    t_new = t - lr_t*np.tanh(grad_t/10)
    w = -lr_R*np.tanh(grad_w/10)
    th=np.linalg.norm(w)
    R_new = rotmat_axis_angle(w/th, th)@R if th>1e-8 else R
    new_e = energies(transform(R_new,t_new)); s_n = hscore(new_e)
    if s_n>s_b: return R,t,s_b,base
    return R_new,t_new,s_n,new_e

def lrmsd(cD):
    return float(np.sqrt(((cD-cD0)**2).sum(1).mean()))
def irmsd(cD):
    bb = np.isin(nameD, ['N','CA','C','O'])
    intf = np.isin(resiD, list(active_D))
    sel = bb & intf
    a = cD[sel]; b = cD0[sel]
    return float(np.sqrt(((a-b)**2).sum(1).mean()))

# Sampling
N_POSES = 120
N_STEPS = 35
results=[]
all_R=[]; all_t=[]
t0=time.time()
air_cenA = cA[np.isin(resiA, list(active_A))].mean(0)
air_cenD0c = cD0_c[np.isin(resiD, list(active_D))].mean(0)

for i in range(N_POSES):
    R = random_rotation()
    direction = rng.normal(size=3); direction/=np.linalg.norm(direction)
    t = comA + direction*rng.uniform(20,30)
    # pull AIR centroid close
    for _ in range(3):
        cur = transform(R,t)
        cur_air = cur[np.isin(resiD, list(active_D))].mean(0)
        t = t + 0.7*(air_cenA - cur_air)
    last_s=1e9; last_e=None
    for step in range(N_STEPS):
        R,t,s,e = gstep(R,t)
        if abs(last_s-s)<5e-2: break
        last_s=s; last_e=e
    cD_final = transform(R,t)
    lrm = lrmsd(cD_final); irm = irmsd(cD_final)
    results.append(dict(idx=i, Evdw=last_e[0], Eelec=last_e[1], Eair=last_e[2], Edes=last_e[3],
                        score=last_s, lrmsd=lrm, irmsd=irm))
    all_R.append(R); all_t.append(t)
    if i%10==0:
        dt=time.time()-t0
        print(f'pose {i}/{N_POSES}: score={last_s:.2f} lrmsd={lrm:.2f} irmsd={irm:.2f} t={dt:.1f}s')
    # incremental save every 20
    if i%20==19 or i==N_POSES-1:
        df = pd.DataFrame(results).sort_values('score').reset_index(drop=True)
        df.to_csv('outputs/poses.csv', index=False)
        np.savez('outputs/poses_full.npz',
                 R=np.array(all_R), t=np.array(all_t),
                 scores=np.array([r['score'] for r in results]),
                 lrmsd=np.array([r['lrmsd'] for r in results]),
                 irmsd=np.array([r['irmsd'] for r in results]))

print('Done. Total:',time.time()-t0,'s')
df = pd.DataFrame(results).sort_values('score').reset_index(drop=True)
print(df.head(10))
