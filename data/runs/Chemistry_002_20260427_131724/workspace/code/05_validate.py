"""Better validation: use multiple residue-level interface descriptors.

Predictors per residue:
  - n_contacts: number of heavy atoms within 5 Å of partner chain
  - n_close_partner_residues: number of distinct partner residues within 5 Å
  - min_distance: minimum heavy atom distance to partner chain
  - HSres: HADDOCK score per-residue contribution
  - delta_HSres: difference between predicted and bound

Outcomes:
  - mean_ddG, max_ddG, sum_abs_ddG (per-residue aggregates)
  - per-mutation ddG (single mutants)
"""
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr, pearsonr

D = np.load('outputs/structure.npz', allow_pickle=True)
chains = D['chains']; coords=D['coords'].astype(np.float64)
chg = D['chg']; vdw = D['vdw']; resi = D['resi']; res=D['res']; name=D['name']
iA = chains=='A'; iD = chains=='D'
cA = coords[iA]; cD_bound = coords[iD]
resiA, resiD = resi[iA], resi[iD]
resA, resD = res[iA], res[iD]
chgA, chgD = chg[iA], chg[iD]
vdwA, vdwD = vdw[iA], vdw[iD]

HP = {'ALA':0.31,'ARG':-1.01,'ASN':-0.60,'ASP':-0.77,'CYS':1.54,'GLN':-0.22,'GLU':-0.64,
      'GLY':0.00,'HIS':0.13,'ILE':1.80,'LEU':1.70,'LYS':-0.99,'MET':1.23,'PHE':1.79,
      'PRO':0.72,'SER':-0.04,'THR':0.26,'TRP':2.25,'TYR':0.96,'VAL':1.22}
hpA = np.array([HP.get(r,0.0) for r in resA])
hpD = np.array([HP.get(r,0.0) for r in resD])

def descriptors(cD):
    diff = cA[:,None,:]-cD[None,:,:]
    d2 = np.einsum('ijk,ijk->ij',diff,diff)
    d2 = np.maximum(d2,1.0)
    d  = np.sqrt(d2)
    sig = (vdwA[:,None]+vdwD[None,:])
    inv2 = sig*sig/d2
    inv6 = inv2**3
    e_lj = 4*0.10*(inv6*inv6 - inv6); e_lj = np.clip(e_lj,-2.0,5.0)
    e_el = 332.0*chgA[:,None]*chgD[None,:]/(10.0*d2)
    close5 = d<5.0
    e_ds = -0.05*np.exp(-((d-3.5)/2.0)**2)*(d<8.0)*hpA[:,None]*hpD[None,:]
    rows=[]
    # chain A
    for r in np.unique(resiA):
        sel = resiA==r
        # n_contacts: heavy atoms with min distance to any D heavy atom <5
        atom_min = d[sel].min(axis=1)  # min dist of each atom to partner
        n_close = int((atom_min<5.0).sum())
        partner_residues = set()
        for i_atom in np.where(sel)[0]:
            j_close = np.where(d[i_atom]<5.0)[0]
            for j in j_close:
                partner_residues.add(int(resiD[j]))
        rows.append(dict(chain='A', resi=int(r), aa=resA[sel][0],
                         n_contacts=n_close,
                         n_close_partner=len(partner_residues),
                         min_dist=float(atom_min.min()),
                         E_vdw=float(e_lj[sel].sum()),
                         E_elec=float(e_el[sel].sum()),
                         E_des=float(e_ds[sel].sum()),
                         HSres=float(e_lj[sel].sum()+0.2*e_el[sel].sum()+e_ds[sel].sum())))
    for r in np.unique(resiD):
        sel = resiD==r
        atom_min = d[:,sel].min(axis=0)
        n_close = int((atom_min<5.0).sum())
        partner_residues = set()
        for j_atom in np.where(sel)[0]:
            i_close = np.where(d[:,j_atom]<5.0)[0]
            for i in i_close:
                partner_residues.add(int(resiA[i]))
        rows.append(dict(chain='D', resi=int(r), aa=resD[sel][0],
                         n_contacts=n_close,
                         n_close_partner=len(partner_residues),
                         min_dist=float(atom_min.min()),
                         E_vdw=float(e_lj[:,sel].sum()),
                         E_elec=float(e_el[:,sel].sum()),
                         E_des=float(e_ds[:,sel].sum()),
                         HSres=float(e_lj[:,sel].sum()+0.2*e_el[:,sel].sum()+e_ds[:,sel].sum())))
    return pd.DataFrame(rows)

bound_desc = descriptors(cD_bound)
bound_desc.to_csv('outputs/per_residue_descriptors_bound.csv', index=False)

T = np.load('outputs/top1_pose.npz')
pred_desc = descriptors(T['cD'])
pred_desc.to_csv('outputs/per_residue_descriptors_top1.csv', index=False)

# merge with skempi
sk = pd.read_csv('outputs/skempi_1brs_perresidue.csv')
mb = sk.merge(bound_desc, on=['chain','resi'], how='left')
mp = sk.merge(pred_desc,  on=['chain','resi'], how='left')

per_mut = pd.read_csv('outputs/skempi_1brs.csv')
per_mut = per_mut[per_mut.n_muts==1].copy()
import re
def parse(s):
    m=re.match(r'^([A-Z])([A-Z])(\d+)([A-Z])$', s.split(',')[0].strip())
    if not m: return (None,None)
    return (m.group(2), int(m.group(3)))
per_mut[['chain','resi']] = per_mut['Mutation(s)_PDB'].apply(lambda s: pd.Series(parse(s)))
per_mut_b = per_mut.merge(bound_desc, on=['chain','resi'], how='left')
per_mut_p = per_mut.merge(pred_desc,  on=['chain','resi'], how='left')

stats = {}
def corrs(name, df, predictors, target):
    out = {}
    df2 = df.dropna(subset=predictors+[target])
    out['n'] = int(len(df2))
    for p in predictors:
        sR, sP = spearmanr(df2[p], df2[target])
        pR, pP = pearsonr(df2[p], df2[target])
        out[p] = dict(spearman=float(sR), spearman_p=float(sP),
                      pearson=float(pR),  pearson_p=float(pP))
    stats[name] = out

predictors = ['n_contacts','n_close_partner','E_vdw','E_elec','E_des','HSres']
corrs('per_residue_bound_meanDDG', mb, predictors, 'mean_ddG')
corrs('per_residue_pred_meanDDG',  mp, predictors, 'mean_ddG')
corrs('per_residue_bound_sumabsDDG', mb, predictors, 'sum_abs_ddG')
corrs('per_residue_pred_sumabsDDG',  mp, predictors, 'sum_abs_ddG')
corrs('per_mutation_bound', per_mut_b, predictors, 'ddG')
corrs('per_mutation_pred',  per_mut_p, predictors, 'ddG')

json.dump(stats, open('outputs/validation_stats.json','w'), indent=2)
# print summary
for k, v in stats.items():
    print(k, 'n=', v['n'])
    for p in predictors:
        print(f'  {p}: spearman={v[p]["spearman"]:+.3f} (p={v[p]["spearman_p"]:.3f}), pearson={v[p]["pearson"]:+.3f}')

# Save merged tables
mb.to_csv('outputs/skempi_vs_bound_descriptors.csv', index=False)
mp.to_csv('outputs/skempi_vs_pred_descriptors.csv', index=False)
per_mut_b.to_csv('outputs/skempi_per_mutation_bound.csv', index=False)
per_mut_p.to_csv('outputs/skempi_per_mutation_pred.csv', index=False)
