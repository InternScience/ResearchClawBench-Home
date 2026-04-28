"""Step 1: parse 1BRS PDB into per-atom array; identify interface; define AIRs."""
import numpy as np, json, os
from collections import defaultdict

# Atomic vdW radii (OPLS-like, Å) and partial charges (very approximate)
VDW_RADIUS = {'C':1.90, 'N':1.85, 'O':1.70, 'S':2.00, 'H':1.20, 'P':2.10}
# Default partial charges by atom name for common backbone & sidechain atoms
DEFAULT_CHG = {'N':-0.40, 'CA':0.05, 'C':0.50, 'O':-0.50, 'CB':0.0}
# Side-chain charged terminal heuristics
SIDECHAIN_CHG = {
    ('LYS','NZ'): 1.0, ('ARG','NH1'): 0.5, ('ARG','NH2'): 0.5, ('ARG','NE'): 0.0,
    ('ASP','OD1'): -0.5, ('ASP','OD2'): -0.5, ('GLU','OE1'): -0.5, ('GLU','OE2'): -0.5,
    ('HIS','ND1'): 0.1, ('HIS','NE2'): 0.1,
}
# Hydrophobicity scale for desolvation (Fauchere-Pliska transfer free energies, kcal/mol)
HYDROPHOB = {
    'ALA':0.31,'ARG':-1.01,'ASN':-0.60,'ASP':-0.77,'CYS':1.54,'GLN':-0.22,'GLU':-0.64,
    'GLY':0.00,'HIS':0.13,'ILE':1.80,'LEU':1.70,'LYS':-0.99,'MET':1.23,'PHE':1.79,
    'PRO':0.72,'SER':-0.04,'THR':0.26,'TRP':2.25,'TYR':0.96,'VAL':1.22
}

def parse_pdb(path):
    atoms=[]
    for ln in open(path):
        if not ln.startswith('ATOM'): continue
        rec = {
            'name': ln[12:16].strip(),
            'res': ln[17:20].strip(),
            'chain': ln[21],
            'resi': int(ln[22:26]),
            'x': float(ln[30:38]),
            'y': float(ln[38:46]),
            'z': float(ln[46:54]),
            'elem': ln[76:78].strip() or ln[12:16].strip()[0],
        }
        atoms.append(rec)
    return atoms

def atom_charge(a):
    if (a['res'],a['name']) in SIDECHAIN_CHG: return SIDECHAIN_CHG[(a['res'],a['name'])]
    return DEFAULT_CHG.get(a['name'],0.0)

def atom_vdw(a):
    return VDW_RADIUS.get(a['elem'],1.80)

def to_arrays(atoms):
    coords = np.array([[a['x'],a['y'],a['z']] for a in atoms])
    chains = np.array([a['chain'] for a in atoms])
    resi = np.array([a['resi'] for a in atoms])
    res = np.array([a['res'] for a in atoms])
    name = np.array([a['name'] for a in atoms])
    elem = np.array([a['elem'] for a in atoms])
    chg  = np.array([atom_charge(a) for a in atoms])
    vdw  = np.array([atom_vdw(a) for a in atoms])
    return dict(coords=coords,chains=chains,resi=resi,res=res,name=name,elem=elem,chg=chg,vdw=vdw)

def main():
    atoms = parse_pdb('data/1brs_AD.pdb')
    A = to_arrays(atoms)
    np.savez('outputs/structure.npz', **A)
    print('atoms total:', len(atoms))
    # interface residues: any heavy atom within 5 Å across chains
    iA = A['chains']=='A'; iD = A['chains']=='D'
    cA = A['coords'][iA]; cD = A['coords'][iD]
    rA = A['resi'][iA]; rD = A['resi'][iD]
    # pairwise distance
    d = np.linalg.norm(cA[:,None,:]-cD[None,:,:],axis=2)
    pairs = np.argwhere(d<5.0)
    actA = sorted(set(rA[pairs[:,0]].tolist()))
    actD = sorted(set(rD[pairs[:,1]].tolist()))
    print('Active barnase residues (chain A):', actA)
    print('Active barstar residues (chain D):', actD)
    # passive: same-chain residues with any heavy atom within 6.5 Å of an active residue
    def passive(chain_mask, resids, active_set):
        coords = A['coords'][chain_mask]
        resi = A['resi'][chain_mask]
        active_idx = np.array([i for i,r in enumerate(resi) if r in active_set])
        if len(active_idx)==0: return []
        ac = coords[active_idx]
        passive=set()
        for i,r in enumerate(resi):
            if r in active_set: continue
            d=np.linalg.norm(coords[i]-ac, axis=1).min()
            if d<6.5: passive.add(int(r))
        return sorted(passive)
    pasA = passive(iA, rA, set(actA))
    pasD = passive(iD, rD, set(actD))
    out = {
        'active_chainA_barnase': actA,
        'active_chainD_barstar': actD,
        'passive_chainA_barnase': pasA,
        'passive_chainD_barstar': pasD,
        'definition': 'active = any heavy atom within 5 A of the partner chain in the bound 1BRS structure; passive = same-chain residues within 6.5 A of an active residue.'
    }
    json.dump(out, open('outputs/airs.json','w'), indent=2)
    print('Active A:', len(actA),'passive A:', len(pasA))
    print('Active D:', len(actD),'passive D:', len(pasD))

if __name__=='__main__':
    main()
