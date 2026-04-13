import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parents[1]
PDB_PATH = ROOT / 'data/sample/2l3r/2l3r_protein.pdb'
SDF_PATH = ROOT / 'data/sample/2l3r/2l3r_ligand.sdf'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report/images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)
sns.set_theme(style='whitegrid')

AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}


def parse_pdb_ca(path):
    atoms=[]
    seq=[]
    seen=[]
    with open(path) as f:
        for line in f:
            if line.startswith('SEQRES'):
                parts=line.split()[4:]
                seq.extend(parts)
            if line.startswith('ATOM'):
                name=line[12:16].strip()
                resn=line[17:20].strip()
                chain=line[21].strip() or 'A'
                resi=int(line[22:26])
                x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
                atoms.append((name,resn,chain,resi,x,y,z))
                if name=='CA':
                    seen.append((resn,chain,resi,x,y,z))
    ca=np.array([[x,y,z] for _,_,_,x,y,z in seen],dtype=float)
    seq1=''.join(AA3_TO_1.get(r,'X') for r in seq)
    return atoms, seen, ca, seq, seq1


def kabsch(P,Q):
    Pc=P-P.mean(0)
    Qc=Q-Q.mean(0)
    C=Pc.T@Qc
    V,S,Wt=np.linalg.svd(C)
    d=np.sign(np.linalg.det(V@Wt))
    D=np.diag([1,1,d])
    U=V@D@Wt
    return U, P.mean(0), Q.mean(0)


def apply_alignment(P,U,cP,cQ):
    return (P-cP)@U + cQ


def rmsd(P,Q):
    return float(np.sqrt(np.mean(np.sum((P-Q)**2,axis=1))))


def random_backbone_predictions(ca, n=32, seed=0):
    rng=np.random.default_rng(seed)
    preds=[]
    for i in range(n):
        noise_scale=0.4 + 0.08*i
        local = ca + rng.normal(scale=noise_scale, size=ca.shape)
        # random rigid transform before alignment to emulate global uncertainty
        # simple random rotation from QR decomposition
        Qm,_r=np.linalg.qr(rng.normal(size=(3,3)))
        t=rng.normal(scale=3.0,size=3)
        pred=local@Qm + t
        preds.append(pred)
    return preds


def parse_ligand(path):
    mol=Chem.SDMolSupplier(str(path), removeHs=False)[0]
    conf=mol.GetConformer()
    coords=np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],dtype=float)
    atoms=[a.GetSymbol() for a in mol.GetAtoms()]
    return mol, coords, atoms


def random_ligand_predictions(coords, n=32, seed=1):
    rng=np.random.default_rng(seed)
    preds=[]
    for i in range(n):
        noise_scale=0.15 + 0.05*i
        pred=coords + rng.normal(scale=noise_scale,size=coords.shape)
        Qm,_r=np.linalg.qr(rng.normal(size=(3,3)))
        t=rng.normal(scale=1.5,size=3)
        pred=pred@Qm + t
        preds.append(pred)
    return preds


def contact_map(ca, threshold=8.0):
    D=np.sqrt(((ca[:,None,:]-ca[None,:,:])**2).sum(-1))
    return D, (D<threshold).astype(int)


def ligand_protein_contacts(ca, lig, threshold=6.0):
    D=np.sqrt(((ca[:,None,:]-lig[None,:,:])**2).sum(-1))
    return D, (D.min(axis=1)<threshold).astype(int)


def main():
    atoms, ca_rows, ca, seq3, seq1 = parse_pdb_ca(PDB_PATH)
    mol, lig_coords, lig_atoms = parse_ligand(SDF_PATH)

    backbone_preds = random_backbone_predictions(ca)
    ligand_preds = random_ligand_predictions(lig_coords)

    bb_metrics=[]
    lig_metrics=[]
    for i,pred in enumerate(backbone_preds):
        U,cP,cQ = kabsch(pred, ca)
        aligned=apply_alignment(pred,U,cP,cQ)
        bb_metrics.append({'sample':i,'rmsd':rmsd(aligned,ca)})
    for i,pred in enumerate(ligand_preds):
        U,cP,cQ = kabsch(pred, lig_coords)
        aligned=apply_alignment(pred,U,cP,cQ)
        lig_metrics.append({'sample':i,'rmsd':rmsd(aligned,lig_coords)})

    D, cmap = contact_map(ca)
    Dpl, pl_contact = ligand_protein_contacts(ca, lig_coords)

    protein_summary = {
        'seqres_length': len(seq3),
        'ca_atoms': int(len(ca_rows)),
        'sequence_preview': seq1[:50],
        'centroid': ca.mean(0).round(3).tolist(),
        'radius_of_gyration_ca': float(np.sqrt(((ca-ca.mean(0))**2).sum()/len(ca))),
    }
    ligand_summary = {
        'num_atoms': int(mol.GetNumAtoms()),
        'num_bonds': int(mol.GetNumBonds()),
        'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
        'mol_wt': float(Descriptors.MolWt(mol)),
        'heavy_atoms': int(mol.GetNumHeavyAtoms()),
        'centroid': lig_coords.mean(0).round(3).tolist(),
    }

    df_bb=pd.DataFrame(bb_metrics)
    df_lig=pd.DataFrame(lig_metrics)
    summary={
        'protein': protein_summary,
        'ligand': ligand_summary,
        'contacts': {
            'protein_residue_residue_contacts_lt8A': int(np.triu(cmap,1).sum()),
            'protein_residues_within_6A_of_ligand': int(pl_contact.sum()),
            'min_protein_ligand_distance': float(Dpl.min()),
        },
        'toy_eval': {
            'protein_rmsd_mean': float(df_bb.rmsd.mean()),
            'protein_rmsd_std': float(df_bb.rmsd.std()),
            'ligand_rmsd_mean': float(df_lig.rmsd.mean()),
            'ligand_rmsd_std': float(df_lig.rmsd.std()),
        }
    }

    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    df_bb.to_csv(OUT/'protein_rmsd_samples.csv',index=False)
    df_lig.to_csv(OUT/'ligand_rmsd_samples.csv',index=False)

    plt.figure(figsize=(6,5))
    plt.scatter(ca[:,0],ca[:,1],s=20,c=np.arange(len(ca)),cmap='viridis')
    plt.scatter(lig_coords[:,0],lig_coords[:,1],s=30,c='red',label='Ligand atoms')
    plt.xlabel('X (Å)'); plt.ylabel('Y (Å)'); plt.title('2L3R protein CA trace and ligand projection')
    plt.legend(); plt.tight_layout(); plt.savefig(IMG/'figure_data_overview.png',dpi=200); plt.close()

    plt.figure(figsize=(6,5))
    sns.heatmap(D, cmap='mako', cbar_kws={'label':'CA-CA distance (Å)'})
    plt.title('Protein intramolecular CA distance map')
    plt.xlabel('Residue index'); plt.ylabel('Residue index')
    plt.tight_layout(); plt.savefig(IMG/'figure_contact_map.png',dpi=200); plt.close()

    plt.figure(figsize=(7,4))
    sns.histplot(df_bb['rmsd'], bins=10, color='steelblue', label='Protein', kde=True)
    sns.histplot(df_lig['rmsd'], bins=10, color='darkorange', label='Ligand', kde=True)
    plt.xlabel('Aligned RMSD (Å)'); plt.ylabel('Count')
    plt.title('Toy diffusion-sample accuracy distribution')
    plt.legend(); plt.tight_layout(); plt.savefig(IMG/'figure_rmsd_distribution.png',dpi=200); plt.close()

    # architecture schematic proxy as a bar plot of module roles
    modules=pd.DataFrame({
        'module':['Sequence encoders','Molecule graph encoder','Cross-modal token fusion','SE(3) diffusion trunk','Confidence heads'],
        'relative_complexity':[2,2,3,5,1]
    })
    plt.figure(figsize=(8,4))
    sns.barplot(data=modules, x='relative_complexity', y='module', palette='crest')
    plt.xlabel('Relative compute budget (conceptual)')
    plt.ylabel('')
    plt.title('Proposed unified framework modules')
    plt.tight_layout(); plt.savefig(IMG/'figure_architecture_modules.png',dpi=200); plt.close()

    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
