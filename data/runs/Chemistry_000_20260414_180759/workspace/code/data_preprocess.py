import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import json

# Scaffold split func
def scaffold_to_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    scaffold = Chem.MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def scaffold_split(df, sizes=(0.8,0.1,0.1), seed=42):
    np.random.seed(seed)
    df_scaff = df.copy()
    df_scaff['scaffold'] = df_scaff['smiles'].apply(scaffold_to_smiles)
    scaffold_groups = df_scaff.groupby('scaffold')
    scaffolds = list(scaffold_groups.groups.keys())
    np.random.shuffle(scaffolds)
    train_cut = int(len(scaffolds) * sizes[0])
    val_cut = int(len(scaffolds) * (sizes[0] + sizes[1]))
    train_scaffs = set(scaffolds[:train_cut])
    val_scaffs = set(scaffolds[train_cut:val_cut])
    test_scaffs = set(scaffolds[val_cut:])
    train_df = df_scaff[df_scaff.scaffold.isin(train_scaffs)]
    val_df = df_scaff[df_scaff.scaffold.isin(val_scaffs)]
    test_df = df_scaff[df_scaff.scaffold.isin(test_scaffs)]
    return train_df, val_df, test_df

# Atom feat func
def atom_featurize(atom):
    atomic_num = [0] * 119
    atomic_num[atom.GetAtomicNum()] = 1
    degree = [0] * 6
    degree[atom.GetDegree()] = 1
    formal_charge = atom.GetFormalCharge()
    formal_charge = [0] * 13
    formal_charge[formal_charge + 5] = 1  # -5 to 5 mapped 0-10
    chiral_tag = [0] * 2
    if atom.HasProp('_ChiralityPossible') or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        chiral_tag[1] = 1
    hcount = atom.GetTotalNumHs()
    hcount = [0] * 5
    hcount[min(hcount,4)] = 1
    return atomic_num + degree + formal_charge + chiral_tag + hcount

# Bond feat
def bond_featurize(bond):
    bt = bond.GetBondType()
    bond_type = [0,0,0,0]  # single, double, triple, aromatic
    if bt == Chem.BondType.SINGLE:
        bond_type[0] = 1
    elif bt == Chem.BondType.DOUBLE:
        bond_type[1] = 1
    elif bt == Chem.BondType.TRIPLE:
        bond_type[2] = 1
    elif bt == Chem.BondType.AROMATIC:
        bond_type[3] = 1
    stereo = [0] * 4  # none, any, e,z
    stereo[bond.GetStereo()] = 1
    return bond_type + stereo

# Smiles to Data
def smiles_to_graph(smiles, label_dict=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    
    n_atoms = mol.GetNumAtoms()
    edge_index = []
    edge_attr = []
    
    # Covalent edges
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        edge_attr.append(bond_featurize(bond))
        edge_attr.append(bond_featurize(bond))
    
    # Non-covalent: dist < 4A
    cutoff = 4.0
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < cutoff:
                edge_type = [0,1]  # [cov=0, noncov=1]
                edge_attr += [edge_type + [0.0]] * 2  # dummy feat for now
                edge_index += [[i, j], [j, i]]
    
    x = torch.tensor([atom_featurize(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=torch.tensor(pos, dtype=torch.float))
    
    if label_dict:
        y = torch.tensor([label_dict.get(k, 0) for k in label_dict.keys() if k in label_dict], dtype=torch.float)
        data.y = y
    else:
        data.y = torch.tensor([0], dtype=torch.long)  # dummy
    return data

# Main
datasets_info = {
    'bace': {'label_col': 'label', 'multi_task': False},
    'bbbp': {'label_col': 'label', 'multi_task': False},
    'clintox': {'label_cols': ['FDA_APPROVED', 'CT_TOX'], 'multi_task': True},
    'hiv': {'label_col': 'label', 'multi_task': False},
    'muv': {'label_cols': [f'MUV-{i}' for i in [466,548,600,644,652,689,692,712,713,733,737,810,832,846,852,858,859]], 'multi_task': True},
}

fig, axes = plt.subplots(2,3, figsize=(15,10))
axes = axes.flatten()
i = 0
for name in datasets_info:
    df = pd.read_csv(f'data/{name}.csv')
    train_df, val_df, test_df = scaffold_split(df)
    # Plot balance for main label
    if not datasets_info[name]['multi_task']:
        label_col = datasets_info[name]['label_col']
        balances = {'train': train_df[label_col].value_counts(normalize=True), 'val': val_df[label_col].value_counts(normalize=True), 'test': test_df[label_col].value_counts(normalize=True)}
        sns.barplot(data=pd.DataFrame(balances).T, ax=axes[i])
        axes[i].set_title(f'{name} label balance')
    else:
        # Avg balance
        label_cols = datasets_info[name]['label_cols']
        avg_pos = df[label_cols].mean().mean()
        axes[i].bar(['pos avg'], [avg_pos])
        axes[i].set_title(f'{name} avg task balance')
    i +=1
axes[-1].remove()
plt.tight_layout()
plt.savefig('report/images/data_overview.png')
plt.close()

# Save stats update
with open('outputs/data_stats.json', 'r') as f:
    stats = json.load(f)
for name in datasets_info:
    stats[name]['scaffold_split_sizes'] = {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)}
with open('outputs/data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print('Data overview plot saved. Stats updated.')
