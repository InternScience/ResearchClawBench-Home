import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')

def smiles_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features (Atom features)
    atom_features = []
    for atom in mol.GetAtoms():
        feature = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
        ]
        atom_features.append(feature)
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge features (Bond features)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        feature = [
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
        
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feature, feature]
        
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
        
    y_tensor = torch.tensor([y], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
    return data

def process_dataset(name, path, smiles_col, target_cols):
    df = pd.read_csv(path)
    data_list = []
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        y = row[target_cols].values.astype(float)
        data = smiles_to_graph(smiles, y)
        if data is not None:
            data_list.append(data)
    print(f"Dataset {name}: {len(data_list)} valid molecules out of {len(df)}")
    return data_list

if __name__ == "__main__":
    bace_data = process_dataset("BACE", "data/bace.csv", "smiles", ["label"])
    torch.save(bace_data, "outputs/bace.pt")
    
    bbbp_data = process_dataset("BBBP", "data/bbbp.csv", "smiles", ["label"])
    torch.save(bbbp_data, "outputs/bbbp.pt")
    
    clintox_data = process_dataset("ClinTox", "data/clintox.csv", "smiles", ["FDA_APPROVED", "CT_TOX"])
    torch.save(clintox_data, "outputs/clintox.pt")
    
    hiv_data = process_dataset("HIV", "data/hiv.csv", "smiles", ["label"])
    torch.save(hiv_data, "outputs/hiv.pt")
    
    muv_cols = [c for c in pd.read_csv("data/muv.csv", nrows=0).columns if c.startswith("MUV-")]
    muv_data = process_dataset("MUV", "data/muv.csv", "smiles", muv_cols)
    torch.save(muv_data, "outputs/muv.pt")
