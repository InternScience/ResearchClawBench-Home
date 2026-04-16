import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data
from multiprocessing import Pool

RDLogger.DisableLog('rdApp.*')

def smiles_to_graph(smiles_y):
    smiles, y = smiles_y
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

def process_dataset_parallel(name, path, smiles_col, target_cols, max_samples=5000):
    df = pd.read_csv(path)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        
    smiles_y_list = []
    for _, row in df.iterrows():
        smiles = row[smiles_col]
        y = row[target_cols].values.astype(float)
        smiles_y_list.append((smiles, y))
    
    with Pool(os.cpu_count()) as p:
        data_list = p.map(smiles_to_graph, smiles_y_list)
        
    data_list = [d for d in data_list if d is not None]
    print(f"Dataset {name}: {len(data_list)} valid molecules out of {len(df)}")
    return data_list

if __name__ == "__main__":
    if os.path.exists("outputs/muv.pt") and os.path.getsize("outputs/muv.pt") == 0:
        os.remove("outputs/muv.pt")
        
    if not os.path.exists("outputs/muv.pt"):
        muv_cols = [c for c in pd.read_csv("data/muv.csv", nrows=0).columns if c.startswith("MUV-")]
        muv_data = process_dataset_parallel("MUV", "data/muv.csv", "smiles", muv_cols)
        torch.save(muv_data, "outputs/muv.pt")
