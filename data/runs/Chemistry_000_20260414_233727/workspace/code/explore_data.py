"""
Data exploration script for molecular property prediction datasets.
"""
import pandas as pd
import numpy as np
import json
from rdkit import Chem

datasets = {
    'bace': 'data/bace.csv',
    'bbbp': 'data/bbbp.csv',
    'clintox': 'data/clintox.csv',
    'hiv': 'data/hiv.csv',
    'muv': 'data/muv.csv',
}

summary = {}

for name, path in datasets.items():
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Find SMILES column
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower() or 'smile' in col.lower():
            smiles_col = col
            break
    
    if smiles_col:
        valid_mols = 0
        total = len(df)
        atom_counts = []
        bond_counts = []
        for smi in df[smiles_col].dropna():
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols += 1
                atom_counts.append(mol.GetNumAtoms())
                bond_counts.append(mol.GetNumBonds())
        
        print(f"SMILES column: {smiles_col}")
        print(f"Valid molecules: {valid_mols}/{total}")
        if atom_counts:
            print(f"Atom count - mean: {np.mean(atom_counts):.1f}, std: {np.std(atom_counts):.1f}, min: {min(atom_counts)}, max: {max(atom_counts)}")
            print(f"Bond count - mean: {np.mean(bond_counts):.1f}, std: {np.std(bond_counts):.1f}, min: {min(bond_counts)}, max: {max(bond_counts)}")
    
    # Find label columns
    label_cols = [c for c in df.columns if c.lower() not in ['smiles', 'mol_id', 'cid', 'id', 'num', 'name'] and df[c].dtype in ['int64', 'float64', 'bool']]
    for lc in label_cols:
        non_null = df[lc].notna().sum()
        val_counts = df[lc].value_counts().to_dict()
        print(f"Label '{lc}': {non_null} non-null, distribution: {val_counts}")
    
    summary[name] = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'smiles_column': smiles_col,
        'label_columns': label_cols,
        'valid_molecules': valid_mols if smiles_col else 0,
        'total_molecules': total if smiles_col else 0,
        'atom_mean': float(np.mean(atom_counts)) if atom_counts else None,
        'atom_std': float(np.std(atom_counts)) if atom_counts else None,
        'atom_min': int(min(atom_counts)) if atom_counts else None,
        'atom_max': int(max(atom_counts)) if atom_counts else None,
    }

with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nData summary saved to outputs/data_summary.json")
