"""
Step 2: Molecular Feature Engineering
- Compute Morgan fingerprints and molecular descriptors from SMILES
- For vitrimer systems: combine acid + epoxide features
- Prepare data for Graph VAE training
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import os
import json
import pickle

os.makedirs('../outputs', exist_ok=True)

# Load data
df_cal = pd.read_csv('../data/tg_calibration.csv')
df_vitrimer = pd.read_csv('../outputs/vitrimer_calibrated.csv')

# ==============================
# Feature computation functions
# ==============================
def compute_morgan_fp(smiles, radius=2, n_bits=1024):
    """Compute Morgan fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def compute_molecular_descriptors(smiles):
    """Compute a set of molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: 0.0 for k in ['mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotbonds', 'aromatic_rings', 'heavy_atoms', 'fraction_csp3']}
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'rotbonds': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
        'heavy_atoms': Descriptors.HeavyAtomCount(mol),
        'fraction_csp3': Descriptors.FractionCSP3(mol),
    }

# ==============================
# Process calibration data
# ==============================
print("Processing calibration data...")
cal_fps = []
cal_descs = []
for i, row in df_cal.iterrows():
    fp = compute_morgan_fp(row['smiles'])
    desc = compute_molecular_descriptors(row['smiles'])
    cal_fps.append(fp)
    cal_descs.append(desc)

cal_fp_array = np.array(cal_fps)
cal_desc_df = pd.DataFrame(cal_descs)

np.save('../outputs/cal_morgan_fps.npy', cal_fp_array)
cal_desc_df.to_csv('../outputs/cal_descriptors.csv', index=False)
print(f"Calibration fingerprints: {cal_fp_array.shape}")
print(f"Calibration descriptors: {cal_desc_df.shape}")

# ==============================
# Process vitrimer data
# ==============================
print("\nProcessing vitrimer data...")
vitrimer_acid_fps = []
vitrimer_epoxide_fps = []
vitrimer_acid_descs = []
vitrimer_epoxide_descs = []
combined_fps = []
combined_descs = []

for i, row in df_vitrimer.iterrows():
    acid_fp = compute_morgan_fp(row['acid'])
    epoxide_fp = compute_morgan_fp(row['epoxide'])
    acid_desc = compute_molecular_descriptors(row['acid'])
    epoxide_desc = compute_molecular_descriptors(row['epoxide'])
    
    vitrimer_acid_fps.append(acid_fp)
    vitrimer_epoxide_fps.append(epoxide_fp)
    vitrimer_acid_descs.append(acid_desc)
    vitrimer_epoxide_descs.append(epoxide_desc)
    
    # Combined: concatenate fingerprints
    combined_fp = np.concatenate([acid_fp, epoxide_fp])
    combined_fps.append(combined_fp)
    
    # Combined descriptors: prefix with acid_/epoxide_
    combined_desc = {}
    for k, v in acid_desc.items():
        combined_desc[f'acid_{k}'] = v
    for k, v in epoxide_desc.items():
        combined_desc[f'epoxide_{k}'] = v
    combined_descs.append(combined_desc)

vitrimer_acid_fps = np.array(vitrimer_acid_fps)
vitrimer_epoxide_fps = np.array(vitrimer_epoxide_fps)
combined_fps = np.array(combined_fps)
combined_desc_df = pd.DataFrame(combined_descs)

np.save('../outputs/vitrimer_acid_fps.npy', vitrimer_acid_fps)
np.save('../outputs/vitrimer_epoxide_fps.npy', vitrimer_epoxide_fps)
np.save('../outputs/vitrimer_combined_fps.npy', combined_fps)
combined_desc_df.to_csv('../outputs/vitrimer_combined_descriptors.csv', index=False)

print(f"Vitrimer acid fingerprints: {vitrimer_acid_fps.shape}")
print(f"Vitrimer epoxide fingerprints: {vitrimer_epoxide_fps.shape}")
print(f"Vitrimer combined fingerprints: {combined_fps.shape}")
print(f"Vitrimer combined descriptors: {combined_desc_df.shape}")

# ==============================
# Build molecular graphs for Graph VAE
# ==============================
print("\nBuilding molecular graphs...")

def smiles_to_graph(smiles):
    """Convert SMILES to graph representation (node features + edge list)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),           # Atomic number
            atom.GetDegree(),              # Degree
            atom.GetFormalCharge(),        # Formal charge
            int(atom.GetIsAromatic()),     # Aromaticity
            atom.GetTotalNumHs(),          # Total Hs
            atom.GetTotalValence(),        # Total valence
            int(atom.IsInRing()),          # In ring
        ]
        atom_features.append(features)
    
    # Edge list (bonds)
    edge_list = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edge_list.append([i, j])
        edge_list.append([j, i])
        edge_features.append([bond_type, int(bond.GetIsAromatic()), int(bond.IsInRing())])
        edge_features.append([bond_type, int(bond.GetIsAromatic()), int(bond.IsInRing())])
    
    return {
        'atom_features': np.array(atom_features, dtype=np.float32),
        'edge_list': np.array(edge_list, dtype=np.int64) if edge_list else np.zeros((0, 2), dtype=np.int64),
        'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 3), dtype=np.float32),
        'n_atoms': len(atom_features),
        'smiles': smiles,
    }

# Build graphs for vitrimer molecules (combine acid + epoxide as concatenated SMILES)
vitrimer_graphs = []
valid_indices = []
for i, row in df_vitrimer.iterrows():
    # Create combined molecule representation
    combined_smiles = row['acid'] + '.' + row['epoxide']
    graph = smiles_to_graph(row['acid'])  # We'll use acid as primary for VAE
    if graph is not None:
        vitrimer_graphs.append(graph)
        valid_indices.append(i)

print(f"Valid vitrimer graphs: {len(vitrimer_graphs)}")

# Save graph data
with open('../outputs/vitrimer_graphs.pkl', 'wb') as f:
    pickle.dump(vitrimer_graphs, f)

# Also build graphs for all unique SMILES (acid + epoxide separately)
all_smiles = set()
for _, row in df_vitrimer.iterrows():
    all_smiles.add(row['acid'])
    all_smiles.add(row['epoxide'])
for _, row in df_cal.iterrows():
    all_smiles.add(row['smiles'])

all_smiles = list(all_smiles)
print(f"Total unique SMILES: {len(all_smiles)}")

all_graphs = []
valid_smiles = []
for smi in all_smiles:
    g = smiles_to_graph(smi)
    if g is not None:
        all_graphs.append(g)
        valid_smiles.append(smi)

print(f"Valid graphs from unique SMILES: {len(all_graphs)}")

with open('../outputs/all_molecule_graphs.pkl', 'wb') as f:
    pickle.dump({'graphs': all_graphs, 'smiles': valid_smiles}, f)

# Save atom feature dimension info
atom_feat_dim = 7  # As defined above
edge_feat_dim = 3

feat_info = {
    'atom_feat_dim': atom_feat_dim,
    'edge_feat_dim': edge_feat_dim,
    'n_unique_molecules': len(valid_smiles),
    'n_vitrimer_systems': len(vitrimer_graphs),
}
with open('../outputs/feature_info.json', 'w') as f:
    json.dump(feat_info, f, indent=2)

print("\nStep 2 complete.")
