"""
Phase 1: Data exploration and fingerprint generation.
Generates Morgan fingerprints from SMILES for both datasets.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import json
import os

os.makedirs('outputs', exist_ok=True)

# Load data
cal_df = pd.read_csv('data/tg_calibration.csv')
vit_df = pd.read_csv('data/tg_vitrimer_MD.csv')

print(f"Calibration data: {len(cal_df)} entries")
print(f"Vitrimer MD data: {len(vit_df)} entries")
print(f"\nCalibration columns: {list(cal_df.columns)}")
print(f"Vitrimer columns: {list(vit_df.columns)}")
print(f"\nCalibration Tg stats:")
print(cal_df[['tg_exp', 'tg_md', 'std']].describe())
print(f"\nVitrimer Tg stats:")
print(vit_df[['tg', 'std']].describe())

# Generate Morgan fingerprints for calibration data
def smiles_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0
    return arr

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

# Calibration fingerprints
cal_fps = []
cal_valid = []
for i, row in cal_df.iterrows():
    fp = smiles_to_fp(row['smiles'])
    if fp is not None:
        cal_fps.append(fp)
        cal_valid.append(i)

cal_fps = np.array(cal_fps)
print(f"\nValid calibration fingerprints: {len(cal_fps)}/{len(cal_df)}")

# Compute basic molecular descriptors for calibration
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = {
        'mol_wt': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_rings': Descriptors.RingCount(mol),
        'num_hba': Descriptors.NumHAcceptors(mol),
        'num_hbd': Descriptors.NumHDonors(mol),
        'num_rot_bonds': Descriptors.NumRotatableBonds(mol),
        'tpsa': Descriptors.TPSA(mol),
    }
    return desc

cal_descs = []
for i, row in cal_df.iterrows():
    d = compute_descriptors(row['smiles'])
    if d is not None:
        cal_descs.append(d)

cal_desc_df = pd.DataFrame(cal_descs)
print(f"\nDescriptor statistics:")
print(cal_desc_df.describe())

# Save fingerprints and descriptors
np.save('outputs/cal_fps.npy', cal_fps)
np.save('outputs/cal_valid_idx.npy', np.array(cal_valid))
cal_desc_df.to_csv('outputs/cal_descriptors.csv', index=False)

# For vitrimer data, we have acid + epoxide pairs
# Generate combined fingerprints for each pair
vit_fps = []
vit_valid = []
vit_acid_fps = []
vit_epox_fps = []

for i, row in vit_df.iterrows():
    acid_fp = smiles_to_fp(row['acid'])
    epox_fp = smiles_to_fp(row['epoxide'])
    if acid_fp is not None and epox_fp is not None:
        # Concatenate acid and epoxide fingerprints
        combined = np.concatenate([acid_fp, epox_fp])
        vit_fps.append(combined)
        vit_acid_fps.append(acid_fp)
        vit_epox_fps.append(epox_fp)
        vit_valid.append(i)

vit_fps = np.array(vit_fps)
print(f"\nValid vitrimer fingerprints: {len(vit_fps)}/{len(vit_df)}")
print(f"Combined fingerprint dimension: {vit_fps.shape[1]}")

# Save vitrimer fingerprints in compressed format
np.savez_compressed('outputs/vit_fps.npz', vit_fps=vit_fps)
np.save('outputs/vit_valid_idx.npy', np.array(vit_valid))

# Save summary stats
summary = {
    'cal_n_total': len(cal_df),
    'cal_names': cal_df['name'].tolist(),
    'cal_smiles': cal_df['smiles'].tolist(),
    'cal_tg_exp': cal_df['tg_exp'].tolist(),
    'cal_tg_md': cal_df['tg_md'].tolist(),
    'cal_n_valid': len(cal_fps),
    'cal_tg_exp_mean': float(cal_df['tg_exp'].mean()),
    'cal_tg_exp_std': float(cal_df['tg_exp'].std()),
    'cal_tg_md_mean': float(cal_df['tg_md'].mean()),
    'cal_tg_md_std': float(cal_df['tg_md'].std()),
    'vit_n_total': len(vit_df),
    'vit_n_valid': len(vit_fps),
    'vit_tg_mean': float(vit_df['tg'].mean()),
    'vit_tg_std': float(vit_df['tg'].std()),
    'vit_acid_smiles': vit_df['acid'].tolist(),
    'vit_epoxide_smiles': vit_df['epoxide'].tolist(),
    'vit_tg_md': vit_df['tg'].tolist(),
    'vit_std': vit_df['std'].tolist(),
    'fp_dim': int(cal_fps.shape[1]),
    'combined_fp_dim': int(vit_fps.shape[1]),
}
with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nData exploration complete. Files saved to outputs/")
