"""
Step 1: Data exploration and molecular descriptor computation.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# Load data
cal = pd.read_csv('data/tg_calibration.csv')
vit = pd.read_csv('data/tg_vitrimer_MD.csv')

print(f"Calibration data: {len(cal)} samples")
print(f"Vitrimer data: {len(vit)} samples")
print(f"\nCalibration columns: {list(cal.columns)}")
print(f"Vitrimer columns: {list(vit.columns)}")

# --- Calibration data descriptors ---
def compute_descriptors(smiles):
    """Compute molecular descriptors from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'RingCount': Descriptors.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),
        'BalabanJ': Descriptors.BalabanJ(mol) if Descriptors.BalabanJ(mol) is not None else 0,
    }
    return desc

# Process calibration data
cal_descs = []
valid_cal_idx = []
for i, row in cal.iterrows():
    d = compute_descriptors(row['smiles'])
    if d is not None:
        cal_descs.append(d)
        valid_cal_idx.append(i)

cal_desc_df = pd.DataFrame(cal_descs, index=valid_cal_idx)
cal_valid = cal.loc[valid_cal_idx].reset_index(drop=True)
cal_desc_df = cal_desc_df.reset_index(drop=True)

print(f"\nValid calibration molecules: {len(cal_valid)}/{len(cal)}")

# Process vitrimer data (combine acid + epoxide SMILES)
vit_descs = []
valid_vit_idx = []
for i, row in vit.iterrows():
    # Try acid first, then epoxide
    d_acid = compute_descriptors(row['acid'])
    d_epox = compute_descriptors(row['epoxide'])
    if d_acid is not None and d_epox is not None:
        # Combine descriptors (sum for additive properties, mean for intensive)
        combined = {}
        for k in d_acid:
            if k in ['LogP', 'FractionCSP3', 'BalabanJ']:
                combined[k] = (d_acid[k] + d_epox[k]) / 2
            else:
                combined[k] = d_acid[k] + d_epox[k]
        vit_descs.append(combined)
        valid_vit_idx.append(i)

vit_desc_df = pd.DataFrame(vit_descs, index=valid_vit_idx)
vit_valid = vit.loc[valid_vit_idx].reset_index(drop=True)
vit_desc_df = vit_desc_df.reset_index(drop=True)

print(f"Valid vitrimer molecules: {len(vit_valid)}/{len(vit)}")

# Save descriptor data
cal_valid.to_csv('outputs/calibration_data.csv', index=False)
cal_desc_df.to_csv('outputs/calibration_descriptors.csv', index=False)
vit_valid.to_csv('outputs/vitrimer_data.csv', index=False)
vit_desc_df.to_csv('outputs/vitrimer_descriptors.csv', index=False)

# --- Data overview plots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Calibration Tg distributions
ax = axes[0, 0]
ax.hist(cal_valid['tg_exp'], bins=30, alpha=0.7, label='Experimental Tg', color='steelblue')
ax.hist(cal_valid['tg_md'], bins=30, alpha=0.7, label='MD Simulated Tg', color='coral')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Count')
ax.set_title('Calibration Dataset: Tg Distributions')
ax.legend()

# Plot 2: MD vs Experimental Tg scatter
ax = axes[0, 1]
ax.scatter(cal_valid['tg_md'], cal_valid['tg_exp'], alpha=0.6, s=30, c='steelblue')
lims = [min(cal_valid['tg_md'].min(), cal_valid['tg_exp'].min()) - 20,
        max(cal_valid['tg_md'].max(), cal_valid['tg_exp'].max()) + 20]
ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
ax.set_xlabel('MD Simulated Tg (K)')
ax.set_ylabel('Experimental Tg (K)')
ax.set_title('MD vs Experimental Tg (Calibration)')
ax.legend()

# Plot 3: Vitrimer MD Tg distribution
ax = axes[0, 2]
ax.hist(vit_valid['tg'], bins=50, alpha=0.7, color='seagreen')
ax.set_xlabel('MD Simulated Tg (K)')
ax.set_ylabel('Count')
ax.set_title('Vitrimer Dataset: MD Tg Distribution')

# Plot 4: Descriptor correlations for calibration
ax = axes[1, 0]
key_descs = ['MolWt', 'LogP', 'TPSA', 'NumRotatableBonds', 'FractionCSP3']
corr_with_tg = cal_desc_df[key_descs].corrwith(cal_valid['tg_exp'])
corr_with_tg.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Correlation with Experimental Tg')
ax.set_title('Descriptor-Tg Correlations')

# Plot 5: Calibration error distribution
ax = axes[1, 1]
error = cal_valid['tg_md'] - cal_valid['tg_exp']
ax.hist(error, bins=30, alpha=0.7, color='coral')
ax.axvline(0, color='k', linestyle='--')
ax.set_xlabel('MD Tg - Experimental Tg (K)')
ax.set_ylabel('Count')
ax.set_title(f'MD Prediction Error (Mean: {error.mean():.1f} K, Std: {error.std():.1f} K)')

# Plot 6: Vitrimer descriptor distributions
ax = axes[1, 2]
ax.hist(vit_desc_df['MolWt'], bins=50, alpha=0.7, color='seagreen')
ax.set_xlabel('Combined Molecular Weight')
ax.set_ylabel('Count')
ax.set_title('Vitrimer Combined Molecular Weight')

plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# Save summary statistics
summary = {
    'calibration_samples': len(cal_valid),
    'vitrimer_samples': len(vit_valid),
    'cal_tg_exp_mean': float(cal_valid['tg_exp'].mean()),
    'cal_tg_exp_std': float(cal_valid['tg_exp'].std()),
    'cal_tg_md_mean': float(cal_valid['tg_md'].mean()),
    'cal_tg_md_std': float(cal_valid['tg_md'].std()),
    'cal_md_error_mean': float(error.mean()),
    'cal_md_error_std': float(error.std()),
    'vit_tg_md_mean': float(vit_valid['tg'].mean()),
    'vit_tg_md_std': float(vit_valid['tg'].std()),
    'descriptor_names': list(cal_desc_df.columns),
}

with open('outputs/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nData summary saved to outputs/data_summary.json")
print("Data overview plot saved to report/images/data_overview.png")
print("Done with Step 1.")
