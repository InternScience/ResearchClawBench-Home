import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import os

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load data
cal = pd.read_csv('data/tg_calibration.csv')
md = pd.read_csv('data/tg_vitrimer_MD.csv')

# Compute molecular descriptors for calibration data
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumHBD': Descriptors.NumHDonors(mol),
        'NumHBA': Descriptors.NumHAcceptors(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
    }

cal_desc = pd.DataFrame([compute_descriptors(s) for s in cal['smiles']])
cal = pd.concat([cal, cal_desc], axis=1)

# Compute descriptors for vitrimer components
def compute_pair_descriptors(acid, epoxide):
    a = compute_descriptors(acid)
    e = compute_descriptors(epoxide)
    result = {}
    for k in a:
        result[f'acid_{k}'] = a[k]
        result[f'epoxide_{k}'] = e[k]
        result[f'sum_{k}'] = a[k] + e[k]
        result[f'diff_{k}'] = abs(a[k] - e[k])
    return result

md_desc = pd.DataFrame([compute_pair_descriptors(a, e) for a, e in zip(md['acid'], md['epoxide'])])
md = pd.concat([md, md_desc], axis=1)

# Save processed data
cal.to_csv('outputs/calibration_with_descriptors.csv', index=False)
md.to_csv('outputs/vitrimer_with_descriptors.csv', index=False)

# Figure 1: Data overview - calibration
def create_figures():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Calibration: MD vs Exp
    ax = axes[0, 0]
    ax.scatter(cal['tg_exp'], cal['tg_md'], alpha=0.6, c='steelblue', edgecolors='k', linewidths=0.3)
    ax.plot([150, 650], [150, 650], 'r--', lw=1.5, label='Perfect agreement')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('MD Simulated Tg (K)')
    ax.set_title('Calibration Data: MD vs Experimental Tg')
    ax.legend()
    
    # Calibration: Residuals
    ax = axes[0, 1]
    residuals = cal['tg_md'] - cal['tg_exp']
    ax.hist(residuals, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('MD - Experimental Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Calibration Residuals Distribution')
    
    # Calibration: Residuals vs Exp
    ax = axes[0, 2]
    ax.scatter(cal['tg_exp'], residuals, alpha=0.6, c='steelblue', edgecolors='k', linewidths=0.3)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('MD - Experimental Tg (K)')
    ax.set_title('Residuals vs Experimental Tg')
    
    # Vitrimer: Tg distribution
    ax = axes[1, 0]
    ax.hist(md['tg'], bins=50, color='darkgreen', edgecolor='k', alpha=0.7)
    ax.set_xlabel('MD Simulated Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Vitrimer MD Tg Distribution')
    
    # Vitrimer: Tg vs std
    ax = axes[1, 1]
    ax.scatter(md['tg'], md['std'], alpha=0.3, c='darkgreen', edgecolors='k', linewidths=0.1)
    ax.set_xlabel('MD Simulated Tg (K)')
    ax.set_ylabel('Standard Deviation (K)')
    ax.set_title('Vitrimer Tg vs Uncertainty')
    
    # Descriptor correlation heatmap (calibration)
    ax = axes[1, 2]
    desc_cols = ['MolWt', 'LogP', 'TPSA', 'NumRotatableBonds', 'NumHBD', 'NumHBA']
    corr = cal[desc_cols + ['tg_exp', 'tg_md']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, square=True, cbar_kws={'shrink': 0.7})
    ax.set_title('Descriptor Correlation (Calibration)')
    
    plt.tight_layout()
    plt.savefig('report/images/fig01_data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Figure 1 saved.")

create_figures()
print("Data exploration complete.")
