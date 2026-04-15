"""
Data exploration and preprocessing for vitrimer inverse design framework.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import DataStructs
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_data():
    """Load calibration and vitrimer data."""
    cal_df = pd.read_csv('data/tg_calibration.csv')
    vit_df = pd.read_csv('data/tg_vitrimer_MD.csv')
    return cal_df, vit_df

def analyze_calibration_data(cal_df):
    """Analyze the calibration dataset."""
    print("=== Calibration Data Analysis ===")
    print(f"Total polymers: {len(cal_df)}")
    print(f"\nTg Statistics (K):")
    print(f"  Experimental Tg: {cal_df['tg_exp'].describe()}")
    print(f"  MD Simulated Tg: {cal_df['tg_md'].describe()}")
    
    # Calculate error
    cal_df['error'] = cal_df['tg_md'] - cal_df['tg_exp']
    cal_df['abs_error'] = np.abs(cal_df['error'])
    cal_df['pct_error'] = 100 * cal_df['abs_error'] / cal_df['tg_exp']
    
    print(f"\nMD Simulation Error:")
    print(f"  Mean Absolute Error: {cal_df['abs_error'].mean():.2f} K")
    print(f"  RMSE: {np.sqrt((cal_df['error']**2).mean()):.2f} K")
    print(f"  Mean % Error: {cal_df['pct_error'].mean():.2f}%")
    
    return cal_df

def analyze_vitrimer_data(vit_df):
    """Analyze the vitrimer MD dataset."""
    print("\n=== Vitrimer MD Data Analysis ===")
    print(f"Total vitrimer systems: {len(vit_df)}")
    print(f"\nMD Tg Statistics (K):")
    print(vit_df['tg'].describe())
    
    # Parse SMILES to get unique acids and epoxides
    unique_acids = vit_df['acid'].nunique()
    unique_epoxides = vit_df['epoxide'].nunique()
    print(f"\nUnique acids: {unique_acids}")
    print(f"Unique epoxides: {unique_epoxides}")
    
    return vit_df

def compute_molecular_features(smiles_list):
    """Compute molecular descriptors from SMILES."""
    features = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            # Compute descriptors
            feat = {
                'mol_weight': Descriptors.MolWt(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                'num_rotatable': Descriptors.NumRotatableBonds(mol),
                'num_hbd': rdMolDescriptors.CalcNumHBD(mol),
                'num_hba': rdMolDescriptors.CalcNumHBA(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            }
            features.append(feat)
            valid_indices.append(i)
        except Exception as e:
            continue
    
    return pd.DataFrame(features), valid_indices

def compute_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    """Compute Morgan fingerprints from SMILES."""
    fps = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
            arr = np.zeros((n_bits,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_indices.append(i)
        except Exception as e:
            continue
    
    return np.array(fps), valid_indices

def create_visualizations(cal_df, vit_df):
    """Create data overview visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Calibration: Experimental vs MD Tg
    ax = axes[0, 0]
    ax.scatter(cal_df['tg_exp'], cal_df['tg_md'], alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.plot([150, 550], [150, 550], 'r--', label='Perfect Agreement')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('MD Simulated Tg (K)')
    ax.set_title('Calibration Data: MD vs Experimental Tg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Calibration error distribution
    ax = axes[0, 1]
    ax.hist(cal_df['error'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='r', linestyle='--', label='Zero Error')
    ax.set_xlabel('Prediction Error (MD - Exp) K')
    ax.set_ylabel('Frequency')
    ax.set_title('MD Prediction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Tg distributions comparison
    ax = axes[0, 2]
    ax.hist(cal_df['tg_exp'], bins=25, alpha=0.6, label='Experimental', edgecolor='black')
    ax.hist(cal_df['tg_md'], bins=25, alpha=0.6, label='MD Simulated', edgecolor='black')
    ax.set_xlabel('Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Tg Distribution: Calibration Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Vitrimer Tg distribution
    ax = axes[1, 0]
    ax.hist(vit_df['tg'], bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
    ax.axvline(vit_df['tg'].mean(), color='r', linestyle='--', 
               label=f'Mean: {vit_df["tg"].mean():.1f} K')
    ax.set_xlabel('MD Simulated Tg (K)')
    ax.set_ylabel('Frequency')
    ax.set_title('Vitrimer Systems: Tg Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Vitrimer Tg vs Std
    ax = axes[1, 1]
    ax.scatter(vit_df['tg'], vit_df['std'], alpha=0.4, s=10)
    ax.set_xlabel('Tg (K)')
    ax.set_ylabel('Standard Deviation (K)')
    ax.set_title('Vitrimer Tg vs Uncertainty')
    ax.grid(True, alpha=0.3)
    
    # 6. Error vs Tg
    ax = axes[1, 2]
    ax.scatter(cal_df['tg_exp'], cal_df['error'], alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Experimental Tg (K)')
    ax.set_ylabel('MD Error (K)')
    ax.set_title('MD Error vs Experimental Tg')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/data_overview.png', dpi=150, bbox_inches='tight')
    print("\nSaved: report/images/data_overview.png")
    plt.close()

def main():
    # Load data
    cal_df, vit_df = load_data()
    
    # Analyze data
    cal_df = analyze_calibration_data(cal_df)
    vit_df = analyze_vitrimer_data(vit_df)
    
    # Create visualizations
    create_visualizations(cal_df, vit_df)
    
    # Save processed data
    cal_df.to_csv('outputs/calibration_processed.csv', index=False)
    vit_df.to_csv('outputs/vitrimer_processed.csv', index=False)
    
    print("\n=== Data Exploration Complete ===")
    return cal_df, vit_df

if __name__ == "__main__":
    main()
