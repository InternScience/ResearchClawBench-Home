import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import os

os.makedirs('outputs', exist_ok=True)

# Load data
md = pd.read_csv('data/tg_vitrimer_MD.csv')

# Load GP model
model_data = joblib.load('outputs/gp_model.pkl')
gp = model_data['gp']
scaler_X = model_data['scaler_X']
scaler_y = model_data['scaler_y']

# Compute fingerprints for vitrimer data
def morgan_fp(smiles, radius=2, n_bits=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))

# For vitrimers, we combine acid and epoxide fingerprints
X_fp_acid = np.array([morgan_fp(s) for s in md['acid']])
X_fp_epoxide = np.array([morgan_fp(s) for s in md['epoxide']])
X_fp = X_fp_acid + X_fp_epoxide  # simple combination
X_md = md['tg'].values.reshape(-1, 1)
X = np.hstack([X_fp, X_md])

# Standardize and predict
X_s = scaler_X.transform(X)
md_pred_s, md_std_s = gp.predict(X_s, return_std=True)
md_pred = scaler_y.inverse_transform(md_pred_s.reshape(-1, 1)).flatten()
md_std = md_std_s * scaler_y.scale_[0]

md['tg_calibrated'] = md_pred
md['tg_calibrated_std'] = md_std
md.to_csv('outputs/vitrimer_calibrated.csv', index=False)

print(f"Applied GP calibration to {len(md)} vitrimer systems.")
print(f"Calibrated Tg range: {md_pred.min():.2f} - {md_pred.max():.2f} K")
print(f"Mean calibrated Tg: {md_pred.mean():.2f} K ± {md_pred.std():.2f} K")
