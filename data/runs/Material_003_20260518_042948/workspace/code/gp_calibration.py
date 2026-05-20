import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
import json

def get_fp_desc(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        fp_arr = np.array(list(fp.ToBitString()), dtype=int)
        descs = [Descriptors.MolWt(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
        return np.concatenate([fp_arr, descs])
    return None

# 1. GP Calibration for tg_exp from tg_md + features
df_cal = pd.read_csv('data/tg_calibration.csv')
X_cal = []
y_cal = []
valid_indices = []

for i, row in df_cal.iterrows():
    feat = get_fp_desc(row['smiles'])
    if feat is not None:
        X_cal.append(np.concatenate([feat, [row['tg_md']]]))
        y_cal.append(row['tg_exp'])
        valid_indices.append(i)

X_cal = np.array(X_cal)
y_cal = np.array(y_cal)
df_cal_v = df_cal.iloc[valid_indices]

X_train, X_test, y_train, y_test = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)

kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
gp.fit(X_train, y_train)

with open('outputs/gp_model.pkl', 'wb') as f:
    pickle.dump(gp, f)

y_pred_cal, y_std_cal = gp.predict(X_test, return_std=True)
rmse_cal = np.sqrt(mean_squared_error(y_test, y_pred_cal))
r2_cal = r2_score(y_test, y_pred_cal)

plt.figure(figsize=(6, 6))
plt.errorbar(y_test, y_pred_cal, yerr=y_std_cal, fmt='o', alpha=0.5, label='Test Data')
plt.plot([150, 550], [150, 550], 'k--', label='Ideal')
plt.xlabel('Experimental Tg (K)')
plt.ylabel('GP Calibrated Tg (K)')
plt.title(f'GP Calibration (RMSE={rmse_cal:.1f}K, R2={r2_cal:.2f})')
plt.legend()
plt.grid(True)
plt.savefig('report/images/gp_calibration.png', dpi=100)
plt.close()

# 2. Surrogate model for tg_md of vitrimers
df_vit = pd.read_csv('data/tg_vitrimer_MD.csv')
X_vit = []
y_vit = []

for i, row in df_vit.iterrows():
    f_acid = get_fp_desc(row['acid'])
    f_epox = get_fp_desc(row['epoxide'])
    if f_acid is not None and f_epox is not None:
        X_vit.append(np.concatenate([f_acid, f_epox]))
        y_vit.append(row['tg'])

X_vit = np.array(X_vit)
y_vit = np.array(y_vit)

# Use a smaller subset for RF training if needed, but 8000 is fine with RF
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_vit, y_vit)

with open('outputs/rf_vit_md_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

metrics = {"gp_rmse": float(rmse_cal), "gp_r2": float(r2_cal)}
with open('outputs/gp_metrics.json', 'w') as f:
    json.dump(metrics, f)

# Data overview plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df_cal['tg_exp'], bins=20, alpha=0.7, label='Experimental Tg', color='blue')
plt.hist(df_cal['tg_md'], bins=20, alpha=0.7, label='MD Simulated Tg', color='orange')
plt.xlabel('Tg (K)')
plt.ylabel('Count')
plt.title('Calibration Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df_vit['tg'], bins=20, alpha=0.7, color='green')
plt.xlabel('MD Simulated Tg (K)')
plt.ylabel('Count')
plt.title('Vitrimer MD Data')
plt.tight_layout()
plt.savefig('report/images/data_overview.png', dpi=100)
plt.close()

print("Done.")
