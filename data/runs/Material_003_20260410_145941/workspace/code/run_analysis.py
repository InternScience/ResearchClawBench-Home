import os
import json
import math
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem import AllChem, DataStructs

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

BASE = Path('.')
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')

def canon_smiles(s):
    try:
        m = Chem.MolFromSmiles(str(s).replace('*',''))
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None

def mol_from_polymer_smiles(s):
    try:
        s = str(s).replace('*','')
        m = Chem.MolFromSmiles(s)
        return m
    except Exception:
        return None

def mol_features(smiles):
    m = mol_from_polymer_smiles(smiles)
    if m is None:
        return None
    feats = {
        'mw': Descriptors.MolWt(m),
        'logp': Crippen.MolLogP(m),
        'tpsa': rdMolDescriptors.CalcTPSA(m),
        'hba': Lipinski.NumHAcceptors(m),
        'hbd': Lipinski.NumHDonors(m),
        'rings': rdMolDescriptors.CalcNumRings(m),
        'arom_rings': rdMolDescriptors.CalcNumAromaticRings(m),
        'hetero': rdMolDescriptors.CalcNumHeteroatoms(m),
        'rot_bonds': Lipinski.NumRotatableBonds(m),
        'frac_csp3': rdMolDescriptors.CalcFractionCSP3(m),
        'heavy_atoms': m.GetNumHeavyAtoms(),
        'valence_e': Descriptors.NumValenceElectrons(m),
        'mr': Crippen.MolMR(m),
    }
    atom_counts = {}
    for atom in m.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1
    for sym in ['C','N','O','S','P','F','Cl','Br','I']:
        feats[f'count_{sym}'] = atom_counts.get(sym, 0)
    total = sum(atom_counts.values()) or 1
    for sym in ['C','N','O','S','P','F','Cl','Br','I']:
        feats[f'frac_{sym}'] = atom_counts.get(sym, 0) / total
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=256)
    arr = np.zeros((256,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    for i, v in enumerate(arr):
        feats[f'fp_{i}'] = int(v)
    return feats

def acid_epoxy_features(acid, epoxide):
    fa = mol_features(acid)
    fe = mol_features(epoxide)
    if fa is None or fe is None:
        return None
    out = {}
    base_keys = [k for k in fa.keys() if not k.startswith('fp_')]
    for k in base_keys:
        out[f'acid_{k}'] = fa[k]
        out[f'epoxy_{k}'] = fe[k]
        out[f'sum_{k}'] = fa[k] + fe[k] if isinstance(fa[k], (int,float,np.floating)) else np.nan
        out[f'diff_{k}'] = fa[k] - fe[k] if isinstance(fa[k], (int,float,np.floating)) else np.nan
    for i in range(256):
        out[f'fp_sum_{i}'] = fa[f'fp_{i}'] + fe[f'fp_{i}']
        out[f'fp_and_{i}'] = int(fa[f'fp_{i}'] and fe[f'fp_{i}'])
    out['acid_smiles'] = acid
    out['epoxide_smiles'] = epoxide
    return out

cal = pd.read_csv('data/tg_calibration.csv')
vit = pd.read_csv('data/tg_vitrimer_MD.csv')

# Calibration analysis
X_basic = cal[['tg_md']].values
lin = LinearRegression().fit(X_basic, cal['tg_exp'])
cal['tg_linear'] = lin.predict(X_basic)

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=40.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=25.0, noise_level_bounds=(1e-5, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=RANDOM_STATE, n_restarts_optimizer=5)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# manual CV for GP because alpha is fold-specific
preds = np.zeros(len(cal))
for tr, te in cv.split(X_basic):
    gp_cv = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=RANDOM_STATE, n_restarts_optimizer=2)
    gp_cv.fit(X_basic[tr], cal['tg_exp'].values[tr])
    preds[te] = gp_cv.predict(X_basic[te])
cal['tg_gp_cv'] = preds
gp.fit(X_basic, cal['tg_exp'])
mu, std = gp.predict(X_basic, return_std=True)
cal['tg_gp_fit'] = mu
cal['tg_gp_fit_std'] = std

metrics = {
    'linear_mae': float(mean_absolute_error(cal['tg_exp'], cal['tg_linear'])),
    'linear_rmse': float(mean_squared_error(cal['tg_exp'], cal['tg_linear'])**0.5),
    'linear_r2': float(r2_score(cal['tg_exp'], cal['tg_linear'])),
    'gp_cv_mae': float(mean_absolute_error(cal['tg_exp'], cal['tg_gp_cv'])),
    'gp_cv_rmse': float(mean_squared_error(cal['tg_exp'], cal['tg_gp_cv'])**0.5),
    'gp_cv_r2': float(r2_score(cal['tg_exp'], cal['tg_gp_cv'])),
    'bias_md_minus_exp_mean': float((cal['tg_md'] - cal['tg_exp']).mean()),
}

# Apply calibration to vitrimer MD set
vit['tg_calibrated'], vit['tg_calibrated_std'] = gp.predict(vit[['tg']].values, return_std=True)
vit['acq_score'] = -np.abs(vit['tg_calibrated'] - 420.0) + 0.15*vit['tg_calibrated_std']

# Generative surrogate using fragment latent space from acids/epoxides
acid_smiles = sorted(vit['acid'].dropna().unique())
epoxy_smiles = sorted(vit['epoxide'].dropna().unique())

acid_feat_rows = []
for s in acid_smiles:
    f = mol_features(s)
    if f is not None:
        f['smiles'] = s
        acid_feat_rows.append(f)
epoxy_feat_rows = []
for s in epoxy_smiles:
    f = mol_features(s)
    if f is not None:
        f['smiles'] = s
        epoxy_feat_rows.append(f)
acid_df = pd.DataFrame(acid_feat_rows).set_index('smiles')
epoxy_df = pd.DataFrame(epoxy_feat_rows).set_index('smiles')

acid_num = acid_df.select_dtypes(include=[np.number]).fillna(0)
epoxy_num = epoxy_df.select_dtypes(include=[np.number]).fillna(0)
acid_scaler = StandardScaler().fit(acid_num)
epoxy_scaler = StandardScaler().fit(epoxy_num)
acid_lat = PCA(n_components=min(8, acid_num.shape[1], acid_num.shape[0])).fit_transform(acid_scaler.transform(acid_num))
epoxy_lat = PCA(n_components=min(8, epoxy_num.shape[1], epoxy_num.shape[0])).fit_transform(epoxy_scaler.transform(epoxy_num))
acid_lat_df = pd.DataFrame(acid_lat, index=acid_num.index, columns=[f'a_z{i+1}' for i in range(acid_lat.shape[1])])
epoxy_lat_df = pd.DataFrame(epoxy_lat, index=epoxy_num.index, columns=[f'e_z{i+1}' for i in range(epoxy_lat.shape[1])])

vit_lat = vit.join(acid_lat_df, on='acid').join(epoxy_lat_df, on='epoxide')
latent_cols = [c for c in vit_lat.columns if c.startswith('a_z') or c.startswith('e_z')]
vit_lat['latent_norm'] = np.sqrt((vit_lat[latent_cols]**2).sum(axis=1))

# Diversity-aware candidate selection for multiple targets
all_candidates = []
for target in [350, 400, 450, 500]:
    tmp = vit.copy()
    tmp['target'] = target
    tmp['score'] = -np.abs(tmp['tg_calibrated'] - target) + 0.10*tmp['tg_calibrated_std']
    sel = tmp.sort_values('score', ascending=False).head(15).copy()
    sel['design_rationale'] = np.where(sel['tg_calibrated_std']>np.quantile(tmp['tg_calibrated_std'],0.75), 'exploratory-high-uncertainty', 'exploitative-near-target')
    all_candidates.append(sel)
selected = pd.concat(all_candidates, ignore_index=True)
selected = selected.drop_duplicates(subset=['acid','epoxide']).reset_index(drop=True)
selected['candidate_id'] = ['C%03d' % (i+1) for i in range(len(selected))]

# Experimental prioritization heuristic
selected['priority_score'] = (
    -np.abs(selected['tg_calibrated'] - selected['target'])/10
    + 0.05*selected['tg_calibrated_std']
    - 0.02*selected['std']
)
selected = selected.sort_values(['target','priority_score'], ascending=[True, False])

# Save outputs
cal.to_csv(OUT/'calibration_predictions.csv', index=False)
vit.to_csv(OUT/'vitrimer_calibrated_predictions.csv', index=False)
selected.to_csv(OUT/'inverse_design_candidates.csv', index=False)
acid_lat_df.reset_index().to_csv(OUT/'acid_latent_space.csv', index=False)
epoxy_lat_df.reset_index().to_csv(OUT/'epoxide_latent_space.csv', index=False)
with open(OUT/'metrics.json','w') as f:
    json.dump(metrics, f, indent=2)

# Figures
plt.figure(figsize=(7,6))
plt.scatter(cal['tg_md'], cal['tg_exp'], alpha=0.7, label='Observed', s=45)
xs = np.linspace(cal['tg_md'].min(), cal['tg_md'].max(), 200)
plt.plot(xs, lin.predict(xs.reshape(-1,1)), color='tab:orange', lw=2, label='Linear fit')
mu_x, std_x = gp.predict(xs.reshape(-1,1), return_std=True)
plt.plot(xs, mu_x, color='tab:red', lw=2, label='GP mean')
plt.fill_between(xs, mu_x-1.96*std_x, mu_x+1.96*std_x, color='tab:red', alpha=0.15, label='95% CI')
plt.xlabel('MD-simulated Tg (K)')
plt.ylabel('Experimental Tg (K)')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(IMG/'figure1_calibration_curve.png', dpi=220)
plt.close()

plt.figure(figsize=(7,6))
plt.scatter(cal['tg_exp'], cal['tg_linear'], alpha=0.6, label='Linear')
plt.scatter(cal['tg_exp'], cal['tg_gp_cv'], alpha=0.6, label='GP CV')
lims = [min(cal['tg_exp'].min(), cal['tg_gp_cv'].min(), cal['tg_linear'].min()), max(cal['tg_exp'].max(), cal['tg_gp_cv'].max(), cal['tg_linear'].max())]
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel('Observed experimental Tg (K)')
plt.ylabel('Predicted Tg (K)')
plt.legend()
plt.tight_layout()
plt.savefig(IMG/'figure2_calibration_parity.png', dpi=220)
plt.close()

plt.figure(figsize=(8,6))
sns.histplot(cal['tg_exp'], color='tab:blue', label='Experimental calibration set', kde=True, stat='density', alpha=0.4)
sns.histplot(vit['tg'], color='tab:gray', label='Raw MD vitrimer set', kde=True, stat='density', alpha=0.3)
sns.histplot(vit['tg_calibrated'], color='tab:green', label='Calibrated vitrimer predictions', kde=True, stat='density', alpha=0.35)
plt.xlabel('Tg (K)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig(IMG/'figure3_tg_distributions.png', dpi=220)
plt.close()

plt.figure(figsize=(8,6))
sc = plt.scatter(vit_lat['a_z1'], vit_lat['e_z1'], c=vit_lat['tg_calibrated'], cmap='viridis', s=18, alpha=0.7)
plt.colorbar(sc, label='Calibrated Tg (K)')
plt.xlabel('Acid latent coordinate z1')
plt.ylabel('Epoxide latent coordinate z1')
plt.tight_layout()
plt.savefig(IMG/'figure4_latent_map.png', dpi=220)
plt.close()

plt.figure(figsize=(9,6))
plot_df = selected.sort_values(['target','tg_calibrated'])
sns.scatterplot(data=plot_df, x='target', y='tg_calibrated', hue='design_rationale', size='tg_calibrated_std', sizes=(60,220), alpha=0.85)
for _, r in plot_df.iterrows():
    plt.plot([r['target'], r['target']], [r['tg_calibrated']-r['tg_calibrated_std'], r['tg_calibrated']+r['tg_calibrated_std']], color='gray', alpha=0.35)
plt.xlabel('Target Tg (K)')
plt.ylabel('Predicted calibrated Tg (K)')
plt.tight_layout()
plt.savefig(IMG/'figure5_candidate_targets.png', dpi=220)
plt.close()

summary = pd.DataFrame({
    'dataset':['calibration','vitrimer_md'],
    'n_rows':[len(cal), len(vit)],
    'mean_tg':[cal['tg_exp'].mean(), vit['tg_calibrated'].mean()],
    'std_tg':[cal['tg_exp'].std(), vit['tg_calibrated'].std()]
})
summary.to_csv(OUT/'dataset_summary.csv', index=False)

print('done')
print(json.dumps(metrics, indent=2))
print(selected[['candidate_id','target','acid','epoxide','tg_calibrated','tg_calibrated_std','priority_score']].head(12).to_string(index=False))
