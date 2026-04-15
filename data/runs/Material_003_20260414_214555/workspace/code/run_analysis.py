import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

ROOT = Path('.')
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='talk')
np.random.seed(42)

cal = pd.read_csv(DATA / 'tg_calibration.csv')
vit = pd.read_csv(DATA / 'tg_vitrimer_MD.csv')

# dataset overview
summary = {
    'calibration_rows': int(len(cal)),
    'vitrimer_rows': int(len(vit)),
    'calibration_columns': list(cal.columns),
    'vitrimer_columns': list(vit.columns),
    'calibration_tg_exp_mean': float(cal.tg_exp.mean()),
    'calibration_tg_md_mean': float(cal.tg_md.mean()),
    'vitrimer_tg_md_mean': float(vit.tg.mean()),
    'unique_acids': int(vit['acid'].nunique()),
    'unique_epoxides': int(vit['epoxide'].nunique())
}
(Path(OUT/'dataset_overview.json')).write_text(json.dumps(summary, indent=2))

# GP calibration
X = cal[['tg_md']].values
y = cal['tg_exp'].values
noise_all = (cal['std'].values / np.median(cal['std'].values))**2
noise_all = np.clip(noise_all * 10.0, 1e-6, 1e4)
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=50.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=10.0, noise_level_bounds=(1e-5, 1e3))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
pred = np.zeros_like(y, dtype=float)
pstd = np.zeros_like(y, dtype=float)
for train_idx, test_idx in kf.split(X):
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=noise_all[train_idx], random_state=42, n_restarts_optimizer=3)
    model.fit(X[train_idx], y[train_idx])
    p, s = model.predict(X[test_idx], return_std=True)
    pred[test_idx] = p
    pstd[test_idx] = s

metrics = {
    'mae': float(mean_absolute_error(y, pred)),
    'rmse': float(mean_squared_error(y, pred) ** 0.5),
    'r2': float(r2_score(y, pred)),
    'pearson_r': float(np.corrcoef(y, pred)[0, 1]),
    'coverage_within_2sigma': float(np.mean(np.abs(y - pred) <= 2 * pstd))
}
full_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=noise_all, random_state=42, n_restarts_optimizer=5)
full_model.fit(X, y)
fit_pred, fit_std = full_model.predict(X, return_std=True)
cal_out = cal.copy()
cal_out['pred_cv'] = pred
cal_out['pred_cv_std'] = pstd
cal_out['residual_cv'] = cal_out['tg_exp'] - cal_out['pred_cv']
cal_out.to_csv(OUT/'calibration_cv_predictions.csv', index=False)

vit_pred, vit_std = full_model.predict(vit[['tg']].values, return_std=True)
vit_out = vit.copy()
vit_out['tg_calibrated'] = vit_pred
vit_out['tg_calibrated_std'] = vit_std
vit_out['target_distance_430'] = np.abs(vit_out['tg_calibrated'] - 430.0)
vit_out['composite_score'] = -vit_out['target_distance_430'] - 0.25 * vit_out['tg_calibrated_std']
vit_out.sort_values(['composite_score', 'tg_calibrated'], ascending=[False, False], inplace=True)
vit_out.to_csv(OUT/'vitrimer_calibrated_predictions.csv', index=False)
vit_out.head(100).to_csv(OUT/'top100_candidates.csv', index=False)
(Path(OUT/'gp_calibration_metrics.json')).write_text(json.dumps({'metrics_cv5': metrics, 'fitted_kernel': str(full_model.kernel_)}, indent=2))

# latent-space surrogate using fingerprints+PCA+GMM nearest-neighbor decode
acids = sorted(set(vit['acid']))
epoxides = sorted(set(vit['epoxide']))
def fp_from_smiles(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
def arr(fp):
    a = np.zeros((256,), dtype=float)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a
acid_fps = {s: fp_from_smiles(s) for s in acids}
ep_fps = {s: fp_from_smiles(s) for s in epoxides}
valid_acids = [s for s,v in acid_fps.items() if v is not None]
valid_eps = [s for s,v in ep_fps.items() if v is not None]
XA = np.vstack([arr(acid_fps[s]) for s in valid_acids])
XE = np.vstack([arr(ep_fps[s]) for s in valid_eps])
pa = PCA(n_components=8, random_state=42).fit(XA)
pe = PCA(n_components=8, random_state=42).fit(XE)
Za = pa.transform(XA)
Ze = pe.transform(XE)
pd.DataFrame({'acid': valid_acids, **{f'pc{i+1}': Za[:,i] for i in range(Za.shape[1])}}).to_csv(OUT/'acid_latent_pca.csv', index=False)
pd.DataFrame({'epoxide': valid_eps, **{f'pc{i+1}': Ze[:,i] for i in range(Ze.shape[1])}}).to_csv(OUT/'epoxide_latent_pca.csv', index=False)

na = NearestNeighbors(n_neighbors=1).fit(Za)
ne = NearestNeighbors(n_neighbors=1).fit(Ze)
gma = GaussianMixture(n_components=min(6, len(valid_acids)), covariance_type='full', random_state=42).fit(Za)
gme = GaussianMixture(n_components=min(6, len(valid_eps)), covariance_type='full', random_state=42).fit(Ze)
new_a = [valid_acids[na.kneighbors([z], return_distance=False)[0,0]] for z in gma.sample(120)[0]]
new_e = [valid_eps[ne.kneighbors([z], return_distance=False)[0,0]] for z in gme.sample(120)[0]]
existing = set(zip(vit['acid'], vit['epoxide']))
seed_pairs = list(dict.fromkeys(zip(vit_out.head(50)['acid'], vit_out.head(50)['epoxide'])))
cands = []
for a in list(dict.fromkeys(new_a))[:25]:
    for e in list(dict.fromkeys(new_e))[:25]:
        if (a,e) not in existing:
            cands.append((a,e))
for a,_ in seed_pairs:
    for e in list(dict.fromkeys(new_e))[:15]:
        if (a,e) not in existing:
            cands.append((a,e))
for _,e in seed_pairs:
    for a in list(dict.fromkeys(new_a))[:15]:
        if (a,e) not in existing:
            cands.append((a,e))
uniq = []
seen = set()
for pair in cands:
    if pair not in seen:
        uniq.append(pair)
        seen.add(pair)
gen = pd.DataFrame(uniq, columns=['acid','epoxide'])
# score generated pairs by additive building-block effects learned from known systems
acid_effect = vit_out.groupby('acid')['tg_calibrated'].mean().rename('acid_mean')
ep_effect = vit_out.groupby('epoxide')['tg_calibrated'].mean().rename('epoxide_mean')
overall = vit_out['tg_calibrated'].mean()
gen = gen.join(acid_effect, on='acid').join(ep_effect, on='epoxide')
gen['tg_calibrated_est'] = gen['acid_mean'] + gen['epoxide_mean'] - overall
gen['uncertainty_proxy'] = vit_out['tg_calibrated_std'].mean()
gen['target_distance_430'] = np.abs(gen['tg_calibrated_est'] - 430.0)
gen['novelty_score'] = gen['target_distance_430'] + 0.25 * gen['uncertainty_proxy']
gen.sort_values('novelty_score', inplace=True)
gen.to_csv(OUT/'generated_recombined_pairs_scored.csv', index=False)
gen.head(20).to_csv(OUT/'top20_generated_candidates.csv', index=False)

# figures
plt.figure(figsize=(7,6))
plt.scatter(cal_out['tg_md'], cal_out['tg_exp'], s=35, alpha=0.7, label='Observed')
ord_idx = np.argsort(cal_out['tg_md'].values)
plt.plot(cal_out['tg_md'].values[ord_idx], cal_out['pred_cv'].values[ord_idx], color='crimson', linewidth=2.5, label='GP CV prediction')
lims = [min(cal_out['tg_md'].min(), cal_out['tg_exp'].min()), max(cal_out['tg_md'].max(), cal_out['tg_exp'].max())]
plt.plot(lims, lims, '--', color='gray', linewidth=1, label='Identity')
plt.xlabel('MD Tg (K)')
plt.ylabel('Experimental Tg or GP-predicted Tg (K)')
plt.title('Gaussian-process calibration of MD to experimental Tg')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(IMG/'calibration_scatter.png', dpi=220)
plt.close()

fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].scatter(cal_out['pred_cv'], cal_out['residual_cv'], s=28, alpha=0.7)
ax[0].axhline(0, color='black', linestyle='--', linewidth=1)
ax[0].set_xlabel('Cross-validated predicted Tg (K)')
ax[0].set_ylabel('Residual (exp - pred) (K)')
ax[0].set_title('Residual diagnostic')
ax[1].hist(cal_out['residual_cv'], bins=24, color='steelblue', alpha=0.85)
ax[1].set_xlabel('Residual (K)')
ax[1].set_ylabel('Count')
ax[1].set_title('Residual distribution')
plt.tight_layout()
plt.savefig(IMG/'residual_diagnostics.png', dpi=220)
plt.close()

plot_df = vit_out.head(500).copy()
plot_df['rank'] = np.arange(1, len(plot_df)+1)
plt.figure(figsize=(8,6))
sc = plt.scatter(plot_df['rank'], plot_df['tg_calibrated'], c=plot_df['tg_calibrated_std'], cmap='viridis', s=42)
plt.axhline(430, color='crimson', linestyle='--', linewidth=1.5, label='Target Tg = 430 K')
plt.xlabel('Candidate rank among top 500 screened systems')
plt.ylabel('Calibrated Tg (K)')
plt.title('Top screened vitrimer candidates after GP calibration')
cb = plt.colorbar(sc)
cb.set_label('Predictive std (K)')
plt.legend()
plt.tight_layout()
plt.savefig(IMG/'candidate_ranking.png', dpi=220)
plt.close()

acid_plot = pd.read_csv(OUT/'acid_latent_pca.csv').head(1500).copy()
acid_scores = vit_out.groupby('acid')['tg_calibrated'].mean().reset_index().rename(columns={'tg_calibrated':'mean_calibrated_tg'})
acid_plot = acid_plot.merge(acid_scores, on='acid', how='left')
plt.figure(figsize=(8,6))
sc = plt.scatter(acid_plot['pc1'], acid_plot['pc2'], c=acid_plot['mean_calibrated_tg'], cmap='coolwarm', s=18, alpha=0.75)
plt.xlabel('Acid latent PC1')
plt.ylabel('Acid latent PC2')
plt.title('Surrogate latent space of acid components')
cb = plt.colorbar(sc)
cb.set_label('Mean calibrated Tg contribution (K)')
plt.tight_layout()
plt.savefig(IMG/'acid_latent_space.png', dpi=220)
plt.close()

# claim recovery
claim_recovery = pd.DataFrame([
    ['Calibration data support systematic MD-to-experiment correction', 'outputs/calibration_cv_predictions.csv; outputs/gp_calibration_metrics.json; report/images/calibration_scatter.png'],
    ['GP calibration attains moderate predictive fidelity', 'outputs/gp_calibration_metrics.json'],
    ['Screening reprioritizes vitrimer candidates around target Tg', 'outputs/vitrimer_calibrated_predictions.csv; outputs/top100_candidates.csv; report/images/candidate_ranking.png'],
    ['Latent-space-inspired recombination can suggest novel acid/epoxide pairings', 'outputs/generated_recombined_pairs.csv; outputs/generated_recombined_pairs_scored.csv; report/images/acid_latent_space.png'],
    ['Experimental validation is not directly executed here', 'outputs/method_contract.json; outputs/dependency_check.json']
], columns=['claim','supporting_artifacts'])
claim_recovery.to_csv(OUT/'claim_recovery_table.csv', index=False)

# update artifact inventory
artifact_inventory = {
    'artifacts': [
        {'name': 'dataset_overview_table', 'status': 'satisfied', 'path': 'outputs/dataset_overview.json'},
        {'name': 'gp_calibration_metrics', 'status': 'satisfied', 'path': 'outputs/gp_calibration_metrics.json'},
        {'name': 'calibration_scatter_plot', 'status': 'satisfied', 'path': 'report/images/calibration_scatter.png'},
        {'name': 'residual_diagnostics_plot', 'status': 'satisfied', 'path': 'report/images/residual_diagnostics.png'},
        {'name': 'candidate_ranking_table', 'status': 'satisfied', 'path': 'outputs/top100_candidates.csv'},
        {'name': 'latent_or_property_space_plot', 'status': 'satisfied', 'path': 'report/images/acid_latent_space.png'},
        {'name': 'generated_candidate_table', 'status': 'satisfied', 'path': 'outputs/top20_generated_candidates.csv'},
        {'name': 'report', 'status': 'planned', 'path': 'report/report.md'}
    ]
}
(Path(OUT/'target_artifact_inventory.json')).write_text(json.dumps(artifact_inventory, indent=2))
print(json.dumps({'metrics': metrics, 'top_candidate_count': int(len(vit_out.head(100))), 'generated_candidate_count': int(len(gen))}, indent=2))
