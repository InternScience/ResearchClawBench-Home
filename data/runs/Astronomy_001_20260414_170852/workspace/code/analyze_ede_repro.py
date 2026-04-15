import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'DESI_EDE_Repro_Data.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')


def load_namespace(path: Path):
    ns = {}
    text = path.read_text()
    exec(compile(text, str(path), 'exec'), {}, ns)
    return ns


def dict_to_df(model_name, d):
    rows = []
    for p, (mean, sigma) in d.items():
        rows.append({'model': model_name, 'parameter': p, 'mean': float(mean), 'sigma': float(sigma)})
    return pd.DataFrame(rows)


def points_to_df(kind, points, yname):
    return pd.DataFrame(points, columns=['z', yname, 'error']).assign(dataset=kind)


ns = load_namespace(DATA_PATH)

params = pd.concat([
    dict_to_df('LambdaCDM', ns['lcdm_params']),
    dict_to_df('EDE', ns['ede_params']),
    dict_to_df('w0wa', ns['w0wa_params']),
], ignore_index=True)
params.to_csv(OUT / 'parameter_constraints.csv', index=False)

shared = sorted(set(params.loc[params['model']=='LambdaCDM','parameter']) & set(params.loc[params['model']=='EDE','parameter']) & set(params.loc[params['model']=='w0wa','parameter']))
shared_df = params[params['parameter'].isin(shared)].copy()
shared_pivot = shared_df.pivot(index='parameter', columns='model', values='mean')
shared_sigma = shared_df.pivot(index='parameter', columns='model', values='sigma')

# Model comparison summary via standardized shifts relative to LambdaCDM for shared parameters.
rows = []
for model in ['EDE', 'w0wa']:
    sub = shared_df[shared_df['model']==model].set_index('parameter')
    ref = shared_df[shared_df['model']=='LambdaCDM'].set_index('parameter')
    for p in shared:
        delta = sub.loc[p, 'mean'] - ref.loc[p, 'mean']
        sigma_comb = float(np.hypot(sub.loc[p, 'sigma'], ref.loc[p, 'sigma']))
        z = delta / sigma_comb if sigma_comb > 0 else np.nan
        rows.append({'comparison_to': 'LambdaCDM', 'model': model, 'parameter': p, 'delta_mean': delta, 'combined_sigma': sigma_comb, 'z_shift': z})
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / 'model_comparison.csv', index=False)

# EDE-only posterior-style summary
ede_only = params[params['model']=='EDE'].copy()
ede_only[ede_only['parameter'].isin(['f_EDE', 'log10_ac'])].to_csv(OUT / 'ede_parameter_summary.csv', index=False)

# Distance data
points = pd.concat([
    points_to_df('DESI_DV_over_rd', ns['desi_dvrd_points'], 'value'),
    points_to_df('DESI_F_AP', ns['desi_fap_points'], 'value'),
    points_to_df('Union3_SN', ns['sne_mu_points'], 'value'),
], ignore_index=True)
points.to_csv(OUT / 'distance_points.csv', index=False)

# Data summary JSON
summary = {
    'n_parameter_rows': int(len(params)),
    'models': sorted(params['model'].unique().tolist()),
    'shared_parameters_all_models': shared,
    'ede_specific_parameters': sorted(set(ede_only['parameter']) - set(shared)),
    'w0wa_specific_parameters': sorted(set(params[params['model']=='w0wa']['parameter']) - set(shared)),
    'distance_datasets': points['dataset'].value_counts().sort_index().to_dict(),
}
(OUT / 'data_summary.json').write_text(json.dumps(summary, indent=2))

# Figure 1: parameter constraints for key parameters
plot_params = ['omega_m', 'H0', 'sigma8', 'ns', 'ombh2', 'tau']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
palette = {'LambdaCDM': '#4C72B0', 'EDE': '#DD8452', 'w0wa': '#55A868'}
for ax, p in zip(axes, plot_params):
    sub = params[params['parameter']==p].copy()
    sub = sub.set_index('model').loc[['LambdaCDM', 'EDE', 'w0wa']].reset_index()
    y = np.arange(len(sub))
    ax.errorbar(sub['mean'], y, xerr=sub['sigma'], fmt='o', color='black', ecolor='black', capsize=4)
    for yi, (_, row) in enumerate(sub.iterrows()):
        ax.scatter(row['mean'], yi, s=80, color=palette[row['model']], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['model'])
    ax.set_title(p)
    ax.invert_yaxis()
plt.tight_layout()
fig.savefig(IMG / 'parameter_constraints.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Figure 2: EDE-specific parameter constraints
fig, ax = plt.subplots(figsize=(8, 4.8))
sub = params[(params['model']=='EDE') & (params['parameter'].isin(['f_EDE', 'log10_ac']))].copy()
y = np.arange(len(sub))
ax.errorbar(sub['mean'], y, xerr=sub['sigma'], fmt='o', color='black', capsize=5)
for yi, (_, row) in enumerate(sub.iterrows()):
    ax.scatter(row['mean'], yi, s=90, color='#C44E52', zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(sub['parameter'])
ax.set_title('EDE-specific reproduced constraints')
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(IMG / 'ede_parameters.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Figure 3: distance comparison panels
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=False)
for ax, ds, title, ylabel in zip(
    axes,
    ['DESI_DV_over_rd', 'DESI_F_AP', 'Union3_SN'],
    ['DESI Δ(D_V/r_d)', 'DESI ΔF_AP', 'Union3 Δμ'],
    ['relative shift', 'relative shift', 'mag']
):
    sub = points[points['dataset']==ds]
    ax.errorbar(sub['z'], sub['value'], yerr=sub['error'], fmt='o', color='#4C72B0', capsize=4)
    ax.axhline(0, color='gray', lw=1, ls='--')
    ax.set_title(title)
    ax.set_xlabel('redshift z')
    ax.set_ylabel(ylabel)
plt.tight_layout()
fig.savefig(IMG / 'distance_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Figure 4: standardized shifts heatmap
heat = comparison.pivot(index='parameter', columns='model', values='z_shift').loc[shared]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(heat, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'shift / combined sigma'})
ax.set_title('Parameter shifts relative to LambdaCDM')
plt.tight_layout()
fig.savefig(IMG / 'parameter_shift_heatmap.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Direct answer table for key parameters
key = params[params['parameter'].isin(['omega_m', 'H0', 'sigma8', 'f_EDE', 'log10_ac', 'w0', 'wa'])].copy()
key.to_csv(OUT / 'direct_answer_table.csv', index=False)

# Claim recovery scaffold
claims = [
    {
        'claim': 'EDE shifts H0 upward relative to LambdaCDM in the reproduced summary constraints.',
        'supported_by': ['outputs/parameter_constraints.csv', 'outputs/model_comparison.csv', 'report/images/parameter_constraints.png'],
        'status': 'supported_directly_from_workspace_data'
    },
    {
        'claim': 'w0wa shifts H0 downward and Omega_m upward relative to LambdaCDM in the reproduced summary constraints.',
        'supported_by': ['outputs/parameter_constraints.csv', 'outputs/model_comparison.csv', 'report/images/parameter_constraints.png', 'report/images/parameter_shift_heatmap.png'],
        'status': 'supported_directly_from_workspace_data'
    },
    {
        'claim': 'The reproduced EDE summary prefers nonzero f_EDE with log10_ac near -3.56.',
        'supported_by': ['outputs/ede_parameter_summary.csv', 'report/images/ede_parameters.png'],
        'status': 'supported_directly_from_workspace_data'
    },
    {
        'claim': 'Delta chi^2 comparison among LambdaCDM, EDE, and w0wa can be established from the current workspace.',
        'supported_by': [],
        'status': 'unsupported_missing_numeric_values'
    }
]
(OUT / 'claim_recovery_table.json').write_text(json.dumps(claims, indent=2))

print('WROTE', OUT / 'parameter_constraints.csv')
print('WROTE', OUT / 'model_comparison.csv')
print('WROTE', OUT / 'ede_parameter_summary.csv')
print('WROTE', OUT / 'distance_points.csv')
print('WROTE', OUT / 'data_summary.json')
print('WROTE', OUT / 'direct_answer_table.csv')
print('WROTE', OUT / 'claim_recovery_table.json')
print('WROTE', IMG / 'parameter_constraints.png')
print('WROTE', IMG / 'ede_parameters.png')
print('WROTE', IMG / 'distance_comparison.png')
print('WROTE', IMG / 'parameter_shift_heatmap.png')
