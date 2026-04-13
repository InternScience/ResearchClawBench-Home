
import os, glob, yaml, h5py, json, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

ROOT = Path('.')
DATA = ROOT/'data'/'flow'/'0000'
OUT = ROOT/'outputs'
IMG = ROOT/'report'/'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')

# 1) collect model metadata and validation losses
rows = []
for meta in sorted(DATA.glob('[0-9][0-9][0-9]/_meta.yaml')):
    model_id = meta.parent.name
    with open(meta) as f:
        d = yaml.safe_load(f)
    cfg = d['config']
    loss_path = meta.parent/'validation_loss.h5'
    with h5py.File(loss_path, 'r') as f:
        val_loss = float(f['data'][()])
    rows.append({
        'model_id': model_id,
        'validation_loss': val_loss,
        'n_iters': cfg['task'].get('n_iters'),
        'batch_size': cfg['task'].get('batch_size'),
        'fold': cfg['task'].get('fold'),
        'seed': cfg['task'].get('seed'),
        'dt': cfg['task']['dataset'].get('dt'),
        'n_frames': cfg['task']['dataset'].get('n_frames'),
        'extent': cfg['network']['connectome'].get('extent'),
        'connectome_file': cfg['network']['connectome'].get('file'),
        'decoder_type': cfg['task']['decoder']['flow'].get('type'),
        'decoder_kernel_size': cfg['task']['decoder']['flow'].get('kernel_size'),
        'time_const_init': cfg['network']['node_config']['time_const'].get('value'),
        'bias_mean_init': cfg['network']['node_config']['bias'].get('mean'),
        'bias_std_init': cfg['network']['node_config']['bias'].get('std'),
        'syn_strength_scale': cfg['network']['edge_config']['syn_strength'].get('scale'),
        'n_syn_fill': cfg['network']['connectome'].get('n_syn_fill'),
        'activation': cfg['network']['dynamics']['activation'].get('type')
    })
models = pd.DataFrame(rows).sort_values('validation_loss').reset_index(drop=True)
models.to_csv(OUT/'model_validation_summary.csv', index=False)

# 2) inspect clustering pickles without custom imports
pickle_rows = []
for p in sorted((DATA/'umap_and_clustering').glob('*.pickle')):
    txt = os.popen(f"strings '{p}' | head -160").read()
    n_components = None
    m = re.search(r'n_components\\nscalar\\nNNNJ\\n(.*?)\\n', txt, re.S)
    if m:
        pass
    reduced = 'GaussianMixtureClustering' in txt
    has_scaler = 'MinMaxScaler' in txt
    # cluster labels are not safely recoverable without flyvis, so record toolchain evidence
    pickle_rows.append({
        'cell_type': p.stem,
        'has_gaussian_mixture': reduced,
        'has_minmax_scaler': has_scaler,
        'raw_strings_excerpt': txt[:500].replace('\n',' | ')
    })
clust = pd.DataFrame(pickle_rows)
clust.to_csv(OUT/'umap_pickle_inventory.csv', index=False)

# 3) derive pathway groups from cell-type names
cell_types = sorted([p.stem for p in (DATA/'umap_and_clustering').glob('*.pickle')])

def classify_cell_type(name):
    if name.startswith('R'):
        return 'photoreceptor/retina'
    if name.startswith('L') or name.startswith('Lawf'):
        return 'lamina'
    if name.startswith('Mi') or name.startswith('Tm') or name.startswith('TmY') or name in ['T1','T2','T2a','T3']:
        return 'medulla/intermediate'
    if name.startswith('T4'):
        return 'ON motion output (T4)'
    if name.startswith('T5'):
        return 'OFF motion output (T5)'
    if name.startswith('CT1') or name in ['Am','C2','C3']:
        return 'wide-field/modulatory'
    return 'other'

ct = pd.DataFrame({'cell_type': cell_types})
ct['group'] = ct['cell_type'].map(classify_cell_type)
ct.to_csv(OUT/'cell_type_groups.csv', index=False)

# 4) figures
# fig1 validation distribution
plt.figure(figsize=(10,6))
ax = sns.histplot(models['validation_loss'], bins=12, kde=True, color='#4C78A8')
ax.axvline(models['validation_loss'].mean(), color='crimson', linestyle='--', label=f"mean={models['validation_loss'].mean():.3f}")
ax.axvline(models['validation_loss'].min(), color='darkgreen', linestyle=':', label=f"best={models['validation_loss'].min():.3f}")
ax.set_title('Distribution of validation loss across 50 pretrained DMN models')
ax.set_xlabel('Validation loss')
ax.set_ylabel('Number of models')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(IMG/'validation_loss_distribution.png', dpi=200)
plt.close()

# fig2 ranked models
plt.figure(figsize=(12,6))
ranked = models.sort_values('validation_loss').reset_index(drop=True)
ax = sns.lineplot(data=ranked, x=ranked.index+1, y='validation_loss', marker='o', linewidth=2)
ax.set_title('Ensemble ranking of pretrained connectome-constrained DMNs')
ax.set_xlabel('Model rank (best to worst)')
ax.set_ylabel('Validation loss')
plt.tight_layout()
plt.savefig(IMG/'model_ranking_validation_loss.png', dpi=200)
plt.close()

# fig3 cell-type groups
plt.figure(figsize=(10,6))
order = ct['group'].value_counts().index
ax = sns.countplot(data=ct, y='group', order=order, palette='viridis')
ax.set_title('Cell-type coverage represented in clustering outputs')
ax.set_xlabel('Count of cell types')
ax.set_ylabel('Functional/anatomical group')
for c in ax.containers:
    ax.bar_label(c, fmt='%d', padding=3)
plt.tight_layout()
plt.savefig(IMG/'cell_type_group_counts.png', dpi=200)
plt.close()

# fig4 overview schematic heatmap of canonical motion pathway stages
stage_names = ['Retina','Lamina','Medulla/intermediate','T4 ON outputs','T5 OFF outputs','Wide-field/modulatory']
stage_counts = [
    (ct['group']=='photoreceptor/retina').sum(),
    (ct['group']=='lamina').sum(),
    (ct['group']=='medulla/intermediate').sum(),
    (ct['group']=='ON motion output (T4)').sum(),
    (ct['group']=='OFF motion output (T5)').sum(),
    (ct['group']=='wide-field/modulatory').sum(),
]
mat = np.diag(stage_counts)
plt.figure(figsize=(8,6))
ax = sns.heatmap(mat, annot=True, fmt='d', cmap='mako', cbar=False,
                 xticklabels=stage_names, yticklabels=stage_names)
ax.set_title('Stage-wise inventory of motion-pathway cell types in the released analysis bundle')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(IMG/'pathway_stage_inventory_heatmap.png', dpi=200)
plt.close()

summary = {
    'n_models': int(len(models)),
    'best_model_id': str(models.iloc[0]['model_id']),
    'best_validation_loss': float(models.iloc[0]['validation_loss']),
    'mean_validation_loss': float(models['validation_loss'].mean()),
    'std_validation_loss': float(models['validation_loss'].std()),
    'n_cluster_pickles': int(len(clust)),
    'n_cell_types': int(len(ct)),
    'group_counts': ct['group'].value_counts().to_dict(),
    'config_constants': {
        'n_frames': int(models['n_frames'].iloc[0]),
        'dt': float(models['dt'].iloc[0]),
        'extent': int(models['extent'].iloc[0]),
        'connectome_file': str(models['connectome_file'].iloc[0]),
        'decoder_type': str(models['decoder_type'].iloc[0]),
        'activation': str(models['activation'].iloc[0]),
        'time_const_init': float(models['time_const_init'].iloc[0]),
        'bias_mean_init': float(models['bias_mean_init'].iloc[0]),
        'bias_std_init': float(models['bias_std_init'].iloc[0]),
        'syn_strength_scale': float(models['syn_strength_scale'].iloc[0]),
    }
}
with open(OUT/'analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
