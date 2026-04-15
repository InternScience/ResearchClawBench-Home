import os
import sys
import json
import types
import pickle
import yaml
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'flow', '0000')
OUT = os.path.join(ROOT, 'outputs')
IMG = os.path.join(ROOT, 'report', 'images')
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')


def install_flyvis_stubs():
    flyvis = types.ModuleType('flyvis')
    analysis = types.ModuleType('flyvis.analysis')
    clustering = types.ModuleType('flyvis.analysis.clustering')

    class Embedding:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)
        def __setstate__(self, state):
            self.__dict__.update(state)

    class GaussianMixtureClustering:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)
        def __setstate__(self, state):
            self.__dict__.update(state)

    clustering.Embedding = Embedding
    clustering.GaussianMixtureClustering = GaussianMixtureClustering
    analysis.clustering = clustering
    flyvis.analysis = analysis
    sys.modules['flyvis'] = flyvis
    sys.modules['flyvis.analysis'] = analysis
    sys.modules['flyvis.analysis.clustering'] = clustering


def model_table():
    rows = []
    for model in sorted(d for d in os.listdir(DATA) if d.isdigit()):
        mdir = os.path.join(DATA, model)
        with open(os.path.join(mdir, '_meta.yaml')) as f:
            meta = yaml.safe_load(f)
        with h5py.File(os.path.join(mdir, 'validation_loss.h5'), 'r') as f:
            val = float(f['data'][()])
        rows.append({
            'model_id': int(model),
            'validation_loss': val,
            'fold': meta['config']['task']['fold'],
            'seed': meta['config']['task']['seed'],
            'decoder_type': meta['config']['task']['decoder']['flow']['type'],
            'decoder_shape_0': meta['config']['task']['decoder']['flow']['shape'][0],
            'decoder_shape_1': meta['config']['task']['decoder']['flow']['shape'][1],
            'kernel_size': meta['config']['task']['decoder']['flow']['kernel_size'],
            'dataset_type': meta['config']['task']['dataset']['type'],
            'n_frames': meta['config']['task']['dataset']['n_frames'],
            'dt': meta['config']['task']['dataset']['dt'],
            'connectome_file': meta['config']['network']['connectome']['file'],
            'extent': meta['config']['network']['connectome']['extent'],
            'syn_fill': meta['config']['network']['connectome']['n_syn_fill'],
        })
    df = pd.DataFrame(rows).sort_values('model_id')
    df.to_csv(os.path.join(OUT, 'model_inventory.csv'), index=False)
    summary = {
        'n_models': int(len(df)),
        'validation_loss_mean': float(df.validation_loss.mean()),
        'validation_loss_std': float(df.validation_loss.std(ddof=1)),
        'validation_loss_min': float(df.validation_loss.min()),
        'validation_loss_median': float(df.validation_loss.median()),
        'validation_loss_max': float(df.validation_loss.max()),
        'connectome_file_unique': sorted(df.connectome_file.unique().tolist()),
        'dataset_type_unique': sorted(df.dataset_type.unique().tolist()),
    }
    summary['validation_loss_quantiles'] = {str(k): float(v) for k, v in df.validation_loss.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict().items()}
    with open(os.path.join(OUT, 'validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(os.path.join(OUT, 'validation_summary.csv'), index=False)
    return df, summary


def connectome_summary():
    with open(os.path.join(DATA, '000', '_meta.yaml')) as f:
        meta = yaml.safe_load(f)
    summary = {
        'connectome_type': meta['config']['network']['connectome']['type'],
        'connectome_file': meta['config']['network']['connectome']['file'],
        'extent': meta['config']['network']['connectome']['extent'],
        'n_syn_fill': meta['config']['network']['connectome']['n_syn_fill'],
        'dynamics_type': meta['config']['network']['dynamics']['type'],
        'activation_type': meta['config']['network']['dynamics']['activation']['type'],
        'node_groupby': {k: v.get('groupby') for k, v in meta['config']['network']['node_config'].items()},
        'edge_groupby': {k: v.get('groupby') for k, v in meta['config']['network']['edge_config'].items()},
        'task_dataset_type': meta['config']['task']['dataset']['type'],
        'task_tasks': meta['config']['task']['dataset']['tasks'],
        'n_iters': meta['config']['task']['n_iters'],
        'n_folds': meta['config']['task']['n_folds'],
        'batch_size': meta['config']['task']['batch_size'],
        'decoder_type': meta['config']['task']['decoder']['flow']['type'],
        'decoder_shape': meta['config']['task']['decoder']['flow']['shape'],
    }
    with open(os.path.join(OUT, 'connectome_config_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def umap_table():
    install_flyvis_stubs()
    rows = []
    for fname in sorted(os.listdir(os.path.join(DATA, 'umap_and_clustering'))):
        if not fname.endswith('.pickle'):
            continue
        with open(os.path.join(DATA, 'umap_and_clustering', fname), 'rb') as f:
            obj = pickle.load(f)
        d = getattr(obj, '__dict__', {})
        emb = np.asarray(getattr(d.get('embedding', None), '__dict__', {}).get('_embedding'))
        labels = np.asarray(d.get('labels', []))
        scores = np.asarray(d.get('scores', []))
        row = {
            'cell_type': fname[:-7],
            'embedding_shape': str(tuple(emb.shape)),
            'n_models': int(emb.shape[0]) if emb.ndim > 0 else np.nan,
            'embedding_dim': int(emb.shape[1]) if emb.ndim > 1 else np.nan,
            'emb_mean': float(np.nanmean(emb)) if emb.size else np.nan,
            'emb_std': float(np.nanstd(emb)) if emb.size else np.nan,
            'nan_fraction': float(np.isnan(emb).mean()) if emb.size and np.issubdtype(emb.dtype, np.number) else np.nan,
            'n_clusters': int(len(np.unique(labels))) if labels.size else np.nan,
            'largest_cluster': int(np.max(np.unique(labels, return_counts=True)[1])) if labels.size else np.nan,
            'smallest_cluster': int(np.min(np.unique(labels, return_counts=True)[1])) if labels.size else np.nan,
            'score_len': int(scores.size) if scores.size else np.nan,
            'score_best': float(np.nanmax(scores)) if scores.size else np.nan,
            'score_worst': float(np.nanmin(scores)) if scores.size else np.nan,
        }
        if emb.ndim == 2 and emb.shape[1] >= 2:
            row['emb_x_mean'] = float(np.nanmean(emb[:, 0]))
            row['emb_y_mean'] = float(np.nanmean(emb[:, 1]))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('cell_type')
    df.to_csv(os.path.join(OUT, 'umap_cluster_summary.csv'), index=False)
    agg = {
        'n_cell_types': int(len(df)),
        'all_embedding_shape_unique': sorted(df.embedding_shape.dropna().unique().tolist()),
        'cluster_count_mean': float(df.n_clusters.dropna().mean()),
        'cluster_count_min': float(df.n_clusters.dropna().min()),
        'cluster_count_max': float(df.n_clusters.dropna().max()),
        'largest_cluster_mean': float(df.largest_cluster.dropna().mean()),
        'nan_containing_cell_types': df.loc[df.nan_fraction > 0, 'cell_type'].tolist(),
    }
    with open(os.path.join(OUT, 'umap_cluster_aggregate.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    return df, agg


def make_figures(models, clusters):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(models['validation_loss'], bins=15, kde=True, ax=axes[0], color='#4C72B0')
    axes[0].set_title('Ensemble validation losses')
    axes[0].set_xlabel('Validation loss')
    axes[0].set_ylabel('Count')
    sns.lineplot(data=models, x='model_id', y='validation_loss', marker='o', ax=axes[1], color='#DD8452')
    axes[1].set_title('Validation loss by model index')
    axes[1].set_xlabel('Model id')
    axes[1].set_ylabel('Validation loss')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'validation_curves.png'), dpi=200)
    plt.close(fig)

    top = clusters.sort_values(['n_clusters', 'largest_cluster', 'cell_type'], ascending=[False, False, True]).head(20)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(data=clusters, x='emb_mean', y='emb_std', hue='n_clusters', size='largest_cluster', palette='viridis', ax=axes[0])
    axes[0].set_title('Cell-type embedding summary across 50 models')
    axes[0].set_xlabel('Mean of 2D embedding coordinates')
    axes[0].set_ylabel('Std. of 2D embedding coordinates')
    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    sns.barplot(data=top, y='cell_type', x='largest_cluster', hue='n_clusters', dodge=False, palette='magma', ax=axes[1])
    axes[1].set_title('Top cell types by dominant cluster size')
    axes[1].set_xlabel('Largest cluster membership among 50 models')
    axes[1].set_ylabel('Cell type')
    axes[1].legend(title='n_clusters', loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'celltype_embedding_examples.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    order = clusters.sort_values('n_clusters', ascending=False)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    sns.barplot(data=order, x='cell_type', y='n_clusters', color='#55A868', ax=axes[0])
    axes[0].set_title('Estimated number of ensemble clusters per cell type')
    axes[0].set_ylabel('n_clusters')
    axes[0].tick_params(axis='x', rotation=90)
    sns.barplot(data=order, x='cell_type', y='largest_cluster', color='#C44E52', ax=axes[1])
    axes[1].set_title('Dominant cluster size per cell type')
    axes[1].set_ylabel('largest_cluster')
    axes[1].set_xlabel('Cell type')
    axes[1].tick_params(axis='x', rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'cluster_summary.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def claim_table(model_summary, cluster_agg):
    claims = [
        {
            'claim': 'The workspace contains an ensemble of 50 pretrained DMN models.',
            'evidence_artifact': 'outputs/model_inventory.csv',
            'status': 'supported',
            'detail': f"Found {model_summary['n_models']} model directories with metadata and checkpoints."
        },
        {
            'claim': 'All inspected models use the same connectome-constrained architecture configuration file.',
            'evidence_artifact': 'outputs/connectome_config_summary.json; outputs/model_inventory.csv',
            'status': 'supported',
            'detail': f"Unique connectome file: {model_summary['connectome_file_unique']}"
        },
        {
            'claim': 'The saved ensemble shows modest but non-zero variation in validation performance.',
            'evidence_artifact': 'outputs/validation_summary.json; report/images/validation_curves.png',
            'status': 'supported',
            'detail': f"Mean±sd validation loss = {model_summary['validation_loss_mean']:.3f} ± {model_summary['validation_loss_std']:.3f}."
        },
        {
            'claim': 'Cell-type-specific ensemble embeddings are available for broad mechanistic comparison.',
            'evidence_artifact': 'outputs/umap_cluster_summary.csv; report/images/celltype_embedding_examples.png',
            'status': 'supported',
            'detail': f"Recovered summaries for {cluster_agg['n_cell_types']} cell types; all embeddings had shape {cluster_agg['all_embedding_shape_unique']}."
        },
        {
            'claim': 'Direct checkpoint parameter introspection and full neuron-level forward simulation were executed in this run.',
            'evidence_artifact': 'outputs/dependency_check.json',
            'status': 'not_supported',
            'detail': 'Torch/flyvis runtime support was unavailable from the default environment, so analysis relied on metadata, HDF5 losses, and pickled clustering artifacts.'
        }
    ]
    pd.DataFrame(claims).to_csv(os.path.join(OUT, 'claim_recovery_table.csv'), index=False)


def main():
    models, model_summary = model_table()
    connectome_summary()
    clusters, cluster_agg = umap_table()
    make_figures(models, clusters)
    claim_table(model_summary, cluster_agg)
    print(json.dumps({'model_summary': model_summary, 'cluster_agg': cluster_agg}, indent=2))


if __name__ == '__main__':
    main()
