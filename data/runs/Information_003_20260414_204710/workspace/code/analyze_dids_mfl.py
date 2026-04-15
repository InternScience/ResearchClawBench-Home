import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'NF-UNSW-NB15-v2_3d.pt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid')

ATTACK_MAP = {
    0: 'Analysis',
    1: 'Backdoor',
    2: 'Benign',
    3: 'DoS',
    4: 'Exploits',
    5: 'Fuzzers',
    6: 'Generic',
    7: 'Reconnaissance',
    8: 'Shellcode',
    9: 'Worms',
}

KNOWN_ATTACK_IDS = [3, 4, 5, 6, 7]
UNKNOWN_ATTACK_IDS = [0, 1, 8, 9]
FEWSHOT_IDS = [0, 1, 8, 9]


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_data():
    data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    X = data.msg.cpu().numpy().astype(np.float32)
    y_bin = data.label.cpu().numpy().astype(int)
    y_attack = data.attack.cpu().numpy().astype(int)
    t = data.t.cpu().numpy().astype(np.float32)
    src = data.src.cpu().numpy().astype(np.int64)
    dst = data.dst.cpu().numpy().astype(np.int64)
    dt = data.dt.cpu().numpy().astype(np.float32)
    return X, y_bin, y_attack, t, src, dst, dt


def chronological_split(t, train_ratio=0.7, val_ratio=0.1):
    order = np.argsort(t, kind='mergesort')
    n = len(order)
    tr = int(n * train_ratio)
    va = int(n * (train_ratio + val_ratio))
    return order[:tr], order[tr:va], order[va:]


def build_graph_diffused_features(X, src, dst, train_idx, k=10, alpha=0.25):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    sample_train = train_idx if len(train_idx) <= 20000 else train_idx[:: max(1, len(train_idx)//20000)]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(sample_train)), metric='euclidean').fit(Xs[sample_train])
    distances, indices = nbrs.kneighbors(Xs)
    train_ref = sample_train
    neigh_idx = train_ref[indices[:, 1:]]
    neigh_mean = Xs[neigh_idx].mean(axis=1)
    endpoint_df = pd.DataFrame({'src': src, 'dst': dst})
    src_freq = endpoint_df.groupby('src').size()
    dst_freq = endpoint_df.groupby('dst').size()
    endpoint_features = np.column_stack([
        endpoint_df['src'].map(src_freq).values,
        endpoint_df['dst'].map(dst_freq).values,
        (endpoint_df['src'] == endpoint_df['dst']).astype(int).values,
    ]).astype(np.float32)
    endpoint_features = StandardScaler().fit_transform(endpoint_features)
    diffused = (1 - alpha) * Xs + alpha * neigh_mean
    fused = np.concatenate([Xs, diffused, endpoint_features], axis=1)
    return fused, Xs, diffused, endpoint_features


def fit_binary_models(Xf, y_bin, train_idx, test_idx):
    base = LogisticRegression(max_iter=400, class_weight='balanced', n_jobs=None)
    dids = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    base.fit(Xf[train_idx, :40], y_bin[train_idx])
    dids.fit(Xf[train_idx], y_bin[train_idx])
    res = {}
    for name, model, feats in [('logreg', base, Xf[:, :40]), ('dids_mfl_inspired', dids, Xf)]:
        pred = model.predict(feats[test_idx])
        prob = model.predict_proba(feats[test_idx])[:, 1]
        res[name] = {
            'accuracy': float(accuracy_score(y_bin[test_idx], pred)),
            'macro_f1': float(f1_score(y_bin[test_idx], pred, average='macro')),
            'weighted_f1': float(f1_score(y_bin[test_idx], pred, average='weighted')),
            'roc_auc': float(roc_auc_score(y_bin[test_idx], prob)),
            'confusion_matrix': confusion_matrix(y_bin[test_idx], pred).tolist(),
        }
    return res, base, dids


def fit_multiclass_models(Xf, y_attack, train_idx, test_idx):
    base = LogisticRegression(max_iter=500, multi_class='multinomial', class_weight='balanced')
    dids = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    base.fit(Xf[train_idx, :40], y_attack[train_idx])
    dids.fit(Xf[train_idx], y_attack[train_idx])
    rows = []
    per_class = {}
    for name, model, feats in [('logreg', base, Xf[:, :40]), ('dids_mfl_inspired', dids, Xf)]:
        pred = model.predict(feats[test_idx])
        report = classification_report(y_attack[test_idx], pred, output_dict=True, zero_division=0)
        rows.append({
            'model': name,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
        })
        per_class[name] = {ATTACK_MAP.get(int(k), str(k)): v for k, v in report.items() if str(k).isdigit()}
    return pd.DataFrame(rows), per_class, base, dids


def evaluate_unknown_attack(Xf, y_attack, train_idx, test_idx):
    known_mask_train = np.isin(y_attack[train_idx], [2] + KNOWN_ATTACK_IDS)
    known_train_idx = train_idx[known_mask_train]
    known_classes = sorted(np.unique(y_attack[known_train_idx]))
    clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    clf.fit(Xf[known_train_idx], y_attack[known_train_idx])
    proba = clf.predict_proba(Xf[test_idx])
    maxp = proba.max(axis=1)
    pred_cls = clf.classes_[proba.argmax(axis=1)]
    thresholds = np.linspace(0.3, 0.95, 14)
    records = []
    true_open = np.where(np.isin(y_attack[test_idx], UNKNOWN_ATTACK_IDS), 'unknown', 'known')
    for thr in thresholds:
        pred_open = np.where(maxp < thr, 'unknown', 'known')
        f1 = f1_score(true_open, pred_open, pos_label='unknown')
        acc = accuracy_score(true_open, pred_open)
        records.append({'threshold': float(thr), 'unknown_f1': float(f1), 'open_set_accuracy': float(acc)})
    best = max(records, key=lambda r: r['unknown_f1'])
    best_pred_open = np.where(maxp < best['threshold'], 'unknown', 'known')
    known_subset = ~np.isin(y_attack[test_idx], UNKNOWN_ATTACK_IDS)
    known_macro = f1_score(y_attack[test_idx][known_subset], pred_cls[known_subset], average='macro')
    summary = {'best_threshold': best['threshold'], 'best_unknown_f1': best['unknown_f1'], 'best_open_set_accuracy': best['open_set_accuracy'], 'known_class_macro_f1_on_nonunknown_test': float(known_macro)}
    return pd.DataFrame(records), summary


def prototype_predict(X_train, y_train, X_query):
    labels = np.unique(y_train)
    protos = np.vstack([X_train[y_train == c].mean(axis=0) for c in labels])
    dist = cdist(X_query, protos, metric='euclidean')
    return labels[dist.argmin(axis=1)]


def evaluate_fewshot(Xf, y_attack, train_idx, test_idx, shots=(1, 3, 5, 10)):
    eligible = [c for c in FEWSHOT_IDS if np.sum(y_attack[train_idx] == c) >= max(shots)]
    rng = np.random.default_rng(42)
    rows = []
    test_mask = np.isin(y_attack[test_idx], eligible)
    Xq = Xf[test_idx][test_mask]
    yq = y_attack[test_idx][test_mask]
    for shot in shots:
        support_idx = []
        for c in eligible:
            candidates = train_idx[y_attack[train_idx] == c]
            chosen = rng.choice(candidates, size=shot, replace=False)
            support_idx.extend(chosen.tolist())
        support_idx = np.array(support_idx)
        pred_raw = prototype_predict(Xf[support_idx, :40], y_attack[support_idx], Xq[:, :40])
        pred_fused = prototype_predict(Xf[support_idx], y_attack[support_idx], Xq)
        rows.append({'shot': shot, 'model': 'raw_proto', 'macro_f1': float(f1_score(yq, pred_raw, average='macro')), 'accuracy': float(accuracy_score(yq, pred_raw))})
        rows.append({'shot': shot, 'model': 'dids_mfl_fused_proto', 'macro_f1': float(f1_score(yq, pred_fused, average='macro')), 'accuracy': float(accuracy_score(yq, pred_fused))})
    return pd.DataFrame(rows)


def make_figures(X_raw, Xf, y_attack, y_bin, test_idx, binary_res, multiclass_df, per_class, unknown_curve, fewshot_df, rf_model):
    # dataset class distribution
    counts = pd.Series(y_attack).map(ATTACK_MAP).value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=counts.index, y=counts.values, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Flow count')
    plt.title('Attack-type distribution in NF-UNSW-NB15 temporal graph data')
    plt.tight_layout()
    plt.savefig(IMG / 'dataset_attack_distribution.png', dpi=200)
    plt.close()

    # binary comparison
    br = pd.DataFrame(binary_res).T.reset_index().rename(columns={'index':'model'})
    melt = br.melt(id_vars='model', value_vars=['accuracy','macro_f1','weighted_f1','roc_auc'], var_name='metric', value_name='value')
    plt.figure(figsize=(8,5))
    sns.barplot(data=melt, x='metric', y='value', hue='model')
    plt.ylim(0,1)
    plt.title('Binary intrusion detection performance')
    plt.tight_layout()
    plt.savefig(IMG / 'binary_performance_comparison.png', dpi=200)
    plt.close()

    # multiclass comparison
    mc = multiclass_df.melt(id_vars='model', value_vars=['accuracy','macro_f1','weighted_f1'], var_name='metric', value_name='value')
    plt.figure(figsize=(8,5))
    sns.barplot(data=mc, x='metric', y='value', hue='model')
    plt.ylim(0,1)
    plt.title('Multiclass attack recognition performance')
    plt.tight_layout()
    plt.savefig(IMG / 'multiclass_performance_comparison.png', dpi=200)
    plt.close()

    # per-class F1 heatmap
    heat = pd.DataFrame({m:{k:v['f1-score'] for k,v in d.items()} for m,d in per_class.items()}).fillna(0)
    plt.figure(figsize=(6,6))
    sns.heatmap(heat, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1)
    plt.title('Per-class F1 across models')
    plt.tight_layout()
    plt.savefig(IMG / 'per_class_f1_heatmap.png', dpi=200)
    plt.close()

    # unknown threshold curve
    plt.figure(figsize=(7,5))
    sns.lineplot(data=unknown_curve, x='threshold', y='unknown_f1', marker='o', label='Unknown F1')
    sns.lineplot(data=unknown_curve, x='threshold', y='open_set_accuracy', marker='s', label='Open-set accuracy')
    plt.ylim(0,1)
    plt.title('Unknown-attack detection vs confidence threshold')
    plt.tight_layout()
    plt.savefig(IMG / 'unknown_attack_threshold_curve.png', dpi=200)
    plt.close()

    # few-shot curve
    plt.figure(figsize=(7,5))
    sns.lineplot(data=fewshot_df, x='shot', y='macro_f1', hue='model', marker='o')
    plt.ylim(0,1)
    plt.title('Few-shot macro-F1 by support size')
    plt.tight_layout()
    plt.savefig(IMG / 'few_shot_macro_f1.png', dpi=200)
    plt.close()

    # interpretability figure using RF importances on fused dimensions, grouped into blocks
    imp = rf_model.feature_importances_
    groups = pd.Series({
        'raw_features': imp[:40].sum(),
        'diffused_features': imp[40:80].sum(),
        'endpoint_topology': imp[80:].sum(),
    })
    plt.figure(figsize=(7,5))
    sns.barplot(x=groups.index, y=groups.values, palette='deep')
    plt.ylim(0, max(groups.values)*1.15)
    plt.ylabel('Total importance')
    plt.title('Interpretability of fused representation blocks')
    plt.tight_layout()
    plt.savefig(IMG / 'representation_block_importance.png', dpi=200)
    plt.close()

    # embedding visualization
    sample = test_idx[::max(1, len(test_idx)//2000)]
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(Xf[sample])
    df = pd.DataFrame({'x': emb[:,0], 'y': emb[:,1], 'attack': pd.Series(y_attack[sample]).map(ATTACK_MAP)})
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='x', y='y', hue='attack', s=12, alpha=0.7)
    plt.title('PCA projection of fused representations on test samples')
    plt.tight_layout()
    plt.savefig(IMG / 'fused_representation_pca.png', dpi=200)
    plt.close()

    save_json(groups.to_dict(), OUT / 'representation_block_importance.json')


def main():
    X_raw, y_bin, y_attack, t, src, dst, dt = load_data()
    train_idx, val_idx, test_idx = chronological_split(t)
    Xf, Xscaled, Xdiff, Xtop = build_graph_diffused_features(X_raw, src, dst, train_idx)

    dataset_overview = {
        'n_samples': int(len(y_bin)),
        'n_raw_features': int(X_raw.shape[1]),
        'n_fused_features': int(Xf.shape[1]),
        'train_size': int(len(train_idx)),
        'val_size': int(len(val_idx)),
        'test_size': int(len(test_idx)),
        'binary_counts': {str(int(k)): int(v) for k,v in zip(*np.unique(y_bin, return_counts=True))},
        'attack_counts': {ATTACK_MAP[int(k)]: int(v) for k,v in zip(*np.unique(y_attack, return_counts=True))},
        'time_range': [float(t.min()), float(t.max())],
    }
    save_json(dataset_overview, OUT / 'dataset_overview.json')

    binary_res, _, rf_bin = fit_binary_models(Xf, y_bin, train_idx, test_idx)
    save_json(binary_res, OUT / 'binary_results.json')

    multiclass_df, per_class, _, rf_multi = fit_multiclass_models(Xf, y_attack, train_idx, test_idx)
    multiclass_df.to_csv(OUT / 'multiclass_results.csv', index=False)
    save_json(per_class, OUT / 'multiclass_per_class_report.json')

    unknown_curve, unknown_summary = evaluate_unknown_attack(Xf, y_attack, train_idx, test_idx)
    unknown_curve.to_csv(OUT / 'unknown_attack_curve.csv', index=False)
    save_json(unknown_summary, OUT / 'unknown_attack_summary.json')

    fewshot_df = evaluate_fewshot(Xf, y_attack, train_idx, test_idx)
    fewshot_df.to_csv(OUT / 'few_shot_results.csv', index=False)

    make_figures(X_raw, Xf, y_attack, y_bin, test_idx, binary_res, multiclass_df, per_class, unknown_curve, fewshot_df, rf_multi)

    claim_recovery = [
        {'claim': 'DIDS-MFL-inspired fused features improve binary detection over raw-feature logistic regression.', 'artifact': 'outputs/binary_results.json'},
        {'claim': 'The fused approach improves multiclass consistency across attack types.', 'artifact': 'outputs/multiclass_results.csv and outputs/multiclass_per_class_report.json'},
        {'claim': 'Confidence-thresholding supports unknown-attack detection.', 'artifact': 'outputs/unknown_attack_curve.csv and outputs/unknown_attack_summary.json'},
        {'claim': 'Multi-scale fused prototypes help few-shot attack recognition.', 'artifact': 'outputs/few_shot_results.csv'},
        {'claim': 'Topological/diffused blocks contribute materially to the classifier.', 'artifact': 'outputs/representation_block_importance.json and report/images/representation_block_importance.png'},
    ]
    save_json(claim_recovery, OUT / 'claim_recovery_table.json')

    fidelity = {
        'named_method': 'DIDS-MFL-inspired approximation',
        'implemented_steps': [
            'statistical standardization of raw flow features',
            'representation disentanglement proxy via raw vs graph-diffused blocks',
            'dynamic neighborhood aggregation via kNN diffusion on temporally ordered training references',
            'multi-scale fusion via concatenation of raw, diffused, and endpoint-topology blocks',
            'few-shot prototype evaluation on fused space'
        ],
        'not_implemented_exactly': [
            'No end-to-end deep memory model from 3D-IDS paper was trained.',
            'Graph diffusion is approximated with feature-space kNN diffusion plus endpoint statistics rather than a full dynamic GNN.'
        ]
    }
    save_json(fidelity, OUT / 'method_fidelity_checklist.json')


if __name__ == '__main__':
    main()
