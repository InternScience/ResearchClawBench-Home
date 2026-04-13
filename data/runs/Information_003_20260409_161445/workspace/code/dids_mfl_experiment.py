import json, math, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data.temporal import TemporalData
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'NF-UNSW-NB15-v2_3d.pt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)


def set_style():
    sns.set_theme(style='whitegrid', context='talk')


def load_data():
    add_safe_globals([TemporalData])
    obj = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    X = obj.msg.numpy().astype(np.float32)
    y_bin = obj.label.numpy().astype(int)
    attack = obj.attack.numpy().astype(int)
    t = obj.t.numpy().astype(np.int64)
    src = obj.src.numpy().astype(np.int64)
    dst = obj.dst.numpy().astype(np.int64)
    dt = obj.dt.numpy().astype(np.float32)
    src_layer = obj.src_layer.numpy().astype(np.int64) if hasattr(obj, 'src_layer') else np.zeros(len(y_bin), dtype=int)
    dst_layer = obj.dst_layer.numpy().astype(np.int64) if hasattr(obj, 'dst_layer') else np.zeros(len(y_bin), dtype=int)
    benign_id = int(pd.Series(attack[y_bin == 0]).mode().iloc[0])
    y_multi = attack.copy()
    return X, y_bin, y_multi, t, src, dst, dt, src_layer, dst_layer, benign_id


def temporal_features(X, t, src, dst, dt):
    n = len(t)
    order = np.argsort(t, kind='mergesort')
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    t_sorted = t[order]
    src_sorted = src[order]
    dst_sorted = dst[order]

    prev_src = {}
    prev_dst = {}
    prev_pair = {}
    src_gap = np.zeros(n, dtype=np.float32)
    dst_gap = np.zeros(n, dtype=np.float32)
    pair_gap = np.zeros(n, dtype=np.float32)
    src_freq = np.zeros(n, dtype=np.float32)
    dst_freq = np.zeros(n, dtype=np.float32)
    pair_freq = np.zeros(n, dtype=np.float32)

    count_src = {}
    count_dst = {}
    count_pair = {}
    for i, (tt, s, d) in enumerate(zip(t_sorted, src_sorted, dst_sorted)):
        src_gap[i] = tt - prev_src.get(s, tt)
        dst_gap[i] = tt - prev_dst.get(d, tt)
        pair_gap[i] = tt - prev_pair.get((s, d), tt)
        count_src[s] = count_src.get(s, 0) + 1
        count_dst[d] = count_dst.get(d, 0) + 1
        count_pair[(s, d)] = count_pair.get((s, d), 0) + 1
        src_freq[i] = count_src[s]
        dst_freq[i] = count_dst[d]
        pair_freq[i] = count_pair[(s, d)]
        prev_src[s] = tt
        prev_dst[d] = tt
        prev_pair[(s, d)] = tt

    deg_src = pd.Series(src).map(pd.Series(src).value_counts()).values.astype(np.float32)
    deg_dst = pd.Series(dst).map(pd.Series(dst).value_counts()).values.astype(np.float32)
    hour = (t % 86400) / 3600.0
    hour_sin = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    topo = np.column_stack([
        np.log1p(src_gap[inv]), np.log1p(dst_gap[inv]), np.log1p(pair_gap[inv]),
        np.log1p(src_freq[inv]), np.log1p(dst_freq[inv]), np.log1p(pair_freq[inv]),
        np.log1p(deg_src), np.log1p(deg_dst), dt, hour_sin, hour_cos
    ]).astype(np.float32)
    return topo


def disentangle_transform(X_train, X_test, n_components=16):
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)
    corr = np.corrcoef(Xs_train, rowvar=False)
    w = 1.0 - np.nanmean(np.abs(corr - np.eye(corr.shape[0])), axis=1)
    w = np.clip(w, 0.1, None)
    Xw_train = Xs_train * w
    Xw_test = Xs_test * w
    pca = PCA(n_components=n_components, random_state=SEED)
    Z_train = pca.fit_transform(Xw_train)
    Z_test = pca.transform(Xw_test)
    return {
        'scaler': scaler,
        'weights': w,
        'pca': pca,
        'Z_train': Z_train.astype(np.float32),
        'Z_test': Z_test.astype(np.float32),
        'explained_variance': pca.explained_variance_ratio_.tolist(),
    }


def multiscale_fusion(base_train, base_test, topo_train, topo_test, rep_train, rep_test):
    return np.concatenate([base_train, topo_train, rep_train], axis=1), np.concatenate([base_test, topo_test, rep_test], axis=1)


def evaluate_binary(X_train, X_test, y_train, y_test):
    models = {
        'logreg': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
        'rf': RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight='balanced_subsample'),
        'linsvc': LinearSVC(class_weight='balanced', random_state=SEED)
    }
    results = {}
    for name, model in models.items():
        clf = clone(model)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        scores = None
        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X_test)
        res = {
            'accuracy': float(accuracy_score(y_test, pred)),
            'f1_macro': float(f1_score(y_test, pred, average='macro')),
            'f1_weighted': float(f1_score(y_test, pred, average='weighted')),
            'f1_attack': float(f1_score(y_test, pred, pos_label=1)),
            'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
        }
        if scores is not None:
            try:
                res['roc_auc'] = float(roc_auc_score(y_test, scores))
            except Exception:
                pass
        results[name] = {'model': clf, 'metrics': res, 'pred': pred.tolist()}
    return results


def evaluate_multiclass(X_train, X_test, y_train, y_test):
    models = {
        'logreg': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED, multi_class='auto'),
        'rf': RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight='balanced_subsample')
    }
    results = {}
    labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    for name, model in models.items():
        clf = clone(model)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
        per_class = {str(lbl): {'precision': float(pp), 'recall': float(rr), 'f1': float(ff), 'support': int(ss)}
                     for lbl, pp, rr, ff, ss in zip(labels, p, r, f, s)}
        results[name] = {
            'model': clf,
            'metrics': {
                'accuracy': float(accuracy_score(y_test, pred)),
                'f1_macro': float(f1_score(y_test, pred, average='macro', zero_division=0)),
                'f1_weighted': float(f1_score(y_test, pred, average='weighted', zero_division=0)),
                'per_class': per_class,
            },
            'pred': pred.tolist(),
            'labels': labels,
        }
    return results


def unknown_attack_eval(X, y_multi, benign_id, topo):
    attack_ids = sorted([int(v) for v in np.unique(y_multi) if int(v) != benign_id])
    rows = []
    for unknown_id in attack_ids:
        known_mask = (y_multi != unknown_id)
        test_mask = np.ones(len(y_multi), dtype=bool)
        # temporal split among known data
        idx_known = np.where(known_mask)[0]
        split = int(len(idx_known) * 0.8)
        train_idx = idx_known[:split]
        test_idx = np.where(test_mask)[0][int(len(y_multi)*0.8):]
        # ensure unknown present in test
        test_idx = np.unique(np.concatenate([test_idx, np.where(y_multi == unknown_id)[0]]))
        yb_train = (y_multi[train_idx] != benign_id).astype(int)
        yb_test = (y_multi[test_idx] != benign_id).astype(int)
        rep = disentangle_transform(X[train_idx], X[test_idx], n_components=16)
        Xtr, Xte = multiscale_fusion(X[train_idx], X[test_idx], topo[train_idx], topo[test_idx], rep['Z_train'], rep['Z_test'])
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
        clf.fit(Xtr, yb_train)
        pred = clf.predict(Xte)
        mask_unknown = (y_multi[test_idx] == unknown_id)
        if mask_unknown.sum() == 0:
            continue
        f1_unknown = f1_score(yb_test[mask_unknown], pred[mask_unknown], zero_division=0)
        recall_unknown = (pred[mask_unknown] == 1).mean()
        overall = f1_score(yb_test, pred)
        rows.append({'unknown_attack_id': int(unknown_id), 'n_unknown_test': int(mask_unknown.sum()), 'unknown_recall_as_attack': float(recall_unknown), 'unknown_f1_binary': float(f1_unknown), 'overall_attack_f1': float(overall)})
    return pd.DataFrame(rows)


def fewshot_eval(X, y_multi, benign_id, topo, shots=(1,5,10,20)):
    rng = np.random.default_rng(SEED)
    attack_ids = [int(v) for v in np.unique(y_multi) if int(v) != benign_id]
    rows = []
    benign_idx = np.where(y_multi == benign_id)[0]
    for shot in shots:
        train_parts = []
        for aid in attack_ids:
            idx = np.where(y_multi == aid)[0]
            k = min(shot, len(idx) // 2 if len(idx) > 1 else 1)
            chosen = rng.choice(idx, size=max(1, k), replace=False)
            train_parts.append(chosen)
        benign_k = min(len(benign_idx)//3, max(50, shot * len(attack_ids) * 2))
        benign_train = rng.choice(benign_idx, size=benign_k, replace=False)
        train_idx = np.unique(np.concatenate(train_parts + [benign_train]))
        test_idx = np.setdiff1d(np.arange(len(y_multi)), train_idx)
        rep = disentangle_transform(X[train_idx], X[test_idx], n_components=16)
        Xtr, Xte = multiscale_fusion(X[train_idx], X[test_idx], topo[train_idx], topo[test_idx], rep['Z_train'], rep['Z_test'])
        clf = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED, multi_class='auto')
        clf.fit(Xtr, y_multi[train_idx])
        pred = clf.predict(Xte)
        rows.append({'shots_per_attack': int(shot), 'train_size': int(len(train_idx)), 'test_size': int(len(test_idx)), 'macro_f1': float(f1_score(y_multi[test_idx], pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y_multi[test_idx], pred, average='weighted', zero_division=0)), 'accuracy': float(accuracy_score(y_multi[test_idx], pred))})
    return pd.DataFrame(rows)


def plot_overview(y_bin, y_multi, topo):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    pd.Series(y_bin).map({0:'Benign',1:'Attack'}).value_counts().plot(kind='bar', ax=axes[0], color=['#4daf4a','#e41a1c'])
    axes[0].set_title('Binary label distribution')
    axes[0].set_ylabel('Count')
    pd.Series(y_multi).value_counts().sort_index().plot(kind='bar', ax=axes[1], color='#377eb8')
    axes[1].set_title('Attack ID distribution')
    axes[1].set_ylabel('Count')
    corr = np.corrcoef(topo, rowvar=False)
    sns.heatmap(corr, ax=axes[2], cmap='coolwarm', center=0)
    axes[2].set_title('Correlation of temporal-topological features')
    plt.tight_layout()
    plt.savefig(IMG / 'data_overview.png', dpi=200)
    plt.close()


def plot_embedding(Xf, y_multi, sample_n=4000):
    set_style()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(np.arange(len(Xf)), size=min(sample_n, len(Xf)), replace=False)
    emb = TSNE(n_components=2, random_state=SEED, init='pca', learning_rate='auto', perplexity=30).fit_transform(Xf[idx])
    df = pd.DataFrame({'x': emb[:,0], 'y': emb[:,1], 'attack': y_multi[idx].astype(str)})
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='x', y='y', hue='attack', s=18, alpha=0.75, palette='tab10', linewidth=0)
    plt.title('t-SNE of fused representations')
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', title='Attack ID')
    plt.tight_layout()
    plt.savefig(IMG / 'embedding_tsne.png', dpi=200)
    plt.close()


def plot_binary_compare(results_baseline, results_fused):
    set_style()
    rows = []
    for family, results in [('Baseline', results_baseline), ('DIDS-MFL', results_fused)]:
        for name, pack in results.items():
            m = pack['metrics']
            rows.append({'setting': family, 'model': name, 'metric': 'Attack F1', 'value': m['f1_attack']})
            rows.append({'setting': family, 'model': name, 'metric': 'Macro F1', 'value': m['f1_macro']})
            if 'roc_auc' in m:
                rows.append({'setting': family, 'model': name, 'metric': 'ROC-AUC', 'value': m['roc_auc']})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='model', y='value', hue='setting')
    plt.title('Binary intrusion detection performance')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(IMG / 'binary_comparison.png', dpi=200)
    plt.close()


def plot_multiclass_perclass(per_class_base, per_class_fused):
    rows = []
    for lbl, vals in per_class_base.items():
        rows.append({'attack_id': lbl, 'setting': 'Baseline', 'f1': vals['f1']})
    for lbl, vals in per_class_fused.items():
        rows.append({'attack_id': lbl, 'setting': 'DIDS-MFL', 'f1': vals['f1']})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(11,6))
    sns.barplot(data=df, x='attack_id', y='f1', hue='setting')
    plt.title('Per-class F1 in multiclass classification')
    plt.ylim(0,1.05)
    plt.tight_layout()
    plt.savefig(IMG / 'multiclass_perclass_f1.png', dpi=200)
    plt.close()


def plot_unknown(df):
    plt.figure(figsize=(10,5))
    sns.barplot(data=df, x='unknown_attack_id', y='unknown_recall_as_attack', color='#984ea3')
    plt.title('Unknown attack detection recall')
    plt.ylim(0,1.05)
    plt.tight_layout()
    plt.savefig(IMG / 'unknown_attack_recall.png', dpi=200)
    plt.close()


def plot_fewshot(df):
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df, x='shots_per_attack', y='macro_f1', marker='o', linewidth=3)
    plt.title('Few-shot multiclass performance vs. shots per attack')
    plt.ylim(0,1.05)
    plt.tight_layout()
    plt.savefig(IMG / 'fewshot_curve.png', dpi=200)
    plt.close()


def main():
    X, y_bin, y_multi, t, src, dst, dt, src_layer, dst_layer, benign_id = load_data()
    topo = temporal_features(X, t, src, dst, dt)
    plot_overview(y_bin, y_multi, topo)

    # Temporal split for primary evaluation
    n = len(X)
    split = int(n * 0.8)
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)

    baseline_scaler = StandardScaler()
    Xtr_base = baseline_scaler.fit_transform(X[train_idx])
    Xte_base = baseline_scaler.transform(X[test_idx])

    rep = disentangle_transform(X[train_idx], X[test_idx], n_components=16)
    Xtr_fused, Xte_fused = multiscale_fusion(Xtr_base, Xte_base, topo[train_idx], topo[test_idx], rep['Z_train'], rep['Z_test'])

    binary_base = evaluate_binary(Xtr_base, Xte_base, y_bin[train_idx], y_bin[test_idx])
    binary_fused = evaluate_binary(Xtr_fused, Xte_fused, y_bin[train_idx], y_bin[test_idx])
    multiclass_base = evaluate_multiclass(Xtr_base, Xte_base, y_multi[train_idx], y_multi[test_idx])
    multiclass_fused = evaluate_multiclass(Xtr_fused, Xte_fused, y_multi[train_idx], y_multi[test_idx])

    unknown_df = unknown_attack_eval(X, y_multi, benign_id, topo)
    fewshot_df = fewshot_eval(X, y_multi, benign_id, topo)

    plot_embedding(Xte_fused, y_multi[test_idx])
    plot_binary_compare(binary_base, binary_fused)
    plot_multiclass_perclass(multiclass_base['rf']['metrics']['per_class'], multiclass_fused['rf']['metrics']['per_class'])
    plot_unknown(unknown_df)
    plot_fewshot(fewshot_df)

    dataset_summary = {
        'n_samples': int(len(X)),
        'n_raw_features': int(X.shape[1]),
        'n_temporal_topological_features': int(topo.shape[1]),
        'benign_attack_id': int(benign_id),
        'binary_counts': {str(int(k)): int(v) for k, v in pd.Series(y_bin).value_counts().sort_index().items()},
        'attack_counts': {str(int(k)): int(v) for k, v in pd.Series(y_multi).value_counts().sort_index().items()},
        'disentangled_explained_variance_first5': rep['explained_variance'][:5],
    }

    serializable = {
        'dataset_summary': dataset_summary,
        'binary_baseline': {k:v['metrics'] for k,v in binary_base.items()},
        'binary_dids_mfl': {k:v['metrics'] for k,v in binary_fused.items()},
        'multiclass_baseline': {k:v['metrics'] for k,v in multiclass_base.items()},
        'multiclass_dids_mfl': {k:v['metrics'] for k,v in multiclass_fused.items()},
        'unknown_attack_results': unknown_df.to_dict(orient='records'),
        'fewshot_results': fewshot_df.to_dict(orient='records'),
        'train_size': int(len(train_idx)),
        'test_size': int(len(test_idx)),
    }
    with open(OUT / 'results_summary.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    pd.DataFrame(serializable['unknown_attack_results']).to_csv(OUT / 'unknown_attack_results.csv', index=False)
    pd.DataFrame(serializable['fewshot_results']).to_csv(OUT / 'fewshot_results.csv', index=False)
    with open(OUT / 'dataset_summary.json', 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    print(json.dumps({
        'binary_rf_baseline': binary_base['rf']['metrics'],
        'binary_rf_dids_mfl': binary_fused['rf']['metrics'],
        'multiclass_rf_baseline': {'accuracy': multiclass_base['rf']['metrics']['accuracy'], 'f1_macro': multiclass_base['rf']['metrics']['f1_macro']},
        'multiclass_rf_dids_mfl': {'accuracy': multiclass_fused['rf']['metrics']['accuracy'], 'f1_macro': multiclass_fused['rf']['metrics']['f1_macro']},
    }, indent=2))

if __name__ == '__main__':
    main()
