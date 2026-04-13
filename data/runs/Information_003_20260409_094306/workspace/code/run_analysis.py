import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals
from torch_geometric.data.temporal import TemporalData


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "NF-UNSW-NB15-v2_3d.pt"
OUTPUTS = ROOT / "outputs"
IMAGES = ROOT / "report" / "images"


def ensure_dirs():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)


def load_temporal_data():
    add_safe_globals([TemporalData])
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    return data


def build_dataframe(data: TemporalData) -> pd.DataFrame:
    msg = data.msg.numpy()
    cols = [f"f{i:02d}" for i in range(msg.shape[1])]
    df = pd.DataFrame(msg, columns=cols)
    df["src"] = data.src.numpy()
    df["dst"] = data.dst.numpy()
    df["t"] = data.t.numpy()
    df["label"] = data.label.numpy()
    df["attack"] = data.attack.numpy()
    return df


def add_temporal_topology_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["t"] / 3600.0
    out["time_sin"] = np.sin(2 * np.pi * out["t"] / 86400.0)
    out["time_cos"] = np.cos(2 * np.pi * out["t"] / 86400.0)

    src_count = out.groupby("src").cumcount()
    dst_count = out.groupby("dst").cumcount()
    pair_count = out.groupby(["src", "dst"]).cumcount()
    out["src_prev_count"] = src_count
    out["dst_prev_count"] = dst_count
    out["pair_prev_count"] = pair_count
    out["degree_prev_sum"] = src_count + dst_count

    feature_cols = [c for c in out.columns if c.startswith("f")]
    flow_intensity = out[feature_cols[:10]].mean(axis=1)
    out["flow_intensity"] = flow_intensity
    out["src_roll_mean"] = (
        out.assign(flow_intensity=flow_intensity)
        .groupby("src")["flow_intensity"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    out["dst_roll_mean"] = (
        out.assign(flow_intensity=flow_intensity)
        .groupby("dst")["flow_intensity"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    return out


def build_disentangled_views(df: pd.DataFrame):
    base_features = [c for c in df.columns if c.startswith("f")]
    temporal_features = [
        "hour",
        "time_sin",
        "time_cos",
        "src_prev_count",
        "dst_prev_count",
        "pair_prev_count",
        "degree_prev_sum",
        "src_roll_mean",
        "dst_roll_mean",
        "flow_intensity",
    ]

    groups = {
        "stat_low": base_features[:14],
        "stat_mid": base_features[14:27],
        "stat_high": base_features[27:],
        "dynamic": temporal_features,
    }
    return groups


def make_representation(train_df: pd.DataFrame, test_df: pd.DataFrame, groups):
    train_views = []
    test_views = []
    view_dims = {}
    for name, cols in groups.items():
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[cols])
        test_scaled = scaler.transform(test_df[cols])
        n_comp = min(4, train_scaled.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)
        train_views.append(train_pca)
        test_views.append(test_pca)
        view_dims[name] = int(n_comp)

    train_repr = np.hstack(train_views)
    test_repr = np.hstack(test_views)
    return train_repr, test_repr, view_dims


def evaluate_binary(train_df, test_df, train_repr, test_repr):
    x_train_raw = train_df[[c for c in train_df.columns if c.startswith("f")] + [
        "hour", "time_sin", "time_cos", "src_prev_count", "dst_prev_count",
        "pair_prev_count", "degree_prev_sum", "src_roll_mean", "dst_roll_mean",
        "flow_intensity"
    ]]
    x_test_raw = test_df[x_train_raw.columns]
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    baseline = ExtraTreesClassifier(
        n_estimators=120, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
    proposed = RandomForestClassifier(
        n_estimators=160, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
    baseline.fit(x_train_raw, y_train)
    proposed.fit(train_repr, y_train)

    baseline_pred = baseline.predict(x_test_raw)
    proposed_pred = proposed.predict(test_repr)
    baseline_prob = baseline.predict_proba(x_test_raw)[:, 1]
    proposed_prob = proposed.predict_proba(test_repr)[:, 1]

    return {
        "baseline": {
            "accuracy": accuracy_score(y_test, baseline_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, baseline_pred),
            "macro_f1": f1_score(y_test, baseline_pred, average="macro"),
            "roc_auc": roc_auc_score(y_test, baseline_prob),
        },
        "proposed": {
            "accuracy": accuracy_score(y_test, proposed_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, proposed_pred),
            "macro_f1": f1_score(y_test, proposed_pred, average="macro"),
            "roc_auc": roc_auc_score(y_test, proposed_prob),
        },
        "y_test": y_test,
        "y_pred": proposed_pred,
        "y_prob": proposed_prob,
    }


def evaluate_multiclass(train_df, test_df, train_repr, test_repr):
    y_train = train_df["attack"].to_numpy()
    y_test = test_df["attack"].to_numpy()

    raw_cols = [c for c in train_df.columns if c.startswith("f")] + [
        "hour", "time_sin", "time_cos", "src_prev_count", "dst_prev_count",
        "pair_prev_count", "degree_prev_sum", "src_roll_mean", "dst_roll_mean",
        "flow_intensity"
    ]
    baseline = ExtraTreesClassifier(
        n_estimators=120, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
    proposed = RandomForestClassifier(
        n_estimators=180, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
    baseline.fit(train_df[raw_cols], y_train)
    proposed.fit(train_repr, y_train)

    baseline_pred = baseline.predict(test_df[raw_cols])
    proposed_pred = proposed.predict(test_repr)
    labels = sorted(np.unique(y_test).tolist())

    return {
        "baseline": {
            "accuracy": accuracy_score(y_test, baseline_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, baseline_pred),
            "macro_f1": f1_score(y_test, baseline_pred, average="macro"),
        },
        "proposed": {
            "accuracy": accuracy_score(y_test, proposed_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, proposed_pred),
            "macro_f1": f1_score(y_test, proposed_pred, average="macro"),
        },
        "report": classification_report(y_test, proposed_pred, labels=labels, output_dict=True, zero_division=0),
        "labels": labels,
        "y_test": y_test,
        "y_pred": proposed_pred,
    }


def evaluate_unknown_attack(df, groups):
    labels = sorted(df["attack"].unique().tolist())
    attack_labels = [x for x in labels if x != 2]
    holdout_attack = min(attack_labels, key=lambda x: df[df["attack"] == x].shape[0])

    known_train = df[df["attack"] != holdout_attack].copy()
    unknown_test = df[df["attack"] == holdout_attack].copy()
    known_train["known_binary"] = (known_train["attack"] != 2).astype(int)
    unknown_test["known_binary"] = 1

    train_df, calib_df = train_test_split(
        known_train, test_size=0.2, stratify=known_train["known_binary"], random_state=42
    )
    train_repr, calib_repr, _ = make_representation(train_df, calib_df, groups)
    _, unknown_repr, _ = make_representation(train_df, unknown_test, groups)

    clf = RandomForestClassifier(
        n_estimators=140, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
    clf.fit(train_repr, train_df["known_binary"])
    calib_prob = clf.predict_proba(calib_repr)[:, 1]
    threshold = np.quantile(calib_prob[calib_df["known_binary"].to_numpy() == 0], 0.95)
    unknown_prob = clf.predict_proba(unknown_repr)[:, 1]

    return {
        "holdout_attack": int(holdout_attack),
        "threshold": float(threshold),
        "unknown_attack_mean_score": float(np.mean(unknown_prob)),
        "unknown_attack_rejection_rate": float(np.mean(unknown_prob < threshold)),
    }


def prototype_scores(query, proto):
    qn = np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
    pn = np.linalg.norm(proto, axis=1, keepdims=True).T + 1e-8
    cosine = (query @ proto.T) / (qn * pn)
    dists = np.linalg.norm(query[:, None, :] - proto[None, :, :], axis=2)
    euclid = -dists
    return cosine + euclid / (np.std(dists) + 1e-8)


def evaluate_few_shot(train_df, test_df, train_repr, test_repr):
    rare_classes = train_df["attack"].value_counts().sort_values().head(3).index.tolist()
    subset_train = train_df[train_df["attack"].isin(rare_classes)]
    subset_test = test_df[test_df["attack"].isin(rare_classes)]
    train_idx = subset_train.index.to_numpy()
    test_idx = subset_test.index.to_numpy()
    full_train_idx = train_df.index.to_numpy()
    full_test_idx = test_df.index.to_numpy()
    pos_train = {idx: i for i, idx in enumerate(full_train_idx)}
    pos_test = {idx: i for i, idx in enumerate(full_test_idx)}

    support_idx = []
    query_idx = []
    rng = np.random.default_rng(42)
    for cls in rare_classes:
        cls_idx = subset_train[subset_train["attack"] == cls].index.to_numpy()
        take = min(5, len(cls_idx))
        chosen = rng.choice(cls_idx, size=take, replace=False)
        support_idx.extend(chosen.tolist())

        q_idx = subset_test[subset_test["attack"] == cls].index.to_numpy()
        query_idx.extend(q_idx.tolist())

    support_pos = np.array([pos_train[i] for i in support_idx], dtype=int)
    query_pos = np.array([pos_test[i] for i in query_idx], dtype=int)
    support_labels = train_df.loc[support_idx, "attack"].to_numpy()
    query_labels = test_df.loc[query_idx, "attack"].to_numpy()

    support_repr = train_repr[support_pos]
    query_repr = test_repr[query_pos]
    prototypes = []
    proto_labels = []
    for cls in rare_classes:
        mask = support_labels == cls
        prototypes.append(support_repr[mask].mean(axis=0))
        proto_labels.append(cls)
    prototypes = np.vstack(prototypes)
    scores = prototype_scores(query_repr, prototypes)
    pred = np.array([proto_labels[i] for i in scores.argmax(axis=1)])

    return {
        "classes": [int(x) for x in rare_classes],
        "support_per_class": 5,
        "macro_f1": f1_score(query_labels, pred, average="macro"),
        "accuracy": accuracy_score(query_labels, pred),
    }


def save_figures(df, train_df, test_df, train_repr, test_repr, binary_res, multi_res):
    sns.set_theme(style="whitegrid")

    attack_counts = df["attack"].value_counts().sort_index()
    plt.figure(figsize=(9, 4))
    sns.barplot(x=attack_counts.index.astype(str), y=attack_counts.values, color="#4C72B0")
    plt.title("Attack Label Distribution")
    plt.xlabel("Attack Label")
    plt.ylabel("Flow Count")
    plt.tight_layout()
    plt.savefig(IMAGES / "attack_distribution.png", dpi=200)
    plt.close()

    sample_n = min(3000, len(test_repr))
    sample_idx = np.linspace(0, len(test_repr) - 1, sample_n, dtype=int)
    emb2 = PCA(n_components=2, random_state=42).fit_transform(test_repr[sample_idx])
    plot_df = pd.DataFrame({
        "x": emb2[:, 0],
        "y": emb2[:, 1],
        "attack": test_df.iloc[sample_idx]["attack"].astype(str).to_numpy(),
    })
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="attack", s=18, linewidth=0, alpha=0.8, palette="tab10")
    plt.title("Disentangled Representation of Test Flows")
    plt.tight_layout()
    plt.savefig(IMAGES / "representation_tsne.png", dpi=200)
    plt.close()

    cm = confusion_matrix(multi_res["y_test"], multi_res["y_pred"], labels=multi_res["labels"], normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="mako", ax=ax, xticklabels=multi_res["labels"], yticklabels=multi_res["labels"])
    ax.set_title("Normalized Multi-class Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(IMAGES / "multiclass_confusion.png", dpi=200)
    plt.close()

    comp = pd.DataFrame([
        {"task": "Binary Macro-F1", "model": "Baseline", "value": binary_res["baseline"]["macro_f1"]},
        {"task": "Binary Macro-F1", "model": "DIDS-MFL Local", "value": binary_res["proposed"]["macro_f1"]},
        {"task": "Multi Macro-F1", "model": "Baseline", "value": multi_res["baseline"]["macro_f1"]},
        {"task": "Multi Macro-F1", "model": "DIDS-MFL Local", "value": multi_res["proposed"]["macro_f1"]},
    ])
    plt.figure(figsize=(7, 4))
    sns.barplot(data=comp, x="task", y="value", hue="model", palette=["#9ecae1", "#08519c"])
    plt.ylim(0, 1.0)
    plt.title("Baseline vs Proposed Macro-F1")
    plt.tight_layout()
    plt.savefig(IMAGES / "performance_comparison.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()
    data = load_temporal_data()
    df = build_dataframe(data)
    df = add_temporal_topology_features(df)

    split_point = np.quantile(df["t"], 0.7)
    train_df = df[df["t"] <= split_point].copy()
    test_df = df[df["t"] > split_point].copy()

    train_cap = min(len(train_df), 60000)
    test_cap = min(len(test_df), 30000)
    train_df = train_df.sample(train_cap, random_state=42).sort_values("t").copy()
    test_df = test_df.sample(test_cap, random_state=42).sort_values("t").copy()

    groups = build_disentangled_views(df)
    train_repr, test_repr, view_dims = make_representation(train_df, test_df, groups)

    binary_res = evaluate_binary(train_df, test_df, train_repr, test_repr)
    multi_res = evaluate_multiclass(train_df, test_df, train_repr, test_repr)
    unknown_res = evaluate_unknown_attack(df, groups)
    few_shot_res = evaluate_few_shot(train_df, test_df, train_repr, test_repr)

    dataset_summary = {
        "num_flows": int(len(df)),
        "num_features": int(len([c for c in df.columns if c.startswith("f")])),
        "num_nodes_est": int(max(df["src"].max(), df["dst"].max()) + 1),
        "time_range": [int(df["t"].min()), int(df["t"].max())],
        "binary_label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "attack_counts": {str(k): int(v) for k, v in df["attack"].value_counts().sort_index().items()},
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "representation_dim": int(train_repr.shape[1]),
        "view_dims": view_dims,
    }

    results = {
        "dataset_summary": dataset_summary,
        "binary": binary_res,
        "multiclass": {
            "baseline": multi_res["baseline"],
            "proposed": multi_res["proposed"],
            "report": multi_res["report"],
        },
        "unknown_attack": unknown_res,
        "few_shot": few_shot_res,
    }

    save_figures(df, train_df, test_df, train_repr, test_repr, binary_res, multi_res)

    (OUTPUTS / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))
    (OUTPUTS / "results.json").write_text(json.dumps(results, indent=2))
    (OUTPUTS / "method_notes.md").write_text(
        "\n".join(
            [
                "# Local DIDS-MFL Approximation",
                "",
                "- Statistical disentanglement: split 40 flow features into three PCA-compressed subspaces.",
                "- Dynamic/topological context: previous source/destination/pair counts, cyclic time encoding, and rolling source/destination intensity.",
                "- Multi-scale fusion: concatenate all subspaces into a joint representation.",
                "- Few-shot inference: dual-similarity prototype scoring with cosine and Euclidean similarity.",
                "- Unknown attack evaluation: hold out the rarest malicious class during training and threshold the maliciousness score on benign calibration flows.",
            ]
        )
    )

    print("analysis complete")


if __name__ == "__main__":
    main()
