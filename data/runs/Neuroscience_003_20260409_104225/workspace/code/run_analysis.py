#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import trustworthiness
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "adata_RPE.h5ad"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


@dataclass
class EvaluationResult:
    method: str
    n_features: int
    pseudotime_age_spearman: float
    knn_jaccard_vs_all: float
    age_rmse: float
    state_auc: float
    phase_macro_auc: float
    embedding_trustworthiness: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    adata = ad.read_h5ad(DATA_PATH)
    x = np.asarray(adata.X, dtype=float)
    obs = adata.obs.copy()
    obs["state"] = obs["state"].astype(str).replace("nan", "unassigned")
    obs["phase"] = obs["phase"].astype(str)
    var_names = pd.Index(adata.var_names.astype(str), name="feature")
    feature_df = pd.DataFrame(x, columns=var_names)
    return obs, feature_df, x


def rolling_dynamic_score(values: np.ndarray, age_order: np.ndarray, window: int) -> float:
    ordered = values[age_order]
    smooth = pd.Series(ordered).rolling(window=window, center=True, min_periods=window // 2).mean()
    residual = ordered - smooth.to_numpy()
    signal_var = np.nanvar(smooth.to_numpy())
    noise_var = np.nanvar(residual)
    return float(signal_var / (noise_var + 1e-8))


def score_features(obs: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    age = obs["annotated_age"].to_numpy()
    batch = obs["batch"].astype(str).to_numpy()
    age_order = np.argsort(age)
    window = max(25, len(age) // 40)
    rows = []
    for feature in feature_df.columns:
        values = feature_df[feature].to_numpy()
        age_rho = float(spearmanr(values, age).statistic)
        dynamic = rolling_dynamic_score(values, age_order, window)
        batch_gap = 0.0
        unique_batches = np.unique(batch)
        if len(unique_batches) == 2:
            v0 = values[batch == unique_batches[0]]
            v1 = values[batch == unique_batches[1]]
            pooled = np.sqrt((np.var(v0) + np.var(v1)) / 2.0 + 1e-8)
            batch_gap = float(abs(v0.mean() - v1.mean()) / pooled)
        variance = float(np.var(values))
        score = dynamic * abs(age_rho) / (1.0 + batch_gap)
        rows.append(
            {
                "feature": feature,
                "dynamic_score": dynamic,
                "age_spearman": age_rho,
                "batch_effect": batch_gap,
                "variance": variance,
                "score": score,
            }
        )
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def build_embedding(feature_df: pd.DataFrame, features: list[str], n_components: int = 8) -> np.ndarray:
    x = feature_df[features].to_numpy()
    x = StandardScaler().fit_transform(x)
    n_components = min(n_components, x.shape[1], x.shape[0] - 1)
    if n_components < 2:
        n_components = min(2, x.shape[1])
    return PCA(n_components=n_components, random_state=0).fit_transform(x)


def pseudotime_from_embedding(embedding: np.ndarray, root_idx: int) -> np.ndarray:
    pseudotime = embedding[:, 0].copy()
    if pseudotime[root_idx] > np.median(pseudotime):
        pseudotime *= -1
    pseudotime -= pseudotime.min()
    pseudotime /= pseudotime.max() + 1e-8
    return pseudotime


def neighborhood_jaccard(x_ref: np.ndarray, x_query: np.ndarray, k: int = 15) -> float:
    nn_ref = NearestNeighbors(n_neighbors=k + 1).fit(x_ref)
    nn_q = NearestNeighbors(n_neighbors=k + 1).fit(x_query)
    ref_idx = nn_ref.kneighbors(return_distance=False)[:, 1:]
    q_idx = nn_q.kneighbors(return_distance=False)[:, 1:]
    scores = []
    for a, b in zip(ref_idx, q_idx):
        sa, sb = set(a.tolist()), set(b.tolist())
        scores.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(scores))


def phase_auc(embedding: np.ndarray, phases: np.ndarray) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=4000, multi_class="multinomial")
    prob = cross_val_predict(clf, embedding, phases, cv=skf, method="predict_proba")
    classes = np.unique(phases)
    y_true = pd.get_dummies(phases)[classes].to_numpy()
    return float(roc_auc_score(y_true, prob, average="macro", multi_class="ovr"))


def state_auc(embedding: np.ndarray, states: np.ndarray) -> float:
    mask = states != "unassigned"
    y = (states[mask] == "cycling").astype(int)
    x = embedding[mask]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000)
    prob = cross_val_predict(clf, x, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def age_rmse(embedding: np.ndarray, age: np.ndarray) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    pred = cross_val_predict(Ridge(alpha=1.0), embedding, age, cv=kf)
    return float(np.sqrt(mean_squared_error(age, pred)))


def evaluate_methods(obs: pd.DataFrame, feature_df: pd.DataFrame, rankings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    all_features = feature_df.columns.tolist()
    var_rank = rankings.sort_values("variance", ascending=False)["feature"].tolist()
    age_rank = rankings.reindex(rankings["age_spearman"].abs().sort_values(ascending=False).index)["feature"].tolist()
    dynamic_rank = rankings["feature"].tolist()

    selected = {
        "all_features": all_features,
        "variance_top30": var_rank[:30],
        "agecorr_top30": age_rank[:30],
        "dynamic_top30": dynamic_rank[:30],
    }

    base_embedding = build_embedding(feature_df, all_features)
    age = obs["annotated_age"].to_numpy()
    root_idx = int(np.argmin(age))

    embeddings: dict[str, np.ndarray] = {"all_features": base_embedding}
    results: list[EvaluationResult] = []
    for method, features in selected.items():
        emb = embeddings.get(method)
        if emb is None:
            emb = build_embedding(feature_df, features)
            embeddings[method] = emb
        pseudo = pseudotime_from_embedding(emb, root_idx=root_idx)
        result = EvaluationResult(
            method=method,
            n_features=len(features),
            pseudotime_age_spearman=float(spearmanr(pseudo, age).statistic),
            knn_jaccard_vs_all=neighborhood_jaccard(base_embedding, emb),
            age_rmse=age_rmse(emb, age),
            state_auc=state_auc(emb, obs["state"].to_numpy()),
            phase_macro_auc=phase_auc(emb, obs["phase"].to_numpy()),
            embedding_trustworthiness=float(trustworthiness(StandardScaler().fit_transform(feature_df[features].to_numpy()), emb, n_neighbors=15)),
        )
        results.append(result)

    return pd.DataFrame([r.__dict__ for r in results]), embeddings


def save_feature_rankings(rankings: pd.DataFrame) -> None:
    rankings.to_csv(OUTPUT_DIR / "feature_rankings.csv", index=False)
    rankings.head(30).to_csv(OUTPUT_DIR / "selected_dynamic_features.csv", index=False)


def save_summary(obs: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    summary = {
        "n_cells": int(len(obs)),
        "n_features": int(feature_df.shape[1]),
        "age_min": float(obs["annotated_age"].min()),
        "age_median": float(obs["annotated_age"].median()),
        "age_max": float(obs["annotated_age"].max()),
        "phase_counts": obs["phase"].value_counts().to_dict(),
        "state_counts": obs["state"].value_counts().to_dict(),
        "batch_counts": obs["batch"].astype(str).value_counts().to_dict(),
    }
    with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_data_overview(obs: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(obs["annotated_age"], bins=30, color="#3d6ea8", ax=axes[0])
    axes[0].set_title("Annotated Age Distribution")
    axes[0].set_xlabel("Annotated age")

    phase_counts = obs["phase"].value_counts().reindex(["G0", "G1", "S", "G2"]).fillna(0)
    sns.barplot(x=phase_counts.index, y=phase_counts.values, color="#d27d2d", ax=axes[1])
    axes[1].set_title("Cell-Cycle Phase Counts")
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel("Cells")

    state_counts = obs["state"].value_counts()
    sns.barplot(x=state_counts.index, y=state_counts.values, palette=["#2a9d8f", "#e76f51", "#8d99ae"], ax=axes[2])
    axes[2].set_title("State Labels")
    axes[2].set_xlabel("State")
    axes[2].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "data_overview.png", dpi=220)
    plt.close(fig)


def plot_feature_scores(rankings: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.scatterplot(
        data=rankings,
        x="batch_effect",
        y="dynamic_score",
        hue="age_spearman",
        palette="coolwarm",
        ax=axes[0],
        s=28,
    )
    axes[0].set_title("Dynamic Score Versus Batch Effect")
    axes[0].set_xlabel("Standardized batch gap")
    axes[0].set_ylabel("Dynamic score")

    top = rankings.head(15).iloc[::-1]
    sns.barplot(data=top, y="feature", x="score", color="#457b9d", ax=axes[1])
    axes[1].set_title("Top Dynamic Features")
    axes[1].set_xlabel("Composite selection score")
    axes[1].set_ylabel("")
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "feature_selection_summary.png", dpi=220)
    plt.close(fig)


def plot_embeddings(obs: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> None:
    methods = ["all_features", "variance_top30", "agecorr_top30", "dynamic_top30"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    age = obs["annotated_age"].to_numpy()
    phase = obs["phase"].to_numpy()
    phase_palette = {"G0": "#6c757d", "G1": "#1d3557", "S": "#2a9d8f", "G2": "#e76f51"}

    for idx, method in enumerate(methods):
        emb = embeddings[method]
        ax = axes[0, idx]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=age, cmap="viridis", s=12, linewidths=0)
        ax.set_title(f"{method}: age")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if idx == len(methods) - 1:
            plt.colorbar(sc, ax=ax, fraction=0.046)

        ax2 = axes[1, idx]
        for ph in ["G0", "G1", "S", "G2"]:
            mask = phase == ph
            ax2.scatter(emb[mask, 0], emb[mask, 1], s=10, linewidths=0, label=ph, color=phase_palette[ph], alpha=0.85)
        ax2.set_title(f"{method}: phase")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
    handles, labels = axes[1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(REPORT_IMG_DIR / "trajectory_embeddings.png", dpi=220)
    plt.close(fig)


def plot_method_comparison(results: pd.DataFrame) -> None:
    melted = results.melt(
        id_vars=["method", "n_features"],
        value_vars=[
            "pseudotime_age_spearman",
            "knn_jaccard_vs_all",
            "state_auc",
            "phase_macro_auc",
            "embedding_trustworthiness",
        ],
        var_name="metric",
        value_name="value",
    )
    metric_names = {
        "pseudotime_age_spearman": "Pseudotime-age rho",
        "knn_jaccard_vs_all": "kNN overlap",
        "state_auc": "Cycling state AUC",
        "phase_macro_auc": "Phase macro AUC",
        "embedding_trustworthiness": "Trustworthiness",
    }
    melted["metric"] = melted["metric"].map(metric_names)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="method", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=15)
    ax.set_title("Trajectory Preservation Benchmark")
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "method_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    obs, feature_df, _ = load_data()
    save_summary(obs, feature_df)
    rankings = score_features(obs, feature_df)
    save_feature_rankings(rankings)
    results, embeddings = evaluate_methods(obs, feature_df, rankings)
    results = results.sort_values("method").reset_index(drop=True)
    results.to_csv(OUTPUT_DIR / "method_comparison.csv", index=False)

    dynamic_features = rankings.head(30)["feature"].tolist()
    dynamic_embedding = embeddings["dynamic_top30"]
    pseudo = pseudotime_from_embedding(dynamic_embedding, root_idx=int(np.argmin(obs["annotated_age"].to_numpy())))
    cell_table = obs.copy()
    cell_table["pseudotime_dynamic"] = pseudo
    cell_table.to_csv(OUTPUT_DIR / "cell_level_results.csv", index=True)

    plot_data_overview(obs)
    plot_feature_scores(rankings)
    plot_embeddings(obs, embeddings)
    plot_method_comparison(results)

    with open(OUTPUT_DIR / "analysis_notes.txt", "w", encoding="utf-8") as f:
        f.write("Dynamic feature subset (top 30):\n")
        for feat in dynamic_features:
            f.write(f"- {feat}\n")
        f.write("\nMethod comparison:\n")
        f.write(results.to_string(index=False))
        f.write("\n")


if __name__ == "__main__":
    main()
