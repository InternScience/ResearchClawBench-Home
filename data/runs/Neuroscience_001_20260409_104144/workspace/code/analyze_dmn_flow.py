#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "flow" / "0000"
RELATED_WORK = ROOT / "related_work"
OUTPUTS = ROOT / "outputs"
REPORT_IMAGES = ROOT / "report" / "images"


class DummyClass:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        self._state = state
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            self._tuple_state = state
        else:
            self._raw_state = state


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # The clustering pickles reference flyvis classes that are not shipped here.
        # Returning a dummy class is enough because the stored state is still accessible.
        return DummyClass


def safe_load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return SafeUnpickler(handle).load()


def ensure_dirs() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)


def list_runs() -> list[Path]:
    return sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit()])


def read_validation_loss(run_dir: Path) -> float:
    with h5py.File(run_dir / "validation_loss.h5", "r") as handle:
        return float(handle["data"][()])


def load_checkpoint(run_dir: Path) -> dict[str, Any]:
    return torch.load(run_dir / "best_chkpt", map_location="cpu")


def summarize_literature() -> pd.DataFrame:
    rows = []
    for pdf_path in sorted(RELATED_WORK.glob("*.pdf")):
        doc = fitz.open(pdf_path)
        text = " ".join(page.get_text() for page in doc[:2]).replace("\x00", " ")
        rows.append(
            {
                "paper": pdf_path.name,
                "title_snippet": " ".join(text.split())[:240],
                "n_pages": len(doc),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS / "literature_overview.csv", index=False)
    return df


def compute_ensemble_tables(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loss_rows = []
    checkpoints = []
    for run_dir in runs:
        ckpt = load_checkpoint(run_dir)
        checkpoints.append(ckpt)
        loss_rows.append(
            {
                "run": run_dir.name,
                "validation_loss": read_validation_loss(run_dir),
            }
        )
    loss_df = pd.DataFrame(loss_rows).sort_values("validation_loss", ignore_index=True)

    network_keys = list(checkpoints[0]["network"].keys())
    param_frames = []
    summary_rows = []
    for key in network_keys:
        arr = np.stack([ckpt["network"][key].detach().cpu().numpy().ravel() for ckpt in checkpoints])
        entry = {
            "parameter": key,
            "n_values": arr.shape[1],
            "global_mean": float(arr.mean()),
            "global_std": float(arr.std()),
            "across_run_mean_std": float(arr.std(axis=0).mean()),
            "across_run_max_std": float(arr.std(axis=0).max()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        summary_rows.append(entry)
        param_frames.append(
            pd.DataFrame(
                {
                    "parameter": key,
                    "index": np.tile(np.arange(arr.shape[1]), arr.shape[0]),
                    "run": np.repeat([r.name for r in runs], arr.shape[1]),
                    "value": arr.reshape(-1),
                }
            )
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("across_run_mean_std", ascending=False)
    long_df = pd.concat(param_frames, ignore_index=True)

    loss_df.to_csv(OUTPUTS / "ensemble_validation_losses.csv", index=False)
    summary_df.to_csv(OUTPUTS / "parameter_summary.csv", index=False)
    long_df.to_csv(OUTPUTS / "parameter_values_long.csv", index=False)
    return loss_df, summary_df, long_df


def object_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {"repr": repr(obj)}


def unwrap_dummy(obj: Any) -> Any:
    if isinstance(obj, DummyClass):
        if hasattr(obj, "_state") and isinstance(obj._state, dict):
            if "data" in obj._state:
                return unwrap_dummy(obj._state["data"])
            return {k: unwrap_dummy(v) for k, v in obj._state.items()}
        if hasattr(obj, "_tuple_state"):
            return tuple(unwrap_dummy(v) for v in obj._tuple_state)
        return obj
    if isinstance(obj, dict):
        return {k: unwrap_dummy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(unwrap_dummy(v) for v in obj)
    return obj


def clustering_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cluster_rows = []
    for pickle_path in sorted((DATA_ROOT / "umap_and_clustering").glob("*.pickle")):
        obj = safe_load_pickle(pickle_path)
        data = unwrap_dummy(object_to_dict(obj))
        row = {"cell_type": pickle_path.stem}
        for key, value in data.items():
            if hasattr(value, "shape"):
                row[f"{key}_shape"] = json.dumps(list(value.shape))
            elif isinstance(value, (list, tuple)):
                row[f"{key}_len"] = len(value)
            elif np.isscalar(value):
                row[key] = value.item() if hasattr(value, "item") else value
            else:
                row[key] = type(value).__name__
        rows.append(row)

        labels = None
        embeddings = None
        for key, value in data.items():
            if hasattr(value, "shape") and len(value.shape) == 1 and value.shape[0] > 2:
                if np.issubdtype(np.asarray(value).dtype, np.integer):
                    labels = np.asarray(value)
            if hasattr(value, "shape") and len(value.shape) == 2 and value.shape[1] in (2, 3):
                embeddings = np.asarray(value)
        if labels is not None:
            counts = pd.Series(labels).value_counts().sort_index()
            for cluster_id, count in counts.items():
                cluster_rows.append(
                    {
                        "cell_type": pickle_path.stem,
                        "cluster_id": int(cluster_id),
                        "count": int(count),
                        "n_clusters": int(counts.size),
                        "has_embedding": embeddings is not None,
                    }
                )

    overview_df = pd.DataFrame(rows).sort_values("cell_type", ignore_index=True)
    cluster_df = pd.DataFrame(cluster_rows)
    if not cluster_df.empty:
        cluster_df = cluster_df.sort_values(["cell_type", "cluster_id"], ignore_index=True)
    overview_df.to_csv(OUTPUTS / "clustering_overview.csv", index=False)
    cluster_df.to_csv(OUTPUTS / "clustering_cluster_counts.csv", index=False)
    return overview_df, cluster_df


def plot_loss_distribution(loss_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(loss_df["validation_loss"], bins=15, kde=True, ax=ax, color="#2b6cb0")
    ax.axvline(loss_df["validation_loss"].mean(), color="#c53030", linestyle="--", linewidth=2, label="ensemble mean")
    ax.set_xlabel("Validation loss")
    ax.set_ylabel("Number of models")
    ax.set_title("Distribution of validation loss across 50 DMN models")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "ensemble_validation_loss.png", dpi=200)
    plt.close(fig)


def plot_parameter_variability(summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = summary_df.sort_values("across_run_mean_std", ascending=True)
    ax.barh(ordered["parameter"], ordered["across_run_mean_std"], color="#dd6b20")
    ax.set_xlabel("Mean std across runs")
    ax.set_ylabel("Parameter group")
    ax.set_title("Learned-parameter variability across the DMN ensemble")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "parameter_variability.png", dpi=200)
    plt.close(fig)


def plot_cluster_counts(cluster_df: pd.DataFrame) -> None:
    if cluster_df.empty:
        return
    per_type = (
        cluster_df.groupby("cell_type")
        .agg(total_points=("count", "sum"), n_clusters=("n_clusters", "max"))
        .sort_values("total_points", ascending=False)
        .head(20)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=per_type,
        x="n_clusters",
        y="total_points",
        size="total_points",
        hue="n_clusters",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    for _, row in per_type.iterrows():
        ax.text(row["n_clusters"] + 0.03, row["total_points"], row["cell_type"], fontsize=8)
    ax.set_xlabel("Number of discovered clusters")
    ax.set_ylabel("Samples per cell type")
    ax.set_title("Top cell types by clustering complexity and sample count")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "cluster_complexity.png", dpi=200)
    plt.close(fig)


def plot_clustering_metadata(overview_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    n_init_counts = overview_df["n_init"].value_counts().sort_index()
    axes[0].bar(n_init_counts.index.astype(str), n_init_counts.values, color="#2f855a")
    axes[0].set_title("Gaussian-mixture restarts per cell type")
    axes[0].set_xlabel("n_init")
    axes[0].set_ylabel("Cell types")

    cluster_range_counts = overview_df["range_n_clusters_len"].value_counts().sort_index()
    axes[1].bar(cluster_range_counts.index.astype(str), cluster_range_counts.values, color="#805ad5")
    axes[1].set_title("Candidate cluster counts tested")
    axes[1].set_xlabel("Number of candidate cluster counts")
    axes[1].set_ylabel("Cell types")

    fig.suptitle("Clustering protocol metadata across 65 cell types")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "clustering_metadata.png", dpi=200)
    plt.close(fig)


def build_summary_json(loss_df: pd.DataFrame, summary_df: pd.DataFrame, cluster_df: pd.DataFrame) -> None:
    out = {
        "n_models": int(loss_df.shape[0]),
        "validation_loss_mean": float(loss_df["validation_loss"].mean()),
        "validation_loss_std": float(loss_df["validation_loss"].std()),
        "best_validation_loss": float(loss_df["validation_loss"].min()),
        "worst_validation_loss": float(loss_df["validation_loss"].max()),
        "most_variable_parameter": str(summary_df.iloc[0]["parameter"]),
        "least_variable_parameter": str(summary_df.iloc[-1]["parameter"]),
        "n_clustered_cell_types": int(cluster_df["cell_type"].nunique()) if not cluster_df.empty else 0,
    }
    (OUTPUTS / "analysis_summary.json").write_text(json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    ensure_dirs()
    runs = list_runs()
    literature_df = summarize_literature()
    loss_df, summary_df, _ = compute_ensemble_tables(runs)
    clustering_overview_df, cluster_df = clustering_tables()
    plot_loss_distribution(loss_df)
    plot_parameter_variability(summary_df)
    plot_cluster_counts(cluster_df)
    plot_clustering_metadata(clustering_overview_df)
    build_summary_json(loss_df, summary_df, cluster_df)

    print("Literature papers:", len(literature_df))
    print("Models analyzed:", len(runs))
    print("Cluster overview rows:", len(clustering_overview_df))
    print("Figures written to:", REPORT_IMAGES)


if __name__ == "__main__":
    main()
