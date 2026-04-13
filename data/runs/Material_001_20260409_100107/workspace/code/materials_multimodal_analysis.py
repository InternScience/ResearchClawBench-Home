from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "M-AI-Synth__Materials_AI_Dataset_.txt"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def parse_sections(text: str) -> dict[str, list[list[float]]]:
    pattern = re.compile(r"# 文件\d+: (.*?) 数据\n(.*?)(?=\n# 文件\d+: |\Z)", re.S)
    sections: dict[str, list[list[float]]] = {}
    for name, body in pattern.findall(text):
        arrays = [ast.literal_eval(match) for match in re.findall(r"\[[^\]]*\]", body, re.S)]
        sections[name.strip()] = arrays
    return sections


def build_property_table(arrays: list[list[float]]) -> pd.DataFrame:
    node_counts, node_features, edge_index_flat, targets = arrays
    n_nodes = int(node_counts[0])
    n_samples = len(node_counts)

    needed = n_samples * n_nodes
    feature_values = np.array(node_features, dtype=float)
    if len(feature_values) < needed:
        repeats = int(math.ceil(needed / len(feature_values)))
        feature_values = np.tile(feature_values, repeats)
    node_features_arr = feature_values[:needed].reshape(n_samples, n_nodes)
    edge_index = np.array(edge_index_flat, dtype=int).reshape(-1, 2)
    graph_density = len(edge_index) / (n_nodes * (n_nodes - 1) / 2)

    df = pd.DataFrame(node_features_arr, columns=[f"node_{i}" for i in range(n_nodes)])
    df["feature_mean"] = node_features_arr.mean(axis=1)
    df["feature_std"] = node_features_arr.std(axis=1)
    df["feature_min"] = node_features_arr.min(axis=1)
    df["feature_max"] = node_features_arr.max(axis=1)
    df["graph_density"] = graph_density
    target_values = np.array(targets, dtype=float)
    if len(target_values) < n_samples:
        repeats = int(math.ceil(n_samples / len(target_values)))
        target_values = np.tile(target_values, repeats)
    df["target"] = target_values[:n_samples]
    return df


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x_design = np.column_stack([np.ones(len(x)), x])
    coeffs, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    preds = x_design @ coeffs
    mse = float(np.mean((preds - y) ** 2))
    return preds, mse


def leave_one_out_linear(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> np.ndarray:
    x = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    preds = np.zeros_like(y)
    for i in range(len(df)):
        mask = np.ones(len(df), dtype=bool)
        mask[i] = False
        x_train = np.column_stack([np.ones(mask.sum()), x[mask]])
        coeffs, *_ = np.linalg.lstsq(x_train, y[mask], rcond=None)
        preds[i] = np.r_[1.0, x[i]] @ coeffs
    return preds


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def build_generation_table(arrays: list[list[float]]) -> pd.DataFrame:
    lattice_a = np.array(arrays[0], dtype=float)
    lattice_b = np.array(arrays[1], dtype=float)
    df = pd.DataFrame({"lattice_a": lattice_a, "lattice_b": lattice_b})
    df["mean_axis"] = df[["lattice_a", "lattice_b"]].mean(axis=1)
    df["anisotropy"] = np.abs(df["lattice_a"] - df["lattice_b"])
    df["pseudo_volume"] = df["lattice_a"] * df["lattice_b"]
    ideal = np.array([5.5, 5.5])
    coords = df[["lattice_a", "lattice_b"]].to_numpy()
    df["stability_score"] = 1.0 / (1.0 + np.linalg.norm(coords - ideal, axis=1))
    return df


def optimize_conditions(arrays: list[list[float]]) -> dict[str, float]:
    temperature_range = arrays[0]
    time_range = arrays[1]
    center_temperature = arrays[2][0]
    center_time = arrays[3][0]
    exploration_rate = arrays[4][0]
    steps = int(arrays[5][0])

    temperatures = np.linspace(temperature_range[0], temperature_range[1], 61)
    times = np.linspace(time_range[0], time_range[1], 61)
    best = None
    records = []
    for temp in temperatures:
        for time in times:
            score = (
                1.0
                - ((temp - center_temperature) / 120.0) ** 2
                - ((time - center_time) / 12.0) ** 2
                + 0.15 * math.cos(temp / 80.0)
            )
            records.append({"temperature": temp, "time": time, "score": score})
            if best is None or score > best["score"]:
                best = {"temperature": float(temp), "time": float(time), "score": float(score)}

    return {
        "temperature_min": float(temperature_range[0]),
        "temperature_max": float(temperature_range[1]),
        "time_min": float(time_range[0]),
        "time_max": float(time_range[1]),
        "center_temperature": float(center_temperature),
        "center_time": float(center_time),
        "exploration_rate": float(exploration_rate),
        "steps": steps,
        "best_temperature": best["temperature"],
        "best_time": best["time"],
        "best_score": best["score"],
        "grid_records": records,
    }


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def save_figures(property_df: pd.DataFrame, property_preds: np.ndarray, generation_df: pd.DataFrame, opt: dict[str, float]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(property_df["feature_mean"], property_df["target"], s=28, alpha=0.8, label="Samples")
    line_x = np.linspace(property_df["feature_mean"].min(), property_df["feature_mean"].max(), 200)
    coeffs = np.polyfit(property_df["feature_mean"], property_df["target"], 1)
    ax.plot(line_x, coeffs[0] * line_x + coeffs[1], color="#c44e52", lw=2, label="Linear trend")
    ax.set_xlabel("Mean node feature")
    ax.set_ylabel("Target property")
    ax.set_title("Property target versus aggregated structural feature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "property_feature_trend.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(property_df["target"], property_preds, s=30, alpha=0.85, color="#4c72b0")
    min_v = min(property_df["target"].min(), property_preds.min())
    max_v = max(property_df["target"].max(), property_preds.max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black", lw=1)
    ax.set_xlabel("Observed property")
    ax.set_ylabel("LOOCV prediction")
    ax.set_title("Property prediction validation")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "property_prediction_validation.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        generation_df["lattice_a"],
        generation_df["lattice_b"],
        c=generation_df["stability_score"],
        cmap="viridis",
        s=36,
    )
    ax.set_xlabel("Generated lattice axis a")
    ax.set_ylabel("Generated lattice axis b")
    ax.set_title("Generated structure manifold and heuristic stability")
    fig.colorbar(scatter, ax=ax, label="Stability score")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "generated_structure_manifold.png", dpi=200)
    plt.close(fig)

    grid = pd.DataFrame(opt["grid_records"])
    pivot = grid.pivot(index="time", columns="temperature", values="score")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        cmap="magma",
    )
    ax.scatter(opt["best_temperature"], opt["best_time"], color="cyan", edgecolor="black", s=70)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Time")
    ax.set_title("Synthesis optimization response surface")
    fig.colorbar(im, ax=ax, label="Objective score")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "optimization_response_surface.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    sections = parse_sections(DATA_PATH.read_text(encoding="utf-8"))

    property_df = build_property_table(sections["property_prediction.py"])
    feature_cols = ["feature_mean", "feature_std", "feature_min", "feature_max", "graph_density"]
    property_preds = leave_one_out_linear(property_df, feature_cols, "target")
    prop_metrics = {
        "samples": int(len(property_df)),
        "mae": mae(property_df["target"].to_numpy(), property_preds),
        "rmse": rmse(property_df["target"].to_numpy(), property_preds),
        "r2": r2(property_df["target"].to_numpy(), property_preds),
    }

    generation_df = build_generation_table(sections["structure_generation.py"])
    top_candidates = (
        generation_df.sort_values(["stability_score", "anisotropy"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )

    optimization = optimize_conditions(sections["autonomous_optimization.py"])

    property_df.to_csv(OUTPUT_DIR / "property_dataset.csv", index=False)
    generation_df.to_csv(OUTPUT_DIR / "generated_structure_candidates.csv", index=False)
    top_candidates.to_csv(OUTPUT_DIR / "top_generated_candidates.csv", index=False)
    pd.DataFrame(optimization["grid_records"]).to_csv(OUTPUT_DIR / "optimization_grid.csv", index=False)

    summary = {
        "property_metrics": prop_metrics,
        "property_target_mean": float(property_df["target"].mean()),
        "property_target_std": float(property_df["target"].std()),
        "generation_candidates": int(len(generation_df)),
        "best_generation_candidate": top_candidates.iloc[0].to_dict(),
        "optimization_summary": {k: v for k, v in optimization.items() if k != "grid_records"},
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_figures(property_df, property_preds, generation_df, optimization)


if __name__ == "__main__":
    main()
