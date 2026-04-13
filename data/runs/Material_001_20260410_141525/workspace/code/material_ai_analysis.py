import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

sns.set_theme(style="whitegrid", context="talk")

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "M-AI-Synth__Materials_AI_Dataset_.txt"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)


def parse_dataset(path: Path):
    text = path.read_text(encoding="utf-8")
    blocks = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# 文件"):
            current = line
            blocks[current] = []
        elif line.startswith("["):
            blocks[current].append(ast.literal_eval(line))
    return blocks


def summarize_array(arr):
    a = np.asarray(arr, dtype=float)
    return {
        "length": int(len(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def build_property_dataset(block):
    composition = np.asarray(block[0], dtype=float)
    spectral = np.asarray(block[1], dtype=float)
    edges = np.asarray(block[2], dtype=int)
    target = np.asarray(block[3], dtype=float)

    n = len(target)
    spectral = spectral[:n]
    composition = composition[:n]

    if edges.ndim == 1:
        if len(edges) % 2 != 0:
            edges = edges[:-1]
        edges = edges.reshape(-1, 2)

    edge_mean = edges.mean(axis=1).mean()
    edge_span = (edges[:, 1] - edges[:, 0]).mean()
    edge_count = len(edges)

    df = pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "composition_signal": composition,
            "spectral_signal": spectral,
            "spectral_sq": spectral**2,
            "spectral_sin": np.sin(spectral),
            "graph_edge_mean": edge_mean,
            "graph_edge_span": edge_span,
            "graph_edge_count": edge_count,
            "target_property": target,
        }
    )
    return df, {"n_edges": int(edge_count), "edge_mean": float(edge_mean), "edge_span": float(edge_span)}


def run_property_prediction(df):
    features = [
        "composition_signal",
        "spectral_signal",
        "spectral_sq",
        "spectral_sin",
        "graph_edge_mean",
        "graph_edge_span",
        "graph_edge_count",
    ]
    X = df[features]
    y = df["target_property"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    baseline = LinearRegression()
    model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=5)
    baseline.fit(X_train, y_train)
    model.fit(X_train, y_train)

    pred_base = baseline.predict(X_test)
    pred = model.predict(X_test)

    metrics = {
        "linear": {
            "mae": float(mean_absolute_error(y_test, pred_base)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_base))),
            "r2": float(r2_score(y_test, pred_base)),
        },
        "random_forest": {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        },
    }

    results = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred_linear": pred_base,
            "y_pred_rf": pred,
        }
    )
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return metrics, results, importances


def run_structure_generation(block):
    a = np.asarray(block[0], dtype=float)
    b = np.asarray(block[1], dtype=float)
    X = np.column_stack([a, b])

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm.fit(X)
    samples, _ = gmm.sample(200)

    kde = KernelDensity(kernel="gaussian", bandwidth=0.08).fit(X)
    scores = kde.score_samples(X)

    real_df = pd.DataFrame(X, columns=["lattice_a", "lattice_b"])
    real_df["type"] = "real"
    gen_df = pd.DataFrame(samples, columns=["lattice_a", "lattice_b"])
    gen_df["type"] = "generated"
    combined = pd.concat([real_df, gen_df], ignore_index=True)

    summary = {
        "real_mean_a": float(real_df["lattice_a"].mean()),
        "real_mean_b": float(real_df["lattice_b"].mean()),
        "generated_mean_a": float(gen_df["lattice_a"].mean()),
        "generated_mean_b": float(gen_df["lattice_b"].mean()),
        "avg_log_density": float(scores.mean()),
    }
    return combined, summary


def objective(temp, time, init_temp, init_time):
    # Smooth toy objective for workflow validation: maximize near a plausible optimum.
    return (
        82
        + 14 * np.exp(-((temp - (init_temp + 20)) ** 2) / (2 * 55**2))
        + 9 * np.exp(-((time - (init_time + 2)) ** 2) / (2 * 4.5**2))
        - 0.00022 * (temp - 370) ** 2
        - 0.055 * (time - 22) ** 2
    )


def run_optimization(block):
    temp_bounds = np.asarray(block[0], dtype=float)
    time_bounds = np.asarray(block[1], dtype=float)
    init_temp = float(block[2][0])
    init_time = float(block[3][0])
    learning_rate = float(block[4][0])
    n_iter = int(block[5][0])

    temps = np.linspace(temp_bounds[0], temp_bounds[1], 121)
    times = np.linspace(time_bounds[0], time_bounds[1], 81)

    grid = []
    for t in temps:
        for tm in times:
            grid.append((t, tm, objective(t, tm, init_temp, init_time)))
    grid_df = pd.DataFrame(grid, columns=["temperature", "time", "score"])
    best = grid_df.loc[grid_df["score"].idxmax()].to_dict()

    current_t, current_tm = init_temp, init_time
    history = []
    for i in range(n_iter + 1):
        score = objective(current_t, current_tm, init_temp, init_time)
        history.append({"iteration": i, "temperature": current_t, "time": current_tm, "score": score})
        grad_t = (objective(current_t + 1, current_tm, init_temp, init_time) - objective(current_t - 1, current_tm, init_temp, init_time)) / 2
        grad_tm = (objective(current_t, current_tm + 0.5, init_temp, init_time) - objective(current_t, current_tm - 0.5, init_temp, init_time))
        current_t = np.clip(current_t + learning_rate * 18 * grad_t, temp_bounds[0], temp_bounds[1])
        current_tm = np.clip(current_tm + learning_rate * 5 * grad_tm, time_bounds[0], time_bounds[1])

    history_df = pd.DataFrame(history)
    summary = {
        "initial_score": float(history_df.iloc[0]["score"]),
        "final_score": float(history_df.iloc[-1]["score"]),
        "best_grid_temperature": float(best["temperature"]),
        "best_grid_time": float(best["time"]),
        "best_grid_score": float(best["score"]),
    }
    return grid_df, history_df, summary


def make_figures(prop_df, prop_results, importances, struct_combined, opt_grid, opt_history):
    # Figure 1
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(prop_df["spectral_signal"], bins=20, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Property-workflow spectral signal distribution")
    axes[0].set_xlabel("Spectral signal")
    axes[0].set_ylabel("Count")
    sns.scatterplot(data=prop_df, x="spectral_signal", y="target_property", ax=axes[1], color="#DD8452", s=60)
    axes[1].set_title("Synthetic property target vs. spectral signal")
    axes[1].set_xlabel("Spectral signal")
    axes[1].set_ylabel("Target property")
    fig.tight_layout()
    fig.savefig(IMG / "data_overview.png", dpi=220)
    plt.close(fig)

    # Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x="y_true", y="y_pred_rf", data=prop_results, ax=axes[0], color="#55A868", s=70)
    lims = [
        min(prop_results[["y_true", "y_pred_rf"]].min().min(), prop_results[["y_true", "y_pred_linear"]].min().min()),
        max(prop_results[["y_true", "y_pred_rf"]].max().max(), prop_results[["y_true", "y_pred_linear"]].max().max()),
    ]
    axes[0].plot(lims, lims, "k--", linewidth=1)
    axes[0].set_title("Random forest: predicted vs true property")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")
    importances.sort_values().plot(kind="barh", ax=axes[1], color="#C44E52")
    axes[1].set_title("Feature importance")
    axes[1].set_xlabel("Importance")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(IMG / "property_prediction_results.png", dpi=220)
    plt.close(fig)

    # Figure 3
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.kdeplot(data=struct_combined, x="lattice_a", hue="type", fill=True, common_norm=False, ax=axes[0])
    axes[0].set_title("Distribution of lattice parameter a")
    axes[0].set_xlabel("Lattice a")
    axes[0].set_ylabel("Density")
    sns.scatterplot(data=struct_combined.sample(min(250, len(struct_combined)), random_state=42), x="lattice_a", y="lattice_b", hue="type", ax=axes[1], s=60)
    axes[1].set_title("Real vs generated structure manifold")
    axes[1].set_xlabel("Lattice a")
    axes[1].set_ylabel("Lattice b")
    fig.tight_layout()
    fig.savefig(IMG / "structure_generation_results.png", dpi=220)
    plt.close(fig)

    # Figure 4
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pivot = opt_grid.pivot(index="time", columns="temperature", values="score")
    sns.heatmap(pivot, ax=axes[0], cmap="viridis")
    axes[0].set_title("Optimization landscape")
    axes[0].set_xlabel("Temperature grid")
    axes[0].set_ylabel("Time grid")
    sns.lineplot(data=opt_history, x="iteration", y="score", marker="o", ax=axes[1], color="#8172B2")
    axes[1].set_title("Autonomous optimization trajectory")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Objective score")
    fig.tight_layout()
    fig.savefig(IMG / "optimization_results.png", dpi=220)
    plt.close(fig)


def main():
    blocks = parse_dataset(DATA_FILE)

    property_block = blocks["# 文件1: property_prediction.py 数据"]
    structure_block = blocks["# 文件2: structure_generation.py 数据"]
    optimization_block = blocks["# 文件3: autonomous_optimization.py 数据"]

    summary = {
        "raw_data_summary": {
            k: [summarize_array(arr) for arr in v] for k, v in blocks.items()
        }
    }

    property_df, graph_summary = build_property_dataset(property_block)
    prop_metrics, prop_results, importances = run_property_prediction(property_df)
    struct_combined, struct_summary = run_structure_generation(structure_block)
    opt_grid, opt_history, opt_summary = run_optimization(optimization_block)

    summary["property_prediction"] = {"graph_summary": graph_summary, "metrics": prop_metrics}
    summary["structure_generation"] = struct_summary
    summary["optimization"] = opt_summary

    property_df.to_csv(OUT / "property_dataset.csv", index=False)
    prop_results.to_csv(OUT / "property_predictions.csv", index=False)
    importances.rename("importance").to_csv(OUT / "feature_importance.csv")
    struct_combined.to_csv(OUT / "structure_generation_samples.csv", index=False)
    opt_grid.to_csv(OUT / "optimization_grid.csv", index=False)
    opt_history.to_csv(OUT / "optimization_history.csv", index=False)
    with open(OUT / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_figures(property_df, prop_results, importances, struct_combined, opt_grid, opt_history)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
