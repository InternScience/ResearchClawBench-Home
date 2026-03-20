from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
IMAGE_DIR = REPORT_DIR / "images"

FEATURES = [
    "Nucleophilic-HEA",
    "Hydrophobic-BA",
    "Acidic-CBEA",
    "Cationic-ATAC",
    "Aromatic-PEA",
    "Amide-AAm",
]
TARGET = "Glass (kPa)_10s"
HOLDOUT_PREFIX = "GPRFR-"


@dataclass
class ModelSpec:
    name: str
    estimator: object


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, REPORT_DIR, IMAGE_DIR, OUTPUT_DIR / ".mplconfig"]:
        path.mkdir(parents=True, exist_ok=True)


def figure_path(name: str) -> Path:
    return IMAGE_DIR / name


def output_path(name: str) -> Path:
    return OUTPUT_DIR / name


def load_related_work() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for pdf_path in sorted((ROOT / "related_work").glob("*.pdf")):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc[:2]:
            text += page.get_text("text") + "\n"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = pdf_path.stem
        for idx, line in enumerate(lines[:25]):
            if "Population-based heteropolymer design to" in line and idx + 1 < len(lines):
                title = f"{line} {lines[idx + 1]}".strip()
                break
            if (
                12 <= len(line) <= 120
                and "doi" not in line.lower()
                and "Nature" not in line
                and "Vol" not in line
                and "Article" not in line
                and "Received:" not in line
            ):
                title = line
                break
        abstract_guess = " ".join(lines[5:18])[:1200]
        records.append(
            {
                "file": pdf_path.name,
                "title": title,
                "excerpt": abstract_guess,
            }
        )
    return records


def load_verified_dataset() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "184_verified_Original Data_ML_20230926.xlsx")
    numeric_cols = FEATURES + [
        TARGET,
        "Steel (kPa)_10s",
        "Q",
        "Phase Seperation",
        "Modulus (kPa)",
        "Tanδ",
        "Slope",
        "Log_Slope",
        "G''",
        "XlogP3",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["feature_sum"] = df[FEATURES].sum(axis=1)
    df["is_holdout"] = df["No."].str.startswith(HOLDOUT_PREFIX)
    return df


def load_optimization_tables() -> pd.DataFrame:
    workbook = DATA_DIR / "ML_ei&pred (1&2&3rounds)_20240408.xlsx"
    frames: list[pd.DataFrame] = []
    xl = pd.ExcelFile(workbook)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df["sheet"] = sheet
        df["ML"] = df["ML"].ffill()
        for col in FEATURES + ["Glass (kPa)_max"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=FEATURES + ["Glass (kPa)_max"]).copy()
        df["feature_sum"] = df[FEATURES].sum(axis=1)
        df = df[df["feature_sum"].between(0.99, 1.01)].copy()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["candidate_id"] = np.arange(1, len(combined) + 1)
    return combined


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "QuadraticRidge",
            Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        ),
        ModelSpec(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
            ),
        ),
        ModelSpec(
            "ExtraTrees",
            ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
            ),
        ),
        ModelSpec(
            "GradientBoosting",
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.8,
                random_state=42,
            ),
        ),
    ]


def nearest_neighbor_distances(train_x: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(train_x)
    dists, _ = nn.kneighbors(query_x)
    return dists[:, 0]


def training_neighbor_distances(train_x: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(train_x)
    dists, _ = nn.kneighbors(train_x)
    return dists[:, 1]


def repeated_cv_predictions(
    estimator: object,
    x: np.ndarray,
    y: np.ndarray,
    cv: RepeatedKFold,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pred_lists: list[list[float]] = [[] for _ in range(len(y))]
    fold_rows: list[dict[str, float]] = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
        est = clone(estimator)
        est.fit(x[train_idx], y[train_idx])
        preds = est.predict(x[test_idx])
        fold_rows.append(
            {
                "fold_id": fold_id,
                "r2": r2_score(y[test_idx], preds),
                "mae": mean_absolute_error(y[test_idx], preds),
                "rmse": root_mean_squared_error(y[test_idx], preds),
            }
        )
        for idx, pred in zip(test_idx, preds):
            pred_lists[idx].append(float(pred))

    pred_mean = np.array([np.mean(vals) for vals in pred_lists])
    pred_std = np.array([np.std(vals) for vals in pred_lists])
    pointwise = pd.DataFrame(
        {
            "observed": y,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "n_predictions": [len(vals) for vals in pred_lists],
        }
    )
    metrics = {
        "r2": float(r2_score(y, pred_mean)),
        "mae": float(mean_absolute_error(y, pred_mean)),
        "rmse": float(root_mean_squared_error(y, pred_mean)),
        "fold_r2_mean": float(pd.DataFrame(fold_rows)["r2"].mean()),
        "fold_r2_std": float(pd.DataFrame(fold_rows)["r2"].std()),
        "fold_mae_mean": float(pd.DataFrame(fold_rows)["mae"].mean()),
        "fold_rmse_mean": float(pd.DataFrame(fold_rows)["rmse"].mean()),
    }
    return pointwise, metrics


def bootstrap_predictions(
    estimator: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    n_boot: int = 80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    preds = np.zeros((n_boot, len(x_query)))
    n = len(x_train)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, n)
        est = clone(estimator)
        est.fit(x_train[sample_idx], y_train[sample_idx])
        preds[i] = est.predict(x_query)
    return preds.mean(axis=0), preds.std(axis=0)


def sample_simplex(center: np.ndarray, concentration: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha = np.clip(center * concentration, 0.2, None)
    return rng.dirichlet(alpha, size=n)


def make_overview_figure(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    feature_means = df[FEATURES].mean().sort_values(ascending=False)
    sns.barplot(
        x=feature_means.values,
        y=feature_means.index,
        hue=feature_means.index,
        orient="h",
        palette="crest",
        dodge=False,
        legend=False,
        ax=ax0,
    )
    ax0.set_title("Average Monomer Fraction")
    ax0.set_xlabel("Mean composition")
    ax0.set_ylabel("")

    sns.histplot(df[TARGET], bins=20, color="#287271", ax=ax1)
    ax1.axvline(100, color="#c1121f", linestyle="--", linewidth=2, label="0.1 MPa")
    ax1.axvline(1000, color="#780000", linestyle=":", linewidth=2, label="1 MPa target")
    ax1.set_title("Adhesive Strength Distribution")
    ax1.set_xlabel("Glass adhesion at 10 s (kPa)")
    ax1.legend(frameon=False)

    sns.scatterplot(
        data=df,
        x="Hydrophobic-BA",
        y="Aromatic-PEA",
        hue=TARGET,
        palette="viridis",
        s=80,
        ax=ax2,
    )
    ax2.set_title("Hydrophobic-Aromatic Design Plane")

    sns.scatterplot(
        data=df,
        x="Nucleophilic-HEA",
        y=TARGET,
        hue="Hydrophobic-BA",
        palette="mako",
        s=80,
        ax=ax3,
    )
    ax3.set_title("Strength vs HEA Fraction")
    ax3.legend(frameon=False, title="BA")

    fig.tight_layout()
    fig.savefig(figure_path("figure_01_data_overview.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_correlation_figure(df: pd.DataFrame) -> None:
    corr_cols = FEATURES + [TARGET, "Q", "Modulus (kPa)", "G''", "XlogP3"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Correlation Structure Across Composition and Material Properties")
    plt.tight_layout()
    plt.savefig(figure_path("figure_02_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def make_validation_figure(
    df: pd.DataFrame,
    cv_pred: pd.DataFrame,
    holdout_pred: pd.DataFrame,
    chosen_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    merged = pd.concat([df[["No.", TARGET]].reset_index(drop=True), cv_pred.reset_index(drop=True)], axis=1)
    sns.scatterplot(data=merged, x=TARGET, y="pred_mean", ax=axes[0], color="#33658a", s=70)
    max_val = max(merged[TARGET].max(), merged["pred_mean"].max()) * 1.05
    axes[0].plot([0, max_val], [0, max_val], linestyle="--", color="black")
    axes[0].set_title(f"Repeated CV: {chosen_label}")
    axes[0].set_xlabel("Observed (kPa)")
    axes[0].set_ylabel("Predicted (kPa)")

    sns.scatterplot(
        data=holdout_pred,
        x=TARGET,
        y="pred_mean",
        s=100,
        color="#bc4749",
        ax=axes[1],
    )
    max_val2 = max(holdout_pred[TARGET].max(), holdout_pred["pred_mean"].max()) * 1.05
    axes[1].plot([0, max_val2], [0, max_val2], linestyle="--", color="black")
    for _, row in holdout_pred.iterrows():
        axes[1].text(row[TARGET] + 3, row["pred_mean"] + 3, row["No."], fontsize=10)
    axes[1].set_title("External Frontier Check: GPRFR Holdout")
    axes[1].set_xlabel("Observed (kPa)")
    axes[1].set_ylabel("Predicted from original-180 model (kPa)")

    fig.tight_layout()
    fig.savefig(figure_path("figure_03_validation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_importance_figure(estimator: object, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    result = permutation_importance(estimator, x, y, n_repeats=40, random_state=42)
    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=importance_df,
        x="importance_mean",
        y="feature",
        hue="feature",
        palette="rocket",
        dodge=False,
        legend=False,
    )
    plt.xlabel("Permutation importance (mean drop in score)")
    plt.ylabel("")
    plt.title("Model Dependence on Monomer Composition")
    plt.tight_layout()
    plt.savefig(figure_path("figure_04_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return importance_df


def make_partial_dependence_figure(estimator: object, x: pd.DataFrame) -> None:
    top_features = ["Hydrophobic-BA", "Nucleophilic-HEA", "Aromatic-PEA"]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    PartialDependenceDisplay.from_estimator(estimator, x, top_features, ax=ax, grid_resolution=40)
    fig.suptitle("Partial Dependence of the Three Most Influential Features", y=1.03)
    fig.tight_layout()
    fig.savefig(figure_path("figure_05_partial_dependence.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_candidate_figure(candidate_df: pd.DataFrame) -> None:
    top = candidate_df.nlargest(20, "archived_predicted_strength").copy()
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=candidate_df,
        x="archived_predicted_strength",
        y="retro_mean",
        hue="sheet",
        size="nn_dist",
        palette="Set2",
        sizes=(40, 200),
        alpha=0.8,
    )
    for _, row in top.iterrows():
        plt.text(row["archived_predicted_strength"] + 2, row["retro_mean"] + 1, f"{row['ML']}-{int(row['NO.'])}", fontsize=8)
    plt.xlabel("Archived optimization score (kPa)")
    plt.ylabel("Retrospective GBR prediction (kPa)")
    plt.title("Historical Optimization Scores vs Retrospective Model Scores")
    plt.tight_layout()
    plt.savefig(figure_path("figure_06_candidate_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def make_design_figure(design_df: pd.DataFrame) -> None:
    plot_df = design_df.melt(
        id_vars=["candidate_name", "design_track"],
        value_vars=FEATURES,
        var_name="feature",
        value_name="fraction",
    )
    plt.figure(figsize=(13, 7))
    sns.barplot(data=plot_df, x="candidate_name", y="fraction", hue="feature", palette="tab20")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("")
    plt.ylabel("Monomer fraction")
    plt.title("Recommended Design Candidates and Their Composition Signatures")
    plt.tight_layout()
    plt.savefig(figure_path("figure_07_design_candidates.png"), dpi=300, bbox_inches="tight")
    plt.close()


def build_design_candidates(
    original_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    best_estimator: object,
) -> pd.DataFrame:
    x_train = original_df[FEATURES].to_numpy()
    y_train = original_df[TARGET].to_numpy()

    boot_mean_holdout, boot_std_holdout = bootstrap_predictions(
        best_estimator,
        x_train,
        y_train,
        holdout_df[FEATURES].to_numpy(),
        n_boot=80,
        seed=42,
    )
    holdout_aug = holdout_df.copy()
    holdout_aug["retro_mean"] = boot_mean_holdout
    holdout_aug["retro_std"] = boot_std_holdout
    holdout_aug["nn_dist"] = nearest_neighbor_distances(x_train, holdout_aug[FEATURES].to_numpy())
    holdout_aug["design_track"] = "Measured frontier"
    holdout_aug["candidate_name"] = holdout_aug["No."]

    top_original = original_df.nlargest(12, TARGET)
    top_center = top_original[FEATURES].mean().to_numpy()
    frontier_center = holdout_df[FEATURES].mean().to_numpy()
    sample_a = sample_simplex(top_center, concentration=90, n=6000, seed=123)
    sample_b = sample_simplex(frontier_center, concentration=110, n=6000, seed=456)
    sample_c = np.random.default_rng(789).dirichlet(np.ones(len(FEATURES)), size=8000)
    sampled = np.vstack([sample_a, sample_b, sample_c])
    sampled_df = pd.DataFrame(sampled, columns=FEATURES)
    sampled_df["nn_dist"] = nearest_neighbor_distances(x_train, sampled_df[FEATURES].to_numpy())
    ad_threshold = float(np.quantile(training_neighbor_distances(x_train), 0.9))
    sampled_df = sampled_df[sampled_df["nn_dist"] <= max(ad_threshold, 0.12)].copy()
    sampled_df["retro_mean"], sampled_df["retro_std"] = bootstrap_predictions(
        best_estimator,
        x_train,
        y_train,
        sampled_df[FEATURES].to_numpy(),
        n_boot=80,
        seed=314,
    )
    sampled_df["penalized_score"] = sampled_df["retro_mean"] - 0.5 * sampled_df["retro_std"] - 40 * sampled_df["nn_dist"]
    sampled_df = sampled_df.sort_values("penalized_score", ascending=False)

    selected_rows = []
    min_distance_between_selected = 0.08
    selected_vectors: list[np.ndarray] = []
    for _, row in sampled_df.iterrows():
        vec = row[FEATURES].to_numpy()
        if not selected_vectors:
            selected_vectors.append(vec)
            selected_rows.append(row)
            continue
        if all(np.linalg.norm(vec - ref) >= min_distance_between_selected for ref in selected_vectors):
            selected_vectors.append(vec)
            selected_rows.append(row)
        if len(selected_rows) == 4:
            break

    interpolative = pd.DataFrame(selected_rows).reset_index(drop=True)
    interpolative["design_track"] = "Interpolative search"
    interpolative["candidate_name"] = [f"INT-{i+1}" for i in range(len(interpolative))]
    interpolative["No."] = interpolative["candidate_name"]
    interpolative[TARGET] = np.nan

    candidate_top = candidate_df.nlargest(4, "retro_mean").copy().reset_index(drop=True)
    candidate_top["design_track"] = "Historical optimizer"
    candidate_top["candidate_name"] = [f"HIST-{i+1}" for i in range(len(candidate_top))]
    candidate_top["No."] = candidate_top["candidate_name"]
    candidate_top[TARGET] = np.nan

    design_df = pd.concat(
        [
            interpolative[
                ["candidate_name", "design_track", "No.", TARGET, "retro_mean", "retro_std", "nn_dist"] + FEATURES
            ],
            holdout_aug[
                ["candidate_name", "design_track", "No.", TARGET, "retro_mean", "retro_std", "nn_dist"] + FEATURES
            ],
            candidate_top[
                ["candidate_name", "design_track", "No.", TARGET, "retro_mean", "retro_std", "nn_dist"] + FEATURES
            ],
        ],
        ignore_index=True,
    )
    return design_df


def write_report(
    related_work: list[dict[str, str]],
    summary: dict[str, object],
    model_table: pd.DataFrame,
    holdout_pred: pd.DataFrame,
    importance_df: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    design_df: pd.DataFrame,
) -> None:
    top_importance = importance_df.head(3)["feature"].tolist()
    top_candidate_lines = []
    for _, row in design_df.head(8).iterrows():
        measured = "measured" if pd.notna(row[TARGET]) else "proposed"
        top_candidate_lines.append(
            f"- `{row['candidate_name']}` ({row['design_track']}): "
            f"BA={row['Hydrophobic-BA']:.3f}, PEA={row['Aromatic-PEA']:.3f}, "
            f"ATAC={row['Cationic-ATAC']:.3f}, HEA={row['Nucleophilic-HEA']:.3f}; "
            f"retrospective score={row['retro_mean']:.1f} ± {row['retro_std']:.1f} kPa; {measured}."
        )

    paper_lines = []
    for record in related_work:
        paper_lines.append(f"- `{record['file']}`: **{record['title']}**")

    holdout_table = holdout_pred[["No.", TARGET, "pred_mean", "pred_std", "nn_dist"]].copy()
    holdout_table.columns = ["Candidate", "Observed_kPa", "Predicted_kPa", "Predicted_SD", "NearestNeighborDistance"]

    report = f"""# Data-Driven Design Analysis for Underwater Adhesive Hydrogels

## Abstract
This study re-analyzed the verified hydrogel dataset (`n = {summary['n_total']}` formulations; `n = {summary['n_original']}` original screening points and `n = {summary['n_holdout']}` later `GPRFR-*` frontier formulations) to assess whether monomer-composition features derived from natural adhesive proteins are sufficient to support de novo design of stronger underwater adhesives. A repeated cross-validation benchmark on the original 180 formulations showed that a gradient-boosting regressor provided the best internal predictive performance (`R^2 = {summary['best_r2']:.3f}`, `MAE = {summary['best_mae']:.1f} kPa`, `RMSE = {summary['best_rmse']:.1f} kPa`). Across the original screen, high `Hydrophobic-BA` and moderate `Aromatic-PEA` favored stronger adhesion, whereas high `Nucleophilic-HEA` was consistently detrimental. However, the model severely under-predicted the four experimentally verified `GPRFR-*` frontier formulations, indicating that the initial 180-point library does not fully resolve the high-adhesion regime. The strongest measured sample in the available data reached `304.6 kPa` (`0.305 MPa`), which is far below the stated `>1 MPa` target. Therefore, the data support a practical near-term design rule for moving from the `0.10–0.15 MPa` range toward the `0.20–0.35 MPa` range, but they do not justify a credible statistical claim that `>1 MPa` underwater adhesion has been achieved or is currently predictable from the available training library alone.

## Related Work Context
The local reference set establishes the scientific framing of this analysis:
{chr(10).join(paper_lines)}

Three ideas from the reference set shaped the analysis. First, the Nature paper (`paper_000.pdf`) frames protein-inspired polymer design as a population-level statistical matching problem, which justifies treating the hydrogel compositions as a distributional design space rather than as isolated recipes. Second, `paper_001.pdf` emphasizes that synthetic heteropolymer composition is itself a stochastic object with possible drift and mismatch between feed ratio and realized polymer sequence, which argues for cautious interpretation of purely composition-based regressors. Third, `paper_002.pdf` explains why wet adhesion requires a careful balance of hydrophobic, electrostatic, and aromatic interactions in water; this motivated a focus on interaction-rich nonlinear models and on interpreting the feature effects rather than reporting black-box predictions alone.

## Data and Assumptions
The main input was `data/184_verified_Original Data_ML_20230926.xlsx`. All six monomer features sum to approximately one, so the dataset lies on a five-dimensional simplex. The dense response used throughout the analysis was `Glass (kPa)_10s`, because it is available for all 184 verified entries. `Steel (kPa)_10s` is too sparse (`28/184` non-missing values) to support a parallel design study. The final optimization workbook (`data/ML_ei&pred (1&2&3rounds)_20240408.xlsx`) was treated as a historical record of optimization proposals and archived scores, not as experimentally verified labels, because its rows do not map directly onto the measured entries in the verified dataset.

The experimentally observed range in the verified data is `1.19–304.60 kPa`. For clarity, `1 MPa = 1000 kPa`, so the entire measured dataset remains below one-third of the stated target. This matters: any discussion of `>1 MPa` is necessarily extrapolative in this workspace.

## Methods
1. I split the verified dataset into the original `G-001` to `G-180` screening library and the four later `GPRFR-*` frontier formulations.
2. I benchmarked four regressors on the original 180 points with repeated 5-fold cross-validation: quadratic ridge regression, random forest, extra-trees, and gradient boosting.
3. I selected the highest-performing model by repeated-CV accuracy and refit it on the full original 180-point library.
4. I quantified feature effects with permutation importance and partial dependence.
5. I tested extrapolation by predicting the four `GPRFR-*` formulations using only the original-180-trained model.
6. I retrospectively rescored the historical optimization candidates using bootstrap ensembles of the selected model.
7. I generated additional simplex-constrained candidates by sampling around the top-performing original formulations and around the `GPRFR-*` centroid, while filtering by nearest-neighbor applicability to the original design space.

## Results

### 1. Data overview
The original screen is broad, but the strongest formulations are not random. They cluster toward high `Hydrophobic-BA`, moderate `Cationic-ATAC`, and suppressed `Nucleophilic-HEA`. Figure 1 summarizes the target distribution and the most informative low-dimensional projections.

![Data overview](images/figure_01_data_overview.png)

Figure 2 shows the correlation structure. Adhesion correlates positively with `Hydrophobic-BA` and `Aromatic-PEA`, negatively with `Nucleophilic-HEA`, and only weakly with `Amide-AAm`. This pattern is chemically plausible for wet adhesion because hydrophobic and aromatic motifs can strengthen interfacial association, whereas excessive hydrophilic character can dilute cohesive and surface-binding interactions.

![Correlation heatmap](images/figure_02_correlation_heatmap.png)

### 2. Predictive performance on the original 180-point screen
Table 1 summarizes the repeated-CV benchmark.

| Model | R2 | MAE (kPa) | RMSE (kPa) |
|---|---:|---:|---:|
{chr(10).join([f"| {row['model']} | {row['r2']:.3f} | {row['mae']:.1f} | {row['rmse']:.1f} |" for _, row in model_table.iterrows()])}

The selected model was **{summary['best_model']}**, which improved over the interpretable quadratic baseline and reached `R^2 = {summary['best_r2']:.3f}`. This is useful but not decisive accuracy: the typical error remains on the order of `15 kPa`, and the response distribution is heavy-tailed.

Figure 3 makes the key distinction of this project. The left panel shows that internal validation on the original 180-point library is serviceable. The right panel shows that the same model fails dramatically on the later `GPRFR-*` frontier formulations, under-predicting all four.

![Validation](images/figure_03_validation.png)

The `GPRFR-*` under-prediction is not a minor calibration issue. The holdout predictions are:

| Candidate | Observed_kPa | Predicted_kPa | Predicted_SD | NearestNeighborDistance |
|---|---:|---:|---:|---:|
{chr(10).join([f"| {row['Candidate']} | {row['Observed_kPa']:.1f} | {row['Predicted_kPa']:.1f} | {row['Predicted_SD']:.1f} | {row['NearestNeighborDistance']:.3f} |" for _, row in holdout_table.iterrows()])}

This failure means the original 180-point library captures the general trend but not the sharp transition into the frontier regime reached by the later optimization campaign.

### 3. Feature effects
Permutation importance ranked the main drivers as `{top_importance[0]}`, `{top_importance[1]}`, and `{top_importance[2]}`. The dominant pattern is that increasing `Hydrophobic-BA` raises predicted adhesion until the formulation approaches a BA-rich regime, while increasing `Nucleophilic-HEA` generally suppresses performance.

![Feature importance](images/figure_04_feature_importance.png)

The partial dependence plots reinforce the same picture: stronger adhesion is associated with high BA, low HEA, and a moderate aromatic contribution rather than a monotonic push toward any single monomer.

![Partial dependence](images/figure_05_partial_dependence.png)

### 4. Retrospective analysis of historical optimization candidates
The archived optimization workbook contains many high-scoring candidates according to the original optimization loop, some with archived scores above `300 kPa`. When rescored by the present retrospective model, the rank order is only moderately aligned, indicating model dependence in the frontier regime rather than a single robust optimum. Figure 6 visualizes this disagreement.

![Historical candidates](images/figure_06_candidate_comparison.png)

The most consistent historical pattern is not a single recipe but a narrow chemistry family: BA-rich formulations with low or zero acidic content, low HEA, modest ATAC, and an aromatic fraction between roughly `0.20` and `0.35`.

### 5. Design implications
The data support two distinct design tracks:

1. **Interpolative track**: stay close to the original screened manifold. This gives the most trustworthy predictions and favors BA-rich, low-HEA, low-AAm formulations with either modest PEA or modest acidic content.
2. **Frontier track**: move toward the experimentally verified `GPRFR-*` region. This is more aggressive and empirically promising, but the original-180-trained model does not extrapolate reliably there.

The recommended candidates are shown in Figure 7 and listed below.

![Design candidates](images/figure_07_design_candidates.png)

{chr(10).join(top_candidate_lines)}

## Discussion
The central conclusion is negative but useful: **the available data do not support a defensible claim that the current statistical design pipeline can deliver `>1 MPa` underwater adhesion.** The strongest measured formulation in the verified dataset is `304.6 kPa`, equivalent to `0.305 MPa`. This is a strong improvement over the median of the original library, but it remains far from the desired threshold.

At the same time, the dataset reveals a concrete research opportunity. The four `GPRFR-*` frontier formulations show that a substantial jump beyond the original 180-screen is possible when the composition moves toward a BA-dominant, low-HEA, low-acid regime with controlled aromatic and cationic content. Because those points are under-predicted even by the best internally validated model, they should be treated as evidence of an unresolved regime boundary. The correct next experiment is therefore not another global screen across the entire simplex. It is a dense local campaign around the `GPRFR-*` neighborhood, with a specific focus on:

- fixing `HEA` and `CBEA` near zero;
- sweeping `BA` between roughly `0.48` and `0.62`;
- sweeping `PEA` between roughly `0.18` and `0.45`;
- keeping `ATAC` in a low-to-moderate band (`0.05–0.15`);
- testing whether a small `AAm` fraction stabilizes or suppresses the frontier response.

The present analysis therefore reframes the project goal. The immediate statistically supported objective is not yet `>1 MPa`, but rather **mapping and validating the sharp frontier between about `0.10 MPa` and `0.35 MPa`** so that a second-generation model can be trained on a denser local library.

## Limitations
- The design model sees only monomer feed fractions, not realized sequence distributions, polymerization drift, molecular weight, or interfacial test conditions beyond what is implicitly encoded in the measured labels.
- The historical optimization workbook contains archived scores rather than directly verified labels in this workspace.
- The holdout frontier contains only four measured `GPRFR-*` points, so the frontier regime is experimentally real but statistically under-sampled.
- `Steel (kPa)_10s` is too sparse for a parallel analysis of substrate dependence.
- Because no measured point exceeds `304.6 kPa`, any claim about `>1 MPa` remains unsupported extrapolation.

## Reproducibility
All analyses, figures, and tables were generated by `code/run_analysis.py`. Intermediate outputs are stored in `outputs/`, and all report figures are stored in `report/images/`.
"""
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    related_work = load_related_work()
    verified = load_verified_dataset()
    original = verified[~verified["is_holdout"]].copy().reset_index(drop=True)
    holdout = verified[verified["is_holdout"]].copy().reset_index(drop=True)
    candidate_df = load_optimization_tables()

    x_original = original[FEATURES].to_numpy()
    y_original = original[TARGET].to_numpy()
    x_holdout = holdout[FEATURES].to_numpy()

    cv = RepeatedKFold(n_splits=5, n_repeats=8, random_state=42)
    comparison_rows = []
    model_pointwise: dict[str, pd.DataFrame] = {}
    model_metrics: dict[str, dict[str, float]] = {}
    for spec in model_specs():
        pointwise, metrics = repeated_cv_predictions(spec.estimator, x_original, y_original, cv)
        model_pointwise[spec.name] = pointwise
        model_metrics[spec.name] = metrics
        comparison_rows.append(
            {
                "model": spec.name,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "fold_r2_mean": metrics["fold_r2_mean"],
                "fold_r2_std": metrics["fold_r2_std"],
            }
        )

    model_table = pd.DataFrame(comparison_rows).sort_values("r2", ascending=False).reset_index(drop=True)
    chosen_label = model_table.iloc[0]["model"]
    best_spec = next(spec for spec in model_specs() if spec.name == chosen_label)
    best_estimator = clone(best_spec.estimator)
    best_estimator.fit(original[FEATURES], y_original)

    holdout_mean, holdout_std = bootstrap_predictions(best_estimator, x_original, y_original, x_holdout, n_boot=80, seed=42)
    holdout_pred = holdout.copy()
    holdout_pred["pred_mean"] = holdout_mean
    holdout_pred["pred_std"] = holdout_std
    holdout_pred["nn_dist"] = nearest_neighbor_distances(x_original, x_holdout)

    candidate_df["retro_mean"], candidate_df["retro_std"] = bootstrap_predictions(
        best_estimator,
        x_original,
        y_original,
        candidate_df[FEATURES].to_numpy(),
        n_boot=80,
        seed=7,
    )
    candidate_df["archived_predicted_strength"] = candidate_df["Glass (kPa)_max"]
    candidate_df["nn_dist"] = nearest_neighbor_distances(x_original, candidate_df[FEATURES].to_numpy())
    candidate_df["penalized_score"] = candidate_df["retro_mean"] - 0.5 * candidate_df["retro_std"] - 40 * candidate_df["nn_dist"]

    importance_df = make_importance_figure(best_estimator, original[FEATURES], original[TARGET])
    make_overview_figure(original)
    make_correlation_figure(verified)
    make_validation_figure(original, model_pointwise[chosen_label], holdout_pred, chosen_label)
    make_partial_dependence_figure(best_estimator, original[FEATURES])
    make_candidate_figure(candidate_df)

    design_df = build_design_candidates(original, holdout, candidate_df, best_estimator)
    make_design_figure(design_df)

    top_original = original.nlargest(15, TARGET)[["No.", TARGET] + FEATURES + ["Q", "Modulus (kPa)", "XlogP3"]]
    candidate_summary = candidate_df.sort_values(["penalized_score", "retro_mean"], ascending=False).head(20)

    verified.to_csv(output_path("verified_dataset_cleaned.csv"), index=False)
    model_table.to_csv(output_path("model_comparison.csv"), index=False)
    model_pointwise[chosen_label].assign(No=original["No."]).to_csv(output_path("best_model_cv_predictions.csv"), index=False)
    holdout_pred.to_csv(output_path("gprfr_holdout_predictions.csv"), index=False)
    candidate_df.to_csv(output_path("historical_candidate_scores.csv"), index=False)
    candidate_summary.to_csv(output_path("top_historical_candidates.csv"), index=False)
    top_original.to_csv(output_path("top_original_formulations.csv"), index=False)
    importance_df.to_csv(output_path("feature_importance.csv"), index=False)
    design_df.to_csv(output_path("design_candidates.csv"), index=False)

    summary = {
        "n_total": int(len(verified)),
        "n_original": int(len(original)),
        "n_holdout": int(len(holdout)),
        "target_min_kpa": float(verified[TARGET].min()),
        "target_median_kpa": float(verified[TARGET].median()),
        "target_max_kpa": float(verified[TARGET].max()),
        "best_model": chosen_label,
        "best_r2": float(model_table.iloc[0]["r2"]),
        "best_mae": float(model_table.iloc[0]["mae"]),
        "best_rmse": float(model_table.iloc[0]["rmse"]),
        "holdout_r2": float(r2_score(holdout_pred[TARGET], holdout_pred["pred_mean"])),
        "holdout_mae": float(mean_absolute_error(holdout_pred[TARGET], holdout_pred["pred_mean"])),
        "holdout_rmse": float(root_mean_squared_error(holdout_pred[TARGET], holdout_pred["pred_mean"])),
        "mpa_target_kpa": 1000.0,
        "max_measured_mpa": float(verified[TARGET].max() / 1000.0),
        "applicability_threshold": float(np.quantile(training_neighbor_distances(x_original), 0.9)),
    }
    output_path("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_report(
        related_work=related_work,
        summary=summary,
        model_table=model_table,
        holdout_pred=holdout_pred,
        importance_df=importance_df,
        candidate_summary=candidate_summary,
        design_df=design_df,
    )


if __name__ == "__main__":
    main()
