#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def token_stats(text: str, prefix: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    stats[f"{prefix}_length"] = len(text)
    tokens = re.findall(r"[A-Z][a-z]?|\(|\)|=|#|\*|[a-z]+|\d", text)
    for tok in ["C", "N", "O", "S", "P", "F", "Cl", "Br", "c", "n", "=", "#", "(", ")", "*"]:
        stats[f"{prefix}_count_{tok}"] = tokens.count(tok)
    stats[f"{prefix}_ring_digits"] = sum(ch.isdigit() for ch in text)
    stats[f"{prefix}_aromatic_chars"] = sum(ch in "cnops" for ch in text)
    stats[f"{prefix}_uppercase_chars"] = sum(ch.isupper() for ch in text)
    stats[f"{prefix}_hetero_ratio"] = (
        (stats[f"{prefix}_count_N"] + stats[f"{prefix}_count_O"] + stats[f"{prefix}_count_S"] + stats[f"{prefix}_count_P"])
        / max(1.0, stats[f"{prefix}_count_C"] + stats[f"{prefix}_count_c"])
    )
    return stats


def featurize_polymer_smiles(smiles: str) -> dict[str, float]:
    return token_stats(smiles, "poly")


def featurize_vitrimer_pair(acid: str, epoxide: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(token_stats(acid, "acid"))
    feats.update(token_stats(epoxide, "epoxy"))
    feats["pair_total_length"] = len(acid) + len(epoxide)
    feats["pair_length_ratio"] = len(acid) / max(1.0, len(epoxide))
    feats["pair_oxygen_total"] = feats["acid_count_O"] + feats["epoxy_count_O"]
    feats["pair_nitrogen_total"] = feats["acid_count_N"] + feats["epoxy_count_N"]
    feats["pair_aromatic_total"] = feats["acid_aromatic_chars"] + feats["epoxy_aromatic_chars"]
    feats["pair_ring_total"] = feats["acid_ring_digits"] + feats["epoxy_ring_digits"]
    feats["pair_hetero_ratio_mean"] = 0.5 * (feats["acid_hetero_ratio"] + feats["epoxy_hetero_ratio"])
    return feats


COMMON_TOKENS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "c", "n", "=", "#", "(", ")", "*"]


def composition_from_stats(stats: dict[str, float], prefix: str) -> dict[str, float]:
    comp: dict[str, float] = {
        "length": stats[f"{prefix}_length"],
        "ring_digits": stats[f"{prefix}_ring_digits"],
        "aromatic_chars": stats[f"{prefix}_aromatic_chars"],
        "uppercase_chars": stats[f"{prefix}_uppercase_chars"],
        "hetero_ratio": stats[f"{prefix}_hetero_ratio"],
    }
    for tok in COMMON_TOKENS:
        comp[f"count_{tok}"] = stats[f"{prefix}_count_{tok}"]
    return comp


def calibration_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        raw = featurize_polymer_smiles(row.smiles)
        feat = composition_from_stats(raw, "poly")
        feat["tg_md"] = row.tg_md
        feat["std"] = row.std
        rows.append(feat)
    return pd.DataFrame(rows)


def vitrimer_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        raw = featurize_vitrimer_pair(row.acid, row.epoxide)
        acid = composition_from_stats(raw, "acid")
        epoxy = composition_from_stats(raw, "epoxy")
        feat: dict[str, float] = {}
        for key in acid:
            feat[key] = acid[key] + epoxy[key] if key.startswith("count_") else 0.5 * (acid[key] + epoxy[key])
        feat["pair_length_ratio"] = raw["pair_length_ratio"]
        feat["pair_oxygen_total"] = raw["pair_oxygen_total"]
        feat["pair_nitrogen_total"] = raw["pair_nitrogen_total"]
        feat["pair_aromatic_total"] = raw["pair_aromatic_total"]
        feat["pair_ring_total"] = raw["pair_ring_total"]
        feat["pair_hetero_ratio_mean"] = raw["pair_hetero_ratio_mean"]
        feat["tg_md"] = row.tg
        feat["std"] = row.std
        rows.append(feat)
    return pd.DataFrame(rows)


def fit_gp_calibration(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        noise_level=1.0, noise_level_bounds=(1e-5, 1e2)
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6, random_state=42, n_restarts_optimizer=1)
    model = Pipeline([("scale", StandardScaler()), ("gp", gp)])
    model.fit(X, y)
    return model


def cross_validated_predictions(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(X))
    for train_idx, test_idx in cv.split(X):
        model = Pipeline([("scale", StandardScaler()), ("ridge", RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2))])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = model.predict(X.iloc[test_idx])
    return preds


def build_latent_embedding(X: pd.DataFrame) -> tuple[np.ndarray, Pipeline]:
    embedder = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=8, random_state=42))])
    Z = embedder.fit_transform(X)
    return Z, embedder


def inverse_design(
    df: pd.DataFrame,
    target_tgs: list[float],
    latent_cols: list[str],
) -> pd.DataFrame:
    Z = df[latent_cols].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=min(25, len(df))).fit(Z)
    records: list[dict[str, object]] = []
    seen: set[tuple[float, str, str]] = set()
    for target in target_tgs:
        anchor = df.iloc[(df["tg_calibrated"] - target).abs().argsort()[:15]].copy()
        for idx, row in anchor.iterrows():
            _, neigh_idx = nbrs.kneighbors(Z[idx].reshape(1, -1), return_distance=True)
            neighbors = df.iloc[neigh_idx[0]]
            sampled = neighbors["tg_calibrated"].sub(target).abs().sort_values().index[:5]
            for j in sampled:
                cand = df.loc[j]
                key = (target, cand["acid"], cand["epoxide"])
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "target_tg": target,
                        "acid": cand["acid"],
                        "epoxide": cand["epoxide"],
                        "tg_md": cand["tg"],
                        "tg_calibrated": cand["tg_calibrated"],
                        "calibration_std": cand["calibration_std"],
                        "target_error": abs(cand["tg_calibrated"] - target),
                        "novelty_score": cand["novelty_score"],
                        "design_score": abs(cand["tg_calibrated"] - target) + 0.15 * cand["novelty_score"] + 0.05 * cand["calibration_std"],
                    }
                )
    out = pd.DataFrame(records).sort_values(["target_tg", "design_score"]).groupby("target_tg").head(10).reset_index(drop=True)
    return out


def plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=40, alpha=0.8)
    lims = [min(y_true.min(), y_pred.min()) - 10, max(y_true.max(), y_pred.max()) + 10]
    plt.plot(lims, lims, linestyle="--", color="black", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Experimental Tg (K)")
    plt.ylabel("Cross-validated calibrated Tg (K)")
    plt.title("GP calibration from MD Tg to experimental Tg")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_md_vs_calibrated(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df.sample(min(2000, len(df)), random_state=42), x="tg", y="tg_calibrated", hue="calibration_std", palette="viridis", s=22)
    plt.xlabel("MD Tg (K)")
    plt.ylabel("Calibrated Tg (K)")
    plt.title("Calibration transfer to vitrimer candidates")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_latent_map(df: pd.DataFrame, selected: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    base = df.sample(min(2500, len(df)), random_state=42)
    plt.scatter(base["latent_1"], base["latent_2"], c=base["tg_calibrated"], cmap="coolwarm", s=12, alpha=0.45)
    if not selected.empty:
        merged = selected.merge(df, on=["acid", "epoxide", "tg_calibrated", "tg"], how="left")
        plt.scatter(merged["latent_1"], merged["latent_2"], c="black", s=28, marker="x")
    plt.xlabel("Latent axis 1")
    plt.ylabel("Latent axis 2")
    plt.title("Latent vitrimer design space")
    cbar = plt.colorbar()
    cbar.set_label("Calibrated Tg (K)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_target_hits(selected: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=selected, x="target_tg", y="tg_calibrated", hue="target_error", size="novelty_score", palette="magma", sizes=(35, 140))
    plt.plot([selected["target_tg"].min() - 5, selected["target_tg"].max() + 5], [selected["target_tg"].min() - 5, selected["target_tg"].max() + 5], "--", color="black", linewidth=1)
    plt.xlabel("Requested Tg target (K)")
    plt.ylabel("Candidate calibrated Tg (K)")
    plt.title("Inverse design candidates versus targets")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    calib = pd.read_csv(DATA_DIR / "tg_calibration.csv")
    vit = pd.read_csv(DATA_DIR / "tg_vitrimer_MD.csv")

    X_calib = calibration_features(calib)
    y_calib = calib["tg_exp"]

    y_cv = cross_validated_predictions(X_calib, y_calib)

    metrics = {
        "n_calibration": int(len(calib)),
        "n_vitrimer_candidates": int(len(vit)),
        "gp_cv_r2": float(r2_score(y_calib, y_cv)),
        "gp_cv_mae": float(mean_absolute_error(y_calib, y_cv)),
        "gp_cv_rmse": safe_rmse(y_calib, y_cv),
        "baseline_md_r2": float(r2_score(y_calib, calib["tg_md"])),
        "baseline_md_mae": float(mean_absolute_error(y_calib, calib["tg_md"])),
        "baseline_md_rmse": float(safe_rmse(y_calib, calib["tg_md"])),
    }

    gp_model = fit_gp_calibration(X_calib, y_calib)
    X_vit = vitrimer_features(vit)
    calibration_cols = list(X_calib.columns)
    X_vit_cal = X_vit.reindex(columns=calibration_cols, fill_value=0.0)
    tg_pred, tg_std = gp_model.predict(X_vit_cal, return_std=True)
    vit = vit.copy()
    vit["tg_calibrated"] = tg_pred
    vit["calibration_std"] = tg_std

    feature_cols = [c for c in X_vit.columns if c not in {"tg_md", "std"}]
    Z, embedder = build_latent_embedding(X_vit[feature_cols])
    latent_cols = [f"latent_{i+1}" for i in range(Z.shape[1])]
    for i, col in enumerate(latent_cols):
        vit[col] = Z[:, i]

    density = NearestNeighbors(n_neighbors=15).fit(Z)
    dists, _ = density.kneighbors(Z)
    vit["novelty_score"] = dists[:, 1:].mean(axis=1)

    target_tgs = [350.0, 400.0, 450.0, 500.0]
    selected = inverse_design(vit, target_tgs, latent_cols)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3)
    rf.fit(Z, vit["tg_calibrated"])
    vit["latent_rf_pred"] = rf.predict(Z)
    metrics["latent_surrogate_r2"] = float(r2_score(vit["tg_calibrated"], vit["latent_rf_pred"]))

    calib_out = calib.copy()
    calib_out["tg_cv_calibrated"] = y_cv

    calib_out.to_csv(OUTPUT_DIR / "calibration_predictions.csv", index=False)
    vit.to_csv(OUTPUT_DIR / "vitrimer_calibrated_predictions.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "inverse_design_candidates.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    summary = {
        "metrics": metrics,
        "targets": target_tgs,
        "top_candidates": selected.groupby("target_tg").head(3).to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_calibration(y_calib.to_numpy(), y_cv, REPORT_IMG_DIR / "calibration_parity.png")
    plot_md_vs_calibrated(vit, REPORT_IMG_DIR / "md_vs_calibrated.png")
    plot_latent_map(vit, selected[["acid", "epoxide", "tg_calibrated", "tg_md"]].rename(columns={"tg_md": "tg"}).drop_duplicates(), REPORT_IMG_DIR / "latent_space_map.png")
    plot_target_hits(selected, REPORT_IMG_DIR / "inverse_design_targets.png")


if __name__ == "__main__":
    main()
