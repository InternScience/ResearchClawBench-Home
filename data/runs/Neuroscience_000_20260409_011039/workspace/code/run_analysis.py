from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"

TARGETS = ["Attack", "Sniffing"]
SEED = 42


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(DATA_DIR / "Together_1_features_extracted.csv")
    targets = pd.read_csv(DATA_DIR / "Together_1_targets_inserted.csv")
    reference = pd.read_csv(DATA_DIR / "Together_1_machine_results_reference.csv")
    return features, targets, reference


def build_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    x = features.copy()
    x = x.rename(columns={"Unnamed: 0": "frame_index"})
    return x


def classifier_library() -> dict[str, Pipeline]:
    return {
        "logistic_l2": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def cross_validated_model_selection(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[str, pd.DataFrame]:
    models = classifier_library()
    cv = TimeSeriesSplit(n_splits=5)
    rows: list[dict[str, float | str | int]] = []

    for model_name, pipeline in models.items():
        for fold, (tr_idx, val_idx) in enumerate(cv.split(x_train), start=1):
            if np.unique(y_train.iloc[tr_idx]).size < 2 or np.unique(
                y_train.iloc[val_idx]
            ).size < 2:
                continue
            model = clone(pipeline)
            model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
            val_prob = model.predict_proba(x_train.iloc[val_idx])[:, 1]
            val_pred = (val_prob >= 0.5).astype(int)
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "average_precision": average_precision_score(
                        y_train.iloc[val_idx], val_prob
                    ),
                    "roc_auc": roc_auc_score(y_train.iloc[val_idx], val_prob),
                    "f1": f1_score(y_train.iloc[val_idx], val_pred, zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(
                        y_train.iloc[val_idx], val_pred
                    ),
                }
            )

    cv_results = pd.DataFrame(rows)
    if cv_results.empty:
        raise RuntimeError("No valid cross-validation folds contained both classes.")
    summary = (
        cv_results.groupby("model", as_index=False)
        .agg(
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            f1_mean=("f1", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
        )
        .sort_values(
            ["average_precision_mean", "roc_auc_mean", "f1_mean"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    best_model_name = str(summary.loc[0, "model"])
    return best_model_name, summary


def fit_calibrated_model(
    model_name: str, x_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline | CalibratedClassifierCV:
    pipeline = classifier_library()[model_name]
    class_counts = y_train.value_counts()
    positive_count = int(class_counts.get(1, 0))
    negative_count = int(class_counts.get(0, 0))
    cv_folds = max(2, min(5, positive_count, negative_count))
    calibrated = CalibratedClassifierCV(
        estimator=clone(pipeline), cv=cv_folds, method="sigmoid"
    )
    calibrated.fit(x_train, y_train)
    return calibrated


def select_threshold_from_training(
    model_name: str, x_train: pd.DataFrame, y_train: pd.Series
) -> float:
    cv = TimeSeriesSplit(n_splits=5)
    probs = np.full(len(y_train), np.nan)

    for tr_idx, val_idx in cv.split(x_train):
        if np.unique(y_train.iloc[tr_idx]).size < 2:
            continue
        model = clone(classifier_library()[model_name])
        model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
        probs[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]

    valid = ~np.isnan(probs)
    if valid.sum() == 0:
        return 0.5

    y_valid = y_train.iloc[valid].to_numpy()
    prob_valid = probs[valid]
    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidate_thresholds:
        pred = (prob_valid >= threshold).astype(int)
        score = f1_score(y_valid, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics(
    y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }
    return metrics


def bootstrap_metric_interval(
    y_true: np.ndarray, y_prob: np.ndarray, metric_name: str, n_boot: int = 1000
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    stats: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if np.unique(y_b).size < 2:
            continue
        p_b = y_prob[idx]
        if metric_name == "average_precision":
            stats.append(average_precision_score(y_b, p_b))
        elif metric_name == "roc_auc":
            stats.append(roc_auc_score(y_b, p_b))
        else:
            raise ValueError(metric_name)
    lower, upper = np.percentile(stats, [2.5, 97.5])
    return float(lower), float(upper)


def extract_feature_importance(
    fitted_model: CalibratedClassifierCV, feature_names: list[str]
) -> pd.DataFrame:
    estimator = fitted_model.calibrated_classifiers_[0].estimator
    final_model = estimator.named_steps["model"]
    if hasattr(final_model, "feature_importances_"):
        scores = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        scores = np.abs(final_model.coef_[0])
    else:
        raise TypeError("Model does not expose feature importance.")
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance


def compare_with_reference(
    reference: pd.DataFrame, target: str, y_prob: np.ndarray
) -> dict[str, float | None]:
    preferred = f"Probability_{target}"
    if preferred in reference.columns:
        ref_col = preferred
    else:
        ref_cols = [c for c in reference.columns if c.lower() == target.lower()]
        if not ref_cols:
            return {"reference_probability_correlation": None}
        ref_col = ref_cols[0]
    ref = reference[ref_col].to_numpy()
    n = min(len(ref), len(y_prob))
    if n < 3:
        return {"reference_probability_correlation": None}
    corr = float(np.corrcoef(ref[:n], y_prob[:n])[0, 1])
    return {"reference_probability_correlation": corr}


def plot_class_balance(targets: pd.DataFrame) -> None:
    rows = []
    for target in TARGETS:
        counts = targets[target].value_counts().sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    "behavior": target,
                    "label": "positive" if label == 1 else "negative",
                    "count": int(count),
                }
            )
    plot_df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=plot_df, x="behavior", y="count", hue="label", palette="deep")
    ax.set_title("Class balance in frame-level behavior labels")
    ax.set_xlabel("")
    ax.set_ylabel("Frames")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "class_balance.png", dpi=200)
    plt.close()


def plot_pr_curve(y_true: pd.Series, y_prob: np.ndarray, target: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}", linewidth=2)
    plt.hlines(
        baseline,
        xmin=0,
        xmax=1,
        linestyles="--",
        colors="gray",
        label=f"Chance = {baseline:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-recall curve: {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / f"pr_curve_{target.lower()}.png", dpi=200)
    plt.close()


def plot_confusion_matrix(y_true: pd.Series, y_prob: np.ndarray, target: str) -> None:
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(f"Confusion matrix: {target}")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / f"confusion_matrix_{target.lower()}.png", dpi=200)
    plt.close()


def plot_probability_trace(
    frame_index: pd.Series, y_true: pd.Series, y_prob: np.ndarray, target: str
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(frame_index, y_prob, label="Predicted probability", linewidth=1.5)
    plt.plot(frame_index, y_true, label="True label", linewidth=1.2, alpha=0.8)
    plt.xlabel("Frame index")
    plt.ylabel("Probability / label")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Temporal prediction trace: {target}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / f"probability_trace_{target.lower()}.png", dpi=200)
    plt.close()


def plot_feature_importance(importance: pd.DataFrame, target: str) -> None:
    top = importance.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"], color="#3a7")
    plt.xlabel("Importance")
    plt.title(f"Top feature importances: {target}")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / f"feature_importance_{target.lower()}.png", dpi=200)
    plt.close()


def write_report_tables(
    dataset_summary: pd.DataFrame,
    model_results: pd.DataFrame,
    cv_results: dict[str, pd.DataFrame],
    feature_tables: dict[str, pd.DataFrame],
) -> None:
    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    model_results.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    for target, table in cv_results.items():
        table.to_csv(OUTPUT_DIR / f"cv_summary_{target.lower()}.csv", index=False)
    for target, table in feature_tables.items():
        table.to_csv(OUTPUT_DIR / f"feature_importance_{target.lower()}.csv", index=False)


def main() -> None:
    ensure_dirs()
    features, targets_full, reference = load_data()
    x = build_feature_matrix(features)
    y = targets_full[TARGETS].copy()

    split_idx = int(len(x) * 0.8)
    x_train = x.iloc[:split_idx].reset_index(drop=True)
    x_test = x.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)

    dataset_summary = pd.DataFrame(
        [
            {
                "n_frames_total": len(x),
                "n_train": len(x_train),
                "n_test": len(x_test),
                "n_features": x.shape[1],
                "attack_positive_rate": y["Attack"].mean(),
                "sniffing_positive_rate": y["Sniffing"].mean(),
                "reference_rows": len(reference),
                "reference_columns": reference.shape[1],
            }
        ]
    )

    plot_class_balance(y)

    model_rows: list[dict[str, float | str]] = []
    cv_summaries: dict[str, pd.DataFrame] = {}
    feature_tables: dict[str, pd.DataFrame] = {}
    prediction_tables: list[pd.DataFrame] = []

    feature_names = list(x.columns)

    for target in TARGETS:
        best_model_name, cv_summary = cross_validated_model_selection(
            x_train, y_train[target]
        )
        cv_summaries[target] = cv_summary
        threshold = select_threshold_from_training(best_model_name, x_train, y_train[target])
        fitted = fit_calibrated_model(best_model_name, x_train, y_train[target])
        y_prob = fitted.predict_proba(x_test)[:, 1]

        metrics = compute_metrics(y_test[target], y_prob, threshold=threshold)
        ap_ci_low, ap_ci_high = bootstrap_metric_interval(
            y_test[target].to_numpy(), y_prob, "average_precision"
        )
        roc_ci_low, roc_ci_high = bootstrap_metric_interval(
            y_test[target].to_numpy(), y_prob, "roc_auc"
        )

        ref_stats = compare_with_reference(reference, target, y_prob)
        model_rows.append(
            {
                "target": target,
                "selected_model": best_model_name,
                "decision_threshold": threshold,
                **metrics,
                "average_precision_ci_low": ap_ci_low,
                "average_precision_ci_high": ap_ci_high,
                "roc_auc_ci_low": roc_ci_low,
                "roc_auc_ci_high": roc_ci_high,
                **ref_stats,
            }
        )

        plot_pr_curve(y_test[target], y_prob, target)
        plot_confusion_matrix(y_test[target], y_prob, target)
        plot_probability_trace(x_test["frame_index"], y_test[target], y_prob, target)

        importance = extract_feature_importance(fitted, feature_names)
        feature_tables[target] = importance
        plot_feature_importance(importance, target)

        prediction_tables.append(
            pd.DataFrame(
                {
                    "frame_index": x_test["frame_index"],
                    "target": target,
                    "y_true": y_test[target],
                    "y_prob": y_prob,
                    "y_pred": (y_prob >= threshold).astype(int),
                }
            )
        )

    model_results = pd.DataFrame(model_rows)
    write_report_tables(dataset_summary, model_results, cv_summaries, feature_tables)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "test_set_predictions.csv", index=False
    )

    summary = {
        "dataset_summary": dataset_summary.to_dict(orient="records")[0],
        "model_results": model_results.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
