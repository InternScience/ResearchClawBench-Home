#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "report" / "images"


def metric_dict(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(pred.mean()),
    }


def pick_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.1, 0.9, 161):
        pred = (prob >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        score = f1 - 0.05 * max(0.0, 0.5 - precision) - 0.05 * max(0.0, 0.5 - recall)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    train = pd.read_csv(ROOT / "data" / "train_simulated.csv")
    test = pd.read_csv(ROOT / "data" / "test_simulated.csv")

    feature_cols = [str(i) for i in range(20)]
    group_col = "degradation"
    label_col = "label"
    train_fit = (
        train.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), 30000), random_state=42))
        .reset_index(drop=True)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                feature_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [group_col],
            ),
        ]
    )
    model = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    X_train = train_fit[feature_cols + [group_col]]
    y_train = train_fit[label_col].astype(int).to_numpy()
    X_test = test[feature_cols + [group_col]]
    y_test = test[label_col].astype(int).to_numpy()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    pipeline.fit(X_fit, y_fit)
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(y_val, val_prob)
    val_pred = (val_prob >= threshold).astype(int)
    cv_metrics = metric_dict(y_val, val_prob, val_pred)

    pipeline.fit(X_train, y_train)
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = metric_dict(y_test, test_prob, test_pred)

    leaderboard = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "threshold": threshold,
                **{f"cv_{k}": v for k, v in cv_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        ]
    )
    leaderboard.to_csv(OUTPUT_DIR / "model_leaderboard.csv", index=False)

    pred_df = test.copy()
    pred_df["pred_prob"] = test_prob
    pred_df["pred_label"] = test_pred
    pred_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    by_deg = []
    for degradation, df in pred_df.groupby(group_col):
        m = metric_dict(
            df[label_col].astype(int).to_numpy(),
            df["pred_prob"].to_numpy(),
            df["pred_label"].astype(int).to_numpy(),
        )
        m["degradation"] = degradation
        m["count"] = int(len(df))
        by_deg.append(m)
    by_deg_df = pd.DataFrame(by_deg).sort_values("degradation")
    by_deg_df.to_csv(OUTPUT_DIR / "metrics_by_degradation.csv", index=False)

    dataset_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train),
                "fit_rows": len(train_fit),
                "positive_rate": train[label_col].mean(),
            },
            {"split": "test", "rows": len(test), "positive_rate": test[label_col].mean()},
        ]
    )
    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = np.abs(pipeline.named_steps["model"].coef_[0])
    coef_df = pd.DataFrame({"feature": feature_names, "importance": coefs}).sort_values("importance", ascending=False)
    coef_df.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

    precision, recall, _ = precision_recall_curve(y_test, test_prob)
    plt.figure(figsize=(6, 4.5))
    plt.plot(recall, precision, label=f"AP={test_metrics['average_precision']:.3f}")
    plt.axhline(y_test.mean(), linestyle="--", color="gray", label=f"Prevalence={y_test.mean():.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "precision_recall_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plot_df = pd.DataFrame(
        {
            "metric": ["Average precision", "F1", "ROC AUC", "Balanced accuracy"],
            "value": [
                test_metrics["average_precision"],
                test_metrics["f1"],
                test_metrics["roc_auc"],
                test_metrics["balanced_accuracy"],
            ],
        }
    )
    sns.barplot(data=plot_df, x="metric", y="value", color="#4C72B0")
    plt.ylim(0, 1)
    plt.title("Held-out logistic regression performance")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "model_comparison.png", dpi=180)
    plt.close()

    prev_df = pd.concat(
        [
            train.groupby(group_col)[label_col].mean().rename("positive_rate").reset_index().assign(split="train"),
            test.groupby(group_col)[label_col].mean().rename("positive_rate").reset_index().assign(split="test"),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=prev_df, x=group_col, y="positive_rate", hue="split")
    plt.xticks(rotation=15)
    plt.title("Positive prevalence by degradation")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "class_prevalence_by_degradation.png", dpi=180)
    plt.close()

    heatmap_df = by_deg_df.set_index("degradation")[["average_precision", "f1", "precision", "recall", "balanced_accuracy"]]
    plt.figure(figsize=(7, 3.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1)
    plt.title("Performance by degradation type")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "degradation_heatmap.png", dpi=180)
    plt.close()

    top = coef_df.head(12).iloc[::-1]
    plt.figure(figsize=(8, 5.5))
    plt.barh(top["feature"], top["importance"], color="#DD8452")
    plt.title("Top absolute logistic coefficients")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "feature_importance.png", dpi=180)
    plt.close()

    (OUTPUT_DIR / "literature_summary.txt").write_text(
        "Local literature note\\n\\n"
        "- paper_000 is directly relevant to scalable EM segmentation and proofreading.\\n"
        "- the other local papers are generic ML references rather than connectomics-specific merge predictors.\\n"
        "- the benchmark provides tabular engineered features, so a disciplined tabular classifier is the strongest local equivalent.\\n"
    )

    with open(OUTPUT_DIR / "summary_metrics.json", "w") as f:
        json.dump(
            {
                "model": "logistic_regression",
                "threshold": threshold,
                "cv_metrics": cv_metrics,
                "test_metrics": test_metrics,
                "by_degradation": by_deg,
            },
            f,
            indent=2,
        )

    print(json.dumps({"threshold": threshold, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
