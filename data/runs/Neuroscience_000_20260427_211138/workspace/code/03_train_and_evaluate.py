"""
03_train_and_evaluate.py
Train SimBA-style supervised classifiers (Random Forest as the primary
estimator, Gradient Boosting as comparison) for both Attack and Sniffing.

Two evaluation regimes:
  (a) Stratified K-fold cross-validation (5-fold) — primary, comparable to
      SimBA's train/test reporting.
  (b) Single chronological train/test hold-out (last 30% of frames) — gives
      a temporally honest estimate that respects bout structure.

Saves:
  - outputs/metrics_<behavior>.json
  - outputs/predictions_<behavior>.csv (per-frame OOF probabilities)
  - outputs/feature_importance_<behavior>.csv  (mean over CV folds, RF)
  - outputs/cv_fold_metrics_<behavior>.csv
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score)

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"

df = pd.read_csv(OUT / "engineered_features.csv")
y_attack   = df["__Attack__"].values.astype(int)
y_sniffing = df["__Sniffing__"].values.astype(int)
X = df.drop(columns=["__Attack__", "__Sniffing__"])
feature_names = X.columns.tolist()
X_arr = X.values

print(f"X: {X_arr.shape}, attack pos rate {y_attack.mean():.3f}, "
      f"sniffing pos rate {y_sniffing.mean():.3f}")

RF_PARAMS = dict(n_estimators=400, max_depth=None, min_samples_leaf=2,
                 max_features="sqrt", n_jobs=-1, random_state=42,
                 class_weight="balanced")
GB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                 random_state=42)


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_cv(model_factory, X_arr, y, n_splits=5, seed=42, name="model"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y), dtype=float)
    fold_records = []
    importances = np.zeros(X_arr.shape[1])
    for fi, (tr, te) in enumerate(skf.split(X_arr, y)):
        m = model_factory()
        m.fit(X_arr[tr], y[tr])
        prob = m.predict_proba(X_arr[te])[:, 1]
        oof_prob[te] = prob
        rec = evaluate(y[te], prob, threshold=0.5)
        rec["fold"] = fi
        rec["model"] = name
        fold_records.append(rec)
        if hasattr(m, "feature_importances_"):
            importances += m.feature_importances_
    importances /= n_splits
    return oof_prob, fold_records, importances


def best_threshold_by_f1(y, prob):
    p, r, t = precision_recall_curve(y, prob)
    # t has length len(p)-1; align
    f1s = 2 * p * r / (p + r + 1e-12)
    if len(t) == 0:
        return 0.5, float(f1s.max() if len(f1s) else 0.0)
    idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    return float(t[idx]), float(f1s[idx])


def chronological_holdout(model_factory, X_arr, y, frac=0.3):
    n = len(y)
    cut = int(n * (1 - frac))
    m = model_factory()
    m.fit(X_arr[:cut], y[:cut])
    prob = m.predict_proba(X_arr[cut:])[:, 1]
    return cut, prob


for behavior, y in [("attack", y_attack), ("sniffing", y_sniffing)]:
    print(f"\n=== Behavior: {behavior} ===")
    rf_factory = lambda: RandomForestClassifier(**RF_PARAMS)
    gb_factory = lambda: GradientBoostingClassifier(**GB_PARAMS)

    rf_prob, rf_folds, rf_imp = run_cv(rf_factory, X_arr, y, name="RandomForest")
    gb_prob, gb_folds, _      = run_cv(gb_factory, X_arr, y, name="GradientBoosting")

    rf_overall = evaluate(y, rf_prob, threshold=0.5)
    gb_overall = evaluate(y, gb_prob, threshold=0.5)

    rf_best_thr, rf_best_f1 = best_threshold_by_f1(y, rf_prob)
    rf_at_best = evaluate(y, rf_prob, threshold=rf_best_thr)

    cut, holdout_prob = chronological_holdout(rf_factory, X_arr, y, frac=0.3)
    rf_holdout = evaluate(y[cut:], holdout_prob, threshold=0.5)

    metrics = {
        "behavior": behavior,
        "n_frames": int(len(y)),
        "n_pos": int(int(y.sum())),
        "n_neg": int(int((y == 0).sum())),
        "models": {
            "RandomForest_CV": rf_overall,
            "RandomForest_CV_best_threshold": {
                "best_threshold": rf_best_thr,
                "best_f1": rf_best_f1,
                **rf_at_best,
            },
            "GradientBoosting_CV": gb_overall,
            "RandomForest_chronological_holdout": {
                "train_end_idx": int(cut),
                "test_n": int(len(y) - cut),
                **rf_holdout,
            },
        },
        "fold_records": rf_folds + gb_folds,
        "n_features": int(X_arr.shape[1]),
    }
    with open(OUT / f"metrics_{behavior}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "frame": np.arange(len(y)),
        "y_true": y,
        "rf_prob": rf_prob,
        "gb_prob": gb_prob,
        "rf_pred_thr0p5": (rf_prob >= 0.5).astype(int),
        "rf_pred_best_thr": (rf_prob >= rf_best_thr).astype(int),
        "gb_pred_thr0p5": (gb_prob >= 0.5).astype(int),
    })
    pred_df.to_csv(OUT / f"predictions_{behavior}.csv", index=False)

    imp_df = (pd.DataFrame({"feature": feature_names, "importance": rf_imp})
              .sort_values("importance", ascending=False).reset_index(drop=True))
    imp_df.to_csv(OUT / f"feature_importance_{behavior}.csv", index=False)

    fold_df = pd.DataFrame(rf_folds + gb_folds)
    fold_df.to_csv(OUT / f"cv_fold_metrics_{behavior}.csv", index=False)

    print(f"RF CV  ROC-AUC={rf_overall['roc_auc']:.3f}  PR-AUC={rf_overall['pr_auc']:.3f}  "
          f"F1@0.5={rf_overall['f1']:.3f}  best F1={rf_best_f1:.3f} @ thr={rf_best_thr:.3f}")
    print(f"GB CV  ROC-AUC={gb_overall['roc_auc']:.3f}  PR-AUC={gb_overall['pr_auc']:.3f}  "
          f"F1@0.5={gb_overall['f1']:.3f}")
    print(f"RF chronological hold-out (last 30%): F1={rf_holdout['f1']:.3f} "
          f"PR-AUC={rf_holdout['pr_auc']}")

print("\nSaved metrics_*.json, predictions_*.csv, feature_importance_*.csv, cv_fold_metrics_*.csv.")
