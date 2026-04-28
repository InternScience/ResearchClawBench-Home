"""
04_diagnostic_figures.py
Generate publication-style diagnostic figures from the saved metrics and
predictions:
  - 03_confusion_matrices.png
  - 04_pr_roc_curves.png
  - 05_feature_importance.png
  - 06_probability_timeseries.png
  - 07_model_comparison.png
  - 08_reference_probability_comparison.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_recall_curve, roc_curve,
                             confusion_matrix, average_precision_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
IMG  = ROOT / "report" / "images"
DATA = ROOT / "data"

sns.set_context("notebook")

BEHAVIORS = ["attack", "sniffing"]
COLORS = {"attack": "#cc3344", "sniffing": "#3388aa"}

preds = {b: pd.read_csv(OUT / f"predictions_{b}.csv") for b in BEHAVIORS}
metrics = {b: json.load(open(OUT / f"metrics_{b}.json")) for b in BEHAVIORS}
imps = {b: pd.read_csv(OUT / f"feature_importance_{b}.csv") for b in BEHAVIORS}

# ---------- Figure 3: Confusion matrices (RF, default 0.5 and best F1) ----------
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for col, b in enumerate(BEHAVIORS):
    p = preds[b]
    y = p["y_true"].values
    rf_p = p["rf_prob"].values
    best_thr = metrics[b]["models"]["RandomForest_CV_best_threshold"]["best_threshold"]
    for row, (thr, label) in enumerate([(0.5, "default thr=0.5"),
                                        (best_thr, f"best F1 thr={best_thr:.2f}")]):
        cm = confusion_matrix(y, (rf_p >= thr).astype(int))
        ax = axes[row, col]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds" if b == "attack" else "Blues",
                    cbar=False, ax=ax,
                    xticklabels=["pred 0", "pred 1"],
                    yticklabels=["true 0", "true 1"])
        ax.set_title(f"{b.capitalize()} — RF CV ({label})")
plt.tight_layout()
plt.savefig(IMG / "03_confusion_matrices.png", dpi=140)
plt.close()

# ---------- Figure 4: PR + ROC curves for both behaviors and both models ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for col, b in enumerate(BEHAVIORS):
    p = preds[b]; y = p["y_true"].values
    # PR
    ax = axes[0, col]
    for label, prob, ls in [("RF", p["rf_prob"].values, "-"),
                            ("GB", p["gb_prob"].values, "--")]:
        pr, rc, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(rc, pr, ls, label=f"{label} (AP={ap:.3f})")
    ax.axhline(y.mean(), color="gray", lw=0.7, ls=":",
               label=f"prevalence={y.mean():.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR — {b.capitalize()}"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    # ROC
    ax = axes[1, col]
    for label, prob, ls in [("RF", p["rf_prob"].values, "-"),
                            ("GB", p["gb_prob"].values, "--")]:
        fpr, tpr, _ = roc_curve(y, prob)
        au = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, ls, label=f"{label} (AUC={au:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.7, ls=":")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {b.capitalize()}")
    ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(IMG / "04_pr_roc_curves.png", dpi=140)
plt.close()

# ---------- Figure 5: Top-25 feature importances per behavior ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 8))
for ax, b in zip(axes, BEHAVIORS):
    top = imps[b].head(25).iloc[::-1]
    ax.barh(np.arange(len(top)), top["importance"].values,
            color=COLORS[b], edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.set_xlabel("Mean RF impurity importance (5-fold CV)")
    ax.set_title(f"Top-25 features — {b.capitalize()}")
plt.tight_layout()
plt.savefig(IMG / "05_feature_importance.png", dpi=140)
plt.close()

# ---------- Figure 6: Probability time-series vs ground truth ----------
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
for ax, b in zip(axes, BEHAVIORS):
    p = preds[b]
    ax.plot(p["frame"], p["rf_prob"], color="black", lw=0.6, label="RF probability (OOF)")
    ax.fill_between(p["frame"], 0, p["y_true"], step="pre",
                    color=COLORS[b], alpha=0.25, label="Ground-truth bouts")
    best_thr = metrics[b]["models"]["RandomForest_CV_best_threshold"]["best_threshold"]
    ax.axhline(0.5, color="gray", ls=":", lw=0.6, label="thr=0.5")
    ax.axhline(best_thr, color="orange", ls="--", lw=0.6, label=f"best-F1 thr={best_thr:.2f}")
    ax.set_ylim(-0.02, 1.05); ax.set_ylabel(f"P({b})")
    ax.set_title(f"{b.capitalize()} — RF CV-OOF probability vs ground-truth ethogram")
    ax.legend(loc="upper right", ncols=4, fontsize=8)
axes[-1].set_xlabel("Frame index")
plt.tight_layout()
plt.savefig(IMG / "06_probability_timeseries.png", dpi=140)
plt.close()

# ---------- Figure 7: Model comparison bar chart ----------
rows = []
for b in BEHAVIORS:
    m = metrics[b]["models"]
    rows.append({"behavior": b, "model": "RF (5-fold CV)",
                 "ROC-AUC": m["RandomForest_CV"]["roc_auc"],
                 "PR-AUC":  m["RandomForest_CV"]["pr_auc"],
                 "F1@0.5":  m["RandomForest_CV"]["f1"]})
    rows.append({"behavior": b, "model": "GB (5-fold CV)",
                 "ROC-AUC": m["GradientBoosting_CV"]["roc_auc"],
                 "PR-AUC":  m["GradientBoosting_CV"]["pr_auc"],
                 "F1@0.5":  m["GradientBoosting_CV"]["f1"]})
    rows.append({"behavior": b, "model": "RF (chrono hold-out)",
                 "ROC-AUC": m["RandomForest_chronological_holdout"]["roc_auc"],
                 "PR-AUC":  m["RandomForest_chronological_holdout"]["pr_auc"],
                 "F1@0.5":  m["RandomForest_chronological_holdout"]["f1"]})
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(OUT / "model_comparison.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
metrics_to_show = ["ROC-AUC", "PR-AUC", "F1@0.5"]
for ax, m_name in zip(axes, metrics_to_show):
    sub = cmp_df.pivot(index="model", columns="behavior", values=m_name)
    sub.plot(kind="bar", ax=ax, color=[COLORS[c] for c in sub.columns],
             edgecolor="black", linewidth=0.4)
    ax.set_title(m_name); ax.set_ylim(0, 1.0)
    ax.set_ylabel(m_name); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    for p_ in ax.patches:
        h = p_.get_height()
        if not np.isnan(h):
            ax.text(p_.get_x()+p_.get_width()/2, h+0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(IMG / "07_model_comparison.png", dpi=140)
plt.close()

# ---------- Figure 8: Reference vs reproduced probability distributions ----------
ref = pd.read_csv(DATA / "Together_1_machine_results_reference.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, b in zip(axes, BEHAVIORS):
    p = preds[b]
    ax.hist(p["rf_prob"], bins=30, alpha=0.6,
            color=COLORS[b], label="Reproduced RF (CV-OOF, 1738 frames)",
            density=True, edgecolor="black", linewidth=0.4)
    ax.hist(ref[f"Probability_{b.capitalize()}"], bins=30, alpha=0.5,
            color="gray", label="SimBA reference (300 frames)",
            density=True, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(f"P({b})")
    ax.set_ylabel("Density")
    ax.set_title(f"{b.capitalize()} probability distribution: reproduced vs reference")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG / "08_reference_probability_comparison.png", dpi=140)
plt.close()

print("Saved all diagnostic figures to report/images/")
