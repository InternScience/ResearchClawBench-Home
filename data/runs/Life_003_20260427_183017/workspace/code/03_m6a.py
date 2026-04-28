"""m6A modification detection: PR/ROC comparison Uncalled4 vs Nanopolish via m6Anet."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve,
                             average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)

ROOT = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_003_20260427_183017"
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report/images")

labels = pd.read_csv(os.path.join(DATA, "m6a_labels.csv"))
unc   = pd.read_csv(os.path.join(DATA, "m6a_predictions_uncalled4.csv"))
nano  = pd.read_csv(os.path.join(DATA, "m6a_predictions_nanopolish.csv"))
print("labels:", labels.shape, "uncalled4:", unc.shape, "nanopolish:", nano.shape)
print("class balance:", labels.label.value_counts().to_dict())

merged = labels.merge(unc.rename(columns={"probability": "prob_uncalled4"}), on="site_id") \
                .merge(nano.rename(columns={"probability": "prob_nanopolish"}), on="site_id")
print("merged rows:", len(merged))
merged.to_csv(os.path.join(OUT, "m6a_merged.csv"), index=False)
y = merged.label.values

results = {}
for tool, col in [("Uncalled4", "prob_uncalled4"), ("Nanopolish", "prob_nanopolish")]:
    s = merged[col].values
    auroc = roc_auc_score(y, s)
    auprc = average_precision_score(y, s)
    # threshold 0.5 metrics
    pred05 = (s >= 0.5).astype(int)
    p05, r05, f105 = precision_score(y, pred05, zero_division=0), recall_score(y, pred05), f1_score(y, pred05, zero_division=0)
    # best F1
    prec, rec, thr = precision_recall_curve(y, s)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    bidx = int(np.nanargmax(f1s))
    best_thr = thr[bidx - 1] if bidx > 0 and bidx - 1 < len(thr) else 0.5
    results[tool] = {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        "precision@0.5": float(p05), "recall@0.5": float(r05), "f1@0.5": float(f105),
        "best_F1": float(f1s[bidx]),
        "best_F1_threshold": float(best_thr),
        "best_F1_precision": float(prec[bidx]),
        "best_F1_recall": float(rec[bidx]),
        "score_mean_pos": float(s[y == 1].mean()),
        "score_mean_neg": float(s[y == 0].mean()),
    }

print(json.dumps(results, indent=2))
with open(os.path.join(OUT, "m6a_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

# --- PR curves ---
fig, ax = plt.subplots(figsize=(6, 5))
for tool, col, color in [("Uncalled4", "prob_uncalled4", "#1f77b4"),
                         ("Nanopolish", "prob_nanopolish", "#d62728")]:
    prec, rec, _ = precision_recall_curve(y, merged[col].values)
    ap = results[tool]["AUPRC"]
    ax.plot(rec, prec, lw=2, color=color, label=f"{tool}  AP = {ap:.3f}")
baseline = y.mean()
ax.axhline(baseline, ls="--", color="grey", label=f"random (prevalence = {baseline:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("m6A detection — Precision-Recall (m6Anet probabilities)")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "m6a_pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- ROC curves ---
fig, ax = plt.subplots(figsize=(6, 5))
for tool, col, color in [("Uncalled4", "prob_uncalled4", "#1f77b4"),
                         ("Nanopolish", "prob_nanopolish", "#d62728")]:
    fpr, tpr, _ = roc_curve(y, merged[col].values)
    auc = results[tool]["AUROC"]
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{tool}  AUROC = {auc:.3f}")
ax.plot([0, 1], [0, 1], "--", color="grey", label="chance")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("m6A detection — ROC (m6Anet probabilities)")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "m6a_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Score distributions ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (tool, col) in zip(axes, [("Uncalled4", "prob_uncalled4"),
                                  ("Nanopolish", "prob_nanopolish")]):
    s = merged[col].values
    ax.hist(s[y == 0], bins=40, alpha=0.6, label="negative", color="#888")
    ax.hist(s[y == 1], bins=40, alpha=0.6, label="positive (m6A)", color="#d6604d")
    ax.set_title(f"{tool} score distribution")
    ax.set_xlabel("Predicted m6A probability")
    ax.set_ylabel("# sites")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(IMG, "m6a_score_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Calibration curve (reliability) ---
from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(5.5, 5))
for tool, col, color in [("Uncalled4", "prob_uncalled4", "#1f77b4"),
                         ("Nanopolish", "prob_nanopolish", "#d62728")]:
    frac_pos, mean_pred = calibration_curve(y, merged[col].values, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=color, label=tool)
ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
ax.set_xlabel("Mean predicted probability (per quantile bin)")
ax.set_ylabel("Empirical fraction positive")
ax.set_title("Reliability (calibration) of m6Anet probabilities")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "m6a_calibration.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("DONE m6a")
