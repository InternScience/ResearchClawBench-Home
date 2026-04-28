"""
01_data_overview.py
Load all three CSVs, characterize them, and save a JSON summary plus a data
overview figure (label class balance + label timeline).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = ROOT / "outputs"
IMG  = ROOT / "report" / "images"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

feat = pd.read_csv(DATA / "Together_1_features_extracted.csv")
tgt  = pd.read_csv(DATA / "Together_1_targets_inserted.csv")
ref  = pd.read_csv(DATA / "Together_1_machine_results_reference.csv")

# Body parts for the two mice
BODYPARTS = ["Nose", "Ear_left", "Ear_right", "Center", "Lat_left",
             "Lat_right", "Tail_base", "Tail_end"]
ANIMALS = [1, 2]

# Sanity counts
summary = {
    "features_extracted": {
        "shape": list(feat.shape),
        "columns_first10": feat.columns[:10].tolist(),
        "columns_last5": feat.columns[-5:].tolist(),
    },
    "targets_inserted": {
        "shape": list(tgt.shape),
        "labels": ["Attack", "Sniffing"],
        "Attack_counts": {int(k): int(v) for k, v in tgt["Attack"].value_counts().to_dict().items()},
        "Sniffing_counts": {int(k): int(v) for k, v in tgt["Sniffing"].value_counts().to_dict().items()},
        "Attack_prevalence": float(tgt["Attack"].mean()),
        "Sniffing_prevalence": float(tgt["Sniffing"].mean()),
        "co_occurrence": int(((tgt["Attack"] == 1) & (tgt["Sniffing"] == 1)).sum()),
    },
    "machine_results_reference": {
        "shape": list(ref.shape),
        "n_engineered_features": int(ref.shape[1] - 6),  # minus pose? approx; tracked separately below
        "Probability_Attack_summary": {
            "mean": float(ref["Probability_Attack"].mean()),
            "std":  float(ref["Probability_Attack"].std()),
            "min":  float(ref["Probability_Attack"].min()),
            "max":  float(ref["Probability_Attack"].max()),
        },
        "Probability_Sniffing_summary": {
            "mean": float(ref["Probability_Sniffing"].mean()),
            "std":  float(ref["Probability_Sniffing"].std()),
            "min":  float(ref["Probability_Sniffing"].min()),
            "max":  float(ref["Probability_Sniffing"].max()),
        },
        "Attack_counts": {int(k): int(v) for k, v in ref["Attack"].value_counts().to_dict().items()},
        "Sniffing_counts": {int(k): int(v) for k, v in ref["Sniffing"].value_counts().to_dict().items()},
    },
    "bodyparts_used": BODYPARTS,
    "animals": ANIMALS,
}

# Tracker confidence (mean across all _p columns) for the main session
p_cols = [c for c in feat.columns if c.endswith("_p")]
summary["tracker_confidence_mean"] = float(feat[p_cols].mean().mean())
summary["tracker_confidence_per_bodypart"] = {
    c: float(feat[c].mean()) for c in p_cols
}

with open(OUT / "data_overview.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved outputs/data_overview.json")

# ---------- Figure 1: Data overview ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

# (a) class prevalence bar
ax = axes[0, 0]
labels = ["Attack", "Sniffing"]
pos = [tgt["Attack"].sum(), tgt["Sniffing"].sum()]
neg = [(tgt["Attack"] == 0).sum(), (tgt["Sniffing"] == 0).sum()]
x = np.arange(len(labels))
ax.bar(x - 0.2, neg, width=0.4, label="negative (0)", color="#888888")
ax.bar(x + 0.2, pos, width=0.4, label="positive (1)", color="#cc3344")
for i, (n, p) in enumerate(zip(neg, pos)):
    ax.text(i - 0.2, n, str(n), ha="center", va="bottom", fontsize=9)
    ax.text(i + 0.2, p, str(p), ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Number of frames"); ax.set_title("(a) Class balance (1738 frames)")
ax.legend()

# (b) tracker confidence per bodypart
ax = axes[0, 1]
mean_p = feat[p_cols].mean().sort_values()
ax.barh(np.arange(len(mean_p)), mean_p.values, color="#3366aa")
ax.set_yticks(np.arange(len(mean_p))); ax.set_yticklabels(mean_p.index, fontsize=7)
ax.set_xlabel("Mean tracking confidence")
ax.set_title("(b) Pose-tracker confidence per body part")
ax.set_xlim(0, 1.05)

# (c) Attack timeline
ax = axes[1, 0]
ax.plot(tgt.index, tgt["Attack"].values, color="#cc3344", lw=0.6)
ax.fill_between(tgt.index, 0, tgt["Attack"].values, color="#cc3344", alpha=0.4)
ax.set_xlabel("Frame"); ax.set_ylabel("Label"); ax.set_ylim(-0.05, 1.05)
ax.set_title("(c) Attack ground-truth ethogram")

# (d) Sniffing timeline
ax = axes[1, 1]
ax.plot(tgt.index, tgt["Sniffing"].values, color="#3388aa", lw=0.6)
ax.fill_between(tgt.index, 0, tgt["Sniffing"].values, color="#3388aa", alpha=0.4)
ax.set_xlabel("Frame"); ax.set_ylabel("Label"); ax.set_ylim(-0.05, 1.05)
ax.set_title("(d) Sniffing ground-truth ethogram")

plt.tight_layout()
plt.savefig(IMG / "01_data_overview.png", dpi=140)
plt.close()
print("Saved report/images/01_data_overview.png")

# ---------- Figure 2: Label timeline (combined, with co-occurrence) ----------
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.fill_between(tgt.index, 1.05, 1.95, where=tgt["Attack"] == 1,
                color="#cc3344", alpha=0.85, step="pre", label="Attack")
ax.fill_between(tgt.index, 0.05, 0.95, where=tgt["Sniffing"] == 1,
                color="#3388aa", alpha=0.85, step="pre", label="Sniffing")
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Sniffing", "Attack"])
ax.set_xlabel("Frame index")
ax.set_title("Frame-aligned annotation ethograms (positive bouts shaded)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(IMG / "02_label_timeline.png", dpi=140)
plt.close()
print("Saved report/images/02_label_timeline.png")
