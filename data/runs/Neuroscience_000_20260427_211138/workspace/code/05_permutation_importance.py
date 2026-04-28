"""
05_permutation_importance.py
Compute scikit-learn permutation importances for the Random Forest classifier
on a held-out fold (last-fold of stratified split) for both behaviors. This is
a post-hoc interpretability artifact independent of impurity-based importance.

Outputs:
  - outputs/perm_importance_<behavior>.csv
  - report/images/09_permutation_importance.png
  - outputs/feature_group_importance.csv (impurity importance aggregated)
  - report/images/10_feature_group_importance.png
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
IMG  = ROOT / "report" / "images"

df = pd.read_csv(OUT / "engineered_features.csv")
y_attack   = df["__Attack__"].values.astype(int)
y_sniffing = df["__Sniffing__"].values.astype(int)
X = df.drop(columns=["__Attack__", "__Sniffing__"])
feat_names = X.columns.tolist()
X_arr = X.values

# Group mapping
def group_of(name):
    if name.startswith("prob_"): return "probabilities"
    if name.startswith("d_inter_"): return "inter-animal distance"
    if name.startswith("d_cross_"): return "inter-animal cross-distance"
    if name.startswith("d_") and name.endswith(("_a1", "_a2")): return "within-animal distance"
    if name.startswith("vel_"): return "velocity"
    if name.startswith("acc_"): return "acceleration"
    if name.startswith("bbox_"): return "bbox / size"
    if name.startswith("angle_") or name.startswith("angvel_") or name.startswith("rel_angle"): return "angle"
    if name.startswith("inter_center"): return "inter-center kinematics"
    if name.endswith("_rmean15") or name.endswith("_rstd15"): return "rolling stats"
    return "other"

groups = pd.Series([group_of(n) for n in feat_names], name="group")

RF_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
                 max_features="sqrt", n_jobs=-1, random_state=42,
                 class_weight="balanced")

fig, axes = plt.subplots(1, 2, figsize=(13, 8))
COLORS = {"attack": "#cc3344", "sniffing": "#3388aa"}
all_perm = {}
for ax, (b, y) in zip(axes, [("attack", y_attack), ("sniffing", y_sniffing)]):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X_arr, y))
    tr, te = splits[-1]   # last fold
    m = RandomForestClassifier(**RF_PARAMS)
    m.fit(X_arr[tr], y[tr])
    pi = permutation_importance(m, X_arr[te], y[te],
                                n_repeats=10, random_state=42,
                                scoring="average_precision", n_jobs=-1)
    pdf = (pd.DataFrame({"feature": feat_names,
                         "perm_mean": pi.importances_mean,
                         "perm_std":  pi.importances_std})
           .sort_values("perm_mean", ascending=False)
           .reset_index(drop=True))
    pdf.to_csv(OUT / f"perm_importance_{b}.csv", index=False)
    all_perm[b] = pdf

    top = pdf.head(20).iloc[::-1]
    ax.barh(np.arange(len(top)), top["perm_mean"].values,
            xerr=top["perm_std"].values,
            color=COLORS[b], edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.set_xlabel("Permutation Δ Average Precision (mean ± std, n=10)")
    ax.set_title(f"Permutation importance — {b.capitalize()} (held-out fold)")

plt.tight_layout()
plt.savefig(IMG / "09_permutation_importance.png", dpi=140)
plt.close()

# Aggregated impurity importance per feature group
group_rows = []
for b in ["attack", "sniffing"]:
    imp = pd.read_csv(OUT / f"feature_importance_{b}.csv")
    imp["group"] = imp["feature"].map(group_of)
    g = imp.groupby("group")["importance"].sum().reset_index()
    g["behavior"] = b
    group_rows.append(g)
grp = pd.concat(group_rows, ignore_index=True)
grp.to_csv(OUT / "feature_group_importance.csv", index=False)

pivot = grp.pivot(index="group", columns="behavior", values="importance").fillna(0)
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

fig, ax = plt.subplots(figsize=(9, 5.5))
y_pos = np.arange(len(pivot))
ax.barh(y_pos - 0.2, pivot["attack"].values, height=0.4,
        color=COLORS["attack"], label="Attack", edgecolor="black", linewidth=0.4)
ax.barh(y_pos + 0.2, pivot["sniffing"].values, height=0.4,
        color=COLORS["sniffing"], label="Sniffing", edgecolor="black", linewidth=0.4)
ax.set_yticks(y_pos); ax.set_yticklabels(pivot.index)
ax.set_xlabel("Summed RF impurity importance (5-fold CV mean)")
ax.set_title("Feature-group contribution to each behavior classifier")
ax.legend()
plt.tight_layout()
plt.savefig(IMG / "10_feature_group_importance.png", dpi=140)
plt.close()

print("Saved permutation importance and feature-group importance artifacts.")
