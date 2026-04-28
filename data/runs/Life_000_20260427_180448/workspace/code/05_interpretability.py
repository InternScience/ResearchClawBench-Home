"""
05_interpretability.py — permutation importance + partial-dependence
artifacts and a leave-one-feature-out ablation for the top-2 surrogates.
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")

MONOMERS = ["Nucleophilic-HEA","Hydrophobic-BA","Acidic-CBEA",
            "Cationic-ATAC","Aromatic-PEA","Amide-AAm"]
SHORT = [m.split("-",1)[1] for m in MONOMERS]

def main():
    df = pd.read_csv(os.path.join(OUT,"training_184_clean.csv"))
    X = df[MONOMERS].copy(); y = df["Glass_kPa"].copy()
    with open(os.path.join(OUT,"full_models.pkl"),"rb") as fh:
        models = pickle.load(fh)

    # Permutation importance for GP and RF on the training set
    rng = np.random.default_rng(0)
    perm_results = {}
    for name in ["GP-Matern","RandomForest","GradientBoost"]:
        m = models[name]
        r = permutation_importance(m, X, y, n_repeats=30, random_state=0, n_jobs=1,
                                    scoring="r2")
        perm_results[name] = {
            "mean": r.importances_mean.tolist(),
            "std":  r.importances_std.tolist(),
            "features": MONOMERS,
        }

    with open(os.path.join(OUT,"permutation_importance.json"),"w") as fh:
        json.dump(perm_results, fh, indent=2)

    # Bar fig
    sns.set_theme(context="talk", style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    for ax,(name,res) in zip(axes, perm_results.items()):
        order = np.argsort(res["mean"])
        ax.barh(np.array(SHORT)[order],
                np.array(res["mean"])[order],
                xerr=np.array(res["std"])[order],
                color="#3b7dd8", edgecolor="k")
        ax.set_title(f"{name}")
        ax.set_xlabel("Δ R² when feature is permuted")
    fig.suptitle("Permutation importance (training set, 30 repeats)", fontsize=15, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig11_permutation_importance.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Partial dependence — RF (fast)
    rf = models["RandomForest"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, idx in zip(axes.flat, range(len(MONOMERS))):
        pd_res = partial_dependence(rf, X, [idx], grid_resolution=40, kind="average")
        # sklearn returns dict-like
        grid = pd_res["grid_values"][0] if "grid_values" in pd_res else pd_res["values"][0]
        avg = pd_res["average"][0]
        ax.plot(grid, avg, lw=2, color="#3b7dd8")
        ax.fill_between(grid, avg, alpha=.2, color="#3b7dd8")
        ax.set_xlabel(SHORT[idx])
        ax.set_ylabel("PD on Glass kPa")
        ax.set_title(SHORT[idx])
    fig.suptitle("Partial-dependence curves (RandomForest)", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig12_partial_dependence.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Leave-one-feature-out ablation on RF (5-fold CV mean R²)
    base_r2 = []
    drop_r2 = {m:[] for m in MONOMERS}
    for s in range(3):
        kf = KFold(5, shuffle=True, random_state=s)
        for tr, te in kf.split(X):
            full = clone(rf).fit(X.iloc[tr], y.iloc[tr])
            base_r2.append(r2_score(y.iloc[te], full.predict(X.iloc[te])))
            for m in MONOMERS:
                Xtr2 = X.iloc[tr].drop(columns=[m])
                Xte2 = X.iloc[te].drop(columns=[m])
                m2 = clone(rf).fit(Xtr2, y.iloc[tr])
                drop_r2[m].append(r2_score(y.iloc[te], m2.predict(Xte2)))
    abl = pd.DataFrame({
        "feature_dropped": MONOMERS,
        "mean_r2_drop": [np.mean(drop_r2[m]) for m in MONOMERS],
        "delta_vs_full": [np.mean(base_r2)-np.mean(drop_r2[m]) for m in MONOMERS],
    }).sort_values("delta_vs_full", ascending=False)
    abl["base_r2"] = np.mean(base_r2)
    abl.to_csv(os.path.join(OUT, "ablation_dropfeature.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.barh([SHORT[MONOMERS.index(m)] for m in abl["feature_dropped"]],
            abl["delta_vs_full"], color="#dc8b3a", edgecolor="k")
    ax.set_xlabel("R² drop when feature is removed (RandomForest)")
    ax.set_title(f"Leave-one-feature-out ablation (full R²={abl['base_r2'].iloc[0]:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig13_loo_ablation.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("interpretability done")

if __name__ == "__main__":
    main()
