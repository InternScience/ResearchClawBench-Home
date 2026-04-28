"""
03_bo_design.py — Surrogate-driven de-novo design of hydrogel formulations.

Pipeline:
  1) Use the full-data GP-Matern surrogate from 02_models.py.
  2) Sample N candidate compositions on the 6-simplex (Dirichlet draws +
     a vertex/edge schema; constrained to sum-to-one, x_i in [0,1]).
  3) For each candidate, predict mean μ(x) and std σ(x) of Glass adhesion
     using the GP, then evaluate the Expected Improvement (EI) acquisition
     function vs the best observed value y* in the training set.
  4) Rank candidates by EI and by predicted μ; report the top suggestions
     and write them to outputs/bo_suggestions.csv.
  5) Quantify what fraction of the simplex the surrogate predicts above
     several thresholds (incl. 1 MPa) — exposing how far the bio-inspired
     monomer space is from achieving robust >1 MPa underwater adhesion.
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)

MONOMERS = ["Nucleophilic-HEA","Hydrophobic-BA","Acidic-CBEA",
            "Cationic-ATAC","Aromatic-PEA","Amide-AAm"]
N_DIM = len(MONOMERS)

def sample_simplex(n, rng, alpha=1.0):
    """Dirichlet on the simplex."""
    return rng.dirichlet(alpha=np.full(N_DIM, alpha), size=n)

def main():
    train = pd.read_csv(os.path.join(OUT, "training_184_clean.csv"))
    Xtr = train[MONOMERS].values; ytr = train["Glass_kPa"].values
    y_best = ytr.max()

    with open(os.path.join(OUT, "full_models.pkl"), "rb") as fh:
        models = pickle.load(fh)
    gp = models["GP-Matern"]
    rf = models["RandomForest"]

    rng = np.random.default_rng(0)
    cand_blocks = []
    # broad sampling
    cand_blocks.append(sample_simplex(40000, rng, alpha=1.0))
    # focus near observed high-adhesion formulas (top-20 used as Dirichlet means)
    top = train.sort_values("Glass_kPa", ascending=False).head(20)[MONOMERS].values
    for r in top:
        # Dirichlet centered approximately on r (alpha = c*r + small)
        a = 30.0*r + 0.5
        cand_blocks.append(rng.dirichlet(a, size=1500))
    # vertices/edges
    eye = np.eye(N_DIM)
    cand_blocks.append(eye)
    for i in range(N_DIM):
        for j in range(i+1, N_DIM):
            for w in np.linspace(0.05,0.95,11):
                v = np.zeros(N_DIM); v[i]=w; v[j]=1-w
                cand_blocks.append(v[None,:])
    cand = np.vstack(cand_blocks)
    cand = np.clip(cand,0,1)
    cand = cand/cand.sum(axis=1,keepdims=True)

    # GP predictions with std
    mu, sigma = gp.predict(cand, return_std=True)
    # RF predictions for comparison
    mu_rf = rf.predict(cand)

    # Expected improvement (maximization)
    eps = 0.0
    impr = mu - y_best - eps
    z = np.zeros_like(impr)
    nonzero = sigma > 1e-9
    z[nonzero] = impr[nonzero] / sigma[nonzero]
    ei = np.where(nonzero, impr*norm.cdf(z) + sigma*norm.pdf(z), np.maximum(impr,0.0))

    df = pd.DataFrame(cand, columns=MONOMERS)
    df["mu_GP"]=mu; df["sigma_GP"]=sigma; df["mu_RF"]=mu_rf; df["EI"]=ei

    # near-duplicate suppression — quantize to 0.02 grid
    quant = (cand*50).round().astype(int)
    df["_key"] = [tuple(r) for r in quant]
    df_sorted = df.sort_values("EI", ascending=False).drop_duplicates("_key").drop(columns=["_key"])
    top_ei  = df_sorted.head(50).reset_index(drop=True)
    top_mu  = df.sort_values("mu_GP", ascending=False).drop_duplicates(subset=None).head(50).reset_index(drop=True)

    top_ei.to_csv(os.path.join(OUT, "bo_suggestions_topEI.csv"), index=False)
    top_mu.to_csv(os.path.join(OUT, "bo_suggestions_topMu.csv"), index=False)

    # threshold analysis
    thresholds = [50, 100, 200, 300, 500, 1000]
    frac_above = {t: float((mu>=t).mean()) for t in thresholds}
    frac_plus_2sigma = {t: float(((mu+2*sigma)>=t).mean()) for t in thresholds}
    summary = {
        "y_best_train_kPa": float(y_best),
        "n_candidates": int(len(cand)),
        "GP_mu_max": float(mu.max()),
        "GP_mu_p99": float(np.percentile(mu,99)),
        "GP_mu_plus_2sigma_max": float((mu+2*sigma).max()),
        "RF_mu_max": float(mu_rf.max()),
        "frac_GPmu_above_kPa": frac_above,
        "frac_GPmu_plus2sigma_above_kPa": frac_plus_2sigma,
        "ei_max": float(ei.max()),
        "best_top_EI_formula": top_ei.iloc[0][MONOMERS].to_dict(),
        "best_top_mu_formula": top_mu.iloc[0][MONOMERS].to_dict(),
    }
    with open(os.path.join(OUT, "bo_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    # ============== FIGURES ==============
    sns.set_theme(context="talk", style="whitegrid")

    # Fig: EI / μ landscape on dominant 2-D projection (PEA vs HEA)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.7))

    # (a) μ density on (PEA, HEA)
    ax = axes[0]
    sc = ax.scatter(cand[:,4], cand[:,0], c=mu, s=4, cmap="viridis",
                    vmin=0, vmax=np.percentile(mu,99))
    ax.scatter(Xtr[:,4], Xtr[:,0], c="white", edgecolor="k", lw=.5, s=20, label="training")
    ax.set_xlabel("Aromatic-PEA"); ax.set_ylabel("Nucleophilic-HEA")
    ax.set_title("(a) GP μ — Aromatic vs Nucleophilic")
    plt.colorbar(sc, ax=ax, label="μ Glass (kPa)")
    ax.legend(loc="upper right", fontsize=10)

    # (b) σ density
    ax = axes[1]
    sc = ax.scatter(cand[:,4], cand[:,0], c=sigma, s=4, cmap="magma",
                    vmin=0, vmax=np.percentile(sigma,99))
    ax.scatter(Xtr[:,4], Xtr[:,0], c="lime", edgecolor="k", lw=.4, s=15, label="training")
    ax.set_xlabel("Aromatic-PEA"); ax.set_ylabel("Nucleophilic-HEA")
    ax.set_title("(b) GP σ — uncertainty")
    plt.colorbar(sc, ax=ax, label="σ Glass (kPa)")
    ax.legend(loc="upper right", fontsize=10)

    # (c) EI landscape with top picks
    ax = axes[2]
    sc = ax.scatter(cand[:,4], cand[:,0], c=ei, s=4, cmap="cividis",
                    vmin=0, vmax=np.percentile(ei,99.5))
    ax.scatter(top_ei["Aromatic-PEA"], top_ei["Nucleophilic-HEA"],
               c="red", marker="*", s=110, edgecolor="white", lw=.5, label="top-EI")
    ax.set_xlabel("Aromatic-PEA"); ax.set_ylabel("Nucleophilic-HEA")
    ax.set_title("(c) Expected Improvement")
    plt.colorbar(sc, ax=ax, label="EI (kPa)")
    ax.legend(loc="upper right", fontsize=10)

    fig.suptitle("Surrogate-driven exploration of the 6-monomer simplex",
                 fontsize=16, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig05_bo_landscape.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig: distribution of GP μ and RF μ across candidates
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(mu, bins=60, alpha=.6, label="GP μ", color="#3b7dd8", density=True)
    ax.hist(mu_rf, bins=60, alpha=.5, label="RF μ", color="#dc8b3a", density=True)
    ax.axvline(y_best, color="green", ls="--", label=f"best train = {y_best:.0f} kPa")
    ax.axvline(1000, color="crimson", ls="--", label="1 MPa target")
    ax.set_xlabel("Predicted Glass adhesion (kPa)"); ax.set_ylabel("density")
    ax.set_title("Predicted-adhesion distribution across the simplex")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig06_pred_distribution.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig: top-EI formulas as stacked composition bars
    fig, ax = plt.subplots(figsize=(11,5.5))
    n_show = 20
    bottoms = np.zeros(n_show)
    palette = sns.color_palette("Set2", N_DIM)
    for i, m in enumerate(MONOMERS):
        ax.bar(range(n_show), top_ei[m].values[:n_show], bottom=bottoms,
               color=palette[i], label=m.split("-",1)[1])
        bottoms += top_ei[m].values[:n_show]
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"#{i+1}\nμ={top_ei['mu_GP'].iloc[i]:.0f}\n±{top_ei['sigma_GP'].iloc[i]:.0f}"
                        for i in range(n_show)], fontsize=8)
    ax.set_ylim(0,1)
    ax.set_ylabel("Mole fraction")
    ax.set_title("Top-20 EI candidate formulas (GP surrogate)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig07_top_ei_compositions.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
