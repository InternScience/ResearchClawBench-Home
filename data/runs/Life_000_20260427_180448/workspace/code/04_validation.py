"""
04_validation.py — Validate the surrogate against the multi-round
experimental SMBO results, and produce trajectory + parity diagnostics.
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")
DATA = os.path.join(ROOT, "data")

MONOMERS = ["Nucleophilic-HEA","Hydrophobic-BA","Acidic-CBEA",
            "Cationic-ATAC","Aromatic-PEA","Amide-AAm"]

def main():
    # rounds dataset
    df = pd.read_excel(os.path.join(DATA,"ML_ei&pred (1&2&3rounds)_20240408.xlsx"),
                       sheet_name="EI")
    df["ML"] = df["ML"].ffill()
    df["Glass (kPa)_max"] = pd.to_numeric(df["Glass (kPa)_max"], errors="coerce")
    df = df.dropna(subset=MONOMERS+["Glass (kPa)_max"]).copy()
    for m in MONOMERS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    # compositions in some rows are like 0.519 or close to 0; ensure they sum~1
    s = df[MONOMERS].sum(axis=1)
    print("Composition sum stats:", s.describe())
    # round assignment from the ML method label
    def round_of(label):
        if "3rd" in label: return 3
        if "2rd" in label or "2nd" in label: return 2
        return 1
    df["round"] = df["ML"].apply(round_of)

    # load surrogate
    with open(os.path.join(OUT,"full_models.pkl"),"rb") as fh:
        models = pickle.load(fh)
    gp = models["GP-Matern"]; rf = models["RandomForest"]

    X = df[MONOMERS].values
    df["pred_GP"]  = gp.predict(X)
    df["pred_RF"]  = rf.predict(X)
    df.to_csv(os.path.join(OUT,"rounds_predicted.csv"), index=False)

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    metrics = {}
    for m_name in ["pred_GP","pred_RF"]:
        metrics[m_name] = {
            "r2":   float(r2_score(df["Glass (kPa)_max"], df[m_name])),
            "mae":  float(mean_absolute_error(df["Glass (kPa)_max"], df[m_name])),
            "rmse": float(np.sqrt(mean_squared_error(df["Glass (kPa)_max"], df[m_name]))),
            "spearman": float(pd.Series(df["Glass (kPa)_max"]).corr(df[m_name], method="spearman")),
            "pearson":  float(pd.Series(df["Glass (kPa)_max"]).corr(df[m_name], method="pearson")),
        }
    with open(os.path.join(OUT,"validation_rounds_metrics.json"),"w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))

    # trajectory: best-so-far measured per round
    sns.set_theme(context="talk", style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.7))

    # (a) violin per round
    ax = axes[0]
    sns.boxplot(data=df, x="round", y="Glass (kPa)_max", ax=ax,
                palette="Set2", showfliers=False, width=.55)
    sns.stripplot(data=df, x="round", y="Glass (kPa)_max", ax=ax,
                  color="black", alpha=.5, size=4)
    ax.axhline(1000, color="crimson", ls="--", label="1 MPa target")
    ax.axhline(304.6, color="grey", ls=":", label="best in 184-train")
    ax.set_title("(a) Measured adhesion across SMBO rounds")
    ax.set_ylabel("Glass adhesion (kPa)"); ax.legend(fontsize=9)

    # (b) parity: surrogate prediction vs measured on rounds suggestions
    ax = axes[1]
    cmap = {1:"#3b7dd8", 2:"#dc8b3a", 3:"#4caf72"}
    for r in [1,2,3]:
        sub = df[df["round"]==r]
        ax.scatter(sub["Glass (kPa)_max"], sub["pred_GP"],
                   c=cmap[r], label=f"round {r}", s=45, edgecolor="k", lw=.3, alpha=.8)
    lo, hi = 0, max(df["Glass (kPa)_max"].max(), df["pred_GP"].max())*1.05
    ax.plot([lo,hi],[lo,hi],"--",color="grey")
    ax.set_xlabel("Measured Glass adhesion (kPa)")
    ax.set_ylabel("GP prediction trained on 184 (kPa)")
    ax.set_title(f"(b) Surrogate vs measurement on rounds  (R²={metrics['pred_GP']['r2']:.2f})")
    ax.legend(fontsize=9)

    fig.suptitle("Validation against multi-round wet-lab SMBO data", fontsize=16, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig08_round_validation.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Per-strategy summary
    strat = df.groupby("ML").agg(
        n=("Glass (kPa)_max","count"),
        max=("Glass (kPa)_max","max"),
        mean=("Glass (kPa)_max","mean"),
        median=("Glass (kPa)_max","median"))
    strat = strat.sort_values("max", ascending=False)
    strat.to_csv(os.path.join(OUT,"per_strategy_summary.csv"))

    # Fig: per-strategy max bar chart
    fig, ax = plt.subplots(figsize=(13,5))
    bars = ax.bar(strat.index, strat["max"], color=sns.color_palette("Set3", len(strat)),
                  edgecolor="k")
    ax.axhline(1000, color="crimson", ls="--", label="1 MPa")
    ax.axhline(304.6, color="grey", ls=":", label="train best")
    ax.set_ylabel("Best measured Glass adhesion (kPa)")
    ax.set_title("Best wet-lab adhesion per BO/SMBO strategy")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig09_per_strategy_best.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Combined progress chart: cumulative max measured vs index
    df_sorted = df.sort_values(["round"]).reset_index(drop=True)
    df_sorted["cum_max"] = df_sorted["Glass (kPa)_max"].cummax()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1,len(df_sorted)+1), df_sorted["cum_max"], "-", lw=2, color="#3b7dd8",
            label="cumulative max measured")
    ax.scatter(range(1,len(df_sorted)+1), df_sorted["Glass (kPa)_max"],
               c=df_sorted["round"].map(cmap), s=25, edgecolor="k", lw=.3,
               label="individual measurements")
    ax.axhline(1000, color="crimson", ls="--", label="1 MPa target")
    ax.axhline(304.6, color="grey", ls=":", label="train best")
    ax.set_xlabel("Suggestion index (rounds 1→3)")
    ax.set_ylabel("Glass adhesion (kPa)")
    ax.set_title("Best-so-far adhesion across the SMBO campaign")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig10_progress_curve.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Save report numbers
    with open(os.path.join(OUT,"validation_report_numbers.json"),"w") as fh:
        json.dump({
            "rounds_total": int(len(df)),
            "rounds_max": float(df["Glass (kPa)_max"].max()),
            "rounds_mean": float(df["Glass (kPa)_max"].mean()),
            "rounds_above_1MPa": int((df["Glass (kPa)_max"]>=1000).sum()),
            "metrics_GP": metrics["pred_GP"],
            "metrics_RF": metrics["pred_RF"],
            "best_strategy": strat.index[0],
            "best_strategy_max_kPa": float(strat["max"].iloc[0]),
        }, fh, indent=2)

if __name__ == "__main__":
    main()
