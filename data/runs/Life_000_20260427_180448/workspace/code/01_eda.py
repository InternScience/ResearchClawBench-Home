"""
01_eda.py — Exploratory data analysis for the bio-inspired hydrogel datasets.

Saves a multi-panel data overview figure plus a CSV summary of distributions.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)

MONOMERS = ["Nucleophilic-HEA","Hydrophobic-BA","Acidic-CBEA",
            "Cationic-ATAC","Aromatic-PEA","Amide-AAm"]

def load_184():
    df = pd.read_excel(os.path.join(DATA, "184_verified_Original Data_ML_20230926.xlsx"),
                       sheet_name="Data_to_HU")
    # primary target: max of 10s/60s on glass; only 10s is non-null in 184 set
    df["Glass_kPa"] = df[["Glass (kPa)_10s","Glass (kPa)_60s"]].max(axis=1)
    return df

def load_rounds():
    df = pd.read_excel(os.path.join(DATA, "ML_ei&pred (1&2&3rounds)_20240408.xlsx"),
                       sheet_name="EI")
    df["ML"] = df["ML"].ffill()
    df["Glass (kPa)_max"] = pd.to_numeric(df["Glass (kPa)_max"], errors="coerce")
    return df

def main():
    df184 = load_184()
    dfr   = load_rounds()
    df184.to_csv(os.path.join(OUT, "training_184_clean.csv"), index=False)
    dfr.to_csv(os.path.join(OUT, "rounds_EI_results.csv"), index=False)

    # quick stats
    summary = {
        "n_train": int(len(df184)),
        "max_Glass_kPa_train": float(df184["Glass_kPa"].max()),
        "median_Glass_kPa_train": float(df184["Glass_kPa"].median()),
        "n_rounds_records": int(len(dfr)),
        "max_Glass_kPa_rounds": float(dfr["Glass (kPa)_max"].max()),
        "n_rounds_above_1MPa": int((dfr["Glass (kPa)_max"]>=1000).sum()),
        "n_rounds_above_300kPa": int((dfr["Glass (kPa)_max"]>=300).sum()),
        "round_groups": list(dfr["ML"].unique()),
    }
    with open(os.path.join(OUT, "data_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    # === Figure: data overview ===
    sns.set_theme(context="talk", style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) target distribution training
    ax = axes[0,0]
    ax.hist(df184["Glass_kPa"], bins=30, color="#3b7dd8", edgecolor="white")
    ax.set_xlabel("Glass adhesion (kPa)")
    ax.set_ylabel("Count")
    ax.set_title(f"(a) Training set (n={len(df184)})")
    ax.axvline(1000, color="crimson", ls="--", lw=1.5, label="1 MPa target")
    ax.legend()

    # (b) target distribution rounds
    ax = axes[0,1]
    ax.hist(dfr["Glass (kPa)_max"].dropna(), bins=30, color="#dc8b3a", edgecolor="white")
    ax.set_xlabel("Glass adhesion (kPa)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Multi-round suggestions (measured)")
    ax.axvline(1000, color="crimson", ls="--", lw=1.5, label="1 MPa target")
    ax.legend()

    # (c) monomer composition statistics (training)
    ax = axes[0,2]
    comp_stats = df184[MONOMERS].describe().T[["mean","std"]]
    bars = ax.bar(range(len(MONOMERS)), comp_stats["mean"],
                  yerr=comp_stats["std"], color="#4caf72", capsize=4)
    ax.set_xticks(range(len(MONOMERS)))
    ax.set_xticklabels([m.split("-",1)[1] for m in MONOMERS], rotation=30, ha="right")
    ax.set_ylabel("Mole fraction")
    ax.set_title("(c) Monomer composition (training)")

    # (d) correlation heatmap
    ax = axes[1,0]
    cols = MONOMERS + ["Glass_kPa","Q","Modulus (kPa)"]
    corr = df184[cols].apply(pd.to_numeric, errors="coerce").corr()
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                annot_kws={"size":8}, cbar_kws={"shrink":0.7})
    ax.set_title("(d) Correlation matrix")

    # (e) scatter: Aromatic-PEA vs Glass adhesion
    ax = axes[1,1]
    sc = ax.scatter(df184["Aromatic-PEA"], df184["Glass_kPa"],
                    c=df184["Hydrophobic-BA"], cmap="viridis", s=40, edgecolor="k", lw=.3)
    ax.set_xlabel("Aromatic-PEA mole fraction")
    ax.set_ylabel("Glass adhesion (kPa)")
    ax.set_title("(e) PEA aromatic content vs adhesion")
    plt.colorbar(sc, ax=ax, label="Hydrophobic-BA")

    # (f) Q vs adhesion
    ax = axes[1,2]
    ax.scatter(df184["Q"], df184["Glass_kPa"], c="#888", s=40, edgecolor="k", lw=.3)
    ax.set_xlabel("Swelling Q")
    ax.set_ylabel("Glass adhesion (kPa)")
    ax.set_title("(f) Swelling vs adhesion")
    ax.set_xscale("log")

    fig.suptitle("Bio-inspired adhesive hydrogels — data overview", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig01_data_overview.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
