"""Pore model exploratory analysis across chemistries."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_003_20260427_183017"
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report/images")
os.makedirs(IMG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

CHEMS = [
    ("DNA r9.4.1 (6-mer)",  "dna_r9.4.1_400bps_6mer_uncalled4.csv",  6, "DNA"),
    ("DNA r10.4.1 (9-mer)", "dna_r10.4.1_400bps_9mer_uncalled4.csv", 9, "DNA"),
    ("RNA001 (5-mer)",      "rna_r9.4.1_70bps_5mer_uncalled4.csv",   5, "RNA"),
    ("RNA004 (9-mer)",      "rna004_130bps_9mer_uncalled4.csv",      9, "RNA"),
]

models = {}
summary_rows = []
for label, fname, k, mol in CHEMS:
    df = pd.read_csv(os.path.join(DATA, fname))
    models[label] = (df, k, mol)
    summary_rows.append({
        "chemistry": label, "molecule": mol, "k": k, "n_kmers": len(df),
        "current_mean_mean": df.current_mean.mean(),
        "current_mean_std":  df.current_mean.std(),
        "current_std_mean":  df.current_std.mean(),
        "dwell_time_mean":   df.dwell_time.mean(),
        "dwell_time_median": df.dwell_time.median(),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT, "pore_model_summary.csv"), index=False)
print(summary_df.to_string(index=False))

# --- Figure 1: distributions of current_mean ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (label, (df, k, mol)) in zip(axes.flat, models.items()):
    ax.hist(df.current_mean, bins=80, color="#3a6ea5", alpha=0.85, edgecolor="white")
    ax.set_title(f"{label}  (n={len(df):,})")
    ax.set_xlabel("Normalized current mean (z-score units)")
    ax.set_ylabel("# k-mers")
fig.suptitle("Pore-model k-mer current distributions across chemistries", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "kmer_current_distributions.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Figure 2: dwell time ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (label, (df, k, mol)) in zip(axes.flat, models.items()):
    d = df.dwell_time.clip(upper=df.dwell_time.quantile(0.99))
    ax.hist(d, bins=60, color="#b2545b", alpha=0.85, edgecolor="white")
    ax.set_title(f"{label}  median={df.dwell_time.median():.0f}")
    ax.set_xlabel("Dwell time (samples)")
    ax.set_ylabel("# k-mers")
fig.suptitle("Pore-model dwell-time distributions (clipped at 99th pct)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "kmer_dwell_distributions.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Figure 3: base-position effect (mean current vs base at center position) ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
bases_dna = list("ACGT"); bases_rna = list("ACGU")  # files use ACGT regardless
for ax, (label, (df, k, mol)) in zip(axes.flat, models.items()):
    center = k // 2
    df = df.copy()
    df["base_c"] = df.kmer.str[center]
    order = ["A", "C", "G", "T"]
    sns.boxplot(data=df, x="base_c", y="current_mean", order=order, ax=ax,
                palette="Set2", fliersize=0.5, linewidth=0.8)
    ax.set_title(f"{label}  center pos {center}")
    ax.set_xlabel("Center base")
    ax.set_ylabel("Current mean")
fig.suptitle("Effect of central base on k-mer current (per chemistry)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "base_position_effect.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Figure 4: position-wise mean current variability for the 9-mer models ---
def position_var(df, k):
    out = []
    for pos in range(k):
        for b in "ACGT":
            sub = df[df.kmer.str[pos] == b]
            out.append({"pos": pos, "base": b, "mean_current": sub.current_mean.mean()})
    return pd.DataFrame(out)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, label in zip(axes, ["DNA r10.4.1 (9-mer)", "RNA004 (9-mer)"]):
    df, k, _ = models[label]
    pv = position_var(df, k)
    pv_wide = pv.pivot(index="pos", columns="base", values="mean_current")
    pv_wide.plot(ax=ax, marker="o")
    ax.set_title(label)
    ax.set_xlabel("k-mer position")
    ax.set_ylabel("Mean current (avg over k-mers)")
    ax.grid(alpha=0.3)
fig.suptitle("Position-wise base sensitivity in 9-mer pore models", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "position_base_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Figure 5: compare overlapping central 5-mers between RNA001 (5-mer) and RNA004 (9-mer) ---
rna5 = models["RNA001 (5-mer)"][0].set_index("kmer")
rna9 = models["RNA004 (9-mer)"][0].copy()
rna9["central5"] = rna9.kmer.str[2:7]
agg9 = rna9.groupby("central5", as_index=True)["current_mean"].mean()
common = sorted(set(rna5.index) & set(agg9.index))
x = rna5.loc[common, "current_mean"].values
y = agg9.loc[common].values
corr = np.corrcoef(x, y)[0, 1]

fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(x, y, s=4, alpha=0.4, color="#2a7f62")
lim = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("RNA001 5-mer current mean")
ax.set_ylabel("RNA004 (avg over 9-mers sharing central 5-mer)")
ax.set_title(f"RNA001 vs RNA004 central-5mer agreement\nPearson r = {corr:.3f}, n={len(common)}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(IMG, "rna001_vs_rna004_kmer_agreement.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("RNA001 vs RNA004 central 5-mer Pearson r =", corr, "n=", len(common))

# Save k-mer overlap stats
with open(os.path.join(OUT, "kmer_chemistry_overlap.txt"), "w") as f:
    f.write(f"RNA001 vs RNA004 (central 5-mer of 9-mer): Pearson r = {corr:.4f}, n_common = {len(common)}\n")

print("DONE pore model EDA")
