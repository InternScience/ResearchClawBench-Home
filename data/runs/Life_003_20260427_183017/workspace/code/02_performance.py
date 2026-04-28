"""Performance benchmark figure & wide-format table (Table 1 reproduction)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Life_003_20260427_183017"
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report/images")

df = pd.read_csv(os.path.join(DATA, "performance_summary.csv"))
print(df)

CHEM_ORDER = ["DNA r9.4", "DNA r10.4", "RNA001", "RNA004"]
TOOL_ORDER = ["Uncalled4", "f5c", "Nanopolish", "Tombo"]
TOOL_COLORS = {"Uncalled4": "#1f77b4", "f5c": "#2ca02c",
               "Nanopolish": "#d62728", "Tombo": "#9467bd"}

# Wide tables
time_w  = df.pivot(index="Chemistry", columns="Tool", values="Time_min").reindex(CHEM_ORDER)[TOOL_ORDER]
size_w  = df.pivot(index="Chemistry", columns="Tool", values="FileSize_MB").reindex(CHEM_ORDER)[TOOL_ORDER]
time_w.to_csv(os.path.join(OUT, "performance_time_min.csv"))
size_w.to_csv(os.path.join(OUT, "performance_filesize_mb.csv"))

# Speed-up of Uncalled4 vs each tool
speedups = []
for chem in CHEM_ORDER:
    u_time = time_w.loc[chem, "Uncalled4"]
    u_size = size_w.loc[chem, "Uncalled4"]
    for tool in TOOL_ORDER:
        if tool == "Uncalled4": continue
        t = time_w.loc[chem, tool]; s = size_w.loc[chem, tool]
        if pd.notna(t) and pd.notna(u_time):
            speedups.append({"chemistry": chem, "vs": tool,
                             "time_speedup": float(t / u_time),
                             "size_savings": float(s / u_size) if pd.notna(s) and u_size else np.nan})
sp = pd.DataFrame(speedups)
sp.to_csv(os.path.join(OUT, "performance_speedups.csv"), index=False)
print("\nSpeed-ups (vs Uncalled4):"); print(sp.to_string(index=False))

# --- Figure: bar chart, 2 subplots (time + filesize), grouped by chemistry, log-y ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
x = np.arange(len(CHEM_ORDER))
width = 0.2

for ax, mat, ylab, title in [
    (axes[0], time_w, "Alignment time (min, log-scale)", "Alignment time"),
    (axes[1], size_w, "Output file size (MB, log-scale)", "Output file size"),
]:
    for i, tool in enumerate(TOOL_ORDER):
        vals = mat[tool].values.astype(float)
        offset = (i - 1.5) * width
        ax.bar(x + offset, np.where(np.isnan(vals), 0, vals),
               width=width, color=TOOL_COLORS[tool], label=tool,
               edgecolor="black", linewidth=0.4)
        for xi, v in zip(x + offset, vals):
            if np.isnan(v):
                ax.text(xi, 1.2, "n/a", ha="center", va="bottom", fontsize=7, color="grey")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(CHEM_ORDER)
    ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(axis="y", which="both", alpha=0.25)

axes[0].legend(ncol=4, loc="upper left", fontsize=9)
fig.suptitle("Table 1 reproduction: Uncalled4 vs f5c vs Nanopolish vs Tombo", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "performance_benchmark.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Figure: heatmap-style speedup ---
piv_t = sp.pivot(index="chemistry", columns="vs", values="time_speedup").reindex(CHEM_ORDER)
piv_s = sp.pivot(index="chemistry", columns="vs", values="size_savings").reindex(CHEM_ORDER)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, piv, title, fmt in [
    (axes[0], piv_t, "Time speed-up (×) of Uncalled4", "{:.1f}×"),
    (axes[1], piv_s, "File-size reduction (×)", "{:.1f}×"),
]:
    im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for (r, c), v in np.ndenumerate(piv.values):
        ax.text(c, r, "n/a" if np.isnan(v) else fmt.format(v),
                ha="center", va="center", fontsize=10,
                color="black" if np.isnan(v) or v < 8 else "white")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "performance_speedup_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nWide time table:\n", time_w.round(1))
print("\nWide size table:\n", size_w.round(1))
print("DONE performance")
