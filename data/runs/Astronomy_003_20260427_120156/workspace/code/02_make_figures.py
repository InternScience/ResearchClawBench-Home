"""
Reproduce SXS catalog Figs. 6, 7, 8 and produce additional analysis figures.

Outputs:
  report/images/fig6_resolution_mismatch.png
  report/images/fig7_per_ell_mismatch.png
  report/images/fig8_extrapolation.png
  report/images/fig_overview.png
  report/images/fig_ell_scaling.png
  report/images/fig_cdf_summary.png
  outputs/figure_data_summary.json
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report", "images")
os.makedirs(IMG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig6 = pd.read_csv(os.path.join(DATA, "fig6_data.csv"))["waveform_difference"].values
fig7 = pd.read_csv(os.path.join(DATA, "fig7_data.csv"))
fig8 = pd.read_csv(os.path.join(DATA, "fig8_data.csv"))

# ---------- Fig. 6 reproduction ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
bins = np.logspace(np.log10(fig6.min()*0.9), np.log10(fig6.max()*1.1), 40)
ax.hist(fig6, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.85)
med = np.median(fig6)
ax.axvline(med, color="crimson", ls="--", lw=1.5, label=f"median = {med:.2e}")
ax.axvline(4e-4, color="black", ls=":", lw=1.2, label="reported median (4e-4)")
ax.set_xscale("log")
ax.set_xlabel(r"Waveform difference between two highest resolutions $\delta h_{\rm res}$")
ax.set_ylabel("Number of simulations")
ax.set_title(f"Fig. 6 — resolution-mismatch distribution (N={len(fig6)})")
ax.legend(loc="upper left", fontsize=9)

ax = axes[1]
sorted6 = np.sort(fig6)
cdf = np.arange(1, len(sorted6)+1) / len(sorted6)
ax.plot(sorted6, cdf, color="#1f77b4", lw=1.6)
ax.axhline(0.5, color="gray", ls=":", lw=0.8)
ax.axvline(med, color="crimson", ls="--", lw=1.2, label=f"median = {med:.2e}")
for thr in [1e-3, 1e-2]:
    frac = np.mean(fig6 > thr)
    ax.axvline(thr, color="black", ls=":", lw=0.8)
    ax.text(thr, 0.05, f"P(>{thr:g}) = {frac:.2%}", rotation=90, va="bottom", ha="right", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel(r"$\delta h_{\rm res}$")
ax.set_ylabel("Cumulative fraction")
ax.set_title("Empirical CDF of resolution mismatch")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig6_resolution_mismatch.png"))
plt.close(fig)

# ---------- Fig. 7 reproduction ----------
ells = list(range(2, 9))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# panel (a): histogram-like step plots per ell
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(ells)))
all_vals = pd.concat([fig7[c] for c in fig7.columns]).values
bin_edges = np.logspace(np.log10(max(all_vals.min(), 1e-7)), np.log10(all_vals.max()*1.05), 35)
for ell, col, color in zip(ells, fig7.columns, colors):
    vals = fig7[col].values
    counts, _ = np.histogram(vals, bins=bin_edges)
    centers = np.sqrt(bin_edges[:-1]*bin_edges[1:])
    ax.step(centers, counts, where="mid", color=color, lw=1.6, label=fr"$\ell={ell}$")
ax.set_xscale("log")
ax.set_xlabel(r"Per-mode waveform difference $\delta h_\ell$")
ax.set_ylabel("Number of simulations")
ax.set_title(f"Fig. 7 — modal mismatch distributions (N={len(fig7)})")
ax.legend(ncol=2, fontsize=8, loc="upper left")

# panel (b): violin / box of medians vs ell
ax = axes[1]
data_per_ell = [np.log10(fig7[c].values) for c in fig7.columns]
bp = ax.boxplot(data_per_ell, positions=ells, widths=0.55, patch_artist=True,
                showfliers=False, medianprops=dict(color="crimson", lw=1.5))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
medians = [np.median(fig7[c].values) for c in fig7.columns]
ax.plot(ells, np.log10(medians), "ko-", lw=1.4, label="median")
ax.set_xticks(ells)
ax.set_xlabel(r"Spherical-harmonic index $\ell$")
ax.set_ylabel(r"$\log_{10}\,\delta h_\ell$")
ax.set_title("Per-$\\ell$ summary (boxplot in $\\log_{10}$ space)")
ax.legend(fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig7_per_ell_mismatch.png"))
plt.close(fig)

# ---------- Fig. 8 reproduction ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
all8 = np.concatenate([fig8["N2vsN3"].values, fig8["N2vsN4"].values])
b8 = np.logspace(np.log10(all8.min()*0.9), np.log10(all8.max()*1.1), 35)
ax.hist(fig8["N2vsN3"], bins=b8, color="#1f77b4", alpha=0.6, label=r"$N=2$ vs $N=3$",
        edgecolor="white")
ax.hist(fig8["N2vsN4"], bins=b8, color="#d62728", alpha=0.55, label=r"$N=2$ vs $N=4$",
        edgecolor="white")
m23 = np.median(fig8["N2vsN3"]); m24 = np.median(fig8["N2vsN4"])
ax.axvline(m23, color="#1f77b4", ls="--", lw=1.2)
ax.axvline(m24, color="#d62728", ls="--", lw=1.2)
ax.set_xscale("log")
ax.set_xlabel(r"Extrapolation-order waveform difference $\delta h_{\rm extr}$")
ax.set_ylabel("Number of simulations")
ax.set_title(f"Fig. 8 — extrapolation-order comparison (N={len(fig8)})")
ax.legend(loc="upper left", fontsize=9)
ax.text(0.98, 0.95,
        f"median(N2 vs N3) = {m23:.2e}\nmedian(N2 vs N4) = {m24:.2e}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))

ax = axes[1]
for col, color in [("N2vsN3", "#1f77b4"), ("N2vsN4", "#d62728")]:
    s = np.sort(fig8[col].values)
    c = np.arange(1, len(s)+1) / len(s)
    ax.plot(s, c, color=color, lw=1.6, label=col.replace("vs"," vs "))
ax.set_xscale("log")
ax.set_xlabel(r"$\delta h_{\rm extr}$")
ax.set_ylabel("Cumulative fraction")
ax.set_title("Empirical CDFs (extrapolation comparison)")
ax.legend(loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig8_extrapolation.png"))
plt.close(fig)

# ---------- Cross-cutting overview ----------
fig, ax = plt.subplots(figsize=(8.5, 5))
sources = [
    ("Resolution (Fig. 6)", fig6, "#1f77b4"),
    ("Mode ℓ=2 (Fig. 7)", fig7["ell2"].values, "#2ca02c"),
    ("Mode ℓ=8 (Fig. 7)", fig7["ell8"].values, "#9467bd"),
    ("Extrap N2 vs N3 (Fig. 8)", fig8["N2vsN3"].values, "#ff7f0e"),
    ("Extrap N2 vs N4 (Fig. 8)", fig8["N2vsN4"].values, "#d62728"),
]
edges = np.logspace(-7, 0, 50)
for name, vals, c in sources:
    s = np.sort(vals)
    cdf = np.arange(1, len(s)+1)/len(s)
    ax.plot(s, cdf, lw=1.8, color=c, label=f"{name}  (med={np.median(vals):.1e})")
ax.set_xscale("log")
ax.set_xlim(1e-7, 1)
ax.set_xlabel(r"Waveform difference $\delta h$")
ax.set_ylabel("Cumulative fraction of simulations")
ax.set_title("Cross-source CDF overview of waveform error metrics")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_overview.png"))
plt.close(fig)

# ---------- Per-ell scaling ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
medians = np.array([np.median(fig7[c].values) for c in fig7.columns])
p05 = np.array([np.quantile(fig7[c].values, 0.05) for c in fig7.columns])
p95 = np.array([np.quantile(fig7[c].values, 0.95) for c in fig7.columns])
ax.fill_between(ells, p05, p95, color="#1f77b4", alpha=0.18, label="5–95% range")
ax.plot(ells, medians, "o-", color="#1f77b4", lw=2, label="median")
# Power-law fit on the log–log relation
coef = np.polyfit(np.log10(ells), np.log10(medians), 1)
fit = 10**(coef[1]) * np.array(ells)**coef[0]
ax.plot(ells, fit, "k--", lw=1.2,
        label=fr"power-law fit: $\delta h_\ell \propto \ell^{{{coef[0]:.2f}}}$")
ax.set_yscale("log")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\delta h_\ell$")
ax.set_title("Modal mismatch scaling with spherical-harmonic index")
ax.legend(fontsize=9)
ax.set_xticks(ells)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_ell_scaling.png"))
plt.close(fig)

# ---------- Combined CDF tail summary ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
thresholds = np.logspace(-6, -1, 40)
for name, vals, c in [
    ("Resolution", fig6, "#1f77b4"),
    ("Modal ℓ=2", fig7["ell2"].values, "#2ca02c"),
    ("Modal ℓ=4", fig7["ell4"].values, "#bcbd22"),
    ("Modal ℓ=6", fig7["ell6"].values, "#9467bd"),
    ("Modal ℓ=8", fig7["ell8"].values, "#9467bd"),
]:
    surv = np.array([np.mean(vals > t) for t in thresholds])
    ax.plot(thresholds, surv, lw=1.5, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Threshold $\delta h_*$")
ax.set_ylabel(r"Fraction with $\delta h > \delta h_*$")
ax.set_title("Tail probability — modal & resolution errors")
ax.legend(fontsize=9)

ax = axes[1]
for name, vals, c in [("N2 vs N3", fig8["N2vsN3"].values, "#1f77b4"),
                      ("N2 vs N4", fig8["N2vsN4"].values, "#d62728")]:
    surv = np.array([np.mean(vals > t) for t in thresholds])
    ax.plot(thresholds, surv, lw=1.6, label=name, color=c)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Threshold $\delta h_*$")
ax.set_ylabel(r"Fraction with $\delta h > \delta h_*$")
ax.set_title("Tail probability — extrapolation order")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_cdf_summary.png"))
plt.close(fig)

# ---------- Save figure-derived numbers ----------
fig_summary = {
    "fig6_median": float(np.median(fig6)),
    "fig6_p95": float(np.quantile(fig6, 0.95)),
    "fig6_frac_above_1e-3": float(np.mean(fig6 > 1e-3)),
    "fig6_frac_above_1e-2": float(np.mean(fig6 > 1e-2)),
    "fig7_median_per_ell": {c: float(np.median(fig7[c].values)) for c in fig7.columns},
    "fig7_powerlaw_slope": float(coef[0]),
    "fig7_powerlaw_intercept_log10": float(coef[1]),
    "fig8_median_N2vsN3": float(np.median(fig8["N2vsN3"].values)),
    "fig8_median_N2vsN4": float(np.median(fig8["N2vsN4"].values)),
    "fig8_ratio_N4overN3_median": float(np.median(fig8["N2vsN4"].values) /
                                       np.median(fig8["N2vsN3"].values)),
}
with open(os.path.join(OUT, "figure_data_summary.json"), "w") as f:
    json.dump(fig_summary, f, indent=2)
print(json.dumps(fig_summary, indent=2))
print("All figures written to", IMG)
