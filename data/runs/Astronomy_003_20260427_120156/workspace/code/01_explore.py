"""
Data exploration: load the three CSVs, compute summary statistics in log10 space,
and dump them to outputs/summary_stats.json.
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def stats(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    la = np.log10(a)
    out = {
        "n": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "p05": float(np.quantile(a, 0.05)),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
        "p95": float(np.quantile(a, 0.95)),
        "log10_median": float(np.median(la)),
        "log10_mean": float(la.mean()),
        "log10_std": float(la.std(ddof=1)),
        "frac_above_1e-2": float(np.mean(a > 1e-2)),
        "frac_above_1e-3": float(np.mean(a > 1e-3)),
        "frac_below_1e-4": float(np.mean(a < 1e-4)),
    }
    return out


fig6 = pd.read_csv(os.path.join(DATA, "fig6_data.csv"))
fig7 = pd.read_csv(os.path.join(DATA, "fig7_data.csv"))
fig8 = pd.read_csv(os.path.join(DATA, "fig8_data.csv"))

summary = {
    "fig6_resolution_mismatch": stats(fig6["waveform_difference"].values),
    "fig7_per_ell": {col: stats(fig7[col].values) for col in fig7.columns},
    "fig8_extrapolation": {col: stats(fig8[col].values) for col in fig8.columns},
    "shapes": {
        "fig6": list(fig6.shape),
        "fig7": list(fig7.shape),
        "fig8": list(fig8.shape),
    },
}

with open(os.path.join(OUT, "summary_stats.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
