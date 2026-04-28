"""
Validation script.

Cross-checks two independent paths to per-cell P(response):
  (a) values reported in `sim-specific-response-likelihoods.csv`
  (b) the aggregation we compute ourselves from the rep-* score files.

Saves a parity-plot figure and an agreement summary JSON.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
IMG  = ROOT / "report" / "images"

agg = pd.read_csv(OUT / "aggregated_cell_response.csv")
sim = pd.read_csv(ROOT / "data" / "sim-specific-response-likelihoods.csv")
sim["repetition"] = sim["vaccine"].str.extract(r"rep-(\d+)").astype(int)
sim = sim.rename(columns={"name": "cell_id"})

merged = agg.merge(sim[["cell_id", "repetition", "p_response"]],
                   on=["cell_id", "repetition"], how="inner")

# Spearman & Pearson
from scipy.stats import pearsonr, spearmanr
pcc = pearsonr(merged.p_response_agg, merged.p_response)
scc = spearmanr(merged.p_response_agg, merged.p_response)
mae = float(np.mean(np.abs(merged.p_response_agg - merged.p_response)))
rmse = float(np.sqrt(np.mean((merged.p_response_agg - merged.p_response) ** 2)))
summary = {
    "n_pairs": int(len(merged)),
    "pearson_r": float(pcc[0]),
    "spearman_r": float(scc[0]),
    "MAE": mae,
    "RMSE": rmse
}
with open(OUT / "validation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(summary)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(merged.p_response, merged.p_response_agg, s=18, alpha=0.55,
           color="teal", edgecolor="white")
ax.plot([0, 1], [0, 1], "r--", lw=1, label="y = x")
ax.set_xlabel("Reported per-cell P(response) (sim-specific csv)")
ax.set_ylabel("Re-aggregated P(response) (from rep-* score files)")
ax.set_title(
    f"Validation parity plot (n={summary['n_pairs']})\n"
    f"Pearson r = {summary['pearson_r']:.4f}, "
    f"MAE = {summary['MAE']:.2e}"
)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.legend()
plt.tight_layout()
plt.savefig(IMG / "fig7_validation_parity.png", dpi=150)
plt.close()
print("Wrote", IMG / "fig7_validation_parity.png")
