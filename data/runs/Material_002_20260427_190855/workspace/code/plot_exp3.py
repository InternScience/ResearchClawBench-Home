"""Plot reaction barriers for Experiment 3."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUTS, IMAGES

with open(os.path.join(OUTPUTS, "exp3_barriers.json")) as f:
    data = json.load(f)
rows = data["rows"]
labels = ["Rxn 1\n(cyclobutene\nring-open)",
          "Rxn 11\n(methoxy\ndecomp.)",
          "Rxn 20\n(cyclopropane\nring-open)"]
mace = [r["barrier_MACE_eV"] for r in rows]
dft = [r["barrier_DFT_eV"] for r in rows]
x = np.arange(len(rows))
w = 0.36

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.bar(x - w/2, dft, w, label="DFT (CRBH20 ref.)", color="#7f7f7f")
ax.bar(x + w/2, mace, w, label="MACE-MP-0b3 (single-point)", color="#1f77b4")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Barrier $E_{TS}-E_R$ (eV)")
ax.set_title("Reaction barriers on placeholder reactant/TS geometries\n"
             f"MAE={data['MAE_eV']:.2f} eV, RMSE={data['RMSE_eV']:.2f} eV")
ax.legend(frameon=False)
fig.tight_layout()
out = os.path.join(IMAGES, "reaction_barriers.png")
fig.savefig(out, dpi=150)
print("wrote", out)
