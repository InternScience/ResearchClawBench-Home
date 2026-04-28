"""Plot adsorption-energy scaling for Experiment 2."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUTS, IMAGES

with open(os.path.join(OUTPUTS, "exp2_adsorption.json")) as f:
    data = json.load(f)

rows = data["rows"]
metals = [r["metal"] for r in rows]
Eo = np.array([r["Eads_O_eV"] for r in rows])
Eoh = np.array([r["Eads_OH_eV"] for r in rows])

slope, intercept = data["slope"], data["intercept_eV"]
r2 = data["r2"]
xx = np.linspace(Eo.min() - 0.3, Eo.max() + 0.3, 100)
yy = slope * xx + intercept

fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.scatter(Eo, Eoh, s=70, color="#1f77b4", zorder=3)
for m, x, y in zip(metals, Eo, Eoh):
    ax.annotate(m, (x, y), xytext=(6, 5), textcoords="offset points",
                fontsize=11)
ax.plot(xx, yy, color="#d62728", lw=1.8,
        label=f"MACE-MP-0b3 fit: E$_{{OH}}$ = {slope:.2f}·E$_O$ + {intercept:.2f} eV  (R$^2$={r2:.2f})")
# Reference scaling line from Abild-Pedersen 2007 / Calle-Vallejo 2014
# For OH vs O on close-packed transition metals, slope ~ 0.5 (CHEMBOND)
ax.plot(xx, 0.5 * xx + (Eoh.mean() - 0.5 * Eo.mean()), color="grey", ls="--", lw=1.2,
        label="Reference slope 0.5 (Abild-Pedersen-style)")
ax.set_xlabel("E$_{ads}$(O*) (eV)")
ax.set_ylabel("E$_{ads}$(OH*) (eV)")
ax.set_title("OH–O adsorption-energy scaling on fcc(111) — MACE-MP-0b3-medium")
ax.legend(frameon=False, loc="best")
fig.tight_layout()
out = os.path.join(IMAGES, "adsorption_scaling.png")
fig.savefig(out, dpi=150)
print("wrote", out)


# ---- companion bar plot of raw Eads ----
fig, ax = plt.subplots(figsize=(7.0, 4.0))
x = np.arange(len(metals))
w = 0.36
ax.bar(x - w/2, Eo, w, label="E$_{ads}$(O*)", color="#1f77b4")
ax.bar(x + w/2, Eoh, w, label="E$_{ads}$(OH*)", color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(metals)
ax.set_ylabel("Adsorption energy (eV)")
ax.axhline(0, lw=0.6, color="k")
ax.set_title("Adsorption energies on fcc(111) hollow site — MACE-MP-0b3-medium")
ax.legend(frameon=False)
fig.tight_layout()
out = os.path.join(IMAGES, "adsorption_energies.png")
fig.savefig(out, dpi=150)
print("wrote", out)
