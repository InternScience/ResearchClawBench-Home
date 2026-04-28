"""Build a 2x2 summary panel."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUTS, IMAGES

with open(os.path.join(OUTPUTS, "exp1_rdf.json")) as f:
    rdf = json.load(f)
with open(os.path.join(OUTPUTS, "exp1_water_md_log.json")) as f:
    log = json.load(f)
with open(os.path.join(OUTPUTS, "exp2_adsorption.json")) as f:
    ads = json.load(f)
with open(os.path.join(OUTPUTS, "exp3_barriers.json")) as f:
    bar = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

# --- (0,0) MD energy/T ---
ax = axes[0, 0]
t = [d["time_fs"] for d in log]
T = [d["T_K"] for d in log]
ax.plot(t, T, color="#d62728", lw=1.2, label="instantaneous T")
ax.axhline(330, color="grey", ls="--", lw=1, label="target 330 K")
ax.set_xlabel("Time (fs)"); ax.set_ylabel("Temperature (K)")
ax.set_title("(a) MD thermostat trace")
ax.legend(frameon=False, fontsize=9)

# --- (0,1) intermolecular RDF ---
ax = axes[0, 1]
labels = {"OO": ("O–O", "#1f77b4"), "OH": ("O–H", "#d62728"), "HH": ("H–H", "#2ca02c")}
ref = {"OO": 2.80, "OH": 1.85, "HH": 2.45}
for key in ["OO", "OH", "HH"]:
    name, c = labels[key]
    r = np.array(rdf["inter"][key]["r"]); g = np.array(rdf["inter"][key]["g"])
    ax.plot(r, g, color=c, lw=1.6, label=name)
    ax.axvline(ref[key], color=c, ls=":", alpha=0.4)
ax.set_xlim(0, 6); ax.set_xlabel("r (Å)"); ax.set_ylabel("g$_{inter}$(r)")
ax.set_title("(b) Liquid water intermolecular RDF")
ax.legend(frameon=False, fontsize=9)

# --- (1,0) adsorption scaling ---
ax = axes[1, 0]
rows = ads["rows"]
metals = [r["metal"] for r in rows]
Eo = np.array([r["Eads_O_eV"] for r in rows])
Eoh = np.array([r["Eads_OH_eV"] for r in rows])
ax.scatter(Eo, Eoh, s=70, color="#1f77b4")
for m, x, y in zip(metals, Eo, Eoh):
    ax.annotate(m, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=10)
xx = np.linspace(Eo.min() - 0.3, Eo.max() + 0.3, 50)
ax.plot(xx, ads["slope"] * xx + ads["intercept_eV"], color="#d62728", lw=1.6,
        label=f"slope={ads['slope']:.2f}, R²={ads['r2']:.2f}")
ax.plot(xx, 0.5 * xx + (Eoh.mean() - 0.5 * Eo.mean()), color="grey", ls="--", lw=1.0,
        label="ref. slope 0.5")
ax.set_xlabel("E$_{ads}$(O*) (eV)"); ax.set_ylabel("E$_{ads}$(OH*) (eV)")
ax.set_title("(c) OH–O scaling on fcc(111)")
ax.legend(frameon=False, fontsize=9)

# --- (1,1) reaction barriers ---
ax = axes[1, 1]
brows = bar["rows"]
labels_r = ["Rxn 1", "Rxn 11", "Rxn 20"]
mace = [r["barrier_MACE_eV"] for r in brows]
dft = [r["barrier_DFT_eV"] for r in brows]
xpos = np.arange(len(brows)); w = 0.36
ax.bar(xpos - w/2, dft, w, label="DFT (CRBH20)", color="#7f7f7f")
ax.bar(xpos + w/2, mace, w, label="MACE-MP-0b3", color="#1f77b4")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(xpos); ax.set_xticklabels(labels_r)
ax.set_ylabel("$E_{TS} - E_R$ (eV)")
ax.set_title(f"(d) Reaction barriers (MAE={bar['MAE_eV']:.2f} eV)")
ax.legend(frameon=False, fontsize=9)

fig.suptitle("MACE-MP-0b3-medium reproduction: liquid water, surface adsorption, "
             "reaction barriers", y=1.00, fontsize=12)
fig.tight_layout()
out = os.path.join(IMAGES, "summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
