"""
Phase 4: apply the calibrated GP to all 8424 vitrimer MD points.

Outputs:
  - outputs/vitrimer_calibrated_tg.csv
  - report/images/fig_calibrated_vitrimer_tg.png
"""
import os, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gp_pkl = pickle.load(open(os.path.join(ROOT, "outputs/gp_calibration.pkl"), "rb"))
gp = gp_pkl["gp"]

vit = pd.read_csv(os.path.join(ROOT, "data/tg_vitrimer_MD.csv"))
Xv = vit[["tg"]].values.astype(float)
mu, sd = gp.predict(Xv, return_std=True)

# Total uncertainty: GP posterior std (epistemic + WhiteKernel noise) combined
# with each point's MD-internal std propagated through the linearized GP.
md_std = vit["std"].values.astype(float)
# Local sensitivity dmu/dtg_md ~ finite difference
eps = 1.0
mu_eps = gp.predict(Xv + eps, return_std=False)
sens = (mu_eps - mu) / eps
sd_total = np.sqrt(sd**2 + (sens * md_std)**2)

vit_out = vit.copy()
vit_out["tg_md"] = vit_out["tg"]
vit_out["tg_calibrated"] = mu
vit_out["tg_calibrated_std_gp"] = sd
vit_out["tg_calibrated_std_total"] = sd_total
vit_out.to_csv(os.path.join(ROOT, "outputs/vitrimer_calibrated_tg.csv"), index=False)

# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axes[0]
ax.scatter(vit_out["tg_md"], vit_out["tg_calibrated"], s=4, alpha=0.25, color='#1f77b4')
lo, hi = 280, 580
ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label="y = x")
ax.set_xlabel("MD Tg [K]"); ax.set_ylabel("GP-calibrated Tg [K]")
ax.set_title("Vitrimer MD → calibrated Tg (8424 pairs)")
ax.legend(loc="upper left")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

ax = axes[1]
ax.hist(vit_out["tg_md"], bins=60, alpha=0.55, label="MD Tg", color='#ff7f0e')
ax.hist(vit_out["tg_calibrated"], bins=60, alpha=0.55, label="Calibrated Tg", color='#1f77b4')
ax.set_xlabel("Tg [K]"); ax.set_ylabel("count")
ax.set_title("Calibration shifts and tightens the Tg distribution")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "report/images/fig_calibrated_vitrimer_tg.png"), dpi=150)
plt.close(fig)

print("Calibrated Tg: mean", float(mu.mean()), "std", float(mu.std()))
print("MD Tg        : mean", float(Xv.mean()),"std", float(Xv.std()))
print("Mean GP sigma:", float(sd.mean()), "Mean total sigma:", float(sd_total.mean()))
print("Wrote vitrimer_calibrated_tg.csv shape", vit_out.shape)
