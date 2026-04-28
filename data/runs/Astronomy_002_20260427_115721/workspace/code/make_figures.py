"""
Generate figures for the Local Distance Network analysis.
Reads outputs/* and writes report/images/*.png
"""
import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
IMG  = os.path.join(ROOT, "report", "images")
os.makedirs(IMG, exist_ok=True)

variants = pd.read_csv(os.path.join(OUT, "h0_variants.csv"))
baseline = json.load(open(os.path.join(OUT, "h0_baseline.json")))
parms    = pd.read_csv(os.path.join(OUT, "gls_parameters.csv"))
resids   = pd.read_csv(os.path.join(OUT, "residuals.csv"))
weights  = pd.read_csv(os.path.join(OUT, "info_weights.csv"))

# ----------------------------------------------------------------------
# 1. Data overview — primary host distance moduli by indicator/anchor
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
prim = resids[resids["block"] == "primary"].copy()
markers = {("Cepheid","N4258"):("o","tab:blue"),
           ("Cepheid","LMC"):  ("s","tab:cyan"),
           ("TRGB","N4258"):   ("D","tab:red")}
for (m,a), (mk,col) in markers.items():
    sub = prim[(prim["method"]==m) & (prim["anchor"]==a)]
    ax.errorbar(sub["host"], sub["y"], yerr=sub["sigma"], fmt=mk, color=col,
                label=f"{m} / {a}", capsize=3, ms=7, lw=1.4)
# best-fit per host
hosts = sorted(prim["host"].unique())
mu_fit = {row["parameter"].replace("mu_",""): row["value"]
          for _, row in parms.iterrows() if row["parameter"].startswith("mu_") and not row["parameter"].startswith("mu_grp_")}
ax.plot(hosts, [mu_fit[h] for h in hosts], "k_", ms=24, mew=2,
        label="GLS best-fit μ_host")
ax.set_ylabel(r"Distance modulus $\mu$ (mag)")
ax.set_xlabel("Primary host")
ax.set_title("Primary host distance moduli — data and GLS fit")
ax.grid(alpha=0.3)
ax.legend()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(IMG, "data_overview.png"), dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 2. Hubble diagram for HF SNe Ia and HF SBF, with best-fit a_H
# ----------------------------------------------------------------------
M_B   = float(parms.loc[parms["parameter"]=="M_B","value"].iloc[0])
M_SBF = float(parms.loc[parms["parameter"]=="M_SBF","value"].iloc[0])
a_H   = float(parms.loc[parms["parameter"]=="a_H","value"].iloc[0])

c_km = 299792.458
z_grid = np.linspace(0.005, 0.10, 200)
mu_grid = 5*np.log10(c_km*z_grid) + 25 - a_H

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
# HF SNe Ia
hfsn = resids[resids["block"]=="hf_sneia"]
ax = axes[0]
mB_obs = hfsn["y"] + 5*np.log10(c_km*hfsn["z"]) + 25
ax.errorbar(hfsn["z"], mB_obs - M_B, yerr=hfsn["sigma"], fmt="o", color="tab:blue",
            label="HF SNe Ia (observed μ)")
ax.plot(z_grid, mu_grid, "-", color="black", label=f"GLS fit (5log10(cz/H0)+25)")
ax.set_xlabel("redshift z")
ax.set_ylabel(r"$\mu = m_B - M_B$ (mag)")
ax.set_title("Hubble diagram — SNe Ia")
ax.set_xscale("log")
ax.legend()
ax.grid(alpha=0.3)

# HF SBF
hfsbf = resids[resids["block"]=="hf_sbf"]
ax = axes[1]
mF_obs = hfsbf["y"] + 5*np.log10(c_km*hfsbf["z"]) + 25
ax.errorbar(hfsbf["z"], mF_obs - M_SBF, yerr=hfsbf["sigma"], fmt="s", color="tab:red",
            label="HF SBF (observed μ)")
ax.plot(z_grid, mu_grid, "-", color="black", label="GLS fit")
ax.set_xlabel("redshift z")
ax.set_ylabel(r"$\mu = m_{F110W} - M_{SBF}$ (mag)")
ax.set_title("Hubble diagram — SBF")
ax.set_xscale("log")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(IMG, "hubble_diagram.png"), dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 3. H0 across variants — forest plot
# ----------------------------------------------------------------------
order = ["baseline","only_N4258","only_LMC","N4258+LMC",
         "Cepheids_only","TRGB_only","SNeIa_only",
         "drop_NGC1309","drop_NGC1365","drop_NGC1448","drop_NGC1559",
         "drop_M101","drop_NGC1316","drop_NGC5643"]
order = [n for n in order if n in set(variants["name"])]
df = variants.set_index("name").loc[order].reset_index()

fig, ax = plt.subplots(figsize=(8, 7))
ypos = np.arange(len(df))
ax.errorbar(df["H0"], ypos, xerr=df["sigma_H0"], fmt="o", color="tab:blue",
            ecolor="gray", capsize=3)
# Highlight baseline
i_base = list(df["name"]).index("baseline")
ax.errorbar([df["H0"].iloc[i_base]], [ypos[i_base]],
            xerr=[df["sigma_H0"].iloc[i_base]], fmt="o", color="black",
            ecolor="black", capsize=4, ms=9, label="baseline")
ax.axvline(73.04, color="tab:green", linestyle="--", label="SH0ES 2022 (73.04)")
ax.axvline(67.4, color="tab:red", linestyle="--", label="Planck 2018 (67.4)")
ax.set_yticks(ypos)
ax.set_yticklabels(df["name"])
ax.invert_yaxis()
ax.set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
ax.set_title("Distance Network: H₀ across analysis variants")
ax.grid(alpha=0.3, axis="x")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(IMG, "h0_variants.png"), dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 4. Residual diagnostic
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
colors = {"primary":"tab:blue","sneia_cal":"tab:orange","sbf_cal":"tab:green",
          "hf_sneia":"tab:purple","hf_sbf":"tab:brown"}
for blk, c in colors.items():
    sub = resids[resids["block"]==blk]
    ax.scatter(np.arange(len(sub))+ (resids["block"]==blk).cumsum().iloc[0]*0,
               sub["std_resid"], color=c, label=blk)
# Better: use original index
ax.clear()
for blk, c in colors.items():
    sub = resids[resids["block"]==blk]
    ax.scatter(sub.index, sub["std_resid"], color=c, label=blk, s=40)
ax.axhline(0, color="black", lw=1)
ax.axhline(2, color="gray", linestyle=":")
ax.axhline(-2, color="gray", linestyle=":")
ax.set_xlabel("observation index")
ax.set_ylabel("standardized residual (resid / σ)")
ax.set_title("GLS residuals (baseline)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(resids["std_resid"], bins=15, color="tab:blue", edgecolor="black", alpha=0.7)
xs = np.linspace(-6, 6, 200)
ax.plot(xs, len(resids)*1.0/np.sqrt(2*np.pi)*np.exp(-xs**2/2), color="red",
        label="Standard normal (expected)")
ax.set_xlabel("standardized residual")
ax.set_ylabel("count")
ax.set_title("Residual distribution")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(IMG, "residuals.png"), dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 5. Information weights — which observations drive H0
# ----------------------------------------------------------------------
weights["label"] = weights.apply(
    lambda r: ("{block}/{host}".format(**r) if r["block"] in ("primary","sneia_cal","sbf_cal")
               else "{block}/z={z:.3f}".format(**r)), axis=1)
ws = weights.sort_values("info_share", ascending=True)
fig, ax = plt.subplots(figsize=(8, 9))
colors = ws["block"].map({"primary":"tab:blue","sneia_cal":"tab:orange",
                          "sbf_cal":"tab:green","hf_sneia":"tab:purple",
                          "hf_sbf":"tab:brown"})
ax.barh(ws["label"], ws["info_share"], color=colors)
ax.set_xlabel("Information share on $a_H = 5\\log_{10} H_0$")
ax.set_title("Per-observation contribution to H₀ constraint")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
fig.savefig(os.path.join(IMG, "info_weights.png"), dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 6. Anchor consistency
# ----------------------------------------------------------------------
anchor_df = pd.read_csv(os.path.join(OUT, "h0_by_anchor.csv"))
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(anchor_df["H0"], range(len(anchor_df)),
            xerr=anchor_df["sigma_H0"], fmt="o", color="tab:blue", capsize=4)
ax.axvline(73.04, color="tab:green", linestyle="--", label="SH0ES 2022")
ax.axvline(67.4, color="tab:red", linestyle="--", label="Planck 2018")
ax.set_yticks(range(len(anchor_df)))
ax.set_yticklabels(anchor_df["label"])
ax.set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
ax.set_title("H₀ from each anchor configuration")
ax.legend()
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
fig.savefig(os.path.join(IMG, "anchor_consistency.png"), dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 7. Local vs CMB
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["This work\n(Distance Network, baseline)",
          "SH0ES (Riess+ 2022)",
          "Planck 2018 (CMB+ΛCDM)"]
H0s = [baseline["H0"], 73.04, 67.4]
errs = [baseline["sigma_H0"], 1.04, 0.5]
colors = ["tab:blue","tab:green","tab:red"]
ypos = range(len(labels))
for y,(h,e,l,c) in enumerate(zip(H0s, errs, labels, colors)):
    ax.errorbar([h],[y], xerr=[e], fmt="o", color=c, capsize=5, ms=9)
ax.set_yticks(list(ypos))
ax.set_yticklabels(labels)
ax.set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)")
ax.set_title("Local distance ladder vs early-universe (CMB)")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
fig.savefig(os.path.join(IMG, "h0_vs_cmb.png"), dpi=150)
plt.close(fig)

print("Figures saved to", IMG)
print("Files:", sorted(os.listdir(IMG)))
