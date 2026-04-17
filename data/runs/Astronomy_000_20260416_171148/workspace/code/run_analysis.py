#!/usr/bin/env python3
"""
Full analysis pipeline for constraining ultralight bosons via BH superradiance.
Produces all figures and output tables.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import json
import os
import sys

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from superradiance import (
    alpha_grav, critical_spin_lm, instability_timescale,
    is_excluded_by_superradiance, bayesian_exclusion_probability,
    bosenova_fa_limit, load_samples, r_g, omega_plus,
    superradiance_rate_nlm, yr_to_s, M_Pl, c_light, G_N, M_sun, hbar, eV_to_kg
)

# ============================================================
# Configuration
# ============================================================
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, "data")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report", "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# BH system parameters
BH_SYSTEMS = {
    "M33_X-7": {
        "file": "M33_X-7_samples.dat",
        "type": "stellar",
        "max_age_yr": 5e6,  # young stellar-mass BH, ~few Myr
        "label": "M33 X-7 (Stellar)",
        "color": "royalblue",
        "mu_range": np.logspace(-14, -11, 200),  # eV - stellar mass range
    },
    "IRAS_09149-6206": {
        "file": "IRAS_09149-6206_samples.dat",
        "type": "smbh",
        "max_age_yr": 1e10,  # Hubble time for SMBH
        "label": "IRAS 09149-6206 (SMBH)",
        "color": "crimson",
        "mu_range": np.logspace(-21, -16, 200),  # eV - SMBH range
    }
}

# ============================================================
# Load Data
# ============================================================
print("=" * 60)
print("Loading posterior samples...")
print("=" * 60)

data = {}
for name, info in BH_SYSTEMS.items():
    filepath = os.path.join(DATA_DIR, info["file"])
    M_samples, a_samples = load_samples(filepath)
    data[name] = {"M": M_samples, "a": a_samples}
    print(f"\n{info['label']}:")
    print(f"  N samples: {len(M_samples)}")
    print(f"  Mass: {M_samples.mean():.2e} +/- {M_samples.std():.2e} M_sun")
    print(f"  Spin: {a_samples.mean():.4f} +/- {a_samples.std():.4f}")

# ============================================================
# Figure 1: Posterior Distributions
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 1: Posterior Distributions")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (name, info) in enumerate(BH_SYSTEMS.items()):
    M = data[name]["M"]
    a = data[name]["a"]
    
    # Scatter plot
    ax = axes[idx, 0]
    ax.scatter(M, a, alpha=0.1, s=1, color=info["color"])
    ax.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]", fontsize=12)
    ax.set_ylabel(r"$a_*$", fontsize=12)
    ax.set_title(info["label"], fontsize=13)
    
    # Mass histogram
    ax = axes[idx, 1]
    ax.hist(M, bins=50, color=info["color"], alpha=0.7, density=True)
    ax.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Mass Distribution", fontsize=13)
    
    # Spin histogram
    ax = axes[idx, 2]
    ax.hist(a, bins=50, color=info["color"], alpha=0.7, density=True)
    ax.set_xlabel(r"$a_*$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Spin Distribution", fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "fig1_posterior_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_posterior_distributions.png")

# ============================================================
# Figure 2: Regge Plane with Exclusion Regions
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 2: Regge Plane")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (name, info) in enumerate(BH_SYSTEMS.items()):
    ax = axes[idx]
    M = data[name]["M"]
    a = data[name]["a"]
    
    # Plot posterior samples
    ax.scatter(M, a, alpha=0.15, s=2, color='gray', label='Posterior samples', zorder=1)
    
    # Plot critical spin curves for several boson masses
    if info["type"] == "stellar":
        mu_values = [5e-13, 1e-12, 2e-12, 5e-12, 1e-11]
        M_grid = np.linspace(M.min() * 0.8, M.max() * 1.2, 500)
    else:
        mu_values = [1e-19, 5e-19, 1e-18, 5e-18, 1e-17]
        M_grid = np.linspace(M.min() * 0.5, M.max() * 1.5, 500)
    
    colors_regge = plt.cm.viridis(np.linspace(0.2, 0.9, len(mu_values)))
    
    for j, mu in enumerate(mu_values):
        a_crit_l1 = np.array([critical_spin_lm(alpha_grav(mu, m), 1, 1) for m in M_grid])
        a_crit_l2 = np.array([critical_spin_lm(alpha_grav(mu, m), 2, 2) for m in M_grid])
        
        ax.plot(M_grid, a_crit_l1, '-', color=colors_regge[j], linewidth=2,
                label=f'$\\mu = {mu:.0e}$ eV ($\\ell=1$)')
        ax.plot(M_grid, a_crit_l2, '--', color=colors_regge[j], linewidth=1.5)
    
    ax.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]", fontsize=13)
    ax.set_ylabel(r"$a_*$", fontsize=13)
    ax.set_title(f"Regge Plane: {info['label']}", fontsize=14)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0, 1)
    
    if info["type"] == "smbh":
        ax.set_xscale('log')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "fig2_regge_plane.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_regge_plane.png")

# ============================================================
# Figure 3: Instability Timescale vs Boson Mass
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 3: Instability Timescales")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (name, info) in enumerate(BH_SYSTEMS.items()):
    ax = axes[idx]
    M_median = np.median(data[name]["M"])
    a_median = np.median(data[name]["a"])
    
    mu_range = info["mu_range"]
    
    for l in [1, 2]:
        taus = []
        for mu in mu_range:
            alpha = alpha_grav(mu, M_median)
            tau = instability_timescale(alpha, a_median, M_median, l=l, m=l, n=0)
            taus.append(tau / yr_to_s)
        taus = np.array(taus)
        ax.plot(mu_range, taus, linewidth=2, label=f'$\\ell = m = {l}$')
    
    ax.axhline(y=info["max_age_yr"], color='red', linestyle='--', linewidth=1.5,
               label=f'BH age ({info["max_age_yr"]:.0e} yr)')
    ax.axhline(y=4.35e17 / yr_to_s, color='orange', linestyle=':', linewidth=1.5,
               label='Hubble time')
    
    ax.set_xlabel(r"$\mu_b$ [eV]", fontsize=13)
    ax.set_ylabel(r"$\tau_{\rm SR}$ [yr]", fontsize=13)
    ax.set_title(f"Instability Timescale: {info['label']}\n($M = {M_median:.1e}$ $M_\\odot$, $a_* = {a_median:.3f}$)", fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e25)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "fig3_instability_timescales.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_instability_timescales.png")

# ============================================================
# Compute Bayesian Exclusion Curves
# ============================================================
print("\n" + "=" * 60)
print("Computing Bayesian Exclusion Curves (this may take a while)...")
print("=" * 60)

exclusion_results = {}

for name, info in BH_SYSTEMS.items():
    print(f"\n  Processing {info['label']}...")
    M = data[name]["M"]
    a = data[name]["a"]
    mu_range = info["mu_range"]
    
    # Use subsampling for speed if needed
    N = len(M)
    if N > 2000:
        idx_sub = np.random.choice(N, 2000, replace=False)
        M_sub = M[idx_sub]
        a_sub = a[idx_sub]
    else:
        M_sub = M
        a_sub = a
    
    probs = []
    for i, mu in enumerate(mu_range):
        if i % 50 == 0:
            print(f"    {i}/{len(mu_range)}...")
        p = bayesian_exclusion_probability(mu, M_sub, a_sub,
                                            max_age_yr=info["max_age_yr"],
                                            l_max=2)
        probs.append(p)
    
    probs = np.array(probs)
    exclusion_results[name] = {
        "mu_range": mu_range,
        "probs": probs
    }
    
    # Find 95% exclusion range
    excluded_95 = mu_range[probs >= 0.95]
    if len(excluded_95) > 0:
        print(f"    95% exclusion range: [{excluded_95.min():.2e}, {excluded_95.max():.2e}] eV")
    else:
        # Find maximum exclusion probability
        max_p = probs.max()
        print(f"    Maximum exclusion probability: {max_p:.3f}")
        excluded_50 = mu_range[probs >= 0.50]
        if len(excluded_50) > 0:
            print(f"    50% exclusion range: [{excluded_50.min():.2e}, {excluded_50.max():.2e}] eV")

# ============================================================
# Figure 4: Bayesian Exclusion Probability
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 4: Bayesian Exclusion Probability")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

for name, info in BH_SYSTEMS.items():
    mu_range = exclusion_results[name]["mu_range"]
    probs = exclusion_results[name]["probs"]
    
    ax.plot(mu_range, probs, linewidth=2.5, color=info["color"],
            label=info["label"])

ax.axhline(y=0.95, color='black', linestyle='--', linewidth=1, alpha=0.7,
           label='95% CL')
ax.axhline(y=0.90, color='gray', linestyle=':', linewidth=1, alpha=0.5,
           label='90% CL')

ax.set_xlabel(r"Boson mass $\mu_b$ [eV]", fontsize=14)
ax.set_ylabel(r"Exclusion probability $P(\mathrm{excluded} | \mu_b)$", fontsize=14)
ax.set_title("Bayesian Exclusion of Ultralight Bosons from Black Hole Superradiance", fontsize=15)
ax.set_xscale('log')
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=12, loc='center left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "fig4_exclusion_probability.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_exclusion_probability.png")

# ============================================================
# Figure 5: Per-mode Exclusion
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 5: Per-mode Exclusion")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (name, info) in enumerate(BH_SYSTEMS.items()):
    ax = axes[idx]
    M = data[name]["M"]
    a = data[name]["a"]
    mu_range = info["mu_range"]
    
    N = len(M)
    if N > 2000:
        idx_sub = np.random.choice(N, 2000, replace=False)
        M_sub = M[idx_sub]
        a_sub = a[idx_sub]
    else:
        M_sub = M
        a_sub = a
    
    for l_max_val in [1, 2]:
        probs_mode = []
        for mu in mu_range:
            p = bayesian_exclusion_probability(mu, M_sub, a_sub,
                                                max_age_yr=info["max_age_yr"],
                                                l_max=l_max_val)
            probs_mode.append(p)
        probs_mode = np.array(probs_mode)
        ax.plot(mu_range, probs_mode, linewidth=2,
                label=f'$\\ell_{{\\max}} = {l_max_val}$')
    
    ax.axhline(y=0.95, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel(r"$\mu_b$ [eV]", fontsize=13)
    ax.set_ylabel(r"Exclusion probability", fontsize=13)
    ax.set_title(f"{info['label']}", fontsize=14)
    ax.set_xscale('log')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "fig5_per_mode_exclusion.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_per_mode_exclusion.png")

# ============================================================
# Figure 6: Self-Interaction Coupling Constraints
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 6: Self-Interaction Coupling Constraints")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

for name, info in BH_SYSTEMS.items():
    M = data[name]["M"]
    a = data[name]["a"]
    mu_range = info["mu_range"]
    
    # For each boson mass, compute the median f_a upper limit across posterior samples
    fa_limits_median = []
    fa_limits_95 = []
    
    for mu in mu_range:
        fa_vals = []
        for i in range(min(500, len(M))):
            fa = bosenova_fa_limit(mu, M[i], a[i], l=1)
            if np.isfinite(fa) and fa > 0:
                fa_vals.append(fa)
        
        if len(fa_vals) > 0:
            fa_limits_median.append(np.median(fa_vals))
            fa_limits_95.append(np.percentile(fa_vals, 95))
        else:
            fa_limits_median.append(np.nan)
            fa_limits_95.append(np.nan)
    
    fa_limits_median = np.array(fa_limits_median)
    fa_limits_95 = np.array(fa_limits_95)
    
    valid = np.isfinite(fa_limits_median)
    if np.any(valid):
        ax.plot(mu_range[valid], fa_limits_median[valid], linewidth=2.5,
                color=info["color"], label=f"{info['label']} (median)")
        ax.fill_between(mu_range[valid], 
                        fa_limits_median[valid],
                        fa_limits_95[valid],
                        alpha=0.2, color=info["color"])

# Reference lines
ax.axhline(y=2.4e18, color='green', linestyle=':', linewidth=1.5, 
           label=r'$M_{\rm Pl}$')
ax.axhline(y=2e16, color='purple', linestyle='--', linewidth=1.5,
           label=r'$M_{\rm GUT}$')

ax.set_xlabel(r"Boson mass $\mu_b$ [eV]", fontsize=14)
ax.set_ylabel(r"$f_a$ upper limit [GeV]", fontsize=14)
ax.set_title("Self-Interaction Coupling Constraints from Bosenova", fontsize=15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "fig6_self_interaction_constraints.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6_self_interaction_constraints.png")

# ============================================================
# Figure 7: Combined Exclusion Plot
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 7: Combined Exclusion Plot")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# Plot exclusion regions as filled areas
for name, info in BH_SYSTEMS.items():
    mu_range = exclusion_results[name]["mu_range"]
    probs = exclusion_results[name]["probs"]
    
    # 95% exclusion region
    excluded_95 = probs >= 0.95
    if np.any(excluded_95):
        ax.fill_between(mu_range, 0, 1, where=excluded_95,
                        alpha=0.3, color=info["color"],
                        label=f"{info['label']} (95% CL)")
    
    # 90% exclusion region
    excluded_90 = probs >= 0.90
    if np.any(excluded_90):
        ax.fill_between(mu_range, 0, 1, where=excluded_90 & ~excluded_95,
                        alpha=0.15, color=info["color"],
                        label=f"{info['label']} (90% CL)")
    
    ax.plot(mu_range, probs, linewidth=2, color=info["color"])

ax.axhline(y=0.95, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel(r"Boson mass $\mu_b$ [eV]", fontsize=14)
ax.set_ylabel(r"Exclusion probability", fontsize=14)
ax.set_title("Combined Ultralight Boson Exclusion from Black Hole Superradiance", fontsize=15)
ax.set_xscale('log')
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add mass regime annotations
ax.annotate('Stellar BH\nregime', xy=(1e-12, 0.02), fontsize=11, ha='center',
            color='royalblue', fontweight='bold')
ax.annotate('SMBH\nregime', xy=(1e-18, 0.02), fontsize=11, ha='center',
            color='crimson', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "fig7_combined_exclusion.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig7_combined_exclusion.png")

# ============================================================
# Figure 8: Alpha-Spin Exclusion Plane
# ============================================================
print("\n" + "=" * 60)
print("Generating Figure 8: Alpha-Spin Exclusion Plane")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

alpha_grid = np.linspace(0.001, 0.5, 500)

for l in [1, 2, 3]:
    a_crit = np.array([critical_spin_lm(a, l, l) for a in alpha_grid])
    ax.plot(alpha_grid, a_crit, linewidth=2.5, label=f'$\\ell = m = {l}$')
    
    # Fill exclusion region (above the curve)
    ax.fill_between(alpha_grid, a_crit, 1.0, alpha=0.1)

# Overlay data points
for name, info in BH_SYSTEMS.items():
    M = data[name]["M"]
    a = data[name]["a"]
    
    # Compute alpha for a representative boson mass
    if info["type"] == "stellar":
        mu_rep = 1e-12
    else:
        mu_rep = 1e-18
    
    alphas = np.array([alpha_grav(mu_rep, m) for m in M])
    
    # Subsample for plotting
    idx_sub = np.random.choice(len(M), min(500, len(M)), replace=False)
    ax.scatter(alphas[idx_sub], a[idx_sub], alpha=0.3, s=5,
               color=info["color"], label=f"{info['label']} ($\\mu = {mu_rep:.0e}$ eV)")

ax.set_xlabel(r"Gravitational coupling $\alpha = G M \mu / (\hbar c)$", fontsize=13)
ax.set_ylabel(r"Black hole spin $a_*$", fontsize=13)
ax.set_title("Superradiance Exclusion in the $\\alpha$-$a_*$ Plane", fontsize=14)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "fig8_alpha_spin_plane.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig8_alpha_spin_plane.png")

# ============================================================
# Save Numerical Results
# ============================================================
print("\n" + "=" * 60)
print("Saving numerical results...")
print("=" * 60)

results_summary = {}

for name, info in BH_SYSTEMS.items():
    M = data[name]["M"]
    a = data[name]["a"]
    mu_range = exclusion_results[name]["mu_range"]
    probs = exclusion_results[name]["probs"]
    
    # Find exclusion ranges at different CLs
    excluded_95 = mu_range[probs >= 0.95]
    excluded_90 = mu_range[probs >= 0.90]
    excluded_50 = mu_range[probs >= 0.50]
    
    result = {
        "bh_system": name,
        "label": info["label"],
        "n_samples": len(M),
        "mass_mean_Msun": float(M.mean()),
        "mass_std_Msun": float(M.std()),
        "spin_mean": float(a.mean()),
        "spin_std": float(a.std()),
        "max_exclusion_prob": float(probs.max()),
        "mu_at_max_exclusion_eV": float(mu_range[np.argmax(probs)]),
    }
    
    if len(excluded_95) > 0:
        result["excluded_95CL_min_eV"] = float(excluded_95.min())
        result["excluded_95CL_max_eV"] = float(excluded_95.max())
    if len(excluded_90) > 0:
        result["excluded_90CL_min_eV"] = float(excluded_90.min())
        result["excluded_90CL_max_eV"] = float(excluded_90.max())
    if len(excluded_50) > 0:
        result["excluded_50CL_min_eV"] = float(excluded_50.min())
        result["excluded_50CL_max_eV"] = float(excluded_50.max())
    
    results_summary[name] = result
    print(f"\n{info['label']}:")
    for k, v in result.items():
        print(f"  {k}: {v}")

# Save results
with open(os.path.join(OUTPUT_DIR, "exclusion_results.json"), 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved exclusion_results.json")

# Save exclusion curves as CSV
for name in BH_SYSTEMS:
    mu_range = exclusion_results[name]["mu_range"]
    probs = exclusion_results[name]["probs"]
    np.savetxt(os.path.join(OUTPUT_DIR, f"exclusion_curve_{name}.csv"),
               np.column_stack([mu_range, probs]),
               header="mu_eV,exclusion_probability",
               delimiter=",")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
