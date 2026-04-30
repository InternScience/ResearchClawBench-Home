#!/usr/bin/env python3
"""
Bayesian black-hole superradiance constraints from mass/spin posterior samples.

This script implements a compact, reproducible approximation to a Bayesian ULB
constraint framework. It ingests the full posterior samples for black-hole mass
and spin, evaluates a superradiance saturation boundary for scalar bosons, and
exports quantitative grids, summaries, validation tables, and PNG figures.

Physics approximation:
  alpha = G M mu/(hbar c) = 7.485e-11 * (M/Msun) * (mu/eV)
  chi_sat(alpha, m=1) = 4 alpha / (1 + 4 alpha^2), from alpha < m Omega_H r_g
  exclusion probability p_excl(mu) = P_posterior[chi_obs > chi_sat(alpha)]
  Growth-time proxy: require alpha in [0.03, 0.5] for efficient scalar l=m=1.

Self-interaction proxy:
  Related work (Arvanitaki & Dubovsky) gives nonlinear importance when
  M_cloud/M_BH ~= epsilon >= 2 l^4 alpha^2 f_a^2/M_Pl^2. We invert for the
  maximum decay constant f_a such that self-interaction can quench a cloud before
  extracting a fiducial spin-down mass fraction epsilon_required. This is a
  decay-constant threshold, not a full dynamical quartic-coupling posterior.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

ALPHA_PER_MSUN_EV = 7.485232053823733e9  # G Msun/(hbar c^3) in 1/eV times eV; alpha≈0.2246 for 30 Msun and 1e-12 eV
MPL_GEV = 1.220890e19
HBAR_EV_S = 6.582119569e-16
EPSILON_REQUIRED = 1e-4

SOURCES = {
    "M33_X-7": DATA / "M33_X-7_samples.dat",
    "IRAS_09149-6206": DATA / "IRAS_09149-6206_samples.dat",
}

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 220, "font.size": 10})


def load_samples(path: Path) -> pd.DataFrame:
    arr = np.loadtxt(path, comments="#")
    return pd.DataFrame({"mass_msun": arr[:, 0], "spin": arr[:, 1]})


def q(x, probs=(0.05, 0.16, 0.5, 0.84, 0.95)):
    return np.quantile(np.asarray(x), probs)


def chi_sat(alpha: np.ndarray, m: int = 1) -> np.ndarray:
    # Solve alpha/m = chi/[2(1+sqrt(1-chi^2))] => chi = 4 beta/(1+4 beta^2)
    beta = alpha / m
    cs = 4 * beta / (1 + 4 * beta**2)
    return np.where((beta > 0) & (beta < 0.5), cs, np.nan)


def alpha_from_mu(mass_msun, mu_ev):
    return ALPHA_PER_MSUN_EV * np.asarray(mass_msun) * mu_ev


def growth_time_proxy_s(alpha, mu_ev):
    # Hydrogenic l=m=1 rate scaling Gamma ~ alpha^9 mu / 24; deliberately only a proxy.
    alpha = np.asarray(alpha)
    gamma_ev = np.where(alpha > 0, (alpha**9) * mu_ev / 24.0, np.nan)
    gamma_s = gamma_ev / HBAR_EV_S
    return 1 / gamma_s


def hdi_intervals(log_mu, p, threshold=0.95):
    mask = np.asarray(p) >= threshold
    x = np.asarray(log_mu)
    if not mask.any():
        return []
    intervals = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if start is not None and ((not val) or i == len(mask) - 1):
            end = i - 1 if not val else i
            intervals.append((10 ** x[start], 10 ** x[end], int(end - start + 1)))
            start = None
    return intervals


def main():
    # Save dependency check.
    dep = {}
    for mod in ["numpy", "pandas", "matplotlib", "seaborn", "scipy", "pypdf"]:
        try:
            __import__(mod)
            dep[mod] = {"available": True}
        except Exception as e:
            dep[mod] = {"available": False, "error": repr(e)}
    dep["capability_notes"] = [
        "Exact Teukolsky/Kerr superradiance rates are not implemented; analysis uses an analytic scalar l=m=1 saturation boundary and growth-window proxy.",
        "Self-interaction treatment uses the related-work Bosenova/nonlinearity threshold inverted to a decay-constant/coupling proxy, not a full nonlinear cloud simulation."
    ]
    (OUT / "dependency_check.json").write_text(json.dumps(dep, indent=2))

    samples = {name: load_samples(path) for name, path in SOURCES.items()}
    summary_rows = []
    for name, df in samples.items():
        mass_q = q(df.mass_msun)
        spin_q = q(df.spin)
        summary_rows.append({
            "source": name, "n_samples": len(df),
            "mass_mean_msun": df.mass_msun.mean(), "mass_sd_msun": df.mass_msun.std(ddof=1),
            "mass_q05_msun": mass_q[0], "mass_q16_msun": mass_q[1], "mass_median_msun": mass_q[2],
            "mass_q84_msun": mass_q[3], "mass_q95_msun": mass_q[4],
            "spin_mean": df.spin.mean(), "spin_sd": df.spin.std(ddof=1),
            "spin_q05": spin_q[0], "spin_q16": spin_q[1], "spin_median": spin_q[2],
            "spin_q84": spin_q[3], "spin_q95": spin_q[4],
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "data_summary.csv", index=False)

    # Source-adapted mass grids plus broad combined grid.
    grids = {}
    for name, df in samples.items():
        mmed = df.mass_msun.median()
        mu_center = 0.25 / (ALPHA_PER_MSUN_EV * mmed)  # alpha ~0.25 peak sensitivity
        grids[name] = np.logspace(np.log10(mu_center) - 1.3, np.log10(mu_center) + 1.3, 700)
    combined_grid = np.unique(np.concatenate(list(grids.values())))
    combined_grid.sort()

    grid_rows = []
    posterior_alpha_rows = []
    rng = np.random.default_rng(12345)
    for name, df in samples.items():
        mass = df.mass_msun.to_numpy()
        spin = df.spin.to_numpy()
        for mu in combined_grid:
            alpha = alpha_from_mu(mass, mu)
            cs = chi_sat(alpha, 1)
            in_window = np.isfinite(cs) & (alpha >= 0.03) & (alpha <= 0.5)
            superspin = in_window & (spin > cs)
            # Strong if posterior spin is sufficiently above boundary and alpha in efficient window.
            p_excl = float(np.mean(superspin))
            # Also evaluate pure kinematic superradiance condition without growth-window lower cutoff.
            p_kin = float(np.mean(np.isfinite(cs) & (spin > cs)))
            if np.any(in_window):
                med_chi_sat = float(np.nanmedian(cs[in_window]))
                med_alpha = float(np.median(alpha[in_window]))
                med_tau = float(np.nanmedian(growth_time_proxy_s(alpha[in_window], mu)))
            else:
                med_chi_sat = np.nan; med_alpha = np.nan; med_tau = np.nan
            # self-interaction: maximum f_a for nonlinear effects before epsilon_required.
            # epsilon >= 2 alpha^2 f_a^2/Mpl^2 => f_a_crit = Mpl sqrt(eps/(2 alpha^2)).
            with np.errstate(divide='ignore', invalid='ignore'):
                fa_crit = MPL_GEV * np.sqrt(EPSILON_REQUIRED / (2 * alpha**2))
                lambda_crit = (mu * 1e-9 / fa_crit)**2  # dimensionless quartic proxy lambda~(mu/fa)^2; mu eV -> GeV
            fa_med = float(np.nanmedian(fa_crit[in_window])) if np.any(in_window) else np.nan
            fa_q16 = float(np.nanquantile(fa_crit[in_window], 0.16)) if np.any(in_window) else np.nan
            fa_q84 = float(np.nanquantile(fa_crit[in_window], 0.84)) if np.any(in_window) else np.nan
            lam_med = float(np.nanmedian(lambda_crit[in_window])) if np.any(in_window) else np.nan
            grid_rows.append({
                "source": name, "mu_ev": mu, "log10_mu_ev": np.log10(mu),
                "p_exclusion_growth_window": p_excl,
                "p_exclusion_kinematic": p_kin,
                "posterior_fraction_in_alpha_window": float(np.mean(in_window)),
                "median_alpha_in_window": med_alpha,
                "median_chi_sat_in_window": med_chi_sat,
                "growth_time_proxy_s_median": med_tau,
                "fa_crit_GeV_median": fa_med,
                "fa_crit_GeV_q16": fa_q16,
                "fa_crit_GeV_q84": fa_q84,
                "quartic_lambda_crit_median": lam_med,
            })
        # selected posterior alpha distributions at source's max exclusion point later; save all only downsampled
        idx = rng.choice(len(df), size=min(4000, len(df)), replace=False)
        for mu in np.geomspace(grids[name].min(), grids[name].max(), 15):
            al = alpha_from_mu(mass[idx], mu)
            posterior_alpha_rows.extend({"source": name, "mu_ev": mu, "alpha": a} for a in al)

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "exclusion_grid.csv", index=False)
    pd.DataFrame(posterior_alpha_rows).to_csv(OUT / "posterior_alpha_samples_for_validation.csv", index=False)

    # Constraint intervals and direct summaries.
    cons_rows = []
    for name, g in grid.groupby("source"):
        g = g.sort_values("mu_ev")
        imax = g.p_exclusion_growth_window.idxmax()
        row = g.loc[imax]
        intervals95 = hdi_intervals(g.log10_mu_ev, g.p_exclusion_growth_window, 0.95)
        intervals68 = hdi_intervals(g.log10_mu_ev, g.p_exclusion_growth_window, 0.68)
        cons_rows.append({
            "source": name,
            "peak_mu_ev": row.mu_ev,
            "peak_p_exclusion": row.p_exclusion_growth_window,
            "peak_p_kinematic": row.p_exclusion_kinematic,
            "peak_median_alpha": row.median_alpha_in_window,
            "peak_median_chi_sat": row.median_chi_sat_in_window,
            "excluded_intervals_p_ge_0p95": json.dumps([(a,b) for a,b,n in intervals95]),
            "excluded_intervals_p_ge_0p68": json.dumps([(a,b) for a,b,n in intervals68]),
            "n_95_intervals": len(intervals95),
            "n_68_intervals": len(intervals68),
        })
    cons = pd.DataFrame(cons_rows)
    cons.to_csv(OUT / "constraint_summary.csv", index=False)

    # Self-interaction limits at masses with p_exclusion >= 0.68 and peak.
    si_rows = []
    for name, g in grid.groupby("source"):
        for thresh in [0.68, 0.95]:
            gg = g[g.p_exclusion_growth_window >= thresh]
            if len(gg):
                si_rows.append({
                    "source": name, "p_exclusion_threshold": thresh,
                    "mu_min_ev": gg.mu_ev.min(), "mu_max_ev": gg.mu_ev.max(),
                    "fa_crit_min_GeV": gg.fa_crit_GeV_median.min(),
                    "fa_crit_max_GeV": gg.fa_crit_GeV_median.max(),
                    "quartic_lambda_crit_min": gg.quartic_lambda_crit_median.min(),
                    "quartic_lambda_crit_max": gg.quartic_lambda_crit_median.max(),
                    "interpretation": "For f_a below fa_crit, attractive self-interactions can become nonlinear before a fiducial epsilon=1e-4 cloud extracts the modeled spin; lambda~(mu/fa)^2 is a proxy."
                })
            else:
                si_rows.append({"source": name, "p_exclusion_threshold": thresh, "mu_min_ev": np.nan, "mu_max_ev": np.nan,
                                "fa_crit_min_GeV": np.nan, "fa_crit_max_GeV": np.nan,
                                "quartic_lambda_crit_min": np.nan, "quartic_lambda_crit_max": np.nan,
                                "interpretation": "No mass grid points reached this posterior exclusion threshold."})
    si = pd.DataFrame(si_rows)
    si.to_csv(OUT / "self_interaction_limits.csv", index=False)

    # Figure 1: posterior overview.
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.8))
    for axrow, (name, df) in zip(axes, samples.items()):
        ax = axrow[0]
        ax.scatter(df.mass_msun, df.spin, s=5, alpha=0.16, rasterized=True)
        ax.set_xscale('log')
        ax.set_xlabel(r"$M_{\rm BH}\,[M_\odot]$")
        ax.set_ylabel(r"spin $a_*$")
        ax.set_title(f"{name}: posterior samples")
        ax.set_ylim(max(0, df.spin.min()-0.05), min(1.02, df.spin.max()+0.05))
        ax = axrow[1]
        ax.hist(df.spin, bins=45, alpha=0.75, density=True, label='spin')
        ax2 = ax.twiny()
        ax2.hist(np.log10(df.mass_msun), bins=45, alpha=0.35, density=True, color='tab:orange', label='log10 mass')
        ax.set_xlabel(r"spin $a_*$")
        ax.set_ylabel("posterior density")
        ax2.set_xlabel(r"$\log_{10}(M/M_\odot)$")
        ax.set_title(f"{name}: 1D marginals")
    fig.tight_layout()
    fig.savefig(IMG / "figure1_posterior_overview.png")
    plt.close(fig)

    # Figure 2: Regge-plane validation with boundaries for representative mu.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (name, df) in zip(axes, samples.items()):
        ax.scatter(df.mass_msun, df.spin, s=5, alpha=0.12, color='k', label='posterior')
        mu_peak = cons.loc[cons.source == name, 'peak_mu_ev'].iloc[0]
        mgrid = np.logspace(np.log10(df.mass_msun.quantile(0.005))-0.2, np.log10(df.mass_msun.quantile(0.995))+0.2, 500)
        for fac, ls in [(0.5, '--'), (1.0, '-'), (2.0, ':')]:
            mu = mu_peak * fac
            al = alpha_from_mu(mgrid, mu)
            cs = chi_sat(al)
            valid = np.isfinite(cs) & (al >= 0.03) & (al <= 0.5)
            ax.plot(mgrid[valid], cs[valid], ls=ls, lw=2, label=fr"$\mu={fac:g}\mu_{{peak}}$")
        ax.set_xscale('log')
        ax.set_xlabel(r"$M_{\rm BH}\,[M_\odot]$")
        ax.set_title(name)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"spin $a_*$")
    fig.suptitle("Regge-plane validation: posterior samples versus scalar superradiance saturation curves")
    fig.tight_layout()
    fig.savefig(IMG / "figure2_regge_validation.png")
    plt.close(fig)

    # Figure 3: mass exclusion curves.
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for name, g in grid.groupby('source'):
        g = g.sort_values('mu_ev')
        ax.plot(g.mu_ev, g.p_exclusion_growth_window, lw=2, label=f"{name}: growth-window posterior")
        ax.plot(g.mu_ev, g.p_exclusion_kinematic, lw=1, alpha=0.45, ls='--', label=f"{name}: kinematic")
        peak = cons[cons.source == name].iloc[0]
        ax.axvline(peak.peak_mu_ev, lw=1, alpha=0.5)
    ax.axhline(0.95, color='grey', ls=':', lw=1)
    ax.axhline(0.68, color='grey', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"ULB mass $\mu$ [eV]")
    ax.set_ylabel(r"posterior exclusion probability")
    ax.set_title("Bayesian superradiance exclusion probability from full posteriors")
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(IMG / "figure3_mass_exclusion.png")
    plt.close(fig)

    # Figure 4: self-interaction f_a threshold / quartic proxy.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, g in grid.groupby('source'):
        g = g.sort_values('mu_ev')
        mask = g.posterior_fraction_in_alpha_window > 0
        axes[0].plot(g.mu_ev[mask], g.fa_crit_GeV_median[mask], lw=2, label=name)
        axes[0].fill_between(g.mu_ev[mask], g.fa_crit_GeV_q16[mask], g.fa_crit_GeV_q84[mask], alpha=0.18)
        axes[1].plot(g.mu_ev[mask], g.quartic_lambda_crit_median[mask], lw=2, label=name)
    for ax in axes:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(fontsize=8)
        ax.set_xlabel(r"ULB mass $\mu$ [eV]")
    axes[0].set_ylabel(r"$f_{a,crit}$ [GeV]")
    axes[0].set_title(r"Decay-constant threshold for nonlinear cloud effects")
    axes[1].set_ylabel(r"quartic proxy $\lambda_{crit}\sim(\mu/f_a)^2$")
    axes[1].set_title("Equivalent coupling-strength proxy")
    fig.tight_layout()
    fig.savefig(IMG / "figure4_self_interaction.png")
    plt.close(fig)

    # Source data for figures.
    summary.to_json(OUT / "figure1_source_data.json", orient="records", indent=2)
    grid.to_csv(OUT / "figure3_source_data.csv", index=False)
    grid[["source","mu_ev","fa_crit_GeV_median","fa_crit_GeV_q16","fa_crit_GeV_q84","quartic_lambda_crit_median"]].to_csv(OUT / "figure4_source_data.csv", index=False)

    # Related-work contract extraction.
    rw = {
        "papers_read_with_pypdf": [
            {"file": "paper_000.pdf", "title": "Exploring the String Axiverse with Precision Black Hole Physics", "task_relevant_points": [
                "Axions with Compton wavelength comparable to black-hole size form gravitational atoms and spin down black holes by superradiance.",
                "Observable signature is a gap in the black-hole mass-spin (Regge) plane; Regge trajectories mark saturation boundaries.",
                "Attractive self-interactions can cause Bosenova collapse; nonlinear importance condition reported as M_cloud/M_BH >= 2 l^4 alpha^2 f_a^2/M_Pl^2.",
                "Existing spin data were used to bound QCD axion decay constants, motivating f_a/self-interaction reporting."
            ]},
            {"file": "paper_001.pdf", "title": "The Spectrum of the Axion Dark Sector, Cosmological Observable and Black Hole Superradiance Constraints", "task_relevant_points": [
                "Black hole superradiance probes approximately 10^-20 to 10^-11 eV boson masses.",
                "Constraints are expressed as isocontour exclusion regions in the black-hole mass-spin Regge plane."
            ]},
            {"file": "paper_002.pdf", "title": "Black Hole Mergers and the QCD Axion at Advanced LIGO", "task_relevant_points": [
                "Defines alpha = G_N M_BH mu_a ~ 0.22 (M/30 Msun)(mu/1e-12 eV).",
                "Superradiance condition omega/m < Omega_H; scalar l=m=1 fastest level has growth rate scaling with alpha.",
                "High-spin X-ray binaries disfavor axion masses around 6e-13 to 2e-11 eV in previous point-estimate analyses."
            ]},
            {"file": "paper_003.pdf", "title": "Superradiant instabilities in astrophysical systems", "task_relevant_points": [
                "Massive scalar/vector fields around spinning black holes can develop quasi-bound states and superradiant instabilities.",
                "Numerical work supports that spin measurements can bound ultralight bosonic field masses."
            ]}
        ],
        "contract_updates": [
            "Use Regge-plane mass-spin exclusion plots as a central validation/comparison figure.",
            "Report the gravitational fine-structure parameter alpha and scalar saturation boundary.",
            "Treat self-interaction constraints as decay-constant/nonlinear-threshold proxies unless full Bosenova dynamics is implemented."
        ]
    }
    (OUT / "related_work_contract.json").write_text(json.dumps(rw, indent=2))

    # Refresh method contract and inventory.
    method_contract = {
        "task": "Bayesian ULB constraints from black-hole mass/spin posterior samples.",
        "implemented_commitments": [
            "Full posterior samples are loaded and each boson mass is scored by posterior probability that a draw falls in the forbidden superradiant region.",
            "Scalar l=m=1 Regge saturation curve is used: chi_sat=4 alpha/(1+4 alpha^2) for alpha<0.5.",
            "Mass constraints are reported as p>=0.68 and p>=0.95 exclusion intervals for each source.",
            "Self-interaction strength is constrained with a related-work Bosenova/nonlinearity threshold inverted to f_a and lambda~(mu/f_a)^2 proxies."
        ],
        "known_deviations": [
            "No exact Kerr/Teukolsky eigenvalue calculation or source-age/accretion modeling is performed.",
            "Self-interaction limits are not full upper limits from nonlinear simulations; they are threshold proxies tied to epsilon_required=1e-4.",
            "Only scalar l=m=1 is modeled; vector fields and higher levels are discussed but not fit."
        ],
        "constants": {"alpha_per_msun_ev": ALPHA_PER_MSUN_EV, "Mpl_GeV": MPL_GEV, "epsilon_required": EPSILON_REQUIRED}
    }
    (OUT / "method_contract.json").write_text(json.dumps(method_contract, indent=2))

    fidelity = {
        "named_method": "Bayesian posterior-sample black-hole superradiance Regge-plane exclusion",
        "definition": [
            "For every boson mass, map each posterior mass sample to alpha=G M mu/(hbar c).",
            "Compute scalar saturation spin from the horizon superradiance condition alpha/m < Omega_H r_g.",
            "Estimate exclusion probability as the posterior mass-spin fraction above the saturation curve within an efficient alpha window.",
            "Summarize exclusions by source-specific posterior probabilities, not point estimates."
        ],
        "invariants_checked": {
            "posterior_samples_used_directly": True,
            "source_specific_outputs": True,
            "regge_plane_figure": True,
            "self_interaction_proxy_documented": True,
            "exact_growth_rates": False,
            "full_nonlinear_bosenova_simulation": False
        },
        "fallbacks": [
            "Analytic saturation boundary substituted for exact Kerr spectrum.",
            "Alpha-window growth proxy substituted for source-age-dependent instability integration."
        ]
    }
    (OUT / "method_fidelity_checklist.json").write_text(json.dumps(fidelity, indent=2))

    # Claim recovery table.
    claims = [
        ["The analysis used full posterior samples rather than point estimates.", "data_summary.csv; exclusion_grid.csv", "Each p_exclusion value is a posterior fraction over all mass/spin draws."],
        ["M33 X-7 constrains stellar-mass ULB masses near its peak_mu_ev.", "constraint_summary.csv; figure3_mass_exclusion.png", "Peak and interval values are source-specific."],
        ["IRAS 09149-6206 constrains supermassive-BH ULB masses near its peak_mu_ev.", "constraint_summary.csv; figure3_mass_exclusion.png", "Peak and interval values are source-specific."],
        ["Regge-plane boundary explains the exclusion calculation.", "figure2_regge_validation.png; method_fidelity_checklist.json", "Posterior points above the saturation curve contribute to exclusion probability."],
        ["Self-interaction coupling results are threshold proxies, not exact nonlinear limits.", "self_interaction_limits.csv; dependency_check.json", "Based on related-work condition for nonlinear cloud importance."
        ]
    ]
    pd.DataFrame(claims, columns=["claim", "supporting_artifacts", "recovery_note"]).to_csv(OUT / "claim_recovery.csv", index=False)

    # Mark inventory complete.
    inventory = {
        "primary_quantitative_outputs": [
            {"artifact": "source posterior summaries", "target_path": "outputs/data_summary.csv", "status": "satisfied"},
            {"artifact": "mass-grid exclusion probabilities for each source", "target_path": "outputs/exclusion_grid.csv", "status": "satisfied"},
            {"artifact": "direct ULB mass constraints", "target_path": "outputs/constraint_summary.csv", "status": "satisfied"},
            {"artifact": "self-interaction coupling proxy constraints", "target_path": "outputs/self_interaction_limits.csv", "status": "satisfied"}
        ],
        "figures": [
            {"artifact": "posterior data overview", "target_path": "report/images/figure1_posterior_overview.png", "status": "satisfied"},
            {"artifact": "superradiance Regge-plane validation/comparison", "target_path": "report/images/figure2_regge_validation.png", "status": "satisfied"},
            {"artifact": "mass exclusion probabilities", "target_path": "report/images/figure3_mass_exclusion.png", "status": "satisfied"},
            {"artifact": "self-interaction coupling limits", "target_path": "report/images/figure4_self_interaction.png", "status": "satisfied"}
        ],
        "validation_artifacts": [
            {"artifact": "dependency check", "target_path": "outputs/dependency_check.json", "status": "satisfied"},
            {"artifact": "method fidelity checklist", "target_path": "outputs/method_fidelity_checklist.json", "status": "satisfied"},
            {"artifact": "claim recovery table", "target_path": "outputs/claim_recovery.csv", "status": "satisfied"}
        ],
        "unsatisfied_or_limited": [
            {"artifact": "exact self-interaction upper limits from nonlinear simulations", "status": "limited", "reason": "not available in workspace and beyond compact analysis; threshold proxy reported instead"},
            {"artifact": "exact source-age/accretion superradiance timescale likelihood", "status": "limited", "reason": "source ages/accretion histories absent from provided data; alpha-window and growth-rate proxy reported"}
        ]
    }
    (OUT / "target_artifact_inventory.json").write_text(json.dumps(inventory, indent=2))

    print(json.dumps({
        "data_summary": summary.to_dict(orient='records'),
        "constraints": cons.to_dict(orient='records'),
        "self_interaction": si.to_dict(orient='records'),
        "figures": [str(p.relative_to(ROOT)) for p in sorted(IMG.glob('*.png'))]
    }, indent=2))

if __name__ == "__main__":
    main()
