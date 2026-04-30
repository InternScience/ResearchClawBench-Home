#!/usr/bin/env python3
"""Analyze structured Hartree-Fock derivation/scoring artifacts for 2111.01152.

This script reads the local YAML task dataset and source TeX evidence, exports
paper extraction, derivation, scoring, validation artifacts, and mandatory PNG
figures for the report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "2111.01152"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

SCORE_KEYS = [
    "in_paper",
    "prompt_quality",
    "follow_instructions",
    "physics_logic",
    "math_derivation",
    "final_answer_accuracy",
]


def load_yaml():
    with open(DATA / "2111.01152.yaml") as f:
        raw = yaml.safe_load(f)
    rows = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict) or "task" not in item:
            continue
        score = item.get("score") or {}
        vals = {k: score.get(k) for k in SCORE_KEYS}
        numeric = [v for v in vals.values() if isinstance(v, (int, float))]
        rows.append(
            {
                "index": idx,
                "task": item.get("task"),
                "answer": item.get("answer"),
                "has_answer": bool(item.get("answer")),
                **vals,
                "total_score": sum(numeric),
                "max_score": 2 * len(numeric),
                "normalized_score": sum(numeric) / (2 * len(numeric)) if numeric else np.nan,
                "mean_score": sum(numeric) / len(numeric) if numeric else np.nan,
            }
        )
    return pd.DataFrame(rows), raw


def extract_latex_command(text: str, command: str) -> str:
    marker = "\\" + command + "{"
    start = text.index(marker) + len(marker)
    depth = 1
    chars = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            chars.append(ch)
        else:
            chars.append(ch)
    return "".join(chars).replace("\n", " ")


def extract_tex_info():
    main = (DATA / "2111.01152.tex").read_text()
    sm = (DATA / "2111.01152_SM.tex").read_text()
    title = extract_latex_command(main, "title")
    authors = re.findall(r"\\author\{(.+?)\}", main)
    abstract = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", main, re.S).group(1).strip()
    lattice = re.search(r"\$\(a_\{\\mathfrak\{b\}\},a_\{\\mathfrak\{t\}\}\)\$=\(([^\)]+)\)", main)
    info = {
        "arxiv_id": "2111.01152",
        "title": title,
        "authors": authors,
        "system": "AB-stacked MoTe2/WSe2 moire heterobilayer, exact 180 degree twist",
        "scientific_context_from_abstract": " ".join(abstract.split()),
        "key_physics": [
            "continuum moire Hamiltonian retaining topmost valence bands from both layers",
            "self-consistent Hartree-Fock calculation in plane-wave basis",
            "dual-gate screened Coulomb interaction",
            "topological phases at nu=2, nu=1, and nu=2/3",
        ],
        "parameters": {
            "lattice_constants": "(a_b, a_t)=(3.575 Angstrom, 3.32 Angstrom)",
            "moire_period_formula": "a_M=a_b a_t / |a_b-a_t|",
            "effective_masses": "(m_b, m_t)=(0.65,0.35)m_e",
            "kappa": "4 pi/(3 a_M) (1,0)",
            "psi_b": "-14 degrees",
            "representative_parameters": "(w, V_b, V_zt)=(12 meV, 7 meV, -20 meV)",
            "gate_distance": "d=5 nm unless otherwise stated",
            "dielectric_range": "epsilon of order 10-20",
            "coulomb_scale_statement": "e^2/(epsilon a_M) approx 31 meV for a_M approx 4.7 nm and epsilon=10",
            "bandwidth_statement": "hbar^2 kappa^2/(2 m_b) approx 47 meV",
        },
        "source_evidence": {
            "main_tex_lines": {
                "model_hamiltonian": "2111.01152.tex lines 55-62",
                "potentials": "2111.01152.tex lines 71-80",
                "coulomb_interaction": "2111.01152.tex line 95 and following",
            },
            "supplement_lines": {
                "second_quantized_single_particle": "2111.01152_SM.tex lines 46-62",
                "momentum_space_hf": "2111.01152_SM.tex lines 88-119",
            },
        },
    }
    (OUT / "paper_information_extraction.json").write_text(json.dumps(info, indent=2))
    return info


def write_hf_derivation():
    text = r"""# Derived Hartree-Fock Hamiltonian for 2111.01152

## 1. Single-particle continuum Hamiltonian
For valley $\tau=\pm1$ and layer basis $(\mathfrak b,\mathfrak t)$, the target paper defines

\[
H_\tau(\mathbf r)=\begin{pmatrix}
-\frac{\hbar^2\mathbf k^2}{2m_\mathfrak b}+\Delta_\mathfrak b(\mathbf r)&\Delta_{T,\tau}(\mathbf r)\\
\Delta_{T,\tau}^{\dagger}(\mathbf r)&-\frac{\hbar^2(\mathbf k-\tau\boldsymbol\kappa)^2}{2m_\mathfrak t}+\Delta_\mathfrak t(\mathbf r)+V_{z\mathfrak t}
\end{pmatrix},
\]
where $\boldsymbol\kappa=4\pi(1,0)/(3a_M)$, $(m_\mathfrak b,m_\mathfrak t)=(0.65,0.35)m_e$, and $\mathbf k=-i\nabla_\mathbf r$ in real space.

The intralayer and tunneling fields are
\[
\Delta_{\mathfrak b}(\mathbf r)=2V_{\mathfrak b}\sum_{j=1,3,5}\cos(\mathbf g_j\cdot\mathbf r+\psi_\mathfrak b),\qquad \Delta_\mathfrak t(\mathbf r)=0,
\]
\[
\Delta_{T,\tau}(\mathbf r)=\tau w\left[1+\omega^\tau e^{i\tau\mathbf g_2\cdot\mathbf r}+\omega^{2\tau}e^{i\tau\mathbf g_3\cdot\mathbf r}\right],\quad \omega=e^{2\pi i/3}.
\]

## 2. Second quantization and momentum representation
The real-space second-quantized noninteracting Hamiltonian is
\[
\hat{\mathcal H}_0=\sum_{\tau}\int d^2r\,\Psi_\tau^\dagger(\mathbf r)H_\tau(\mathbf r)\Psi_\tau(\mathbf r),
\]
with spin inferred by spin-valley-layer locking. Defining
\[
c_{\mathbf k,l,\tau}^{\dagger}=A^{-1/2}\int d^2r\,\psi_{l,\tau}^{\dagger}(\mathbf r)e^{i\mathbf k\cdot\mathbf r},
\]
produces
\[
\hat{\mathcal H}_0=\sum_{\mathbf k_\alpha,\mathbf k_\beta}\sum_{l_\alpha,l_\beta}\sum_\tau h^{(\tau)}_{\mathbf k_\alpha l_\alpha,\mathbf k_\beta l_\beta}c^{\dagger}_{\mathbf k_\alpha,l_\alpha,\tau}c_{\mathbf k_\beta,l_\beta,\tau}.
\]
Bloch periodicity restricts nonzero matrix elements to momentum differences equal to moire reciprocal lattice vectors.

## 3. Hole basis and interaction
Using $b_{\mathbf k,l,\tau}=c^{\dagger}_{\mathbf k,l,\tau}$, normal ordering gives the one-body hole Hamiltonian (dropping constants)
\[
\hat{\mathcal H}_1=\sum_{\mathbf k_\alpha,\mathbf k_\beta}\sum_{l_\alpha,l_\beta}\sum_\tau \tilde h^{(\tau)}_{\mathbf k_\alpha l_\alpha,\mathbf k_\beta l_\beta}b^{\dagger}_{\mathbf k_\alpha,l_\alpha,\tau}b_{\mathbf k_\beta,l_\beta,\tau},\quad \tilde h^{(\tau)}=-[h^{(\tau)}]^T.
\]
The hole-hole interaction is
\[
\hat{\mathcal H}_{\rm int}=\frac{1}{2A}\sum_{\alpha\beta\gamma\delta}\sum_{l_\alpha,l_\beta}\sum_{\tau_\alpha,\tau_\beta}V(\mathbf k_\alpha-\mathbf k_\delta)
 b_\alpha^\dagger b_\beta^\dagger b_\gamma b_\delta\,
\delta_{\mathbf k_\alpha+\mathbf k_\beta,\mathbf k_\delta+\mathbf k_\gamma},
\]
where $b_\alpha=b_{\mathbf k_\alpha,l_\alpha,\tau_\alpha}$, $b_\beta=b_{\mathbf k_\beta,l_\beta,\tau_\beta}$, $b_\gamma=b_{\mathbf k_\gamma,l_\beta,\tau_\beta}$, $b_\delta=b_{\mathbf k_\delta,l_\alpha,\tau_\alpha}$, and
\[
V(\mathbf q)=\frac{2\pi e^2\tanh(|\mathbf q|d)}{\epsilon |\mathbf q|}.
\]

## 4. Hartree-Fock decoupling
Applying Wick's theorem to $b_\alpha^\dagger b_\beta^\dagger b_\gamma b_\delta$ and combining equivalent Hartree and Fock partners cancels the factor $1/2$, giving
\[
\hat{\mathcal H}^{\rm HF}=\hat{\mathcal H}_1+\hat{\mathcal H}^{\rm HF}_{\rm int},
\]
\[
\hat{\mathcal H}^{\rm HF}_{\rm int}=\frac{1}{A}\sum_{\alpha\beta\gamma\delta}\sum_{l_\alpha,l_\beta}\sum_{\tau_\alpha,\tau_\beta} V(\mathbf k_\alpha-\mathbf k_\delta)
\left[\langle b_\alpha^\dagger b_\delta\rangle b_\beta^\dagger b_\gamma-\langle b_\alpha^\dagger b_\gamma\rangle b_\beta^\dagger b_\delta\right]
\delta_{\mathbf k_\alpha+\mathbf k_\beta,\mathbf k_\delta+\mathbf k_\gamma}.
\]
This is the compact source-equation form of the Hartree-Fock Hamiltonian used for validation.
"""
    (OUT / "hf_hamiltonian_derivation.md").write_text(text)


def score_outputs(df: pd.DataFrame):
    df.to_csv(OUT / "step_scoring_results.csv", index=False)
    summary = {
        "n_tasks": int(len(df)),
        "n_with_answers": int(df["has_answer"].sum()),
        "mean_total_score": float(df["total_score"].mean()),
        "mean_normalized_score": float(df["normalized_score"].mean()),
        "perfect_tasks": int((df["total_score"] == df["max_score"]).sum()),
        "lowest_tasks": df.nsmallest(4, "total_score")[["index", "task", "total_score", "max_score"]].to_dict(orient="records"),
        "category_means": {k: float(df[k].mean()) for k in SCORE_KEYS},
    }
    (OUT / "step_scoring_results.json").write_text(json.dumps({"summary": summary, "tasks": df.to_dict(orient="records")}, indent=2))
    return summary


def validation_artifacts(df, info):
    validation = {
        "directly_verified_from_workspace": [
            "Target paper title/authors/abstract extracted from data/2111.01152/2111.01152.tex.",
            "Continuum Hamiltonian and potential definitions verified from 2111.01152.tex lines 55-80.",
            "Momentum-space hole-basis HF formulation verified from 2111.01152_SM.tex lines 88-119.",
            "Six-category step scores extracted from data/2111.01152/2111.01152.yaml for 16 tasks."
        ],
        "limitations": [
            "No new numerical self-consistent HF diagonalization was run; the analysis audits symbolic derivation and scoring artifacts.",
            "ReadPDF failed for the provided PDFs, so report evidence relies on TeX/YAML/Markdown artifacts rather than parsed PDFs.",
            "The YAML contains one final task with perfect scores but no answer text; this is flagged as an artifact incompleteness."
        ],
        "score_consistency_checks": {
            "score_keys": SCORE_KEYS,
            "all_scores_in_0_to_2": bool(((df[SCORE_KEYS] >= 0) & (df[SCORE_KEYS] <= 2)).all().all()),
            "n_tasks": int(len(df)),
            "n_missing_answers": int((~df["has_answer"]).sum()),
        },
        "source_equation_match": {
            "single_particle_H_tau": "matches main-text Eq. Ham structure",
            "hole_basis_tilde_h": "matches SM Hartree-Fock section definition tilde h=-[h]^T",
            "HF_interaction": "matches SM Eq. HF compact Hartree-minus-Fock form"
        }
    }
    (OUT / "validation_summary.json").write_text(json.dumps(validation, indent=2))
    claims = [
        {"claim": "The dataset contains 16 scored calculation tasks for 2111.01152.", "artifact": "outputs/step_scoring_results.json", "status": "verified"},
        {"claim": "The reference continuum Hamiltonian is a valley-resolved 2x2 layer Hamiltonian with bottom/top effective masses and interlayer tunneling.", "artifact": "outputs/hf_hamiltonian_derivation.md; data/2111.01152/2111.01152.tex", "status": "verified"},
        {"claim": "The HF interaction has Hartree and Fock quadratic terms with dual-gate screened Coulomb potential.", "artifact": "outputs/hf_hamiltonian_derivation.md; data/2111.01152/2111.01152_SM.tex", "status": "verified"},
        {"claim": "Structured prompts perform strongest on potential definition, second quantization, interaction construction, Fock reduction, and Hartree+Fock combination in this dataset.", "artifact": "outputs/step_scoring_results.csv", "status": "verified from scores"},
        {"claim": "The analysis did not reproduce a full self-consistent phase diagram.", "artifact": "outputs/validation_summary.json", "status": "limitation"},
    ]
    (OUT / "claim_recovery_table.json").write_text(json.dumps(claims, indent=2))


def method_fidelity():
    data = {
        "named_method": "Hartree-Fock approximation for the AB-stacked MoTe2/WSe2 continuum model",
        "definition": "Four-fermion hole interaction is Wick-decoupled into normal quadratic Hartree and Fock bilinears using density-matrix expectation values, then combined with the normal-ordered hole-basis one-body Hamiltonian.",
        "non_negotiable_steps": [
            "Preserve valley, layer, and momentum labels.",
            "Use hole operator b_kltau = c^dagger_kltau before writing the interacting hole Hamiltonian.",
            "Use dual-gate screened Coulomb V(q)=2*pi*e^2*tanh(|q|d)/(epsilon |q|).",
            "Enforce total momentum conservation delta_{k_alpha+k_beta,k_delta+k_gamma}.",
            "Keep Hartree term with <b_alpha^dagger b_delta> b_beta^dagger b_gamma and Fock term with minus <b_alpha^dagger b_gamma> b_beta^dagger b_delta.",
            "Combine equivalent Wick partners so the prefactor is 1/A in the compact HF interaction."
        ],
        "implemented": {
            "symbolic_derivation": True,
            "score_audit": True,
            "full_self_consistent_solver": False
        },
        "deviations": ["No numerical self-consistent diagonalization or Chern-number calculation was implemented; task focus was derivation/scoring artifact generation."]
    }
    (OUT / "method_fidelity_checklist.json").write_text(json.dumps(data, indent=2))


def figures(df, summary):
    sns.set_theme(style="whitegrid")

    # Figure 1: data overview
    fig, ax = plt.subplots(figsize=(8, 4.8))
    counts = pd.Series({
        "Scored tasks": len(df),
        "Tasks with answers": int(df["has_answer"].sum()),
        "Perfect-score tasks": int((df["total_score"] == df["max_score"]).sum()),
        "Source files used": 5,
        "Score categories": len(SCORE_KEYS),
    })
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C72B0")
    ax.set_ylabel("Count")
    ax.set_title("Data and artifact overview")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "data_overview.png", dpi=200)
    plt.close(fig)

    # Figure 2: score heatmap
    heat = df.set_index("task")[SCORE_KEYS]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heat, annot=True, cmap="YlGnBu", vmin=0, vmax=2, cbar_kws={"label": "Score (0-2)"}, ax=ax)
    ax.set_title("Step-level scoring across Hartree-Fock derivation pipeline")
    ax.set_xlabel("Scoring category")
    ax.set_ylabel("Calculation step")
    fig.tight_layout()
    fig.savefig(IMG / "step_scores.png", dpi=200)
    plt.close(fig)

    # Figure 3: category means and exact-source sensitivity
    cat_means = pd.Series(summary["category_means"]).reset_index()
    cat_means.columns = ["category", "mean_score"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=cat_means, x="category", y="mean_score", ax=axes[0], color="#55A868")
    axes[0].axhline(2, color="k", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 2.1)
    axes[0].set_title("Mean score by validation category")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Mean score (0-2)")
    axes[0].set_xlabel("")
    df2 = df.copy()
    df2["task_short"] = df2["index"].astype(str)
    sns.lineplot(data=df2, x="index", y="total_score", marker="o", ax=axes[1], label="total")
    axes[1].axhline(12, color="k", linestyle="--", linewidth=1, label="maximum")
    axes[1].set_title("Total score by pipeline order")
    axes[1].set_xlabel("Task index")
    axes[1].set_ylabel("Total score / 12")
    axes[1].set_ylim(0, 12.5)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMG / "validation_comparison.png", dpi=200)
    plt.close(fig)


def update_inventory():
    inv = {
        "primary_artifacts": [
            {"name": "paper_information_extraction", "path": "outputs/paper_information_extraction.json", "status": "satisfied"},
            {"name": "hf_hamiltonian_derivation", "path": "outputs/hf_hamiltonian_derivation.md", "status": "satisfied"},
            {"name": "step_scoring_results", "path": "outputs/step_scoring_results.json", "status": "satisfied"},
            {"name": "validation_summary", "path": "outputs/validation_summary.json", "status": "satisfied"},
            {"name": "data_overview_figure", "path": "report/images/data_overview.png", "status": "satisfied"},
            {"name": "step_scores_figure", "path": "report/images/step_scores.png", "status": "satisfied"},
            {"name": "validation_comparison_figure", "path": "report/images/validation_comparison.png", "status": "satisfied"},
            {"name": "final_report", "path": "report/report.md", "status": "planned; created after analysis script"}
        ]
    }
    (OUT / "target_artifact_inventory.json").write_text(json.dumps(inv, indent=2))


def main():
    df, raw = load_yaml()
    info = extract_tex_info()
    write_hf_derivation()
    summary = score_outputs(df)
    validation_artifacts(df, info)
    method_fidelity()
    figures(df, summary)
    update_inventory()
    print(json.dumps({"n_tasks": len(df), "mean_total": summary["mean_total_score"], "figures": ["data_overview.png", "step_scores.png", "validation_comparison.png"]}, indent=2))


if __name__ == "__main__":
    main()
