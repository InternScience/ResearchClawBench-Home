#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "2111.01152"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_equation_block(text: str, label: str) -> str:
    pattern = re.compile(r"\\begin\{equation\}(?:\\label\{[^}]+\})?(.*?)\\end\{equation\}", re.S)
    for match in pattern.finditer(text):
        block = match.group(0)
        if label in block or match.group(1).find(label) >= 0:
            return re.sub(r"\s+", " ", block).strip()
    return ""


def extract_section(text: str, section_name: str) -> str:
    pattern = re.compile(
        rf"\\section\{{{re.escape(section_name)}\}}(.*?)(?:\\section\{{|\\end\{{document\}})",
        re.S,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def parse_score_value(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if v in {"(?)", "?", ""}:
            return None
        try:
            return float(v)
        except ValueError:
            return None
    return None


def summarize_yaml(yaml_data: list[dict]) -> dict:
    task_rows = []
    placeholder_rows = []
    category_scores = defaultdict(list)
    reviewer_scores = defaultdict(list)

    for item in yaml_data:
        if "task" not in item:
            continue
        task_name = item["task"]
        score_map = item.get("score", {})
        numeric_scores = {k: float(v) for k, v in score_map.items() if isinstance(v, (int, float))}
        avg_score = mean(numeric_scores.values()) if numeric_scores else math.nan
        task_rows.append(
            {
                "task": task_name,
                "average_score": avg_score,
                "scores": numeric_scores,
                "answer": item.get("answer", ""),
            }
        )
        for k, v in numeric_scores.items():
            category_scores[k].append(v)

        for placeholder_name, placeholder_info in item.get("placeholder", {}).items():
            review = placeholder_info.get("score", {})
            row = {"task": task_name, "placeholder": placeholder_name}
            for reviewer, score in review.items():
                parsed = parse_score_value(score)
                if parsed is not None:
                    row[reviewer] = parsed
                    reviewer_scores[reviewer].append(parsed)
            placeholder_rows.append(row)

    category_summary = {k: mean(v) for k, v in category_scores.items() if v}
    reviewer_summary = {k: mean(v) for k, v in reviewer_scores.items() if v}
    return {
        "tasks": task_rows,
        "placeholders": placeholder_rows,
        "category_summary": category_summary,
        "reviewer_summary": reviewer_summary,
    }


def classify_placeholder_mismatches(yaml_data: list[dict]) -> list[dict]:
    rows = []
    for item in yaml_data:
        if "task" not in item:
            continue
        task_name = item["task"]
        for placeholder_name, placeholder_info in item.get("placeholder", {}).items():
            llm = placeholder_info.get("LLM")
            human = placeholder_info.get("human")
            if llm is None and human is None:
                continue
            llm_s = "" if llm is None else str(llm).strip()
            human_s = "" if human is None else str(human).strip()
            mismatch = llm_s != human_s and human_s != ""
            if mismatch:
                rows.append(
                    {
                        "task": task_name,
                        "placeholder": placeholder_name,
                        "llm": llm_s,
                        "human": human_s,
                    }
                )
    return rows


def derive_claim_discipline(task_rows: list[dict]) -> list[dict]:
    claims = []
    for row in task_rows:
        avg_score = row["average_score"]
        if avg_score >= 1.6:
            support = "supported"
        elif avg_score >= 1.1:
            support = "partially supported"
        else:
            support = "not supported"
        claims.append(
            {
                "task": row["task"],
                "average_score": round(avg_score, 3),
                "claim_support": support,
                "evidence": row["scores"],
            }
        )
    return claims


def extract_hf_summary(sm_text: str) -> dict:
    section = extract_section(sm_text, "Hartree-Fock calculation")
    full_eq_match = re.search(
        r"\\begin\{equation\}\\label\{eq:full\}(.*?)\\end\{equation\}",
        sm_text,
        re.S,
    )
    full_eq = re.sub(r"\s+", " ", full_eq_match.group(0)).strip() if full_eq_match else ""

    mf_matches = re.findall(r"\\hat\{\\mathcal\{H\}\}_\{HF\}|\\hat\{\\mathcal\{H\}\}_\{\\mathrm\{MF\}\}", sm_text)
    density_matches = re.findall(r"P_[A-Za-z]+|\\langle .*? \\rangle", section, re.S)

    keywords = {
        "hartree_terms": len(re.findall(r"Hartree", section)),
        "fock_terms": len(re.findall(r"Fock", section)),
        "density_matrix_mentions": len(re.findall(r"density matrix|order parameter|expectation value", section, re.I)),
        "plane_wave_mentions": len(re.findall(r"plane-wave basis|plane-wave", section, re.I)),
    }
    return {
        "section_excerpt": re.sub(r"\s+", " ", section[:5000]).strip(),
        "full_hamiltonian_equation": full_eq,
        "mf_symbol_mentions": len(mf_matches),
        "density_like_mentions": len(density_matches),
        "keywords": keywords,
    }


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_markdown_summary(path: Path, summary: dict, mismatches: list[dict], equations: dict, related_titles: list[str]) -> None:
    lines = [
        "# Analysis Summary",
        "",
        "## Task-Level Scores",
        "",
    ]
    for row in summary["tasks"]:
        lines.append(f"- {row['task']}: average rubric score {row['average_score']:.2f}")
    lines.extend(
        [
            "",
            "## Category Means",
            "",
        ]
    )
    for key, val in sorted(summary["category_summary"].items()):
        lines.append(f"- {key}: {val:.2f}")
    lines.extend(
        [
            "",
            "## High-Signal Placeholder Mismatches",
            "",
        ]
    )
    for row in mismatches[:20]:
        lines.append(
            f"- {row['task']} | {row['placeholder']} | LLM=`{row['llm']}` | reference=`{row['human']}`"
        )
    lines.extend(
        [
            "",
            "## Continuum Hamiltonian",
            "",
            f"`{equations['continuum']}`",
            "",
            "## Full Interacting Hamiltonian",
            "",
            f"`{equations['full']}`",
            "",
            "## Related Work Titles",
            "",
        ]
    )
    for title in related_titles:
        lines.append(f"- {title}")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(summary: dict, mismatches: list[dict], claims: list[dict]) -> None:
    task_names = [row["task"] for row in summary["tasks"]]
    task_scores = [row["average_score"] for row in summary["tasks"]]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(task_names)), task_scores, color="#1f77b4")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.set_ylabel("Average rubric score")
    ax.set_title("Task-level Hartree-Fock prompt performance")
    ax.set_ylim(0, 2.1)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "task_scores.png", dpi=200)
    plt.close(fig)

    cat_names = list(sorted(summary["category_summary"].keys()))
    cat_vals = [summary["category_summary"][k] for k in cat_names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(cat_names, cat_vals, color="#d62728")
    ax.set_xlabel("Mean score")
    ax.set_xlim(0, 2.1)
    ax.set_title("Rubric category means")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "category_scores.png", dpi=200)
    plt.close(fig)

    counter = Counter(row["placeholder"] for row in mismatches)
    top_items = counter.most_common(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([k for k, _ in top_items][::-1], [v for _, v in top_items][::-1], color="#2ca02c")
    ax.set_xlabel("Mismatch count")
    ax.set_title("Most frequent prompt-field mismatches")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "mismatch_counts.png", dpi=200)
    plt.close(fig)

    support_order = ["supported", "partially supported", "not supported"]
    support_counter = Counter(row["claim_support"] for row in claims)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(support_order, [support_counter.get(k, 0) for k in support_order], color=["#4c78a8", "#f58518", "#e45756"])
    ax.set_ylabel("Number of tasks")
    ax.set_title("Claim-support discipline over tasks")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "claim_support.png", dpi=200)
    plt.close(fig)


def extract_related_titles() -> list[str]:
    titles = []
    try:
        import subprocess

        for pdf in sorted((ROOT / "related_work").glob("*.pdf")):
            try:
                txt = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True, stderr=subprocess.DEVNULL)
            except Exception:
                txt = ""
            lines = [line.strip() for line in txt.splitlines() if line.strip()]
            title = ""
            for line in lines[:20]:
                if len(line) > 20 and not line.lower().startswith(("citation:", "publisher:", "version:", "terms of use:")):
                    title = line
                    break
            titles.append(f"{pdf.name}: {title or 'title not recovered'}")
    except Exception:
        pass
    return titles


def main() -> None:
    ensure_dirs()

    yaml_data = yaml.safe_load(read_text(DATA_DIR / "2111.01152.yaml"))
    tex_text = read_text(DATA_DIR / "2111.01152.tex")
    sm_text = read_text(DATA_DIR / "2111.01152_SM.tex")

    summary = summarize_yaml(yaml_data)
    mismatches = classify_placeholder_mismatches(yaml_data)
    claims = derive_claim_discipline(summary["tasks"])
    hf_summary = extract_hf_summary(sm_text)

    continuum_eq = extract_equation_block(tex_text, "eq:Ham")
    full_eq = hf_summary["full_hamiltonian_equation"]
    related_titles = extract_related_titles()

    results = {
        "paper_id": "2111.01152",
        "task_count": len(summary["tasks"]),
        "placeholder_count": len(summary["placeholders"]),
        "task_summary": summary["tasks"],
        "category_summary": summary["category_summary"],
        "reviewer_summary": summary["reviewer_summary"],
        "mismatches": mismatches,
        "claim_discipline": claims,
        "equations": {
            "continuum_hamiltonian": continuum_eq,
            "full_interacting_hamiltonian": full_eq,
        },
        "hf_summary": hf_summary,
        "related_work_titles": related_titles,
    }

    save_json(OUTPUT_DIR / "analysis_results.json", results)
    save_json(OUTPUT_DIR / "claim_discipline.json", claims)
    save_markdown_summary(
        OUTPUT_DIR / "analysis_summary.md",
        summary,
        mismatches,
        {"continuum": continuum_eq, "full": full_eq},
        related_titles,
    )
    make_figures(summary, mismatches, claims)


if __name__ == "__main__":
    main()
