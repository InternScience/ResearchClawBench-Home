#!/usr/bin/env python3
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
REPORT_IMG_DIR = os.path.join(ROOT, "report", "images")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_IMG_DIR, exist_ok=True)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_problem_entry(problem_id, body):
    if "?" not in body:
        raise ValueError(f"Malformed problem body: {problem_id}")
    prefix, goal = body.split("?", 1)
    constructions = [p.strip() for p in prefix.split(";") if p.strip()]
    goal = goal.strip()
    return problem_id, constructions, goal


def tokenize_args(text):
    return [t for t in text.strip().split() if t]


def parse_construction(stmt):
    left, right = [x.strip() for x in stmt.split("=", 1)]
    lhs_tokens = tokenize_args(left)
    rhs_tokens = tokenize_args(right)
    op = rhs_tokens[0]
    args = rhs_tokens[1:]
    return {
        "raw": stmt,
        "lhs": lhs_tokens,
        "op": op,
        "args": args,
        "arity": len(args),
        "new_points": len(lhs_tokens),
    }


def goal_predicate(goal):
    return tokenize_args(goal)[0]


def parse_rule_head(rule_line):
    if "=>" not in rule_line:
        return None
    _, rhs = rule_line.split("=>", 1)
    rhs = rhs.strip()
    return tokenize_args(rhs)[0]


def parse_defs_ops(defs_text):
    ops = []
    blocks = [b.strip() for b in defs_text.split("\n\n") if b.strip()]
    for block in blocks:
        first = block.splitlines()[0].strip()
        tokens = first.split()
        if tokens:
            ops.append(tokens[0])
    ordered = []
    seen = set()
    for op in ops:
        if op not in seen:
            ordered.append(op)
            seen.add(op)
    return ordered


def infer_families(constructions):
    families = Counter()
    for c in constructions:
        op = c["op"]
        if "circle" in op:
            families["circle"] += 1
        if "line" in op:
            families["line"] += 1
        if "mid" in op:
            families["midpoint"] += 1
        if "center" in op:
            families["center"] += 1
        if "angle" in op or "bisector" in op or "mirror" in op or "reflect" in op:
            families["angle_transform"] += 1
        if op in {"on_tline", "foot"} or "ortho" in op:
            families["perpendicular"] += 1
        if op in {"on_pline", "parallelogram"}:
            families["parallel"] += 1
    return families


def problem_metrics(problem_id, constructions, goal, supported_goals):
    ops = [c["op"] for c in constructions]
    op_counts = Counter(ops)
    point_refs = Counter()
    for c in constructions:
        for token in c["lhs"] + c["args"]:
            if re.match(r"^[A-Za-z][A-Za-z0-9@._-]*$", token):
                point_refs[token] += 1

    families = infer_families(constructions)
    goal_type = goal_predicate(goal)
    support = int(goal_type in supported_goals)

    complexity = (
        len(constructions)
        + 1.4 * len(op_counts)
        + 0.8 * sum(1 for op in ops if op.startswith("on_"))
        + 1.2 * sum(c["new_points"] > 1 for c in constructions)
        + 0.7 * families["circle"]
        + 0.9 * families["angle_transform"]
    )
    proof_pressure = complexity * (1.0 + 0.15 * max(0, len(goal.split()) - 5))
    branching_factor = sum(max(0, c["new_points"] - 1) for c in constructions)
    reused_symbols = sum(1 for _, v in point_refs.items() if v >= 3)

    return {
        "problem_id": problem_id,
        "num_constructions": len(constructions),
        "num_unique_ops": len(op_counts),
        "goal_type": goal_type,
        "goal_supported_by_rules": support,
        "num_on_ops": sum(1 for op in ops if op.startswith("on_")),
        "num_circle_ops": families["circle"],
        "num_line_ops": families["line"],
        "num_midpoint_ops": families["midpoint"],
        "num_center_ops": families["center"],
        "num_angle_transform_ops": families["angle_transform"],
        "num_perpendicular_ops": families["perpendicular"],
        "num_parallel_ops": families["parallel"],
        "max_op_frequency": max(op_counts.values()) if op_counts else 0,
        "branching_factor_proxy": branching_factor,
        "reused_symbol_count": reused_symbols,
        "complexity_score": round(complexity, 3),
        "proof_pressure_score": round(proof_pressure, 3),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def plot_with_matplotlib(metrics):
    import matplotlib.pyplot as plt

    ids = [m["problem_id"].replace("translated_", "") for m in metrics]
    pressure = [m["proof_pressure_score"] for m in metrics]
    complexity = [m["complexity_score"] for m in metrics]
    support = [m["goal_supported_by_rules"] for m in metrics]
    goal_types = [m["goal_type"] for m in metrics]

    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(ids)), pressure, color="#355C7D")
    ax.set_title("Problem-wise symbolic proof pressure")
    ax.set_ylabel("Proof pressure score")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=75, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_IMG_DIR, "proof_pressure.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#C06C84" if s == 0 else "#6C5B7B" for s in support]
    ax.scatter(complexity, pressure, c=colors, s=55, edgecolor="black", linewidth=0.4)
    ax.set_title("Complexity vs proof pressure")
    ax.set_xlabel("Complexity score")
    ax.set_ylabel("Proof pressure score")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_IMG_DIR, "complexity_vs_pressure.png"), dpi=200)
    plt.close(fig)

    goal_counter = Counter(goal_types)
    labels = list(goal_counter.keys())
    values = [goal_counter[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="#F67280")
    ax.set_title("Goal predicate distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_IMG_DIR, "goal_distribution.png"), dpi=200)
    plt.close(fig)


def main():
    ensure_dirs()

    raw_lines = [line.strip() for line in read_text(os.path.join(DATA_DIR, "imo_ag_30.txt")).splitlines() if line.strip()]
    if len(raw_lines) % 2 != 0:
        raise ValueError("Expected alternating problem-id/body lines in imo_ag_30.txt")
    problem_entries = [(raw_lines[i], raw_lines[i + 1]) for i in range(0, len(raw_lines), 2)]
    defs_text = read_text(os.path.join(DATA_DIR, "defs.txt"))
    rules_text = read_text(os.path.join(DATA_DIR, "rules.txt"))

    def_ops = parse_defs_ops(defs_text)
    rule_lines = [line.strip() for line in rules_text.splitlines() if line.strip()]
    supported_goals = {head for head in (parse_rule_head(line) for line in rule_lines) if head}

    metrics = []
    op_usage = Counter()
    family_usage = Counter()

    for problem_id, body in problem_entries:
        problem_id, construction_stmts, goal = parse_problem_entry(problem_id, body)
        constructions = [parse_construction(stmt) for stmt in construction_stmts]
        for c in constructions:
            op_usage[c["op"]] += 1
        family_usage.update(infer_families(constructions))
        metrics.append(problem_metrics(problem_id, constructions, goal, supported_goals))

    metrics.sort(key=lambda x: x["proof_pressure_score"], reverse=True)

    summary = {
        "num_problems": len(metrics),
        "supported_goal_predicates": sorted(supported_goals),
        "definition_operations": def_ops,
        "goal_type_counts": dict(Counter(m["goal_type"] for m in metrics)),
        "goal_support_rate": round(sum(m["goal_supported_by_rules"] for m in metrics) / len(metrics), 4),
        "avg_num_constructions": round(sum(m["num_constructions"] for m in metrics) / len(metrics), 3),
        "avg_complexity_score": round(sum(m["complexity_score"] for m in metrics) / len(metrics), 3),
        "avg_proof_pressure_score": round(sum(m["proof_pressure_score"] for m in metrics) / len(metrics), 3),
        "top_operations": op_usage.most_common(12),
        "family_usage": dict(family_usage),
        "hardest_by_pressure": metrics[:5],
        "easiest_by_pressure": sorted(metrics, key=lambda x: x["proof_pressure_score"])[:5],
    }

    write_csv(os.path.join(OUTPUT_DIR, "problem_metrics.csv"), metrics)
    save_json(os.path.join(OUTPUT_DIR, "benchmark_summary.json"), summary)

    ranked_for_goal = sorted(metrics, key=lambda x: (x["goal_supported_by_rules"], x["proof_pressure_score"]))
    write_csv(os.path.join(OUTPUT_DIR, "goal_support_ranked.csv"), ranked_for_goal)

    plot_with_matplotlib(metrics)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
