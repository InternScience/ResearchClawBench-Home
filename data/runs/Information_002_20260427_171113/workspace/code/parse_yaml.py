"""Parse 2111.01152.yaml and produce structured per-task / per-placeholder
score tables for the LLM-driven Hartree-Fock derivation benchmark.

Outputs are written under ../outputs:
    - parsed_tasks.json           : full structured list of tasks
    - placeholder_scores.csv      : one row per (task, placeholder, grader)
    - answer_scores.csv           : one row per task with the 6 final-answer axes
    - per_task_summary.csv        : mean placeholder + final-answer scores per task
    - summary.json                : top-level summary numbers used by the report
"""
import json
import os
import re
from collections import defaultdict

import yaml
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "data", "2111.01152", "2111.01152.yaml")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTDIR, exist_ok=True)


def to_score(x):
    """Convert a raw grader entry to a 0/1/2 numeric score, NaN otherwise.
    Accepts ints, '0'/'1'/'2', '(?)' -> NaN, None -> NaN."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            v = int(x)
        except Exception:
            return None
        if v in (0, 1, 2):
            return float(v)
        return None
    s = str(x).strip()
    if s in ("(?)", "?", "", "check", "n/a", "N/A", "None"):
        return None
    m = re.match(r"^([012])$", s)
    if m:
        return float(m.group(1))
    return None


def main():
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)

    tasks = []
    placeholder_rows = []
    answer_rows = []
    GRADERS = ["Haining", "Will", "Yasaman"]
    AXES = [
        "in_paper",
        "prompt_quality",
        "follow_instructions",
        "physics_logic",
        "math_derivation",
        "final_answer_accuracy",
    ]

    task_index = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if "task" not in entry:
            # branch / metadata header
            continue
        task_index += 1
        name = entry["task"]
        placeholders = entry.get("placeholder", {}) or {}
        ans_score = entry.get("score", {}) or {}

        # placeholder rows
        for ph_name, ph in placeholders.items():
            if not isinstance(ph, dict):
                continue
            ph_scores = ph.get("score", {}) or {}
            llm_val = ph.get("LLM")
            human_val = ph.get("human")
            for g in GRADERS:
                v = to_score(ph_scores.get(g))
                placeholder_rows.append({
                    "task_index": task_index,
                    "task": name,
                    "placeholder": ph_name,
                    "grader": g,
                    "score": v,
                    "llm_value": str(llm_val) if llm_val is not None else None,
                    "human_value": str(human_val) if human_val is not None else None,
                })

        # final-answer / completion axis row
        ans_row = {"task_index": task_index, "task": name}
        for ax in AXES:
            ans_row[ax] = to_score(ans_score.get(ax))
        answer_rows.append(ans_row)

        tasks.append({
            "task_index": task_index,
            "task": name,
            "n_placeholders": sum(1 for ph in placeholders.values()
                                  if isinstance(ph, dict)),
            "answer_axes": {ax: ans_row[ax] for ax in AXES},
        })

    # DataFrames
    df_ph = pd.DataFrame(placeholder_rows)
    df_ans = pd.DataFrame(answer_rows)

    # per-task summaries
    per_task = (df_ph.groupby(["task_index", "task"])["score"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "placeholder_mean_score",
                                 "count": "n_grader_judgements"})
                .reset_index())
    per_task = per_task.merge(df_ans, on=["task_index", "task"], how="left")
    per_task["answer_mean_score"] = per_task[AXES].mean(axis=1)

    # save
    df_ph.to_csv(os.path.join(OUTDIR, "placeholder_scores.csv"), index=False)
    df_ans.to_csv(os.path.join(OUTDIR, "answer_scores.csv"), index=False)
    per_task.to_csv(os.path.join(OUTDIR, "per_task_summary.csv"), index=False)
    with open(os.path.join(OUTDIR, "parsed_tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2)

    # summary numbers
    summary = {
        "n_tasks": int(task_index),
        "n_placeholder_rows": int(len(df_ph)),
        "n_placeholder_judged": int(df_ph["score"].notna().sum()),
        "global_placeholder_mean": float(df_ph["score"].mean()),
        "global_placeholder_full_credit_rate": float(
            (df_ph["score"] == 2).mean(skipna=True)
        ),
        "global_placeholder_zero_rate": float(
            (df_ph["score"] == 0).mean(skipna=True)
        ),
        "per_grader_mean": {
            g: float(df_ph.loc[df_ph["grader"] == g, "score"].mean())
            for g in GRADERS
        },
        "per_axis_mean": {
            ax: float(df_ans[ax].mean()) for ax in AXES
        },
        "per_task_answer_mean": {
            row["task"]: float(row["answer_mean_score"])
            for _, row in per_task.iterrows()
        },
        "per_task_placeholder_mean": {
            row["task"]: float(row["placeholder_mean_score"])
            for _, row in per_task.iterrows()
        },
    }
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # short text
    print(f"Tasks: {summary['n_tasks']}")
    print(f"Placeholder rows: {summary['n_placeholder_rows']} "
          f"(judged: {summary['n_placeholder_judged']})")
    print(f"Global placeholder mean score: "
          f"{summary['global_placeholder_mean']:.3f}")
    print(f"Full-credit rate: "
          f"{summary['global_placeholder_full_credit_rate']:.3f}")
    print("Per grader mean:")
    for g, v in summary["per_grader_mean"].items():
        print(f"  {g}: {v:.3f}")
    print("Per axis mean (final-answer):")
    for ax, v in summary["per_axis_mean"].items():
        print(f"  {ax}: {v:.3f}")


if __name__ == "__main__":
    main()
