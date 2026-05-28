"""Import external runs as Home leaderboard-only records.

This keeps full run artifacts out of GitHub Pages while preserving table scores.
"""

import argparse
import json
from pathlib import Path

from export_static import (
    DATA_DIR,
    LEADERBOARD_ONLY_RUNS_PATH,
    RCB_SOURCE,
    _estimate_run_cost_usd,
    _format_model_display,
    _normalize_model_name,
)


TASKS_DIR = RCB_SOURCE / "tasks"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _valid_tasks():
    return {
        path.name
        for path in TASKS_DIR.iterdir()
        if path.is_dir() and (path / "task_info.json").exists()
    }


def _load_existing():
    if not LEADERBOARD_ONLY_RUNS_PATH.exists():
        return []
    with open(LEADERBOARD_ONLY_RUNS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"{LEADERBOARD_ONLY_RUNS_PATH} must contain a JSON list")
    return data


def _collect_runs(source_dir, agent_name, model):
    valid_tasks = _valid_tasks()
    candidates = []
    errors = []
    for run_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        meta_path = run_dir / "_meta.json"
        score_path = run_dir / "_score.json"
        if not meta_path.exists() or not score_path.exists():
            errors.append(f"{run_dir.name}: missing _meta.json or _score.json")
            continue

        meta = _load_json(meta_path)
        score = _load_json(score_path)
        task_id = meta.get("task_id") or score.get("task_id")
        total_score = score.get("total_score")
        duration = meta.get("duration_seconds")
        if task_id not in valid_tasks:
            errors.append(f"{run_dir.name}: invalid task_id {task_id!r}")
            continue
        if meta.get("status") != "completed":
            errors.append(f"{run_dir.name}: status is {meta.get('status')!r}")
            continue
        if total_score is None:
            errors.append(f"{run_dir.name}: missing total_score")
            continue
        if duration is None:
            errors.append(f"{run_dir.name}: missing duration_seconds")
            continue

        normalized_model = _normalize_model_name(model)
        candidates.append({
            "run_id": meta.get("run_id") or run_dir.name,
            "task_id": task_id,
            "timestamp": meta.get("timestamp") or run_dir.name.rsplit("_", 2)[-2] + "_" + run_dir.name.rsplit("_", 2)[-1],
            "status": "completed",
            "agent_name": agent_name,
            "model": normalized_model,
            "model_display": _format_model_display(normalized_model),
            "duration_seconds": duration,
            "cost_usd": _estimate_run_cost_usd(normalized_model, duration),
            "total_score": total_score,
            "details_exported": False,
        })
    if errors:
        raise RuntimeError("Invalid leaderboard-only source runs:\n" + "\n".join(errors))
    return candidates


def _pick_best_per_task(candidates):
    best = {}
    skipped = []
    for item in candidates:
        current = best.get(item["task_id"])
        if current is None or item["total_score"] > current["total_score"]:
            if current is not None:
                skipped.append(current)
            best[item["task_id"]] = item
        else:
            skipped.append(item)
    return list(best.values()), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="External run directory containing run subdirectories")
    parser.add_argument("--agent-name", required=True, help="Canonical Home agent column name")
    parser.add_argument("--model", required=True, help="Canonical model key/name used for display and cost")
    parser.add_argument("--replace-agent", action="store_true", help="Replace existing leaderboard-only records for this agent")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory not found: {source_dir}")

    candidates = _collect_runs(source_dir, args.agent_name, args.model)
    selected, skipped = _pick_best_per_task(candidates)
    existing = _load_existing()
    if args.replace_agent:
        existing = [item for item in existing if item.get("agent_name") != args.agent_name]

    existing_ids = {item.get("run_id") for item in existing}
    duplicate_ids = sorted(item["run_id"] for item in selected if item["run_id"] in existing_ids)
    if duplicate_ids:
        raise RuntimeError(f"Run IDs already exist in {LEADERBOARD_ONLY_RUNS_PATH}: {duplicate_ids}")

    output = existing + selected
    output.sort(key=lambda item: (item.get("agent_name", ""), item.get("task_id", ""), item.get("run_id", "")))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_ONLY_RUNS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    missing_tasks = sorted(_valid_tasks() - {item["task_id"] for item in selected})
    print(f"Imported {len(selected)} leaderboard-only runs from {len(candidates)} candidates")
    if skipped:
        print("Skipped lower-scoring duplicates:")
        for item in skipped:
            print(f"  {item['task_id']}: {item['run_id']} score={item['total_score']}")
    if missing_tasks:
        print("Missing tasks:")
        for task_id in missing_tasks:
            print(f"  {task_id}")
    print(f"Wrote {LEADERBOARD_ONLY_RUNS_PATH}")


if __name__ == "__main__":
    main()
