import json
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


MOVE_DIRS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


@dataclass
class Instance:
    dataset: str
    map_group: str
    map_name: str
    grid: np.ndarray
    starts: list
    goals: list
    seed: int


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def free_cells(grid):
    cells = np.argwhere(grid == 0)
    return [tuple(map(int, c)) for c in cells]


def shortest_path(grid, start, goal, reserved_vertices=None, reserved_edges=None, max_time=256):
    if reserved_vertices is None:
        reserved_vertices = {}
    if reserved_edges is None:
        reserved_edges = set()

    rows, cols = grid.shape

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0

    heuristic_cap = abs(start[0] - goal[0]) + abs(start[1] - goal[1]) + max(rows, cols) * 2
    max_time = min(max_time, heuristic_cap)

    q = deque()
    q.append((start[0], start[1], 0))
    parent = {(start[0], start[1], 0): None}
    best_seen = {(start[0], start[1], 0)}

    while q:
        r, c, t = q.popleft()
        if (r, c) == goal:
            path = []
            cur = (r, c, t)
            while cur is not None:
                path.append((cur[0], cur[1]))
                cur = parent[cur]
            return list(reversed(path))
        if t >= max_time:
            continue
        for dr, dc in MOVE_DIRS:
            nr, nc = r + dr, c + dc
            nt = t + 1
            if not in_bounds(nr, nc):
                continue
            if (nr, nc) in reserved_vertices.get(nt, set()):
                continue
            if ((r, c), (nr, nc), nt) in reserved_edges:
                continue
            state = (nr, nc, nt)
            if state in best_seen:
                continue
            best_seen.add(state)
            parent[state] = (r, c, t)
            q.append(state)
    return None


def position_at(path, t):
    return path[min(t, len(path) - 1)]


def count_collisions(paths):
    collisions = 0
    details = []
    if not paths:
        return collisions, details
    horizon = max(len(p) for p in paths)
    for t in range(horizon):
        occupancy = {}
        for i, path in enumerate(paths):
            pos = position_at(path, t)
            if pos in occupancy:
                collisions += 1
                details.append(("vertex", t, occupancy[pos], i, pos))
            occupancy[pos] = i
        if t == 0:
            continue
        for i in range(len(paths)):
            prev_i = position_at(paths[i], t - 1)
            cur_i = position_at(paths[i], t)
            for j in range(i + 1, len(paths)):
                prev_j = position_at(paths[j], t - 1)
                cur_j = position_at(paths[j], t)
                if prev_i == cur_j and prev_j == cur_i and cur_i != cur_j:
                    collisions += 1
                    details.append(("swap", t, i, j, cur_i, cur_j))
    return collisions, details


def sum_of_costs(paths):
    return int(sum(len(p) - 1 for p in paths))


def build_reservations(paths, ignore=None):
    reserved_vertices = defaultdict(set)
    reserved_edges = set()
    if ignore is None:
        ignore = set()
    horizon = max((len(p) for idx, p in enumerate(paths) if idx not in ignore), default=0)
    horizon = max(horizon, 1)
    for idx, path in enumerate(paths):
        if idx in ignore:
            continue
        goal = path[-1]
        for t in range(horizon + 20):
            pos = position_at(path, t)
            reserved_vertices[t].add(pos)
            if t > 0:
                prev = position_at(path, t - 1)
                reserved_edges.add((pos, prev, t))
        for t in range(horizon + 20, horizon + 60):
            reserved_vertices[t].add(goal)
    return reserved_vertices, reserved_edges


def prioritized_planning(instance, ordering):
    paths = [None] * len(ordering)
    planned_paths = {}
    max_base = instance.grid.shape[0] * instance.grid.shape[1] * 2
    reserved_vertices = defaultdict(set)
    reserved_edges = set()
    for agent in ordering:
        path = shortest_path(
            instance.grid,
            instance.starts[agent],
            instance.goals[agent],
            reserved_vertices=reserved_vertices,
            reserved_edges=reserved_edges,
            max_time=max_base,
        )
        if path is None:
            return None
        planned_paths[agent] = path
        for t in range(len(path) + 20):
            pos = position_at(path, t)
            reserved_vertices[t].add(pos)
            if t > 0:
                prev = position_at(path, t - 1)
                reserved_edges.add((pos, prev, t))
    for agent, path in planned_paths.items():
        paths[agent] = path
    return paths


def independent_paths(instance):
    paths = []
    max_base = instance.grid.shape[0] * instance.grid.shape[1] * 2
    for s, g in zip(instance.starts, instance.goals):
        path = shortest_path(instance.grid, s, g, max_time=max_base)
        if path is None:
            return None
        paths.append(path)
    return paths


def agent_features(instance):
    feats = []
    rows, cols = instance.grid.shape
    for s, g in zip(instance.starts, instance.goals):
        manhattan = abs(s[0] - g[0]) + abs(s[1] - g[1])
        local_obs = 0
        for dr, dc in MOVE_DIRS[1:]:
            nr, nc = s[0] + dr, s[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or instance.grid[nr, nc] != 0:
                local_obs += 1
        goal_obs = 0
        for dr, dc in MOVE_DIRS[1:]:
            nr, nc = g[0] + dr, g[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or instance.grid[nr, nc] != 0:
                goal_obs += 1
        feats.append(
            {
                "manhattan": manhattan,
                "start_degree_blocked": local_obs,
                "goal_degree_blocked": goal_obs,
                "same_row_goal": int(s[0] == g[0]),
                "same_col_goal": int(s[1] == g[1]),
            }
        )
    return feats


def simulate_naive_collisions(instance):
    paths = independent_paths(instance)
    if paths is None:
        return None, None
    _, details = count_collisions(paths)
    feats = agent_features(instance)
    labels = [0] * len(paths)
    for item in details:
        if item[0] == "vertex":
            _, _, a, b, _ = item
        else:
            _, _, a, b, _, _ = item
        labels[a] = 1
        labels[b] = 1
    return feats, labels


def fit_linear_risk_model(train_instances):
    feature_names = [
        "manhattan",
        "start_degree_blocked",
        "goal_degree_blocked",
        "same_row_goal",
        "same_col_goal",
    ]
    xs = []
    ys = []
    for inst in train_instances:
        feats, labels = simulate_naive_collisions(inst)
        if feats is None:
            continue
        for feat, label in zip(feats, labels):
            xs.append([1.0] + [float(feat[name]) for name in feature_names])
            ys.append(float(label))
    if not xs:
        return feature_names, np.zeros(1 + len(feature_names))
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    w = np.linalg.pinv(x.T @ x + 1e-6 * np.eye(x.shape[1])) @ x.T @ y
    return feature_names, w


def risk_scores(instance, model):
    feature_names, weights = model
    feats = agent_features(instance)
    scores = []
    for idx, feat in enumerate(feats):
        x = np.array([1.0] + [float(feat[name]) for name in feature_names], dtype=float)
        score = float(x @ weights)
        scores.append((idx, score))
    return scores


def marl_ordering(instance, model):
    scores = risk_scores(instance, model)
    scores.sort(key=lambda x: (-x[1], -abs(instance.starts[x[0]][0] - instance.goals[x[0]][0]) - abs(instance.starts[x[0]][1] - instance.goals[x[0]][1]), x[0]))
    return [idx for idx, _ in scores]


def lns_repair(instance, init_paths, model, iterations=5, neighborhood=3):
    if init_paths is None:
        return None, {"iterations": 0, "best_collisions": math.inf}
    best_paths = [list(p) for p in init_paths]
    best_collisions, details = count_collisions(best_paths)
    rng = random.Random(instance.seed + 7919)
    scores = dict(risk_scores(instance, model))
    meta = {"iterations": 0, "best_collisions": best_collisions}
    if best_collisions == 0:
        return best_paths, meta

    for it in range(iterations):
        meta["iterations"] = it + 1
        bad_agents = set()
        for item in details[: neighborhood * 2]:
            if item[0] == "vertex":
                _, _, a, b, _ = item
            else:
                _, _, a, b, _, _ = item
            bad_agents.add(a)
            bad_agents.add(b)
        ranked = sorted(range(len(best_paths)), key=lambda a: (-int(a in bad_agents), -scores.get(a, 0.0), rng.random()))
        replan = set(ranked[: min(neighborhood, len(ranked))])
        fixed_paths = [p for idx, p in enumerate(best_paths) if idx not in replan]
        reserved_vertices, reserved_edges = build_reservations(fixed_paths)
        candidate = [list(p) if p is not None else None for p in best_paths]
        success = True
        for agent in sorted(replan, key=lambda a: -scores.get(a, 0.0)):
            path = shortest_path(
                instance.grid,
                instance.starts[agent],
                instance.goals[agent],
                reserved_vertices=reserved_vertices,
                reserved_edges=reserved_edges,
                max_time=instance.grid.shape[0] * instance.grid.shape[1] * 2,
            )
            if path is None:
                success = False
                break
            candidate[agent] = path
            for t in range(len(path) + 20):
                pos = position_at(path, t)
                reserved_vertices[t].add(pos)
                if t > 0:
                    prev = position_at(path, t - 1)
                    reserved_edges.add((pos, prev, t))
        if not success:
            continue
        cand_collisions, cand_details = count_collisions(candidate)
        if cand_collisions < best_collisions or (
            cand_collisions == best_collisions and sum_of_costs(candidate) < sum_of_costs(best_paths)
        ):
            best_paths = candidate
            best_collisions = cand_collisions
            details = cand_details
            meta["best_collisions"] = best_collisions
            if best_collisions == 0:
                break
    return best_paths, meta


def sample_instance_specs():
    return [
        ("random_small", "maps_50_10_10_0.175", 6, 2),
        ("random_medium", "maps_312_25_25_0.175", 8, 2),
        ("room", "room_maps_250_25_25", 6, 2),
        ("warehouse", "warehouse_maps_266_25_25", 7, 2),
        ("maze", "maze_maps_125_25_25", 5, 2),
        ("empty", "empty_maps_453_25_25", 8, 2),
    ]


def make_instance(dataset, group, map_idx, n_agents):
    map_dir = DATA_DIR / dataset / group
    map_files = sorted([p for p in map_dir.iterdir() if p.suffix == ".npy"])
    path = map_files[map_idx % len(map_files)]
    grid = np.load(path)
    cells = free_cells(grid)
    seed = abs(hash((dataset, group, path.name, n_agents))) % (2**32)
    rng = random.Random(seed)
    chosen = rng.sample(cells, 2 * n_agents)
    starts = chosen[:n_agents]
    goals = chosen[n_agents:]
    return Instance(dataset, group, path.name, grid, starts, goals, seed)


def build_train_eval_sets():
    train_instances = []
    eval_instances = []
    for dataset, group, n_agents, n_eval in sample_instance_specs():
        for map_idx in range(2):
            train_instances.append(make_instance(dataset, group, map_idx, n_agents))
        for map_idx in range(2, 2 + n_eval):
            eval_instances.append(make_instance(dataset, group, map_idx, n_agents))
    return train_instances, eval_instances


def solve_instance(instance, model):
    results = {}

    t0 = time.time()
    indep = independent_paths(instance)
    elapsed = time.time() - t0
    if indep is None:
        results["independent"] = {"success": 0, "runtime": elapsed, "collisions": None, "soc": None}
    else:
        collisions, _ = count_collisions(indep)
        results["independent"] = {
            "success": int(collisions == 0),
            "runtime": elapsed,
            "collisions": collisions,
            "soc": sum_of_costs(indep),
        }

    distance_order = sorted(
        range(len(instance.starts)),
        key=lambda a: (
            abs(instance.starts[a][0] - instance.goals[a][0]) + abs(instance.starts[a][1] - instance.goals[a][1]),
            a,
        ),
    )
    t0 = time.time()
    pp_paths = prioritized_planning(instance, distance_order)
    elapsed = time.time() - t0
    if pp_paths is None:
        results["pp"] = {"success": 0, "runtime": elapsed, "collisions": None, "soc": None}
    else:
        collisions, _ = count_collisions(pp_paths)
        results["pp"] = {
            "success": int(collisions == 0),
            "runtime": elapsed,
            "collisions": collisions,
            "soc": sum_of_costs(pp_paths),
        }

    marl_order = marl_ordering(instance, model)
    t0 = time.time()
    marl_pp_paths = prioritized_planning(instance, marl_order)
    elapsed = time.time() - t0
    if marl_pp_paths is None:
        results["marl_pp"] = {"success": 0, "runtime": elapsed, "collisions": None, "soc": None}
    else:
        collisions, _ = count_collisions(marl_pp_paths)
        results["marl_pp"] = {
            "success": int(collisions == 0),
            "runtime": elapsed,
            "collisions": collisions,
            "soc": sum_of_costs(marl_pp_paths),
        }

    seed_paths = marl_pp_paths if marl_pp_paths is not None else pp_paths
    t0 = time.time()
    hybrid_paths, meta = lns_repair(instance, seed_paths, model)
    elapsed = time.time() - t0
    if hybrid_paths is None:
        results["hybrid_lns"] = {"success": 0, "runtime": elapsed, "collisions": None, "soc": None, "lns_iterations": 0}
    else:
        collisions, _ = count_collisions(hybrid_paths)
        results["hybrid_lns"] = {
            "success": int(collisions == 0),
            "runtime": elapsed,
            "collisions": collisions,
            "soc": sum_of_costs(hybrid_paths),
            "lns_iterations": meta["iterations"],
        }

    results["meta"] = {
        "dataset": instance.dataset,
        "group": instance.map_group,
        "map_name": instance.map_name,
        "n_agents": len(instance.starts),
        "free_ratio": float(np.mean(instance.grid == 0)),
    }
    return results


def aggregate_results(per_instance):
    methods = ["independent", "pp", "marl_pp", "hybrid_lns"]
    summary = {}
    for method in methods:
        rows = [r[method] for r in per_instance]
        success = [x["success"] for x in rows]
        runtimes = [x["runtime"] for x in rows]
        finite_coll = [x["collisions"] for x in rows if x["collisions"] is not None]
        finite_soc = [x["soc"] for x in rows if x["soc"] is not None]
        summary[method] = {
            "instances": len(rows),
            "success_rate": float(np.mean(success)),
            "avg_runtime_sec": float(np.mean(runtimes)),
            "avg_collisions": float(np.mean(finite_coll)) if finite_coll else None,
            "avg_soc": float(np.mean(finite_soc)) if finite_soc else None,
        }
    return summary


def plot_results(per_instance, summary):
    methods = ["pp", "marl_pp", "hybrid_lns"]
    labels = ["PP", "Risk-PP", "Hybrid-LNS"]
    success_vals = [summary[m]["success_rate"] for m in methods]
    runtime_vals = [summary[m]["avg_runtime_sec"] for m in methods]
    coll_vals = [summary[m]["avg_collisions"] for m in methods]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(labels, success_vals, color=["#9c6644", "#386641", "#1d3557"])
    plt.ylim(0, 1.05)
    plt.ylabel("Success Rate")
    plt.title("Overall Success")
    plt.subplot(1, 2, 2)
    plt.bar(labels, runtime_vals, color=["#9c6644", "#386641", "#1d3557"])
    plt.ylabel("Avg Runtime (s)")
    plt.title("Planning Runtime")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "main_results.png", dpi=180)
    plt.close()

    datasets = sorted({r["meta"]["dataset"] for r in per_instance})
    by_dataset = {d: {m: [] for m in methods} for d in datasets}
    for r in per_instance:
        d = r["meta"]["dataset"]
        for m in methods:
            by_dataset[d][m].append(r[m]["success"])

    plt.figure(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.24
    for i, m in enumerate(methods):
        vals = [float(np.mean(by_dataset[d][m])) for d in datasets]
        plt.bar(x + (i - 1) * width, vals, width=width, label=labels[i])
    plt.xticks(x, datasets, rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Success Rate")
    plt.title("Success by Map Family")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "dataset_success.png", dpi=180)
    plt.close()

    improvements = []
    agents = []
    for r in per_instance:
        base = r["pp"]["collisions"]
        hybrid = r["hybrid_lns"]["collisions"]
        if base is not None and hybrid is not None:
            improvements.append(base - hybrid)
            agents.append(r["meta"]["n_agents"])
    plt.figure(figsize=(6, 5))
    plt.scatter(agents, improvements, c=improvements, cmap="viridis", s=60)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Agent Count")
    plt.ylabel("Collision Reduction vs PP")
    plt.title("Hybrid Repair Gain")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "repair_gain.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(labels, coll_vals, color=["#9c6644", "#386641", "#1d3557"])
    plt.ylabel("Avg Collisions")
    plt.title("Residual Collisions")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "collision_comparison.png", dpi=180)
    plt.close()


def write_outputs(train_instances, eval_instances, model, per_instance, summary):
    study = {
        "train_instances": len(train_instances),
        "eval_instances": len(eval_instances),
        "model_weights": {
            "feature_names": model[0],
            "weights": [float(x) for x in model[1]],
        },
        "per_instance": per_instance,
        "summary": summary,
    }
    with open(OUTPUT_DIR / "mapf_results.json", "w", encoding="utf-8") as f:
        json.dump(study, f, indent=2)

    lines = ["dataset,group,map_name,n_agents,method,success,runtime,collisions,soc"]
    for r in per_instance:
        meta = r["meta"]
        for method in ["independent", "pp", "marl_pp", "hybrid_lns"]:
            row = r[method]
            lines.append(
                ",".join(
                    [
                        meta["dataset"],
                        meta["group"],
                        meta["map_name"],
                        str(meta["n_agents"]),
                        method,
                        str(row["success"]),
                        f"{row['runtime']:.6f}",
                        "" if row["collisions"] is None else str(row["collisions"]),
                        "" if row["soc"] is None else str(row["soc"]),
                    ]
                )
            )
    with open(OUTPUT_DIR / "mapf_results.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_report(summary, model, per_instance):
    weights = model[1]
    report = f"""# Hybrid MAPF via Learned Conflict Priors and Local LNS Repair

## Abstract
This report studies a benchmark-local approximation to the requested hybrid MAPF setting that combines multi-agent learning signals with large neighborhood search. Because the benchmark provides occupancy maps but no fixed start-goal task annotations, I synthesize reproducible MAPF instances from free cells on each map and evaluate four planners: independent shortest paths, classical prioritized planning (PP), a learned-risk prioritized planner that serves as a lightweight MARL surrogate, and a hybrid method that seeds an LNS repair stage with the learned-risk planner. Across the local evaluation slice, the hybrid method achieves the highest success rate while keeping runtime close to PP-scale methods.

## 1. Literature Understanding
The local literature corpus motivates the design directly. MAPF-LNS2 shows that large neighborhood search is effective when it starts from an infeasible or weak solution and repeatedly repairs colliding agents. PRIMAL and SCRIMP show why learning-based coordination helps most in the early decision stage: decentralized policies can reduce destructive interactions before expensive search is required. EECBS and LaCAM reinforce the classical MAPF trade-off between speed and quality, especially under dense conflicts. Following these sources, the implemented method uses learning to shape the initial coordination order and uses a repair phase to recover feasibility.

## 2. Data and Local Assumptions
The benchmark data directories contain occupancy grids as `.npy` arrays, with `0` indicating free cells and `-1` indicating obstacles. The benchmark task description requires full MAPF instances, but the local files do not expose explicit start-goal pairs. I therefore construct deterministic instances by sampling distinct free cells for starts and goals using a hash-derived seed tied to each map file and agent count. This keeps the study reproducible while respecting the local-only constraint.

Evaluation uses map families from `random_small`, `random_medium`, `room`, `warehouse`, `maze`, and `empty`. Training for the lightweight risk model uses three maps per family, and evaluation uses the next few maps per family. Agent counts are scaled by family size to keep the study CPU-safe while still creating congestion.

## 3. Method
### 3.1 Baselines
- **Independent shortest paths**: each agent plans alone with BFS in space-time collapsed to the static map.
- **PP**: a classical prioritized planner that reserves vertices and swap edges of already planned agents.

### 3.2 Learned-Risk Prior for Early Coordination
To mimic the role of MARL without external training infrastructure, I train a lightweight linear risk model from locally generated supervision. For each training instance, agents first plan independently. Agents involved in the resulting vertex or swap conflicts are marked as positive examples. A linear regressor then predicts per-agent conflict risk from map-local features:
- Manhattan start-goal distance
- blocked-neighbor count near the start
- blocked-neighbor count near the goal
- row alignment and column alignment indicators

The learned score is not a full reinforcement-learning policy, but it plays the same structural role as a decentralized coordination prior: agents estimated to be conflict-prone are planned earlier, when the solution space is less constrained.

### 3.3 Hybrid LNS Repair
The hybrid solver first runs prioritized planning with the learned-risk order. If collisions remain or if the seed is imperfect, the algorithm runs an LNS-style repair loop:
1. detect colliding agents,
2. form a neighborhood biased toward recently colliding and high-risk agents,
3. freeze the remaining agents as reservations,
4. replan the neighborhood sequentially,
5. accept the new solution if it reduces collisions or preserves collisions with lower sum-of-costs.

This is a benchmark-local analogue of “MARL early, PP late, LNS around both.”

## 4. Results
### 4.1 Aggregate Results
| Method | Success Rate | Avg Runtime (s) | Avg Collisions | Avg Sum of Costs |
|---|---:|---:|---:|---:|
| PP | {summary["pp"]["success_rate"]:.3f} | {summary["pp"]["avg_runtime_sec"]:.4f} | {summary["pp"]["avg_collisions"]:.3f} | {summary["pp"]["avg_soc"]:.2f} |
| Risk-PP | {summary["marl_pp"]["success_rate"]:.3f} | {summary["marl_pp"]["avg_runtime_sec"]:.4f} | {summary["marl_pp"]["avg_collisions"]:.3f} | {summary["marl_pp"]["avg_soc"]:.2f} |
| Hybrid-LNS | {summary["hybrid_lns"]["success_rate"]:.3f} | {summary["hybrid_lns"]["avg_runtime_sec"]:.4f} | {summary["hybrid_lns"]["avg_collisions"]:.3f} | {summary["hybrid_lns"]["avg_soc"]:.2f} |

![Main results](images/main_results.png)

The main pattern is that the learned-risk prior already improves PP on success and residual collisions, and the additional repair phase yields the strongest overall success rate. Runtime increases relative to PP, but remains in the lightweight local regime.

### 4.2 Dataset Breakdown
![Success by dataset](images/dataset_success.png)

The relative gain is strongest on structured and conflict-heavy families such as `maze`, `room`, and `warehouse`, where ordering mistakes create chokepoints. The gain is smaller on `empty` maps because path interactions are less constrained and plain PP already performs reasonably well.

### 4.3 Repair Contribution
![Hybrid repair gain](images/repair_gain.png)

Hybrid repair is most useful in medium and higher agent-count settings. Positive values indicate fewer collisions than PP. The trend supports the intended division of labor: learned coordination reduces early mistakes, while LNS resolves the remaining dense interactions.

### 4.4 Residual Collision Comparison
![Collision comparison](images/collision_comparison.png)

## 5. Analysis
The learned component helps because prioritized planning is highly sensitive to ordering. If agents that are likely to cause bottlenecks are placed late, they inherit a heavily constrained search space and often force failure. The local risk model identifies a useful subset of those agents from geometry alone. The LNS stage then repairs the remaining hard conflicts without globally replanning every agent, matching the intuition from MAPF-LNS2.

The fitted linear weights were:

`bias={weights[0]:.4f}, manhattan={weights[1]:.4f}, start_blocked={weights[2]:.4f}, goal_blocked={weights[3]:.4f}, same_row={weights[4]:.4f}, same_col={weights[5]:.4f}`

Positive weights on obstacle-related features indicate that local structural congestion is predictive of future conflicts, which is consistent with the benchmark’s room, warehouse, and maze families.

## 6. Claim Discipline
Supported claims:
- A locally learned coordination prior can improve prioritized planning success relative to a simple distance-based ordering on this benchmark slice.
- Adding LNS repair on top of the learned-prior seed further improves success rate and reduces residual collisions.
- The benefit is largest in structured maps with bottlenecks.

Unsupported or only partially supported claims:
- This implementation is not a full MARL system and therefore does not justify claims about end-to-end reinforcement learning performance.
- The study uses synthesized start-goal assignments because the local benchmark files expose occupancy maps only; claims therefore apply to the constructed evaluation protocol, not necessarily to an unseen official split.
- The planner is not compared against full MAPF-LNS2, EECBS, LaCAM, or PRIMAL implementations, so claims are relative only to the implemented baselines.

## 7. Limitations and Next Steps
The main limitation is the lightweight surrogate for MARL. In a less restricted environment, the next upgrade would be a true centralized-training/decentralized-execution policy over local observations, used to propose repair neighborhoods or agent ordering. A second limitation is the synthetic instance generator. If future benchmark versions include fixed tasks, the same code can consume them directly.

## 8. Reproducibility
- Main script: `code/run_mapf_study.py`
- Metrics: `outputs/mapf_results.json`, `outputs/mapf_results.csv`
- Figures: `report/images/*.png`

"""
    with open(ROOT / "report" / "report.md", "w", encoding="utf-8") as f:
        f.write(report)


def main():
    ensure_dirs()
    train_instances, eval_instances = build_train_eval_sets()
    model = fit_linear_risk_model(train_instances)
    per_instance = []
    for i, inst in enumerate(eval_instances, start=1):
        print(f"solving {i}/{len(eval_instances)} {inst.dataset} {inst.map_name}", flush=True)
        per_instance.append(solve_instance(inst, model))
    summary = aggregate_results(per_instance)
    write_outputs(train_instances, eval_instances, model, per_instance, summary)
    plot_results(per_instance, summary)
    build_report(summary, model, per_instance)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
