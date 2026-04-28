"""
mapf_lns.py

Large Neighborhood Search (LNS2-style) for MAPF, plus the proposed hybrid MARL-LNS.

Workflow (MAPF-LNS2 style):
  1. Build an initial path set that may contain collisions (using PP without
     reservations + per-agent shortest paths).
  2. Repeat for a budget of iterations:
       - destroy: select a neighborhood N of agents to replan (we use
         collision-based selection: pick the most-conflicted agent and follow
         its conflicts)
       - repair: replan their paths to reduce collisions. We provide two repair
         operators:
           * pp_repair: classical Prioritized Planning repair against the
             reservations of the agents NOT in N
           * marl_repair: cooperative MARL-style decentralized rollout with a
             learned/policy-iterated value function on the local graph
       - if total collisions decreases, accept; otherwise revert.
  3. Termination when collisions == 0 (success) or budget exhausted (failure).

For the hybrid, we run MARL repair for the first `marl_iters` iterations and
PP repair afterwards (the warm "early-stage" MARL phase, then "late-stage" PP).
"""
from __future__ import annotations
import time
import random
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Set, Callable

from mapf_core import (
    Pos, Path, bfs_distance, st_astar, build_reservation_from_paths,
    detect_collisions, sum_of_costs, num_collisions, is_solution_valid,
    prioritized_planning,
)


# ---------- initial path construction ----------

def initial_paths_shortest(grid, starts, goals, h_tables, max_time=256) -> Dict[int, Path]:
    """Each agent gets its individual shortest path; ignores other agents.
    Used as the LNS2 initial (collision-prone) path set."""
    paths = {}
    empty_res = {'vertex': set(), 'edge': set(), 'final': {}}
    for i, (s, g) in enumerate(zip(starts, goals)):
        p = st_astar(grid, s, g, empty_res, h_table=h_tables[i], max_time=max_time)
        if p is None:
            # fallback: stay still then teleport later (return single-cell)
            paths[i] = [s]
        else:
            paths[i] = p
    return paths


# ---------- destroy operators ----------

def select_neighborhood_collision(
    paths: Dict[int, Path], k: int, rng: random.Random
) -> List[int]:
    cols = detect_collisions(paths)
    if not cols:
        return []
    # count conflicts per agent
    cnt: Dict[int, int] = {}
    for a, b, t, kind in cols:
        cnt[a] = cnt.get(a, 0) + 1
        cnt[b] = cnt.get(b, 0) + 1
    # pick most-conflicted as anchor
    anchor = max(cnt, key=cnt.get)
    nbhd = {anchor}
    # walk BFS over conflict graph
    # build adjacency
    adj: Dict[int, Set[int]] = {}
    for a, b, t, kind in cols:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    frontier = [anchor]
    while frontier and len(nbhd) < k:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in nbhd:
                    nbhd.add(v)
                    nxt.append(v)
                    if len(nbhd) >= k:
                        break
            if len(nbhd) >= k:
                break
        frontier = nxt
    # pad with random colliding agents if still short
    while len(nbhd) < k:
        candidates = [a for a in cnt if a not in nbhd]
        if not candidates:
            break
        nbhd.add(rng.choice(candidates))
    return list(nbhd)


def select_neighborhood_random(paths: Dict[int, Path], k: int, rng: random.Random) -> List[int]:
    aids = list(paths.keys())
    rng.shuffle(aids)
    return aids[:k]


# ---------- PP repair ----------

def pp_repair(
    grid: np.ndarray,
    starts: List[Pos],
    goals: List[Pos],
    h_tables: List[np.ndarray],
    paths: Dict[int, Path],
    nbhd: List[int],
    max_time: int,
    seed: int,
) -> Dict[int, Path]:
    """Replan paths for `nbhd` against the reservation of the others, in random order."""
    rng = random.Random(seed)
    others = {i: paths[i] for i in paths if i not in nbhd}
    reservation = build_reservation_from_paths(others)
    order = nbhd[:]
    rng.shuffle(order)
    new_paths = dict(paths)
    for aid in order:
        p = st_astar(grid, starts[aid], goals[aid], reservation,
                     h_table=h_tables[aid], max_time=max_time)
        if p is None:
            # repair failed for this agent: keep existing path (still colliding)
            continue
        new_paths[aid] = p
        # add to reservation
        for t, c in enumerate(p):
            reservation['vertex'].add((t, c))
        for t in range(len(p) - 1):
            a, b = p[t], p[t + 1]
            if a != b:
                reservation['edge'].add((t, a, b))
        prev = reservation['final'].get(goals[aid])
        ft = len(p) - 1
        if prev is None or ft < prev:
            reservation['final'][goals[aid]] = ft
    return new_paths


# ---------- MARL-style cooperative repair ----------
#
# Faithful-but-lightweight realisation of the PRIMAL/SCRIMP idea:
#   * each agent observes a local FOV (relative goal direction, nearby agents,
#     nearby obstacles)
#   * a shared policy maps observation to action (5 actions: stay + 4 moves)
#   * the policy is built from a value function V(cell) = -BFS_dist_to_goal
#     plus a SHARED tabular Q correction Q[(state_feat, action)] learned by
#     cooperative Q-learning over batches of mini-rollouts
#   * decentralized execution: at each step every agent in the neighborhood
#     samples an action from the policy; vertex collisions are resolved by
#     priority (random per-step shuffling); swap collisions cause the lower-
#     priority agent to wait.
#
# The "early-stage" MARL repair runs T rollout steps and produces full path
# segments for the neighborhood; if all reach their goals we accept. Otherwise
# the partial result is rejected and the iteration falls through.

ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # stay, up, down, left, right


def _local_obs_key(grid: np.ndarray,
                   pos: Pos,
                   goal: Pos,
                   others_now: Dict[Pos, int],
                   fov: int = 2) -> Tuple:
    """Compact tuple key for the local observation.
    Features:
      * sign of (goal - pos) per axis (-1, 0, +1)
      * occupancy of the 4 adjacent cells (0=free,1=obstacle,2=other-agent)
    """
    H, W = grid.shape
    dr = goal[0] - pos[0]
    dc = goal[1] - pos[1]
    sg_r = (dr > 0) - (dr < 0)
    sg_c = (dc > 0) - (dc < 0)
    feats = [sg_r, sg_c]
    for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = pos[0] + d[0], pos[1] + d[1]
        if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] == -1:
            feats.append(1)
        elif (nr, nc) in others_now:
            feats.append(2)
        else:
            feats.append(0)
    return tuple(feats)


class SharedQTable:
    """Tabular shared Q used by all MARL agents."""

    def __init__(self):
        self.q: Dict[Tuple, np.ndarray] = {}

    def get(self, key: Tuple) -> np.ndarray:
        v = self.q.get(key)
        if v is None:
            v = np.zeros(len(ACTIONS), dtype=np.float32)
            self.q[key] = v
        return v

    def update(self, key: Tuple, a_idx: int, target: float, lr: float):
        v = self.get(key)
        v[a_idx] += lr * (target - v[a_idx])

    def size(self) -> int:
        return len(self.q)


def _neighbors_free(grid: np.ndarray, pos: Pos) -> List[Pos]:
    H, W = grid.shape
    out = []
    for dr, dc in ACTIONS:
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
            out.append((nr, nc))
    return out


def marl_train_episodes(
    grid: np.ndarray,
    h_tables_for_train: List[np.ndarray],
    starts: List[Pos],
    goals: List[Pos],
    q: SharedQTable,
    n_episodes: int = 40,
    horizon: int = 64,
    epsilon: float = 0.2,
    lr: float = 0.3,
    gamma: float = 0.95,
    seed: int = 0,
) -> None:
    """Cooperative Q-learning episodes on a small set of training instances.
    Reward shaping: -1 per step, -0.5 per stay, -5 per blocked move attempt,
    -10 per vertex collision, +20 on reaching goal. The BFS distance is
    used as a potential-based shaping signal: phi(s) = -dist(s, goal).
    """
    rng = random.Random(seed)
    H, W = grid.shape
    n = len(starts)
    for ep in range(n_episodes):
        positions = list(starts)
        done = [False] * n
        for t in range(horizon):
            occ_now = {p: i for i, p in enumerate(positions)}
            actions = [0] * n
            order = list(range(n))
            rng.shuffle(order)
            new_positions = list(positions)
            new_occ: Dict[Pos, int] = {}
            for i in order:
                if done[i]:
                    new_positions[i] = goals[i]
                    new_occ[goals[i]] = i
                    continue
                key = _local_obs_key(grid, positions[i], goals[i], occ_now)
                qvals = q.get(key)
                if rng.random() < epsilon:
                    a = rng.randrange(len(ACTIONS))
                else:
                    a = int(np.argmax(qvals))
                actions[i] = a
                dr, dc = ACTIONS[a]
                cand = (positions[i][0] + dr, positions[i][1] + dc)
                # boundary / obstacle
                if not (0 <= cand[0] < H and 0 <= cand[1] < W) or grid[cand] == -1:
                    cand = positions[i]
                    pen_blocked = 5.0
                else:
                    pen_blocked = 0.0
                # vertex conflict at next step
                if cand in new_occ:
                    # blocked: stay
                    cand = positions[i]
                    pen_collision = 10.0
                else:
                    pen_collision = 0.0
                new_positions[i] = cand
                new_occ[cand] = i
                # reward
                d_old = float(h_tables_for_train[i][positions[i]])
                d_new = float(h_tables_for_train[i][cand])
                progress = d_old - d_new  # +1 for moving toward goal
                reward = -1.0 + 1.5 * progress - pen_blocked - pen_collision
                if cand == goals[i]:
                    reward += 20.0
                next_key = _local_obs_key(grid, cand, goals[i], new_occ)
                next_qvals = q.get(next_key)
                target = reward + gamma * float(np.max(next_qvals))
                q.update(key, a, target, lr)
            positions = new_positions
            for i in range(n):
                if positions[i] == goals[i]:
                    done[i] = True
            if all(done):
                break


def marl_rollout_repair(
    grid: np.ndarray,
    starts: List[Pos],
    goals: List[Pos],
    h_tables: List[np.ndarray],
    paths: Dict[int, Path],
    nbhd: List[int],
    q: SharedQTable,
    max_time: int,
    epsilon: float = 0.05,
    seed: int = 0,
    n_rollouts: int = 4,
) -> Dict[int, Path]:
    """Decentralized synchronous rollout for the agents in ``nbhd``. Other
    agents' paths are honoured via reservation. The rollout is rejected (we
    return the unchanged paths) if it cannot bring every nbhd agent to its
    goal."""
    rng = random.Random(seed)
    others = {i: paths[i] for i in paths if i not in nbhd}
    reservation = build_reservation_from_paths(others)
    vres = reservation['vertex']
    eres = reservation['edge']
    final_res: Dict[Pos, int] = reservation['final']
    H, W = grid.shape

    best_new = None
    best_score = (10**9, 10**9, 10**9)
    for ridx in range(n_rollouts):
        positions = {i: starts[i] for i in nbhd}
        traj: Dict[int, List[Pos]] = {i: [starts[i]] for i in nbhd}
        done = {i: starts[i] == goals[i] for i in nbhd}
        rng_local = random.Random(seed * 7919 + ridx * 17 + 1)
        for t in range(max_time):
            if all(done[i] for i in nbhd):
                break
            occ_now = {positions[i]: i for i in nbhd}
            order = list(nbhd)
            rng_local.shuffle(order)
            order.sort(key=lambda i: int(h_tables[i][positions[i]]))
            new_positions: Dict[int, Pos] = {}
            new_occ: Dict[Pos, int] = {}
            for i in order:
                if done[i]:
                    new_positions[i] = goals[i]
                    new_occ[goals[i]] = i
                    continue
                key = _local_obs_key(grid, positions[i], goals[i], occ_now)
                qvals = q.get(key)
                if rng_local.random() < epsilon:
                    a = rng_local.randrange(len(ACTIONS))
                else:
                    a = int(np.argmax(qvals))
                # BFS-biased fallback ordering (so progress is made even when
                # Q is uninformative)
                bfs_score = []
                for k_a, (dr_, dc_) in enumerate(ACTIONS):
                    nr_, nc_ = positions[i][0] + dr_, positions[i][1] + dc_
                    if 0 <= nr_ < H and 0 <= nc_ < W and grid[nr_, nc_] == 0:
                        bfs_score.append((int(h_tables[i][nr_, nc_]), k_a))
                    else:
                        bfs_score.append((10**9, k_a))
                bfs_score.sort()
                fallback = [k for _, k in bfs_score]
                tried = []
                chosen = None
                for choice in [a] + fallback:
                    if choice in tried:
                        continue
                    tried.append(choice)
                    dr, dc = ACTIONS[choice]
                    cand = (positions[i][0] + dr, positions[i][1] + dc)
                    if not (0 <= cand[0] < H and 0 <= cand[1] < W):
                        continue
                    if grid[cand] == -1:
                        continue
                    if (t + 1, cand) in vres:
                        continue
                    ft = final_res.get(cand)
                    if ft is not None and (t + 1) >= ft and cand != goals[i]:
                        continue
                    if (dr, dc) != (0, 0) and (t, cand, positions[i]) in eres:
                        continue
                    if cand in new_occ:
                        continue
                    new_positions[i] = cand
                    new_occ[cand] = i
                    chosen = cand
                    break
                if chosen is None:
                    cand = positions[i]
                    new_positions[i] = cand
                    if cand not in new_occ:
                        new_occ[cand] = i
            positions = new_positions
            for i in nbhd:
                traj[i].append(positions[i])
                if positions[i] == goals[i]:
                    done[i] = True
        # Trim trailing waits at goal so cost = arrival time
        for i in nbhd:
            tr = traj[i]
            if tr[-1] == goals[i]:
                k_arr = next((k for k, p in enumerate(tr) if p == goals[i]),
                             len(tr) - 1)
                traj[i] = tr[: k_arr + 1]
        cand_paths = dict(paths)
        for i in nbhd:
            cand_paths[i] = traj[i]
        unreached = sum(1 for i in nbhd if traj[i][-1] != goals[i])
        col = num_collisions(cand_paths)
        soc = sum(len(traj[i]) - 1 for i in nbhd)
        score = (unreached, col, soc)
        if score < best_score:
            best_score = score
            best_new = cand_paths
    if best_new is None or best_score[0] > 0:
        return paths
    return best_new


def lns_solve(
    grid: np.ndarray,
    starts: List[Pos],
    goals: List[Pos],
    repair: str = 'pp',          # 'pp' | 'marl' | 'hybrid'
    max_iters: int = 80,
    nbhd_size: int = 8,
    time_limit: float = 30.0,
    max_time: int = 256,
    seed: int = 0,
    h_tables: Optional[List[np.ndarray]] = None,
    q: Optional['SharedQTable'] = None,
    marl_iters_frac: float = 0.4,  # fraction of iters that use MARL in 'hybrid'
    log: Optional[List] = None,
) -> Tuple[Optional[Dict[int, Path]], Dict]:
    """Run LNS2-style search. Returns (paths or None, stats)."""
    rng = random.Random(seed)
    if h_tables is None:
        h_tables = [bfs_distance(grid, g) for g in goals]
    paths = initial_paths_shortest(grid, starts, goals, h_tables, max_time=max_time)
    if log is None:
        log = []
    cols = num_collisions(paths)
    log.append({'iter': 0, 'collisions': cols, 'soc': sum(len(p) - 1 for p in paths.values())})
    t0 = time.time()
    if cols == 0:
        ok, _ = is_solution_valid(paths, dict(enumerate(starts)), dict(enumerate(goals)))
        if ok:
            return paths, {'iters': 0, 'time': time.time() - t0, 'log': log,
                           'final_collisions': 0, 'success': True}
    marl_budget = int(max_iters * marl_iters_frac)
    # Smaller neighborhood for MARL repair (rollouts scale poorly with size)
    marl_nbhd = max(3, min(5, nbhd_size))
    marl_failures = 0   # consecutive failures of MARL to reduce collisions
    marl_active = (repair in ('marl', 'hybrid'))
    it = 0
    for it in range(1, max_iters + 1):
        if time.time() - t0 > time_limit:
            break
        if cols == 0:
            break
        # Determine which repair this iteration uses
        if repair == 'pp':
            phase = 'pp'
        elif repair == 'marl':
            phase = 'marl'
        elif repair == 'hybrid':
            # adaptive: MARL while in budget AND not too many consecutive
            # failures; otherwise switch to PP for the remainder.
            if marl_active and it <= marl_budget and marl_failures < 5:
                phase = 'marl'
            else:
                phase = 'pp'
                marl_active = False  # latch: never go back
        else:
            raise ValueError(repair)

        # Pick neighborhood (smaller for MARL phase)
        nb_size = marl_nbhd if phase == 'marl' else nbhd_size
        nbhd = select_neighborhood_collision(paths, nb_size, rng)
        if not nbhd:
            break

        if phase == 'pp':
            cand = pp_repair(grid, starts, goals, h_tables, paths, nbhd,
                             max_time, seed=seed * 31 + it)
        else:
            cand = marl_rollout_repair(grid, starts, goals, h_tables, paths, nbhd,
                                       q=q, max_time=max_time, seed=seed * 31 + it)

        new_cols = num_collisions(cand)
        improved = new_cols < cols
        if improved:
            paths = cand
            cols = new_cols
            if phase == 'marl':
                marl_failures = 0
        else:
            if phase == 'marl':
                marl_failures += 1
            if phase == 'pp' and new_cols == cols and rng.random() < 0.2:
                paths = cand
        soc = sum(len(p) - 1 for p in paths.values())
        log.append({'iter': it, 'collisions': cols, 'soc': soc,
                    'nbhd': len(nbhd), 'phase': phase})
    success = (cols == 0)
    if success:
        ok, _ = is_solution_valid(paths, dict(enumerate(starts)), dict(enumerate(goals)))
        success = ok
    return paths, {
        'iters': it if 'it' in dir() else 0,
        'time': time.time() - t0,
        'log': log,
        'final_collisions': cols,
        'success': success,
    }
