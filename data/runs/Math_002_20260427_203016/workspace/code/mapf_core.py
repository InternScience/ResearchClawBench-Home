"""
mapf_core.py

Core utilities:
  - load grid map
  - generate MAPF instance (start/goal sampling) deterministically
  - single-agent space-time A* with reservation table
  - collision detection (vertex + swap/edge)
  - prioritized planning (PP)

A grid map: numpy int array, 0 = free, -1 = obstacle.
A path: list of (r, c) positions, length T+1, path[0] = start, path[-1] = goal.
Time index t corresponds to position path[t]. Agents that have already arrived
are assumed to remain at their goal cell forever (we extend paths with their
goal when checking collisions among different lengths).
"""
from __future__ import annotations
import os
import heapq
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Set

Pos = Tuple[int, int]
Path = List[Pos]


# ---------- map / instance utilities ----------

def load_grid(path: str) -> np.ndarray:
    g = np.load(path, allow_pickle=True)
    if g.dtype != np.int64:
        g = g.astype(np.int64)
    return g


def free_cells(grid: np.ndarray) -> List[Pos]:
    rs, cs = np.where(grid == 0)
    return list(zip(rs.tolist(), cs.tolist()))


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_distance(grid: np.ndarray, goal: Pos) -> np.ndarray:
    """All-pairs single-source BFS distance from goal across free cells."""
    H, W = grid.shape
    dist = np.full((H, W), np.iinfo(np.int32).max // 4, dtype=np.int32)
    dist[goal] = 0
    from collections import deque
    q = deque([goal])
    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and dist[nr, nc] > d + 1:
                dist[nr, nc] = d + 1
                q.append((nr, nc))
    return dist


def generate_instance(grid: np.ndarray, n_agents: int, seed: int) -> Tuple[List[Pos], List[Pos]]:
    """Sample n_agents distinct starts and distinct goals among free cells.
    Each agent must have a feasible path from its start to its goal (we check
    by ensuring start and goal are in the same BFS component)."""
    rng = random.Random(seed)
    free = free_cells(grid)
    if len(free) < 2 * n_agents:
        # not enough cells; fall back to as many as feasible
        n_agents = max(1, len(free) // 2)
    # Connected component check via flood fill
    H, W = grid.shape
    seen = np.zeros_like(grid, dtype=np.int32)
    comp_id = 0
    comps: List[List[Pos]] = []
    for cell in free:
        if seen[cell]:
            continue
        comp_id += 1
        stack = [cell]
        comp_cells = []
        seen[cell] = comp_id
        while stack:
            r, c = stack.pop()
            comp_cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and seen[nr, nc] == 0:
                    seen[nr, nc] = comp_id
                    stack.append((nr, nc))
        comps.append(comp_cells)
    # Pick the largest component
    comps.sort(key=len, reverse=True)
    comp = comps[0]
    if len(comp) < 2 * n_agents:
        n_agents = max(1, len(comp) // 2)
    rng.shuffle(comp)
    starts = comp[:n_agents]
    goals_pool = comp.copy()
    rng.shuffle(goals_pool)
    # Ensure start_i != goal_i
    goals: List[Pos] = []
    used = set()
    for s in starts:
        for g in goals_pool:
            if g == s or g in used:
                continue
            goals.append(g)
            used.add(g)
            break
        else:
            # fallback: any cell
            for g in comp:
                if g != s and g not in used:
                    goals.append(g)
                    used.add(g)
                    break
    return starts, goals


# ---------- space-time A* ----------

def st_astar(
    grid: np.ndarray,
    start: Pos,
    goal: Pos,
    reservation: Dict,
    h_table: Optional[np.ndarray] = None,
    max_time: int = 256,
    final_blocked: Optional[Set[Pos]] = None,
) -> Optional[Path]:
    """Single-agent space-time A*.
    reservation: dict with two keys
      'vertex': set of (t, (r,c)) booked
      'edge':   set of (t, (r,c), (r2,c2)) movement booked from t->t+1 by another
                agent that we cannot swap with.
      'final':  dict mapping cell -> earliest time t* from which the cell is
                permanently occupied (by an agent that has reached its goal and
                is staying). If t >= t* and cell == that goal, blocked.
    final_blocked: optional set of cells that can never be entered (e.g. other
                   agents' starts when planning order matters). Currently unused.
    h_table: precomputed BFS distance to goal.
    """
    H, W = grid.shape
    if h_table is None:
        h_table = bfs_distance(grid, goal)
    if h_table[start] >= np.iinfo(np.int32).max // 4:
        return None
    vres: Set = reservation.get('vertex', set())
    eres: Set = reservation.get('edge', set())
    final_res: Dict[Pos, int] = reservation.get('final', {})

    def is_v_blocked(t: int, cell: Pos) -> bool:
        if (t, cell) in vres:
            return True
        # final reservations
        ft = final_res.get(cell)
        if ft is not None and t >= ft:
            return True
        return False

    def is_e_blocked(t: int, a: Pos, b: Pos) -> bool:
        # swap conflict: another agent moves b -> a at the same step t -> t+1
        if (t, b, a) in eres:
            return True
        return False

    # Heuristic-tight bound: best g + h <= max_time
    open_heap: List = []
    start_h = int(h_table[start])
    counter = 0
    heapq.heappush(open_heap, (start_h, 0, counter, start, 0, None))
    counter += 1
    # parents map keyed by (t, cell)
    parents: Dict[Tuple[int, Pos], Tuple[int, Pos]] = {}
    g_best: Dict[Tuple[int, Pos], int] = {(0, start): 0}
    closed: Set[Tuple[int, Pos]] = set()

    while open_heap:
        f, g, _, cell, t, parent = heapq.heappop(open_heap)
        if (t, cell) in closed:
            continue
        closed.add((t, cell))
        if parent is not None:
            parents[(t, cell)] = parent
        # Goal check: at goal AND no future vertex reservation forces us to leave
        if cell == goal:
            # Need to ensure no future reserved entry on goal beyond t.
            # If yes we must keep waiting and exit only when clear.
            ok = True
            # search any (t', goal) with t' > t in vres (and final_res)
            future_block = False
            # quick approximation: scan small horizon
            for tp in range(t + 1, min(max_time, t + 64) + 1):
                if (tp, goal) in vres:
                    future_block = True
                    break
            ft = final_res.get(goal)
            if ft is not None and ft <= t:
                future_block = True
            if not future_block:
                # reconstruct
                path = [cell]
                cur = (t, cell)
                while cur in parents:
                    pcur = parents[cur]
                    path.append(pcur[1])
                    cur = pcur
                path.reverse()
                return path
        if t >= max_time:
            continue
        # Expand
        r, c = cell
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc] == -1:
                continue
            ncell = (nr, nc)
            nt = t + 1
            if is_v_blocked(nt, ncell):
                continue
            if (dr, dc) != (0, 0) and is_e_blocked(t, cell, ncell):
                continue
            ng = g + 1
            key = (nt, ncell)
            if key in g_best and g_best[key] <= ng:
                continue
            g_best[key] = ng
            h = int(h_table[ncell])
            if ng + h > max_time:
                continue
            counter += 1
            heapq.heappush(open_heap, (ng + h, ng, counter, ncell, nt, (t, cell)))
            parents[key] = (t, cell)
    return None


def build_reservation_from_paths(paths: Dict[int, Path], skip: Optional[Set[int]] = None) -> Dict:
    """Build vertex/edge/final reservation from existing paths (keyed by agent id).
    Skip a set of agent ids (the ones being replanned)."""
    skip = skip or set()
    vres: Set = set()
    eres: Set = set()
    final_res: Dict[Pos, int] = {}
    for aid, path in paths.items():
        if aid in skip or path is None:
            continue
        for t, p in enumerate(path):
            vres.add((t, p))
        for t in range(len(path) - 1):
            a, b = path[t], path[t + 1]
            if a != b:
                eres.add((t, a, b))
        # final occupy
        goal = path[-1]
        ft = len(path) - 1
        prev = final_res.get(goal)
        if prev is None or ft < prev:
            final_res[goal] = ft
    return {'vertex': vres, 'edge': eres, 'final': final_res}


# ---------- collision detection ----------

def detect_collisions(paths: Dict[int, Path]) -> List[Tuple[int, int, int, str]]:
    """Return list of (agent_a, agent_b, t, kind) collisions. kind in {'vertex','swap'}."""
    if not paths:
        return []
    T = max(len(p) for p in paths.values()) if paths else 0
    aids = list(paths.keys())
    cols: List = []
    # extend implicitly: position(a, t) = paths[a][min(t, len-1)]
    def pos(a, t):
        p = paths[a]
        if t >= len(p):
            return p[-1]
        return p[t]
    for t in range(T):
        seen_v: Dict[Pos, int] = {}
        for a in aids:
            p = pos(a, t)
            if p in seen_v:
                cols.append((seen_v[p], a, t, 'vertex'))
            else:
                seen_v[p] = a
    for t in range(T - 1):
        # swap: a goes p->q while b goes q->p
        for i, a in enumerate(aids):
            pa = pos(a, t); qa = pos(a, t + 1)
            if pa == qa:
                continue
            for b in aids[i + 1:]:
                pb = pos(b, t); qb = pos(b, t + 1)
                if pa == qb and qa == pb:
                    cols.append((a, b, t, 'swap'))
    return cols


def num_collisions(paths: Dict[int, Path]) -> int:
    return len(detect_collisions(paths))


def sum_of_costs(paths: Dict[int, Path], goals: Dict[int, Pos]) -> int:
    s = 0
    for a, p in paths.items():
        # cost = arrival time (last step where agent moves) -- here = len(p)-1
        # but if agent already at goal we still include trailing waits as cost
        # Standard SoC: time of arrival at goal (final step).
        s += len(p) - 1
    return s


def makespan(paths: Dict[int, Path]) -> int:
    return max((len(p) - 1 for p in paths.values()), default=0)


def is_solution_valid(paths: Dict[int, Path], starts: Dict[int, Pos], goals: Dict[int, Pos]) -> Tuple[bool, str]:
    for a, p in paths.items():
        if not p:
            return False, f"empty path for agent {a}"
        if p[0] != starts[a]:
            return False, f"start mismatch agent {a}"
        if p[-1] != goals[a]:
            return False, f"goal mismatch agent {a}"
    cols = detect_collisions(paths)
    if cols:
        return False, f"{len(cols)} collisions e.g. {cols[0]}"
    return True, "ok"


# ---------- Prioritized Planning ----------

def prioritized_planning(
    grid: np.ndarray,
    starts: List[Pos],
    goals: List[Pos],
    h_tables: Optional[List[np.ndarray]] = None,
    max_time: int = 256,
    seed: int = 0,
    order: Optional[List[int]] = None,
) -> Tuple[Optional[Dict[int, Path]], int]:
    """Standard PP: plan agents in order, each respects previous reservations.
    Returns (paths or None, agents_failed)."""
    n = len(starts)
    if h_tables is None:
        h_tables = [bfs_distance(grid, g) for g in goals]
    if order is None:
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
    paths: Dict[int, Path] = {}
    reservation = {'vertex': set(), 'edge': set(), 'final': {}}
    failed = 0
    for aid in order:
        path = st_astar(grid, starts[aid], goals[aid], reservation,
                        h_table=h_tables[aid], max_time=max_time)
        if path is None:
            failed += 1
            paths[aid] = None  # mark failure
            continue
        paths[aid] = path
        # extend reservation
        for t, p in enumerate(path):
            reservation['vertex'].add((t, p))
        for t in range(len(path) - 1):
            a, b = path[t], path[t + 1]
            if a != b:
                reservation['edge'].add((t, a, b))
        prev = reservation['final'].get(goals[aid])
        ft = len(path) - 1
        if prev is None or ft < prev:
            reservation['final'][goals[aid]] = ft
    if failed > 0:
        return None, failed
    return paths, 0
