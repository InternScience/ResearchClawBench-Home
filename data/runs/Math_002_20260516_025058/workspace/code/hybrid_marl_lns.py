#!/usr/bin/env python3
"""
Hybrid MARL + LNS solver for MAPF
- Early iterations: MARL-guided destroy/repair (PRIMAL-style decentralized policy)
- Later iterations: Prioritized Planning repair
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict, Optional
import random

# -----------------------------
# Environment & MAPF utilities
# -----------------------------

class MAPFEnv:
    def __init__(self, grid: np.ndarray, starts: List[Tuple[int,int]], goals: List[Tuple[int,int]]):
        self.grid = grid  # 0 free, -1 obstacle
        self.starts = starts
        self.goals = goals
        self.n_agents = len(starts)
        self.rows, self.cols = grid.shape
        self.directions = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]  # N S W E Wait

    def is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] == 0

    def get_neighbors(self, pos):
        r, c = pos
        neigh = []
        for dr, dc in self.directions:
            nr, nc = r+dr, c+dc
            if self.is_valid((nr,nc)):
                neigh.append((nr,nc))
        return neigh

    def check_collisions(self, paths: List[List[Tuple[int,int]]]) -> int:
        """Return number of collisions (vertex + edge)"""
        collisions = 0
        max_len = max(len(p) for p in paths)
        for t in range(max_len):
            positions = {}
            for a in range(self.n_agents):
                pos = paths[a][min(t, len(paths[a])-1)]
                if pos in positions:
                    collisions += 1
                positions[pos] = a
            # edge collisions
            if t > 0:
                for a in range(self.n_agents):
                    prev = paths[a][min(t-1, len(paths[a])-1)]
                    curr = paths[a][min(t, len(paths[a])-1)]
                    for b in range(a+1, self.n_agents):
                        bprev = paths[b][min(t-1, len(paths[b])-1)]
                        bcurr = paths[b][min(t, len(paths[b])-1)]
                        if prev == bcurr and curr == bprev:
                            collisions += 1
        return collisions

    def compute_makespan(self, paths):
        return max(len(p) for p in paths)

    def compute_sum_of_costs(self, paths):
        return sum(len(p) for p in paths)


# -----------------------------
# Simple PRIMAL-style MARL Policy
# -----------------------------

class MARLPolicy(nn.Module):
    def __init__(self, obs_dim=5, hidden=64, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        with torch.no_grad():
            logits = self.forward(obs)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action


def local_observation(env: MAPFEnv, agent: int, pos: Tuple[int,int], goal: Tuple[int,int]) -> torch.Tensor:
    """Very simple local observation: relative goal + 4-dir obstacle flags"""
    r, c = pos
    gr, gc = goal
    dr, dc = gr - r, gc - c
    obs = torch.tensor([dr/10.0, dc/10.0], dtype=torch.float32)
    # 4 direction free flags
    for drr, dcc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+drr, c+dcc
        free = 1.0 if env.is_valid((nr,nc)) else 0.0
        obs = torch.cat([obs, torch.tensor([free])])
    return obs


# -----------------------------
# LNS Framework
# -----------------------------

def prioritized_planning(env: MAPFEnv, max_time=10.0) -> List[List[Tuple[int,int]]]:
    """Simple prioritized planning using A*"""
    paths = []
    occupied = set()
    start_time = time.time()

    for a in range(env.n_agents):
        if time.time() - start_time > max_time:
            break
        start = env.starts[a]
        goal = env.goals[a]
        path = a_star(env, start, goal, occupied)
        if path is None:
            # fallback: stay at start
            path = [start]
        paths.append(path)
        for p in path:
            occupied.add(p)
    return paths


def a_star(env: MAPFEnv, start, goal, occupied=None):
    if occupied is None:
        occupied = set()
    open_set = []
    import heapq
    heapq.heappush(open_set, (0, start, [start]))
    visited = set()
    while open_set:
        _, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        for neigh in env.get_neighbors(current):
            if neigh not in visited and neigh not in occupied:
                new_path = path + [neigh]
                cost = len(new_path) + abs(neigh[0]-goal[0]) + abs(neigh[1]-goal[1])
                heapq.heappush(open_set, (cost, neigh, new_path))
    return None


def marl_repair(env: MAPFEnv, policy: MARLPolicy, paths: List[List], destroy_ratio=0.3):
    """MARL-guided repair on a subset of agents"""
    n = env.n_agents
    destroy_count = max(1, int(n * destroy_ratio))
    destroy_agents = random.sample(range(n), destroy_count)

    new_paths = [p[:] for p in paths]
    for a in destroy_agents:
        start = env.starts[a]
        goal = env.goals[a]
        # simple MARL rollout (very crude)
        pos = start
        path = [pos]
        for _ in range(50):
            obs = local_observation(env, a, pos, goal)
            action = policy.act(obs)
            dr, dc = env.directions[action]
            new_pos = (pos[0]+dr, pos[1]+dc)
            if env.is_valid(new_pos):
                pos = new_pos
            path.append(pos)
            if pos == goal:
                break
        new_paths[a] = path
    return new_paths


def hybrid_lns_solver(env: MAPFEnv, max_iter=30, marl_iters=15):
    """Main hybrid algorithm"""
    policy = MARLPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Initial solution with PP
    paths = prioritized_planning(env)
    best_paths = [p[:] for p in paths]
    best_cost = env.compute_sum_of_costs(best_paths)

    for it in range(max_iter):
        if it < marl_iters:
            # MARL-guided repair
            paths = marl_repair(env, policy, paths, destroy_ratio=0.4)
        else:
            # Switch to PP repair
            paths = prioritized_planning(env)

        cost = env.compute_sum_of_costs(paths)
        if cost < best_cost:
            best_cost = cost
            best_paths = [p[:] for p in paths]

        # crude policy update (reward = -collisions)
        collisions = env.check_collisions(paths)
        reward = -collisions
        loss = -reward * torch.tensor(1.0)  # dummy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return best_paths, best_cost


# -----------------------------
# Evaluation & Plotting
# -----------------------------

def load_map(map_path: str, n_agents: int = 10):
    grid = np.load(map_path, allow_pickle=True)
    rows, cols = grid.shape
    free_cells = [(r,c) for r in range(rows) for c in range(cols) if grid[r,c]==0]
    random.shuffle(free_cells)
    starts = free_cells[:n_agents]
    goals = free_cells[n_agents:2*n_agents]
    return MAPFEnv(grid, starts, goals)


def evaluate_on_dataset(data_dir: str, n_maps=5, n_agents=10):
    results = []
    map_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')][:n_maps]
    for mf in map_files:
        env = load_map(os.path.join(data_dir, mf), n_agents)
        start = time.time()
        paths, cost = hybrid_lns_solver(env)
        runtime = time.time() - start
        success = env.check_collisions(paths) == 0
        results.append({
            'map': mf,
            'success': success,
            'cost': cost,
            'runtime': runtime,
            'makespan': env.compute_makespan(paths)
        })
    return results


def plot_results(results, out_path):
    success = [r['success'] for r in results]
    costs = [r['cost'] for r in results]
    runtimes = [r['runtime'] for r in results]

    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].bar(range(len(success)), success)
    axs[0].set_title('Success Rate')
    axs[0].set_ylim(0,1.1)

    axs[1].plot(costs, marker='o')
    axs[1].set_title('Sum of Costs')

    axs[2].plot(runtimes, marker='o', color='orange')
    axs[2].set_title('Runtime (s)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    # Quick test on warehouse
    data_dir = "data/warehouse"
    res = evaluate_on_dataset(data_dir, n_maps=3, n_agents=8)
    print(res)
    plot_results(res, "report/images/results_warehouse.png")
    print("Done. Figure saved.")