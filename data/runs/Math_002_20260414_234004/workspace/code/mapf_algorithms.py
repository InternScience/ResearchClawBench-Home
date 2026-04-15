"""
MAPF Environment and Algorithm Implementation
Hybrid MARL-LNS for Multi-Agent Path Finding
"""
import numpy as np
import heapq
import time
import json
import os
from collections import defaultdict

# ============================================================
# MAPF Environment
# ============================================================

class MAPFEnv:
    """Multi-Agent Path Finding Environment on a 2D grid."""
    
    MOVES = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # wait, right, left, down, up
    
    def __init__(self, grid, starts, goals):
        self.grid = grid  # 2D numpy array: 0=free, -1=obstacle
        self.H, self.W = grid.shape
        self.starts = starts  # list of (r, c)
        self.goals = goals    # list of (r, c)
        self.num_agents = len(starts)
    
    def is_valid(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W and self.grid[r, c] == 0
    
    def neighbors(self, r, c):
        result = []
        for dr, dc in self.MOVES:
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc):
                result.append((nr, nc))
        return result
    
    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ============================================================
# Single-Agent A* Search
# ============================================================

def astar(env, start, goal, obstacles=None, max_steps=50000):
    """A* search for a single agent, avoiding obstacles (set of (r,c,t) tuples)."""
    if obstacles is None:
        obstacles = set()
    
    open_set = []
    heapq.heappush(open_set, (env.manhattan(start, goal), 0, start, [start]))
    visited = set()
    
    steps = 0
    while open_set and steps < max_steps:
        steps += 1
        f, g, pos, path = heapq.heappop(open_set)
        
        if pos == goal:
            return path
        
        if (pos, g) in visited:
            continue
        visited.add((pos, g))
        
        for nr, nc in env.neighbors(pos[0], pos[1]):
            ng = g + 1
            if ((nr, nc), ng) not in visited and ((nr, nc), ng) not in obstacles:
                nf = ng + env.manhattan((nr, nc), goal)
                heapq.heappush(open_set, (nf, ng, (nr, nc), path + [(nr, nc)]))
    
    return None


# ============================================================
# Prioritized Planning (PP)
# ============================================================

def prioritized_planning(env, order=None, time_limit=30.0):
    """Prioritized Planning: plan agents in priority order."""
    if order is None:
        order = list(range(env.num_agents))
    
    start_time = time.time()
    paths = [None] * env.num_agents
    occupied = set()  # (r, c, t)
    
    for idx in order:
        if time.time() - start_time > time_limit:
            return None, time.time() - start_time
        
        s = env.starts[idx]
        g = env.goals[idx]
        
        # Build obstacles from already-planned paths
        obs = set(occupied)
        path = astar(env, s, g, obstacles=obs)
        
        if path is None:
            return None, time.time() - start_time
        
        paths[idx] = path
        # Add path to occupied set
        for t, (r, c) in enumerate(path):
            occupied.add(((r, c), t))
        # Agent stays at goal after reaching it
        last_t = len(path) - 1
        last_pos = path[-1]
        for t in range(last_t, last_t + env.H * env.W):
            occupied.add((last_pos, t))
    
    return paths, time.time() - start_time


# ============================================================
# Collision Detection
# ============================================================

def detect_collisions(paths):
    """Detect vertex and swap collisions among paths."""
    collisions = []
    max_len = max(len(p) for p in paths)
    
    for t in range(max_len):
        pos_at_t = {}
        for i, p in enumerate(paths):
            pos = p[min(t, len(p) - 1)]
            if pos in pos_at_t:
                collisions.append(('vertex', pos_at_t[pos], i, t))
            else:
                pos_at_t[pos] = i
        
        # Swap collisions
        if t < max_len - 1:
            pos_at_t1 = {}
            for i, p in enumerate(paths):
                pos = p[min(t + 1, len(p) - 1)]
                pos_at_t1[i] = pos
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    pi_t = paths[i][min(t, len(paths[i]) - 1)]
                    pj_t = paths[j][min(t, len(paths[j]) - 1)]
                    pi_t1 = pos_at_t1[i]
                    pj_t1 = pos_at_t1[j]
                    if pi_t == pj_t1 and pi_t1 == pj_t:
                        collisions.append(('swap', i, j, t))
    
    return collisions


def count_collisions(paths):
    """Count total collisions."""
    return len(detect_collisions(paths))


# ============================================================
# Large Neighborhood Search (LNS)
# ============================================================

def lns_destroy_random(paths, num_destroy):
    """Randomly select agents to destroy (replan)."""
    n = len(paths)
    k = min(num_destroy, n)
    indices = np.random.choice(n, k, replace=False).tolist()
    return indices


def lns_destroy_colliding(paths, num_destroy):
    """Select colliding agents to destroy."""
    collisions = detect_collisions(paths)
    colliding_agents = set()
    for c in collisions:
        if c[0] == 'vertex':
            colliding_agents.add(c[1])
            colliding_agents.add(c[2])
        else:
            colliding_agents.add(c[1])
            colliding_agents.add(c[2])
    
    colliding_list = list(colliding_agents)
    if not colliding_list:
        return lns_destroy_random(paths, num_destroy)
    
    k = min(num_destroy, len(colliding_list))
    indices = np.random.choice(colliding_list, k, replace=False).tolist()
    return indices


def lns_repair_pp(env, paths, destroy_indices, occupied_cache=None):
    """Repair destroyed paths using Prioritized Planning."""
    new_paths = list(paths)
    
    # Build occupied set from non-destroyed paths
    occupied = set()
    for i, p in enumerate(new_paths):
        if i not in destroy_indices and p is not None:
            for t, (r, c) in enumerate(p):
                occupied.add(((r, c), t))
            last_t = len(p) - 1
            for t in range(last_t, last_t + env.H * env.W):
                occupied.add((p[-1], t))
    
    # Replan destroyed paths in random order
    order = list(destroy_indices)
    np.random.shuffle(order)
    
    for idx in order:
        s = env.starts[idx]
        g = env.goals[idx]
        path = astar(env, s, g, obstacles=occupied)
        if path is None:
            return None
        new_paths[idx] = path
        for t, (r, c) in enumerate(path):
            occupied.add(((r, c), t))
        last_t = len(path) - 1
        for t in range(last_t, last_t + env.H * env.W):
            occupied.add((path[-1], t))
    
    return new_paths


def lns(env, initial_paths=None, time_limit=60.0, num_destroy=2, max_iter=10000):
    """Large Neighborhood Search for MAPF."""
    start_time = time.time()
    
    if initial_paths is None:
        # Start with individual shortest paths (may have collisions)
        initial_paths = []
        for i in range(env.num_agents):
            p = astar(env, env.starts[i], env.goals[i])
            if p is None:
                return None, 0, float('inf')
            initial_paths.append(p)
    
    best_paths = list(initial_paths)
    best_collisions = count_collisions(best_paths)
    
    if best_collisions == 0:
        return best_paths, time.time() - start_time, 0
    
    iteration = 0
    while time.time() - start_time < time_limit and iteration < max_iter:
        iteration += 1
        
        # Alternate between destroy strategies
        if iteration % 3 == 0:
            destroy_idx = lns_destroy_colliding(best_paths, num_destroy)
        else:
            destroy_idx = lns_destroy_random(best_paths, num_destroy)
        
        new_paths = lns_repair_pp(env, best_paths, destroy_idx)
        
        if new_paths is not None:
            new_collisions = count_collisions(new_paths)
            if new_collisions < best_collisions:
                best_paths = new_paths
                best_collisions = new_collisions
                if best_collisions == 0:
                    return best_paths, time.time() - start_time, 0
        
        # Adapt destroy size
        if iteration % 100 == 0:
            num_destroy = min(num_destroy + 1, env.num_agents // 2)
    
    return best_paths, time.time() - start_time, best_collisions


# ============================================================
# MARL-inspired Policy (Simplified Q-learning style)
# ============================================================

class MARLPolicy:
    """Simplified MARL policy using learned Q-values for action selection.
    
    Instead of training a full neural network, we use a feature-based
    Q-function that captures:
    1. Distance to goal (primary signal)
    2. Local congestion (agents nearby)
    3. Obstacle proximity
    """
    
    def __init__(self, env, learning_rate=0.1, gamma=0.95, epsilon=0.3):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: state_hash -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.visit_count = defaultdict(lambda: defaultdict(int))
    
    def get_state_features(self, agent_pos, other_positions, goal):
        """Extract discrete state features."""
        r, c = agent_pos
        gr, gc = goal
        
        # Direction to goal (discretized)
        dr = np.sign(gr - r)
        dc = np.sign(gc - c)
        
        # Local congestion (agents within 3 cells)
        congestion = sum(1 for op in other_positions 
                        if abs(op[0] - r) <= 3 and abs(op[1] - c) <= 3 and op != agent_pos)
        congestion_bin = min(congestion, 4)
        
        # Distance to goal (binned)
        dist = self.env.manhattan(agent_pos, goal)
        dist_bin = min(dist // 5, 5)
        
        return (dr, dc, congestion_bin, dist_bin)
    
    def select_action(self, agent_idx, current_pos, all_positions):
        """Select action using epsilon-greedy with learned Q-values."""
        goal = self.env.goals[agent_idx]
        other_pos = [all_positions[i] for i in range(len(all_positions)) if i != agent_idx]
        state = self.get_state_features(current_pos, other_pos, goal)
        
        valid_actions = []
        for a_idx, (dr, dc) in enumerate(self.env.MOVES):
            nr, nc = current_pos[0] + dr, current_pos[1] + dc
            if self.env.is_valid(nr, nc):
                valid_actions.append(a_idx)
        
        if not valid_actions:
            return 0  # wait
        
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Greedy: pick action with highest Q-value
        best_action = valid_actions[0]
        best_q = self.q_table[state][best_action]
        for a in valid_actions[1:]:
            q = self.q_table[state][a]
            if q > best_q:
                best_q = q
                best_action = a
        return best_action
    
    def compute_reward(self, agent_idx, old_pos, new_pos, all_new_positions, collisions_this_step):
        """Compute reward for the agent."""
        goal = self.env.goals[agent_idx]
        
        reward = 0.0
        
        # Goal proximity reward
        old_dist = self.env.manhattan(old_pos, goal)
        new_dist = self.env.manhattan(new_pos, goal)
        reward += (old_dist - new_dist) * 1.0
        
        # Reached goal bonus
        if new_pos == goal:
            reward += 5.0
        
        # Collision penalty
        for c in collisions_this_step:
            if c[0] == 'vertex' and (c[1] == agent_idx or c[2] == agent_idx):
                reward -= 10.0
            elif c[0] == 'swap' and (c[1] == agent_idx or c[2] == agent_idx):
                reward -= 10.0
        
        # Congestion penalty
        congestion = sum(1 for i, p in enumerate(all_new_positions) 
                        if i != agent_idx and 
                        abs(p[0] - new_pos[0]) <= 1 and abs(p[1] - new_pos[1]) <= 1)
        reward -= congestion * 0.5
        
        return reward
    
    def update(self, state, action, reward, next_state):
        """Q-learning update."""
        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state][action] = new_q
        self.visit_count[state][action] += 1


def run_marl_episode(env, policy, max_steps=200):
    """Run one MARL episode, return paths and collision count."""
    positions = list(env.starts)
    paths = [[pos] for pos in positions]
    all_collisions = []
    
    for step in range(max_steps):
        # Check if all agents reached goals
        if all(pos == env.goals[i] for i, pos in enumerate(positions)):
            break
        
        # Select actions
        new_positions = list(positions)
        for i in range(env.num_agents):
            if positions[i] == env.goals[i]:
                new_positions[i] = positions[i]  # stay at goal
                continue
            action = policy.select_action(i, positions[i], positions)
            dr, dc = env.MOVES[action]
            nr, nc = positions[i][0] + dr, positions[i][1] + dc
            if env.is_valid(nr, nc):
                new_positions[i] = (nr, nc)
        
        # Detect collisions
        step_collisions = []
        for i in range(env.num_agents):
            for j in range(i + 1, env.num_agents):
                if new_positions[i] == new_positions[j]:
                    step_collisions.append(('vertex', i, j, step))
                if positions[i] == new_positions[j] and positions[j] == new_positions[i]:
                    step_collisions.append(('swap', i, j, step))
        
        all_collisions.extend(step_collisions)
        
        # Update Q-values
        for i in range(env.num_agents):
            if positions[i] == env.goals[i]:
                continue
            other_pos = [positions[k] for k in range(env.num_agents) if k != i]
            state = policy.get_state_features(positions[i], other_pos, env.goals[i])
            action = policy.select_action(i, positions[i], positions)
            reward = policy.compute_reward(i, positions[i], new_positions[i], new_positions, step_collisions)
            other_new = [new_positions[k] for k in range(env.num_agents) if k != i]
            next_state = policy.get_state_features(new_positions[i], other_new, env.goals[i])
            policy.update(state, action, reward, next_state)
        
        positions = new_positions
        for i in range(env.num_agents):
            paths[i].append(positions[i])
    
    return paths, len(all_collisions)


def train_marl(env, num_episodes=50, time_limit=30.0):
    """Train MARL policy and return best paths found."""
    policy = MARLPolicy(env, epsilon=0.5)
    best_paths = None
    best_collisions = float('inf')
    
    start_time = time.time()
    for ep in range(num_episodes):
        if time.time() - start_time > time_limit:
            break
        
        # Decay epsilon
        policy.epsilon = max(0.1, 0.5 * (0.97 ** ep))
        
        paths, collisions = run_marl_episode(env, policy, max_steps=env.H * env.W)
        
        if collisions < best_collisions:
            best_collisions = collisions
            best_paths = paths
            if collisions == 0:
                break
    
    return best_paths, time.time() - start_time, best_collisions, policy


# ============================================================
# Hybrid MARL-LNS Algorithm
# ============================================================

def hybrid_marl_lns(env, time_limit=60.0, marl_time_ratio=0.3, marl_episodes=30):
    """
    Hybrid algorithm: MARL for initial collision reduction, then LNS for refinement.
    
    Phase 1 (MARL): Use learned policy to generate initial paths with fewer collisions.
    Phase 2 (LNS): Use LNS with PP repair to eliminate remaining collisions.
    """
    total_start = time.time()
    marl_time = time_limit * marl_time_ratio
    lns_time = time_limit * (1 - marl_time_ratio)
    
    # Phase 1: MARL - generate initial paths with reduced collisions
    marl_paths, marl_elapsed, marl_collisions, policy = train_marl(
        env, num_episodes=marl_episodes, time_limit=marl_time
    )
    
    if marl_paths is None:
        # Fallback: start with individual shortest paths
        marl_paths = []
        for i in range(env.num_agents):
            p = astar(env, env.starts[i], env.goals[i])
            if p is None:
                return None, time.time() - total_start, float('inf'), {'marl_collisions': float('inf'), 'lns_collisions': float('inf')}
            marl_paths.append(p)
        marl_collisions = count_collisions(marl_paths)
    
    if marl_collisions == 0:
        return marl_paths, time.time() - total_start, 0, {'marl_collisions': 0, 'lns_collisions': 0}
    
    # Phase 2: LNS - refine to eliminate remaining collisions
    remaining_time = time_limit - (time.time() - total_start)
    if remaining_time <= 0:
        return marl_paths, time.time() - total_start, marl_collisions, {'marl_collisions': marl_collisions, 'lns_collisions': marl_collisions}
    
    lns_paths, lns_elapsed, lns_collisions = lns(
        env, initial_paths=marl_paths, time_limit=remaining_time, num_destroy=max(2, env.num_agents // 10)
    )
    
    total_time = time.time() - total_start
    
    if lns_paths is not None:
        return lns_paths, total_time, lns_collisions, {'marl_collisions': marl_collisions, 'lns_collisions': lns_collisions}
    else:
        return marl_paths, total_time, marl_collisions, {'marl_collisions': marl_collisions, 'lns_collisions': marl_collisions}


# ============================================================
# Data Loading
# ============================================================

def load_map(map_path):
    """Load a map from .npy file."""
    return np.load(map_path)


def generate_agents(grid, num_agents, seed=42):
    """Generate random start and goal positions on free cells."""
    rng = np.random.RandomState(seed)
    free_cells = list(zip(*np.where(grid == 0)))
    
    if len(free_cells) < 2 * num_agents:
        return None, None
    
    rng.shuffle(free_cells)
    starts = free_cells[:num_agents]
    goals = free_cells[num_agents:2 * num_agents]
    
    # Ensure starts and goals don't overlap
    start_set = set(starts)
    filtered_goals = [g for g in goals if g not in start_set]
    if len(filtered_goals) < num_agents:
        # Regenerate
        return generate_agents(grid, num_agents, seed + 1)
    
    return starts, filtered_goals[:num_agents]


def parse_agent_count(dirname):
    """Extract agent count from directory name like 'empty_maps_453_25_25'."""
    parts = dirname.split('_')
    for p in parts:
        if p.isdigit():
            return int(p)
    return None
