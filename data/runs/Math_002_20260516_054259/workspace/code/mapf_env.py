"""
MAPF Environment: Grid world with obstacles, agents, and path planning.

Provides:
- Map loading from .npy files
- Instance generation (random start/goal positions)
- Collision detection
- Path planning utilities (A*, space-time A*)
"""

import numpy as np
from collections import deque
import heapq
import os

# Movement directions: 0=wait, 1=up, 2=down, 3=left, 4=right
DIRS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ['wait', 'up', 'down', 'left', 'right']


class MAPFInstance:
    """A single MAPF problem instance."""
    
    def __init__(self, grid, starts, goals, name=""):
        """
        Args:
            grid: 2D numpy array, 0=free, -1=obstacle
            starts: list of (r, c) tuples
            goals: list of (r, c) tuples
            name: instance identifier
        """
        self.grid = grid.astype(np.int32)
        self.H, self.W = grid.shape
        self.starts = [tuple(s) for s in starts]
        self.goals = [tuple(g) for g in goals]
        self.n_agents = len(starts)
        self.name = name
        
        # Precompute free cells
        self.free_cells = [(r, c) for r in range(self.H) for c in range(self.W) if grid[r, c] == 0]
        
    def is_free(self, pos):
        """Check if a position is within bounds and not an obstacle."""
        r, c = pos
        if r < 0 or r >= self.H or c < 0 or c >= self.W:
            return False
        return self.grid[r, c] == 0
    
    def get_neighbors(self, pos):
        """Get valid neighbor positions (including wait)."""
        r, c = pos
        neighbors = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if self.is_free((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors


class MAPFInstanceGenerator:
    """Generate MAPF instances from grid maps."""
    
    @staticmethod
    def generate_from_grid(grid, n_agents, seed=None, ensure_solvable=False):
        """Generate a MAPF instance with random starts and goals.
        
        Args:
            grid: 2D numpy array
            n_agents: number of agents
            seed: random seed
            ensure_solvable: if True, verify solvability (slow)
        
        Returns:
            MAPFInstance
        """
        if seed is not None:
            np.random.seed(seed)
        
        free_cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r, c] == 0]
        
        if len(free_cells) < 2 * n_agents:
            n_agents = len(free_cells) // 2
            if n_agents < 1:
                n_agents = 1
        
        # Sample positions
        indices = np.random.choice(len(free_cells), size=2 * n_agents, replace=False)
        starts = [free_cells[i] for i in indices[:n_agents]]
        goals = [free_cells[i] for i in indices[n_agents:]]
        
        return MAPFInstance(grid, starts, goals)


def load_map_dataset(data_dir, dataset_name, agent_counts=None, max_instances_per_config=20):
    """Load MAPF instances from the data directory.
    
    Args:
        data_dir: path to data directory
        dataset_name: one of 'empty', 'maze', 'random_large', 'random_medium', 
                      'random_small', 'room', 'warehouse', 'maps_60_10_10_0.175'
        agent_counts: list of agent counts to use. If None, inferred from directory names.
        max_instances_per_config: max instances per (map_type, agent_count) combination
    
    Returns:
        list of MAPFInstance
    """
    instances = []
    base = os.path.join(data_dir, dataset_name)
    
    if not os.path.isdir(base):
        return instances
    
    subdirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    
    # If no subdirs, this is a flat directory (maps_60_10_10_0.175)
    if not subdirs:
        files = sorted([f for f in os.listdir(base) if f.endswith('.npy')])
        # Parse: maps_60_10_10_0.175 -> 60 maps, 10x10, 0.175 density
        # Agent count not directly specified; infer from naming
        parts = dataset_name.split('_')
        if len(parts) >= 2:
            n_agents = int(parts[1])  # Heuristic: second number is agent count
        else:
            n_agents = 10
        
        for i, f in enumerate(files[:max_instances_per_config]):
            grid = np.load(os.path.join(base, f), allow_pickle=True)
            inst = MAPFInstanceGenerator.generate_from_grid(grid, n_agents, seed=i)
            inst.name = f"{dataset_name}/{f}"
            instances.append(inst)
    else:
        for sd in subdirs:
            # Parse agent count from directory name
            # e.g., maps_312_25_25_0.175 -> 312 agents? Or 312 maps?
            # Actually, looking at the pattern: maps_{agent_count}_{H}_{W}_{density}
            # But 453 agents on 25x25 empty grid is possible
            parts = sd.split('_')
            if len(parts) >= 2:
                n_agents = int(parts[1])
            else:
                n_agents = 50
            
            sd_path = os.path.join(base, sd)
            files = sorted([f for f in os.listdir(sd_path) if f.endswith('.npy')])
            
            for i, f in enumerate(files[:max_instances_per_config]):
                grid = np.load(os.path.join(sd_path, f), allow_pickle=True)
                inst = MAPFInstanceGenerator.generate_from_grid(grid, n_agents, seed=i)
                inst.name = f"{dataset_name}/{sd}/{f}"
                instances.append(inst)
    
    return instances


def detect_collisions(paths, max_t=None):
    """Detect vertex and edge collisions in a set of paths.
    
    Args:
        paths: list of paths, each path is list of (r, c) tuples
        max_t: max timestep to check
    
    Returns:
        list of collision tuples (agent_i, agent_j, timestep, type)
    """
    collisions = []
    n = len(paths)
    if max_t is None:
        max_t = max(len(p) for p in paths)
    
    # Vertex collisions
    for t in range(max_t):
        pos_to_agents = {}
        for i, path in enumerate(paths):
            if t < len(path):
                pos = path[t]
            else:
                pos = path[-1]  # Agent stays at goal
            if pos not in pos_to_agents:
                pos_to_agents[pos] = []
            pos_to_agents[pos].append(i)
        
        for pos, agents in pos_to_agents.items():
            if len(agents) > 1:
                for a in range(len(agents)):
                    for b in range(a + 1, len(agents)):
                        collisions.append((agents[a], agents[b], t, 'vertex'))
    
    # Edge (swapping) collisions
    for t in range(1, max_t):
        for i in range(n):
            for j in range(i + 1, n):
                pos_i_prev = paths[i][t-1] if t-1 < len(paths[i]) else paths[i][-1]
                pos_i_curr = paths[i][t] if t < len(paths[i]) else paths[i][-1]
                pos_j_prev = paths[j][t-1] if t-1 < len(paths[j]) else paths[j][-1]
                pos_j_curr = paths[j][t] if t < len(paths[j]) else paths[j][-1]
                
                if pos_i_prev == pos_j_curr and pos_i_curr == pos_j_prev:
                    collisions.append((i, j, t, 'edge'))
    
    return collisions


def count_colliding_pairs(paths):
    """Count number of colliding agent pairs."""
    collisions = detect_collisions(paths)
    pairs = set()
    for c in collisions:
        pairs.add((min(c[0], c[1]), max(c[0], c[1])))
    return len(pairs)


def compute_path_cost(path):
    """Compute the cost (length) of a path."""
    return len(path) - 1  # number of moves


def manhattan_distance(pos1, pos2):
    """Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def a_star(grid, start, goal, constraints=None):
    """Standard A* single-agent pathfinding.
    
    Args:
        grid: 2D numpy array
        start: (r, c) tuple
        goal: (r, c) tuple
        constraints: set of (pos, timestep) that are forbidden
    
    Returns:
        path as list of (r, c) tuples, or None if no path found
    """
    H, W = grid.shape
    if constraints is None:
        constraints = set()
    
    open_set = []
    heapq.heappush(open_set, (manhattan_distance(start, goal), 0, start, 0))
    came_from = {}
    g_score = {start: 0}
    
    # Limit search
    max_steps = H * W * 5
    steps = 0
    
    while open_set and steps < max_steps:
        steps += 1
        _, _, current, _ = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        r, c = current
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if grid[nr, nc] == -1:
                continue
            
            tentative_g = g_score[current] + 1
            
            # Check constraints
            t = tentative_g
            if (neighbor, t) in constraints:
                continue
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan_distance(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_set, (f, tentative_g, neighbor, id(neighbor)))
    
    return None


def space_time_a_star(grid, start, goal, dynamic_obstacles, max_t=500):
    """Space-time A* that avoids dynamic obstacles.
    
    Args:
        grid: 2D numpy array
        start: (r, c) tuple
        goal: (r, c) tuple
        dynamic_obstacles: dict mapping (pos, timestep) -> bool or list of occupied positions over time
        max_t: maximum timestep
    
    Returns:
        path as list of (r, c) tuples, or None
    """
    from collections import defaultdict
    
    H, W = grid.shape
    start_state = (start, 0)
    open_set = []
    heapq.heappush(open_set, (manhattan_distance(start, goal), 0, start_state))
    came_from = {}
    g_score = {start_state: 0}
    max_steps = H * W * 20
    
    steps = 0
    while open_set and steps < max_steps:
        steps += 1
        _, _, current_state = heapq.heappop(open_set)
        current, t = current_state
        
        if current == goal:
            # Reconstruct path
            path = [current]
            ct = t
            cc = current
            while (cc, ct) in came_from:
                cc, ct = came_from[(cc, ct)]
                path.append(cc)
            path.reverse()
            return path
        
        if t >= max_t:
            continue
        
        r, c = current
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            nt = t + 1
            
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if grid[nr, nc] == -1:
                continue
            
            # Check dynamic obstacles
            if (nr, nc, nt) in dynamic_obstacles:
                continue
            # Check edge collision
            if (current, (nr, nc), nt) in dynamic_obstacles:
                continue
            
            state = ((nr, nc), nt)
            tentative_g = g_score[(current, t)] + 1
            
            if state not in g_score or tentative_g < g_score[state]:
                g_score[state] = tentative_g
                f = tentative_g + manhattan_distance((nr, nc), goal)
                came_from[state] = (current, t)
                heapq.heappush(open_set, (f, tentative_g, state))
    
    return None


def paths_to_obstacles(paths):
    """Convert a set of paths to dynamic obstacle constraints.
    
    Returns set of (r, c, t) tuples representing occupied space-time positions.
    """
    obstacles = set()
    for path in paths:
        for t, pos in enumerate(path):
            obstacles.add((pos[0], pos[1], t))
        # Agent stays at goal
        if len(path) > 0:
            goal = path[-1]
            for t in range(len(path), 500):
                obstacles.add((goal[0], goal[1], t))
    return obstacles
