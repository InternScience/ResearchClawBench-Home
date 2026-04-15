"""
Core MAPF utilities: map loading, agent generation, collision detection, A* search.
"""

import numpy as np
import heapq
import os
import glob
from typing import List, Tuple, Dict, Optional, Set


class MapLoader:
    """Load and manage MAPF grid maps."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.map_cache = {}
    
    def list_datasets(self) -> List[str]:
        """List available dataset directories."""
        datasets = []
        for name in sorted(os.listdir(self.data_dir)):
            path = os.path.join(self.data_dir, name)
            if os.path.isdir(path):
                datasets.append(name)
        return datasets
    
    def list_maps(self, dataset: str) -> List[str]:
        """List all .npy map files in a dataset."""
        pattern = os.path.join(self.data_dir, dataset, "**", "*.npy")
        return sorted(glob.glob(pattern, recursive=True))
    
    def load_map(self, filepath: str) -> np.ndarray:
        """Load a single map file. Returns 2D array (0=free, -1=obstacle)."""
        if filepath not in self.map_cache:
            m = np.load(filepath)
            self.map_cache[filepath] = m
        return self.map_cache[filepath].copy()
    
    def get_free_cells(self, map_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Get list of free cell coordinates."""
        return list(zip(*np.where(map_grid == 0)))


def generate_agents(map_grid: np.ndarray, num_agents: int, seed: int = 42) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Generate random start-goal pairs for agents on a map.
    Returns list of ((start_row, start_col), (goal_row, goal_col)).
    """
    rng = np.random.RandomState(seed)
    free_cells = list(zip(*np.where(map_grid == 0)))
    
    if len(free_cells) < 2 * num_agents:
        num_agents = len(free_cells) // 2
    
    # Sample distinct starts and goals
    indices = rng.choice(len(free_cells), size=2*num_agents, replace=False)
    starts = [free_cells[i] for i in indices[:num_agents]]
    goals = [free_cells[i] for i in indices[num_agents:2*num_agents]]
    
    # Ensure starts != goals for each agent
    for i in range(num_agents):
        if starts[i] == goals[i]:
            # Swap with another goal
            j = (i + 1) % num_agents
            goals[i], goals[j] = goals[j], goals[i]
    
    return [(starts[i], goals[i]) for i in range(num_agents)]


class CollisionChecker:
    """Detect vertex and swap collisions between agent paths."""
    
    @staticmethod
    def check_vertex_collision(paths: List[List[Tuple[int,int]]], t: int) -> Optional[Tuple[int, int]]:
        """Check for vertex collision at time t. Returns (agent_i, agent_j) or None."""
        positions = {}
        for i, path in enumerate(paths):
            if t < len(path):
                pos = path[t]
            else:
                pos = path[-1]  # Agent stays at goal
            if pos in positions:
                return (positions[pos], i)
            positions[pos] = i
        return None
    
    @staticmethod
    def check_swap_collision(paths: List[List[Tuple[int,int]]], t: int) -> Optional[Tuple[int, int]]:
        """Check for swap collision between t and t+1."""
        positions_t = {}
        for i, path in enumerate(paths):
            if t < len(path):
                positions_t[path[t]] = i
        
        for i, path in enumerate(paths):
            if t + 1 < len(path):
                next_pos = path[t + 1]
            else:
                next_pos = path[-1]
            
            if t < len(path):
                curr_pos = path[t]
            else:
                curr_pos = path[-1]
            
            # Check if agent i moves to where agent j was, and j moves to where i was
            if curr_pos in positions_t:
                j = positions_t[curr_pos]
                if i != j:
                    # Check if j moves to curr_pos's old position
                    if t + 1 < len(paths[j]):
                        j_next = paths[j][t + 1]
                    else:
                        j_next = paths[j][-1]
                    
                    if j_next == curr_pos and next_pos == (paths[j][t] if t < len(paths[j]) else paths[j][-1]):
                        return (min(i, j), max(i, j))
        return None
    
    @staticmethod
    def count_collisions(paths: List[List[Tuple[int,int]]]) -> int:
        """Count total number of collisions across all timesteps."""
        if not paths or not any(paths):
            return 0
        
        max_len = max(len(p) for p in paths)
        collisions = set()
        
        for t in range(max_len):
            # Vertex collisions
            vc = CollisionChecker.check_vertex_collision(paths, t)
            if vc:
                collisions.add((vc[0], vc[1], t, 'vertex'))
            
            # Swap collisions
            if t < max_len - 1:
                sc = CollisionChecker.check_swap_collision(paths, t)
                if sc:
                    collisions.add((sc[0], sc[1], t, 'swap'))
        
        return len(collisions)
    
    @staticmethod
    def get_colliding_agents(paths: List[List[Tuple[int,int]]]) -> Set[int]:
        """Get set of agent indices involved in any collision."""
        colliding = set()
        if not paths or not any(paths):
            return colliding
        
        max_len = max(len(p) for p in paths)
        
        for t in range(max_len):
            vc = CollisionChecker.check_vertex_collision(paths, t)
            if vc:
                colliding.add(vc[0])
                colliding.add(vc[1])
            
            if t < max_len - 1:
                sc = CollisionChecker.check_swap_collision(paths, t)
                if sc:
                    colliding.add(sc[0])
                    colliding.add(sc[1])
        
        return colliding


class AStarSearch:
    """A* pathfinding for single agent on grid with obstacles."""
    
    # 4-connected grid + wait action
    MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
    
    @staticmethod
    def heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        """Manhattan distance."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    @staticmethod
    def find_path(
        map_grid: np.ndarray,
        start: Tuple[int,int],
        goal: Tuple[int,int],
        constraints: Optional[Set[Tuple[int,int,int]]] = None,
        max_time: int = 200
    ) -> Optional[List[Tuple[int,int]]]:
        """
        Find shortest path from start to goal avoiding obstacles and constraints.
        constraints: set of (row, col, time) tuples that must be avoided.
        Returns path as list of (row, col) or None if no path found.
        """
        rows, cols = map_grid.shape
        constraints = constraints or set()
        
        # (f, g, row, col, path)
        start_h = AStarSearch.heuristic(start, goal)
        open_list = [(start_h, 0, start[0], start[1])]
        closed = set()
        came_from = {}  # (r, c, t) -> (prev_r, prev_c, prev_t)
        
        # Track best g-score for state (r, c, t)
        best_g = {}
        best_g[(start[0], start[1], 0)] = 0
        
        while open_list:
            f, g, r, c = heapq.heappop(open_list)
            t = g  # time = cost for unit-cost grid
            
            state = (r, c, t)
            if state in closed:
                continue
            closed.add(state)
            
            if (r, c) == goal:
                # Reconstruct path
                path = [(r, c)]
                curr = state
                while curr in came_from:
                    prev = came_from[curr]
                    path.append((prev[0], prev[1]))
                    curr = prev
                path.reverse()
                return path
            
            if t >= max_time:
                continue
            
            for dr, dc in AStarSearch.MOVES:
                nr, nc = r + dr, c + dc
                nt = t + 1
                
                if 0 <= nr < rows and 0 <= nc < cols and map_grid[nr, nc] != -1:
                    if (nr, nc, nt) not in constraints:
                        new_g = g + 1
                        new_state = (nr, nc, nt)
                        
                        if new_state not in best_g or new_g < best_g[new_state]:
                            best_g[new_state] = new_g
                            h = AStarSearch.heuristic((nr, nc), goal)
                            heapq.heappush(open_list, (new_g + h, new_g, nr, nc))
                            came_from[new_state] = state
        
        return None
    
    @staticmethod
    def find_path_with_existing(
        map_grid: np.ndarray,
        start: Tuple[int,int],
        goal: Tuple[int,int],
        existing_paths: List[Optional[List[Tuple[int,int]]]],
        exclude_agent: int,
        max_time: int = 200
    ) -> Optional[List[Tuple[int,int]]]:
        """
        Find path avoiding collisions with existing agent paths.
        """
        constraints = set()
        for i, path in enumerate(existing_paths):
            if i == exclude_agent or path is None:
                continue
            for t, pos in enumerate(path):
                constraints.add((pos[0], pos[1], t))
                # Also prevent swap: don't go to pos at t if agent i was there at t-1
                if t > 0:
                    prev_pos = path[t-1]
                    constraints.add((pos[0], pos[1], t))  # vertex constraint already covers this
        
        return AStarSearch.find_path(map_grid, start, goal, constraints, max_time)


def compute_shortest_path_lengths(map_grid: np.ndarray, agents: List[Tuple[Tuple[int,int], Tuple[int,int]]]) -> List[int]:
    """Compute shortest path lengths for each agent ignoring other agents."""
    lengths = []
    for (start, goal) in agents:
        path = AStarSearch.find_path(map_grid, start, goal, max_time=500)
        if path:
            lengths.append(len(path) - 1)
        else:
            lengths.append(-1)
    return lengths
