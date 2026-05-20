"""
MAPF Environment and Core Utilities
Multi-Agent Path Finding environment with grid maps, agent management,
and collision detection.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import random
import heapq
from collections import defaultdict


class MAPFInstance:
    """Represents a MAPF problem instance."""
    
    def __init__(self, grid: np.ndarray, starts: List[Tuple[int, int]], 
                 goals: List[Tuple[int, int]], agent_count: int):
        self.grid = grid  # 2D array: 0=free, -1=obstacle
        self.starts = starts  # List of (row, col) start positions
        self.goals = goals    # List of (row, col) goal positions
        self.agent_count = agent_count
        self.height, self.width = grid.shape
        
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not an obstacle."""
        r, c = pos
        return (0 <= r < self.height and 0 <= c < self.width and 
                self.grid[r, c] != -1)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)."""
        r, c = pos
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if self.is_valid_position((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors


class Path:
    """Represents a path for a single agent."""
    
    def __init__(self, positions: List[Tuple[int, int]]):
        self.positions = positions  # List of (row, col) at each timestep
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx]
    
    def __repr__(self):
        return f"Path(length={len(self.positions)}, start={self.positions[0]}, goal={self.positions[-1]})"


class Solution:
    """Represents a complete MAPF solution."""
    
    def __init__(self, paths: List[Path], instance: MAPFInstance):
        self.paths = paths
        self.instance = instance
        self.agent_count = len(paths)
        
    def sum_of_costs(self) -> int:
        """Calculate sum of path lengths."""
        return sum(len(p) for p in self.paths)
    
    def count_collisions(self) -> int:
        """Count total vertex and swap collisions."""
        return count_all_collisions(self.paths, self.instance)
    
def count_all_collisions(paths: List[Path], instance: MAPFInstance) -> int:
    """Count all vertex and swap collisions in a set of paths."""
    collisions = 0
    max_time = max(len(p) for p in paths) if paths else 0
    
    for t in range(max_time):
        # Get positions at time t
        positions_t = {}
        for i, path in enumerate(paths):
            if t < len(path):
                pos = path[t]
                if pos not in positions_t:
                    positions_t[pos] = []
                positions_t[pos].append(i)
        
        # Vertex collisions
        for pos, agents in positions_t.items():
            if len(agents) > 1:
                collisions += len(agents) * (len(agents) - 1) // 2
        
        # Swap collisions
        if t + 1 < max_time:
            for i, path_i in enumerate(paths):
                if t + 1 < len(path_i):
                    pos_i_t = path_i[t]
                    pos_i_next = path_i[t + 1]
                    for j, path_j in enumerate(paths):
                        if i < j and t + 1 < len(path_j):
                            pos_j_t = path_j[t]
                            pos_j_next = path_j[t + 1]
                            # Swap: i goes to j's position, j goes to i's position
                            if pos_i_t == pos_j_next and pos_j_t == pos_i_next:
                                collisions += 1
    
    return collisions


def generate_random_instance(grid: np.ndarray, agent_count: int, 
                            seed: Optional[int] = None) -> MAPFInstance:
    """Generate a random MAPF instance on a given grid."""
    if seed is not None:
        np.random.seed(seed)
    
    free_cells = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 0:
                free_cells.append((r, c))
    
    if len(free_cells) < 2 * agent_count:
        raise ValueError(f"Not enough free cells for {agent_count} agents")
    
    # Sample start and goal positions
    indices = np.random.choice(len(free_cells), size=2 * agent_count, replace=False)
    starts = [free_cells[i] for i in indices[:agent_count]]
    goals = [free_cells[i] for i in indices[agent_count:]]
    
    return MAPFInstance(grid, starts, goals, agent_count)


class AStarPlanner:
    """A* pathfinding for single agents."""
    
    def __init__(self, instance: MAPFInstance):
        self.instance = instance
        
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int],
             constraints: Set[Tuple[int, Tuple[int, int]]] = None,
             time_limit: int = 1000) -> Optional[Path]:
        """Find shortest path from start to goal avoiding constraints."""
        if constraints is None:
            constraints = set()
        
        # Priority queue: (f_cost, g_cost, time, position, path)
        open_list = [(self._heuristic(start, goal), 0, 0, start, [start])]
        visited = set()
        
        while open_list:
            f, g, t, pos, path = heapq.heappop(open_list)
            
            if pos == goal:
                return Path(path)
            
            if t >= time_limit:
                continue
                
            state = (t, pos)
            if state in visited:
                continue
            visited.add(state)
            
            # Expand to neighbors (including wait action)
            neighbors = self.instance.get_neighbors(pos) + [pos]
            
            for next_pos in neighbors:
                next_t = t + 1
                next_state = (next_t, next_pos)
                
                if next_state in visited:
                    continue
                if (next_t, next_pos) in constraints:
                    continue
                # Also check swap constraints: (t, (pos, next_pos))
                if (t, (pos, next_pos)) in constraints:
                    continue
                    
                new_g = g + 1
                new_f = new_g + self._heuristic(next_pos, goal)
                new_path = path + [next_pos]
                
                heapq.heappush(open_list, (new_f, new_g, next_t, next_pos, new_path))
        
        return None  # No path found
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


class PrioritizedPlanner:
    """Prioritized Planning algorithm for MAPF."""
    
    def __init__(self, instance: MAPFInstance):
        self.instance = instance
        self.planner = AStarPlanner(instance)
        
    def plan(self, priority_order: Optional[List[int]] = None, 
             time_limit: int = 1000) -> Optional[List[Path]]:
        """Plan paths according to priority order."""
        if priority_order is None:
            priority_order = list(range(self.instance.agent_count))
        
        paths = []
        constraints = set()  # (timestep, position) constraints
        
        for agent_id in priority_order:
            start = self.instance.starts[agent_id]
            goal = self.instance.goals[agent_id]
            
            # Add constraints from already-planned paths
            agent_constraints = set()
            for i, existing_path in enumerate(paths):
                if i < len(priority_order):
                    for t, pos in enumerate(existing_path):
                        agent_constraints.add((t, pos))
                        if t + 1 < len(existing_path):
                            # Swap constraint
                            next_pos = existing_path[t + 1]
                            agent_constraints.add((t, (pos, next_pos)))
            
            path = self.planner.plan(start, goal, 
                                     {(t, pos) for t, pos in [(t, p) for path in paths for t, p in enumerate(path)]},
                                     time_limit)
            
            if path is None:
                return None  # Failed to find path for this agent
            
            paths.append(path)
        
        # Reorder paths according to original agent order
        reordered_paths = [None] * self.instance.agent_count
        for i, agent_id in enumerate(priority_order):
            reordered_paths[agent_id] = paths[i]
        
        return reordered_paths


def load_map(file_path: str) -> np.ndarray:
    """Load a map from .npy file."""
    return np.load(file_path, allow_pickle=True)
