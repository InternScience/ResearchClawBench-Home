"""
Utility functions for Multi-Agent Path Finding (MAPF)
"""
import numpy as np
import heapq
from typing import List, Tuple, Dict, Set, Optional
from collections import deque
import random


class MAPFMap:
    """Represents a 2D grid map for MAPF."""
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.height, self.width = grid.shape
        self.obstacles = set(zip(*np.where(grid == -1)))
        
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not obstacle)."""
        x, y = pos
        return (0 <= x < self.height and 0 <= y < self.width and 
                pos not in self.obstacles)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)."""
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:  # including stay
            new_pos = (x + dx, y + dy)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def get_free_positions(self) -> List[Tuple[int, int]]:
        """Get all free positions in the map."""
        return list(zip(*np.where(self.grid == 0)))


class MAPFInstance:
    """Represents a MAPF instance with map, starts, and goals."""
    
    def __init__(self, map_obj: MAPFMap, starts: List[Tuple[int, int]], 
                 goals: List[Tuple[int, int]]):
        self.map = map_obj
        self.starts = starts
        self.goals = goals
        self.num_agents = len(starts)
        
    @classmethod
    def generate_random(cls, map_obj: MAPFMap, num_agents: int, seed: Optional[int] = None):
        """Generate random start and goal positions."""
        if seed is not None:
            random.seed(seed)
        
        free_positions = map_obj.get_free_positions()
        if len(free_positions) < 2 * num_agents:
            raise ValueError(f"Not enough free positions for {num_agents} agents")
        
        selected = random.sample(free_positions, 2 * num_agents)
        starts = selected[:num_agents]
        goals = selected[num_agents:]
        
        return cls(map_obj, starts, goals)


class Path:
    """Represents a path for an agent."""
    
    def __init__(self, positions: List[Tuple[int, int]]):
        self.positions = positions
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx]
    
    def get_position(self, time: int) -> Tuple[int, int]:
        """Get position at time t (stays at goal if beyond path length)."""
        if time < len(self.positions):
            return self.positions[time]
        return self.positions[-1]


def detect_collision(path1: Path, path2: Path, agent1: int, agent2: int) -> Optional[Tuple]:
    """
    Detect collisions between two paths.
    Returns (time, type, agent1, agent2, pos) or None if no collision.
    Types: 'vertex' or 'edge' (swapping collision)
    """
    max_len = max(len(path1), len(path2))
    
    for t in range(max_len):
        pos1 = path1.get_position(t)
        pos2 = path2.get_position(t)
        
        # Vertex collision
        if pos1 == pos2:
            return (t, 'vertex', agent1, agent2, pos1)
        
        # Edge (swapping) collision
        if t > 0:
            prev_pos1 = path1.get_position(t - 1)
            prev_pos2 = path2.get_position(t - 1)
            if pos1 == prev_pos2 and pos2 == prev_pos1:
                return (t, 'edge', agent1, agent2, (prev_pos1, pos1))
    
    return None


def count_collisions(solution: List[Path]) -> int:
    """Count total number of collisions in a solution."""
    num_agents = len(solution)
    collisions = 0
    
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if detect_collision(solution[i], solution[j], i, j) is not None:
                collisions += 1
    
    return collisions


def get_colliding_agents(solution: List[Path]) -> Set[int]:
    """Get set of agents involved in collisions."""
    num_agents = len(solution)
    colliding = set()
    
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if detect_collision(solution[i], solution[j], i, j) is not None:
                colliding.add(i)
                colliding.add(j)
    
    return colliding


def astar(map_obj: MAPFMap, start: Tuple[int, int], goal: Tuple[int, int],
          constraints: Optional[Set] = None, other_paths: Optional[List[Path]] = None) -> Optional[Path]:
    """
    A* pathfinding with optional constraints and collision avoidance.
    
    Args:
        map_obj: The map
        start: Start position
        goal: Goal position
        constraints: Set of (position, time) tuples that are forbidden
        other_paths: Other agents' paths to avoid
    """
    if constraints is None:
        constraints = set()
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def is_constrained(pos, time):
        if (pos, time) in constraints:
            return True
        if other_paths is not None:
            for path in other_paths:
                if path.get_position(time) == pos:
                    return True
                # Check edge collision
                if time > 0:
                    prev_pos = path.get_position(time - 1)
                    curr_pos = path.get_position(time)
                    if pos == prev_pos and path.get_position(time - 1) == pos:
                        return True
        return False
    
    # (f_score, g_score, position, path)
    open_set = [(heuristic(start), 0, start, [start])]
    closed_set = set()
    
    while open_set:
        f, g, pos, path = heapq.heappop(open_set)
        
        if pos == goal:
            return Path(path)
        
        state = (pos, g)
        if state in closed_set:
            continue
        closed_set.add(state)
        
        for neighbor in map_obj.get_neighbors(pos):
            if neighbor != pos:  # Skip wait action for now
                new_g = g + 1
                new_path = path + [neighbor]
                
                if not is_constrained(neighbor, new_g):
                    new_f = new_g + heuristic(neighbor)
                    heapq.heappush(open_set, (new_f, new_g, neighbor, new_path))
    
    return None


def space_time_a_star(map_obj: MAPFMap, start: Tuple[int, int], goal: Tuple[int, int],
                     other_paths: Optional[List[Path]] = None, 
                     max_time: int = 100) -> Optional[Path]:
    """
    Space-Time A* that avoids collisions with other agents.
    """
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def has_collision(pos, time):
        if other_paths is None:
            return False
        for path in other_paths:
            # Vertex collision
            if path.get_position(time) == pos:
                return True
            # Edge collision (swapping)
            if time > 0:
                if path.get_position(time - 1) == pos and path.get_position(time) == path.get_position(time - 1):
                    return True
        return False
    
    open_set = [(heuristic(start), 0, start, [start])]
    closed_set = set()
    
    while open_set:
        f, g, pos, path = heapq.heappop(open_set)
        
        if pos == goal and g >= heuristic(start):
            # Stay at goal if needed
            final_path = path
            while len(final_path) < max_time and any(p.get_position(len(final_path)) == goal 
                                                      for p in (other_paths or [])):
                final_path = final_path + [goal]
            return Path(final_path)
        
        state = (pos, g)
        if state in closed_set:
            continue
        closed_set.add(state)
        
        if g >= max_time:
            continue
        
        for neighbor in map_obj.get_neighbors(pos):
            new_g = g + 1
            new_path = path + [neighbor]
            
            if not has_collision(neighbor, new_g):
                new_f = new_g + heuristic(neighbor)
                heapq.heappush(open_set, (new_f, new_g, neighbor, new_path))
    
    return None


def prioritized_planning(instance: MAPFInstance, priority_order: Optional[List[int]] = None) -> List[Path]:
    """
    Prioritized Planning algorithm.
    Plans paths for agents in order of priority, avoiding collisions with higher-priority agents.
    """
    num_agents = instance.num_agents
    
    if priority_order is None:
        priority_order = list(range(num_agents))
    
    solution = []
    
    for agent_id in priority_order:
        start = instance.starts[agent_id]
        goal = instance.goals[agent_id]
        
        path = space_time_a_star(instance.map, start, goal, solution)
        
        if path is None:
            # If no collision-free path found, use A* without collision avoidance
            path = astar(instance.map, start, goal)
            if path is None:
                # If even that fails, create a direct path
                path = Path([start, goal])
        
        solution.append(path)
    
    # Reorder solution back to original agent order
    reordered = [None] * num_agents
    for i, agent_id in enumerate(priority_order):
        reordered[agent_id] = solution[i]
    
    return reordered


def compute_solution_cost(solution: List[Path]) -> int:
    """Compute sum of costs for a solution."""
    return sum(len(p) - 1 for p in solution)


def makespan(solution: List[Path]) -> int:
    """Compute makespan (max path length) for a solution."""
    return max(len(p) for p in solution)
