"""
MAPF Core Components: Grid, A*, Space-Time A*, Collision Detection
"""
import numpy as np
import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set

class MAPFGrid:
    """2D grid environment for MAPF"""
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.free_cells = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] == 0:
                    self.free_cells.add((r, c))
    
    def is_free(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions (4-connected + wait)"""
        r, c = pos
        neighbors = [(r, c)]  # wait action
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                neighbors.append((nr, nc))
        return neighbors
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar(grid: MAPFGrid, start: Tuple, goal: Tuple) -> Optional[List[Tuple]]:
    """Standard A* pathfinding on the grid"""
    if start == goal:
        return [start]
    
    open_list = [(grid.manhattan_distance(start, goal), 0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    
    while open_list:
        f, g, current = heapq.heappop(open_list)
        
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        if g > g_score.get(current, float('inf')):
            continue
        
        for neighbor in grid.get_neighbors(current):
            if neighbor == current:
                continue  # skip wait for basic A*
            new_g = g + 1
            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                f = new_g + grid.manhattan_distance(neighbor, goal)
                heapq.heappush(open_list, (f, new_g, neighbor))
    
    return None  # No path found


def spacetime_astar(grid: MAPFGrid, start: Tuple, goal: Tuple, 
                     constraints: Dict, max_time: int = 200) -> Optional[List[Tuple]]:
    """
    Space-Time A* with hard and soft constraints.
    constraints: dict with:
      'vertex': set of (pos, t) - vertex constraints (hard)
      'edge': set of (pos1, pos2, t) - edge constraints (hard)
      'soft_vertex': set of (pos, t) - soft vertex constraints (minimize)
      'soft_edge': set of (pos1, pos2, t) - soft edge constraints (minimize)
    Returns path as list of positions (index = timestep)
    """
    vertex_constraints = constraints.get('vertex', set())
    edge_constraints = constraints.get('edge', set())
    soft_vertex = constraints.get('soft_vertex', set())
    soft_edge = constraints.get('soft_edge', set())
    
    # State: (pos, timestep)
    # Priority: (collisions, f_value, g_value, counter)
    counter = 0
    start_state = (start, 0)
    h = grid.manhattan_distance(start, goal)
    
    # (collisions, f, g, counter, state)
    open_list = [(0, h, 0, counter, start_state)]
    came_from = {start_state: None}
    g_score = {start_state: 0}
    c_score = {start_state: 0}  # collision count
    
    while open_list:
        c, f, g, _, current_state = heapq.heappop(open_list)
        pos, t = current_state
        
        if pos == goal and t >= max_time // 4:
            # Check no future constraints at goal
            has_future = False
            for ft in range(t, min(t + 20, max_time)):
                if (goal, ft) in vertex_constraints:
                    has_future = True
                    break
            if not has_future:
                path = []
                state = current_state
                while state is not None:
                    path.append(state[0])
                    state = came_from[state]
                return path[::-1]
        
        if t >= max_time:
            continue
        
        if g > g_score.get(current_state, float('inf')):
            continue
        
        for next_pos in grid.get_neighbors(pos):
            next_t = t + 1
            next_state = (next_pos, next_t)
            
            # Check hard constraints
            if (next_pos, next_t) in vertex_constraints:
                continue
            if (pos, next_pos, t) in edge_constraints:
                continue
            # Check swapping constraint
            if (next_pos, pos, t) in edge_constraints:
                continue
            
            new_g = g + 1
            # Count soft collisions
            new_c = c
            if (next_pos, next_t) in soft_vertex:
                new_c += 1
            if (pos, next_pos, t) in soft_edge:
                new_c += 1
            
            if new_g < g_score.get(next_state, float('inf')) or \
               (new_g == g_score.get(next_state, float('inf')) and new_c < c_score.get(next_state, float('inf'))):
                g_score[next_state] = new_g
                c_score[next_state] = new_c
                came_from[next_state] = current_state
                h = grid.manhattan_distance(next_pos, goal)
                counter += 1
                heapq.heappush(open_list, (new_c, new_g + h, new_g, counter, next_state))
    
    # If no path found, return shortest path ignoring time constraints
    return astar(grid, start, goal)


def detect_collisions(paths: List[List[Tuple]]) -> List[Dict]:
    """
    Detect vertex and edge (swapping) collisions between paths.
    Returns list of collision dicts with agents involved, type, location, timestep.
    """
    collisions = []
    n = len(paths)
    if n == 0:
        return collisions
    
    max_t = max(len(p) for p in paths)
    
    def get_pos(path, t):
        if t < len(path):
            return path[t]
        return path[-1]  # agent stays at goal
    
    for i in range(n):
        for j in range(i + 1, n):
            for t in range(max_t):
                pos_i = get_pos(paths[i], t)
                pos_j = get_pos(paths[j], t)
                
                # Vertex collision
                if pos_i == pos_j:
                    collisions.append({
                        'agents': (i, j),
                        'type': 'vertex',
                        'location': pos_i,
                        'timestep': t
                    })
                
                # Edge/swapping collision
                if t > 0:
                    prev_i = get_pos(paths[i], t - 1)
                    prev_j = get_pos(paths[j], t - 1)
                    if pos_i == prev_j and pos_j == prev_i:
                        collisions.append({
                            'agents': (i, j),
                            'type': 'edge',
                            'location': (prev_i, pos_i),
                            'timestep': t
                        })
    
    return collisions


def count_collisions(paths: List[List[Tuple]]) -> int:
    """Count total number of collisions"""
    return len(detect_collisions(paths))


def get_colliding_agents(paths: List[List[Tuple]]) -> Set[int]:
    """Get set of agents involved in collisions"""
    collisions = detect_collisions(paths)
    agents = set()
    for c in collisions:
        agents.add(c['agents'][0])
        agents.add(c['agents'][1])
    return agents


def generate_agent_tasks(grid: MAPFGrid, num_agents: int, rng=None) -> List[Tuple]:
    """Generate random start-goal pairs for agents on free cells"""
    if rng is None:
        rng = np.random.RandomState(42)
    
    free_cells = list(grid.free_cells)
    if len(free_cells) < 2 * num_agents:
        num_agents = len(free_cells) // 2
    
    selected = rng.choice(len(free_cells), size=2 * num_agents, replace=False)
    tasks = []
    for i in range(num_agents):
        start = free_cells[selected[2 * i]]
        goal = free_cells[selected[2 * i + 1]]
        tasks.append((start, goal))
    
    return tasks
