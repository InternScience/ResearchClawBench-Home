import numpy as np
from collections import namedtuple
import networkx as nx
from heapq import heappush, heappop

State = namedtuple('State', ['pos', 'time'])

def load_map(map_path):
    grid = np.load(map_path)
    h, w = grid.shape
    obs = (grid == -1)
    return obs, h, w

def free_cells(obs):
    return np.argwhere(~obs)

def generate_agents(obs, num_agents):
    free = free_cells(obs)
    if len(free) < 2 * num_agents:
        return None, None
    starts = free[np.random.choice(len(free), num_agents, replace=False)]
    goals = free[np.random.choice(len(free), num_agents, replace=False)]
    while np.any(np.all(starts == goals, axis=1)):
        goals = free[np.random.choice(len(free), num_agents, replace=False)]
    return starts, goals

def a_star(grid, start, goal, obstacles_paths=None):
    \"\"\"A* for single agent avoiding static grid and dynamic obstacles_paths dict of lists of (pos,time)\"\"\"
    h, w = grid.shape
    directions = [(-1,0), (1,0), (0,-1), (0,1), (0,0)]  # include wait
    open_set = []
    came_from = {}
    g_score = {}
    f_score = {}
    start_state = State(start, 0)
    g_score[start_state] = 0
    f_score[start_state] = manhattan(start, goal)
    heappush(open_set, (f_score[start_state], start_state))
    
    while open_set:
        _, current = heappop(open_set)
        if np.array_equal(current.pos, goal):
            path = reconstruct_path(came_from, current)
            return path
        
        for dx, dy in directions:
            nx_pos = current.pos + np.array([dx, dy])
            if not (0 <= nx_pos[0] < h and 0 <= nx_pos[1] < w) or grid[tuple(nx_pos)]:
                continue
            nt = current.time + 1
            nstate = State(nx_pos, nt)
            
            # Check dynamic collision with other paths
            collide = False
            if obstacles_paths:
                for agent_path in obstacles_paths.values():
                    if len(agent_path) > nt and np.array_equal(agent_path[nt], nx_pos):
                        collide = True
                        break
            if collide:
                continue
                
            tentative_g = current.time + 1
            if nstate not in g_score or tentative_g < g_score[nstate]:
                came_from[nstate] = current
                g_score[nstate] = tentative_g
                f_score[nstate] = tentative_g + manhattan(nx_pos, goal)
                heappush(open_set, (f_score[nstate], nstate))
    return None

def manhattan(p1, p2):
    return np.sum(np.abs(p1 - p2))

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current.pos)
        current = came_from[current]
    path.append(current.pos)
    return path[::-1]

def prioritized_planning(grid, starts, goals):
    \"\"\"PP: random order, plan sequentially avoiding previous.\"\"\"
    n = len(starts)
    perm = np.random.permutation(n)
    paths = {}
    for i in perm:
        obstacles_paths = {j: paths[j] for j in paths}
        path = a_star(grid, starts[i], goals[i], obstacles_paths)
        if path is None:
            return None
        paths[i] = path
    return paths

def has_collision(paths):
    \"\"\"Check vertex and edge collisions.\"\"\"
    # Simple check for vertex collision
    for t in range(1000):  # max time
        positions = {}
        for i, path in paths.items():
            if t < len(path):
                pos = tuple(path[t])
                if pos in positions:
                    return True
                positions[pos] = i
    return False

def evaluate_solution(paths):
    if paths is None:
        return False, np.inf, np.inf
    success = not has_collision(paths)
    makespan = max(len(p) for p in paths.values())
    sum_ic = sum(len(p) - manhattan(starts[i], goals[i]) for i,p in paths.items())
    return success, makespan, sum_ic