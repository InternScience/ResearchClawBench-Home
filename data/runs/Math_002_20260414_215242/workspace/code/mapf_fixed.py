import numpy as np
import os
import glob
from collections import namedtuple
from heapq import heappush, heappop
import time
import json

State = namedtuple('State', ['pos', 'time'])

def manhattan(p1, p2):
    return np.sum(np.abs(p1 - p2))

def a_star(grid, start, goal, reservations=None):
    if reservations is None:
        reservations = {}
    h, w = grid.shape
    directions = np.array([[-1,0], [1,0], [0,-1], [0,1], [0,0]])
    open_set = []
    came_from = {}
    g_score = {}
    f_score = {}
    start_state = State(start.copy(), 0)
    g_score[start_state] = 0
    f_score[start_state] = manhattan(start, goal)
    heappush(open_set, (f_score[start_state], np.random.random(), start_state))
    
    visited = set()
    
    while open_set:
        _, _, current = heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        
        if np.array_equal(current.pos, goal):
            path = []
            while current in came_from:
                path.append(current.pos.copy())
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for d in directions:
            nx_pos = current.pos + d
            if not (0 <= nx_pos[0] < h and 0 <= nx_pos[1] < w) or grid[int(nx_pos[0]), int(nx_pos[1])] == -1:
                continue
            nt = current.time + 1
            nstate = State(nx_pos.copy(), nt)
            
            # Check reservation collision
            collide = False
            for agent_paths in reservations.values():
                for path in agent_paths:
                    if nt < len(path) and np.array_equal(path[nt], nx_pos):
                        collide = True
                        break
                if collide:
                    break
            if collide:
                continue
            
            tentative_g = current.time + 1
            if nstate not in g_score or tentative_g < g_score[nstate]:
                came_from[nstate] = current
                g_score[nstate] = tentative_g
                f_score[nstate] = tentative_g + manhattan(nx_pos, goal)
                heappush(open_set, (f_score[nstate], np.random.random(), nstate))
    return None

def prioritized_planning(grid, starts, goals):
    n = len(starts)
    perm = np.random.permutation(n)
    paths = {}
    reservations = {}
    for i in perm:
        path = a_star(grid, starts[i], goals[i], reservations)
        if path is None:
            return None
        paths[i] = path
        reservations[i] = [path]
    return paths

def has_collisions(paths):
    max_t = max(len(p) for p in paths.values())
    for t in range(max_t):
        pos_t = {}
        for i, path in paths.items():
            if t < len(path):
                pos = tuple(path[t])
                if pos in pos_t:
                    return True
                pos_t[pos] = i
        # Edge collisions: swap check
        for i, path in paths.items():
            if t + 1 < len(path):
                e1 = (tuple(path[t]), tuple(path[t+1]))
                for j, pathj in paths.items():
                    if i == j or t + 1 >= len(pathj):
                        continue
                    e2 = (tuple(pathj[t+1]), tuple(pathj[t]))
                    if e1 == e2:
                        return True
    return False

def evaluate(paths, starts, goals):
    if paths is None:
        return False, np.inf, np.inf
    success = not has_collisions(paths)
    makespan = max(len(p) for p in paths.values())
    sum_ic = sum(len(p) - manhattan(starts[i], goals[i]) for i,p in paths.items())
    return success, makespan, sum_ic

def run_pp(dataset_dir, num_maps=5, num_trials=3):
    results = {'success_rates': [], 'mean_makespan': [], 'mean_sum_ic': [], 'mean_runtime': [], 'agent_nums': []}
    subdirs = glob.glob(f'{dataset_dir}/*_maps_*')
    for subdir in subdirs[:num_maps]:
        npy_files = glob.glob(f'{subdir}/*.npy')
        agent_num = int(subdir.split('_maps_')[1].split('_')[0])
        for trial in range(num_trials):
            map_path = np.random.choice(npy_files)
            grid = np.load(map_path)
            free = np.argwhere(grid == 0)
            if len(free) < 2 * agent_num:
                continue
            starts_idx = np.random.choice(len(free), agent_num, replace=False)
            goals_idx = np.random.choice(len(free), agent_num, replace=False)
            starts = free[starts_idx]
            goals = free[goals_idx]
            t0 = time.time()
            paths = prioritized_planning(grid, starts, goals)
            dt = time.time() - t0
            success, makespan, sum_ic = evaluate(paths, starts, goals)
            results['success_rates'].append(success)
            results['mean_makespan'].append(makespan)
            results['mean_sum_ic'].append(sum_ic)
            results['mean_runtime'].append(dt)
            results['agent_nums'].append(agent_num)
    return results

datasets = {
    'small': 'data/random_small',
    'maps60': 'data/maps_60_10_10_0.175',
    'room': 'data/room'
}
all_results = {}
for name, ds in datasets.items():
    all_results[name] = run_pp(ds)

with open('outputs/pp_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print('PP results computed')