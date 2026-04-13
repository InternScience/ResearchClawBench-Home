#!/usr/bin/env python3
import os, json, math, time, random, heapq, glob
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DIRS = [
    ('random_small','data/random_small/maps_60_10_10_0.175', 60),
    ('random_medium','data/random_medium/maps_312_25_25_0.175', 312),
    ('random_large','data/random_large/maps_1250_50_50_0.175', 1250),
    ('empty','data/empty/empty_maps_453_25_25', 453),
    ('maze','data/maze/maze_maps_125_25_25', 125),
    ('room','data/room/room_maps_250_25_25', 250),
    ('warehouse','data/warehouse/warehouse_maps_266_25_25', 266),
]
MOVES = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]


def neighbors(pos, grid):
    h,w = grid.shape
    x,y = pos
    for dx,dy in MOVES:
        nx,ny = x+dx,y+dy
        if 0 <= nx < h and 0 <= ny < w and grid[nx,ny] != -1:
            yield (nx,ny)


def bfs_dist(grid, start, goal):
    if start == goal:
        return 0
    dq = deque([(start,0)])
    seen = {start}
    while dq:
        v,d = dq.popleft()
        for u in neighbors(v, grid):
            if u == goal:
                return d+1
            if u not in seen:
                seen.add(u)
                dq.append((u,d+1))
    return None


def sample_instance(grid, n_agents, seed):
    rng = random.Random(seed)
    free = [(i,j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j] != -1]
    starts = rng.sample(free, n_agents)
    remain = [p for p in free if p not in starts]
    goals = rng.sample(remain, n_agents)
    return starts, goals


def manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def reconstruct(came, state):
    out = [state]
    while state in came:
        state = came[state]
        out.append(state)
    out.reverse()
    return [s[0] for s in out]


def conflict_cost(path, reservations):
    c = 0
    for t in range(len(path)):
        p = path[t]
        if p in reservations['vertex'].get(t, set()):
            c += 1
        if t > 0 and (path[t-1], path[t]) in reservations['edge'].get(t, set()):
            c += 1
    return c


def build_reservations(paths, exclude=None):
    res = {'vertex':defaultdict(set), 'edge':defaultdict(set), 'goal':[]}
    for idx,p in enumerate(paths):
        if exclude is not None and idx in exclude:
            continue
        for t,loc in enumerate(p):
            res['vertex'][t].add(loc)
            if t > 0:
                res['edge'][t].add((p[t], p[t-1]))
        res['goal'].append((p[-1], len(p)-1))
    return res


def plan_single_soft(grid, start, goal, reservations, max_time=80):
    h = manhattan(start, goal)
    pq = [(0, h, 0, start, 0)]
    came = {}
    best = {(start,0):(0,h)}
    while pq:
        soft, f, g, pos, t = heapq.heappop(pq)
        if pos == goal:
            path = reconstruct(came, (pos,t))
            return path, soft
        if t >= max_time:
            continue
        for nxt in neighbors(pos, grid):
            nt = t + 1
            add = 0
            if nxt in reservations['vertex'].get(nt, set()):
                add += 1
            if (pos, nxt) in reservations['edge'].get(nt, set()):
                add += 1
            for gpos, gt in reservations['goal']:
                if nxt == gpos and nt >= gt:
                    add += 1
                    break
            ng = g + 1
            nh = manhattan(nxt, goal)
            ns = soft + add
            state = (nxt, nt)
            key = (ns, ng + nh)
            if state not in best or key < best[state]:
                best[state] = key
                came[state] = (pos, t)
                heapq.heappush(pq, (ns, ng + nh, ng, nxt, nt))
    return None, 10**9


def detect_conflicts(paths):
    conflicts = []
    T = max(len(p) for p in paths)
    ext = []
    for p in paths:
        ext.append(p + [p[-1]]*(T-len(p)))
    for t in range(T):
        occ = {}
        for i,p in enumerate(ext):
            if p[t] in occ:
                conflicts.append(('vertex', t, occ[p[t]], i, p[t]))
            occ[p[t]] = i
        if t > 0:
            for i in range(len(ext)):
                for j in range(i+1, len(ext)):
                    if ext[i][t-1] == ext[j][t] and ext[j][t-1] == ext[i][t]:
                        conflicts.append(('swap', t, i, j, (ext[i][t], ext[j][t])))
    return conflicts


def prioritized_planning(grid, starts, goals, order=None, max_time=80):
    n = len(starts)
    if order is None:
        order = list(range(n))
    paths = [None]*n
    for idx in order:
        res = build_reservations([p for p in paths if p is not None])
        path, soft = plan_single_soft(grid, starts[idx], goals[idx], res, max_time=max_time)
        if path is None:
            return None
        paths[idx] = path
    return paths


def marl_scores(grid, starts, goals):
    n = len(starts)
    dists = [bfs_dist(grid, s, g) or 999 for s,g in zip(starts, goals)]
    scores = []
    for i in range(n):
        congestion = 0
        for j in range(n):
            if i == j: continue
            if manhattan(starts[i], starts[j]) <= 3:
                congestion += 1
            if manhattan(goals[i], goals[j]) <= 3:
                congestion += 1
        obstacle_pressure = 0
        x,y = starts[i]
        for dx,dy in MOVES[1:]:
            nx,ny = x+dx,y+dy
            if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]) or grid[nx,ny] == -1:
                obstacle_pressure += 1
        score = 0.6*congestion + 0.3*obstacle_pressure + 0.1*dists[i]
        scores.append(score)
    return scores


def hybrid_lns(grid, starts, goals, destroy_frac=0.35, max_iter=25, max_time=80):
    n = len(starts)
    scores = marl_scores(grid, starts, goals)
    order = sorted(range(n), key=lambda i: (-scores[i], manhattan(starts[i], goals[i])))
    paths = prioritized_planning(grid, starts, goals, order=order, max_time=max_time)
    if paths is None:
        return None, {'init_collisions':None, 'final_collisions':None, 'accepted':0, 'iters':0}
    init_conf = len(detect_conflicts(paths))
    accepted = 0
    for _ in range(max_iter):
        conflicts = detect_conflicts(paths)
        if not conflicts:
            break
        hot = set()
        for c in conflicts:
            hot.add(c[2]); hot.add(c[3])
        k = max(2, int(math.ceil(destroy_frac * n)))
        ranked = sorted(range(n), key=lambda i: ((i in hot), scores[i], random.random()), reverse=True)
        subset = set(ranked[:k])
        new_paths = paths[:]
        fixed = [p for idx,p in enumerate(paths) if idx not in subset]
        fixed_res = build_reservations(fixed)
        suborder = sorted(list(subset), key=lambda i: (-(scores[i] + 2*(i in hot)), manhattan(starts[i], goals[i])))
        ok = True
        temp = {}
        for idx in suborder:
            current_fixed = fixed + [temp[j] for j in temp]
            res = build_reservations(current_fixed)
            p, soft = plan_single_soft(grid, starts[idx], goals[idx], res, max_time=max_time)
            if p is None:
                ok = False; break
            temp[idx] = p
        if ok:
            for idx,p in temp.items():
                new_paths[idx] = p
            if len(detect_conflicts(new_paths)) <= len(conflicts):
                paths = new_paths
                accepted += 1
    final_conf = len(detect_conflicts(paths))
    return paths, {'init_collisions':init_conf, 'final_collisions':final_conf, 'accepted':accepted, 'iters':max_iter}


def metrics(paths):
    conflicts = len(detect_conflicts(paths)) if paths is not None else None
    soc = sum(len(p)-1 for p in paths) if paths is not None else None
    makespan = max(len(p)-1 for p in paths) if paths is not None else None
    success = int(paths is not None and conflicts == 0)
    return {'success':success, 'collisions':conflicts, 'soc':soc, 'makespan':makespan}


def run_benchmark():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    rows = []
    for dname, dpath, nag in DIRS:
        files = sorted(glob.glob(os.path.join(dpath, '*.npy')))[:8]
        for file_idx, f in enumerate(files[:6]):
            grid = np.load(f)
            for rep in range(3):
                seed = 1000 + file_idx*10 + rep
                free_count = int((grid != -1).sum())
                starts, goals = sample_instance(grid, min(nag, max(8, min(32, free_count//5))), seed)
                for method in ['pp_random','pp_marl','hybrid_lns']:
                    t0 = time.time()
                    if method == 'pp_random':
                        order = list(range(len(starts)))
                        random.Random(seed).shuffle(order)
                        paths = prioritized_planning(grid, starts, goals, order=order, max_time=max(80, grid.shape[0]+grid.shape[1]))
                        info = {'init_collisions':None,'final_collisions':None,'accepted':0,'iters':0}
                    elif method == 'pp_marl':
                        scores = marl_scores(grid, starts, goals)
                        order = sorted(range(len(starts)), key=lambda i: (-scores[i], manhattan(starts[i], goals[i])))
                        paths = prioritized_planning(grid, starts, goals, order=order, max_time=max(80, grid.shape[0]+grid.shape[1]))
                        info = {'init_collisions':None,'final_collisions':None,'accepted':0,'iters':0}
                    else:
                        paths, info = hybrid_lns(grid, starts, goals, destroy_frac=0.35, max_iter=10, max_time=max(60, grid.shape[0]+grid.shape[1]))
                    runtime = time.time() - t0
                    m = metrics(paths)
                    rows.append({
                        'dataset': dname,
                        'map_file': os.path.basename(f),
                        'map_h': grid.shape[0],
                        'map_w': grid.shape[1],
                        'agents': len(starts),
                        'obstacle_ratio': float((grid==-1).mean()),
                        'method': method,
                        'runtime_sec': runtime,
                        **m,
                        **info,
                    })
    df = pd.DataFrame(rows)
    df.to_csv('outputs/results.csv', index=False)
    summary = df.groupby(['dataset','method']).agg(
        success_rate=('success','mean'),
        mean_collisions=('collisions','mean'),
        mean_soc=('soc','mean'),
        mean_makespan=('makespan','mean'),
        mean_runtime=('runtime_sec','mean')
    ).reset_index()
    summary.to_csv('outputs/summary.csv', index=False)

    sns.set_style('whitegrid')
    plt.figure(figsize=(10,5))
    s1 = summary.pivot(index='dataset', columns='method', values='success_rate').reindex([d[0] for d in DIRS])
    s1.plot(kind='bar', ax=plt.gca())
    plt.ylabel('Success rate')
    plt.title('Success rate by dataset and method')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('report/images/success_rate.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10,5))
    s2 = summary.pivot(index='dataset', columns='method', values='mean_runtime').reindex([d[0] for d in DIRS])
    s2.plot(kind='bar', ax=plt.gca())
    plt.ylabel('Runtime (s)')
    plt.title('Average runtime by dataset and method')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('report/images/runtime.png', dpi=200)
    plt.close()

    succ_df = df[df['success']==1].copy()
    plt.figure(figsize=(10,5))
    sns.boxplot(data=succ_df, x='dataset', y='soc', hue='method', order=[d[0] for d in DIRS])
    plt.ylabel('Sum of costs')
    plt.title('Solution quality on solved instances')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('report/images/soc_boxplot.png', dpi=200)
    plt.close()

    over = df.groupby('method').agg(success_rate=('success','mean'), mean_runtime=('runtime_sec','mean'), mean_soc=('soc','mean')).reset_index()
    over.to_json('outputs/overall.json', orient='records', indent=2)
    print(summary)
    print('\nOVERALL\n', over)

if __name__ == '__main__':
    run_benchmark()
