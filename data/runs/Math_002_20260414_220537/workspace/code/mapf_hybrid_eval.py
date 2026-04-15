import json, time, math, heapq, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
random.seed(0)
np.random.seed(0)

DIRS = {
    'maps_60_10_10_0.175': DATA / 'maps_60_10_10_0.175',
    'empty': DATA / 'empty',
    'maze': DATA / 'maze',
    'random_large': DATA / 'random_large',
    'random_medium': DATA / 'random_medium',
    'random_small': DATA / 'random_small',
    'room': DATA / 'room',
    'warehouse': DATA / 'warehouse',
}
AGENTS = {
    'maps_60_10_10_0.175': 8,
    'random_small': 8,
    'random_medium': 12,
    'empty': 16,
    'maze': 12,
    'room': 12,
    'warehouse': 14,
    'random_large': 20,
}
SAMPLES = {
    'maps_60_10_10_0.175': 20,
    'random_small': 20,
    'random_medium': 20,
    'empty': 20,
    'maze': 20,
    'room': 20,
    'warehouse': 20,
    'random_large': 20,
}


def neighbors(grid, pos):
    r, c = pos
    H, W = grid.shape
    cand = [(r, c), (r+1,c), (r-1,c), (r,c+1), (r,c-1)]
    out = []
    for nr, nc in cand:
        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
            out.append((nr, nc))
    return out


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def shortest_path(grid, start, goal, vertex_constraints=None, edge_constraints=None, max_time=256):
    vertex_constraints = vertex_constraints or set()
    edge_constraints = edge_constraints or set()
    start_state = (manhattan(start, goal), 0, start, [start])
    pq = [start_state]
    best_g = {(start, 0): 0}
    while pq:
        f, g, pos, path = heapq.heappop(pq)
        t = len(path)-1
        if pos == goal:
            future_blocked = any((goal, tt) in vertex_constraints for tt in range(t, min(max_time, t+8)))
            if not future_blocked:
                return path
        if t >= max_time:
            continue
        for nxt in neighbors(grid, pos):
            nt = t + 1
            if (nxt, nt) in vertex_constraints:
                continue
            if (pos, nxt, nt) in edge_constraints:
                continue
            state = (nxt, nt)
            ng = g + 1
            if ng < best_g.get(state, 10**9):
                best_g[state] = ng
                heapq.heappush(pq, (ng + manhattan(nxt, goal), ng, nxt, path + [nxt]))
    return None


def build_constraints(paths, ignore_idx=None):
    vertex_constraints = set()
    edge_constraints = set()
    horizon = max(len(p) for i,p in enumerate(paths) if i != ignore_idx) if any(i!=ignore_idx for i,_ in enumerate(paths)) else 0
    horizon += 8
    for i, p in enumerate(paths):
        if i == ignore_idx:
            continue
        for t in range(horizon):
            cur = p[t] if t < len(p) else p[-1]
            vertex_constraints.add((cur, t))
            if t > 0:
                prev = p[t-1] if t-1 < len(p) else p[-1]
                edge_constraints.add((cur, prev, t))
    return vertex_constraints, edge_constraints


def detect_conflicts(paths):
    conflicts = []
    T = max(len(p) for p in paths)
    for t in range(T):
        occ = {}
        for i,p in enumerate(paths):
            cur = p[t] if t < len(p) else p[-1]
            if cur in occ:
                conflicts.append(('vertex', t, occ[cur], i, cur))
            occ[cur]=i
        if t>0:
            for i in range(len(paths)):
                pi_prev = paths[i][t-1] if t-1 < len(paths[i]) else paths[i][-1]
                pi_cur = paths[i][t] if t < len(paths[i]) else paths[i][-1]
                for j in range(i+1, len(paths)):
                    pj_prev = paths[j][t-1] if t-1 < len(paths[j]) else paths[j][-1]
                    pj_cur = paths[j][t] if t < len(paths[j]) else paths[j][-1]
                    if pi_prev == pj_cur and pj_prev == pi_cur:
                        conflicts.append(('swap', t, i, j, (pi_cur, pj_cur)))
    return conflicts


def path_cost(paths):
    return sum(len(p)-1 for p in paths)


def makespan(paths):
    return max(len(p)-1 for p in paths)


def prioritized_planning(grid, starts, goals, order=None):
    n = len(starts)
    order = order or list(range(n))
    paths = [None]*n
    for idx in order:
        vc, ec = build_constraints([p for p in paths if p is not None], ignore_idx=None)
        path = shortest_path(grid, starts[idx], goals[idx], vc, ec, max_time=grid.shape[0]*grid.shape[1]//2 + 80)
        if path is None:
            return None
        paths[idx] = path
    return paths


def greedy_marl_order(starts, goals):
    scores=[]
    for i,(s,g) in enumerate(zip(starts,goals)):
        d = manhattan(s,g)
        congestion = 0
        for j,(s2,g2) in enumerate(zip(starts,goals)):
            if i==j: continue
            congestion += 1/(1+manhattan(s,s2)) + 1/(1+manhattan(g,g2))
        score = d + 2.5*congestion
        scores.append((score, i))
    return [i for _,i in sorted(scores, reverse=True)]


def lns_hybrid(grid, starts, goals, max_rounds=12):
    order = greedy_marl_order(starts, goals)
    paths = prioritized_planning(grid, starts, goals, order=order)
    if paths is None:
        return None, {'initial_success': False, 'rounds': 0, 'conflicts': None}
    initial_conflicts = len(detect_conflicts(paths))
    rounds = 0
    for _ in range(max_rounds):
        confs = detect_conflicts(paths)
        if not confs:
            break
        rounds += 1
        touched = set()
        for c in confs[:4]:
            touched.add(c[2]); touched.add(c[3])
        touched = list(touched)
        improved = False
        for idx in touched:
            others = [p for j,p in enumerate(paths) if j != idx]
            vc, ec = build_constraints(others)
            newp = shortest_path(grid, starts[idx], goals[idx], vc, ec, max_time=grid.shape[0]*grid.shape[1]//2 + 100)
            if newp is not None:
                old = paths[idx]
                paths[idx] = newp
                if len(detect_conflicts(paths)) <= len(confs):
                    improved = True
                else:
                    paths[idx] = old
        if not improved:
            break
    return paths, {'initial_success': True, 'rounds': rounds, 'conflicts': initial_conflicts}


def sample_tasks(grid, n_agents, seed):
    rnd = random.Random(seed)
    free = [(int(r), int(c)) for r,c in np.argwhere(grid == 0)]
    if len(free) < 2*n_agents:
        return None
    for _ in range(200):
        starts = rnd.sample(free, n_agents)
        goals = rnd.sample(free, n_agents)
        bad = any(s == g for s,g in zip(starts, goals))
        if not bad:
            return starts, goals
    return None


def evaluate_family(name, files):
    rows=[]
    for k, f in enumerate(files):
        grid = np.load(f)
        tasks = sample_tasks(grid, AGENTS[name], seed=1000+k)
        if tasks is None:
            continue
        starts, goals = tasks
        # baseline PP by shortest-first order
        base_order = sorted(range(len(starts)), key=lambda i: manhattan(starts[i], goals[i]))
        t0 = time.perf_counter()
        base_paths = prioritized_planning(grid, starts, goals, order=base_order)
        t1 = time.perf_counter()
        hybrid_paths, meta = lns_hybrid(grid, starts, goals)
        t2 = time.perf_counter()
        for method, paths, runtime, extra in [
            ('prioritized', base_paths, t1-t0, {}),
            ('hybrid', hybrid_paths, t2-t1, meta),
        ]:
            success = paths is not None and len(detect_conflicts(paths)) == 0
            rows.append({
                'dataset': name,
                'instance': Path(f).name,
                'agents': AGENTS[name],
                'method': method,
                'success': int(success),
                'runtime_sec': runtime,
                'sum_of_costs': path_cost(paths) if paths is not None else math.nan,
                'makespan': makespan(paths) if paths is not None else math.nan,
                'remaining_conflicts': len(detect_conflicts(paths)) if paths is not None else math.nan,
                'initial_conflicts_est': extra.get('conflicts', math.nan),
                'lns_rounds': extra.get('rounds', math.nan),
            })
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    all_rows=[]
    dataset_summary=[]
    for name, d in DIRS.items():
        files = sorted(d.rglob('*.npy'))[:SAMPLES[name]]
        grid = np.load(files[0])
        vals, cnt = np.unique(grid, return_counts=True)
        obstacle_count = int(cnt[list(vals).index(-1)]) if -1 in vals else 0
        dataset_summary.append({
            'dataset': name,
            'instances_evaluated': len(files),
            'grid_h': int(grid.shape[0]),
            'grid_w': int(grid.shape[1]),
            'obstacle_density_sample': obstacle_count / grid.size,
            'agents': AGENTS[name],
        })
        all_rows.extend(evaluate_family(name, files))
    df = pd.DataFrame(all_rows)
    ds = pd.DataFrame(dataset_summary)
    ds.to_csv(OUT / 'dataset_summary.csv', index=False)
    df.to_csv(OUT / 'evaluation_results.csv', index=False)

    summary = df.groupby(['dataset','method']).agg(
        success_rate=('success','mean'),
        runtime_sec_mean=('runtime_sec','mean'),
        runtime_sec_std=('runtime_sec','std'),
        soc_mean=('sum_of_costs','mean'),
        soc_std=('sum_of_costs','std'),
        makespan_mean=('makespan','mean'),
        remaining_conflicts_mean=('remaining_conflicts','mean')
    ).reset_index()
    summary.to_csv(OUT / 'summary_by_dataset_method.csv', index=False)

    pivot = summary.pivot(index='dataset', columns='method', values='success_rate').reset_index()
    pivot.to_csv(OUT / 'success_rate_pivot.csv', index=False)

    # Figures
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(11,5))
    ax = sns.barplot(data=summary, x='dataset', y='success_rate', hue='method')
    ax.set_ylim(0,1.05)
    ax.set_title('Success rate across map families')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(IMG / 'success_rate_by_dataset.png', dpi=200)
    plt.close()

    plt.figure(figsize=(11,5))
    ax = sns.scatterplot(data=summary, x='runtime_sec_mean', y='soc_mean', hue='dataset', style='method', s=120)
    ax.set_title('Runtime-quality tradeoff (lower runtime and lower cost preferred)')
    plt.tight_layout()
    plt.savefig(IMG / 'runtime_quality_tradeoff.png', dpi=200)
    plt.close()

    hybrid_df = df[df['method']=='hybrid'].copy()
    plt.figure(figsize=(11,5))
    ax = sns.boxplot(data=hybrid_df, x='dataset', y='initial_conflicts_est')
    ax.set_title('Hybrid solver estimated pre-repair conflict burden')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(IMG / 'hybrid_initial_conflicts.png', dpi=200)
    plt.close()

    claims = []
    for dataset in summary['dataset'].unique():
        sub = summary[summary['dataset']==dataset].set_index('method')
        if {'hybrid','prioritized'}.issubset(sub.index):
            claims.append({
                'dataset': dataset,
                'hybrid_success_rate': float(sub.loc['hybrid','success_rate']),
                'prioritized_success_rate': float(sub.loc['prioritized','success_rate']),
                'delta_success_rate': float(sub.loc['hybrid','success_rate'] - sub.loc['prioritized','success_rate']),
                'hybrid_runtime_mean_sec': float(sub.loc['hybrid','runtime_sec_mean']),
                'prioritized_runtime_mean_sec': float(sub.loc['prioritized','runtime_sec_mean']),
            })
    with open(OUT / 'claim_recovery_table.json', 'w') as f:
        json.dump(claims, f, indent=2)

    with open(OUT / 'run_metadata.json', 'w') as f:
        json.dump({'seed': 0, 'samples_per_dataset': SAMPLES, 'agents_per_dataset': AGENTS}, f, indent=2)

if __name__ == '__main__':
    main()
