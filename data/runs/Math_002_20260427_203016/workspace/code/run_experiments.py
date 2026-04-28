"""
run_experiments.py

Run benchmark across map families with three solvers:
    - PP (Prioritized Planning, baseline)
    - LNS-PP (LNS2-style with PP repair)
    - LNS-Hybrid (proposed: MARL repair early, PP repair late)

For each (family, agent_count): pick a small number of instances, generate
deterministic start/goal placements, run all three solvers, record metrics.
"""
from __future__ import annotations
import os, sys, time, json, csv
import numpy as np
import random

sys.path.insert(0, os.path.dirname(__file__))
from mapf_core import (load_grid, generate_instance, prioritized_planning,
                       bfs_distance, is_solution_valid, sum_of_costs, makespan,
                       num_collisions)
from mapf_lns import (lns_solve, SharedQTable, marl_train_episodes,
                      initial_paths_shortest)


# ---------- benchmark configuration ----------

# (family_name, base_dir, max_n_agents, n_instances, time_limit_s, max_time_horizon)
# Reduced agent counts so the evaluation finishes in a reasonable wall clock.
BENCHMARK = [
    ('random_small_10x10', 'data/random_small/maps_50_10_10_0.175', [15, 25, 35],    4, 4.0,  80),
    ('target_60a_10x10',   'data/maps_60_10_10_0.175',              [30, 45, 60],    4, 5.0, 100),
    ('random_medium_25x25','data/random_medium/maps_312_25_25_0.175',[40, 80, 120],  3, 6.0, 140),
    ('empty_25x25',        'data/empty/empty_maps_453_25_25',        [80, 140, 200], 3, 6.0, 140),
    ('maze_25x25',         'data/maze/maze_maps_125_25_25',         [15, 30, 45],    3, 6.0, 180),
    ('room_25x25',         'data/room/room_maps_250_25_25',         [40, 80, 120],   3, 6.0, 140),
    ('warehouse_25x25',    'data/warehouse/warehouse_maps_266_25_25',[60, 100, 140], 3, 6.0, 140),
    ('random_large_50x50', 'data/random_large/maps_1250_50_50_0.175',[60, 120, 200], 2, 10.0, 220),
]


def list_maps(d: str, k: int):
    files = sorted([f for f in os.listdir(d) if f.endswith('.npy')])
    return [os.path.join(d, f) for f in files[:k]]


def train_q_for_family(grid, starts, goals, h_tables, train_episodes=40, horizon=60, seed=0):
    q = SharedQTable()
    marl_train_episodes(grid, h_tables, starts, goals, q,
                        n_episodes=train_episodes, horizon=horizon,
                        epsilon=0.3, lr=0.3, gamma=0.95, seed=seed)
    return q


def run_one(map_path, n_agents, family_name, time_limit, max_time, seed, q_pretrained=None):
    grid = load_grid(map_path)
    starts, goals = generate_instance(grid, n_agents=n_agents, seed=seed)
    n_actual = len(starts)
    ht = [bfs_distance(grid, g) for g in goals]
    rec = {
        'family': family_name, 'map': os.path.basename(map_path),
        'n_agents': n_actual, 'seed': seed,
    }

    # ----- PP (no LNS) -----
    t0 = time.time()
    paths, fail = prioritized_planning(grid, starts, goals, h_tables=ht,
                                       max_time=max_time, seed=seed)
    t_pp = time.time() - t0
    if paths is not None:
        ok, _ = is_solution_valid(paths, dict(enumerate(starts)),
                                   dict(enumerate(goals)))
    else:
        ok = False
    rec.update({
        'pp_success': bool(ok),
        'pp_time_s': t_pp,
        'pp_soc': sum_of_costs(paths, dict(enumerate(goals))) if (paths and ok) else None,
        'pp_makespan': makespan(paths) if (paths and ok) else None,
    })

    # ----- LNS-PP -----
    t0 = time.time()
    paths_lp, stats_lp = lns_solve(grid, starts, goals, repair='pp',
                                    max_iters=200, nbhd_size=8,
                                    time_limit=time_limit, max_time=max_time,
                                    seed=seed, h_tables=ht)
    t_lp = time.time() - t0
    ok_lp = stats_lp['success']
    if ok_lp:
        ok2, _ = is_solution_valid(paths_lp, dict(enumerate(starts)),
                                    dict(enumerate(goals)))
        ok_lp = ok2
    rec.update({
        'lnspp_success': bool(ok_lp),
        'lnspp_time_s': t_lp,
        'lnspp_iters': stats_lp['iters'],
        'lnspp_soc': sum_of_costs(paths_lp, dict(enumerate(goals))) if ok_lp else None,
        'lnspp_makespan': makespan(paths_lp) if ok_lp else None,
        'lnspp_log': stats_lp['log'],
    })

    # ----- LNS-Hybrid (MARL early + PP late) -----
    t_train_0 = time.time()
    if q_pretrained is None:
        q = train_q_for_family(grid, starts, goals, ht,
                               train_episodes=30, horizon=60, seed=seed)
    else:
        q = q_pretrained
        # extra fine-tune on this instance for a few episodes
        marl_train_episodes(grid, ht, starts, goals, q,
                            n_episodes=6, horizon=50, epsilon=0.2,
                            lr=0.2, gamma=0.95, seed=seed)
    t_train = time.time() - t_train_0
    t0 = time.time()
    paths_h, stats_h = lns_solve(grid, starts, goals, repair='hybrid',
                                  max_iters=200, nbhd_size=8,
                                  time_limit=time_limit, max_time=max_time,
                                  seed=seed, h_tables=ht, q=q,
                                  marl_iters_frac=0.4)
    t_h = time.time() - t0
    ok_h = stats_h['success']
    if ok_h:
        ok2, _ = is_solution_valid(paths_h, dict(enumerate(starts)),
                                    dict(enumerate(goals)))
        ok_h = ok2
    rec.update({
        'hybrid_success': bool(ok_h),
        'hybrid_time_s': t_h,
        'hybrid_train_s': t_train,
        'hybrid_iters': stats_h['iters'],
        'hybrid_soc': sum_of_costs(paths_h, dict(enumerate(goals))) if ok_h else None,
        'hybrid_makespan': makespan(paths_h) if ok_h else None,
        'hybrid_log': stats_h['log'],
        'hybrid_qsize': q.size(),
    })
    return rec


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    all_records = []
    t_global = time.time()
    for family, base_dir, n_list, n_inst, tl, mt in BENCHMARK:
        if not os.path.isdir(base_dir):
            print(f'skip missing {base_dir}')
            continue
        maps = list_maps(base_dir, n_inst)
        for n_agents in n_list:
            q_cache = None
            for k_idx, mp in enumerate(maps):
                seed = abs(hash((family, mp, n_agents))) % (2**31)
                t0 = time.time()
                try:
                    rec = run_one(mp, n_agents, family, tl, mt, seed,
                                   q_pretrained=q_cache)
                    rec['wall_s'] = time.time() - t0
                    rec['cumulative_min'] = (time.time() - t_global) / 60.0
                    if q_cache is None:
                        # rebuild q from disk: easier to pull it from the
                        # function -- here we just train once and reuse.
                        # We do a one-shot training on the first instance
                        # outside run_one so subsequent runs reuse it.
                        from mapf_lns import SharedQTable
                        q_cache = SharedQTable()
                        gg = load_grid(mp)
                        ss, gs = generate_instance(gg, n_agents=n_agents, seed=seed)
                        ht_ = [bfs_distance(gg, g_) for g_ in gs]
                        marl_train_episodes(gg, ht_, ss, gs, q_cache,
                                            n_episodes=30, horizon=60,
                                            epsilon=0.3, lr=0.3, gamma=0.95,
                                            seed=seed)
                    print(f"[{rec['cumulative_min']:5.2f}m] {family:24s} n={n_agents:3d} "
                          f"{os.path.basename(mp):28s} pp={rec['pp_success']:1d} "
                          f"lns={rec['lnspp_success']:1d} hyb={rec['hybrid_success']:1d} "
                          f"t_h={rec['hybrid_time_s']:.1f}s",
                          flush=True)
                    all_records.append(rec)
                except Exception as e:
                    print(f'ERR {family} {mp} {n_agents}: {e}', flush=True)
                    continue
        # save partial after each family
        with open(os.path.join(out_dir, 'results_per_instance.json'), 'w') as f:
            # strip log lists out of instance file (kept for plots separately)
            slim = []
            for r in all_records:
                rr = {k: v for k, v in r.items() if not k.endswith('_log')}
                slim.append(rr)
            json.dump(slim, f, indent=2, default=str)
        # keep logs in a separate npz/json for convergence plots
        with open(os.path.join(out_dir, 'lns_logs.json'), 'w') as f:
            json.dump([{k: r.get(k) for k in
                        ('family','map','n_agents','seed',
                         'lnspp_log','hybrid_log',
                         'lnspp_success','hybrid_success')}
                       for r in all_records], f, indent=2, default=str)
    print(f'\n=== total {(time.time()-t_global)/60:.2f} min, {len(all_records)} records ===')


if __name__ == '__main__':
    main()
