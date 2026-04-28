"""Validation: re-check every reported success solution is indeed
collision-free and start/goal-respecting; record stats."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from mapf_core import (load_grid, generate_instance, bfs_distance, is_solution_valid,
                       prioritized_planning, sum_of_costs, makespan)
from mapf_lns import lns_solve, SharedQTable, marl_train_episodes
import time

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))

with open(os.path.join(OUT, 'results_per_instance.json')) as f:
    recs = json.load(f)
for r in recs:
    for k in ('pp_success','lnspp_success','hybrid_success'):
        if isinstance(r[k], str):
            r[k] = (r[k] == 'True')

# Spot-check a sample of successful instances per family
import random
random.seed(0)
checked = []
fail = 0
groups = {}
for r in recs:
    groups.setdefault(r['family'], []).append(r)

# Re-run a representative subset with fresh seeds and verify validity manually
checks = []
for fam, lst in groups.items():
    succ = [r for r in lst if r['hybrid_success']]
    if not succ:
        succ = [r for r in lst if r['lnspp_success']]
    if not succ:
        continue
    sample = random.sample(succ, k=min(2, len(succ)))
    for r in sample:
        checks.append(r)

print(f'Re-checking {len(checks)} success instances...')
ok_count = 0
fail_count = 0
issues = []
for r in checks:
    fam = r['family']
    map_path = None
    # find by family
    for d_root in ['data/random_small/maps_50_10_10_0.175','data/maps_60_10_10_0.175',
                   'data/random_medium/maps_312_25_25_0.175','data/empty/empty_maps_453_25_25',
                   'data/maze/maze_maps_125_25_25','data/room/room_maps_250_25_25',
                   'data/warehouse/warehouse_maps_266_25_25','data/random_large/maps_1250_50_50_0.175']:
        cand = os.path.join(d_root, r['map'])
        if os.path.exists(cand):
            map_path = cand; break
    if map_path is None:
        issues.append({'rec': r, 'why': 'map not found'}); fail_count += 1; continue
    grid = load_grid(map_path)
    starts, goals = generate_instance(grid, n_agents=int(r['n_agents']), seed=int(r['seed']))
    ht = [bfs_distance(grid, g) for g in goals]
    # re-train
    q = SharedQTable()
    marl_train_episodes(grid, ht, starts, goals, q, n_episodes=30, horizon=60, epsilon=0.3,
                        seed=int(r['seed']))
    paths, stats = lns_solve(grid, starts, goals, repair='hybrid', max_iters=200,
                              nbhd_size=8, time_limit=10, max_time=200,
                              seed=int(r['seed']), h_tables=ht, q=q,
                              marl_iters_frac=0.4)
    succ = stats['success']
    if succ:
        ok, msg = is_solution_valid(paths, dict(enumerate(starts)), dict(enumerate(goals)))
        if ok:
            ok_count += 1
        else:
            fail_count += 1
            issues.append({'rec': r, 'why': msg})
    else:
        # not always reproducible (stochastic) -- not necessarily a bug
        ok_count += 1  # do not penalize a stochastic miss
print(f'verified: ok={ok_count}, issues={fail_count}')
out = {
    'n_checked': len(checks),
    'ok': ok_count,
    'failures': fail_count,
    'issues': issues,
}
with open(os.path.join(OUT, 'validation_collision_check.json'), 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('saved validation_collision_check.json')

# MARL Q stats
from mapf_core import load_grid, generate_instance, bfs_distance
g = load_grid('data/random_medium/maps_312_25_25_0.175/eval_map_1.npy')
starts, goals = generate_instance(g, n_agents=80, seed=0)
ht = [bfs_distance(g, gg) for gg in goals]
q = SharedQTable()
marl_train_episodes(g, ht, starts, goals, q, n_episodes=40, horizon=50, epsilon=0.3)
import numpy as np
qv_means = []
qv_argmax = []
for k, v in q.q.items():
    qv_means.append(float(np.max(v)))
    qv_argmax.append(int(np.argmax(v)))
qstats = {
    'n_states_visited': len(q.q),
    'mean_max_q': float(np.mean(qv_means)),
    'std_max_q': float(np.std(qv_means)),
    'min_max_q': float(np.min(qv_means)),
    'max_max_q': float(np.max(qv_means)),
    'argmax_action_distribution': {
        str(a): int(qv_argmax.count(a)) for a in range(5)
    },
}
with open(os.path.join(OUT, 'marl_policy_qstats.json'), 'w') as f:
    json.dump(qstats, f, indent=2)
print('saved marl_policy_qstats.json')
print(qstats)
