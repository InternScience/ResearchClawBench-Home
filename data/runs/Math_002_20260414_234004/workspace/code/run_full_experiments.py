import sys; sys.path.insert(0, 'code')
import numpy as np, json, os, time
from mapf_algorithms import MAPFEnv, load_map, generate_agents, prioritized_planning, lns, hybrid_marl_lns, count_collisions

os.makedirs('outputs', exist_ok=True)

map_configs = [
    ('random_small', 'data/random_small/maps_50_10_10_0.175'),
    ('random_medium', 'data/random_medium/maps_312_25_25_0.175'),
    ('maze', 'data/maze/maze_maps_125_25_25'),
    ('room', 'data/room/room_maps_250_25_25'),
    ('empty', 'data/empty/empty_maps_453_25_25'),
    ('warehouse', 'data/warehouse/warehouse_maps_266_25_25'),
]

agent_counts = [5, 10, 15]
time_limit = 5.0
all_results = []

for map_type, map_dir in map_configs:
    for inst in range(1, 3):
        map_path = None
        for f in os.listdir(map_dir):
            if f.endswith(f'_{inst}.npy') or f == f'eval_map_{inst}.npy':
                map_path = os.path.join(map_dir, f); break
        if map_path is None: continue
        grid = load_map(map_path)
        free_cells = int(np.sum(grid == 0))
        for n_agents in agent_counts:
            actual = min(n_agents, free_cells // 2)
            if actual < 3: continue
            starts, goals = generate_agents(grid, actual, seed=inst*100+n_agents)
            if starts is None: continue
            env = MAPFEnv(grid, starts, goals)
            pp_paths, pp_t = prioritized_planning(env, time_limit=time_limit)
            pp_coll = count_collisions(pp_paths) if pp_paths else 999
            pp_cost = sum(len(p) for p in pp_paths) if pp_paths else -1
            lns_p, lns_t, lns_c = lns(env, time_limit=time_limit, num_destroy=max(2,actual//5), max_iter=500)
            lns_cost = sum(len(p) for p in lns_p) if lns_p else -1
            hyb_p, hyb_t, hyb_c, hyb_d = hybrid_marl_lns(env, time_limit=time_limit, marl_episodes=5)
            hyb_cost = sum(len(p) for p in hyb_p) if hyb_p else -1
            r = {'map_type':map_type,'instance':inst,'n_agents':actual,'map_shape':list(grid.shape),
                 'obstacle_density':round(float(np.sum(grid==-1))/grid.size,4),
                 'PP':{'success':pp_coll==0,'time':round(pp_t,3),'collisions':pp_coll,'cost':pp_cost},
                 'LNS':{'success':lns_c==0,'time':round(lns_t,3),'collisions':lns_c,'cost':lns_cost},
                 'Hybrid_MARL_LNS':{'success':hyb_c==0,'time':round(hyb_t,3),'collisions':hyb_c,'cost':hyb_cost}}
            all_results.append(r)
            print(f"{map_type} n={actual} i={inst}: PP[c={pp_coll}] LNS[c={lns_c}] HYB[c={hyb_c}]", flush=True)

with open('outputs/experiment_results.json','w') as f:
    json.dump(all_results, f, indent=2)
print(f"Total: {len(all_results)}")
