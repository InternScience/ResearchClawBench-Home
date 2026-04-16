"""
Run MAPF experiments - optimized for speed with reduced agent counts and time limits
"""
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mapf_algorithms import GridEnv, pp_solve, marl_policy, lns_repair, rr_pp, marl_lns, count_cp, count_collisions_fast

BASE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_002_20260416_105258"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")

# Reduced config for speed: fewer maps, fewer agents, shorter time limits
DATASETS = {
    "random_small": {
        "subdir": "maps_50_10_10_0.175",
        "agents": [5, 10, 15],
        "nmaps": 3,
    },
    "random_medium": {
        "subdir": "maps_312_25_25_0.175",
        "agents": [10, 20, 40],
        "nmaps": 3,
    },
    "maze": {
        "subdir": "maze_maps_125_25_25",
        "agents": [5, 15, 30],
        "nmaps": 3,
    },
    "room": {
        "subdir": "room_maps_250_25_25",
        "agents": [10, 20, 40],
        "nmaps": 3,
    },
    "warehouse": {
        "subdir": "warehouse_maps_266_25_25",
        "agents": [10, 20, 40],
        "nmaps": 3,
    },
    "empty": {
        "subdir": "empty_maps_453_25_25",
        "agents": [10, 20, 40],
        "nmaps": 3,
    },
}

ALGORITHMS = ["PP", "LNS", "RRPP", "MARL-LNS"]

def load_map(ds, sd):
    d = os.path.join(DATA, ds, sd)
    fs = sorted([f for f in os.listdir(d) if f.endswith('.npy')])
    return [np.load(os.path.join(d,f), allow_pickle=True) for f in fs]

def soc(paths):
    if not paths: return 9999
    return sum(len(p)-1 for p in paths)

def run_algo(env, s, g, algo, tl=30):
    t0=time.time()
    res = {"algo":algo,"success":False,"soc":9999,"time":0,"cp":999,"coll":999,"cp_hist":[]}
    
    try:
        if algo=="PP":
            p=pp_solve(env,s,g,tl=tl)
            if p:
                cp=count_cp(p); res["success"]=cp==0; res["soc"]=soc(p); res["cp"]=cp; res["coll"]=count_collisions_fast(p)
        elif algo=="LNS":
            init=pp_solve(env,s,g,tl=tl*0.3)
            if init is None:
                init=marl_policy(env,s,g,max_steps=128,tl=tl*0.3)
            if init:
                rem=tl-(time.time()-t0)
                p,ch=lns_repair(env,s,g,init,tl=max(rem,5),nh_size=min(4,len(s)),seed=42)
                cp=count_cp(p); res["success"]=cp==0; res["soc"]=soc(p); res["cp"]=cp; res["coll"]=count_collisions_fast(p); res["cp_hist"]=ch
        elif algo=="RRPP":
            p=rr_pp(env,s,g,max_r=5,ttl=tl)
            if p:
                cp=count_cp(p); res["success"]=cp==0; res["soc"]=soc(p); res["cp"]=cp; res["coll"]=count_collisions_fast(p)
        elif algo=="MARL-LNS":
            p,ch,el=marl_lns(env,s,g,marl_tl=tl*0.25,lns_tl=tl*0.75,seed=42)
            if p:
                cp=count_cp(p); res["success"]=cp==0; res["soc"]=soc(p); res["cp"]=cp; res["coll"]=count_collisions_fast(p); res["cp_hist"]=ch; res["time"]=el
    except Exception as e:
        res["error"]=str(e)
    
    if res["time"]==0: res["time"]=time.time()-t0
    return res

def main():
    results=[]
    for ds,cfg in DATASETS.items():
        print(f"\n=== {ds} ===")
        maps=load_map(ds,cfg["subdir"])
        nm=min(cfg["nmaps"],len(maps))
        for na in cfg["agents"]:
            for mi in range(nm):
                env=GridEnv(maps[mi])
                ag=env.gen_agents(na,seed=mi*100+na)
                if ag is None:
                    print(f"  agents={na} map={mi}: SKIP (not enough free cells)")
                    continue
                s,g=ag
                print(f"  agents={na} map={mi}:")
                for algo in ALGORITHMS:
                    r=run_algo(env,s,g,algo,tl=20)
                    r["dataset"]=ds; r["num_agents"]=na; r["map_idx"]=mi
                    results.append(r)
                    print(f"    {algo}: succ={r['success']} soc={r['soc']} cp={r['cp']} t={r['time']:.2f}s")
    
    with open(os.path.join(OUT,"experiment_results.json"),"w") as f:
        json.dump(results,f,indent=2)
    print(f"\nTotal: {len(results)} experiments saved")

if __name__=="__main__":
    main()