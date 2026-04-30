#!/usr/bin/env python3
"""Bounded MAPF benchmark with prioritized planning and MARL-inspired LNS repair.

The implementation is self-contained and uses only numpy/pandas/matplotlib.
Input data contain obstacle grids only, so start/goal tasks are generated
reproducibly from free cells per map instance and agent count.
"""
from __future__ import annotations
import argparse, csv, heapq, json, math, os, random, re, time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

Pos = Tuple[int, int]
PathT = List[Pos]

MOVES = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]

@dataclass
class Instance:
    family: str
    group: str
    map_id: str
    path: str
    grid: np.ndarray
    starts: List[Pos]
    goals: List[Pos]
    agents: int


def stable_seed(s: str) -> int:
    v = 2166136261
    for ch in s:
        v = (v ^ ord(ch)) * 16777619 & 0xffffffff
    return v


def parse_group_agents(group: str, default: int = 20) -> int:
    nums = [int(x) for x in re.findall(r"\d+", group)]
    if not nums:
        return default
    # Directory names encode free cells, height, width, obstacle density; use free-cell
    # count as a density proxy and map it to a manageable number of agents.
    if len(nums) >= 3:
        free, h, w = nums[0], nums[1], nums[2]
        area = h*w
        density = free / max(1, area)
        if h <= 10: return 12 if density > 0.65 else 8
        if h >= 50: return 40 if density > 0.65 else 30
        return 28 if density > 0.70 else 22
    return min(default, max(6, nums[0]//10))


def choose_agent_count(family: str, group: str, grid: np.ndarray) -> int:
    free = int(np.sum(grid == 0))
    base = parse_group_agents(group, 20)
    # Keep benchmark bounded while preserving high-density differences.
    if grid.shape[0] <= 10:
        base = min(base, 12)
    elif grid.shape[0] >= 50:
        base = min(base, 36)
    else:
        base = min(base, 24)
    return max(4, min(base, free//3))


def generate_tasks(grid: np.ndarray, n: int, seed: int) -> Tuple[List[Pos], List[Pos]]:
    free = [(int(r), int(c)) for r,c in zip(*np.where(grid == 0))]
    rng = random.Random(seed)
    # Prefer starts/goals far apart: sample starts and greedily choose distant goals.
    starts = rng.sample(free, n)
    remaining = [p for p in free if p not in set(starts)]
    goals=[]
    for s in starts:
        candidates = rng.sample(remaining, min(len(remaining), 80)) if len(remaining)>80 else list(remaining)
        candidates.sort(key=lambda p: abs(p[0]-s[0])+abs(p[1]-s[1]), reverse=True)
        g = candidates[0]
        goals.append(g); remaining.remove(g)
    return starts, goals


def neighbors(grid: np.ndarray, p: Pos) -> Iterable[Pos]:
    h,w=grid.shape
    for dr,dc in MOVES:
        q=(p[0]+dr,p[1]+dc)
        if 0 <= q[0] < h and 0 <= q[1] < w and grid[q] == 0:
            yield q


def unconstrained_shortest(grid: np.ndarray, s: Pos, g: Pos) -> Optional[PathT]:
    q=deque([s]); parent={s:None}
    while q:
        p=q.popleft()
        if p==g: break
        for nb in neighbors(grid,p):
            if nb not in parent:
                parent[nb]=p; q.append(nb)
    if g not in parent: return None
    path=[]; p=g
    while p is not None:
        path.append(p); p=parent[p]
    return path[::-1]


def pos_at(path: PathT, t: int) -> Pos:
    return path[t] if t < len(path) else path[-1]


def build_reservations(paths: List[Optional[PathT]], exclude: Set[int]=set()) -> Tuple[Dict[int,Set[Pos]], Dict[int,Set[Tuple[Pos,Pos]]]]:
    max_t=max((len(p or []) for i,p in enumerate(paths) if i not in exclude), default=1) + 80
    verts=defaultdict(set); edges=defaultdict(set)
    for i,p in enumerate(paths):
        if i in exclude or p is None: continue
        for t in range(max_t+1):
            a=pos_at(p,t); b=pos_at(p,t+1)
            verts[t].add(a); edges[t+1].add((a,b))
    return verts, edges


def astar_time(grid: np.ndarray, s: Pos, g: Pos, vertex_res: Dict[int,Set[Pos]], edge_res: Dict[int,Set[Tuple[Pos,Pos]]], max_time: int) -> Optional[PathT]:
    def h(p): return abs(p[0]-g[0])+abs(p[1]-g[1])
    pq=[(h(s),0,s)]
    parent={(s,0):None}
    best_goal=None
    while pq:
        f,t,p=heapq.heappop(pq)
        if t>max_time: continue
        if p==g:
            # goal must be safe for a short suffix in reservations
            safe=True
            for ft in range(t, min(max_time, t+10)+1):
                if g in vertex_res.get(ft,set()): safe=False; break
            if safe:
                best_goal=(p,t); break
        for q in neighbors(grid,p):
            nt=t+1
            if q in vertex_res.get(nt,set()): continue
            # Swapping collision: another planned edge q->p at nt conflicts with p->q.
            if (q,p) in edge_res.get(nt,set()): continue
            state=(q,nt)
            if state in parent: continue
            parent[state]=(p,t)
            heapq.heappush(pq,(nt+h(q),nt,q))
    if best_goal is None: return None
    path=[]; cur=best_goal
    while cur is not None:
        p,t=cur; path.append(p); cur=parent[cur]
    return path[::-1]


def prioritized_planning(grid: np.ndarray, starts: List[Pos], goals: List[Pos], order: Optional[List[int]]=None, max_time_factor: int=4) -> List[Optional[PathT]]:
    n=len(starts); order=order or list(range(n))
    paths=[None]*n
    max_manh=max(abs(s[0]-g[0])+abs(s[1]-g[1]) for s,g in zip(starts,goals))
    max_time=max(40, max_time_factor*(grid.shape[0]+grid.shape[1]+max_manh))
    for aid in order:
        v,e=build_reservations(paths)
        p=astar_time(grid, starts[aid], goals[aid], v, e, max_time)
        if p is None:
            p=unconstrained_shortest(grid, starts[aid], goals[aid])
        paths[aid]=p
    return paths


def detect_collisions(paths: List[Optional[PathT]]) -> Dict:
    bad_unplanned=[i for i,p in enumerate(paths) if p is None]
    maxlen=max((len(p) for p in paths if p), default=0)
    vertex=[]; swap=[]
    for t in range(maxlen + 1):
        occ={}
        for i,p in enumerate(paths):
            if p is None: continue
            x=pos_at(p,t)
            if x in occ: vertex.append({'t':t,'a1':occ[x],'a2':i,'pos':x})
            else: occ[x]=i
        if t>0:
            seen={}
            for i,p in enumerate(paths):
                if p is None: continue
                edge=(pos_at(p,t-1), pos_at(p,t))
                rev=(edge[1],edge[0])
                if rev in seen and edge[0]!=edge[1]: swap.append({'t':t,'a1':seen[rev],'a2':i,'edge':edge})
                seen[edge]=i
    return {'vertex':vertex,'swap':swap,'unplanned':bad_unplanned,'total':len(vertex)+len(swap)+len(bad_unplanned)}


def metrics(paths: List[Optional[PathT]], runtime: float) -> Dict:
    col=detect_collisions(paths)
    valid = col['total']==0 and all(p is not None for p in paths)
    costs=[len(p)-1 for p in paths if p]
    return {
        'success': int(valid), 'collisions_total': col['total'], 'vertex_collisions': len(col['vertex']),
        'swap_collisions': len(col['swap']), 'unplanned_agents': len(col['unplanned']),
        'sum_of_costs': int(sum(costs)) if costs else 0, 'makespan': int(max(costs) if costs else 0),
        'runtime_seconds': runtime
    }


def feature_agent(grid: np.ndarray, paths: List[Optional[PathT]], starts: List[Pos], goals: List[Pos], aid: int, conflict_counts: Counter) -> Dict[str,float]:
    p=paths[aid]
    length=(len(p)-1) if p else 999
    man=abs(starts[aid][0]-goals[aid][0])+abs(starts[aid][1]-goals[aid][1])
    # local obstacle density around current start/goal corridor bounding box
    r0,r1=sorted([starts[aid][0],goals[aid][0]]); c0,c1=sorted([starts[aid][1],goals[aid][1]])
    r0=max(0,r0-2); r1=min(grid.shape[0],r1+3); c0=max(0,c0-2); c1=min(grid.shape[1],c1+3)
    dens=float(np.mean(grid[r0:r1,c0:c1] != 0))
    return {'conflicts':float(conflict_counts[aid]), 'excess_length':float(length-man), 'manhattan':float(man), 'corridor_obstacle_density':dens}

POLICY_WEIGHTS={'conflicts':4.0,'excess_length':0.15,'manhattan':0.02,'corridor_obstacle_density':1.5}

def select_neighborhood(grid, paths, starts, goals, k: int, mode: str, rng: random.Random) -> List[int]:
    col=detect_collisions(paths)
    n=len(starts)
    cc=Counter(col['unplanned'])
    for c in col['vertex']+col['swap']:
        cc[c['a1']]+=1; cc[c['a2']]+=1
    if mode=='random':
        return rng.sample(range(n), min(k,n))
    scores=[]
    for aid in range(n):
        f=feature_agent(grid,paths,starts,goals,aid,cc)
        s=sum(POLICY_WEIGHTS[x]*f[x] for x in POLICY_WEIGHTS)
        # Gumbel-like small exploration.
        s += rng.random()*0.01
        scores.append((s,aid))
    scores.sort(reverse=True)
    chosen=[aid for _,aid in scores[:min(k,n)]]
    if not chosen and n: chosen=[rng.randrange(n)]
    return chosen


def lns_repair(grid: np.ndarray, starts: List[Pos], goals: List[Pos], initial: List[Optional[PathT]], iterations: int=12, k: int=6, mode: str='policy', seed: int=0) -> Tuple[List[Optional[PathT]], List[Dict]]:
    rng=random.Random(seed)
    paths=list(initial)
    history=[]
    best=list(paths); best_col=detect_collisions(paths)['total']
    for it in range(iterations):
        before=detect_collisions(paths)['total']
        neigh=select_neighborhood(grid, paths, starts, goals, k, mode, rng)
        proposal=list(paths)
        for aid in neigh: proposal[aid]=None
        order=list(neigh)
        # Late-stage efficiency: prioritize most conflicted agents first and use PP repair.
        cc=Counter()
        col=detect_collisions(paths)
        for c in col['vertex']+col['swap']:
            cc[c['a1']]+=1; cc[c['a2']]+=1
        order.sort(key=lambda a:(-cc[a], a))
        for aid in order:
            v,e=build_reservations(proposal)
            max_time=max(50, 4*(grid.shape[0]+grid.shape[1]+abs(starts[aid][0]-goals[aid][0])+abs(starts[aid][1]-goals[aid][1])))
            proposal[aid]=astar_time(grid, starts[aid], goals[aid], v, e, max_time)
            if proposal[aid] is None:
                proposal[aid]=unconstrained_shortest(grid, starts[aid], goals[aid])
        after=detect_collisions(proposal)['total']
        old_cost=sum(len(p)-1 for p in paths if p)
        new_cost=sum(len(p)-1 for p in proposal if p)
        accept = (after < before) or (after==before and new_cost <= old_cost) or (rng.random()<0.03 and after <= before+1)
        if accept: paths=proposal
        cur=detect_collisions(paths)['total']
        if cur < best_col:
            best_col=cur; best=list(paths)
        history.append({'iteration':it,'before_collisions':before,'proposal_collisions':after,'accepted':int(accept),'current_collisions':cur,'best_collisions':best_col,'neighborhood_size':len(neigh),'mode':mode})
        if best_col==0:
            # still record bounded early stop
            break
    return best, history


def load_instances(data_dir: Path, per_family: int, seed: int) -> List[Instance]:
    rng=random.Random(seed)
    instances=[]
    for top in sorted(data_dir.iterdir()):
        if not top.is_dir(): continue
        files=sorted([p for p in top.glob('**/*.npy')])
        if not files: continue
        rng.shuffle(files)
        for f in files[:per_family]:
            grid=np.load(f)
            group=f.parent.name if f.parent != top else top.name
            n=choose_agent_count(top.name, group, grid)
            starts,goals=generate_tasks(grid,n,stable_seed(str(f.relative_to(data_dir))))
            instances.append(Instance(top.name, group, f.stem, str(f), grid, starts, goals, n))
    return instances


def data_overview(data_dir: Path, out: Path) -> pd.DataFrame:
    rows=[]
    for top in sorted(data_dir.iterdir()):
        if not top.is_dir(): continue
        files=sorted([p for p in top.glob('**/*.npy')])
        for f in files:
            g=np.load(f)
            group=f.parent.name if f.parent != top else top.name
            rows.append({'family':top.name,'group':group,'file':str(f),'height':g.shape[0],'width':g.shape[1],'free_cells':int((g==0).sum()),'obstacles':int((g!=0).sum()),'obstacle_density':float((g!=0).mean()),'agent_count_generated':choose_agent_count(top.name,group,g)})
    df=pd.DataFrame(rows)
    df.to_csv(out/'data_overview.csv',index=False)
    return df


def run(args):
    out=Path(args.outputs); out.mkdir(parents=True, exist_ok=True)
    overview=data_overview(Path(args.data), out)
    insts=load_instances(Path(args.data), args.per_family, args.seed)
    results=[]; histories=[]; validations=[]; solutions={}
    for idx,ins in enumerate(insts):
        methods=[]
        # Baseline 1: independent shortest paths (fast but collision-prone) for contrast.
        t0=time.perf_counter(); indep=[unconstrained_shortest(ins.grid,s,g) for s,g in zip(ins.starts,ins.goals)]; rt=time.perf_counter()-t0
        methods.append(('IndependentShortest', indep, rt, []))
        # Baseline 2: prioritized planning.
        t0=time.perf_counter(); pp=prioritized_planning(ins.grid,ins.starts,ins.goals); rt=time.perf_counter()-t0
        methods.append(('PrioritizedPlanning', pp, rt, []))
        # Ablation: random LNS.
        t0=time.perf_counter(); rand,hist_r=lns_repair(ins.grid,ins.starts,ins.goals,pp,iterations=args.iterations,k=args.neighborhood,mode='random',seed=args.seed+idx); rt=time.perf_counter()-t0
        methods.append(('RandomLNS+PP', rand, rt, hist_r))
        # Proposed hybrid: policy/MARL-inspired neighborhood LNS.
        t0=time.perf_counter(); hyb,hist_h=lns_repair(ins.grid,ins.starts,ins.goals,pp,iterations=args.iterations,k=args.neighborhood,mode='policy',seed=args.seed+1000+idx); rt=time.perf_counter()-t0
        methods.append(('PolicyLNS+PP', hyb, rt, hist_h))
        for name,paths,rt,hist in methods:
            m=metrics(paths,rt); m.update({'family':ins.family,'group':ins.group,'map_id':ins.map_id,'file':ins.path,'agents':ins.agents,'method':name})
            results.append(m)
            col=detect_collisions(paths)
            if len(validations)<20 and name in ('PrioritizedPlanning','PolicyLNS+PP'):
                validations.append({'family':ins.family,'map_id':ins.map_id,'method':name,'collision_total':col['total'],'vertex':len(col['vertex']),'swap':len(col['swap']),'unplanned':len(col['unplanned']),'starts':ins.starts[:5],'goals':ins.goals[:5]})
            if hist:
                for h in hist:
                    h.update({'family':ins.family,'group':ins.group,'map_id':ins.map_id,'method':name})
                    histories.append(h)
        if idx < 8:
            solutions[f'{ins.family}/{ins.map_id}']={'starts':ins.starts,'goals':ins.goals,'policy_solution':hyb}
        print(f'[{idx+1}/{len(insts)}] {ins.family} {ins.map_id} agents={ins.agents}')
    df=pd.DataFrame(results); df.to_csv(out/'benchmark_results.csv',index=False)
    summ=df.groupby(['family','method']).agg(success_rate=('success','mean'), mean_collisions=('collisions_total','mean'), median_collisions=('collisions_total','median'), mean_runtime_s=('runtime_seconds','mean'), mean_sum_of_costs=('sum_of_costs','mean'), mean_makespan=('makespan','mean'), n=('success','size')).reset_index()
    summ.to_csv(out/'summary_by_family_method.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'lns_history.csv',index=False)
    with open(out/'validation_examples.json','w') as f: json.dump(validations,f,indent=2,default=lambda x: list(x) if isinstance(x,tuple) else x)
    with open(out/'sample_solutions.json','w') as f: json.dump(solutions,f,indent=2,default=lambda x: list(x) if isinstance(x,tuple) else x)
    pd.DataFrame([{'feature':k,'weight':v,'interpretation':{'conflicts':'agents involved in vertex/swap conflicts are repaired first','excess_length':'paths much longer than Manhattan distance suggest detours','manhattan':'long tasks are slightly prioritized','corridor_obstacle_density':'dense/bottleneck corridors receive repair attention'}[k]} for k,v in POLICY_WEIGHTS.items()]).to_csv(out/'neighborhood_policy_importance.csv',index=False)
    print('WROTE', len(df), 'rows')

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data')
    ap.add_argument('--outputs',default='outputs')
    ap.add_argument('--per-family',type=int,default=8)
    ap.add_argument('--iterations',type=int,default=12)
    ap.add_argument('--neighborhood',type=int,default=6)
    ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args(); run(args)
