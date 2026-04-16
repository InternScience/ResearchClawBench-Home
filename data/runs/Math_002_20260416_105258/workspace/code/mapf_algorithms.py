"""
Optimized MAPF Core Algorithms - Fast implementation for experiments
"""
import numpy as np
import heapq
import time
import random
from copy import deepcopy

class GridEnv:
    def __init__(self, grid):
        self.grid = grid.copy()
        self.R, self.C = grid.shape
        self.free = set()
        for r in range(self.R):
            for c in range(self.C):
                if grid[r,c] == 0:
                    self.free.add((r,c))
    
    def neighbors(self, pos):
        r,c = pos
        result = []
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr,nc = r+dr, c+dc
            if 0<=nr<self.R and 0<=nc<self.C and self.grid[nr,nc]==0:
                result.append((nr,nc))
        result.append((r,c))
        return result
    
    def h(self, a, b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])
    
    def gen_agents(self, n, seed=0):
        rng = random.Random(seed)
        fl = sorted(list(self.free))
        if len(fl) < 2*n: return None
        pts = rng.sample(fl, 2*n)
        return pts[:n], pts[n:]

def spacetime_astar(env, start, goal, occupied_times=None, max_t=100):
    """Fast space-time A*. occupied_times: dict (pos,t)->True for hard obstacles."""
    if occupied_times is None:
        occupied_times = {}
    
    h0 = env.h(start, goal)
    open_list = [(h0, 0, start[0], start[1])]
    g_vals = {(start,0): 0}
    came_from = {}
    
    while open_list:
        f_val, t, r, c = heapq.heappop(open_list)
        pos = (r,c)
        state = (pos, t)
        
        if pos == goal:
            path = []
            cur = state
            while cur in came_from:
                path.append(cur[0])
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path
        
        g = g_vals.get(state, float('inf'))
        if g + env.h(pos, goal) > f_val + 0.01:
            continue
        
        if t >= max_t:
            continue
        
        for npos in env.neighbors(pos):
            nt = t + 1
            nstate = (npos, nt)
            if (npos, nt) in occupied_times:
                continue
            ng = g + 1
            old_g = g_vals.get(nstate, float('inf'))
            if ng < old_g:
                g_vals[nstate] = ng
                came_from[nstate] = state
                nf = ng + env.h(npos, goal)
                heapq.heappush(open_list, (nf, nt, npos[0], npos[1]))
    
    return None

def count_cp(paths):
    """Count colliding pairs quickly."""
    if not paths: return 999
    ml = max(len(p) for p in paths)
    padded = [p + [p[-1]]*(ml-len(p)) if p else [] for p in paths]
    
    pairs = set()
    for t in range(ml):
        seen = {}
        for i,p in enumerate(padded):
            if t<len(p):
                pos = p[t]
                if pos in seen:
                    j = seen[pos]
                    pairs.add((min(i,j),max(i,j)))
                else:
                    seen[pos] = i
    for t in range(ml-1):
        for i in range(len(padded)):
            if t+1>=len(padded[i]): continue
            for j in range(i+1,len(padded)):
                if t+1>=len(padded[j]): continue
                if padded[i][t]==padded[j][t+1] and padded[i][t+1]==padded[j][t]:
                    pairs.add((min(i,j),max(i,j)))
    return len(pairs)

def count_collisions_fast(paths):
    """Count total collision events."""
    if not paths: return 999
    ml = max(len(p) for p in paths)
    padded = [p+[p[-1]]*(ml-len(p)) for p in paths]
    count = 0
    for t in range(ml):
        seen = {}
        for i,pa in enumerate(padded):
            if t<len(pa):
                pos=pa[t]
                if pos in seen: count+=1
                else: seen[pos]=i
    for t in range(ml-1):
        for i in range(len(padded)):
            if t+1>=len(padded[i]): continue
            for j in range(i+1,len(padded)):
                if t+1>=len(padded[j]): continue
                if padded[i][t]==padded[j][t+1] and padded[i][t+1]==padded[j][t]:
                    count+=1
    return count

def pp_solve(env, starts, goals, order=None, tl=15, max_t=80):
    """Prioritized Planning."""
    n = len(starts)
    if order is None: order = list(range(n))
    paths = [None]*n
    t0 = time.time()
    
    for idx in order:
        if time.time()-t0 > tl: return None
        occ = {}
        max_pl = max_t
        for oi in order:
            if oi==idx or paths[oi] is None: continue
            p = paths[oi]
            for ts in range(len(p)):
                occ[(p[ts],ts)] = True
            for ts in range(len(p), max_pl):
                occ[(p[-1],ts)] = True
        
        path = spacetime_astar(env, starts[idx], goals[idx], occ, max_t=max_pl)
        if path is None: return None
        paths[idx] = path
    return paths

def marl_policy(env, starts, goals, max_steps=128, tl=10):
    """MARL-inspired cooperative decentralized policy."""
    n = len(starts)
    t0 = time.time()
    pos = list(starts)
    paths = [[p] for p in pos]
    done = [p==goals[i] for i,p in enumerate(pos)]
    
    for step in range(max_steps):
        if time.time()-t0>tl: break
        if all(done): break
        
        dists = [env.h(pos[i],goals[i]) if not done[i] else 0 for i in range(n)]
        order = sorted(range(n), key=lambda i: dists[i])
        
        claimed = {}
        moves = {}
        
        for i in order:
            if done[i]:
                moves[i] = pos[i]
                if pos[i] not in claimed: claimed[pos[i]]=i
                continue
            
            nbrs = env.neighbors(pos[i])
            avail = [n for n in nbrs if n not in claimed]
            if not avail:
                moves[i] = pos[i]
                if pos[i] not in claimed: claimed[pos[i]]=i
                continue
            
            best = min(avail, key=lambda n: env.h(n, goals[i]))
            if best==pos[i] and len(avail)>1:
                mv = [n for n in avail if n!=pos[i]]
                if mv: best = min(mv, key=lambda n: env.h(n, goals[i]))
            
            moves[i] = best
            claimed[best] = i
        
        for i in range(n):
            for j in range(i+1,n):
                if moves[i]==pos[j] and moves[j]==pos[i]:
                    if dists[i]<=dists[j]: moves[j]=pos[j]
                    else: moves[i]=pos[i]
        
        for i in range(n):
            pos[i] = moves[i]
            paths[i].append(pos[i])
            done[i] = (pos[i]==goals[i])
    
    return paths

def lns_repair(env, starts, goals, init_paths, max_iter=200, tl=30, nh_size=4, seed=42):
    """LNS repair."""
    rng = random.Random(seed)
    paths = [list(p) for p in init_paths]
    t0 = time.time()
    cp_hist = []
    
    for it in range(max_iter):
        if time.time()-t0>tl: break
        cp = count_cp(paths)
        cp_hist.append(cp)
        if cp==0: break
        
        ml = max(len(p) for p in paths)
        padded = [p+[p[-1]]*(ml-len(p)) for p in paths]
        coll_agents = set()
        for t in range(ml):
            seen={}
            for i,pa in enumerate(padded):
                if t<len(pa):
                    pos=pa[t]
                    if pos in seen:
                        j=seen[pos]
                        coll_agents.add(i); coll_agents.add(j)
                    else: seen[pos]=i
        for t in range(ml-1):
            for i in range(len(padded)):
                if t+1>=len(padded[i]): continue
                for j in range(i+1,len(padded)):
                    if t+1>=len(padded[j]): continue
                    if padded[i][t]==padded[j][t+1] and padded[i][t+1]==padded[j][t]:
                        coll_agents.add(i); coll_agents.add(j)
        
        sel = list(coll_agents) if len(coll_agents)<=nh_size else rng.sample(sorted(coll_agents), nh_size)
        non_coll = [i for i in range(len(paths)) if i not in coll_agents]
        if non_coll and len(sel)<nh_size:
            sel.extend(rng.sample(non_coll, min(nh_size-len(sel),len(non_coll))))
        
        fixed = [i for i in range(len(paths)) if i not in sel]
        max_pl = max(len(p) for p in paths)+20
        occ = {}
        for fi in fixed:
            p = paths[fi]
            for ts in range(len(p)): occ[(p[ts],ts)] = True
            for ts in range(len(p),max_pl): occ[(p[-1],ts)] = True
        
        prio = rng.sample(sel, len(sel))
        new_p = {}
        ok = True
        for idx in prio:
            cur_occ = dict(occ)
            for j,np_path in new_p.items():
                for ts in range(len(np_path)): cur_occ[(np_path[ts],ts)] = True
                for ts in range(len(np_path),max_pl): cur_occ[(np_path[-1],ts)] = True
            
            path = spacetime_astar(env, starts[idx], goals[idx], cur_occ, max_t=max_pl)
            if path is None: ok=False; break
            new_p[idx] = path
        
        if ok:
            new_plan = [list(p) for p in paths]
            for idx,p in new_p.items(): new_plan[idx]=p
            new_cp = count_cp(new_plan)
            if new_cp <= cp: paths = new_plan
    
    return paths, cp_hist

def rr_pp(env, starts, goals, max_r=5, ttl=30):
    """Random-restart PP."""
    t0=time.time()
    best=None; best_cp=999
    for _ in range(max_r):
        if time.time()-t0>ttl: break
        order=list(range(len(starts)))
        random.shuffle(order)
        remaining = ttl-(time.time()-t0)
        if remaining<1: break
        p=pp_solve(env,starts,goals,order,tl=min(8,remaining))
        if p is not None:
            cp=count_cp(p)
            if cp==0: return p
            if cp<best_cp: best_cp=cp; best=p
    return best

def marl_lns(env, starts, goals, marl_tl=8, lns_tl=25, seed=42):
    """MARL-LNS Hybrid."""
    t0=time.time()
    init = marl_policy(env, starts, goals, max_steps=128, tl=marl_tl)
    if init is None: return None,[],0
    n=len(init)
    nh=min(4,n)
    paths,cp_hist = lns_repair(env,starts,goals,init,max_iter=200,tl=lns_tl,nh_size=nh,seed=seed)
    elapsed=time.time()-t0
    return paths,cp_hist,elapsed