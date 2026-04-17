"""
MAPF Solvers: Prioritized Planning, MAPF-LNS2, MARL-LNS Hybrid
Improved version with better MARL integration and adaptive switching.
"""
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Set
from mapf_core import (MAPFGrid, astar, spacetime_astar, detect_collisions, 
                       count_collisions, get_colliding_agents, generate_agent_tasks)


class PrioritizedPlanning:
    """Prioritized Planning (PP) baseline."""
    def __init__(self, grid: MAPFGrid, max_time: int = 200):
        self.grid = grid
        self.max_time = max_time
    
    def solve(self, tasks: List[Tuple], priority_order: List[int] = None) -> Tuple[List[List[Tuple]], dict]:
        n = len(tasks)
        if priority_order is None:
            priority_order = list(range(n))
        
        paths = [None] * n
        constraints = {'vertex': set(), 'edge': set()}
        
        start_time = time.time()
        
        for idx in priority_order:
            start, goal = tasks[idx]
            path = spacetime_astar(self.grid, start, goal, constraints, self.max_time)
            if path is None:
                path = [start]
            paths[idx] = path
            
            for t, pos in enumerate(path):
                constraints['vertex'].add((pos, t))
            goal_pos = path[-1]
            for t in range(len(path), self.max_time):
                constraints['vertex'].add((goal_pos, t))
            for t in range(1, len(path)):
                constraints['edge'].add((path[t], path[t-1], t))
        
        elapsed = time.time() - start_time
        collisions = detect_collisions(paths)
        
        info = {
            'runtime': elapsed,
            'num_collisions': len(collisions),
            'success': len(collisions) == 0,
            'makespan': max(len(p) for p in paths) if paths else 0,
            'sum_of_costs': sum(len(p) for p in paths) if paths else 0,
        }
        return paths, info


class PPWithRandomRestarts:
    """Prioritized Planning with random restarts."""
    def __init__(self, grid: MAPFGrid, max_time: int = 200, max_restarts: int = 20, time_limit: float = 30.0):
        self.grid = grid
        self.max_time = max_time
        self.max_restarts = max_restarts
        self.time_limit = time_limit
        self.pp = PrioritizedPlanning(grid, max_time)
    
    def solve(self, tasks, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        
        start_time = time.time()
        best_paths = None
        best_collisions = float('inf')
        
        for restart in range(self.max_restarts):
            if time.time() - start_time > self.time_limit:
                break
            
            order = list(range(len(tasks)))
            rng.shuffle(order)
            paths, info = self.pp.solve(tasks, priority_order=order)
            
            if info['num_collisions'] < best_collisions:
                best_collisions = info['num_collisions']
                best_paths = paths
            
            if best_collisions == 0:
                break
        
        elapsed = time.time() - start_time
        info = {
            'runtime': elapsed,
            'num_collisions': best_collisions,
            'success': best_collisions == 0,
            'makespan': max(len(p) for p in best_paths) if best_paths else 0,
            'sum_of_costs': sum(len(p) for p in best_paths) if best_paths else 0,
        }
        return best_paths, info


def _select_neighborhood(paths, collisions, neighborhood_size):
    """Select agents to replan based on collision graph."""
    if not collisions:
        return []
    
    collision_count = {}
    collision_graph = {}
    for c in collisions:
        for a in c['agents']:
            collision_count[a] = collision_count.get(a, 0) + 1
        a1, a2 = c['agents']
        collision_graph.setdefault(a1, set()).add(a2)
        collision_graph.setdefault(a2, set()).add(a1)
    
    sorted_agents = sorted(collision_count.keys(), key=lambda a: -collision_count[a])
    seed = sorted_agents[0]
    neighborhood = {seed}
    queue = [seed]
    while len(neighborhood) < neighborhood_size and queue:
        agent = queue.pop(0)
        for neighbor in collision_graph.get(agent, set()):
            if neighbor not in neighborhood:
                neighborhood.add(neighbor)
                queue.append(neighbor)
                if len(neighborhood) >= neighborhood_size:
                    break
    return list(neighborhood)


def _replan_pp(grid, tasks, paths, agents_to_replan, rng, max_time):
    """Replan selected agents using Prioritized Planning."""
    new_paths = list(paths)
    
    constraints = {'vertex': set(), 'edge': set()}
    for i, path in enumerate(paths):
        if i not in agents_to_replan:
            for t, pos in enumerate(path):
                constraints['vertex'].add((pos, t))
            goal_pos = path[-1]
            for t in range(len(path), max_time):
                constraints['vertex'].add((goal_pos, t))
            for t in range(1, len(path)):
                constraints['edge'].add((path[t], path[t-1], t))
    
    order = list(agents_to_replan)
    rng.shuffle(order)
    
    local_constraints = {
        'vertex': set(constraints['vertex']),
        'edge': set(constraints['edge'])
    }
    
    for idx in order:
        start, goal = tasks[idx]
        path = spacetime_astar(grid, start, goal, local_constraints, max_time)
        if path is None:
            path = [start]
        new_paths[idx] = path
        
        for t, pos in enumerate(path):
            local_constraints['vertex'].add((pos, t))
        goal_pos = path[-1]
        for t in range(len(path), max_time):
            local_constraints['vertex'].add((goal_pos, t))
        for t in range(1, len(path)):
            local_constraints['edge'].add((path[t], path[t-1], t))
    
    return new_paths


class MAPFLNS2:
    """MAPF-LNS2: Large Neighborhood Search for MAPF."""
    def __init__(self, grid: MAPFGrid, max_time: int = 200, max_iterations: int = 100,
                 neighborhood_size: int = 5, time_limit: float = 30.0):
        self.grid = grid
        self.max_time = max_time
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size
        self.time_limit = time_limit
    
    def solve(self, tasks, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        
        start_time = time.time()
        
        # Initial paths using individual A*
        paths = []
        for start, goal in tasks:
            path = astar(self.grid, start, goal)
            if path is None:
                path = [start]
            paths.append(path)
        
        collisions = detect_collisions(paths)
        collision_history = [len(collisions)]
        best_paths = list(paths)
        best_collisions = len(collisions)
        
        iteration = 0
        while len(collisions) > 0 and iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                break
            
            neighborhood = _select_neighborhood(paths, collisions, self.neighborhood_size)
            if not neighborhood:
                break
            
            new_paths = _replan_pp(self.grid, tasks, paths, neighborhood, rng, self.max_time)
            new_collisions = detect_collisions(new_paths)
            
            if len(new_collisions) <= len(collisions):
                paths = new_paths
                collisions = new_collisions
                if len(collisions) < best_collisions:
                    best_paths = list(paths)
                    best_collisions = len(collisions)
            
            collision_history.append(len(collisions))
            iteration += 1
        
        elapsed = time.time() - start_time
        
        if best_collisions < len(collisions):
            paths = best_paths
            collisions = detect_collisions(paths)
        
        info = {
            'runtime': elapsed,
            'num_collisions': len(collisions),
            'success': len(collisions) == 0,
            'makespan': max(len(p) for p in paths) if paths else 0,
            'sum_of_costs': sum(len(p) for p in paths) if paths else 0,
            'iterations': iteration,
            'collision_history': collision_history,
        }
        return paths, info


class MARLPolicy:
    """
    MARL-style policy for MAPF.
    Uses learned heuristics based on local observations to guide path decisions.
    Simulates PRIMAL/SCRIMP-style decentralized policies.
    """
    def __init__(self, grid: MAPFGrid, fov_size: int = 7, seed: int = 42):
        self.grid = grid
        self.fov_size = fov_size
        self.rng = np.random.RandomState(seed)
        
        # Policy parameters (simulating trained weights)
        self.goal_weight = 4.0
        self.collision_weight = -6.0
        self.density_weight = -2.0
        self.progress_weight = 3.0
        self.history_weight = -1.0
    
    def score_action(self, pos, next_pos, goal, agent_positions, collision_map, visit_count):
        """Score a potential action based on MARL-style heuristics."""
        score = 0.0
        
        # Goal progress
        curr_dist = self.grid.manhattan_distance(pos, goal)
        next_dist = self.grid.manhattan_distance(next_pos, goal)
        score += self.progress_weight * (curr_dist - next_dist)
        
        # Direct collision avoidance
        if next_pos in agent_positions:
            score += self.collision_weight
        
        # Density penalty
        density = sum(1 for ap in agent_positions 
                     if self.grid.manhattan_distance(next_pos, ap) <= 2)
        score += self.density_weight * density * 0.15
        
        # Collision hotspot avoidance
        if next_pos in collision_map:
            score += self.collision_weight * 0.3 * collision_map[next_pos]
        
        # Anti-loop: penalize revisiting
        if next_pos in visit_count:
            score += self.history_weight * visit_count[next_pos]
        
        # Goal proximity bonus
        if next_pos == goal:
            score += 10.0
        
        return score
    
    def plan_path(self, start, goal, other_paths, collision_map, max_steps=100):
        """Plan a full path using the MARL policy with awareness of other paths over time."""
        path = [start]
        pos = start
        visit_count = {}
        
        for step in range(max_steps):
            if pos == goal:
                break
            
            visit_count[pos] = visit_count.get(pos, 0) + 1
            
            # Get agent positions at current timestep
            agent_positions = set()
            for op in other_paths:
                if step < len(op):
                    agent_positions.add(op[step])
                elif op:
                    agent_positions.add(op[-1])
            
            neighbors = self.grid.get_neighbors(pos)
            temp = 0.2 + 0.15 * min(visit_count.get(pos, 0), 8)
            
            scores = []
            for n in neighbors:
                s = self.score_action(pos, n, goal, agent_positions, collision_map, visit_count)
                scores.append(s)
            
            scores = np.array(scores)
            exp_scores = np.exp((scores - np.max(scores)) / max(temp, 0.01))
            probs = exp_scores / exp_scores.sum()
            
            idx = self.rng.choice(len(neighbors), p=probs)
            next_pos = neighbors[idx]
            path.append(next_pos)
            pos = next_pos
        
        return path


class MARL_LNS:
    """
    Hybrid MARL-LNS: Integrates MARL into LNS framework.
    - Early stages: Use MARL policy for neighborhood repair (better collision avoidance via learned heuristics)
    - Later stages: Switch to PP for efficient final refinement
    - Adaptive switching based on collision reduction progress
    """
    def __init__(self, grid: MAPFGrid, max_time: int = 200, max_iterations: int = 100,
                 neighborhood_size: int = 5, time_limit: float = 30.0,
                 switch_ratio: float = 0.5, marl_fov: int = 7):
        self.grid = grid
        self.max_time = max_time
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size
        self.time_limit = time_limit
        self.switch_ratio = switch_ratio
        self.marl_policy = MARLPolicy(grid, marl_fov)
    
    def _build_collision_map(self, paths):
        """Build a collision heatmap from existing paths."""
        collision_map = {}
        collisions = detect_collisions(paths)
        for c in collisions:
            loc = c['location']
            if isinstance(loc, tuple) and len(loc) == 2 and isinstance(loc[0], int):
                collision_map[loc] = collision_map.get(loc, 0) + 1
            elif isinstance(loc, tuple):
                for l in loc:
                    if isinstance(l, tuple):
                        collision_map[l] = collision_map.get(l, 0) + 1
        return collision_map
    
    def _replan_marl(self, tasks, paths, agents_to_replan, rng):
        """Replan selected agents using MARL policy with time-aware planning."""
        new_paths = list(paths)
        collision_map = self._build_collision_map(paths)
        
        # Get other agents' paths (not being replanned)
        other_paths = [paths[i] for i in range(len(paths)) if i not in agents_to_replan]
        
        # Plan each agent in the neighborhood
        replanned_paths = []
        for idx in agents_to_replan:
            start, goal = tasks[idx]
            
            # Include already replanned agents' paths
            all_other = other_paths + replanned_paths
            
            path = self.marl_policy.plan_path(start, goal, all_other, collision_map, self.max_time)
            new_paths[idx] = path
            replanned_paths.append(path)
        
        return new_paths
    
    def solve(self, tasks, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        
        start_time = time.time()
        
        # Initial paths using individual A*
        paths = []
        for start, goal in tasks:
            path = astar(self.grid, start, goal)
            if path is None:
                path = [start]
            paths.append(path)
        
        collisions = detect_collisions(paths)
        initial_collisions = max(len(collisions), 1)
        
        collision_history = [len(collisions)]
        phase_history = []
        best_paths = list(paths)
        best_collisions = len(collisions)
        
        # Adaptive switching: track progress
        stagnation_counter = 0
        last_improvement = len(collisions)
        
        iteration = 0
        while len(collisions) > 0 and iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                break
            
            # Determine phase based on collision ratio and progress
            collision_ratio = len(collisions) / initial_collisions
            
            # Use MARL when collision ratio is high (early phase)
            # Switch to PP when collisions are reduced enough
            # Also switch to PP if MARL is stagnating
            use_marl = (collision_ratio > self.switch_ratio) and (stagnation_counter < 5)
            
            neighborhood = _select_neighborhood(paths, collisions, self.neighborhood_size)
            if not neighborhood:
                break
            
            if use_marl:
                new_paths = self._replan_marl(tasks, paths, neighborhood, rng)
                phase_history.append('MARL')
            else:
                new_paths = _replan_pp(self.grid, tasks, paths, neighborhood, rng, self.max_time)
                phase_history.append('PP')
            
            new_collisions = detect_collisions(new_paths)
            
            # Accept improvement
            if len(new_collisions) <= len(collisions):
                if len(new_collisions) < len(collisions):
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                paths = new_paths
                collisions = new_collisions
                if len(collisions) < best_collisions:
                    best_paths = list(paths)
                    best_collisions = len(collisions)
            else:
                stagnation_counter += 1
                # During MARL phase, accept with small probability for exploration
                if use_marl and rng.random() < 0.05:
                    paths = new_paths
                    collisions = new_collisions
            
            collision_history.append(len(collisions))
            iteration += 1
        
        elapsed = time.time() - start_time
        
        if best_collisions < len(collisions):
            paths = best_paths
            collisions = detect_collisions(paths)
        
        info = {
            'runtime': elapsed,
            'num_collisions': len(collisions),
            'success': len(collisions) == 0,
            'makespan': max(len(p) for p in paths) if paths else 0,
            'sum_of_costs': sum(len(p) for p in paths) if paths else 0,
            'iterations': iteration,
            'collision_history': collision_history,
            'phase_history': phase_history,
            'initial_collisions': initial_collisions,
            'marl_iterations': sum(1 for p in phase_history if p == 'MARL'),
            'pp_iterations': sum(1 for p in phase_history if p == 'PP'),
        }
        return paths, info


class PureMARLSolver:
    """Pure MARL solver (no LNS) for comparison."""
    def __init__(self, grid: MAPFGrid, max_time: int = 200, max_restarts: int = 10):
        self.grid = grid
        self.max_time = max_time
        self.max_restarts = max_restarts
    
    def solve(self, tasks, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        
        start_time = time.time()
        best_paths = None
        best_collisions = float('inf')
        
        for restart in range(self.max_restarts):
            marl_policy = MARLPolicy(self.grid, seed=42 + restart)
            paths = []
            collision_map = {}
            
            # Plan sequentially with awareness of previous paths
            for idx, (start, goal) in enumerate(tasks):
                path = marl_policy.plan_path(start, goal, paths, collision_map, self.max_time)
                paths.append(path)
            
            n_collisions = count_collisions(paths)
            if n_collisions < best_collisions:
                best_collisions = n_collisions
                best_paths = paths
            
            if best_collisions == 0:
                break
        
        elapsed = time.time() - start_time
        
        info = {
            'runtime': elapsed,
            'num_collisions': best_collisions,
            'success': best_collisions == 0,
            'makespan': max(len(p) for p in best_paths) if best_paths else 0,
            'sum_of_costs': sum(len(p) for p in best_paths) if best_paths else 0,
        }
        return best_paths, info
