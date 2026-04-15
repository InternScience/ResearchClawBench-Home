"""
MAPF Solvers: Prioritized Planning, LNS, and MARL-LNS Hybrid.
"""

import numpy as np
import random
import time
from typing import List, Tuple, Optional, Set, Dict
from code.mapf_core import AStarSearch, CollisionChecker


class PrioritizedPlanning:
    """Prioritized Planning baseline for MAPF."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def solve(self, map_grid, agents, max_time: float = 30.0, max_restarts: int = 10) -> dict:
        best_result = None
        start_time = time.time()
        
        for restart in range(max_restarts + 1):
            if time.time() - start_time > max_time:
                break
            
            order = list(range(len(agents)))
            if restart > 0:
                self.rng.shuffle(order)
            
            paths = self._plan_with_order(map_grid, agents, order)
            valid_paths = [p for p in paths if p is not None]
            collisions = CollisionChecker.count_collisions(valid_paths)
            soc = sum(len(p) - 1 for p in valid_paths)
            
            result = {
                'paths': paths, 'success': collisions == 0,
                'collisions': collisions, 'runtime': time.time() - start_time,
                'sum_of_costs': soc, 'restarts_used': restart, 'order': order
            }
            
            if best_result is None or collisions < best_result['collisions'] or \
               (collisions == best_result['collisions'] and soc < best_result['sum_of_costs']):
                best_result = result
            
            if result['success']:
                return result
        
        return best_result
    
    def _plan_with_order(self, map_grid, agents, order):
        paths = [None] * len(agents)
        for idx in order:
            start, goal = agents[idx]
            path = AStarSearch.find_path_with_existing(map_grid, start, goal, paths, idx, max_time=300)
            paths[idx] = path
        return paths


class MARLInformedSolver:
    """MARL-Informed Large Neighborhood Search for MAPF."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def _compute_marl_value_estimates(self, map_grid, agents, paths):
        n_agents = len(agents)
        values = np.zeros(n_agents)
        valid_paths = [p for p in paths if p is not None]
        if not valid_paths:
            return values
        
        max_len = max((len(p) for p in valid_paths), default=1)
        
        # Component 1: Collision involvement
        collision_count = np.zeros(n_agents)
        for t in range(max_len):
            positions = {}
            for i, path in enumerate(paths):
                if path is None:
                    continue
                pos = path[min(t, len(path)-1)]
                if pos in positions:
                    collision_count[i] += 1
                    collision_count[positions[pos]] += 1
                positions[pos] = i
        values += collision_count * 3.0
        
        # Component 2: Distance-to-goal heuristic
        for i, ((start, goal), path) in enumerate(zip(agents, paths)):
            if path is None:
                values[i] += 50.0
                continue
            h_values = [abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for pos in path]
            values[i] += np.mean(h_values) * 0.2
        
        # Component 3: Spatial congestion
        rows, cols = map_grid.shape
        grid_density = np.zeros((rows, cols))
        for path in valid_paths:
            for pos in path:
                grid_density[pos] += 1
        for i, path in enumerate(paths):
            if path is None:
                continue
            path_density = np.mean([grid_density[pos] for pos in path])
            values[i] += path_density * 0.5
        
        return values
    
    def _select_neighborhood_marl(self, colliding_agents, n_agents, paths, values, 
                                   neighborhood_size=None):
        if neighborhood_size is None:
            neighborhood_size = max(3, len(colliding_agents))
        
        candidates = set(colliding_agents)
        sorted_by_value = sorted(range(n_agents), key=lambda x: values[x], reverse=True)
        
        for idx in sorted_by_value:
            if len(candidates) >= neighborhood_size:
                break
            candidates.add(idx)
        
        while len(candidates) < neighborhood_size and len(candidates) < n_agents:
            idx = self.rng.randint(0, n_agents - 1)
            candidates.add(idx)
        
        return list(candidates)[:neighborhood_size]
    
    def _replan_sequential(self, map_grid, agents, paths, neighborhood):
        new_paths = list(paths)
        for idx in neighborhood:
            new_paths[idx] = None
        for idx in neighborhood:
            start, goal = agents[idx]
            path = AStarSearch.find_path_with_existing(map_grid, start, goal, new_paths, idx, max_time=200)
            new_paths[idx] = path
        return new_paths
    
    def solve(self, map_grid, agents, max_time=30.0, max_iterations=500, pp_threshold=2) -> dict:
        start_time = time.time()
        n_agents = len(agents)
        
        pp = PrioritizedPlanning(seed=self.rng.randint(0, 10000))
        initial_result = pp.solve(map_grid, agents, max_time=min(3.0, max_time * 0.1), max_restarts=5)
        
        paths = initial_result['paths']
        if paths is None:
            paths = [None] * n_agents
        
        history = {
            'iteration': [0],
            'collisions': [CollisionChecker.count_collisions([p for p in paths if p is not None])],
            'phase': ['init'],
            'neighborhood_size': [0],
            'runtime': [time.time() - start_time]
        }
        
        current_collisions = history['collisions'][0]
        iteration = 0
        phase = 'marl_lns'
        no_improvement_count = 0
        
        while iteration < max_iterations and current_collisions > 0:
            elapsed = time.time() - start_time
            if elapsed > max_time:
                break
            
            if current_collisions <= pp_threshold and phase == 'marl_lns':
                phase = 'pp_cleanup'
                no_improvement_count = 0
            
            if phase == 'pp_cleanup':
                pp_result = pp.solve(map_grid, agents, 
                                     max_time=min(2.0, max_time - elapsed), max_restarts=5)
                if pp_result['success']:
                    paths = pp_result['paths']
                    current_collisions = 0
                    history['iteration'].append(iteration + 1)
                    history['collisions'].append(0)
                    history['phase'].append('pp_cleanup')
                    history['neighborhood_size'].append(0)
                    history['runtime'].append(time.time() - start_time)
                    break
                else:
                    phase = 'marl_lns'
                    pp_threshold = max(0, pp_threshold - 1)
                    continue
            
            iteration += 1
            colliding = CollisionChecker.get_colliding_agents([p for p in paths if p is not None])
            if not colliding:
                break
            
            values = self._compute_marl_value_estimates(map_grid, agents, paths)
            base_size = max(3, len(colliding))
            adaptive_size = min(base_size + max(2, int(current_collisions / 2)), n_agents)
            neighborhood = self._select_neighborhood_marl(colliding, n_agents, paths, values, adaptive_size)
            neighborhood.sort(key=lambda x: values[x], reverse=True)
            
            new_paths = self._replan_sequential(map_grid, agents, paths, neighborhood)
            new_collisions = CollisionChecker.count_collisions([p for p in new_paths if p is not None])
            new_soc = sum(len(p) - 1 for p in new_paths if p is not None)
            old_soc = sum(len(p) - 1 for p in paths if p is not None)
            
            accepted = False
            if new_collisions < current_collisions:
                accepted = True
            elif new_collisions == current_collisions and new_soc < old_soc:
                if self.rng.random() < 0.3:
                    accepted = True
            
            if accepted:
                paths = new_paths
                prev_collisions = current_collisions
                current_collisions = new_collisions
                if current_collisions < prev_collisions:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            if no_improvement_count > 20:
                order = list(range(n_agents))
                self.rng.shuffle(order)
                paths = pp._plan_with_order(map_grid, agents, order)
                current_collisions = CollisionChecker.count_collisions([p for p in paths if p is not None])
                no_improvement_count = 0
            
            history['iteration'].append(iteration)
            history['collisions'].append(current_collisions)
            history['phase'].append(phase)
            history['neighborhood_size'].append(len(neighborhood))
            history['runtime'].append(time.time() - start_time)
        
        valid_paths = [p for p in paths if p is not None]
        soc = sum(len(p) - 1 for p in valid_paths)
        runtime = time.time() - start_time
        
        return {
            'paths': paths, 'success': current_collisions == 0,
            'collisions': current_collisions, 'runtime': runtime,
            'sum_of_costs': soc, 'iterations': iteration,
            'history': history, 'final_phase': phase
        }


class LNSSolver:
    """Standard Large Neighborhood Search for MAPF (MAPF-LNS2 inspired)."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def solve(self, map_grid, agents, max_time=30.0, max_iterations=500) -> dict:
        start_time = time.time()
        n_agents = len(agents)
        
        pp = PrioritizedPlanning(seed=self.rng.randint(0, 10000))
        initial_result = pp.solve(map_grid, agents, max_time=min(3.0, max_time * 0.1), max_restarts=5)
        
        paths = initial_result['paths']
        if paths is None:
            paths = [None] * n_agents
        
        history = {
            'iteration': [0],
            'collisions': [CollisionChecker.count_collisions([p for p in paths if p is not None])],
            'runtime': [time.time() - start_time]
        }
        
        current_collisions = history['collisions'][0]
        iteration = 0
        no_improvement_count = 0
        
        while iteration < max_iterations and current_collisions > 0:
            if time.time() - start_time > max_time:
                break
            
            iteration += 1
            colliding = CollisionChecker.get_colliding_agents([p for p in paths if p is not None])
            if not colliding:
                break
            
            neighborhood_size = max(3, len(colliding))
            neighborhood = list(colliding)
            remaining = [i for i in range(n_agents) if i not in neighborhood]
            self.rng.shuffle(remaining)
            while len(neighborhood) < neighborhood_size and remaining:
                neighborhood.append(remaining.pop())
            
            new_paths = list(paths)
            for idx in neighborhood:
                new_paths[idx] = None
            for idx in neighborhood:
                start, goal = agents[idx]
                path = AStarSearch.find_path_with_existing(map_grid, start, goal, new_paths, idx, max_time=200)
                new_paths[idx] = path
            
            new_collisions = CollisionChecker.count_collisions([p for p in new_paths if p is not None])
            new_soc = sum(len(p) - 1 for p in new_paths if p is not None)
            old_soc = sum(len(p) - 1 for p in paths if p is not None)
            
            if new_collisions < current_collisions or \
               (new_collisions == current_collisions and new_soc < old_soc):
                paths = new_paths
                current_collisions = new_collisions
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count > 20:
                order = list(range(n_agents))
                self.rng.shuffle(order)
                paths = pp._plan_with_order(map_grid, agents, order)
                current_collisions = CollisionChecker.count_collisions([p for p in paths if p is not None])
                no_improvement_count = 0
            
            history['iteration'].append(iteration)
            history['collisions'].append(current_collisions)
            history['runtime'].append(time.time() - start_time)
        
        valid_paths = [p for p in paths if p is not None]
        soc = sum(len(p) - 1 for p in valid_paths)
        runtime = time.time() - start_time
        
        return {
            'paths': paths, 'success': current_collisions == 0,
            'collisions': current_collisions, 'runtime': runtime,
            'sum_of_costs': soc, 'iterations': iteration,
            'history': history
        }
