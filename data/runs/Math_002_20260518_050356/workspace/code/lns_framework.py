"""
Large Neighborhood Search (LNS) Framework for MAPF
Implements the core LNS algorithm with support for different neighborhood selection strategies.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Set
import random
import time
from mapf_core import (
    MAPFInstance, Path, Solution, 
    count_all_collisions, AStarPlanner
)


class LNS:
    """Large Neighborhood Search for MAPF."""
    
    def __init__(self, instance: MAPFInstance, max_iterations: int = 1000,
                 timeout: float = 30.0):
        self.instance = instance
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.planner = AStarPlanner(instance)
    
    def solve(self, initial_paths: Optional[List[Path]] = None,
              neighborhood_selector: Callable = None,
              neighborhood_size: int = 10) -> Tuple[Optional[List[Path]], dict]:
        start_time = time.time()
        
        if initial_paths is None:
            paths = self._generate_initial_paths()
        else:
            paths = [Path(list(p.positions)) for p in initial_paths]
        
        if paths is None:
            return None, {"success": False, "error": "Failed to generate initial paths",
                         "final_collisions": float('inf'), "final_soc": float('inf'),
                         "runtime": time.time() - start_time}
        
        stats = {
            "iterations": 0,
            "collisions_history": [],
            "soc_history": [],
            "time_history": [],
            "success": False
        }
        
        current_collisions = count_all_collisions(paths, self.instance)
        stats["collisions_history"].append(current_collisions)
        stats["soc_history"].append(sum(len(p) for p in paths))
        
        for iteration in range(self.max_iterations):
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                break
            
            if current_collisions == 0:
                stats["success"] = True
                stats["iterations"] = iteration
                stats["time"] = elapsed
                stats["final_collisions"] = 0
                stats["final_soc"] = sum(len(p) for p in paths)
                stats["runtime"] = elapsed
                return paths, stats
            
            # Select neighborhood
            if neighborhood_selector:
                colliding_agents = self._get_colliding_agents(paths)
                if not colliding_agents:
                    colliding_agents = list(range(self.instance.agent_count))
                neighborhood = neighborhood_selector(
                    self.instance, paths, colliding_agents, neighborhood_size
                )
            else:
                neighborhood = self._select_random_neighborhood(paths, neighborhood_size)
            
            if not neighborhood:
                continue
            
            # Repair neighborhood using prioritized planning
            new_paths = self._repair_neighborhood(paths, neighborhood)
            
            if new_paths is not None:
                new_collisions = count_all_collisions(new_paths, self.instance)
                
                if new_collisions < current_collisions or random.random() < 0.1:
                    paths = new_paths
                    current_collisions = new_collisions
            
            elapsed = time.time() - start_time
            stats["iterations"] = iteration + 1
            stats["collisions_history"].append(current_collisions)
            stats["soc_history"].append(sum(len(p) for p in paths))
            stats["time_history"].append(elapsed)
        
        stats["time"] = time.time() - start_time
        stats["final_collisions"] = count_all_collisions(paths, self.instance)
        stats["final_soc"] = sum(len(p) for p in paths)
        stats["runtime"] = stats["time"]
        return paths, stats
    
    def _generate_initial_paths(self) -> Optional[List[Path]]:
        paths = []
        planner = AStarPlanner(self.instance)
        
        for i in range(self.instance.agent_count):
            start = self.instance.starts[i]
            goal = self.instance.goals[i]
            path = planner.plan(start, goal, time_limit=100)
            
            if path is None:
                path = self._random_walk(start, goal)
            
            if path is None:
                return None
            
            paths.append(path)
        
        return paths
    
    def _random_walk(self, start, goal, max_steps=100):
        path = [start]
        current = start
        
        for _ in range(max_steps):
            if current == goal:
                return Path(path)
            
            neighbors = self.instance.get_neighbors(current)
            if not neighbors:
                break
            
            if random.random() < 0.3:
                best = min(neighbors, key=lambda p: 
                          abs(p[0] - goal[0]) + abs(p[1] - goal[1]))
                current = best
            else:
                current = random.choice(neighbors)
            
            path.append(current)
        
        if current == goal:
            return Path(path)
        return None
    
    def _select_random_neighborhood(self, paths, k):
        colliding = self._get_colliding_agents(paths)
        if not colliding:
            colliding = list(range(self.instance.agent_count))
        
        k = min(k, len(colliding))
        return random.sample(colliding, k)
    
    def _get_colliding_agents(self, paths):
        colliding = set()
        max_time = max(len(p) for p in paths) if paths else 0
        
        for t in range(max_time):
            positions_t = {}
            for i, path in enumerate(paths):
                if t < len(path):
                    pos = path[t]
                    if pos not in positions_t:
                        positions_t[pos] = []
                    positions_t[pos].append(i)
            
            for pos, agents in positions_t.items():
                if len(agents) > 1:
                    colliding.update(agents)
            
            if t + 1 < max_time:
                for i, path_i in enumerate(paths):
                    if t + 1 < len(path_i):
                        pos_i_t = path_i[t]
                        pos_i_next = path_i[t + 1]
                        for j, path_j in enumerate(paths):
                            if i < j and t + 1 < len(path_j):
                                pos_j_t = path_j[t]
                                pos_j_next = path_j[t + 1]
                                if pos_i_t == pos_j_next and pos_j_t == pos_i_next:
                                    colliding.add(i)
                                    colliding.add(j)
        
        return list(colliding)
    
    def _repair_neighborhood(self, paths, neighborhood):
        """Repair selected neighborhood using prioritized planning."""
        new_paths = [Path(list(p.positions)) for p in paths]
        
        fixed_agents = [i for i in range(self.instance.agent_count) 
                       if i not in neighborhood]
        
        # Create constraints from fixed paths
        constraints = set()
        for agent_id in fixed_agents:
            path = new_paths[agent_id]
            for t, pos in enumerate(path):
                constraints.add((t, pos))
        
        # Plan paths for neighborhood agents using prioritized planning
        planned_paths = {}
        for agent_id in neighborhood:
            start = self.instance.starts[agent_id]
            goal = self.instance.goals[agent_id]
            
            # Create constraints for this agent (fixed + previously planned neighborhood)
            agent_constraints = set(constraints)
            for prev_id, prev_path in planned_paths.items():
                for t, pos in enumerate(prev_path):
                    agent_constraints.add((t, pos))
                    if t + 1 < len(prev_path):
                        next_pos = prev_path[t + 1]
                        agent_constraints.add((t, (pos, next_pos)))
            
            path = self.planner.plan(start, goal, agent_constraints, time_limit=100)
            
            if path is None:
                return None
            
            for t, pos in enumerate(path):
                constraints.add((t, pos))
            
            planned_paths[agent_id] = path
            new_paths[agent_id] = path
        
        return new_paths


class MARLGuidedLNS(LNS):
    """LNS with MARL-guided neighborhood selection."""
    
    def __init__(self, instance, marl_agent, max_iterations=1000,
                 timeout=30.0, marl_ratio=0.7):
        super().__init__(instance, max_iterations, timeout)
        self.marl_agent = marl_agent
        self.marl_ratio = marl_ratio
    
    def solve(self, initial_paths=None, neighborhood_size=10):
        selector = lambda inst, paths, agents, k: \
            self.marl_agent.select_neighborhood_with_random(
                inst, paths, agents, k, self.marl_ratio
            )
        
        return super().solve(initial_paths, selector, neighborhood_size)
