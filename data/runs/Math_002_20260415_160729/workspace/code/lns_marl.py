"""
Large Neighborhood Search with MARL-inspired guidance for MAPF
"""
import numpy as np
import random
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict
import heapq

from mapf_utils import (
    MAPFMap, MAPFInstance, Path, 
    detect_collision, count_collisions, get_colliding_agents,
    space_time_a_star, compute_solution_cost, makespan
)


class LNS_MARL:
    """
    Hybrid LNS with MARL-inspired guidance for MAPF.
    
    Combines:
    - Prioritized Planning for fast initial solutions
    - Large Neighborhood Search for repair
    - MARL-inspired heuristics for neighborhood selection and path guidance
    """
    
    def __init__(self, instance: MAPFInstance, seed: Optional[int] = None):
        self.instance = instance
        self.num_agents = instance.num_agents
        self.map = instance.map
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # MARL-inspired agent features (simulated)
        self.agent_q_values = defaultdict(lambda: defaultdict(float))
        self.agent_visit_counts = defaultdict(lambda: defaultdict(int))
        self.exploration_bonus = 0.1
        
    def initialize_solution(self) -> List[Path]:
        """
        Initialize solution using Prioritized Planning with MARL-inspired ordering.
        Agents with more difficult paths (longer distance to goal) get higher priority.
        """
        # Compute difficulty scores (distance to goal)
        difficulties = []
        for i in range(self.num_agents):
            start = self.instance.starts[i]
            goal = self.instance.goals[i]
            dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            difficulties.append((dist, i))
        
        # Sort by difficulty (harder first - MARL-inspired priority)
        difficulties.sort(reverse=True)
        priority_order = [i for _, i in difficulties]
        
        # Run prioritized planning
        solution = self._prioritized_planning(priority_order)
        return solution
    
    def _prioritized_planning(self, priority_order: List[int]) -> List[Path]:
        """Prioritized Planning with collision avoidance."""
        solution = []
        
        for agent_id in priority_order:
            start = self.instance.starts[agent_id]
            goal = self.instance.goals[agent_id]
            
            # Try to find collision-free path
            path = space_time_a_star(self.map, start, goal, solution)
            
            if path is None:
                # Fallback: simple path without collision avoidance
                path = self._simple_path(start, goal)
            
            solution.append(path)
        
        # Reorder to original agent IDs
        reordered = [None] * self.num_agents
        for i, agent_id in enumerate(priority_order):
            reordered[agent_id] = solution[i]
        
        return reordered
    
    def _simple_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Path:
        """Create a simple path using greedy approach."""
        if start == goal:
            return Path([start])
        
        # BFS for shortest path
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            pos, path = queue.pop(0)
            
            if pos == goal:
                return Path(path)
            
            for neighbor in self.map.get_neighbors(pos):
                if neighbor != pos and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        # If no path found, return direct path
        return Path([start, goal])
    
    def select_neighborhood(self, solution: List[Path], size: int) -> Set[int]:
        """
        Select a neighborhood of agents to replan using MARL-inspired strategy.
        Prioritizes agents involved in collisions and their neighbors.
        """
        colliding = get_colliding_agents(solution)
        
        if not colliding:
            # Random selection if no collisions
            return set(random.sample(range(self.num_agents), min(size, self.num_agents)))
        
        # Start with colliding agents
        neighborhood = set(colliding)
        
        # Add neighboring agents (agents that might interact with colliding ones)
        for agent in list(neighborhood):
            path = solution[agent]
            for t, pos in enumerate(path.positions):
                for other in range(self.num_agents):
                    if other != agent and other not in neighborhood:
                        other_pos = solution[other].get_position(t)
                        # If close in space-time, they might interact
                        if abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1]) <= 2:
                            neighborhood.add(other)
                            break
        
        # If neighborhood too small, add random agents
        available = set(range(self.num_agents)) - neighborhood
        while len(neighborhood) < size and available:
            neighborhood.add(available.pop())
        
        # If too large, prioritize by collision count
        if len(neighborhood) > size:
            # Count collisions per agent
            collision_counts = defaultdict(int)
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    if detect_collision(solution[i], solution[j], i, j) is not None:
                        collision_counts[i] += 1
                        collision_counts[j] += 1
            
            sorted_agents = sorted(neighborhood, key=lambda a: collision_counts[a], reverse=True)
            neighborhood = set(sorted_agents[:size])
        
        return neighborhood
    
    def replan_neighborhood(self, solution: List[Path], neighborhood: Set[int]) -> List[Path]:
        """
        Replan paths for agents in the neighborhood.
        Uses MARL-inspired guidance for path selection.
        """
        neighborhood = list(neighborhood)
        new_solution = solution.copy()
        
        # Get paths of agents outside neighborhood (fixed)
        fixed_paths = [solution[i] for i in range(self.num_agents) if i not in neighborhood]
        
        # Sort neighborhood agents by MARL-inspired priority
        # Agents with more collisions and longer paths get priority
        priorities = []
        for agent in neighborhood:
            collisions = sum(1 for j in range(self.num_agents) 
                           if j != agent and detect_collision(solution[agent], solution[j], agent, j) is not None)
            path_len = len(solution[agent])
            priorities.append((collisions, path_len, agent))
        
        priorities.sort(reverse=True)
        
        # Replan each agent in priority order
        for _, _, agent in priorities:
            start = self.instance.starts[agent]
            goal = self.instance.goals[agent]
            
            # Add other neighborhood agents that have been replanned
            other_paths = fixed_paths.copy()
            for other_agent in neighborhood:
                if other_agent != agent:
                    other_paths.append(new_solution[other_agent])
            
            # Try to find collision-free path with MARL exploration bonus
            path = self._st_a_star_with_exploration(start, goal, other_paths, agent)
            
            if path is None:
                # Fallback to simple path
                path = self._simple_path(start, goal)
            
            new_solution[agent] = path
        
        return new_solution
    
    def _st_a_star_with_exploration(self, start: Tuple[int, int], goal: Tuple[int, int],
                                    other_paths: List[Path], agent_id: int) -> Optional[Path]:
        """
        Space-Time A* with MARL-inspired exploration bonus.
        Encourages visiting less-visited states.
        """
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        def has_collision(pos, time):
            for path in other_paths:
                if path.get_position(time) == pos:
                    return True
            return False
        
        def get_exploration_bonus(pos, time):
            # MARL-inspired bonus for less-visited states
            state_key = (pos, time % 10)  # Time abstraction
            visits = self.agent_visit_counts[agent_id][state_key]
            return self.exploration_bonus / (1 + visits)
        
        # (f_score, g_score, position, path)
        open_set = [(heuristic(start), 0, start, [start])]
        closed_set = set()
        max_iterations = 10000
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            f, g, pos, path = heapq.heappop(open_set)
            
            if pos == goal:
                return Path(path)
            
            state = (pos, g)
            if state in closed_set:
                continue
            closed_set.add(state)
            
            # Update visit count
            state_key = (pos, g % 10)
            self.agent_visit_counts[agent_id][state_key] += 1
            
            if g >= 100:  # Max time limit
                continue
            
            for neighbor in self.map.get_neighbors(pos):
                new_g = g + 1
                new_path = path + [neighbor]
                
                if not has_collision(neighbor, new_g):
                    exploration_bonus = get_exploration_bonus(neighbor, new_g)
                    new_f = new_g + heuristic(neighbor) - exploration_bonus
                    heapq.heappush(open_set, (new_f, new_g, neighbor, new_path))
        
        return None
    
    def solve(self, max_iterations: int = 100, neighborhood_size: int = 5, 
              time_limit: float = 60.0) -> Tuple[List[Path], Dict]:
        """
        Solve MAPF using LNS with MARL guidance.
        
        Returns:
            solution: List of paths
            stats: Dictionary with solving statistics
        """
        import time
        start_time = time.time()
        
        # Initialize
        solution = self.initialize_solution()
        initial_collisions = count_collisions(solution)
        initial_cost = compute_solution_cost(solution)
        
        stats = {
            'initial_collisions': initial_collisions,
            'initial_cost': initial_cost,
            'iterations': 0,
            'time': 0.0,
            'final_collisions': initial_collisions,
            'final_cost': initial_cost,
            'success': initial_collisions == 0
        }
        
        best_solution = solution
        best_collisions = initial_collisions
        best_cost = initial_cost
        
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Select neighborhood
            neighborhood_size = min(neighborhood_size + iteration // 10, self.num_agents // 2 + 1)
            neighborhood = self.select_neighborhood(solution, neighborhood_size)
            
            # Replan neighborhood
            new_solution = self.replan_neighborhood(solution, neighborhood)
            new_collisions = count_collisions(new_solution)
            new_cost = compute_solution_cost(new_solution)
            
            # Accept if better (fewer collisions or same collisions but lower cost)
            if new_collisions < best_collisions or (new_collisions == best_collisions and new_cost < best_cost):
                best_solution = new_solution
                best_collisions = new_collisions
                best_cost = new_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            solution = new_solution
            
            # Adaptive neighborhood size based on progress
            if no_improvement_count > 10:
                neighborhood_size = min(neighborhood_size + 2, self.num_agents)
                no_improvement_count = 0
            
            # Early termination if collision-free
            if best_collisions == 0:
                break
        
        stats['iterations'] = iteration + 1
        stats['time'] = time.time() - start_time
        stats['final_collisions'] = best_collisions
        stats['final_cost'] = best_cost
        stats['success'] = best_collisions == 0
        stats['makespan'] = makespan(best_solution)
        
        return best_solution, stats


class BaselinePP:
    """Baseline Prioritized Planning algorithm."""
    
    def __init__(self, instance: MAPFInstance):
        self.instance = instance
        self.map = instance.map
        self.num_agents = instance.num_agents
    
    def solve(self) -> Tuple[List[Path], Dict]:
        """Solve using simple prioritized planning."""
        import time
        start_time = time.time()
        
        solution = []
        
        for agent_id in range(self.num_agents):
            start = self.instance.starts[agent_id]
            goal = self.instance.goals[agent_id]
            
            path = space_time_a_star(self.map, start, goal, solution)
            
            if path is None:
                # Fallback
                path = self._simple_path(start, goal)
            
            solution.append(path)
        
        elapsed = time.time() - start_time
        
        stats = {
            'time': elapsed,
            'collisions': count_collisions(solution),
            'cost': compute_solution_cost(solution),
            'success': count_collisions(solution) == 0,
            'makespan': makespan(solution)
        }
        
        return solution, stats
    
    def _simple_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Path:
        """Create a simple path using BFS."""
        if start == goal:
            return Path([start])
        
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            pos, path = queue.pop(0)
            
            if pos == goal:
                return Path(path)
            
            for neighbor in self.map.get_neighbors(pos):
                if neighbor != pos and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return Path([start, goal])
