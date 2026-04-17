#!/usr/bin/env python3
"""
Hybrid MAPF Solver: MARL-LNS-PP
Integrates Multi-Agent Reinforcement Learning into Large Neighborhood Search framework,
with Prioritized Planning for efficient finalization.
"""

import numpy as np
import heapq
import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class Agent:
    """Represents a single agent in the MAPF problem."""
    id: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    path: List[Tuple[int, int]] = None
    
@dataclass
class MAPFInstance:
    """Represents a MAPF instance with grid map and agents."""
    grid: np.ndarray  # 0 = free, -1 = obstacle
    agents: List[Agent]
    height: int
    width: int
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not obstacle)."""
        x, y = pos
        return (0 <= x < self.height and 
                0 <= y < self.width and 
                self.grid[x, y] != -1)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (including wait action)."""
        x, y = pos
        neighbors = [(x, y)]  # Wait action
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (x + dx, y + dy)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors

class CollisionDetector:
    """Detects vertex and edge (swapping) collisions between paths."""
    
    @staticmethod
    def detect_collisions(paths: List[List[Tuple[int, int]]]) -> List[Dict]:
        """
        Detect all collisions in a set of paths.
        Returns list of collision dictionaries with type, agents, location, timestep.
        """
        collisions = []
        
        # Build occupancy map: (timestep, x, y) -> list of agent ids
        occupancy = defaultdict(list)
        # Build edge occupancy: (timestep, x1, y1, x2, y2) -> agent id
        edge_occupancy = {}
        
        for agent_id, path in enumerate(paths):
            for t, pos in enumerate(path):
                key = (t, pos[0], pos[1])
                occupancy[key].append(agent_id)
                
                if t > 0:
                    prev_pos = path[t-1]
                    edge_key = (t-1, prev_pos[0], prev_pos[1], pos[0], pos[1])
                    edge_occupancy[edge_key] = agent_id
        
        # Check vertex collisions
        for key, agents in occupancy.items():
            if len(agents) > 1:
                t, x, y = key
                for i in range(len(agents)):
                    for j in range(i+1, len(agents)):
                        collisions.append({
                            'type': 'vertex',
                            'agents': (agents[i], agents[j]),
                            'location': (x, y),
                            'timestep': t
                        })
        
        # Check edge (swapping) collisions
        for edge_key, agent1 in edge_occupancy.items():
            t, x1, y1, x2, y2 = edge_key
            reverse_edge = (t, x2, y2, x1, y1)
            if reverse_edge in edge_occupancy:
                agent2 = edge_occupancy[reverse_edge]
                if agent1 < agent2:  # Avoid duplicates
                    collisions.append({
                        'type': 'edge',
                        'agents': (agent1, agent2),
                        'location': ((x1, y1), (x2, y2)),
                        'timestep': t
                    })
        
        return collisions

class SpaceTimeAStar:
    """Space-Time A* for single-agent pathfinding with dynamic obstacles."""
    
    def __init__(self, instance: MAPFInstance):
        self.instance = instance
        
    def find_path(self, agent: Agent, constraints: Set = None, 
                  other_paths: Dict[int, List[Tuple[int, int]]] = None,
                  max_timesteps: int = 512) -> Optional[List[Tuple[int, int]]]:
        """
        Find path for agent avoiding constraints and other agents' paths.
        
        Args:
            agent: The agent to plan for
            constraints: Set of (timestep, x, y) that are forbidden
            other_paths: Dict mapping agent_id to their paths
            max_timesteps: Maximum path length
        
        Returns:
            Path as list of (x, y) positions, or None if no path found
        """
        constraints = constraints or set()
        other_paths = other_paths or {}
        
        start = agent.start
        goal = agent.goal
        
        # Priority queue: (f_score, g_score, timestep, position, path)
        # f = g + heuristic
        heuristic = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        open_set = [(heuristic, 0, 0, start, [start])]
        
        # Closed set: (timestep, position) -> best g_score
        closed = {}
        
        while open_set:
            f, g, t, pos, path = heapq.heappop(open_set)
            
            if pos == goal:
                return path
            
            state = (t, pos)
            if state in closed and closed[state] <= g:
                continue
            closed[state] = g
            
            if t >= max_timesteps:
                continue
            
            for next_pos in self.instance.get_neighbors(pos):
                next_t = t + 1
                
                # Check constraints
                if (next_t, next_pos[0], next_pos[1]) in constraints:
                    continue
                
                # Check collisions with other paths
                collision = False
                for other_agent_id, other_path in other_paths.items():
                    if other_agent_id == agent.id:
                        continue
                    if next_t < len(other_path):
                        other_pos = other_path[next_t]
                        # Vertex collision
                        if other_pos == next_pos:
                            collision = True
                            break
                        # Edge collision (swapping)
                        if t < len(other_path) and other_path[t] == next_pos and other_pos == pos:
                            collision = True
                            break
                
                if collision:
                    continue
                
                new_g = g + 1
                h = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                new_f = new_g + h
                
                new_state = (next_t, next_pos)
                if new_state not in closed or closed[new_state] > new_g:
                    new_path = path + [next_pos]
                    heapq.heappush(open_set, (new_f, new_g, next_t, next_pos, new_path))
        
        return None

class PrioritizedPlanning:
    """Prioritized Planning baseline for MAPF."""
    
    def __init__(self, instance: MAPFInstance):
        self.instance = instance
        self.planner = SpaceTimeAStar(instance)
        
    def solve(self, priority_order: List[int] = None) -> Tuple[Optional[List[List[Tuple[int, int]]]], float]:
        """
        Solve MAPF using prioritized planning.
        
        Args:
            priority_order: Order in which to plan for agents (by agent id)
                           If None, uses agent id order
        
        Returns:
            Tuple of (paths list or None if failed, runtime)
        """
        start_time = time.time()
        
        if priority_order is None:
            priority_order = list(range(len(self.instance.agents)))
        
        paths = {}
        constraints = set()
        
        for agent_id in priority_order:
            agent = self.instance.agents[agent_id]
            path = self.planner.find_path(agent, constraints, paths)
            
            if path is None:
                return None, time.time() - start_time
            
            paths[agent_id] = path
            
            # Add constraints for lower priority agents
            for t, pos in enumerate(path):
                constraints.add((t, pos[0], pos[1]))
            # Agent stays at goal
            goal = agent.goal
            for t in range(len(path), len(path) + 100):  # Extended wait
                constraints.add((t, goal[0], goal[1]))
        
        # Convert to ordered list
        result_paths = [paths[i] for i in range(len(self.instance.agents))]
        return result_paths, time.time() - start_time

class MARLGuidedPlanner:
    """
    Simplified MARL-inspired planner that uses learned-style heuristics
    for collision avoidance. In a full implementation, this would use
    neural networks trained via reinforcement learning.
    
    This implementation uses:
    1. Potential field-based guidance to avoid congested areas
    2. Cooperative path selection that considers other agents' likely paths
    3. Imitation of expert behavior (space-time A* with cooperation)
    """
    
    def __init__(self, instance: MAPFInstance):
        self.instance = instance
        self.planner = SpaceTimeAStar(instance)
        
    def compute_congestion_map(self, agents: List[Agent]) -> np.ndarray:
        """Compute a congestion heatmap based on agent start/goal positions."""
        congestion = np.zeros_like(self.instance.grid, dtype=float)
        
        for agent in agents:
            # Add potential at start and goal
            congestion[agent.start] += 1.0
            congestion[agent.goal] += 1.0
            
        # Smooth the congestion map
        from scipy.ndimage import gaussian_filter
        congestion = gaussian_filter(congestion, sigma=2.0)
        
        return congestion
    
    def find_cooperative_path(self, agent: Agent, other_agents: List[Agent],
                              congestion_map: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Find path that considers congestion and other agents.
        Uses modified cost function that penalizes congested areas.
        """
        start = agent.start
        goal = agent.goal
        
        # Priority queue with congestion-aware costs
        heuristic = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        # Initial cost includes congestion at start
        initial_cost = 1.0 + congestion_map[start] * 0.5
        open_set = [(heuristic + initial_cost, initial_cost, 0, start, [start])]
        
        closed = {}
        max_timesteps = 256
        
        while open_set:
            f, g, t, pos, path = heapq.heappop(open_set)
            
            if pos == goal:
                return path
            
            state = (t, pos)
            if state in closed and closed[state] <= g:
                continue
            closed[state] = g
            
            if t >= max_timesteps:
                continue
            
            for next_pos in self.instance.get_neighbors(pos):
                next_t = t + 1
                
                # Cost includes congestion penalty
                congestion_penalty = congestion_map[next_pos] * 0.3
                new_g = g + 1.0 + congestion_penalty
                
                h = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                new_f = new_g + h
                
                new_state = (next_t, next_pos)
                if new_state not in closed or closed[new_state] > new_g:
                    new_path = path + [next_pos]
                    heapq.heappush(open_set, (new_f, new_g, next_t, next_pos, new_path))
        
        return None
    
    def generate_initial_solution(self) -> Tuple[Optional[List[List[Tuple[int, int]]]], float]:
        """
        Generate initial solution using MARL-inspired cooperative planning.
        Plans for all agents considering global congestion.
        """
        start_time = time.time()
        
        # Compute global congestion map
        congestion_map = self.compute_congestion_map(self.instance.agents)
        
        # Plan for each agent considering congestion
        paths = {}
        planned_paths = []
        
        # Sort agents by distance (shorter paths first - similar to priority)
        agent_order = sorted(
            range(len(self.instance.agents)),
            key=lambda i: abs(self.instance.agents[i].start[0] - self.instance.agents[i].goal[0]) +
                         abs(self.instance.agents[i].start[1] - self.instance.agents[i].goal[1])
        )
        
        for idx, agent_id in enumerate(agent_order):
            agent = self.instance.agents[agent_id]
            
            # Update congestion with already planned paths
            for path in planned_paths:
                for pos in path:
                    congestion_map[pos] += 0.5
            
            path = self.find_cooperative_path(agent, 
                                              [self.instance.agents[i] for i in agent_order if i != agent_id],
                                              congestion_map)
            
            if path is None:
                # Fallback to standard space-time A*
                other_paths = {i: planned_paths[j] for j, i in enumerate(agent_order[:idx])}
                path = self.planner.find_path(agent, other_paths=other_paths)
            
            if path is None:
                return None, time.time() - start_time
            
            paths[agent_id] = path
            planned_paths.append(path)
        
        # Convert to ordered list
        result_paths = [paths[i] for i in range(len(self.instance.agents))]
        return result_paths, time.time() - start_time

class LargeNeighborhoodSearch:
    """
    Large Neighborhood Search framework for MAPF.
    Iteratively selects subsets of agents and replans their paths.
    """
    
    def __init__(self, instance: MAPFInstance, 
                 initial_solver: str = 'marl',
                 repair_solver: str = 'pp'):
        self.instance = instance
        self.collision_detector = CollisionDetector()
        self.initial_solver = initial_solver
        self.repair_solver = repair_solver
        
        if initial_solver == 'marl':
            self.initial_planner = MARLGuidedPlanner(instance)
        else:
            self.initial_planner = PrioritizedPlanning(instance)
        
        self.pp_planner = PrioritizedPlanning(instance)
        self.max_iterations = 50
        self.neighborhood_size = 3  # Number of agents to replan per iteration
        
    def select_neighborhood(self, paths: List[List[Tuple[int, int]]], 
                           collisions: List[Dict]) -> Set[int]:
        """
        Select subset of agents to replan (neighborhood).
        Prioritizes agents involved in collisions.
        """
        # Count collisions per agent
        collision_count = defaultdict(int)
        for collision in collisions:
            for agent_id in collision['agents']:
                collision_count[agent_id] += 1
        
        # Select agents with most collisions
        sorted_agents = sorted(collision_count.keys(), 
                               key=lambda x: collision_count[x], 
                               reverse=True)
        
        neighborhood = set(sorted_agents[:self.neighborhood_size])
        
        # If not enough colliding agents, add random agents
        all_agents = set(range(len(self.instance.agents)))
        while len(neighborhood) < min(self.neighborhood_size, len(all_agents)):
            remaining = list(all_agents - neighborhood)
            if remaining:
                neighborhood.add(random.choice(remaining))
            else:
                break
        
        return neighborhood
    
    def repair_paths(self, paths: List[List[Tuple[int, int]]], 
                     neighborhood: Set[int]) -> Optional[List[List[Tuple[int, int]]]]:
        """
        Repair paths for agents in neighborhood using Prioritized Planning.
        """
        # Build constraints from non-neighborhood paths
        constraints = set()
        other_paths = {}
        
        for agent_id, path in enumerate(paths):
            if agent_id not in neighborhood:
                other_paths[agent_id] = path
                for t, pos in enumerate(path):
                    constraints.add((t, pos[0], pos[1]))
                # Extended wait at goal
                goal = self.instance.agents[agent_id].goal
                for t in range(len(path), len(path) + 100):
                    constraints.add((t, goal[0], goal[1]))
        
        # Replan for neighborhood agents
        new_paths = paths.copy()
        neighborhood_list = list(neighborhood)
        
        # Random priority within neighborhood
        random.shuffle(neighborhood_list)
        
        for agent_id in neighborhood_list:
            agent = self.instance.agents[agent_id]
            
            # Combine constraints with other paths
            combined_other = dict(other_paths)
            path = self.pp_planner.planner.find_path(agent, constraints, combined_other)
            
            if path is None:
                return None  # Repair failed
            
            new_paths[agent_id] = path
            other_paths[agent_id] = path
            
            # Update constraints
            for t, pos in enumerate(path):
                constraints.add((t, pos[0], pos[1]))
        
        return new_paths
    
    def solve(self, time_limit: float = 60.0) -> Tuple[Optional[List[List[Tuple[int, int]]]], float, Dict]:
        """
        Solve MAPF using LNS framework.
        
        Returns:
            Tuple of (paths, total_runtime, statistics)
        """
        start_time = time.time()
        stats = {
            'initial_solver': self.initial_solver,
            'iterations': 0,
            'initial_collisions': 0,
            'final_collisions': 0
        }
        
        # Get initial solution
        if self.initial_solver == 'marl':
            paths, initial_time = self.initial_planner.generate_initial_solution()
        else:
            paths, initial_time = self.initial_planner.solve()
        
        if paths is None:
            return None, time.time() - start_time, stats
        
        stats['initial_time'] = initial_time
        
        # Check for collisions
        collisions = self.collision_detector.detect_collisions(paths)
        stats['initial_collisions'] = len(collisions)
        
        if len(collisions) == 0:
            stats['final_collisions'] = 0
            return paths, time.time() - start_time, stats
        
        # LNS iterations
        best_paths = paths
        best_collision_count = len(collisions)
        
        for iteration in range(self.max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            stats['iterations'] = iteration + 1
            
            # Select neighborhood
            neighborhood = self.select_neighborhood(paths, collisions)
            
            # Repair paths
            new_paths = self.repair_paths(paths, neighborhood)
            
            if new_paths is not None:
                new_collisions = self.collision_detector.detect_collisions(new_paths)
                
                if len(new_collisions) < len(collisions):
                    paths = new_paths
                    collisions = new_collisions
                    
                    if len(collisions) < best_collision_count:
                        best_paths = paths
                        best_collision_count = len(collisions)
                    
                    if len(collisions) == 0:
                        break
        
        stats['final_collisions'] = best_collision_count
        
        if best_collision_count == 0:
            return best_paths, time.time() - start_time, stats
        else:
            # Return best solution even if not collision-free
            return best_paths, time.time() - start_time, stats

class HybridMARLLNSPP:
    """
    Hybrid MAPF solver combining MARL, LNS, and PP.
    
    Strategy:
    1. Use MARL-guided planning for initial solution (good collision avoidance)
    2. Use LNS with MARL-based neighborhood selection for repair
    3. Use PP for fast finalization when few collisions remain
    """
    
    def __init__(self, instance: MAPFInstance):
        self.instance = instance
        self.lns = LargeNeighborhoodSearch(instance, 
                                           initial_solver='marl',
                                           repair_solver='pp')
        self.pp = PrioritizedPlanning(instance)
        
    def solve(self, time_limit: float = 60.0) -> Tuple[Optional[List[List[Tuple[int, int]]]], float, Dict]:
        """
        Solve MAPF using hybrid approach.
        
        Returns:
            Tuple of (paths, runtime, statistics)
        """
        start_time = time.time()
        
        # Phase 1: MARL-LNS for initial solution and major repairs
        paths, lns_time, lns_stats = self.lns.solve(time_limit * 0.7)
        
        if paths is None:
            # Fallback to pure PP with random restarts
            for seed in range(5):
                random.seed(seed)
                priority_order = list(range(len(self.instance.agents)))
                random.shuffle(priority_order)
                paths, pp_time = self.pp.solve(priority_order)
                if paths is not None:
                    return paths, time.time() - start_time, {
                        'method': 'pp_restart',
                        'restarts': seed + 1
                    }
            return None, time.time() - start_time, {'method': 'failed'}
        
        # Check if solution is already good
        collisions = CollisionDetector.detect_collisions(paths)
        
        if len(collisions) == 0:
            lns_stats['method'] = 'marl-lns'
            return paths, time.time() - start_time, lns_stats
        
        # Phase 2: PP refinement for remaining collisions
        if len(collisions) <= 5:
            # Extract colliding agents
            colliding_agents = set()
            for c in collisions:
                colliding_agents.update(c['agents'])
            
            # Replan colliding agents with PP
            non_colliding_paths = {i: paths[i] for i in range(len(paths)) 
                                   if i not in colliding_agents}
            
            # Build constraints from non-colliding paths
            constraints = set()
            for agent_id, path in non_colliding_paths.items():
                for t, pos in enumerate(path):
                    constraints.add((t, pos[0], pos[1]))
            
            # Try to replan colliding agents
            new_paths = paths.copy()
            success = True
            
            for agent_id in sorted(colliding_agents):
                agent = self.instance.agents[agent_id]
                other_paths = dict(non_colliding_paths)
                other_paths.update({i: new_paths[i] for i in colliding_agents 
                                   if i != agent_id and new_paths[i] is not None})
                
                new_path = self.pp.planner.find_path(agent, constraints, other_paths)
                
                if new_path is None:
                    success = False
                    break
                
                new_paths[agent_id] = new_path
            
            if success:
                final_collisions = CollisionDetector.detect_collisions(new_paths)
                if len(final_collisions) == 0:
                    lns_stats['method'] = 'marl-lns-pp-hybrid'
                    lns_stats['pp_refinement'] = True
                    return new_paths, time.time() - start_time, lns_stats
        
        lns_stats['method'] = 'marl-lns'
        return paths, time.time() - start_time, lns_stats


def load_instance_from_npy(filepath: str) -> MAPFInstance:
    """
    Load a MAPF instance from .npy file.
    
    Note: The current data files only contain grid maps.
    For a complete implementation, agent start/goal positions would need
    to be loaded from additional files or generated.
    """
    grid = np.load(filepath, allow_pickle=True)
    
    height, width = grid.shape
    
    # Generate sample agents (in real scenario, these would be loaded from data)
    # Place agents at random free cells
    free_cells = list(zip(*np.where(grid == 0)))
    
    if len(free_cells) < 10:
        # Not enough space for agents
        agents = []
    else:
        # Generate 10 agents for testing
        num_agents = min(10, len(free_cells) // 2)
        selected_starts = random.sample(free_cells, num_agents)
        selected_goals = random.sample([c for c in free_cells if c not in selected_starts], 
                                       min(num_agents, len(free_cells) - num_agents))
        
        agents = [
            Agent(id=i, start=selected_starts[i], goal=selected_goals[i % len(selected_goals)])
            for i in range(num_agents)
        ]
    
    return MAPFInstance(grid=grid, agents=agents, height=height, width=width)


if __name__ == "__main__":
    # Test the solvers
    print("Testing MAPF Solvers...")
    
    # Create a simple test instance
    grid = np.zeros((10, 10))
    grid[3, 3] = -1
    grid[3, 4] = -1
    grid[4, 3] = -1
    grid[7, 7] = -1
    
    agents = [
        Agent(id=0, start=(0, 0), goal=(9, 9)),
        Agent(id=1, start=(0, 9), goal=(9, 0)),
        Agent(id=2, start=(5, 0), goal=(5, 9)),
        Agent(id=3, start=(0, 5), goal=(9, 5)),
    ]
    
    instance = MAPFInstance(grid=grid, agents=agents, height=10, width=10)
    
    # Test Prioritized Planning
    print("\n--- Testing Prioritized Planning ---")
    pp = PrioritizedPlanning(instance)
    paths, runtime = pp.solve()
    if paths:
        print(f"PP succeeded in {runtime:.3f}s")
        collisions = CollisionDetector.detect_collisions(paths)
        print(f"Collisions: {len(collisions)}")
    else:
        print("PP failed")
    
    # Test MARL-guided planning
    print("\n--- Testing MARL-Guided Planning ---")
    marl = MARLGuidedPlanner(instance)
    paths, runtime = marl.generate_initial_solution()
    if paths:
        print(f"MARL succeeded in {runtime:.3f}s")
        collisions = CollisionDetector.detect_collisions(paths)
        print(f"Collisions: {len(collisions)}")
    else:
        print("MARL failed")
    
    # Test Hybrid LNS
    print("\n--- Testing Hybrid MARL-LNS-PP ---")
    hybrid = HybridMARLLNSPP(instance)
    paths, runtime, stats = hybrid.solve(time_limit=10.0)
    if paths:
        print(f"Hybrid succeeded in {runtime:.3f}s")
        print(f"Stats: {stats}")
        collisions = CollisionDetector.detect_collisions(paths)
        print(f"Final collisions: {len(collisions)}")
    else:
        print("Hybrid failed")
        print(f"Stats: {stats}")
