"""
Hybrid MARL-LNS Algorithm for MAPF
Combines Multi-Agent Reinforcement Learning with Large Neighborhood Search
for improved collision reduction and solution quality.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import time
import random
from mapf_core import (
    MAPFInstance, Path, Solution, 
    count_all_collisions, AStarPlanner, PrioritizedPlanner
)
from marl_agent import MARLAgent, MAPFEnvironment, pretrain_marl
from lns_framework import LNS, MARLGuidedLNS


class HybridMARLLNS:
    """
    Hybrid MARL-LNS algorithm for MAPF.
    
    Key Innovation:
    - Early stage: Use MARL to reduce collisions via collision-aware selection
    - Late stage: Switch to pure Prioritized Planning for efficient local repair
    - Adaptive switching based on collision count threshold
    """
    
    def __init__(self, instance: MAPFInstance, 
                 marl_agent: Optional[MARLAgent] = None,
                 max_iterations: int = 500,
                 timeout: float = 30.0,
                 early_stage_threshold: float = 0.3,
                 marl_ratio: float = 0.7):
        """
        Args:
            instance: MAPF problem instance
            marl_agent: Pre-trained MARL agent (if None, will be created)
            max_iterations: Maximum LNS iterations
            timeout: Time limit in seconds
            early_stage_threshold: Switch to PP when collision ratio < threshold
            marl_ratio: Ratio of MARL vs random selection in early stage
        """
        self.instance = instance
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.early_stage_threshold = early_stage_threshold
        self.marl_ratio = marl_ratio
        
        # Initialize or use provided MARL agent
        if marl_agent is None:
            self.marl_agent = MARLAgent(fov_size=5, hidden_size=32, lr=1e-3)
        else:
            self.marl_agent = marl_agent
        
        # Initialize LNS components
        self.marl_lns = MARLGuidedLNS(
            instance, self.marl_agent, max_iterations, timeout, marl_ratio
        )
        self.plain_lns = LNS(instance, max_iterations, timeout)
        self.planner = AStarPlanner(instance)
    
    def solve(self, initial_paths: Optional[List[Path]] = None) -> Tuple[Optional[List[Path]], dict]:
        """
        Solve MAPF using hybrid MARL-LNS.
        
        Returns:
            Tuple of (solution paths, statistics)
        """
        start_time = time.time()
        
        # Statistics
        stats = {
            "algorithm": "HybridMARL-LNS",
            "instance_size": self.instance.agent_count,
            "iterations": 0,
            "collisions_history": [],
            "soc_history": [],
            "stage_history": [],
            "time_history": [],
            "success": False,
            "final_collisions": 0,
            "final_soc": 0,
            "runtime": 0
        }
        
        # Phase 1: Generate initial solution
        if initial_paths is None:
            paths = self._generate_initial_solution()
        else:
            paths = [Path(list(p.positions)) for p in initial_paths]
        
        if paths is None:
            stats["error"] = "Failed to generate initial solution"
            return None, stats
        
        initial_collisions = count_all_collisions(paths, self.instance)
        max_possible_collisions = initial_collisions if initial_collisions > 0 else 1
        
        stats["collisions_history"].append(initial_collisions)
        stats["soc_history"].append(sum(len(p) for p in paths))
        stats["stage_history"].append("initial")
        
        # Phase 2: Iterative improvement with stage switching
        for iteration in range(self.max_iterations):
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                break
            
            current_collisions = count_all_collisions(paths, self.instance)
            collision_ratio = current_collisions / max_possible_collisions
            
            # Determine stage based on collision ratio
            if collision_ratio > self.early_stage_threshold:
                # Early stage: Use MARL-guided LNS
                stage = "marl_lns"
                result = self.marl_lns.solve(paths, neighborhood_size=min(10, self.instance.agent_count))
            else:
                # Late stage: Use pure Prioritized Planning for efficiency
                stage = "pp_lns"
                result = self.plain_lns.solve(paths, neighborhood_size=min(5, self.instance.agent_count))
            
            new_paths, iteration_stats = result
            
            if new_paths is not None:
                new_collisions = count_all_collisions(new_paths, self.instance)
                
                # Accept if improved
                if new_collisions < current_collisions:
                    paths = new_paths
                    current_collisions = new_collisions
                
                # Train MARL agent with experience
                self._train_marl_from_iteration(paths, iteration)
            
            # Record statistics
            stats["iterations"] = iteration + 1
            stats["collisions_history"].append(current_collisions)
            stats["soc_history"].append(sum(len(p) for p in paths))
            stats["stage_history"].append(stage)
            stats["time_history"].append(time.time() - start_time)
            
            # Check if solution is feasible
            if current_collisions == 0:
                stats["success"] = True
                break
        
        # Final statistics
        stats["runtime"] = time.time() - start_time
        stats["final_collisions"] = count_all_collisions(paths, self.instance)
        stats["final_soc"] = sum(len(p) for p in paths)
        stats["success"] = (stats["final_collisions"] == 0)
        
        return paths, stats
    
    def _generate_initial_solution(self) -> Optional[List[Path]]:
        """Generate initial solution using prioritized planning."""
        # Try different priority orders
        for _ in range(10):
            priority_order = list(range(self.instance.agent_count))
            random.shuffle(priority_order)
            
            pp = PrioritizedPlanner(self.instance)
            paths = pp.plan(priority_order)
            
            if paths is not None:
                return paths
        
        # Fallback: random walks
        return self.plain_lns._generate_initial_paths()
    
    def _train_marl_from_iteration(self, paths: List[Path], iteration: int):
        """Train MARL agent based on current iteration experience."""
        # Create environment and collect experience
        env = MAPFEnvironment(self.instance, self.marl_agent.fov_size)
        observations = env.reset()
        
        # Simulate one step with current collision state
        actions = []
        for i in range(self.instance.agent_count):
            # Use path information to guide action
            if paths and i < len(paths) and len(paths[i]) > 1:
                # Move towards next position in path
                current = paths[i][0]
                next_pos = paths[i][1]
                dr = next_pos[0] - current[0]
                dc = next_pos[1] - current[1]
                
                # Convert to action
                if dr == 0 and dc == 1:
                    action = 1  # right
                elif dr == 1 and dc == 0:
                    action = 2  # down
                elif dr == 0 and dc == -1:
                    action = 3  # left
                elif dr == -1 and dc == 0:
                    action = 4  # up
                else:
                    action = 0  # wait
            else:
                action = random.randint(0, 4)
            
            actions.append(action)
        
        # Step environment
        next_observations, rewards, done = env.step(actions)
        
        # Store experience
        for i in range(self.instance.agent_count):
            self.marl_agent.buffer.append((
                observations[i],
                actions[i],
                rewards[i],
                next_observations[i],
                float(done)
            ))
        
        # Train periodically
        if iteration % 5 == 0 and len(self.marl_agent.buffer) >= 32:
            self.marl_agent.train_step(batch_size=32)


def compare_algorithms(instance: MAPFInstance, 
                      max_iterations: int = 300,
                      timeout: float = 15.0) -> Dict:
    """
    Compare different algorithms on a MAPF instance.
    
    Returns:
        Dictionary with results for each algorithm
    """
    results = {}
    
    # 1. Pure Prioritized Planning
    print("Running Pure PP...")
    start = time.time()
    pp = PrioritizedPlanner(instance)
    pp_paths = None
    for _ in range(5):  # Try multiple random orders
        order = list(range(instance.agent_count))
        random.shuffle(order)
        pp_paths = pp.plan(order)
        if pp_paths is not None:
            break
    
    pp_time = time.time() - start
    pp_collisions = count_all_collisions(pp_paths, instance) if pp_paths else float('inf')
    pp_soc = sum(len(p) for p in pp_paths) if pp_paths else float('inf')
    
    results["PurePP"] = {
        "success": pp_collisions == 0,
        "collisions": pp_collisions,
        "soc": pp_soc,
        "runtime": pp_time
    }
    
    # 2. MAPF-LNS2 (LNS with random selection)
    print("Running MAPF-LNS2...")
    lns = LNS(instance, max_iterations, timeout)
    lns_paths, lns_stats = lns.solve()
    
    results["MAPF-LNS2"] = {
        "success": lns_stats["success"],
        "collisions": lns_stats["final_collisions"],
        "soc": lns_stats["final_soc"],
        "runtime": lns_stats["runtime"],
        "iterations": lns_stats.get("iterations", 0)
    }
    
    # 3. Hybrid MARL-LNS (our method)
    print("Running Hybrid MARL-LNS...")
    # Pretrain MARL agent
    marl_agent = MARLAgent(fov_size=5, hidden_size=32, lr=1e-3)
    pretrain_marl(marl_agent, instance, episodes=50, max_steps=100)
    
    hybrid = HybridMARLLNS(
        instance, marl_agent, max_iterations, timeout,
        early_stage_threshold=0.3, marl_ratio=0.7
    )
    hybrid_paths, hybrid_stats = hybrid.solve()
    
    results["HybridMARL-LNS"] = {
        "success": hybrid_stats["success"],
        "collisions": hybrid_stats["final_collisions"],
        "soc": hybrid_stats["final_soc"],
        "runtime": hybrid_stats["runtime"],
        "iterations": hybrid_stats["iterations"],
        "collisions_history": hybrid_stats["collisions_history"]
    }
    
    # 4. MARL-LNS without stage switching (ablation)
    print("Running MARL-LNS (no switching)...")
    marl_lns = MARLGuidedLNS(
        instance, marl_agent, max_iterations, timeout, 0.7
    )
    marl_only_paths, marl_only_stats = marl_lns.solve()
    
    results["MARL-LNS-Only"] = {
        "success": marl_only_stats["success"],
        "collisions": marl_only_stats["final_collisions"],
        "soc": marl_only_stats["final_soc"],
        "runtime": marl_only_stats["runtime"],
        "iterations": marl_only_stats["iterations"]
    }
    
    return results
