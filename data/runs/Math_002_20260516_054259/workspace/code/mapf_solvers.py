"""
MAPF Solvers: Prioritized Planning, LNS, and MARL-LNS Hybrid.
"""

import numpy as np
import random
import time
from collections import defaultdict, deque
import heapq

from mapf_env import (
    MAPFInstance, DIRS, manhattan_distance, 
    detect_collisions, count_colliding_pairs,
    a_star, space_time_a_star, paths_to_obstacles
)


class PrioritizedPlanning:
    """Prioritized Planning (PP) solver for MAPF."""
    
    def __init__(self, time_limit=30, max_t=500):
        self.time_limit = time_limit
        self.max_t = max_t
        self.stats = {}
    
    def solve(self, instance, seed=None, use_space_time=True):
        """Solve MAPF instance using prioritized planning.
        
        Args:
            instance: MAPFInstance
            seed: random seed for agent ordering
            use_space_time: if True, use space-time A*; else use plain A* with constraints
        
        Returns:
            (paths, success, stats)
        """
        if seed is not None:
            random.seed(seed)
        
        start_time = time.time()
        grid = instance.grid
        agents = list(range(instance.n_agents))
        
        # Random priority ordering
        random.shuffle(agents)
        
        paths = [None] * instance.n_agents
        all_obstacles = set()
        
        for idx in agents:
            if time.time() - start_time > self.time_limit:
                # Plan remaining with minimal collision avoidance
                for remaining in agents[agents.index(idx):]:
                    path = a_star(grid, instance.starts[remaining], instance.goals[remaining])
                    if path:
                        paths[remaining] = path
                break
            
            start = instance.starts[idx]
            goal = instance.goals[idx]
            
            if use_space_time:
                path = space_time_a_star(grid, start, goal, all_obstacles, self.max_t)
            else:
                # Convert obstacles to constraints
                constraints = set()
                for (r, c, t) in all_obstacles:
                    constraints.add(((r, c), t))
                path = a_star(grid, start, goal, constraints)
            
            if path is not None:
                paths[idx] = path
                # Add to obstacles
                for t, pos in enumerate(path):
                    all_obstacles.add((pos[0], pos[1], t))
                # Goal occupation
                for t in range(len(path), self.max_t):
                    all_obstacles.add((goal[0], goal[1], t))
            else:
                # Try plain A* without obstacle avoidance
                path = a_star(grid, start, goal)
                if path:
                    paths[idx] = path
                    for t, pos in enumerate(path):
                        all_obstacles.add((pos[0], pos[1], t))
        
        # Check for None paths
        for i in range(instance.n_agents):
            if paths[i] is None:
                # Use trivial path (stay at start)
                paths[i] = [instance.starts[i]]
        
        collisions = detect_collisions(paths)
        success = len(collisions) == 0
        
        elapsed = time.time() - start_time
        
        self.stats = {
            'success': success,
            'n_collisions': len(collisions),
            'n_colliding_pairs': count_colliding_pairs(paths),
            'time': elapsed,
            'sum_of_costs': sum(len(p) - 1 for p in paths),
            'makespan': max(len(p) - 1 for p in paths),
        }
        
        return paths, success, self.stats


class LargeNeighborhoodSearch:
    """MAPF-LNS2 style Large Neighborhood Search solver."""
    
    def __init__(self, time_limit=30, max_iterations=1000, neighborhood_size=0.3, max_t=500):
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size  # fraction of agents to replan
        self.max_t = max_t
        self.stats = {}
    
    def solve(self, instance, seed=None, initial_paths=None):
        """Solve MAPF using LNS.
        
        Args:
            instance: MAPFInstance
            seed: random seed
            initial_paths: optional initial paths (from another solver)
        
        Returns:
            (paths, success, stats)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        start_time = time.time()
        grid = instance.grid
        
        # Get initial paths
        if initial_paths is not None:
            paths = [list(p) for p in initial_paths]
        else:
            # Use PP for initial paths
            pp = PrioritizedPlanning(time_limit=min(5, self.time_limit), max_t=self.max_t)
            paths, _, _ = pp.solve(instance, seed=seed)
        
        n_agents = instance.n_agents
        best_cp = count_colliding_pairs(paths)
        best_paths = [list(p) for p in paths]
        
        iteration = 0
        while iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                break
            
            iteration += 1
            
            # Select neighborhood: agents involved in collisions
            collisions = detect_collisions(paths)
            if len(collisions) == 0:
                # Solution found!
                break
            
            colliding_agents = set()
            for c in collisions:
                colliding_agents.add(c[0])
                colliding_agents.add(c[1])
            
            # Select subset to replan
            n_select = max(2, int(self.neighborhood_size * n_agents))
            candidates = list(colliding_agents)
            if len(candidates) < n_select:
                # Add some non-colliding agents
                others = [i for i in range(n_agents) if i not in candidates]
                random.shuffle(others)
                candidates.extend(others[:n_select - len(candidates)])
            
            selected = random.sample(candidates, min(n_select, len(candidates)))
            
            # Store old paths
            old_selected_paths = {i: list(paths[i]) for i in selected}
            
            # Build obstacle set from non-selected agents
            obstacles = set()
            for i in range(n_agents):
                if i not in selected:
                    for t, pos in enumerate(paths[i]):
                        obstacles.add((pos[0], pos[1], t))
                    goal = paths[i][-1]
                    for t in range(len(paths[i]), self.max_t):
                        obstacles.add((goal[0], goal[1], t))
            
            # Replan selected agents (using PP within the neighborhood)
            sel_list = list(selected)
            random.shuffle(sel_list)
            
            for idx in sel_list:
                start = instance.starts[idx]
                goal = instance.goals[idx]
                path = space_time_a_star(grid, start, goal, obstacles, self.max_t)
                if path is not None:
                    paths[idx] = path
                    for t, pos in enumerate(path):
                        obstacles.add((pos[0], pos[1], t))
                    for t in range(len(path), self.max_t):
                        obstacles.add((goal[0], goal[1], t))
                # else keep old path
            
            # Evaluate
            new_cp = count_colliding_pairs(paths)
            
            if new_cp <= best_cp:
                best_cp = new_cp
                best_paths = [list(p) for p in paths]
            else:
                # Revert to old paths for selected agents
                for i in selected:
                    paths[i] = old_selected_paths[i]
        
        final_collisions = detect_collisions(best_paths)
        success = len(final_collisions) == 0
        
        elapsed = time.time() - start_time
        
        self.stats = {
            'success': success,
            'n_collisions': len(final_collisions),
            'n_colliding_pairs': count_colliding_pairs(best_paths),
            'time': elapsed,
            'iterations': iteration,
            'sum_of_costs': sum(len(p) - 1 for p in best_paths),
            'makespan': max(len(p) - 1 for p in best_paths),
        }
        
        return best_paths, success, self.stats


class MARLAgent:
    """Simple MARL agent using Q-learning with coordination heuristics."""
    
    def __init__(self, agent_id, start, goal, grid, learning_rate=0.1, discount=0.95, epsilon=0.3):
        self.agent_id = agent_id
        self.start = start
        self.goal = goal
        self.grid = grid
        self.H, self.W = grid.shape
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = {}  # (pos, state_key) -> {action: value}
        self.episode_memory = []
    
    def get_state_key(self, pos, other_positions):
        """Create a simplified state key."""
        dist_to_goal = manhattan_distance(pos, self.goal)
        # Count nearby agents
        nearby = 0
        for op in other_positions:
            if manhattan_distance(pos, op) <= 2:
                nearby += 1
        return (pos[0], pos[1], min(dist_to_goal, 20), min(nearby, 5))
    
    def get_valid_actions(self, pos):
        """Get valid moves from current position."""
        r, c = pos
        actions = []
        for a_idx, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] == 0:
                actions.append(a_idx)
        return actions
    
    def select_action(self, pos, other_positions, explore=True):
        """Select action using epsilon-greedy with heuristic bias."""
        state = self.get_state_key(pos, other_positions)
        valid_actions = self.get_valid_actions(pos)
        
        if not valid_actions:
            return 0  # wait
        
        if explore and random.random() < self.epsilon:
            # Heuristic: prefer actions that move toward goal
            best_dist = float('inf')
            best_actions = []
            for a in valid_actions:
                dr, dc = DIRS[a]
                nr, nc = pos[0] + dr, pos[1] + dc
                d = manhattan_distance((nr, nc), self.goal)
                if d < best_dist:
                    best_dist = d
                    best_actions = [a]
                elif d == best_dist:
                    best_actions.append(a)
            
            # Avoid positions where other agents are
            safe_actions = [a for a in best_actions if (pos[0] + DIRS[a][0], pos[1] + DIRS[a][1]) not in other_positions]
            if safe_actions:
                return random.choice(safe_actions)
            return random.choice(best_actions)
        else:
            # Greedy from Q-table
            if state not in self.q_table:
                self.q_table[state] = {}
            
            best_val = float('-inf')
            best_action = valid_actions[0]
            for a in valid_actions:
                val = self.q_table[state].get(a, 0.0)
                if val > best_val:
                    best_val = val
                    best_action = a
            return best_action
    
    def update(self, state, action, reward, next_state):
        """Q-learning update."""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        old_val = self.q_table[state].get(action, 0.0)
        
        # Max Q for next state
        if next_state in self.q_table and self.q_table[next_state]:
            max_next = max(self.q_table[next_state].values())
        else:
            max_next = 0.0
        
        new_val = old_val + self.lr * (reward + self.discount * max_next - old_val)
        self.q_table[state][action] = new_val


class MARLPathPlanner:
    """MARL-based path planner for initial collision reduction."""
    
    def __init__(self, n_episodes=50, max_steps=200, time_limit=30):
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.stats = {}
    
    def solve(self, instance, seed=None):
        """Generate paths using MARL-based coordination.
        
        Uses Q-learning with heuristic coordination to find
        collision-free paths.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        start_time = time.time()
        grid = instance.grid
        
        agents = [MARLAgent(i, instance.starts[i], instance.goals[i], grid) 
                  for i in range(instance.n_agents)]
        
        # Run episodes
        best_paths = None
        best_collisions = float('inf')
        
        for episode in range(self.n_episodes):
            if time.time() - start_time > self.time_limit:
                break
            
            # Initialize positions
            positions = [instance.starts[i] for i in range(instance.n_agents)]
            paths = [[instance.starts[i]] for i in range(instance.n_agents)]
            done = [instance.starts[i] == instance.goals[i] for i in range(instance.n_agents)]
            
            for step in range(self.max_steps):
                if all(done):
                    break
                
                # Each agent selects action
                actions = []
                new_positions = [None] * instance.n_agents
                
                # Order agents randomly
                order = list(range(instance.n_agents))
                random.shuffle(order)
                
                for i in order:
                    if done[i]:
                        actions.append(0)
                        new_positions[i] = positions[i]
                        continue
                    
                    other_pos = [positions[j] for j in range(instance.n_agents) if j != i]
                    other_pos.extend([new_positions[j] for j in range(instance.n_agents) 
                                     if j != i and new_positions[j] is not None])
                    
                    action = agents[i].select_action(positions[i], other_pos, explore=(episode < self.n_episodes * 0.7))
                    actions.append(action)
                    
                    dr, dc = DIRS[action]
                    nr, nc = positions[i][0] + dr, positions[i][1] + dc
                    
                    # Check for collisions with already-moved agents
                    collision = False
                    for j in order:
                        if j == i:
                            break
                        if new_positions[j] is not None and new_positions[j] == (nr, nc):
                            collision = True
                            break
                        # Check swap collision
                        if (new_positions[j] is not None and 
                            new_positions[j] == positions[i] and 
                            positions[j] == (nr, nc)):
                            collision = True
                            break
                    
                    if collision:
                        new_positions[i] = positions[i]  # stay
                        actions[-1] = 0
                    else:
                        new_positions[i] = (nr, nc)
                
                # Update positions and paths
                for i in range(instance.n_agents):
                    positions[i] = new_positions[i]
                    paths[i].append(positions[i])
                    if positions[i] == instance.goals[i]:
                        done[i] = True
                    
                    # Reward: negative distance to goal
                    dist = manhattan_distance(positions[i], instance.goals[i])
                    
                    # Simple Q-update (not full, just heuristic)
                    state = agents[i].get_state_key(positions[i], 
                        [positions[j] for j in range(instance.n_agents) if j != i])
                    reward = -0.1 if not done[i] else 10.0
                    agents[i].q_table.setdefault(state, {})[actions[i]] = (
                        agents[i].q_table.setdefault(state, {}).get(actions[i], 0.0) + 
                        agents[i].lr * reward
                    )
            
            # Evaluate
            collisions = detect_collisions(paths)
            n_cp = count_colliding_pairs(paths)
            
            if n_cp < best_collisions:
                best_collisions = n_cp
                best_paths = [list(p) for p in paths]
        
        if best_paths is None:
            # Fallback to simple A* for each agent
            best_paths = []
            for i in range(instance.n_agents):
                path = a_star(grid, instance.starts[i], instance.goals[i])
                if path is None:
                    path = [instance.starts[i]]
                best_paths.append(path)
        
        elapsed = time.time() - start_time
        final_collisions = detect_collisions(best_paths)
        
        self.stats = {
            'success': len(final_collisions) == 0,
            'n_collisions': len(final_collisions),
            'n_colliding_pairs': count_colliding_pairs(best_paths),
            'time': elapsed,
            'sum_of_costs': sum(len(p) - 1 for p in best_paths),
            'makespan': max(len(p) - 1 for p in best_paths),
        }
        
        return best_paths, len(final_collisions) == 0, self.stats


class HybridMARLLNS:
    """Hybrid MARL-LNS solver: MARL for initial collision reduction, LNS for refinement."""
    
    def __init__(self, time_limit=30, marl_episodes=30, lns_iterations=500, 
                 marl_fraction=0.4, neighborhood_size=0.3, max_t=500):
        """
        Args:
            time_limit: total time limit in seconds
            marl_episodes: number of MARL training episodes
            lns_iterations: max LNS repair iterations
            marl_fraction: fraction of time allocated to MARL phase
            neighborhood_size: fraction of agents to replan in LNS
            max_t: maximum timestep
        """
        self.time_limit = time_limit
        self.marl_episodes = marl_episodes
        self.lns_iterations = lns_iterations
        self.marl_fraction = marl_fraction
        self.neighborhood_size = neighborhood_size
        self.max_t = max_t
        self.stats = {}
    
    def solve(self, instance, seed=None):
        """Solve MAPF using hybrid MARL-LNS approach.
        
        Phase 1 (MARL): Use multi-agent Q-learning to get initial paths with low collisions
        Phase 2 (LNS): Use large neighborhood search with prioritized planning to resolve remaining collisions
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        total_start = time.time()
        grid = instance.grid
        
        # Phase 1: MARL for initial paths
        marl_time_limit = self.time_limit * self.marl_fraction
        marl_planner = MARLPathPlanner(
            n_episodes=self.marl_episodes,
            max_steps=200,
            time_limit=marl_time_limit
        )
        
        initial_paths, marl_success, marl_stats = marl_planner.solve(instance, seed=seed)
        marl_time = marl_stats['time']
        
        # Phase 2: LNS repair
        remaining_time = self.time_limit - (time.time() - total_start)
        if remaining_time < 1:
            remaining_time = 1
        
        lns = LargeNeighborhoodSearch(
            time_limit=remaining_time,
            max_iterations=self.lns_iterations,
            neighborhood_size=self.neighborhood_size,
            max_t=self.max_t
        )
        
        final_paths, lns_success, lns_stats = lns.solve(instance, seed=seed, initial_paths=initial_paths)
        
        total_time = time.time() - total_start
        
        # Combine stats
        final_collisions = detect_collisions(final_paths)
        
        self.stats = {
            'success': len(final_collisions) == 0,
            'n_collisions': len(final_collisions),
            'n_colliding_pairs': count_colliding_pairs(final_paths),
            'time': total_time,
            'marl_time': marl_time,
            'lns_time': lns_stats['time'],
            'lns_iterations': lns_stats['iterations'],
            'marl_collisions_initial': marl_stats['n_colliding_pairs'],
            'sum_of_costs': sum(len(p) - 1 for p in final_paths),
            'makespan': max(len(p) - 1 for p in final_paths),
        }
        
        return final_paths, self.stats['success'], self.stats
