"""
Deep MARL for MAPF using PyTorch - Neural network policy for collision reduction.
Implements a simplified version of the PRIMAL/SCRIMP approach.
"""

import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from mapf_env import DIRS, manhattan_distance, detect_collisions, count_colliding_pairs, a_star


class MAPFPolicyNet(nn.Module):
    """Neural network policy for MAPF agents.
    
    Input: local observation (obstacles, other agents, goal direction)
    Output: action probabilities over {wait, up, down, left, right}
    """
    
    def __init__(self, fov_size=7, n_channels=5, hidden_dim=128):
        super().__init__()
        self.fov_size = fov_size
        
        # CNN for spatial features
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Compute conv output size
        conv_out_size = 64 * fov_size * fov_size
        
        # Additional features: goal distance (2), goal direction (2)
        self.extra_dim = 4
        
        self.fc1 = nn.Linear(conv_out_size + self.extra_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 5)  # 5 actions
        self.value_head = nn.Linear(hidden_dim, 1)    # State value
        
    def forward(self, obs_grid, extra_features):
        """Forward pass.
        
        Args:
            obs_grid: (batch, n_channels, fov_size, fov_size) tensor
            extra_features: (batch, extra_dim) tensor
        
        Returns:
            action_logits: (batch, 5) tensor
            value: (batch, 1) tensor
        """
        x = F.relu(self.conv1(obs_grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, extra_features], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        value = self.value_head(x)
        return action_logits, value


class DeepMARLPlanner:
    """Deep MARL-based path planner using neural network policies."""
    
    def __init__(self, fov_size=7, hidden_dim=64, n_episodes=100, max_steps=200, 
                 time_limit=30, lr=3e-4, gamma=0.95, device='cpu'):
        self.fov_size = fov_size
        self.hidden_dim = hidden_dim
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.gamma = gamma
        self.device = device
        
        self.policy_net = MAPFPolicyNet(fov_size, 5, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.stats = {}
    
    def _get_observation(self, pos, goal, grid, agent_positions, goal_positions):
        """Build observation tensor for an agent.
        
        Channels:
        0: obstacles
        1: other agents
        2: own goal (if in FOV)
        3: other agents' goals (if in FOV)
        4: visited/to-avoid positions
        """
        H, W = grid.shape
        half = self.fov_size // 2
        
        obs = np.zeros((5, self.fov_size, self.fov_size), dtype=np.float32)
        
        for fi in range(self.fov_size):
            for fj in range(self.fov_size):
                wr = pos[0] - half + fi
                wc = pos[1] - half + fj
                
                if wr < 0 or wr >= H or wc < 0 or wc >= W:
                    obs[0, fi, fj] = 1.0  # Out of bounds = obstacle
                    continue
                
                # Channel 0: obstacles
                if grid[wr, wc] == -1:
                    obs[0, fi, fj] = 1.0
                
                # Channel 1: other agents
                if (wr, wc) in agent_positions:
                    obs[1, fi, fj] = 1.0
                
                # Channel 2: own goal
                if (wr, wc) == goal:
                    obs[2, fi, fj] = 1.0
                
                # Channel 3: other agents' goals
                if (wr, wc) in goal_positions:
                    obs[3, fi, fj] = 1.0
        
        # Channel 4: distance heuristic (normalized)
        for fi in range(self.fov_size):
            for fj in range(self.fov_size):
                wr = pos[0] - half + fi
                wc = pos[1] - half + fj
                if 0 <= wr < H and 0 <= wc < W and grid[wr, wc] == 0:
                    obs[4, fi, fj] = 1.0 - min(manhattan_distance((wr, wc), goal), 20) / 20.0
        
        # Extra features
        goal_dist = manhattan_distance(pos, goal)
        goal_dr = goal[0] - pos[0]
        goal_dc = goal[1] - pos[1]
        max_dist = max(H + W, 1)
        
        extra = np.array([
            goal_dist / max_dist,
            0 if goal_dist == 0 else goal_dr / goal_dist,
            0 if goal_dist == 0 else goal_dc / goal_dist,
            len(agent_positions) / max(1, H * W)
        ], dtype=np.float32)
        
        return obs, extra
    
    def _get_valid_actions(self, pos, grid):
        """Get valid actions from a position."""
        r, c = pos
        H, W = grid.shape
        valid = []
        for a_idx, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                valid.append(a_idx)
        return valid
    
    def solve(self, instance, seed=None):
        """Generate paths using deep MARL policy."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        start_time = time.time()
        grid = instance.grid
        n_agents = instance.n_agents
        
        # Quick training
        self._quick_train(instance)
        
        # Inference with trained policy
        positions = [instance.starts[i] for i in range(n_agents)]
        paths = [[instance.starts[i]] for i in range(n_agents)]
        done = [instance.starts[i] == instance.goals[i] for i in range(n_agents)]
        
        for step in range(self.max_steps):
            if time.time() - start_time > self.time_limit:
                break
            if all(done):
                break
            
            actions = [0] * n_agents
            new_positions = [None] * n_agents
            
            # Order agents
            order = list(range(n_agents))
            random.shuffle(order)
            
            for idx in order:
                if done[idx]:
                    new_positions[idx] = positions[idx]
                    continue
                
                other_pos = set()
                for j in range(n_agents):
                    if j != idx:
                        other_pos.add(positions[j])
                
                goal_positions_set = set()
                for j in range(n_agents):
                    if j != idx:
                        goal_positions_set.add(instance.goals[j])
                
                obs_grid, extra = self._get_observation(
                    positions[idx], instance.goals[idx], grid, other_pos, goal_positions_set
                )
                
                obs_tensor = torch.FloatTensor(obs_grid).unsqueeze(0).to(self.device)
                extra_tensor = torch.FloatTensor(extra).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, _ = self.policy_net(obs_tensor, extra_tensor)
                    valid_actions = self._get_valid_actions(positions[idx], grid)
                    
                    # Mask invalid actions
                    mask = torch.ones(5).to(self.device) * -1e9
                    for a in valid_actions:
                        mask[a] = 0
                    masked_logits = logits[0] + mask
                    probs = F.softmax(masked_logits, dim=0)
                    
                    action = torch.multinomial(probs, 1).item()
                
                dr, dc = DIRS[action]
                nr, nc = positions[idx][0] + dr, positions[idx][1] + dc
                
                # Collision check with already-moved agents
                collision = False
                for j in order:
                    if j == idx:
                        break
                    if new_positions[j] == (nr, nc):
                        collision = True
                        break
                    # Swap check
                    if (new_positions[j] is not None and 
                        new_positions[j] == positions[idx] and
                        positions[j] == (nr, nc)):
                        collision = True
                        break
                
                if collision:
                    new_positions[idx] = positions[idx]
                else:
                    new_positions[idx] = (nr, nc)
            
            for i in range(n_agents):
                positions[i] = new_positions[i]
                paths[i].append(positions[i])
                if positions[i] == instance.goals[i]:
                    done[i] = True
        
        elapsed = time.time() - start_time
        
        # Fill in missing paths
        for i in range(n_agents):
            if paths[i][-1] != instance.goals[i]:
                # Try A* to complete
                remaining = a_star(grid, paths[i][-1], instance.goals[i])
                if remaining:
                    paths[i].extend(remaining[1:])
        
        final_collisions = detect_collisions(paths)
        
        self.stats = {
            'success': len(final_collisions) == 0,
            'n_collisions': len(final_collisions),
            'n_colliding_pairs': count_colliding_pairs(paths),
            'time': elapsed,
            'sum_of_costs': sum(len(p) - 1 for p in paths),
            'makespan': max(len(p) - 1 for p in paths),
        }
        
        return paths, len(final_collisions) == 0, self.stats
    
    def _quick_train(self, instance):
        """Quick training loop on the given instance."""
        n_agents = instance.n_agents
        grid = instance.grid
        
        for episode in range(min(self.n_episodes, 50)):
            positions = [instance.starts[i] for i in range(n_agents)]
            done = [instance.starts[i] == instance.goals[i] for i in range(n_agents)]
            
            # Storage for training
            all_obs = [[] for _ in range(n_agents)]
            all_extra = [[] for _ in range(n_agents)]
            all_actions = [[] for _ in range(n_agents)]
            all_rewards = [[] for _ in range(n_agents)]
            
            for step in range(self.max_steps):
                if all(done):
                    break
                
                new_positions = [None] * n_agents
                order = list(range(n_agents))
                random.shuffle(order)
                
                for idx in order:
                    if done[idx]:
                        new_positions[idx] = positions[idx]
                        continue
                    
                    other_pos = set(positions[j] for j in range(n_agents) if j != idx)
                    goal_set = set(instance.goals[j] for j in range(n_agents) if j != idx)
                    
                    obs_grid, extra = self._get_observation(
                        positions[idx], instance.goals[idx], grid, other_pos, goal_set
                    )
                    
                    obs_tensor = torch.FloatTensor(obs_grid).unsqueeze(0).to(self.device)
                    extra_tensor = torch.FloatTensor(extra).unsqueeze(0).to(self.device)
                    
                    logits, value = self.policy_net(obs_tensor, extra_tensor)
                    valid_actions = self._get_valid_actions(positions[idx], grid)
                    
                    mask = torch.ones(5).to(self.device) * -1e9
                    for a in valid_actions:
                        mask[a] = 0
                    masked_logits = logits[0] + mask
                    probs = F.softmax(masked_logits, dim=0)
                    
                    # Epsilon-greedy during training
                    if random.random() < max(0.05, 0.3 * (1 - episode / 50)):
                        action = random.choice(valid_actions)
                    else:
                        action = torch.multinomial(probs, 1).item()
                    
                    dr, dc = DIRS[action]
                    nr, nc = positions[idx][0] + dr, positions[idx][1] + dc
                    
                    # Collision check
                    collision = False
                    for j in order:
                        if j == idx:
                            break
                        if new_positions[j] == (nr, nc):
                            collision = True
                            break
                        if (new_positions[j] is not None and 
                            new_positions[j] == positions[idx] and
                            positions[j] == (nr, nc)):
                            collision = True
                            break
                    
                    if collision:
                        new_positions[idx] = positions[idx]
                        reward = -2.0
                    else:
                        new_positions[idx] = (nr, nc)
                        if (nr, nc) == instance.goals[idx]:
                            reward = 10.0
                        else:
                            reward = -0.1
                    
                    all_obs[idx].append(np.copy(obs_grid))
                    all_extra[idx].append(np.copy(extra))
                    all_actions[idx].append(action)
                    all_rewards[idx].append(reward)
                
                for i in range(n_agents):
                    positions[i] = new_positions[i]
                    if positions[i] == instance.goals[i]:
                        done[i] = True
            
            # Policy gradient update
            self.optimizer.zero_grad()
            total_loss = 0.0
            
            for i in range(n_agents):
                if len(all_actions[i]) == 0:
                    continue
                
                T = len(all_actions[i])
                returns = np.zeros(T)
                G = 0
                for t in reversed(range(T)):
                    G = all_rewards[i][t] + self.gamma * G
                    returns[t] = G
                
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                returns = torch.FloatTensor(returns).to(self.device)
                
                obs_batch = torch.FloatTensor(np.array(all_obs[i])).to(self.device)
                extra_batch = torch.FloatTensor(np.array(all_extra[i])).to(self.device)
                actions_batch = torch.LongTensor(all_actions[i]).to(self.device)
                
                logits, values = self.policy_net(obs_batch, extra_batch)
                
                # Policy loss (REINFORCE with baseline)
                log_probs = F.log_softmax(logits, dim=1)
                selected_log_probs = log_probs[range(len(actions_batch)), actions_batch]
                advantage = returns - values.squeeze()
                policy_loss = -(selected_log_probs * advantage.detach()).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), returns)
                
                loss = policy_loss + 0.5 * value_loss
                total_loss += loss
            
            if total_loss != 0.0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
