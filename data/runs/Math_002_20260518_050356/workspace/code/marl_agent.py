"""
MARL Component for MAPF
Multi-Agent Reinforcement Learning with shared policy for:
1. Neighborhood selection in LNS
2. Early-stage collision reduction
3. Collision-aware path guidance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
import random
from collections import deque
from mapf_core import MAPFInstance, Path, Solution, count_all_collisions


class MARLNetwork(nn.Module):
    """Neural network for MARL policy."""
    
    def __init__(self, fov_size: int = 5, hidden_size: int = 64, num_actions: int = 5):
        """
        Args:
            fov_size: Field of view size (fov_size x fov_size local observation)
            hidden_size: Hidden layer size
            num_actions: 4 directions + wait
        """
        super().__init__()
        
        # Local observation encoder
        input_size = fov_size * fov_size * 3  # 3 channels: grid, agents, goals
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value."""
        features = self.encoder(observation)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, observation: torch.Tensor, stochastic: bool = True) -> int:
        """Get action from observation."""
        with torch.no_grad():
            logits, value = self.forward(observation)
            if stochastic:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                action = torch.argmax(logits, dim=-1).item()
        return action
    
    def evaluate(self, observation: torch.Tensor, action: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action for training."""
        logits, value = self.forward(observation)
        probs = torch.softmax(logits, dim=-1)
        log_prob = torch.log(probs[action] + 1e-8)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return log_prob, value.squeeze(), entropy


class MAPFEnvironment:
    """RL environment for MAPF."""
    
    ACTION_MAP = {
        0: (0, 0),   # wait
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
        4: (-1, 0)   # up
    }
    
    def __init__(self, instance: MAPFInstance, fov_size: int = 5):
        self.instance = instance
        self.fov_size = fov_size
        self.reset()
    
    def reset(self) -> List[torch.Tensor]:
        """Reset environment and return initial observations."""
        self.agent_positions = list(self.instance.starts)
        self.timestep = 0
        self.done = False
        self.collisions = 0
        self.goals_reached = [False] * self.instance.agent_count
        return self._get_observations()
    
    def _get_observations(self) -> List[torch.Tensor]:
        """Get observations for all agents."""
        observations = []
        half_fov = self.fov_size // 2
        
        for i in range(self.instance.agent_count):
            obs = np.zeros((3, self.fov_size, self.fov_size))
            r, c = self.agent_positions[i]
            
            # Channel 0: Grid (obstacles)
            for dr in range(-half_fov, half_fov + 1):
                for dc in range(-half_fov, half_fov + 1):
                    gr, gc = r + dr, c + dc
                    if (0 <= gr < self.instance.height and 
                        0 <= gc < self.instance.width):
                        obs[0, dr + half_fov, dc + half_fov] = (
                            1.0 if self.instance.grid[gr, gc] == -1 else 0.0
                        )
            
            # Channel 1: Other agents
            for j, pos in enumerate(self.agent_positions):
                if i != j and not self.goals_reached[j]:
                    jr, jc = pos
                    dr, dc = jr - r, jc - c
                    if abs(dr) <= half_fov and abs(dc) <= half_fov:
                        obs[1, dr + half_fov, dc + half_fov] = 1.0
            
            # Channel 2: Goal direction
            goal = self.instance.goals[i]
            gr, gc = goal
            dr, dc = gr - r, gc - c
            # Normalize and clip
            norm = max(abs(dr), abs(dc), 1)
            dr_norm, dc_norm = dr / norm, dc / norm
            # Mark goal direction
            for d in range(1, half_fov + 1):
                gr_pos = min(half_fov + int(d * dr_norm), self.fov_size - 1)
                gc_pos = min(half_fov + int(d * dc_norm), self.fov_size - 1)
                gr_pos = max(0, gr_pos)
                gc_pos = max(0, gc_pos)
                obs[2, gr_pos, gc_pos] = 1.0
            
            observations.append(torch.FloatTensor(obs.flatten()))
        
        return observations
    
    def step(self, actions: List[int]) -> Tuple[List[torch.Tensor], List[float], bool]:
        """Execute actions for all agents."""
        new_positions = []
        rewards = []
        
        for i, action in enumerate(actions):
            dr, dc = self.ACTION_MAP[action]
            r, c = self.agent_positions[i]
            new_r, new_c = r + dr, c + dc
            
            # Check bounds and obstacles
            if (0 <= new_r < self.instance.height and 
                0 <= new_c < self.instance.width and
                self.instance.grid[new_r, new_c] != -1):
                new_positions.append((new_r, new_c))
            else:
                new_positions.append((r, c))  # Stay in place
        
        # Check collisions
        vertex_collisions = 0
        swap_collisions = 0
        
        # Vertex collisions
        pos_counts = {}
        for pos in new_positions:
            if pos not in pos_counts:
                pos_counts[pos] = 0
            pos_counts[pos] += 1
        for pos, count in pos_counts.items():
            if count > 1:
                vertex_collisions += count - 1
        
        # Swap collisions
        for i in range(self.instance.agent_count):
            for j in range(i + 1, self.instance.agent_count):
                if (self.agent_positions[i] == new_positions[j] and
                    self.agent_positions[j] == new_positions[i]):
                    swap_collisions += 1
        
        total_collisions = vertex_collisions + swap_collisions
        self.collisions += total_collisions
        
        # Calculate rewards
        for i in range(self.instance.agent_count):
            reward = 0.0
            
            # Goal progress reward
            old_dist = abs(self.agent_positions[i][0] - self.instance.goals[i][0]) + \
                       abs(self.agent_positions[i][1] - self.instance.goals[i][1])
            new_dist = abs(new_positions[i][0] - self.instance.goals[i][0]) + \
                       abs(new_positions[i][1] - self.instance.goals[i][1])
            
            if new_dist < old_dist:
                reward += 1.0  # Getting closer
            elif new_dist > old_dist:
                reward -= 0.5  # Moving away
            
            # Collision penalty
            reward -= total_collisions * 0.1
            
            # Goal reached bonus
            if new_positions[i] == self.instance.goals[i] and not self.goals_reached[i]:
                reward += 10.0
                self.goals_reached[i] = True
            
            rewards.append(reward)
        
        # Update positions
        self.agent_positions = new_positions
        self.timestep += 1
        
        # Check if done
        all_goals_reached = all(self.goals_reached)
        max_time_exceeded = self.timestep > max(100, self.instance.agent_count * 10)
        self.done = all_goals_reached or max_time_exceeded
        
        return self._get_observations(), rewards, self.done
    
    def get_collision_matrix(self) -> np.ndarray:
        """Get pairwise collision matrix between agents."""
        matrix = np.zeros((self.instance.agent_count, self.instance.agent_count))
        for i in range(self.instance.agent_count):
            for j in range(i + 1, self.instance.agent_count):
                dist = abs(self.agent_positions[i][0] - self.agent_positions[j][0]) + \
                       abs(self.agent_positions[i][1] - self.agent_positions[j][1])
                if dist <= 2:  # Close agents have higher collision risk
                    matrix[i, j] = 1.0 / max(dist, 1)
                    matrix[j, i] = matrix[i, j]
        return matrix


class MARLAgent:
    """MARL agent for MAPF."""
    
    def __init__(self, fov_size: int = 5, hidden_size: int = 64, 
                 lr: float = 1e-4, gamma: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = MARLNetwork(fov_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.fov_size = fov_size
        
        # Experience buffer
        self.buffer = deque(maxlen=10000)
    
    def select_neighborhood(self, instance: MAPFInstance, 
                           current_paths: List[Path],
                           collision_agents: List[int],
                           k: int = 10) -> List[int]:
        """Select neighborhood for LNS using MARL."""
        if not collision_agents:
            return []
        
        # Create environment with current state
        env = MAPFEnvironment(instance, self.fov_size)
        
        # Calculate collision scores for each agent
        scores = []
        for agent_id in collision_agents:
            # Create observation
            obs = self._create_observation(instance, agent_id, current_paths)
            with torch.no_grad():
                logits, value = self.network(obs)
                scores.append((value.item(), agent_id))
        
        # Sort by value (higher value = more important to replan)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Select top-k agents (or all if fewer)
        selected = [agent_id for _, agent_id in scores[:k]]
        
        return selected
    
    def select_neighborhood_with_random(self, instance: MAPFInstance,
                                       current_paths: List[Path],
                                       collision_agents: List[int],
                                       k: int = 10,
                                       marl_ratio: float = 0.7) -> List[int]:
        """Select neighborhood using MARL + random for diversity."""
        if not collision_agents:
            return []
        
        # MARL selection
        marl_k = int(k * marl_ratio)
        marl_selected = self.select_neighborhood(instance, current_paths, 
                                                 collision_agents, marl_k)
        
        # Random selection for remaining
        remaining = [a for a in collision_agents if a not in marl_selected]
        random_k = min(k - len(marl_selected), len(remaining))
        random_selected = random.sample(remaining, random_k) if random_k > 0 else []
        
        return marl_selected + random_selected
    
    def _create_observation(self, instance: MAPFInstance, agent_id: int,
                          current_paths: List[Path]) -> torch.Tensor:
        """Create observation for a specific agent."""
        half_fov = self.fov_size // 2
        obs = np.zeros((3, self.fov_size, self.fov_size))
        
        # Get current position from path
        if current_paths and agent_id < len(current_paths):
            path = current_paths[agent_id]
            # Use position at timestep 0 or last known
            if len(path) > 0:
                r, c = path[0]
            else:
                r, c = instance.starts[agent_id]
        else:
            r, c = instance.starts[agent_id]
        
        # Channel 0: Grid
        for dr in range(-half_fov, half_fov + 1):
            for dc in range(-half_fov, half_fov + 1):
                gr, gc = r + dr, c + dc
                if (0 <= gr < instance.height and 
                    0 <= gc < instance.width):
                    obs[0, dr + half_fov, dc + half_fov] = (
                        1.0 if instance.grid[gr, gc] == -1 else 0.0
                    )
        
        # Channel 1: Other agents
        for j, path in enumerate(current_paths):
            if j != agent_id and len(path) > 0:
                jr, jc = path[0]
                dr, dc = jr - r, jc - c
                if abs(dr) <= half_fov and abs(dc) <= half_fov:
                    obs[1, dr + half_fov, dc + half_fov] = 1.0
        
        # Channel 2: Goal direction
        goal = instance.goals[agent_id]
        gr, gc = goal
        dr, dc = gr - r, gc - c
        norm = max(abs(dr), abs(dc), 1)
        
        return torch.FloatTensor(obs.flatten()).to(self.device)
    
    def train_step(self, batch_size: int = 32):
        """Train the MARL network on collected experiences."""
        if len(self.buffer) < batch_size:
            return
        
        batch = random.sample(self.buffer, batch_size)
        
        observations = torch.stack([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_observations = torch.stack([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)
        
        # Calculate returns
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Forward pass
        logits, values = self.network(observations)
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate loss
        advantages = returns - values.squeeze().detach()
        log_probs = torch.log(probs + 1e-8)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        actor_loss = -(selected_log_probs * advantages).mean()
        critic_loss = nn.MSELoss()(values.squeeze(), returns)
        entropy_loss = -0.01 * (probs * log_probs).sum(dim=-1).mean()
        
        loss = actor_loss + 0.5 * critic_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()


def pretrain_marl(agent: MARLAgent, instance: MAPFInstance, 
                  episodes: int = 100, max_steps: int = 200):
    """Pretrain MARL agent using random episodes."""
    env = MAPFEnvironment(instance, agent.fov_size)
    
    for episode in range(episodes):
        observations = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Random actions
            actions = [random.randint(0, 4) for _ in range(instance.agent_count)]
            
            next_observations, rewards, done = env.step(actions)
            
            # Store experience
            for i in range(instance.agent_count):
                agent.buffer.append((
                    observations[i],
                    actions[i],
                    rewards[i],
                    next_observations[i],
                    float(done)
                ))
            
            observations = next_observations
            episode_reward += sum(rewards)
            
            if done:
                break
        
        # Train periodically
        if episode % 10 == 0:
            agent.train_step(batch_size=min(32, len(agent.buffer)))
    
    return agent
