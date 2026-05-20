"""
Neural guidance for geometry theorem proving using Graph Neural Networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Set, Dict, Tuple
from collections import defaultdict

from src.geometry_engine import Fact, GeometryState


class GeometryGraphBuilder:
    """Convert a geometry state to a graph representation."""
    
    # Predicate types for edge features
    PREDICATE_TYPES = [
        'cong', 'eqangle', 'eqratio', 'perp', 'para', 'coll', 'cyclic',
        'midp', 'pmirror', 'diff', 'ncoll', 'bisect', 'amirror',
        'contri', 'simtri', 'contri2', 'simtri2', 'eqratio3',
        'free', 'circle', 'foot', 'reflect', 'shift',
        'rconst', 's_angle', 'eqangle2', 'eqangle3', 'eqdistance',
        'parallelogram', 'square', 'rectangle', 'trapezoid',
        'iso_triangle', 'r_triangle', 'eq_triangle', 'ieq_triangle',
        'on_line', 'on_circle', 'on_tline', 'on_pline', 'on_bline',
        'tangent', 'cc_tangent', 'intersection', 'other'
    ]
    
    PREDICATE_TO_IDX = {p: i for i, p in enumerate(PREDICATE_TYPES)}
    NUM_EDGE_TYPES = len(PREDICATE_TYPES)
    
    def __init__(self):
        self.point_to_idx = {}
        self.idx_to_point = {}
    
    def build_graph(self, state: GeometryState, goal: Fact) -> Data:
        """Build a PyG Data object from geometry state and goal."""
        points = sorted(state.points)
        point_to_idx = {p: i for i, p in enumerate(points)}
        n_points = len(points)
        
        # Node features: one-hot encoding + goal relevance
        node_feats = torch.zeros(n_points, 3)
        goal_points = set(goal.args)
        for i, p in enumerate(points):
            node_feats[i, 0] = 1.0  # Point exists
            if p in goal_points:
                node_feats[i, 1] = 1.0  # In goal
        
        # Edge features from facts
        edge_index = []
        edge_attr = []
        
        # Add edges for binary/ternary relations
        for fact in state.facts:
            pred_idx = self.PREDICATE_TO_IDX.get(fact.predicate, self.NUM_EDGE_TYPES - 1)
            
            # Create edges between all pairs of points in the fact
            args = list(fact.args)
            for i in range(len(args)):
                for j in range(i + 1, len(args)):
                    if args[i] in point_to_idx and args[j] in point_to_idx:
                        u = point_to_idx[args[i]]
                        v = point_to_idx[args[j]]
                        edge_index.append([u, v])
                        edge_index.append([v, u])
                        
                        # Edge feature: one-hot predicate type
                        feat = torch.zeros(self.NUM_EDGE_TYPES + 1)
                        feat[pred_idx] = 1.0
                        feat[-1] = 1.0 if fact.predicate == goal.predicate else 0.0
                        edge_attr.append(feat)
                        edge_attr.append(feat)
        
        if len(edge_index) == 0:
            # Add self-loops for isolated nodes
            edge_index = [[i, i] for i in range(n_points)]
            edge_attr = [torch.zeros(self.NUM_EDGE_TYPES + 1) for _ in range(n_points)]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
        
        # Goal encoding as global feature
        goal_pred_idx = self.PREDICATE_TO_IDX.get(goal.predicate, self.NUM_EDGE_TYPES - 1)
        goal_feat = torch.zeros(self.NUM_EDGE_TYPES + 1)
        goal_feat[goal_pred_idx] = 1.0
        goal_feat = goal_feat.unsqueeze(0).repeat(n_points, 1)
        
        # Concatenate node features with goal
        x = torch.cat([node_feats, goal_feat], dim=-1)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   num_nodes=n_points, goal=goal)


class GeometryGNN(nn.Module):
    """Graph Neural Network for geometry proof state evaluation."""
    
    def __init__(self, node_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Edge MLP for edge features
        edge_dim = GeometryGraphBuilder.NUM_EDGE_TYPES + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Readout layers
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value head: estimates how close state is to goal
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Policy head: scores actions (rules)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Edge features processing
        edge_weight = self.edge_mlp(edge_attr).norm(dim=-1, keepdim=True).squeeze(-1)
        
        # GNN forward
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Global pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        
        hidden = self.readout(pooled)
        
        # Outputs
        value = self.value_head(hidden).squeeze(-1)  # Scalar per graph
        policy = self.policy_head(hidden).squeeze(-1)  # Scalar per graph
        
        return value, policy


class NeuralHeuristic:
    """Neural network heuristic for proof search."""
    
    def __init__(self, model: GeometryGNN, graph_builder: GeometryGraphBuilder, device='cpu'):
        self.model = model.to(device)
        self.graph_builder = graph_builder
        self.device = device
        self.model.eval()
    
    def score_state(self, facts: Set[Fact], goal: Fact) -> float:
        """Score a proof state (lower is better)."""
        with torch.no_grad():
            # Build a dummy state
            state = GeometryState()
            state.facts = facts
            state.points = set()
            for f in facts:
                for a in f.args:
                    state.points.add(a)
            
            data = self.graph_builder.build_graph(state, goal)
            data = data.to(self.device)
            
            value, _ = self.model(data)
            return -value.item()  # Higher value = closer to goal, so negate for score
    
    def train_step(self, batch_data: List[Tuple[Data, float]]) -> float:
        """Train on a batch of (state, target_value) pairs."""
        self.model.train()
        
        graphs = [d for d, _ in batch_data]
        targets = torch.tensor([v for _, v in batch_data], dtype=torch.float, device=self.device)
        
        batch = Batch.from_data_list(graphs)
        values, _ = self.model(batch)
        
        loss = F.mse_loss(values, targets)
        
        return loss.item()


def generate_synthetic_data(problems, rules, num_episodes=100, max_depth=8):
    """Generate training data by running random proof walks."""
    from src.prover import SearchProver, RuleMatcher, normalize_fact
    from src.geometry_engine import problem_to_state
    import random
    
    matcher = RuleMatcher(rules)
    graph_builder = GeometryGraphBuilder()
    
    data = []
    
    for _ in range(num_episodes):
        problem = random.choice(problems)
        state = problem_to_state(problem)
        goal = normalize_fact(Fact(problem.goal_predicate, tuple(problem.goal_args)))
        
        facts = set(normalize_fact(f) for f in state.facts)
        
        # Random walk
        for step in range(max_depth):
            if goal in facts:
                value = 1.0 - step / max_depth
                data.append((graph_builder.build_graph(state, goal), value))
                break
            
            results = matcher.apply_all(facts)
            if not results:
                value = 0.0
                data.append((graph_builder.build_graph(state, goal), value))
                break
            
            # Score and pick best
            rule, sub, new_facts = random.choice(results)
            for nf in new_facts:
                facts.add(nf)
                state.facts.add(nf)
            
            # Intermediate reward
            value = max(0.0, 0.5 - step / max_depth)
            data.append((graph_builder.build_graph(state, goal), value))
        else:
            value = 0.0
            data.append((graph_builder.build_graph(state, goal), value))
    
    return data
