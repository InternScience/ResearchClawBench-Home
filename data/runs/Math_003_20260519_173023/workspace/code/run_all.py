"""
Streamlined experiment runner for neuro-symbolic geometry theorem proving.
"""
import os
import json
import time
import random
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from src.parser import parse_problems, parse_rules
from src.geometry_engine import problem_to_state, Fact, GeometryState
from src.prover import SearchProver, RuleMatcher, normalize_fact, goal_distance_heuristic
from src.neural_guidance import GeometryGraphBuilder, GeometryGNN, NeuralHeuristic

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

problems = parse_problems('data/imo_ag_30.txt')
rules = parse_rules('data/rules.txt')
print(f"Loaded {len(problems)} problems, {len(rules)} rules")

# ============================================================
# 1. PROBLEM ANALYSIS
# ============================================================
print("\n=== 1. Problem Analysis ===")
analysis = []
pred_types_all = defaultdict(int)
for p in problems:
    state = problem_to_state(p)
    pred_counts = defaultdict(int)
    for f in state.facts:
        pred_counts[f.predicate] += 1
        pred_types_all[f.predicate] += 1
    analysis.append({
        'name': p.name,
        'num_points': len(state.points),
        'num_facts': len(state.facts),
        'num_constructions': len(p.constructions),
        'goal_predicate': p.goal_predicate,
        'predicate_distribution': dict(pred_counts)
    })

with open('outputs/problem_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"  Avg points: {np.mean([a['num_points'] for a in analysis]):.1f}")
print(f"  Avg facts: {np.mean([a['num_facts'] for a in analysis]):.1f}")
print(f"  Avg constructions: {np.mean([a['num_constructions'] for a in analysis]):.1f}")

# ============================================================
# 2. BASELINE EXPERIMENTS (fast)
# ============================================================
print("\n=== 2. Baseline Search Experiments ===")
prover = SearchProver(rules)

baseline_configs = [
    ('BFS-500', lambda s, g: prover.prove_bfs(s, g, max_depth=6, max_nodes=500)),
    ('Beam-3-D8', lambda s, g: prover.prove_beam(s, g, beam_width=3, max_depth=8)),
    ('Beam-5-D8', lambda s, g: prover.prove_beam(s, g, beam_width=5, max_depth=8)),
    ('Heuristic-BFS-1k', lambda s, g: prover.prove_bfs(s, g, max_depth=8, max_nodes=1000, heuristic=goal_distance_heuristic)),
]

baseline_results = {}
for strat_name, strat_fn in baseline_configs:
    solved = 0
    times = []
    nodes = []
    for p in problems:
        state = problem_to_state(p)
        goal = normalize_fact(Fact(p.goal_predicate, tuple(p.goal_args)))
        res = strat_fn(state.facts, goal)
        if res['success']:
            solved += 1
        times.append(res['time'])
        nodes.append(res['nodes'])
    baseline_results[strat_name] = {
        'solved': solved, 'total': len(problems),
        'solve_rate': solved / len(problems),
        'avg_time': float(np.mean(times)),
        'avg_nodes': float(np.mean(nodes)),
        'max_time': float(np.max(times)),
    }
    print(f"  {strat_name}: {solved}/{len(problems)} solve_rate={solved/len(problems):.3f} avg_time={np.mean(times):.3f}s")

with open('outputs/baseline_results.json', 'w') as f:
    json.dump(baseline_results, f, indent=2)

# ============================================================
# 3. GENERATE TRAINING DATA
# ============================================================
print("\n=== 3. Generate Training Data ===")
matcher = RuleMatcher(rules)
graph_builder = GeometryGraphBuilder()

train_data = []
num_episodes = 200
max_depth = 8

for ep in range(num_episodes):
    p = random.choice(problems)
    state = problem_to_state(p)
    goal = normalize_fact(Fact(p.goal_predicate, tuple(p.goal_args)))
    facts = set(normalize_fact(f) for f in state.facts)
    
    for step in range(max_depth):
        if goal in facts:
            train_data.append((graph_builder.build_graph(state, goal), 1.0))
            break
        
        results = matcher.apply_all(facts)
        if not results:
            train_data.append((graph_builder.build_graph(state, goal), 0.1))
            break
        
        # Heuristic-guided selection
        if len(results) > 1:
            best = max(results, key=lambda r: sum(
                (10 if nf.predicate == goal.predicate else 0) + 
                len(set(nf.args) & set(goal.args)) 
                for nf in r[2]
            ))
            rule, sub, new_facts = best
        else:
            rule, sub, new_facts = results[0]
        
        for nf in new_facts:
            nf_norm = normalize_fact(nf)
            if nf_norm not in facts:
                facts.add(nf_norm)
                state.facts.add(nf_norm)
        
        # Compute value
        goal_facts = [f for f in facts if f.predicate == goal.predicate]
        goal_overlap = sum(len(set(f.args) & set(goal.args)) for f in goal_facts)
        value = 0.1 + 0.6 * min(1.0, goal_overlap / max(1, len(goal.args))) + 0.3 * (1 - step / max_depth)
        value = min(1.0, value)
        train_data.append((graph_builder.build_graph(state, goal), value))
    else:
        train_data.append((graph_builder.build_graph(state, goal), 0.1))

print(f"  Generated {len(train_data)} samples")

# ============================================================
# 4. TRAIN NEURAL MODEL
# ============================================================
print("\n=== 4. Train Neural Guidance Model ===")
node_dim = 3 + GeometryGraphBuilder.NUM_EDGE_TYPES + 1
model = GeometryGNN(node_dim=node_dim, hidden_dim=128, num_layers=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

random.shuffle(train_data)
split = int(0.8 * len(train_data))
train_set = train_data[:split]
val_set = train_data[split:]

train_losses = []
val_losses = []

for epoch in range(40):
    model.train()
    epoch_loss = 0
    num_batches = 0
    for i in range(0, len(train_set), 32):
        batch = train_set[i:i+32]
        graphs = [d for d, _ in batch]
        targets = torch.tensor([v for _, v in batch], dtype=torch.float, device=DEVICE)
        batch_data = Batch.from_data_list(graphs).to(DEVICE)
        values, _ = model(batch_data)
        loss = nn.MSELoss()(values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(val_set), 32):
            batch = val_set[i:i+32]
            graphs = [d for d, _ in batch]
            targets = torch.tensor([v for _, v in batch], dtype=torch.float, device=DEVICE)
            batch_data = Batch.from_data_list(graphs).to(DEVICE)
            values, _ = model(batch_data)
            val_loss += nn.MSELoss()(values, targets).item()
    
    avg_train = epoch_loss / max(num_batches, 1)
    avg_val = val_loss / max(len(val_set) // 32, 1)
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}")

torch.save(model.state_dict(), 'outputs/gnn_model.pt')
with open('outputs/training_curves.json', 'w') as f:
    json.dump({'train_loss': train_losses, 'val_loss': val_losses}, f)

# ============================================================
# 5. NEURAL-GUIDED SEARCH
# ============================================================
print("\n=== 5. Neural-Guided Search ===")
neural_heuristic = NeuralHeuristic(model, graph_builder, device=DEVICE)

neural_configs = [
    ('Neural-BF-1k-D8', lambda s, g: prover.prove_bfs(s, g, max_depth=8, max_nodes=1000, heuristic=lambda f, gg: neural_heuristic.score_state(f, gg))),
    ('Neural-BF-2k-D10', lambda s, g: prover.prove_bfs(s, g, max_depth=10, max_nodes=2000, heuristic=lambda f, gg: neural_heuristic.score_state(f, gg))),
    ('Neural-Beam-5-D10', lambda s, g: prover.prove_beam(s, g, beam_width=5, max_depth=10)),
]

neural_results = {}
for strat_name, strat_fn in neural_configs:
    solved = 0
    times = []
    nodes = []
    for p in problems:
        state = problem_to_state(p)
        goal = normalize_fact(Fact(p.goal_predicate, tuple(p.goal_args)))
        res = strat_fn(state.facts, goal)
        if res['success']:
            solved += 1
        times.append(res['time'])
        nodes.append(res['nodes'])
    neural_results[strat_name] = {
        'solved': solved, 'total': len(problems),
        'solve_rate': solved / len(problems),
        'avg_time': float(np.mean(times)),
        'avg_nodes': float(np.mean(nodes)),
        'max_time': float(np.max(times)),
    }
    print(f"  {strat_name}: {solved}/{len(problems)} solve_rate={solved/len(problems):.3f} avg_time={np.mean(times):.3f}s")

with open('outputs/neural_results.json', 'w') as f:
    json.dump(neural_results, f, indent=2)

# ============================================================
# 6. COMBINED COMPARISON
# ============================================================
print("\n=== 6. Combined Comparison ===")
all_results = {**baseline_results, **neural_results}
with open('outputs/combined_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nAll experiments complete!")
