"""
Main experiment script for neuro-symbolic geometry theorem proving.
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

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def run_baseline_experiments(problems, rules, output_dir):
    """Run baseline symbolic search experiments."""
    print("\n=== Running Baseline Experiments ===")
    prover = SearchProver(rules)
    
    strategies = {
        'BFS': lambda s, g: prover.prove_bfs(s, g, max_depth=10, max_nodes=5000),
        'Beam-3': lambda s, g: prover.prove_beam(s, g, beam_width=3, max_depth=10),
        'Beam-5': lambda s, g: prover.prove_beam(s, g, beam_width=5, max_depth=10),
        'Beam-10': lambda s, g: prover.prove_beam(s, g, beam_width=10, max_depth=10),
        'Heuristic-BFS': lambda s, g: prover.prove_bfs(s, g, max_depth=10, max_nodes=5000, heuristic=goal_distance_heuristic),
    }
    
    results = defaultdict(dict)
    
    for strategy_name, strategy_fn in strategies.items():
        print(f"\n  Strategy: {strategy_name}")
        solved = 0
        total_time = 0
        total_nodes = 0
        
        for p in problems:
            state = problem_to_state(p)
            goal = normalize_fact(Fact(p.goal_predicate, tuple(p.goal_args)))
            
            result = strategy_fn(state.facts, goal)
            results[p.name][strategy_name] = result
            
            if result['success']:
                solved += 1
            total_time += result['time']
            total_nodes += result['nodes']
        
        print(f"    Solved: {solved}/{len(problems)} ({100*solved/len(problems):.1f}%)")
        print(f"    Avg time: {total_time/len(problems):.3f}s")
        print(f"    Avg nodes: {total_nodes/len(problems):.0f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/baseline_results.json', 'w') as f:
        # Convert sets to lists for JSON serialization
        json_results = {}
        for pname, strats in results.items():
            json_results[pname] = {}
            for sname, res in strats.items():
                json_results[pname][sname] = {
                    'success': res['success'],
                    'depth': res['depth'],
                    'nodes': res['nodes'],
                    'time': res['time'],
                    'facts': res['facts']
                }
        json.dump(json_results, f, indent=2)
    
    return results


def analyze_problems(problems, rules, output_dir):
    """Analyze problem characteristics."""
    print("\n=== Analyzing Problems ===")
    
    analysis = []
    for p in problems:
        state = problem_to_state(p)
        
        # Count predicates
        pred_counts = defaultdict(int)
        for f in state.facts:
            pred_counts[f.predicate] += 1
        
        analysis.append({
            'name': p.name,
            'num_points': len(state.points),
            'num_facts': len(state.facts),
            'num_constructions': len(p.constructions),
            'goal_predicate': p.goal_predicate,
            'goal_arity': len(p.goal_args),
            'predicate_distribution': dict(pred_counts)
        })
    
    with open(f'{output_dir}/problem_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Summary statistics
    num_points = [a['num_points'] for a in analysis]
    num_facts = [a['num_facts'] for a in analysis]
    num_constructions = [a['num_constructions'] for a in analysis]
    
    print(f"  Points: {np.mean(num_points):.1f} ± {np.std(num_points):.1f} (range {min(num_points)}-{max(num_points)})")
    print(f"  Facts: {np.mean(num_facts):.1f} ± {np.std(num_facts):.1f} (range {min(num_facts)}-{max(num_facts)})")
    print(f"  Constructions: {np.mean(num_constructions):.1f} ± {np.std(num_constructions):.1f}")
    
    return analysis


def generate_training_data(problems, rules, num_episodes=200, max_depth=10):
    """Generate synthetic training data via random and heuristic walks."""
    print("\n=== Generating Training Data ===")
    matcher = RuleMatcher(rules)
    graph_builder = GeometryGraphBuilder()
    
    data = []
    
    for ep in range(num_episodes):
        problem = random.choice(problems)
        state = problem_to_state(problem)
        goal = normalize_fact(Fact(problem.goal_predicate, tuple(problem.goal_args)))
        
        facts = set(normalize_fact(f) for f in state.facts)
        
        # Mix of random and heuristic-guided walks
        use_heuristic = random.random() < 0.5
        
        for step in range(max_depth):
            if goal in facts:
                value = 1.0
                data.append((graph_builder.build_graph(state, goal), value))
                break
            
            results = matcher.apply_all(facts)
            if not results:
                value = 0.1
                data.append((graph_builder.build_graph(state, goal), value))
                break
            
            if use_heuristic and len(results) > 1:
                # Heuristic: prefer rules that introduce goal-related facts
                best_score = -1
                best_result = results[0]
                for rule, sub, new_facts in results:
                    score = 0
                    for nf in new_facts:
                        overlap = len(set(nf.args) & set(goal.args))
                        if nf.predicate == goal.predicate:
                            score += 10
                        score += overlap
                    if score > best_score:
                        best_score = score
                        best_result = (rule, sub, new_facts)
                rule, sub, new_facts = best_result
            else:
                rule, sub, new_facts = random.choice(results)
            
            for nf in new_facts:
                nf_norm = normalize_fact(nf)
                if nf_norm not in facts:
                    facts.add(nf_norm)
                    state.facts.add(nf_norm)
            
            # Value based on progress
            goal_overlap = sum(len(set(f.args) & set(goal.args)) for f in facts if f.predicate == goal.predicate)
            value = 0.1 + 0.7 * (goal_overlap / max(1, len(goal.args) * 2)) + 0.2 * (1 - step / max_depth)
            value = min(1.0, value)
            data.append((graph_builder.build_graph(state, goal), value))
        else:
            value = 0.1
            data.append((graph_builder.build_graph(state, goal), value))
        
        if ep % 50 == 0:
            print(f"  Episode {ep}/{num_episodes}, data size: {len(data)}")
    
    print(f"  Generated {len(data)} training samples")
    return data


def train_neural_model(train_data, output_dir, epochs=50, batch_size=32):
    """Train the GNN guidance model."""
    print("\n=== Training Neural Guidance Model ===")
    
    node_dim = 3 + GeometryGraphBuilder.NUM_EDGE_TYPES + 1  # node_feats + goal_feat
    model = GeometryGNN(node_dim=node_dim, hidden_dim=128, num_layers=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Split train/val
    random.shuffle(train_data)
    split = int(0.8 * len(train_data))
    train_set = train_data[:split]
    val_set = train_data[split:]
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_set), batch_size):
            batch = train_set[i:i+batch_size]
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
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_set), batch_size):
                batch = val_set[i:i+batch_size]
                graphs = [d for d, _ in batch]
                targets = torch.tensor([v for _, v in batch], dtype=torch.float, device=DEVICE)
                batch_data = Batch.from_data_list(graphs).to(DEVICE)
                values, _ = model(batch_data)
                val_loss += nn.MSELoss()(values, targets).item()
        
        avg_train = epoch_loss / max(num_batches, 1)
        avg_val = val_loss / max(len(val_set) // batch_size, 1)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{output_dir}/gnn_model.pt')
    
    # Save training curves
    with open(f'{output_dir}/training_curves.json', 'w') as f:
        json.dump({'train_loss': train_losses, 'val_loss': val_losses}, f)
    
    print(f"  Model saved to {output_dir}/gnn_model.pt")
    return model, train_losses, val_losses


def run_neural_guided_search(problems, rules, model, output_dir):
    """Run search with neural guidance."""
    print("\n=== Running Neural-Guided Search ===")
    
    graph_builder = GeometryGraphBuilder()
    neural_heuristic = NeuralHeuristic(model, graph_builder, device=DEVICE)
    prover = SearchProver(rules)
    
    results = {}
    solved = 0
    total_time = 0
    total_nodes = 0
    
    for p in problems:
        state = problem_to_state(p)
        goal = normalize_fact(Fact(p.goal_predicate, tuple(p.goal_args)))
        
        # Use neural heuristic in best-first search
        result = prover.prove_bfs(
            state.facts, goal,
            max_depth=12, max_nodes=10000,
            heuristic=lambda f, g: neural_heuristic.score_state(f, g)
        )
        
        results[p.name] = result
        if result['success']:
            solved += 1
        total_time += result['time']
        total_nodes += result['nodes']
    
    print(f"  Solved: {solved}/{len(problems)} ({100*solved/len(problems):.1f}%)")
    print(f"  Avg time: {total_time/len(problems):.3f}s")
    print(f"  Avg nodes: {total_nodes/len(problems):.0f}")
    
    with open(f'{output_dir}/neural_results.json', 'w') as f:
        json_results = {}
        for pname, res in results.items():
            json_results[pname] = {
                'success': res['success'],
                'depth': res['depth'],
                'nodes': res['nodes'],
                'time': res['time'],
                'facts': res['facts']
            }
        json.dump(json_results, f, indent=2)
    
    return results


def run_ablation_studies(problems, rules, model, output_dir):
    """Run ablation studies on search parameters."""
    print("\n=== Running Ablation Studies ===")
    
    graph_builder = GeometryGraphBuilder()
    neural_heuristic = NeuralHeuristic(model, graph_builder, device=DEVICE)
    prover = SearchProver(rules)
    
    configs = [
        ('Neural-BF-5k-D10', lambda s, g: prover.prove_bfs(s, g, max_depth=10, max_nodes=5000, heuristic=lambda f, gg: neural_heuristic.score_state(f, gg))),
        ('Neural-BF-10k-D12', lambda s, g: prover.prove_bfs(s, g, max_depth=12, max_nodes=10000, heuristic=lambda f, gg: neural_heuristic.score_state(f, gg))),
        ('Neural-Beam-5-D10', lambda s, g: prover.prove_beam(s, g, beam_width=5, max_depth=10)),
        ('Neural-Beam-10-D12', lambda s, g: prover.prove_beam(s, g, beam_width=10, max_depth=12)),
    ]
    
    ablation_results = {}
    for name, fn in configs:
        solved = 0
        total_nodes = 0
        total_time = 0
        for p in problems:
            state = problem_to_state(p)
            goal = normalize_fact(Fact(p.goal_predicate, tuple(p.goal_args)))
            res = fn(state.facts, goal)
            if res['success']:
                solved += 1
            total_nodes += res['nodes']
            total_time += res['time']
        
        ablation_results[name] = {
            'solved': solved,
            'total': len(problems),
            'solve_rate': solved / len(problems),
            'avg_nodes': total_nodes / len(problems),
            'avg_time': total_time / len(problems)
        }
        print(f"  {name}: {solved}/{len(problems)} ({100*solved/len(problems):.1f}%) avg_nodes={total_nodes/len(problems):.0f}")
    
    with open(f'{output_dir}/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    return ablation_results


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    
    problems = parse_problems('data/imo_ag_30.txt')
    rules = parse_rules('data/rules.txt')
    
    print(f"Loaded {len(problems)} problems and {len(rules)} rules")
    
    # Run analysis
    analysis = analyze_problems(problems, rules, 'outputs')
    
    # Run baselines
    baseline_results = run_baseline_experiments(problems, rules, 'outputs')
    
    # Generate training data and train model
    train_data = generate_training_data(problems, rules, num_episodes=300, max_depth=10)
    model, train_losses, val_losses = train_neural_model(train_data, 'outputs', epochs=50, batch_size=32)
    
    # Run neural-guided search
    neural_results = run_neural_guided_search(problems, rules, model, 'outputs')
    
    # Run ablations
    ablation_results = run_ablation_studies(problems, rules, model, 'outputs')
    
    print("\n=== All experiments complete ===")
