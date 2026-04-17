"""
Main experiment runner for DIDS-MFL framework.
Runs all experiments: binary, multi-class, unknown attack, few-shot, ablation.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter

# Add code dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dids_mfl import (
    DIDS_MFL, MLPBaseline, TGNBaseline,
    train_epoch, evaluate, get_attack_names, get_few_shot_attack_types,
    prepare_data_splits, prepare_unknown_attack_split, prepare_few_shot_split,
    compute_loss, RepresentationalDisentanglement
)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(WORKSPACE, 'data', 'NF-UNSW-NB15-v2_3d.pt')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def remap_nodes(data):
    """Remap node IDs to consecutive integers for memory efficiency."""
    all_nodes = torch.cat([data.src, data.dst]).unique()
    node_map = {int(n): i for i, n in enumerate(all_nodes)}
    num_nodes = len(node_map)
    
    new_src = torch.tensor([node_map[int(s)] for s in data.src], dtype=torch.long)
    new_dst = torch.tensor([node_map[int(d)] for d in data.dst], dtype=torch.long)
    
    data.src = new_src
    data.dst = new_dst
    
    return data, num_nodes


def run_standard_experiment(data, num_nodes, model_name='DIDS_MFL', 
                           epochs=15, batch_size=512, lr=0.001):
    """Run standard binary + multi-class classification experiment."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} - Standard Classification")
    print(f"{'='*60}")
    
    train_mask, val_mask, test_mask = prepare_data_splits(data)
    print(f"Train: {len(train_mask)}, Val: {len(val_mask)}, Test: {len(test_mask)}")
    
    # Create model
    if model_name == 'DIDS_MFL':
        model = DIDS_MFL(num_nodes, feat_dim=40, memory_dim=32, hidden_dim=32)
    elif model_name == 'TGN':
        model = TGNBaseline(num_nodes, feat_dim=40, memory_dim=32, hidden_dim=32)
    elif model_name == 'MLP':
        model = MLPBaseline(feat_dim=40, hidden_dim=64)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    best_val_f1 = 0
    best_model_state = None
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Train
        loss_dict = train_epoch(model, data, train_mask, batch_size=batch_size, 
                               optimizer=optimizer)
        train_losses.append(loss_dict)
        
        # Validate
        val_results = evaluate(model, data, val_mask, batch_size=batch_size)
        val_metrics_history.append({
            'binary_f1': val_results['binary_f1'],
            'binary_auc': val_results['binary_auc'],
            'multi_f1_macro': val_results['multi_f1_macro'],
        })
        
        scheduler.step()
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_dict['total']:.4f} | "
              f"Val F1: {val_results['binary_f1']:.4f} | "
              f"Val AUC: {val_results['binary_auc']:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        if val_results['binary_f1'] > best_val_f1:
            best_val_f1 = val_results['binary_f1']
            best_model_state = {k: v.clone() for k, v in model.state_dict().items() 
                               if not k.startswith('memory_module.memory') and 
                               not k.startswith('memory_module.last_update')}
    
    # Load best model and evaluate on test
    if best_model_state is not None:
        current_state = model.state_dict()
        for k, v in best_model_state.items():
            current_state[k] = v
        model.load_state_dict(current_state)
    
    test_results = evaluate(model, data, test_mask, batch_size=batch_size)
    
    print(f"\n--- {model_name} Test Results ---")
    print(f"Binary F1: {test_results['binary_f1']:.4f}")
    print(f"Binary AUC: {test_results['binary_auc']:.4f}")
    print(f"Binary Precision: {test_results['binary_precision']:.4f}")
    print(f"Binary Recall: {test_results['binary_recall']:.4f}")
    print(f"Multi-class F1 (macro): {test_results['multi_f1_macro']:.4f}")
    print(f"Multi-class F1 (weighted): {test_results['multi_f1_weighted']:.4f}")
    print(f"Per-attack F1: {json.dumps(test_results['per_attack_f1'], indent=2)}")
    
    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_metrics': val_metrics_history,
        'test_results': {k: v for k, v in test_results.items() 
                        if k not in ['representations', 'labels', 'attacks', 
                                    'binary_preds', 'multi_preds']},
        'test_labels': test_results.get('labels'),
        'test_attacks': test_results.get('attacks'),
        'test_binary_preds': test_results.get('binary_preds'),
        'test_multi_preds': test_results.get('multi_preds'),
        'representations': test_results.get('representations'),
    }


def run_unknown_attack_experiment(data, num_nodes, model_name='DIDS_MFL',
                                  epochs=10, batch_size=512, lr=0.001):
    """Run unknown attack detection experiment."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} - Unknown Attack Detection")
    print(f"{'='*60}")
    
    attack_names = get_attack_names()
    # Test unknown detection for several attack types
    unknown_attacks = [0, 1, 3, 8, 9]  # Analysis, Backdoor, DoS, Shellcode, Worms
    
    results = {}
    for attack_id in unknown_attacks:
        attack_name = attack_names.get(attack_id, f'class_{attack_id}')
        print(f"\n--- Unknown attack: {attack_name} (id={attack_id}) ---")
        
        train_mask, val_mask, test_mask = prepare_unknown_attack_split(
            data, attack_id
        )
        
        # Create model
        if model_name == 'DIDS_MFL':
            model = DIDS_MFL(num_nodes, feat_dim=40, memory_dim=32, hidden_dim=32)
        elif model_name == 'TGN':
            model = TGNBaseline(num_nodes, feat_dim=40, memory_dim=32, hidden_dim=32)
        elif model_name == 'MLP':
            model = MLPBaseline(feat_dim=40, hidden_dim=64)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        for epoch in range(epochs):
            loss_dict = train_epoch(model, data, train_mask, batch_size=batch_size,
                                   optimizer=optimizer)
        
        # Evaluate on test set, focusing on the unknown attack
        test_results = evaluate(model, data, test_mask, batch_size=batch_size)
        
        # Get F1 for the unknown attack specifically (binary: can it detect as attack?)
        test_attacks = data.attack[test_mask].numpy()
        test_labels = data.label[test_mask].numpy()
        
        # For binary: check if unknown attack samples are detected as attacks
        unknown_mask = test_attacks == attack_id
        if unknown_mask.sum() > 0:
            unknown_binary_f1 = float(
                (test_results['binary_preds'][unknown_mask] == 1).mean()
            )
        else:
            unknown_binary_f1 = 0.0
        
        results[attack_name] = {
            'binary_f1': test_results['binary_f1'],
            'binary_auc': test_results['binary_auc'],
            'unknown_detection_rate': unknown_binary_f1,
            'per_attack_f1': test_results['per_attack_f1'],
        }
        
        print(f"  Binary F1: {test_results['binary_f1']:.4f}")
        print(f"  Unknown detection rate: {unknown_binary_f1:.4f}")
    
    return results


def run_few_shot_experiment(data, num_nodes, model_name='DIDS_MFL',
                           epochs=10, batch_size=512, lr=0.001):
    """Run few-shot attack detection experiment."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} - Few-Shot Attack Detection")
    print(f"{'='*60}")
    
    attack_names = get_attack_names()
    few_shot_types = get_few_shot_attack_types(data)
    print(f"Few-shot attack types: {few_shot_types}")
    
    n_shots_list = [1, 5, 10, 20]
    results = {}
    
    for attack_id, count in few_shot_types.items():
        attack_name = attack_names.get(attack_id, f'class_{attack_id}')
        results[attack_name] = {}
        
        for n_shots in n_shots_list:
            if n_shots > count:
                continue
                
            print(f"\n--- {attack_name}: {n_shots}-shot ---")
            
            train_mask, val_mask, test_mask = prepare_few_shot_split(
                data, attack_id, n_shots=n_shots
            )
            
            if model_name == 'DIDS_MFL':
                model = DIDS_MFL(num_nodes, feat_dim=40, memory_dim=32, hidden_dim=32)
            elif model_name == 'TGN':
                model = TGNBaseline(num_nodes, feat_dim=40, memory_dim=32, hidden_dim=32)
            elif model_name == 'MLP':
                model = MLPBaseline(feat_dim=40, hidden_dim=64)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            
            for epoch in range(epochs):
                loss_dict = train_epoch(model, data, train_mask, batch_size=batch_size,
                                       optimizer=optimizer)
            
            test_results = evaluate(model, data, test_mask, batch_size=batch_size)
            
            # Get F1 for the few-shot attack
            test_attacks = data.attack[test_mask].numpy()
            from sklearn.metrics import f1_score
            few_shot_mask = test_attacks == attack_id
            if few_shot_mask.sum() > 0:
                # Per-class F1 for the few-shot attack
                few_shot_f1 = test_results['per_attack_f1'].get(attack_name, 0.0)
            else:
                few_shot_f1 = 0.0
            
            results[attack_name][f'{n_shots}-shot'] = {
                'binary_f1': test_results['binary_f1'],
                'multi_f1_macro': test_results['multi_f1_macro'],
                'few_shot_attack_f1': few_shot_f1,
                'per_attack_f1': test_results['per_attack_f1'],
            }
            
            print(f"  Binary F1: {test_results['binary_f1']:.4f}")
            print(f"  Few-shot attack F1: {few_shot_f1:.4f}")
    
    return results


def run_ablation_study(data, num_nodes, epochs=10, batch_size=512, lr=0.001):
    """Run ablation study for DIDS-MFL components."""
    print(f"\n{'='*60}")
    print(f"Running Ablation Study")
    print(f"{'='*60}")
    
    train_mask, val_mask, test_mask = prepare_data_splits(data)
    
    variants = {
        'DIDS-MFL (Full)': {'use_sd': True, 'use_rd': True, 'use_gd': True, 'use_mfl': True},
        'w/o SD': {'use_sd': False, 'use_rd': True, 'use_gd': True, 'use_mfl': True},
        'w/o RD': {'use_sd': True, 'use_rd': False, 'use_gd': True, 'use_mfl': True},
        'w/o GD': {'use_sd': True, 'use_rd': True, 'use_gd': False, 'use_mfl': True},
        'w/o MFL': {'use_sd': True, 'use_rd': True, 'use_gd': True, 'use_mfl': False},
    }
    
    results = {}
    
    for variant_name, config in variants.items():
        print(f"\n--- {variant_name} ---")
        
        model = DIDS_MFL(num_nodes, feat_dim=40, memory_dim=32, hidden_dim=32)
        
        # Disable components based on config
        if not config['use_sd']:
            # Replace statistical disentanglement with identity
            model.stat_disentangle = torch.nn.Identity()
        if not config['use_rd']:
            model.rep_disentangle = torch.nn.Identity()
        if not config['use_gd']:
            # Replace graph diffusion with identity pass-through
            class IdentityDiffusion(torch.nn.Module):
                def forward(self, x_src, x_dst, src_layer, dst_layer, dt):
                    return x_src, x_dst
            model.graph_diffusion = IdentityDiffusion()
        if not config['use_mfl']:
            model.multi_scale_fusion = torch.nn.Identity()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        for epoch in range(epochs):
            # Custom training for ablation (handle Identity modules)
            model.train()
            if hasattr(model, 'reset_memory'):
                model.reset_memory()
            
            indices = train_mask
            n_train = len(indices)
            
            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                batch_idx = indices[start:end]
                
                src = data.src[batch_idx]
                dst = data.dst[batch_idx]
                msg = data.msg[batch_idx]
                t = data.t[batch_idx]
                dt_batch = data.dt[batch_idx]
                src_layer = data.src_layer[batch_idx]
                dst_layer = data.dst_layer[batch_idx]
                labels = data.label[batch_idx]
                attacks = data.attack[batch_idx]
                
                optimizer.zero_grad()
                
                try:
                    binary_logits, multi_logits, x_prev, x_curr = model(
                        src, dst, msg, t, dt_batch, src_layer, dst_layer
                    )
                except Exception as e:
                    # Fallback for ablated models
                    binary_logits, multi_logits, x_prev, x_curr = model(
                        src, dst, msg, t, dt_batch, src_layer, dst_layer
                    )
                
                loss = F.cross_entropy(binary_logits, labels) + F.cross_entropy(multi_logits, attacks)
                
                if config['use_sd'] and hasattr(model.stat_disentangle, 'disentangle_loss'):
                    loss += 0.05 * model.stat_disentangle.disentangle_loss(msg)
                if config['use_rd'] and hasattr(model.rep_disentangle, 'disentangle_loss'):
                    loss += 0.1 * RepresentationalDisentanglement.disentangle_loss(x_curr, x_prev)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # Evaluate
        test_results = evaluate(model, data, test_mask, batch_size=batch_size)
        
        results[variant_name] = {
            'binary_f1': test_results['binary_f1'],
            'binary_auc': test_results['binary_auc'],
            'binary_precision': test_results['binary_precision'],
            'binary_recall': test_results['binary_recall'],
            'multi_f1_macro': test_results['multi_f1_macro'],
        }
        
        print(f"  Binary F1: {test_results['binary_f1']:.4f}")
        print(f"  Binary AUC: {test_results['binary_auc']:.4f}")
        print(f"  Multi F1: {test_results['multi_f1_macro']:.4f}")
    
    return results


def main():
    print("Loading data...")
    data = torch.load(DATA_PATH, weights_only=False)
    print(f"Data loaded: {len(data.src)} edges")
    
    # Remap nodes
    print("Remapping node IDs...")
    data, num_nodes = remap_nodes(data)
    print(f"Num nodes: {num_nodes}")
    
    all_results = {}
    
    # ============================================================
    # Experiment 1: Standard Classification (all models)
    # ============================================================
    for model_name in ['MLP', 'TGN', 'DIDS_MFL']:
        result = run_standard_experiment(
            data, num_nodes, model_name=model_name,
            epochs=15, batch_size=512, lr=0.001
        )
        all_results[f'standard_{model_name}'] = result
        
        # Save intermediate
        save_results = {k: v for k, v in result.items() 
                       if k not in ['test_labels', 'test_attacks', 'test_binary_preds',
                                   'test_multi_preds', 'representations']}
        with open(os.path.join(OUTPUT_DIR, f'standard_{model_name}.json'), 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
    
    # Save representations for DIDS-MFL
    if 'standard_DIDS_MFL' in all_results and all_results['standard_DIDS_MFL'].get('representations') is not None:
        np.save(os.path.join(OUTPUT_DIR, 'dids_mfl_representations.npy'),
                all_results['standard_DIDS_MFL']['representations'])
        np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'),
                all_results['standard_DIDS_MFL']['test_labels'])
        np.save(os.path.join(OUTPUT_DIR, 'test_attacks.npy'),
                all_results['standard_DIDS_MFL']['test_attacks'])
    
    # ============================================================
    # Experiment 2: Unknown Attack Detection
    # ============================================================
    unknown_results = {}
    for model_name in ['MLP', 'TGN', 'DIDS_MFL']:
        result = run_unknown_attack_experiment(
            data, num_nodes, model_name=model_name,
            epochs=10, batch_size=512, lr=0.001
        )
        unknown_results[model_name] = result
    
    with open(os.path.join(OUTPUT_DIR, 'unknown_attack_results.json'), 'w') as f:
        json.dump(unknown_results, f, indent=2, default=str)
    all_results['unknown'] = unknown_results
    
    # ============================================================
    # Experiment 3: Few-Shot Attack Detection
    # ============================================================
    few_shot_results = {}
    for model_name in ['MLP', 'TGN', 'DIDS_MFL']:
        result = run_few_shot_experiment(
            data, num_nodes, model_name=model_name,
            epochs=10, batch_size=512, lr=0.001
        )
        few_shot_results[model_name] = result
    
    with open(os.path.join(OUTPUT_DIR, 'few_shot_results.json'), 'w') as f:
        json.dump(few_shot_results, f, indent=2, default=str)
    all_results['few_shot'] = few_shot_results
    
    # ============================================================
    # Experiment 4: Ablation Study
    # ============================================================
    ablation_results = run_ablation_study(
        data, num_nodes, epochs=10, batch_size=512, lr=0.001
    )
    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    all_results['ablation'] = ablation_results
    
    # Save all results
    print("\n\nAll experiments completed!")
    print("Results saved to outputs/")
    
    return all_results


if __name__ == '__main__':
    main()
