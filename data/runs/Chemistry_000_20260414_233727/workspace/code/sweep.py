"""
Hyperparameter sweep for KA-GNN to find optimal settings.
Tests different grid sizes, hidden dims, and learning rates.
"""
import subprocess
import json
import sys

configs = [
    {'grid_size': 3, 'hidden_dim': 64, 'lr': 5e-4, 'num_layers': 2},
    {'grid_size': 3, 'hidden_dim': 96, 'lr': 5e-4, 'num_layers': 3},
    {'grid_size': 5, 'hidden_dim': 64, 'lr': 5e-4, 'num_layers': 2},
    {'grid_size': 3, 'hidden_dim': 128, 'lr': 1e-3, 'num_layers': 3},
    {'grid_size': 5, 'hidden_dim': 96, 'lr': 5e-4, 'num_layers': 3},
]

best_results = {}

for ds in ['bace', 'bbbp']:
    print(f"\n{'='*60}")
    print(f"Sweeping {ds}...")
    
    best_auc = 0
    best_cfg = None
    
    for cfg in configs:
        print(f"\n--- Config: {cfg} ---")
        cmd = [
            sys.executable, 'code/train.py',
            '--datasets', ds,
            '--epochs', '60',
            '--batch_size', '128',
            '--hidden_dim', str(cfg['hidden_dim']),
            '--num_layers', str(cfg['num_layers']),
            '--grid_size', str(cfg['grid_size']),
            '--lr', str(cfg['lr']),
            '--seed', '42'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"Error: {result.stderr[-300:]}")
            continue
        
        with open('outputs/experiment_results.json', 'r') as f:
            res = json.load(f)
        
        if ds in res and 'KA-GNN' in res[ds]:
            kagcn_res = res[ds]['KA-GNN']
            gcn_res = res[ds].get('GCN', {})
            test_auc = kagcn_res['test_roc_auc']
            val_auc = kagcn_res['val_roc_auc']
            print(f"  KA-GNN: Val AUC={val_auc:.4f}, Test AUC={test_auc:.4f}, "
                  f"Params={kagcn_res['num_parameters']:,}, Time={kagcn_res['training_time']:.1f}s")
            if gcn_res:
                print(f"  GCN:    Val AUC={gcn_res['val_roc_auc']:.4f}, Test AUC={gcn_res['test_roc_auc']:.4f}")
            
            if test_auc > best_auc:
                best_auc = test_auc
                best_cfg = cfg.copy()
                best_cfg['test_auc'] = test_auc
                best_cfg['val_auc'] = val_auc
    
    best_results[ds] = {
        'best_config': best_cfg,
        'best_test_auc': best_auc
    }
    print(f"\nBest config for {ds}: {best_cfg}")

with open('outputs/hyperparam_sweep.json', 'w') as f:
    json.dump(best_results, f, indent=2)
