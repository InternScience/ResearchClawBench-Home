"""
Multi-seed experiment runner for robust statistical comparison.
"""
import subprocess
import json
import sys

datasets = ['bace', 'bbbp', 'clintox']  # HIV takes too long
seeds = [42, 123, 456, 789, 1024]
all_results = {}

for ds in datasets:
    print(f"\n{'='*60}")
    print(f"Running {ds} with multiple seeds...")
    
    ds_results = {}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        cmd = [
            sys.executable, 'code/train.py',
            '--datasets', ds,
            '--epochs', '60',
            '--batch_size', '128',
            '--hidden_dim', '96',
            '--num_layers', '3',
            '--grid_size', '5',
            '--lr', '1e-3',
            '--seed', str(seed)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            print(f"Error: {result.stderr[-500:]}")
            continue
        
        # Load results for this seed
        with open('outputs/experiment_results.json', 'r') as f:
            res = json.load(f)
        
        if ds in res:
            ds_results[f'seed_{seed}'] = res[ds]
            for model_name in ['GCN', 'KA-GNN']:
                if model_name in res[ds]:
                    r = res[ds][model_name]
                    print(f"  {model_name}: Val AUC={r['val_roc_auc']:.4f}, Test AUC={r['test_roc_auc']:.4f}")
    
    all_results[ds] = ds_results

# Save multi-seed results
with open('outputs/multi_seed_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n\nMulti-seed Summary:")
for ds, seeds_res in all_results.items():
    print(f"\n{ds}:")
    for model_name in ['GCN', 'KA-GNN']:
        val_aucs = []
        test_aucs = []
        times = []
        params = []
        for seed_key, res in seeds_res.items():
            if model_name in res:
                val_aucs.append(res[model_name]['val_roc_auc'])
                test_aucs.append(res[model_name]['test_roc_auc'])
                times.append(res[model_name]['training_time'])
                params.append(res[model_name]['num_parameters'])
        if val_aucs:
            import numpy as np
            print(f"  {model_name}: Val AUC={np.mean(val_aucs):.4f}±{np.std(val_aucs):.4f}, "
                  f"Test AUC={np.mean(test_aucs):.4f}±{np.std(test_aucs):.4f}, "
                  f"Time={np.mean(times):.1f}s, Params={params[0]:,}")
