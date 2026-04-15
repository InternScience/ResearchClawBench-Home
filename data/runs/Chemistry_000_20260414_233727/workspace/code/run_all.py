"""
Run full experiments on all datasets with optimized settings.
"""
import subprocess
import json
import sys

datasets = ['bace', 'bbbp', 'clintox', 'hiv']
results_all = {}

for ds in datasets:
    print(f"\n{'='*60}")
    print(f"Running {ds}...")
    
    cmd = [
        sys.executable, 'code/train.py',
        '--datasets', ds,
        '--epochs', '80',
        '--batch_size', '128',
        '--hidden_dim', '96',
        '--num_layers', '3',
        '--grid_size', '5',
        '--lr', '1e-3',
        '--seed', '42'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    print(result.stdout[-2000:])  # Print last 2000 chars
    
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-500:]}")

# Load and merge results
with open('outputs/experiment_results.json', 'r') as f:
    results_all = json.load(f)

print("\n\nFinal Summary:")
print(f"{'Dataset':<12} {'Model':<10} {'Val AUC':<10} {'Test AUC':<10} {'Test Acc':<10} {'Time(s)':<10}")
print("-" * 62)
for ds_name, res in results_all.items():
    for model_name in ['GCN', 'KA-GNN']:
        if model_name in res:
            r = res[model_name]
            print(f"{ds_name:<12} {model_name:<10} {r['val_roc_auc']:<10.4f} "
                  f"{r['test_roc_auc']:<10.4f} {r['test_accuracy']:<10.4f} "
                  f"{r['training_time']:<10.1f}")
