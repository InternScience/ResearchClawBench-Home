"""
Analysis Script 1: Load all 50 model checkpoints and extract learned parameters.
Save aggregated parameter data for downstream analysis.
"""
import torch
import numpy as np
import json
import os
import h5py
import warnings
warnings.filterwarnings('ignore')

BASE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Neuroscience_001_20260417_004607'
DATA = os.path.join(BASE, 'data/flow/0000')
OUT = os.path.join(BASE, 'outputs')

# Load connectome for cell type names
with open('/home/chenyixin/.local/lib/python3.10/site-packages/flyvis/connectome/fib25-fib19_v2.2.json') as f:
    conn = json.load(f)

cell_types = [n['name'] for n in conn['nodes']]
edges = conn['edges']
edge_pairs = [(e['src'], e['tar']) for e in edges]
input_units = conn['input_units']
output_units = conn['output_units']

print(f"Cell types: {len(cell_types)}")
print(f"Edges: {len(edges)}")
print(f"Input units: {input_units}")

# Load all 50 models
n_models = 50
all_bias = []
all_time_const = []
all_sign = []
all_syn_strength = []
all_val_loss = []
all_val_loss_best = []

for i in range(n_models):
    model_dir = os.path.join(DATA, f'{i:03d}')
    
    # Load best checkpoint
    chkpt_path = os.path.join(model_dir, 'best_chkpt')
    if not os.path.exists(chkpt_path):
        print(f"Model {i:03d}: best_chkpt not found, skipping")
        continue
    
    chkpt = torch.load(chkpt_path, map_location='cpu', weights_only=False)
    net = chkpt['network']
    
    all_bias.append(net['nodes_bias'].numpy())
    all_time_const.append(net['nodes_time_const'].numpy())
    all_sign.append(net['edges_sign'].numpy())
    all_syn_strength.append(net['edges_syn_strength'].numpy())
    
    # Load validation loss
    loss_path = os.path.join(model_dir, 'validation/loss.h5')
    if os.path.exists(loss_path):
        with h5py.File(loss_path, 'r') as f:
            all_val_loss_best.append(f['data'][()])
    
    loss_path2 = os.path.join(model_dir, 'validation_loss.h5')
    if os.path.exists(loss_path2):
        with h5py.File(loss_path2, 'r') as f:
            all_val_loss.append(f['data'][()])
    
    if i % 10 == 0:
        print(f"Loaded model {i:03d}")

print(f"\nLoaded {len(all_bias)} models")

# Convert to arrays
all_bias = np.array(all_bias)  # (50, 65)
all_time_const = np.array(all_time_const)  # (50, 65)
all_sign = np.array(all_sign)  # (50, 604)
all_syn_strength = np.array(all_syn_strength)  # (50, 604)
all_val_loss = np.array(all_val_loss)
all_val_loss_best = np.array(all_val_loss_best)

print(f"Bias shape: {all_bias.shape}")
print(f"Time const shape: {all_time_const.shape}")
print(f"Sign shape: {all_sign.shape}")
print(f"Syn strength shape: {all_syn_strength.shape}")
print(f"Val loss shape: {all_val_loss.shape}")
print(f"Val loss best shape: {all_val_loss_best.shape}")

# Save
np.savez(os.path.join(OUT, 'model_parameters.npz'),
         bias=all_bias,
         time_const=all_time_const,
         sign=all_sign,
         syn_strength=all_syn_strength,
         val_loss=all_val_loss,
         val_loss_best=all_val_loss_best,
         cell_types=cell_types)

# Save edge info
edge_info = {
    'edge_pairs': edge_pairs,
    'cell_types': cell_types,
    'input_units': input_units,
    'output_units': output_units,
}
with open(os.path.join(OUT, 'edge_info.json'), 'w') as f:
    json.dump(edge_info, f)

# Summary stats
print("\n=== Parameter Summary ===")
print(f"Bias (resting potential) - mean: {all_bias.mean():.4f}, std: {all_bias.std():.4f}")
print(f"Time constant - mean: {all_time_const.mean():.4f}, std: {all_time_const.std():.4f}")
print(f"Syn strength - mean: {all_syn_strength.mean():.4f}, std: {all_syn_strength.std():.4f}")
print(f"Validation loss - mean: {all_val_loss.mean():.4f}, std: {all_val_loss.std():.4f}")
print(f"Best validation loss - mean: {all_val_loss_best.mean():.4f}, std: {all_val_loss_best.std():.4f}")

# Per-cell-type stats
print("\n=== Per-Cell-Type Bias (mean ± std across 50 models) ===")
for i, ct in enumerate(cell_types):
    print(f"  {ct:12s}: {all_bias[:,i].mean():.4f} ± {all_bias[:,i].std():.4f}")

print("\n=== Per-Cell-Type Time Constant (mean ± std across 50 models) ===")
for i, ct in enumerate(cell_types):
    print(f"  {ct:12s}: {all_time_const[:,i].mean():.4f} ± {all_time_const[:,i].std():.4f}")
