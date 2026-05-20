"""
Load and analyze DMN ensemble data for Drosophila optic lobe motion pathway.
"""
import torch
import numpy as np
import h5py
import yaml
import os
import pickle
import sys
import json

# Add fake flyvis to path for pickle loading
sys.path.insert(0, '/tmp/fake_flyvis_final')

def load_all_checkpoints(base_dir='data/flow/0000'):
    """Load all 50 model checkpoints."""
    checkpoints = []
    losses = []
    model_dirs = sorted([d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()])
    
    for model_dir in model_dirs:
        chkpt_path = os.path.join(base_dir, model_dir, 'best_chkpt')
        loss_path = os.path.join(base_dir, model_dir, 'validation', 'loss.h5')
        
        if os.path.exists(chkpt_path):
            state = torch.load(chkpt_path, map_location='cpu')
            checkpoints.append(state)
        
        if os.path.exists(loss_path):
            with h5py.File(loss_path, 'r') as f:
                losses.append(f['data'][()])
        else:
            losses.append(np.nan)
    
    return checkpoints, np.array(losses), model_dirs


def extract_parameters(checkpoints):
    """Extract network parameters from all checkpoints."""
    n_models = len(checkpoints)
    
    # Each checkpoint has the same structure
    biases = []
    time_consts = []
    signs = []
    syn_counts = []
    syn_strengths = []
    
    for state in checkpoints:
        biases.append(state['network']['nodes_bias'].numpy())
        time_consts.append(state['network']['nodes_time_const'].numpy())
        signs.append(state['network']['edges_sign'].numpy())
        syn_counts.append(state['network']['edges_syn_count'].numpy())
        syn_strengths.append(state['network']['edges_syn_strength'].numpy())
    
    return {
        'bias': np.stack(biases),           # (n_models, 65)
        'time_const': np.stack(time_consts), # (n_models, 65)
        'sign': np.stack(signs),             # (n_models, 604)
        'syn_count': np.stack(syn_counts),   # (n_models, 2355)
        'syn_strength': np.stack(syn_strengths), # (n_models, 604)
    }


def load_clustering_data(base_dir='data/flow/0000/umap_and_clustering'):
    """Load UMAP and clustering data for all cell types."""
    clustering = {}
    for fname in sorted(os.listdir(base_dir)):
        if fname.endswith('.pickle'):
            cell_type = fname.replace('.pickle', '')
            with open(os.path.join(base_dir, fname), 'rb') as f:
                clustering[cell_type] = pickle.load(f)
    return clustering


if __name__ == '__main__':
    print("Loading checkpoints...")
    checkpoints, losses, model_dirs = load_all_checkpoints()
    print(f"Loaded {len(checkpoints)} checkpoints")
    print(f"Losses range: {np.nanmin(losses):.4f} - {np.nanmax(losses):.4f}")
    
    params = extract_parameters(checkpoints)
    print(f"\nParameter shapes:")
    for k, v in params.items():
        print(f"  {k}: {v.shape}")
    
    print("\nLoading clustering data...")
    clustering = load_clustering_data()
    print(f"Loaded clustering for {len(clustering)} cell types")
    
    # Save extracted data for downstream analysis
    os.makedirs('outputs', exist_ok=True)
    np.savez('outputs/ensemble_parameters.npz', **params)
    np.save('outputs/validation_losses.npy', losses)
    
    # Save clustering labels
    clustering_labels = {}
    for ct, cl in clustering.items():
        clustering_labels[ct] = cl.labels.tolist()
    with open('outputs/clustering_labels.json', 'w') as f:
        json.dump(clustering_labels, f)
    
    print("\nData saved to outputs/")
