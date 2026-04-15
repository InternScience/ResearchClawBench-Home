"""
Analysis of 50 pre-trained Deep Mechanistic Network (DMN) models 
constrained by the Drosophila optic lobe connectome and optimized for optic flow estimation.

This script:
1. Loads all 50 model checkpoints
2. Extracts and analyzes learned parameters (bias, time constants, synapse strengths)
3. Computes ensemble statistics across models
4. Generates publication-quality figures
"""

import torch
import numpy as np
import h5py
import os
import glob
import yaml
import json
import pickle

# ============================================================
# 1. Load all models and extract parameters
# ============================================================

def load_all_models(base_dir='data/flow/0000'):
    """Load all 50 model checkpoints and extract parameters."""
    models = []
    losses = []
    
    model_dirs = sorted(glob.glob(os.path.join(base_dir, '*/')))
    # Filter out non-model directories
    model_dirs = [d for d in model_dirs if os.path.basename(d.strip('/')).isdigit()]
    
    print(f"Found {len(model_dirs)} model directories")
    
    for d in model_dirs:
        chkpt_path = os.path.join(d, 'best_chkpt')
        loss_path = os.path.join(d, 'validation_loss.h5')
        meta_path = os.path.join(d, '_meta.yaml')
        
        if not os.path.exists(chkpt_path):
            continue
            
        # Load checkpoint
        data = torch.load(chkpt_path, map_location='cpu', weights_only=False)
        net = data['network']
        decoder = data.get('decoder', {})
        
        # Load validation loss
        val_loss = None
        if os.path.exists(loss_path):
            with h5py.File(loss_path, 'r') as f:
                val_loss = float(f['data'][()])
        
        # Load meta config
        config = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                config = yaml.safe_load(f)
        
        model_info = {
            'dir': os.path.basename(d.strip('/')),
            'network': net,
            'decoder': decoder,
            'val_loss': val_loss,
            'config': config,
            'bias': net['nodes_bias'].numpy().copy(),
            'time_const': net['nodes_time_const'].numpy().copy(),
            'sign': net['edges_sign'].numpy().copy(),
            'syn_count': net['edges_syn_count'].numpy().copy(),
            'syn_strength': net['edges_syn_strength'].numpy().copy(),
        }
        
        models.append(model_info)
        losses.append(val_loss)
    
    print(f"Successfully loaded {len(models)} models")
    return models, losses


def compute_ensemble_statistics(models):
    """Compute mean and std across all 50 models for each parameter."""
    stats = {}
    
    # Stack parameters across models
    biases = np.array([m['bias'] for m in models])
    time_consts = np.array([m['time_const'] for m in models])
    signs = np.array([m['sign'] for m in models])
    syn_counts = np.array([m['syn_count'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    
    stats['bias'] = {
        'mean': biases.mean(axis=0),
        'std': biases.std(axis=0),
        'all': biases,
    }
    stats['time_const'] = {
        'mean': time_consts.mean(axis=0),
        'std': time_consts.std(axis=0),
        'all': time_consts,
    }
    stats['sign'] = {
        'mean': signs.mean(axis=0),
        'all': signs,
    }
    stats['syn_count'] = {
        'mean': syn_counts.mean(axis=0),
        'std': syn_counts.std(axis=0),
        'all': syn_counts,
    }
    stats['syn_strength'] = {
        'mean': syn_strengths.mean(axis=0),
        'std': syn_strengths.std(axis=0),
        'all': syn_strengths,
    }
    
    return stats


def save_outputs(models, losses, stats):
    """Save intermediate results to outputs/ directory."""
    os.makedirs('outputs', exist_ok=True)
    
    # Save validation losses
    loss_data = {
        'model_ids': [m['dir'] for m in models],
        'losses': [m['val_loss'] for m in models],
        'mean_loss': np.mean([m['val_loss'] for m in models]),
        'std_loss': np.std([m['val_loss'] for m in models]),
        'min_loss': np.min([m['val_loss'] for m in models]),
        'max_loss': np.max([m['val_loss'] for m in models]),
        'median_loss': np.median([m['val_loss'] for m in models]),
    }
    with open('outputs/validation_losses.json', 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    # Save parameter statistics
    param_stats = {
        'bias_mean': stats['bias']['mean'].tolist(),
        'bias_std': stats['bias']['std'].tolist(),
        'time_const_mean': stats['time_const']['mean'].tolist(),
        'time_const_std': stats['time_const']['std'].tolist(),
        'sign_mean': stats['sign']['mean'].tolist(),
        'syn_count_mean': stats['syn_count']['mean'].tolist(),
        'syn_count_std': stats['syn_count']['std'].tolist(),
        'syn_strength_mean': stats['syn_strength']['mean'].tolist(),
        'syn_strength_std': stats['syn_strength']['std'].tolist(),
    }
    with open('outputs/parameter_statistics.json', 'w') as f:
        json.dump(param_stats, f, indent=2)
    
    # Save model summary
    summary = {
        'n_models': len(models),
        'n_cell_types': models[0]['bias'].shape[0],
        'n_edge_signs': models[0]['sign'].shape[0],
        'n_syn_counts': models[0]['syn_count'].shape[0],
        'n_syn_strengths': models[0]['syn_strength'].shape[0],
        'validation_losses': loss_data,
    }
    with open('outputs/model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Saved outputs to outputs/")


if __name__ == '__main__':
    print("=" * 60)
    print("Loading all 50 DMN models...")
    print("=" * 60)
    
    models, losses = load_all_models()
    
    print("\nComputing ensemble statistics...")
    stats = compute_ensemble_statistics(models)
    
    print("\nSaving outputs...")
    save_outputs(models, losses, stats)
    
    print("\nDone!")
