#!/usr/bin/env python3
"""
Analysis script for Drosophila optic lobe Deep Mechanistic Network (DMN) models.

This script analyzes 50 pre-trained DMN models constrained by the fly connectome
and optimized for optic flow estimation.
"""

import os
import sys
import numpy as np
import h5py
import yaml
import zipfile
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Neuroscience_001_20260416_211156"
DATA_DIR = os.path.join(WORKSPACE, "data/flow/0000")
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_model_data(model_id):
    """Load checkpoint data from a single model."""
    checkpoint_path = os.path.join(DATA_DIR, model_id, "best_chkpt")
    val_loss_path = os.path.join(DATA_DIR, model_id, "validation_loss.h5")
    meta_path = os.path.join(DATA_DIR, model_id, "_meta.yaml")
    
    # Load metadata
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    
    # Load validation loss
    with h5py.File(val_loss_path, 'r') as f:
        val_loss = float(f['data'][()])
    
    # Load checkpoint parameters
    params = {}
    with zipfile.ZipFile(checkpoint_path, 'r') as z:
        param_names = ['data/0', 'data/1', 'data/2', 'data/3', 'data/4', 
                       'data/5', 'data/6', 'data/7', 'data/8', 'data/9',
                       'data/10', 'data/11', 'data/12', 'data/13']
        param_keys = ['resting_potentials', 'time_constants', 'synapse_signs',
                      'synapse_strengths', 'synapse_scaling', 'decoder_w1',
                      'decoder_b1', 'decoder_w2', 'decoder_b2', 'decoder_w3',
                      'decoder_b3', 'decoder_out', 'decoder_extra', 'misc']
        
        for pname, pkey in zip(param_names, param_keys):
            try:
                with z.open(f'best_chkpt/{pname}') as f:
                    raw = f.read()
                    params[pkey] = np.frombuffer(raw, dtype=np.float32)
            except:
                params[pkey] = None
    
    return {
        'model_id': model_id,
        'meta': meta,
        'val_loss': val_loss,
        'params': params
    }


def load_all_models():
    """Load data from all 50 models."""
    model_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.isdigit()])
    models = []
    for model_id in model_dirs:
        print(f"Loading model {model_id}...")
        models.append(load_model_data(model_id))
    return models


def analyze_parameters(models):
    """Analyze parameter distributions across models."""
    # Extract arrays
    rp = np.array([m['params']['resting_potentials'] for m in models])
    tc = np.array([m['params']['time_constants'] for m in models])
    ss = np.array([m['params']['synapse_signs'] for m in models])
    st = np.array([m['params']['synapse_strengths'] for m in models])
    sc = np.array([m['params']['synapse_scaling'] for m in models])
    val_losses = np.array([m['val_loss'] for m in models])
    
    stats = {
        'resting_potentials': {
            'shape': rp.shape,
            'mean': float(rp.mean()),
            'std': float(rp.std()),
            'min': float(rp.min()),
            'max': float(rp.max()),
            'mean_per_type': rp.mean(axis=0).tolist(),
            'std_per_type': rp.std(axis=0).tolist()
        },
        'time_constants': {
            'shape': tc.shape,
            'mean': float(tc.mean()),
            'std': float(tc.std()),
            'min': float(tc.min()),
            'max': float(tc.max()),
            'mean_per_type': tc.mean(axis=0).tolist(),
            'std_per_type': tc.std(axis=0).tolist()
        },
        'synapse_signs': {
            'shape': ss.shape,
            'excitatory_count': int((ss > 0).sum()),
            'inhibitory_count': int((ss < 0).sum()),
            'excitatory_fraction': float((ss > 0).mean()),
            'sign_consistency': float((ss == ss[0]).all(axis=0).mean())
        },
        'synapse_strengths': {
            'shape': st.shape,
            'mean': float(st.mean()),
            'std': float(st.std()),
            'min': float(st.min()),
            'max': float(st.max())
        },
        'synapse_scaling': {
            'shape': sc.shape,
            'mean': float(sc.mean()),
            'std': float(sc.std()),
            'min': float(sc.min()),
            'max': float(sc.max())
        },
        'validation_losses': {
            'mean': float(val_losses.mean()),
            'std': float(val_losses.std()),
            'min': float(val_losses.min()),
            'max': float(val_losses.max()),
            'values': val_losses.tolist()
        }
    }
    
    return stats, rp, tc, ss, st, sc, val_losses


def create_visualizations(stats, rp, tc, ss, st, sc, val_losses):
    """Generate all analysis figures."""
    figures = {}
    
    # Figure 1: Validation loss distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(val_losses, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(val_losses.mean(), color='red', linestyle='--', label=f'Mean: {val_losses.mean():.3f}')
    ax.set_xlabel('Validation Loss')
    ax.set_ylabel('Number of Models')
    ax.set_title('Distribution of Validation Losses Across 50 DMN Models')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'fig1_validation_loss.png')
    plt.savefig(path, dpi=150)
    plt.close()
    figures['validation_loss'] = path
    
    # Figure 2: Resting potential distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(rp.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Resting Potential')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Resting Potentials\n(65 cell types × 50 models)')
    
    # Mean and std per cell type
    rp_mean = rp.mean(axis=0)
    rp_std = rp.std(axis=0)
    axes[1].errorbar(range(len(rp_mean)), rp_mean, yerr=rp_std, fmt='o', capsize=3)
    axes[1].set_xlabel('Cell Type Index')
    axes[1].set_ylabel('Resting Potential')
    axes[1].set_title('Resting Potential by Cell Type\n(Mean ± Std across models)')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'fig2_resting_potentials.png')
    plt.savefig(path, dpi=150)
    plt.close()
    figures['resting_potentials'] = path
    
    # Figure 3: Time constant distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(tc.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Time Constant')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Time Constants\n(65 cell types × 50 models)')
    
    # Mean and std per cell type
    tc_mean = tc.mean(axis=0)
    tc_std = tc.std(axis=0)
    axes[1].errorbar(range(len(tc_mean)), tc_mean, yerr=tc_std, fmt='o', capsize=3, color='orange')
    axes[1].set_xlabel('Cell Type Index')
    axes[1].set_ylabel('Time Constant')
    axes[1].set_title('Time Constant by Cell Type\n(Mean ± Std across models)')
    
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'fig3_time_constants.png')
    plt.savefig(path, dpi=150)
    plt.close()
    figures['time_constants'] = path
    
    # Figure 4: Synapse sign distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    exc_count = (ss > 0).sum()
    inh_count = (ss < 0).sum()
    axes[0].pie([exc_count, inh_count], labels=['Excitatory', 'Inhibitory'], 
                autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title(f'Synapse Sign Distribution\n(604 edge types × 50 models)')
    
    # Sign consistency across models
    sign_consistency = [(ss[:, i] == ss[0, i]).all() for i in range(ss.shape[1])]
    consistent_count = sum(sign_consistency)
    axes[1].bar(['Consistent', 'Variable'], [consistent_count, len(sign_consistency) - consistent_count],
                color=['#95E1D3', '#F38181'])
    axes[1].set_ylabel('Number of Edge Types')
    axes[1].set_title('Sign Consistency Across Models')
    
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'fig4_synapse_signs.png')
    plt.savefig(path, dpi=150)
    plt.close()
    figures['synapse_signs'] = path
    
    # Figure 5: Synapse strength distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(st.flatten(), bins=100, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Synapse Strength')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Synapse Strengths\n(2355 connections × 50 models)')
    
    # Strength vs scaling relationship (aggregate)
    st_mean = st.mean(axis=0)
    sc_mean = sc.mean(axis=0)
    # Note: shapes differ, so we can't directly compare
    axes[1].scatter(range(len(st_mean)), st_mean, s=10, alpha=0.5)
    axes[1].set_xlabel('Connection Index')
    axes[1].set_ylabel('Synapse Strength')
    axes[1].set_title('Synapse Strengths by Connection\n(Mean across models)')
    
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'fig5_synapse_strengths.png')
    plt.savefig(path, dpi=150)
    plt.close()
    figures['synapse_strengths'] = path
    
    # Figure 6: Parameter correlation heatmap (resting potentials)
    fig, ax = plt.subplots(figsize=(10, 8))
    rp_corr = np.corrcoef(rp.T)
    im = ax.imshow(rp_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Model Index')
    ax.set_ylabel('Cell Type Index')
    ax.set_title('Correlation of Resting Potentials\nAcross Models and Cell Types')
    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, 'fig6_parameter_correlation.png')
    plt.savefig(path, dpi=150)
    plt.close()
    figures['correlation'] = path
    
    return figures


def main():
    print("="*60)
    print("Drosophila Optic Lobe DMN Analysis")
    print("="*60)
    
    # Load all models
    print("\nLoading 50 DMN models...")
    models = load_all_models()
    
    # Analyze parameters
    print("\nAnalyzing parameters...")
    stats, rp, tc, ss, st, sc, val_losses = analyze_parameters(models)
    
    # Save statistics
    stats_path = os.path.join(OUTPUTS_DIR, 'parameter_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    figures = create_visualizations(stats, rp, tc, ss, st, sc, val_losses)
    
    print("\nGenerated figures:")
    for name, path in figures.items():
        print(f"  {name}: {path}")
    
    # Save figure manifest
    manifest_path = os.path.join(OUTPUTS_DIR, 'figure_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(figures, f, indent=2)
    
    print("\nAnalysis complete!")
    print(f"Total models analyzed: {len(models)}")
    print(f"Mean validation loss: {stats['validation_losses']['mean']:.4f} ± {stats['validation_losses']['std']:.4f}")
    print(f"Cell types: 65 (including glia)")
    print(f"Edge types: 604")
    print(f"Excitatory synapses: {stats['synapse_signs']['excitatory_fraction']*100:.1f}%")
    

if __name__ == "__main__":
    main()
