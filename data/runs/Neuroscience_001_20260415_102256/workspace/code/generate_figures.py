"""
Generate publication-quality figures for the Drosophila optic lobe DMN analysis.
"""

import torch
import numpy as np
import h5py
import os
import glob
import yaml
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


def load_all_models(base_dir='data/flow/0000'):
    """Load all 50 model checkpoints."""
    models = []
    model_dirs = sorted(glob.glob(os.path.join(base_dir, '*/')))
    model_dirs = [d for d in model_dirs if os.path.basename(d.strip('/')).isdigit()]
    
    for d in model_dirs:
        chkpt_path = os.path.join(d, 'best_chkpt')
        loss_path = os.path.join(d, 'validation_loss.h5')
        if not os.path.exists(chkpt_path):
            continue
        data = torch.load(chkpt_path, map_location='cpu', weights_only=False)
        net = data['network']
        val_loss = None
        if os.path.exists(loss_path):
            with h5py.File(loss_path, 'r') as f:
                val_loss = float(f['data'][()])
        config = None
        meta_path = os.path.join(d, '_meta.yaml')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                config = yaml.safe_load(f)
        model_info = {
            'dir': os.path.basename(d.strip('/')),
            'val_loss': val_loss,
            'config': config,
            'bias': net['nodes_bias'].numpy().copy(),
            'time_const': net['nodes_time_const'].numpy().copy(),
            'sign': net['edges_sign'].numpy().copy(),
            'syn_count': net['edges_syn_count'].numpy().copy(),
            'syn_strength': net['edges_syn_strength'].numpy().copy(),
        }
        models.append(model_info)
    print(f"Loaded {len(models)} models")
    return models


def figure_1_validation_losses(models):
    losses = np.array([m['val_loss'] for m in models])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sorted_idx = np.argsort(losses)
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(losses)))
    ax.bar(range(len(losses)), losses[sorted_idx], color=colors, edgecolor='none', width=0.8)
    ax.axhline(y=losses.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {losses.mean():.4f}')
    ax.axhline(y=losses.min(), color='green', linestyle=':', linewidth=1.5, label=f'Best: {losses.min():.4f}')
    ax.set_xlabel('Model (sorted by loss)')
    ax.set_ylabel('Validation Loss (L2 norm)')
    ax.set_title('Validation Loss Across 50 Ensemble Models')
    ax.legend(loc='upper right')
    ax.set_xticks([])
    ax = axes[1]
    ax.hist(losses, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=losses.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {losses.mean():.4f}')
    ax.axvline(x=np.median(losses), color='orange', linestyle='-.', linewidth=1.5, label=f'Median: {np.median(losses):.4f}')
    ax.set_xlabel('Validation Loss')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Validation Losses')
    ax.legend()
    plt.tight_layout()
    plt.savefig('report/images/fig1_validation_losses.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_validation_losses.png")


def figure_2_parameter_distributions(models):
    biases = np.array([m['bias'] for m in models])
    time_consts = np.array([m['time_const'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    syn_counts = np.array([m['syn_count'] for m in models])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    n_cell = biases.shape[1]  # 65
    
    ax = axes[0, 0]
    bias_mean = biases.mean(axis=0)
    bias_std = biases.std(axis=0)
    x = np.arange(n_cell)
    ax.bar(x, bias_mean, yerr=bias_std, capsize=2, color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Cell Type Index')
    ax.set_ylabel('Resting Potential (mean +/- std)')
    ax.set_title('A: Learned Resting Potentials (65 cell types)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    ax = axes[0, 1]
    tc_mean = time_consts.mean(axis=0)
    tc_std = time_consts.std(axis=0)
    ax.bar(x, tc_mean, yerr=tc_std, capsize=2, color='coral', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Cell Type Index')
    ax.set_ylabel('Time Constant (mean +/- std)')
    ax.set_title('B: Learned Time Constants (65 cell types)')
    
    ax = axes[1, 0]
    ss_mean = syn_strengths.mean(axis=0)
    ss_std = syn_strengths.std(axis=0)
    n_conn = len(ss_mean)
    ax.bar(np.arange(n_conn), ss_mean, yerr=ss_std, capsize=1, color='mediumseagreen', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Connection Index (604 connections)')
    ax.set_ylabel('Synaptic Strength (mean +/- std)')
    ax.set_title('C: Learned Synaptic Strengths')
    
    ax = axes[1, 1]
    sc_mean = syn_counts.mean(axis=0)
    sc_std = syn_counts.std(axis=0)
    n_sc = len(sc_mean)
    ax.bar(np.arange(n_sc), sc_mean, yerr=sc_std, capsize=1, color='goldenrod', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Connection Index (2355 entries)')
    ax.set_ylabel('Synapse Count (mean +/- std)')
    ax.set_title('D: Synapse Counts (from connectome)')
    
    plt.tight_layout()
    plt.savefig('report/images/fig2_parameter_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_parameter_distributions.png")


def figure_3_parameter_summary_statistics(models):
    biases = np.array([m['bias'] for m in models])
    time_consts = np.array([m['time_const'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    signs = np.array([m['sign'] for m in models])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax = axes[0, 0]
    bias_mean = biases.mean(axis=0)
    bias_std = biases.std(axis=0)
    ax.scatter(bias_mean, bias_std, s=50, c='steelblue', alpha=0.7, edgecolors='darkblue')
    ax.set_xlabel('Mean Resting Potential')
    ax.set_ylabel('Std Across Models')
    ax.set_title('A: Resting Potential Variability')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    tc_all = time_consts.flatten()
    ax.hist(tc_all, bins=30, color='coral', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Time Constant Value')
    ax.set_ylabel('Count')
    ax.set_title('B: Time Constant Distribution (all cell types x models)')
    
    ax = axes[1, 0]
    sign_mean = signs.mean(axis=0)
    excitatory_mask = sign_mean > 0.5
    inhibitory_mask = ~excitatory_mask
    exc_strengths = syn_strengths[:, excitatory_mask].flatten()
    inh_strengths = syn_strengths[:, inhibitory_mask].flatten()
    ax.hist(exc_strengths, bins=30, color='red', alpha=0.6, label='Excitatory', edgecolor='white')
    ax.hist(inh_strengths, bins=30, color='blue', alpha=0.6, label='Inhibitory', edgecolor='white')
    ax.set_xlabel('Synaptic Strength')
    ax.set_ylabel('Count')
    ax.set_title('C: Excitatory vs Inhibitory Strengths')
    ax.legend()
    
    ax = axes[1, 1]
    data_to_plot = [biases.flatten(), time_consts.flatten(), syn_strengths.flatten()]
    labels = ['Bias', 'Time Const', 'Syn Strength']
    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    colors = ['steelblue', 'coral', 'mediumseagreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Parameter Value')
    ax.set_title('D: Parameter Range Comparison')
    
    plt.tight_layout()
    plt.savefig('report/images/fig3_parameter_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_parameter_summary.png")


def figure_4_connectome_analysis(models):
    syn_counts = np.array([m['syn_count'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    signs = np.array([m['sign'] for m in models])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax = axes[0, 0]
    sc_mean = syn_counts.mean(axis=0)
    ax.hist(sc_mean, bins=30, color='goldenrod', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Mean Synapse Count')
    ax.set_ylabel('Number of Connections')
    ax.set_title('A: Synapse Count Distribution')
    ax.set_yscale('log')
    
    ax = axes[0, 1]
    sc_sorted = np.sort(sc_mean)[::-1]
    cumsum = np.cumsum(sc_sorted)
    total = cumsum[-1]
    ax.plot(cumsum / total * 100, color='darkgoldenrod', linewidth=2)
    ax.set_xlabel('Connection Rank')
    ax.set_ylabel('Cumulative Fraction (%)')
    ax.set_title('B: Cumulative Synapse Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    sign_mean = signs.mean(axis=0)
    n_excitatory = int(np.sum(sign_mean > 0.5))
    n_inhibitory = int(np.sum(sign_mean <= 0.5))
    ax.bar(['Excitatory', 'Inhibitory'], [n_excitatory, n_inhibitory], 
           color=['red', 'blue'], alpha=0.7, edgecolor='white')
    ax.set_ylabel('Number of Connections')
    ax.set_title('C: Excitatory vs Inhibitory Connections')
    ax.text(0, n_excitatory + max(5, n_excitatory*0.05), str(n_excitatory), ha='center', fontweight='bold')
    ax.text(1, n_inhibitory + max(5, n_inhibitory*0.05), str(n_inhibitory), ha='center', fontweight='bold')
    
    ax = axes[1, 1]
    ss_mean_vals = syn_strengths.mean(axis=0)
    n_ss = len(ss_mean_vals)
    sc_subset = sc_mean[:n_ss]
    ax.scatter(sc_subset, ss_mean_vals, s=20, c='teal', alpha=0.5)
    ax.set_xlabel('Mean Synapse Count')
    ax.set_ylabel('Mean Synaptic Strength')
    ax.set_title('D: Synapse Count vs Strength')
    corr = np.corrcoef(sc_subset, ss_mean_vals)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig4_connectome_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_connectome_analysis.png")


def figure_5_model_comparison(models):
    biases = np.array([m['bias'] for m in models])
    time_consts = np.array([m['time_const'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    losses = np.array([m['val_loss'] for m in models])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    best_idx = np.argmin(losses)
    worst_idx = np.argmax(losses)
    n_cell = biases.shape[1]
    
    ax = axes[0, 0]
    x = np.arange(n_cell)
    ax.scatter(x, biases[best_idx], s=30, c='green', alpha=0.7, label='Best model', zorder=3)
    ax.scatter(x, biases[worst_idx], s=30, c='red', alpha=0.7, label='Worst model', zorder=3)
    ax.plot(x, biases.mean(axis=0), 'k-', linewidth=1.5, label='Ensemble mean')
    ax.fill_between(x, 
                    biases.mean(axis=0) - biases.std(axis=0),
                    biases.mean(axis=0) + biases.std(axis=0),
                    alpha=0.2, color='gray', label='+/-1 std')
    ax.set_xlabel('Cell Type Index')
    ax.set_ylabel('Resting Potential')
    ax.set_title('A: Best vs Worst Model Biases')
    ax.legend(fontsize=8)
    
    ax = axes[0, 1]
    cv_bias = biases.std(axis=0) / (np.abs(biases.mean(axis=0)) + 1e-8)
    cv_tc = time_consts.std(axis=0) / (np.abs(time_consts.mean(axis=0)) + 1e-8)
    ss_mean = syn_strengths.mean(axis=0)
    cv_ss = syn_strengths.std(axis=0) / (np.abs(ss_mean) + 1e-8)
    
    ax.scatter(x, cv_bias, s=20, c='steelblue', alpha=0.6, label='Bias')
    ax.scatter(x, cv_tc, s=20, c='coral', alpha=0.6, label='Time const')
    ax.hist(cv_ss, bins=20, color='mediumseagreen', alpha=0.5, label='Syn strength (hist)', edgecolor='white')
    ax.set_xlabel('Parameter Index / Bin')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('B: Parameter Variability (CV)')
    ax.legend(fontsize=8)
    
    ax = axes[1, 0]
    n_models = len(models)
    corr_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i, n_models):
            c = np.corrcoef(biases[i], biases[j])[0, 1]
            corr_matrix[i, j] = c
            corr_matrix[j, i] = c
    im = ax.imshow(corr_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Model Index')
    ax.set_ylabel('Model Index')
    ax.set_title('C: Pairwise Bias Correlation')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1, 1]
    param_diversity = biases.std(axis=1)  # shape (50,) - diversity per model
    ax.scatter(param_diversity, losses, s=50, c='purple', alpha=0.7)
    ax.set_xlabel('Average Parameter Diversity')
    ax.set_ylabel('Validation Loss')
    ax.set_title('D: Parameter Diversity vs Performance')
    corr = np.corrcoef(param_diversity, losses)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig5_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_model_comparison.png")


def figure_6_ensemble_consensus(models):
    biases = np.array([m['bias'] for m in models])
    time_consts = np.array([m['time_const'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    losses = np.array([m['val_loss'] for m in models])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax = axes[0, 0]
    sorted_idx = np.argsort(losses)
    sorted_losses = losses[sorted_idx]
    n_boot = 1000
    boot_means = []
    for _ in range(n_boot):
        sample_idx = np.random.choice(len(losses), len(losses), replace=True)
        boot_means.append(losses[sample_idx].mean())
    boot_ci = np.percentile(boot_means, [2.5, 97.5])
    ax.plot(sorted_losses, 'o-', color='steelblue', markersize=4, linewidth=1)
    ax.fill_between(range(len(losses)), boot_ci[0], boot_ci[1], alpha=0.2, color='gray',
                    label=f'95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]')
    ax.set_xlabel('Model Rank')
    ax.set_ylabel('Validation Loss')
    ax.set_title('A: Model Performance Ranking')
    ax.legend(fontsize=8)
    
    ax = axes[0, 1]
    n_range = np.arange(5, 51, 5)
    bias_stability = []
    for n in n_range:
        subset_biases = biases[:n]
        mean_estimate = subset_biases.mean(axis=0)
        full_mean = biases.mean(axis=0)
        error = np.sqrt(((mean_estimate - full_mean)**2).mean())
        bias_stability.append(error)
    ax.plot(n_range, bias_stability, 's-', color='coral', markersize=6, linewidth=2)
    ax.set_xlabel('Number of Models in Ensemble')
    ax.set_ylabel('RMSE vs Full Ensemble')
    ax.set_title('B: Parameter Estimate Stability')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    bias_mean = biases.mean(axis=0)
    bias_norm = (bias_mean - bias_mean.min()) / (bias_mean.max() - bias_mean.min() + 1e-8)
    im = ax.imshow(bias_norm.reshape(1, -1), cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xlabel('Cell Type Index')
    ax.set_title('C: Normalized Resting Potential Heatmap')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Value')
    
    ax = axes[1, 1]
    within_var = biases.var(axis=1)
    between_var = biases.var(axis=0)
    ax.hist(within_var.flatten(), bins=20, alpha=0.6, color='steelblue', label='Within-model variance', edgecolor='white')
    ax.hist(between_var.flatten(), bins=20, alpha=0.6, color='coral', label='Between-model variance', edgecolor='white')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Count')
    ax.set_title('D: Variance Decomposition')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('report/images/fig6_ensemble_consensus.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig6_ensemble_consensus.png")


def figure_7_motion_pathway_architecture(models):
    syn_counts = np.array([m['syn_count'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    signs = np.array([m['sign'] for m in models])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax = axes[0, 0]
    ss_mean = syn_strengths.mean(axis=0)
    sorted_idx = np.argsort(ss_mean)[::-1]
    top_n = 50
    ax.barh(range(top_n), ss_mean[sorted_idx][:top_n], color='steelblue', edgecolor='white')
    ax.set_xlabel('Mean Synaptic Strength')
    ax.set_ylabel('Connection Rank')
    ax.set_title('A: Top 50 Strongest Connections')
    ax.invert_yaxis()
    
    ax = axes[0, 1]
    sign_mean = signs.mean(axis=0)
    exc_mask = sign_mean > 0.5
    inh_mask = ~exc_mask
    exc_strength = syn_strengths[:, exc_mask]
    inh_strength = syn_strengths[:, inh_mask]
    ax.boxplot([exc_strength.flatten(), inh_strength.flatten()], 
               tick_labels=['Excitatory', 'Inhibitory'], patch_artist=True)
    for patch, color in zip(ax.patches, ['red', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('Synaptic Strength')
    ax.set_title('B: Strength by Connection Type')
    
    ax = axes[1, 0]
    sc_mean = syn_counts.mean(axis=0)
    ss_mean = syn_strengths.mean(axis=0)
    n_common = min(len(sc_mean), len(ss_mean))
    mask = (sc_mean[:n_common] > 0) & (ss_mean[:n_common] > 0)
    ax.scatter(np.log10(sc_mean[:n_common][mask]), np.log10(ss_mean[:n_common][mask]), 
               s=15, c='teal', alpha=0.5)
    ax.set_xlabel('log10(Synapse Count)')
    ax.set_ylabel('log10(Synaptic Strength)')
    ax.set_title('C: Count-Strength Relationship')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    n_cell_types = 65
    adj_matrix = np.zeros((n_cell_types, n_cell_types))
    sc_mean = syn_counts.mean(axis=0)
    max_count = sc_mean.max()
    block_size = int(np.sqrt(len(sc_mean)))
    for i in range(min(n_cell_types, block_size)):
        for j in range(min(n_cell_types, block_size)):
            idx = i * block_size + j
            if idx < len(sc_mean):
                adj_matrix[i, j] = sc_mean[idx] / max_count
    im = ax.imshow(adj_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Target Cell Type')
    ax.set_ylabel('Source Cell Type')
    ax.set_title('D: Connectivity Matrix (Sample)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized Count')
    
    plt.tight_layout()
    plt.savefig('report/images/fig7_motion_pathway.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig7_motion_pathway.png")


def figure_8_task_optimization_results(models):
    biases = np.array([m['bias'] for m in models])
    time_consts = np.array([m['time_const'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    losses = np.array([m['val_loss'] for m in models])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax = axes[0, 0]
    params = np.hstack([biases, time_consts, syn_strengths])
    params_centered = params - params.mean(axis=0)
    U, S, Vt = np.linalg.svd(params_centered, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    pc2 = U[:, 1] * S[1]
    scatter = ax.scatter(pc1, pc2, c=losses, cmap='viridis_r', s=80, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('A: Parameter Space (colored by loss)')
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label='Validation Loss')
    
    ax = axes[0, 1]
    metrics = {
        'Mean Loss': losses.mean(),
        'Median Loss': np.median(losses),
        'Min Loss': losses.min(),
        'Max Loss': losses.max(),
        'Std Loss': losses.std(),
    }
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), 
                   color=['steelblue', 'coral', 'mediumseagreen', 'goldenrod', 'purple'])
    for bar, val in zip(bars, metrics.values()):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=9)
    ax.set_xlabel('Value')
    ax.set_title('B: Performance Summary Statistics')
    
    ax = axes[1, 0]
    var_bias = biases.var(axis=0).sum()
    var_tc = time_consts.var(axis=0).sum()
    var_ss = syn_strengths.var(axis=0).sum()
    total_var = var_bias + var_tc + var_ss
    contributions = [var_bias/total_var*100, var_tc/total_var*100, var_ss/total_var*100]
    labels = ['Bias\n(Resting Potential)', 'Time Constant', 'Synaptic Strength']
    colors = ['steelblue', 'coral', 'mediumseagreen']
    ax.pie(contributions, labels=labels, autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 9})
    ax.set_title('C: Parameter Variance Contribution')
    
    ax = axes[1, 1]
    bias_mean = biases.mean(axis=0)
    bias_se = biases.std(axis=0) / np.sqrt(len(models))
    x = np.arange(65)
    ax.errorbar(x, bias_mean, yerr=2*bias_se, fmt='.', color='steelblue', 
                alpha=0.6, capsize=2, markersize=4)
    ax.fill_between(x, bias_mean - 2*bias_se, bias_mean + 2*bias_se,
                    alpha=0.2, color='steelblue')
    ax.set_xlabel('Cell Type Index')
    ax.set_ylabel('Resting Potential\n(mean +/- 2xSE)')
    ax.set_title('D: Ensemble Prediction Reliability')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig8_task_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig8_task_optimization.png")


def save_detailed_outputs(models):
    biases = np.array([m['bias'] for m in models])
    time_consts = np.array([m['time_const'] for m in models])
    syn_strengths = np.array([m['syn_strength'] for m in models])
    syn_counts = np.array([m['syn_count'] for m in models])
    signs = np.array([m['sign'] for m in models])
    losses = np.array([m['val_loss'] for m in models])
    
    detailed_stats = {
        'n_models': len(models),
        'n_cell_types': 65,
        'n_connections': 604,
        'validation_losses': {
            'values': losses.tolist(),
            'mean': float(losses.mean()),
            'std': float(losses.std()),
            'min': float(losses.min()),
            'max': float(losses.max()),
            'median': float(np.median(losses)),
            'q25': float(np.percentile(losses, 25)),
            'q75': float(np.percentile(losses, 75)),
        },
        'bias': {
            'mean': biases.mean(axis=0).tolist(),
            'std': biases.std(axis=0).tolist(),
            'min': biases.min(axis=0).tolist(),
            'max': biases.max(axis=0).tolist(),
        },
        'time_constant': {
            'mean': time_consts.mean(axis=0).tolist(),
            'std': time_consts.std(axis=0).tolist(),
        },
        'synaptic_strength': {
            'mean': syn_strengths.mean(axis=0).tolist(),
            'std': syn_strengths.std(axis=0).tolist(),
        },
        'synapse_count': {
            'mean': syn_counts.mean(axis=0).tolist(),
            'std': syn_counts.std(axis=0).tolist(),
        },
        'connection_sign': {
            'mean': signs.mean(axis=0).tolist(),
            'n_excitatory': int((signs.mean(axis=0) > 0.5).sum()),
            'n_inhibitory': int((signs.mean(axis=0) <= 0.5).sum()),
        },
    }
    with open('outputs/detailed_statistics.json', 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    
    rankings = []
    for i, m in enumerate(models):
        rankings.append({
            'model_id': m['dir'],
            'validation_loss': float(m['val_loss']),
            'rank': int(np.argsort(losses)[i]) + 1,
        })
    rankings.sort(key=lambda x: x['validation_loss'])
    with open('outputs/model_rankings.json', 'w') as f:
        json.dump(rankings, f, indent=2)
    print("Saved detailed outputs")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating publication-quality figures...")
    print("=" * 60)
    models = load_all_models()
    
    print("\nGenerating Figure 1: Validation Losses...")
    figure_1_validation_losses(models)
    print("\nGenerating Figure 2: Parameter Distributions...")
    figure_2_parameter_distributions(models)
    print("\nGenerating Figure 3: Parameter Summary Statistics...")
    figure_3_parameter_summary_statistics(models)
    print("\nGenerating Figure 4: Connectome Analysis...")
    figure_4_connectome_analysis(models)
    print("\nGenerating Figure 5: Model Comparison...")
    figure_5_model_comparison(models)
    print("\nGenerating Figure 6: Ensemble Consensus...")
    figure_6_ensemble_consensus(models)
    print("\nGenerating Figure 7: Motion Pathway Architecture...")
    figure_7_motion_pathway_architecture(models)
    print("\nGenerating Figure 8: Task Optimization Results...")
    figure_8_task_optimization_results(models)
    print("\nSaving detailed outputs...")
    save_detailed_outputs(models)
    print("\nAll figures generated successfully!")
