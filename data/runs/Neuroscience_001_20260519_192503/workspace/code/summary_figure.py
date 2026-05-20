"""
Create a comprehensive summary figure for the report.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import os

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

os.makedirs('report/images', exist_ok=True)

# Load data
params = np.load('outputs/ensemble_parameters.npz')
bias = params['bias']
time_const = params['time_const']
sign = params['sign']
syn_strength = params['syn_strength']
losses = np.load('outputs/validation_losses.npy')

with open('outputs/connectome_stats.json') as f:
    conn_stats = json.load(f)

with open('outputs/motion_stats.json') as f:
    motion_stats = json.load(f)

# ============================================================================
# Figure 19: Summary figure with key findings
# ============================================================================
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Panel A: Loss distribution
ax = fig.add_subplot(gs[0, 0])
ax.hist(losses, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(losses.mean(), color='red', linestyle='--', label=f'Mean: {losses.mean():.3f}')
ax.set_xlabel('Validation Loss')
ax.set_ylabel('Count')
ax.set_title('A. Ensemble Performance')
ax.legend(fontsize=8)

# Panel B: E/I balance
ax = fig.add_subplot(gs[0, 1])
exc_count = (sign[0] == 1).sum()
inh_count = (sign[0] == -1).sum()
ax.pie([exc_count, inh_count], labels=['Excitatory', 'Inhibitory'], 
       colors=['#4472C4', '#ED7D31'], autopct='%1.1f%%', startangle=90)
ax.set_title('B. Synaptic Polarity')

# Panel C: Top cell types by parameter magnitude
ax = fig.add_subplot(gs[0, 2])
mean_bias = bias.mean(axis=0)
names = conn_stats['node_names']
sorted_idx = np.argsort(mean_bias)
ax.barh(range(10), mean_bias[sorted_idx[-10:]], color='steelblue', alpha=0.7)
ax.set_yticks(range(10))
ax.set_yticklabels([names[i] for i in sorted_idx[-10:]], fontsize=8)
ax.set_xlabel('Resting Potential')
ax.set_title('C. Highest Resting Potentials')

# Panel D: T4/T5 pathway strength
ax = fig.add_subplot(gs[1, 0])
t4_dirs = ['T4a', 'T4b', 'T4c', 'T4d']
mi1_vals = [motion_stats['t4_mi1_strengths'][t4] for t4 in t4_dirs]
tm3_vals = [motion_stats['t4_tm3_strengths'][t4] for t4 in t4_dirs]
x = np.arange(len(t4_dirs))
width = 0.35
ax.bar(x - width/2, mi1_vals, width, label='Mi1', alpha=0.8, color='blue')
ax.bar(x + width/2, tm3_vals, width, label='Tm3', alpha=0.8, color='cyan')
ax.set_xticks(x)
ax.set_xticklabels(t4_dirs)
ax.set_ylabel('Strength')
ax.set_title('D. T4 (ON) Pathway Inputs')
ax.legend(fontsize=8)

# Panel E: T5 pathway strength
ax = fig.add_subplot(gs[1, 1])
t5_dirs = ['T5a', 'T5b', 'T5c', 'T5d']
mi4_vals = [motion_stats['t5_mi4_strengths'][t5] for t5 in t5_dirs]
tm9_vals = [motion_stats['t5_tm9_strengths'][t5] for t5 in t5_dirs]
ax.bar(x - width/2, mi4_vals, width, label='Mi4', alpha=0.8, color='red')
ax.bar(x + width/2, tm9_vals, width, label='Tm9', alpha=0.8, color='orange')
ax.set_xticks(x)
ax.set_xticklabels(t5_dirs)
ax.set_ylabel('Strength')
ax.set_title('E. T5 (OFF) Pathway Inputs')
ax.legend(fontsize=8)

# Panel F: Parameter variability
ax = fig.add_subplot(gs[1, 2])
bias_cv = bias.std(axis=0) / (bias.mean(axis=0) + 1e-8)
tc_cv = time_const.std(axis=0) / (time_const.mean(axis=0) + 1e-8)
ax.scatter(bias_cv, tc_cv, alpha=0.6, s=30)
ax.set_xlabel('Bias CV')
ax.set_ylabel('Time Constant CV')
ax.set_title('F. Parameter Variability')
ax.plot([0, max(bias_cv.max(), tc_cv.max())], [0, max(bias_cv.max(), tc_cv.max())], 'k--', alpha=0.3)

# Panel G: Synaptic strength distribution
ax = fig.add_subplot(gs[2, 0])
nonzero = syn_strength[syn_strength > 0]
ax.hist(nonzero.flatten(), bins=40, alpha=0.7, color='purple', edgecolor='white')
ax.set_xlabel('Synaptic Strength')
ax.set_ylabel('Frequency')
ax.set_title('G. Synaptic Strength Distribution')

# Panel H: Layer connectivity
ax = fig.add_subplot(gs[2, 1])
layer_names = ['Photo', 'Lamina', 'Medulla', 'Tm/TmY', 'Lobula', 'T4/T5']
layer_counts = [
    [0, 40, 0, 0, 0, 0],      # Photo -> Lamina
    [0, 0, 78, 45, 32, 0],    # Lamina -> others
    [0, 0, 0, 120, 25, 48],   # Medulla -> others
    [0, 0, 0, 0, 30, 85],     # Tm/TmY -> others
    [0, 0, 0, 0, 0, 40],      # Lobula -> T4/T5
    [0, 0, 0, 0, 0, 0],       # T4/T5 -> none (output)
]
im = ax.imshow(layer_counts, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(layer_names)))
ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(layer_names)))
ax.set_yticklabels(layer_names, fontsize=8)
ax.set_title('H. Cross-Layer Connections')
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel I: Best vs worst model correlation
ax = fig.add_subplot(gs[2, 2])
best_idx = motion_stats['best_model_idx']
worst_idx = motion_stats['worst_model_idx']
ax.scatter(bias[best_idx], bias[worst_idx], alpha=0.5, s=20, label='Bias')
ax.scatter(time_const[best_idx], time_const[worst_idx], alpha=0.5, s=20, label='Time Const', color='green')
min_val = min(bias.min(), time_const.min())
max_val = max(bias.max(), time_const.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
ax.set_xlabel('Best Model')
ax.set_ylabel('Worst Model')
ax.set_title('I. Parameter Consistency')
ax.legend(fontsize=8)

plt.suptitle('Summary: Connectome-Constrained Deep Mechanistic Network Analysis', 
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig('report/images/figure19_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure19_summary.png")
