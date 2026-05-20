"""
Analyze motion detection mechanisms in the DMN ensemble.
Focus on T4/T5 pathways and direction selectivity.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

os.makedirs('report/images', exist_ok=True)

# Load data
with open('/tmp/flyvis/flyvis/connectome/fib25-fib19_v2.2.json') as f:
    connectome = json.load(f)

node_names = [n['name'] for n in connectome['nodes']]
name_to_idx = {name: i for i, name in enumerate(node_names)}

params = np.load('outputs/ensemble_parameters.npz')
bias = params['bias']
time_const = params['time_const']
sign = params['sign']
syn_strength = params['syn_strength']
losses = np.load('outputs/validation_losses.npy')

n_models = syn_strength.shape[0]

# Build edge list
edges = []
for i, edge in enumerate(connectome['edges'][:syn_strength.shape[1]]):
    edges.append({
        'src': edge['src'],
        'tar': edge['tar'],
        'src_idx': name_to_idx[edge['src']],
        'tar_idx': name_to_idx[edge['tar']],
        'sign': int(sign[0, i]),
        'mean_strength': float(syn_strength[:, i].mean()),
        'std_strength': float(syn_strength[:, i].std()),
    })

# ============================================================================
# Figure 13: Direction-selective circuit motif analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Known motion detection pathways from literature:
# Mi1 -> T4 (ON pathway)
# Tm3 -> T4 (ON pathway)
# Mi4 -> T5 (OFF pathway)
# Tm9 -> T5 (OFF pathway)
# Mi9 -> T4/T5 (inhibitory)

motion_pairs = [
    ('Mi1', 'T4a'), ('Mi1', 'T4b'), ('Mi1', 'T4c'), ('Mi1', 'T4d'),
    ('Tm3', 'T4a'), ('Tm3', 'T4b'), ('Tm3', 'T4c'), ('Tm3', 'T4d'),
    ('Mi4', 'T5a'), ('Mi4', 'T5b'), ('Mi4', 'T5c'), ('Mi4', 'T5d'),
    ('Tm9', 'T5a'), ('Tm9', 'T5b'), ('Tm9', 'T5c'), ('Tm9', 'T5d'),
    ('Mi9', 'T4a'), ('Mi9', 'T4b'), ('Mi9', 'T4c'), ('Mi9', 'T4d'),
    ('Mi9', 'T5a'), ('Mi9', 'T5b'), ('Mi9', 'T5c'), ('Mi9', 'T5d'),
]

# Extract strengths for these pairs
pair_strengths = {}
pair_signs = {}
for src, tar in motion_pairs:
    for edge in edges:
        if edge['src'] == src and edge['tar'] == tar:
            pair_strengths[(src, tar)] = edge['mean_strength']
            pair_signs[(src, tar)] = edge['sign']

# Plot 1: Mi1/Tm3 -> T4 strengths
ax = axes[0, 0]
t4_dirs = ['T4a', 'T4b', 'T4c', 'T4d']
mi1_strengths = [pair_strengths.get(('Mi1', t4), 0) for t4 in t4_dirs]
tm3_strengths = [pair_strengths.get(('Tm3', t4), 0) for t4 in t4_dirs]

x = np.arange(len(t4_dirs))
width = 0.35
ax.bar(x - width/2, mi1_strengths, width, label='Mi1 -> T4', alpha=0.8, color='blue')
ax.bar(x + width/2, tm3_strengths, width, label='Tm3 -> T4', alpha=0.8, color='cyan')
ax.set_xticks(x)
ax.set_xticklabels(t4_dirs)
ax.set_ylabel('Mean Synaptic Strength')
ax.set_title('ON Pathway Inputs to T4')
ax.legend()

# Plot 2: Mi4/Tm9 -> T5 strengths
ax = axes[0, 1]
t5_dirs = ['T5a', 'T5b', 'T5c', 'T5d']
mi4_strengths = [pair_strengths.get(('Mi4', t5), 0) for t5 in t5_dirs]
tm9_strengths = [pair_strengths.get(('Tm9', t5), 0) for t5 in t5_dirs]

x = np.arange(len(t5_dirs))
ax.bar(x - width/2, mi4_strengths, width, label='Mi4 -> T5', alpha=0.8, color='red')
ax.bar(x + width/2, tm9_strengths, width, label='Tm9 -> T5', alpha=0.8, color='orange')
ax.set_xticks(x)
ax.set_xticklabels(t5_dirs)
ax.set_ylabel('Mean Synaptic Strength')
ax.set_title('OFF Pathway Inputs to T5')
ax.legend()

# Plot 3: Inhibitory Mi9 inputs
ax = axes[1, 0]
mi9_t4 = [pair_strengths.get(('Mi9', t4), 0) for t4 in t4_dirs]
mi9_t5 = [pair_strengths.get(('Mi9', t5), 0) for t5 in t5_dirs]

x_t4 = np.arange(len(t4_dirs))
x_t5 = np.arange(len(t5_dirs)) + 0.3
ax.bar(x_t4, mi9_t4, 0.3, label='Mi9 -> T4', alpha=0.8, color='purple')
ax.bar(x_t5, mi9_t5, 0.3, label='Mi9 -> T5', alpha=0.8, color='magenta')
ax.set_xticks(x_t4 + 0.15)
ax.set_xticklabels(t4_dirs)
ax.set_ylabel('Mean Synaptic Strength')
ax.set_title('Mi9 Inhibitory Inputs to T4/T5')
ax.legend()

# Plot 4: Ensemble variability in key motion connections
ax = axes[1, 1]
key_connections = [
    ('Mi1', 'T4a'), ('Tm3', 'T4a'),
    ('Mi4', 'T5a'), ('Tm9', 'T5a'),
    ('Mi9', 'T4a'), ('Mi9', 'T5a')
]

# Get per-model strengths
conn_values = {}
for src, tar in key_connections:
    for i, edge in enumerate(connectome['edges'][:syn_strength.shape[1]]):
        if edge['src'] == src and edge['tar'] == tar:
            conn_values[(src, tar)] = syn_strength[:, i]

positions = []
data_to_plot = []
labels = []
for idx, (src, tar) in enumerate(key_connections):
    if (src, tar) in conn_values:
        positions.append(idx)
        data_to_plot.append(conn_values[(src, tar)])
        labels.append(f'{src}→{tar}')

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], ['blue', 'cyan', 'red', 'orange', 'purple', 'magenta']):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Synaptic Strength')
ax.set_title('Variability in Key Motion Connections')

plt.tight_layout()
plt.savefig('report/images/figure13_motion_circuits.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure13_motion_circuits.png")

# ============================================================================
# Figure 14: Parameter consistency analysis - which parameters are most constrained?
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Coefficient of variation for each parameter
bias_cv = bias.std(axis=0) / (bias.mean(axis=0) + 1e-8)
tc_cv = time_const.std(axis=0) / (time_const.mean(axis=0) + 1e-8)
ss_cv = syn_strength.std(axis=0) / (syn_strength.mean(axis=0) + 1e-8)

ax = axes[0]
sorted_idx = np.argsort(bias_cv)
ax.barh(range(len(bias_cv)), bias_cv[sorted_idx], color='steelblue', alpha=0.7)
ax.set_yticks(range(0, len(bias_cv), 5))
ax.set_yticklabels([node_names[i] for i in sorted_idx[::5]], fontsize=7)
ax.set_xlabel('Coefficient of Variation')
ax.set_title('Bias Variability (CV)')

ax = axes[1]
sorted_idx = np.argsort(tc_cv)
ax.barh(range(len(tc_cv)), tc_cv[sorted_idx], color='green', alpha=0.7)
ax.set_yticks(range(0, len(tc_cv), 5))
ax.set_yticklabels([node_names[i] for i in sorted_idx[::5]], fontsize=7)
ax.set_xlabel('Coefficient of Variation')
ax.set_title('Time Constant Variability (CV)')

ax = axes[2]
# For edges, show top 20 most and least variable
sorted_idx = np.argsort(ss_cv)
edge_labels = [f"{edges[i]['src']}→{edges[i]['tar']}" for i in sorted_idx]
ax.barh(range(len(ss_cv)), ss_cv[sorted_idx], color='purple', alpha=0.7)
ax.set_yticks(range(0, len(ss_cv), 50))
ax.set_yticklabels([edge_labels[i] for i in range(0, len(ss_cv), 50)], fontsize=5)
ax.set_xlabel('Coefficient of Variation')
ax.set_title('Synaptic Strength Variability (CV)')

plt.tight_layout()
plt.savefig('report/images/figure14_parameter_consistency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure14_parameter_consistency.png")

# ============================================================================
# Figure 15: Best vs worst model comparison
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

best_idx = int(np.argmin(losses))
worst_idx = int(np.argmax(losses))

ax = axes[0]
ax.scatter(bias[best_idx], bias[worst_idx], alpha=0.6, s=30)
ax.plot([bias.min(), bias.max()], [bias.min(), bias.max()], 'k--', alpha=0.3)
ax.set_xlabel(f'Best Model Bias (loss={losses[best_idx]:.4f})')
ax.set_ylabel(f'Worst Model Bias (loss={losses[worst_idx]:.4f})')
ax.set_title('Resting Potential: Best vs Worst')

ax = axes[1]
ax.scatter(time_const[best_idx], time_const[worst_idx], alpha=0.6, s=30, color='green')
ax.plot([time_const.min(), time_const.max()], [time_const.min(), time_const.max()], 'k--', alpha=0.3)
ax.set_xlabel('Best Model Time Constant')
ax.set_ylabel('Worst Model Time Constant')
ax.set_title('Time Constant: Best vs Worst')

ax = axes[2]
ax.scatter(syn_strength[best_idx], syn_strength[worst_idx], alpha=0.6, s=20, color='purple')
max_ss = syn_strength.max()
ax.plot([0, max_ss], [0, max_ss], 'k--', alpha=0.3)
ax.set_xlabel('Best Model Synaptic Strength')
ax.set_ylabel('Worst Model Synaptic Strength')
ax.set_title('Synaptic Strength: Best vs Worst')

plt.tight_layout()
plt.savefig('report/images/figure15_best_vs_worst.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure15_best_vs_worst.png")

# ============================================================================
# Save motion detection stats
# ============================================================================
motion_stats = {
    't4_mi1_strengths': {t4: pair_strengths.get(('Mi1', t4), 0) for t4 in t4_dirs},
    't4_tm3_strengths': {t4: pair_strengths.get(('Tm3', t4), 0) for t4 in t4_dirs},
    't5_mi4_strengths': {t5: pair_strengths.get(('Mi4', t5), 0) for t5 in t5_dirs},
    't5_tm9_strengths': {t5: pair_strengths.get(('Tm9', t5), 0) for t5 in t5_dirs},
    'mi9_inhibition': {
        't4': {t4: pair_strengths.get(('Mi9', t4), 0) for t4 in t4_dirs},
        't5': {t5: pair_strengths.get(('Mi9', t5), 0) for t5 in t5_dirs}
    },
    'best_model_idx': best_idx,
    'worst_model_idx': worst_idx,
    'best_loss': float(losses[best_idx]),
    'worst_loss': float(losses[worst_idx]),
}

with open('outputs/motion_stats.json', 'w') as f:
    json.dump(motion_stats, f, indent=2)

print("\nMotion detection analysis complete. Stats saved to outputs/motion_stats.json")
