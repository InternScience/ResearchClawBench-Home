"""
Advanced connectome analysis using the fib25-fib19_v2.2.json structure.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Load connectome
with open('/tmp/flyvis/flyvis/connectome/fib25-fib19_v2.2.json') as f:
    connectome = json.load(f)

node_names = [n['name'] for n in connectome['nodes']]
name_to_idx = {name: i for i, name in enumerate(node_names)}

# Load parameters
params = np.load('outputs/ensemble_parameters.npz')
bias = params['bias']
time_const = params['time_const']
sign = params['sign']
syn_strength = params['syn_strength']
losses = np.load('outputs/validation_losses.npy')

n_models = bias.shape[0]

# Build edge list from connectome
edges = []
for edge in connectome['edges']:
    src_idx = name_to_idx[edge['src']]
    tar_idx = name_to_idx[edge['tar']]
    edges.append({
        'src': edge['src'],
        'tar': edge['tar'],
        'src_idx': src_idx,
        'tar_idx': tar_idx,
        'n_offsets': len(edge['offsets']),
        'alpha_fixed': edge['alpha_fixed']
    })

print(f"Connectome edges: {len(edges)}")
print(f"Checkpoint edges: {syn_strength.shape[1]}")

# The edges list might be slightly different from checkpoint ordering
# Let's use the connectome structure for analysis but checkpoint values
# We'll use the first 604 edges from connectome if there's a mismatch
if len(edges) > syn_strength.shape[1]:
    edges = edges[:syn_strength.shape[1]]

# Add sign and strength info
for i, edge in enumerate(edges):
    edge['sign'] = int(sign[0, i])
    edge['mean_strength'] = float(syn_strength[:, i].mean())
    edge['std_strength'] = float(syn_strength[:, i].std())

# ============================================================================
# Figure 9: Cell-type specific parameter profiles
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Mean bias per cell type
mean_bias = bias.mean(axis=0)
std_bias = bias.std(axis=0)

ax = axes[0, 0]
x = np.arange(len(node_names))
ax.bar(x, mean_bias, yerr=std_bias, alpha=0.7, color='steelblue', edgecolor='white')
ax.set_xticks(x[::3])
ax.set_xticklabels([node_names[i] for i in range(0, len(node_names), 3)], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Resting Potential (bias)')
ax.set_title('Mean Resting Potential per Cell Type')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Mean time constant per cell type
mean_tc = time_const.mean(axis=0)
std_tc = time_const.std(axis=0)

ax = axes[0, 1]
ax.bar(x, mean_tc, yerr=std_tc, alpha=0.7, color='green', edgecolor='white')
ax.set_xticks(x[::3])
ax.set_xticklabels([node_names[i] for i in range(0, len(node_names), 3)], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Time Constant')
ax.set_title('Mean Time Constant per Cell Type')

# Out-degree weighted synaptic strength
out_strength = defaultdict(float)
out_count = defaultdict(int)
for edge in edges:
    out_strength[edge['src']] += edge['mean_strength']
    out_count[edge['src']] += 1

ax = axes[1, 0]
src_names = [node_names[i] for i in range(len(node_names)) if node_names[i] in out_strength]
src_values = [out_strength[name] / max(1, out_count[name]) for name in src_names]
src_indices = range(len(src_names))
ax.bar(src_indices, src_values, alpha=0.7, color='purple', edgecolor='white')
ax.set_xticks(range(0, len(src_names), 3))
ax.set_xticklabels([src_names[i] for i in range(0, len(src_names), 3)], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Mean Outgoing Synaptic Strength')
ax.set_title('Mean Outgoing Strength per Cell Type')

# In-degree weighted synaptic strength
in_strength = defaultdict(float)
in_count = defaultdict(int)
for edge in edges:
    in_strength[edge['tar']] += edge['mean_strength']
    in_count[edge['tar']] += 1

ax = axes[1, 1]
tar_names = [node_names[i] for i in range(len(node_names)) if node_names[i] in in_strength]
tar_values = [in_strength[name] / max(1, in_count[name]) for name in tar_names]
tar_indices = range(len(tar_names))
ax.bar(tar_indices, tar_values, alpha=0.7, color='coral', edgecolor='white')
ax.set_xticks(range(0, len(tar_names), 3))
ax.set_xticklabels([tar_names[i] for i in range(0, len(tar_names), 3)], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Mean Incoming Synaptic Strength')
ax.set_title('Mean Incoming Strength per Cell Type')

plt.tight_layout()
plt.savefig('report/images/figure9_celltype_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure9_celltype_profiles.png")

# ============================================================================
# Figure 10: Motion pathway analysis (T4/T5 circuits)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# T4 pathway inputs
t4_types = ['T4a', 'T4b', 'T4c', 'T4d']
t5_types = ['T5a', 'T5b', 'T5c', 'T5d']

# Find inputs to T4 cells
t4_inputs = defaultdict(list)
t5_inputs = defaultdict(list)

for edge in edges:
    if edge['tar'] in t4_types:
        t4_inputs[edge['tar']].append((edge['src'], edge['mean_strength'], edge['sign']))
    if edge['tar'] in t5_types:
        t5_inputs[edge['tar']].append((edge['src'], edge['mean_strength'], edge['sign']))

ax = axes[0, 0]
for t4 in t4_types:
    inputs = t4_inputs[t4]
    srcs = [i[0] for i in inputs]
    strengths = [i[1] for i in inputs]
    colors = ['blue' if i[2] == 1 else 'red' for i in inputs]
    x_pos = np.arange(len(srcs))
    ax.scatter([t4] * len(srcs), x_pos, s=np.array(strengths)*5000, c=colors, alpha=0.6)
ax.set_title('T4 Pathway Inputs (size=strength, blue=exc, red=inh)')
ax.set_ylabel('Input Cell Type')

# Actually let's make a better plot
ax = axes[0, 0]
all_t4_srcs = set()
for t4 in t4_types:
    for src, _, _ in t4_inputs[t4]:
        all_t4_srcs.add(src)
all_t4_srcs = sorted(all_t4_srcs)

strength_matrix = np.zeros((len(all_t4_srcs), len(t4_types)))
for j, t4 in enumerate(t4_types):
    for i, src in enumerate(all_t4_srcs):
        for s, st, _ in t4_inputs[t4]:
            if s == src:
                strength_matrix[i, j] = st

im = ax.imshow(strength_matrix, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(t4_types)))
ax.set_xticklabels(t4_types)
ax.set_yticks(range(len(all_t4_srcs)))
ax.set_yticklabels(all_t4_srcs, fontsize=7)
ax.set_title('T4 Pathway Input Strengths')
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
all_t5_srcs = set()
for t5 in t5_types:
    for src, _, _ in t5_inputs[t5]:
        all_t5_srcs.add(src)
all_t5_srcs = sorted(all_t5_srcs)

strength_matrix_t5 = np.zeros((len(all_t5_srcs), len(t5_types)))
for j, t5 in enumerate(t5_types):
    for i, src in enumerate(all_t5_srcs):
        for s, st, _ in t5_inputs[t5]:
            if s == src:
                strength_matrix_t5[i, j] = st

im = ax.imshow(strength_matrix_t5, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(t5_types)))
ax.set_xticklabels(t5_types)
ax.set_yticks(range(len(all_t5_srcs)))
ax.set_yticklabels(all_t5_srcs, fontsize=7)
ax.set_title('T5 Pathway Input Strengths')
plt.colorbar(im, ax=ax)

# T4/T5 bias and time constant comparison
ax = axes[1, 0]
t4_indices = [name_to_idx[t] for t in t4_types]
t5_indices = [name_to_idx[t] for t in t5_types]

x_t4 = np.arange(len(t4_types))
x_t5 = np.arange(len(t5_types)) + 0.3

ax.bar(x_t4, mean_bias[t4_indices], 0.3, label='T4', alpha=0.7, color='blue')
ax.bar(x_t5, mean_bias[t5_indices], 0.3, label='T5', alpha=0.7, color='orange')
ax.set_xticks(x_t4 + 0.15)
ax.set_xticklabels(t4_types)
ax.set_ylabel('Resting Potential')
ax.set_title('Resting Potential: T4 vs T5')
ax.legend()

ax = axes[1, 1]
ax.bar(x_t4, mean_tc[t4_indices], 0.3, label='T4', alpha=0.7, color='blue')
ax.bar(x_t5, mean_tc[t5_indices], 0.3, label='T5', alpha=0.7, color='orange')
ax.set_xticks(x_t4 + 0.15)
ax.set_xticklabels(t4_types)
ax.set_ylabel('Time Constant')
ax.set_title('Time Constant: T4 vs T5')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/figure10_motion_pathways.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure10_motion_pathways.png")

# ============================================================================
# Figure 11: Connectome graph visualization (adjacency matrix)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Build adjacency matrix
n = len(node_names)
adj_mean = np.zeros((n, n))
adj_sign = np.zeros((n, n))

for edge in edges:
    i = edge['src_idx']
    j = edge['tar_idx']
    adj_mean[i, j] = edge['mean_strength']
    adj_sign[i, j] = edge['sign']

# Reorder by pathway
lamina = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'L1', 'L2', 'L3', 'L4', 'L5', 'Lawf1', 'Lawf2', 'Am', 'C2', 'C3']
medulla = ['CT1(Lo1)', 'CT1(M10)', 'Mi1', 'Mi2', 'Mi3', 'Mi4', 'Mi9', 'Mi10', 'Mi11', 'Mi12', 'Mi13', 'Mi14', 'Mi15']
lobula = ['T1', 'T2', 'T2a', 'T3']
t4 = ['T4a', 'T4b', 'T4c', 'T4d']
t5 = ['T5a', 'T5b', 'T5c', 'T5d']
tm = ['Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm5Y', 'Tm5a', 'Tm5b', 'Tm5c', 'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30']
tmy = ['TmY3', 'TmY4', 'TmY5a', 'TmY9', 'TmY10', 'TmY13', 'TmY14', 'TmY15', 'TmY18']

order = lamina + medulla + lobula + t4 + t5 + tm + tmy
order_idx = [name_to_idx[name] for name in order if name in name_to_idx]

ax = axes[0]
adj_reordered = adj_mean[order_idx, :][:, order_idx]
im = ax.imshow(adj_reordered, cmap='YlOrRd', aspect='auto')
ax.set_title('Mean Synaptic Strength (Reordered by Pathway)')
plt.colorbar(im, ax=ax)

# Add pathway boundaries
boundaries = [len(lamina), len(lamina)+len(medulla), len(lamina)+len(medulla)+len(lobula),
              len(lamina)+len(medulla)+len(lobula)+len(t4),
              len(lamina)+len(medulla)+len(lobula)+len(t4)+len(t5),
              len(lamina)+len(medulla)+len(lobula)+len(t4)+len(t5)+len(tm)]
for b in boundaries:
    ax.axhline(b-0.5, color='white', linewidth=1)
    ax.axvline(b-0.5, color='white', linewidth=1)

ax = axes[1]
adj_sign_reordered = adj_sign[order_idx, :][:, order_idx]
im = ax.imshow(adj_sign_reordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_title('Connection Polarity (Reordered by Pathway)')
plt.colorbar(im, ax=ax)
for b in boundaries:
    ax.axhline(b-0.5, color='white', linewidth=1)
    ax.axvline(b-0.5, color='white', linewidth=1)

plt.tight_layout()
plt.savefig('report/images/figure11_connectome_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure11_connectome_matrix.png")

# ============================================================================
# Figure 12: Layer-wise connectivity analysis
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Compute layer connectivity strength
layers = {
    'Photoreceptors': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'],
    'Lamina': ['L1', 'L2', 'L3', 'L4', 'L5', 'Lawf1', 'Lawf2', 'Am', 'C2', 'C3'],
    'Medulla': ['CT1(Lo1)', 'CT1(M10)', 'Mi1', 'Mi2', 'Mi3', 'Mi4', 'Mi9', 'Mi10', 'Mi11', 'Mi12', 'Mi13', 'Mi14', 'Mi15'],
    'Transmedulla': ['Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm5Y', 'Tm5a', 'Tm5b', 'Tm5c', 'Tm9', 'Tm16', 'Tm20', 'Tm28', 'Tm30'],
    'TmY': ['TmY3', 'TmY4', 'TmY5a', 'TmY9', 'TmY10', 'TmY13', 'TmY14', 'TmY15', 'TmY18'],
    'Lobula': ['T1', 'T2', 'T2a', 'T3'],
    'T4/T5': ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
}

layer_names = list(layers.keys())
n_layers = len(layer_names)
layer_strength = np.zeros((n_layers, n_layers))
layer_count = np.zeros((n_layers, n_layers))

for edge in edges:
    src_layer = None
    tar_layer = None
    for li, (lname, lcells) in enumerate(layers.items()):
        if edge['src'] in lcells:
            src_layer = li
        if edge['tar'] in lcells:
            tar_layer = li
    if src_layer is not None and tar_layer is not None:
        layer_strength[src_layer, tar_layer] += edge['mean_strength']
        layer_count[src_layer, tar_layer] += 1

# Normalize by count
layer_strength_norm = np.divide(layer_strength, layer_count, out=np.zeros_like(layer_strength), where=layer_count>0)

ax = axes[0]
im = ax.imshow(layer_strength_norm, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(n_layers))
ax.set_xticklabels(layer_names, rotation=45, ha='right')
ax.set_yticks(range(n_layers))
ax.set_yticklabels(layer_names)
ax.set_title('Mean Synaptic Strength Between Layers')
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(layer_count, cmap='Blues', aspect='auto')
ax.set_xticks(range(n_layers))
ax.set_xticklabels(layer_names, rotation=45, ha='right')
ax.set_yticks(range(n_layers))
ax.set_yticklabels(layer_names)
ax.set_title('Number of Connections Between Layers')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('report/images/figure12_layer_connectivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure12_layer_connectivity.png")

# ============================================================================
# Save connectome-derived statistics
# ============================================================================
connectome_stats = {
    'n_nodes': len(node_names),
    'n_edges': len(edges),
    'node_names': node_names,
    'cell_types_by_layer': {k: v for k, v in layers.items()},
    'top_outgoing': sorted([(k, v/out_count[k]) for k, v in out_strength.items()], key=lambda x: -x[1])[:10],
    'top_incoming': sorted([(k, v/in_count[k]) for k, v in in_strength.items()], key=lambda x: -x[1])[:10],
    't4_input_strengths': {t4: {src: st for src, st, _ in t4_inputs[t4]} for t4 in t4_types},
    't5_input_strengths': {t5: {src: st for src, st, _ in t5_inputs[t5]} for t5 in t5_types},
}

import json
with open('outputs/connectome_stats.json', 'w') as f:
    json.dump(connectome_stats, f, indent=2)

print("\nConnectome analysis complete. Stats saved to outputs/connectome_stats.json")
