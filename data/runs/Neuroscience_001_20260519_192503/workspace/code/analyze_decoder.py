"""
Analyze decoder parameters and structure-function relationships.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import os

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

os.makedirs('report/images', exist_ok=True)

# Load all decoder parameters
decoder_params = {
    'base_weight': [],
    'base_bias': [],
    'decoder_weight': [],
    'decoder_bias': [],
}

model_dirs = sorted([d for d in os.listdir('data/flow/0000') if d.isdigit()])
for model_dir in model_dirs:
    state = torch.load(f'data/flow/0000/{model_dir}/best_chkpt', map_location='cpu')
    flow = state['decoder']['flow']
    decoder_params['base_weight'].append(flow['base.0.weight'].numpy())
    decoder_params['base_bias'].append(flow['base.0.bias'].numpy())
    decoder_params['decoder_weight'].append(flow['decoder.0.weight'].numpy())
    decoder_params['decoder_bias'].append(flow['decoder.0.bias'].numpy())

for k in decoder_params:
    decoder_params[k] = np.stack(decoder_params[k])

losses = np.load('outputs/validation_losses.npy')

print(f"Base weight shape: {decoder_params['base_weight'].shape}")
print(f"Decoder weight shape: {decoder_params['decoder_weight'].shape}")

# ============================================================================
# Figure 16: Decoder parameter analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Base layer weight distribution
ax = axes[0, 0]
bw = decoder_params['base_weight']
ax.hist(bw.flatten(), bins=50, alpha=0.6, color='steelblue', edgecolor='white')
ax.set_xlabel('Weight Value')
ax.set_ylabel('Frequency')
ax.set_title(f'Base Layer Weight Distribution\nmean={bw.mean():.4f}, std={bw.std():.4f}')

# Base layer bias distribution
ax = axes[0, 1]
bb = decoder_params['base_bias']
ax.hist(bb.flatten(), bins=30, alpha=0.6, color='green', edgecolor='white')
ax.set_xlabel('Bias Value')
ax.set_ylabel('Frequency')
ax.set_title(f'Base Layer Bias Distribution\nmean={bb.mean():.4f}, std={bb.std():.4f}')

# Decoder weight distribution
ax = axes[1, 0]
dw = decoder_params['decoder_weight']
ax.hist(dw.flatten(), bins=50, alpha=0.6, color='purple', edgecolor='white')
ax.set_xlabel('Weight Value')
ax.set_ylabel('Frequency')
ax.set_title(f'Decoder Weight Distribution\nmean={dw.mean():.4f}, std={dw.std():.4f}')

# Decoder bias distribution
ax = axes[1, 1]
db = decoder_params['decoder_bias']
ax.hist(db.flatten(), bins=20, alpha=0.6, color='coral', edgecolor='white')
ax.set_xlabel('Bias Value')
ax.set_ylabel('Frequency')
ax.set_title(f'Decoder Bias Distribution\nmean={db.mean():.4f}, std={db.std():.4f}')

plt.tight_layout()
plt.savefig('report/images/figure16_decoder_params.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure16_decoder_params.png")

# ============================================================================
# Figure 17: Decoder readout pattern (which neurons are read?)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# The decoder base.0.weight has shape (8, 34, 5, 5)
# 34 input features correspond to 34 output units from connectome
with open('/tmp/flyvis/flyvis/connectome/fib25-fib19_v2.2.json') as f:
    connectome = json.load(f)
output_units = connectome['output_units']

# Average absolute weight per output unit
mean_abs_weight = np.abs(decoder_params['base_weight']).mean(axis=0)  # (8, 34, 5, 5)
weight_per_unit = mean_abs_weight.mean(axis=(0, 2, 3))  # (34,)

ax = axes[0]
sorted_idx = np.argsort(weight_per_unit)[::-1]
ax.barh(range(len(output_units)), weight_per_unit[sorted_idx], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(output_units)))
ax.set_yticklabels([output_units[i] for i in sorted_idx], fontsize=7)
ax.set_xlabel('Mean Absolute Weight')
ax.set_title('Decoder Readout Weight per Output Unit')

# Correlation between readout weight and node properties
ax = axes[1]
params = np.load('outputs/ensemble_parameters.npz')
bias = params['bias']
tc = params['time_const']

# Get indices of output units in node list
node_names = [n['name'] for n in connectome['nodes']]
output_indices = [node_names.index(u) for u in output_units]

mean_bias_out = bias[:, output_indices].mean(axis=0)
mean_tc_out = tc[:, output_indices].mean(axis=0)

ax.scatter(mean_bias_out, weight_per_unit, alpha=0.6, s=50, label='Bias vs Readout')
ax.set_xlabel('Mean Resting Potential')
ax.set_ylabel('Mean Absolute Readout Weight')
ax.set_title('Resting Potential vs Decoder Readout')

# Add text labels for top units
for i in sorted_idx[:5]:
    ax.annotate(output_units[i], (mean_bias_out[i], weight_per_unit[i]), fontsize=6, alpha=0.7)

plt.tight_layout()
plt.savefig('report/images/figure17_decoder_readout.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure17_decoder_readout.png")

# ============================================================================
# Figure 18: Training convergence analysis (if available)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# The validation losses are just single values, but we can show ensemble statistics
ax.hist(losses, bins=20, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(losses.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {losses.mean():.4f}')
ax.axvline(np.median(losses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(losses):.4f}')
ax.axvline(losses.min(), color='blue', linestyle=':', linewidth=2, label=f'Best: {losses.min():.4f}')
ax.set_xlabel('Validation Loss (L2 norm)')
ax.set_ylabel('Number of Models')
ax.set_title('Ensemble Performance Distribution\n(50 independently trained models)')
ax.legend()

plt.tight_layout()
plt.savefig('report/images/figure18_ensemble_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure18_ensemble_performance.png")

print("\nDecoder analysis complete.")
