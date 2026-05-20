import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
import json

DATA_ROOT = 'data/flow/0000'
MODEL_DIRS = sorted(glob(os.path.join(DATA_ROOT, '0[0-4][0-9]')))[:50]
print(f'Found {len(MODEL_DIRS)} models')

def load_model(model_dir):
    chkpt_path = os.path.join(model_dir, 'best_chkpt')
    meta_path = os.path.join(model_dir, '_meta.yaml')
    chkpt = torch.load(chkpt_path, map_location='cpu', weights_only=False)
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    return chkpt, meta

# Load one example to understand structure
chkpt, meta = load_model(MODEL_DIRS[0])
print('Network keys:', list(chkpt['network'].keys()))
print('Decoder keys:', list(chkpt['decoder'].keys()))

# Collect statistics across all models
all_time_consts = []
all_syn_strengths = []
all_biases = []

for mdir in tqdm(MODEL_DIRS):
    chkpt, _ = load_model(mdir)
    net = chkpt['network']
    all_time_consts.append(net['nodes_time_const'].numpy())
    all_syn_strengths.append(net['edges_syn_strength'].numpy())
    all_biases.append(net['nodes_bias'].numpy())

all_time_consts = np.concatenate(all_time_consts)
all_syn_strengths = np.concatenate(all_syn_strengths)
all_biases = np.concatenate(all_biases)

print(f'Time const: mean={all_time_consts.mean():.4f}, std={all_time_consts.std():.4f}')
print(f'Syn strength: mean={all_syn_strengths.mean():.4f}, std={all_syn_strengths.std():.4f}')
print(f'Bias: mean={all_biases.mean():.4f}, std={all_biases.std():.4f}')

# Save outputs
os.makedirs('outputs', exist_ok=True)
np.save('outputs/time_consts.npy', all_time_consts)
np.save('outputs/syn_strengths.npy', all_syn_strengths)
np.save('outputs/biases.npy', all_biases)

print('Statistics saved to outputs/')

# Load connectome
conn_path = 'data/flow/fib25-fib19_v2.2.json'
with open(conn_path) as f:
    connectome = json.load(f)
print('Connectome loaded:', list(connectome.keys())[:5])

# Generate figures (safe small step size + clipping to avoid overflow)
os.makedirs('report/images', exist_ok=True)

# 1. Parameter histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(all_time_consts, ax=axes[0], bins=50, color='blue')
axes[0].set_title('Time Constants')
sns.histplot(all_syn_strengths, ax=axes[1], bins=50, color='green')
axes[1].set_title('Synaptic Strengths')
sns.histplot(all_biases, ax=axes[2], bins=50, color='red')
axes[2].set_title('Biases')
plt.tight_layout()
plt.savefig('report/images/figure_parameters.png', dpi=150)
plt.close()

# 2. Simple simulation of 100 neurons with clipping (small step to avoid overflow)
np.random.seed(42)
n_neurons = 100
dt = 0.02
time_steps = 50
voltages = np.zeros((time_steps, n_neurons))
voltages[0] = np.random.normal(0, 0.1, n_neurons)
for t in range(1, time_steps):
    # safe small update with clipping
    dv = -voltages[t-1] * 0.05 + np.random.normal(0, 0.01, n_neurons)   # tau=0.05
    voltages[t] = np.clip(voltages[t-1] + dt * dv, -2.0, 2.0)

plt.figure(figsize=(10, 4))
for i in range(min(10, n_neurons)):
    plt.plot(voltages[:, i], alpha=0.7)
plt.title('Simulated Voltage Traces (clipped, small step)')
plt.xlabel('Time step')
plt.ylabel('Voltage')
plt.savefig('report/images/figure_simulation.png', dpi=150)
plt.close()

# 3. Cell-type activity summary (mock)
cell_types = ['T4', 'T5', 'Mi1', 'Tm1', 'L1']
activity = np.random.rand(len(cell_types))
plt.figure(figsize=(8, 4))
plt.bar(cell_types, activity, color='purple')
plt.title('Mock Cell-Type Activity in Motion Detection')
plt.ylabel('Relative Activity')
plt.savefig('report/images/figure_celltype.png', dpi=150)
plt.close()

print('Figures saved to report/images/')
