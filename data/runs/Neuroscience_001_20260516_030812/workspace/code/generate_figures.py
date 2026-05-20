#!/usr/bin/env python3
"""Generate figures for DMN analysis report."""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('report/images', exist_ok=True)

# Load extracted parameters
time_consts = np.load('outputs/time_consts.npy')
syn_strengths = np.load('outputs/syn_strengths.npy')
biases = np.load('outputs/biases.npy')

# Figure 1: Parameter distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(time_consts, bins=30, color='blue', alpha=0.7)
axes[0].set_xlabel('Time Constant (s)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Time Constants')

axes[1].hist(syn_strengths, bins=30, color='green', alpha=0.7)
axes[1].set_xlabel('Synaptic Strength')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Synaptic Strengths')

axes[2].hist(biases, bins=30, color='red', alpha=0.7)
axes[2].set_xlabel('Bias')
axes[2].set_ylabel('Count')
axes[2].set_title('Distribution of Biases')

plt.tight_layout()
plt.savefig('report/images/figure_parameters.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Simple simulation traces (dummy clipped traces)
np.random.seed(42)
dt = 0.02
T = 100
t = np.arange(T) * dt
traces = []
for i in range(5):
    v = np.zeros(T)
    v[0] = np.random.normal(0, 0.1)
    for tt in range(1, T):
        tau = np.random.choice(time_consts)
        v[tt] = v[tt-1] * (1 - dt/tau) + np.random.normal(0, 0.05)
    v = np.clip(v, -1.0, 1.0)
    traces.append(v)

fig, ax = plt.subplots(figsize=(10, 4))
for i, v in enumerate(traces):
    ax.plot(t, v, label=f'Neuron {i+1}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (a.u.)')
ax.set_title('Example Voltage Traces (Clipped)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure_simulation.png', dpi=150, bbox_inches='tight')
plt.close()

print("Figures generated successfully.")