#!/usr/bin/env python3
"""Generate plots for XEB fidelity analysis."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_ROOT = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_002_20260415_145205/outputs"
REPORT_IMG = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_002_20260415_145205/report/images"

with open(f"{OUTPUT_ROOT}/fidelity_results.json") as f:
    results = json.load(f)

# Group results
from collections import defaultdict
grouped = defaultdict(list)
for r in results:
    grouped[(r['N'], r['d'])].append(r)

# Compute summary stats per (N,d)
summary = {}
for (N, d), items in grouped.items():
    fids = [x['f_xeb'] for x in items]
    summary[(N, d)] = {
        'N': N, 'd': d, 'n': len(fids),
        'mean': np.mean(fids), 'std': np.std(fids),
        'se_mean': np.std(fids) / np.sqrt(len(fids)),
        'fids': fids
    }

# --- Figure 1: Depth scan at N=40 ---
fig, ax = plt.subplots(figsize=(8, 5))
n40_depths = sorted([d for (N, d) in summary if N == 40])
means = [summary[(40, d)]['mean'] for d in n40_depths]
ses = [summary[(40, d)]['se_mean'] for d in n40_depths]
stds = [summary[(40, d)]['std'] for d in n40_depths]

ax.errorbar(n40_depths, means, yerr=ses, fmt='o-', color='#2196F3', linewidth=2, markersize=8, capsize=5, label='Mean F_XEB ± SE')
ax.fill_between(n40_depths, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.15, color='#2196F3')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Uniform sampling (F=0)')
ax.set_xlabel('Circuit Depth (d)', fontsize=13)
ax.set_ylabel('XEB Fidelity', fontsize=13)
ax.set_title('XEB Fidelity vs Circuit Depth (N=40 qubits)', fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(n40_depths)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{REPORT_IMG}/depth_scan.png", dpi=150)
plt.close()
print("Saved depth_scan.png")

# --- Figure 2: N scan at d=12 ---
fig, ax = plt.subplots(figsize=(8, 5))
d12_ns = sorted([N for (N, d) in summary if d == 12 and N in [16, 24, 32, 40]])
means_n = [summary[(N, 12)]['mean'] for N in d12_ns]
ses_n = [summary[(N, 12)]['se_mean'] for N in d12_ns]
stds_n = [summary[(N, 12)]['std'] for N in d12_ns]

ax.errorbar(d12_ns, means_n, yerr=ses_n, fmt='s-', color='#E91E63', linewidth=2, markersize=8, capsize=5, label='Mean F_XEB ± SE')
ax.fill_between(d12_ns, np.array(means_n) - np.array(stds_n), np.array(means_n) + np.array(stds_n), alpha=0.15, color='#E91E63')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Uniform sampling (F=0)')
ax.set_xlabel('Number of Qubits (N)', fontsize=13)
ax.set_ylabel('XEB Fidelity', fontsize=13)
ax.set_title('XEB Fidelity vs Qubit Count (depth d=12)', fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(d12_ns)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{REPORT_IMG}/n_scan.png", dpi=150)
plt.close()
print("Saved n_scan.png")

# --- Figure 3: Heatmap ---
all_ns = sorted(set(r['N'] for r in results))
all_ds = sorted(set(r['d'] for r in results))
heatmap = np.full((len(all_ns), len(all_ds)), np.nan)
for i, N in enumerate(all_ns):
    for j, d in enumerate(all_ds):
        if (N, d) in summary:
            heatmap[i, j] = summary[(N, d)]['mean']

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(all_ds)))
ax.set_xticklabels(all_ds)
ax.set_yticks(range(len(all_ns)))
ax.set_yticklabels(all_ns)
ax.set_xlabel('Circuit Depth (d)', fontsize=13)
ax.set_ylabel('Number of Qubits (N)', fontsize=13)
ax.set_title('Mean XEB Fidelity Heatmap', fontsize=14)
for i in range(len(all_ns)):
    for j in range(len(all_ds)):
        if not np.isnan(heatmap[i, j]):
            ax.text(j, i, f'{heatmap[i,j]:.2f}', ha='center', va='center', fontsize=10, color='black')
plt.colorbar(im, ax=ax, label='Mean F_XEB')
plt.tight_layout()
plt.savefig(f"{REPORT_IMG}/fidelity_heatmap.png", dpi=150)
plt.close()
print("Saved fidelity_heatmap.png")

# --- Figure 4: Per-instance scatter for depth scan ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
axes = axes.flatten()
for idx, d in enumerate(n40_depths):
    ax = axes[idx]
    items = grouped[(40, d)]
    fids = [x['f_xeb'] for x in items]
    rs = [x['r'] for x in items]
    ax.scatter(rs, fids, s=20, alpha=0.6, color='#2196F3')
    ax.axhline(y=np.mean(fids), color='red', linestyle='--', linewidth=1.5, label=f'mean={np.mean(fids):.3f}')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'd={d}', fontsize=12)
    ax.set_xlabel('Instance r')
    if idx == 0:
        ax.set_ylabel('F_XEB')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
if len(n40_depths) < 8:
    for idx in range(len(n40_depths), 8):
        axes[idx].set_visible(False)
plt.suptitle('Per-Instance XEB Fidelity at N=40 (Depth Scan)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{REPORT_IMG}/per_instance_depth_scan.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved per_instance_depth_scan.png")

# --- Figure 5: Per-instance scatter for N scan ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for idx, N in enumerate(d12_ns):
    ax = axes[idx]
    items = grouped[(N, 12)]
    fids = [x['f_xeb'] for x in items]
    rs = [x['r'] for x in items]
    ax.scatter(rs, fids, s=20, alpha=0.6, color='#E91E63')
    ax.axhline(y=np.mean(fids), color='red', linestyle='--', linewidth=1.5, label=f'mean={np.mean(fids):.3f}')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'N={N}', fontsize=12)
    ax.set_xlabel('Instance r')
    if idx == 0:
        ax.set_ylabel('F_XEB')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
plt.suptitle('Per-Instance XEB Fidelity at d=12 (N Scan)', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig(f"{REPORT_IMG}/per_instance_n_scan.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved per_instance_n_scan.png")

print("\nAll plots generated successfully.")
