"""
Generate visualization figures from experiment results
"""
import os
import sys
sys.path.insert(0, "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_002_20260416_105258/code")
from mapf_algorithms import GridEnv, marl_lns
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_002_20260416_105258"
OUT = os.path.join(BASE, "outputs")
IMG = os.path.join(BASE, "report", "images")

os.makedirs(IMG, exist_ok=True)

# Load results
with open(os.path.join(OUT, "experiment_results.json")) as f:
    results = json.load(f)

sns.set_theme(style="whitegrid", font_scale=1.2)

# ============================================================
# 1. Success Rate Comparison across Map Types
# ============================================================
datasets = ["random_small", "random_medium", "maze", "room", "warehouse", "empty"]
algos = ["PP", "LNS", "RRPP", "MARL-LNS"]

# Aggregate success rate per dataset and algorithm
success_data = {}
for ds in datasets:
    success_data[ds] = {}
    for algo in algos:
        entries = [r for r in results if r["dataset"]==ds and r["algo"]==algo]
        if entries:
            succ_rate = sum(1 for e in entries if e["success"]) / len(entries)
            success_data[ds][algo] = succ_rate
        else:
            success_data[ds][algo] = 0

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(datasets))
width = 0.2
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, algo in enumerate(algos):
    rates = [success_data[ds][algo] for ds in datasets]
    bars = ax.bar(x + i*width, rates, width, label=algo, color=colors[i])
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{rate:.0%}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Map Type')
ax.set_ylabel('Success Rate')
ax.set_title('Success Rate Comparison Across Map Types')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(datasets)
ax.legend()
ax.set_ylim(0, 1.15)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "success_rate_comparison.png"), dpi=150)
plt.close()

# ============================================================
# 2. Success Rate vs Agent Count (per map type)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()

for ax_idx, ds in enumerate(datasets):
    ax = axes_flat[ax_idx]
    # Get unique agent counts for this dataset
    agent_counts = sorted(set(r["num_agents"] for r in results if r["dataset"]==ds))
    
    for i, algo in enumerate(algos):
        rates = []
        for na in agent_counts:
            entries = [r for r in results if r["dataset"]==ds and r["algo"]==algo and r["num_agents"]==na]
            if entries:
                rate = sum(1 for e in entries if e["success"]) / len(entries)
            else:
                rate = 0
            rates.append(rate)
        ax.plot(agent_counts, rates, marker='o', label=algo, color=colors[i], linewidth=2)
    
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Success Rate')
    ax.set_title(f'{ds}')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

plt.suptitle('Success Rate vs Agent Count Across Map Types', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "success_rate_vs_agents.png"), dpi=150)
plt.close()

# ============================================================
# 3. Colliding Pairs Comparison (bar chart)
# ============================================================
cp_data = {}
for ds in datasets:
    cp_data[ds] = {}
    for algo in algos:
        entries = [r for r in results if r["dataset"]==ds and r["algo"]==algo and r["cp"]<999]
        if entries:
            avg_cp = np.mean([e["cp"] for e in entries])
            cp_data[ds][algo] = avg_cp
        else:
            cp_data[ds][algo] = 0

fig, ax = plt.subplots(figsize=(10, 6))
for i, algo in enumerate(algos):
    vals = [cp_data[ds][algo] for ds in datasets]
    ax.bar(x + i*width, vals, width, label=algo, color=colors[i])

ax.set_xlabel('Map Type')
ax.set_ylabel('Average Colliding Pairs')
ax.set_title('Average Colliding Pairs Across Map Types (excluding failures)')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(datasets)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG, "colliding_pairs_comparison.png"), dpi=150)
plt.close()

# ============================================================
# 4. Computation Time Comparison
# ============================================================
time_data = {}
for ds in datasets:
    time_data[ds] = {}
    for algo in algos:
        entries = [r for r in results if r["dataset"]==ds and r["algo"]==algo]
        if entries:
            avg_time = np.mean([e["time"] for e in entries])
            time_data[ds][algo] = avg_time
        else:
            time_data[ds][algo] = 0

fig, ax = plt.subplots(figsize=(10, 6))
for i, algo in enumerate(algos):
    vals = [time_data[ds][algo] for ds in datasets]
    ax.bar(x + i*width, vals, width, label=algo, color=colors[i])

ax.set_xlabel('Map Type')
ax.set_ylabel('Average Computation Time (s)')
ax.set_title('Average Computation Time Across Map Types')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(datasets)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG, "computation_time_comparison.png"), dpi=150)
plt.close()

# ============================================================
# 5. Collision Progression during LNS iterations
# ============================================================
# Find a good example with cp_history
best_example = None
best_hist_len = 0
for r in results:
    if r["algo"]=="MARL-LNS" and r["cp_hist"] and len(r["cp_hist"])>best_hist_len:
        best_hist_len = len(r["cp_hist"])
        best_example = r

if best_example and best_example["cp_hist"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    hist = best_example["cp_hist"]
    ax.plot(range(len(hist)), hist, 'b-', linewidth=2, label=f'MARL-LNS ({best_example["dataset"]}, {best_example["num_agents"]} agents)')
    ax.set_xlabel('LNS Iteration')
    ax.set_ylabel('Colliding Pairs')
    ax.set_title('Collision Reduction Progression During LNS Repair')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, "collision_progression.png"), dpi=150)
    plt.close()

# Also plot multiple examples
fig, ax = plt.subplots(figsize=(12, 7))
plotted = 0
for r in results:
    if r["algo"]=="MARL-LNS" and r["cp_hist"] and len(r["cp_hist"])>3 and plotted<6:
        hist = r["cp_hist"]
        label = f'{r["dataset"]}, {r["num_agents"]}ag (succ={r["success"]})'
        ax.plot(range(len(hist)), hist, linewidth=1.5, label=label)
        plotted += 1

ax.set_xlabel('LNS Iteration')
ax.set_ylabel('Colliding Pairs')
ax.set_title('Collision Reduction Progression - Multiple Examples')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG, "collision_progression_multi.png"), dpi=150)
plt.close()

# ============================================================
# 6. Heatmap of Success Rates
# ============================================================
heatmap_data = np.zeros((len(datasets), len(algos)))
for i, ds in enumerate(datasets):
    for j, algo in enumerate(algos):
        heatmap_data[i,j] = success_data[ds][algo]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.2f', xticklabels=algos, yticklabels=datasets,
            cmap='YlGn', vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Success Rate'})
ax.set_title('Success Rate Heatmap: Algorithm × Map Type')
plt.tight_layout()
plt.savefig(os.path.join(IMG, "algorithm_comparison_heatmap.png"), dpi=150)
plt.close()

# ============================================================
# 7. Map Visualization with Agent Paths
# ============================================================
# Load a map and show paths
map_ds = "warehouse"
map_subdir = "warehouse_maps_266_25_25"
map_path = os.path.join(BASE, "data", map_ds, map_subdir, "eval_map_warehouse_1.npy")
grid = np.load(map_path, allow_pickle=True)

env = GridEnv(grid)
agents_cfg = env.gen_agents(10, seed=0)

if agents_cfg:
    starts, goals = agents_cfg
    # Run MARL-LNS to get paths
    paths, _, _ = marl_lns(env, starts, goals, marl_tl=5, lns_tl=15, seed=42)
    
    if paths:
        fig, ax = plt.subplots(figsize=(8, 8))
        # Draw grid
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r,c] == -1:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#404040'))
                else:
                    ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='#f0f0f0'))
        
        # Draw paths
        path_colors = plt.cm.Set2(np.linspace(0, 1, len(paths)))
        max_steps = min(max(len(p) for p in paths), 30)
        for i, p in enumerate(paths):
            truncated = p[:max_steps]
            xs = [pos[1] for pos in truncated]
            ys = [pos[0] for pos in truncated]
            ax.plot(xs, ys, '-', color=path_colors[i], linewidth=1.5, alpha=0.7)
            # Start
            ax.plot(starts[i][1], starts[i][0], 'o', color=path_colors[i], markersize=8)
            # Goal
            ax.plot(goals[i][1], goals[i][0], '*', color=path_colors[i], markersize=12)
        
        ax.set_xlim(-0.5, grid.shape[1]-0.5)
        ax.set_ylim(-0.5, grid.shape[0]-0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Warehouse Map: 10 Agents (MARL-LNS Solution)')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(IMG, "map_visualization.png"), dpi=150)
        plt.close()

print("All figures generated successfully!")
print("Figures saved to:", IMG)