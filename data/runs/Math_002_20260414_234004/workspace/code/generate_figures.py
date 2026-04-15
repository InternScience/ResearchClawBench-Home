import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('report/images', exist_ok=True)

with open('outputs/experiment_results.json') as f:
    data = json.load(f)

# Aggregate by map_type and n_agents
from collections import defaultdict
agg = defaultdict(lambda: defaultdict(lambda: {'collisions': [], 'success': [], 'time': [], 'cost': []}))

for r in data:
    key = (r['map_type'], r['n_agents'])
    for algo in ['PP', 'LNS', 'Hybrid_MARL_LNS']:
        agg[key][algo]['collisions'].append(r[algo]['collisions'])
        agg[key][algo]['success'].append(r[algo]['success'])
        agg[key][algo]['time'].append(r[algo]['time'])
        c = r[algo]['cost']
        agg[key][algo]['cost'].append(c if c != -1 else float('inf'))

# Figure 1: Success rate by map type
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
map_types = ['random_small', 'random_medium', 'maze', 'room', 'empty', 'warehouse']
algo_colors = {'PP': '#e74c3c', 'LNS': '#3498db', 'Hybrid_MARL_LNS': '#2ecc71'}
algo_names = {'PP': 'Prioritized Planning', 'LNS': 'LNS', 'Hybrid_MARL_LNS': 'Hybrid MARL-LNS'}

for idx, mt in enumerate(map_types):
    ax = axes[idx // 3][idx % 3]
    keys = sorted([k for k in agg if k[0] == mt], key=lambda x: x[1])
    if not keys:
        continue
    
    x = np.arange(len(keys))
    width = 0.25
    for i, algo in enumerate(['PP', 'LNS', 'Hybrid_MARL_LNS']):
        rates = [np.mean(agg[k][algo]['success']) * 100 for k in keys]
        ax.bar(x + i * width, rates, width, label=algo_names[algo], color=algo_colors[algo], alpha=0.85)
    
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(mt.replace('_', ' ').title())
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(k[1]) for k in keys])
    ax.set_ylim(0, 110)
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle('Success Rate Comparison Across Map Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig1_success_rate.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Average collisions by map type
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, mt in enumerate(map_types):
    ax = axes[idx // 3][idx % 3]
    keys = sorted([k for k in agg if k[0] == mt], key=lambda x: x[1])
    if not keys: continue
    
    x = np.arange(len(keys))
    width = 0.25
    for i, algo in enumerate(['PP', 'LNS', 'Hybrid_MARL_LNS']):
        colls = [np.mean(agg[k][algo]['collisions']) for k in keys]
        ax.bar(x + i * width, colls, width, label=algo_names[algo], color=algo_colors[algo], alpha=0.85)
    
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Avg Collisions')
    ax.set_title(mt.replace('_', ' ').title())
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(k[1]) for k in keys])
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle('Average Collisions Comparison Across Map Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig2_avg_collisions.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Runtime comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, mt in enumerate(map_types):
    ax = axes[idx // 3][idx % 3]
    keys = sorted([k for k in agg if k[0] == mt], key=lambda x: x[1])
    if not keys: continue
    
    x = np.arange(len(keys))
    width = 0.25
    for i, algo in enumerate(['PP', 'LNS', 'Hybrid_MARL_LNS']):
        times = [np.mean(agg[k][algo]['time']) for k in keys]
        ax.bar(x + i * width, times, width, label=algo_names[algo], color=algo_colors[algo], alpha=0.85)
    
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Avg Runtime (s)')
    ax.set_title(mt.replace('_', ' ').title())
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(k[1]) for k in keys])
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle('Runtime Comparison Across Map Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig3_runtime.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Overall summary - grouped bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
overall = defaultdict(lambda: {'collisions': [], 'success': [], 'time': []})
for r in data:
    for algo in ['PP', 'LNS', 'Hybrid_MARL_LNS']:
        overall[algo]['collisions'].append(r[algo]['collisions'])
        overall[algo]['success'].append(r[algo]['success'])
        overall[algo]['time'].append(r[algo]['time'])

# Success rate
algorithms = ['PP', 'LNS', 'Hybrid_MARL_LNS']
colors = [algo_colors[a] for a in algorithms]
labels = [algo_names[a] for a in algorithms]

rates = [np.mean(overall[a]['success']) * 100 for a in algorithms]
axes[0].bar(labels, rates, color=colors, alpha=0.85)
axes[0].set_ylabel('Success Rate (%)')
axes[0].set_title('Overall Success Rate')
axes[0].set_ylim(0, 100)
for i, v in enumerate(rates):
    axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

colls = [np.mean(overall[a]['collisions']) for a in algorithms]
axes[1].bar(labels, colls, color=colors, alpha=0.85)
axes[1].set_ylabel('Avg Collisions')
axes[1].set_title('Overall Average Collisions')
for i, v in enumerate(colls):
    axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

times = [np.mean(overall[a]['time']) for a in algorithms]
axes[2].bar(labels, times, color=colors, alpha=0.85)
axes[2].set_ylabel('Avg Runtime (s)')
axes[2].set_title('Overall Average Runtime')
for i, v in enumerate(times):
    axes[2].text(i, v + 0.05, f'{v:.2f}s', ha='center', fontweight='bold')

plt.suptitle('Overall Algorithm Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig4_overall_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 5: Collision reduction heatmap (Hybrid vs PP)
fig, ax = plt.subplots(figsize=(10, 6))
heatmap_data = []
row_labels = []
for mt in map_types:
    for n in sorted(set(k[1] for k in agg if k[0] == mt)):
        key = (mt, n)
        if key in agg:
            pp_c = np.mean(agg[key]['PP']['collisions'])
            hyb_c = np.mean(agg[key]['Hybrid_MARL_LNS']['collisions'])
            if pp_c > 0:
                reduction = (pp_c - hyb_c) / pp_c * 100
            else:
                reduction = 0 if hyb_c == 0 else -100
            heatmap_data.append(reduction)
            row_labels.append(f"{mt[:4]}-{n}")

if heatmap_data:
    heatmap_arr = np.array(heatmap_data).reshape(-1, 1)
    im = ax.imshow(heatmap_arr, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(['Collision Reduction %'])
    ax.set_title('Collision Reduction: Hybrid MARL-LNS vs PP')
    plt.colorbar(im, ax=ax, label='Reduction %')
    plt.tight_layout()
    plt.savefig('report/images/fig5_collision_reduction.png', dpi=150, bbox_inches='tight')
    plt.close()

# Figure 6: Example map visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
map_files = [
    ('data/random_medium/maps_312_25_25_0.175/eval_map_1.npy', 'Random Medium'),
    ('data/maze/maze_maps_125_25_25/eval_map_maze_1.npy', 'Maze'),
    ('data/room/room_maps_250_25_25/eval_map_room_1.npy', 'Room'),
]
for idx, (mf, title) in enumerate(map_files):
    grid = np.load(mf)
    ax = axes[idx]
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    ax.imshow(grid, cmap=cmap, interpolation='nearest')
    ax.set_title(f'{title} ({grid.shape[0]}x{grid.shape[1]})')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

plt.suptitle('Example Map Structures', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig6_map_examples.png', dpi=150, bbox_inches='tight')
plt.close()

print("All figures generated successfully!")
print("Files:", os.listdir('report/images/'))
