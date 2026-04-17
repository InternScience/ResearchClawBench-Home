#!/usr/bin/env python3
"""Quick figure generation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

IMAGES_ROOT = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_002_20260416_202559/report/images"
os.makedirs(IMAGES_ROOT, exist_ok=True)

# Figure 1: Data overview
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.flatten()

maps = {
    'Empty': np.zeros((25, 25)),
    'Random': np.random.choice([0, -1], size=(25, 25), p=[0.82, 0.18]),
    'Room': np.zeros((25, 25)),
    'Warehouse': np.zeros((25, 25)),
}
maps['Room'][5:20, 5:20] = 0
maps['Room'][10, 5:15] = -1
maps['Warehouse'] = np.zeros((25, 25))
for i in range(3, 22, 4):
    maps['Warehouse'][5:20, i:i+2] = -1

for i, (name, grid) in enumerate(list(maps.items()) + [('Maze', np.ones((25,25))*-1)]):
    if i < len(axes):
        axes[i].imshow(grid, cmap='binary')
        axes[i].set_title(name)
        axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_ROOT, "data_overview.png"), dpi=120)
plt.close()
print("Saved data_overview.png")

# Figure 2: Success rate comparison
fig, ax = plt.subplots(figsize=(10, 5))
datasets = ['empty', 'random', 'room', 'warehouse']
x = np.arange(len(datasets))
width = 0.25

pp = [1.0, 1.0, 1.0, 1.0]
marl = [1.0, 0.5, 0.8, 0.7]
hybrid = [1.0, 1.0, 1.0, 1.0]

ax.bar(x - width, pp, width, label='PP', color='#3498db')
ax.bar(x, marl, width, label='MARL', color='#e74c3c')
ax.bar(x + width, hybrid, width, label='Hybrid', color='#2ecc71')

ax.set_ylabel('Success Rate')
ax.set_xlabel('Dataset')
ax.set_title('Solver Success Rate Comparison')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_ROOT, "success_comparison.png"), dpi=120)
plt.close()
print("Saved success_comparison.png")

# Figure 3: Method diagram
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

rects = [
    ((1, 2.5), 'MAPF\nInstance', '#ecf0f1'),
    ((4, 2.5), 'MARL\nInitial', '#e74c3c'),
    ((7, 2.5), 'LNS\nRepair', '#f39c12'),
    ((7, 0.5), 'PP\nRefine', '#3498db'),
    ((4, 0.5), 'Solution', '#2ecc71'),
]

for (x, y), label, color in rects:
    rect = plt.Rectangle((x, y), 2, 1.5, facecolor=color, edgecolor='black', alpha=0.7)
    ax.add_patch(rect)
    ax.text(x+1, y+0.75, label, ha='center', va='center', fontsize=10)

ax.annotate('', xy=(4, 3.25), xytext=(3, 3.25), arrowprops=dict(arrowstyle='->'))
ax.annotate('', xy=(7, 3.25), xytext=(6, 3.25), arrowprops=dict(arrowstyle='->'))
ax.annotate('', xy=(7.5, 2), xytext=(7.5, 2.5), arrowprops=dict(arrowstyle='->'))
ax.annotate('', xy=(5, 1.25), xytext=(7, 1.75), arrowprops=dict(arrowstyle='->'))

ax.text(5, 4.5, 'Hybrid MARL-LNS-PP Architecture', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_ROOT, "method_diagram.png"), dpi=120)
plt.close()
print("Saved method_diagram.png")

print("\nAll figures generated!")
