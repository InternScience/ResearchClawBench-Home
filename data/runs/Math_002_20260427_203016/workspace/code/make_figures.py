"""Generate all figures used in the report."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 130,
                     'font.size': 10, 'axes.titlesize': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
IMG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'report', 'images'))
os.makedirs(IMG, exist_ok=True)

with open(os.path.join(OUT, 'results_per_instance.json')) as f:
    recs = json.load(f)
with open(os.path.join(OUT, 'lns_logs.json')) as f:
    logs = json.load(f)

for r in recs:
    for k in ('pp_success', 'lnspp_success', 'hybrid_success'):
        if isinstance(r[k], str):
            r[k] = (r[k] == 'True')

df = pd.DataFrame(recs)
for col in ['pp_time_s','pp_soc','pp_makespan','lnspp_time_s','lnspp_soc',
            'lnspp_makespan','lnspp_iters','hybrid_time_s','hybrid_soc',
            'hybrid_makespan','hybrid_iters','hybrid_train_s','hybrid_qsize']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# -------- Fig 1: Data overview ------------
import sys
sys.path.insert(0, os.path.dirname(__file__))
from mapf_core import load_grid

example_files = [
    ('random_small\n10x10', 'data/random_small/maps_50_10_10_0.175/eval_map_1.npy'),
    ('target_60a\n10x10',   'data/maps_60_10_10_0.175/eval_map_1.npy'),
    ('random_medium\n25x25','data/random_medium/maps_312_25_25_0.175/eval_map_1.npy'),
    ('empty\n25x25',        'data/empty/empty_maps_453_25_25/eval_map_empty_1.npy'),
    ('maze\n25x25',         'data/maze/maze_maps_125_25_25/eval_map_maze_1.npy'),
    ('room\n25x25',         'data/room/room_maps_250_25_25/eval_map_room_1.npy'),
    ('warehouse\n25x25',    'data/warehouse/warehouse_maps_266_25_25/eval_map_warehouse_1.npy'),
    ('random_large\n50x50', 'data/random_large/maps_1250_50_50_0.175/eval_map_1.npy'),
]
fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for ax, (name, fn) in zip(axes.flatten(), example_files):
    g = load_grid(fn)
    ax.imshow(g, cmap='Greys', vmin=-1, vmax=0)
    ax.set_title(name + f'\n({g.shape[0]}x{g.shape[1]}, '
                 f'{(g==0).sum()} free)')
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Map families used for the MAPF benchmark '
             '(black = obstacle, white = free)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_data_overview.png'), bbox_inches='tight')
plt.close()


# -------- Fig 2: Success rate by family x agent count ---------
fams_order = ['random_small_10x10','target_60a_10x10','random_medium_25x25',
              'empty_25x25','maze_25x25','room_25x25','warehouse_25x25',
              'random_large_50x50']
methods = [('PP', 'pp_success', '#888'),
           ('LNS-PP', 'lnspp_success', '#1f77b4'),
           ('LNS-Hybrid', 'hybrid_success', '#d62728')]

fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharey=True)
for ax, fam in zip(axes.flatten(), fams_order):
    sub = df[df['family'] == fam]
    if sub.empty:
        ax.set_visible(False); continue
    grp = sub.groupby('n_agents').mean(numeric_only=True).reset_index()
    x = np.arange(len(grp))
    w = 0.27
    for k, (label, col, color) in enumerate(methods):
        ax.bar(x + (k - 1) * w, grp[col].values, width=w, label=label, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in grp['n_agents']])
    ax.set_title(fam)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('# agents')
axes[0,0].set_ylabel('success rate')
axes[1,0].set_ylabel('success rate')
axes[0,0].legend(loc='upper right', fontsize=9)
fig.suptitle('Success rate by map family × number of agents', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_success_rate_by_map.png'), bbox_inches='tight')
plt.close()


# -------- Fig 3: Runtime vs # agents --------
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharey=False)
for ax, fam in zip(axes.flatten(), fams_order):
    sub = df[df['family'] == fam]
    if sub.empty:
        ax.set_visible(False); continue
    grp = sub.groupby('n_agents')[['pp_time_s','lnspp_time_s','hybrid_time_s']].mean().reset_index()
    ax.plot(grp['n_agents'], grp['pp_time_s'], 'o-', label='PP', color='#888')
    ax.plot(grp['n_agents'], grp['lnspp_time_s'], 's-', label='LNS-PP', color='#1f77b4')
    ax.plot(grp['n_agents'], grp['hybrid_time_s'], '^-', label='LNS-Hybrid', color='#d62728')
    ax.set_xlabel('# agents'); ax.set_ylabel('runtime (s)')
    ax.set_title(fam)
axes[0,0].legend(loc='upper left', fontsize=9)
fig.suptitle('Runtime vs number of agents', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_runtime_vs_agents.png'), bbox_inches='tight')
plt.close()


# -------- Fig 4: LNS convergence ---------
# Pick one representative successful instance per family from logs
selected = []
for fam in fams_order:
    fam_logs = [l for l in logs if l['family'] == fam and l.get('hybrid_success')
                and l.get('lnspp_success')]
    if not fam_logs:
        fam_logs = [l for l in logs if l['family'] == fam and l.get('lnspp_log')]
    if fam_logs:
        # pick one with reasonably long log
        fam_logs.sort(key=lambda x: -len(x.get('lnspp_log') or []))
        selected.append(fam_logs[0])
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
for ax, l in zip(axes.flatten(), selected):
    lns_log = l.get('lnspp_log') or []
    hyb_log = l.get('hybrid_log') or []
    if lns_log:
        ax.plot([e['iter'] for e in lns_log],
                [e['collisions'] for e in lns_log],
                's-', label='LNS-PP', color='#1f77b4', markersize=3)
    if hyb_log:
        # color by phase
        xs = [e['iter'] for e in hyb_log]
        ys = [e['collisions'] for e in hyb_log]
        ax.plot(xs, ys, '^-', label='LNS-Hybrid', color='#d62728', markersize=3)
        # mark MARL phase region
        marl_iters = [e['iter'] for e in hyb_log if e.get('phase') == 'marl']
        if marl_iters:
            ax.axvspan(min(marl_iters)-0.5, max(marl_iters)+0.5,
                       color='#d62728', alpha=0.08, label='MARL phase')
    ax.set_xlabel('LNS iteration')
    ax.set_ylabel('# collisions')
    ax.set_title(f"{l['family']} n={l['n_agents']}")
    ax.set_yscale('symlog', linthresh=1)
axes[0,0].legend(loc='upper right', fontsize=8)
fig.suptitle('LNS convergence: collisions vs iteration\n'
             '(red shaded = MARL repair phase, then switch to PP)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_lns_convergence.png'), bbox_inches='tight')
plt.close()


# -------- Fig 5: Target benchmark (10x10 with 30/45/60 agents) ---------
sub = df[df['family'] == 'target_60a_10x10']
if not sub.empty:
    grp = sub.groupby('n_agents')[['pp_success','lnspp_success','hybrid_success']].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(grp)); w=0.27
    ax.bar(x-w, grp['pp_success'], w, label='PP', color='#888')
    ax.bar(x, grp['lnspp_success'], w, label='LNS-PP', color='#1f77b4')
    ax.bar(x+w, grp['hybrid_success'], w, label='LNS-Hybrid', color='#d62728')
    ax.set_xticks(x); ax.set_xticklabels([f'n={int(n)}' for n in grp['n_agents']])
    ax.set_ylabel('success rate'); ax.set_ylim(0,1.05)
    ax.set_title('Target benchmark: maps_60_10_10_0.175 (10×10 grid)')
    ax.legend()
    # annotate counts
    for i, n in enumerate(grp['n_agents']):
        cnt = (sub['n_agents']==n).sum()
        ax.text(i, -0.06, f'({cnt} inst)', ha='center', fontsize=8, color='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'fig_success_rate_target_benchmark.png'), bbox_inches='tight')
    plt.close()


# -------- Fig 6: Trajectory visualization (one example) ---------
# Re-solve a small instance and draw paths
from mapf_core import load_grid, generate_instance, bfs_distance
from mapf_lns import lns_solve, SharedQTable, marl_train_episodes

g = load_grid('data/random_small/maps_50_10_10_0.175/eval_map_1.npy')
starts, goals = generate_instance(g, n_agents=15, seed=0)
ht = [bfs_distance(g, gg) for gg in goals]
q = SharedQTable()
marl_train_episodes(g, ht, starts, goals, q, n_episodes=30, horizon=50, epsilon=0.3)
paths_lp, _ = lns_solve(g, starts, goals, repair='pp', max_iters=200,
                         nbhd_size=8, time_limit=4, max_time=80, seed=0,
                         h_tables=ht)
paths_h, _ = lns_solve(g, starts, goals, repair='hybrid', max_iters=200,
                        nbhd_size=8, time_limit=4, max_time=80, seed=0,
                        h_tables=ht, q=q, marl_iters_frac=0.4)

def draw_paths(ax, grid, paths, starts, goals, title):
    ax.imshow(grid, cmap='Greys', vmin=-1, vmax=0)
    cmap = plt.get_cmap('tab20', len(starts))
    for i, p in paths.items():
        if p is None: continue
        rs = [c[0] for c in p]
        cs = [c[1] for c in p]
        ax.plot(cs, rs, '-', color=cmap(i), alpha=0.7, linewidth=1.4)
        ax.plot(starts[i][1], starts[i][0], 'o', color=cmap(i), markersize=6,
                markeredgecolor='black', markeredgewidth=0.4)
        ax.plot(goals[i][1], goals[i][0], 's', color=cmap(i), markersize=6,
                markeredgecolor='black', markeredgewidth=0.4)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
draw_paths(axes[0], g, paths_lp, starts, goals, 'LNS-PP solution (15 agents)')
draw_paths(axes[1], g, paths_h, starts, goals, 'LNS-Hybrid solution (15 agents)')
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_example_trajectories.png'), bbox_inches='tight')
plt.close()


# -------- Fig 7: MARL Q-value heatmap (interpretability) ---------
# For a fixed map, train Q and visualise the value V(cell) = max_a Q(obs(cell), a)
# over a grid of starting cells with a fixed goal in lower-right corner.
g = load_grid('data/random_medium/maps_312_25_25_0.175/eval_map_1.npy')
# pick a goal in a free cell
free = [(int(r),int(c)) for r,c in zip(*np.where(g==0))]
goal = free[len(free) // 2]
fake_starts = [free[k] for k in np.linspace(0, len(free) - 1, 10).astype(int)]
fake_goals = [goal] * len(fake_starts)
ht = [bfs_distance(g, goal)] * len(fake_starts)
q = SharedQTable()
marl_train_episodes(g, ht, fake_starts, fake_goals, q, n_episodes=40, horizon=50, epsilon=0.3)

# now query V for every cell
from mapf_lns import _local_obs_key, ACTIONS
H, W = g.shape
V = np.full((H, W), np.nan, dtype=float)
best_a = np.full((H, W), -1, dtype=int)
for r in range(H):
    for c in range(W):
        if g[r, c] != 0:
            continue
        # observation with no other agents (free environment)
        key = _local_obs_key(g, (r, c), goal, {})
        qv = q.get(key)
        V[r, c] = float(np.max(qv))
        best_a[r, c] = int(np.argmax(qv))

bfs = bfs_distance(g, goal).astype(float)
bfs[bfs > 1e6] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
ax0 = axes[0]
im0 = ax0.imshow(np.where(np.isnan(V), 0, V), cmap='viridis')
ax0.imshow(np.where(g == -1, 1, np.nan), cmap='Greys', vmin=0, vmax=1, alpha=0.95)
ax0.scatter(goal[1], goal[0], marker='*', s=160, c='red',
            edgecolors='black', label='goal')
ax0.set_title('Learned MARL value V(cell) = max$_a$ Q(obs, a)\n'
              'after 40 episodes (random 25×25)')
plt.colorbar(im0, ax=ax0, fraction=0.046, label='V')
# arrows for best action
for r in range(0, H, 2):
    for c in range(0, W, 2):
        if g[r, c] != 0 or best_a[r, c] < 0:
            continue
        dr, dc = ACTIONS[best_a[r, c]]
        if dr == 0 and dc == 0:
            ax0.plot(c, r, 'o', color='white', markersize=2)
        else:
            ax0.arrow(c, r, dc * 0.4, dr * 0.4,
                      color='white', head_width=0.25, alpha=0.75)
ax0.legend(loc='lower right'); ax0.set_xticks([]); ax0.set_yticks([])

ax1 = axes[1]
im1 = ax1.imshow(bfs, cmap='magma_r')
ax1.scatter(goal[1], goal[0], marker='*', s=160, c='red',
            edgecolors='black', label='goal')
ax1.set_title('BFS shortest-path distance to goal\n(reference)')
plt.colorbar(im1, ax=ax1, fraction=0.046, label='steps')
ax1.set_xticks([]); ax1.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_marl_value_heatmap.png'), bbox_inches='tight')
plt.close()


# -------- Fig 8: aggregate success-rate / runtime per family ---------
fam_summary = (df.groupby('family')
    .agg(pp_succ=('pp_success', 'mean'),
         lns_succ=('lnspp_success', 'mean'),
         hyb_succ=('hybrid_success', 'mean'),
         pp_time=('pp_time_s', 'mean'),
         lns_time=('lnspp_time_s', 'mean'),
         hyb_time=('hybrid_time_s', 'mean'))
    .reindex(fams_order)
    .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(fam_summary)); w = 0.26
ax = axes[0]
ax.bar(x - w, fam_summary['pp_succ'], w, label='PP', color='#888')
ax.bar(x,     fam_summary['lns_succ'], w, label='LNS-PP', color='#1f77b4')
ax.bar(x + w, fam_summary['hyb_succ'], w, label='LNS-Hybrid', color='#d62728')
ax.set_xticks(x); ax.set_xticklabels(fam_summary['family'], rotation=30, ha='right')
ax.set_ylabel('success rate (averaged over agent counts and instances)')
ax.set_title('Per-family success rate'); ax.set_ylim(0, 1.05); ax.legend()

ax = axes[1]
ax.bar(x - w, fam_summary['pp_time'], w, label='PP', color='#888')
ax.bar(x,     fam_summary['lns_time'], w, label='LNS-PP', color='#1f77b4')
ax.bar(x + w, fam_summary['hyb_time'], w, label='LNS-Hybrid', color='#d62728')
ax.set_xticks(x); ax.set_xticklabels(fam_summary['family'], rotation=30, ha='right')
ax.set_ylabel('mean runtime (s)')
ax.set_title('Per-family mean runtime')
plt.tight_layout()
plt.savefig(os.path.join(IMG, 'fig_per_family_summary.png'), bbox_inches='tight')
plt.close()

print('all figures generated')
print(sorted(os.listdir(IMG)))
