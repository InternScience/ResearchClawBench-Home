#!/usr/bin/env python3
"""
Multi-Component Icosahedral Nanocluster Analysis - Main Script
==============================================================
Generates all figures and outputs for the research report.
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar

# ============================================================
# 1. DATA
# ============================================================

# Hexagonal coordinate sequence
hex_coords = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
              (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),
              (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),
              (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),
              (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),
              (5,0),(5,1),(5,2),(5,3),(5,4),(5,5)]

mackay_seq = [1, 13, 55, 147, 309]
new_seq_b5 = [1, 13, 45, 117, 239, 431]

chiral_labels = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
shell_colors = {'MC':'#1f77b4','BG':'#ff7f0e','Ch1':'#2ca02c',
                'Ch2':'#d62728','Ch3':'#9467bd','Ch4':'#8c564b','Ch5':'#e377c2'}

atomic_radii = {'Na':1.86,'K':2.27,'Rb':2.48,'Cs':2.65,'Ag':1.44,'Cu':1.28,'Ni':1.24}

atomic_pairs = [('Na','Rb',0.22),('Ag','Cu',0.12),('Ag','Ni',0.15),('Cu','Ni',0.032)]

optimal_mismatch = [('MC','MC',0.03,0.05),('MC','Ch1',0.12,0.16),
                    ('MC','Ch2',0.19,0.22),('MC','BG',0.08,0.10)]

clusters = [('Na13@Rb32','Na','Rb','MC','Ch1'),
            ('K13@Cs42','K','Cs','MC','Ch2'),
            ('Ag13@Cu45','Ag','Cu','MC','Ch1')]

shell_energies = [(1,'MC',0.00),(2,'MC',-2.35),(2,'Ch1',-2.15),
                  (3,'MC',-4.82),(3,'Ch1',-4.61),(3,'BG',-4.55)]

mismatch_params = [(1,2,'MC','MC',0.04),(1,2,'MC','Ch1',0.14),
                   (2,3,'MC','MC',0.038),(2,3,'MC','Ch1',0.136),(2,3,'Ch1','Ch2',0.21)]

exp_points = [(1,3,0.048,0.045),(3,4,0.042,0.044),
              (4,7,0.138,0.142),(7,12,0.132,0.139)]

growth_params = {'temperature':300.0,'deposition_rate':0.01,
                 'simulation_steps':1000,'beta_factor':1.0,'delta_opt':0.04,'random_seed':42}
path_weights = {'conservative_step':0.65,'mismatch_driven_step':0.25,'random_step':0.10}

lj_params = {'Na-Na':(1.0,3.72),'Rb-Rb':(1.0,4.96),'Cs-Cs':(1.0,5.30),
             'Ag-Ag':(1.0,2.88),'Cu-Cu':(1.0,2.56),'Na-Rb':(1.0,4.34),'Ag-Cu':(1.0,2.72)}

growth_results = [(0,'MC',0.00),(10,'MC',0.01),(20,'MC',0.02),
                  (30,'MC',0.025),(40,'MC',0.03),(50,'MC',0.035),
                  (0,'Ch1',0.00),(10,'Ch1',0.12),(20,'Ch1',0.14),
                  (30,'Ch1',0.138),(40,'Ch1',0.136),(50,'Ch1',0.135),
                  (0,'MC',0.00),(10,'MC',0.08),(20,'Ch1',0.14),
                  (30,'Ch1',0.15),(40,'Ch1',0.145),(50,'Ch1',0.142)]

path_stats = {'Conservative path':325,'Mismatch-driven path':125,
              'Random path':50,'Reverse step':100}

# ============================================================
# 2. CORE FUNCTIONS
# ============================================================

def triangulation(h, k):
    return h**2 + h*k + k**2

def size_mismatch(e1, e2):
    r1, r2 = atomic_radii[e1], atomic_radii[e2]
    return abs(r1 - r2) / ((r1 + r2) / 2)

def hex_to_cart(h, k, a=1.0):
    return a*(h + k/2.0), a*(k*np.sqrt(3)/2.0)

def lj_energy(r, eps, sig):
    sr6 = (sig/r)**6
    return 4.0*eps*(sr6**2 - sr6)

def classify_chiral(T):
    if T <= 3: return 'MC'
    elif T <= 7: return 'BG'
    elif T <= 12: return 'Ch1'
    elif T <= 19: return 'Ch2'
    elif T <= 27: return 'Ch3'
    elif T <= 37: return 'Ch4'
    else: return 'Ch5'

def cluster_energy(core_type, shell_type, n_core, n_shell, sm):
    e_core = next((e for s,c,e in shell_energies if s==1 and c==core_type), 0)
    e_shell = next((e for s,c,e in shell_energies if s==2 and c==shell_type), 0)
    e = (n_core*e_core + n_shell*e_shell)/(n_core + n_shell)
    e -= 1.5*(sm - 0.14)**2
    return e

def simulate_growth(n_steps=50, seed_mismatch=0.0, target_mismatch=0.14):
    rng = np.random.RandomState(42)
    mm = [seed_mismatch]
    for _ in range(1, n_steps):
        r = rng.random()
        if r < 0.65:
            mm.append(mm[-1] + rng.normal(0, 0.008))
        elif r < 0.90:
            delta = target_mismatch - mm[-1]
            mm.append(mm[-1] + 0.015*np.sign(delta) + rng.normal(0, 0.005))
        else:
            mm.append(rng.uniform(0, 0.25))
    return np.clip(mm, 0, 0.3)

# ============================================================
# 3. FIGURE 1: Hexagonal Lattice + Shell Path
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for h, k in hex_coords[:36]:
    x, y = hex_to_cart(h, k)
    T = triangulation(h, k)
    ch = classify_chiral(T)
    c = shell_colors[ch]
    circle = plt.Circle((x, y), 0.3, color=c, alpha=0.7, ec='black', lw=0.5)
    ax.add_patch(circle)
    ax.text(x, y, f'({h},{k})\nT={T}', ha='center', va='center', fontsize=4.5, fontweight='bold')

ax.set_xlim(-0.5, 9); ax.set_ylim(-0.5, 6); ax.set_aspect('equal')
ax.set_xlabel('x (lattice units)', fontsize=11); ax.set_ylabel('y (lattice units)', fontsize=11)
ax.set_title('(a) Hexagonal Lattice with Triangulation Numbers', fontsize=12, fontweight='bold')
legend_el = [mpatches.Patch(facecolor=shell_colors[l], label=f'{l}') for l in ['MC','BG','Ch1','Ch2','Ch3']]
ax.legend(handles=legend_el, loc='upper left', fontsize=8)

ax2 = axes[1]
path_c = hex_coords[:6]
for i, (h, k) in enumerate(path_c):
    x, y = hex_to_cart(h, k)
    T = triangulation(h, k)
    ch = classify_chiral(T)
    c = shell_colors[ch]
    circle = plt.Circle((x, y), 0.25, color=c, alpha=0.8, ec='black', lw=1)
    ax2.add_patch(circle)
    ax2.text(x, y+0.45, f'Step {i+1}\n({h},{k})\nT={T}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    if i < len(path_c)-1:
        h2, k2 = path_c[i+1]
        x2, y2 = hex_to_cart(h2, k2)
        ax2.annotate('', xy=(x2, y2), xytext=(x, y), arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax2.set_xlim(-0.5, 3); ax2.set_ylim(-0.5, 4); ax2.set_aspect('equal')
ax2.set_xlabel('x (lattice units)', fontsize=11); ax2.set_ylabel('y (lattice units)', fontsize=11)
ax2.set_title('(b) Shell Sequence Path (MC→Ch1→Ch2→...)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/figure1_hexagonal_lattice.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 done")

# ============================================================
# 4. FIGURE 2: Energy Landscape
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
shells = [1, 2, 3]; chiral_types_e = ['MC', 'Ch1', 'BG']
bw = 0.25
for i, ch in enumerate(chiral_types_e):
    energies = [next((e for s,c,e in shell_energies if s==s_ and c==ch), 0) for s_ in shells]
    pos = np.arange(len(shells)) + i*bw
    bars = ax.bar(pos, energies, bw, color=shell_colors[ch], label=ch, alpha=0.8, edgecolor='black')
    for b, en in zip(bars, energies):
        if en != 0:
            ax.text(b.get_x()+b.get_width()/2., b.get_height()-0.15, f'{en:.2f}', ha='center', va='top', fontsize=7)
ax.set_xlabel('Shell Layer', fontsize=11); ax.set_ylabel('Normalized Energy', fontsize=11)
ax.set_title('(a) Shell Energy by Chiral Type', fontsize=12, fontweight='bold')
ax.set_xticks(np.arange(len(shells))+bw); ax.set_xticklabels([f'Shell {s}' for s in shells])
ax.legend(fontsize=9); ax.axhline(y=0, color='black', linestyle='--', lw=0.5)

ax2 = axes[1]
labels = [f"S{i[0]}→{i[1]}\n{i[2]}→{i[3]}" for i in mismatch_params]
vals = [i[4] for i in mismatch_params]
colors = [shell_colors.get(i[3],'#666') for i in mismatch_params]
bars = ax2.bar(range(len(labels)), vals, color=colors, alpha=0.8, edgecolor='black')
for b, v in zip(bars, vals):
    ax2.text(b.get_x()+b.get_width()/2., b.get_height()+0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax2.set_xlabel('Shell Transition', fontsize=11); ax2.set_ylabel('Optimal Size Mismatch', fontsize=11)
ax2.set_title('(b) Inter-shell Size Mismatch', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=7)

ax3 = axes[2]
c_names = [c[0] for c in clusters]
core_r = [atomic_radii[c[1]] for c in clusters]
shell_r = [atomic_radii[c[2]] for c in clusters]
sm_calc = [size_mismatch(c[1], c[2]) for c in clusters]
x = np.arange(len(c_names))
ax3.bar(x-0.15, core_r, 0.3, label='Core radius (Å)', color='#1f77b4', alpha=0.8, edgecolor='black')
ax3.bar(x+0.15, shell_r, 0.3, label='Shell radius (Å)', color='#ff7f0e', alpha=0.8, edgecolor='black')
ax3t = ax3.twinx()
ax3t.plot(x, sm_calc, 'ko-', ms=8, label='Size mismatch', lw=2)
ax3t.set_ylabel('Size Mismatch', fontsize=11)
ax3.set_xlabel('Cluster', fontsize=11); ax3.set_ylabel('Atomic Radius (Å)', fontsize=11)
ax3.set_title('(c) Predicted Stable Clusters', fontsize=12, fontweight='bold')
ax3.set_xticks(x); ax3.set_xticklabels(c_names, fontsize=9)
l1, lb1 = ax3.get_legend_handles_labels()
l2, lb2 = ax3t.get_legend_handles_labels()
ax3.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('report/images/figure2_energy_landscape.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 done")

# ============================================================
# 5. FIGURE 3: Mismatch Compatibility
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
elements = list(atomic_radii.keys())
n_e = len(elements)
mm_mat = np.zeros((n_e, n_e))
for i, e1 in enumerate(elements):
    for j, e2 in enumerate(elements):
        mm_mat[i, j] = size_mismatch(e1, e2)

im = ax.imshow(mm_mat, cmap='YlOrRd', aspect='auto')
for i in range(n_e):
    for j in range(n_e):
        v = mm_mat[i, j]
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7, color='white' if v>0.15 else 'black')
ax.set_xticks(range(n_e)); ax.set_xticklabels(elements, fontsize=10)
ax.set_yticks(range(n_e)); ax.set_yticklabels(elements, fontsize=10)
plt.colorbar(im, ax=ax, label='Size Mismatch')
ax.set_title('(a) Atomic Size Mismatch Matrix', fontsize=12, fontweight='bold')

ax2 = axes[1]
opt_labels = [f"{o[0]}@{o[1]}" for o in optimal_mismatch]
for i, o in enumerate(optimal_mismatch):
    w = o[3] - o[2]
    ax2.barh(i, w, left=o[2], height=0.5, color=shell_colors[o[1]], alpha=0.7, edgecolor='black')
    ax2.text(o[2]+w/2, i+0.3, f"[{o[2]:.2f}, {o[3]:.2f}]", ha='center', va='bottom', fontsize=8)

pair_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, (e1, e2, sm) in enumerate(atomic_pairs):
    ax2.axvline(x=sm, color=pair_colors[i], ls='--', lw=2, alpha=0.8)
    ax2.text(sm, len(opt_labels)-0.3+i*0.15, f'{e1}-{e2}', ha='center', va='bottom', fontsize=8, color=pair_colors[i], fontweight='bold')

ax2.set_yticks(range(len(opt_labels))); ax2.set_yticklabels(opt_labels, fontsize=10)
ax2.set_xlabel('Size Mismatch', fontsize=11)
ax2.set_title('(b) Optimal Mismatch Ranges vs Element Pairs', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 0.35)
plt.tight_layout()
plt.savefig('report/images/figure3_mismatch_compatibility.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 done")

# ============================================================
# 6. FIGURE 4: Growth Paths
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0,0]
mc_pts = [(s, mm) for s, ch, mm in growth_results if ch == 'MC']
ch1_pts = [(s, mm) for s, ch, mm in growth_results if ch == 'Ch1']
ax.plot([p[0] for p in mc_pts[:6]], [p[1] for p in mc_pts[:6]], 'o-', color=shell_colors['MC'], label='MC seed', ms=6, lw=2)
ax.plot([p[0] for p in mc_pts[6:]], [p[1] for p in mc_pts[6:]], 's--', color=shell_colors['MC'], alpha=0.6, label='MC seed (alt)', ms=6, lw=2)
ax.plot([p[0] for p in ch1_pts], [p[1] for p in ch1_pts], 'o-', color=shell_colors['Ch1'], label='Ch1 seed', ms=6, lw=2)
ax.set_xlabel('Growth Steps', fontsize=11); ax.set_ylabel('Average Size Mismatch', fontsize=11)
ax.set_title('(a) Growth Trajectory (Simulation Data)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax2 = axes[0,1]
labels = list(path_stats.keys()); values = list(path_stats.values())
colors_pie = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
wedges, texts, autotexts = ax2.pie(values, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90, textprops={'fontsize':9})
for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
ax2.set_title('(b) Growth Path Selection Statistics', fontsize=12, fontweight='bold')

ax3 = axes[1,0]
mm_cons = simulate_growth(50, 0.0, 0.04)
mm_md = simulate_growth(50, 0.0, 0.14)
rng2 = np.random.RandomState(42)
mm_rand = np.clip([rng2.uniform(0, 0.25) for _ in range(50)], 0, 0.3)
ax3.plot(range(50), mm_cons, label='Conservative', color='#2ecc71', lw=1.5, alpha=0.8)
ax3.plot(range(50), mm_md, label='Mismatch-driven', color='#3498db', lw=1.5, alpha=0.8)
ax3.plot(range(50), mm_rand, label='Random', color='#e74c3c', lw=1.5, alpha=0.8)
ax3.axhline(y=0.04, color='gray', ls='--', alpha=0.5, label='MC optimal')
ax3.axhline(y=0.14, color='gray', ls=':', alpha=0.5, label='Ch1 optimal')
ax3.set_xlabel('Growth Steps', fontsize=11); ax3.set_ylabel('Size Mismatch', fontsize=11)
ax3.set_title('(c) Simulated Growth Path Trajectories', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

ax4 = axes[1,1]
dep_data = [('Na₁₃+Na', 0.00), ('Na₁₃@Rb₃₂+Rb', 0.00), ('Ag₁₃+Cu', 0.08),
            ('Ag₁₃+Cu (mid)', 0.14), ('Ag₁₃+Cu (late)', 0.15), ('Ag₁₃+Cu (final)', 0.145)]
names = [d[0] for d in dep_data]
mms = [d[1] for d in dep_data]
clrs = ['#1f77b4', '#1f77b4', '#ff7f0e', '#2ca02c', '#2ca02c', '#2ca02c']
ax4.barh(range(len(names)), mms, color=clrs, alpha=0.8, edgecolor='black')
for i, (n, mm) in enumerate(zip(names, mms)):
    ax4.text(mm+0.002, i, f'{mm:.3f}', va='center', fontsize=8)
ax4.set_yticks(range(len(names))); ax4.set_yticklabels(names, fontsize=9)
ax4.set_xlabel('Final Size Mismatch', fontsize=11)
ax4.set_title('(d) Deposition Sequence Outcomes', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('report/images/figure4_growth_paths.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 done")

# ============================================================
# 7. FIGURE 5: Stability Phase Diagram
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
elem_pairs = []
for i, e1 in enumerate(elements):
    for j, e2 in enumerate(elements):
        if i < j:
            elem_pairs.append((e1, e2, size_mismatch(e1, e2)))
chiral_types_s = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3']
stab_map = np.zeros((len(elem_pairs), len(chiral_types_s)))
for i, (e1, e2, sm) in enumerate(elem_pairs):
    for j, ch in enumerate(chiral_types_s):
        stab_map[i, j] = cluster_energy('MC', ch, 13, 32, sm)

im = ax.imshow(stab_map, cmap='RdYlGn_r', aspect='auto')
pair_lbl = [f'{e1}-{e2}' for e1, e2, _ in elem_pairs]
ax.set_yticks(range(len(pair_lbl))); ax.set_yticklabels(pair_lbl, fontsize=7)
ax.set_xticks(range(len(chiral_types_s))); ax.set_xticklabels(chiral_types_s, fontsize=10)
ax.set_xlabel('Chiral Type', fontsize=11); ax.set_ylabel('Element Pair', fontsize=11)
ax.set_title('(a) Stability Map: Energy by Element Pair and Chiral Type', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized Energy')
for i in range(len(elem_pairs)):
    best_j = np.argmin(stab_map[i, :])
    ax.plot(best_j, i, 'k*', ms=10)

ax2 = axes[1]
sm_range = np.linspace(0, 0.35, 100)
for ch in ['MC', 'Ch1', 'Ch2', 'BG']:
    energies = [cluster_energy('MC', ch, 13, 32, sm) for sm in sm_range]
    ax2.plot(sm_range, energies, label=ch, color=shell_colors[ch], lw=2)
for e1, e2, sm in atomic_pairs:
    ax2.axvline(x=sm, color='gray', ls=':', alpha=0.4)
    ax2.text(sm+0.003, ax2.get_ylim()[0]+0.05 if ax2.get_ylim()[0] else -3, f'{e1}-{e2}', fontsize=7, rotation=90, va='bottom')
ax2.set_xlabel('Size Mismatch', fontsize=11); ax2.set_ylabel('Normalized Energy', fontsize=11)
ax2.set_title('(b) Energy vs Size Mismatch for Different Chiral Types', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure5_stability_phase.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 done")

# ============================================================
# 8. FIGURE 6: Shell Energy Comparison + Validation
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
shell_labels = ['Shell 1\n(MC)', 'Shell 2\n(MC)', 'Shell 2\n(Ch1)', 'Shell 3\n(MC)', 'Shell 3\n(Ch1)', 'Shell 3\n(BG)']
energies = [e for _, _, e in shell_energies]
clrs = [shell_colors['MC'], shell_colors['MC'], shell_colors['Ch1'], 
        shell_colors['MC'], shell_colors['Ch1'], shell_colors['BG']]
bars = ax.bar(range(len(energies)), energies, color=clrs, alpha=0.8, edgecolor='black')
for b, en in zip(bars, energies):
    ax.text(b.get_x()+b.get_width()/2., min(en, 0)-0.2 if en < 0 else 0.05, f'{en:.2f}', ha='center', va='top' if en < 0 else 'bottom', fontsize=8)
ax.set_xticks(range(len(shell_labels))); ax.set_xticklabels(shell_labels, fontsize=8)
ax.set_ylabel('Normalized Energy', fontsize=11)
ax.set_title('(a) Shell Energy Comparison Across Configurations', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', ls='--', lw=0.5); ax.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
exp_Ti = [p[0] for p in exp_points]
exp_Tj = [p[1] for p in exp_points]
exp_meas = [p[2] for p in exp_points]
exp_theory = [p[3] for p in exp_points]
x_exp = np.arange(len(exp_points))
ax2.bar(x_exp-0.15, exp_meas, 0.3, label='Measured sm', color='#3498db', alpha=0.8, edgecolor='black')
ax2.bar(x_exp+0.15, exp_theory, 0.3, label='Theoretical sm', color='#e74c3c', alpha=0.8, edgecolor='black')
for i, (m, t) in enumerate(zip(exp_meas, exp_theory)):
    ax2.text(i-0.15, m+0.001, f'{m:.3f}', ha='center', va='bottom', fontsize=7)
    ax2.text(i+0.15, t+0.001, f'{t:.3f}', ha='center', va='bottom', fontsize=7)
ax2.set_xticks(x_exp)
ax2.set_xticklabels([f'T({Ti},{Tj})' for Ti, Tj in zip(exp_Ti, exp_Tj)], fontsize=9)
ax2.set_ylabel('Size Mismatch', fontsize=11)
ax2.set_title('(b) Theory vs Experiment Validation', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('report/images/figure6_shell_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 done")

# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

# Save stability landscape
landscape = []
for e1, e2, sm in elem_pairs:
    for ch in chiral_types_s:
        landscape.append({'elem1': e1, 'elem2': e2, 'mismatch': round(sm, 4),
                         'chiral': ch, 'energy': round(cluster_energy('MC', ch, 13, 32, sm), 4)})
with open('outputs/stability_landscape.json', 'w') as f:
    json.dump(landscape, f, indent=2)

# Save predicted clusters
pred_clusters = []
for formula, core, shell, ct, st in clusters:
    sm = size_mismatch(core, shell)
    e = cluster_energy(ct, st, 13, 32, sm)
    pred_clusters.append({'formula': formula, 'core': core, 'shell': shell,
                         'core_type': ct, 'shell_type': st, 'size_mismatch': round(sm, 4),
                         'energy': round(e, 4)})
with open('outputs/predicted_clusters.json', 'w') as f:
    json.dump(pred_clusters, f, indent=2)

# Save triangulation table
tri_table = []
for h, k in hex_coords:
    T = triangulation(h, k)
    ch = classify_chiral(T)
    tri_table.append({'h': h, 'k': k, 'T': T, 'chiral': ch})
with open('outputs/triangulation_table.json', 'w') as f:
    json.dump(tri_table, f, indent=2)

# Save validation results
validation = []
for Ti, Tj, meas, theor in exp_points:
    validation.append({'T_i': Ti, 'T_j': Tj, 'measured': meas, 'theoretical': theor,
                       'residual': round(abs(meas - theor), 4)})
with open('outputs/validation_results.json', 'w') as f:
    json.dump(validation, f, indent=2)

print("All outputs saved successfully!")
print("Figures: report/images/figure1-6 .png")
print("Outputs: stability_landscape.json, predicted_clusters.json, triangulation_table.json, validation_results.json")
