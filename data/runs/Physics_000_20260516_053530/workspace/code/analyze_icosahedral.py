#!/usr/bin/env python3
"""
Analysis script for Multi-component Icosahedral Structures
Reproduces key results from the reproduction data and generates figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Ensure output dir
os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Data from Multi-component Icosahedral Reproduction Data.txt
mackay_sequence = [1, 13, 55, 147, 309]
new_sequence_b5 = [1, 13, 45, 117, 239, 431]
chiral_labels = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
shell_colors = {'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c', 'Ch2': '#d62728', 'Ch3': '#9467bd', 'Ch4': '#8c564b', 'Ch5': '#e377c2'}

atomic_radii = {'Na': 1.86, 'K': 2.27, 'Rb': 2.48, 'Cs': 2.65, 'Ag': 1.44, 'Cu': 1.28, 'Ni': 1.24}
multicomponent_clusters = [('Na13@Rb32', 'Na', 'Rb', 'MC', 'Ch1'), ('K13@Cs42', 'K', 'Cs', 'MC', 'Ch2'), ('Ag13@Cu45', 'Ag', 'Cu', 'MC', 'Ch1')]
shell_energies = [(1, 'MC', 0.00), (2, 'MC', -2.35), (2, 'Ch1', -2.15), (3, 'MC', -4.82), (3, 'Ch1', -4.61), (3, 'BG', -4.55)]
mismatch_params = [(1, 2, 'MC', 'MC', 0.04), (1, 2, 'MC', 'Ch1', 0.14), (2, 3, 'MC', 'MC', 0.038), (2, 3, 'MC', 'Ch1', 0.136), (2, 3, 'Ch1', 'Ch2', 0.21)]
experimental_points = [(1, 3, 0.048, 0.045), (3, 4, 0.042, 0.044), (4, 7, 0.138, 0.142), (7, 12, 0.132, 0.139)]
growth_results = [(0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02), (30, 'MC', 0.025), (40, 'MC', 0.03), (50, 'MC', 0.035),
                  (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14), (30, 'Ch1', 0.138), (40, 'Ch1', 0.136), (50, 'Ch1', 0.135),
                  (0, 'MC', 0.00), (10, 'MC', 0.08), (20, 'Ch1', 0.14), (30, 'Ch1', 0.15), (40, 'Ch1', 0.145), (50, 'Ch1', 0.142)]
path_selection_stats = [('Conservative path', 325), ('Mismatch-driven path', 125), ('Random path', 50), ('Reverse step', 100)]
optimal_mismatch_ranges = [('MC', 'MC', 0.03, 0.05), ('MC', 'Ch1', 0.12, 0.16), ('MC', 'Ch2', 0.19, 0.22), ('MC', 'BG', 0.08, 0.10)]

# Figure 1: Magic Number Sequences Comparison
fig, ax = plt.subplots(figsize=(8, 5))
shells = np.arange(1, 6)
ax.plot(shells, mackay_sequence, 'o-', label='Mackay (standard)', color='#1f77b4', linewidth=2, markersize=8)
shells_new = np.arange(1, len(new_sequence_b5)+1)
ax.plot(shells_new, new_sequence_b5, 's--', label='New (b=5)', color='#ff7f0e', linewidth=2, markersize=8)
ax.set_xlabel('Shell Index', fontsize=12)
ax.set_ylabel('Number of Atoms', fontsize=12)
ax.set_title('Comparison of Icosahedral Magic Number Sequences', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure1_magic_numbers.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Shell Energy Landscape
fig, ax = plt.subplots(figsize=(8, 5))
shells_e = [1,2,2,3,3,3]
energies = [0.00, -2.35, -2.15, -4.82, -4.61, -4.55]
labels_e = ['MC(1)', 'MC(2)', 'Ch1(2)', 'MC(3)', 'Ch1(3)', 'BG(3)']
colors_e = [shell_colors['MC'], shell_colors['MC'], shell_colors['Ch1'], shell_colors['MC'], shell_colors['Ch1'], shell_colors['BG']]
bars = ax.bar(labels_e, energies, color=colors_e, edgecolor='black')
ax.set_ylabel('Relative Energy (normalized)', fontsize=12)
ax.set_title('Shell Energy Landscape for Multi-Shell Icosahedra', fontsize=14)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
for bar, e in zip(bars, energies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{e:.2f}', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('report/images/figure2_shell_energies.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Growth Simulation Results - Mismatch Evolution
fig, ax = plt.subplots(figsize=(8, 5))
steps_mc = [r[0] for r in growth_results if r[1]=='MC' and r[0]<=50][:6]
mismatch_mc = [r[2] for r in growth_results if r[1]=='MC' and r[0]<=50][:6]
steps_ch1 = [r[0] for r in growth_results if r[1]=='Ch1' and r[0]<=50][:6]
mismatch_ch1 = [r[2] for r in growth_results if r[1]=='Ch1' and r[0]<=50][:6]
ax.plot(steps_mc, mismatch_mc, 'o-', label='MC path (conservative)', color=shell_colors['MC'], linewidth=2)
ax.plot(steps_ch1, mismatch_ch1, 's-', label='Ch1 path (mismatch-driven)', color=shell_colors['Ch1'], linewidth=2)
ax.set_xlabel('Simulation Steps', fontsize=12)
ax.set_ylabel('Average Size Mismatch', fontsize=12)
ax.set_title('Dynamic Growth: Size Mismatch Evolution', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure3_growth_mismatch.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Path Selection Statistics (Pie Chart)
fig, ax = plt.subplots(figsize=(7, 7))
labels_p = [p[0] for p in path_selection_stats]
sizes = [p[1] for p in path_selection_stats]
colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
ax.pie(sizes, labels=labels_p, autopct='%1.1f%%', colors=colors_p, startangle=90, textprops={'fontsize':10})
ax.set_title('Growth Path Selection Statistics in Simulations', fontsize=14)
plt.tight_layout()
plt.savefig('report/images/figure4_path_stats.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 5: Experimental vs Theoretical Size Mismatch
fig, ax = plt.subplots(figsize=(8, 5))
exp_idx = [f'T{i}-{j}' for i,j,_,_ in experimental_points]
meas = [p[2] for p in experimental_points]
theo = [p[3] for p in experimental_points]
x = np.arange(len(exp_idx))
width = 0.35
bars1 = ax.bar(x - width/2, meas, width, label='Measured', color='#1f77b4')
bars2 = ax.bar(x + width/2, theo, width, label='Theoretical', color='#ff7f0e')
ax.set_ylabel('Size Mismatch (Δr)', fontsize=12)
ax.set_title('Validation: Experimental vs Theoretical Size Mismatch', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(exp_idx)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{bar.get_height():.3f}', ha='center', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{bar.get_height():.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('report/images/figure5_validation_mismatch.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 6: Optimal Mismatch Ranges by Category
fig, ax = plt.subplots(figsize=(8, 5))
cats = [f"{r[0]}-{r[1]}" for r in optimal_mismatch_ranges]
low = [r[2] for r in optimal_mismatch_ranges]
high = [r[3] for r in optimal_mismatch_ranges]
x = np.arange(len(cats))
ax.bar(x, [h-l for l,h in zip(low,high)], bottom=low, width=0.6, color=['#1f77b4','#2ca02c','#d62728','#ff7f0e'], edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=15)
ax.set_ylabel('Size Mismatch Range', fontsize=12)
ax.set_title('Optimal Size Mismatch Ranges for Shell Categories', fontsize=14)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/figure6_optimal_ranges.png', dpi=150, bbox_inches='tight')
plt.close()

# Save key data to outputs
np.savez('outputs/icosahedral_data.npz',
         mackay=mackay_sequence,
         new_b5=new_sequence_b5,
         shell_energies=np.array(shell_energies, dtype=object),
         growth_results=np.array(growth_results, dtype=object),
         experimental_points=np.array(experimental_points))

print("Analysis complete. Figures saved to report/images/. Data saved to outputs/.")
print("Generated figures:")
for f in sorted(os.listdir('report/images')):
    print(f"  - {f}")