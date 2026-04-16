import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create images directory
os.makedirs('report/images', exist_ok=True)

def parse_data_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract variable assignments
    pattern = r'^([a-zA-Z0-9_]+)\s*=\s*(.+)$'
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            var_name = match.group(1)
            var_value_str = match.group(2)
            try:
                # Handle specific cases that ast.literal_eval fails on
                if var_name == 'deposition_sequences':
                    data[var_name] = [
                        ('Na13 + Na', ['Na']*50), 
                        ('Na13@Rb32 + Rb', ['Rb']*30), 
                        ('Ag13 + Cu', ['Cu']*20 + ['Ag']*10 + ['Cu']*20), 
                        ('Rb72 + Cs', ['Cs']*40)
                    ]
                else:
                    var_value = ast.literal_eval(var_value_str)
                    data[var_name] = var_value
            except Exception as e:
                print(f"Error parsing {var_name}: {e}")
    return data

data = parse_data_file('data/Multi-component Icosahedral Reproduction Data.txt')

# Figure 1: Magic Numbers and Shell Sequences
plt.figure(figsize=(10, 6))
shells = list(range(1, len(data['mackay_sequence']) + 1))
plt.plot(shells, data['mackay_sequence'], 'o-', label='Mackay Sequence (Achiral)', linewidth=2, markersize=8)
shells_b5 = list(range(1, len(data['new_sequence_b5']) + 1))
plt.plot(shells_b5, data['new_sequence_b5'], 's--', label='New Sequence b=5 (Chiral)', linewidth=2, markersize=8)
plt.xlabel('Shell Index', fontsize=14)
plt.ylabel('Number of Atoms (Magic Number)', fontsize=14)
plt.title('Magic Numbers for Icosahedral Shells', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('report/images/fig1_magic_numbers.png', dpi=300)
plt.close()

# Figure 2: Shell Energies
plt.figure(figsize=(10, 6))
energies = data['shell_energies']
# (shell_idx, chiral_label, energy)
shell_idx = [e[0] for e in energies]
chiral_labels = [e[1] for e in energies]
energy_vals = [e[2] for e in energies]

# Group by chiral label
labels = list(set([e[1] for e in energies]))
colors = data['shell_colors']

for label in labels:
    x = [e[0] for e in energies if e[1] == label]
    y = [e[2] for e in energies if e[1] == label]
    plt.plot(x, y, marker='o', linestyle='-', label=label, color=colors.get(label, 'black'), markersize=10, linewidth=2)

plt.xlabel('Shell Index', fontsize=14)
plt.ylabel('Normalized Shell Energy', fontsize=14)
plt.title('Relative Shell Energies for Different Chiral Categories', fontsize=16)
plt.xticks([1, 2, 3])
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('report/images/fig2_shell_energies.png', dpi=300)
plt.close()

# Figure 3: Optimal Size Mismatch Ranges
plt.figure(figsize=(10, 6))
mismatch_ranges = data['optimal_mismatch_ranges']
# ('MC', 'MC', 0.03, 0.05)
categories = [f"{m[0]}-{m[1]}" for m in mismatch_ranges]
mins = [m[2] for m in mismatch_ranges]
maxs = [m[3] for m in mismatch_ranges]
means = [(m[2]+m[3])/2 for m in mismatch_ranges]
errors = [(m[3]-m[2])/2 for m in mismatch_ranges]

plt.errorbar(categories, means, yerr=errors, fmt='o', capsize=8, markersize=10, linewidth=2, color='darkblue')
plt.xlabel('Shell Transition (Category 1 -> Category 2)', fontsize=14)
plt.ylabel('Optimal Size Mismatch Range', fontsize=14)
plt.title('Optimal Size Mismatch for Shell Transitions', fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('report/images/fig3_optimal_mismatch.png', dpi=300)
plt.close()

# Figure 4: Experimental vs Theoretical Mismatch
plt.figure(figsize=(8, 8))
exp_points = data['experimental_points']
# (T_i, T_{i+1}, measured sm, theoretical sm)
measured = [p[2] for p in exp_points]
theoretical = [p[3] for p in exp_points]
labels = [f"T={p[0]}->{p[1]}" for p in exp_points]

plt.scatter(theoretical, measured, s=100, color='red', zorder=5)
for i, txt in enumerate(labels):
    plt.annotate(txt, (theoretical[i], measured[i]), xytext=(5, 5), textcoords='offset points', fontsize=12)

# Plot y=x line
min_val = min(min(measured), min(theoretical)) - 0.02
max_val = max(max(measured), max(theoretical)) + 0.02
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Agreement')

plt.xlabel('Theoretical Size Mismatch', fontsize=14)
plt.ylabel('Measured Size Mismatch', fontsize=14)
plt.title('Experimental vs Theoretical Size Mismatch', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('report/images/fig4_exp_vs_theo_mismatch.png', dpi=300)
plt.close()

# Figure 5: Growth Dynamics - Mismatch over time
plt.figure(figsize=(10, 6))
growth_res = data['growth_results']
# (steps, chiral category, average mismatch)

seq1_steps, seq1_mis = [], []
seq2_steps, seq2_mis = [], []
seq3_steps, seq3_mis = [], []

current_seq = 1
for r in growth_res:
    if r[0] == 0:
        if current_seq == 1 and len(seq1_steps) > 0:
            current_seq = 2
        elif current_seq == 2 and len(seq2_steps) > 0:
            current_seq = 3
            
    if current_seq == 1:
        seq1_steps.append(r[0])
        seq1_mis.append(r[2])
    elif current_seq == 2:
        seq2_steps.append(r[0])
        seq2_mis.append(r[2])
    elif current_seq == 3:
        seq3_steps.append(r[0])
        seq3_mis.append(r[2])

plt.plot(seq1_steps, seq1_mis, 'o-', label='Sequence 1 (MC)', linewidth=2, markersize=8, color=colors.get('MC', 'blue'))
plt.plot(seq2_steps, seq2_mis, 's--', label='Sequence 2 (Ch1)', linewidth=2, markersize=8, color=colors.get('Ch1', 'green'))
plt.plot(seq3_steps, seq3_mis, '^:', label='Sequence 3 (MC->Ch1)', linewidth=2, markersize=8, color='purple')

plt.xlabel('Simulation Steps', fontsize=14)
plt.ylabel('Average Mismatch', fontsize=14)
plt.title('Average Mismatch Evolution during Growth Simulation', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('report/images/fig5_growth_dynamics.png', dpi=300)
plt.close()

# Figure 6: Path Selection Statistics
plt.figure(figsize=(10, 6))
path_stats = data['path_selection_stats']
labels = [p[0] for p in path_stats]
counts = [p[1] for p in path_stats]

colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_pie, textprops={'fontsize': 12})
plt.axis('equal')
plt.title('Path Selection Statistics in Growth Simulation', fontsize=16)
plt.tight_layout()
plt.savefig('report/images/fig6_path_selection.png', dpi=300)
plt.close()

print("Figures generated successfully.")
