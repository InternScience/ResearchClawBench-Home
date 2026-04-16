"""
DIDS-MFL: Unknown Attack Visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import json, os

plt.style.use('seaborn-v0_8-whitegrid')

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(WORKSPACE, 'report', 'images')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')

with open(os.path.join(OUTPUT_DIR, 'unknown_attack_results.json')) as f:
    unknown_results = json.load(f)

# ===================== Figure 13: Unknown Attack Detection =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scenario 1: Backdoor and Worms as unknown
s1 = unknown_results['scenario1']
s1_names = list(s1['per_unknown_type'].keys())
s1_det_rates = [s1['per_unknown_type'][n]['detection_rate'] for n in s1_names]
s1_f1s = [s1['per_unknown_type'][n]['binary_f1'] for n in s1_names]

x = np.arange(len(s1_names))
width = 0.35
axes[0].bar(x - width/2, s1_det_rates, width, label='Detection Rate', color='#F44336')
axes[0].bar(x + width/2, s1_f1s, width, label='Binary F1', color='#4CAF50')
axes[0].set_title(f'Scenario 1: Unknown {", ".join(s1_names)}\nOverall Binary F1={s1["overall_binary_f1"]:.4f}', fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(s1_names)
axes[0].legend()
axes[0].set_ylim(0, 1.1)
for i, (dr, f1v) in enumerate(zip(s1_det_rates, s1_f1s)):
    axes[0].annotate(f'{dr:.3f}', xy=(i-width/2, dr), ha='center', fontsize=10)
    axes[0].annotate(f'{f1v:.3f}', xy=(i+width/2, f1v), ha='center', fontsize=10)

# Scenario 2: Analysis and Shellcode as unknown
s2 = unknown_results['scenario2']
s2_names = list(s2['per_unknown_type'].keys())
s2_det_rates = [s2['per_unknown_type'][n]['detection_rate'] for n in s2_names]
s2_f1s = [s2['per_unknown_type'][n]['binary_f1'] for n in s2_names]

x = np.arange(len(s2_names))
axes[1].bar(x - width/2, s2_det_rates, width, label='Detection Rate', color='#F44336')
axes[1].bar(x + width/2, s2_f1s, width, label='Binary F1', color='#4CAF50')
axes[1].set_title(f'Scenario 2: Unknown {", ".join(s2_names)}\nOverall Binary F1={s2["overall_binary_f1"]:.4f}', fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(s2_names)
axes[1].legend()
axes[1].set_ylim(0, 1.1)
for i, (dr, f1v) in enumerate(zip(s2_det_rates, s2_f1s)):
    axes[1].annotate(f'{dr:.3f}', xy=(i-width/2, dr), ha='center', fontsize=10)
    axes[1].annotate(f'{f1v:.3f}', xy=(i+width/2, f1v), ha='center', fontsize=10)

plt.suptitle('Unknown Attack Detection Performance (DIDS-MFL)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig13_unknown_attack_detection.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Unknown attack visualization complete.")