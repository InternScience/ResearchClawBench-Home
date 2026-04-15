"""
DIDS-MFL: Generate visualization figures for the report
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('report/images', exist_ok=True)

# Load results
with open('outputs/evaluation_results.json') as f:
    results = json.load(f)

with open('outputs/training_histories.json') as f:
    histories = json.load(f)

with open('outputs/data_statistics.json') as f:
    stats = json.load(f)

# ============================================================
# Figure 1: Binary Classification Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
models = ['DIDS-MFL', 'MLP', 'GraphSAGE', 'RandomForest']
metrics = ['accuracy', 'f1_macro', 'precision', 'recall']
metric_labels = ['Accuracy', 'F1 (Macro)', 'Precision', 'Recall']
x = np.arange(len(metrics))
width = 0.2
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for i, model in enumerate(models):
    vals = [results[model]['binary'][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=model, color=colors[i], edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('Score')
ax.set_title('Binary Classification: Benign vs Attack')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_labels)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/binary_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: binary_comparison.png")

# ============================================================
# Figure 2: Multi-class F1 Comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overall multi-class F1
models_dl = ['DIDS-MFL', 'MLP', 'GraphSAGE']
multi_f1 = [results[m]['multiclass']['f1_macro'] for m in models_dl]
multi_acc = [results[m]['multiclass']['accuracy'] for m in models_dl]
multi_f1.append(results['RandomForest']['multiclass']['f1_macro'])
multi_acc.append(results['RandomForest']['multiclass']['accuracy'])
all_models = models_dl + ['RandomForest']

bars = axes[0].bar(all_models, multi_f1, color=colors, edgecolor='white')
for bar, val in zip(bars, multi_f1):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
axes[0].set_ylabel('F1 Score (Macro)')
axes[0].set_title('Multi-class Classification: F1 (Macro)')
axes[0].set_ylim(0, 1.0)
axes[0].grid(axis='y', alpha=0.3)

bars2 = axes[1].bar(all_models, multi_acc, color=colors, edgecolor='white')
for bar, val in zip(bars2, multi_acc):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Multi-class Classification: Accuracy')
axes[1].set_ylim(0, 1.0)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/multiclass_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: multiclass_comparison.png")

# ============================================================
# Figure 3: Per-class F1 for each deep learning model
# ============================================================
attack_types = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 
                'Generic', 'Reconnaissance', 'Shellcode', 'Worms']

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(attack_types))
width = 0.25

for i, model in enumerate(['DIDS-MFL', 'MLP', 'GraphSAGE']):
    per_class = results[model]['multiclass'].get('per_class', {})
    f1_vals = [per_class.get(at, {}).get('f1', 0) for at in attack_types]
    bars = ax.bar(x + i * width, f1_vals, width, label=model, color=colors[i], edgecolor='white')

ax.set_ylabel('F1 Score')
ax.set_title('Per-class F1 Score by Attack Type (Few-shot classes on left)')
ax.set_xticks(x + width)
ax.set_xticklabels(attack_types, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Mark few-shot classes
few_shot = ['Analysis', 'Backdoor', 'Exploits', 'Fuzzers', 'Shellcode', 'Worms']
for i, at in enumerate(attack_types):
    if at in few_shot:
        ax.axvspan(i - 0.1, i + 3 * width + 0.1, alpha=0.1, color='red')

ax.text(1.5, 0.95, 'Few-shot region', ha='center', fontsize=10, color='red', style='italic')
plt.tight_layout()
plt.savefig('report/images/per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: per_class_f1.png")

# ============================================================
# Figure 4: Training curves
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for model_name, hist in histories.items():
    epochs = range(1, len(hist['train_loss']) + 1)
    axes[0, 0].plot(epochs, hist['train_loss'], label=model_name, linewidth=2)
    axes[0, 1].plot(epochs, hist['val_loss'], label=model_name, linewidth=2)
    axes[1, 0].plot(epochs, hist['val_binary_f1'], label=model_name, linewidth=2)
    axes[1, 1].plot(epochs, hist['val_multi_f1'], label=model_name, linewidth=2)

axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].set_title('Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].set_title('Validation Binary F1')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].set_title('Validation Multi-class F1')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_curves.png")

# ============================================================
# Figure 5: Few-shot vs Known attack performance
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

few_shot_types = ['Analysis', 'Backdoor', 'Exploits', 'Fuzzers', 'Shellcode', 'Worms']
known_types = ['DoS', 'Generic', 'Reconnaissance']

for model_name in ['DIDS-MFL', 'MLP', 'GraphSAGE']:
    per_class = results[model_name]['multiclass'].get('per_class', {})
    fs_f1 = [per_class.get(at, {}).get('f1', 0) for at in few_shot_types]
    kn_f1 = [per_class.get(at, {}).get('f1', 0) for at in known_types]
    
    fs_counts = [per_class.get(at, {}).get('count', 0) for at in few_shot_types]
    kn_counts = [per_class.get(at, {}).get('count', 0) for at in known_types]
    
    print(f"\n{model_name}:")
    print(f"  Few-shot avg F1: {np.mean(fs_f1):.4f}")
    print(f"  Known avg F1: {np.mean(kn_f1):.4f}")
    print(f"  Few-shot classes: {few_shot_types}")
    print(f"  Few-shot F1s: {fs_f1}")
    print(f"  Known F1s: {kn_f1}")

# Bar chart: few-shot vs known
x = np.arange(2)
width = 0.25
for i, model_name in enumerate(['DIDS-MFL', 'MLP', 'GraphSAGE']):
    per_class = results[model_name]['multiclass'].get('per_class', {})
    fs_f1 = np.mean([per_class.get(at, {}).get('f1', 0) for at in few_shot_types])
    kn_f1 = np.mean([per_class.get(at, {}).get('f1', 0) for at in known_types])
    ax.bar(x + i * width, [fs_f1, kn_f1], width, label=model_name, color=colors[i], edgecolor='white')

ax.set_ylabel('Average F1 Score')
ax.set_title('Few-shot vs Known Attack Detection Performance')
ax.set_xticks(x + width)
ax.set_xticklabels(['Few-shot Attacks\n(<500 samples)', 'Known Attacks\n(>500 samples)'])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('report/images/fewshot_vs_known.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fewshot_vs_known.png")

# ============================================================
# Figure 6: Radar chart of model capabilities
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

categories = ['Binary\nAccuracy', 'Binary\nF1', 'Multi\nF1 Macro', 'Multi\nAccuracy', 'Weighted\nF1']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, model in enumerate(['DIDS-MFL', 'MLP', 'GraphSAGE', 'RandomForest']):
    values = [
        results[model]['binary']['accuracy'],
        results[model]['binary']['f1_macro'],
        results[model]['multiclass']['f1_macro'],
        results[model]['multiclass']['accuracy'],
        results[model]['multiclass'].get('f1_weighted', 0)
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('Model Capability Comparison', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('report/images/radar_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: radar_comparison.png")

print("\nAll figures generated successfully!")
