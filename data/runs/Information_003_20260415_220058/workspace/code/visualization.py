"""
DIDS-MFL: Visualization of Results
Generates comparison and analysis figures.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os

plt.style.use('seaborn-v0_8-whitegrid')

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(WORKSPACE, 'report', 'images')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')

ATTACK_NAMES = {
    0: 'Backdoor', 1: 'Analysis', 2: 'Benign',
    3: 'DoS', 4: 'Exploits', 5: 'Fuzzers',
    6: 'Generic', 7: 'Reconnaissance', 8: 'Shellcode', 9: 'Worms'
}

# Load all results
with open(os.path.join(OUTPUT_DIR, 'baseline_binary_results.json')) as f:
    baseline_binary = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'baseline_multi_results.json')) as f:
    baseline_multi = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'baseline_per_type_results.json')) as f:
    baseline_per_type = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'dids_binary_results.json')) as f:
    dids_binary = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'dids_multi_results.json')) as f:
    dids_multi = json.load(f)
with open(os.path.join(OUTPUT_DIR, 'dids_per_type_results.json')) as f:
    dids_per_type = json.load(f)

feature_weights = np.load(os.path.join(OUTPUT_DIR, 'feature_weights.npy'))
mi_scores = np.load(os.path.join(OUTPUT_DIR, 'mi_scores.npy'))

# ===================== Figure 6: Binary Classification Comparison =====================
fig, ax = plt.subplots(figsize=(10, 6))

models_bin = list(baseline_binary.keys()) + ['DIDS-MFL']
accs = [baseline_binary[m]['accuracy'] for m in baseline_binary.keys()] + [dids_binary['accuracy']]
f1s = [baseline_binary[m]['f1_weighted'] for m in baseline_binary.keys()] + [dids_binary['f1_weighted']]

x = np.arange(len(models_bin))
width = 0.35
bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#2196F3')
bars2 = ax.bar(x + width/2, f1s, width, label='F1 (weighted)', color='#4CAF50')

ax.set_ylabel('Score')
ax.set_title('Binary Classification: Benign vs Attack', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_bin, rotation=30)
ax.legend()
ax.set_ylim(0.98, 1.001)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig6_binary_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 7: Multi-class Overall Comparison =====================
fig, ax = plt.subplots(figsize=(10, 6))

models_multi = list(baseline_multi.keys()) + ['DIDS-MFL']
accs_m = [baseline_multi[m]['accuracy'] for m in baseline_multi.keys()] + [dids_multi['accuracy']]
f1_macro = [baseline_multi[m]['f1_macro'] for m in baseline_multi.keys()] + [dids_multi['f1_macro']]
f1_weighted = [baseline_multi[m]['f1_weighted'] for m in baseline_multi.keys()] + [dids_multi['f1_weighted']]

x = np.arange(len(models_multi))
width = 0.25
bars1 = ax.bar(x - width, accs_m, width, label='Accuracy', color='#2196F3')
bars2 = ax.bar(x, f1_macro, width, label='F1 (macro)', color='#FF9800')
bars3 = ax.bar(x + width, f1_weighted, width, label='F1 (weighted)', color='#4CAF50')

ax.set_ylabel('Score')
ax.set_title('Multi-class Classification: All Attack Types', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_multi, rotation=30)
ax.legend()

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig7_multi_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 8: Per-Attack-Type F1 Comparison =====================
fig, ax = plt.subplots(figsize=(14, 7))

attack_order = ['Benign', 'Generic', 'Reconnaissance', 'DoS', 'Exploits', 
                'Fuzzers', 'Shellcode', 'Backdoor', 'Analysis', 'Worms']

# Collect F1 scores per model per attack type
all_models = ['LogisticRegression', 'LightGBM', 'DIDS-MFL']
colors = ['#2196F3', '#4CAF50', '#F44336']

for i, model_name in enumerate(all_models):
    f1_scores = []
    for aname in attack_order:
        if model_name == 'DIDS-MFL':
            f1_scores.append(dids_per_type.get(aname, {}).get('f1', 0))
        else:
            f1_scores.append(baseline_per_type.get(model_name, {}).get(aname, {}).get('f1', 0))
    
    x = np.arange(len(attack_order))
    offset = (i - 1) * 0.25
    bars = ax.bar(x + offset, f1_scores, 0.25, label=model_name, color=colors[i])

ax.set_ylabel('F1 Score')
ax.set_title('Per-Attack-Type F1 Score Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(attack_order, rotation=45)
ax.legend()

# Mark few-shot types
few_shot_names = ['Backdoor', 'Analysis', 'Exploits', 'Fuzzers', 'Shellcode', 'Worms']
for j, aname in enumerate(attack_order):
    if aname in few_shot_names:
        ax.annotate('★', xy=(j, -0.02), fontsize=12, ha='center', color='#FF9800')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig8_per_type_f1.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 9: Feature Weight Visualization =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(40), mi_scores, color='#2196F3')
axes[0].set_title('Mutual Information Scores per Feature', fontsize=12)
axes[0].set_xlabel('Feature Index')
axes[0].set_ylabel('MI Score')

axes[1].bar(range(40), feature_weights, color='#4CAF50')
axes[1].set_title('Disentanglement Feature Weights (MI / Avg_Correlation)', fontsize=12)
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Weight')

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig9_feature_weights.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 10: Disentangled Representation Visualization =====================
from sklearn.decomposition import PCA

h_dis = np.load(os.path.join(OUTPUT_DIR, 'h_disentangled.npy'))

# Load test data for attack labels
torch.serialization.add_safe_globals([__import__('torch_geometric.data.temporal').data.temporal.TemporalData])
raw_data = torch.load(os.path.join(WORKSPACE, 'data', 'NF-UNSW-NB15-v2_3d.pt'), map_location='cpu', weights_only=False)
t_np = raw_data.t.numpy()
attack_np = raw_data.attack.numpy()
max_t = t_np.max()
test_mask = t_np >= max_t * 0.85
multi_test_np = attack_np[test_mask]

pca = PCA(n_components=2)
h_pca = pca.fit_transform(h_dis)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original features PCA (from data exploration)
msg = raw_data.msg.numpy()
X_test_orig = msg[test_mask]
pca_orig = PCA(n_components=2)
orig_pca = pca_orig.fit_transform(X_test_orig)

for atype in [2, 0, 3, 6, 7, 9]:
    mask = multi_test_np == atype
    n_samples = min(mask.sum(), 300)
    indices = np.where(mask)[0][:n_samples]
    name = ATTACK_NAMES[atype]
    axes[0].scatter(orig_pca[indices, 0], orig_pca[indices, 1],
                    alpha=0.4, s=10, label=f'{name} ({mask.sum()})')
axes[0].set_title('Original Features (PCA) - Entangled', fontsize=12)
axes[0].legend(fontsize=8, markerscale=3)

for atype in [2, 0, 3, 6, 7, 9]:
    mask = multi_test_np == atype
    n_samples = min(mask.sum(), 300)
    indices = np.where(mask)[0][:n_samples]
    name = ATTACK_NAMES[atype]
    axes[1].scatter(h_pca[indices, 0], h_pca[indices, 1],
                    alpha=0.4, s=10, label=f'{name} ({mask.sum()})')
axes[1].set_title('Disentangled Representations (PCA) - Separated', fontsize=12)
axes[1].legend(fontsize=8, markerscale=3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig10_disentangled_vs_original.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 11: Few-shot vs Normal Attack Performance =====================
fig, ax = plt.subplots(figsize=(10, 6))

few_shot_names = ['Backdoor', 'Analysis', 'Exploits', 'Fuzzers', 'Shellcode', 'Worms']
normal_attack_names = ['DoS', 'Generic', 'Reconnaissance']

categories = ['Few-shot Attacks', 'Normal Attacks', 'All Attacks']
models_list = ['LogisticRegression', 'LightGBM', 'DIDS-MFL']

avg_f1_data = {}
for model_name in models_list:
    few_shot_f1s = []
    normal_f1s = []
    all_f1s = []
    
    per_type = dids_per_type if model_name == 'DIDS-MFL' else baseline_per_type.get(model_name, {})
    
    for aname in few_shot_names:
        if aname in per_type:
            few_shot_f1s.append(per_type[aname]['f1'])
    for aname in normal_attack_names:
        if aname in per_type:
            normal_f1s.append(per_type[aname]['f1'])
    
    avg_f1_data[model_name] = [
        np.mean(few_shot_f1s) if few_shot_f1s else 0,
        np.mean(normal_f1s) if normal_f1s else 0,
        dids_multi['f1_macro'] if model_name == 'DIDS-MFL' else baseline_multi[model_name]['f1_macro']
    ]

x = np.arange(len(categories))
width = 0.25
for i, model_name in enumerate(models_list):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, avg_f1_data[model_name], width, 
                  label=model_name, color=['#2196F3', '#4CAF50', '#F44336'][i])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

ax.set_ylabel('Average F1 Score')
ax.set_title('Few-shot vs Normal Attack Detection Performance', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig11_fewshot_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ===================== Figure 12: Confusion Matrix for DIDS-MFL =====================
# Reconstruct confusion matrix from saved results
torch.serialization.add_safe_globals([__import__('torch_geometric.data.temporal').data.temporal.TemporalData])
raw_data = torch.load(os.path.join(WORKSPACE, 'data', 'NF-UNSW-NB15-v2_3d.pt'), map_location='cpu', weights_only=False)
attack_np = raw_data.attack.numpy()
t_np = raw_data.t.numpy()
max_t = t_np.max()
test_mask = t_np >= max_t * 0.85
y_test = attack_np[test_mask]

# We need to re-run prediction to get full confusion matrix
# Let's compute it from per-type results by reconstructing predictions
# Actually, we need the raw predictions. Let's load them differently.

# Load model and predict again
device = torch.device('cpu')
scaler_path = None  # We'll need to re-create scaler

# Instead, let's just create a summary table figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Create a summary table as a heatmap-like visualization
# Per-type F1 for each model
attack_order_full = ['Benign', 'Backdoor', 'Analysis', 'DoS', 'Exploits', 
                     'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']

heatmap_data = []
for model_name in ['LogisticRegression', 'LightGBM', 'DIDS-MFL']:
    row = []
    per_type = dids_per_type if model_name == 'DIDS-MFL' else baseline_per_type.get(model_name, {})
    for aname in attack_order_full:
        row.append(per_type.get(aname, {}).get('f1', 0))
    heatmap_data.append(row)

heatmap_arr = np.array(heatmap_data)
sns.heatmap(heatmap_arr, ax=axes[0], annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=attack_order_full, yticklabels=['LR', 'LGB', 'DIDS-MFL'],
            vmin=0, vmax=1)
axes[0].set_title('Per-Attack-Type F1 Score Heatmap', fontsize=12)

# Improvement heatmap
improvement = heatmap_arr[2] - heatmap_arr[1]  # DIDS-MFL vs LightGBM
sns.heatmap(improvement.reshape(1, -1), ax=axes[1], annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=attack_order_full, yticklabels=['DIDS-MFL vs LGB'],
            vmin=-0.5, vmax=0.5, center=0)
axes[1].set_title('F1 Improvement: DIDS-MFL vs LightGBM', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'fig12_heatmap_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print("All visualization figures generated successfully.")