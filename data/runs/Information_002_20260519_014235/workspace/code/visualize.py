"""
Generate figures for LLM Hartree-Fock performance analysis.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('outputs/task_scores.csv')
score_dimensions = ['in_paper', 'prompt_quality', 'follow_instructions', 
                    'physics_logic', 'math_derivation', 'final_answer_accuracy']

# Clean task names for plotting
df['task_short'] = df['task'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Figure 1: Heatmap of task scores across dimensions
fig, ax = plt.subplots(figsize=(10, 10))
heatmap_data = df[score_dimensions].values
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
ax.set_xticks(np.arange(len(score_dimensions)))
ax.set_yticks(np.arange(len(df)))
ax.set_xticklabels([d.replace('_', '\n') for d in score_dimensions], fontsize=9)
ax.set_yticklabels(df['task_short'], fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Add text annotations
for i in range(len(df)):
    for j in range(len(score_dimensions)):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            text = ax.text(j, i, f'{val:.0f}', ha="center", va="center", 
                          color="black" if val > 1 else "white", fontsize=9, fontweight='bold')
ax.set_title('LLM Performance Heatmap: Task Scores Across Evaluation Dimensions', fontsize=12, fontweight='bold')
fig.colorbar(im, ax=ax, label='Score (0-2)', shrink=0.6)
plt.tight_layout()
plt.savefig('report/images/fig1_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig1_heatmap.png")

# Figure 2: Average score per dimension
fig, ax = plt.subplots(figsize=(9, 5))
means = df[score_dimensions].mean().sort_values(ascending=True)
colors = plt.cm.RdYlGn(means / 2)
bars = ax.barh(range(len(means)), means.values, color=colors, edgecolor='black')
ax.set_yticks(range(len(means)))
ax.set_yticklabels([m.replace('_', ' ').title() for m in means.index], fontsize=10)
ax.set_xlim(0, 2)
ax.set_xlabel('Mean Score (0-2)', fontsize=11)
ax.set_title('Average LLM Performance by Evaluation Dimension', fontsize=12, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, means.values)):
    ax.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig2_dimension_scores.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig2_dimension_scores.png")

# Figure 3: Average score per task
fig, ax = plt.subplots(figsize=(10, 7))
task_means = df[score_dimensions].mean(axis=1)
task_names = df['task_short']
# Sort by mean score
sorted_idx = task_means.argsort()
task_means_sorted = task_means.iloc[sorted_idx]
task_names_sorted = task_names.iloc[sorted_idx]
colors = plt.cm.RdYlGn(task_means_sorted / 2)
bars = ax.barh(range(len(task_means_sorted)), task_means_sorted.values, color=colors, edgecolor='black')
ax.set_yticks(range(len(task_means_sorted)))
ax.set_yticklabels(task_names_sorted, fontsize=8)
ax.set_xlim(0, 2)
ax.set_xlabel('Mean Score (0-2)', fontsize=11)
ax.set_title('Average LLM Performance by Calculation Task', fontsize=12, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, task_means_sorted.values)):
    ax.text(val + 0.03, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig3_task_scores.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_task_scores.png")

# Figure 4: Performance by calculation phase
categories = {
    'Hamiltonian Construction': [0, 1, 2, 3],
    'Second Quantization': [4, 5],
    'Space Transformation': [6],
    'Particle-Hole': [7, 8],
    'Interaction & Wick': [9, 10, 11],
    'Hartree-Fock Reduction': [12, 13, 14, 15]
}
phase_data = []
for phase, indices in categories.items():
    scores = df.iloc[indices][score_dimensions].values.flatten()
    scores = scores[~np.isnan(scores)]
    phase_data.append({
        'phase': phase,
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    })
phase_df = pd.DataFrame(phase_data)
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlGn(phase_df['mean'] / 2)
bars = ax.bar(range(len(phase_df)), phase_df['mean'], yerr=phase_df['std'], 
              color=colors, edgecolor='black', capsize=5, alpha=0.85)
ax.set_xticks(range(len(phase_df)))
ax.set_xticklabels(phase_df['phase'], rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, 2)
ax.set_ylabel('Mean Score ± Std (0-2)', fontsize=11)
ax.set_title('LLM Performance by Calculation Phase', fontsize=12, fontweight='bold')
for i, (bar, val, std) in enumerate(zip(bars, phase_df['mean'], phase_df['std'])):
    ax.text(i, val + std + 0.05, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig4_phase_scores.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig4_phase_scores.png")

# Figure 5: Inter-annotator agreement (placeholder-level)
df_ph = pd.read_csv('outputs/placeholder_scores.csv')
annotator_pivot = df_ph.pivot_table(index=['task', 'placeholder'], columns='annotator', values='score')
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pairs = [('Haining', 'Will'), ('Haining', 'Yasaman'), ('Will', 'Yasaman')]
for ax, (a1, a2) in zip(axes, pairs):
    x = annotator_pivot[a1].dropna()
    y = annotator_pivot[a2].dropna()
    common = pd.concat([x, y], axis=1).dropna()
    if len(common) > 0:
        ax.scatter(common[a1], common[a2], alpha=0.6, edgecolors='black')
        ax.plot([0, 2], [0, 2], 'r--', lw=2)
        r, p = pearsonr(common[a1], common[a2])
        ax.set_title(f'{a1} vs {a2}\nr={r:.3f}, p={p:.3e}', fontsize=10)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 2.1)
    ax.set_xlabel(a1, fontsize=9)
    ax.set_ylabel(a2, fontsize=9)
    ax.set_aspect('equal')
fig.suptitle('Inter-Annotator Agreement on Placeholder-Level Scores', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('report/images/fig5_interannotator.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig5_interannotator.png")

# Figure 6: Stacked area / radar for dimension profiles of weak vs strong tasks
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
weak_tasks = df[df[score_dimensions].mean(axis=1) < 1.7]['task_short'].tolist()
strong_tasks = df[df[score_dimensions].mean(axis=1) >= 1.9]['task_short'].tolist()
weak_scores = df[df['task_short'].isin(weak_tasks)][score_dimensions].mean()
strong_scores = df[df['task_short'].isin(strong_tasks)][score_dimensions].mean()
x = np.arange(len(score_dimensions))
width = 0.35
axes[0].bar(x - width/2, weak_scores, width, label='Weak Tasks (mean<1.7)', color='salmon', edgecolor='black')
axes[0].bar(x + width/2, strong_scores, width, label='Strong Tasks (mean≥1.9)', color='lightgreen', edgecolor='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels([d.replace('_', '\n') for d in score_dimensions], fontsize=8)
axes[0].set_ylim(0, 2)
axes[0].set_ylabel('Mean Score', fontsize=10)
axes[0].set_title('Dimension Profiles: Weak vs Strong Tasks', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)

# Task progression line
axes[1].plot(range(len(df)), df[score_dimensions].mean(axis=1), marker='o', linewidth=2, markersize=6, color='steelblue')
axes[1].axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Pass threshold')
axes[1].set_xticks(range(len(df)))
axes[1].set_xticklabels([f'T{i+1}' for i in range(len(df))], fontsize=8)
axes[1].set_ylim(0.5, 2.2)
axes[1].set_ylabel('Mean Score', fontsize=10)
axes[1].set_xlabel('Task Index (in pipeline order)', fontsize=10)
axes[1].set_title('Score Progression Through Calculation Pipeline', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('report/images/fig6_weak_strong_progression.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig6_weak_strong_progression.png")

# Figure 7: Error mode analysis - which dimensions fail most often
fig, ax = plt.subplots(figsize=(8, 5))
error_counts = {}
for dim in score_dimensions:
    counts = df[dim].value_counts().to_dict()
    error_counts[dim] = {
        'perfect (2)': counts.get(2, 0),
        'partial (1)': counts.get(1, 0),
        'failed (0)': counts.get(0, 0)
    }
error_df = pd.DataFrame(error_counts).T
error_df = error_df[['failed (0)', 'partial (1)', 'perfect (2)']]
error_df.plot(kind='barh', stacked=True, ax=ax, 
              color=['#d62728', '#ffbb78', '#2ca02c'], edgecolor='black')
ax.set_xlabel('Number of Tasks', fontsize=11)
ax.set_title('Score Distribution by Evaluation Dimension', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('report/images/fig7_error_modes.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig7_error_modes.png")

print("\nAll figures generated successfully!")
