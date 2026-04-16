#!/usr/bin/env python3
"""
Analyze Hartree-Fock method calculation tasks from paper 2111.01152.
Parse YAML scoring data, compute statistics, generate figures, and derive Hamiltonians.
"""

import yaml
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_002_20260415_215045'
YAML_PATH = os.path.join(WORKSPACE, 'data/2111.01152/2111.01152.yaml')
OUTPUTS_DIR = os.path.join(WORKSPACE, 'outputs')
IMAGES_DIR = os.path.join(WORKSPACE, 'report/images')
CODE_DIR = os.path.join(WORKSPACE, 'code')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load YAML
with open(YAML_PATH, 'r') as f:
    data = yaml.safe_load(f)

tasks = [item for item in data if 'task' in item]

# ============================================================
# 1. Parse all data into structured format
# ============================================================

task_records = []
placeholder_records = []

for i, task in enumerate(tasks):
    task_name = task.get('task', 'N/A')
    answer = task.get('answer', '')
    score = task.get('score', {})
    
    # Extract numeric scores
    score_vals = {}
    score_comments = {}
    for k, v in score.items():
        if isinstance(v, (int, float)):
            score_vals[k] = v
        elif isinstance(v, str):
            if v.startswith('#'):
                score_comments[k] = v.lstrip('#').strip()
            else:
                score_comments[k] = v
    
    task_record = {
        'task_idx': i,
        'task_name': task_name,
        'answer': answer,
        'scores': score_vals,
        'comments': score_comments,
        'avg_score': np.mean(list(score_vals.values())) if score_vals else 0,
        'total_score': sum(score_vals.values()) if score_vals else 0,
        'max_possible': len(score_vals) * 2 if score_vals else 0,
    }
    task_records.append(task_record)
    
    # Parse placeholder-level scores
    placeholder = task.get('placeholder', {})
    for field_name, field_data in placeholder.items():
        if isinstance(field_data, dict) and 'score' in field_data:
            score_dict = field_data['score']
            llm_val = field_data.get('LLM', '')
            human_val = field_data.get('human', '')
            
            for evaluator, score_val in score_dict.items():
                if evaluator.startswith('#') or not isinstance(score_val, (int, float)):
                    continue
                placeholder_records.append({
                    'task_idx': i,
                    'task_name': task_name,
                    'field': field_name,
                    'evaluator': evaluator,
                    'score': score_val,
                    'llm_value': str(llm_val)[:200] if llm_val else '',
                    'human_value': str(human_val)[:200] if human_val else '',
                })

# Save structured outputs
with open(os.path.join(OUTPUTS_DIR, 'task_records.json'), 'w') as f:
    json.dump(task_records, f, indent=2, default=str)

with open(os.path.join(OUTPUTS_DIR, 'placeholder_records.json'), 'w') as f:
    json.dump(placeholder_records, f, indent=2)

print(f"Saved {len(task_records)} task records and {len(placeholder_records)} placeholder records")

# ============================================================
# 2. Compute aggregate statistics
# ============================================================

# Per-task average scores by category
categories = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']

category_stats = {}
for cat in categories:
    vals = [t['scores'].get(cat, None) for t in task_records if cat in t['scores']]
    if vals:
        category_stats[cat] = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'count': len(vals),
            'values': vals,
        }

# Per-evaluator placeholder-level stats
evaluator_stats = {}
for ev in ['Haining', 'Will', 'Yasaman']:
    ev_scores = [p['score'] for p in placeholder_records if p['evaluator'] == ev]
    if ev_scores:
        evaluator_stats[ev] = {
            'mean': np.mean(ev_scores),
            'std': np.std(ev_scores),
            'count': len(ev_scores),
            'scores': ev_scores,
        }

# Overall statistics
all_task_scores = []
for t in task_records:
    for cat in categories:
        if cat in t['scores']:
            all_task_scores.append(t['scores'][cat])

overall_mean = np.mean(all_task_scores)
overall_std = np.std(all_task_scores)
print(f"Overall task-level score: {overall_mean:.3f} ± {overall_std:.3f} (scale 0-2)")

stats_summary = {
    'overall_mean': overall_mean,
    'overall_std': overall_std,
    'num_tasks': len(task_records),
    'num_placeholder_scores': len(placeholder_records),
    'category_stats': category_stats,
    'evaluator_stats': evaluator_stats,
}

with open(os.path.join(OUTPUTS_DIR, 'stats_summary.json'), 'w') as f:
    json.dump(stats_summary, f, indent=2, default=str)

# ============================================================
# 3. Derive the correct Hartree-Fock Hamiltonian
# ============================================================

hamiltonian_derivation = {
    'paper_info': {
        'arxiv_id': '2111.01152',
        'title': 'Topological Phases in AB-stacked MoTe2/WSe2',
        'authors': 'Haining Pan, Ming Xie, Fengcheng Wu, Sankar Das Sarma',
        'system': 'AB-stacked MoTe2/WSe2 moiré heterobilayer',
    },
    'single_particle_hamiltonian': {
        'description': 'Valley-dependent continuum Hamiltonian for AB-stacked MoTe2/WSe2',
        'formula': r'H_\tau = \begin{pmatrix} -\frac{\hbar^2 k^2}{2m_b} + \Delta_b(r) & \Delta_{T,\tau}(r) \\ \Delta_{T,\tau}^\dag(r) & -\frac{\hbar^2 (k-\tau\kappa)^2}{2m_t} + \Delta_t(r) + V_{zt} \end{pmatrix}',
        'parameters': {
            'tau': '\pm 1 for \pm K valleys',
            'kappa': '4\pi/(3a_M) (1,0) - moiré BZ corner',
            'm_b': '0.65 m_e (MoTe2 effective mass)',
            'm_t': '0.35 m_e (WSe2 effective mass)',
            'V_zt': 'Band offset (tunable by displacement field)',
        },
    },
    'intralayer_potential': {
        'bottom_layer': r'\Delta_b(r) = 2V_b \sum_{j=1,3,5} \cos(g_j \cdot r + \psi_b)',
        'top_layer': r'\Delta_t(r) = 0 (or V_{zt} as constant offset)',
    },
    'interlayer_tunneling': {
        'plus_K': r'\Delta_{T,+}(r) = w(1 + \omega e^{ig_2 \cdot r} + \omega^2 e^{ig_3 \cdot r})',
        'minus_K': r'\Delta_{T,-}(r) = -w(1 + \omega^{-1} e^{-ig_2 \cdot r} + \omega^{-2} e^{-ig_3 \cdot r})',
        'omega': 'e^{i2\pi/3}',
    },
    'second_quantized_form': {
        'real_space': r'\hat{H}_0 = \sum_\tau \int d^2r \Psi_\tau^\dag(r) H_\tau \Psi_\tau(r)',
        'momentum_space': r'\hat{H}_0 = \sum_{k_\alpha,k_\beta,l_\alpha,l_\beta,\tau} h_{k_\alpha l_\alpha, k_\beta l_\beta}^{(\tau)} c_{k_\alpha,l_\alpha,\tau}^\dag c_{k_\beta,l_\beta,\tau}',
    },
    'particle_hole_transform': {
        'definition': r'b_{k,l,\tau} = c_{k,l,\tau}^\dag',
        'hole_hamiltonian': r'\hat{H}_0 = \sum_\tau Tr h^{(\tau)} - \sum_{k_\alpha,k_\beta,l_\alpha,l_\beta,\tau} [h^{(\tau)}]^T_{k_\alpha l_\alpha, k_\beta l_\beta} b_{k_\alpha,l_\alpha,\tau}^\dag b_{k_\beta,l_\beta,\tau}',
    },
    'interaction_hamiltonian': {
        'momentum_space': r'\hat{H}_{int} = \frac{1}{2A} \sum_{k_\alpha,k_\beta,k_\gamma,k_\delta,l_\alpha,l_\beta,\tau_\alpha,\tau_\beta} V(k_\alpha-k_\delta) b_{k_\alpha,l_\alpha,\tau_\alpha}^\dag b_{k_\beta,l_\beta,\tau_\beta}^\dag b_{k_\gamma,l_\beta,\tau_\beta} b_{k_\delta,l_\alpha,\tau_\alpha} \delta_{k_\alpha+k_\beta,k_\delta+k_\gamma}',
        'coulomb_form': r'V(k) = 2\pi e^2 \tanh(|k|d)/(\epsilon |k|)',
    },
    'hartree_fock_hamiltonian': {
        'full_HF': r'\hat{H}^{HF} = \hat{H}_1 + \hat{H}_{int}^{HF}',
        'single_particle_in_hole_basis': r'\hat{H}_1 = \sum_{k_\alpha,k_\beta,l_\alpha,l_\beta,\tau} \tilde{h}^{(\tau)}_{k_\alpha l_\alpha, k_\beta l_\beta} b_{k_\alpha,l_\alpha,\tau}^\dag b_{k_\beta,l_\beta,\tau}',
        'HF_interaction': r'\hat{H}_{int}^{HF} = \frac{1}{A} \sum_{k_\alpha,k_\beta,k_\gamma,k_\delta,l_\alpha,l_\beta,\tau_\alpha,\tau_\beta} V(k_\alpha-k_\delta) [\langle b_{k_\alpha,l_\alpha,\tau_\alpha}^\dag b_{k_\delta,l_\alpha,\tau_\alpha} \rangle b_{k_\beta,l_\beta,\tau_\beta}^\dag b_{k_\gamma,l_\beta,\tau_\beta} - \langle b_{k_\alpha,l_\alpha,\tau_\alpha}^\dag b_{k_\gamma,l_\beta,\tau_\beta} \rangle b_{k_\beta,l_\beta,\tau_\beta}^\dag b_{k_\delta,l_\alpha,\tau_\alpha}] \delta_{k_\alpha+k_\beta,k_\delta+k_\gamma}',
        'hartree_term': 'Direct (Hartree) term with same-valley/layer expectation values',
        'fock_term': 'Exchange (Fock) term with cross-valley/layer expectation values',
    },
}

with open(os.path.join(OUTPUTS_DIR, 'hamiltonian_derivation.json'), 'w') as f:
    json.dump(hamiltonian_derivation, f, indent=2, default=str)

print("Hamiltonian derivation saved")

# ============================================================
# 4. Generate Figures
# ============================================================

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# Figure 1: Task-level score distribution across steps
fig, ax = plt.subplots(figsize=(14, 6))
task_names_short = [f"Step {i}" for i in range(len(task_records))]
x_pos = np.arange(len(task_records))

# Plot stacked bar for each category
bottom = np.zeros(len(task_records))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#607D8B']
for ci, cat in enumerate(categories):
    vals = [task_records[i]['scores'].get(cat, 0) for i in range(len(task_records))]
    ax.bar(x_pos, vals, bottom=bottom, label=cat, color=colors[ci], width=0.7)
    bottom += np.array(vals)

ax.set_xticks(x_pos)
ax.set_xticklabels(task_names_short, rotation=45, ha='right')
ax.set_ylabel('Cumulative Score (max=12 per step)')
ax.set_title('Step-by-Step Score Distribution by Category\n(Paper 2111.01152: AB-stacked MoTe₂/WSe₂)')
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(0, 13)

# Add total score annotations
for i in range(len(task_records)):
    total = task_records[i]['total_score']
    ax.annotate(f'{total}/12', (x_pos[i], total+0.3), ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'step_score_distribution.png'))
plt.close()
print("Figure 1 saved: step_score_distribution.png")

# Figure 2: Per-category average scores
fig, ax = plt.subplots(figsize=(8, 5))
cat_means = [category_stats[cat]['mean'] for cat in categories]
cat_stds = [category_stats[cat]['std'] for cat in categories]
bars = ax.bar(categories, cat_means, yerr=cat_stds, color=colors, capsize=5)
ax.set_ylabel('Average Score (scale 0-2)')
ax.set_title('Average Score per Evaluation Category\n(Paper 2111.01152)')
ax.set_ylim(0, 2.3)
for bar, mean in zip(bars, cat_means):
    ax.annotate(f'{mean:.2f}', (bar.get_x() + bar.get_width()/2, mean+0.05), 
                ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'category_avg_scores.png'))
plt.close()
print("Figure 2 saved: category_avg_scores.png")

# Figure 3: Evaluator agreement heatmap
fig, ax = plt.subplots(figsize=(10, 8))
# Build matrix: rows=placeholder fields, columns=evaluators
# Use only fields with all 3 evaluators scored
field_eval_matrix = {}
for p in placeholder_records:
    key = f"Step{p['task_idx']}_{p['field']}"
    if key not in field_eval_matrix:
        field_eval_matrix[key] = {}
    field_eval_matrix[key][p['evaluator']] = p['score']

# Filter to fields with at least 2 evaluators
consistent_fields = {k: v for k, v in field_eval_matrix.items() if len(v) >= 2}
if consistent_fields:
    field_names = list(consistent_fields.keys())[:30]  # Limit to 30 for readability
    evaluators = ['Haining', 'Will', 'Yasaman']
    matrix = np.zeros((len(field_names), len(evaluators)))
    for fi, fn in enumerate(field_names):
        for ei, ev in enumerate(evaluators):
            matrix[fi, ei] = consistent_fields[fn].get(ev, np.nan)
    
    sns.heatmap(matrix, xticklabels=evaluators, yticklabels=field_names,
                cmap='RdYlGn', vmin=0, vmax=2, ax=ax, annot=True, fmt='.0f')
    ax.set_title('Evaluator Scores per Placeholder Field\n(Top 30 Fields)')
    ax.set_ylabel('Placeholder Field')
    ax.set_xlabel('Evaluator')
else:
    ax.text(0.5, 0.5, 'No consistent fields found', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'evaluator_heatmap.png'))
plt.close()
print("Figure 3 saved: evaluator_heatmap.png")

# Figure 4: Score trajectory along HF derivation pipeline
fig, ax = plt.subplots(figsize=(14, 5))
pipeline_steps = [
    'Kinetic Ham.', 'Define Kinetic Terms', 'Potential Ham.', 'Define Potential Terms',
    'Second Quantized (matrix)', 'Second Quantized (summation)', 'Real→Momentum Space',
    'Particle-Hole Transform', 'Simplify Hole Basis', 'Interaction Ham.',
    'Wick\'s Theorem', 'Extract Quadratic', 'Combine H/F Terms',
    'Reduce Hartree Momentum', 'Reduce Fock Momentum', 'Combine Final H/F'
]

avg_scores = [t['avg_score'] for t in task_records]
in_paper_scores = [t['scores'].get('in_paper', 0) for t in task_records]
accuracy_scores = [t['scores'].get('final_answer_accuracy', 0) for t in task_records]

ax.plot(range(len(avg_scores)), avg_scores, 'o-', label='Average Overall Score', linewidth=2, markersize=8)
ax.plot(range(len(in_paper_scores)), in_paper_scores, 's--', label='In-Paper Score', linewidth=1.5, markersize=6)
ax.plot(range(len(accuracy_scores)), accuracy_scores, '^:', label='Final Answer Accuracy', linewidth=1.5, markersize=6)

ax.set_xticks(range(len(pipeline_steps)))
ax.set_xticklabels(pipeline_steps, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Score (scale 0-2)')
ax.set_title('Score Trajectory Along Hartree-Fock Derivation Pipeline\n(Paper 2111.01152)')
ax.legend()
ax.set_ylim(-0.1, 2.3)
ax.axhline(y=2, color='gray', linestyle='--', alpha=0.3, label='Perfect score')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'score_trajectory.png'))
plt.close()
print("Figure 4 saved: score_trajectory.png")

# Figure 5: Evaluator comparison (average per evaluator)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: average score per evaluator
ev_names = list(evaluator_stats.keys())
ev_means = [evaluator_stats[ev]['mean'] for ev in ev_names]
ev_stds = [evaluator_stats[ev]['std'] for ev in ev_names]
axes[0].bar(ev_names, ev_means, yerr=ev_stds, color=['#2196F3', '#4CAF50', '#FF9800'], capsize=5)
axes[0].set_ylabel('Average Placeholder-Level Score (0-2)')
axes[0].set_title('Average Score per Evaluator')
axes[0].set_ylim(0, 2.3)
for i, (name, mean) in enumerate(zip(ev_names, ev_means)):
    axes[0].annotate(f'{mean:.2f}', (i, mean+0.05), ha='center')

# Right: score distribution histograms per evaluator
for ev, color in zip(ev_names, ['#2196F3', '#4CAF50', '#FF9800']):
    scores = evaluator_stats[ev]['scores']
    axes[1].hist(scores, bins=np.arange(-0.25, 2.75, 0.5), alpha=0.6, label=ev, color=color, edgecolor='black')
axes[1].set_xlabel('Score')
axes[1].set_ylabel('Count')
axes[1].set_title('Score Distribution per Evaluator')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'evaluator_comparison.png'))
plt.close()
print("Figure 5 saved: evaluator_comparison.png")

# Figure 6: Physics derivation flow diagram (text-based visualization)
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# Draw boxes for each step
step_positions = [
    (1, 7), (4, 7), (7, 7), (10, 7),  # Row 1: Kinetic/Potential
    (1, 5.5), (4, 5.5), (7, 5.5),     # Row 2: Second quantized, Fourier, P-H
    (10, 5.5),                          # Simplify
    (1, 4), (4, 4), (7, 4),            # Row 3: Interaction, Wick, Extract
    (10, 4),                            # Combine H/F
    (1, 2.5), (4, 2.5), (7, 2.5),      # Row 4: Reduce momentum
    (10, 2.5),                          # Final combine
]

short_names = [
    'Kinetic Ham.', 'Define Kin.', 'Potential Ham.', 'Define Pot.',
    '2nd Quant (mat)', '2nd Quant (sum)', 'Real→Mom.',
    'P-H Transform', 'Simplify Hole', 'Interaction',
    'Wick\'s Thm', 'Extract Quad', 'Combine H/F',
    'Reduce Hartree', 'Reduce Fock', 'Final H/F'
]

score_colors = []
for t in task_records:
    avg = t['avg_score']
    if avg >= 1.8:
        score_colors.append('#4CAF50')  # Green
    elif avg >= 1.5:
        score_colors.append('#FF9800')  # Orange
    elif avg >= 1.0:
        score_colors.append('#FFC107')  # Yellow
    else:
        score_colors.append('#F44336')  # Red

for i, (pos, name, color) in enumerate(zip(step_positions, short_names, score_colors)):
    x, y = pos
    rect = plt.Rectangle((x-0.8, y-0.4), 1.6, 0.8, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y+0.15, name, ha='center', va='center', fontsize=7, weight='bold')
    ax.text(x, y-0.15, f'{task_records[i]["avg_score"]:.2f}/2.0', ha='center', va='center', fontsize=6)

# Draw arrows connecting steps
arrow_style = dict(arrowstyle='->', color='gray', lw=1)
for i in range(len(step_positions)-1):
    x1, y1 = step_positions[i]
    x2, y2 = step_positions[i+1]
    if abs(y1-y2) < 1:  # Same row
        ax.annotate('', xy=(x2-0.8, y2), xytext=(x1+0.8, y1), arrowprops=arrow_style)

ax.set_title('Hartree-Fock Derivation Pipeline with Step Scores\n(Green≥1.8, Orange≥1.5, Yellow≥1.0, Red<1.0)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'derivation_pipeline.png'))
plt.close()
print("Figure 6 saved: derivation_pipeline.png")

print("\n=== All figures and outputs generated successfully ===")