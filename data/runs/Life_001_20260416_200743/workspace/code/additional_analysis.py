#!/usr/bin/env python3
"""
Additional analyses: Budget sweep, sim-specific comparisons, and deeper element analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
})
sns.set_style("whitegrid")

DATA_DIR = "data"
IMG_DIR = "report/images"
OUT_DIR = "outputs"

# Load data
vaccine_comp = pd.read_csv(f"{DATA_DIR}/vaccine.budget-10.minsum.adaptive.csv")
selected_elements = pd.read_csv(f"{DATA_DIR}/selected-vaccine-elements.budget-10.minsum.adaptive.csv")
cell_pop = pd.read_csv(f"{DATA_DIR}/cell-populations.csv")
final_resp = pd.read_csv(f"{DATA_DIR}/final-response-likelihoods.csv")
sim_resp = pd.read_csv(f"{DATA_DIR}/sim-specific-response-likelihoods.csv")

rep_scores = {}
for i in range(10):
    rep_scores[i] = pd.read_csv(f"{DATA_DIR}/vaccine-elements.scores.100-cells.10x.rep-{i}.csv")

# ============================================================
# Budget Sweep Analysis
# ============================================================
print("=== BUDGET SWEEP ANALYSIS ===")

# Simulate different budgets by selecting top-k elements
# Order elements by their contribution (mean response probability)
elem_order = []
for elem in vaccine_comp['peptide'].values:
    mean_resp = []
    for rep in range(10):
        df = rep_scores[rep]
        elem_data = df[df['vaccine_element'] == elem]
        mean_resp.append(elem_data['p_response'].mean())
    elem_order.append((elem, np.mean(mean_resp)))

elem_order.sort(key=lambda x: x[1], reverse=True)
print("Element ranking by mean response:")
for i, (e, v) in enumerate(elem_order):
    print(f"  {i+1}. {e}: {v:.4f}")

budget_results = []
for budget in range(1, 11):
    top_elements = [e for e, v in elem_order[:budget]]
    
    all_responses = []
    for rep in range(10):
        df = rep_scores[rep]
        df_sel = df[df['vaccine_element'].isin(top_elements)]
        cell_agg = df_sel.groupby('cell_id').agg(
            sum_log_p_no_response=('log_p_no_response', 'sum')
        ).reset_index()
        cell_agg['p_response'] = 1 - np.exp(cell_agg['sum_log_p_no_response'])
        all_responses.extend(cell_agg['p_response'].tolist())
    
    responses = np.array(all_responses)
    budget_results.append({
        'budget': budget,
        'elements': top_elements,
        'mean_response': responses.mean(),
        'median_response': np.median(responses),
        'coverage_50': (responses >= 0.5).mean(),
        'coverage_80': (responses >= 0.8).mean(),
        'coverage_90': (responses >= 0.9).mean()
    })

budget_df = pd.DataFrame(budget_results)
budget_df.to_csv(f"{OUT_DIR}/budget_sweep.csv", index=False)

print("\nBudget sweep results:")
for _, row in budget_df.iterrows():
    print(f"  Budget {int(row['budget'])}: mean={row['mean_response']:.4f}, "
          f"cov50={row['coverage_50']:.4f}, cov80={row['coverage_80']:.4f}, cov90={row['coverage_90']:.4f}")

# ============================================================
# Figure 10: Budget Sweep
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 10a: Mean response vs budget
axes[0].plot(budget_df['budget'], budget_df['mean_response'], 'o-', color='steelblue', linewidth=2, markersize=8)
axes[0].fill_between(budget_df['budget'], budget_df['mean_response'], alpha=0.2, color='steelblue')
axes[0].set_xlabel('Budget (Number of Vaccine Elements)')
axes[0].set_ylabel('Mean Response Probability')
axes[0].set_title('(A) Mean Response vs Budget')
axes[0].set_xticks(range(1, 11))

# 10b: Coverage at different thresholds vs budget
axes[1].plot(budget_df['budget'], budget_df['coverage_50'], 'o-', label='Coverage ≥ 0.5', linewidth=2)
axes[1].plot(budget_df['budget'], budget_df['coverage_80'], 's-', label='Coverage ≥ 0.8', linewidth=2)
axes[1].plot(budget_df['budget'], budget_df['coverage_90'], '^-', label='Coverage ≥ 0.9', linewidth=2)
axes[1].set_xlabel('Budget (Number of Vaccine Elements)')
axes[1].set_ylabel('Fraction of Cells Covered')
axes[1].set_title('(B) Coverage at Different Thresholds vs Budget')
axes[1].set_xticks(range(1, 11))
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig10_budget_sweep.png", bbox_inches='tight')
plt.close()
print("Saved fig10_budget_sweep.png")

# ============================================================
# Sim-Specific vs Adaptive Comparison
# ============================================================
print("\n=== SIM-SPECIFIC vs ADAPTIVE COMPARISON ===")

sim_resp['rep'] = sim_resp['vaccine'].apply(lambda x: int(x.split('rep-')[1]))
sim_resp['pop_rep'] = sim_resp['population'].apply(lambda x: int(x.split(', ')[1]))
final_resp['rep'] = final_resp['population'].apply(lambda x: int(x.split(', ')[1]))

# Compare: for each cell in each population, adaptive vs sim-specific response
comparison = []
for rep in range(10):
    adaptive_data = final_resp[final_resp['rep'] == rep].sort_values('name')
    sim_data = sim_resp[sim_resp['rep'] == rep].sort_values('name')
    
    if len(adaptive_data) == len(sim_data):
        merged = pd.merge(adaptive_data[['name', 'p_response']], 
                          sim_data[['name', 'p_response']], 
                          on='name', suffixes=('_adaptive', '_sim'))
        merged['rep'] = rep
        comparison.append(merged)

if comparison:
    comp_df = pd.concat(comparison, ignore_index=True)
    
    print(f"Adaptive mean: {comp_df['p_response_adaptive'].mean():.4f}")
    print(f"Sim-specific mean: {comp_df['p_response_sim'].mean():.4f}")
    print(f"Correlation: {comp_df['p_response_adaptive'].corr(comp_df['p_response_sim']):.4f}")
    
    # Figure 11: Adaptive vs Sim-specific scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(comp_df['p_response_sim'], comp_df['p_response_adaptive'], 
                    alpha=0.3, s=10, c='steelblue')
    axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
    axes[0].set_xlabel('Sim-Specific Response Probability')
    axes[0].set_ylabel('Adaptive Response Probability')
    axes[0].set_title('(A) Adaptive vs Sim-Specific Response')
    axes[0].legend()
    
    # Per-replicate comparison
    rep_comparison = comp_df.groupby('rep').agg(
        adaptive_mean=('p_response_adaptive', 'mean'),
        sim_mean=('p_response_sim', 'mean')
    ).reset_index()
    
    x = np.arange(10)
    width = 0.35
    axes[1].bar(x - width/2, rep_comparison['adaptive_mean'], width, label='Adaptive', color='steelblue')
    axes[1].bar(x + width/2, rep_comparison['sim_mean'], width, label='Sim-Specific', color='coral')
    axes[1].set_xlabel('Replicate')
    axes[1].set_ylabel('Mean Response Probability')
    axes[1].set_title('(B) Per-Replicate Mean Response Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Rep-{i}' for i in range(10)], rotation=45)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/fig11_adaptive_vs_sim.png", bbox_inches='tight')
    plt.close()
    print("Saved fig11_adaptive_vs_sim.png")

# ============================================================
# Peptide Diversity Analysis
# ============================================================
print("\n=== PEPTIDE DIVERSITY ANALYSIS ===")

# How many peptides does each mutation contribute?
mut_peptides = cell_pop.groupby('mutation')['presented_peptides'].nunique().sort_values(ascending=False)
print("Peptides per mutation:")
for m, v in mut_peptides.items():
    print(f"  {m}: {v} unique peptides")

# Figure 12: Peptide diversity
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['steelblue' if m in vaccine_comp['peptide'].values else 'lightcoral' 
          for m in mut_peptides.index]
ax.barh(mut_peptides.index, mut_peptides.values, color=colors)
ax.set_xlabel('Number of Unique Peptides')
ax.set_ylabel('Mutation')
ax.set_title('Peptide Diversity per Mutation')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', label='In Vaccine'),
                   Patch(facecolor='lightcoral', label='Not in Vaccine')]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig12_peptide_diversity.png", bbox_inches='tight')
plt.close()
print("Saved fig12_peptide_diversity.png")

# ============================================================
# Marginal Element Contribution (Leave-One-Out)
# ============================================================
print("\n=== LEAVE-ONE-OUT ANALYSIS ===")

vaccine_elements = vaccine_comp['peptide'].tolist()
loo_results = []

# Full vaccine response
full_responses = []
for rep in range(10):
    df = rep_scores[rep]
    df_sel = df[df['vaccine_element'].isin(vaccine_elements)]
    cell_agg = df_sel.groupby('cell_id')['log_p_no_response'].sum().reset_index()
    cell_agg['p_response'] = 1 - np.exp(cell_agg['log_p_no_response'])
    full_responses.extend(cell_agg['p_response'].tolist())
full_mean = np.mean(full_responses)

for elem in vaccine_elements:
    reduced = [e for e in vaccine_elements if e != elem]
    reduced_responses = []
    for rep in range(10):
        df = rep_scores[rep]
        df_sel = df[df['vaccine_element'].isin(reduced)]
        cell_agg = df_sel.groupby('cell_id')['log_p_no_response'].sum().reset_index()
        cell_agg['p_response'] = 1 - np.exp(cell_agg['log_p_no_response'])
        reduced_responses.extend(cell_agg['p_response'].tolist())
    
    reduced_mean = np.mean(reduced_responses)
    marginal = full_mean - reduced_mean
    
    loo_results.append({
        'element': elem,
        'full_mean': full_mean,
        'reduced_mean': reduced_mean,
        'marginal_contribution': marginal
    })

loo_df = pd.DataFrame(loo_results).sort_values('marginal_contribution', ascending=False)
loo_df.to_csv(f"{OUT_DIR}/leave_one_out.csv", index=False)

print("Leave-one-out marginal contributions:")
for _, row in loo_df.iterrows():
    print(f"  {row['element']}: Δ = {row['marginal_contribution']:.6f} "
          f"(reduced mean = {row['reduced_mean']:.4f})")

# Figure 13: LOO analysis
fig, ax = plt.subplots(figsize=(10, 6))
loo_sorted = loo_df.sort_values('marginal_contribution', ascending=True)
ax.barh(loo_sorted['element'], loo_sorted['marginal_contribution'], color='steelblue')
ax.set_xlabel('Marginal Contribution to Mean Response')
ax.set_ylabel('Vaccine Element')
ax.set_title('Leave-One-Out Marginal Contribution Analysis')
ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/fig13_loo_analysis.png", bbox_inches='tight')
plt.close()
print("Saved fig13_loo_analysis.png")

print("\n=== ADDITIONAL ANALYSES COMPLETE ===")
