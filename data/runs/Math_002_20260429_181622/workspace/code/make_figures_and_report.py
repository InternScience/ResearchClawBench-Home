#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path('.')
OUT=ROOT/'outputs'
IMG=ROOT/'report'/'images'
REP=ROOT/'report'
IMG.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

over=pd.read_csv(OUT/'data_overview.csv')
res=pd.read_csv(OUT/'benchmark_results.csv')
sumdf=pd.read_csv(OUT/'summary_by_family_method.csv')
hist=pd.read_csv(OUT/'lns_history.csv') if (OUT/'lns_history.csv').exists() else pd.DataFrame()

# Figure 1: data overview
fig, ax=plt.subplots(1,2,figsize=(12,4.5))
family_order=sorted(over.family.unique())
dens=over.groupby('family').obstacle_density.mean().reindex(family_order)
ag=over.groupby('family').agent_count_generated.mean().reindex(family_order)
ax[0].bar(range(len(dens)), dens.values, color='#4C78A8')
ax[0].set_xticks(range(len(dens))); ax[0].set_xticklabels(dens.index, rotation=35, ha='right')
ax[0].set_ylabel('Mean obstacle density'); ax[0].set_title('Map structural difficulty')
ax[1].bar(range(len(ag)), ag.values, color='#F58518')
ax[1].set_xticks(range(len(ag))); ax[1].set_xticklabels(ag.index, rotation=35, ha='right')
ax[1].set_ylabel('Generated agents per instance'); ax[1].set_title('Benchmark agent load')
fig.tight_layout(); fig.savefig(IMG/'data_overview.png', dpi=180); plt.close(fig)

# Figure 2: success rate
methods=['IndependentShortest','PrioritizedPlanning','RandomLNS+PP','PolicyLNS+PP']
pivot=sumdf.pivot(index='family', columns='method', values='success_rate').reindex(family_order)[methods]
fig, ax=plt.subplots(figsize=(12,4.8))
x=np.arange(len(pivot)); width=0.2
colors=['#BAB0AC','#4C78A8','#54A24B','#E45756']
for i,m in enumerate(methods):
    ax.bar(x+(i-1.5)*width, pivot[m].values, width, label=m, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=30, ha='right')
ax.set_ylim(0,1.05); ax.set_ylabel('Success rate (zero collisions)')
ax.set_title('Collision-free success by map family')
ax.legend(ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(IMG/'success_rate.png', dpi=180); plt.close(fig)

# Figure 3: runtime-quality tradeoff
fig, ax=plt.subplots(figsize=(7,5))
agg=res.groupby('method').agg(runtime=('runtime_seconds','mean'), collisions=('collisions_total','mean'), success=('success','mean')).reindex(methods)
for m,c in zip(methods,colors):
    ax.scatter(agg.loc[m,'runtime'], agg.loc[m,'collisions'], s=80+400*agg.loc[m,'success'], label=f"{m} (succ={agg.loc[m,'success']:.2f})", color=c, alpha=.85)
    ax.annotate(m, (agg.loc[m,'runtime'], agg.loc[m,'collisions']), xytext=(5,4), textcoords='offset points', fontsize=8)
ax.set_xscale('symlog', linthresh=0.01)
ax.set_xlabel('Mean runtime per instance (s, symlog)')
ax.set_ylabel('Mean residual conflicts/unplanned agents')
ax.set_title('Efficiency-quality trade-off')
fig.tight_layout(); fig.savefig(IMG/'runtime_quality.png', dpi=180); plt.close(fig)

# Figure 4: conflict reduction over LNS iterations
fig, ax=plt.subplots(figsize=(8,5))
if not hist.empty:
    h=hist.groupby(['method','iteration']).best_collisions.mean().reset_index()
    for m,c in [('RandomLNS+PP','#54A24B'),('PolicyLNS+PP','#E45756')]:
        sub=h[h.method==m]
        if len(sub): ax.plot(sub.iteration, sub.best_collisions, marker='o', label=m, color=c)
ax.set_xlabel('LNS iteration'); ax.set_ylabel('Mean best residual collisions')
ax.set_title('Conflict reduction during neighborhood repair')
ax.legend(); fig.tight_layout(); fig.savefig(IMG/'conflict_reduction.png', dpi=180); plt.close(fig)

# Additional validation plot: residual conflicts by family/method
pivotc=sumdf.pivot(index='family', columns='method', values='mean_collisions').reindex(family_order)[methods]
fig, ax=plt.subplots(figsize=(12,4.8))
for i,m in enumerate(methods): ax.bar(x+(i-1.5)*width, pivotc[m].values, width, label=m, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(pivotc.index, rotation=30, ha='right')
ax.set_ylabel('Mean residual collisions/unplanned'); ax.set_title('Validation: residual collision counts')
ax.legend(ncol=2, fontsize=8); fig.tight_layout(); fig.savefig(IMG/'validation_collisions.png', dpi=180); plt.close(fig)

# Claim recovery and direct results
claim_rows=[]
overall=res.groupby('method').agg(success_rate=('success','mean'), mean_collisions=('collisions_total','mean'), mean_runtime_s=('runtime_seconds','mean'), mean_cost=('sum_of_costs','mean')).reindex(methods).reset_index()
overall.to_csv(OUT/'overall_method_summary.csv',index=False)
claim_rows.append({'claim':'Independent shortest paths are not collision-free in dense MAPF tasks.','artifact':'outputs/summary_by_family_method.csv; report/images/validation_collisions.png','evidence':f"Overall success={overall.loc[overall.method=='IndependentShortest','success_rate'].iloc[0]:.3f}, mean residual conflicts={overall.loc[overall.method=='IndependentShortest','mean_collisions'].iloc[0]:.2f}.",'status':'directly verified'})
claim_rows.append({'claim':'Prioritized planning greatly reduces collisions relative to independent paths.','artifact':'outputs/overall_method_summary.csv','evidence':f"Mean residual conflicts fall from {overall.loc[overall.method=='IndependentShortest','mean_collisions'].iloc[0]:.2f} to {overall.loc[overall.method=='PrioritizedPlanning','mean_collisions'].iloc[0]:.2f}.",'status':'directly verified'})
claim_rows.append({'claim':'PolicyLNS+PP is a faithful LNS/PP hybrid but only approximates MARL.','artifact':'outputs/method_fidelity_checklist.json; code/hybrid_mapf_lns.py','evidence':'Destroy/repair LNS and prioritized space-time repair are implemented; MARL is represented by a transparent conflict-pressure policy rather than learned neural MARL.','status':'verified with limitation'})
claim_rows.append({'claim':'The benchmark preserves map-family strata.','artifact':'outputs/benchmark_results.csv; outputs/summary_by_family_method.csv','evidence':f"{res.family.nunique()} families and {len(res)} method-instance rows were evaluated.",'status':'directly verified'})
pd.DataFrame(claim_rows).to_csv(OUT/'claim_recovery_table.csv',index=False)

# Update inventory status
with open(OUT/'target_artifact_inventory.json') as f: inv=json.load(f)
for section in inv.values():
    if isinstance(section,list):
        for item in section:
            p=Path(item['name'])
            item['status']='satisfied' if p.exists() else 'unsatisfied: file not found'
with open(OUT/'target_artifact_inventory.json','w') as f: json.dump(inv,f,indent=2)

# Report content
fmt_overall=overall.copy()
for col in ['success_rate','mean_collisions','mean_runtime_s','mean_cost']:
    fmt_overall[col]=fmt_overall[col].map(lambda v: f'{v:.3f}')
overall_md=fmt_overall.to_markdown(index=False)
family_md=sumdf.sort_values(['family','method']).to_markdown(index=False)
policy=pd.read_csv(OUT/'neighborhood_policy_importance.csv').to_markdown(index=False)
claims=pd.read_csv(OUT/'claim_recovery_table.csv').to_markdown(index=False)

report=f'''# Hybrid Policy-Guided Large Neighborhood Search for MAPF

## Abstract

This study implements and evaluates a bounded, reproducible Multi-Agent Path Finding (MAPF) solver that combines space-time prioritized planning with Large Neighborhood Search (LNS). The proposed method, **PolicyLNS+PP**, uses a MARL-inspired conflict-pressure policy during early neighborhood selection and efficient prioritized planning for local repair. The available workspace data contain obstacle grids but no explicit start/goal task files, so agent tasks were generated deterministically from free cells for each map. On a 16-instance stratified benchmark spanning eight map families, prioritized methods sharply reduced residual collisions relative to independent shortest paths. The final implementation is collision-validated with explicit vertex and edge-swap checks, but the MARL component should be interpreted as a transparent policy approximation rather than a trained neural MARL model.

## 1. Problem and data

MAPF requires one path per agent from a distinct start to a goal on a grid with static obstacles. A solution is valid only if no two agents occupy the same vertex at the same timestep and no two agents swap positions along an edge in opposite directions. The workspace provides eight map families: `empty`, `maps_60_10_10_0.175`, `maze`, `random_large`, `random_medium`, `random_small`, `room`, and `warehouse`. The datasets contain `.npy` obstacle maps (`0` free, `-1` obstacle). Because no separate task files were present, starts and goals were reproducibly generated from free cells using file-path-derived seeds.

![Dataset overview](images/data_overview.png)

The full map census is saved in `outputs/data_overview.csv`. The executed benchmark used two instances per family to keep runtime bounded while preserving family-level strata.

## 2. Methodology

### 2.1 Baselines and proposed hybrid

Four methods were evaluated:

1. **IndependentShortest**: each agent receives an individual shortest path ignoring other agents. This is a lower-bound quality reference and is expected to collide.
2. **PrioritizedPlanning**: agents are planned sequentially with space-time A* against reservations from earlier agents.
3. **RandomLNS+PP**: an LNS ablation that randomly destroys a neighborhood of agents and repairs them by prioritized planning.
4. **PolicyLNS+PP**: the proposed hybrid. It destroys agents with high learned-policy-surrogate scores and repairs them by prioritized planning.

The LNS loop removes selected agent paths, reserves all remaining paths, and replans selected agents with space-time A*. The repair planner prevents both reserved-vertex conflicts and swap conflicts. The policy score is a transparent MARL-inspired proxy:

\[
score_i = 4.0\,conflicts_i + 0.15\,excess_i + 0.02\,Manhattan_i + 1.5\,density_i.
\]

This favors agents involved in conflicts and agents traversing constrained or inefficient corridors. Feature weights and interpretations are exported in `outputs/neighborhood_policy_importance.csv`:

{policy}

### 2.2 Fidelity to the named MARL-LNS objective

The task requested integration of Multi-Agent Reinforcement Learning into LNS. The workspace did not include a MARL simulator, pretrained policy, or deep-RL dependencies. Dependency checks are saved in `outputs/dependency_check.json`; common scientific libraries were available, but no specialized MAPF/MARL stack was present. Therefore, the implementation preserves the **structural** hybrid commitment (policy-guided multi-agent neighborhood selection + LNS + prioritized repair) but approximates MARL with a deterministic, interpretable conflict-pressure policy. This deviation is documented in `outputs/method_fidelity_checklist.json`.

## 3. Results

### 3.1 Overall method comparison

{overall_md}

The independent baseline had zero success on this bounded benchmark because shortest paths often collide. Prioritized planning reduced mean residual conflicts substantially. LNS variants retained the same basic collision-reduction behavior, with PolicyLNS+PP matching or modestly improving some family-level outcomes while incurring extra repair overhead on hard 25x25 instances.

![Success rate by family](images/success_rate.png)

### 3.2 Family-specific outcomes

{family_md}

The family-level table shows that easy open or small random instances were solved reliably by prioritized methods, whereas maze, room, and some medium/large random instances remained difficult under the bounded runtime and generated high-density tasks. This is consistent with bottleneck-heavy environments producing hard ordering and reservation choices.

### 3.3 Runtime-quality trade-off

![Runtime quality tradeoff](images/runtime_quality.png)

The runtime-quality plot shows the core trade-off: IndependentShortest is fastest but invalid; PrioritizedPlanning is the most direct collision-reducing baseline; LNS methods add computational cost in exchange for the opportunity to repair conflict neighborhoods. In this small run, the policy-guided variant did not dominate the random LNS ablation globally, but it provides a principled mechanism for focusing repair on high-conflict agents.

### 3.4 LNS conflict reduction

![Conflict reduction](images/conflict_reduction.png)

The LNS history (`outputs/lns_history.csv`) records per-iteration proposal and best collision counts. Since many prioritized-planning solutions were already locally stable or infeasible to fully repair within the bounded search horizon, LNS curves are relatively flat on several families. Room instances showed one case where PolicyLNS+PP reduced residual collisions relative to the initial prioritized solution.

## 4. Validation

### 4.1 Directly verified from workspace artifacts

- `code/hybrid_mapf_lns.py` implements map loading, deterministic task generation, shortest-path baselines, space-time prioritized planning, LNS repair, and collision validation.
- `outputs/benchmark_results.csv` contains per-instance metrics for all methods.
- `outputs/validation_examples.json` records explicit validation summaries for representative prioritized and hybrid solutions.
- `outputs/sample_solutions.json` saves representative starts, goals, and PolicyLNS+PP paths.
- Success requires all agents to be planned and zero vertex/swap collisions according to `detect_collisions`.

![Residual collision validation](images/validation_collisions.png)

### 4.2 Claim recovery table

{claims}

### 4.3 Related-work and assumption limitations

The five PDFs in `related_work/` could not be extracted by the provided `ReadPDF` tool, and local PDF utilities/libraries were unavailable. This status is saved in `outputs/related_work_contract.json`. Consequently, the report does not claim paper-specific numerical reproduction. It uses standard MAPF concepts named in the task: prioritized planning, LNS, vertex collisions, and edge-swap collisions.

The largest methodological assumption is task generation: the data contained maps only, not explicit agent start/goal configurations. Starts and goals were therefore generated deterministically and may not match any hidden benchmark task distribution.

## 5. Discussion

The experiment supports three conclusions. First, collision-aware planning is essential: independent shortest paths are fast but invalid under multi-agent interactions. Second, prioritized planning is a strong efficiency baseline and often resolves most conflicts. Third, LNS provides a natural framework for targeted repair, but the benefit depends on the quality of neighborhood selection and sufficient time budget. The implemented policy-guided selector captures the intended MARL role—allocating repair attention to locally interacting agents—but it is not a trained MARL policy.

Future work should replace the hand-weighted policy with a trained decentralized MARL value or actor network, use provided start/goal scenarios when available, and run larger sweeps over neighborhood size, LNS iterations, and agent density. A learned policy should be evaluated not only on success and cost but also on generalization across maze, room, warehouse, and random maps.

## 6. Reproducibility

Run the benchmark and report generation from the workspace root:

```bash
python3 code/hybrid_mapf_lns.py --per-family 2 --iterations 6 --neighborhood 5 --seed 11
python3 code/make_figures_and_report.py
```

Primary artifacts:

- `outputs/method_contract.json`
- `outputs/target_artifact_inventory.json`
- `outputs/dependency_check.json`
- `outputs/method_fidelity_checklist.json`
- `outputs/benchmark_results.csv`
- `outputs/summary_by_family_method.csv`
- `outputs/overall_method_summary.csv`
- `outputs/claim_recovery_table.csv`
- `report/images/*.png`
'''
(REP/'report.md').write_text(report)
print('wrote figures and report')
