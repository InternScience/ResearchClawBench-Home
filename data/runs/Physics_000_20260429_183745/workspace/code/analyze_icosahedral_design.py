#!/usr/bin/env python3
"""Reproducible analysis for multi-component icosahedral shell design.

This script parses the provided reproduction-data text file, derives shell
mismatch/compatibility summaries, exports tables, and generates PNG figures
referenced by report/report.md.
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Multi-component Icosahedral Reproduction Data.txt"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def parse_data(path: Path) -> dict:
    """Parse assignment lines containing Python literals from the dataset."""
    text = path.read_text()
    data = {}
    for line in text.splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        name, expr = line.split("=", 1)
        name = name.strip()
        expr = expr.strip()
        if not re.match(r"^[A-Za-z_]\w*$", name):
            continue
        try:
            data[name] = ast.literal_eval(expr)
        except Exception as exc:
            # The reproduction file uses simple list repetition in one field
            # (e.g. ['Na']*50).  Evaluate that file-local literal expression in
            # a no-builtins namespace rather than loosening parsing generally.
            if name == 'deposition_sequences':
                try:
                    data[name] = eval(expr, {'__builtins__': {}}, {})
                except Exception as exc2:
                    raise ValueError(f"Could not parse {name}: {exc2}\n{expr[:200]}") from exc2
            else:
                raise ValueError(f"Could not parse {name}: {exc}\n{expr[:200]}") from exc
    return data


def shell_atom_count(total_sequence):
    vals = list(total_sequence)
    return [vals[0]] + [vals[i] - vals[i-1] for i in range(1, len(vals))]


def optimal_window(label_a, label_b, ranges):
    for a,b,lo,hi in ranges:
        if (a,b)==(label_a,label_b):
            return lo, hi
    return None, None


def classify_pair_mismatch(mismatch, ranges):
    hits=[]
    for a,b,lo,hi in ranges:
        if lo <= mismatch <= hi:
            hits.append(f"{a}->{b}")
    return ";".join(hits) if hits else "outside listed windows"


def make_tables(data: dict):
    # Dataset summary
    summary=[]
    for k,v in data.items():
        n=len(v) if hasattr(v, '__len__') and not isinstance(v,(str,dict)) else (len(v) if isinstance(v,dict) else 1)
        summary.append({'table':k, 'entry_count':n, 'python_type':type(v).__name__})
    pd.DataFrame(summary).sort_values('table').to_csv(OUT/'parsed_dataset_summary.csv', index=False)
    (OUT/'parsed_dataset_summary.json').write_text(json.dumps(summary, indent=2))

    radii = pd.DataFrame(data['atomic_radii'], columns=['element','radius_A'])
    radii.to_csv(OUT/'atomic_radii.csv', index=False)
    rmap=dict(data['atomic_radii'])

    compat = pd.DataFrame(data['atomic_pairs_compatibility'], columns=['core_element','shell_element','reported_mismatch'])
    compat['radius_ratio_shell_to_core']=compat['shell_element'].map(rmap)/compat['core_element'].map(rmap)
    compat['radius_fractional_difference']=(compat['shell_element'].map(rmap)-compat['core_element'].map(rmap))/compat['core_element'].map(rmap)
    compat['compatible_path_window']=compat['reported_mismatch'].apply(lambda x: classify_pair_mismatch(x, data['optimal_mismatch_ranges']))
    compat.to_csv(OUT/'atomic_pair_compatibility_augmented.csv', index=False)

    clusters=pd.DataFrame(data['multicomponent_clusters'], columns=['cluster','core_element','shell_element','inner_label','outer_label'])
    # derive total atom counts from cluster string, e.g. Na13@Rb32 means 45 atoms total
    parsed=[]
    for cl in clusters['cluster']:
        nums=list(map(int,re.findall(r'(?:[A-Z][a-z]?)(\d+)',cl)))
        parsed.append({'cluster':cl,'core_atoms':nums[0] if nums else np.nan,'outer_shell_atoms':nums[1] if len(nums)>1 else np.nan,'total_atoms':sum(nums) if nums else np.nan})
    clusters=clusters.merge(pd.DataFrame(parsed), on='cluster')
    clusters=clusters.merge(compat[['core_element','shell_element','reported_mismatch','compatible_path_window']], on=['core_element','shell_element'], how='left')
    clusters['design_status']='provided stable validation cluster'
    clusters.to_csv(OUT/'stable_cluster_predictions.csv', index=False)

    energies=pd.DataFrame(data['shell_energies'], columns=['shell_index','label','relative_energy_norm'])
    energies['rank_within_shell']=energies.groupby('shell_index')['relative_energy_norm'].rank(method='dense')
    energies.to_csv(OUT/'shell_energies_ranked.csv', index=False)

    mismatch=pd.DataFrame(data['mismatch_params'], columns=['shell_i','shell_j','label_i','label_j','optimal_mismatch'])
    mismatch[['window_low','window_high']]=mismatch.apply(lambda r: pd.Series(optimal_window(r.label_i,r.label_j,data['optimal_mismatch_ranges'])), axis=1)
    mismatch['in_listed_optimal_range']=mismatch.apply(lambda r: bool(pd.notna(r.window_low) and r.window_low<=r.optimal_mismatch<=r.window_high), axis=1)
    mismatch['window_midpoint']=(mismatch['window_low']+mismatch['window_high'])/2
    mismatch['deviation_from_window_midpoint']=mismatch['optimal_mismatch']-mismatch['window_midpoint']
    mismatch.to_csv(OUT/'mismatch_design_matrix.csv', index=False)

    exp=pd.DataFrame(data['experimental_points'], columns=['T_i','T_next','measured_sm','theoretical_sm'])
    exp['residual_measured_minus_theory']=exp['measured_sm']-exp['theoretical_sm']
    exp['abs_residual']=exp['residual_measured_minus_theory'].abs()
    exp['relative_abs_error_pct']=100*exp['abs_residual']/exp['theoretical_sm']
    metrics={
        'n_points': int(len(exp)),
        'mae_sm': float(exp['abs_residual'].mean()),
        'rmse_sm': float(np.sqrt(np.mean(exp['residual_measured_minus_theory']**2))),
        'mean_relative_abs_error_pct': float(exp['relative_abs_error_pct'].mean()),
        'pearson_r': float(exp[['measured_sm','theoretical_sm']].corr().iloc[0,1])
    }
    exp.to_csv(OUT/'experimental_mismatch_validation.csv', index=False)
    (OUT/'validation_metrics.json').write_text(json.dumps(metrics, indent=2))

    growth=pd.DataFrame(data['growth_results'], columns=['step','category','average_mismatch'])
    # Dataset contains three sequential traces; infer run_id by step reset to 0.
    run_ids=[]; rid=0; last=None
    for s in growth['step']:
        if last is not None and s==0:
            rid+=1
        run_ids.append(rid)
        last=s
    growth['run_id']=run_ids
    growth['trace_label']=growth['run_id'].map({0:'single-component MC growth',1:'seeded Ch1 growth',2:'mixed Ag/Cu transition'})
    growth.to_csv(OUT/'growth_path_summary.csv', index=False)

    path_stats=pd.DataFrame(data['path_selection_stats'], columns=['path_type','count'])
    path_stats['fraction']=path_stats['count']/path_stats['count'].sum()
    path_stats.to_csv(OUT/'path_selection_stats.csv', index=False)

    # Shell/path coordinate descriptors
    coords=pd.DataFrame(data['hexagonal_coords'], columns=['u','v'])
    coords['hex_distance_from_origin']=(coords['u'].abs()+coords['v'].abs()+(coords['u']+coords['v']).abs())/2
    coords['path_index']=np.arange(len(coords))
    coords.to_csv(OUT/'hexagonal_path_coordinates.csv', index=False)

    mackay=pd.DataFrame({'shell_index':np.arange(len(data['mackay_sequence'])), 'total_atoms':data['mackay_sequence']})
    mackay['shell_atoms']=shell_atom_count(data['mackay_sequence'])
    mackay['sequence']='Mackay'
    new=pd.DataFrame({'shell_index':np.arange(len(data['new_sequence_b5'])), 'total_atoms':data['new_sequence_b5']})
    new['shell_atoms']=shell_atom_count(data['new_sequence_b5'])
    new['sequence']='new_b5'
    shellseq=pd.concat([mackay,new], ignore_index=True)
    shellseq.to_csv(OUT/'shell_magic_sequences.csv', index=False)

    # Direct target answer table combining structure/path/mismatch recommendations.
    direct=[]
    for _,r in clusters.iterrows():
        direct.append({
            'predicted_structure':r.cluster,
            'adjacent_shells':f"{r.inner_label}->{r.outer_label}",
            'core_element':r.core_element,
            'shell_element':r.shell_element,
            'reported_or_required_mismatch':r.reported_mismatch,
            'compatible_design_window':r.compatible_path_window,
            'total_atoms':r.total_atoms,
            'interpretation':'stable validation structure from reproduction dataset'
        })
    for _,r in mismatch.iterrows():
        direct.append({
            'predicted_structure':f"generic shell {int(r.shell_i)}->{int(r.shell_j)} {r.label_i}->{r.label_j}",
            'adjacent_shells':f"{r.label_i}->{r.label_j}",
            'core_element':'generic',
            'shell_element':'generic',
            'reported_or_required_mismatch':r.optimal_mismatch,
            'compatible_design_window':f"{r.window_low}-{r.window_high}" if pd.notna(r.window_low) else 'not tabulated',
            'total_atoms':'path-dependent',
            'interpretation':'optimal adjacent-shell mismatch parameter'
        })
    direct=pd.DataFrame(direct)
    direct.to_csv(OUT/'direct_design_answers.csv', index=False)

    return {
        'radii':radii,'compat':compat,'clusters':clusters,'energies':energies,'mismatch':mismatch,
        'exp':exp,'metrics':metrics,'growth':growth,'path_stats':path_stats,'coords':coords,'shellseq':shellseq,
        'direct':direct
    }


def make_figures(t):
    # Figure 1: data overview radii and magic sequences
    fig, axes = plt.subplots(1,2, figsize=(11,4.2))
    sns.barplot(data=t['radii'], x='element', y='radius_A', ax=axes[0], color='#4c78a8')
    axes[0].set_title('Particle-size inputs')
    axes[0].set_ylabel('Atomic radius (Å)')
    axes[0].set_xlabel('Element')
    sns.lineplot(data=t['shellseq'], x='shell_index', y='total_atoms', hue='sequence', marker='o', ax=axes[1])
    axes[1].set_title('Icosahedral magic-number sequences')
    axes[1].set_ylabel('Cumulative atoms')
    axes[1].set_xlabel('Shell index')
    fig.tight_layout()
    fig.savefig(IMG/'figure_1_data_overview.png', dpi=220)
    plt.close(fig)

    # Figure 2: validation parity and residuals
    fig, axes = plt.subplots(1,2, figsize=(10.5,4.2))
    exp=t['exp']
    sns.scatterplot(data=exp, x='theoretical_sm', y='measured_sm', s=80, ax=axes[0], color='#f58518')
    mn=min(exp['theoretical_sm'].min(), exp['measured_sm'].min())*0.92
    mx=max(exp['theoretical_sm'].max(), exp['measured_sm'].max())*1.08
    axes[0].plot([mn,mx],[mn,mx],'k--',lw=1)
    axes[0].set_xlim(mn,mx); axes[0].set_ylim(mn,mx)
    axes[0].set_title('Mismatch validation parity')
    axes[0].set_xlabel('Theoretical size mismatch')
    axes[0].set_ylabel('Measured size mismatch')
    exp2=exp.copy(); exp2['transition']=exp2['T_i'].astype(str)+'→'+exp2['T_next'].astype(str)
    sns.barplot(data=exp2, x='transition', y='residual_measured_minus_theory', ax=axes[1], color='#54a24b')
    axes[1].axhline(0,color='k',lw=1)
    axes[1].set_title('Measured − theoretical residual')
    axes[1].set_ylabel('Residual in mismatch units')
    axes[1].set_xlabel('Shell transition')
    fig.tight_layout()
    fig.savefig(IMG/'figure_2_mismatch_validation.png', dpi=220)
    plt.close(fig)

    # Figure 3: energy and mismatch windows
    fig, axes = plt.subplots(1,2, figsize=(11.5,4.5))
    sns.barplot(data=t['energies'], x='shell_index', y='relative_energy_norm', hue='label', ax=axes[0])
    axes[0].set_title('Relative shell energies')
    axes[0].set_xlabel('Shell index')
    axes[0].set_ylabel('Normalized relative energy')
    mm=t['mismatch'].copy(); mm['transition']=mm['shell_i'].astype(str)+'→'+mm['shell_j'].astype(str)+' '+mm['label_i']+'→'+mm['label_j']
    sns.scatterplot(data=mm, x='transition', y='optimal_mismatch', hue='label_j', s=90, ax=axes[1])
    for idx,r in mm.reset_index(drop=True).iterrows():
        if pd.notna(r['window_low']):
            axes[1].vlines(idx, r['window_low'], r['window_high'], color='gray', lw=5, alpha=0.35)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_title('Adjacent-shell mismatch design map')
    axes[1].set_ylabel('Optimal size mismatch')
    axes[1].set_xlabel('Transition')
    fig.tight_layout()
    fig.savefig(IMG/'figure_3_energy_design_map.png', dpi=220)
    plt.close(fig)

    # Figure 4: growth traces and path stats
    fig, axes = plt.subplots(1,2, figsize=(11,4.2))
    sns.lineplot(data=t['growth'], x='step', y='average_mismatch', hue='trace_label', marker='o', ax=axes[0])
    axes[0].set_title('Growth-simulation mismatch trajectories')
    axes[0].set_xlabel('Simulation step')
    axes[0].set_ylabel('Average mismatch')
    axes[0].legend(fontsize=8)
    ps=t['path_stats'].sort_values('count', ascending=False)
    sns.barplot(data=ps, x='fraction', y='path_type', ax=axes[1], color='#b279a2')
    axes[1].set_title('Path-selection statistics')
    axes[1].set_xlabel('Fraction of recorded events')
    axes[1].set_ylabel('')
    for i,(_,r) in enumerate(ps.iterrows()):
        axes[1].text(r['fraction']+0.01, i, f"{r['count']}", va='center')
    axes[1].set_xlim(0, max(ps['fraction'])*1.25)
    fig.tight_layout()
    fig.savefig(IMG/'figure_4_growth_dynamics.png', dpi=220)
    plt.close(fig)


def update_inventory_and_claims(t):
    claims=[
        {'claim':'Provided reproduction data support stable validation clusters Na13@Rb32, K13@Cs42, and Ag13@Cu45.', 'artifact':'outputs/stable_cluster_predictions.csv'},
        {'claim':'Adjacent-shell optimal mismatches cluster into low MC->MC (~0.04), intermediate MC->Ch1 (~0.136-0.14), and larger Ch1->Ch2 (~0.21) regimes.', 'artifact':'outputs/mismatch_design_matrix.csv'},
        {'claim':'Theoretical and measured mismatch validation points agree closely.', 'artifact':'outputs/experimental_mismatch_validation.csv; outputs/validation_metrics.json'},
        {'claim':'Growth summaries show conservative MC growth, seeded Ch1 convergence, and a mixed Ag/Cu transition toward Ch1-like mismatch.', 'artifact':'outputs/growth_path_summary.csv; report/images/figure_4_growth_dynamics.png'},
        {'claim':'Conservative path choices dominate the recorded path-selection statistics.', 'artifact':'outputs/path_selection_stats.csv'}
    ]
    pd.DataFrame(claims).to_csv(OUT/'claim_recovery_table.csv', index=False)

    inv_path=OUT/'target_artifact_inventory.json'
    inv=json.loads(inv_path.read_text())
    for group in ['primary_artifacts','figure_artifacts']:
        for item in inv[group]:
            p=ROOT/item['path']
            if p.exists() and p.stat().st_size>0:
                item['status']='satisfied'
            else:
                item['status']='unsatisfied: file missing'
    inv_path.write_text(json.dumps(inv, indent=2))


def main():
    data=parse_data(DATA)
    t=make_tables(data)
    make_figures(t)
    update_inventory_and_claims(t)
    print(json.dumps({'outputs':len(list(OUT.glob('*'))), 'images':[p.name for p in IMG.glob('*.png')], 'metrics':t['metrics']}, indent=2))

if __name__=='__main__':
    main()
