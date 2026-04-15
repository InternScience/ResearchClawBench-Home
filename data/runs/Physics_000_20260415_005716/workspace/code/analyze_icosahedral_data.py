from pathlib import Path
import ast, json, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = Path(__file__).resolve().parents[1]
text = (root / 'data' / 'Multi-component Icosahedral Reproduction Data.txt').read_text()
vars = {}
for line in text.splitlines():
    s = line.strip()
    if not s or s.startswith('#') or '=' not in s:
        continue
    k, v = s.split('=', 1)
    k = k.strip(); v = v.strip()
    try:
        vars[k] = ast.literal_eval(v)
    except Exception:
        if k == 'deposition_sequences':
            seqs = []
            pattern = r"\('([^']+)',\s*(\[[^\)]*\])\)"
            for name, arr in re.findall(pattern, v):
                seqs.append((name, ast.literal_eval(arr)))
            vars[k] = seqs
        else:
            vars[k] = v

out = root / 'outputs'; out.mkdir(exist_ok=True)
img = root / 'report' / 'images'; img.mkdir(parents=True, exist_ok=True)

radii = pd.DataFrame(vars['atomic_radii'], columns=['element', 'radius_A'])
compat = pd.DataFrame(vars['atomic_pairs_compatibility'], columns=['inner', 'outer', 'reported_mismatch'])
clusters = pd.DataFrame(vars['multicomponent_clusters'], columns=['cluster', 'inner_element', 'outer_element', 'inner_shell', 'outer_shell'])
rmap = dict(vars['atomic_radii'])
clusters['inner_radius'] = clusters['inner_element'].map(rmap)
clusters['outer_radius'] = clusters['outer_element'].map(rmap)
clusters['computed_mismatch'] = (clusters['outer_radius'] - clusters['inner_radius']) / clusters['inner_radius']

ranges = pd.DataFrame(vars['optimal_mismatch_ranges'], columns=['inner_shell', 'outer_shell', 'range_low', 'range_high'])
ranges['range_mid'] = (ranges['range_low'] + ranges['range_high']) / 2
mismatch = pd.DataFrame(vars['mismatch_params'], columns=['shell_i', 'shell_j', 'type_i', 'type_j', 'theoretical_sm'])
energies = pd.DataFrame(vars['shell_energies'], columns=['shell_number', 'chiral_category', 'relative_energy'])
exp = pd.DataFrame(vars['experimental_points'], columns=['Ti', 'Tip1', 'measured_sm', 'theoretical_sm'])
exp['abs_error'] = (exp['measured_sm'] - exp['theoretical_sm']).abs()
exp['rel_error_pct'] = 100 * exp['abs_error'] / exp['theoretical_sm']

growth = pd.DataFrame(vars['growth_results'], columns=['step', 'category', 'avg_mismatch'])
path_stats = pd.DataFrame(vars['path_selection_stats'], columns=['path', 'count'])
path_stats['fraction'] = path_stats['count'] / path_stats['count'].sum()

dep_rows = []
for name, seq in vars['deposition_sequences']:
    counts = pd.Series(seq).value_counts().to_dict()
    dep_rows.append({'experiment': name, 'length': len(seq), 'composition': counts})
dep = pd.DataFrame(dep_rows)

range_map = {(a, b): (lo, hi, (lo + hi) / 2) for a, b, lo, hi in vars['optimal_mismatch_ranges']}
preds = []
for _, row in compat.iterrows():
    best = None
    for (a, b), (lo, hi, mid) in range_map.items():
        dist = abs(row['reported_mismatch'] - mid)
        inside = lo <= row['reported_mismatch'] <= hi
        score = (0 if inside else 1, dist)
        if best is None or score < best['score']:
            best = {'inner_shell': a, 'outer_shell': b, 'target_mid': mid, 'range_low': lo, 'range_high': hi, 'inside_range': inside, 'score': score}
    preds.append({
        'pair': f"{row['inner']}-{row['outer']}",
        'inner_element': row['inner'],
        'outer_element': row['outer'],
        'reported_mismatch': row['reported_mismatch'],
        **{k: v for k, v in best.items() if k != 'score'}
    })
pred_df = pd.DataFrame(preds)

radii.to_csv(out / 'atomic_radii.csv', index=False)
compat.to_csv(out / 'atomic_pair_compatibility.csv', index=False)
clusters.to_csv(out / 'stable_structure_table.csv', index=False)
ranges.to_csv(out / 'size_mismatch_summary.csv', index=False)
mismatch.to_csv(out / 'mismatch_params.csv', index=False)
energies.to_csv(out / 'shell_energy_table.csv', index=False)
exp.to_csv(out / 'validation_experimental_vs_theory.csv', index=False)
growth.to_csv(out / 'growth_results.csv', index=False)
path_stats.to_csv(out / 'path_selection_summary.csv', index=False)
dep.to_json(out / 'deposition_sequence_summary.json', orient='records', indent=2)
pred_df.to_csv(out / 'predicted_pair_shell_mapping.csv', index=False)

summary = {
    'n_atomic_species': int(len(radii)),
    'n_compatible_pairs': int(len(compat)),
    'n_validated_clusters': int(len(clusters)),
    'mean_validation_abs_error': float(exp['abs_error'].mean()),
    'max_validation_abs_error': float(exp['abs_error'].max()),
    'dominant_path': path_stats.sort_values('count', ascending=False).iloc[0]['path'],
    'dominant_path_fraction': float(path_stats['fraction'].max()),
    'lowest_energy_entry': energies.sort_values('relative_energy').iloc[0].to_dict()
}
(out / 'analysis_summary.json').write_text(json.dumps(summary, indent=2))

claim_rows = [
    {'claim': 'Dataset provides explicit shell sequences, mismatch parameters, growth results, and validation points.', 'artifact': 'outputs/data_schema_summary.json'},
    {'claim': 'Validated multi-component clusters include Na13@Rb32, K13@Cs42, and Ag13@Cu45.', 'artifact': 'outputs/stable_structure_table.csv'},
    {'claim': 'MC-MC and MC-Ch1 mismatches cluster near ~0.04 and ~0.14 respectively.', 'artifact': 'outputs/size_mismatch_summary.csv; outputs/mismatch_params.csv'},
    {'claim': 'Theory agrees with experimental mismatch values within small absolute error.', 'artifact': 'outputs/validation_experimental_vs_theory.csv'},
    {'claim': 'Conservative path dominates growth-path selection statistics.', 'artifact': 'outputs/path_selection_summary.csv'}
]
pd.DataFrame(claim_rows).to_csv(out / 'claim_recovery_table.csv', index=False)

sns.set_theme(style='whitegrid')
plt.figure(figsize=(6, 4))
sns.scatterplot(data=exp, x='theoretical_sm', y='measured_sm', s=80)
lims = [min(exp['theoretical_sm'].min(), exp['measured_sm'].min()) - 0.005, max(exp['theoretical_sm'].max(), exp['measured_sm'].max()) + 0.005]
plt.plot(lims, lims, '--', color='gray', linewidth=1)
for _, r in exp.iterrows():
    plt.text(r['theoretical_sm'] + 0.001, r['measured_sm'] + 0.001, f"{int(r['Ti'])}->{int(r['Tip1'])}", fontsize=8)
plt.xlim(lims); plt.ylim(lims)
plt.xlabel('Theoretical mismatch')
plt.ylabel('Measured mismatch')
plt.title('Validation of mismatch theory')
plt.tight_layout(); plt.savefig(img / 'validation_theory_vs_experiment.png', dpi=200); plt.close()

plt.figure(figsize=(6, 4))
ord = energies.sort_values(['shell_number', 'relative_energy'])
sns.barplot(data=ord, x='shell_number', y='relative_energy', hue='chiral_category')
plt.title('Relative shell energies by shell number and category')
plt.ylabel('Relative energy (normalized)')
plt.tight_layout(); plt.savefig(img / 'shell_energy_comparison.png', dpi=200); plt.close()

plt.figure(figsize=(6, 4))
path_stats_sorted = path_stats.sort_values('count', ascending=False)
sns.barplot(data=path_stats_sorted, x='path', y='count', hue='path', dodge=False, legend=False)
plt.xticks(rotation=20, ha='right')
plt.title('Growth path selection statistics')
plt.tight_layout(); plt.savefig(img / 'growth_path_statistics.png', dpi=200); plt.close()

plt.figure(figsize=(7, 4))
for cat, grp in growth.groupby('category'):
    grp = grp.sort_values('step')
    plt.plot(grp['step'], grp['avg_mismatch'], marker='o', label=cat)
plt.xlabel('Growth step'); plt.ylabel('Average mismatch'); plt.title('Mismatch trajectory during growth simulations')
plt.legend(); plt.tight_layout(); plt.savefig(img / 'growth_mismatch_trajectories.png', dpi=200); plt.close()

plt.figure(figsize=(6, 4))
merged = clusters.merge(pred_df[['inner_element', 'outer_element', 'inside_range', 'target_mid', 'inner_shell', 'outer_shell']], on=['inner_element', 'outer_element'], how='left')
plotdf = merged[['cluster', 'computed_mismatch', 'target_mid']].melt(id_vars='cluster', var_name='kind', value_name='mismatch')
sns.barplot(data=plotdf, x='cluster', y='mismatch', hue='kind')
plt.xticks(rotation=15, ha='right')
plt.ylabel('Mismatch')
plt.title('Observed cluster mismatch vs mapped shell target')
plt.tight_layout(); plt.savefig(img / 'cluster_mismatch_vs_target.png', dpi=200); plt.close()

print(json.dumps(summary, indent=2))
