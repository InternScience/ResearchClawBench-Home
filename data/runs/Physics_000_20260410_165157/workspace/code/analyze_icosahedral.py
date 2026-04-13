import ast
import math
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'Multi-component Icosahedral Reproduction Data.txt'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')


def parse_dataset(path):
    raw = path.read_text(encoding='utf-8')
    data = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        try:
            data[key] = ast.literal_eval(value)
        except Exception:
            data[key] = value
    return data


def shell_count(n):
    return 10 * n * n + 2


def candidate_shell_sizes(max_n=6):
    return {n: shell_count(n) for n in range(1, max_n + 1)}


def classify_delta(delta):
    if 0.03 <= delta <= 0.05:
        return 'MC'
    if 0.08 <= delta <= 0.10:
        return 'BG'
    if 0.12 <= delta <= 0.16:
        return 'Ch1'
    if 0.19 <= delta <= 0.22:
        return 'Ch2'
    return 'off-window'


def pick_shell_size(size_ratio, shell_map):
    best_n, best_count, best_err = None, None, 1e9
    for n, count in shell_map.items():
        geom_ratio = math.sqrt(count / 12.0)
        err = abs(geom_ratio - size_ratio)
        if err < best_err:
            best_n, best_count, best_err = n, count, err
    return best_n, best_count, best_err


def pair_predictions(radii):
    shell_map = candidate_shell_sizes(6)
    rows = []
    for core, r1 in radii:
        for outer, r2 in radii:
            if outer == core or r2 <= r1:
                continue
            delta = round((r2 - r1) / r1, 3)
            category = classify_delta(delta)
            n_shell, count, geom_err = pick_shell_size(r2 / r1, shell_map)
            stable_score = max(0.0, 1.0 - geom_err) * (1.0 if category != 'off-window' else 0.25)
            rows.append({
                'core': core,
                'outer': outer,
                'r_core': r1,
                'r_outer': r2,
                'size_ratio': r2 / r1,
                'delta': delta,
                'predicted_category': category,
                'outer_shell_index': n_shell,
                'outer_shell_atoms': count,
                'predicted_cluster': f'{core}13@{outer}{count}',
                'geometric_error': geom_err,
                'stability_score': round(stable_score, 3),
            })
    return pd.DataFrame(rows).sort_values(['stability_score', 'delta'], ascending=[False, True])


def main():
    data = parse_dataset(DATA_FILE)

    radii_df = pd.DataFrame(data['atomic_radii'], columns=['element', 'radius'])
    pair_df = pd.DataFrame(data['atomic_pairs_compatibility'], columns=['core', 'outer', 'compatibility'])
    mismatch_df = pd.DataFrame(data['optimal_mismatch_ranges'], columns=['inner_shell', 'outer_shell', 'delta_min', 'delta_max'])
    clusters_df = pd.DataFrame(data['multicomponent_clusters'], columns=['cluster', 'core', 'outer', 'shell1', 'shell2'])
    exp_df = pd.DataFrame(data['experimental_points'], columns=['Ti', 'Tip1', 'measured', 'theoretical'])
    growth_df = pd.DataFrame(data['growth_results'], columns=['step', 'category', 'avg_mismatch'])
    path_df = pd.DataFrame(data['path_selection_stats'], columns=['path', 'count'])
    energy_df = pd.DataFrame(data['shell_energies'], columns=['shell_index', 'category', 'energy'])

    pred_df = pair_predictions(data['atomic_radii'])
    pred_df.to_csv(OUT / 'predicted_pairs.csv', index=False)
    radii_df.to_csv(OUT / 'atomic_radii.csv', index=False)
    exp_df.to_csv(OUT / 'experimental_points.csv', index=False)

    validated = clusters_df.merge(pred_df[['core','outer','delta','predicted_category','outer_shell_atoms','predicted_cluster']], on=['core','outer'], how='left')
    validated.to_csv(OUT / 'validated_clusters.csv', index=False)

    # Figure 1: atomic radii
    plt.figure(figsize=(7,4))
    sns.barplot(data=radii_df.sort_values('radius'), x='element', y='radius', palette='viridis')
    plt.ylabel('Atomic radius (Å)')
    plt.xlabel('Element')
    plt.title('Atomic size hierarchy used for shell design')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_atomic_radii.png', dpi=200)
    plt.close()

    # Figure 2: mismatch windows
    mm_plot = mismatch_df.copy()
    mm_plot['center'] = (mm_plot['delta_min'] + mm_plot['delta_max']) / 2
    mm_plot['width'] = mm_plot['delta_max'] - mm_plot['delta_min']
    plt.figure(figsize=(7,4))
    plt.hlines(y=np.arange(len(mm_plot)), xmin=mm_plot['delta_min'], xmax=mm_plot['delta_max'], color='tab:blue', lw=6)
    plt.plot(mm_plot['center'], np.arange(len(mm_plot)), 'o', color='black')
    plt.yticks(np.arange(len(mm_plot)), mm_plot['inner_shell'] + '→' + mm_plot['outer_shell'])
    plt.xlabel('Optimal size mismatch δ')
    plt.title('Reported stability windows for adjacent shells')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_mismatch_windows.png', dpi=200)
    plt.close()

    # Figure 3: experiment validation
    plt.figure(figsize=(5,5))
    sns.scatterplot(data=exp_df, x='theoretical', y='measured', s=80)
    lims = [min(exp_df[['measured','theoretical']].min()) - 0.01, max(exp_df[['measured','theoretical']].max()) + 0.01]
    plt.plot(lims, lims, '--', color='gray')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Theoretical mismatch')
    plt.ylabel('Measured mismatch')
    plt.title('Agreement between theory and experiment')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_validation_scatter.png', dpi=200)
    plt.close()

    # Figure 4: growth trajectories
    plt.figure(figsize=(7,4))
    sns.lineplot(data=growth_df, x='step', y='avg_mismatch', hue='category', marker='o')
    plt.ylabel('Average mismatch')
    plt.xlabel('Simulation step')
    plt.title('Self-assembly trajectories in growth simulations')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_growth_trajectories.png', dpi=200)
    plt.close()

    # Figure 5: path statistics
    plt.figure(figsize=(7,4))
    sns.barplot(data=path_df, x='path', y='count', palette='magma')
    plt.xticks(rotation=25, ha='right')
    plt.ylabel('Selection count')
    plt.xlabel('Path type')
    plt.title('Relative frequency of shell-growth path choices')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_path_statistics.png', dpi=200)
    plt.close()

    # Figure 6: top predictions
    top = pred_df.head(10).copy()
    plt.figure(figsize=(8,5))
    sns.barplot(data=top, y='predicted_cluster', x='stability_score', hue='predicted_category', dodge=False)
    plt.xlabel('Composite stability score')
    plt.ylabel('Predicted cluster')
    plt.title('Top candidate binary icosahedral clusters')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_top_predictions.png', dpi=200)
    plt.close()

    summary = {
        'top_predictions': top[['predicted_cluster','delta','predicted_category','stability_score']].to_dict(orient='records'),
        'validated_clusters': validated.to_dict(orient='records'),
        'experimental_rmse': float(np.sqrt(np.mean((exp_df['measured'] - exp_df['theoretical'])**2))),
        'path_probabilities': (path_df.assign(probability=path_df['count'] / path_df['count'].sum())[['path','probability']].to_dict(orient='records')),
        'energy_table': energy_df.to_dict(orient='records')
    }
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
