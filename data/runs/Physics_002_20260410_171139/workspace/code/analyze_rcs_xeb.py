import ast
import json
import math
import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'data' / 'results'
AMPS_DIR = ROOT / 'data' / 'amplitudes'
OUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')


def parse_complex_or_prob(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, complex):
        return abs(v) ** 2
    if isinstance(v, str):
        try:
            obj = ast.literal_eval(v)
        except Exception:
            obj = None
        if isinstance(obj, complex):
            return abs(obj) ** 2
        if isinstance(obj, (int, float)):
            return float(obj)
    raise ValueError(f'Unsupported amplitude/probability value: {v!r}')


def parse_counts_key(k):
    if isinstance(k, str):
        return k
    return str(k)


def infer_metadata(path):
    m = re.search(r'N(?P<N>\d+)_d(?P<d>\d+)_r(?P<r>\d+)_XEB', path.name)
    if not m:
        raise ValueError(f'Cannot parse metadata from {path}')
    return int(m.group('N')), int(m.group('d')), int(m.group('r'))


def bootstrap_mean(values, n_boot=2000, seed=0):
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(arr.mean()), float(lo), float(hi)


def compute_record(count_file):
    count_file = Path(count_file)
    N, d, r = infer_metadata(count_file)
    amp_file = Path(str(count_file).replace(str(RESULTS_DIR), str(AMPS_DIR)).replace('_counts.json', '_amplitudes.json'))

    with open(count_file, 'r') as f:
        counts_raw = json.load(f)
    with open(amp_file, 'r') as f:
        amps_raw = json.load(f)

    counts = {parse_counts_key(k): int(v) for k, v in counts_raw.items()}
    ideal_probs = {str(k): parse_complex_or_prob(v) for k, v in amps_raw.items()}

    matched = sorted(set(counts) & set(ideal_probs))
    total_counts = sum(counts.values())
    sample_probs = np.array([ideal_probs[k] for k in matched for _ in range(counts[k])], dtype=float)
    weighted_probs = np.array([ideal_probs[k] * counts[k] for k in matched], dtype=float)

    mean_p = float(sample_probs.mean()) if len(sample_probs) else np.nan
    xeb = (2 ** N) * mean_p - 1.0
    stat_se = (2 ** N) * float(sample_probs.std(ddof=1) / math.sqrt(len(sample_probs))) if len(sample_probs) > 1 else np.nan
    boot_mean, boot_lo, boot_hi = bootstrap_mean(sample_probs, seed=N * 10000 + d * 100 + r)
    boot_xeb_lo = (2 ** N) * boot_lo - 1.0
    boot_xeb_hi = (2 ** N) * boot_hi - 1.0

    # MB-style linear regression surrogate: regress observed frequencies on ideal probabilities.
    # With sparse unique samples (mostly count 1), interpret slope around 2^N * total_counts.
    pvals = np.array([ideal_probs[k] for k in matched], dtype=float)
    freqs = np.array([counts[k] / total_counts for k in matched], dtype=float)
    X = np.vstack([np.ones(len(pvals)), pvals]).T
    beta, _, _, _ = np.linalg.lstsq(X, freqs, rcond=None)
    intercept, slope = beta
    mb_fidelity = float((2 ** N) * slope / total_counts)

    uniform_p = 2.0 ** (-N)
    log_ratio = np.log(np.maximum(sample_probs, 1e-300) / uniform_p)

    return {
        'N': N, 'd': d, 'r': r,
        'count_file': str(count_file.relative_to(ROOT)),
        'amp_file': str(amp_file.relative_to(ROOT)),
        'n_unique_counts': len(counts),
        'n_matched': len(matched),
        'total_counts': total_counts,
        'subset_prob_sum': float(sum(ideal_probs.values())),
        'mean_ideal_prob_observed': mean_p,
        'xeb_fidelity': float(xeb),
        'xeb_stat_se': float(stat_se),
        'xeb_boot_ci_low': float(boot_xeb_lo),
        'xeb_boot_ci_high': float(boot_xeb_hi),
        'mb_intercept': float(intercept),
        'mb_slope': float(slope),
        'mb_fidelity_surrogate': float(mb_fidelity),
        'mean_log_prob_ratio_to_uniform': float(log_ratio.mean()),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    count_files = sorted(glob.glob(str(RESULTS_DIR / 'N40_verification' / 'N40_d*_XEB' / '*_counts.json')))
    records = [compute_record(f) for f in count_files]
    df = pd.DataFrame(records).sort_values(['N', 'd', 'r']).reset_index(drop=True)
    df.to_csv(OUT_DIR / 'xeb_instance_results.csv', index=False)

    depth_summary = df.groupby(['N', 'd']).agg(
        n_instances=('r', 'count'),
        mean_xeb=('xeb_fidelity', 'mean'),
        std_xeb=('xeb_fidelity', 'std'),
        sem_xeb=('xeb_fidelity', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        mean_mb=('mb_fidelity_surrogate', 'mean'),
        mean_matches=('n_matched', 'mean'),
        mean_subset_prob_sum=('subset_prob_sum', 'mean'),
    ).reset_index()
    depth_summary['ci95'] = 1.96 * depth_summary['sem_xeb']
    depth_summary['classical_approx_floor'] = 0.0
    depth_summary.to_csv(OUT_DIR / 'xeb_depth_summary.csv', index=False)

    overview = {
        'n_instances': int(len(df)),
        'qubit_counts': sorted(df['N'].unique().tolist()),
        'depths': sorted(df['d'].unique().tolist()),
        'instances_per_depth': df.groupby('d')['r'].count().to_dict(),
        'mean_total_counts': float(df['total_counts'].mean()),
        'mean_matches': float(df['n_matched'].mean()),
        'mean_xeb_overall': float(df['xeb_fidelity'].mean()),
    }
    with open(OUT_DIR / 'data_overview.json', 'w') as f:
        json.dump(overview, f, indent=2)

    # Figures
    plt.figure(figsize=(10, 6))
    sns.histplot(df['xeb_fidelity'], bins=30, kde=True, color='#4c72b0')
    plt.xlabel('Instance-level linear XEB fidelity')
    plt.ylabel('Count')
    plt.title('Distribution of instance-level XEB fidelities')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'xeb_histogram.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=depth_summary, x='d', y='mean_xeb', marker='o', linewidth=2.5, label='Mean XEB fidelity')
    plt.fill_between(depth_summary['d'], depth_summary['mean_xeb'] - depth_summary['ci95'], depth_summary['mean_xeb'] + depth_summary['ci95'], alpha=0.25)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Uniform / classical baseline')
    plt.xlabel('Circuit depth d')
    plt.ylabel('Fidelity estimate')
    plt.title('Fidelity versus depth for N=40 arbitrary-geometry verification circuits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fidelity_vs_depth.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='d', y='xeb_fidelity', alpha=0.55, s=60)
    sns.lineplot(data=depth_summary, x='d', y='mean_xeb', color='crimson', marker='o', linewidth=2.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel('Circuit depth d')
    plt.ylabel('Instance-level XEB fidelity')
    plt.title('Instance spread and depth-averaged fidelity')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'instance_scatter_vs_depth.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=depth_summary, x='d', y='mean_xeb', marker='o', linewidth=2.5, label='Linear XEB')
    sns.lineplot(data=depth_summary, x='d', y='mean_mb', marker='s', linewidth=2.0, label='MB-regression surrogate')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel('Circuit depth d')
    plt.ylabel('Fidelity estimate')
    plt.title('Comparison of fidelity estimators across depth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'estimator_comparison.png', dpi=200)
    plt.close()

    # Simple gate-error back-fit model: exponential in depth
    ds = depth_summary['d'].to_numpy(dtype=float)
    ys = np.clip(depth_summary['mean_xeb'].to_numpy(dtype=float), 1e-6, None)
    coef = np.polyfit(ds, np.log(ys), 1)
    b, a = coef[0], coef[1]
    depth_summary['exp_model_fit'] = np.exp(a + b * depth_summary['d'])
    depth_summary['effective_error_per_cycle'] = float(1 - math.exp(b))
    depth_summary.to_csv(OUT_DIR / 'xeb_depth_summary.csv', index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=depth_summary, x='d', y='mean_xeb', marker='o', linewidth=2.5, label='Observed mean XEB')
    sns.lineplot(data=depth_summary, x='d', y='exp_model_fit', marker='D', linewidth=2.0, linestyle='--', label='Exponential error-propagation fit')
    plt.yscale('log')
    plt.xlabel('Circuit depth d')
    plt.ylabel('Fidelity estimate (log scale)')
    plt.title('Depth-dependent decay and effective error-propagation model')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'error_model_fit.png', dpi=200)
    plt.close()

    print('Wrote:', OUT_DIR / 'xeb_instance_results.csv')
    print('Wrote:', OUT_DIR / 'xeb_depth_summary.csv')
    print('Figures in', FIG_DIR)
    print(depth_summary[['d', 'mean_xeb', 'ci95', 'mean_mb', 'exp_model_fit', 'effective_error_per_cycle']].to_string(index=False))

if __name__ == '__main__':
    main()
