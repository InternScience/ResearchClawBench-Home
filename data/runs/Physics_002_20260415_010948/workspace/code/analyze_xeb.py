import os
import glob
import json
import math
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_amp_to_prob(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            c = complex(s)
            return (c.real * c.real + c.imag * c.imag)
        except Exception:
            try:
                return float(s)
            except Exception:
                return float(ast.literal_eval(s))
    raise TypeError(f'Unsupported amplitude value type: {type(v)}')


def main():
    result_paths = sorted(glob.glob('data/results/N40_verification/N40_d*_XEB/*_counts.json'))
    rows = []
    pat = re.compile(r'N(?P<N>\d+)_d(?P<d>\d+)_r(?P<r>\d+)_XEB')

    for rp in result_paths:
        m = pat.search(os.path.basename(rp))
        if not m:
            continue
        depth = int(m.group('d'))
        instance = int(m.group('r'))
        n_qubits = int(m.group('N'))
        ap = rp.replace('data/results', 'data/amplitudes').replace('_counts.json', '_amplitudes.json')
        if not os.path.exists(ap):
            continue

        with open(rp) as f:
            counts = json.load(f)
        with open(ap) as f:
            amps = json.load(f)

        matched_terms = []
        for bitstring, count in counts.items():
            if bitstring not in amps:
                continue
            p = parse_amp_to_prob(amps[bitstring])
            matched_terms.extend([2**n_qubits * p - 1] * int(count))

        matched_terms = np.asarray(matched_terms, dtype=float)
        f_xeb = float(matched_terms.mean())
        se_within = float(matched_terms.std(ddof=1) / math.sqrt(len(matched_terms))) if len(matched_terms) > 1 else 0.0
        rows.append({
            'N': n_qubits,
            'depth': depth,
            'instance': instance,
            'shots_matched': int(len(matched_terms)),
            'f_xeb': f_xeb,
            'se_within': se_within,
            'ci95_within': 1.96 * se_within,
            'mean_ideal_prob': float(((matched_terms + 1) / (2**n_qubits)).mean()),
            'median_term': float(np.median(matched_terms)),
            'min_term': float(matched_terms.min()),
            'max_term': float(matched_terms.max())
        })

    per = pd.DataFrame(rows).sort_values(['depth', 'instance'])
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    per.to_csv('outputs/per_instance_fidelity.csv', index=False)

    summary = (
        per.groupby('depth')
        .agg(
            N=('N', 'first'),
            n_instances=('instance', 'count'),
            mean_f_xeb=('f_xeb', 'mean'),
            std_across_instances=('f_xeb', 'std'),
            sem_across_instances=('f_xeb', lambda x: x.std(ddof=1) / math.sqrt(len(x))),
            mean_within_se=('se_within', 'mean'),
            median_f_xeb=('f_xeb', 'median'),
            q25_f_xeb=('f_xeb', lambda x: x.quantile(0.25)),
            q75_f_xeb=('f_xeb', lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    summary['classical_threshold'] = 2 ** (-summary['depth'] / 2)
    summary['gap_vs_threshold'] = summary['mean_f_xeb'] - summary['classical_threshold']
    summary.to_csv('outputs/depth_summary.csv', index=False)

    with open('outputs/analysis_summary.json', 'w') as f:
        json.dump({
            'n_instances': int(len(per)),
            'depths': [int(x) for x in summary['depth'].tolist()],
            'overall_mean_f_xeb': float(per['f_xeb'].mean()),
            'overall_std_f_xeb': float(per['f_xeb'].std(ddof=1)),
            'best_depth_by_mean': int(summary.loc[summary['mean_f_xeb'].idxmax(), 'depth']),
            'worst_depth_by_mean': int(summary.loc[summary['mean_f_xeb'].idxmin(), 'depth'])
        }, f, indent=2)

    sns.set_theme(style='whitegrid')

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(summary['depth'], summary['mean_f_xeb'], yerr=summary['sem_across_instances'], marker='o', capsize=4)
    plt.xlabel('Circuit depth d')
    plt.ylabel('Linear XEB fidelity')
    plt.title('N=40 random-circuit fidelity versus depth')
    plt.tight_layout()
    plt.savefig('report/images/fidelity_vs_depth.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    sns.boxplot(data=per, x='depth', y='f_xeb', color='skyblue')
    sns.stripplot(data=per, x='depth', y='f_xeb', color='black', alpha=0.45, size=3)
    plt.xlabel('Circuit depth d')
    plt.ylabel('Per-instance linear XEB fidelity')
    plt.title('Distribution of per-instance fidelities across depths')
    plt.tight_layout()
    plt.savefig('report/images/fidelity_distribution_by_depth.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(summary['depth'], summary['mean_f_xeb'], yerr=summary['sem_across_instances'], marker='o', capsize=4, label='Estimated experimental fidelity')
    plt.plot(summary['depth'], summary['classical_threshold'], marker='s', linestyle='--', label=r'Reference threshold $2^{-d/2}$')
    plt.yscale('log')
    plt.xlabel('Circuit depth d')
    plt.ylabel('Fidelity / reference scale')
    plt.title('Experimental fidelity and a depth-dependent reference threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/fidelity_vs_classical_threshold.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
