import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')


def is_protein_residue(res):
    hetflag = res.id[0]
    if hetflag.strip():
        return False
    atoms = {a.get_name() for a in res.get_atoms()}
    return 'CA' in atoms


def parse_structure(path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    chains = {}
    for chain in model:
        residues = [res for res in chain if is_protein_residue(res)]
        if not residues:
            continue
        coords = np.array([res['CA'].coord.astype(float) for res in residues], dtype=float)
        res_ids = [f"{res.id[1]}{res.id[2].strip()}" for res in residues]
        chains[chain.id] = {
            'coords': coords,
            'res_ids': res_ids,
            'length': len(residues),
        }
    return chains


def center_coords(coords):
    centroid = coords.mean(axis=0)
    return coords - centroid, centroid


def kabsch(P, Q):
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t


def apply_transform(coords, R, t):
    return coords @ R + t


def residue_distance_matrix(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def needleman_wunsch_score(D, gap_penalty=-1.0, d0=3.0):
    m, n = D.shape
    S = 1.0 / (1.0 + (D / d0) ** 2)
    F = np.zeros((m + 1, n + 1), dtype=float)
    TB = np.zeros((m + 1, n + 1), dtype=np.int8)
    for i in range(1, m + 1):
        F[i, 0] = F[i - 1, 0] + gap_penalty
        TB[i, 0] = 1
    for j in range(1, n + 1):
        F[0, j] = F[0, j - 1] + gap_penalty
        TB[0, j] = 2
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            vals = [
                F[i - 1, j - 1] + S[i - 1, j - 1],
                F[i - 1, j] + gap_penalty,
                F[i, j - 1] + gap_penalty,
            ]
            tb = int(np.argmax(vals))
            F[i, j] = vals[tb]
            TB[i, j] = tb
    i, j = m, n
    pairs = []
    while i > 0 or j > 0:
        tb = TB[i, j]
        if tb == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs, F[m, n]


def tm_score(distances, Lnorm):
    if Lnorm <= 0 or len(distances) == 0:
        return 0.0
    d0 = max(0.5, 1.24 * ((max(Lnorm, 15) - 15) ** (1/3)) - 1.8)
    return float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / Lnorm)


def align_chain_pair(c1, c2, max_iter=5):
    A = c1['coords']
    B = c2['coords']
    # initialize by centered truncation
    k = min(len(A), len(B))
    A0 = A[:k]
    B0 = B[:k]
    R, t = kabsch(A0 - A0.mean(0), B0 - B0.mean(0))
    for _ in range(max_iter):
        A_t = apply_transform(A, R, t)
        D = residue_distance_matrix(A_t, B)
        pairs, _ = needleman_wunsch_score(D, gap_penalty=-0.6, d0=4.0)
        if len(pairs) < 3:
            break
        P = np.array([A[i] for i, j in pairs], dtype=float)
        Q = np.array([B[j] for i, j in pairs], dtype=float)
        Pc = P - P.mean(0)
        Qc = Q - Q.mean(0)
        R, t = kabsch(Pc, Qc)
        t = Q.mean(0) - P.mean(0) @ R
    A_t = apply_transform(A, R, t)
    D = residue_distance_matrix(A_t, B)
    pairs, score = needleman_wunsch_score(D, gap_penalty=-0.6, d0=4.0)
    if pairs:
        distances = np.array([np.linalg.norm(A_t[i] - B[j]) for i, j in pairs])
        rmsd = float(np.sqrt(np.mean(distances ** 2)))
    else:
        distances = np.array([])
        rmsd = float('nan')
    return {
        'pairs': pairs,
        'R': R,
        't': t,
        'aligned_len': len(pairs),
        'rmsd': rmsd,
        'tm_q': tm_score(distances, len(A)),
        'tm_t': tm_score(distances, len(B)),
        'score': score,
        'distances': distances.tolist(),
    }


def main():
    q = parse_structure(DATA / '7xg4.pdb')
    t = parse_structure(DATA / '6n40.pdb')

    q_chains = sorted(q)
    t_chains = sorted(t)

    pair_results = {}
    rows = []
    cost = np.full((len(q_chains), len(t_chains)), 1e6, dtype=float)
    for i, qc in enumerate(q_chains):
        for j, tc in enumerate(t_chains):
            res = align_chain_pair(q[qc], t[tc])
            pair_results[(qc, tc)] = res
            avg_tm = 0.5 * (res['tm_q'] + res['tm_t'])
            rows.append({
                'query_chain': qc,
                'target_chain': tc,
                'query_len': q[qc]['length'],
                'target_len': t[tc]['length'],
                'aligned_len': res['aligned_len'],
                'rmsd': res['rmsd'],
                'tm_query_norm': res['tm_q'],
                'tm_target_norm': res['tm_t'],
                'tm_avg': avg_tm,
                'dp_score': res['score'],
            })
            cost[i, j] = -avg_tm

    df = pd.DataFrame(rows).sort_values('tm_avg', ascending=False)
    df.to_csv(OUT / 'chain_pair_metrics.csv', index=False)

    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = []
    matched = []
    for i, j in zip(row_ind, col_ind):
        qc = q_chains[i]
        tc = t_chains[j]
        res = pair_results[(qc, tc)]
        avg_tm = 0.5 * (res['tm_q'] + res['tm_t'])
        if avg_tm <= 0.10:
            continue
        assignment.append({
            'query_chain': qc,
            'target_chain': tc,
            'tm_avg': avg_tm,
            'aligned_len': res['aligned_len'],
            'rmsd': res['rmsd'],
            'tm_query_norm': res['tm_q'],
            'tm_target_norm': res['tm_t'],
            'rotation_matrix': np.array(res['R']).round(6).tolist(),
            'translation_vector': np.array(res['t']).round(6).tolist(),
            'residue_pairs': [
                {
                    'query_residue': q[qc]['res_ids'][a],
                    'target_residue': t[tc]['res_ids'][b],
                    'distance_after_superposition': round(float(d), 3)
                }
                for (a, b), d in zip(res['pairs'], res['distances'])
            ]
        })
        matched.append((qc, tc, res))

    summary = {
        'query_chain_count': len(q_chains),
        'target_chain_count': len(t_chains),
        'query_protein_chains': q_chains,
        'target_protein_chains': t_chains,
        'selected_chain_correspondence_count': len(assignment),
        'selected_chain_correspondence': assignment,
    }
    (OUT / 'alignment_summary.json').write_text(json.dumps(summary, indent=2))

    # Figures
    plt.figure(figsize=(8, 4.8))
    pivot = df.pivot(index='query_chain', columns='target_chain', values='tm_avg').reindex(index=q_chains, columns=t_chains)
    sns.heatmap(pivot, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Average TM-score'})
    plt.title('Chain-to-chain structural similarity: 7xg4 vs 6n40')
    plt.xlabel('Target chain')
    plt.ylabel('Query chain')
    plt.tight_layout()
    plt.savefig(IMG / 'chain_tm_heatmap.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plot_df = df.copy()
    plot_df['pair'] = plot_df['query_chain'] + '-' + plot_df['target_chain']
    sns.barplot(data=plot_df.sort_values('tm_avg', ascending=False).head(12), x='pair', y='tm_avg', color='#4C72B0')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average TM-score')
    plt.xlabel('Chain pair')
    plt.title('Top chain-pair alignments')
    plt.tight_layout()
    plt.savefig(IMG / 'top_chain_pairs.png', dpi=200)
    plt.close()

    selected_rows = []
    for qc, tc, res in matched:
        d = np.array(res['distances'])
        if len(d) == 0:
            continue
        selected_rows.extend([{'pair': f'{qc}-{tc}', 'distance': x} for x in d])
    if selected_rows:
        plt.figure(figsize=(8, 4.8))
        sel_df = pd.DataFrame(selected_rows)
        sns.boxplot(data=sel_df, x='pair', y='distance', color='#55A868')
        plt.ylabel('Cα distance after superposition (Å)')
        plt.xlabel('Selected chain pair')
        plt.title('Residue-level distance distributions for selected correspondences')
        plt.tight_layout()
        plt.savefig(IMG / 'selected_pair_distance_boxplot.png', dpi=200)
        plt.close()

    # Data overview
    overview = []
    for label, chains in [('7xg4', q), ('6n40', t)]:
        for cid, obj in chains.items():
            overview.append({'structure': label, 'chain': cid, 'length': obj['length']})
    ov = pd.DataFrame(overview)
    plt.figure(figsize=(10, 4.8))
    sns.barplot(data=ov, x='chain', y='length', hue='structure')
    plt.ylabel('Protein chain length (residues)')
    plt.xlabel('Chain ID')
    plt.title('Protein-chain composition of the two input structures')
    plt.tight_layout()
    plt.savefig(IMG / 'chain_length_overview.png', dpi=200)
    plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
