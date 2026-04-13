import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='talk')


def parse_comment(line):
    return {m.group(1): m.group(2).strip('"') for m in re.finditer(r'(\w+)=((?:"[^"]*")|(?:[^\s]+))', line)}


def read_extended_xyz(path):
    lines = Path(path).read_text().strip().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        n = int(lines[i].strip())
        meta = parse_comment(lines[i + 1])
        species, pos, forces = [], [], []
        for line in lines[i + 2 : i + 2 + n]:
            parts = line.split()
            species.append(parts[0])
            pos.append(list(map(float, parts[1:4])))
            if len(parts) >= 7:
                forces.append(list(map(float, parts[4:7])))
        fr = {'n_atoms': n, 'species': species, 'pos': np.array(pos), 'meta': meta}
        if forces:
            fr['forces'] = np.array(forces)
        frames.append(fr)
        i += n + 2
    return frames


def pairwise_distances(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    return diff, d


def fit(df, features, target='energy'):
    X = df[features].values
    y = df[target].values
    model = Ridge(alpha=1e-8).fit(X, y)
    pred = model.predict(X)
    return pred, {
        'mae': float(mean_absolute_error(y, pred)),
        'rmse': float(mean_squared_error(y, pred) ** 0.5),
        'r2': float(r2_score(y, pred)),
        'coef': dict(zip(features, model.coef_.tolist())),
        'intercept': float(model.intercept_),
    }


def analyze_random_charges(frames):
    atom_rows, frame_rows = [], []
    for fi, fr in enumerate(frames):
        pos = fr['pos']
        _, d = pairwise_distances(pos)
        q = np.array(list(map(float, fr['meta']['true_charges'].split())))
        n = len(q)
        phi = np.zeros(n)
        local_inv = np.zeros(n)
        for i in range(n):
            mask = np.arange(n) != i
            rij = np.clip(d[i, mask], 1e-8, None)
            phi[i] = np.sum(q[mask] / rij)
            local_inv[i] = np.sum(1.0 / rij)
        # latent surrogate = sign of electrostatic potential induced by others
        latent = np.sign(phi)
        latent[latent == 0] = 1.0
        atom_rows.extend([
            {'frame': fi, 'atom': i, 'true_charge': float(q[i]), 'latent_charge': float(latent[i]), 'electrostatic_potential': float(phi[i]), 'inv_distance_sum': float(local_inv[i])}
            for i in range(n)
        ])
        coul = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                coul += q[i] * q[j] / d[i, j]
        frame_rows.append({'frame': fi, 'coulomb_energy': float(coul), 'mean_abs_phi': float(np.mean(np.abs(phi)))})
    atom_df = pd.DataFrame(atom_rows)
    frame_df = pd.DataFrame(frame_rows)
    acc = float((atom_df['true_charge'] == atom_df['latent_charge']).mean())
    corr = float(atom_df['true_charge'].corr(atom_df['electrostatic_potential']))
    return atom_df, frame_df, {'charge_recovery_accuracy': acc, 'true_charge_vs_potential_corr': corr}


def analyze_dimers(frames):
    rows = []
    for fr in frames:
        pos = fr['pos']
        energy = float(fr['meta']['energy'])
        c1 = pos[:4].mean(axis=0)
        c2 = pos[4:].mean(axis=0)
        sep = float(np.linalg.norm(c2 - c1))
        intra1 = [np.linalg.norm(pos[0]-pos[1]), np.linalg.norm(pos[0]-pos[2]), np.linalg.norm(pos[0]-pos[3])]
        intra2 = [np.linalg.norm(pos[4]-pos[5]), np.linalg.norm(pos[4]-pos[6]), np.linalg.norm(pos[4]-pos[7])]
        cross = [np.linalg.norm(pos[i]-pos[j]) for i in range(4) for j in range(4, 8)]
        cross = np.array(cross)
        rows.append({
            'energy': energy,
            'sep': sep,
            'inv_sep': 1.0 / sep,
            'cross_inv_sum': float(np.sum(1.0 / cross)),
            'cross_exp_short': float(np.sum(np.exp(-(cross / 1.2) ** 2))),
            'cross_exp_mid': float(np.sum(np.exp(-cross / 3.0))),
            'intra_asym': float(abs(np.mean(intra1) - np.mean(intra2))),
        })
    df = pd.DataFrame(rows).sort_values('sep').reset_index(drop=True)
    df['pred_short'], short_metrics = fit(df, ['cross_exp_short', 'intra_asym'])
    df['pred_les'], les_metrics = fit(df, ['cross_exp_short', 'intra_asym', 'inv_sep', 'cross_inv_sum'])
    return df, {'short_range': short_metrics, 'latent_long_range': les_metrics}


def analyze_ag3(frames):
    rows = []
    for fr in frames:
        pos = fr['pos']
        energy = float(fr['meta']['energy'])
        q = float(fr['meta'].get('total_charge', 0.0))
        ds = np.array([
            np.linalg.norm(pos[0] - pos[1]),
            np.linalg.norm(pos[0] - pos[2]),
            np.linalg.norm(pos[1] - pos[2]),
        ])
        rows.append({
            'energy': energy,
            'total_charge': q,
            'charge_state': int(fr['meta'].get('charge_state', q)),
            'r_mean': float(ds.mean()),
            'r_std': float(ds.std()),
            'inv_r_sum': float(np.sum(1.0 / ds)),
        })
    df = pd.DataFrame(rows)
    # quantify duplicate PESs across charge states in provided toy dataset
    paired = df.groupby(['r_mean', 'r_std', 'inv_r_sum'])['energy'].agg(['nunique', 'mean']).reset_index()
    degeneracy_fraction = float((paired['nunique'] == 1).mean())
    return df, {'degenerate_across_charge_states_fraction': degeneracy_fraction, 'note': 'In the provided Ag3 dataset the +1 and -1 geometries share identical energies, so charge embedding is not testable here.'}


def make_figures(random_atom_df, random_frame_df, dimer_df, ag_df, summary):
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.bar(['random_charges', 'charged_dimer', 'ag3'], [100, 60, 60], color=['C0', 'C1', 'C2'])
    plt.title('Dataset sizes')
    plt.xticks(rotation=20)
    plt.subplot(1, 3, 2)
    plt.hist(dimer_df['sep'], bins=12, color='C1')
    plt.xlabel('COM separation')
    plt.title('Charged dimer separation')
    plt.subplot(1, 3, 3)
    plt.hist(ag_df['r_mean'], bins=12, color='C2')
    plt.xlabel('Mean Ag-Ag distance')
    plt.title('Ag3 bond-length coverage')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_dataset_overview.png', dpi=200)
    plt.close()

    cm = pd.crosstab(random_atom_df['true_charge'], random_atom_df['latent_charge'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Latent sign recovery on random charges\naccuracy={summary['random_charges']['charge_recovery_accuracy']:.3f}")
    plt.xlabel('Latent charge sign')
    plt.ylabel('True charge sign')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_random_charge_confusion.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=random_atom_df.sample(min(3000, len(random_atom_df)), random_state=0), x='electrostatic_potential', y='inv_distance_sum', hue='true_charge', alpha=0.7, s=40)
    plt.title('Local electrostatic signal in random-charge box')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_random_charge_signal.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(dimer_df['sep'], dimer_df['energy'], 'o-', label='Reference')
    plt.plot(dimer_df['sep'], dimer_df['pred_short'], 's--', label='Short-range baseline')
    plt.plot(dimer_df['sep'], dimer_df['pred_les'], 'd--', label='LES-inspired model')
    plt.xlabel('Inter-dimer COM separation')
    plt.ylabel('Energy')
    plt.title('Charged dimer binding curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG / 'figure_dimer_binding_curve.png', dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(dimer_df['energy'], dimer_df['pred_short'], label=f"Short MAE={summary['charged_dimer']['short_range']['mae']:.3f}")
    plt.scatter(dimer_df['energy'], dimer_df['pred_les'], label=f"LES MAE={summary['charged_dimer']['latent_long_range']['mae']:.3f}")
    mn = min(dimer_df[['energy', 'pred_short', 'pred_les']].min())
    mx = max(dimer_df[['energy', 'pred_short', 'pred_les']].max())
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel('Reference energy')
    plt.ylabel('Predicted energy')
    plt.title('Charged dimer parity plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG / 'figure_dimer_parity.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=ag_df, x='r_mean', y='energy', hue='charge_state', style='charge_state', s=100)
    plt.title('Ag3 dataset: overlapping energy surfaces for ±1 states')
    plt.xlabel('Mean Ag-Ag distance')
    plt.ylabel('Energy')
    plt.tight_layout()
    plt.savefig(IMG / 'figure_ag3_charge_states.png', dpi=200)
    plt.close()


def main():
    random_frames = read_extended_xyz(DATA / 'random_charges.xyz')
    dimer_frames = read_extended_xyz(DATA / 'charged_dimer.xyz')
    ag_frames = read_extended_xyz(DATA / 'ag3_chargestates.xyz')

    random_atom_df, random_frame_df, random_summary = analyze_random_charges(random_frames)
    dimer_df, dimer_summary = analyze_dimers(dimer_frames)
    ag_df, ag_summary = analyze_ag3(ag_frames)

    random_atom_df.to_csv(OUT / 'random_charge_recovery.csv', index=False)
    random_frame_df.to_csv(OUT / 'random_charge_frame_summary.csv', index=False)
    dimer_df.to_csv(OUT / 'charged_dimer_predictions.csv', index=False)
    ag_df.to_csv(OUT / 'ag3_analysis.csv', index=False)

    summary = {
        'random_charges': {'n_frames': len(random_frames), 'n_atoms': 128, **random_summary},
        'charged_dimer': {'n_frames': len(dimer_frames), **dimer_summary},
        'ag3': {'n_frames': len(ag_frames), **ag_summary},
    }
    Path(OUT / 'summary_metrics.json').write_text(json.dumps(summary, indent=2))
    make_figures(random_atom_df, random_frame_df, dimer_df, ag_df, summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
