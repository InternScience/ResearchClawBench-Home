import json, math, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='talk')


def parse_comment(line):
    pairs = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
    out = {}
    for k, v in pairs.items():
        out[k] = v[1:-1] if v.startswith('"') and v.endswith('"') else v
    return out


def read_xyz(path):
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            natoms = int(line.strip())
            meta = parse_comment(f.readline().strip())
            species, pos, forces = [], [], []
            for _ in range(natoms):
                parts = f.readline().split()
                species.append(parts[0])
                pos.append([float(x) for x in parts[1:4]])
                if len(parts) >= 7:
                    forces.append([float(x) for x in parts[4:7]])
            frame = {
                'species': species,
                'pos': np.array(pos, float),
                'meta': meta,
            }
            if forces:
                frame['forces'] = np.array(forces, float)
            if 'true_charges' in meta:
                frame['true_charges'] = np.array([float(x) for x in meta['true_charges'].split()], float)
            frames.append(frame)
    return frames


def pairwise_distances(pos):
    n = len(pos)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(np.linalg.norm(pos[i] - pos[j]))
    return np.array(vals)


def pairwise_inverse_distances(pos):
    d = pairwise_distances(pos)
    return 1.0 / d


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def eval_regression(X, y, alpha=1.0):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    loo = LeaveOneOut()
    pred = cross_val_predict(model, X, y, cv=loo)
    return {
        'mae': float(mean_absolute_error(y, pred)),
        'rmse': float(rmse(y, pred)),
        'r2': float(r2_score(y, pred)),
        'pred': pred.tolist(),
    }


def eval_force_regression(X, Y, alpha=1.0):
    # flattened force regression with shared geometric descriptor
    metrics = []
    preds = np.zeros_like(Y)
    for k in range(Y.shape[1]):
        out = eval_regression(X, Y[:, k], alpha=alpha)
        preds[:, k] = np.array(out['pred'])
        metrics.append(out)
    mae = float(np.mean(np.abs(Y - preds)))
    rmse_val = float(np.sqrt(np.mean((Y - preds) ** 2)))
    return {'mae': mae, 'rmse': rmse_val, 'pred': preds.tolist()}


def charged_dimer_analysis(frames):
    rows = []
    X_local, X_long, yE, yF = [], [], [], []
    for fr in frames:
        pos = fr['pos']
        molA, molB = pos[:4], pos[4:]
        intra = np.concatenate([pairwise_distances(molA), pairwise_distances(molB)])
        inter = np.array([np.linalg.norm(a - b) for a in molA for b in molB])
        com_sep = np.linalg.norm(molA.mean(0) - molB.mean(0))
        X_local.append(np.sort(intra))
        X_long.append(np.concatenate([np.sort(intra), np.sort(1.0 / inter)]))
        yE.append(float(fr['meta']['energy']))
        yF.append(fr['forces'].reshape(-1))
        rows.append({'center_separation': com_sep, 'energy': yE[-1], 'mean_inter_inv_r': float(np.mean(1.0/inter))})
    X_local = np.array(X_local)
    X_long = np.array(X_long)
    yE = np.array(yE)
    yF = np.array(yF)
    res_local_E = eval_regression(X_local, yE)
    res_long_E = eval_regression(X_long, yE)
    res_local_F = eval_force_regression(X_local, yF)
    res_long_F = eval_force_regression(X_long, yF)
    df = pd.DataFrame(rows)
    df['pred_local'] = res_local_E['pred']
    df['pred_long'] = res_long_E['pred']
    return {
        'table': df,
        'metrics': {
            'energy_local_only': {k: res_local_E[k] for k in ['mae','rmse','r2']},
            'energy_long_range_aware': {k: res_long_E[k] for k in ['mae','rmse','r2']},
            'forces_local_only': {k: res_local_F[k] for k in ['mae','rmse']},
            'forces_long_range_aware': {k: res_long_F[k] for k in ['mae','rmse']},
        }
    }


def ag3_analysis(frames):
    rows = []
    X_geom, X_geom_charge, yE, yCharge = [], [], [], []
    for fr in frames:
        pos = fr['pos']
        d = np.sort(pairwise_inverse_distances(pos))
        charge = int(fr['meta']['charge_state'])
        energy = float(fr['meta']['energy'])
        X_geom.append(d)
        X_geom_charge.append(np.concatenate([d, [charge]]))
        yE.append(energy)
        yCharge.append(1 if charge > 0 else 0)
        rows.append({'charge_state': charge, 'energy': energy, 'inv_r12': d[0], 'inv_r23': d[1], 'inv_r13': d[2]})
    X_geom = np.array(X_geom)
    X_geom_charge = np.array(X_geom_charge)
    yE = np.array(yE)
    yCharge = np.array(yCharge)
    geom_only = eval_regression(X_geom, yE)
    geom_plus_charge = eval_regression(X_geom_charge, yE)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pred_charge = cross_val_predict(clf, X_geom, yCharge, cv=LeaveOneOut())
    df = pd.DataFrame(rows)
    df['pred_geom_only'] = geom_only['pred']
    df['pred_geom_plus_charge'] = geom_plus_charge['pred']
    return {
        'table': df,
        'metrics': {
            'energy_geom_only': {k: geom_only[k] for k in ['mae','rmse','r2']},
            'energy_geom_plus_charge_state': {k: geom_plus_charge[k] for k in ['mae','rmse','r2']},
            'charge_state_from_geometry_accuracy': float(accuracy_score(yCharge, pred_charge))
        }
    }


def random_charges_analysis(frames):
    charges = np.stack([fr['true_charges'] for fr in frames])
    pos = np.stack([fr['pos'] for fr in frames])
    n_frames, n_atoms = charges.shape
    # Build atom-local geometry descriptors independent of atom index.
    feats, labels = [], []
    for f in range(n_frames):
        pf = pos[f]
        qf = charges[f]
        for i in range(n_atoms):
            d = np.linalg.norm(pf - pf[i], axis=1)
            d = d[np.arange(n_atoms) != i]
            near = np.sort(d)[:8]
            feats.append(np.concatenate([near, [near.mean(), near.std(), 1.0 / near.mean()]]))
            labels.append(1 if qf[i] > 0 else 0)
    X = np.array(feats)
    y = np.array(labels)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    yhat = cross_val_predict(clf, X, y, cv=5)
    overall_acc = float(accuracy_score(y, yhat))
    # Chance-level framewise assignment consistency after reshaping.
    pred = np.where(yhat > 0, 1.0, -1.0).reshape(n_frames, n_atoms)
    frame_acc = ((pred == charges).mean(axis=1))
    dipoles = np.einsum('fni,fn->fi', pos, charges)
    dipole_norm = np.linalg.norm(dipoles, axis=1)
    out = {
        'pooled_local_geometry_only_accuracy': overall_acc,
        'framewise_accuracy_mean': float(frame_acc.mean()),
        'framewise_accuracy_std': float(frame_acc.std()),
        'dipole_norm_mean': float(dipole_norm.mean()),
        'dipole_norm_std': float(dipole_norm.std()),
        'net_charge_unique': sorted(np.unique(charges.sum(axis=1)).tolist())
    }
    return out, dipole_norm


def make_figures(charged, ag3, dipole_norm):
    # dataset overview
    ds = pd.read_json(OUT / 'dataset_summary.json')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(ax=axes[0], data=ds, x='dataset', y='n_frames', palette='deep')
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].set_title('Number of configurations')
    sns.barplot(ax=axes[1], data=ds, x='dataset', y=ds['natoms_values'].apply(lambda x: x[0]), palette='muted')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].set_title('Atoms per configuration')
    force_flags = ds['has_forces'].astype(int)
    sns.barplot(ax=axes[2], x=ds['dataset'], y=force_flags, palette='pastel')
    axes[2].tick_params(axis='x', rotation=25)
    axes[2].set_yticks([0,1])
    axes[2].set_yticklabels(['no','yes'])
    axes[2].set_title('Force labels available')
    fig.tight_layout()
    fig.savefig(IMG / 'dataset_overview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # charged dimer binding curve
    df = charged['table'].sort_values('center_separation')
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(df['center_separation'], df['energy'], 'o-', label='Reference energy')
    ax.plot(df['center_separation'], df['pred_local'], 's--', label='Local-only ridge')
    ax.plot(df['center_separation'], df['pred_long'], 'd--', label='Long-range-aware ridge')
    ax.set_xlabel('Center-of-mass separation (Å)')
    ax.set_ylabel('Energy (arb. units)')
    ax.set_title('Charged dimers: separation-dependent energy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / 'charged_dimer_binding_curve.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ag3 plot
    df = ag3['table'].copy()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df, x='inv_r13', y='energy', hue='charge_state', palette='coolwarm', s=80, ax=ax)
    sns.lineplot(data=df.sort_values('inv_r13'), x='inv_r13', y='pred_geom_only', color='black', label='Geom-only LOO pred', ax=ax)
    sns.lineplot(data=df.sort_values('inv_r13'), x='inv_r13', y='pred_geom_plus_charge', color='green', label='Geom+charge-state LOO pred', ax=ax)
    ax.set_xlabel('Largest inverse Ag–Ag distance (Å$^{-1}$)')
    ax.set_ylabel('Energy (arb. units)')
    ax.set_title('Ag$_3$ charge states require global charge information')
    fig.tight_layout()
    fig.savefig(IMG / 'ag3_charge_state_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # random charges / dipole figure
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    sns.histplot(dipole_norm, bins=20, kde=True, ax=axes[0], color='purple')
    axes[0].set_title('Random-charges dipole norm distribution')
    axes[0].set_xlabel('Dipole norm (eÅ)')
    metrics = json.loads((OUT / 'random_charges_metrics.json').read_text())
    keys = ['pooled_local_geometry_only_accuracy','framewise_accuracy_mean']
    vals = [metrics[k] for k in keys]
    sns.barplot(x=['pooled local acc.','framewise acc.'], y=vals, ax=axes[1], color='gray')
    axes[1].axhline(0.5, color='red', ls='--', lw=2, label='chance')
    axes[1].set_ylim(0,1)
    axes[1].set_title('Geometry alone does not reveal random labels')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMG / 'random_charges_interpretability.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    charged_frames = read_xyz(DATA / 'charged_dimer.xyz')
    ag3_frames = read_xyz(DATA / 'ag3_chargestates.xyz')
    random_frames = read_xyz(DATA / 'random_charges.xyz')

    charged = charged_dimer_analysis(charged_frames)
    ag3 = ag3_analysis(ag3_frames)
    random_metrics, dipole_norm = random_charges_analysis(random_frames)

    (OUT / 'charged_dimer_metrics.json').write_text(json.dumps(charged['metrics'], indent=2))
    charged['table'].to_csv(OUT / 'charged_dimer_predictions.csv', index=False)
    (OUT / 'ag3_metrics.json').write_text(json.dumps(ag3['metrics'], indent=2))
    ag3['table'].to_csv(OUT / 'ag3_predictions.csv', index=False)
    (OUT / 'random_charges_metrics.json').write_text(json.dumps(random_metrics, indent=2))
    pd.DataFrame({'dipole_norm': dipole_norm}).to_csv(OUT / 'random_charges_dipoles.csv', index=False)

    metrics_table = pd.DataFrame([
        {'dataset': 'charged_dimer', 'model': 'local_only_energy', **charged['metrics']['energy_local_only']},
        {'dataset': 'charged_dimer', 'model': 'long_range_aware_energy', **charged['metrics']['energy_long_range_aware']},
        {'dataset': 'charged_dimer', 'model': 'local_only_forces', **charged['metrics']['forces_local_only']},
        {'dataset': 'charged_dimer', 'model': 'long_range_aware_forces', **charged['metrics']['forces_long_range_aware']},
        {'dataset': 'ag3_chargestates', 'model': 'geom_only_energy', **ag3['metrics']['energy_geom_only']},
        {'dataset': 'ag3_chargestates', 'model': 'geom_plus_charge_state_energy', **ag3['metrics']['energy_geom_plus_charge_state']},
        {'dataset': 'ag3_chargestates', 'model': 'geometry_to_charge_state_classifier', 'accuracy': ag3['metrics']['charge_state_from_geometry_accuracy']},
        {'dataset': 'random_charges', 'model': 'geometry_only_charge_guess', 'accuracy': random_metrics['pooled_local_geometry_only_accuracy']},
    ])
    metrics_table.to_csv(OUT / 'metrics_summary.csv', index=False)

    make_figures(charged, ag3, dipole_norm)

    claim_recovery = [
        {
            'claim': 'Long-range-aware descriptors improve charged-dimer energy prediction relative to local-only descriptors.',
            'artifact': 'outputs/charged_dimer_metrics.json; report/images/charged_dimer_binding_curve.png',
            'status': 'supported' if charged['metrics']['energy_long_range_aware']['mae'] < charged['metrics']['energy_local_only']['mae'] else 'not_supported'
        },
        {
            'claim': 'Charge-state information materially improves Ag3 energy prediction.',
            'artifact': 'outputs/ag3_metrics.json; report/images/ag3_charge_state_comparison.png',
            'status': 'supported' if ag3['metrics']['energy_geom_plus_charge_state']['mae'] < ag3['metrics']['energy_geom_only']['mae'] else 'not_supported'
        },
        {
            'claim': 'Random charge labels cannot be recovered from geometry alone in the provided random_charges dataset.',
            'artifact': 'outputs/random_charges_metrics.json; report/images/random_charges_interpretability.png',
            'status': 'supported' if random_metrics['pooled_local_geometry_only_accuracy'] < 0.6 else 'not_supported'
        }
    ]
    (OUT / 'claim_recovery_table.json').write_text(json.dumps(claim_recovery, indent=2))

if __name__ == '__main__':
    main()
