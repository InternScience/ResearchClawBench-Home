import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style='whitegrid', context='talk')
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'outputs')
IMG = os.path.join(BASE, 'report', 'images')
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

PARAM_BOUNDS = {
    'Qmax_Ah': (0.9, 1.2),
    'R0_Ohm': (0.01, 0.08),
    'kappa_V': (0.035, 0.16),
    'tau_s': (40.0, 450.0),
    'alpha_ocv': (0.9, 1.1),
    'beta_temp': (0.0, 0.02),
    'gamma_age': (0.0, 0.18),
}
PARAM_TO_PHYSICAL = {
    'Qmax_Ah': 'effective stoichiometric capacity / lithium inventory',
    'R0_Ohm': 'lumped ohmic + charge-transfer resistance',
    'kappa_V': 'polarization amplitude (proxy for kinetic limitation)',
    'tau_s': 'diffusion relaxation time (proxy for particle radius^2 / diffusivity)',
    'alpha_ocv': 'OCV curve scaling / active material utilization factor',
    'beta_temp': 'thermal-voltage coupling coefficient',
    'gamma_age': 'aging intensity factor',
}


def lhs(n, d, seed=42):
    rng = np.random.default_rng(seed)
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.random(n)) / n
    return result


def denorm(u):
    names = list(PARAM_BOUNDS)
    arr = np.zeros_like(u)
    for i, k in enumerate(names):
        lo, hi = PARAM_BOUNDS[k]
        arr[:, i] = lo + u[:, i] * (hi - lo)
    return pd.DataFrame(arr, columns=names)


def extract_cs2_curves():
    curves = []
    for fname in sorted(os.listdir(os.path.join(DATA, 'CS2_36'))):
        if not fname.endswith('.xlsx'):
            continue
        path = os.path.join(DATA, 'CS2_36', fname)
        df = pd.read_excel(path, sheet_name='Channel_1-009')
        discharge = df[df['Current(A)'] < -0.2].copy()
        for cyc, g in discharge.groupby('Cycle_Index'):
            g = g.sort_values('Test_Time(s)')
            t = g['Test_Time(s)'].to_numpy() - g['Test_Time(s)'].iloc[0]
            q = g['Discharge_Capacity(Ah)'].to_numpy()
            if len(t) < 20 or np.nanmax(q) < 0.5:
                continue
            curves.append({
                'dataset': 'CS2_36', 'source_file': fname, 'cycle': int(cyc),
                'time_s': t, 'current_a': g['Current(A)'].to_numpy(), 'voltage_v': g['Voltage(V)'].to_numpy(),
                'temp_c': np.full(len(t), 25.0), 'capacity_ah': q - q.min(),
            })
    return curves


def extract_nasa_curves():
    curves = []
    base = os.path.join(DATA, 'NASA PCoE Dataset Repository', '1. BatteryAgingARC-FY08Q4')
    for fname in sorted(os.listdir(base)):
        if not fname.endswith('.mat'):
            continue
        m = loadmat(os.path.join(base, fname), squeeze_me=True, struct_as_record=False)
        key = [k for k in m if not k.startswith('__')][0]
        cell = m[key]
        for idx, cyc in enumerate(np.atleast_1d(cell.cycle), start=1):
            if getattr(cyc, 'type', '') != 'discharge':
                continue
            d = cyc.data
            t = np.asarray(d.Time, dtype=float)
            v = np.asarray(d.Voltage_measured, dtype=float)
            i = np.asarray(d.Current_measured, dtype=float)
            temp = np.asarray(d.Temperature_measured, dtype=float)
            if len(t) < 20:
                continue
            cap = np.cumsum(np.maximum(-i, 0) * np.gradient(t) / 3600.0)
            curves.append({
                'dataset': 'NASA', 'source_file': fname, 'cycle': idx,
                'time_s': t - t[0], 'current_a': i, 'voltage_v': v, 'temp_c': temp,
                'capacity_ah': cap,
            })
    return curves


def extract_oxford_curves():
    curves = []
    path = os.path.join(DATA, 'Oxford Battery Degradation Dataset', 'ExampleDC_C1.mat')
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = [k for k in m if not k.startswith('__')][0]
    dc = m[key].dc
    t = np.asarray(dc.t, dtype=float)
    i = np.asarray(dc.i, dtype=float) / 1000.0
    v = np.asarray(dc.v, dtype=float)
    temp = np.asarray(dc.T, dtype=float)
    q = np.abs(np.asarray(dc.q, dtype=float)) / 1000.0
    curves.append({
        'dataset': 'Oxford', 'source_file': 'ExampleDC_C1.mat', 'cycle': 1,
        'time_s': t - t[0], 'current_a': i, 'voltage_v': v, 'temp_c': temp,
        'capacity_ah': q - q.min(),
    })
    return curves


def simulate_voltage(curve, p):
    t = np.asarray(curve['time_s'], dtype=float)
    i = np.asarray(curve['current_a'], dtype=float)
    temp = np.asarray(curve['temp_c'], dtype=float)
    q = np.asarray(curve['capacity_ah'], dtype=float)
    dt = np.diff(t, prepend=t[0])
    soc = 1 - np.clip(q / max(p['Qmax_Ah'], 1e-6), 0, 1.2)
    soc = np.clip(soc, 0.02, 0.98)
    z = np.zeros_like(t)
    for k in range(1, len(t)):
        z[k] = z[k-1] + dt[k] * (-z[k-1] / p['tau_s'] + abs(i[k]) / p['Qmax_Ah'])
    ocv = 3.0 + 1.18 * soc + 0.12 * np.tanh((soc - 0.78) / 0.08) - 0.1 * np.tanh((0.18 - soc) / 0.05)
    ocv *= p['alpha_ocv']
    aging_factor = 1 - p['gamma_age'] * min(curve['cycle'] / 200.0, 1.5)
    v = ocv - i * p['R0_Ohm'] * aging_factor - p['kappa_V'] * np.log1p(z) - p['beta_temp'] * (temp - np.nanmean(temp))
    return v


def summarize_curve(curve):
    t = curve['time_s']
    v = curve['voltage_v']
    q = curve['capacity_ah']
    temp = curve['temp_c']
    curr = curve['current_a']
    return {
        'dataset': curve['dataset'], 'source_file': curve['source_file'], 'cycle': curve['cycle'],
        'n_points': len(t), 'duration_s': float(t[-1] - t[0]), 'capacity_ah': float(np.nanmax(q)),
        'v_mean': float(np.nanmean(v)), 'v_min': float(np.nanmin(v)), 'v_max': float(np.nanmax(v)),
        'i_mean': float(np.nanmean(curr)), 'i_std': float(np.nanstd(curr)),
        'temp_mean': float(np.nanmean(temp)), 'temp_max': float(np.nanmax(temp)),
    }


def features_from_curve(curve):
    t = np.asarray(curve['time_s'])
    v = np.asarray(curve['voltage_v'])
    q = np.asarray(curve['capacity_ah'])
    temp = np.asarray(curve['temp_c'])
    i = np.asarray(curve['current_a'])
    frac = np.clip(q / max(np.nanmax(q), 1e-8), 0, 1)
    interp = interp1d(frac, v, bounds_error=False, fill_value='extrapolate')
    pts = np.linspace(0.05, 0.95, 10)
    vq = interp(pts)
    return np.r_[np.nanmax(q), np.nanmean(i), np.nanstd(i), np.nanmean(temp), np.nanmax(temp), vq]


def main():
    curves = extract_cs2_curves() + extract_nasa_curves() + extract_oxford_curves()
    summaries = pd.DataFrame([summarize_curve(c) for c in curves])
    summaries.to_csv(os.path.join(OUT, 'dataset_overview.csv'), index=False)

    cs2 = [c for c in curves if c['dataset'] == 'CS2_36']
    train_curves = cs2[:80]
    val_curves = cs2[80:120]
    target_curves = cs2[120:160]
    nasa_targets = [c for c in curves if c['dataset'] == 'NASA'][:20]
    oxford_target = [c for c in curves if c['dataset'] == 'Oxford'][:1]

    U = lhs(800, len(PARAM_BOUNDS), seed=7)
    param_df = denorm(U)
    X, Y = [], []
    representative = train_curves[:6]
    for _, row in param_df.iterrows():
        p = row.to_dict()
        feats = []
        for c in representative:
            sim = simulate_voltage(c, p)
            frac = np.clip(c['capacity_ah'] / max(np.nanmax(c['capacity_ah']), 1e-8), 0, 1)
            interp = interp1d(frac, sim, bounds_error=False, fill_value='extrapolate')
            feats.extend(interp(np.linspace(0.05, 0.95, 8)))
        X.append(feats)
        Y.append(row.values)
    X = np.asarray(X)
    Y = np.asarray(Y)

    xtr, xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=3000, random_state=0))
    ])
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    metrics = {}
    for i, name in enumerate(PARAM_BOUNDS):
        rmse = math.sqrt(mean_squared_error(yte[:, i], pred[:, i]))
        metrics[name] = {'rmse': rmse, 'r2': r2_score(yte[:, i], pred[:, i])}
    with open(os.path.join(OUT, 'surrogate_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    def estimate_params(curve):
        frac = np.clip(curve['capacity_ah'] / max(np.nanmax(curve['capacity_ah']), 1e-8), 0, 1)
        interp = interp1d(frac, curve['voltage_v'], bounds_error=False, fill_value='extrapolate')
        feat = []
        feat.extend(interp(np.linspace(0.05, 0.95, 8)))
        for c in representative[1:]:
            sim = c['voltage_v']
            frac2 = np.clip(c['capacity_ah'] / max(np.nanmax(c['capacity_ah']), 1e-8), 0, 1)
            feat.extend(interp1d(frac2, sim, bounds_error=False, fill_value='extrapolate')(np.linspace(0.05, 0.95, 8)))
        init = model.predict(np.asarray(feat).reshape(1, -1))[0]
        names = list(PARAM_BOUNDS)
        lbs = np.array([PARAM_BOUNDS[k][0] for k in names])
        ubs = np.array([PARAM_BOUNDS[k][1] for k in names])
        init = np.clip(init, lbs, ubs)
        def resid(x):
            p = dict(zip(names, x))
            sim = simulate_voltage(curve, p)
            v = curve['voltage_v']
            n = min(len(sim), len(v))
            return (sim[:n] - v[:n]) / max(np.std(v[:n]), 1e-6)
        res = least_squares(resid, init, bounds=(lbs, ubs), max_nfev=100)
        return dict(zip(names, res.x)), res.cost

    records = []
    for c in target_curves + nasa_targets + oxford_target:
        p, cost = estimate_params(c)
        sim = simulate_voltage(c, p)
        n = min(len(sim), len(c['voltage_v']))
        rmse = math.sqrt(mean_squared_error(c['voltage_v'][:n], sim[:n]))
        mae = mean_absolute_error(c['voltage_v'][:n], sim[:n])
        rec = {'dataset': c['dataset'], 'source_file': c['source_file'], 'cycle': c['cycle'], 'rmse_v': rmse, 'mae_v': mae, 'cost': cost}
        rec.update(p)
        records.append(rec)
    identified = pd.DataFrame(records)
    identified.to_csv(os.path.join(OUT, 'identified_parameters.csv'), index=False)

    grouped = identified.groupby('dataset')[list(PARAM_BOUNDS)].agg(['mean', 'std'])
    grouped.to_csv(os.path.join(OUT, 'identified_parameters_grouped.csv'))

    # figures
    plt.figure(figsize=(12,6))
    sns.countplot(data=summaries, x='dataset', order=summaries['dataset'].value_counts().index)
    plt.title('Dataset composition across validation sources')
    plt.tight_layout(); plt.savefig(os.path.join(IMG, 'dataset_counts.png'), dpi=200); plt.close()

    plt.figure(figsize=(12,6))
    for ds, g in summaries.groupby('dataset'):
        plt.scatter(g['cycle'], g['capacity_ah'], label=ds, alpha=0.7)
    plt.xlabel('Cycle index'); plt.ylabel('Discharge capacity (Ah)'); plt.title('Capacity trajectories across datasets')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(IMG, 'capacity_trajectories.png'), dpi=200); plt.close()

    example = target_curves[len(target_curves)//2]
    ex_params = identified[(identified['dataset']=='CS2_36')].iloc[len(target_curves)//2][list(PARAM_BOUNDS)].to_dict()
    sim = simulate_voltage(example, ex_params)
    plt.figure(figsize=(12,6))
    plt.plot(example['capacity_ah'], example['voltage_v'], label='Experiment', lw=3)
    plt.plot(example['capacity_ah'], sim, label='ANN-assisted identified model', lw=2)
    plt.xlabel('Discharge capacity (Ah)'); plt.ylabel('Voltage (V)'); plt.title('Representative CS2_36 discharge curve fit')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(IMG, 'cs2_fit_example.png'), dpi=200); plt.close()

    nasa = nasa_targets[min(5, len(nasa_targets)-1)]
    nasa_params = identified[identified['dataset']=='NASA'].iloc[min(5, len(identified[identified['dataset']== 'NASA'])-1)][list(PARAM_BOUNDS)].to_dict()
    sim = simulate_voltage(nasa, nasa_params)
    plt.figure(figsize=(12,6))
    plt.plot(nasa['time_s']/60, nasa['voltage_v'], label='NASA experiment', lw=3)
    plt.plot(nasa['time_s']/60, sim, label='Transfer fit', lw=2)
    plt.xlabel('Time (min)'); plt.ylabel('Voltage (V)'); plt.title('Transfer validation on NASA aging data')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(IMG, 'nasa_transfer_validation.png'), dpi=200); plt.close()

    ox = oxford_target[0]
    ox_params = identified[identified['dataset']=='Oxford'].iloc[0][list(PARAM_BOUNDS)].to_dict()
    sim = simulate_voltage(ox, ox_params)
    plt.figure(figsize=(12,6))
    plt.plot(ox['time_s']/60, ox['voltage_v'], label='Oxford dynamic experiment', lw=3)
    plt.plot(ox['time_s']/60, sim, label='Transfer fit', lw=2)
    plt.xlabel('Time (min)'); plt.ylabel('Voltage (V)'); plt.title('Dynamic-profile validation on Oxford dataset')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(IMG, 'oxford_dynamic_validation.png'), dpi=200); plt.close()

    metric_df = identified.groupby('dataset')[['rmse_v','mae_v']].mean().reset_index()
    plt.figure(figsize=(10,6))
    x = np.arange(len(metric_df))
    w = 0.35
    plt.bar(x-w/2, metric_df['rmse_v'], width=w, label='RMSE (V)')
    plt.bar(x+w/2, metric_df['mae_v'], width=w, label='MAE (V)')
    plt.xticks(x, metric_df['dataset'])
    plt.ylabel('Voltage error (V)')
    plt.title('Cross-dataset error summary of the identified surrogate framework')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(IMG, 'error_summary.png'), dpi=200); plt.close()

    param_long = identified.melt(id_vars=['dataset','cycle'], value_vars=list(PARAM_BOUNDS), var_name='parameter', value_name='value')
    plt.figure(figsize=(14,7))
    sns.boxplot(data=param_long, x='parameter', y='value', hue='dataset')
    plt.xticks(rotation=30, ha='right')
    plt.title('Distribution of identified latent ECAT proxy parameters')
    plt.tight_layout(); plt.savefig(os.path.join(IMG, 'identified_parameter_distributions.png'), dpi=200); plt.close()

    manifest = {
        'n_curves_total': len(curves), 'n_cs2': len(cs2), 'n_nasa': len([c for c in curves if c['dataset']=='NASA']),
        'n_oxford': len([c for c in curves if c['dataset']=='Oxford']), 'param_mapping': PARAM_TO_PHYSICAL,
        'average_errors': identified.groupby('dataset')[['rmse_v','mae_v']].mean().round(4).to_dict(),
    }
    with open(os.path.join(OUT, 'analysis_summary.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == '__main__':
    main()
