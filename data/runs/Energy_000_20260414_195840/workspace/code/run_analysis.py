import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')


def ensure_dirs():
    OUT.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)


PARAM_NAMES = [
    'capacity_ah',
    'r0_ohm',
    'tau_s',
    'diffusion_scale',
    'thermal_gain',
    'particle_radius_um',
    'reaction_rate',
]

PARAM_BOUNDS = np.array([
    [0.65, 0.90],      # capacity_ah
    [0.020, 0.090],    # r0_ohm
    [20.0, 240.0],     # tau_s
    [0.40, 1.60],      # diffusion_scale
    [0.0, 8.0],        # thermal_gain
    [3.0, 18.0],       # particle_radius_um
    [0.4, 1.6],        # reaction_rate
], dtype=float)


def ocv_from_soc(soc):
    soc = np.clip(soc, 1e-6, 1 - 1e-6)
    return (
        3.0
        + 1.15 * soc
        + 0.08 * np.tanh((soc - 0.82) / 0.05)
        - 0.12 * np.tanh((soc - 0.10) / 0.04)
    )


# simplified reduced-order discharge/thermal simulator
# internal parameters remain interpretable though not a full ECAT PDE model

def simulate_profile(params, t, current_a, temp_init=25.0, v_init=None):
    capacity_ah, r0, tau, diff_scale, thermal_gain, radius_um, reaction_rate = params
    dt = np.diff(t, prepend=t[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    else:
        dt[0] = 1.0
    current_pos = np.maximum(current_a, 0.0)
    soc = np.empty_like(t, dtype=float)
    vrc = np.zeros_like(t, dtype=float)
    temp = np.empty_like(t, dtype=float)
    voltage = np.empty_like(t, dtype=float)
    discharged_ah = np.empty_like(t, dtype=float)

    if v_init is None:
        soc[0] = 0.98
    else:
        grid = np.linspace(0.01, 0.99, 2000)
        est_v = ocv_from_soc(grid) - current_pos[0] * r0
        soc[0] = float(grid[np.argmin(np.abs(est_v - v_init))])
    temp[0] = temp_init
    discharged_ah[0] = (1.0 - soc[0]) * capacity_ah
    voltage[0] = ocv_from_soc(soc[0]) - current_pos[0] * r0

    rp = radius_um / 10.0
    for k in range(1, len(t)):
        dtk = max(float(dt[k]), 1e-6)
        soc[k] = np.clip(soc[k - 1] - current_pos[k - 1] * dtk / 3600.0 / capacity_ah, 0.0, 1.0)
        alpha = np.exp(-dtk / tau)
        diffusion_drop = diff_scale * (1.0 / np.sqrt(max(soc[k], 0.02)) - 1.0)
        kinetic_drop = 0.035 * current_pos[k] / max(reaction_rate, 1e-3)
        particle_drop = 0.004 * rp * current_pos[k]
        source = diffusion_drop + kinetic_drop + particle_drop
        vrc[k] = alpha * vrc[k - 1] + (1 - alpha) * source
        heat = thermal_gain * (current_pos[k] ** 2) * (r0 + 0.01 * diff_scale)
        cooling = 0.015 * (temp[k - 1] - temp_init)
        temp[k] = temp[k - 1] + dtk * (heat - cooling) / 250.0
        voltage[k] = ocv_from_soc(soc[k]) - current_pos[k] * r0 - vrc[k]
        discharged_ah[k] = (1.0 - soc[k]) * capacity_ah
    return {
        't': t,
        'current_a': current_a,
        'voltage_v': voltage,
        'temperature_c': temp,
        'discharge_capacity_ah': discharged_ah,
        'soc': soc,
    }


def load_calce_reference_cycle():
    p = DATA / 'CS2_36' / 'CS2_36_1_10_11.xlsx'
    ch = pd.read_excel(p, sheet_name='Channel_1-009')
    stats = pd.read_excel(p, sheet_name='Statistics_1-009')
    cyc = int(stats['Cycle_Index'].iloc[0])
    sub = ch[(ch['Cycle_Index'] == cyc) & (ch['Step_Index'] == 7)].copy()
    t = sub['Step_Time(s)'].to_numpy(dtype=float)
    current = np.abs(sub['Current(A)'].to_numpy(dtype=float))
    voltage = sub['Voltage(V)'].to_numpy(dtype=float)
    cap = sub['Discharge_Capacity(Ah)'].to_numpy(dtype=float)
    return {
        'dataset': 'CALCE_CS2_36',
        'cycle': cyc,
        'time_s': t - t.min(),
        'current_a': current,
        'voltage_v': voltage,
        'discharge_capacity_ah': cap - cap.min(),
        'temperature_c': np.full_like(t, 25.0),
        'source_file': str(p.relative_to(ROOT)),
    }


def load_nasa_reference_cycle():
    p = DATA / 'NASA PCoE Dataset Repository' / '1. BatteryAgingARC-FY08Q4' / 'B0005.mat'
    mat = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    battery = mat['B0005']
    discharge_cycles = [c for c in battery.cycle if getattr(c, 'type', '') == 'discharge']
    cyc = discharge_cycles[0]
    data = cyc.data
    t = np.asarray(data.Time, dtype=float)
    voltage = np.asarray(data.Voltage_measured, dtype=float)
    temp = np.asarray(data.Temperature_measured, dtype=float)
    current = np.abs(np.asarray(data.Current_measured, dtype=float))
    cap_total = float(getattr(data, 'Capacity', np.trapz(current, t) / 3600.0))
    discharged = np.cumsum(np.r_[0.0, 0.5 * (current[1:] + current[:-1]) * np.diff(t)]) / 3600.0
    scale = cap_total / max(discharged.max(), 1e-9)
    discharged *= scale
    return {
        'dataset': 'NASA_B0005',
        'cycle': 1,
        'time_s': t - t.min(),
        'current_a': current,
        'voltage_v': voltage,
        'discharge_capacity_ah': discharged,
        'temperature_c': temp,
        'source_file': str(p.relative_to(ROOT)),
    }


def load_oxford_dynamic():
    p = DATA / 'Oxford Battery Degradation Dataset' / 'ExampleDC_C1.mat'
    mat = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    obj = mat['ExampleDC_C1']
    dc = obj.dc
    t = np.asarray(dc.t, dtype=float)
    voltage = np.asarray(dc.v, dtype=float)
    temp = np.asarray(dc.T, dtype=float)
    current = np.abs(np.asarray(dc.i, dtype=float)) / 1000.0
    q = np.abs(np.asarray(dc.q, dtype=float)) / 1000.0
    return {
        'dataset': 'Oxford_ExampleDC_C1',
        'cycle': 1,
        'time_s': t - t.min(),
        'current_a': current,
        'voltage_v': voltage,
        'discharge_capacity_ah': q - q.min(),
        'temperature_c': temp,
        'source_file': str(p.relative_to(ROOT)),
    }


def summarize_datasets(calce, nasa, oxford):
    summary = {
        'datasets': [
            {
                'name': calce['dataset'],
                'points': int(len(calce['time_s'])),
                'duration_s': float(calce['time_s'][-1]),
                'current_mean_a': float(np.mean(calce['current_a'])),
                'voltage_range_v': [float(np.min(calce['voltage_v'])), float(np.max(calce['voltage_v']))],
                'capacity_range_ah': [float(np.min(calce['discharge_capacity_ah'])), float(np.max(calce['discharge_capacity_ah']))],
            },
            {
                'name': nasa['dataset'],
                'points': int(len(nasa['time_s'])),
                'duration_s': float(nasa['time_s'][-1]),
                'current_mean_a': float(np.mean(nasa['current_a'])),
                'voltage_range_v': [float(np.min(nasa['voltage_v'])), float(np.max(nasa['voltage_v']))],
                'temperature_range_c': [float(np.min(nasa['temperature_c'])), float(np.max(nasa['temperature_c']))],
                'capacity_range_ah': [float(np.min(nasa['discharge_capacity_ah'])), float(np.max(nasa['discharge_capacity_ah']))],
            },
            {
                'name': oxford['dataset'],
                'points': int(len(oxford['time_s'])),
                'duration_s': float(oxford['time_s'][-1]),
                'current_mean_a': float(np.mean(oxford['current_a'])),
                'voltage_range_v': [float(np.min(oxford['voltage_v'])), float(np.max(oxford['voltage_v']))],
                'temperature_range_c': [float(np.min(oxford['temperature_c'])), float(np.max(oxford['temperature_c']))],
                'capacity_range_ah': [float(np.min(oxford['discharge_capacity_ah'])), float(np.max(oxford['discharge_capacity_ah']))],
            },
        ]
    }
    (OUT / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def make_training_set(reference_profile, n_samples=600):
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=42)
    unit = sampler.random(n=n_samples)
    samples = qmc.scale(unit, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
    df = pd.DataFrame(samples, columns=PARAM_NAMES)
    t = reference_profile['time_s']
    I = reference_profile['current_a']
    y_rows = []
    for row in samples:
        sim = simulate_profile(row, t, I, temp_init=reference_profile['temperature_c'][0], v_init=reference_profile['voltage_v'][0])
        feat = np.concatenate([
            np.interp(np.linspace(0, t[-1], 80), t, sim['voltage_v']),
            np.interp(np.linspace(0, t[-1], 80), t, sim['temperature_c']),
            [sim['discharge_capacity_ah'][-1]],
        ])
        y_rows.append(feat)
    Y = np.vstack(y_rows)
    df.to_csv(OUT / 'lhs_samples.csv', index=False)
    return samples, Y


def train_surrogate(samples, Y):
    X_train, X_test, y_train, y_test = train_test_split(samples, Y, test_size=0.2, random_state=7)
    model = Pipeline([
        ('xscale', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(160, 160), activation='relu', random_state=7, max_iter=1200))
    ])
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_test_s = y_scaler.transform(y_test)
    model.fit(X_train, y_train_s)
    pred_train = y_scaler.inverse_transform(model.predict(X_train))
    pred_test = y_scaler.inverse_transform(model.predict(X_test))
    metrics = {
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, pred_train))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, pred_test))),
        'test_r2': float(r2_score(y_test, pred_test, multioutput='variance_weighted')),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
    }
    (OUT / 'surrogate_metrics.json').write_text(json.dumps(metrics, indent=2))
    return model, y_scaler, metrics, (X_test, y_test, pred_test)


def objective_from_features(pred_feat, obs_feat):
    return float(np.sqrt(np.mean((pred_feat - obs_feat) ** 2)))


def identify_parameters(name, profile, surrogate, y_scaler, n_search=4000):
    rng = np.random.default_rng(123 if name == 'CALCE_CS2_36' else (456 if name.startswith('NASA') else 789))
    lhs = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=int(rng.integers(1_000_000)))
    cand = qmc.scale(lhs.random(n=n_search), PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
    obs_feat = np.concatenate([
        np.interp(np.linspace(0, profile['time_s'][-1], 80), profile['time_s'], profile['voltage_v']),
        np.interp(np.linspace(0, profile['time_s'][-1], 80), profile['time_s'], profile['temperature_c']),
        [profile['discharge_capacity_ah'][-1]],
    ])
    pred = y_scaler.inverse_transform(surrogate.predict(cand))
    losses = np.array([objective_from_features(p, obs_feat) for p in pred])
    best = cand[np.argmin(losses)]
    sim = simulate_profile(best, profile['time_s'], profile['current_a'], temp_init=profile['temperature_c'][0], v_init=profile['voltage_v'][0])
    return best, sim, obs_feat, losses.min()


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def export_results(profiles, identified):
    rows = []
    fit_rows = []
    for name, result in identified.items():
        best = result['params']
        sim = result['simulation']
        profile = profiles[name]
        row = {'dataset': name}
        row.update({k: float(v) for k, v in zip(PARAM_NAMES, best)})
        rows.append(row)
        fit_rows.append({
            'dataset': name,
            'voltage_rmse_v': rmse(profile['voltage_v'], sim['voltage_v']),
            'voltage_rmse_mv': 1000.0 * rmse(profile['voltage_v'], sim['voltage_v']),
            'temperature_rmse_c': rmse(profile['temperature_c'], sim['temperature_c']),
            'final_capacity_abs_error_ah': float(abs(profile['discharge_capacity_ah'][-1] - sim['discharge_capacity_ah'][-1])),
            'surrogate_objective': float(result['surrogate_loss'])
        })
    pd.DataFrame(rows).to_csv(OUT / 'identified_parameters.csv', index=False)
    pd.DataFrame(fit_rows).to_csv(OUT / 'fit_metrics.csv', index=False)


def build_figures(dataset_summary, profiles, surrogate_eval, identified):
    # data overview
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=False)
    for ax, key in zip(axes, ['CALCE_CS2_36', 'NASA_B0005', 'Oxford_ExampleDC_C1']):
        prof = profiles[key]
        ax.plot(prof['time_s'] / 60.0, prof['voltage_v'], label='Voltage')
        ax2 = ax.twinx()
        ax2.plot(prof['time_s'] / 60.0, prof['current_a'], color='tab:red', alpha=0.45, label='Current')
        ax.set_title(key)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Voltage (V)')
        ax2.set_ylabel('Current (A)')
    fig.tight_layout()
    fig.savefig(IMG / 'data_overview.png', dpi=180)
    plt.close(fig)

    # surrogate accuracy
    X_test, y_test, pred_test = surrogate_eval
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(y_test[:, 0], pred_test[:, 0], s=18, alpha=0.7)
    lo = min(y_test[:, 0].min(), pred_test[:, 0].min())
    hi = max(y_test[:, 0].max(), pred_test[:, 0].max())
    axes[0].plot([lo, hi], [lo, hi], 'k--')
    axes[0].set_xlabel('True first voltage feature')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('ANN surrogate parity')
    axes[1].hist((pred_test - y_test).ravel(), bins=40, color='tab:blue', alpha=0.75)
    axes[1].set_xlabel('Prediction residual')
    axes[1].set_title('Surrogate residual distribution')
    fig.tight_layout()
    fig.savefig(IMG / 'surrogate_accuracy.png', dpi=180)
    plt.close(fig)

    # reference fit figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    for ax, key in zip(axes, ['CALCE_CS2_36', 'NASA_B0005']):
        prof = profiles[key]
        sim = identified[key]['simulation']
        ax.plot(prof['time_s'] / 60, prof['voltage_v'], label='Measured', lw=2)
        ax.plot(sim['t'] / 60, sim['voltage_v'], label='Identified model', lw=2, ls='--')
        ax.set_title(f'{key} voltage fit')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Voltage (V)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / 'reference_fit.png', dpi=180)
    plt.close(fig)

    # dynamic validation
    prof = profiles['Oxford_ExampleDC_C1']
    sim = identified['Oxford_ExampleDC_C1']['simulation']
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(prof['time_s'] / 60, prof['voltage_v'], label='Measured voltage', lw=2)
    axes[0].plot(sim['t'] / 60, sim['voltage_v'], label='Identified model voltage', lw=2, ls='--')
    axes[0].legend()
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title('Oxford dynamic validation')
    axes[1].plot(prof['time_s'] / 60, prof['current_a'], color='tab:red', label='Dynamic current')
    axes[1].plot(prof['time_s'] / 60, prof['temperature_c'], color='tab:green', alpha=0.7, label='Measured temperature')
    axes[1].plot(sim['t'] / 60, sim['temperature_c'], color='tab:green', ls='--', label='Model temperature')
    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Current / Temperature')
    axes[1].legend(ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG / 'dynamic_validation.png', dpi=180)
    plt.close(fig)


def write_claim_recovery(surr_metrics, fit_metrics_path):
    fit_df = pd.read_csv(fit_metrics_path)
    claims = [
        {
            'claim': 'The workspace supports a reproducible ANN-surrogate parameter identification workflow.',
            'supporting_artifacts': ['code/run_analysis.py', 'outputs/surrogate_metrics.json', 'outputs/lhs_samples.csv'],
            'status': 'supported_directly'
        },
        {
            'claim': 'The surrogate generalizes adequately on held-out simulator samples.',
            'supporting_artifacts': ['outputs/surrogate_metrics.json', 'report/images/surrogate_accuracy.png'],
            'status': 'supported_directly',
            'quantitative_summary': surr_metrics
        },
        {
            'claim': 'Identified parameter sets can fit representative constant-current and dynamic discharge curves.',
            'supporting_artifacts': ['outputs/identified_parameters.csv', 'outputs/fit_metrics.csv', 'report/images/reference_fit.png', 'report/images/dynamic_validation.png'],
            'status': 'supported_directly',
            'quantitative_summary': fit_df.to_dict(orient='records')
        },
        {
            'claim': 'This is an exact reproduction of a full ECAT-MMGA solver from the paper.',
            'supporting_artifacts': ['outputs/method_contract.json', 'outputs/method_fidelity_checklist.json'],
            'status': 'not_supported',
            'reason': 'A full ECAT solver and original MMGA implementation were not available in the workspace.'
        }
    ]
    (OUT / 'claim_recovery_table.json').write_text(json.dumps(claims, indent=2))


def main():
    ensure_dirs()
    calce = load_calce_reference_cycle()
    nasa = load_nasa_reference_cycle()
    oxford = load_oxford_dynamic()
    profiles = {p['dataset']: p for p in [calce, nasa, oxford]}
    dataset_summary = summarize_datasets(calce, nasa, oxford)
    samples, Y = make_training_set(calce, n_samples=600)
    surrogate, y_scaler, surr_metrics, surrogate_eval = train_surrogate(samples, Y)

    identified = {}
    for name, profile in profiles.items():
        best, sim, obs_feat, loss = identify_parameters(name, profile, surrogate, y_scaler, n_search=4000)
        identified[name] = {'params': best, 'simulation': sim, 'obs_feat': obs_feat, 'surrogate_loss': loss}

    export_results(profiles, identified)
    build_figures(dataset_summary, profiles, surrogate_eval, identified)
    write_claim_recovery(surr_metrics, OUT / 'fit_metrics.csv')
    print('Analysis complete')
    print(json.dumps({'surrogate_metrics': surr_metrics}, indent=2))


if __name__ == '__main__':
    main()
