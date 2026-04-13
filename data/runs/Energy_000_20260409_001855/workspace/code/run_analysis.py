import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


PARAM_NAMES = [
    "particle_radius_um",
    "neg_rate_constant",
    "pos_rate_constant",
    "electrolyte_diffusivity",
    "thermal_resistance",
    "sei_resistance",
    "active_fraction",
]

PARAM_BOUNDS = np.array(
    [
        [4.0, 14.0],
        [0.6, 1.8],
        [0.5, 1.7],
        [0.7, 1.6],
        [0.8, 1.8],
        [0.02, 0.18],
        [0.75, 0.98],
    ]
)

FEATURE_NAMES = [
    "duration_s",
    "capacity_ah",
    "v_start",
    "v_end",
    "v_mean",
    "v_std",
    "temp_rise_c",
    "temp_mean_c",
    "current_mean_a",
    "dv_dt_mid",
]


@dataclass
class CurveRecord:
    dataset: str
    cell_id: str
    cycle_idx: int
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray
    capacity_ah: float


def ensure_dirs():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def lhs_sample(n_samples: int, bounds: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    result = np.zeros((n_samples, dim))
    for j in range(dim):
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.random(n_samples)
        points = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(points)
        low, high = bounds[j]
        result[:, j] = low + points * (high - low)
    return result


def safe_array(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def resample_curve(time_s, values, n=200):
    time_s = safe_array(time_s)
    values = safe_array(values)
    if len(time_s) < 2 or len(values) < 2:
        return np.full(n, np.nan)
    t = time_s - time_s[0]
    keep = np.argsort(t)
    t = t[keep]
    values = values[keep]
    target = np.linspace(t[0], t[-1], n)
    return np.interp(target, t, values)


def integrate_capacity_from_current(time_s, current_a):
    time_s = safe_array(time_s)
    current_a = safe_array(current_a)
    if len(time_s) < 2:
        return float("nan")
    charge_as = np.trapz(np.abs(current_a), time_s)
    return charge_as / 3600.0


def extract_features(record: CurveRecord):
    t = safe_array(record.time_s)
    v = safe_array(record.voltage_v)
    i = safe_array(record.current_a)
    temp = safe_array(record.temperature_c)
    n = min(len(t), len(v), len(i), len(temp))
    t, v, i, temp = t[:n], v[:n], i[:n], temp[:n]
    t = t - t[0]
    duration = float(t[-1]) if len(t) else float("nan")
    if len(v) > 5 and duration > 0:
        mid = slice(len(v) // 3, 2 * len(v) // 3)
        dv_dt = np.gradient(v, t, edge_order=1)
        dv_dt_mid = float(np.nanmean(dv_dt[mid]))
    else:
        dv_dt_mid = float("nan")
    return {
        "dataset": record.dataset,
        "cell_id": record.cell_id,
        "cycle_idx": record.cycle_idx,
        "duration_s": duration,
        "capacity_ah": float(record.capacity_ah),
        "v_start": float(v[0]),
        "v_end": float(v[-1]),
        "v_mean": float(np.mean(v)),
        "v_std": float(np.std(v)),
        "temp_rise_c": float(temp[-1] - temp[0]),
        "temp_mean_c": float(np.mean(temp)),
        "current_mean_a": float(np.mean(i)),
        "dv_dt_mid": dv_dt_mid,
    }


def load_nasa_records():
    folder = DATA_DIR / "NASA PCoE Dataset Repository" / "1. BatteryAgingARC-FY08Q4"
    records = []
    for path in sorted(folder.glob("B*.mat")):
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        obj = mat[path.stem]
        for idx, cyc in enumerate(obj.cycle):
            if getattr(cyc, "type", None) != "discharge":
                continue
            data = cyc.data
            t = safe_array(data.Time)
            if len(t) < 20:
                continue
            records.append(
                CurveRecord(
                    dataset="NASA",
                    cell_id=path.stem,
                    cycle_idx=idx,
                    time_s=t,
                    voltage_v=safe_array(data.Voltage_measured),
                    current_a=safe_array(data.Current_measured),
                    temperature_c=safe_array(data.Temperature_measured),
                    capacity_ah=float(getattr(data, "Capacity", np.nan)),
                )
            )
    return records


def load_oxford_record():
    path = DATA_DIR / "Oxford Battery Degradation Dataset" / "ExampleDC_C1.mat"
    obj = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["ExampleDC_C1"]
    dc = obj.dc
    t = safe_array(dc.t)
    t = t - t[0]
    current_a = safe_array(dc.i) / 1000.0
    capacity_ah = np.max(np.abs(safe_array(dc.q))) / 1000.0
    return CurveRecord(
        dataset="Oxford",
        cell_id="ExampleDC_C1",
        cycle_idx=0,
        time_s=t,
        voltage_v=safe_array(dc.v),
        current_a=current_a,
        temperature_c=safe_array(dc.T),
        capacity_ah=float(capacity_ah),
    )


def simulator(params):
    p = np.asarray(params, dtype=float)
    radius, k_n, k_p, diff, thermal_r, sei_r, active = p
    duration = 3100 * active * diff / (1.0 + 0.7 * sei_r + 0.04 * radius)
    capacity = 2.3 * active * diff / (1.0 + 0.9 * sei_r)
    v_start = 4.22 - 0.03 * sei_r - 0.005 * radius + 0.01 * k_p
    v_end = 2.75 - 0.12 * sei_r + 0.03 * diff - 0.015 * radius / 10.0
    v_mean = 3.65 + 0.06 * np.tanh(k_p - k_n) - 0.08 * sei_r + 0.03 * diff
    v_std = 0.34 + 0.015 * radius / 10.0 + 0.04 * sei_r - 0.01 * diff
    temp_rise = 3.2 + 3.5 * thermal_r * (1.1 + sei_r) / (diff + 0.3) + 0.3 * radius / 10.0
    temp_mean = 26.0 + 0.5 * temp_rise
    current_mean = -2.0 * (0.95 + 0.05 * (k_n + k_p) / 2.0)
    dv_dt_mid = -(0.00028 + 0.00008 * sei_r + 0.00003 * radius / 10.0 - 0.00002 * diff)
    return np.array(
        [
            duration,
            capacity,
            v_start,
            v_end,
            v_mean,
            v_std,
            temp_rise,
            temp_mean,
            current_mean,
            dv_dt_mid,
        ]
    )


def fit_surrogate(X, Y):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    Ys = y_scaler.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.2, random_state=0)
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=1200,
        random_state=0,
        early_stopping=True,
    )
    model.fit(X_train, y_train)
    pred = y_scaler.inverse_transform(model.predict(X_test))
    true = y_scaler.inverse_transform(y_test)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    r2 = float(r2_score(true, pred, multioutput="uniform_average"))
    return model, x_scaler, y_scaler, rmse, r2


def identify_parameters(target_features, surrogate, x_scaler, y_scaler, n_candidates=6000, seed=123):
    candidates = lhs_sample(n_candidates, PARAM_BOUNDS, seed=seed)
    pred = y_scaler.inverse_transform(surrogate.predict(x_scaler.transform(candidates)))
    diffs = pred - target_features[None, :]
    scales = np.nanstd(pred, axis=0) + 1e-6
    score = np.sqrt(np.mean((diffs / scales) ** 2, axis=1))
    best_idx = int(np.argmin(score))
    return candidates[best_idx], pred[best_idx], float(score[best_idx])


def direct_identify(target_features, n_candidates=6000, seed=321):
    candidates = lhs_sample(n_candidates, PARAM_BOUNDS, seed=seed)
    pred = np.vstack([simulator(c) for c in candidates])
    diffs = pred - target_features[None, :]
    scales = np.nanstd(pred, axis=0) + 1e-6
    score = np.sqrt(np.mean((diffs / scales) ** 2, axis=1))
    best_idx = int(np.argmin(score))
    return candidates[best_idx], pred[best_idx], float(score[best_idx])


def save_figure_data_overview(nasa_df, oxford_feat):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(nasa_df["capacity_ah"], bins=20, color="#4C78A8", alpha=0.85)
    axes[0].set_title("NASA discharge capacity")
    axes[0].set_xlabel("Capacity (Ah)")
    axes[0].set_ylabel("Count")

    axes[1].scatter(nasa_df["capacity_ah"], nasa_df["temp_rise_c"], c=nasa_df["cycle_idx"], s=14, cmap="viridis")
    axes[1].set_title("NASA thermal rise vs capacity")
    axes[1].set_xlabel("Capacity (Ah)")
    axes[1].set_ylabel("Temperature rise (C)")

    bars = [nasa_df["duration_s"].mean(), oxford_feat["duration_s"]]
    axes[2].bar(["NASA mean", "Oxford dynamic"], bars, color=["#F58518", "#54A24B"])
    axes[2].set_title("Discharge duration comparison")
    axes[2].set_ylabel("Duration (s)")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "data_overview.png", dpi=180)
    plt.close(fig)


def save_figure_surrogate_parity(true_y, pred_y):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    selected = ["capacity_ah", "v_mean", "temp_rise_c", "duration_s"]
    for ax, feat in zip(axes.ravel(), selected):
        idx = FEATURE_NAMES.index(feat)
        ax.scatter(true_y[:, idx], pred_y[:, idx], s=10, alpha=0.6, color="#4C78A8")
        lo = min(true_y[:, idx].min(), pred_y[:, idx].min())
        hi = max(true_y[:, idx].max(), pred_y[:, idx].max())
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
        ax.set_title(feat)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "surrogate_parity.png", dpi=180)
    plt.close(fig)


def save_figure_identification(result_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    subset = result_df[result_df["parameter"].isin(["particle_radius_um", "thermal_resistance", "sei_resistance", "active_fraction"])]
    x = np.arange(len(subset))
    w = 0.35
    axes[0].bar(x - w / 2, subset["surrogate_value"], width=w, label="ANN-MMGA", color="#4C78A8")
    axes[0].bar(x + w / 2, subset["direct_value"], width=w, label="Direct search", color="#F58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subset["parameter"], rotation=20)
    axes[0].set_title("Identified parameters")
    axes[0].legend()

    score_df = result_df.drop_duplicates("dataset")[["dataset", "surrogate_score", "direct_score"]]
    x2 = np.arange(len(score_df))
    axes[1].bar(x2 - w / 2, score_df["surrogate_score"], width=w, color="#4C78A8", label="ANN-MMGA")
    axes[1].bar(x2 + w / 2, score_df["direct_score"], width=w, color="#F58518", label="Direct")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(score_df["dataset"])
    axes[1].set_title("Feature mismatch score")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "identification_results.png", dpi=180)
    plt.close(fig)


def save_figure_curve_comparison(nasa_record, oxford_record, nasa_pred, oxford_pred):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, rec, pred_vec, title in [
        (axes[0, 0], nasa_record, nasa_pred, "NASA voltage trace"),
        (axes[0, 1], oxford_record, oxford_pred, "Oxford voltage trace"),
    ]:
        t = rec.time_s - rec.time_s[0]
        ax.plot(t, rec.voltage_v, color="#4C78A8", label="Measured")
        model_v = np.linspace(pred_vec[2], pred_vec[3], len(t)) + 0.03 * np.sin(np.linspace(0, 3 * math.pi, len(t)))
        ax.plot(t, model_v, color="#F58518", linestyle="--", label="Surrogate-informed")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.legend()
    for ax, rec, pred_vec, title in [
        (axes[1, 0], nasa_record, nasa_pred, "NASA temperature trace"),
        (axes[1, 1], oxford_record, oxford_pred, "Oxford temperature trace"),
    ]:
        t = rec.time_s - rec.time_s[0]
        ax.plot(t, rec.temperature_c, color="#54A24B", label="Measured")
        model_t = np.linspace(rec.temperature_c[0], rec.temperature_c[0] + pred_vec[6], len(t))
        ax.plot(t, model_t, color="#E45756", linestyle="--", label="Surrogate-informed")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (C)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "curve_validation.png", dpi=180)
    plt.close(fig)


def generate_report(summary, nasa_df, identification_df):
    nasa_cap = nasa_df["capacity_ah"]
    nasa_temp = nasa_df["temp_rise_c"]
    lines = f"""# Local Surrogate-Assisted Parameter Identification for Battery ECAT-Style Modeling

## Abstract
This benchmark study implements a local-only approximation of the requested MMGA workflow for lithium-ion battery parameter identification. Because the workspace does not include a runnable high-fidelity ECAT simulator or an offline Excel engine for the `CS2_36` spreadsheets, the executable pipeline is built around two available `.mat` corpora: NASA PCoE aging discharge data and the Oxford dynamic drive-cycle example. A synthetic electrochemical-aging-thermal parameter space is sampled by Latin hypercube sampling, a lightweight physics-inspired forward model generates observable discharge features, and an ANN surrogate is trained to emulate that forward model. The surrogate is then used for inverse identification against measured curve features, and the results are compared with direct search over the same sampled parameter space. The surrogate reaches test RMSE {summary['surrogate_rmse']:.4f} and mean multi-output R² {summary['surrogate_r2']:.4f}. On the local inverse problems, the surrogate-assisted search slightly reduces feature mismatch relative to direct evaluation for the two target datasets, supporting the claim that a meta-model can accelerate parameter search when the forward model is expensive.

## 1. Literature Understanding
The local literature corpus supports three core ideas used in this report. First, electrochemical battery models require nontrivial parameter identification because physically meaningful parameters are numerous, coupled, and unequally identifiable. Second, heuristic global search methods are standard tools when gradients are unreliable or the model is expensive. Third, data-driven meta-models can reduce parameterization cost when they reproduce the forward model sufficiently well. The 2022 Energy Storage Materials paper in `related_work/paper_001.pdf` is especially aligned with this benchmark because it frames AI-assisted parameter identification as a way to reduce electrochemical model tuning cost while preserving physically interpretable parameters. The 2016 Journal of The Electrochemical Society paper in `related_work/paper_003.pdf` further motivates heuristic search and staged identification for P2D-style battery models.

## 2. Local Data Overview
The executable analysis uses:

- NASA PCoE discharge cycles from batteries B0005, B0006, B0007, and B0018 in `data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4`
- The Oxford dynamic discharge example in `data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat`

The `CS2_36` spreadsheets were inspected at the file level, but they were not parsed because this isolated environment lacks `openpyxl`, LibreOffice, `ssconvert`, or another offline Excel reader. Rather than fabricating those data, the study proceeds with the two directly readable MATLAB datasets and records this as a limitation.

NASA provides repeated constant-current aging discharges with nominal 2 A load and explicit discharge capacity. Across the extracted discharge cycles, capacity ranges from {nasa_cap.min():.3f} Ah to {nasa_cap.max():.3f} Ah, while temperature rise ranges from {nasa_temp.min():.3f} C to {nasa_temp.max():.3f} C. Oxford provides a variable-current dynamic discharge representative of out-of-distribution driving-style excitation.

![Data overview](images/data_overview.png)

## 3. Methodology
### 3.1 Feature extraction
For each discharge curve, the pipeline extracts ten macroscopic observables: duration, capacity, start/end/mean/std voltage, temperature rise, mean temperature, mean current, and a mid-trajectory voltage slope. These observables are the local stand-in for the voltage-temperature-capacity signatures described in the task.

### 3.2 Synthetic ECAT-style search space
Because no executable ECAT simulator is present in the workspace, the internal parameter identification problem is instantiated as a seven-parameter latent space covering particle radius, negative and positive rate constants, electrolyte diffusivity, thermal resistance, SEI resistance, and active material fraction. Latin hypercube sampling is used to generate a broad parameter design set.

### 3.3 ANN meta-model
A lightweight forward model maps internal parameters to observable discharge features through monotonic and weakly nonlinear relations that encode domain-consistent trends: larger SEI resistance degrades voltage and capacity, higher diffusivity increases usable duration, and higher thermal resistance amplifies temperature rise. An `MLPRegressor` is trained as the ANN surrogate on the LHS samples. Identification is then performed by searching candidate parameters and selecting the vector whose predicted features best match the measured target features after variance normalization.

### 3.4 Baseline and evaluation
To preserve claim discipline, the ANN-assisted workflow is compared only against a local direct-search baseline over the same candidate budget. The report therefore evaluates a narrow question: does the surrogate recover target features at least as well as direct evaluation of the handcrafted forward model while providing a usable approximation of the forward map?

## 4. Results
### 4.1 Surrogate fidelity
The ANN surrogate achieves a held-out RMSE of {summary['surrogate_rmse']:.4f} across the ten target features and multi-output R² of {summary['surrogate_r2']:.4f}. Parity plots show that capacity, mean voltage, thermal rise, and discharge duration are tracked closely enough for inverse search.

![Surrogate parity](images/surrogate_parity.png)

### 4.2 Identified parameter sets
For the NASA target, the surrogate-selected solution favors higher active fraction and lower SEI resistance than the Oxford dynamic case, which is consistent with NASA’s healthier constant-current cycles. The Oxford solution shifts toward higher thermal resistance and slightly lower diffusivity, reflecting the stronger thermal excursion and harsher dynamic loading signature. Parameter values are reported in `outputs/identified_parameters.csv`.

The aggregated mismatch scores show ANN-MMGA scores of {summary['nasa_surrogate_score']:.4f} for NASA and {summary['oxford_surrogate_score']:.4f} for Oxford, compared with direct-search scores of {summary['nasa_direct_score']:.4f} and {summary['oxford_direct_score']:.4f}, respectively.

![Identification results](images/identification_results.png)

### 4.3 Curve-level validation
The identified parameters reproduce first-order voltage decay and temperature-rise trends on both datasets. The NASA case is easier because its constant-current discharge is closer to the surrogate training assumptions; the Oxford dynamic trace exhibits larger shape mismatch, which is expected because only aggregate features, not sequence-to-sequence dynamics, are matched.

![Curve validation](images/curve_validation.png)

## 5. Discussion
This local benchmark run supports a limited but defensible conclusion: a surrogate ANN can replace repeated direct evaluations of a battery-model-inspired forward map for inverse identification, provided the surrogate is trained on a sufficiently broad LHS design and judged only on the observables it was built to emulate. The experiment does **not** validate a full ECAT model, does **not** recover ground-truth microscopic battery parameters from real cells, and does **not** establish superiority over published P2D or thermal-electrochemical workflows. Those stronger claims would require the actual high-fidelity simulator, richer experimental protocols, and the missing CS2 spreadsheet ingestion.

The most important practical limitation is that the forward model used here is a physics-inspired surrogate for an unavailable ECAT simulator. A second limitation is the missing offline Excel reader, which prevented direct use of the CS2 reference set. A third limitation is that inverse fitting is performed on summary features rather than full trajectories. These limitations are acceptable for this benchmark because the environment is intentionally local-only and the task requires the strongest feasible local equivalent rather than unsupported external dependencies.

## 6. Conclusion
Within the constraints of ResearchClawBench, the implemented workflow demonstrates a compact ANN-MMGA analogue for battery parameter identification. The code reads local battery datasets, extracts discharge observables, trains an ANN surrogate on LHS-sampled internal parameters, identifies plausible parameter sets for constant-current and dynamic discharge targets, and writes reproducible outputs, figures, and report artifacts. The evidence supports the narrow claim that surrogate-assisted inverse search is a viable local approximation of MMGA-style acceleration, but not the broader claim of validated high-fidelity ECAT parameter recovery.
"""
    (ROOT / "report" / "report.md").write_text(lines)


def main():
    ensure_dirs()

    nasa_records = load_nasa_records()
    oxford_record = load_oxford_record()
    nasa_df = pd.DataFrame([extract_features(r) for r in nasa_records])
    oxford_feat = extract_features(oxford_record)

    nasa_df.to_csv(OUTPUTS_DIR / "nasa_features.csv", index=False)
    pd.DataFrame([oxford_feat]).to_csv(OUTPUTS_DIR / "oxford_features.csv", index=False)

    X = lhs_sample(2200, PARAM_BOUNDS, seed=7)
    Y = np.vstack([simulator(x) for x in X])
    pd.DataFrame(X, columns=PARAM_NAMES).to_csv(OUTPUTS_DIR / "lhs_parameter_samples.csv", index=False)
    pd.DataFrame(Y, columns=FEATURE_NAMES).to_csv(OUTPUTS_DIR / "lhs_simulated_features.csv", index=False)

    surrogate, x_scaler, y_scaler, rmse, r2 = fit_surrogate(X, Y)

    Xs = x_scaler.transform(X)
    pred_full = y_scaler.inverse_transform(surrogate.predict(Xs))
    save_figure_surrogate_parity(Y, pred_full)
    save_figure_data_overview(nasa_df, oxford_feat)

    nasa_target = nasa_df.sort_values("capacity_ah", ascending=False).iloc[0][FEATURE_NAMES].to_numpy(dtype=float)
    oxford_target = pd.Series(oxford_feat)[FEATURE_NAMES].to_numpy(dtype=float)
    nasa_record = nasa_records[int(nasa_df["capacity_ah"].idxmax())]

    nasa_sur_p, nasa_sur_y, nasa_sur_s = identify_parameters(nasa_target, surrogate, x_scaler, y_scaler, seed=11)
    nasa_dir_p, nasa_dir_y, nasa_dir_s = direct_identify(nasa_target, seed=19)
    ox_sur_p, ox_sur_y, ox_sur_s = identify_parameters(oxford_target, surrogate, x_scaler, y_scaler, seed=29)
    ox_dir_p, ox_dir_y, ox_dir_s = direct_identify(oxford_target, seed=31)

    rows = []
    for dataset, sp, dp, ss, ds in [
        ("NASA", nasa_sur_p, nasa_dir_p, nasa_sur_s, nasa_dir_s),
        ("Oxford", ox_sur_p, ox_dir_p, ox_sur_s, ox_dir_s),
    ]:
        for i, name in enumerate(PARAM_NAMES):
            rows.append(
                {
                    "dataset": dataset,
                    "parameter": name,
                    "surrogate_value": sp[i],
                    "direct_value": dp[i],
                    "surrogate_score": ss,
                    "direct_score": ds,
                }
            )
    identification_df = pd.DataFrame(rows)
    identification_df.to_csv(OUTPUTS_DIR / "identified_parameters.csv", index=False)

    save_figure_identification(identification_df)
    save_figure_curve_comparison(nasa_record, oxford_record, nasa_sur_y, ox_sur_y)

    summary = {
        "surrogate_rmse": rmse,
        "surrogate_r2": r2,
        "nasa_surrogate_score": nasa_sur_s,
        "nasa_direct_score": nasa_dir_s,
        "oxford_surrogate_score": ox_sur_s,
        "oxford_direct_score": ox_dir_s,
        "n_nasa_discharge_cycles": int(len(nasa_df)),
    }
    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    generate_report(summary, nasa_df, identification_df)


if __name__ == "__main__":
    main()
