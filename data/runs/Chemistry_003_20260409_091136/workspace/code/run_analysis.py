from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REPORT_IMAGES = ROOT / "report" / "images"


@dataclass
class Frame:
    meta: Dict[str, object]
    species: List[str]
    positions: np.ndarray
    forces: np.ndarray | None


def parse_meta(line: str) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    for key, value in re.findall(r'([A-Za-z_]+)=(".*?"|\S+)', line):
        if value.startswith('"') and value.endswith('"'):
            parsed = value[1:-1]
        else:
            parsed = value
        if key in {"energy", "charge_state", "total_charge"}:
            meta[key] = float(parsed)
        elif key == "true_charges":
            meta[key] = np.array([float(x) for x in parsed.split()], dtype=float)
        else:
            meta[key] = parsed
    return meta


def read_xyz(path: Path) -> List[Frame]:
    frames: List[Frame] = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            n_atoms = int(line)
            meta = parse_meta(f.readline().strip())
            species: List[str] = []
            positions = []
            forces = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                species.append(parts[0])
                xyz = [float(x) for x in parts[1:4]]
                positions.append(xyz)
                if len(parts) >= 7:
                    forces.append([float(x) for x in parts[4:7]])
            frames.append(
                Frame(
                    meta=meta,
                    species=species,
                    positions=np.asarray(positions, dtype=float),
                    forces=np.asarray(forces, dtype=float) if forces else None,
                )
            )
    return frames


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    delta = positions[:, None, :] - positions[None, :, :]
    d = np.linalg.norm(delta, axis=-1)
    iu = np.triu_indices(len(positions), 1)
    return d[iu]


def coulomb_matrix_from_positions(positions: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    delta = positions[:, None, :] - positions[None, :, :]
    d = np.linalg.norm(delta, axis=-1)
    iu = np.triu_indices(len(positions), 1)
    inv_r = 1.0 / np.clip(d[iu], 1e-8, None)
    n = len(positions)
    A = np.zeros((len(inv_r), n), dtype=float)
    row = 0
    for i, j in zip(*iu):
        A[row, i] = inv_r[row]
        A[row, j] = -inv_r[row]
        row += 1
    return A, iu


def analyze_random_charges(frames: List[Frame]) -> Dict[str, float]:
    first = frames[0]
    true_q = np.asarray(first.meta["true_charges"], dtype=float)
    A, _ = coulomb_matrix_from_positions(first.positions)
    b = A @ true_q
    recovered, *_ = np.linalg.lstsq(A, b, rcond=None)
    recovered = np.sign(recovered)

    same_features = []
    abs_force_targets = []
    per_atom_charge_signal = []
    for frame in frames:
        pos = frame.positions
        q = np.asarray(frame.meta["true_charges"], dtype=float)
        delta = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(d, np.inf)
        inv_r = 1.0 / d
        inv_r3 = 1.0 / np.power(d, 3)
        same = ((q[:, None] * q[None, :]) > 0).astype(float)
        opposite = ((q[:, None] * q[None, :]) < 0).astype(float)
        per_atom_charge_signal.append((inv_r * q[None, :]).sum(axis=1))
        same_features.append(np.stack([(inv_r * same).sum(axis=1), (inv_r * opposite).sum(axis=1)], axis=1))
        force_vec = ((q[:, None, None] * q[None, :, None]) * delta * inv_r3[:, :, None]).sum(axis=1)
        abs_force_targets.append(np.linalg.norm(force_vec, axis=1))
    X = np.concatenate(same_features, axis=0)
    y = np.concatenate(abs_force_targets, axis=0)
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    charge_signal = np.concatenate(per_atom_charge_signal)
    signed_truth = np.concatenate([np.asarray(f.meta["true_charges"], dtype=float) for f in frames])
    signed_pred = np.sign(charge_signal)
    signed_pred[signed_pred == 0] = 1

    with (OUTPUTS / "random_charges_recovery.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["atom_index", "true_charge", "recovered_charge"])
        for i, (t, r) in enumerate(zip(true_q.tolist(), recovered.tolist())):
            writer.writerow([i, t, r])

    plt.figure(figsize=(6, 4))
    plt.scatter(signed_truth, charge_signal, s=8, alpha=0.25)
    plt.axvline(-1, color="grey", lw=1, ls="--")
    plt.axvline(1, color="grey", lw=1, ls="--")
    plt.xlabel("True charge")
    plt.ylabel("Electrostatic signal")
    plt.title("Random-charge latent signal separates + and - ions")
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "random_charge_signal.png", dpi=200)
    plt.close()

    return {
        "single_frame_charge_recovery_acc": float(np.mean(recovered == true_q)),
        "dataset_charge_sign_acc": float(np.mean(signed_pred == signed_truth)),
        "force_magnitude_r2_from_charge_aware_features": float(r2_score(y, pred)),
        "force_magnitude_mae": float(mean_absolute_error(y, pred)),
    }


def molecule_descriptors(frame: Frame) -> Dict[str, float]:
    p = frame.positions
    c1 = p[:4]
    c2 = p[4:]
    com1 = c1.mean(axis=0)
    com2 = c2.mean(axis=0)
    all_pairs = pairwise_distances(p)
    intra1 = pairwise_distances(c1)
    intra2 = pairwise_distances(c2)
    return {
        "com_distance": float(np.linalg.norm(com1 - com2)),
        "min_interatomic": float(np.min(np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=-1))),
        "mean_intra1": float(intra1.mean()),
        "mean_intra2": float(intra2.mean()),
        "max_pair": float(all_pairs.max()),
    }


def analyze_charged_dimer(frames: List[Frame]) -> Dict[str, float]:
    rows = []
    for frame in frames:
        row = molecule_descriptors(frame)
        row["energy"] = float(frame.meta["energy"])
        rows.append(row)
    keys = [k for k in rows[0] if k != "energy"]
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    y = np.array([r["energy"] for r in rows], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    short_model = LinearRegression().fit(X_train[:, 1:], y_train)
    electro_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1e-6))
    electro_model.fit(X_train, y_train)

    short_pred = short_model.predict(X_test[:, 1:])
    electro_pred = electro_model.predict(X_test)

    order = np.argsort(X[:, 0])
    plt.figure(figsize=(6, 4))
    plt.plot(X[order, 0], y[order], label="Reference energy", lw=2)
    plt.plot(X[order, 0], short_model.predict(X[order, 1:]), label="Short-range baseline", lw=2)
    plt.plot(X[order, 0], electro_model.predict(X[order]), label="Electrostatic-aware surrogate", lw=2)
    plt.xlabel("Center-of-mass distance")
    plt.ylabel("Energy")
    plt.title("Charged dimer binding curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "charged_dimer_binding.png", dpi=200)
    plt.close()

    with (OUTPUTS / "charged_dimer_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["com_distance", "energy", "short_pred", "electro_pred"])
        for i in order:
            writer.writerow([X[i, 0], y[i], short_model.predict(X[i:i + 1, 1:])[0], electro_model.predict(X[i:i + 1])[0]])

    return {
        "short_range_test_mae": float(mean_absolute_error(y_test, short_pred)),
        "short_range_test_rmse": float(math.sqrt(mean_squared_error(y_test, short_pred))),
        "electrostatic_test_mae": float(mean_absolute_error(y_test, electro_pred)),
        "electrostatic_test_rmse": float(math.sqrt(mean_squared_error(y_test, electro_pred))),
        "electrostatic_test_r2": float(r2_score(y_test, electro_pred)),
    }


def ag3_features(frame: Frame) -> Dict[str, float]:
    d = pairwise_distances(frame.positions)
    return {
        "d_mean": float(d.mean()),
        "d_std": float(d.std()),
        "d_min": float(d.min()),
        "d_max": float(d.max()),
        "charge_state": float(frame.meta["charge_state"]),
        "energy": float(frame.meta["energy"]),
    }


def analyze_ag3(frames: List[Frame]) -> Dict[str, float]:
    rows = [ag3_features(f) for f in frames]
    base_keys = ["d_mean", "d_std", "d_min", "d_max"]
    X_geom = np.array([[r[k] for k in base_keys] for r in rows], dtype=float)
    X_charge = np.array([[r[k] for k in base_keys + ["charge_state"]] for r in rows], dtype=float)
    y_energy = np.array([r["energy"] for r in rows], dtype=float)
    y_charge = np.array([1 if r["charge_state"] > 0 else 0 for r in rows], dtype=int)

    idx_train, idx_test = train_test_split(np.arange(len(rows)), test_size=0.3, random_state=0, stratify=y_charge)

    geom_model = LinearRegression().fit(X_geom[idx_train], y_energy[idx_train])
    charge_model = LinearRegression().fit(X_charge[idx_train], y_energy[idx_train])
    classifier = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_geom[idx_train], y_charge[idx_train])

    geom_pred = geom_model.predict(X_geom[idx_test])
    charge_pred = charge_model.predict(X_charge[idx_test])
    class_prob = classifier.predict_proba(X_geom)[:, 1]

    order = np.argsort(X_geom[:, 0])
    plt.figure(figsize=(6, 4))
    mask_pos = np.array([r["charge_state"] > 0 for r in rows])
    plt.scatter(X_geom[mask_pos, 0], y_energy[mask_pos], label="q=+1", s=28)
    plt.scatter(X_geom[~mask_pos, 0], y_energy[~mask_pos], label="q=-1", s=28)
    plt.plot(X_geom[order, 0], charge_model.predict(X_charge[order]), color="black", lw=2, label="Charge-conditioned fit")
    plt.xlabel("Mean Ag-Ag distance")
    plt.ylabel("Energy")
    plt.title("Ag3 charge states are not geometry-separable")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / "ag3_charge_states.png", dpi=200)
    plt.close()

    with (OUTPUTS / "ag3_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d_mean", "energy", "charge_state", "geom_only_pred", "charge_cond_pred", "p_positive"])
        for i in order:
            writer.writerow([X_geom[i, 0], y_energy[i], rows[i]["charge_state"], geom_model.predict(X_geom[i:i + 1])[0], charge_model.predict(X_charge[i:i + 1])[0], class_prob[i]])

    return {
        "geom_only_energy_mae": float(mean_absolute_error(y_energy[idx_test], geom_pred)),
        "charge_conditioned_energy_mae": float(mean_absolute_error(y_energy[idx_test], charge_pred)),
        "charge_state_classification_acc_from_geometry": float(np.mean((class_prob[idx_test] >= 0.5) == y_charge[idx_test])),
    }


def summarize_datasets(random_frames: List[Frame], dimer_frames: List[Frame], ag3_frames: List[Frame]) -> List[Dict[str, object]]:
    return [
        {
            "dataset": "random_charges",
            "n_frames": len(random_frames),
            "n_atoms": len(random_frames[0].species),
            "has_energy": False,
            "has_forces": True,
            "notes": "Synthetic Coulomb plus repulsive benchmark with provided true charges.",
        },
        {
            "dataset": "charged_dimer",
            "n_frames": len(dimer_frames),
            "n_atoms": len(dimer_frames[0].species),
            "has_energy": True,
            "has_forces": True,
            "notes": "Two charged methane-like dimers at varying separation.",
        },
        {
            "dataset": "ag3_chargestates",
            "n_frames": len(ag3_frames),
            "n_atoms": len(ag3_frames[0].species),
            "has_energy": True,
            "has_forces": True,
            "notes": "Ag3 trimers with explicit +/- 1 charge-state labels.",
        },
    ]


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)

    random_frames = read_xyz(DATA / "random_charges.xyz")
    dimer_frames = read_xyz(DATA / "charged_dimer.xyz")
    ag3_frames = read_xyz(DATA / "ag3_chargestates.xyz")

    results = {
        "dataset_summary": summarize_datasets(random_frames, dimer_frames, ag3_frames),
        "random_charges": analyze_random_charges(random_frames),
        "charged_dimer": analyze_charged_dimer(dimer_frames),
        "ag3_chargestates": analyze_ag3(ag3_frames),
    }

    with (OUTPUTS / "metrics_summary.json").open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
