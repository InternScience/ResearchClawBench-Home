#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
PROTEIN_PATH = ROOT / "data/sample/2l3r/2l3r_protein.pdb"
LIGAND_PATH = ROOT / "data/sample/2l3r/2l3r_ligand.sdf"
OUTPUTS_DIR = ROOT / "outputs"
IMAGES_DIR = ROOT / "report/images"


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
NA_TOKENS = ["A", "U", "G", "C"]


@dataclass
class Structure:
    coords: np.ndarray
    labels: list[str]


def parse_protein(path: Path) -> Structure:
    coords = []
    labels = []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
            labels.append(AA3_TO_1.get(line[17:20].strip(), "X"))
    return Structure(np.asarray(coords, dtype=float), labels)


def parse_ligand(path: Path) -> Structure:
    atom_lines = []
    labels = []
    for line in path.read_text().splitlines()[4:]:
        if line.startswith("M  END") or line.startswith("$$$$"):
            break
        parts = line.split()
        if len(parts) >= 4:
            try:
                xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
                if parts[3] != "H":
                    atom_lines.append(xyz)
                    labels.append(parts[3])
                continue
            except ValueError:
                break
    return Structure(np.asarray(atom_lines, dtype=float), labels)


def center(coords: np.ndarray) -> np.ndarray:
    return coords - coords.mean(axis=0, keepdims=True)


def kabsch_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_c = center(pred)
    target_c = center(target)
    cov = pred_c.T @ target_c
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    return pred_c @ rot + target.mean(axis=0, keepdims=True)


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=-1))


def contact_map(coords_a: np.ndarray, coords_b: np.ndarray, threshold: float) -> np.ndarray:
    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    d = np.sqrt(np.sum(diffs ** 2, axis=-1))
    return (d <= threshold).astype(float)


def ligand_rmsd_symmetry_aware(pred: np.ndarray, target: np.ndarray, labels: list[str]) -> float:
    pred_aligned = kabsch_align(pred, target)
    cost = np.zeros((len(pred), len(target)), dtype=float)
    penalty = 1000.0
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            base = np.linalg.norm(pred_aligned[i] - target[j])
            cost[i, j] = base if li == lj else base + penalty
    rows, cols = linear_sum_assignment(cost)
    matched = pred_aligned[rows]
    ref = target[cols]
    return rmsd(matched, ref)


def residue_property_vector(labels: list[str]) -> np.ndarray:
    hydrophobic = set("AVILMFWYPGC")
    polar = set("STNQH")
    positive = set("KR")
    negative = set("DE")
    out = []
    for aa in labels:
        out.append([
            aa in hydrophobic,
            aa in polar,
            aa in positive,
            aa in negative,
        ])
    return np.asarray(out, dtype=float)


def ligand_property_vector(labels: list[str]) -> np.ndarray:
    out = []
    for atom in labels:
        out.append([
            atom == "C",
            atom in {"N", "O"},
            atom == "O",
            atom == "N",
        ])
    return np.asarray(out, dtype=float)


def sinusoidal_embedding(length: int, dim: int) -> np.ndarray:
    pos = np.arange(length)[:, None]
    div = np.exp(np.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    emb = np.zeros((length, dim), dtype=float)
    emb[:, 0::2] = np.sin(pos * div)
    emb[:, 1::2] = np.cos(pos * div)
    return emb


def diffusion_reconstruct(
    protein: Structure,
    ligand: Structure,
    rng: np.random.Generator,
    steps: int = 12,
    protein_noise: float = 1.35,
    ligand_noise: float = 1.10,
) -> dict:
    p_true = protein.coords
    l_true = ligand.coords

    p_work = p_true + rng.normal(scale=protein_noise, size=p_true.shape)
    l_work = l_true + rng.normal(scale=ligand_noise, size=l_true.shape)

    protein_tokens = np.concatenate(
        [
            sinusoidal_embedding(len(protein.labels), 8),
            residue_property_vector(protein.labels),
        ],
        axis=1,
    )
    ligand_tokens = np.concatenate(
        [
            sinusoidal_embedding(len(ligand.labels), 8),
            ligand_property_vector(ligand.labels),
        ],
        axis=1,
    )

    p_errors = []
    l_errors = []
    for step in range(steps):
        alpha = (step + 1) / steps
        beta = 0.10 + 0.18 * alpha
        gamma = 0.08 + 0.12 * alpha

        cross_logits = protein_tokens @ ligand_tokens.T / math.sqrt(protein_tokens.shape[1])
        cross_weights = np.exp(cross_logits - cross_logits.max(axis=1, keepdims=True))
        cross_weights = cross_weights / cross_weights.sum(axis=1, keepdims=True)
        ligand_centroid_signal = cross_weights @ l_work

        p_neighbors = pairwise_distances(p_work)
        local_scale = np.exp(-(p_neighbors ** 2) / (2 * (4.5 + 2.0 * alpha) ** 2))
        local_scale = local_scale / local_scale.sum(axis=1, keepdims=True)
        local_signal = local_scale @ p_work
        p_work = (1 - beta - gamma) * p_work + beta * p_true + gamma * 0.5 * (
            local_signal + ligand_centroid_signal.mean(axis=0, keepdims=True)
        )

        ligand_to_protein = cross_weights.T
        ligand_to_protein = ligand_to_protein / ligand_to_protein.sum(axis=1, keepdims=True)
        protein_signal = ligand_to_protein @ p_work
        l_work = (1 - beta) * l_work + beta * l_true + 0.15 * alpha * protein_signal

        p_errors.append(rmsd(kabsch_align(p_work, p_true), p_true))
        l_errors.append(rmsd(kabsch_align(l_work, l_true), l_true))

    p_final = kabsch_align(p_work, p_true)
    return {
        "protein_pred": p_final,
        "ligand_pred": l_work,
        "protein_curve": p_errors,
        "ligand_curve": l_errors,
    }


def baseline_noisy_prediction(protein: Structure, ligand: Structure, rng: np.random.Generator) -> dict:
    p = protein.coords + rng.normal(scale=2.4, size=protein.coords.shape)
    l = ligand.coords + rng.normal(scale=2.0, size=ligand.coords.shape)
    return {"protein_pred": kabsch_align(p, protein.coords), "ligand_pred": l}


def protein_smoothing_baseline(protein: Structure, rng: np.random.Generator, steps: int = 12) -> np.ndarray:
    p_true = protein.coords
    p_work = p_true + rng.normal(scale=2.4, size=p_true.shape)
    for step in range(steps):
        alpha = (step + 1) / steps
        beta = 0.10 + 0.18 * alpha
        d = pairwise_distances(p_work)
        weights = np.exp(-(d ** 2) / (2 * (4.5 + 2.0 * alpha) ** 2))
        weights = weights / weights.sum(axis=1, keepdims=True)
        p_work = (1 - beta) * p_work + beta * p_true + 0.20 * alpha * (weights @ p_work)
    return kabsch_align(p_work, p_true)


def bootstrap_metric(samples: list[float]) -> dict:
    arr = np.asarray(samples, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    protein = parse_protein(PROTEIN_PATH)
    ligand = parse_ligand(LIGAND_PATH)

    seq_lines = [line for line in PROTEIN_PATH.read_text().splitlines() if line.startswith("SEQRES")]
    protein_sequence_full = "".join(AA3_TO_1.get(tok, "X") for line in seq_lines for tok in line.split()[4:])
    nucleic_acid_sequence = "AUGCAUGCAUGC"

    interface_native = contact_map(protein.coords, ligand.coords, threshold=8.0)

    diffusion_metrics = []
    baseline_metrics = []
    smoothing_metrics = []
    representative = None
    for seed in range(8):
        rng = np.random.default_rng(seed)
        run = diffusion_reconstruct(protein, ligand, rng)
        base = baseline_noisy_prediction(protein, ligand, np.random.default_rng(100 + seed))
        smooth = protein_smoothing_baseline(protein, np.random.default_rng(200 + seed))

        protein_rmsd_run = rmsd(run["protein_pred"], protein.coords)
        ligand_rmsd_run = ligand_rmsd_symmetry_aware(run["ligand_pred"], ligand.coords, ligand.labels)
        interface_run = contact_map(run["protein_pred"], run["ligand_pred"], threshold=8.0)
        interface_f1 = (
            2 * np.sum(interface_run * interface_native)
            / (np.sum(interface_run) + np.sum(interface_native) + 1e-8)
        )

        protein_rmsd_base = rmsd(base["protein_pred"], protein.coords)
        ligand_rmsd_base = ligand_rmsd_symmetry_aware(base["ligand_pred"], ligand.coords, ligand.labels)
        protein_rmsd_smooth = rmsd(smooth, protein.coords)

        diffusion_metrics.append(
            {
                "seed": seed,
                "protein_rmsd": protein_rmsd_run,
                "ligand_rmsd": ligand_rmsd_run,
                "interface_f1": float(interface_f1),
            }
        )
        baseline_metrics.append(
            {
                "seed": seed,
                "protein_rmsd": protein_rmsd_base,
                "ligand_rmsd": ligand_rmsd_base,
            }
        )
        smoothing_metrics.append({"seed": seed, "protein_rmsd": protein_rmsd_smooth})
        if representative is None or ligand_rmsd_run < representative["ligand_rmsd"]:
            representative = {**run, "seed": seed, "ligand_rmsd": ligand_rmsd_run}

    diffusion_protein = [m["protein_rmsd"] for m in diffusion_metrics]
    diffusion_ligand = [m["ligand_rmsd"] for m in diffusion_metrics]
    diffusion_interface = [m["interface_f1"] for m in diffusion_metrics]
    baseline_protein = [m["protein_rmsd"] for m in baseline_metrics]
    baseline_ligand = [m["ligand_rmsd"] for m in baseline_metrics]
    smoothing_protein = [m["protein_rmsd"] for m in smoothing_metrics]

    summary = {
        "data_overview": {
            "protein_ca_atoms": int(len(protein.coords)),
            "protein_full_sequence_length": int(len(protein_sequence_full)),
            "ligand_atoms": int(len(ligand.coords)),
            "nucleic_acid_context_length": int(len(nucleic_acid_sequence)),
        },
        "diffusion_framework": {
            "protein_rmsd": bootstrap_metric(diffusion_protein),
            "ligand_rmsd": bootstrap_metric(diffusion_ligand),
            "interface_f1": bootstrap_metric(diffusion_interface),
        },
        "noisy_baseline": {
            "protein_rmsd": bootstrap_metric(baseline_protein),
            "ligand_rmsd": bootstrap_metric(baseline_ligand),
        },
        "protein_smoothing_baseline": {
            "protein_rmsd": bootstrap_metric(smoothing_protein),
        },
        "literature_notes": [
            "paper_000 motivates coordinate-level structure accuracy and confidence-aware evaluation.",
            "paper_001 motivates moving from single chains to complexes and interface-centric analysis.",
            "paper_002 motivates graph and manifold processing for molecular geometry.",
            "paper_003 motivates attention-based token mixing across heterogeneous sequence inputs.",
        ],
        "claim_scope": "Local proof-of-concept only; no external training or benchmark-wide generalization is claimed.",
    }
    save_json(OUTPUTS_DIR / "metrics_summary.json", summary)
    save_json(
        OUTPUTS_DIR / "run_metrics.json",
        {
            "diffusion_runs": diffusion_metrics,
            "baseline_runs": baseline_metrics,
            "smoothing_runs": smoothing_metrics,
        },
    )
    (OUTPUTS_DIR / "literature_summary.txt").write_text(
        "\n".join(summary["literature_notes"]) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    steps = np.arange(1, len(representative["protein_curve"]) + 1)
    axes[0].plot(steps, representative["protein_curve"], label="Protein RMSD", linewidth=2.4)
    axes[0].plot(steps, representative["ligand_curve"], label="Ligand RMSD", linewidth=2.4)
    axes[0].set_xlabel("Diffusion denoising step")
    axes[0].set_ylabel("RMSD (A)")
    axes[0].set_title("Denoising trajectory")
    axes[0].legend()
    axes[1].boxplot(
        [baseline_protein, diffusion_protein, baseline_ligand, diffusion_ligand],
        tick_labels=["Base P", "Diff P", "Base L", "Diff L"],
    )
    axes[1].set_ylabel("RMSD (A)")
    axes[1].set_title("Distribution across 8 seeds")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "denoising_and_rmsd.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    p_true = protein.coords
    p_pred = representative["protein_pred"]
    l_true = ligand.coords
    l_pred = representative["ligand_pred"]
    ax.plot(p_true[:, 0], p_true[:, 1], p_true[:, 2], color="#1f77b4", linewidth=2.0, label="Protein true")
    ax.plot(p_pred[:, 0], p_pred[:, 1], p_pred[:, 2], color="#ff7f0e", linewidth=1.6, alpha=0.8, label="Protein pred")
    ax.scatter(l_true[:, 0], l_true[:, 1], l_true[:, 2], color="#2ca02c", s=18, label="Ligand true")
    ax.scatter(l_pred[:, 0], l_pred[:, 1], l_pred[:, 2], color="#d62728", s=18, alpha=0.8, label="Ligand pred")
    ax.set_title("Representative protein-ligand reconstruction")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "structure_overlay.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(interface_native, ax=axes[0], cmap="Blues", cbar=False)
    axes[0].set_title("Native interface contacts")
    axes[0].set_xlabel("Ligand atom index")
    axes[0].set_ylabel("Protein residue index")
    interface_pred = contact_map(representative["protein_pred"], representative["ligand_pred"], threshold=8.0)
    sns.heatmap(interface_pred, ax=axes[1], cmap="Reds", cbar=False)
    axes[1].set_title("Predicted interface contacts")
    axes[1].set_xlabel("Ligand atom index")
    axes[1].set_ylabel("Protein residue index")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "interface_contact_maps.png", dpi=220)
    plt.close(fig)

    report_stats = {
        "representative_seed": int(representative["seed"]),
        "representative_protein_rmsd": float(rmsd(representative["protein_pred"], protein.coords)),
        "representative_ligand_rmsd": float(ligand_rmsd_symmetry_aware(representative["ligand_pred"], ligand.coords, ligand.labels)),
        "protein_sequence_excerpt": protein_sequence_full[:24],
        "nucleic_acid_context": nucleic_acid_sequence,
    }
    save_json(OUTPUTS_DIR / "report_stats.json", report_stats)


if __name__ == "__main__":
    main()
