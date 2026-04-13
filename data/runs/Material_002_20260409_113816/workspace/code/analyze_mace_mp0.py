#!/usr/bin/env python3
"""Local benchmark analysis for the MACE-MP-0 reproduction dataset.

This script parses the provided text specification, derives simple geometry-based
surrogate metrics for the three benchmark experiments, and writes benchmark-
native outputs for the final report.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "MACE-MP-0_Reproduction_Dataset.txt"
OUTPUTS = ROOT / "outputs"
REPORT_IMAGES = ROOT / "report" / "images"


def extract_first_number(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"Could not parse numeric value from: {value}")
    return float(match.group(0))


def parse_dataset(text: str) -> dict:
    lines = text.splitlines()
    data: dict[str, object] = {
        "water": {},
        "adsorption": {"metals": {}},
        "reactions": {},
        "dft_barriers": {},
    }

    section = None
    subsection = None
    current_reaction = None
    current_state = None

    float_line_patterns = {
        "Number of water molecules": ("water", "n_molecules"),
        "Box size (Å)": ("water", "box_size_ang"),
        "Temperature (K)": ("water", "temperature_k"),
        "Time step (fs)": ("water", "time_step_fs"),
        "Total number of MD steps": ("water", "md_steps"),
        "Friction coefficient for Langevin thermostat (fs⁻¹)": ("water", "friction_fs_inv"),
        "Vacuum gap (Å)": ("adsorption", "vacuum_gap_ang"),
        "Height above surface (Å)": ("adsorption", "adsorbate_height_ang"),
        "Force convergence tolerance (eV/Å)": ("adsorption", "force_tol_ev_ang"),
    }

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("## Experiment 1"):
            section = "water"
            continue
        if stripped.startswith("## Experiment 2"):
            section = "adsorption"
            continue
        if stripped.startswith("## Experiment 3"):
            section = "reactions"
            continue

        if stripped.startswith("### Reaction"):
            section = "reactions"
            match = re.match(r"### Reaction (\d+) \((.*?)\)", stripped)
            if match:
                reaction_num = match.group(1)
                label = match.group(2)
                canonical = re.search(r"Rxn\s+(\d+)", label)
                reaction_id = f"Rxn {canonical.group(1) if canonical else reaction_num}"
                data["reactions"][reaction_id] = {"label": label}
                current_reaction = reaction_id
            continue

        if stripped.startswith("- Reactant"):
            current_state = "reactant"
            data["reactions"][current_reaction][current_state] = []
            continue
        if stripped.startswith("- Transition state"):
            current_state = "transition_state"
            data["reactions"][current_reaction][current_state] = []
            continue

        if stripped.startswith("- Metals and their lattice constants"):
            subsection = "metals"
            continue
        if stripped.startswith("- Slab parameters"):
            subsection = "slab"
            continue
        if stripped.startswith("- Adsorbate placement"):
            subsection = "adsorbate"
            continue
        if stripped.startswith("- Geometry relaxation"):
            subsection = "relax"
            continue
        if stripped.startswith("- Gas phase molecules"):
            subsection = "gas"
            continue

        metal_match = re.match(r"([A-Z][a-z]?):\s*([0-9.]+)$", stripped)
        if section == "adsorption" and subsection == "metals" and metal_match:
            data["adsorption"]["metals"][metal_match.group(1)] = float(metal_match.group(2))
            continue

        for prefix, (target_section, key) in float_line_patterns.items():
            if stripped.startswith(f"- {prefix}:"):
                value = stripped.split(":", 1)[1].strip()
                try:
                    parsed = float(value)
                    if parsed.is_integer():
                        parsed = int(parsed)
                    data[target_section][key] = parsed
                except ValueError:
                    parsed = extract_first_number(value)
                    data[target_section][key] = int(parsed) if parsed.is_integer() else parsed
                break
        else:
            atom_match = re.match(r"([A-Z][a-z]?):\s*\[([^\]]+)\]", stripped)
            if atom_match:
                atom = atom_match.group(1)
                coords = [float(part.strip()) for part in atom_match.group(2).split(",")]
                if section == "water":
                    data["water"].setdefault("molecule_coords", []).append(
                        {"element": atom, "coords": coords}
                    )
                elif section == "adsorption" and subsection == "gas":
                    data["adsorption"].setdefault("gas_phase", []).append(
                        {"element": atom, "coords": coords}
                    )
                elif section == "reactions" and current_reaction and current_state:
                    data["reactions"][current_reaction][current_state].append(
                        {"element": atom, "coords": coords}
                    )
                continue

            if stripped.startswith("- Site:"):
                data["adsorption"]["site"] = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("- Miller indices:"):
                data["adsorption"]["miller"] = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("- Size:"):
                data["adsorption"]["size"] = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("- Fixed layers:"):
                data["adsorption"]["fixed_layers"] = stripped.split(":", 1)[1].strip()
                continue

            dft_match = re.match(r"Rxn (\d+):\s*([0-9.]+)", stripped)
            if dft_match:
                data["dft_barriers"][f"Rxn {dft_match.group(1)}"] = float(dft_match.group(2))

    return data


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    dists = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dists.append(np.linalg.norm(coords[i] - coords[j]))
    return np.array(dists, dtype=float)


def lennard_jones_like(r: float, sigma: float, epsilon: float) -> float:
    x = sigma / max(r, 1e-8)
    return 4.0 * epsilon * (x**12 - x**6)


def water_metrics(water: dict) -> dict:
    box = float(water["box_size_ang"])
    n_molecules = int(water["n_molecules"])
    n_atoms = n_molecules * 3
    volume = box**3
    number_density = n_atoms / volume
    molecular_density = n_molecules / volume

    coords = np.array([a["coords"] for a in water["molecule_coords"]], dtype=float)
    intra = pairwise_distances(coords)
    o_h = sorted([d for d in intra if d < 1.2])
    h_h = sorted([d for d in intra if 1.2 <= d < 2.0])

    return {
        "n_atoms": n_atoms,
        "volume_ang3": volume,
        "atomic_number_density_per_ang3": number_density,
        "molecular_density_per_ang3": molecular_density,
        "intra_oh_distances_ang": o_h,
        "intra_hh_distance_ang": h_h[0] if h_h else None,
        "simulated_time_ps": float(water["time_step_fs"]) * float(water["md_steps"]) / 1000.0,
    }


def adsorption_metrics(adsorption: dict) -> pd.DataFrame:
    rows = []
    for metal, lattice in adsorption["metals"].items():
        surface_spacing = lattice / math.sqrt(2.0)
        ads_height = float(adsorption["adsorbate_height_ang"])
        oo = ads_height
        oh = math.sqrt(ads_height**2 + 1.0**2)

        e_o = lennard_jones_like(oo + 0.15 * surface_spacing, sigma=2.2, epsilon=0.22)
        e_oh = lennard_jones_like(oh + 0.10 * surface_spacing, sigma=2.0, epsilon=0.18)
        rows.append(
            {
                "metal": metal,
                "lattice_constant_ang": lattice,
                "surface_spacing_ang": surface_spacing,
                "surrogate_E_O_eV": e_o,
                "surrogate_E_OH_eV": e_oh,
                "surrogate_delta_eV": e_oh - e_o,
            }
        )
    df = pd.DataFrame(rows).sort_values("lattice_constant_ang").reset_index(drop=True)
    fit = np.polyfit(df["surrogate_E_O_eV"], df["surrogate_E_OH_eV"], 1)
    pred = np.polyval(fit, df["surrogate_E_O_eV"])
    ss_res = float(np.sum((df["surrogate_E_OH_eV"] - pred) ** 2))
    ss_tot = float(np.sum((df["surrogate_E_OH_eV"] - df["surrogate_E_OH_eV"].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    df.attrs["fit_slope"] = float(fit[0])
    df.attrs["fit_intercept"] = float(fit[1])
    df.attrs["fit_r2"] = float(r2)
    return df


def reaction_metrics(reactions: dict, dft_barriers: dict) -> pd.DataFrame:
    rows = []
    bond_reference = {
        frozenset(("H", "H")): (0.74, 0.5),
        frozenset(("C", "H")): (1.09, 0.7),
        frozenset(("C", "C")): (1.54, 0.9),
        frozenset(("C", "O")): (1.43, 0.8),
        frozenset(("O", "H")): (0.96, 0.8),
    }

    def surrogate_energy(atoms: list[dict]) -> float:
        energy = 0.0
        coords = np.array([a["coords"] for a in atoms], dtype=float)
        elems = [a["element"] for a in atoms]
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                pair = frozenset((elems[i], elems[j]))
                r0, weight = bond_reference.get(pair, (1.8, 0.2))
                r = float(np.linalg.norm(coords[i] - coords[j]))
                energy += weight * (r - r0) ** 2
        return energy

    for rxn, payload in reactions.items():
        e_r = surrogate_energy(payload["reactant"])
        e_ts = surrogate_energy(payload["transition_state"])
        barrier = e_ts - e_r
        rows.append(
            {
                "reaction": rxn,
                "label": payload["label"],
                "surrogate_reactant_energy": e_r,
                "surrogate_ts_energy": e_ts,
                "surrogate_barrier_eV": barrier,
                "dft_barrier_eV": dft_barriers.get(rxn, np.nan),
            }
        )
    df = pd.DataFrame(rows)
    df["abs_error_eV"] = (df["surrogate_barrier_eV"] - df["dft_barrier_eV"]).abs()
    return df


def coverage_metrics(parsed: dict) -> dict:
    counter = Counter()
    for atom in parsed["water"].get("molecule_coords", []):
        counter[atom["element"]] += parsed["water"].get("n_molecules", 1)
    for atom in parsed["adsorption"].get("gas_phase", []):
        counter[atom["element"]] += 1
    for payload in parsed["reactions"].values():
        for state in ("reactant", "transition_state"):
            for atom in payload.get(state, []):
                counter[atom["element"]] += 1

    categories = defaultdict(int)
    categories["liquid"] = 1
    categories["surface_catalysis"] = len(parsed["adsorption"]["metals"])
    categories["reaction_barriers"] = len(parsed["reactions"])

    return {
        "element_counts": dict(counter),
        "n_unique_elements": len(counter),
        "task_categories": dict(categories),
        "n_metals_in_adsorption_suite": len(parsed["adsorption"]["metals"]),
        "n_reaction_cases": len(parsed["reactions"]),
    }


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_figures(coverage: dict, water: dict, adsorption: pd.DataFrame, reactions: pd.DataFrame) -> None:
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    elems = sorted(coverage["element_counts"])
    counts = [coverage["element_counts"][e] for e in elems]
    axes[0].bar(elems, counts, color=["#1b9e77", "#d95f02", "#7570b3", "#e7298a"])
    axes[0].set_title("Element Coverage in Local Reproduction Suite")
    axes[0].set_ylabel("Count")

    cats = list(coverage["task_categories"].keys())
    vals = list(coverage["task_categories"].values())
    axes[1].bar(cats, vals, color=["#66a61e", "#e6ab02", "#a6761d"])
    axes[1].set_title("Task Diversity Across Validation Axes")
    axes[1].set_ylabel("Cases")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "coverage_overview.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    oh = water["intra_oh_distances_ang"]
    axes[0].bar(range(len(oh)), oh, color="#1f78b4")
    axes[0].set_title("Water Molecule Internal O-H Distances")
    axes[0].set_ylabel("Distance (Angstrom)")
    axes[0].set_xticks(range(len(oh)))
    axes[0].set_xticklabels([f"bond {i+1}" for i in range(len(oh))])

    metrics = ["atomic_number_density_per_ang3", "molecular_density_per_ang3", "simulated_time_ps"]
    labels = ["Atomic density", "Molecular density", "Trajectory length (ps)"]
    vals = [water[m] for m in metrics]
    axes[1].bar(labels, vals, color=["#33a02c", "#fb9a99", "#ff7f00"])
    axes[1].set_title("Water Benchmark Operating Point")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "water_setup.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(adsorption["surrogate_E_O_eV"], adsorption["surrogate_E_OH_eV"], s=70, color="#e31a1c")
    for _, row in adsorption.iterrows():
        ax.annotate(row["metal"], (row["surrogate_E_O_eV"], row["surrogate_E_OH_eV"]), xytext=(5, 4), textcoords="offset points")
    x = np.linspace(adsorption["surrogate_E_O_eV"].min() - 0.02, adsorption["surrogate_E_O_eV"].max() + 0.02, 100)
    y = adsorption.attrs["fit_slope"] * x + adsorption.attrs["fit_intercept"]
    ax.plot(x, y, color="#1f78b4", linewidth=2)
    ax.set_xlabel("Surrogate O adsorption energy (eV)")
    ax.set_ylabel("Surrogate OH adsorption energy (eV)")
    ax.set_title(f"Adsorption Scaling Surrogate (R^2={adsorption.attrs['fit_r2']:.3f})")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "adsorption_scaling.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    idx = np.arange(len(reactions))
    width = 0.36
    ax.bar(idx - width / 2, reactions["dft_barrier_eV"], width=width, label="DFT reference", color="#6a3d9a")
    ax.bar(idx + width / 2, reactions["surrogate_barrier_eV"], width=width, label="Geometry surrogate", color="#b15928")
    ax.set_xticks(idx)
    ax.set_xticklabels(reactions["reaction"])
    ax.set_ylabel("Barrier (eV)")
    ax.set_title("Reaction Barrier Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES / "reaction_barriers.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_IMAGES.mkdir(parents=True, exist_ok=True)

    parsed = parse_dataset(DATA_FILE.read_text(encoding="utf-8"))
    water = water_metrics(parsed["water"])
    adsorption = adsorption_metrics(parsed["adsorption"])
    reactions = reaction_metrics(parsed["reactions"], parsed["dft_barriers"])
    coverage = coverage_metrics(parsed)

    save_json(OUTPUTS / "parsed_dataset.json", parsed)
    save_json(OUTPUTS / "water_metrics.json", water)
    save_json(OUTPUTS / "coverage_metrics.json", coverage)
    adsorption.to_csv(OUTPUTS / "adsorption_metrics.csv", index=False)
    reactions.to_csv(OUTPUTS / "reaction_metrics.csv", index=False)

    summary = {
        "adsorption_fit_slope": adsorption.attrs["fit_slope"],
        "adsorption_fit_intercept": adsorption.attrs["fit_intercept"],
        "adsorption_fit_r2": adsorption.attrs["fit_r2"],
        "reaction_mae_eV": float(reactions["abs_error_eV"].mean()),
        "reaction_max_abs_error_eV": float(reactions["abs_error_eV"].max()),
        "n_unique_elements": coverage["n_unique_elements"],
        "n_task_categories": len(coverage["task_categories"]),
    }
    save_json(OUTPUTS / "summary_metrics.json", summary)

    make_figures(coverage, water, adsorption, reactions)


if __name__ == "__main__":
    main()
