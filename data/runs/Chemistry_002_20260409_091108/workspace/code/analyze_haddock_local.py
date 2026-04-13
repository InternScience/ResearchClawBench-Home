#!/usr/bin/env python3
import csv
import math
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


RT_KCAL = 0.0019872041 * 298.15
PDB_PATH = "data/1brs_AD.pdb"
SKEMPI_PATH = "data/skempi_v2.csv"
OUT_DIR = "outputs"
FIG_DIR = os.path.join("report", "images")

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_PROPS = {
    "A": {"hydro": 1.8, "vol": 88.6, "charge": 0},
    "R": {"hydro": -4.5, "vol": 173.4, "charge": 1},
    "N": {"hydro": -3.5, "vol": 114.1, "charge": 0},
    "D": {"hydro": -3.5, "vol": 111.1, "charge": -1},
    "C": {"hydro": 2.5, "vol": 108.5, "charge": 0},
    "Q": {"hydro": -3.5, "vol": 143.8, "charge": 0},
    "E": {"hydro": -3.5, "vol": 138.4, "charge": -1},
    "G": {"hydro": -0.4, "vol": 60.1, "charge": 0},
    "H": {"hydro": -3.2, "vol": 153.2, "charge": 0},
    "I": {"hydro": 4.5, "vol": 166.7, "charge": 0},
    "L": {"hydro": 3.8, "vol": 166.7, "charge": 0},
    "K": {"hydro": -3.9, "vol": 168.6, "charge": 1},
    "M": {"hydro": 1.9, "vol": 162.9, "charge": 0},
    "F": {"hydro": 2.8, "vol": 189.9, "charge": 0},
    "P": {"hydro": -1.6, "vol": 112.7, "charge": 0},
    "S": {"hydro": -0.8, "vol": 89.0, "charge": 0},
    "T": {"hydro": -0.7, "vol": 116.1, "charge": 0},
    "W": {"hydro": -0.9, "vol": 227.8, "charge": 0},
    "Y": {"hydro": -1.3, "vol": 193.6, "charge": 0},
    "V": {"hydro": 4.2, "vol": 140.0, "charge": 0},
}


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def parse_pdb(path):
    atoms = []
    residues = {}
    with open(path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21].strip()
            resi = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atoms.append({
                "chain": chain,
                "resi": resi,
                "resn": resn,
                "atom": atom,
                "coord": np.array([x, y, z], dtype=float),
            })
            residues.setdefault((chain, resi), {"resn": resn, "atoms": []})
            residues[(chain, resi)]["atoms"].append(atom)
    return atoms, residues


def residue_atom_map(atoms):
    by_res = defaultdict(list)
    for atom in atoms:
        by_res[(atom["chain"], atom["resi"])].append(atom)
    return by_res


def cb_or_ca(atom_list):
    preferred = None
    fallback = None
    for atom in atom_list:
        if atom["atom"] == "CB":
            preferred = atom["coord"]
        if atom["atom"] == "CA":
            fallback = atom["coord"]
    return preferred if preferred is not None else fallback


def compute_interface_features(by_res):
    a_res = sorted(k for k in by_res if k[0] == "A")
    d_res = sorted(k for k in by_res if k[0] == "D")
    res_features = {}

    for key, atom_list in by_res.items():
        center = np.mean([a["coord"] for a in atom_list], axis=0)
        anchor = cb_or_ca(atom_list)
        res_features[key] = {
            "resn": atom_list[0]["resn"],
            "center": center,
            "anchor": anchor,
            "min_partner_dist": float("inf"),
            "contact_count_5A": 0,
            "contact_count_8A": 0,
        }

    for a_key in a_res:
        for d_key in d_res:
            a_atoms = by_res[a_key]
            d_atoms = by_res[d_key]
            min_dist = min(
                np.linalg.norm(a["coord"] - d["coord"])
                for a in a_atoms
                for d in d_atoms
            )
            if min_dist < res_features[a_key]["min_partner_dist"]:
                res_features[a_key]["min_partner_dist"] = min_dist
            if min_dist < res_features[d_key]["min_partner_dist"]:
                res_features[d_key]["min_partner_dist"] = min_dist
            if min_dist < 5.0:
                res_features[a_key]["contact_count_5A"] += 1
                res_features[d_key]["contact_count_5A"] += 1
            if min_dist < 8.0:
                res_features[a_key]["contact_count_8A"] += 1
                res_features[d_key]["contact_count_8A"] += 1

    centers = np.array([feat["center"] for feat in res_features.values()])
    global_center = centers.mean(axis=0)
    for feat in res_features.values():
        feat["radial_distance"] = float(np.linalg.norm(feat["center"] - global_center))
        feat["is_interface"] = int(feat["min_partner_dist"] < 5.0)

    return res_features


def parse_mutation_token(token):
    token = token.strip()
    match = re.fullmatch(r"([A-Z])([A-Z])(\d+)([A-Z])", token)
    if not match:
        raise ValueError(f"Cannot parse mutation token: {token}")
    wt, chain, resi, mut = match.groups()
    return {"wt": wt, "chain": chain, "resi": int(resi), "mut": mut}


def load_skempi_rows():
    rows = []
    with open(SKEMPI_PATH, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if not row["#Pdb"].upper().startswith("1BRS"):
                continue
            try:
                kd_mut = float(row["Affinity_mut_parsed"])
                kd_wt = float(row["Affinity_wt_parsed"])
            except (TypeError, ValueError):
                continue
            ddg = RT_KCAL * math.log(kd_mut / kd_wt)
            tokens = [parse_mutation_token(tok) for tok in row["Mutation(s)_cleaned"].split(",")]
            rows.append({
                "mutation_cleaned": row["Mutation(s)_cleaned"],
                "mutation_pdb": row["Mutation(s)_PDB"],
                "location": row["iMutation_Location(s)"],
                "protein_1": row["Protein 1"],
                "protein_2": row["Protein 2"],
                "method": row["Method"],
                "temperature": row["Temperature"],
                "kd_mut": kd_mut,
                "kd_wt": kd_wt,
                "ddg": ddg,
                "mutations": tokens,
            })
    return rows


def mutation_features(rows, res_features):
    out = []
    for row in rows:
        muts = row["mutations"]
        known = []
        for mut in muts:
            key = (mut["chain"], mut["resi"])
            if key not in res_features:
                continue
            feat = res_features[key]
            props_wt = AA_PROPS[mut["wt"]]
            props_mut = AA_PROPS[mut["mut"]]
            known.append({
                "chain": mut["chain"],
                "resi": mut["resi"],
                "wt": mut["wt"],
                "mut": mut["mut"],
                "resn": feat["resn"],
                "is_interface": feat["is_interface"],
                "min_partner_dist": feat["min_partner_dist"],
                "contact_count_5A": feat["contact_count_5A"],
                "contact_count_8A": feat["contact_count_8A"],
                "radial_distance": feat["radial_distance"],
                "delta_hydro": props_mut["hydro"] - props_wt["hydro"],
                "delta_vol": props_mut["vol"] - props_wt["vol"],
                "delta_charge": props_mut["charge"] - props_wt["charge"],
                "to_ala": int(mut["mut"] == "A"),
            })
        if not known:
            continue
        out.append({
            **row,
            "n_mut": len(known),
            "interface_fraction": sum(m["is_interface"] for m in known) / len(known),
            "any_interface": int(any(m["is_interface"] for m in known)),
            "mean_min_partner_dist": float(np.mean([m["min_partner_dist"] for m in known])),
            "sum_contacts_5A": sum(m["contact_count_5A"] for m in known),
            "sum_contacts_8A": sum(m["contact_count_8A"] for m in known),
            "mean_radial_distance": float(np.mean([m["radial_distance"] for m in known])),
            "sum_abs_delta_hydro": float(sum(abs(m["delta_hydro"]) for m in known)),
            "sum_abs_delta_vol": float(sum(abs(m["delta_vol"]) for m in known)),
            "sum_abs_delta_charge": float(sum(abs(m["delta_charge"]) for m in known)),
            "all_to_ala": int(all(m["to_ala"] for m in known)),
            "chains": ",".join(sorted({m["chain"] for m in known})),
            "mutation_sites": ",".join(f"{m['chain']}{m['resi']}" for m in known),
            "known_mutations": known,
        })
    return out


def fit_linear_regression(X, y):
    X_design = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    pred = X_design @ coef
    return coef, pred


def corr(a, b):
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(features, res_features, literature_notes, stats):
    ddg = np.array([r["ddg"] for r in features], dtype=float)
    n_mut = np.array([r["n_mut"] for r in features], dtype=float)
    contacts = np.array([r["sum_contacts_5A"] for r in features], dtype=float)
    interface = np.array([r["any_interface"] for r in features], dtype=float)
    pred_geom = np.array([r["pred_geom"] for r in features], dtype=float)
    pred_full = np.array([r["pred_full"] for r in features], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    chains = Counter(k[0] for k in res_features)
    iface = Counter(k[0] for k, v in res_features.items() if v["is_interface"])
    axes[0].bar(["A total", "D total", "A interface", "D interface"], [
        chains["A"], chains["D"], iface["A"], iface["D"]
    ], color=["#5B8FF9", "#61DDAA", "#5B8FF9", "#61DDAA"])
    axes[0].set_ylabel("Residues")
    axes[0].set_title("1BRS structure overview")

    muts_by_cat = Counter()
    for row in features:
        if row["n_mut"] == 1 and row["any_interface"]:
            muts_by_cat["single-interface"] += 1
        elif row["n_mut"] == 1:
            muts_by_cat["single-noninterface"] += 1
        elif row["any_interface"]:
            muts_by_cat["multi-with-interface"] += 1
        else:
            muts_by_cat["multi-noninterface"] += 1
    labels = list(muts_by_cat.keys())
    axes[1].bar(labels, [muts_by_cat[k] for k in labels], color="#F6BD16")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("SKEMPI entries")
    axes[1].set_title("Mutation assay composition")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "data_overview.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    groups = {
        "interface": ddg[interface == 1],
        "non-interface": ddg[interface == 0],
    }
    axes[0].boxplot([groups["interface"], groups["non-interface"]], tick_labels=["Interface", "Non-interface"])
    axes[0].set_ylabel("ddG (kcal/mol)")
    axes[0].set_title("Interface mutations are more disruptive")

    axes[1].scatter(contacts, ddg, c=n_mut, cmap="viridis", edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Sum of inter-chain residue contacts (<5 A)")
    axes[1].set_ylabel("ddG (kcal/mol)")
    axes[1].set_title("Geometric contact burden vs affinity loss")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "interface_validation.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(ddg, pred_geom, color="#5B8FF9", edgecolor="black", linewidth=0.3)
    lim = [min(ddg.min(), pred_geom.min(), pred_full.min()) - 0.5, max(ddg.max(), pred_geom.max(), pred_full.max()) + 0.5]
    axes[0].plot(lim, lim, linestyle="--", color="gray")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Observed ddG (kcal/mol)")
    axes[0].set_ylabel("Predicted ddG")
    axes[0].set_title(f"Geometry baseline (r={stats['corr_geom']:.2f})")

    axes[1].scatter(ddg, pred_full, color="#61DDAA", edgecolor="black", linewidth=0.3)
    axes[1].plot(lim, lim, linestyle="--", color="gray")
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("Observed ddG (kcal/mol)")
    axes[1].set_ylabel("Predicted ddG")
    axes[1].set_title(f"Geometry+chemistry baseline (r={stats['corr_full']:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=200)
    plt.close(fig)


def main():
    ensure_dirs()
    atoms, _ = parse_pdb(PDB_PATH)
    by_res = residue_atom_map(atoms)
    res_features = compute_interface_features(by_res)
    skempi_rows = load_skempi_rows()
    features = mutation_features(skempi_rows, res_features)

    y = np.array([r["ddg"] for r in features], dtype=float)
    X_geom = np.column_stack([
        [r["n_mut"] for r in features],
        [r["any_interface"] for r in features],
        [r["sum_contacts_5A"] for r in features],
        [r["mean_min_partner_dist"] for r in features],
    ])
    X_full = np.column_stack([
        [r["n_mut"] for r in features],
        [r["any_interface"] for r in features],
        [r["sum_contacts_5A"] for r in features],
        [r["mean_min_partner_dist"] for r in features],
        [r["sum_abs_delta_hydro"] for r in features],
        [r["sum_abs_delta_vol"] for r in features],
        [r["sum_abs_delta_charge"] for r in features],
        [r["all_to_ala"] for r in features],
    ])

    coef_geom, pred_geom = fit_linear_regression(X_geom, y)
    coef_full, pred_full = fit_linear_regression(X_full, y)

    for row, pg, pf in zip(features, pred_geom, pred_full):
        row["pred_geom"] = float(pg)
        row["pred_full"] = float(pf)
        row["abs_err_geom"] = abs(row["ddg"] - pg)
        row["abs_err_full"] = abs(row["ddg"] - pf)

    residue_rows = []
    for (chain, resi), feat in sorted(res_features.items()):
        residue_rows.append({
            "chain": chain,
            "resi": resi,
            "resn": feat["resn"],
            "min_partner_dist": round(feat["min_partner_dist"], 4),
            "contact_count_5A": feat["contact_count_5A"],
            "contact_count_8A": feat["contact_count_8A"],
            "radial_distance": round(feat["radial_distance"], 4),
            "is_interface": feat["is_interface"],
        })

    write_csv(
        os.path.join(OUT_DIR, "residue_interface_features.csv"),
        residue_rows,
        ["chain", "resi", "resn", "min_partner_dist", "contact_count_5A", "contact_count_8A", "radial_distance", "is_interface"],
    )

    mutation_rows = []
    for row in features:
        mutation_rows.append({
            "mutation_cleaned": row["mutation_cleaned"],
            "mutation_pdb": row["mutation_pdb"],
            "mutation_sites": row["mutation_sites"],
            "chains": row["chains"],
            "location": row["location"],
            "n_mut": row["n_mut"],
            "any_interface": row["any_interface"],
            "interface_fraction": round(row["interface_fraction"], 4),
            "sum_contacts_5A": row["sum_contacts_5A"],
            "mean_min_partner_dist": round(row["mean_min_partner_dist"], 4),
            "sum_abs_delta_hydro": round(row["sum_abs_delta_hydro"], 4),
            "sum_abs_delta_vol": round(row["sum_abs_delta_vol"], 4),
            "sum_abs_delta_charge": round(row["sum_abs_delta_charge"], 4),
            "all_to_ala": row["all_to_ala"],
            "ddg": round(row["ddg"], 4),
            "pred_geom": round(row["pred_geom"], 4),
            "pred_full": round(row["pred_full"], 4),
            "abs_err_geom": round(row["abs_err_geom"], 4),
            "abs_err_full": round(row["abs_err_full"], 4),
        })

    write_csv(
        os.path.join(OUT_DIR, "skempi_1brs_features.csv"),
        mutation_rows,
        [
            "mutation_cleaned", "mutation_pdb", "mutation_sites", "chains", "location", "n_mut",
            "any_interface", "interface_fraction", "sum_contacts_5A", "mean_min_partner_dist",
            "sum_abs_delta_hydro", "sum_abs_delta_vol", "sum_abs_delta_charge", "all_to_ala",
            "ddg", "pred_geom", "pred_full", "abs_err_geom", "abs_err_full"
        ],
    )

    stats = {
        "n_rows": len(features),
        "n_interface_rows": int(sum(r["any_interface"] for r in features)),
        "mean_ddg": float(np.mean(y)),
        "median_ddg": float(np.median(y)),
        "corr_geom": corr(y, pred_geom),
        "corr_full": corr(y, pred_full),
        "mae_geom": float(np.mean(np.abs(y - pred_geom))),
        "mae_full": float(np.mean(np.abs(y - pred_full))),
        "interface_mean_ddg": float(np.mean([r["ddg"] for r in features if r["any_interface"]])),
        "non_interface_mean_ddg": float(np.mean([r["ddg"] for r in features if not r["any_interface"]])),
        "single_mut_rows": int(sum(r["n_mut"] == 1 for r in features)),
        "double_mut_rows": int(sum(r["n_mut"] == 2 for r in features)),
    }

    with open(os.path.join(OUT_DIR, "analysis_summary.txt"), "w") as handle:
        for key, value in stats.items():
            handle.write(f"{key}\t{value}\n")
        handle.write("coef_geom\t" + ",".join(f"{x:.6f}" for x in coef_geom) + "\n")
        handle.write("coef_full\t" + ",".join(f"{x:.6f}" for x in coef_full) + "\n")

    literature_notes = {
        "paper_000": "Original HADDOCK paper: AIR-driven docking with mutagenesis or NMR restraints and ranking by intermolecular energies.",
        "paper_001": "HADDOCK2.0: multistage docking with rigid-body, semi-flexible refinement, and solvent refinement; interface information improves success.",
        "paper_002": "HADDOCK2.2: user-facing integrative modeling platform for mixed biomolecular complexes and additional restraints.",
        "paper_003": "Recent glycan benchmark: HADDOCK remains useful when binding-site knowledge is available, but flexibility limits performance.",
    }

    make_figures(features, res_features, literature_notes, stats)


if __name__ == "__main__":
    main()
