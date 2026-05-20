#!/usr/bin/env python3
"""
Structural alignment analysis between 7xg4 (CRISPR-Cas complex) and 6n40 (MmpL3).
Uses tmtools for TM-align style computation.
Outputs: metrics JSON, figures, chain info.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, Superimposer
from tmtools import tm_align
from tmtools.io import get_residue_data

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
REPORT_DIR = Path("report/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(Path(pdb_path).stem, pdb_path)

def get_ca_coords(structure):
    """Extract CA coordinates and residue list for first model."""
    model = structure[0]
    coords = []
    residues = []
    for chain in model:
        for res in chain:
            if "CA" in res:
                coords.append(res["CA"].get_coord())
                residues.append((chain.id, res.get_id()[1]))  # (chain, resnum)
    return np.array(coords), residues

def compute_alignment(ref_pdb, target_pdb):
    """Compute TM-align style alignment using tmtools."""
    ref_struct = load_structure(ref_pdb)
    tgt_struct = load_structure(target_pdb)

    ref_coords, ref_res = get_ca_coords(ref_struct)
    tgt_coords, tgt_res = get_ca_coords(tgt_struct)

    # tm_align expects (coords, seq) but we use dummy seq
    ref_seq = "A" * len(ref_coords)
    tgt_seq = "A" * len(tgt_coords)

    res = tm_align(ref_coords, tgt_coords, ref_seq, tgt_seq)
    tm_score = res.tm_norm_chain1  # normalized to ref
    rmsd = res.rmsd
    # rotation and translation from tmtools result
    rotation = res.u  # 3x3
    translation = res.t  # 3

    return {
        "tm_score": float(tm_score),
        "rmsd": float(rmsd),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "ref_chains": [c.id for c in ref_struct[0]],
        "tgt_chains": [c.id for c in tgt_struct[0]],
        "ref_nres": len(ref_coords),
        "tgt_nres": len(tgt_coords),
    }

def plot_metrics(metrics, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # TM-score bar
    ax = axes[0]
    ax.bar(["7xg4 vs 6n40"], [metrics["tm_score"]], color="steelblue")
    ax.set_ylabel("TM-score")
    ax.set_title("TM-score (normalized to 7xg4)")
    ax.axhline(0.5, color="red", linestyle="--", label="Fold similarity threshold")
    ax.legend()
    ax.set_ylim(0, 1)

    # RMSD bar
    ax = axes[1]
    ax.bar(["7xg4 vs 6n40"], [metrics["rmsd"]], color="coral")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD after superposition")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ref = DATA_DIR / "7xg4.pdb"
    tgt = DATA_DIR / "6n40.pdb"

    metrics = compute_alignment(ref, tgt)

    # Save metrics
    with open(OUTPUT_DIR / "alignment_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save superimposition vectors
    np.save(OUTPUT_DIR / "rotation.npy", np.array(metrics["rotation"]))
    np.save(OUTPUT_DIR / "translation.npy", np.array(metrics["translation"]))

    # Plot
    plot_metrics(metrics, REPORT_DIR / "figure1_metrics.png")

    print("Alignment complete. TM-score:", metrics["tm_score"], "RMSD:", metrics["rmsd"])
    print("Outputs saved to", OUTPUT_DIR)
    print("Figures saved to", REPORT_DIR)

if __name__ == "__main__":
    main()
