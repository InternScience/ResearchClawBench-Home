"""
Parse M-AI-Synth__Materials_AI_Dataset_.txt into three structured datasets.

The raw file is partitioned into three blocks (Chinese-comment headers):
  # 文件1: property_prediction.py 数据   -> 4 rows
  # 文件2: structure_generation.py 数据  -> 2 rows
  # 文件3: autonomous_optimization.py 数据 -> 6 single-value rows

Block 1 (property prediction; CGCNN-style toy graph):
  row1: atomic numbers (length N=120)
  row2: 3D coordinate-like scalars (length N=120; we treat them as a
         scalar atomic-feature 'x_pos' on top of Z to demonstrate the
         message-passing graph regression)
  row3: edge index pairs (length 20 -> 10 undirected edges across the
         first 5 atoms; these define a small reusable graph topology)
  row4: per-atom target property values (length 95)

Because the file is intentionally toy-sized, we synthesize a *consistent*
graph regression dataset on top of these primitives:
  - We treat the 95-length vector as 95 graph-level target labels
    y_g for 95 distinct micro-crystals.
  - Each micro-crystal uses the SAME 5-atom topology (row3) but with
    a different 5-atom Z-and-coordinate sub-window from rows 1/2.
  - The graph-level target is taken from row4 (per-graph scalar).
This faithfully captures the property-prediction workflow described
in M-AI-Synth while being deterministically reproducible.

Block 2 (structure generation): two coordinate sequences (a, b),
each of length 100. Treated as 100 (a, b) lattice-parameter
samples for a 2-D VAE.

Block 3 (autonomous optimization):
  T_range = [200.0, 500.0]
  t_range = [10.0, 30.0]
  T_target = 350.0
  t_target = 20.0
  noise   = 0.1
  threshold = 10.0
"""

from __future__ import annotations
import re, json, ast
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "M-AI-Synth__Materials_AI_Dataset_.txt"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def parse_lists(text: str):
    """Return list of python lists, one per `[...]` block."""
    out = []
    for m in re.finditer(r"\[([^\[\]]*)\]", text):
        body = m.group(1).strip()
        if not body:
            out.append([])
            continue
        out.append(ast.literal_eval("[" + body + "]"))
    return out


def parse_dataset():
    text = DATA_PATH.read_text(encoding="utf-8")
    # Split by the three section markers.
    sections = re.split(r"#\s*文件\d+:[^\n]*\n", text)
    # sections[0] is preamble (empty); sections[1..3] are the three blocks.
    sections = [s for s in sections if s.strip()]
    assert len(sections) == 3, f"expected 3 blocks, got {len(sections)}"

    blk1 = parse_lists(sections[0])  # 4 lists
    blk2 = parse_lists(sections[1])  # 2 lists
    blk3 = parse_lists(sections[2])  # 6 single-element lists

    # ---- Block 1: property prediction ----
    Z = np.array(blk1[0], dtype=np.int64)        # atomic numbers (uniform 5 here)
    pos = np.array(blk1[1], dtype=np.float32)    # scalar atomic feature
    edges_flat = np.array(blk1[2], dtype=np.int64)
    edges = edges_flat.reshape(-1, 2)            # 10 edges, indices into 5 atoms
    y_all = np.array(blk1[3], dtype=np.float32)  # 95 graph targets

    # Build sliding 5-atom windows over (Z, pos). Trim to the smallest
    # count supported by all three primitives so every graph has valid
    # (Z, X, y).
    win = 5
    n_graphs = int(min(len(y_all), len(Z) - win + 1, len(pos) - win + 1))
    Z_g = np.stack([Z[i:i + win] for i in range(n_graphs)])
    X_g = np.stack([pos[i:i + win] for i in range(n_graphs)])
    y_all = y_all[:n_graphs]
    # Edge list is shared across graphs.

    pp = dict(
        Z=Z_g, X=X_g, edges=edges, y=y_all,
        n_graphs=n_graphs, n_atoms=win, n_edges=int(edges.shape[0]),
    )

    # ---- Block 2: structure generation ----
    a = np.array(blk2[0], dtype=np.float32)
    b = np.array(blk2[1], dtype=np.float32)
    sg = dict(a=a, b=b, n=len(a))

    # ---- Block 3: autonomous optimization ----
    ao = dict(
        T_range=tuple(blk3[0]),
        t_range=tuple(blk3[1]),
        T_target=float(blk3[2][0]),
        t_target=float(blk3[3][0]),
        noise=float(blk3[4][0]),
        threshold=float(blk3[5][0]),
    )

    return pp, sg, ao


if __name__ == "__main__":
    pp, sg, ao = parse_dataset()
    summary = {
        "property_prediction": {
            "n_graphs": pp["n_graphs"],
            "n_atoms_per_graph": pp["n_atoms"],
            "n_edges": pp["n_edges"],
            "atomic_number_unique": sorted(set(pp["Z"].flatten().tolist())),
            "y_min": float(pp["y"].min()),
            "y_max": float(pp["y"].max()),
            "y_mean": float(pp["y"].mean()),
            "y_std": float(pp["y"].std()),
        },
        "structure_generation": {
            "n_samples": sg["n"],
            "a_mean": float(sg["a"].mean()),
            "a_std": float(sg["a"].std()),
            "b_mean": float(sg["b"].mean()),
            "b_std": float(sg["b"].std()),
            "a_min": float(sg["a"].min()),
            "a_max": float(sg["a"].max()),
            "b_min": float(sg["b"].min()),
            "b_max": float(sg["b"].max()),
        },
        "autonomous_optimization": {
            "T_range": list(ao["T_range"]),
            "t_range": list(ao["t_range"]),
            "T_target": ao["T_target"],
            "t_target": ao["t_target"],
            "noise": ao["noise"],
            "threshold": ao["threshold"],
        },
    }
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(OUT / "parsed_data.npz",
             pp_Z=pp["Z"], pp_X=pp["X"], pp_edges=pp["edges"], pp_y=pp["y"],
             sg_a=sg["a"], sg_b=sg["b"])
    print(json.dumps(summary, indent=2))
