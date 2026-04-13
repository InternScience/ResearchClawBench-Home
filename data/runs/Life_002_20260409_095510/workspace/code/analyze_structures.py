#!/usr/bin/env python3
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_pdb(path: Path):
    seqres = defaultdict(list)
    ca_records = defaultdict(list)
    atom_counts = defaultdict(int)
    with path.open() as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec == "SEQRES":
                chain = line[11].strip()
                residues = line[19:70].split()
                seqres[chain].extend(residues)
            elif rec == "ATOM":
                chain = line[21].strip()
                atom_counts[chain] += 1
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_seq = int(line[22:26])
                icode = line[26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if atom_name == "CA" and res_name in AA3_TO_1:
                    ca_records[chain].append(
                        {
                            "res_name": res_name,
                            "res_seq": res_seq,
                            "icode": icode,
                            "coord": np.array([x, y, z], dtype=float),
                            "aa": AA3_TO_1[res_name],
                        }
                    )
    return {
        "seqres": seqres,
        "ca_records": ca_records,
        "atom_counts": atom_counts,
    }


def seqres_to_1(res_list):
    return "".join(AA3_TO_1.get(res, "X") for res in res_list)


def needleman_wunsch(seq1: str, seq2: str, match=2, mismatch=-1, gap=-2):
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((n + 1, m + 1), dtype=float)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + gap
        trace[i, 0] = 1
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + gap
        trace[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            best = max(diag, up, left)
            score[i, j] = best
            trace[i, j] = 0 if best == diag else (1 if best == up else 2)
    i, j = n, m
    aligned = []
    while i > 0 or j > 0:
        move = trace[i, j]
        if i > 0 and j > 0 and move == 0:
            aligned.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            aligned.append((i - 1, None))
            i -= 1
        else:
            aligned.append((None, j - 1))
            j -= 1
    aligned.reverse()
    return aligned, float(score[n, m])


def kabsch_fit(p: np.ndarray, q: np.ndarray):
    p_centroid = p.mean(axis=0)
    q_centroid = q.mean(axis=0)
    p_centered = p - p_centroid
    q_centered = q - q_centroid
    h = p_centered.T @ q_centered
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = q_centroid - p_centroid @ r.T
    return r, t


def apply_transform(coords: np.ndarray, r: np.ndarray, t: np.ndarray):
    return coords @ r.T + t


def tm_score(distances: np.ndarray, norm_len: int):
    if norm_len <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * ((norm_len - 15) ** (1.0 / 3.0)) - 1.8
        d0 = max(d0, 0.5)
    return float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / norm_len)


def chain_type(seqres_residues):
    if not seqres_residues:
        return "unknown"
    protein_like = sum(1 for r in seqres_residues if r in AA3_TO_1)
    frac = protein_like / len(seqres_residues)
    return "protein" if frac >= 0.8 else "nucleic_acid_or_other"


def extract_chain_sequence(parsed, chain):
    seqres = parsed["seqres"].get(chain, [])
    seq = seqres_to_1(seqres)
    if seq and set(seq) != {"X"}:
        return seq
    return "".join(rec["aa"] for rec in parsed["ca_records"].get(chain, []))


def compare_chain_to_target(query_chain, query_seq, query_records, target_seq, target_records):
    alignment, nw_score = needleman_wunsch(query_seq, target_seq)
    q_map = {i: rec for i, rec in enumerate(query_records)}
    t_map = {i: rec for i, rec in enumerate(target_records)}
    matched = []
    for qi, ti in alignment:
        if qi is None or ti is None:
            continue
        if qi in q_map and ti in t_map:
            matched.append((qi, ti, q_map[qi], t_map[ti]))
    if len(matched) < 3:
        return {
            "query_chain": query_chain,
            "matched_pairs": len(matched),
            "nw_score": nw_score,
        }
    p = np.vstack([m[2]["coord"] for m in matched])
    q = np.vstack([m[3]["coord"] for m in matched])
    r, t = kabsch_fit(p, q)
    p_fit = apply_transform(p, r, t)
    distances = np.linalg.norm(p_fit - q, axis=1)
    rmsd = float(np.sqrt(np.mean(distances ** 2)))
    seq_identity = float(np.mean([query_seq[m[0]] == target_seq[m[1]] for m in matched]))
    return {
        "query_chain": query_chain,
        "query_length": len(query_seq),
        "target_length": len(target_seq),
        "matched_pairs": len(matched),
        "coverage_query": len(matched) / max(1, len(query_seq)),
        "coverage_target": len(matched) / max(1, len(target_seq)),
        "seq_identity": seq_identity,
        "rmsd": rmsd,
        "tm_query_norm": tm_score(distances, len(query_seq)),
        "tm_target_norm": tm_score(distances, len(target_seq)),
        "tm_average_norm": tm_score(distances, int(round((len(query_seq) + len(target_seq)) / 2))),
        "nw_score": nw_score,
        "rotation_matrix": r.tolist(),
        "translation_vector": t.tolist(),
        "distance_summary": {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p90": float(np.percentile(distances, 90)),
            "max": float(np.max(distances)),
        },
        "aligned_residue_pairs": [
            {
                "query_index": int(m[0]),
                "target_index": int(m[1]),
                "query_resseq": int(m[2]["res_seq"]),
                "target_resseq": int(m[3]["res_seq"]),
                "query_aa": m[2]["aa"],
                "target_aa": m[3]["aa"],
                "distance": float(d),
            }
            for m, d in zip(matched, distances)
        ],
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    query = parse_pdb(DATA_DIR / "7xg4.pdb")
    target = parse_pdb(DATA_DIR / "6n40.pdb")

    target_chain = "A"
    target_seq = extract_chain_sequence(target, target_chain)
    target_records = target["ca_records"][target_chain]

    chain_rows = []
    protein_results = []
    for chain in sorted(query["seqres"].keys() | query["ca_records"].keys()):
        residues = query["seqres"].get(chain, [])
        ctype = chain_type(residues)
        seq = extract_chain_sequence(query, chain)
        ca_count = len(query["ca_records"].get(chain, []))
        chain_rows.append(
            {
                "chain": chain,
                "type": ctype,
                "seqres_length": len(residues),
                "ca_count": ca_count,
                "atom_count": query["atom_counts"].get(chain, 0),
            }
        )
        if ctype == "protein" and ca_count >= 20:
            protein_results.append(compare_chain_to_target(chain, seq, query["ca_records"][chain], target_seq, target_records))

    chain_df = pd.DataFrame(chain_rows).sort_values("chain")
    result_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in res.items()
                if k not in {"rotation_matrix", "translation_vector", "distance_summary", "aligned_residue_pairs"}
            }
            for res in protein_results
        ]
    ).sort_values(["tm_average_norm", "tm_query_norm", "matched_pairs"], ascending=False)

    best = max(protein_results, key=lambda x: x.get("tm_average_norm", -math.inf))

    chain_df.to_csv(OUTPUT_DIR / "query_chain_summary.csv", index=False)
    result_df.to_csv(OUTPUT_DIR / "chain_vs_target_metrics.csv", index=False)
    with (OUTPUT_DIR / "best_alignment.json").open("w") as handle:
        json.dump(best, handle, indent=2)

    fig, ax = plt.subplots(figsize=(9, 4))
    palette = {"protein": "#1f77b4", "nucleic_acid_or_other": "#ff7f0e", "unknown": "#7f7f7f"}
    ax.bar(chain_df["chain"], chain_df["ca_count"], color=[palette[t] for t in chain_df["type"]])
    ax.set_xlabel("7xg4 chain")
    ax.set_ylabel("Observed Cα atoms")
    ax.set_title("Chain composition of query complex 7xg4")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "query_chain_overview.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = result_df.sort_values("tm_average_norm", ascending=True)
    ax.barh(ordered["query_chain"], ordered["tm_average_norm"], color="#2ca02c")
    ax.set_xlabel("TM-score (average-length normalization)")
    ax.set_ylabel("7xg4 protein chain")
    ax.set_title("Per-chain similarity to target structure 6n40 chain A")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "chain_tm_scores.png", dpi=200)
    plt.close(fig)

    best_pairs = pd.DataFrame(best["aligned_residue_pairs"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter = ax.scatter(
        best_pairs["query_index"],
        best_pairs["target_index"],
        c=best_pairs["distance"],
        cmap="viridis",
        s=24,
    )
    ax.set_xlabel(f"7xg4 chain {best['query_chain']} aligned residue index")
    ax.set_ylabel("6n40 chain A aligned residue index")
    ax.set_title("Best chain-level correspondence map")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Post-fit Cα distance (Å)")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "best_chain_correspondence.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
