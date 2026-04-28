"""Aggregate all alignment outputs into a single summary JSON for the report."""
import json, re
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]


def parse_usalign_complex():
    txt = (WORK / "outputs" / "usalign" / "usalign_mm1_full.txt").read_text()
    out = {}
    m = re.search(r"Length of Structure_1:\s+(\d+)", txt); out["L_query"] = int(m.group(1))
    m = re.search(r"Length of Structure_2:\s+(\d+)", txt); out["L_target"] = int(m.group(1))
    m = re.search(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+),\s*Seq_ID=n_identical/n_aligned=\s*([\d.]+)", txt)
    out["L_aligned"] = int(m.group(1)); out["RMSD"] = float(m.group(2)); out["seqID_aligned"] = float(m.group(3))
    m = re.search(r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_1", txt)
    out["TM_norm_q"] = float(m.group(1))
    m = re.search(r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_2", txt)
    out["TM_norm_t"] = float(m.group(1))
    # rotation matrix
    mat = (WORK / "outputs" / "usalign" / "7xg4_vs_6n40_mm1.matrix").read_text()
    rows = []
    for ln in mat.splitlines():
        ln = ln.strip()
        if ln and ln[0].isdigit():
            p = ln.split(); rows.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    out["t"] = [r[0] for r in rows]
    out["U"] = [r[1:] for r in rows]
    return out


def parse_foldseek_easy():
    rows = []
    with open(WORK / "outputs" / "foldseek" / "easy_search.tsv") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rows.append({
                "query_chain": f[0].replace("7xg4_prot_", ""),
                "target_chain": "A",
                "qstart": int(f[2]), "qend": int(f[3]),
                "tstart": int(f[4]), "tend": int(f[5]),
                "evalue": float(f[6]), "bits": float(f[7]), "prob": float(f[8]),
                "alntmscore": float(f[9]), "qtmscore": float(f[10]),
                "ttmscore": float(f[11]), "rmsd": float(f[12]),
                "lddt": float(f[13]), "qcov": float(f[14]), "tcov": float(f[15]),
                "u": [float(x) for x in f[16].split(",")],
                "t": [float(x) for x in f[17].split(",")],
            })
    return rows


def parse_chain_table():
    rows = []
    with open(WORK / "outputs" / "usalign" / "tm_matrix.tsv") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, f)))
    return rows


summary = {
    "complex_alignment_USalign_mm1": parse_usalign_complex(),
    "chain_pair_USalign": parse_chain_table(),
    "chain_pair_Foldseek_TMalign": parse_foldseek_easy(),
}

with open(WORK / "outputs" / "summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(json.dumps(summary["complex_alignment_USalign_mm1"], indent=2))
print("chain pairs (US-align):", len(summary["chain_pair_USalign"]))
print("chain pairs (Foldseek):", len(summary["chain_pair_Foldseek_TMalign"]))
