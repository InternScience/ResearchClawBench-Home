"""Build the chain-vs-chain TM-score matrix between 7xg4 (9 protein chains)
and 6n40 (1 protein chain, but for completeness we also include the chain
self-self entry).  Each pair is run through US-align in monomer mode.
The matrix is saved to outputs/usalign/tm_matrix.tsv and used to make
a heatmap figure.
"""
import os, subprocess, json
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
USALIGN = "/tmp/USalign"

q_chains = list("ABCDEFGHL")
t_chains = ["A"]

results = []
for q in q_chains:
    for t in t_chains:
        qpdb = WORK / "outputs" / "chains" / f"7xg4_{q}.pdb"
        tpdb = WORK / "outputs" / "chains" / f"6n40_{t}.pdb"
        out = subprocess.run(
            [USALIGN, str(qpdb), str(tpdb), "-mol", "prot", "-outfmt", "2"],
            capture_output=True, text=True,
        )
        # last non-empty line is data
        line = [l for l in out.stdout.strip().split("\n") if l and not l.startswith("#")][-1]
        f = line.split("\t")
        results.append({
            "query_chain": q,
            "target_chain": t,
            "TM_norm_q": float(f[2]),
            "TM_norm_t": float(f[3]),
            "RMSD": float(f[4]),
            "seqID_norm_q": float(f[5]),
            "seqID_norm_t": float(f[6]),
            "seqID_aligned": float(f[7]),
            "L_q": int(f[8]),
            "L_t": int(f[9]),
            "L_aligned": int(f[10]),
        })

with open(WORK / "outputs" / "usalign" / "tm_matrix.json", "w") as fh:
    json.dump(results, fh, indent=2)

# tsv
with open(WORK / "outputs" / "usalign" / "tm_matrix.tsv", "w") as fh:
    keys = list(results[0].keys())
    fh.write("\t".join(keys) + "\n")
    for r in results:
        fh.write("\t".join(str(r[k]) for k in keys) + "\n")

for r in results:
    print(f"7xg4_{r['query_chain']} vs 6n40_{r['target_chain']}: "
          f"TM_q={r['TM_norm_q']:.3f} TM_t={r['TM_norm_t']:.3f} "
          f"RMSD={r['RMSD']:.2f} Lali={r['L_aligned']}")
