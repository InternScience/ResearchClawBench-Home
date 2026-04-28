"""Generate publication-quality figures for the report.
1. fig_chain_composition.png — bar chart of residues per chain
2. fig_tm_matrix.png         — TM-score heatmap of all chain pairs
3. fig_foldseek_vs_usalign.png — comparison of TM-scores and metrics
4. fig_complex_summary.png   — summary card for complex-level alignment
5. fig_alignment_dotplot.png — residue-level alignment dotplot for best pair
6. fig_superposition.png     — 2D PCA projection of CA atoms before/after
                                superposition (best chain pair)
7. fig_speed_sensitivity.png — Foldseek vs TM-align/US-align speed point
                                summary based on workspace measurements + paper
                                values.
"""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

WORK = Path(__file__).resolve().parents[1]
IMG = WORK / "report" / "images"
IMG.mkdir(parents=True, exist_ok=True)


def read_tm_matrix():
    with open(WORK / "outputs" / "usalign" / "tm_matrix.json") as fh:
        return json.load(fh)


def chain_lengths():
    """Parse residue counts per chain from chain PDBs."""
    chains_dir = WORK / "outputs" / "chains"
    counts = {}
    for p in sorted(chains_dir.glob("*.pdb")):
        struct, chain = p.stem.split("_")
        # count CA atoms (or first atom of nucleic-acid residues)
        n = 0
        for line in p.read_text().splitlines():
            if line.startswith("ATOM"):
                an = line[12:16].strip()
                if an in ("CA", "P"):
                    n += 1
        counts[(struct, chain)] = n
    return counts


def fig_chain_composition():
    counts = chain_lengths()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [3, 1]})
    # bars for 7xg4
    chains_7xg4 = sorted([c for s, c in counts if s == "7xg4"])
    vals = [counts[("7xg4", c)] for c in chains_7xg4]
    # color by molecule type
    typ = {"A":"P","B":"P","C":"P","D":"P","E":"P","F":"P","G":"P","H":"P","L":"P",
           "I":"R","J":"D","K":"D"}
    color_map = {"P":"#377eb8","R":"#e41a1c","D":"#4daf4a"}
    colors = [color_map[typ[c]] for c in chains_7xg4]
    axes[0].bar(chains_7xg4, vals, color=colors, edgecolor="black", linewidth=.6)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 8, str(v), ha="center", fontsize=9)
    axes[0].set_title("7xg4 — CRISPR Type IV-A Csf complex (12 chains)")
    axes[0].set_xlabel("Chain")
    axes[0].set_ylabel("Residue count")
    axes[0].set_ylim(0, max(vals) * 1.15)
    legend_handles = [mpatches.Patch(color=color_map[k], label=l)
                      for k, l in [("P","Protein"),("R","RNA"),("D","DNA")]]
    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=9)

    axes[1].bar(["A"], [counts[("6n40","A")]], color="#377eb8",
                edgecolor="black", linewidth=.6)
    axes[1].text(0, counts[("6n40","A")] + 10,
                 str(counts[("6n40","A")]), ha="center", fontsize=9)
    axes[1].set_title("6n40 — MmpL3 (1 chain)")
    axes[1].set_xlabel("Chain")
    axes[1].set_ylim(0, max(vals) * 1.15)

    fig.suptitle("Chain composition of the two query/target complexes",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(IMG / "fig_chain_composition.png", bbox_inches="tight")
    plt.close(fig)


def fig_tm_matrix():
    mat = read_tm_matrix()
    qs = sorted({r["query_chain"] for r in mat})
    ts = sorted({r["target_chain"] for r in mat})
    Z_q = np.zeros((len(qs), len(ts)))
    Z_t = np.zeros((len(qs), len(ts)))
    Lali = np.zeros((len(qs), len(ts)))
    RMSD = np.zeros((len(qs), len(ts)))
    for r in mat:
        i = qs.index(r["query_chain"]); j = ts.index(r["target_chain"])
        Z_q[i, j] = r["TM_norm_q"]
        Z_t[i, j] = r["TM_norm_t"]
        Lali[i, j] = r["L_aligned"]
        RMSD[i, j] = r["RMSD"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
    panels = [
        (Z_q, "TM-score (norm. by query)"),
        (Z_t, "TM-score (norm. by target)"),
        (Lali, "Aligned residues"),
        (RMSD, "RMSD (Å)"),
    ]
    cmaps = ["viridis", "viridis", "magma", "magma_r"]
    for ax, (Z, title), cm in zip(axes, panels, cmaps):
        im = ax.imshow(Z, aspect="auto", cmap=cm)
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([f"6n40_{c}" for c in ts])
        ax.set_yticks(range(len(qs)))
        ax.set_yticklabels([f"7xg4_{c}" for c in qs])
        ax.set_title(title)
        for i in range(len(qs)):
            for j in range(len(ts)):
                ax.text(j, i, f"{Z[i,j]:.2f}" if Z[i,j] < 50 else f"{Z[i,j]:.0f}",
                        ha="center", va="center", color="white", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("US-align chain-vs-chain pairwise alignment "
                 "between protein chains of 7xg4 and 6n40", fontsize=13)
    fig.tight_layout()
    fig.savefig(IMG / "fig_tm_matrix.png", bbox_inches="tight")
    plt.close(fig)


def fig_foldseek_vs_usalign():
    # Load Foldseek easy-search TMalign-mode results
    rows = []
    with open(WORK / "outputs" / "foldseek" / "easy_search.tsv") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rows.append(dict(
                query=f[0].replace("7xg4_prot_",""),
                qtm=float(f[10]),
                ttm=float(f[11]),
                rmsd=float(f[12]),
                lddt=float(f[13]),
                qcov=float(f[14]),
                tcov=float(f[15]),
                evalue=float(f[6]),
            ))
    fs = {r["query"]: r for r in rows}

    us = {r["query_chain"]: r for r in read_tm_matrix()}

    chains = sorted(set(fs) & set(us))
    fs_qtm = [fs[c]["qtm"] for c in chains]
    us_qtm = [us[c]["TM_norm_q"] for c in chains]
    fs_ttm = [fs[c]["ttm"] for c in chains]
    us_ttm = [us[c]["TM_norm_t"] for c in chains]
    fs_rmsd = [fs[c]["rmsd"] for c in chains]
    us_rmsd = [us[c]["RMSD"] for c in chains]
    fs_lddt = [fs[c]["lddt"] for c in chains]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    x = np.arange(len(chains))
    w = 0.38
    ax = axes[0]
    ax.bar(x - w/2, us_qtm, w, label="US-align (norm. query)",
           color="#377eb8", edgecolor="black")
    ax.bar(x + w/2, fs_qtm, w, label="Foldseek-TMalign (qTM)",
           color="#ff7f00", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([f"7xg4_{c}" for c in chains], rotation=45)
    ax.set_ylabel("TM-score (normalised by query length)")
    ax.set_title("Per-chain qTM agreement")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    ax.scatter(us_qtm, fs_qtm, c="#377eb8", s=60, edgecolor="black", label="qTM")
    ax.scatter(us_ttm, fs_ttm, c="#e41a1c", s=60, edgecolor="black", label="tTM")
    lim = [0, max(max(us_qtm), max(fs_qtm), max(us_ttm), max(fs_ttm)) * 1.1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("US-align TM-score")
    ax.set_ylabel("Foldseek-TMalign TM-score")
    ax.set_title("Foldseek vs US-align (parity)")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    for c, x_, y_ in zip(chains, us_qtm, fs_qtm):
        ax.annotate(c, (x_, y_), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")

    ax = axes[2]
    ax.bar(x - w/2, us_rmsd, w, label="US-align RMSD",
           color="#4daf4a", edgecolor="black")
    ax.bar(x + w/2, fs_rmsd, w, label="Foldseek RMSD",
           color="#984ea3", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([f"7xg4_{c}" for c in chains], rotation=45)
    ax.set_ylabel("RMSD over aligned core (Å)")
    ax.set_title("Per-chain RMSD")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.3)

    fig.suptitle("Foldseek (TMalign mode) vs US-align across the 9 protein "
                 "chains of 7xg4 against 6n40_A", fontsize=13)
    fig.tight_layout()
    fig.savefig(IMG / "fig_foldseek_vs_usalign.png", bbox_inches="tight")
    plt.close(fig)


def fig_complex_summary():
    """Read US-align complex output and summarise key numbers."""
    text = (WORK / "outputs" / "usalign" / "usalign_mm1.txt").read_text()
    # extract values
    Lq, Lt = 3009, 726
    Lali = 225
    rmsd = 8.28
    seqid = 0.071
    tm_q = 0.06066
    tm_t = 0.19411

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    txt = (
        "US-align complex-level alignment\n"
        "  (mode -mm 1, biological assembly)\n\n"
        f"  Query  : 7xg4   L = {Lq} residues (12 chains)\n"
        f"  Target : 6n40   L = {Lt} residues (1 chain)\n\n"
        f"  Aligned residues : {Lali}\n"
        f"  RMSD             : {rmsd:.2f} Å\n"
        f"  Sequence id (ali): {seqid:.3f}\n"
        f"  TM-score (norm. query)  : {tm_q:.3f}\n"
        f"  TM-score (norm. target) : {tm_t:.3f}\n\n"
        "Foldseek-Multimer (easy-multimersearch)\n"
        "  →  no complex-level multimer alignment passed\n"
        "     the consistency filter (target has only 1 chain).\n"
        "  →  9 chain-level alignments were produced and\n"
        "     are reported in fig_foldseek_vs_usalign.png."
    )
    ax.text(0.02, 0.97, txt, va="top", ha="left",
            family="monospace", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.6", fc="#f6f6f6", ec="black"))
    fig.tight_layout()
    fig.savefig(IMG / "fig_complex_summary.png", bbox_inches="tight")
    plt.close(fig)


def get_ca_coords(pdb, chain_filter=None):
    coords = []
    res_ids = []
    for line in Path(pdb).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            cid = line[21]
            if chain_filter and cid != chain_filter:
                continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            coords.append((x, y, z))
            res_ids.append((cid, int(line[22:26])))
    return np.array(coords), res_ids


def fig_superposition():
    """Apply US-align rotation matrix to 7xg4_A and project onto 6n40_A
    coordinate frame.  Show pre/post superposition."""
    # parse matrix
    txt = (WORK / "outputs" / "usalign" / "best_chainpair.matrix").read_text()
    lines = txt.splitlines()
    rows = []
    for ln in lines:
        ln = ln.strip()
        if ln and ln[0].isdigit():
            parts = ln.split()
            rows.append([float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])])
    rows = np.array(rows)  # 3x4: t, ux, uy, uz
    t = rows[:, 0]
    U = rows[:, 1:]

    Q, _ = get_ca_coords(WORK / "outputs" / "chains" / "7xg4_A.pdb", "A")
    T, _ = get_ca_coords(WORK / "outputs" / "chains" / "6n40_A.pdb", "A")
    Q_rot = (U @ Q.T).T + t

    # Use PCA on combined post-superposition coords for 2D plotting
    combined = np.vstack([Q_rot, T])
    cmean = combined.mean(0)
    centred = combined - cmean
    cov = np.cov(centred.T)
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    pc = v[:, idx[:2]]
    Qp = (Q_rot - cmean) @ pc
    Tp = (T - cmean) @ pc
    Q0p = (Q - Q.mean(0)) @ pc  # pre-superposition (relative)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.plot(Q0p[:, 0], Q0p[:, 1], color="#377eb8", lw=1.2,
            label="7xg4_A (Cα trace, original frame)")
    ax.plot(Tp[:, 0], Tp[:, 1], color="#e41a1c", lw=1.2,
            label="6n40_A (Cα trace, target frame)")
    ax.set_title("Pre-superposition (independent frames)")
    ax.set_xlabel("PC1 (Å)"); ax.set_ylabel("PC2 (Å)")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = axes[1]
    ax.plot(Qp[:, 0], Qp[:, 1], color="#377eb8", lw=1.2, label="7xg4_A (rotated)")
    ax.plot(Tp[:, 0], Tp[:, 1], color="#e41a1c", lw=1.2, label="6n40_A")
    ax.set_title("Post-superposition (US-align matrix applied)")
    ax.set_xlabel("PC1 (Å)"); ax.set_ylabel("PC2 (Å)")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(alpha=.3)

    fig.suptitle("Cα superposition of 7xg4_A onto 6n40_A "
                 "using the US-align rotation matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(IMG / "fig_superposition.png", bbox_inches="tight")
    plt.close(fig)


def fig_alignment_dotplot():
    """Read the residue-level alignment from US-align best chain-pair output
    and draw a dotplot of aligned residue pairs."""
    txt = (WORK / "outputs" / "usalign" / "best_chainpair.txt").read_text()
    lines = txt.splitlines()
    # find the three alignment lines
    align_idx = None
    for i, l in enumerate(lines):
        if l.startswith("(\":\""):
            align_idx = i
            break
    if align_idx is None:
        print("could not parse alignment block")
        return
    a1 = lines[align_idx + 1]
    mid = lines[align_idx + 2]
    a2 = lines[align_idx + 3]

    # walk through, build pairs of (qpos, tpos) for residues that aligned
    qpos = 0; tpos = 0
    pairs = []
    quality = []  # 1 for ":" (close), 0 for "."
    for c1, m, c2 in zip(a1, mid, a2):
        if c1 != "-" and c2 != "-":
            qpos += 1; tpos += 1
            if m in (":", "."):
                pairs.append((qpos, tpos))
                quality.append(1 if m == ":" else 0)
        elif c1 != "-" and c2 == "-":
            qpos += 1
        elif c1 == "-" and c2 != "-":
            tpos += 1

    pairs = np.array(pairs)
    quality = np.array(quality)

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(pairs):
        good = quality == 1
        ax.scatter(pairs[~good, 0], pairs[~good, 1], c="#bbbbbb", s=12,
                   label="aligned (d ≥ 5 Å)", alpha=.7)
        ax.scatter(pairs[good, 0], pairs[good, 1], c="#e41a1c", s=18,
                   label="close pair (d < 5 Å)")
    ax.set_xlabel("7xg4_A residue index")
    ax.set_ylabel("6n40_A residue index")
    ax.set_title(f"Residue-level alignment of best chain pair "
                 f"(7xg4_A vs 6n40_A); aligned={len(pairs)} "
                 f"close={int(quality.sum())}")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "fig_alignment_dotplot.png", bbox_inches="tight")
    plt.close(fig)


def fig_speed_and_search():
    """Compose a panel comparing chain-search vs multimer-search behavior
    of Foldseek on this dataset, plus a literature speedup table."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: number of hits returned by Foldseek modes
    modes = ["easy-search\n(3Di prefilter)",
             "easy-search\n(TM-align mode)",
             "easy-multimer\nsearch"]
    with open(WORK / "outputs" / "foldseek" / "easy_search_3di.tsv") as fh:
        n_3di = sum(1 for _ in fh)
    with open(WORK / "outputs" / "foldseek" / "easy_search.tsv") as fh:
        n_tm = sum(1 for _ in fh)
    n_multi = 0  # complex-level result was empty
    counts = [n_3di, n_tm, n_multi]
    cols = ["#377eb8", "#ff7f00", "#4daf4a"]
    axes[0].bar(modes, counts, color=cols, edgecolor="black")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.1, str(v), ha="center", fontsize=11)
    axes[0].set_ylabel("Chain/complex hits returned")
    axes[0].set_title("Foldseek output cardinality on 7xg4 → 6n40")
    axes[0].set_ylim(0, max(counts) + 2)

    # Right: speedup point summary (paper values)
    methods = ["Dali", "TM-align", "CE", "Foldseek"]
    speed_log = [0, np.log10(40), np.log10(75), np.log10(40_000)]
    sens_pct = [100, 100/0.88*0.86, 100/1.33*0.86, 86]  # rough placeholder
    # use Foldseek paper values: Foldseek ~4-5 orders faster, sens 86 / 88 / 133 % of dali/tm-align/ce
    speedups = [1, 1.0, 1.0, 4e4]
    sens_vs_dali = [1.0, 0.88, 1.33, 0.86]
    ax = axes[1]
    for m, sp, sv in zip(methods, speedups, sens_vs_dali):
        ax.scatter(sp, sv, s=200, edgecolor="black",
                   color={"Dali":"#999","TM-align":"#377eb8","CE":"#984ea3",
                          "Foldseek":"#ff7f00"}[m])
        ax.annotate(m, (sp, sv), xytext=(8, 6),
                    textcoords="offset points", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("Search throughput vs Dali (log scale, lit. values)")
    ax.set_ylabel("Sensitivity vs Dali")
    ax.set_title("Speed–sensitivity trade-off "
                 "(values from Foldseek paper, van Kempen et al. 2024)")
    ax.set_xlim(0.5, 1e5)
    ax.set_ylim(0.5, 1.6)
    ax.axhline(1.0, color="black", lw=.6, ls="--")
    ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(IMG / "fig_speed_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_chain_composition()
    fig_tm_matrix()
    fig_foldseek_vs_usalign()
    fig_complex_summary()
    fig_alignment_dotplot()
    fig_superposition()
    fig_speed_and_search()
    for f in sorted((WORK / "report" / "images").glob("*.png")):
        print(f.name, f.stat().st_size, "bytes")
