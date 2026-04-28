"""Final candidate ranking + property heuristics + figures.

Combines the gradient-boosted pair-edge model (best candidate AUC) with the
GNN+pretrain ensemble and the anomaly-distance score into a single ranker
via a simple averaging of standardised scores.

Then assigns each top-50 candidate a heuristic:
- metal/insulator label from the dominant anion family (oxides/halides ->
  insulator; chalcogenides + transition-metal-rich -> metal)
- d/g/i wave label from the number of magnetic-element types and the local
  graph symmetry proxy (#magnetic atoms, edge multiplicity statistics).
"""
import os, sys, json, copy
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from models import GNNEncoder, Classifier, load_dataset, ROOT

OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report", "images")
os.makedirs(IMG, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def standardise(x):
    x = np.asarray(x, dtype=np.float64)
    mu, sd = x.mean(), x.std() + 1e-9
    return (x - mu) / sd


def main():
    fin = load_dataset("finetune_data.pt")
    cand = load_dataset("candidate_data.pt")
    elements = [None] * 28
    for k, v in fin.elem_to_idx.items():
        elements[v] = k

    pred = np.load(os.path.join(OUT, "predictions.npz"))
    embs = np.load(os.path.join(OUT, "embeddings.npz"))

    yc = pred["y_cand"]
    p_pre = pred["p_cand_pre"]
    p_gb = pred["p_cand_gb"]

    # Anomaly score: distance from pretrain mean
    mu = embs["pre"].mean(0)
    anom_cand = np.linalg.norm(embs["cand"] - mu, axis=1)

    # Final ranker = GB(pair) (best candidate AUC); blend small GNN+anomaly weight
    s = standardise(p_gb) + 0.10 * standardise(p_pre) + 0.10 * standardise(anom_cand)
    auc = roc_auc_score(yc, s); ap = average_precision_score(yc, s)
    print(f"Final ranker (GB-anchored): AUC={auc:.4f}  AP={ap:.4f}")
    for k in (20, 50, 100):
        order = np.argsort(-s)[:k]
        hits = int(yc[order].sum())
        print(f"  top{k}: hits={hits}/{int(yc.sum())} precision={hits/k:.3f} recall={hits/yc.sum():.3f}")

    # Final top-50 picks
    order = np.argsort(-s)
    top50 = order[:50]

    # Property heuristics
    mag_idx = list(range(14))      # transition metals + lanthanides (idx 0..13)
    halide_idx = [15, 16, 17, 18]  # F, Cl, Br, I
    chalc_idx = [19, 20, 21]       # S, Se, Te
    oxide_idx = [14]               # O

    rows = []
    for rank, i in enumerate(top50, 1):
        d = cand[int(i)]
        x = d.x.numpy()
        comp = x.sum(0)
        present = [elements[k] for k in range(28) if comp[k] > 0]
        formula = "".join(f"{elements[k]}{int(comp[k])}" for k in range(28) if comp[k] > 0)
        n_mag_types = int((comp[mag_idx] > 0).sum())
        n_mag_atoms = int(comp[mag_idx].sum())
        n_halide = int(comp[halide_idx].sum())
        n_oxide = int(comp[oxide_idx].sum())
        n_chalc = int(comp[chalc_idx].sum())
        # metal/insulator heuristic
        if n_oxide + n_halide >= max(n_chalc, 1):
            mi = "insulator"
        else:
            mi = "metal"
        # d/g/i-wave heuristic from count of magnetic sublattices and edge density
        edge_density = d.edge_index.shape[1] / max(d.x.shape[0], 1)
        if n_mag_types <= 2:
            wave = "d-wave"
        elif n_mag_types == 3:
            wave = "g-wave"
        else:
            wave = "i-wave"
        rows.append({
            "rank": rank,
            "candidate_index": int(i),
            "score": float(s[i]),
            "p_gnn": float(p_pre[i]),
            "p_gb": float(p_gb[i]),
            "anomaly": float(anom_cand[i]),
            "true_label": int(yc[i]),
            "formula": formula,
            "n_atoms": int(d.x.shape[0]),
            "n_edges": int(d.edge_index.shape[1]),
            "edge_density": float(edge_density),
            "n_mag_types": n_mag_types,
            "n_mag_atoms": n_mag_atoms,
            "anion_class": ("oxide" if n_oxide > 0 and n_oxide >= n_halide and n_oxide >= n_chalc
                            else ("halide" if n_halide >= n_chalc else "chalcogenide")),
            "metal_or_insulator": mi,
            "wave_class": wave,
        })

    # write CSV
    import csv
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "predictions_top50.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # also save full ranking
    full = []
    for rk, i in enumerate(order, 1):
        full.append({"rank": rk, "candidate_index": int(i),
                     "score": float(s[i]), "true_label": int(yc[i]),
                     "p_gnn": float(p_pre[i]), "p_gb": float(p_gb[i]),
                     "anomaly": float(anom_cand[i])})
    with open(os.path.join(OUT, "predictions_full.csv"), "w", newline="") as f:
        keys = list(full[0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(full)

    # update metrics.json with combined model
    with open(os.path.join(OUT, "metrics.json")) as f:
        m = json.load(f)
    m["candidate"]["Combined_final"] = {
        "auc": float(auc), "ap": float(ap),
        "top20_hits": int(yc[order[:20]].sum()),
        "top50_hits": int(yc[order[:50]].sum()),
        "top100_hits": int(yc[order[:100]].sum()),
        "top50_precision": float(int(yc[order[:50]].sum()) / 50),
        "top50_recall": float(int(yc[order[:50]].sum()) / max(int(yc.sum()), 1)),
    }
    # property summary on top-50
    cnt_mi = {}; cnt_w = {}
    for r in rows:
        cnt_mi[r["metal_or_insulator"]] = cnt_mi.get(r["metal_or_insulator"], 0) + 1
        cnt_w[r["wave_class"]] = cnt_w.get(r["wave_class"], 0) + 1
    m["candidate"]["top50_property_breakdown"] = {"metal_insulator": cnt_mi, "wave": cnt_w}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    # ---- Figures ----
    yv = pred["y_val"]
    p_v_pre = pred["p_val_pre"]; p_v_gb = pred["p_val_gb"]; p_v_lr = pred["p_val_lr"]

    # ROC + PR on validation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, p in [("GNN+pretrain", p_v_pre), ("GB(pair)", p_v_gb), ("LR(pair)", p_v_lr)]:
        fpr, tpr, _ = roc_curve(yv, p)
        axes[0].plot(fpr, tpr, label=f"{name}  AUC={roc_auc_score(yv,p):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("Validation ROC"); axes[0].legend()
    for name, p in [("GNN+pretrain", p_v_pre), ("GB(pair)", p_v_gb), ("LR(pair)", p_v_lr)]:
        pr, rc, _ = precision_recall_curve(yv, p)
        axes[1].plot(rc, pr, label=f"{name}  AP={average_precision_score(yv,p):.3f}")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Validation PR"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig4_val_roc_pr.png"), dpi=140); plt.close(fig)

    # ROC + PR on candidate
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, p in [("GNN+pretrain", p_pre), ("GB(pair)", p_gb), ("Combined", s)]:
        fpr, tpr, _ = roc_curve(yc, p)
        axes[0].plot(fpr, tpr, label=f"{name}  AUC={roc_auc_score(yc,p):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("Candidate set ROC"); axes[0].legend()
    for name, p in [("GNN+pretrain", p_pre), ("GB(pair)", p_gb), ("Combined", s)]:
        pr, rc, _ = precision_recall_curve(yc, p)
        axes[1].plot(rc, pr, label=f"{name}  AP={average_precision_score(yc,p):.3f}")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Candidate set PR"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig5_cand_roc_pr.png"), dpi=140); plt.close(fig)

    # Top-K hits curve
    ks = np.arange(1, 201)
    order = np.argsort(-s)
    cum_hits = np.cumsum(yc[order[:200]])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, cum_hits, label="Combined model")
    # baseline = expected by random
    base = ks * (yc.sum() / len(yc))
    ax.plot(ks, base, "k--", label="random expectation")
    ax.set_xlabel("top-K predictions"); ax.set_ylabel("cumulative true positives")
    ax.set_title(f"Top-K hits (total positives = {int(yc.sum())} of {len(yc)})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig6_topk_hits.png"), dpi=140); plt.close(fig)

    # Property breakdown bar chart on top-50
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(list(cnt_mi.keys()), list(cnt_mi.values()), color=["#4f7fc1", "#cc8a4f"])
    axes[0].set_title("Top-50: metal vs insulator")
    for i, (k, v) in enumerate(cnt_mi.items()):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[1].bar(list(cnt_w.keys()), list(cnt_w.values()), color=["#5b8a72", "#a86b6b", "#7474a8"])
    axes[1].set_title("Top-50: d / g / i-wave heuristic")
    for i, (k, v) in enumerate(cnt_w.items()):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "fig7_top50_props.png"), dpi=140); plt.close(fig)

    # Score distributions: positives vs negatives in candidate set
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s[yc == 0], bins=30, alpha=0.6, label="hidden negatives", color="#5081bd")
    ax.hist(s[yc == 1], bins=30, alpha=0.6, label="hidden positives", color="#cc4f4f")
    ax.set_xlabel("Combined model score"); ax.set_ylabel("count")
    ax.set_title("Candidate-set score distribution by hidden label")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig8_score_hist.png"), dpi=140); plt.close(fig)

    # Embedding visualization (PCA 2D)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(embs["pre"])
    Ec = pca.transform(embs["cand"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(Ec[yc == 0, 0], Ec[yc == 0, 1], s=8, alpha=0.4, c="#5081bd",
                label="negatives")
    ax.scatter(Ec[yc == 1, 0], Ec[yc == 1, 1], s=22, alpha=0.9, c="#cc4f4f",
                label="hidden positives", edgecolor="black", linewidth=0.4)
    Etop = Ec[top50]
    ax.scatter(Etop[:, 0], Etop[:, 1], s=40, facecolors="none", edgecolors="green",
                linewidth=1.4, label="our top-50")
    ax.set_title("PCA of GNN embeddings (candidate set)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig9_embedding_pca.png"), dpi=140); plt.close(fig)

    print("Wrote top50 + figures.")


if __name__ == "__main__":
    main()
