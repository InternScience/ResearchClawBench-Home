"""EDA: dataset summaries, label distributions, size distributions, element prevalence."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report", "images")
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)


def load(name):
    return torch.load(os.path.join(ROOT, "data", name), weights_only=False)


pre = load("pretrain_data.pt")
fin = load("finetune_data.pt")
cand = load("candidate_data.pt")
elem_to_idx = fin.elem_to_idx
idx_to_elem = {v: k for k, v in elem_to_idx.items()}
elements = [idx_to_elem[i] for i in range(len(idx_to_elem))]


def stats(ds, name):
    ys = np.array([int(d.y.item()) for d in ds])
    nodes = np.array([d.x.shape[0] for d in ds])
    edges = np.array([d.edge_index.shape[1] for d in ds])
    comp = np.zeros((len(ds), len(elements)))
    for i, d in enumerate(ds):
        comp[i] = d.x.sum(0).numpy()
    return {
        "name": name,
        "size": len(ds),
        "pos": int(ys.sum()),
        "neg": int((ys == 0).sum()),
        "node_mean": float(nodes.mean()),
        "node_min": int(nodes.min()),
        "node_max": int(nodes.max()),
        "edge_mean": float(edges.mean()),
        "edge_min": int(edges.min()),
        "edge_max": int(edges.max()),
        "y": ys,
        "nodes": nodes,
        "edges": edges,
        "comp": comp,
    }


sp = stats(pre, "pretrain")
sf = stats(fin, "finetune")
sc = stats(cand, "candidate")

summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("y", "nodes", "edges", "comp")}
           for k, v in [("pretrain", sp), ("finetune", sf), ("candidate", sc)]}
summary["elements"] = elements
with open(os.path.join(OUT, "data_overview.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# ---- Figure 1: dataset sizes / class balance ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, s in zip(axes, [sp, sf, sc]):
    ax.bar(["positive", "negative"], [s["pos"], s["neg"]], color=["#cc4f4f", "#5081bd"])
    ax.set_title(f"{s['name']}  (n={s['size']})")
    ax.set_ylabel("count")
    for i, v in enumerate([s["pos"], s["neg"]]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
fig.suptitle("Dataset sizes and class balance")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig1_dataset_overview.png"), dpi=140)
plt.close(fig)

# ---- Figure 2: graph size distributions ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for s, c in zip([sp, sf, sc], ["#888888", "#cc4f4f", "#2e8b57"]):
    axes[0].hist(s["nodes"], bins=range(0, 30), alpha=0.6, label=s["name"], color=c)
    axes[1].hist(s["edges"], bins=30, alpha=0.6, label=s["name"], color=c)
axes[0].set_xlabel("# atoms / graph"); axes[0].set_ylabel("frequency"); axes[0].legend()
axes[1].set_xlabel("# edges / graph"); axes[1].set_ylabel("frequency"); axes[1].legend()
fig.suptitle("Crystal-graph size distributions")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig2_graph_sizes.png"), dpi=140)
plt.close(fig)

# ---- Figure 3: element prevalence in finetune positives vs negatives ----
ys = sf["y"]; comp = sf["comp"]
mean_pos = comp[ys == 1].mean(0)
mean_neg = comp[ys == 0].mean(0)
order = np.argsort(-(mean_pos - mean_neg))
xs = np.arange(len(elements))
fig, ax = plt.subplots(figsize=(13, 4.5))
w = 0.4
ax.bar(xs - w / 2, mean_pos[order], w, label="altermagnet", color="#cc4f4f")
ax.bar(xs + w / 2, mean_neg[order], w, label="non-altermagnet", color="#5081bd")
ax.set_xticks(xs)
ax.set_xticklabels([elements[i] for i in order], rotation=0)
ax.set_ylabel("mean atom count / graph")
ax.set_title("Element prevalence: altermagnet vs non-altermagnet (fine-tune set)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig3_element_prevalence.png"), dpi=140)
plt.close(fig)

print("EDA done.")
