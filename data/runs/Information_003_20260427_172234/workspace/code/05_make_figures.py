"""
Build figures for the report from saved JSON / NPZ artifacts.
Saves PNGs to report/images/.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "outputs")
IMG = os.path.join(WORK, "report", "images")
os.makedirs(IMG, exist_ok=True)

ATTACK_NAMES = {
    "0": "Analysis", "1": "Backdoor", "2": "Benign", "3": "DoS",
    "4": "Exploits", "5": "Fuzzers", "6": "Generic", "7": "Reconnaissance",
    "8": "Shellcode", "9": "Worms",
}

sns.set_theme(style="whitegrid", context="talk")


def load(p):
    with open(p) as f:
        return json.load(f)


def fig_data_overview():
    stats = load(os.path.join(OUT, "data_stats.json"))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # left: attack distribution overall
    cnt = stats["attack_dist_overall"]
    keys = sorted(cnt.keys(), key=lambda k: -cnt[k])
    names = [ATTACK_NAMES[k] for k in keys]
    vals = [cnt[k] for k in keys]
    colors = ["#4c72b0" if k == "2" else "#dd8452" for k in keys]
    ax[0].bar(names, vals, color=colors)
    ax[0].set_yscale("log")
    ax[0].set_title("Class distribution (NF-UNSW-NB15-v2)")
    ax[0].set_ylabel("# flows (log)")
    ax[0].tick_params(axis="x", rotation=45)
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.1, f"{v}", ha="center", va="bottom", fontsize=10)

    # right: train / val / test split
    ax[1].pie(
        [stats["n_train"], stats["n_val"], stats["n_test"]],
        labels=[f"train\n{stats['n_train']}",
                f"val\n{stats['n_val']}",
                f"test\n{stats['n_test']}"],
        colors=["#4c72b0", "#dd8452", "#55a868"],
        autopct="%.1f%%",
    )
    ax[1].set_title("Temporal 60/20/20 split")
    plt.tight_layout()
    p = os.path.join(IMG, "data_overview.png")
    plt.savefig(p, dpi=130)
    plt.close()
    print(p)


def fig_binary_compare():
    res = load(os.path.join(OUT, "main_results.json"))
    methods = ["mlp", "egraphsage", "ablate_sd", "ablate_rd", "ablate_ms", "didsmfl"]
    labels = {
        "mlp": "MLP", "egraphsage": "E-GraphSAGE",
        "ablate_sd": "DIDS-MFL\nw/o SD",
        "ablate_rd": "DIDS-MFL\nw/o RD",
        "ablate_ms": "DIDS-MFL\nw/o MS",
        "didsmfl": "DIDS-MFL",
    }
    bin_f1 = [res[m]["mean"]["bin_f1"] for m in methods]
    bin_f1_std = [res[m]["std"]["bin_f1"] for m in methods]
    auc = [res[m]["mean"]["bin_auc"] for m in methods]
    auc_std = [res[m]["std"]["bin_auc"] for m in methods]
    multi = [res[m]["mean"]["multi_macro_f1"] for m in methods]
    multi_std = [res[m]["std"]["multi_macro_f1"] for m in methods]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    x = np.arange(len(methods))
    cols = sns.color_palette("Set2", n_colors=len(methods))
    for i, (vals, stds, title, ylim) in enumerate([
        (bin_f1, bin_f1_std, "Binary F1", (0.97, 0.99)),
        (auc, auc_std, "Binary AUC", (0.995, 1.0)),
        (multi, multi_std, "Multi-class macro F1", (0.0, 0.6)),
    ]):
        ax[i].bar(x, vals, yerr=stds, color=cols, capsize=4)
        for j, v in enumerate(vals):
            ax[i].text(j, v + 0.005 * (ylim[1] - ylim[0]) * 5,
                       f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels([labels[m] for m in methods], rotation=20)
        ax[i].set_title(title)
        ax[i].set_ylim(*ylim)
    plt.tight_layout()
    p = os.path.join(IMG, "binary_compare.png")
    plt.savefig(p, dpi=130)
    plt.close()
    print(p)


def fig_per_attack_f1():
    res = load(os.path.join(OUT, "per_attack_results.json"))
    methods = ["mlp", "egraphsage", "didsmfl"]
    labels = {"mlp": "MLP", "egraphsage": "E-GraphSAGE", "didsmfl": "DIDS-MFL"}
    classes = sorted(res["didsmfl"]["mean"].keys(), key=int)
    name_order = [ATTACK_NAMES[c] for c in classes]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(classes))
    w = 0.27
    cols = sns.color_palette("Set1", n_colors=3)
    for i, m in enumerate(methods):
        vals = [res[m]["mean"][c] for c in classes]
        stds = [res[m]["std"][c] for c in classes]
        ax.bar(x + (i - 1) * w, vals, w, yerr=stds, label=labels[m], capsize=2,
               color=cols[i])
    ax.set_xticks(x)
    ax.set_xticklabels(name_order, rotation=30)
    ax.set_ylabel("Per-class F1 (test)")
    ax.set_title("Per-attack F1 — multi-class head (mean ± std over 3 seeds)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p = os.path.join(IMG, "per_attack_f1.png")
    plt.savefig(p, dpi=130)
    plt.close()
    print(p)


def fig_unknown_loao():
    res = load(os.path.join(OUT, "unknown_results.json"))
    methods = ["mlp", "egraphsage", "didsmfl"]
    labels = {"mlp": "MLP", "egraphsage": "E-GraphSAGE", "didsmfl": "DIDS-MFL"}
    aids = sorted(res["didsmfl"].keys(), key=int)
    names = [res["didsmfl"][a]["attack_name"] for a in aids]

    # subplot 1: detection recall on hidden attack
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
    x = np.arange(len(aids))
    w = 0.27
    cols = sns.color_palette("Set1", n_colors=3)
    for i, m in enumerate(methods):
        vals = [res[m][a]["detect_recall"] for a in aids]
        ax[0].bar(x + (i - 1) * w, vals, w, label=labels[m], color=cols[i])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=30)
    ax[0].set_ylabel("Detection recall (TPR)")
    ax[0].set_title("Unknown-attack LOAO — recall on the held-out attack")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend()

    for i, m in enumerate(methods):
        vals = [res[m][a]["f1_subset_attack_vs_benign"] for a in aids]
        ax[1].bar(x + (i - 1) * w, vals, w, label=labels[m], color=cols[i])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=30)
    ax[1].set_ylabel("F1 (attack-vs-benign subset)")
    ax[1].set_title("Unknown-attack LOAO — F1 on attack-vs-benign subset")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend()

    plt.tight_layout()
    p = os.path.join(IMG, "unknown_loao.png")
    plt.savefig(p, dpi=130)
    plt.close()
    print(p)


def fig_fewshot():
    res = load(os.path.join(OUT, "fewshot_results.json"))
    methods = ["mlp", "egraphsage", "didsmfl"]
    labels = {"mlp": "MLP", "egraphsage": "E-GraphSAGE", "didsmfl": "DIDS-MFL"}
    ks = sorted(res["didsmfl"].keys(), key=int)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0))
    cols = sns.color_palette("Set1", n_colors=3)
    for i, m in enumerate(methods):
        vals = [res[m][k]["fewshot_mean_binary_recall"] for k in ks]
        ax[0].plot(ks, vals, "-o", label=labels[m], color=cols[i], linewidth=2.5,
                   markersize=10)
    ax[0].set_xlabel("k (labeled examples per rare class)")
    ax[0].set_ylabel("Mean binary recall on rare classes")
    ax[0].set_title("Few-shot detection of rare attacks\n(Analysis, Backdoor, Fuzzers, Shellcode, Worms)")
    ax[0].set_ylim(0.9, 1.005)
    ax[0].legend()

    for i, m in enumerate(methods):
        vals = [res[m][k]["bin_f1"] for k in ks]
        ax[1].plot(ks, vals, "-o", label=labels[m], color=cols[i], linewidth=2.5,
                   markersize=10)
    ax[1].set_xlabel("k (labeled examples per rare class)")
    ax[1].set_ylabel("Overall binary F1")
    ax[1].set_title("Overall binary F1 vs k")
    ax[1].set_ylim(0.97, 0.99)
    ax[1].legend()
    plt.tight_layout()
    p = os.path.join(IMG, "fewshot_curve.png")
    plt.savefig(p, dpi=130)
    plt.close()
    print(p)


def fig_ablation():
    res = load(os.path.join(OUT, "main_results.json"))
    variants = ["didsmfl", "ablate_sd", "ablate_rd", "ablate_ms"]
    labels = {"didsmfl": "Full DIDS-MFL",
              "ablate_sd": "w/o Statistical Disent.",
              "ablate_rd": "w/o Representational Disent.",
              "ablate_ms": "w/o Multi-Scale fusion"}
    metrics = ["bin_f1", "bin_auc", "multi_macro_f1"]
    titles = {"bin_f1": "Binary F1", "bin_auc": "Binary AUC",
              "multi_macro_f1": "Multi-class macro F1"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    cols = sns.color_palette("Paired", n_colors=4)
    for i, m in enumerate(metrics):
        vals = [res[v]["mean"][m] for v in variants]
        stds = [res[v]["std"][m] for v in variants]
        axes[i].bar(range(len(variants)), vals, yerr=stds, color=cols, capsize=5)
        axes[i].set_xticks(range(len(variants)))
        axes[i].set_xticklabels([labels[v] for v in variants], rotation=20, ha="right")
        axes[i].set_title(titles[m])
        for j, v in enumerate(vals):
            axes[i].text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    p = os.path.join(IMG, "ablation_bar.png")
    plt.savefig(p, dpi=130)
    plt.close()
    print(p)


def fig_tsne():
    """t-SNE before vs after disentanglement on a stratified test sample."""
    from sklearn.manifold import TSNE
    npz = np.load(os.path.join(OUT, "data_clean.npz"))
    msg = npz["msg"]
    attack = npz["attack"]
    idx_te = npz["idx_te"]
    rng = np.random.default_rng(0)
    # stratified 200 per class
    pick = []
    for c in np.unique(attack[idx_te]):
        c_idx = idx_te[attack[idx_te] == c]
        if len(c_idx) > 200:
            c_idx = rng.choice(c_idx, 200, replace=False)
        pick.append(c_idx)
    pick = np.concatenate(pick)
    Xa = msg[pick]
    ya = attack[pick]

    # also load DIDS-MFL learned embedding by re-running on a small batch
    # via loaded model state. To keep it simple, use the raw + a deterministic
    # "disentangled" view: weights estimated by training the SD module alone.
    # To get a real disentangled embedding, we briefly run DIDSMFL forward.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from train_utils import remap_nodes
    from models import DIDSMFL

    src, dst, t = npz["src"], npz["dst"], npz["t"]
    src_n, dst_n, n_nodes = remap_nodes(src, dst)
    model = DIDSMFL(n_nodes, msg.shape[1], emb=64, time_dim=16,
                    n_groups=4, group_dim=16, n_classes=10)
    # quick training: rerun couple of epochs
    import torch.nn.functional as Fnn
    model.mem.reset()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    label = npz["label"]
    idx_tr = npz["idx_tr"]
    cw = np.ones(10, dtype=np.float32)
    counts = np.bincount(attack[idx_tr], minlength=10).astype(np.float32) + 1.0
    cw = (counts.sum() / counts) ** 0.5
    cw = cw / cw.mean()
    cw_t = torch.from_numpy(cw)
    for epoch in range(3):
        model.mem.reset()
        for s in range(0, len(idx_tr), 2048):
            b = idx_tr[s:s+2048]
            ss = torch.from_numpy(src_n[b]).long()
            dd = torch.from_numpy(dst_n[b]).long()
            tt = torch.from_numpy(t[b]).float()
            mm = torch.from_numpy(msg[b]).float()
            yb = torch.from_numpy(label[b]).long()
            ya2 = torch.from_numpy(attack[b]).long()
            blog, mlog, x, eh = model(ss, dd, tt, mm)
            loss = Fnn.cross_entropy(blog, yb) + 0.5 * Fnn.cross_entropy(mlog, ya2, weight=cw_t)
            loss = loss + 0.05 * model.sd.disentangle_loss(mm) + 0.05 * model.rd.ortho_loss(eh)
            opt.zero_grad(); loss.backward(); opt.step()
    # forward only on selected test rows in temporal order to update mem first
    # to keep simple, just compute edge_h ignoring memory effect for picks
    model.eval()
    with torch.no_grad():
        mm = torch.from_numpy(msg[pick]).float()
        x_sd = model.sd(mm).cpu().numpy()
        h_rd = model.rd(model.sd(mm)).cpu().numpy()

    print("running TSNE...")
    tsne1 = TSNE(n_components=2, init="pca", random_state=0, perplexity=30)
    tsne2 = TSNE(n_components=2, init="pca", random_state=0, perplexity=30)
    tsne3 = TSNE(n_components=2, init="pca", random_state=0, perplexity=30)
    Z0 = tsne1.fit_transform(Xa)
    Z1 = tsne2.fit_transform(x_sd)
    Z2 = tsne3.fit_transform(h_rd)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    classes = np.unique(ya)
    pal = sns.color_palette("tab10", n_colors=len(classes))
    for ax, Z, title in zip(axes, [Z0, Z1, Z2],
                            ["Raw NetFlow features",
                             "After Statistical Disentangle (SD)",
                             "After SD + Representational Disentangle (RD)"]):
        for i, c in enumerate(classes):
            sel = ya == c
            ax.scatter(Z[sel, 0], Z[sel, 1], s=10, color=pal[i],
                       label=ATTACK_NAMES[str(c)], alpha=0.7)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout()
    p = os.path.join(IMG, "tsne_disentangle.png")
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(p)


def fig_confusion():
    npz = np.load(os.path.join(OUT, "didsmfl_seed0_test.npz"))
    y, p = npz["y_a"], npz["preds_m"]
    from sklearn.metrics import confusion_matrix
    classes = sorted(set(y.tolist()) | set(p.tolist()))
    cm = confusion_matrix(y, p, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=[ATTACK_NAMES[str(c)] for c in classes],
                yticklabels=[ATTACK_NAMES[str(c)] for c in classes],
                cbar_kws={"label": "Row-normalized rate"}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("DIDS-MFL test confusion matrix\n(annotations = raw counts; color = row-normalized)")
    plt.tight_layout()
    p = os.path.join(IMG, "confusion_didsmfl.png")
    plt.savefig(p, dpi=130)
    plt.close()
    print(p)


if __name__ == "__main__":
    fig_data_overview()
    fig_binary_compare()
    fig_per_attack_f1()
    fig_unknown_loao()
    fig_fewshot()
    fig_ablation()
    fig_confusion()
    fig_tsne()
