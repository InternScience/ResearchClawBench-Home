import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pypdf import PdfReader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
REPORT_DIR = BASE / "report"
IMG_DIR = REPORT_DIR / "images"
CODE_DIR = BASE / "code"

sns.set_theme(style="whitegrid")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Compatible placeholder for serialized datasets
mod = types.ModuleType("data_prepare")
class RealisticCrystalDataset:
    pass
mod.RealisticCrystalDataset = RealisticCrystalDataset
sys.modules["data_prepare"] = mod


def load_dataset(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj


def data_to_feature_vector(d: Data):
    x = d.x.cpu().numpy().astype(np.float32)
    ea = d.edge_attr.cpu().numpy().astype(np.float32)
    n = x.shape[0]
    m = d.edge_index.shape[1]
    feats = [
        x.sum(0),
        x.mean(0),
        np.array([
            n,
            m,
            m / max(n, 1),
            m / max(n * (n - 1), 1),
        ], dtype=np.float32),
        ea.mean(0),
        ea.std(0),
        ea.min(0),
        ea.max(0),
    ]
    return np.concatenate(feats, dtype=np.float32)


def augment_graph(data: Data, drop_edge_p=0.15, mask_node_p=0.15):
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()

    if edge_index.shape[1] > 1:
        keep = torch.rand(edge_index.shape[1]) > drop_edge_p
        if keep.sum() == 0:
            keep[torch.randint(0, edge_index.shape[1], (1,))] = True
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]

    if x.shape[0] > 0:
        mask = torch.rand(x.shape[0]) < mask_node_p
        x[mask] = 0.0

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y)


class GINEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, proj_dim=64):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, data):
        x = self.node_emb(data.x)
        e = self.edge_encoder(data.edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, e)
            x = bn(x)
            x = F.relu(x)
        g_mean = global_mean_pool(x, data.batch)
        g_add = global_add_pool(x, data.batch)
        g = torch.cat([g_mean, g_add], dim=1)
        z = self.projector(g)
        return g, z


class GraphClassifier(nn.Module):
    def __init__(self, encoder: GINEncoder, graph_dim=128, hidden=64):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, data):
        g, _ = self.encoder(data)
        return self.head(g).squeeze(-1)


def nt_xent(z1, z2, temp=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temp
    n = z1.size(0)
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)])
    return F.cross_entropy(sim, targets)


def pretrain_encoder(pretrain_graphs, node_dim, edge_dim, device, epochs=30, batch_size=128):
    encoder = GINEncoder(node_dim, edge_dim).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    for epoch in range(epochs):
        encoder.train()
        total = 0.0
        count = 0
        idx = np.random.permutation(len(pretrain_graphs))
        for start in range(0, len(idx), batch_size):
            batch_graphs = [pretrain_graphs[i] for i in idx[start:start+batch_size]]
            view1 = [augment_graph(g) for g in batch_graphs]
            view2 = [augment_graph(g) for g in batch_graphs]
            dl1 = next(iter(DataLoader(view1, batch_size=len(view1), shuffle=False)))
            dl2 = next(iter(DataLoader(view2, batch_size=len(view2), shuffle=False)))
            dl1 = dl1.to(device)
            dl2 = dl2.to(device)
            _, z1 = encoder(dl1)
            _, z2 = encoder(dl2)
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(view1)
            count += len(view1)
        losses.append(total / count)
    return encoder, losses


def train_classifier(train_graphs, val_graphs, node_dim, edge_dim, device, pretrained_state=None, epochs=60):
    encoder = GINEncoder(node_dim, edge_dim).to(device)
    if pretrained_state is not None:
        encoder.load_state_dict(pretrained_state)
    model = GraphClassifier(encoder).to(device)

    y_train = np.array([int(g.y.item()) for g in train_graphs])
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False)

    best_state = None
    best_ap = -1
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.float().view(-1)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * y.numel()
            n += y.numel()
        val_metrics = evaluate_classifier(model, val_loader, device)
        val_metrics["train_loss"] = total / n
        history.append(val_metrics)
        if val_metrics["ap"] > best_ap:
            best_ap = val_metrics["ap"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def evaluate_classifier(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(batch.y.cpu().numpy().reshape(-1))
            ps.append(probs.reshape(-1))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    ap = average_precision_score(y, p)
    roc = roc_auc_score(y, p)
    pred = (p >= 0.5).astype(int)
    bal = balanced_accuracy_score(y, pred)
    return {"y": y, "p": p, "ap": float(ap), "roc": float(roc), "bal_acc": float(bal)}


def logistic_baseline_cv(graphs, seed=42):
    X = np.vstack([data_to_feature_vector(g) for g in graphs])
    y = np.array([int(g.y.item()) for g in graphs])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y), 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        rows.append({
            "fold": fold,
            "ap": average_precision_score(y[te], p),
            "roc": roc_auc_score(y[te], p),
            "bal_acc": balanced_accuracy_score(y[te], (p >= 0.5).astype(int)),
        })
    return pd.DataFrame(rows)


def plot_dataset_overview(finetune_graphs, elem_to_idx):
    inv = {v: k for k, v in elem_to_idx.items()}
    rows = []
    for d in finetune_graphs:
        rows.append({
            "label": int(d.y.item()),
            "nodes": d.x.shape[0],
            "edges": d.edge_index.shape[1],
            "density": d.edge_index.shape[1] / max(d.x.shape[0] * (d.x.shape[0] - 1), 1),
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(data=df, x="nodes", hue="label", bins=15, element="step", stat="density", common_norm=False, ax=axes[0])
    sns.histplot(data=df, x="edges", hue="label", bins=20, element="step", stat="density", common_norm=False, ax=axes[1])
    sns.histplot(data=df, x="density", hue="label", bins=20, element="step", stat="density", common_norm=False, ax=axes[2])
    axes[0].set_title("Node count")
    axes[1].set_title("Edge count")
    axes[2].set_title("Graph density")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "dataset_overview.png", dpi=200)
    plt.close(fig)

    elem_counts_pos = np.zeros(len(elem_to_idx))
    elem_counts_neg = np.zeros(len(elem_to_idx))
    for d in finetune_graphs:
        cnt = d.x.sum(0).numpy()
        if int(d.y.item()) == 1:
            elem_counts_pos += cnt
        else:
            elem_counts_neg += cnt
    top = np.argsort(elem_counts_pos + elem_counts_neg)[::-1][:12]
    df2 = pd.DataFrame({
        "element": [inv[int(i)] for i in top] * 2,
        "count": list(elem_counts_pos[top]) + list(elem_counts_neg[top]),
        "class": ["positive"] * len(top) + ["negative"] * len(top),
    })
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df2, x="element", y="count", hue="class", ax=ax)
    ax.set_title("Element frequency in fine-tuning set")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "element_frequencies.png", dpi=200)
    plt.close(fig)


def plot_pretrain_loss(losses):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses)+1), losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title("Self-supervised pretraining convergence")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "pretrain_loss.png", dpi=200)
    plt.close(fig)


def plot_cv_comparison(baseline_df, model_df):
    df = pd.concat([
        baseline_df.assign(model="LogReg baseline"),
        model_df.assign(model="Pretrained GNN"),
    ], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["ap", "roc", "bal_acc"]):
        sns.boxplot(data=df, x="metric_dummy" if False else "model", y=metric, ax=ax)
        sns.stripplot(data=df, x="model", y=metric, color="black", size=4, ax=ax)
        ax.set_title(metric.upper())
        ax.tick_params(axis='x', rotation=12)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "cv_comparison.png", dpi=200)
    plt.close(fig)


def plot_curves(eval_result):
    y = eval_result["y"]
    p = eval_result["p"]
    precision, recall, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(recall, precision)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall (AP={eval_result['ap']:.3f})")

    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title(f"ROC (AUC={eval_result['roc']:.3f})")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "holdout_curves.png", dpi=200)
    plt.close(fig)


def plot_candidate_distribution(candidate_probs, candidate_true=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    if candidate_true is None:
        sns.histplot(candidate_probs, bins=30, ax=ax)
    else:
        df = pd.DataFrame({"prob": candidate_probs, "label": candidate_true})
        sns.histplot(data=df, x="prob", hue="label", bins=30, element="step", stat="count", common_norm=False, ax=ax)
    ax.set_title("Candidate predicted altermagnet probabilities")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "candidate_distribution.png", dpi=200)
    plt.close(fig)


def summarize_papers():
    summaries = []
    for pdf in sorted((BASE / "related_work").glob("*.pdf")):
        reader = PdfReader(str(pdf))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:3])
        snippet = " ".join(text.split())[:1200]
        summaries.append({"file": pdf.name, "snippet": snippet})
    return summaries


def main():
    set_seed(42)
    OUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    pretrain_ds = load_dataset(DATA_DIR / "pretrain_data.pt")
    finetune_ds = load_dataset(DATA_DIR / "finetune_data.pt")
    candidate_ds = load_dataset(DATA_DIR / "candidate_data.pt")

    pretrain_graphs = pretrain_ds.data_list
    finetune_graphs = finetune_ds.data_list
    candidate_graphs = candidate_ds.data_list

    node_dim = finetune_graphs[0].x.shape[1]
    edge_dim = finetune_graphs[0].edge_attr.shape[1]

    plot_dataset_overview(finetune_graphs, finetune_ds.elem_to_idx)

    baseline_df = logistic_baseline_cv(finetune_graphs)
    baseline_df.to_csv(OUT_DIR / "baseline_cv.csv", index=False)

    encoder, losses = pretrain_encoder(pretrain_graphs, node_dim, edge_dim, device, epochs=30)
    torch.save(encoder.state_dict(), OUT_DIR / "pretrained_encoder.pt")
    plot_pretrain_loss(losses)

    y_all = np.array([int(g.y.item()) for g in finetune_graphs])
    idx = np.arange(len(finetune_graphs))
    train_idx, holdout_idx = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=42)
    train_graphs = [finetune_graphs[i] for i in train_idx]
    holdout_graphs = [finetune_graphs[i] for i in holdout_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for fold, (tr_local, te_local) in enumerate(cv.split(np.zeros(len(train_idx)), y_all[train_idx]), 1):
        tr_graphs = [train_graphs[i] for i in tr_local]
        te_graphs = [train_graphs[i] for i in te_local]
        model, _ = train_classifier(
            tr_graphs,
            te_graphs,
            node_dim,
            edge_dim,
            device,
            pretrained_state=encoder.state_dict(),
            epochs=50,
        )
        te_loader = DataLoader(te_graphs, batch_size=128, shuffle=False)
        res = evaluate_classifier(model, te_loader, device)
        cv_rows.append({"fold": fold, "ap": res["ap"], "roc": res["roc"], "bal_acc": res["bal_acc"]})
    model_df = pd.DataFrame(cv_rows)
    model_df.to_csv(OUT_DIR / "gnn_cv.csv", index=False)
    plot_cv_comparison(baseline_df, model_df)

    holdout_train, holdout_val = train_test_split(train_graphs, test_size=0.15, stratify=[int(g.y.item()) for g in train_graphs], random_state=42)
    final_model, history = train_classifier(
        holdout_train,
        holdout_val,
        node_dim,
        edge_dim,
        device,
        pretrained_state=encoder.state_dict(),
        epochs=60,
    )
    torch.save(final_model.state_dict(), OUT_DIR / "final_model.pt")

    holdout_loader = DataLoader(holdout_graphs, batch_size=128, shuffle=False)
    holdout_res = evaluate_classifier(final_model, holdout_loader, device)
    plot_curves(holdout_res)

    cand_loader = DataLoader(candidate_graphs, batch_size=128, shuffle=False)
    final_model.eval()
    cand_probs = []
    cand_true = []
    with torch.no_grad():
        for batch in cand_loader:
            batch = batch.to(device)
            probs = torch.sigmoid(final_model(batch)).cpu().numpy().reshape(-1)
            cand_probs.extend(probs.tolist())
            cand_true.extend(batch.y.cpu().numpy().reshape(-1).tolist())
    cand_probs = np.array(cand_probs)
    cand_true = np.array(cand_true)
    plot_candidate_distribution(cand_probs, cand_true)

    ranking = pd.DataFrame({
        "candidate_id": np.arange(len(candidate_graphs)),
        "predicted_probability": cand_probs,
        "hidden_true_label": cand_true,
    }).sort_values("predicted_probability", ascending=False)
    ranking.to_csv(OUT_DIR / "candidate_ranking.csv", index=False)

    top50 = ranking.head(50)
    discovery_precision = float(top50["hidden_true_label"].mean())
    candidate_ap = float(average_precision_score(cand_true, cand_probs))
    candidate_roc = float(roc_auc_score(cand_true, cand_probs))

    paper_summaries = summarize_papers()

    metrics = {
        "baseline_cv_mean": baseline_df.mean(numeric_only=True).to_dict(),
        "baseline_cv_std": baseline_df.std(numeric_only=True).to_dict(),
        "gnn_cv_mean": model_df.mean(numeric_only=True).to_dict(),
        "gnn_cv_std": model_df.std(numeric_only=True).to_dict(),
        "holdout": {k: v for k, v in holdout_res.items() if k not in ["y", "p"]},
        "candidate": {
            "ap": candidate_ap,
            "roc": candidate_roc,
            "top50_precision": discovery_precision,
            "true_positives_in_top50": int(top50["hidden_true_label"].sum()),
            "total_true_positives": int(cand_true.sum()),
        },
        "pretrain_final_loss": float(losses[-1]),
        "paper_summaries": paper_summaries,
    }
    with open(OUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT_DIR / "final_training_history.csv", index=False)

    top50.to_csv(OUT_DIR / "top50_candidates.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
