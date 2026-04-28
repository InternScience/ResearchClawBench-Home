"""
Training utilities for DIDS-MFL on NF-UNSW-NB15-v2.
Chronological mini-batch training to keep TGN-like memory consistent.
"""
import os
import json
import time
import math
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)

from models import DIDSMFL, MLPBaseline, EGraphSAGEBaseline

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "outputs")
os.makedirs(OUT, exist_ok=True)


def load_data():
    z = np.load(os.path.join(OUT, "data_clean.npz"))
    return {k: z[k] for k in z.files}


def remap_nodes(src, dst):
    all_ids = np.concatenate([src, dst])
    uniq, inv = np.unique(all_ids, return_inverse=True)
    n_nodes = len(uniq)
    new_src = inv[: len(src)]
    new_dst = inv[len(src):]
    return new_src.astype(np.int64), new_dst.astype(np.int64), n_nodes


def chunked_batches(idx, batch_size):
    n = len(idx)
    for s in range(0, n, batch_size):
        yield idx[s:s + batch_size]


def to_tensor(arr, dtype=None):
    t = torch.from_numpy(arr)
    return t.to(dtype) if dtype else t


def evaluate(model, idx, src, dst, t, msg, label, attack, batch_size=2048,
             n_classes=10, with_proba=False):
    model.eval()
    all_bin = []
    all_multi = []
    all_label = []
    all_attack = []
    all_proba_pos = []
    with torch.no_grad():
        for b in chunked_batches(idx, batch_size):
            s = torch.from_numpy(src[b]).long()
            d = torch.from_numpy(dst[b]).long()
            tt = torch.from_numpy(t[b]).float()
            mm = torch.from_numpy(msg[b]).float()
            blog, mlog, _, _ = model(s, d, tt, mm)
            all_bin.append(blog.argmax(-1).cpu().numpy())
            all_multi.append(mlog.argmax(-1).cpu().numpy())
            all_proba_pos.append(torch.softmax(blog, -1)[:, 1].cpu().numpy())
            all_label.append(label[b])
            all_attack.append(attack[b])
    preds_b = np.concatenate(all_bin)
    preds_m = np.concatenate(all_multi)
    prob_pos = np.concatenate(all_proba_pos)
    y_b = np.concatenate(all_label)
    y_a = np.concatenate(all_attack)
    out = {
        "bin_acc": accuracy_score(y_b, preds_b),
        "bin_f1": f1_score(y_b, preds_b, average="binary"),
        "bin_macro_f1": f1_score(y_b, preds_b, average="macro"),
        "multi_acc": accuracy_score(y_a, preds_m),
        "multi_macro_f1": f1_score(y_a, preds_m, average="macro", labels=list(range(n_classes)), zero_division=0),
        "multi_weighted_f1": f1_score(y_a, preds_m, average="weighted", zero_division=0),
    }
    try:
        out["bin_auc"] = roc_auc_score(y_b, prob_pos)
    except Exception:
        out["bin_auc"] = float("nan")

    # per-attack F1 (multi-class)
    classes = list(range(n_classes))
    p, r, f, sup = precision_recall_fscore_support(
        y_a, preds_m, labels=classes, zero_division=0
    )
    out["per_attack_f1"] = {str(c): float(f[i]) for i, c in enumerate(classes)}
    out["per_attack_support"] = {str(c): int(sup[i]) for i, c in enumerate(classes)}
    if with_proba:
        out["preds_b"] = preds_b
        out["preds_m"] = preds_m
        out["prob_pos"] = prob_pos
        out["y_b"] = y_b
        out["y_a"] = y_a
    return out


def train_one(model_kind, seed=0, n_epochs=2, batch_size=1024,
              alpha=0.1, beta=0.1, smooth=0.0, lr=1e-3, verbose=True,
              n_classes=10, mask_train=None):
    """mask_train: optional boolean mask over the training indices to use
    (e.g. for unknown/few-shot settings). The TGN memory is still updated by
    the chronological pass over ALL training flows so that node states are
    realistic, but the supervised loss is only computed where mask_train is True.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_data()
    src, dst, t = data["src"], data["dst"], data["t"]
    msg = data["msg"]
    label, attack = data["label"], data["attack"]
    idx_tr, idx_va, idx_te = data["idx_tr"], data["idx_va"], data["idx_te"]

    src_n, dst_n, n_nodes = remap_nodes(src, dst)

    if model_kind == "didsmfl":
        model = DIDSMFL(n_nodes, msg.shape[1], emb=64, time_dim=16,
                        n_groups=4, group_dim=16, n_classes=n_classes,
                        use_sd=True, use_rd=True, multi_scale=True)
    elif model_kind == "ablate_sd":
        model = DIDSMFL(n_nodes, msg.shape[1], emb=64, time_dim=16,
                        n_groups=4, group_dim=16, n_classes=n_classes,
                        use_sd=False, use_rd=True, multi_scale=True)
    elif model_kind == "ablate_rd":
        model = DIDSMFL(n_nodes, msg.shape[1], emb=64, time_dim=16,
                        n_groups=4, group_dim=16, n_classes=n_classes,
                        use_sd=True, use_rd=False, multi_scale=True)
    elif model_kind == "ablate_ms":
        model = DIDSMFL(n_nodes, msg.shape[1], emb=64, time_dim=16,
                        n_groups=4, group_dim=16, n_classes=n_classes,
                        use_sd=True, use_rd=True, multi_scale=False)
    elif model_kind == "mlp":
        model = MLPBaseline(msg.shape[1], n_classes)
    elif model_kind == "egraphsage":
        model = EGraphSAGEBaseline(n_nodes, msg.shape[1], emb=64,
                                   time_dim=16, n_classes=n_classes)
    else:
        raise ValueError(model_kind)

    # weighted CE for multiclass
    cw = np.ones(n_classes, dtype=np.float32)
    train_attack = attack[idx_tr]
    counts = np.bincount(train_attack, minlength=n_classes).astype(np.float32) + 1.0
    cw = (counts.sum() / counts) ** 0.5
    cw = cw / cw.mean()
    cw_tensor = torch.from_numpy(cw)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val": []}

    for epoch in range(n_epochs):
        # Reset TGN memory at the start of each epoch (single pass through time)
        if hasattr(model, "mem"):
            model.mem.reset()
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for b in chunked_batches(idx_tr, batch_size):
            s = torch.from_numpy(src_n[b]).long()
            d = torch.from_numpy(dst_n[b]).long()
            tt = torch.from_numpy(t[b]).float()
            mm = torch.from_numpy(msg[b]).float()
            yb = torch.from_numpy(label[b]).long()
            ya = torch.from_numpy(attack[b]).long()

            blog, mlog, x, edge_h = model(s, d, tt, mm)

            if mask_train is not None:
                m = torch.from_numpy(mask_train[b]).bool()
                if m.sum() == 0:
                    continue
                ce_b = Fnn.cross_entropy(blog[m], yb[m])
                ce_m = Fnn.cross_entropy(mlog[m], ya[m], weight=cw_tensor)
            else:
                ce_b = Fnn.cross_entropy(blog[:], yb[:])
                ce_m = Fnn.cross_entropy(mlog[:], ya[:], weight=cw_tensor)

            loss = ce_b + 0.5 * ce_m

            # disentanglement losses for didsmfl variants
            if model_kind in ("didsmfl", "ablate_ms"):
                if isinstance(model.sd, type(model.sd)) and hasattr(model.sd, "disentangle_loss"):
                    loss = loss + alpha * model.sd.disentangle_loss(mm)
                if model.rd is not None:
                    loss = loss + beta * model.rd.ortho_loss(edge_h)
            elif model_kind == "ablate_sd":
                if model.rd is not None:
                    loss = loss + beta * model.rd.ortho_loss(edge_h)
            elif model_kind == "ablate_rd":
                if hasattr(model.sd, "disentangle_loss"):
                    loss = loss + alpha * model.sd.disentangle_loss(mm)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        # eval
        # For TGN-style, val/test is a continuation of training memory in temporal order.
        val_metrics = evaluate(model, idx_va, src_n, dst_n, t, msg, label, attack,
                               n_classes=n_classes)
        history["train_loss"].append(avg_loss)
        history["val"].append({k: v for k, v in val_metrics.items() if not isinstance(v, dict)})
        if verbose:
            print(f"[{model_kind} seed={seed}] epoch {epoch} loss={avg_loss:.4f} "
                  f"val_bin_f1={val_metrics['bin_f1']:.4f} "
                  f"val_macro_f1={val_metrics['multi_macro_f1']:.4f} "
                  f"({time.time()-t0:.1f}s)")

    test_metrics = evaluate(model, idx_te, src_n, dst_n, t, msg, label, attack,
                            n_classes=n_classes, with_proba=True)
    return model, history, test_metrics
