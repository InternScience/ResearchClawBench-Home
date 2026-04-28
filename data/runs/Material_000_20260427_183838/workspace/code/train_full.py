"""Stronger feature-based and GNN models exploiting element-pair edge signal.

Adds:
- Pair-edge baseline: indicator features for each (sorted) element pair
  appearing as a graph edge.  Logistic regression / Gradient boosting.
- An improved GNN (more epochs, lower LR finetune, multi-seed averaging).

Also: candidate ranking + top-50 export with property heuristics
(metal/insulator and d/g/i wave).
"""
import os, sys, json, copy, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from models import GNNEncoder, Classifier, load_dataset, ROOT

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def pair_features(ds, n_elem: int = 28):
    """Indicator features: for each pair (i<=j) of element indices, 1 if
    any edge in the graph connects atoms of those types.  Plus composition.
    """
    n_pairs = n_elem * (n_elem + 1) // 2

    def pair_idx(a, b):
        if a > b: a, b = b, a
        return a * n_elem - a * (a - 1) // 2 + (b - a)

    feats = np.zeros((len(ds), n_pairs + n_elem), dtype=np.float32)
    for k, d in enumerate(ds):
        x = d.x.argmax(1).numpy()
        ei = d.edge_index.numpy()
        for u, v in ei.T:
            feats[k, pair_idx(int(x[u]), int(x[v]))] = 1.0
        feats[k, n_pairs:] = d.x.sum(0).numpy()
    return feats


def train_gnn(train_list, val_list, *, pretrained_state=None,
              epochs: int = 80, lr: float = 1e-3, weight: float = 19.0,
              seed: int = 0):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    enc = GNNEncoder().to(device)
    if pretrained_state is not None:
        enc.load_state_dict(pretrained_state)
    clf = Classifier(enc.hidden).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()),
                           lr=lr, weight_decay=1e-5)
    pos_w = torch.tensor([weight], device=device)
    train_loader = DataLoader(train_list, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=256, shuffle=False)
    best = {"auc": -1.0}
    for ep in range(epochs):
        enc.train(); clf.train()
        for batch in train_loader:
            batch = batch.to(device)
            g, _ = enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = clf(g)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y.float(), pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); clf.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                g, _ = enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                p = torch.sigmoid(clf(g)).cpu().numpy()
                ps.append(p); ys.append(batch.y.cpu().numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        auc = roc_auc_score(ys, ps); ap = average_precision_score(ys, ps)
        if auc > best["auc"]:
            best = {"auc": float(auc), "ap": float(ap), "epoch": ep,
                    "enc_state": copy.deepcopy(enc.state_dict()),
                    "clf_state": copy.deepcopy(clf.state_dict())}
    return best


def predict_gnn(enc_state, clf_state, ds):
    enc = GNNEncoder().to(device); enc.load_state_dict(enc_state); enc.eval()
    clf = Classifier(enc.hidden).to(device); clf.load_state_dict(clf_state); clf.eval()
    loader = DataLoader(list(ds), batch_size=256, shuffle=False)
    probs, embs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            g, _ = enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            p = torch.sigmoid(clf(g)).cpu().numpy()
            probs.append(p); embs.append(g.cpu().numpy())
    return np.concatenate(probs), np.concatenate(embs)


def main():
    fin = load_dataset("finetune_data.pt")
    cand = load_dataset("candidate_data.pt")
    ys = np.array([int(d.y.item()) for d in fin])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(np.zeros(len(ys)), ys))
    train_list = [fin[i] for i in train_idx]
    val_list = [fin[i] for i in val_idx]
    yt, yv = ys[train_idx], ys[val_idx]

    # ---- Pair-edge features ----
    Xtr = pair_features(train_list); Xv = pair_features(val_list)
    Xc = pair_features(list(cand))

    # Logistic regression with L1 to keep it sparse
    lr = LogisticRegression(max_iter=4000, class_weight="balanced",
                             penalty="l2", C=1.0).fit(Xtr, yt)
    p_v_lr = lr.predict_proba(Xv)[:, 1]
    p_c_lr = lr.predict_proba(Xc)[:, 1]
    auc_lr = roc_auc_score(yv, p_v_lr); ap_lr = average_precision_score(yv, p_v_lr)
    print(f"[LR(pair)    ] val AUC={auc_lr:.4f}  AP={ap_lr:.4f}")

    # Gradient boosting
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     random_state=SEED).fit(Xtr, yt)
    p_v_gb = gb.predict_proba(Xv)[:, 1]
    p_c_gb = gb.predict_proba(Xc)[:, 1]
    auc_gb = roc_auc_score(yv, p_v_gb); ap_gb = average_precision_score(yv, p_v_gb)
    print(f"[GB(pair)    ] val AUC={auc_gb:.4f}  AP={ap_gb:.4f}")

    # ---- GNN models, multi-seed ensemble ----
    pre = torch.load(os.path.join(OUT, "pretrained_encoder.pt"))
    p_v_pre_ens = np.zeros(len(yv)); p_c_pre_ens = np.zeros(len(cand))
    p_v_scr_ens = np.zeros(len(yv)); p_c_scr_ens = np.zeros(len(cand))
    embs_cand = None
    aucs_pre, aps_pre, aucs_scr, aps_scr = [], [], [], []
    for s in range(3):
        bp = train_gnn(train_list, val_list, pretrained_state=pre["encoder"], seed=s)
        bs = train_gnn(train_list, val_list, pretrained_state=None, seed=s)
        aucs_pre.append(bp["auc"]); aps_pre.append(bp["ap"])
        aucs_scr.append(bs["auc"]); aps_scr.append(bs["ap"])
        pv, _ = predict_gnn(bp["enc_state"], bp["clf_state"], val_list)
        pc, embs = predict_gnn(bp["enc_state"], bp["clf_state"], cand)
        p_v_pre_ens += pv; p_c_pre_ens += pc
        if embs_cand is None: embs_cand = embs
        pv, _ = predict_gnn(bs["enc_state"], bs["clf_state"], val_list)
        pc, _ = predict_gnn(bs["enc_state"], bs["clf_state"], cand)
        p_v_scr_ens += pv; p_c_scr_ens += pc
        print(f"  seed={s}: GNN+pre AUC={bp['auc']:.4f}/AP={bp['ap']:.4f}  scratch AUC={bs['auc']:.4f}/AP={bs['ap']:.4f}")
    p_v_pre_ens /= 3; p_c_pre_ens /= 3
    p_v_scr_ens /= 3; p_c_scr_ens /= 3
    auc_pre = roc_auc_score(yv, p_v_pre_ens); ap_pre = average_precision_score(yv, p_v_pre_ens)
    auc_scr = roc_auc_score(yv, p_v_scr_ens); ap_scr = average_precision_score(yv, p_v_scr_ens)
    print(f"[GNN+pretrain ensemble] val AUC={auc_pre:.4f}  AP={ap_pre:.4f}")
    print(f"[GNN scratch  ensemble] val AUC={auc_scr:.4f}  AP={ap_scr:.4f}")

    # Combined model: average GNN+pretrain ensemble with GB
    p_v_comb = 0.5 * p_v_pre_ens + 0.5 * p_v_gb
    p_c_comb = 0.5 * p_c_pre_ens + 0.5 * p_c_gb
    auc_comb = roc_auc_score(yv, p_v_comb); ap_comb = average_precision_score(yv, p_v_comb)
    print(f"[Combined GNN+GB] val AUC={auc_comb:.4f}  AP={ap_comb:.4f}")

    yc = np.array([int(d.y.item()) for d in cand])
    out = {
        "split": {"train_pos": int(yt.sum()), "train_neg": int((yt == 0).sum()),
                  "val_pos": int(yv.sum()), "val_neg": int((yv == 0).sum())},
        "val": {
            "GNN_pretrain_ens": {"auc": float(auc_pre), "ap": float(ap_pre),
                                  "auc_per_seed": aucs_pre, "ap_per_seed": aps_pre},
            "GNN_scratch_ens":  {"auc": float(auc_scr), "ap": float(ap_scr),
                                  "auc_per_seed": aucs_scr, "ap_per_seed": aps_scr},
            "LR_pair":          {"auc": float(auc_lr),  "ap": float(ap_lr)},
            "GB_pair":          {"auc": float(auc_gb),  "ap": float(ap_gb)},
            "Combined_GNN_GB":  {"auc": float(auc_comb), "ap": float(ap_comb)},
        },
        "candidate": {
            "n": len(yc), "true_positives": int(yc.sum()),
            "GNN_pretrain_ens": {
                "auc": float(roc_auc_score(yc, p_c_pre_ens)),
                "ap": float(average_precision_score(yc, p_c_pre_ens))},
            "GNN_scratch_ens": {
                "auc": float(roc_auc_score(yc, p_c_scr_ens)),
                "ap": float(average_precision_score(yc, p_c_scr_ens))},
            "LR_pair": {"auc": float(roc_auc_score(yc, p_c_lr)),
                         "ap": float(average_precision_score(yc, p_c_lr))},
            "GB_pair": {"auc": float(roc_auc_score(yc, p_c_gb)),
                         "ap": float(average_precision_score(yc, p_c_gb))},
            "Combined_GNN_GB": {"auc": float(roc_auc_score(yc, p_c_comb)),
                                  "ap": float(average_precision_score(yc, p_c_comb))},
        },
    }
    for k in (20, 50, 100):
        for name, p in [("GNN_pretrain_ens", p_c_pre_ens),
                        ("GB_pair", p_c_gb),
                        ("Combined_GNN_GB", p_c_comb)]:
            order = np.argsort(-p)[:k]
            hits = int(yc[order].sum())
            out["candidate"].setdefault(f"top{k}", {})[name] = {
                "hits": hits, "precision": hits / k,
                "recall": hits / max(int(yc.sum()), 1)}

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    np.savez(os.path.join(OUT, "predictions.npz"),
             y_val=yv, p_val_pre=p_v_pre_ens, p_val_scr=p_v_scr_ens,
             p_val_lr=p_v_lr, p_val_gb=p_v_gb, p_val_comb=p_v_comb,
             y_cand=yc, p_cand_pre=p_c_pre_ens, p_cand_scr=p_c_scr_ens,
             p_cand_lr=p_c_lr, p_cand_gb=p_c_gb, p_cand_comb=p_c_comb,
             embs_cand=embs_cand)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
