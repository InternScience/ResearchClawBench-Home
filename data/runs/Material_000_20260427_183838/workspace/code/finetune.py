"""Fine-tune classifier (GNN encoder + MLP head) with stratified split,
class-weighted BCE.  Compares with a from-scratch GNN baseline and a
composition-only logistic-regression baseline.
"""
import os, sys, json, copy, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from models import GNNEncoder, Classifier, load_dataset, ROOT

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(ROOT, "outputs")


def comp_features(ds):
    n = len(ds)
    feats = np.zeros((n, ds[0].x.size(1)), dtype=np.float32)
    for i, d in enumerate(ds):
        feats[i] = d.x.sum(0).numpy()
    return feats


def train_gnn(train_list, val_list, *, pretrained_state=None,
              epochs: int = 60, lr: float = 1e-3, weight: float = 19.0):
    enc = GNNEncoder().to(device)
    if pretrained_state is not None:
        enc.load_state_dict(pretrained_state)
    clf = Classifier(enc.hidden).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()),
                           lr=lr, weight_decay=1e-5)
    pos_w = torch.tensor([weight], device=device)
    train_loader = DataLoader(train_list, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=256, shuffle=False)
    best = {"auc": -1.0, "state": None, "epoch": -1}
    for ep in range(epochs):
        enc.train(); clf.train()
        for batch in train_loader:
            batch = batch.to(device)
            g, _ = enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = clf(g)
            y = batch.y.float()
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        # eval
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
                    "state": (copy.deepcopy(enc.state_dict()),
                              copy.deepcopy(clf.state_dict()))}
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
    print(f"train pos/neg: {ys[train_idx].sum()}/{(ys[train_idx]==0).sum()}  "
          f"val pos/neg: {ys[val_idx].sum()}/{(ys[val_idx]==0).sum()}")

    # 1) pretrained -> finetune
    pre = torch.load(os.path.join(OUT, "pretrained_encoder.pt"))
    best_pre = train_gnn(train_list, val_list, pretrained_state=pre["encoder"])
    print(f"[GNN+pretrain]  val AUC={best_pre['auc']:.4f}  AP={best_pre['ap']:.4f}  ep={best_pre['epoch']}")

    # 2) from scratch
    best_scr = train_gnn(train_list, val_list, pretrained_state=None)
    print(f"[GNN-scratch ]  val AUC={best_scr['auc']:.4f}  AP={best_scr['ap']:.4f}  ep={best_scr['epoch']}")

    # 3) composition-only LR baseline
    Xtr = comp_features([fin[i] for i in train_idx])
    Xv = comp_features([fin[i] for i in val_idx])
    Xc = comp_features(list(cand))
    yt = ys[train_idx]; yv = ys[val_idx]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Xtr, yt)
    p_v = lr.predict_proba(Xv)[:, 1]
    auc_lr = roc_auc_score(yv, p_v); ap_lr = average_precision_score(yv, p_v)
    p_cand_lr = lr.predict_proba(Xc)[:, 1]
    print(f"[LR(comp)    ]  val AUC={auc_lr:.4f}  AP={ap_lr:.4f}")

    # candidate predictions for the two GNN models
    enc_pre, clf_pre = best_pre["state"]
    enc_scr, clf_scr = best_scr["state"]
    p_cand_pre, embs_cand = predict_gnn(enc_pre, clf_pre, cand)
    p_cand_scr, _ = predict_gnn(enc_scr, clf_scr, cand)

    # save val predictions for ROC/PR figures
    p_v_pre, embs_v = predict_gnn(enc_pre, clf_pre, val_list)
    p_v_scr, _ = predict_gnn(enc_scr, clf_scr, val_list)

    np.savez(os.path.join(OUT, "predictions.npz"),
             y_val=yv, p_val_pre=p_v_pre, p_val_scr=p_v_scr, p_val_lr=p_v,
             y_cand=np.array([int(d.y.item()) for d in cand]),
             p_cand_pre=p_cand_pre, p_cand_scr=p_cand_scr, p_cand_lr=p_cand_lr,
             embs_cand=embs_cand)

    # candidate metrics
    yc = np.array([int(d.y.item()) for d in cand])
    out = {
        "split": {"train_pos": int(yt.sum()), "train_neg": int((yt == 0).sum()),
                  "val_pos": int(yv.sum()), "val_neg": int((yv == 0).sum())},
        "val": {
            "GNN_pretrain": {"auc": best_pre["auc"], "ap": best_pre["ap"], "best_epoch": best_pre["epoch"]},
            "GNN_scratch":  {"auc": best_scr["auc"], "ap": best_scr["ap"], "best_epoch": best_scr["epoch"]},
            "LR_composition": {"auc": float(auc_lr), "ap": float(ap_lr)},
        },
        "candidate": {
            "n": len(yc), "true_positives": int(yc.sum()),
            "GNN_pretrain": {
                "auc": float(roc_auc_score(yc, p_cand_pre)),
                "ap": float(average_precision_score(yc, p_cand_pre)),
            },
            "GNN_scratch": {
                "auc": float(roc_auc_score(yc, p_cand_scr)),
                "ap": float(average_precision_score(yc, p_cand_scr)),
            },
            "LR_composition": {
                "auc": float(roc_auc_score(yc, p_cand_lr)),
                "ap": float(average_precision_score(yc, p_cand_lr)),
            },
        },
    }

    # Top-k stats for GNN+pretrain
    for k in (20, 50, 100):
        order = np.argsort(-p_cand_pre)[:k]
        hits = int(yc[order].sum())
        out["candidate"][f"top{k}"] = {"hits": hits, "precision": hits / k,
                                        "recall": hits / max(int(yc.sum()), 1)}
    print(json.dumps(out, indent=2))
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # save encoder + classifier
    torch.save({"enc_pre": enc_pre, "clf_pre": clf_pre,
                "enc_scr": enc_scr, "clf_scr": clf_scr},
               os.path.join(OUT, "models.pt"))


if __name__ == "__main__":
    main()
