"""
Workflow #1 — Property prediction with a CGCNN-lite (Xie & Grossman, PRL 2018).

Architecture (faithful to the named CGCNN ingredients):
  * Per-atom features: learnable embedding of atomic number Z + a scalar
    feature x (the per-atom positional value parsed from the dataset).
  * Edge features: |x_i - x_j| (a 1-D distance proxy expanded into a
    Gaussian basis, matching the CGCNN distance-expansion idea).
  * Graph convolution:
        z_ij = sigma(W_f [v_i || v_j || e_ij]) * softplus(W_s [v_i || v_j || e_ij])
        v_i' = v_i + sum_j z_ij
    (the gated convolution from CGCNN Eq. 5).
  * 2 conv layers, then mean pooling -> MLP head -> graph-level scalar.
Baseline: a vanilla MLP that ignores the graph topology and operates on
mean(x) and mean(Z) per graph.

We compare on RMSE / MAE / R^2 over a held-out 80/20 split. The dataset
is intentionally toy-sized; the goal is workflow demonstration and a
fair head-to-head against a non-graph baseline.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)


def gaussian_basis(d: torch.Tensor, n=8, dmin=0.0, dmax=4.0) -> torch.Tensor:
    centers = torch.linspace(dmin, dmax, n, device=d.device)
    width = (dmax - dmin) / (n - 1)
    return torch.exp(-((d.unsqueeze(-1) - centers) ** 2) / (width ** 2))


class CGCNNConv(nn.Module):
    def __init__(self, atom_dim: int, edge_dim: int):
        super().__init__()
        self.W_f = nn.Linear(2 * atom_dim + edge_dim, atom_dim)
        self.W_s = nn.Linear(2 * atom_dim + edge_dim, atom_dim)
        self.bn = nn.BatchNorm1d(atom_dim)

    def forward(self, V: torch.Tensor, edges: torch.Tensor, E: torch.Tensor):
        # V: (B, N, A); edges: (M, 2); E: (B, M, edge_dim)
        B, N, A = V.shape
        i, j = edges[:, 0], edges[:, 1]
        v_i = V[:, i, :]               # (B, M, A)
        v_j = V[:, j, :]               # (B, M, A)
        cat = torch.cat([v_i, v_j, E], dim=-1)
        z = torch.sigmoid(self.W_f(cat)) * F.softplus(self.W_s(cat))
        msg = torch.zeros_like(V)
        msg = msg.index_add(1, i, z)   # aggregate neighbours by atom index
        out = V + msg
        out = self.bn(out.reshape(-1, A)).reshape(B, N, A)
        return out


class CGCNNLite(nn.Module):
    def __init__(self, n_z=10, atom_dim=32, edge_dim=8):
        super().__init__()
        self.embed = nn.Embedding(n_z, atom_dim - 1)
        self.atom_dim = atom_dim
        self.edge_dim = edge_dim
        self.conv1 = CGCNNConv(atom_dim, edge_dim)
        self.conv2 = CGCNNConv(atom_dim, edge_dim)
        self.head = nn.Sequential(
            nn.Linear(atom_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, Z: torch.Tensor, X: torch.Tensor, edges: torch.Tensor):
        # Z: (B, N) long; X: (B, N) float; edges: (M, 2)
        emb = self.embed(Z)                              # (B, N, A-1)
        V = torch.cat([emb, X.unsqueeze(-1)], dim=-1)    # (B, N, A)
        i, j = edges[:, 0], edges[:, 1]
        d = torch.abs(X[:, i] - X[:, j])                 # (B, M)
        E = gaussian_basis(d, n=self.edge_dim)           # (B, M, edge_dim)
        V = self.conv1(V, edges, E)
        V = F.relu(V)
        V = self.conv2(V, edges, E)
        V = F.relu(V)
        g = V.mean(dim=1)                                # (B, A)
        return self.head(g).squeeze(-1)                  # (B,)


class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, Z: torch.Tensor, X: torch.Tensor, edges: torch.Tensor):
        feat = torch.stack([Z.float().mean(dim=1), X.mean(dim=1)], dim=-1)
        return self.net(feat).squeeze(-1)


def train_model(model, train, val, epochs=400, lr=5e-3, weight_decay=1e-3):
    Z_tr, X_tr, y_tr = train
    Z_va, X_va, y_va = val
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    edges = train[3] if len(train) > 3 else None  # not used; passed via globals
    history = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Z_tr, X_tr, EDGES)
        loss = F.mse_loss(pred, y_tr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(Z_va, X_va, EDGES), y_va).item()
        history["train"].append(float(loss.item()))
        history["val"].append(float(v))
    return history

def train_model_es(model, train, val, epochs=600, lr=5e-3, weight_decay=1e-3, patience=80):
    Z_tr, X_tr, y_tr = train
    Z_va, X_va, y_va = val
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train": [], "val": []}
    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.mse_loss(model(Z_tr, X_tr, EDGES), y_tr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(Z_va, X_va, EDGES), y_va).item()
        history["train"].append(float(loss.item()))
        history["val"].append(float(v))
        if v < best_val - 1e-6:
            best_val = v; best_state = {k: w.detach().clone() for k, w in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def metrics(model, Z, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(Z, X, EDGES).cpu().numpy()
    y_np = y.cpu().numpy()
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_np, pred))),
        "mae": float(mean_absolute_error(y_np, pred)),
        "r2": float(r2_score(y_np, pred)),
    }, pred


def main():
    npz = np.load(OUT / "parsed_data.npz")
    Z = torch.from_numpy(npz["pp_Z"]).long()
    X = torch.from_numpy(npz["pp_X"]).float()
    y = torch.from_numpy(npz["pp_y"]).float()
    edges = torch.from_numpy(npz["pp_edges"]).long()

    # train/val split (80/20)
    rng = np.random.default_rng(0)
    idx = np.arange(Z.shape[0])
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    tr, va = idx[:cut], idx[cut:]

    Z_tr, X_tr, y_tr = Z[tr], X[tr], y[tr]
    Z_va, X_va, y_va = Z[va], X[va], y[va]

    # standardize y for stable training
    y_mean = y_tr.mean()
    y_std = y_tr.std() + 1e-8
    y_tr_n = (y_tr - y_mean) / y_std
    y_va_n = (y_va - y_mean) / y_std

    global EDGES
    EDGES = edges  # used inside model.forward

    # CGCNN-lite
    cg = CGCNNLite(n_z=int(Z.max().item()) + 1, atom_dim=16, edge_dim=8)
    hist_cg = train_model_es(cg,
                          (Z_tr, X_tr, y_tr_n),
                          (Z_va, X_va, y_va_n))
    # de-standardize for reporting
    def _eval(m):
        m.eval()
        with torch.no_grad():
            tr_p = (m(Z_tr, X_tr, EDGES) * y_std + y_mean).cpu().numpy()
            va_p = (m(Z_va, X_va, EDGES) * y_std + y_mean).cpu().numpy()
        return tr_p, va_p
    cg_tr_p, cg_va_p = _eval(cg)

    # MLP baseline
    mlp = MLPBaseline()
    hist_mlp = train_model_es(mlp,
                           (Z_tr, X_tr, y_tr_n),
                           (Z_va, X_va, y_va_n))
    mlp_tr_p, mlp_va_p = _eval(mlp)

    def _m(yp, yt):
        return {
            "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        }

    results = {
        "split": {"n_train": int(len(tr)), "n_val": int(len(va))},
        "CGCNN_lite": {
            "train": _m(cg_tr_p, y_tr.numpy()),
            "val":   _m(cg_va_p, y_va.numpy()),
        },
        "MLP_baseline": {
            "train": _m(mlp_tr_p, y_tr.numpy()),
            "val":   _m(mlp_va_p, y_va.numpy()),
        },
    }
    (OUT / "property_prediction_metrics.json").write_text(json.dumps(results, indent=2))
    np.savez(OUT / "property_prediction_preds.npz",
             y_tr=y_tr.numpy(), y_va=y_va.numpy(),
             cg_tr=cg_tr_p, cg_va=cg_va_p,
             mlp_tr=mlp_tr_p, mlp_va=mlp_va_p,
             hist_cg_train=np.array(hist_cg["train"]),
             hist_cg_val=np.array(hist_cg["val"]),
             hist_mlp_train=np.array(hist_mlp["train"]),
             hist_mlp_val=np.array(hist_mlp["val"]))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
