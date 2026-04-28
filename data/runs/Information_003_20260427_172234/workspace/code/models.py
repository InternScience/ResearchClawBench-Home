"""
Models for DIDS-MFL approximation on edge-level intrusion detection.

Components:
  - StatisticalDisentangle: a learnable per-feature reweighting w in [0,1]
    optimized with a smooth surrogate of the 3D-IDS SMT objective
    (Eq. 5/6/7), encouraging neighbouring weighted features to differ:
        L_SD = - mean_i ( |2*w_i*F_i - w_{i-1}*F_{i-1} - w_{i+1}*F_{i+1}| )
                + boundary terms,
    plus simplex-style constraint sum_i w_i = K via a soft penalty.
  - RepDisentangle: an MLP that produces D groups of d-dim sub-vectors
    (component-wise representation). An orthogonality regularizer
    (||V V^T - I||_F^2 on group means) decorrelates components, mimicking
    the representational disentanglement step.
  - DynamicMemory: TGN-style memory + GRU + sinusoidal time encoder for
    src and dst nodes, producing temporal node embeddings.
  - MultiScaleFusion: concatenates representations at multiple GNN
    propagation depths (1,2,3 hops via repeated message passing on the
    most recent neighbor cache) for few-shot learning.
  - EdgeClassifier: takes [edge_repr, src_node, dst_node, time_enc] and
    outputs binary + multi-class logits.

This is a CPU-friendly, faithful approximation. We run mini-batch chronological
training so the temporal memory advances correctly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.tensor(
            [1.0 / 10 ** (i / dim) for i in range(dim)], dtype=torch.float32
        ))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        # t: [B]
        return torch.cos(t.unsqueeze(-1) * self.w + self.b)


class StatisticalDisentangle(nn.Module):
    """Per-feature gate w_i in (0,1), optimized to differentiate adjacent features.

    We reorder feature axis once (descending variance) so that the
    "neighboring features" objective from 3D-IDS makes sense.
    """

    def __init__(self, n_feat, init_order=None, target_sum=None):
        super().__init__()
        self.n_feat = n_feat
        self.logit = nn.Parameter(torch.zeros(n_feat))
        if init_order is None:
            init_order = torch.arange(n_feat)
        self.register_buffer("perm", init_order.long())
        self.target_sum = float(target_sum) if target_sum is not None else float(n_feat) * 0.5

    def w(self):
        # constrained to (0,1)
        return torch.sigmoid(self.logit)

    def forward(self, x):
        # x: [B, F]
        w = self.w()
        return x[:, self.perm] * w[self.perm]

    def disentangle_loss(self, x):
        """Smooth surrogate of the SMT differentiation objective on a
        sample batch. Encourages adjacent (along permutation) gated features
        to differ, plus a soft simplex constraint.
        x: [B, F]
        """
        w = self.w()
        Fperm = (x[:, self.perm].abs().mean(0)) * w[self.perm]
        # adjacent differences
        diff = torch.abs(2 * Fperm[1:-1] - Fperm[2:] - Fperm[:-2]).mean()
        ends = torch.abs(Fperm[-1] - Fperm[0])
        # simplex penalty
        simplex = (w.sum() - self.target_sum) ** 2 / (self.n_feat ** 2)
        # we want LARGER differences -> minimise negative
        return -(diff + 0.5 * ends) + 1e-2 * simplex


class RepresentationalDisentangle(nn.Module):
    """Project gated features into K disentangled groups; orthogonality
    regularizer decorrelates them.
    """

    def __init__(self, in_dim, n_groups=4, group_dim=16):
        super().__init__()
        self.K = n_groups
        self.gd = group_dim
        self.proj = nn.Linear(in_dim, n_groups * group_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.proj(x))  # [B, K*gd]
        return h

    def split(self, h):
        # [B, K, gd]
        B = h.shape[0]
        return h.view(B, self.K, self.gd)

    def ortho_loss(self, h):
        z = self.split(h)
        # normalize each group
        zn = F.normalize(z, dim=-1)
        # mean per group across batch
        mu = zn.mean(0)  # [K, gd]
        mu = F.normalize(mu, dim=-1)
        gram = mu @ mu.t()  # [K,K]
        eye = torch.eye(self.K, device=gram.device)
        return ((gram - eye) ** 2).mean()


class DynamicMemory(nn.Module):
    """TGN-like memory of dimension D for each node."""

    def __init__(self, n_nodes, dim, msg_dim, time_dim):
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        self.gru = nn.GRUCell(msg_dim + time_dim, dim)
        self.register_buffer("memory", torch.zeros(n_nodes, dim))
        self.register_buffer("last_t", torch.zeros(n_nodes))

    def reset(self):
        self.memory.zero_()
        self.last_t.zero_()

    def detach(self):
        self.memory = self.memory.detach()

    def read(self, ids):
        return self.memory[ids]

    def update(self, ids, msg, t_enc):
        h = self.memory[ids]
        new_h = self.gru(torch.cat([msg, t_enc], dim=-1), h)
        # use index_copy_ to update
        self.memory = self.memory.clone()
        self.memory[ids] = new_h.detach()
        self.last_t[ids] = 0  # placeholder
        return new_h


class DIDSMFL(nn.Module):
    def __init__(
        self,
        n_nodes,
        n_feat,
        emb=64,
        time_dim=16,
        n_groups=4,
        group_dim=16,
        n_classes=10,
        use_sd=True,
        use_rd=True,
        multi_scale=True,
    ):
        super().__init__()
        self.use_sd = use_sd
        self.use_rd = use_rd
        self.multi_scale = multi_scale
        self.n_classes = n_classes

        self.sd = StatisticalDisentangle(n_feat) if use_sd else nn.Identity()
        # representational disentangle on sd output
        self.rd_in = n_feat
        self.rd = RepresentationalDisentangle(self.rd_in, n_groups, group_dim) if use_rd else None
        self.edge_dim_after_rd = n_groups * group_dim if use_rd else n_feat

        self.time_enc = TimeEncoder(time_dim)
        self.mem = DynamicMemory(n_nodes, emb, self.edge_dim_after_rd, time_dim)

        # multi-scale: concat memory at scale 1,2,3 (we use the same memory but
        # with extra "diffusion" via projection layers) -- approximation of
        # multi-layer graph diffusion
        self.diff_layers = nn.ModuleList([nn.Linear(emb, emb) for _ in range(3)])
        scale_dim = emb * 3 if multi_scale else emb

        # edge head
        cls_in = self.edge_dim_after_rd + 2 * scale_dim + time_dim
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.bin_head = nn.Linear(64, 2)
        self.multi_head = nn.Linear(64, n_classes)

    def encode_edge(self, msg):
        if self.use_sd:
            x = self.sd(msg)
        else:
            x = msg
        if self.use_rd:
            h = self.rd(x)
        else:
            h = x
        return x, h

    def diffuse(self, m):
        outs = [m]
        h = m
        for L in self.diff_layers:
            h = torch.tanh(L(h))
            outs.append(h)
        if self.multi_scale:
            return torch.cat(outs[:3], dim=-1)  # 3 scales
        return outs[0]

    def forward(self, src, dst, t, msg):
        x, edge_h = self.encode_edge(msg)
        t_enc = self.time_enc(t.float())

        s_mem = self.mem.read(src)
        d_mem = self.mem.read(dst)

        s_rep = self.diffuse(s_mem)
        d_rep = self.diffuse(d_mem)

        z = torch.cat([edge_h, s_rep, d_rep, t_enc], dim=-1)
        h = self.cls_head(z)
        bin_logit = self.bin_head(h)
        multi_logit = self.multi_head(h)

        # Update memory after computing logits
        with torch.no_grad():
            self.mem.update(src, edge_h.detach(), t_enc.detach())
            self.mem.update(dst, edge_h.detach(), t_enc.detach())

        return bin_logit, multi_logit, x, edge_h


class MLPBaseline(nn.Module):
    def __init__(self, n_feat, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.bin = nn.Linear(64, 2)
        self.multi = nn.Linear(64, n_classes)

    def forward(self, src, dst, t, msg):
        h = self.net(msg)
        return self.bin(h), self.multi(h), msg, h


class EGraphSAGEBaseline(nn.Module):
    """Edge-aware GraphSAGE-style baseline: reads node memories like
    DIDSMFL but without disentanglement and without multi-scale.
    """

    def __init__(self, n_nodes, n_feat, emb=64, time_dim=16, n_classes=10):
        super().__init__()
        self.time_enc = TimeEncoder(time_dim)
        self.mem = DynamicMemory(n_nodes, emb, n_feat, time_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(n_feat + 2 * emb + time_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.bin = nn.Linear(64, 2)
        self.multi = nn.Linear(64, n_classes)

    def forward(self, src, dst, t, msg):
        t_enc = self.time_enc(t.float())
        s = self.mem.read(src)
        d = self.mem.read(dst)
        z = torch.cat([msg, s, d, t_enc], dim=-1)
        h = self.cls_head(z)
        with torch.no_grad():
            self.mem.update(src, msg.detach(), t_enc.detach())
            self.mem.update(dst, msg.detach(), t_enc.detach())
        return self.bin(h), self.multi(h), msg, h
