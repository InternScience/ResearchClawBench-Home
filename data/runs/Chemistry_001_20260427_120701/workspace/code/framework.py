"""
Unified AlphaFold-3-style diffusion framework for biomolecular complexes.

Minimally faithful AF3 architecture at toy / CPU scale.

Architecture:
    Tokenizer (residues / NA-bases / ligand-atoms)
        -> single-token embeddings (dim_s) + pair embeddings (dim_z)
        -> Pairformer-lite trunk (triangle multiplicative + pair self-attention)
        -> Diffusion module: epsilon-prediction over Cartesian coordinates,
           conditioned on (s, z, t).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"          # 20 AAs + unknown
NA_VOCAB = "ACGUTN"                          # 4 RNA + 1 DNA-only + unknown
ATOM_ELEMENTS = ["C","N","O","S","P","H","F","Cl","Br","I","B","Se","X"]

TOKEN_TYPE_PROTEIN = 0
TOKEN_TYPE_NUCLEIC = 1
TOKEN_TYPE_LIGAND  = 2

class Tokenizer:
    def __init__(self):
        self.aa2id = {c: i for i, c in enumerate(AA_VOCAB)}
        self.na2id = {c: i for i, c in enumerate(NA_VOCAB)}
        self.elem2id = {e: i for i, e in enumerate(ATOM_ELEMENTS)}

    def encode(self, protein_seq: str = "",
               nucleic_seq: str = "",
               ligand_elements: list | None = None,
               ligand_bonds: list | None = None):
        types, ids = [], []
        for c in protein_seq:
            types.append(TOKEN_TYPE_PROTEIN)
            ids.append(self.aa2id.get(c, self.aa2id["X"]))
        for c in nucleic_seq:
            types.append(TOKEN_TYPE_NUCLEIC)
            ids.append(self.na2id.get(c, self.na2id["N"]))
        n_off = len(types)
        if ligand_elements:
            for e in ligand_elements:
                types.append(TOKEN_TYPE_LIGAND)
                ids.append(self.elem2id.get(e, self.elem2id["X"]))
        types_t = torch.tensor(types, dtype=torch.long)
        ids_t   = torch.tensor(ids,   dtype=torch.long)
        if ligand_bonds:
            b = torch.tensor(ligand_bonds, dtype=torch.long).T + n_off
        else:
            b = torch.zeros((2,0), dtype=torch.long)
        return types_t, ids_t, b


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class TokenEmbedder(nn.Module):
    def __init__(self, dim_s: int = 48):
        super().__init__()
        self.aa_emb = nn.Embedding(len(AA_VOCAB), dim_s)
        self.na_emb = nn.Embedding(len(NA_VOCAB), dim_s)
        self.atom_emb = nn.Embedding(len(ATOM_ELEMENTS), dim_s)
        self.type_emb = nn.Embedding(3, dim_s)
        self.dim_s = dim_s

    def forward(self, types: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(types.shape[0], self.dim_s, device=types.device)
        m = types == TOKEN_TYPE_PROTEIN
        if m.any(): out[m] = out[m] + self.aa_emb(ids[m])
        m = types == TOKEN_TYPE_NUCLEIC
        if m.any(): out[m] = out[m] + self.na_emb(ids[m])
        m = types == TOKEN_TYPE_LIGAND
        if m.any(): out[m] = out[m] + self.atom_emb(ids[m])
        out = out + self.type_emb(types)
        return out


# ---------------------------------------------------------------------------
# Pairformer-lite trunk
# ---------------------------------------------------------------------------
class TriangleMultiplicativeUpdateLite(nn.Module):
    """Triangle-multiplicative-update with low inner rank for CPU tractability."""
    def __init__(self, dim_z: int, inner: int = 8):
        super().__init__()
        self.left  = nn.Linear(dim_z, inner)
        self.right = nn.Linear(dim_z, inner)
        self.gate  = nn.Linear(dim_z, dim_z)
        self.out   = nn.Linear(inner, dim_z)
        self.norm_in  = nn.LayerNorm(dim_z)
        self.norm_out = nn.LayerNorm(inner)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_n = self.norm_in(z)
        a = self.left(z_n)            # [N,N,inner]
        b = self.right(z_n)           # [N,N,inner]
        # outgoing: t_ij = (1/N) sum_k a_ik * b_jk
        t = torch.einsum("ikd,jkd->ijd", a, b) / max(z.shape[0], 1)
        g = torch.sigmoid(self.gate(z_n))
        return self.out(self.norm_out(t)) * g


class PairAttention(nn.Module):
    """Self-attention on the single representation, biased by pair features."""
    def __init__(self, dim_s: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim_s, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_s)
        self.ff = nn.Sequential(nn.Linear(dim_s, 2*dim_s), nn.GELU(),
                                nn.Linear(2*dim_s, dim_s))
        self.norm2 = nn.LayerNorm(dim_s)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.norm(s).unsqueeze(0)
        a, _ = self.attn(x, x, x)
        s = s + a.squeeze(0)
        s = s + self.ff(self.norm2(s))
        return s


class PairformerLite(nn.Module):
    def __init__(self, dim_s: int = 48, dim_z: int = 16, n_blocks: int = 1):
        super().__init__()
        self.s2z = nn.Linear(dim_s, dim_z)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "tri":  TriangleMultiplicativeUpdateLite(dim_z),
                "attn": PairAttention(dim_s),
                "z_from_s": nn.Linear(2*dim_z, dim_z),
                "norm_z": nn.LayerNorm(dim_z),
            })
            for _ in range(n_blocks)
        ])
        self.dim_s = dim_s
        self.dim_z = dim_z

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        for blk in self.blocks:
            z = z + blk["tri"](z)
            s = blk["attn"](s)
            sz = self.s2z(s)
            op = (sz.unsqueeze(0) + sz.unsqueeze(1)) / 2.0
            z = blk["norm_z"](z + blk["z_from_s"](torch.cat([z, op], dim=-1)))
        return s, z


# ---------------------------------------------------------------------------
# Diffusion module
# ---------------------------------------------------------------------------
def cosine_alpha_bar(T: int) -> torch.Tensor:
    s = 0.008
    steps = torch.arange(T+1, dtype=torch.float32) / T
    f = torch.cos(((steps + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.clamp(min=1e-6, max=0.9999)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionModule(nn.Module):
    """Epsilon-prediction over Cartesian coords, conditioned on (s,z,t)."""
    def __init__(self, dim_s: int = 48, dim_z: int = 16):
        super().__init__()
        self.t_emb  = SinusoidalTimeEmb(dim_s)
        self.x_emb  = nn.Linear(3, dim_s)
        self.cond_norm = nn.LayerNorm(dim_z)
        self.z_to_s = nn.Linear(dim_z, dim_s)
        self.token_block = nn.Sequential(
            nn.Linear(3*dim_s, dim_s), nn.GELU(),
            nn.Linear(dim_s, dim_s), nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(dim_s, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim_s)
        self.head = nn.Sequential(
            nn.Linear(dim_s, dim_s), nn.GELU(),
            nn.Linear(dim_s, 3),
        )

    def forward(self, x_t: torch.Tensor, s: torch.Tensor,
                z: torch.Tensor, t: torch.Tensor):
        N, _ = x_t.shape
        te = self.t_emb(t).expand(N, -1)
        xe = self.x_emb(x_t)
        h  = torch.cat([xe, s, te], dim=-1)
        h  = self.token_block(h)
        h  = h + self.z_to_s(self.cond_norm(z.mean(dim=1)))
        a, _ = self.attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
        h = self.norm(h + a.squeeze(0))
        return self.head(h)


# ---------------------------------------------------------------------------
# Top-level framework
# ---------------------------------------------------------------------------
class UnifiedComplexDiffusion(nn.Module):
    def __init__(self, dim_s: int = 48, dim_z: int = 16,
                 n_trunk: int = 1, T: int = 150):
        super().__init__()
        self.tok = Tokenizer()
        self.embed = TokenEmbedder(dim_s)
        self.s2z_init = nn.Linear(2*dim_s, dim_z)
        self.trunk = PairformerLite(dim_s, dim_z, n_blocks=n_trunk)
        self.diff  = DiffusionModule(dim_s, dim_z)
        self.dim_s = dim_s; self.dim_z = dim_z; self.T = T
        ab = cosine_alpha_bar(T)
        self.register_buffer("alpha_bar", ab)

    # ------- featurisation -------------------------------------------------
    def featurize(self, protein_seq="", nucleic_seq="",
                  ligand_elements=None, ligand_bonds=None):
        types, ids, bonds = self.tok.encode(
            protein_seq, nucleic_seq, ligand_elements, ligand_bonds)
        s0 = self.embed(types, ids)                        # [N,dim_s]
        N = s0.shape[0]
        # initial pair = projection of (s_i, s_j) outer concatenation  [N,N,dim_z]
        op = torch.cat([s0.unsqueeze(0).expand(N,N,-1),
                        s0.unsqueeze(1).expand(N,N,-1)], dim=-1)
        z = self.s2z_init(op)
        if bonds.numel():
            bi, bj = bonds[0], bonds[1]
            z[bi, bj] = z[bi, bj] + 1.0
            z[bj, bi] = z[bj, bi] + 1.0
        return types, s0, z

    # ------- training loss -------------------------------------------------
    def diffusion_loss(self, x0: torch.Tensor, s: torch.Tensor, z: torch.Tensor):
        t = torch.randint(1, self.T+1, (1,), device=x0.device)
        ab = self.alpha_bar[t]
        noise = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        s_t, z_t = self.trunk(s, z)
        eps = self.diff(x_t, s_t, z_t, t.float())
        return F.mse_loss(eps, noise), eps, noise

    # ------- ancestral sampling -------------------------------------------
    @torch.no_grad()
    def sample(self, s: torch.Tensor, z: torch.Tensor, N: int,
               n_save: int = 11, scale: float = 1.0):
        s_t, z_t = self.trunk(s, z)
        x = torch.randn(N, 3, device=s.device) * scale
        traj = [x.clone()]
        save_at = set(torch.linspace(self.T, 1, n_save).long().tolist())
        for t in range(self.T, 0, -1):
            tt = torch.tensor([t], device=s.device, dtype=torch.float32)
            ab_t    = self.alpha_bar[t]
            ab_prev = self.alpha_bar[t-1]
            beta_t  = (1 - ab_t / ab_prev).clamp(min=1e-8, max=0.999)
            alpha_t = 1 - beta_t
            eps = self.diff(x, s_t, z_t, tt)
            # x0 parameterisation (stable)
            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(min=-4.0, max=4.0)
            mean = (ab_prev.sqrt() * beta_t / (1 - ab_t)) * x0_pred +                    (alpha_t.sqrt() * (1 - ab_prev) / (1 - ab_t)) * x
            if t > 1:
                noise = torch.randn_like(x)
                var = ((1 - ab_prev) / (1 - ab_t)) * beta_t
                x = mean + var.sqrt() * noise
            else:
                x = mean
            if t in save_at:
                traj.append(x.clone())
        return x, torch.stack(traj)
