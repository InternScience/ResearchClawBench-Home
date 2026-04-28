"""
Model components for the unified-AR decoupled-encoding prototype.

* `UnderstandingEncoder` — tiny ViT, semantic features (SigLIP/LLaVA flavour).
* `VQTokenizer` — small VQ-VAE, encoder–quantizer–decoder (LlamaGen flavour).
* `UnifiedTransformer` — single causal Transformer trunk that consumes a
  sequence of (text-token | image-VQ-token | understanding-feature) embeddings
  and outputs both text logits and VQ-token logits.

All sized for CPU friendliness (~2 M parameters total).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tiny ViT  -----------------------------------------------------------------
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 64, patch: int = 16, in_ch: int = 3, dim: int = 192):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.n_patches = (img_size // patch) ** 2

    def forward(self, x):
        x = self.proj(x)  # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class UnderstandingEncoder(nn.Module):
    """Tiny SigLIP/CLIP-style ViT producing (B, N+1, D) features. The CLS
    token is used for image-text contrastive alignment."""

    def __init__(self, img_size=64, patch=16, dim=192, depth=4, heads=4):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, 3, dim)
        n = self.patch.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self.n_tokens = n + 1

    def forward(self, x):
        B = x.size(0)
        h = self.patch(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h  # (B, n+1, dim) – use h[:, 0] for CLS


class TextEncoder(nn.Module):
    """A tiny transformer text encoder used together with UnderstandingEncoder
    for the SigLIP-style sigmoid contrastive pre-training."""

    def __init__(self, vocab: int, dim: int = 192, depth: int = 4, heads: int = 4, max_len: int = 32):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.max_len = max_len

    def forward(self, ids):
        B, L = ids.shape
        L = min(L, self.max_len)
        ids = ids[:, :L]
        h = self.tok(ids) + self.pos[:, :L]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h.mean(dim=1)  # mean-pooled sentence embedding


# ---------------------------------------------------------------------------
# VQ-VAE tokenizer ----------------------------------------------------------
# ---------------------------------------------------------------------------

class VQEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(True),  # 64->32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),     # 32->16
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),     # 16->8
            nn.Conv2d(64, dim, 3, 1, 1),
        )
    def forward(self, x):
        return self.net(x)


class VQDecoder(nn.Module):
    def __init__(self, out_ch: int = 3, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, 64, 3, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),  # 8->16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),  # 16->32
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1), nn.Tanh(),  # 32->64
        )
    def forward(self, z):
        return self.net(z)


class VectorQuantizer(nn.Module):
    """Standard VQ with EMA-free straight-through; commitment loss only."""

    def __init__(self, num_embeddings: int = 256, dim: int = 64, beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.dim = dim
        self.beta = beta
        self.embed = nn.Embedding(num_embeddings, dim)
        self.embed.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z: (B, D, H, W)
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (BHW, D)
        d = (
            (flat ** 2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.weight.t()
            + (self.embed.weight ** 2).sum(1)
        )
        idx = d.argmin(1)
        zq = self.embed(idx).view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        commit = F.mse_loss(zq.detach(), z) + self.beta * F.mse_loss(zq, z.detach())
        zq = z + (zq - z).detach()  # straight-through
        return zq, idx.view(B, H, W), commit

    def lookup(self, idx):
        # idx: (B, H, W)
        z = self.embed(idx).permute(0, 3, 1, 2).contiguous()
        return z


class VQTokenizer(nn.Module):
    def __init__(self, num_embeddings=256, dim=64):
        super().__init__()
        self.enc = VQEncoder(3, dim)
        self.vq = VectorQuantizer(num_embeddings, dim)
        self.dec = VQDecoder(3, dim)
        self.K = num_embeddings
        self.tok_grid = 8  # 64/8 = 8 -> 64 image tokens per image

    def forward(self, x):
        z = self.enc(x)
        zq, idx, commit = self.vq(z)
        recon = self.dec(zq)
        return recon, idx, commit, z

    def encode_to_indices(self, x):
        z = self.enc(x)
        _, idx, _ = self.vq(z)
        return idx  # (B, 4, 4)

    def decode_from_indices(self, idx):
        z = self.vq.lookup(idx)
        return self.dec(z)


# ---------------------------------------------------------------------------
# Unified causal Transformer (the "trunk") ----------------------------------
# ---------------------------------------------------------------------------

class CausalBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


@dataclass
class UnifiedConfig:
    text_vocab: int = 64
    vq_codebook: int = 256
    n_special: int = 9
    dim: int = 192
    depth: int = 6
    heads: int = 6
    max_len: int = 256
    n_understand_tokens: int = 17  # 16 patches + 1 CLS


class UnifiedTransformer(nn.Module):
    """Unified AR Transformer.

    The total vocabulary is laid out as:
        [0 ..             text_vocab)              -> text tokens
        [text_vocab ..    text_vocab+vq)            -> VQ image tokens
        [text_vocab+vq .. text_vocab+vq+n_special)  -> special markers
    Continuous understanding-encoder embeddings enter the trunk through a
    learned projector and DO NOT consume vocabulary slots; their corresponding
    output positions are masked out of the cross-entropy loss.
    """

    def __init__(self, cfg: UnifiedConfig, understand_dim: int = 192):
        super().__init__()
        self.cfg = cfg
        self.total_vocab = cfg.text_vocab + cfg.vq_codebook + cfg.n_special
        self.tok_emb = nn.Embedding(self.total_vocab, cfg.dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        # Projector for understanding-encoder features (LLaVA-style)
        self.und_proj = nn.Linear(understand_dim, cfg.dim)
        self.blocks = nn.ModuleList(
            [CausalBlock(cfg.dim, cfg.heads) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, self.total_vocab, bias=False)

        # Indices helpful for slicing
        self.text_range = (0, cfg.text_vocab)
        self.vq_range = (cfg.text_vocab, cfg.text_vocab + cfg.vq_codebook)
        self.spec_base = cfg.text_vocab + cfg.vq_codebook

    # special-token convenience
    def spec(self, name: str) -> int:
        names = ["pad", "bos", "eos", "boi", "eoi", "bog", "eog", "sep", "unk"]
        return self.spec_base + names.index(name)

    def vq_id(self, code: int) -> int:
        return self.cfg.text_vocab + code

    def text_id(self, t: int) -> int:
        return t

    def forward(
        self,
        token_ids: torch.Tensor,            # (B, L) ints — use -1 where embedding is supplied directly
        und_features: Optional[torch.Tensor] = None,  # (B, L, und_dim) — only used where token_ids == -1
        und_mask: Optional[torch.Tensor] = None,     # (B, L) bool (True where features apply)
    ):
        B, L = token_ids.shape
        device = token_ids.device

        safe_ids = token_ids.clamp_min(0)
        emb = self.tok_emb(safe_ids)
        if und_features is not None and und_mask is not None:
            proj = self.und_proj(und_features)
            emb = torch.where(und_mask.unsqueeze(-1), proj, emb)
        emb = emb + self.pos_emb[:, :L]

        mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        x = emb
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------------
# Total parameter count helper ----------------------------------------------
# ---------------------------------------------------------------------------

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    enc = UnderstandingEncoder()
    vq = VQTokenizer()
    cfg = UnifiedConfig()
    trunk = UnifiedTransformer(cfg)
    print(
        "params: enc=", count_params(enc),
        " vq=", count_params(vq),
        " trunk=", count_params(trunk),
        " total=", count_params(enc) + count_params(vq) + count_params(trunk),
    )
    x = torch.randn(2, 3, 64, 64)
    feat = enc(x)
    print("understanding feat:", feat.shape)
    recon, idx, commit, z = vq(x)
    print("vq recon:", recon.shape, "idx:", idx.shape, "commit:", commit.item())
    ids = torch.zeros(2, 16, dtype=torch.long)
    out = trunk(ids)
    print("trunk logits:", out.shape, "vocab:", trunk.total_vocab)
