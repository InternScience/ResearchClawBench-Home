"""
Training pipeline for the unified-AR decoupled-encoding prototype.

Three training stages, all CPU-friendly:

1. **VQ tokenizer pre-training** — train VQEncoder/Quantizer/VQDecoder by
   pixel reconstruction on the synthetic shape corpus. (LlamaGen-style stage 1.)

2. **SigLIP-style alignment** — train the UnderstandingEncoder + a small
   TextEncoder with a sigmoid contrastive loss on (image, caption) pairs.
   (SigLIP / LLaVA flavour.)

3. **Unified autoregressive training** — train the UnifiedTransformer trunk
   with next-token prediction on two interleaved task formats:
     * understanding:  <BOS> <BOI> [16 understanding embeddings] <EOI>
                        <SEP> caption tokens <EOS>
     * generation:     <BOS> caption tokens <SEP> <BOG> [64 VQ ids] <EOG> <EOS>

The result is saved in `outputs/checkpoints/`.

Two model variants are trained:
  * "decoupled"  – understanding pathway uses UnderstandingEncoder features.
  * "shared"     – understanding pathway re-uses VQ encoder features
                   (Chameleon-style baseline ablation).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import (
    Sample,
    WordTokenizer,
    build_synthetic,
    load_real_images,
)
from models import (
    UnderstandingEncoder,
    TextEncoder,
    VQTokenizer,
    UnifiedConfig,
    UnifiedTransformer,
    count_params,
)

WORKSPACE = Path(__file__).resolve().parent.parent
CKPT_DIR = WORKSPACE / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = WORKSPACE / "outputs"

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float().div(127.5).sub(1.0)  # [-1,1]


def stack_images(samples: List[Sample]) -> torch.Tensor:
    return torch.stack([to_tensor(s.image) for s in samples])


# ---------------------------------------------------------------------------
# Stage 1 -- VQ tokenizer ----------------------------------------------------
# ---------------------------------------------------------------------------

def train_vq(samples: List[Sample], epochs: int = 30, lr: float = 3e-3, log_every: int = 5):
    print("[stage1] training VQ tokenizer ...")
    vq = VQTokenizer(num_embeddings=256, dim=64).to(DEVICE)
    print("  params:", count_params(vq))
    opt = torch.optim.Adam(vq.parameters(), lr=lr)
    images = stack_images(samples).to(DEVICE)

    losses = []
    bs = 32
    for ep in range(epochs):
        perm = torch.randperm(images.size(0))
        ep_loss = 0.0
        n = 0
        for i in range(0, images.size(0), bs):
            xb = images[perm[i : i + bs]]
            recon, idx, commit, _ = vq(xb)
            recon_loss = F.mse_loss(recon, xb)
            loss = recon_loss + commit
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        losses.append(ep_loss / n)
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"  epoch {ep+1:>2}/{epochs}  recon+commit={losses[-1]:.4f}")

    torch.save(vq.state_dict(), CKPT_DIR / "vq.pt")
    json.dump(losses, open(LOG_DIR / "vq_losses.json", "w"))
    return vq


# ---------------------------------------------------------------------------
# Stage 2 -- SigLIP-style image-text alignment -------------------------------
# ---------------------------------------------------------------------------

def train_siglip(samples: List[Sample], tokenizer: WordTokenizer,
                 epochs: int = 30, lr: float = 3e-3, log_every: int = 5):
    print("[stage2] training SigLIP-style alignment encoder ...")
    img_enc = UnderstandingEncoder(dim=192, depth=4, heads=4).to(DEVICE)
    txt_enc = TextEncoder(vocab=len(tokenizer), dim=192, depth=4, heads=4).to(DEVICE)
    img_proj = nn.Linear(192, 128).to(DEVICE)
    txt_proj = nn.Linear(192, 128).to(DEVICE)
    bias = nn.Parameter(torch.tensor(-10.0))
    scale = nn.Parameter(torch.tensor(10.0))

    params = list(img_enc.parameters()) + list(txt_enc.parameters())
    params += list(img_proj.parameters()) + list(txt_proj.parameters())
    params += [bias, scale]
    opt = torch.optim.Adam(params, lr=lr)

    images = stack_images(samples).to(DEVICE)
    # Tokenise captions to fixed length
    cap_ids = []
    max_len = 8
    pad = tokenizer.special("<pad>")
    for s in samples:
        ids = tokenizer.encode(s.caption)[:max_len]
        ids = ids + [pad] * (max_len - len(ids))
        cap_ids.append(ids)
    cap_ids = torch.tensor(cap_ids, dtype=torch.long, device=DEVICE)

    losses = []
    bs = 32
    for ep in range(epochs):
        perm = torch.randperm(images.size(0))
        ep_loss = 0.0
        n = 0
        for i in range(0, images.size(0), bs):
            ix = perm[i : i + bs]
            xb = images[ix]
            tb = cap_ids[ix]
            B = xb.size(0)
            ie = img_enc(xb)[:, 0]
            te = txt_enc(tb)
            ie = F.normalize(img_proj(ie), dim=-1)
            te = F.normalize(txt_proj(te), dim=-1)
            logits = ie @ te.t() * scale + bias  # (B, B)
            target = torch.eye(B, device=DEVICE) * 2 - 1  # +1 for matched, -1 else
            loss = -F.logsigmoid(target * logits).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * B
            n += B
        losses.append(ep_loss / n)
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"  epoch {ep+1:>2}/{epochs}  sigloss={losses[-1]:.4f}")

    torch.save(
        {
            "img_enc": img_enc.state_dict(),
            "txt_enc": txt_enc.state_dict(),
            "img_proj": img_proj.state_dict(),
            "txt_proj": txt_proj.state_dict(),
            "bias": bias.detach().cpu(),
            "scale": scale.detach().cpu(),
        },
        CKPT_DIR / "siglip.pt",
    )
    json.dump(losses, open(LOG_DIR / "siglip_losses.json", "w"))
    return img_enc, txt_enc, img_proj, txt_proj


# ---------------------------------------------------------------------------
# Stage 3 -- Unified Transformer training ----------------------------------
# ---------------------------------------------------------------------------

def build_understanding_seq(samples_idx, images, und_features, cap_ids, trunk: UnifiedTransformer):
    """Construct (token_ids, und_features_pad, und_mask, target_ids).

    Sequence layout per example:
      <BOS> <BOI>  [n_und feature-slots]  <EOI> <SEP> [caption ids] <EOS>

    The understanding feature slots use token_id = -1 plus a True entry in
    the und_mask; their *target* is set to PAD so they do not contribute to
    the loss (we only want the model to PREDICT caption tokens conditioned
    on the visual features).
    """
    n_und = und_features.size(1)
    cap_len = cap_ids.size(1)
    L = 1 + 1 + n_und + 1 + 1 + cap_len + 1  # bos boi und eoi sep cap eos
    B = und_features.size(0)
    ids = torch.full((B, L), trunk.spec("pad"), dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    feats = torch.zeros(B, L, und_features.size(-1))
    pos = 0
    ids[:, pos] = trunk.spec("bos"); pos += 1
    ids[:, pos] = trunk.spec("boi"); pos += 1
    ids[:, pos : pos + n_und] = -1
    feats[:, pos : pos + n_und] = und_features
    mask[:, pos : pos + n_und] = True
    pos += n_und
    ids[:, pos] = trunk.spec("eoi"); pos += 1
    ids[:, pos] = trunk.spec("sep"); pos += 1
    ids[:, pos : pos + cap_len] = cap_ids
    pos += cap_len
    ids[:, pos] = trunk.spec("eos"); pos += 1
    return ids, feats, mask


def build_generation_seq(cap_ids, vq_indices, trunk: UnifiedTransformer):
    """<BOS> [caption] <SEP> <BOG> [vq tokens] <EOG> <EOS>"""
    B, cap_len = cap_ids.shape
    n_img = vq_indices.size(1)
    L = 1 + cap_len + 1 + 1 + n_img + 1 + 1
    ids = torch.full((B, L), trunk.spec("pad"), dtype=torch.long)
    pos = 0
    ids[:, pos] = trunk.spec("bos"); pos += 1
    ids[:, pos : pos + cap_len] = cap_ids; pos += cap_len
    ids[:, pos] = trunk.spec("sep"); pos += 1
    ids[:, pos] = trunk.spec("bog"); pos += 1
    # offset VQ indices into the unified vocab
    ids[:, pos : pos + n_img] = vq_indices + trunk.cfg.text_vocab
    pos += n_img
    ids[:, pos] = trunk.spec("eog"); pos += 1
    ids[:, pos] = trunk.spec("eos"); pos += 1
    return ids


def train_unified(
    samples: List[Sample],
    tokenizer: WordTokenizer,
    img_enc: UnderstandingEncoder,
    vq: VQTokenizer,
    cfg: UnifiedConfig,
    variant: str,
    epochs: int = 60,
    lr: float = 3e-3,
):
    print(f"[stage3-{variant}] training unified Transformer ...")
    trunk = UnifiedTransformer(cfg, understand_dim=192).to(DEVICE)
    print("  params:", count_params(trunk))
    opt = torch.optim.Adam(trunk.parameters(), lr=lr)

    images = stack_images(samples).to(DEVICE)
    cap_max = 6
    pad = tokenizer.special("<pad>")
    cap_ids = []
    for s in samples:
        ids = tokenizer.encode(s.caption)[:cap_max]
        ids = ids + [pad] * (cap_max - len(ids))
        cap_ids.append(ids)
    cap_ids = torch.tensor(cap_ids, dtype=torch.long, device=DEVICE)

    # Pre-compute encoder/quantizer features once (frozen front-ends, like LLaVA).
    img_enc.eval(); vq.eval()
    with torch.no_grad():
        if variant == "decoupled":
            und_feats_all = img_enc(images)  # (N, 17, 192)
        elif variant == "shared":
            # Use VQ encoder continuous features projected up to 192 (Chameleon
            # baseline: same encoder for both tasks).
            z = vq.enc(images)  # (N, 64, 8, 8)
            z = z.flatten(2).transpose(1, 2)  # (N, 64, 64)
            # Pad / repeat to dim 192
            und_feats_all = z.repeat(1, 1, 3)[..., :192]  # (N, 64, 192)
            # Sub-sample to 17 tokens to match decoupled budget
            und_feats_all = und_feats_all[:, :17]
        else:
            raise ValueError(variant)
        vq_idx_all = vq.encode_to_indices(images)  # (N, 8, 8)
        vq_idx_all = vq_idx_all.flatten(1)  # (N, 64)

    losses = []
    bs = 16
    for ep in range(epochs):
        perm = torch.randperm(images.size(0))
        ep_loss = 0.0
        n = 0
        for i in range(0, images.size(0), bs):
            ix = perm[i : i + bs]
            B = ix.numel()
            cap_b = cap_ids[ix]

            # Understanding branch
            und_b = und_feats_all[ix]
            u_ids, u_feats, u_mask = build_understanding_seq(ix, images, und_b, cap_b, trunk)
            u_ids, u_feats, u_mask = u_ids.to(DEVICE), u_feats.to(DEVICE), u_mask.to(DEVICE)

            # Generation branch
            vq_b = vq_idx_all[ix]
            g_ids = build_generation_seq(cap_b, vq_b, trunk).to(DEVICE)

            # Pad to common length
            L = max(u_ids.size(1), g_ids.size(1))
            def pad_to(t, L, val):
                if t.size(1) == L: return t
                p = torch.full((t.size(0), L - t.size(1)), val,
                               dtype=t.dtype, device=t.device)
                return torch.cat([t, p], dim=1)

            u_ids_p = pad_to(u_ids, L, trunk.spec("pad"))
            g_ids_p = pad_to(g_ids, L, trunk.spec("pad"))
            u_mask_p = torch.cat(
                [u_mask, torch.zeros(u_mask.size(0), L - u_mask.size(1),
                                     dtype=torch.bool, device=DEVICE)],
                dim=1,
            ) if u_mask.size(1) < L else u_mask
            u_feats_p = torch.cat(
                [u_feats, torch.zeros(u_feats.size(0), L - u_feats.size(1),
                                      u_feats.size(2), device=DEVICE)],
                dim=1,
            ) if u_feats.size(1) < L else u_feats

            # Stack U/G batches
            ids_all = torch.cat([u_ids_p, g_ids_p], dim=0)
            feats_all = torch.cat(
                [u_feats_p, torch.zeros(g_ids_p.size(0), L,
                                        u_feats_p.size(-1), device=DEVICE)], dim=0
            )
            mask_all = torch.cat(
                [u_mask_p,
                 torch.zeros(g_ids_p.size(0), L, dtype=torch.bool, device=DEVICE)], dim=0,
            )

            logits = trunk(ids_all, feats_all, mask_all)  # (2B, L, V)
            # Next-token prediction targets
            tgt = ids_all.clone()
            # Replace -1 with pad (those positions will be masked out anyway via input mask)
            tgt[tgt == -1] = trunk.spec("pad")
            shift_logits = logits[:, :-1].contiguous()
            shift_tgt = tgt[:, 1:].contiguous()
            # Mask: ignore predictions whose CURRENT input was an understanding feature
            input_was_feat = mask_all[:, :-1]
            ce_mask = (~input_was_feat) & (shift_tgt != trunk.spec("pad"))
            loss = F.cross_entropy(
                shift_logits.reshape(-1, logits.size(-1)),
                shift_tgt.reshape(-1),
                reduction="none",
            )
            loss = (loss * ce_mask.reshape(-1).float()).sum() / ce_mask.sum().clamp(min=1)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * B
            n += B
        losses.append(ep_loss / n)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:>3}/{epochs}  ce={losses[-1]:.4f}")

    torch.save(trunk.state_dict(), CKPT_DIR / f"trunk_{variant}.pt")
    json.dump(losses, open(LOG_DIR / f"trunk_{variant}_losses.json", "w"))
    return trunk


def main():
    samples = build_synthetic(n_per_combo=10, seed=0)
    print(f"corpus: {len(samples)} samples")
    real = load_real_images()
    print(f"real images: {len(real)}")

    captions = [s.caption for s in samples]
    tokenizer = WordTokenizer(captions)
    json.dump(tokenizer.itos, open(LOG_DIR / "vocab.json", "w"))
    print(f"vocab size: {len(tokenizer)}")

    cfg = UnifiedConfig(
        text_vocab=len(tokenizer),
        vq_codebook=256,
        n_special=9,
        dim=192,
        depth=6,
        heads=6,
        max_len=128,
    )
    json.dump(asdict(cfg), open(LOG_DIR / "unified_cfg.json", "w"))

    t0 = time.time()
    vq = train_vq(samples, epochs=25)
    print(f"  stage1 took {time.time()-t0:.1f}s")

    t0 = time.time()
    img_enc, txt_enc, img_proj, txt_proj = train_siglip(
        samples, tokenizer, epochs=25
    )
    print(f"  stage2 took {time.time()-t0:.1f}s")

    t0 = time.time()
    trunk_d = train_unified(samples, tokenizer, img_enc, vq, cfg,
                            variant="decoupled", epochs=40)
    print(f"  stage3 decoupled took {time.time()-t0:.1f}s")

    t0 = time.time()
    trunk_s = train_unified(samples, tokenizer, img_enc, vq, cfg,
                            variant="shared", epochs=40)
    print(f"  stage3 shared took {time.time()-t0:.1f}s")

    print("All training done.")


if __name__ == "__main__":
    main()
