"""
ViCrop reproduction: training-free task-guided visual cropping for MLLMs.

We use OpenCLIP ViT-B/16 (the same family of frozen visual encoder used inside
LLaVA-1.5 / InstructBLIP) and:

1. Compute a task-conditioned relevancy map over image patches via
   Chefer-style attention x gradient on each transformer block, w.r.t. the
   CLIP image-text similarity for the question/answer text. We also export
   an attention-rollout map (gradient-free) as a sanity baseline.
2. From the relevancy map we extract a single ROI bounding box (threshold
   + largest connected component + margin), zoom in, and re-encode at the
   native 224x224 resolution.
3. Final answer = weighted ensemble of softmaxed CLIP similarities for
   (global view, cropped view).

Baselines: no-crop, center-crop, random-crop, uniform 2x2 tiling (Monkey-style).
"""

from __future__ import annotations
import os
os.environ.setdefault('HF_HOME','/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/hfcache')
os.environ.setdefault('TORCH_HOME','/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119/torchcache')

import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Dict
from scipy.ndimage import label as cc_label, gaussian_filter
import open_clip

WORKSPACE = '/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260427_165119'
MODEL_TAG = 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K'
DEVICE = 'cpu'

# ---------------------------------------------------------------------------
# Patched MultiheadAttention to expose per-block attention weights & gradients
# ---------------------------------------------------------------------------

class CapturedAttn(nn.Module):
    """Wraps nn.MultiheadAttention so we always get per-head attention weights
    and store them so backward can populate gradients."""

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.attn_weights: torch.Tensor | None = None  # (B, H, N, N)

    def forward(self, q, k, v, need_weights=False, attn_mask=None,
                key_padding_mask=None, is_causal=False, **kwargs):
        # Manual MHA forward so the softmax attention tensor is a graph node
        # whose .grad can be retained for Chefer-style relevancy.
        # open_clip uses batch_first=True so q,k,v are (B, L, E).
        m = self.mha
        embed_dim = m.embed_dim
        num_heads = m.num_heads
        head_dim = embed_dim // num_heads
        if m._qkv_same_embed_dim:
            qkv = F.linear(q, m.in_proj_weight, m.in_proj_bias)
            qp, kp, vp = qkv.chunk(3, dim=-1)
        else:
            b_q = m.in_proj_bias[:embed_dim] if m.in_proj_bias is not None else None
            b_k = m.in_proj_bias[embed_dim:2*embed_dim] if m.in_proj_bias is not None else None
            b_v = m.in_proj_bias[2*embed_dim:] if m.in_proj_bias is not None else None
            qp = F.linear(q, m.q_proj_weight, b_q)
            kp = F.linear(k, m.k_proj_weight, b_k)
            vp = F.linear(v, m.v_proj_weight, b_v)
        # (B, L, E) -> (B, H, L, D)
        B, L, _ = qp.shape
        Sk = kp.shape[1]
        qh = qp.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)
        kh = kp.reshape(B, Sk, num_heads, head_dim).permute(0, 2, 1, 3)
        vh = vp.reshape(B, Sk, num_heads, head_dim).permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(head_dim)
        scores = (qh @ kh.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = scores.softmax(dim=-1)  # (B, H, L, Sk)
        if attn.requires_grad:
            attn.retain_grad()
        self.attn_weights = attn
        out = attn @ vh  # (B, H, L, D)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, L, embed_dim)
        out = F.linear(out, m.out_proj.weight, m.out_proj.bias)
        return out, attn


def patch_clip_attention(model: nn.Module):
    """Replace each block.attn with CapturedAttn so we can read attn weights."""
    blocks = model.visual.transformer.resblocks
    captured = []
    for b in blocks:
        c = CapturedAttn(b.attn)
        b.attn = c
        captured.append(c)
    return captured


# ---------------------------------------------------------------------------
# Relevancy: Chefer-style attention x gradient (works for any cls token).
# ---------------------------------------------------------------------------

def chefer_relevancy(captured: List[CapturedAttn], grid: int = 14) -> np.ndarray:
    """Aggregate (attn * grad).clamp_min(0) averaged over heads, then
    propagate via residual rollout. Returns a (grid, grid) map for CLS->patch."""
    eye = None
    R = None
    for cap in captured:
        attn = cap.attn_weights  # (1, H, N, N)
        grad = cap.attn_weights.grad
        if grad is None:
            continue
        cam = (attn * grad).clamp_min(0).mean(dim=1)  # (1, N, N) -> avg over heads
        cam = cam[0]  # (N, N)
        N = cam.shape[0]
        if eye is None:
            eye = torch.eye(N, device=cam.device)
            R = eye.clone()
        # Add residual identity
        cam = cam + eye
        # Row-normalise
        cam = cam / cam.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        R = cam @ R
    if R is None:
        return np.zeros((grid, grid), dtype=np.float32)
    cls_to_patch = R[0, 1:]  # drop CLS
    cls_to_patch = cls_to_patch[: grid * grid]
    m = cls_to_patch.detach().cpu().numpy().reshape(grid, grid)
    return m


def attention_rollout(captured: List[CapturedAttn], grid: int = 14) -> np.ndarray:
    """Gradient-free attention rollout: average heads, add residual,
    multiply across blocks."""
    R = None
    eye = None
    for cap in captured:
        if cap.attn_weights is None:
            continue
        a = cap.attn_weights[0].mean(dim=0)  # (N,N)
        N = a.shape[0]
        if eye is None:
            eye = torch.eye(N, device=a.device)
            R = eye.clone()
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        R = a @ R
    if R is None:
        return np.zeros((grid, grid), dtype=np.float32)
    cls_to_patch = R[0, 1:][: grid * grid]
    return cls_to_patch.detach().cpu().numpy().reshape(grid, grid)


# ---------------------------------------------------------------------------
# Relevancy map -> ROI bounding box
# ---------------------------------------------------------------------------

def relevancy_to_bbox(rel_map: np.ndarray, image_size: Tuple[int, int],
                      threshold_pct: float = 0.7, margin: float = 0.10
                      ) -> Tuple[int, int, int, int]:
    """rel_map: (grid, grid). image_size: (W, H). Returns (x0, y0, x1, y1) in pixel coords."""
    W, H = image_size
    g = rel_map.shape[0]
    # Smooth a bit
    m = gaussian_filter(rel_map, sigma=0.6)
    if m.max() <= 0:
        return 0, 0, W, H
    thr = np.quantile(m, threshold_pct)
    binm = m >= thr
    if not binm.any():
        # fall back to argmax
        i, j = np.unravel_index(np.argmax(m), m.shape)
        binm = np.zeros_like(m, dtype=bool)
        binm[i, j] = True
    lab, n = cc_label(binm)
    if n == 0:
        return 0, 0, W, H
    # Largest CC by sum of relevancy
    best, best_score = 1, -1.0
    for k in range(1, n + 1):
        score = m[lab == k].sum()
        if score > best_score:
            best_score = score
            best = k
    ys, xs = np.where(lab == best)
    y0g, y1g = ys.min(), ys.max() + 1
    x0g, x1g = xs.min(), xs.max() + 1
    # Convert grid -> pixel coords
    px = W / g
    py = H / g
    x0 = int(max(0, math.floor(x0g * px - margin * W)))
    x1 = int(min(W, math.ceil(x1g * px + margin * W)))
    y0 = int(max(0, math.floor(y0g * py - margin * H)))
    y1 = int(min(H, math.ceil(y1g * py + margin * H)))
    # Ensure non-degenerate
    if x1 - x0 < 32: x1 = min(W, x0 + 32)
    if y1 - y0 < 32: y1 = min(H, y0 + 32)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class VicropModel:
    def __init__(self, tag: str = MODEL_TAG):
        self.model, _, self.prepro = open_clip.create_model_and_transforms(tag)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(tag)
        self.captured = patch_clip_attention(self.model)
        self.grid = self.model.visual.grid_size[0]

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        toks = self.tok(prompts)
        feats = self.model.encode_text(toks)
        feats = F.normalize(feats, dim=-1)
        return feats

    def encode_image(self, pil_image: Image.Image, with_grad: bool = False) -> torch.Tensor:
        x = self.prepro(pil_image).unsqueeze(0)
        if with_grad:
            x.requires_grad_(False)
            feats = self.model.encode_image(x)
        else:
            with torch.no_grad():
                feats = self.model.encode_image(x)
        feats = F.normalize(feats, dim=-1)
        return feats

    def relevancy(self, pil_image: Image.Image, text: str) -> Dict[str, np.ndarray]:
        """Run a forward+backward to obtain a Chefer relevancy map for
        the similarity between the image and the given text."""
        # Reset captured tensors
        for c in self.captured:
            c.attn_weights = None
        x = self.prepro(pil_image).unsqueeze(0)
        x.requires_grad_(False)
        # Get text feature (no grad needed)
        with torch.no_grad():
            txt_feat = self.encode_text([text])  # (1, D)
        # Forward visual with grad
        self.model.zero_grad()
        img_feat = self.model.encode_image(x)  # (1, D)
        img_feat = F.normalize(img_feat, dim=-1)
        # Score = cosine similarity
        score = (img_feat * txt_feat).sum()
        score.backward()
        rel = chefer_relevancy(self.captured, grid=self.grid)
        roll = attention_rollout(self.captured, grid=self.grid)
        return {'chefer': rel, 'rollout': roll, 'score': float(score.detach())}

    @torch.no_grad()
    def score_options(self, pil_image: Image.Image, options_text: List[str]) -> np.ndarray:
        feat = self.encode_image(pil_image)  # (1,D)
        tfeat = self.encode_text(options_text)  # (K,D)
        sims = (feat @ tfeat.T)[0].cpu().numpy()  # (K,)
        return sims


# ---------------------------------------------------------------------------
# Cropping baselines
# ---------------------------------------------------------------------------

def center_crop_box(W: int, H: int, frac: float = 0.5) -> Tuple[int, int, int, int]:
    cw, ch = int(W * frac), int(H * frac)
    x0 = (W - cw) // 2; y0 = (H - ch) // 2
    return x0, y0, x0 + cw, y0 + ch


def random_crop_box(W: int, H: int, frac: float = 0.5, rng=None) -> Tuple[int, int, int, int]:
    if rng is None: rng = np.random.default_rng(0)
    cw, ch = int(W * frac), int(H * frac)
    x0 = int(rng.integers(0, W - cw + 1))
    y0 = int(rng.integers(0, H - ch + 1))
    return x0, y0, x0 + cw, y0 + ch


def uniform_tile_boxes(W: int, H: int, n: int = 2) -> List[Tuple[int, int, int, int]]:
    boxes = []
    for i in range(n):
        for j in range(n):
            x0 = j * W // n
            y0 = i * H // n
            x1 = (j + 1) * W // n
            y1 = (i + 1) * H // n
            boxes.append((x0, y0, x1, y1))
    return boxes


def crop(pil: Image.Image, box) -> Image.Image:
    return pil.crop(box)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, T: float = 0.01) -> np.ndarray:
    z = x / max(T, 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def vicrop_predict(vm: VicropModel, pil: Image.Image, query_text: str,
                    options: List[str], threshold_pct: float = 0.7,
                    margin: float = 0.1, w_crop: float = 0.5
                    ) -> Dict:
    rel = vm.relevancy(pil, query_text)
    bbox = relevancy_to_bbox(rel['chefer'], pil.size,
                             threshold_pct=threshold_pct, margin=margin)
    sims_global = vm.score_options(pil, options)
    sims_crop = vm.score_options(pil.crop(bbox), options)
    p_global = softmax(sims_global)
    p_crop = softmax(sims_crop)
    p_ens = (1 - w_crop) * p_global + w_crop * p_crop
    return {
        'rel_chefer': rel['chefer'],
        'rel_rollout': rel['rollout'],
        'bbox': bbox,
        'sims_global': sims_global.tolist(),
        'sims_crop': sims_crop.tolist(),
        'p_global': p_global.tolist(),
        'p_crop': p_crop.tolist(),
        'p_ens': p_ens.tolist(),
        'pred_global': int(np.argmax(p_global)),
        'pred_crop': int(np.argmax(p_crop)),
        'pred_ens': int(np.argmax(p_ens)),
    }


def baseline_predict(vm: VicropModel, pil: Image.Image, options: List[str],
                     mode: str = 'nocrop', frac: float = 0.5, seed: int = 0
                     ) -> Dict:
    W, H = pil.size
    if mode == 'nocrop':
        sims = vm.score_options(pil, options)
        return {'sims': sims.tolist(), 'pred': int(np.argmax(sims))}
    if mode == 'center':
        b = center_crop_box(W, H, frac)
        sims_c = vm.score_options(pil.crop(b), options)
        sims_g = vm.score_options(pil, options)
        p = 0.5 * softmax(sims_g) + 0.5 * softmax(sims_c)
        return {'bbox': b, 'sims_global': sims_g.tolist(), 'sims_crop': sims_c.tolist(),
                'p_ens': p.tolist(), 'pred': int(np.argmax(p))}
    if mode == 'random':
        rng = np.random.default_rng(seed)
        b = random_crop_box(W, H, frac, rng)
        sims_c = vm.score_options(pil.crop(b), options)
        sims_g = vm.score_options(pil, options)
        p = 0.5 * softmax(sims_g) + 0.5 * softmax(sims_c)
        return {'bbox': b, 'sims_global': sims_g.tolist(), 'sims_crop': sims_c.tolist(),
                'p_ens': p.tolist(), 'pred': int(np.argmax(p))}
    if mode == 'tile':
        boxes = uniform_tile_boxes(W, H, n=2)
        ps = [softmax(vm.score_options(pil.crop(b), options)) for b in boxes]
        sims_g = vm.score_options(pil, options)
        p = 0.5 * softmax(sims_g) + 0.5 * np.mean(ps, axis=0)
        return {'boxes': boxes, 'p_ens': p.tolist(), 'pred': int(np.argmax(p))}
    raise ValueError(mode)


if __name__ == '__main__':
    print('Loading CLIP...', flush=True)
    vm = VicropModel()
    print('Loaded.', flush=True)
    pil = Image.open(os.path.join(WORKSPACE, 'data/demo_imgs/demo1.png')).convert('RGB')
    out = vicrop_predict(vm, pil,
                         'a yellow taxi with the license plate visible',
                         ['a yellow taxi', 'a silver Chevrolet sedan',
                          'a parked motorcycle', 'a red double-decker bus'])
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, np.ndarray)},
                     indent=2))
