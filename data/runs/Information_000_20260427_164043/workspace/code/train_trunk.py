"""Continue training from where train.py left off (unified trunk only)."""

import time
import json
from pathlib import Path

import torch

from data_utils import build_synthetic, WordTokenizer
from models import UnderstandingEncoder, VQTokenizer, UnifiedConfig
from train import train_unified, CKPT_DIR, LOG_DIR

if __name__ == "__main__":
    samples = build_synthetic(n_per_combo=10, seed=0)
    captions = [s.caption for s in samples]
    tokenizer = WordTokenizer(captions)
    cfg = UnifiedConfig(
        text_vocab=len(tokenizer),
        vq_codebook=256,
        n_special=9,
        dim=192,
        depth=6,
        heads=6,
        max_len=128,
    )

    vq = VQTokenizer(num_embeddings=256, dim=64)
    vq.load_state_dict(torch.load(CKPT_DIR / "vq.pt", weights_only=True))
    img_enc = UnderstandingEncoder(dim=192, depth=4, heads=4)
    siglip = torch.load(CKPT_DIR / "siglip.pt", weights_only=True)
    img_enc.load_state_dict(siglip["img_enc"])

    t0 = time.time()
    train_unified(samples, tokenizer, img_enc, vq, cfg, variant="decoupled", epochs=40)
    print(f"decoupled took {time.time()-t0:.1f}s")
    t0 = time.time()
    train_unified(samples, tokenizer, img_enc, vq, cfg, variant="shared", epochs=40)
    print(f"shared took {time.time()-t0:.1f}s")
