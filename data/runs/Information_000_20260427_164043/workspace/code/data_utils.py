"""
Synthetic data generator for the unified-AR prototype.

Builds an in-memory mini-corpus of (image, caption) pairs of procedurally
generated shapes / colours. We also ingest the two real workspace images
(equation.png, doge.png) with hand-written captions used for held-out
qualitative evaluation.
"""

from __future__ import annotations

import json
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE / "data"
OUT_DIR = WORKSPACE / "outputs"

IMG_SIZE = 64  # tiny so that everything fits CPU
VOCAB_SHAPES = ["circle", "square", "triangle"]
VOCAB_COLOURS = {
    "red": (220, 60, 60),
    "green": (60, 200, 90),
    "blue": (60, 120, 240),
    "yellow": (240, 220, 70),
    "purple": (160, 90, 200),
    "orange": (240, 140, 60),
}


@dataclass
class Sample:
    image: np.ndarray  # (3, H, W) uint8
    caption: str
    shape: str
    colour: str


def _draw_shape(shape: str, colour: Tuple[int, int, int], size: int = IMG_SIZE) -> np.ndarray:
    img = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(img)
    pad = 8 + random.randint(0, 4)
    bbox = [pad, pad, size - pad, size - pad]
    if shape == "circle":
        d.ellipse(bbox, fill=colour, outline=(20, 20, 20), width=2)
    elif shape == "square":
        d.rectangle(bbox, fill=colour, outline=(20, 20, 20), width=2)
    else:  # triangle
        cx = size / 2
        d.polygon(
            [(cx, pad), (size - pad, size - pad), (pad, size - pad)],
            fill=colour,
            outline=(20, 20, 20),
        )
    arr = np.array(img).astype(np.uint8).transpose(2, 0, 1)  # (3,H,W)
    return arr


def build_synthetic(n_per_combo: int = 8, seed: int = 0) -> List[Sample]:
    random.seed(seed)
    np.random.seed(seed)
    samples: List[Sample] = []
    templates = [
        "a {colour} {shape}",
        "an image of a {colour} {shape}",
        "{colour} {shape} on light background",
        "picture of a {colour} {shape}",
    ]
    for shape in VOCAB_SHAPES:
        for cname, crgb in VOCAB_COLOURS.items():
            for _ in range(n_per_combo):
                arr = _draw_shape(shape, crgb)
                cap = random.choice(templates).format(colour=cname, shape=shape)
                samples.append(Sample(arr, cap, shape, cname))
    random.shuffle(samples)
    return samples


def load_real_images() -> List[Sample]:
    """Load the two workspace images at IMG_SIZE for inference."""
    out = []
    for fn, cap, shape, colour in [
        ("equation.png", "an equation in latex notation", "equation", "monochrome"),
        ("doge.png", "swole doge vs cheems meme", "meme", "mixed"),
    ]:
        p = DATA_DIR / fn
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img).astype(np.uint8).transpose(2, 0, 1)
        out.append(Sample(arr, cap, shape, colour))
    return out


# Tiny BPE-free word tokenizer ------------------------------------------------

class WordTokenizer:
    """Whitespace tokenizer with reserved special tokens."""

    SPECIAL = ["<pad>", "<bos>", "<eos>", "<boi>", "<eoi>", "<bog>", "<eog>", "<sep>", "<unk>"]

    def __init__(self, captions: List[str]):
        words = set()
        for c in captions:
            for w in c.lower().split():
                w = w.strip(string.punctuation)
                if w:
                    words.add(w)
        self.itos = list(self.SPECIAL) + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        ids = []
        for w in text.lower().split():
            w = w.strip(string.punctuation)
            if not w:
                continue
            ids.append(self.stoi.get(w, self.stoi["<unk>"]))
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    def special(self, name: str) -> int:
        return self.stoi[name]


if __name__ == "__main__":
    s = build_synthetic()
    print(f"built {len(s)} synthetic samples; first caption: {s[0].caption}")
    rs = load_real_images()
    print(f"loaded {len(rs)} real images")
    # Save a tiny preview montage
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for ax, samp in zip(axes.flatten(), s[:12]):
        ax.imshow(samp.image.transpose(1, 2, 0))
        ax.set_title(samp.caption, fontsize=7)
        ax.axis("off")
    OUT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "synthetic_preview.png", dpi=110)
    plt.close(fig)
    print("preview saved")
