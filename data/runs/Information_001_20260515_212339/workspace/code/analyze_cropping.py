#!/usr/bin/env python3
"""
Training-free fine-grained perception via task-guided cropping on demo images.
- Loads demo1.png and demo2.png
- Computes gradient-magnitude saliency map
- Extracts top-ROI, resizes, blends back into global context
- Saves processed images + metrics
- Produces comparison plots (PNG only)
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "demo_imgs"
OUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "report" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def compute_saliency(gray):
    """Simple gradient magnitude saliency (edge energy)."""
    sx = sobel(gray, axis=0)
    sy = sobel(gray, axis=1)
    sal = np.hypot(sx, sy)
    sal = gaussian_filter(sal, sigma=2)
    return sal

def find_roi(sal, top_frac=0.15):
    """Find bounding box of top energy region."""
    thresh = np.percentile(sal, 100 * (1 - top_frac))
    mask = sal >= thresh
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (rmin, rmax, cmin, cmax)

def process_image(img_rgb, name):
    gray = np.mean(img_rgb, axis=2).astype(np.float32)
    sal = compute_saliency(gray)
    roi = find_roi(sal)
    if roi is None:
        roi = (0, img_rgb.shape[0]-1, 0, img_rgb.shape[1]-1)
    rmin, rmax, cmin, cmax = roi
    crop = img_rgb[rmin:rmax+1, cmin:cmax+1]
    # Resize crop to match original aspect ratio (simple upsample)
    crop_pil = Image.fromarray(crop)
    crop_resized = crop_pil.resize((img_rgb.shape[1], img_rgb.shape[0]), Image.BILINEAR)
    crop_arr = np.array(crop_resized)
    # Blend: 70% global + 30% local detail
    blended = (0.7 * img_rgb + 0.3 * crop_arr).astype(np.uint8)
    # Metrics
    orig_entropy = -np.sum((gray/255.0) * np.log2(gray/255.0 + 1e-12))
    crop_entropy = -np.sum((np.mean(crop,2)/255.0) * np.log2(np.mean(crop,2)/255.0 + 1e-12))
    edge_density = np.mean(sal > np.percentile(sal, 90))
    metrics = {
        "orig_entropy": float(orig_entropy),
        "crop_entropy": float(crop_entropy),
        "edge_density": float(edge_density),
        "roi": [int(rmin), int(rmax), int(cmin), int(cmax)]
    }
    # Save images
    Image.fromarray(blended).save(OUT_DIR / f"{name}_blended.png")
    Image.fromarray(crop).save(OUT_DIR / f"{name}_crop.png")
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(crop); axes[1].set_title("Cropped ROI"); axes[1].axis("off")
    axes[2].imshow(blended); axes[2].set_title("Blended (Local+Global)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{name}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    return metrics

def main():
    for fname in ["demo1.png", "demo2.png"]:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"Missing {path}")
            continue
        img = load_image(path)
        name = Path(fname).stem
        metrics = process_image(img, name)
        print(f"{name}: {metrics}")
        # Save metrics
        np.save(OUT_DIR / f"{name}_metrics.npy", metrics)

if __name__ == "__main__":
    main()