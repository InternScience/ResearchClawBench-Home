from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage import filters, measure, transform


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "demo_imgs"
RELATED_DIR = ROOT / "related_work"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


@dataclass
class ImageResult:
    image: str
    width: int
    height: int
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    roi_area_ratio: float
    global_detail_loss: float
    roi_detail_loss: float
    global_detail_retention: float
    roi_detail_retention: float
    recovered_detail_ratio: float
    entropy_global: float
    entropy_roi: float
    sharpness_global: float
    sharpness_roi: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def grayscale(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=2).astype(np.float32) / 255.0


def gradient_map(gray: np.ndarray) -> np.ndarray:
    sobel = filters.sobel(gray)
    return sobel / (sobel.max() + 1e-8)


def patch_scores(grad: np.ndarray, patch: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = grad.shape
    ph = math.ceil(h / patch)
    pw = math.ceil(w / patch)
    padded = np.zeros((ph * patch, pw * patch), dtype=np.float32)
    padded[:h, :w] = grad
    scores = padded.reshape(ph, patch, pw, patch).mean(axis=(1, 3))
    return scores, (h, w)


def top_roi(arr: np.ndarray, patch: int = 32, target_ratio: float = 0.12) -> tuple[int, int, int, int, np.ndarray]:
    gray = grayscale(arr)
    grad = gradient_map(gray)
    scores, (h, w) = patch_scores(grad, patch)
    flat = scores.flatten()
    keep = max(1, int(math.ceil(flat.size * target_ratio)))
    threshold = np.partition(flat, -keep)[-keep]
    mask_small = scores >= threshold
    labels = measure.label(mask_small)
    props = measure.regionprops(labels, intensity_image=scores)
    if props:
        best = max(props, key=lambda p: p.intensity_mean * p.area)
        minr, minc, maxr, maxc = best.bbox
    else:
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        minr = max(idx[0] - 1, 0)
        minc = max(idx[1] - 1, 0)
        maxr = min(idx[0] + 2, scores.shape[0])
        maxc = min(idx[1] + 2, scores.shape[1])

    x1 = max(minc * patch - patch, 0)
    y1 = max(minr * patch - patch, 0)
    x2 = min(maxc * patch + patch, w)
    y2 = min(maxr * patch + patch, h)
    return x1, y1, x2, y2, grad


def resize_roundtrip(gray: np.ndarray, size: int = 224) -> np.ndarray:
    down = transform.resize(gray, (size, size), anti_aliasing=True, preserve_range=True)
    up = transform.resize(down, gray.shape, anti_aliasing=True, preserve_range=True)
    return up.astype(np.float32)


def detail_loss(gray: np.ndarray, degraded: np.ndarray) -> float:
    orig_grad = filters.sobel(gray)
    deg_grad = filters.sobel(degraded)
    return float(np.mean(np.abs(orig_grad - deg_grad)))


def detail_retention(gray: np.ndarray, degraded: np.ndarray) -> float:
    orig_grad = filters.sobel(gray)
    deg_grad = filters.sobel(degraded)
    base = float(np.mean(np.abs(orig_grad)))
    residual = float(np.mean(np.abs(orig_grad - deg_grad)))
    return 1.0 - residual / max(base, 1e-8)


def shannon_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram((gray * 255).astype(np.uint8), bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def sharpness(gray: np.ndarray) -> float:
    lap = filters.laplace(gray)
    return float(np.var(lap))


def analyze_image(path: Path) -> ImageResult:
    arr = load_image(path)
    h, w = arr.shape[:2]
    gray = grayscale(arr)
    x1, y1, x2, y2, grad = top_roi(arr)
    roi = gray[y1:y2, x1:x2]

    global_rt = resize_roundtrip(gray)
    roi_rt = resize_roundtrip(roi)

    global_loss = detail_loss(gray, global_rt)
    roi_loss = detail_loss(roi, roi_rt)
    global_retention = detail_retention(gray, global_rt)
    roi_retention = detail_retention(roi, roi_rt)
    recovered = roi_retention - global_retention

    return ImageResult(
        image=path.name,
        width=w,
        height=h,
        roi_x1=x1,
        roi_y1=y1,
        roi_x2=x2,
        roi_y2=y2,
        roi_area_ratio=float(((x2 - x1) * (y2 - y1)) / (w * h)),
        global_detail_loss=global_loss,
        roi_detail_loss=roi_loss,
        global_detail_retention=global_retention,
        roi_detail_retention=roi_retention,
        recovered_detail_ratio=float(recovered),
        entropy_global=shannon_entropy(gray),
        entropy_roi=shannon_entropy(roi),
        sharpness_global=sharpness(gray),
        sharpness_roi=sharpness(roi),
    )


def add_roi_box(arr: np.ndarray, result: ImageResult) -> Image.Image:
    image = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((result.roi_x1, result.roi_y1, result.roi_x2, result.roi_y2), outline=(255, 80, 80), width=6)
    return image


def save_overview(results: list[ImageResult], paths: Iterable[Path]) -> None:
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4.8 * len(results)))
    if len(results) == 1:
        axes = np.array([axes])
    for row, (result, path) in enumerate(zip(results, paths)):
        arr = load_image(path)
        gray = grayscale(arr)
        roi = arr[result.roi_y1:result.roi_y2, result.roi_x1:result.roi_x2]
        global_rt = resize_roundtrip(gray)
        roi_rt = resize_roundtrip(grayscale(roi))

        axes[row, 0].imshow(add_roi_box(arr, result))
        axes[row, 0].set_title(f"{result.image}: ROI")
        axes[row, 1].imshow(global_rt, cmap="gray")
        axes[row, 1].set_title("Global 224x224 round-trip")
        axes[row, 2].imshow(roi)
        axes[row, 2].set_title("Task-guided crop")
        axes[row, 3].imshow(roi_rt, cmap="gray")
        axes[row, 3].set_title("Crop 224x224 round-trip")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "roi_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(df))
    width = 0.34

    axes[0].bar(x - width / 2, df["global_detail_retention"], width, label="Global")
    axes[0].bar(x + width / 2, df["roi_detail_retention"], width, label="ROI crop")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["image"], rotation=15)
    axes[0].set_ylabel("Gradient-detail retention")
    axes[0].set_title("Relative detail preserved after fixed-resolution encoding")
    axes[0].legend()

    axes[1].bar(df["image"], df["recovered_detail_ratio"], color="#d55e00")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Retention gain")
    axes[1].set_title("Relative gain from task-guided crop")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "detail_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scale_validation(path: Path, result: ImageResult) -> None:
    arr = load_image(path)
    gray = grayscale(arr)
    roi = gray[result.roi_y1:result.roi_y2, result.roi_x1:result.roi_x2]
    sizes = [112, 160, 224, 320, 448]
    global_losses = []
    roi_losses = []
    for size in sizes:
        global_losses.append(detail_retention(gray, resize_roundtrip(gray, size=size)))
        roi_losses.append(detail_retention(roi, resize_roundtrip(roi, size=size)))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(sizes, global_losses, marker="o", label="Global frame")
    ax.plot(sizes, roi_losses, marker="o", label="ROI crop")
    ax.set_xlabel("Encoder input size")
    ax.set_ylabel("Gradient-detail retention")
    ax.set_title(f"Resolution sensitivity on {result.image}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "scale_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_literature_notes() -> None:
    notes = []
    targets = {
        "paper_000.pdf": "Anchor paper proposing LLM-guided visual search and crop-based visual working memory.",
        "paper_001.pdf": "Supporting background on attention explainability as a possible route to region selection.",
        "paper_002.pdf": "Supporting background on frozen image encoders and lightweight bridging modules.",
        "paper_003.pdf": "Additional local corpus item; text extraction may be sparse.",
    }
    for name, framing in targets.items():
        path = RELATED_DIR / name
        try:
            text = subprocess.check_output(["pdftotext", str(path), "-"], text=True, stderr=subprocess.STDOUT)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            excerpt = lines[:20]
        except Exception as exc:
            excerpt = [f"Extraction failed: {exc}"]
        notes.append({"file": name, "framing": framing, "excerpt": excerpt})

    with open(OUTPUT_DIR / "literature_notes.json", "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


def write_summary(df: pd.DataFrame) -> None:
    summary = {
        "num_images": int(len(df)),
        "mean_global_detail_loss": float(df["global_detail_loss"].mean()),
        "mean_roi_detail_loss": float(df["roi_detail_loss"].mean()),
        "mean_recovered_detail_ratio": float(df["recovered_detail_ratio"].mean()),
        "best_image": df.sort_values("recovered_detail_ratio", ascending=False).iloc[0]["image"],
    }
    with open(OUTPUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ensure_dirs()
    build_literature_notes()
    image_paths = sorted(DATA_DIR.glob("*.png"))
    results = [analyze_image(path) for path in image_paths]
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(OUTPUT_DIR / "image_metrics.csv", index=False)
    write_summary(df)
    save_overview(results, image_paths)
    save_metric_plot(df)
    save_scale_validation(image_paths[1], results[1] if len(results) > 1 else results[0])


if __name__ == "__main__":
    main()
