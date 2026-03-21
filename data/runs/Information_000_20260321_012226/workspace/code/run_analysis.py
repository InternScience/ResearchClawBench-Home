from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs"
REPORT_IMG_DIR = WORKSPACE / "report" / "images"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter


DATASETS = {
    "equation": {
        "path": WORKSPACE / "data" / "equation.png",
        "task_focus": "OCR and formula-to-LaTeX",
        "manual_target": r"A_n = a_0 \left[ 1 + \frac{3}{4}\sum_{k=1}^{n}\left(\frac{4}{9}\right)^k \right]",
        "description": "High-contrast mathematical notation with dense symbols and large whitespace.",
    },
    "doge": {
        "path": WORKSPACE / "data" / "doge.png",
        "task_focus": "Meme semantics and joint text-image reasoning",
        "manual_target": "Swole Doge vs. Cheems style meme contrasting decoupled and single-encoder designs.",
        "description": "A meme with two dogs and overlaid text that requires OCR plus semantic interpretation.",
    },
}

UNIFORM_PATCHES = [8, 16, 32, 64]
GENERATION_PATCH = 32
SEED = 7


@dataclass
class QuadToken:
    x0: int
    y0: int
    x1: int
    y1: int
    depth: int
    mean: float
    variance: float
    edge_density: float
    saliency_ratio: float


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def otsu_threshold(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    cumulative = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.linspace(0.0, 1.0, 256))
    global_mean = cumulative_mean[-1]
    denominator = cumulative * (total - cumulative)
    denominator[denominator == 0] = 1.0
    between = (global_mean * cumulative - cumulative_mean) ** 2 / denominator
    idx = int(np.argmax(between))
    return idx / 255.0


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    dy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return np.hypot(dx, dy)


def dilate(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray((mask.astype(np.uint8) * 255))
    expanded = pil.filter(ImageFilter.MaxFilter(size=size))
    return np.asarray(expanded) > 0


def erode(mask: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray((mask.astype(np.uint8) * 255))
    shrunk = pil.filter(ImageFilter.MinFilter(size=size))
    return np.asarray(shrunk) > 0


def connected_components(binary: np.ndarray, min_area: int = 1) -> tuple[int, list[int]]:
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    areas: list[int] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                areas.append(area)
    return len(areas), areas


def compute_saliency(gray: np.ndarray) -> dict[str, np.ndarray | float]:
    threshold = otsu_threshold(gray)
    dark_threshold = min(0.985, max(0.90, threshold - 0.01))
    dark_mask = gray < dark_threshold
    grad = gradient_magnitude(gray)
    edge_mask = grad > np.percentile(grad, 90)
    combined = dilate(dark_mask | edge_mask, size=11)
    combined = erode(combined, size=3)
    return {
        "threshold": threshold,
        "dark_threshold": dark_threshold,
        "dark_mask": dark_mask,
        "edge_map": grad,
        "edge_mask": edge_mask,
        "saliency_mask": combined,
    }


def entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def bbox_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return float(area / mask.size)


def image_profile(name: str, gray: np.ndarray, saliency: dict[str, np.ndarray | float]) -> dict[str, float | str]:
    dark = saliency["dark_mask"]
    saliency_mask = saliency["saliency_mask"]
    edge_mask = saliency["edge_mask"]
    components, component_areas = connected_components(dark, min_area=12)
    return {
        "image": name,
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
        "pixels": int(gray.size),
        "entropy_bits": entropy(gray),
        "threshold": float(saliency["threshold"]),
        "dark_threshold": float(saliency["dark_threshold"]),
        "dark_ratio": float(dark.mean()),
        "saliency_ratio": float(saliency_mask.mean()),
        "edge_ratio": float(edge_mask.mean()),
        "bbox_ratio": bbox_ratio(saliency_mask),
        "components": int(components),
        "median_component_area": float(np.median(component_areas) if component_areas else 0.0),
        "task_focus": DATASETS[name]["task_focus"],
    }


def reconstruct_uniform(gray: np.ndarray, patch: int) -> np.ndarray:
    h, w = gray.shape
    recon = np.ones_like(gray)
    for y0 in range(0, h, patch):
        for x0 in range(0, w, patch):
            y1 = min(h, y0 + patch)
            x1 = min(w, x0 + patch)
            block = gray[y0:y1, x0:x1]
            recon[y0:y1, x0:x1] = float(block.mean())
    return recon


def quadtree_tokens(
    gray: np.ndarray,
    saliency_mask: np.ndarray,
    edge_map: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    depth: int,
    max_depth: int,
    min_size: int,
    variance_threshold: float,
    edge_threshold: float,
) -> list[QuadToken]:
    region = gray[y0:y1, x0:x1]
    region_saliency = saliency_mask[y0:y1, x0:x1]
    if region.size == 0 or region_saliency.mean() < 0.01:
        return []
    region_edge = edge_map[y0:y1, x0:x1]
    token = QuadToken(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        depth=depth,
        mean=float(region.mean()),
        variance=float(region.var()),
        edge_density=float(region_edge.mean()),
        saliency_ratio=float(region_saliency.mean()),
    )
    width = x1 - x0
    height = y1 - y0
    should_split = (
        depth < max_depth
        and min(width, height) > min_size
        and (
            token.variance > variance_threshold
            or token.edge_density > edge_threshold
            or token.saliency_ratio > 0.35
        )
    )
    if not should_split:
        return [token]
    mx = x0 + width // 2
    my = y0 + height // 2
    if mx in (x0, x1) or my in (y0, y1):
        return [token]
    children: list[QuadToken] = []
    boxes = [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]
    for cx0, cy0, cx1, cy1 in boxes:
        children.extend(
            quadtree_tokens(
                gray=gray,
                saliency_mask=saliency_mask,
                edge_map=edge_map,
                x0=cx0,
                y0=cy0,
                x1=cx1,
                y1=cy1,
                depth=depth + 1,
                max_depth=max_depth,
                min_size=min_size,
                variance_threshold=variance_threshold,
                edge_threshold=edge_threshold,
            )
        )
    return children if children else [token]


def reconstruct_quadtree(gray: np.ndarray, tokens: Iterable[QuadToken]) -> np.ndarray:
    recon = np.ones_like(gray)
    for token in tokens:
        recon[token.y0:token.y1, token.x0:token.x1] = token.mean
    return recon


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    if mse == 0:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def edge_f1(reference_edges: np.ndarray, estimate_edges: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.logical_and(reference_edges, estimate_edges).sum())
    fp = float(np.logical_and(~reference_edges, estimate_edges).sum())
    fn = float(np.logical_and(reference_edges, ~estimate_edges).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def evaluate_scheme(
    name: str,
    gray: np.ndarray,
    saliency: dict[str, np.ndarray | float],
    scheme: str,
    recon: np.ndarray,
    token_count: int,
    role: str,
) -> dict[str, float | str]:
    edge_ref = saliency["edge_mask"]
    edge_recon = gradient_magnitude(recon) > np.percentile(gradient_magnitude(recon), 92)
    precision, recall, f1 = edge_f1(edge_ref, edge_recon)
    saliency_mask = saliency["saliency_mask"]
    salient_mae = float(np.mean(np.abs(gray[saliency_mask] - recon[saliency_mask])))
    background = ~saliency_mask
    background_mae = float(np.mean(np.abs(gray[background] - recon[background]))) if background.any() else 0.0
    dark = saliency["dark_mask"]
    dark_contrast = float(1.0 - recon[dark].mean()) if dark.any() else 0.0
    return {
        "image": name,
        "scheme": scheme,
        "role": role,
        "tokens": int(token_count),
        "tokens_per_megapixel": float(token_count / (gray.size / 1_000_000.0)),
        "psnr_db": psnr(gray, recon),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "salient_mae": salient_mae,
        "background_mae": background_mae,
        "dark_contrast": dark_contrast,
    }


def save_partition_overlay(rgb: np.ndarray, tokens: list[QuadToken], out_path: Path, title: str) -> None:
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for token in tokens:
        color = (255, int(60 + 25 * token.depth), 0)
        draw.rectangle((token.x0, token.y0, token.x1 - 1, token.y1 - 1), outline=color, width=2)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def copy_to_report(paths: Iterable[Path]) -> None:
    for path in paths:
        shutil.copy2(path, REPORT_IMG_DIR / path.name)


def plot_input_overview(inputs: dict[str, dict], profiles: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    for ax, name in zip([ax0, ax1], ["equation", "doge"]):
        ax.imshow(inputs[name]["rgb"])
        ax.set_title(f"{name}: {DATASETS[name]['task_focus']}")
        ax.axis("off")

    melted = profiles.melt(
        id_vars=["image"],
        value_vars=["saliency_ratio", "edge_ratio", "dark_ratio", "bbox_ratio"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=melted, x="metric", y="value", hue="image", ax=ax2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel("")
    ax2.set_title("Input complexity profile")
    ax2.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = OUTPUT_DIR / "figure_01_input_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def add_box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.18",
            facecolor=color,
            edgecolor="#222222",
            linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    add_box(0.6, 4.8, 2.0, 1.2, "Input image", "#d9ecff")
    add_box(0.6, 2.0, 2.0, 1.2, "Text prompt", "#ffe7b8")
    add_box(3.1, 5.0, 2.6, 1.0, "Understanding encoder\nadaptive ROI tokens", "#ccebc5")
    add_box(3.1, 2.0, 2.6, 1.0, "Generation tokenizer\ndiscrete visual codes", "#fbb4ae")
    add_box(6.4, 3.4, 2.8, 1.8, "Unified autoregressive\nTransformer", "#decbe4")
    add_box(10.0, 5.0, 2.8, 1.0, "Understanding head\ncaption / answer / OCR", "#ccebc5")
    add_box(10.0, 2.0, 2.8, 1.0, "Image decoder\ntext-to-image output", "#fbb4ae")

    arrows = [
        ((2.6, 5.4), (3.1, 5.5)),
        ((2.6, 2.6), (6.4, 4.0)),
        ((5.7, 5.5), (6.4, 4.8)),
        ((5.7, 2.5), (6.4, 4.0)),
        ((9.2, 4.8), (10.0, 5.5)),
        ((9.2, 3.8), (10.0, 2.5)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.8))

    ax.text(
        7.8,
        0.7,
        "Key idea: understanding tokens are adaptive and sparse; generation tokens stay compressive and discrete.\n"
        "Both streams are consumed by a single Transformer, but they are not forced to share one visual interface.",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "figure_02_architecture.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frontier(metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, image_name in zip(axes, ["equation", "doge"]):
        subset = metrics[(metrics["image"] == image_name) & (metrics["role"] == "understanding")]
        uniform = subset[subset["scheme"].str.startswith("uniform_")].sort_values("tokens")
        dec = subset[subset["scheme"] == "decoupled_understanding"]
        ax.plot(uniform["tokens"], uniform["edge_recall"], marker="o", linewidth=2, label="Single encoder")
        ax.scatter(dec["tokens"], dec["edge_recall"], s=120, color="#d95f02", label="Decoupled")
        for _, row in uniform.iterrows():
            ax.text(row["tokens"] + 3, row["edge_recall"] + 0.003, row["scheme"].replace("uniform_", "p="), fontsize=9)
        for _, row in dec.iterrows():
            ax.text(row["tokens"] + 3, row["edge_recall"] + 0.003, "adaptive", fontsize=9)
        ax.set_title(image_name)
        ax.set_xlabel("Visual tokens")
        ax.set_ylabel("Edge recall")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    path = OUTPUT_DIR / "figure_03_frontier.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_partition_examples(inputs: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name in zip(axes, ["equation", "doge"]):
        ax.imshow(inputs[name]["partition_overlay"])
        ax.set_title(f"Adaptive understanding tokens: {name}")
        ax.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / "figure_04_partitions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tradeoff(metrics: pd.DataFrame) -> Path:
    records = []
    for image_name in ["equation", "doge"]:
        image_metrics = metrics[metrics["image"] == image_name]
        shared_16 = image_metrics[image_metrics["scheme"] == "uniform_16"].iloc[0]
        shared_32 = image_metrics[image_metrics["scheme"] == "uniform_32"].iloc[0]
        dec = image_metrics[image_metrics["scheme"] == "decoupled_understanding"].iloc[0]
        generation_tokens = int(image_metrics[image_metrics["scheme"] == "generation_32"]["tokens"].iloc[0])
        records.extend(
            [
                {"image": image_name, "design": "Shared encoder p=16", "tokens": int(shared_16["tokens"]), "panel": "Understanding tokens"},
                {"image": image_name, "design": "Shared encoder p=32", "tokens": int(shared_32["tokens"]), "panel": "Understanding tokens"},
                {"image": image_name, "design": "Decoupled understanding", "tokens": int(dec["tokens"]), "panel": "Understanding tokens"},
                {"image": image_name, "design": "Shared encoder p=16", "tokens": int(shared_16["tokens"]), "panel": "Generation tokens"},
                {"image": image_name, "design": "Shared encoder p=32", "tokens": int(shared_32["tokens"]), "panel": "Generation tokens"},
                {"image": image_name, "design": "Decoupled generation", "tokens": generation_tokens, "panel": "Generation tokens"},
            ]
        )
    plot_df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, panel in zip(axes, ["Understanding tokens", "Generation tokens"]):
        subset = plot_df[plot_df["panel"] == panel]
        sns.barplot(data=subset, x="design", y="tokens", hue="image", ax=ax)
        ax.set_title(panel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = OUTPUT_DIR / "figure_05_tradeoff.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_partition_preview(rgb: np.ndarray, tokens: list[QuadToken]) -> np.ndarray:
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for token in tokens:
        color = (255, min(255, 80 + 25 * token.depth), 0)
        draw.rectangle((token.x0, token.y0, token.x1 - 1, token.y1 - 1), outline=color, width=2)
    return np.asarray(image)


def main() -> None:
    np.random.seed(SEED)
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    inputs: dict[str, dict] = {}
    profiles = []
    metrics = []
    summary = {"datasets": {}, "framework": {}}

    for name, metadata in DATASETS.items():
        rgb = load_rgb(metadata["path"])
        gray = rgb_to_gray(rgb)
        saliency = compute_saliency(gray)
        profiles.append(image_profile(name, gray, saliency))

        tokens = quadtree_tokens(
            gray=gray,
            saliency_mask=saliency["saliency_mask"],
            edge_map=saliency["edge_map"],
            x0=0,
            y0=0,
            x1=gray.shape[1],
            y1=gray.shape[0],
            depth=0,
            max_depth=6,
            min_size=16,
            variance_threshold=0.0012,
            edge_threshold=float(np.percentile(saliency["edge_map"], 62)),
        )
        partition_overlay = build_partition_preview(rgb, tokens)
        inputs[name] = {
            "rgb": rgb,
            "gray": gray,
            "saliency": saliency,
            "tokens": tokens,
            "partition_overlay": partition_overlay,
        }

        dec_recon = reconstruct_quadtree(gray, tokens)
        metrics.append(
            evaluate_scheme(
                name=name,
                gray=gray,
                saliency=saliency,
                scheme="decoupled_understanding",
                recon=dec_recon,
                token_count=len(tokens),
                role="understanding",
            )
        )
        generation_recon = reconstruct_uniform(gray, GENERATION_PATCH)
        metrics.append(
            evaluate_scheme(
                name=name,
                gray=gray,
                saliency=saliency,
                scheme=f"generation_{GENERATION_PATCH}",
                recon=generation_recon,
                token_count=math.ceil(gray.shape[0] / GENERATION_PATCH) * math.ceil(gray.shape[1] / GENERATION_PATCH),
                role="generation",
            )
        )

        for patch in UNIFORM_PATCHES:
            uniform_recon = reconstruct_uniform(gray, patch)
            metrics.append(
                evaluate_scheme(
                    name=name,
                    gray=gray,
                    saliency=saliency,
                    scheme=f"uniform_{patch}",
                    recon=uniform_recon,
                    token_count=math.ceil(gray.shape[0] / patch) * math.ceil(gray.shape[1] / patch),
                    role="understanding",
                )
            )

        summary["datasets"][name] = {
            "task_focus": metadata["task_focus"],
            "manual_target": metadata["manual_target"],
            "description": metadata["description"],
            "understanding_tokens": len(tokens),
            "generation_tokens": math.ceil(gray.shape[0] / GENERATION_PATCH) * math.ceil(gray.shape[1] / GENERATION_PATCH),
        }

    profiles_df = pd.DataFrame(profiles)
    metrics_df = pd.DataFrame(metrics)

    fig_paths = [
        plot_input_overview(inputs, profiles_df),
        plot_architecture(),
        plot_frontier(metrics_df),
        plot_partition_examples(inputs),
        plot_tradeoff(metrics_df),
    ]

    overlay_paths = []
    for name in DATASETS:
        overlay_path = OUTPUT_DIR / f"{name}_adaptive_overlay.png"
        save_partition_overlay(inputs[name]["rgb"], inputs[name]["tokens"], overlay_path, f"{name} adaptive token partition")
        overlay_paths.append(overlay_path)

    profiles_df.to_csv(OUTPUT_DIR / "data_profile.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "token_metrics.csv", index=False)

    eq_dec = metrics_df[(metrics_df["image"] == "equation") & (metrics_df["scheme"] == "decoupled_understanding")].iloc[0]
    eq_16 = metrics_df[(metrics_df["image"] == "equation") & (metrics_df["scheme"] == "uniform_16")].iloc[0]
    doge_dec = metrics_df[(metrics_df["image"] == "doge") & (metrics_df["scheme"] == "decoupled_understanding")].iloc[0]
    doge_16 = metrics_df[(metrics_df["image"] == "doge") & (metrics_df["scheme"] == "uniform_16")].iloc[0]

    summary["framework"] = {
        "name": "DVE-AR",
        "expansion": "Decoupled Visual Encoding for Autoregression",
        "generation_patch": GENERATION_PATCH,
        "equation_token_reduction_vs_uniform16": float(1.0 - eq_dec["tokens"] / eq_16["tokens"]),
        "doge_token_reduction_vs_uniform16": float(1.0 - doge_dec["tokens"] / doge_16["tokens"]),
        "equation_edge_recall_gain_vs_uniform32": float(
            eq_dec["edge_recall"]
            - metrics_df[(metrics_df["image"] == "equation") & (metrics_df["scheme"] == "uniform_32")]["edge_recall"].iloc[0]
        ),
        "doge_edge_recall_gain_vs_uniform32": float(
            doge_dec["edge_recall"]
            - metrics_df[(metrics_df["image"] == "doge") & (metrics_df["scheme"] == "uniform_32")]["edge_recall"].iloc[0]
        ),
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    copy_to_report(fig_paths + overlay_paths)


if __name__ == "__main__":
    main()
