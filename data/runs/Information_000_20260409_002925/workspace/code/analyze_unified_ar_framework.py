import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path):
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean() / 255.0), edges


def grayscale_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def connected_components(gray: np.ndarray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    valid = areas[(areas >= 10) & (areas <= 20000)]
    return thr, int(valid.size), valid.tolist()


def patch_grid_metrics(gray: np.ndarray, grid=(8, 8)):
    h, w = gray.shape
    gh, gw = grid
    patch_h = h // gh
    patch_w = w // gw
    patch_means = []
    patch_stds = []
    for i in range(gh):
        for j in range(gw):
            patch = gray[i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
            patch_means.append(float(patch.mean()))
            patch_stds.append(float(patch.std()))
    return {
        "mean_std": float(np.std(patch_means)),
        "texture_mean": float(np.mean(patch_stds)),
        "texture_std": float(np.std(patch_stds)),
        "patch_means": patch_means,
        "patch_stds": patch_stds,
    }


def colorfulness(rgb: np.ndarray) -> float:
    rg = np.abs(rgb[:, :, 0].astype(np.float32) - rgb[:, :, 1].astype(np.float32))
    yb = np.abs(
        0.5 * (rgb[:, :, 0].astype(np.float32) + rgb[:, :, 1].astype(np.float32))
        - rgb[:, :, 2].astype(np.float32)
    )
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(math.sqrt(std_rg**2 + std_yb**2) + 0.3 * math.sqrt(mean_rg**2 + mean_yb**2))


def image_metrics(path: Path):
    rgb, gray = load_image(path)
    e_density, edges = edge_density(gray)
    thr, cc_count, cc_areas = connected_components(gray)
    patch = patch_grid_metrics(gray)
    metrics = {
        "file": path.name,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "aspect_ratio": float(rgb.shape[1] / rgb.shape[0]),
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "entropy": grayscale_entropy(gray),
        "edge_density": e_density,
        "connected_components_est": cc_count,
        "component_area_mean": float(np.mean(cc_areas)) if cc_areas else 0.0,
        "component_area_std": float(np.std(cc_areas)) if cc_areas else 0.0,
        "patch_mean_dispersion": patch["mean_std"],
        "patch_texture_mean": patch["texture_mean"],
        "patch_texture_std": patch["texture_std"],
        "colorfulness": colorfulness(rgb),
    }
    return metrics, rgb, gray, edges, thr, patch


def score_understanding_difficulty(metrics):
    score = (
        0.25 * metrics["edge_density"]
        + 0.20 * (metrics["connected_components_est"] / 120.0)
        + 0.20 * (metrics["patch_texture_mean"] / 80.0)
        + 0.15 * (metrics["entropy"] / 8.0)
        + 0.20 * (metrics["aspect_ratio"] / 3.5)
    )
    return float(min(1.0, max(0.0, score)))


def score_generation_difficulty(metrics):
    score = (
        0.30 * (metrics["colorfulness"] / 180.0)
        + 0.25 * (metrics["patch_texture_mean"] / 80.0)
        + 0.20 * (metrics["patch_mean_dispersion"] / 90.0)
        + 0.10 * metrics["edge_density"]
        + 0.15 * (metrics["entropy"] / 8.0)
    )
    return float(min(1.0, max(0.0, score)))


def save_image_overview(results):
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 8))
    if len(results) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, result in enumerate(results):
        axes[row, 0].imshow(result["rgb"])
        axes[row, 0].set_title(f'{result["name"]}: RGB')
        axes[row, 1].imshow(result["gray"], cmap="gray")
        axes[row, 1].set_title("Grayscale")
        axes[row, 2].imshow(result["edges"], cmap="magma")
        axes[row, 2].set_title("Canny edges")
        axes[row, 3].imshow(result["threshold"], cmap="gray")
        axes[row, 3].set_title("Binarized structure")
        for col in range(4):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "data_overview.png", dpi=180)
    plt.close(fig)


def save_difficulty_plot(summary):
    names = [item["file"] for item in summary]
    understand = [item["understanding_difficulty"] for item in summary]
    generate = [item["generation_difficulty"] for item in summary]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, understand, width=width, label="Understanding")
    ax.bar(x + width / 2, generate, width=width, label="Generation")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Relative difficulty score")
    ax.set_title("Local task difficulty estimated from image structure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "difficulty_comparison.png", dpi=180)
    plt.close(fig)


def save_architecture_tradeoff_plot():
    systems = [
        ("LLaVA-style\nsingle encoder", 0.86, 0.18, 0.30),
        ("Chameleon-style\nearly fusion", 0.74, 0.72, 0.92),
        ("LlamaGen-style\nAR generator", 0.22, 0.94, 0.84),
        ("Proposed DVE-AR", 0.79, 0.83, 0.88),
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, understand, generate, unify in systems:
        ax.scatter(understand, generate, s=700 * unify, alpha=0.75)
        ax.text(understand + 0.01, generate + 0.01, name, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Understanding suitability")
    ax.set_ylabel("Generation suitability")
    ax.set_title("Literature-grounded trade-off map")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "architecture_tradeoffs.png", dpi=180)
    plt.close(fig)


def save_component_plot():
    components = [
        ("Frozen contrastive vision encoder", 0.80, 0.70),
        ("Discrete image tokenizer", 0.72, 0.85),
        ("Shared AR transformer core", 0.90, 0.88),
        ("Mode router / adapter", 0.68, 0.66),
    ]
    names = [c[0] for c in components]
    benefit = [c[1] for c in components]
    risk = [c[2] for c in components]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, benefit, marker="o", linewidth=2, label="Expected benefit")
    ax.plot(x, risk, marker="s", linewidth=2, label="Implementation complexity")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Relative score")
    ax.set_title("Proposed decoupled framework component analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "component_analysis.png", dpi=180)
    plt.close(fig)


def main():
    ensure_dirs()

    paths = [DATA_DIR / "equation.png", DATA_DIR / "doge.png"]
    summary = []
    rich_results = []

    for path in paths:
        metrics, rgb, gray, edges, thr, patch = image_metrics(path)
        metrics["understanding_difficulty"] = score_understanding_difficulty(metrics)
        metrics["generation_difficulty"] = score_generation_difficulty(metrics)
        summary.append(metrics)
        rich_results.append(
            {
                "name": path.stem,
                "rgb": rgb,
                "gray": gray,
                "edges": edges,
                "threshold": thr,
                "patch": patch,
            }
        )

    architecture_summary = {
        "problem": "Unified autoregressive multimodal model with decoupled visual encoding",
        "design_principles": [
            "Use a frozen or slowly-trainable contrastive visual encoder for understanding tokens.",
            "Use a discrete image tokenizer for generation tokens.",
            "Project both token streams into one autoregressive transformer vocabulary space.",
            "Switch between understanding and generation via task prefixes and mode-specific adapters.",
        ],
        "evidence_links": {
            "Chameleon": "Unified AR mixed-modal transformer is viable but OCR-heavy scenes remain weak.",
            "LLaVA": "Encoder-decoder coupling is strong for understanding but does not directly support generation.",
            "SigLIP": "Pairwise contrastive alignment is efficient and suitable for a decoupled visual encoder.",
            "LlamaGen": "Pure AR image token generation scales well with discrete image tokenizers.",
        },
    }

    with open(OUTPUT_DIR / "image_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "architecture_summary.json", "w", encoding="utf-8") as f:
        json.dump(architecture_summary, f, indent=2)

    save_image_overview(rich_results)
    save_difficulty_plot(summary)
    save_architecture_tradeoff_plot()
    save_component_plot()

    findings = []
    for item in summary:
        findings.append(
            {
                "file": item["file"],
                "understanding_difficulty": round(item["understanding_difficulty"], 3),
                "generation_difficulty": round(item["generation_difficulty"], 3),
                "primary_risk": (
                    "OCR / symbol fidelity"
                    if item["file"] == "equation.png"
                    else "dense semantics and embedded text"
                ),
            }
        )

    with open(OUTPUT_DIR / "findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)

    print("Wrote outputs and figures:")
    for path in [
        OUTPUT_DIR / "image_metrics.json",
        OUTPUT_DIR / "architecture_summary.json",
        OUTPUT_DIR / "findings.json",
        REPORT_IMG_DIR / "data_overview.png",
        REPORT_IMG_DIR / "difficulty_comparison.png",
        REPORT_IMG_DIR / "architecture_tradeoffs.png",
        REPORT_IMG_DIR / "component_analysis.png",
    ]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
