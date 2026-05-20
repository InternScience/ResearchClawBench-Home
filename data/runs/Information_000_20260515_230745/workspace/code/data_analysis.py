"""
Data Analysis: Process equation.png and doge.png
to evaluate multimodal capabilities.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import json
import os


def analyze_image_properties(image_path: str) -> dict:
    """Extract basic image properties and statistics."""
    img = Image.open(image_path)
    img_array = np.array(img)

    props = {
        "path": image_path,
        "size": list(img.size),
        "mode": img.mode,
        "mean_rgb": [float(c) for c in img_array.mean(axis=(0, 1))],
        "std_rgb": [float(c) for c in img_array.std(axis=(0, 1))],
    }

    if img.mode == "RGB":
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        props["gray_mean"] = float(gray.mean())
        props["gray_std"] = float(gray.std())
        props["entropy_estimate"] = float(
            -np.sum((gray / 255.0) * np.log2(np.clip(gray / 255.0, 1e-10, 1)))
            / gray.size * 255
        )

    return props


def analyze_text_density(image_path: str) -> dict:
    """Estimate text regions and density using edge detection."""
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # Edge detection via gradient
    gy, gx = np.gradient(img_array.astype(float))
    edge_magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Binarize edges
    edge_threshold = np.percentile(edge_magnitude, 90)
    edge_mask = edge_magnitude > edge_threshold

    # Estimate text density
    total_pixels = edge_mask.size
    edge_pixels = edge_mask.sum()

    # Horizontal projection profile (useful for text line detection)
    h_proj = edge_mask.sum(axis=1)
    v_proj = edge_mask.sum(axis=0)

    return {
        "edge_density": float(edge_pixels / total_pixels),
        "edge_pixel_count": int(edge_pixels),
        "total_pixels": int(total_pixels),
        "h_proj_peaks": int(np.sum(h_proj > h_proj.mean() * 1.5)),
        "v_proj_peaks": int(np.sum(v_proj > v_proj.mean() * 1.5)),
    }


def analyze_region_properties(image_path: str, num_regions: int = 4) -> dict:
    """Analyze image by dividing into regions (for multi-panel images)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    img_array = np.array(img)

    # Divide into horizontal strips
    region_h = h // num_regions
    regions = []
    for i in range(num_regions):
        y0 = i * region_h
        y1 = (i + 1) * region_h if i < num_regions - 1 else h
        region = img_array[y0:y1, :, :]

        r_mean = float(region[:, :, 0].mean())
        g_mean = float(region[:, :, 1].mean())
        b_mean = float(region[:, :, 2].mean())

        # Compute gray-level contrast
        gray = 0.299 * region[:, :, 0] + 0.587 * region[:, :, 1] + 0.114 * region[:, :, 2]
        contrast = float(gray.std())

        regions.append({
            "y_range": [y0, y1],
            "mean_rgb": [r_mean, g_mean, b_mean],
            "contrast": contrast,
        })

    return {"num_regions": num_regions, "regions": regions}


def compute_token_efficiency() -> dict:
    """Compare token efficiency of different encoding strategies."""
    strategies = []

    # Chameleon-style: single VQGAN tokenizer
    for img_size in [256, 384, 512]:
        ds = 16
        tokens = (img_size // ds) ** 2
        strategies.append({
            "strategy": "Single VQGAN (Chameleon)",
            "img_size": img_size,
            "tokens": tokens,
            "token_dim": 8,
            "total_dim": tokens * 8,
        })

    # DVE-style: dual pathway
    for img_size in [256, 384, 512]:
        patch_size = 16
        ds = 16
        # Understanding: patches
        u_tokens = (img_size // patch_size) ** 2
        u_dim = 4096
        # Generation: quantized
        g_tokens = (img_size // ds) ** 2
        g_dim = 8
        strategies.append({
            "strategy": "DVE (Ours)",
            "img_size": img_size,
            "understand_tokens": u_tokens,
            "understand_dim": u_dim,
            "generate_tokens": g_tokens,
            "generate_dim": g_dim,
            "total_understand_dim": u_tokens * u_dim,
            "total_generate_dim": g_tokens * g_dim,
        })

    return strategies


def main():
    os.makedirs("outputs", exist_ok=True)

    images = ["data/equation.png", "data/doge.png"]

    all_results = {}
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        props = analyze_image_properties(img_path)
        text_density = analyze_text_density(img_path)
        regions = analyze_region_properties(img_path, num_regions=4 if "doge" in name else 2)

        result = {"properties": props, "text_density": text_density, "regions": regions}
        all_results[name] = result

        print(f"\n=== {name} ===")
        print(f"  Size: {props['size']}")
        print(f"  Mean RGB: [{props['mean_rgb'][0]:.1f}, {props['mean_rgb'][1]:.1f}, {props['mean_rgb'][2]:.1f}]")
        print(f"  Edge density: {text_density['edge_density']:.4f}")

    # Token efficiency
    token_eff = compute_token_efficiency()
    all_results["token_efficiency"] = token_eff

    # Save results
    with open("outputs/data_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to outputs/data_analysis.json")


if __name__ == "__main__":
    main()
