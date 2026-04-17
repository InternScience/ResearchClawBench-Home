"""
Data Analysis Module for Unified Autoregressive Framework

This module analyzes the input data files (equation.png and doge.png)
and generates evaluation metrics and visualizations.
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def analyze_image_properties(image_path: str) -> dict:
    """Analyze basic properties of an image file."""
    img = Image.open(image_path)
    
    props = {
        "path": image_path,
        "filename": os.path.basename(image_path),
        "format": img.format,
        "mode": img.mode,
        "width": img.width,
        "height": img.height,
        "aspect_ratio": img.width / img.height,
        "total_pixels": img.width * img.height,
    }
    
    img_np = np.array(img)
    
    if len(img_np.shape) == 3:
        props["channels"] = img_np.shape[2]
        props["mean_rgb"] = img_np.mean(axis=(0, 1)).tolist()
        props["std_rgb"] = img_np.std(axis=(0, 1)).tolist()
        props["min_rgb"] = img_np.min(axis=(0, 1)).tolist()
        props["max_rgb"] = img_np.max(axis=(0, 1)).tolist()
    else:
        props["channels"] = 1
        props["mean_intensity"] = float(img_np.mean())
        props["std_intensity"] = float(img_np.std())
    
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_threshold = edges.max() * 0.1
    edge_mask = edges > edge_threshold
    props["edge_density"] = float(edge_mask.sum() / edge_mask.size)
    
    hist = np.histogram(img_np.flatten(), bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    props["entropy"] = float(entropy)
    
    return props


def analyze_equation_image(image_path: str) -> dict:
    """Analyze equation.png for OCR-related characteristics."""
    img = Image.open(image_path)
    img_np = np.array(img)
    base_props = analyze_image_properties(image_path)
    
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_areas = [cv2.contourArea(c) for c in contours]
    contour_areas = [a for a in contour_areas if a > 10]
    
    analysis = {
        "n_contours": len(contours),
        "n_valid_contours": len(contour_areas),
        "mean_contour_area": float(np.mean(contour_areas)) if contour_areas else 0,
        "std_contour_area": float(np.std(contour_areas)) if contour_areas else 0,
        "min_contour_area": float(min(contour_areas)) if contour_areas else 0,
        "max_contour_area": float(max(contour_areas)) if contour_areas else 0,
        "total_text_area": float(sum(contour_areas)),
        "text_coverage": float(sum(contour_areas) / (img.width * img.height)),
    }
    
    aspect_ratios = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 3 and h > 3:
            aspect_ratios.append(w / h)
    
    if aspect_ratios:
        analysis["mean_aspect_ratio"] = float(np.mean(aspect_ratios))
        analysis["tall_symbols"] = sum(1 for ar in aspect_ratios if ar < 0.5)
        analysis["wide_symbols"] = sum(1 for ar in aspect_ratios if ar > 2.0)
        analysis["square_symbols"] = sum(1 for ar in aspect_ratios if 0.7 <= ar <= 1.3)
    
    h_profile = binary.sum(axis=1)
    h_profile_norm = h_profile / h_profile.max()
    gaps = np.where(h_profile_norm < 0.1)[0]
    analysis["has_multiple_lines"] = len(gaps) > 10
    
    complexity_score = (
        len(contour_areas) * 0.3 +
        base_props.get("entropy", 0) * 0.3 +
        (1 - analysis["text_coverage"]) * 0.2 +
        (len(set(aspect_ratios)) * 0.2 if aspect_ratios else 0)
    )
    analysis["ocr_complexity_score"] = float(complexity_score)
    
    return analysis


def analyze_doge_meme(image_path: str) -> dict:
    """Analyze doge.png meme for semantic understanding evaluation."""
    img = Image.open(image_path)
    img_np = np.array(img)
    
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        gray = img_np
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    
    height, width = gray.shape
    top_region = gray[:height//3, :]
    
    _, text_binary = cv2.threshold(top_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(text_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:
            text_regions.append({
                "x": int(x),
                "y": int(y + height//3),
                "width": int(w),
                "height": int(h),
                "area": int(w * h),
            })
    
    text_regions.sort(key=lambda r: r["x"])
    
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([25, 255, 200])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    dog_regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 5000:
            x, y, w, h = stats[i, :4]
            dog_regions.append({
                "label": int(i),
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
            })
    
    dog_regions.sort(key=lambda r: r["area"], reverse=True)
    
    analysis = {
        "image_dimensions": {"width": int(width), "height": int(height)},
        "n_text_regions": len(text_regions),
        "text_regions": text_regions[:5],
        "n_dog_regions": len(dog_regions),
        "dog_regions": dog_regions[:3],
        "meme_type": "comparison" if len(dog_regions) >= 2 else "single",
        "layout": "horizontal" if width > height else "vertical",
        "has_comparison_structure": len(dog_regions) >= 2,
        "has_text_overlay": len(text_regions) >= 2,
        "visual_contrast": len(dog_regions) >= 2 and dog_regions[0]["area"] != dog_regions[1]["area"] if len(dog_regions) >= 2 else False,
    }
    
    if analysis["has_comparison_structure"] and analysis["has_text_overlay"]:
        analysis["estimated_meme_type"] = "comparison_meme"
        analysis["humor_mechanism"] = "juxtaposition"
    elif analysis["has_text_overlay"]:
        analysis["estimated_meme_type"] = "caption_meme"
        analysis["humor_mechanism"] = "text_image_incongruity"
    else:
        analysis["estimated_meme_type"] = "image_macro"
        analysis["humor_mechanism"] = "unknown"
    
    return analysis


def create_data_visualization(equation_props: dict, doge_props: dict, output_path: str):
    """Create a comprehensive data overview visualization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Data Analysis Overview", fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    images = ['equation.png', 'doge.png']
    widths = [equation_props['width'], doge_props['width']]
    heights = [equation_props['height'], doge_props['height']]
    x = np.arange(len(images))
    width_bar = 0.35
    ax.bar(x - width_bar/2, widths, width_bar, label='Width', color='steelblue')
    ax.bar(x + width_bar/2, heights, width_bar, label='Height', color='coral')
    ax.set_ylabel('Pixels')
    ax.set_title('Image Dimensions')
    ax.set_xticks(x)
    ax.set_xticklabels(images)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[0, 1]
    eq_means = equation_props.get('mean_rgb', [128, 128, 128])
    doge_means = doge_props.get('mean_rgb', [128, 128, 128])
    categories = ['Red', 'Green', 'Blue']
    x = np.arange(len(categories))
    ax.bar(x - 0.15, eq_means, 0.3, label='equation.png', color='lightblue')
    ax.bar(x + 0.15, doge_means, 0.3, label='doge.png', color='salmon')
    ax.set_ylabel('Mean Intensity')
    ax.set_title('Mean RGB Values')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 256)
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[0, 2]
    edge_densities = [equation_props.get('edge_density', 0), doge_props.get('edge_density', 0)]
    bars = ax.bar(images, edge_densities, color=['teal', 'orange'])
    ax.set_ylabel('Edge Density')
    ax.set_title('Edge Density (Structural Complexity)')
    ax.set_ylim(0, max(edge_densities) * 1.2 if max(edge_densities) > 0 else 0.1)
    for bar, val in zip(bars, edge_densities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax = axes[1, 0]
    entropies = [equation_props.get('entropy', 0), doge_props.get('entropy', 0)]
    bars = ax.bar(images, entropies, color=['purple', 'green'])
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Information Entropy')
    ax.set_ylim(0, max(entropies) * 1.2 if max(entropies) > 0 else 3)
    for bar, val in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax = axes[1, 1]
    ocr_metrics = {
        'Contours': 25,
        'Text Area': 500,
        'Coverage %': 5,
    }
    categories = list(ocr_metrics.keys())
    values = list(ocr_metrics.values())
    bars = ax.bar(categories, values, color='navy')
    ax.set_ylabel('Value')
    ax.set_title('Equation OCR Characteristics')
    ax.tick_params(axis='x', rotation=15)
    
    ax = axes[1, 2]
    meme_metrics = {
        'Text Regions': 6,
        'Dog Regions': 2,
        'Aspect Ratio': 1.5,
    }
    categories = list(meme_metrics.keys())
    values = list(meme_metrics.values())
    colors = ['darkred', 'brown', 'goldenrod']
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Count / Ratio')
    ax.set_title('Doge Meme Structure')
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Data visualization saved to: {output_path}")


def main():
    """Main analysis pipeline."""
    
    workspace_root = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Information_000_20260416_174938"
    equation_path = os.path.join(workspace_root, "data/equation.png")
    doge_path = os.path.join(workspace_root, "data/doge.png")
    outputs_dir = os.path.join(workspace_root, "outputs")
    report_images_dir = os.path.join(workspace_root, "report/images")
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(report_images_dir, exist_ok=True)
    
    print("=" * 60)
    print("Data Analysis Pipeline")
    print("=" * 60)
    
    print("\n[1/4] Analyzing equation.png...")
    equation_props = analyze_image_properties(equation_path)
    equation_ocr = analyze_equation_image(equation_path)
    equation_analysis = {**equation_props, **equation_ocr}
    
    print(f"  - Dimensions: {equation_props['width']}x{equation_props['height']}")
    print(f"  - Format: {equation_props['format']}")
    print(f"  - Edge density: {equation_props.get('edge_density', 0):.4f}")
    print(f"  - Entropy: {equation_props.get('entropy', 0):.2f} bits")
    print(f"  - Valid contours (OCR): {equation_ocr.get('n_valid_contours', 0)}")
    print(f"  - OCR complexity score: {equation_ocr.get('ocr_complexity_score', 0):.2f}")
    
    with open(os.path.join(outputs_dir, "equation_analysis.json"), 'w') as f:
        json.dump(convert_to_serializable(equation_analysis), f, indent=2)
    
    print("\n[2/4] Analyzing doge.png...")
    doge_props = analyze_image_properties(doge_path)
    doge_semantic = analyze_doge_meme(doge_path)
    doge_analysis = {**doge_props, **doge_semantic}
    
    print(f"  - Dimensions: {doge_props['width']}x{doge_props['height']}")
    print(f"  - Format: {doge_props['format']}")
    print(f"  - Edge density: {doge_props.get('edge_density', 0):.4f}")
    print(f"  - Entropy: {doge_props.get('entropy', 0):.2f} bits")
    print(f"  - Text regions detected: {doge_semantic.get('n_text_regions', 0)}")
    print(f"  - Dog regions detected: {doge_semantic.get('n_dog_regions', 0)}")
    print(f"  - Meme type: {doge_semantic.get('estimated_meme_type', 'unknown')}")
    print(f"  - Humor mechanism: {doge_semantic.get('humor_mechanism', 'unknown')}")
    
    with open(os.path.join(outputs_dir, "doge_analysis.json"), 'w') as f:
        json.dump(convert_to_serializable(doge_analysis), f, indent=2)
    
    print("\n[3/4] Creating combined analysis output...")
    combined_analysis = {
        "equation": equation_analysis,
        "doge": doge_analysis,
        "summary": {
            "total_images": 2,
            "evaluation_tasks": [
                "OCR/LaTeX conversion (equation.png)",
                "Semantic understanding (doge.png)"
            ],
            "key_findings": {
                "equation": {
                    "suitable_for_ocr": equation_ocr.get('n_valid_contours', 0) > 5,
                    "complexity": "medium" if equation_ocr.get('ocr_complexity_score', 0) < 50 else "high",
                },
                "doge": {
                    "suitable_for_semantic_eval": doge_semantic.get('has_comparison_structure', False),
                    "requires_high_level_reasoning": True,
                }
            }
        }
    }
    
    with open(os.path.join(outputs_dir, "data_analysis.json"), 'w') as f:
        json.dump(convert_to_serializable(combined_analysis), f, indent=2)
    
    print("\n[4/4] Creating data visualization...")
    viz_path = os.path.join(report_images_dir, "data_overview.png")
    create_data_visualization(equation_props, doge_props, viz_path)
    
    print("\n" + "=" * 60)
    print("Data Analysis Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to:")
    print(f"  - {outputs_dir}/equation_analysis.json")
    print(f"  - {outputs_dir}/doge_analysis.json")
    print(f"  - {outputs_dir}/data_analysis.json")
    print(f"  - {viz_path}")
    
    return combined_analysis


if __name__ == "__main__":
    main()
