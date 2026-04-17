"""
Refined analysis: Compare the effective resolution and detail quality
of the ROI as seen through different processing pipelines.

The key comparison is:
- ROI pixels as they appear in the global 224x224 encoding (very few pixels)
- ROI pixels when directly cropped and encoded at 224x224 or 448x448 (many more pixels)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import json
import os

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260416_190849"
DATA_DIR = os.path.join(WORKSPACE, "data", "demo_imgs")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMG_DIR = os.path.join(WORKSPACE, "report", "images")

demo1 = Image.open(os.path.join(DATA_DIR, "demo1.png")).convert("RGB")
demo2 = Image.open(os.path.join(DATA_DIR, "demo2.png")).convert("RGB")

def compute_effective_resolution(img_size, roi_bbox, encoder_res=224):
    """Calculate how many effective pixels the ROI gets in the global encoding."""
    w, h = img_size
    x1, y1, x2, y2 = roi_bbox
    roi_w = x2 - x1
    roi_h = y2 - y1
    
    # Fraction of image occupied by ROI
    frac_w = roi_w / w
    frac_h = roi_h / h
    
    # Effective pixels in encoder output
    eff_w = int(encoder_res * frac_w)
    eff_h = int(encoder_res * frac_h)
    
    # Total effective pixels
    eff_pixels = eff_w * eff_h
    
    # Compare to what crop-and-zoom gives
    crop_pixels = encoder_res * encoder_res
    
    return {
        "roi_fraction_w": frac_w,
        "roi_fraction_h": frac_h,
        "effective_pixels_global": eff_pixels,
        "effective_resolution_global": f"{eff_w}x{eff_h}",
        "crop_pixels_224": crop_pixels,
        "crop_pixels_448": 448 * 448,
        "pixel_gain_224x": crop_pixels / max(eff_pixels, 1),
        "pixel_gain_448x": (448 * 448) / max(eff_pixels, 1),
    }

def compute_roi_quality(img, roi_bbox, encoder_res=224):
    """Compare ROI quality in global encoding vs direct crop encoding."""
    x1, y1, x2, y2 = roi_bbox
    w, h = img.size
    
    # Ground truth: original ROI at full resolution
    roi_original = img.crop((x1, y1, x2, y2))
    roi_original_arr = np.array(roi_original.convert("L")).astype(np.float64)
    
    # Path 1: Global encoding -> extract ROI region
    global_encoded = img.resize((encoder_res, encoder_res), Image.BILINEAR)
    scale_x = encoder_res / w
    scale_y = encoder_res / h
    gx1, gy1 = int(x1 * scale_x), int(y1 * scale_y)
    gx2, gy2 = int(x2 * scale_x), int(y2 * scale_y)
    gx2 = max(gx2, gx1 + 1)
    gy2 = max(gy2, gy1 + 1)
    roi_from_global = global_encoded.crop((gx1, gy1, gx2, gy2))
    # Resize back to original ROI size for comparison
    roi_from_global_restored = roi_from_global.resize(roi_original.size, Image.BILINEAR)
    roi_from_global_arr = np.array(roi_from_global_restored.convert("L")).astype(np.float64)
    
    # Path 2: Direct crop at 224
    roi_crop_224 = roi_original.resize((224, 224), Image.LANCZOS)
    roi_crop_224_restored = roi_crop_224.resize(roi_original.size, Image.BILINEAR)
    roi_crop_224_arr = np.array(roi_crop_224_restored.convert("L")).astype(np.float64)
    
    # Path 3: Direct crop at 448
    roi_crop_448 = roi_original.resize((448, 448), Image.LANCZOS)
    roi_crop_448_restored = roi_crop_448.resize(roi_original.size, Image.BILINEAR)
    roi_crop_448_arr = np.array(roi_crop_448_restored.convert("L")).astype(np.float64)
    
    # Compute PSNR and SSIM against original
    min_dim = min(roi_original_arr.shape)
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3
    
    results = {}
    for path_name, path_arr in [("global_224", roi_from_global_arr), 
                                  ("crop_224", roi_crop_224_arr),
                                  ("crop_448", roi_crop_448_arr)]:
        p = psnr(roi_original_arr, path_arr, data_range=255)
        s = ssim(roi_original_arr, path_arr, data_range=255, win_size=win_size)
        mse = np.mean((roi_original_arr - path_arr) ** 2)
        results[path_name] = {"psnr": float(p), "ssim": float(s), "mse": float(mse)}
    
    return results, roi_from_global, roi_crop_224, roi_crop_448

# Define scenarios with carefully chosen ROIs
scenarios = [
    {"name": "Demo1-LicensePlate", "img": demo1, "roi": (390, 400, 620, 550),
     "question": "License plate of silver car", "desc": "Small text detail"},
    {"name": "Demo1-OfficerBadge", "img": demo1, "roi": (200, 300, 400, 500),
     "question": "Officer badge details", "desc": "Small object detail"},
    {"name": "Demo1-BuildingSign", "img": demo1, "roi": (50, 0, 400, 150),
     "question": "Building sign text", "desc": "Distant text"},
    {"name": "Demo2-FlowerDetail", "img": demo2, "roi": (1700, 1100, 2250, 1500),
     "question": "Flower colors in corner", "desc": "Specific region detail"},
    {"name": "Demo2-PersonFace", "img": demo2, "roi": (800, 100, 1100, 400),
     "question": "Person details in background", "desc": "Small person detail"},
    {"name": "Demo2-SignLabel", "img": demo2, "roi": (0, 600, 400, 1000),
     "question": "Labels on flower beds", "desc": "Small label text"},
]

print("="*80)
print("REFINED ANALYSIS: Effective Resolution and ROI Quality Comparison")
print("="*80)

all_results = []
eff_res_data = []

for s in scenarios:
    eff = compute_effective_resolution(s["img"].size, s["roi"])
    quality, roi_global, roi_crop224, roi_crop448 = compute_roi_quality(s["img"], s["roi"])
    
    result = {
        "scenario": s["name"],
        "description": s["desc"],
        "effective_resolution": eff,
        "quality_metrics": quality
    }
    all_results.append(result)
    eff_res_data.append(eff)
    
    print(f"\n{s['name']} ({s['desc']}):")
    print(f"  ROI fraction: {eff['roi_fraction_w']:.1%} x {eff['roi_fraction_h']:.1%}")
    print(f"  Effective resolution in global@224: {eff['effective_resolution_global']} ({eff['effective_pixels_global']} pixels)")
    print(f"  Crop@224: 224x224 ({eff['crop_pixels_224']} pixels) -> {eff['pixel_gain_224x']:.1f}x more pixels")
    print(f"  Crop@448: 448x448 ({eff['crop_pixels_448']} pixels) -> {eff['pixel_gain_448x']:.1f}x more pixels")
    print(f"  Quality (PSNR): Global={quality['global_224']['psnr']:.1f}dB, Crop@224={quality['crop_224']['psnr']:.1f}dB, Crop@448={quality['crop_448']['psnr']:.1f}dB")
    print(f"  Quality (SSIM): Global={quality['global_224']['ssim']:.3f}, Crop@224={quality['crop_224']['ssim']:.3f}, Crop@448={quality['crop_448']['ssim']:.3f}")

# Save results
with open(os.path.join(OUTPUT_DIR, "refined_analysis_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# ============================================================
# Generate comprehensive comparison figure
# ============================================================
print("\n\nGenerating comprehensive comparison figure...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Effective pixel count comparison
scenario_labels = [s["name"].split("-")[1] for s in scenarios]
global_pixels = [e["effective_pixels_global"] for e in eff_res_data]
crop224_pixels = [e["crop_pixels_224"] for e in eff_res_data]
crop448_pixels = [e["crop_pixels_448"] for e in eff_res_data]

x = np.arange(len(scenario_labels))
width = 0.25
axes[0, 0].bar(x - width, global_pixels, width, label='Global@224', color='#e74c3c', alpha=0.8)
axes[0, 0].bar(x, crop224_pixels, width, label='Crop@224', color='#3498db', alpha=0.8)
axes[0, 0].bar(x + width, crop448_pixels, width, label='Crop@448', color='#2ecc71', alpha=0.8)
axes[0, 0].set_yscale('log')
axes[0, 0].set_ylabel('Effective Pixels (log scale)')
axes[0, 0].set_title('Effective Pixel Count for ROI')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(scenario_labels, rotation=30, fontsize=8)
axes[0, 0].legend(fontsize=8)

# Plot 2: Pixel gain factor
gain_224 = [e["pixel_gain_224x"] for e in eff_res_data]
gain_448 = [e["pixel_gain_448x"] for e in eff_res_data]
axes[0, 1].bar(x - 0.15, gain_224, 0.3, label='Crop@224 gain', color='#3498db', alpha=0.8)
axes[0, 1].bar(x + 0.15, gain_448, 0.3, label='Crop@448 gain', color='#2ecc71', alpha=0.8)
axes[0, 1].set_ylabel('Pixel Gain Factor (x)')
axes[0, 1].set_title('Resolution Gain from Cropping')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(scenario_labels, rotation=30, fontsize=8)
axes[0, 1].legend(fontsize=8)

# Plot 3: PSNR comparison
psnr_global = [r["quality_metrics"]["global_224"]["psnr"] for r in all_results]
psnr_crop224 = [r["quality_metrics"]["crop_224"]["psnr"] for r in all_results]
psnr_crop448 = [r["quality_metrics"]["crop_448"]["psnr"] for r in all_results]

axes[0, 2].bar(x - width, psnr_global, width, label='Global@224', color='#e74c3c', alpha=0.8)
axes[0, 2].bar(x, psnr_crop224, width, label='Crop@224', color='#3498db', alpha=0.8)
axes[0, 2].bar(x + width, psnr_crop448, width, label='Crop@448', color='#2ecc71', alpha=0.8)
axes[0, 2].set_ylabel('PSNR (dB)')
axes[0, 2].set_title('ROI Reconstruction Quality (PSNR)')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(scenario_labels, rotation=30, fontsize=8)
axes[0, 2].legend(fontsize=8)
axes[0, 2].axhline(y=30, color='gray', linestyle='--', alpha=0.5)

# Plot 4: SSIM comparison
ssim_global = [r["quality_metrics"]["global_224"]["ssim"] for r in all_results]
ssim_crop224 = [r["quality_metrics"]["crop_224"]["ssim"] for r in all_results]
ssim_crop448 = [r["quality_metrics"]["crop_448"]["ssim"] for r in all_results]

axes[1, 0].bar(x - width, ssim_global, width, label='Global@224', color='#e74c3c', alpha=0.8)
axes[1, 0].bar(x, ssim_crop224, width, label='Crop@224', color='#3498db', alpha=0.8)
axes[1, 0].bar(x + width, ssim_crop448, width, label='Crop@448', color='#2ecc71', alpha=0.8)
axes[1, 0].set_ylabel('SSIM')
axes[1, 0].set_title('ROI Structural Similarity (SSIM)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(scenario_labels, rotation=30, fontsize=8)
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_ylim(0, 1.05)

# Plot 5: PSNR improvement (crop vs global)
psnr_improvement_224 = [c - g for c, g in zip(psnr_crop224, psnr_global)]
psnr_improvement_448 = [c - g for c, g in zip(psnr_crop448, psnr_global)]

axes[1, 1].bar(x - 0.15, psnr_improvement_224, 0.3, label='Crop@224 - Global', color='#3498db', alpha=0.8)
axes[1, 1].bar(x + 0.15, psnr_improvement_448, 0.3, label='Crop@448 - Global', color='#2ecc71', alpha=0.8)
axes[1, 1].set_ylabel('PSNR Improvement (dB)')
axes[1, 1].set_title('PSNR Gain from Task-Guided Cropping')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(scenario_labels, rotation=30, fontsize=8)
axes[1, 1].legend(fontsize=8)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)

# Plot 6: SSIM improvement
ssim_improvement_224 = [c - g for c, g in zip(ssim_crop224, ssim_global)]
ssim_improvement_448 = [c - g for c, g in zip(ssim_crop448, ssim_global)]

axes[1, 2].bar(x - 0.15, ssim_improvement_224, 0.3, label='Crop@224 - Global', color='#3498db', alpha=0.8)
axes[1, 2].bar(x + 0.15, ssim_improvement_448, 0.3, label='Crop@448 - Global', color='#2ecc71', alpha=0.8)
axes[1, 2].set_ylabel('SSIM Improvement')
axes[1, 2].set_title('SSIM Gain from Task-Guided Cropping')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(scenario_labels, rotation=30, fontsize=8)
axes[1, 2].legend(fontsize=8)
axes[1, 2].axhline(y=0, color='black', linewidth=0.5)

plt.suptitle('Comprehensive ROI Quality Analysis: Global Encoding vs Task-Guided Cropping', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "comprehensive_roi_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved comprehensive_roi_analysis.png")

# ============================================================
# Generate ROI size vs improvement scatter plot
# ============================================================
print("Generating ROI size vs improvement scatter...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

roi_fractions = [e["roi_fraction_w"] * e["roi_fraction_h"] for e in eff_res_data]
psnr_gains = psnr_improvement_448
ssim_gains = ssim_improvement_448

axes[0].scatter(roi_fractions, psnr_gains, s=100, c='#2196F3', edgecolors='black', zorder=5)
for i, label in enumerate(scenario_labels):
    axes[0].annotate(label, (roi_fractions[i], psnr_gains[i]), 
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
axes[0].set_xlabel('ROI Area Fraction (of full image)')
axes[0].set_ylabel('PSNR Improvement (dB)')
axes[0].set_title('ROI Size vs PSNR Gain from Cropping')
# Fit trend line
z = np.polyfit(roi_fractions, psnr_gains, 1)
p = np.poly1d(z)
x_line = np.linspace(min(roi_fractions), max(roi_fractions), 100)
axes[0].plot(x_line, p(x_line), '--', color='red', alpha=0.5, label=f'Trend (slope={z[0]:.1f})')
axes[0].legend()

axes[1].scatter(roi_fractions, ssim_gains, s=100, c='#4CAF50', edgecolors='black', zorder=5)
for i, label in enumerate(scenario_labels):
    axes[1].annotate(label, (roi_fractions[i], ssim_gains[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
axes[1].set_xlabel('ROI Area Fraction (of full image)')
axes[1].set_ylabel('SSIM Improvement')
axes[1].set_title('ROI Size vs SSIM Gain from Cropping')
z2 = np.polyfit(roi_fractions, ssim_gains, 1)
p2 = np.poly1d(z2)
axes[1].plot(x_line, p2(x_line), '--', color='red', alpha=0.5, label=f'Trend (slope={z2[0]:.2f})')
axes[1].legend()

plt.suptitle('Relationship Between ROI Size and Information Recovery Gain', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "roi_size_vs_improvement.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved roi_size_vs_improvement.png")

print("\nRefined analysis complete!")
