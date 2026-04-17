"""
Training-Free Framework for Fine-Grained MLLM Perception Analysis

This script implements and analyzes the task-guided cropping strategy
for mitigating information loss in fixed-resolution vision encoders.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import cv2
from scipy import ndimage, fft as scipy_fft
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.feature import local_binary_pattern
import seaborn as sns
import json
import os

# Paths
WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260416_190849"
DATA_DIR = os.path.join(WORKSPACE, "data", "demo_imgs")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMG_DIR = os.path.join(WORKSPACE, "report", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Load images
demo1 = Image.open(os.path.join(DATA_DIR, "demo1.png")).convert("RGB")
demo2 = Image.open(os.path.join(DATA_DIR, "demo2.png")).convert("RGB")

print(f"Demo1 size: {demo1.size}")
print(f"Demo2 size: {demo2.size}")

# ============================================================
# PART 1: Information Loss Analysis at Different Resolutions
# ============================================================

def simulate_clip_pipeline(img, target_res=224):
    """Simulate CLIP's fixed-resolution processing: resize down then back up."""
    original_size = img.size
    # Downscale to CLIP resolution
    downscaled = img.resize((target_res, target_res), Image.BILINEAR)
    # Upscale back to original for comparison
    restored = downscaled.resize(original_size, Image.BILINEAR)
    return downscaled, restored

def compute_metrics(original, restored):
    """Compute image quality metrics."""
    orig_arr = np.array(original).astype(np.float64)
    rest_arr = np.array(restored).astype(np.float64)
    
    # MSE
    mse = np.mean((orig_arr - rest_arr) ** 2)
    
    # PSNR
    psnr_val = psnr(orig_arr, rest_arr, data_range=255)
    
    # SSIM (compute on grayscale)
    orig_gray = np.mean(orig_arr, axis=2)
    rest_gray = np.mean(rest_arr, axis=2)
    
    # Ensure minimum dimension for SSIM
    min_dim = min(orig_gray.shape)
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3
    ssim_val = ssim(orig_gray, rest_gray, data_range=255, win_size=win_size)
    
    return {"mse": mse, "psnr": psnr_val, "ssim": ssim_val}

def compute_information_entropy(img):
    """Compute Shannon entropy of image."""
    arr = np.array(img.convert("L"))
    hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def compute_edge_density(img):
    """Compute edge density using Canny edge detection."""
    arr = np.array(img.convert("L"))
    edges = cv2.Canny(arr, 50, 150)
    return np.mean(edges > 0)

def compute_high_freq_energy(img):
    """Compute high-frequency energy ratio using FFT."""
    arr = np.array(img.convert("L")).astype(np.float64)
    f_transform = np.fft.fft2(arr)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    rows, cols = arr.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create mask for high frequencies (outer 75%)
    total_energy = np.sum(magnitude ** 2)
    
    # Low freq mask (center 25%)
    r = min(rows, cols) // 8
    mask = np.zeros((rows, cols), dtype=bool)
    y, x = np.ogrid[:rows, :cols]
    mask[(y - crow)**2 + (x - ccol)**2 <= r**2] = True
    
    low_freq_energy = np.sum(magnitude[mask] ** 2)
    high_freq_energy = total_energy - low_freq_energy
    
    return high_freq_energy / total_energy if total_energy > 0 else 0

# Analyze at multiple resolutions
resolutions = [224, 336, 448, 672, 896]
results = {}

for name, img in [("demo1", demo1), ("demo2", demo2)]:
    results[name] = {}
    orig_entropy = compute_information_entropy(img)
    orig_edge = compute_edge_density(img)
    orig_hf = compute_high_freq_energy(img)
    
    results[name]["original"] = {
        "entropy": orig_entropy,
        "edge_density": orig_edge,
        "high_freq_ratio": orig_hf,
        "resolution": f"{img.size[0]}x{img.size[1]}"
    }
    
    for res in resolutions:
        downscaled, restored = simulate_clip_pipeline(img, res)
        metrics = compute_metrics(img, restored)
        ds_entropy = compute_information_entropy(downscaled)
        ds_edge = compute_edge_density(downscaled)
        ds_hf = compute_high_freq_energy(downscaled)
        
        results[name][f"res_{res}"] = {
            **metrics,
            "entropy_downscaled": ds_entropy,
            "edge_density_downscaled": ds_edge,
            "high_freq_ratio_downscaled": ds_hf,
            "entropy_loss_pct": (orig_entropy - ds_entropy) / orig_entropy * 100,
            "edge_loss_pct": (orig_edge - ds_edge) / orig_edge * 100 if orig_edge > 0 else 0,
            "hf_loss_pct": (orig_hf - ds_hf) / orig_hf * 100 if orig_hf > 0 else 0
        }

# Save results
with open(os.path.join(OUTPUT_DIR, "resolution_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Resolution analysis complete.")
for name in results:
    print(f"\n{name}:")
    print(f"  Original: entropy={results[name]['original']['entropy']:.3f}, "
          f"edge_density={results[name]['original']['edge_density']:.4f}, "
          f"hf_ratio={results[name]['original']['high_freq_ratio']:.4f}")
    for res in resolutions:
        r = results[name][f"res_{res}"]
        print(f"  {res}px: PSNR={r['psnr']:.2f}, SSIM={r['ssim']:.4f}, "
              f"entropy_loss={r['entropy_loss_pct']:.1f}%, "
              f"edge_loss={r['edge_loss_pct']:.1f}%")

# ============================================================
# PART 2: Task-Guided Cropping Strategy
# ============================================================

def generate_saliency_map(img, method='gradient'):
    """Generate a simple saliency map based on gradient magnitude."""
    arr = np.array(img.convert("L")).astype(np.float64)
    
    # Compute gradients
    gx = ndimage.sobel(arr, axis=1)
    gy = ndimage.sobel(arr, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Normalize
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    
    return magnitude

def generate_attention_heatmap(img, question_type='detail'):
    """
    Simulate attention heatmap for a vision encoder.
    Uses a combination of saliency, edge density, and local entropy
    to approximate where a model would attend.
    """
    arr = np.array(img.convert("L")).astype(np.float64)
    h, w = arr.shape
    
    # Component 1: Edge-based saliency
    edges = cv2.Canny(arr.astype(np.uint8), 30, 100).astype(np.float64)
    edges_smooth = ndimage.gaussian_filter(edges, sigma=max(h, w) / 50)
    
    # Component 2: Local entropy (information density)
    from skimage.filters.rank import entropy as rank_entropy
    from skimage.morphology import disk
    arr_uint8 = arr.astype(np.uint8)
    local_ent = rank_entropy(arr_uint8, disk(max(5, min(h, w) // 50)))
    local_ent = local_ent.astype(np.float64)
    
    # Component 3: Color variance (on RGB)
    rgb = np.array(img).astype(np.float64)
    color_var = np.std(rgb, axis=2)
    color_var_smooth = ndimage.gaussian_filter(color_var, sigma=max(h, w) / 50)
    
    # Combine
    edges_norm = (edges_smooth - edges_smooth.min()) / (edges_smooth.max() - edges_smooth.min() + 1e-8)
    ent_norm = (local_ent - local_ent.min()) / (local_ent.max() - local_ent.min() + 1e-8)
    color_norm = (color_var_smooth - color_var_smooth.min()) / (color_var_smooth.max() - color_var_smooth.min() + 1e-8)
    
    if question_type == 'detail':
        # Focus on high-detail regions (small objects)
        heatmap = 0.4 * edges_norm + 0.3 * ent_norm + 0.3 * color_norm
    elif question_type == 'spatial':
        # More uniform with slight center bias
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        center_bias = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * (max(h, w) / 3)**2))
        heatmap = 0.3 * edges_norm + 0.2 * ent_norm + 0.2 * color_norm + 0.3 * center_bias
    else:
        heatmap = 0.33 * edges_norm + 0.33 * ent_norm + 0.34 * color_norm
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap

def identify_roi(heatmap, num_regions=3, min_size_ratio=0.1):
    """Identify regions of interest from attention heatmap."""
    h, w = heatmap.shape
    min_size = int(min(h, w) * min_size_ratio)
    
    # Threshold to get high-attention regions
    threshold = np.percentile(heatmap, 85)
    binary = (heatmap > threshold).astype(np.uint8)
    
    # Connected components
    labeled, num_features = ndimage.label(binary)
    
    regions = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_size * min_size // 4:
            continue
        y1, y2 = max(0, ys.min() - min_size // 4), min(h, ys.max() + min_size // 4)
        x1, x2 = max(0, xs.min() - min_size // 4), min(w, xs.max() + min_size // 4)
        
        # Ensure minimum size
        if (y2 - y1) < min_size:
            cy = (y1 + y2) // 2
            y1 = max(0, cy - min_size // 2)
            y2 = min(h, cy + min_size // 2)
        if (x2 - x1) < min_size:
            cx = (x1 + x2) // 2
            x1 = max(0, cx - min_size // 2)
            x2 = min(w, cx + min_size // 2)
        
        score = np.mean(heatmap[y1:y2, x1:x2])
        regions.append({"bbox": (x1, y1, x2, y2), "score": score})
    
    # Sort by score and return top regions
    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions[:num_regions]

def crop_and_zoom(img, bbox, target_res=448):
    """Crop region and resize to target resolution for detailed analysis."""
    x1, y1, x2, y2 = bbox
    crop = img.crop((x1, y1, x2, y2))
    zoomed = crop.resize((target_res, target_res), Image.LANCZOS)
    return crop, zoomed

# Generate analysis for both images
for name, img in [("demo1", demo1), ("demo2", demo2)]:
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")
    
    # Generate attention heatmap
    heatmap = generate_attention_heatmap(img, 'detail')
    
    # Identify ROIs
    rois = identify_roi(heatmap, num_regions=3)
    print(f"Found {len(rois)} ROIs:")
    for i, roi in enumerate(rois):
        print(f"  ROI {i+1}: bbox={roi['bbox']}, score={roi['score']:.4f}")
    
    # Analyze each ROI
    for i, roi in enumerate(rois):
        crop, zoomed = crop_and_zoom(img, roi['bbox'])
        
        # Compare: original crop at CLIP res vs zoomed crop
        crop_at_clip = crop.resize((224, 224), Image.BILINEAR)
        
        crop_entropy = compute_information_entropy(crop)
        clip_entropy = compute_information_entropy(crop_at_clip)
        zoomed_entropy = compute_information_entropy(zoomed)
        
        crop_edges = compute_edge_density(crop)
        clip_edges = compute_edge_density(crop_at_clip)
        zoomed_edges = compute_edge_density(zoomed)
        
        print(f"\n  ROI {i+1} Analysis:")
        print(f"    Original crop entropy: {crop_entropy:.3f}")
        print(f"    At CLIP-224 entropy:   {clip_entropy:.3f}")
        print(f"    Zoomed-448 entropy:    {zoomed_entropy:.3f}")
        print(f"    Original crop edges:   {crop_edges:.4f}")
        print(f"    At CLIP-224 edges:     {clip_edges:.4f}")
        print(f"    Zoomed-448 edges:      {zoomed_edges:.4f}")

print("\nPart 2 complete.")
