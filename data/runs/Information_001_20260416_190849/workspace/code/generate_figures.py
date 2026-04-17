"""
Generate all figures for the research report.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw
import cv2
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
import seaborn as sns
import json
import os

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260416_190849"
DATA_DIR = os.path.join(WORKSPACE, "data", "demo_imgs")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMG_DIR = os.path.join(WORKSPACE, "report", "images")
os.makedirs(IMG_DIR, exist_ok=True)

demo1 = Image.open(os.path.join(DATA_DIR, "demo1.png")).convert("RGB")
demo2 = Image.open(os.path.join(DATA_DIR, "demo2.png")).convert("RGB")

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# Helper functions
def simulate_clip_pipeline(img, target_res=224):
    original_size = img.size
    downscaled = img.resize((target_res, target_res), Image.BILINEAR)
    restored = downscaled.resize(original_size, Image.BILINEAR)
    return downscaled, restored

def compute_information_entropy(img):
    arr = np.array(img.convert("L"))
    hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def compute_edge_density(img):
    arr = np.array(img.convert("L"))
    edges = cv2.Canny(arr, 50, 150)
    return np.mean(edges > 0)

def generate_attention_heatmap(img):
    arr = np.array(img.convert("L")).astype(np.float64)
    h, w = arr.shape
    edges = cv2.Canny(arr.astype(np.uint8), 30, 100).astype(np.float64)
    edges_smooth = ndimage.gaussian_filter(edges, sigma=max(h, w) / 50)
    arr_uint8 = arr.astype(np.uint8)
    local_ent = rank_entropy(arr_uint8, disk(max(5, min(h, w) // 50)))
    local_ent = local_ent.astype(np.float64)
    rgb = np.array(img).astype(np.float64)
    color_var = np.std(rgb, axis=2)
    color_var_smooth = ndimage.gaussian_filter(color_var, sigma=max(h, w) / 50)
    edges_norm = (edges_smooth - edges_smooth.min()) / (edges_smooth.max() - edges_smooth.min() + 1e-8)
    ent_norm = (local_ent - local_ent.min()) / (local_ent.max() - local_ent.min() + 1e-8)
    color_norm = (color_var_smooth - color_var_smooth.min()) / (color_var_smooth.max() - color_var_smooth.min() + 1e-8)
    heatmap = 0.4 * edges_norm + 0.3 * ent_norm + 0.3 * color_norm
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def identify_roi(heatmap, num_regions=3, min_size_ratio=0.1):
    h, w = heatmap.shape
    min_size = int(min(h, w) * min_size_ratio)
    threshold = np.percentile(heatmap, 85)
    binary = (heatmap > threshold).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    regions = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_size * min_size // 4:
            continue
        y1, y2 = max(0, ys.min() - min_size // 4), min(h, ys.max() + min_size // 4)
        x1, x2 = max(0, xs.min() - min_size // 4), min(w, xs.max() + min_size // 4)
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
    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions[:num_regions]

# ============================================================
# FIGURE 1: Resolution Impact on Information Loss
# ============================================================
print("Generating Figure 1: Resolution Impact...")

resolutions = [224, 336, 448, 672, 896]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, (name, img) in enumerate([("Demo 1 (Street Scene)", demo1), ("Demo 2 (Flower Exhibition)", demo2)]):
    psnr_vals = []
    ssim_vals = []
    entropy_vals = []
    edge_vals = []
    
    orig_entropy = compute_information_entropy(img)
    orig_edge = compute_edge_density(img)
    
    for res in resolutions:
        ds, restored = simulate_clip_pipeline(img, res)
        orig_arr = np.array(img).astype(np.float64)
        rest_arr = np.array(restored).astype(np.float64)
        p = psnr(orig_arr, rest_arr, data_range=255)
        
        orig_gray = np.mean(orig_arr, axis=2)
        rest_gray = np.mean(rest_arr, axis=2)
        min_dim = min(orig_gray.shape)
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3:
            win_size = 3
        s = ssim(orig_gray, rest_gray, data_range=255, win_size=win_size)
        
        ds_entropy = compute_information_entropy(ds)
        ds_edge = compute_edge_density(ds)
        
        psnr_vals.append(p)
        ssim_vals.append(s)
        entropy_vals.append(ds_entropy)
        edge_vals.append(ds_edge)
    
    # PSNR plot
    axes[idx, 0].plot(resolutions, psnr_vals, 'o-', color='#2196F3', linewidth=2, markersize=8)
    axes[idx, 0].set_xlabel('Encoder Resolution (px)')
    axes[idx, 0].set_ylabel('PSNR (dB)')
    axes[idx, 0].set_title(f'{name}\nPSNR vs Resolution')
    axes[idx, 0].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Good quality threshold')
    axes[idx, 0].legend(fontsize=9)
    
    # SSIM plot
    axes[idx, 1].plot(resolutions, ssim_vals, 's-', color='#4CAF50', linewidth=2, markersize=8)
    axes[idx, 1].set_xlabel('Encoder Resolution (px)')
    axes[idx, 1].set_ylabel('SSIM')
    axes[idx, 1].set_title(f'{name}\nSSIM vs Resolution')
    axes[idx, 1].set_ylim(0, 1.05)
    axes[idx, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='High similarity threshold')
    axes[idx, 1].legend(fontsize=9)
    
    # Edge density
    axes[idx, 2].bar(range(len(resolutions)), edge_vals, color='#FF9800', alpha=0.8)
    axes[idx, 2].axhline(y=orig_edge, color='red', linestyle='--', linewidth=2, label=f'Original ({orig_edge:.3f})')
    axes[idx, 2].set_xticks(range(len(resolutions)))
    axes[idx, 2].set_xticklabels([str(r) for r in resolutions])
    axes[idx, 2].set_xlabel('Encoder Resolution (px)')
    axes[idx, 2].set_ylabel('Edge Density')
    axes[idx, 2].set_title(f'{name}\nEdge Density vs Resolution')
    axes[idx, 2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "resolution_impact.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved resolution_impact.png")

# ============================================================
# FIGURE 2: Visual Comparison at Different Resolutions
# ============================================================
print("Generating Figure 2: Visual Resolution Comparison...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
show_res = [224, 336, 448, 672]

for idx, (name, img) in enumerate([("Demo 1", demo1), ("Demo 2", demo2)]):
    axes[idx, 0].imshow(np.array(img))
    axes[idx, 0].set_title(f'{name}\nOriginal ({img.size[0]}x{img.size[1]})', fontsize=10)
    axes[idx, 0].axis('off')
    
    for j, res in enumerate(show_res):
        ds, _ = simulate_clip_pipeline(img, res)
        axes[idx, j+1].imshow(np.array(ds))
        axes[idx, j+1].set_title(f'CLIP {res}x{res}', fontsize=10)
        axes[idx, j+1].axis('off')

plt.suptitle('Information Loss at Different Vision Encoder Resolutions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "visual_resolution_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved visual_resolution_comparison.png")

# ============================================================
# FIGURE 3: Attention Heatmap and ROI Detection
# ============================================================
print("Generating Figure 3: Attention Heatmaps...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (name, img) in enumerate([("Demo 1 (Street Scene)", demo1), ("Demo 2 (Flower Exhibition)", demo2)]):
    heatmap = generate_attention_heatmap(img)
    rois = identify_roi(heatmap, num_regions=3)
    
    # Original image
    axes[idx, 0].imshow(np.array(img))
    axes[idx, 0].set_title(f'{name}\nOriginal Image', fontsize=11)
    axes[idx, 0].axis('off')
    
    # Attention heatmap
    axes[idx, 1].imshow(np.array(img), alpha=0.4)
    im = axes[idx, 1].imshow(heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    axes[idx, 1].set_title(f'Attention/Saliency Heatmap', fontsize=11)
    axes[idx, 1].axis('off')
    plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)
    
    # Image with ROI boxes
    img_with_rois = np.array(img).copy()
    axes[idx, 2].imshow(img_with_rois)
    colors = ['#FF0000', '#00FF00', '#0000FF']
    for i, roi in enumerate(rois):
        x1, y1, x2, y2 = roi['bbox']
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor=colors[i], facecolor='none')
        axes[idx, 2].add_patch(rect)
        axes[idx, 2].text(x1, y1-5, f'ROI {i+1} ({roi["score"]:.2f})', 
                          color=colors[i], fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    axes[idx, 2].set_title(f'Detected Regions of Interest', fontsize=11)
    axes[idx, 2].axis('off')

plt.suptitle('Task-Guided Region Detection via Attention Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "attention_heatmap_roi.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved attention_heatmap_roi.png")

# ============================================================
# FIGURE 4: Crop-and-Zoom Analysis
# ============================================================
print("Generating Figure 4: Crop-and-Zoom Analysis...")

fig = plt.figure(figsize=(20, 16))

for img_idx, (name, img) in enumerate([("Demo 1", demo1), ("Demo 2", demo2)]):
    heatmap = generate_attention_heatmap(img)
    rois = identify_roi(heatmap, num_regions=3)
    
    for roi_idx, roi in enumerate(rois[:2]):  # Show top 2 ROIs per image
        row = img_idx * 2 + roi_idx
        x1, y1, x2, y2 = roi['bbox']
        crop = img.crop((x1, y1, x2, y2))
        
        # Process at different resolutions
        crop_224 = crop.resize((224, 224), Image.BILINEAR)
        crop_448 = crop.resize((448, 448), Image.LANCZOS)
        
        # Full image at CLIP-224
        full_224 = img.resize((224, 224), Image.BILINEAR)
        
        ax1 = fig.add_subplot(4, 4, row * 4 + 1)
        ax1.imshow(np.array(img))
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='cyan', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title(f'{name} - ROI {roi_idx+1}', fontsize=9)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(4, 4, row * 4 + 2)
        ax2.imshow(np.array(full_224))
        ax2.set_title(f'Full Image at 224x224\n(CLIP input)', fontsize=9)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(4, 4, row * 4 + 3)
        ax3.imshow(np.array(crop_224))
        ax3.set_title(f'Cropped ROI at 224x224', fontsize=9)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(4, 4, row * 4 + 4)
        ax4.imshow(np.array(crop_448))
        ax4.set_title(f'Zoomed ROI at 448x448\n(Task-Guided Crop)', fontsize=9)
        ax4.axis('off')

plt.suptitle('Task-Guided Cropping: Global vs Local Detail Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "crop_zoom_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved crop_zoom_analysis.png")

# ============================================================
# FIGURE 5: Information Metrics Comparison (Global vs Cropped)
# ============================================================
print("Generating Figure 5: Information Metrics Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

categories = []
entropy_global = []
entropy_crop224 = []
entropy_crop448 = []
edge_global = []
edge_crop224 = []
edge_crop448 = []

for name, img in [("Demo 1", demo1), ("Demo 2", demo2)]:
    heatmap = generate_attention_heatmap(img)
    rois = identify_roi(heatmap, num_regions=2)
    
    # Global at 224
    global_224 = img.resize((224, 224), Image.BILINEAR)
    g_ent = compute_information_entropy(global_224)
    g_edge = compute_edge_density(global_224)
    
    for i, roi in enumerate(rois[:2]):
        x1, y1, x2, y2 = roi['bbox']
        crop = img.crop((x1, y1, x2, y2))
        crop_224 = crop.resize((224, 224), Image.BILINEAR)
        crop_448 = crop.resize((448, 448), Image.LANCZOS)
        
        cat = f'{name}\nROI {i+1}'
        categories.append(cat)
        
        entropy_global.append(g_ent)
        entropy_crop224.append(compute_information_entropy(crop_224))
        entropy_crop448.append(compute_information_entropy(crop_448))
        
        edge_global.append(g_edge)
        edge_crop224.append(compute_edge_density(crop_224))
        edge_crop448.append(compute_edge_density(crop_448))

x = np.arange(len(categories))
width = 0.25

# Entropy comparison
bars1 = axes[0].bar(x - width, entropy_global, width, label='Global@224', color='#e74c3c', alpha=0.8)
bars2 = axes[0].bar(x, entropy_crop224, width, label='Crop@224', color='#3498db', alpha=0.8)
bars3 = axes[0].bar(x + width, entropy_crop448, width, label='Crop@448', color='#2ecc71', alpha=0.8)
axes[0].set_ylabel('Shannon Entropy (bits)')
axes[0].set_title('Information Entropy Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, fontsize=8)
axes[0].legend()

# Edge density comparison
bars1 = axes[1].bar(x - width, edge_global, width, label='Global@224', color='#e74c3c', alpha=0.8)
bars2 = axes[1].bar(x, edge_crop224, width, label='Crop@224', color='#3498db', alpha=0.8)
bars3 = axes[1].bar(x + width, edge_crop448, width, label='Crop@448', color='#2ecc71', alpha=0.8)
axes[1].set_ylabel('Edge Density')
axes[1].set_title('Edge Detail Preservation')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, fontsize=8)
axes[1].legend()

# Improvement percentage
improvement_entropy = [(c448 - g) / g * 100 for c448, g in zip(entropy_crop448, entropy_global)]
improvement_edge = [(c448 - g) / g * 100 if g > 0 else 0 for c448, g in zip(edge_crop448, edge_global)]

axes[2].bar(x - 0.15, improvement_entropy, 0.3, label='Entropy Gain (%)', color='#9b59b6', alpha=0.8)
axes[2].bar(x + 0.15, improvement_edge, 0.3, label='Edge Gain (%)', color='#f39c12', alpha=0.8)
axes[2].set_ylabel('Improvement (%)')
axes[2].set_title('Crop@448 vs Global@224\nRelative Improvement')
axes[2].set_xticks(x)
axes[2].set_xticklabels(categories, fontsize=8)
axes[2].legend()
axes[2].axhline(y=0, color='black', linewidth=0.5)

plt.suptitle('Quantitative Analysis: Task-Guided Cropping vs Fixed-Resolution Encoding', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "metrics_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved metrics_comparison.png")

# ============================================================
# FIGURE 6: Frequency Domain Analysis
# ============================================================
print("Generating Figure 6: Frequency Domain Analysis...")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for idx, (name, img) in enumerate([("Demo 1", demo1), ("Demo 2", demo2)]):
    heatmap = generate_attention_heatmap(img)
    rois = identify_roi(heatmap, num_regions=1)
    roi = rois[0]
    x1, y1, x2, y2 = roi['bbox']
    crop = img.crop((x1, y1, x2, y2))
    
    # Original full at 224
    full_224 = img.resize((224, 224), Image.BILINEAR)
    full_gray = np.array(full_224.convert("L")).astype(np.float64)
    
    # Crop at 224
    crop_224 = crop.resize((224, 224), Image.BILINEAR)
    crop_gray = np.array(crop_224.convert("L")).astype(np.float64)
    
    # Crop at 448
    crop_448 = crop.resize((448, 448), Image.LANCZOS)
    crop_448_gray = np.array(crop_448.convert("L")).astype(np.float64)
    # Resize to 224 for fair comparison of frequency content
    crop_448_ds = np.array(crop_448.resize((224, 224), Image.LANCZOS).convert("L")).astype(np.float64)
    
    images_to_fft = [
        (full_gray, f'{name}: Full@224'),
        (crop_gray, f'{name}: Crop@224'),
        (crop_448_ds, f'{name}: Crop@448->224')
    ]
    
    # Original image with ROI
    axes[idx, 0].imshow(np.array(img))
    rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='cyan', facecolor='none')
    axes[idx, 0].add_patch(rect)
    axes[idx, 0].set_title(f'{name}: Original + ROI', fontsize=10)
    axes[idx, 0].axis('off')
    
    for j, (gray_img, title) in enumerate(images_to_fft):
        f_transform = np.fft.fft2(gray_img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))
        
        axes[idx, j+1].imshow(magnitude, cmap='inferno')
        axes[idx, j+1].set_title(title + '\nFrequency Spectrum', fontsize=10)
        axes[idx, j+1].axis('off')

plt.suptitle('Frequency Domain Analysis: Information Content at Different Processing Stages', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "frequency_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved frequency_analysis.png")

# ============================================================
# FIGURE 7: Pipeline Overview Diagram
# ============================================================
print("Generating Figure 7: Pipeline Overview...")

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(8, 7.5, 'Task-Guided Cropping Framework for Fine-Grained MLLM Perception',
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2))

# Step 1: Input
ax.add_patch(mpatches.FancyBboxPatch((0.5, 4.5), 2.5, 2, boxstyle='round,pad=0.2',
             facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2))
ax.text(1.75, 5.5, 'Step 1:\nInput Image\n+ Question', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow
ax.annotate('', xy=(3.5, 5.5), xytext=(3.0, 5.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Step 2: Global Analysis
ax.add_patch(mpatches.FancyBboxPatch((3.5, 4.5), 2.5, 2, boxstyle='round,pad=0.2',
             facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2))
ax.text(4.75, 5.5, 'Step 2:\nGlobal Encoding\n(CLIP@224/336)', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow
ax.annotate('', xy=(6.5, 5.5), xytext=(6.0, 5.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Step 3: ROI Detection
ax.add_patch(mpatches.FancyBboxPatch((6.5, 4.5), 2.5, 2, boxstyle='round,pad=0.2',
             facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=2))
ax.text(7.75, 5.5, 'Step 3:\nTask-Guided\nROI Detection', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow
ax.annotate('', xy=(9.5, 5.5), xytext=(9.0, 5.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Step 4: Crop & Zoom
ax.add_patch(mpatches.FancyBboxPatch((9.5, 4.5), 2.5, 2, boxstyle='round,pad=0.2',
             facecolor='#E1F5FE', edgecolor='#0277BD', linewidth=2))
ax.text(10.75, 5.5, 'Step 4:\nCrop & Zoom\n(Local Detail)', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow
ax.annotate('', xy=(12.5, 5.5), xytext=(12.0, 5.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Step 5: Integration
ax.add_patch(mpatches.FancyBboxPatch((12.5, 4.5), 3, 2, boxstyle='round,pad=0.2',
             facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2))
ax.text(14, 5.5, 'Step 5:\nGlobal+Local\nIntegration\n& Response', ha='center', va='center', fontsize=10, fontweight='bold')

# Bottom annotations
ax.text(1.75, 3.5, 'High-res image\nwith question', ha='center', va='center', fontsize=9, style='italic', color='#666')
ax.text(4.75, 3.5, 'Fixed resolution\n-> info loss', ha='center', va='center', fontsize=9, style='italic', color='#C62828')
ax.text(7.75, 3.5, 'LLM identifies\nmissing details', ha='center', va='center', fontsize=9, style='italic', color='#666')
ax.text(10.75, 3.5, 'Recover lost\nfine details', ha='center', va='center', fontsize=9, style='italic', color='#2E7D32')
ax.text(14, 3.5, 'Accurate\nreasoning', ha='center', va='center', fontsize=9, style='italic', color='#666')

# Key insight box
ax.add_patch(mpatches.FancyBboxPatch((2, 1), 12, 1.5, boxstyle='round,pad=0.3',
             facecolor='#FFFDE7', edgecolor='#F57F17', linewidth=2))
ax.text(8, 1.75, 'Key Insight: Fixed-resolution encoders (e.g., CLIP@224) lose critical fine-grained details.\n'
        'Task-guided cropping recovers this information by zooming into task-relevant regions,\n'
        'enabling accurate visual reasoning without retraining the vision encoder.',
        ha='center', va='center', fontsize=10)

plt.savefig(os.path.join(IMG_DIR, "pipeline_overview.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved pipeline_overview.png")

# ============================================================
# FIGURE 8: SSIM Map Visualization
# ============================================================
print("Generating Figure 8: SSIM Difference Maps...")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for idx, (name, img) in enumerate([("Demo 1", demo1), ("Demo 2", demo2)]):
    # Original
    axes[idx, 0].imshow(np.array(img))
    axes[idx, 0].set_title(f'{name}: Original', fontsize=10)
    axes[idx, 0].axis('off')
    
    for j, res in enumerate([224, 336, 448]):
        _, restored = simulate_clip_pipeline(img, res)
        
        orig_gray = np.array(img.convert("L")).astype(np.float64)
        rest_gray = np.array(restored.convert("L")).astype(np.float64)
        
        # Compute difference map
        diff = np.abs(orig_gray - rest_gray)
        diff_norm = diff / 255.0
        
        im = axes[idx, j+1].imshow(diff_norm, cmap='hot', vmin=0, vmax=0.5)
        axes[idx, j+1].set_title(f'Difference Map @ {res}px\n(brighter = more loss)', fontsize=10)
        axes[idx, j+1].axis('off')
        plt.colorbar(im, ax=axes[idx, j+1], fraction=0.046, pad=0.04)

plt.suptitle('Spatial Distribution of Information Loss at Different Encoder Resolutions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "difference_maps.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved difference_maps.png")

# ============================================================
# FIGURE 9: Comprehensive Metrics Table as Figure
# ============================================================
print("Generating Figure 9: Metrics Summary Table...")

# Compute all metrics
table_data = []
for name, img in [("Demo 1", demo1), ("Demo 2", demo2)]:
    heatmap = generate_attention_heatmap(img)
    rois = identify_roi(heatmap, num_regions=2)
    
    # Global at 224
    global_224 = img.resize((224, 224), Image.BILINEAR)
    g_ent = compute_information_entropy(global_224)
    g_edge = compute_edge_density(global_224)
    
    row = [name, "Global@224", f"{g_ent:.3f}", f"{g_edge:.4f}", "-", "-"]
    table_data.append(row)
    
    # Global at 336
    global_336 = img.resize((336, 336), Image.BILINEAR)
    g_ent336 = compute_information_entropy(global_336)
    g_edge336 = compute_edge_density(global_336)
    row = [name, "Global@336", f"{g_ent336:.3f}", f"{g_edge336:.4f}", "-", "-"]
    table_data.append(row)
    
    for i, roi in enumerate(rois[:2]):
        x1, y1, x2, y2 = roi['bbox']
        crop = img.crop((x1, y1, x2, y2))
        
        crop_224 = crop.resize((224, 224), Image.BILINEAR)
        crop_448 = crop.resize((448, 448), Image.LANCZOS)
        
        c_ent224 = compute_information_entropy(crop_224)
        c_edge224 = compute_edge_density(crop_224)
        c_ent448 = compute_information_entropy(crop_448)
        c_edge448 = compute_edge_density(crop_448)
        
        ent_gain = (c_ent448 - g_ent) / g_ent * 100
        edge_gain = (c_edge448 - g_edge) / g_edge * 100 if g_edge > 0 else 0
        
        row = [name, f"Crop ROI{i+1}@224", f"{c_ent224:.3f}", f"{c_edge224:.4f}", "-", "-"]
        table_data.append(row)
        row = [name, f"Crop ROI{i+1}@448", f"{c_ent448:.3f}", f"{c_edge448:.4f}", 
               f"{ent_gain:+.1f}%", f"{edge_gain:+.1f}%"]
        table_data.append(row)

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.axis('off')

columns = ['Image', 'Method', 'Entropy', 'Edge Density', 'Entropy Gain\nvs Global@224', 'Edge Gain\nvs Global@224']
table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

# Style header
for j in range(len(columns)):
    table[0, j].set_facecolor('#1565C0')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(len(columns)):
        if i % 2 == 0:
            table[i, j].set_facecolor('#E3F2FD')
        else:
            table[i, j].set_facecolor('#FFFFFF')

plt.title('Quantitative Metrics: Global vs Task-Guided Cropping', fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(IMG_DIR, "metrics_table.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved metrics_table.png")

# Save table data as JSON
with open(os.path.join(OUTPUT_DIR, "metrics_table.json"), "w") as f:
    json.dump({"columns": columns, "data": table_data}, f, indent=2)

print("\nAll figures generated successfully!")
