"""
Analysis script: Demonstrating the resolution bottleneck in fixed-resolution vision encoders
and the effectiveness of task-guided cropping for fine-grained perception.

This script:
1. Shows information loss when downsampling images to CLIP-style resolutions (224x224, 336x336)
2. Simulates task-guided cropping (zoom) strategy on demo images
3. Quantifies detail retention at different scales
4. Generates comparison figures
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import json

# Paths
WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_001_20260415_214024"
DATA_DIR = os.path.join(WORKSPACE, "data/demo_imgs")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
FIGURE_DIR = os.path.join(WORKSPACE, "report/images")
CODE_DIR = os.path.join(WORKSPACE, "code")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Load demo images
demo1_path = os.path.join(DATA_DIR, "demo1.png")
demo2_path = os.path.join(DATA_DIR, "demo2.png")
method_case_path = os.path.join(DATA_DIR, "method_case.png")

img_demo1 = Image.open(demo1_path).convert('RGB')
img_demo2 = Image.open(demo2_path).convert('RGB')
img_method = Image.open(method_case_path).convert('RGB')

print(f"Demo1 size: {img_demo1.size}")
print(f"Demo2 size: {img_demo2.size}")
print(f"Method case size: {img_method.size}")

# ============================================================
# 1. Resolution Bottleneck Analysis
# ============================================================

def compute_resolution_metrics(img, target_sizes):
    """Compute metrics showing information loss at different resolutions."""
    img_np = np.array(img)
    results = []
    
    for target_size in target_sizes:
        # Downsample to target resolution then upsample back
        small_img = img.resize((target_size, target_size), Image.LANCZOS)
        # Upsample back to original size for comparison
        reconstructed = small_img.resize(img.size, Image.LANCZOS)
        recon_np = np.array(reconstructed)
        
        # Compute SSIM between original and reconstructed
        # Need same size - use min of both dimensions for fair comparison
        h_min = min(img_np.shape[0], recon_np.shape[0])
        w_min = min(img_np.shape[1], recon_np.shape[1])
        
        orig_crop = img_np[:h_min, :w_min]
        recon_crop = recon_np[:h_min, :w_min]
        
        ssim_val = ssim(orig_crop, recon_crop, channel_axis=2)
        
        # Compute MSE
        mse = np.mean((orig_crop.astype(float) - recon_crop.astype(float))**2)
        
        # Compute PSNR
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
        else:
            psnr = float('inf')
        
        # Pixel count ratio
        pixel_ratio = (target_size * target_size) / (img.size[0] * img.size[1])
        
        results.append({
            'target_size': target_size,
            'ssim': ssim_val,
            'mse': mse,
            'psnr': psnr,
            'pixel_ratio': pixel_ratio,
            'original_pixels': img.size[0] * img.size[1],
            'downsampled_pixels': target_size * target_size
        })
    
    return results

# Common CLIP-style resolutions
target_sizes = [224, 336, 448, 672, 896]

# Compute metrics for both demo images
metrics_demo1 = compute_resolution_metrics(img_demo1, target_sizes)
metrics_demo2 = compute_resolution_metrics(img_demo2, target_sizes)

print("\n=== Resolution Metrics for Demo1 ===")
for m in metrics_demo1:
    print(f"Size {m['target_size']}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f}, PixelRatio={m['pixel_ratio']:.4f}")

print("\n=== Resolution Metrics for Demo2 ===")
for m in metrics_demo2:
    print(f"Size {m['target_size']}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f}, PixelRatio={m['pixel_ratio']:.4f}")

# Save metrics to JSON
all_metrics = {
    'demo1': metrics_demo1,
    'demo2': metrics_demo2,
    'target_sizes': target_sizes,
    'demo1_original_size': list(img_demo1.size),
    'demo2_original_size': list(img_demo2.size)
}
with open(os.path.join(OUTPUT_DIR, 'resolution_metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

# ============================================================
# 2. Figure: Resolution Comparison (Downsampling Effect)
# ============================================================

def create_resolution_comparison_figure(img, name, target_sizes, save_path):
    """Create a figure showing progressive information loss at different resolutions."""
    n_cols = len(target_sizes) + 1  # +1 for original
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    
    # Original image
    axes[0].imshow(np.array(img))
    axes[0].set_title(f'Original\n{img.size[0]}×{img.size[1]}', fontsize=12)
    axes[0].axis('off')
    
    # Downsampled versions
    for i, ts in enumerate(target_sizes):
        small_img = img.resize((ts, ts), Image.LANCZOS)
        axes[i+1].imshow(np.array(small_img))
        axes[i+1].set_title(f'CLIP {ts}×{ts}\n{(ts*ts)/(img.size[0]*img.size[1])*100:.1f}% pixels', fontsize=11)
        axes[i+1].axis('off')
    
    fig.suptitle(f'Resolution Bottleneck: Information Loss in Fixed-Resolution Encoders ({name})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

create_resolution_comparison_figure(img_demo1, "Demo1", [224, 336, 448], 
                                    os.path.join(FIGURE_DIR, "resolution_comparison_demo1.png"))
create_resolution_comparison_figure(img_demo2, "Demo2", [224, 336, 448], 
                                    os.path.join(FIGURE_DIR, "resolution_comparison_demo2.png"))

# ============================================================
# 3. Figure: SSIM/PSNR vs Resolution Plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SSIM plot
sizes = [m['target_size'] for m in metrics_demo1]
ssim_vals_1 = [m['ssim'] for m in metrics_demo1]
ssim_vals_2 = [m['ssim'] for m in metrics_demo2]

axes[0].plot(sizes, ssim_vals_1, 'o-', label='Demo1 (1024×768)', color='#2196F3', linewidth=2, markersize=8)
axes[0].plot(sizes, ssim_vals_2, 's-', label='Demo2 (2250×1500)', color='#FF5722', linewidth=2, markersize=8)
axes[0].axvline(x=224, color='gray', linestyle='--', alpha=0.7, label='CLIP ViT-L/14')
axes[0].axvline(x=336, color='green', linestyle='--', alpha=0.7, label='CLIP ViT-L/336')
axes[0].set_xlabel('Encoder Input Resolution', fontsize=12)
axes[0].set_ylabel('SSIM (vs Original)', fontsize=12)
axes[0].set_title('Structural Similarity vs Encoder Resolution', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1.05)

# PSNR plot
psnr_vals_1 = [m['psnr'] for m in metrics_demo1]
psnr_vals_2 = [m['psnr'] for m in metrics_demo2]

axes[1].plot(sizes, psnr_vals_1, 'o-', label='Demo1 (1024×768)', color='#2196F3', linewidth=2, markersize=8)
axes[1].plot(sizes, psnr_vals_2, 's-', label='Demo2 (2250×1500)', color='#FF5722', linewidth=2, markersize=8)
axes[1].axvline(x=224, color='gray', linestyle='--', alpha=0.7, label='CLIP ViT-L/14')
axes[1].axvline(x=336, color='green', linestyle='--', alpha=0.7, label='CLIP ViT-L/336')
axes[1].set_xlabel('Encoder Input Resolution', fontsize=12)
axes[1].set_ylabel('PSNR (dB)', fontsize=12)
axes[1].set_title('Peak Signal-to-Noise Ratio vs Encoder Resolution', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "quality_vs_resolution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: quality_vs_resolution.png")

# ============================================================
# 4. Task-Guided Cropping Simulation
# ============================================================

def simulate_task_guided_crop(img, crop_regions, crop_size=448):
    """Simulate the task-guided cropping strategy.
    
    Args:
        img: PIL Image
        crop_regions: list of (x, y, w, h) tuples defining regions of interest
        crop_size: target crop size (matching encoder input resolution)
    
    Returns:
        List of cropped PIL Images at crop_size resolution
    """
    crops = []
    for (x, y, w, h) in crop_regions:
        # Crop the region from the original high-res image
        crop = img.crop((x, y, x+w, y+h))
        # Resize to encoder input resolution
        crop_resized = crop.resize((crop_size, crop_size), Image.LANCZOS)
        crops.append(crop_resized)
    return crops

# Define simulated regions of interest for demo images
# These represent what a task-guided search would identify
crop_regions_demo1 = [
    (100, 200, 300, 300),   # Region with small text/detail
    (400, 100, 250, 250),   # Another region of interest
    (600, 400, 200, 200),   # Small object region
]

crop_regions_demo2 = [
    (200, 300, 400, 400),   # Region with fine detail
    (800, 200, 350, 350),   # Another region of interest
    (1200, 500, 300, 300),  # Small object/text region
]

crops_demo1 = simulate_task_guided_crop(img_demo1, crop_regions_demo1, crop_size=448)
crops_demo2 = simulate_task_guided_crop(img_demo2, crop_regions_demo2, crop_size=448)

# ============================================================
# 5. Figure: Task-Guided Cropping Visualization
# ============================================================

def create_cropping_visualization(img, crop_regions, crops, name, save_path):
    """Create a figure showing the global image with crop boxes and the zoomed crops."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, len(crop_regions)+1, height_ratios=[1.5, 1], 
                           hspace=0.3, wspace=0.2)
    
    # Top row: Global image with crop boxes
    ax_global = fig.add_subplot(gs[0, :])
    ax_global.imshow(np.array(img))
    
    colors = ['#FF5722', '#4CAF50', '#2196F3']
    for i, (x, y, w, h) in enumerate(crop_regions):
        rect = Rectangle((x, y), w, h, linewidth=3, edgecolor=colors[i], 
                         facecolor='none', linestyle='-')
        ax_global.add_patch(rect)
        ax_global.annotate(f'Crop {i+1}', (x+w/2, y-15), fontsize=11, 
                          color=colors[i], fontweight='bold', ha='center')
    
    ax_global.set_title(f'Task-Guided Cropping Strategy: Global View with ROI Boxes ({name})', 
                       fontsize=14, fontweight='bold')
    ax_global.axis('off')
    
    # Bottom row: Zoomed crops + downsampled global comparison
    for i, crop in enumerate(crops):
        ax_crop = fig.add_subplot(gs[1, i])
        ax_crop.imshow(np.array(crop))
        ax_crop.set_title(f'Zoomed Crop {i+1}\n(448×448)', fontsize=11, color=colors[i])
        ax_crop.axis('off')
    
    # Also show the 224x224 downsampled global for comparison
    ax_downsampled = fig.add_subplot(gs[1, len(crop_regions)])
    global_224 = img.resize((224, 224), Image.LANCZOS)
    ax_downsampled.imshow(np.array(global_224))
    ax_downsampled.set_title('Global @ 224×224\n(CLIP Standard)', fontsize=11, color='gray')
    ax_downsampled.axis('off')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

create_cropping_visualization(img_demo1, crop_regions_demo1, crops_demo1, "Demo1",
                              os.path.join(FIGURE_DIR, "task_guided_cropping_demo1.png"))
create_cropping_visualization(img_demo2, crop_regions_demo2, crops_demo2, "Demo2",
                              os.path.join(FIGURE_DIR, "task_guided_cropping_demo2.png"))

# ============================================================
# 6. Figure: Detail Preservation Comparison (Crop vs Global Downsample)
# ============================================================

def create_detail_comparison(img, crop_region, crop_size=448, name="", save_path=None):
    """Compare detail preservation: local crop at high-res vs same region in downsampled global."""
    x, y, w, h = crop_region
    
    # Method A: Crop from original then resize to 448
    crop_from_original = img.crop((x, y, x+w, y+h)).resize((crop_size, crop_size), Image.LANCZOS)
    
    # Method B: Downsample entire image to 224 first, then extract same relative region
    scale_x = 224 / img.size[0]
    scale_y = 224 / img.size[1]
    rx, ry = int(x * scale_x), int(y * scale_y)
    rw, rh = max(int(w * scale_x), 1), max(int(h * scale_y), 1)
    global_224 = img.resize((224, 224), Image.LANCZOS)
    region_from_downsampled = global_224.crop((rx, ry, rx+rw, ry+rh)).resize((crop_size, crop_size), Image.LANCZOS)
    
    # Method C: Downsample entire image to 336 first, then extract same relative region
    scale_x3 = 336 / img.size[0]
    scale_y3 = 336 / img.size[1]
    rx3, ry3 = int(x * scale_x3), int(y * scale_y3)
    rw3, rh3 = max(int(w * scale_x3), 1), max(int(h * scale_y3), 1)
    global_336 = img.resize((336, 336), Image.LANCZOS)
    region_from_336 = global_336.crop((rx3, ry3, rx3+rw3, ry3+rh3)).resize((crop_size, crop_size), Image.LANCZOS)
    
    # Compute SSIM between crop_from_original and each method
    crop_arr = np.array(crop_from_original)
    region_224_arr = np.array(region_from_downsampled)
    region_336_arr = np.array(region_from_336)
    
    ssim_224 = ssim(crop_arr, region_224_arr, channel_axis=2)
    ssim_336 = ssim(crop_arr, region_336_arr, channel_axis=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(crop_arr)
    axes[0].set_title(f'Task-Guided Crop\n(High-res → 448×448)\nReference', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(region_224_arr)
    axes[1].set_title(f'Global Downsample → Extract\n(224×224 → Crop)\nSSIM={ssim_224:.3f}', fontsize=12, color='red')
    axes[1].axis('off')
    
    axes[2].imshow(region_336_arr)
    axes[2].set_title(f'Global Downsample → Extract\n(336×336 → Crop)\nSSIM={ssim_336:.3f}', fontsize=12, color='orange')
    axes[2].axis('off')
    
    fig.suptitle(f'Detail Preservation: Task-Guided Crop vs Global Downsample ({name})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'ssim_224': ssim_224, 'ssim_336': ssim_336}

detail_metrics_demo1 = []
for i, region in enumerate(crop_regions_demo1):
    dm = create_detail_comparison(img_demo1, region, name=f"Demo1-Region{i+1}",
                                  save_path=os.path.join(FIGURE_DIR, f"detail_comparison_demo1_region{i+1}.png"))
    detail_metrics_demo1.append(dm)

detail_metrics_demo2 = []
for i, region in enumerate(crop_regions_demo2):
    dm = create_detail_comparison(img_demo2, region, name=f"Demo2-Region{i+1}",
                                  save_path=os.path.join(FIGURE_DIR, f"detail_comparison_demo2_region{i+1}.png"))
    detail_metrics_demo2.append(dm)

# Save detail metrics
with open(os.path.join(OUTPUT_DIR, 'detail_preservation_metrics.json'), 'w') as f:
    json.dump({
        'demo1': detail_metrics_demo1,
        'demo2': detail_metrics_demo2
    }, f, indent=2)

# ============================================================
# 7. Figure: Bar Chart - Detail Preservation Across Methods
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Demo1
labels_d1 = [f'Region {i+1}' for i in range(len(detail_metrics_demo1))]
ssim_224_d1 = [m['ssim_224'] for m in detail_metrics_demo1]
ssim_336_d1 = [m['ssim_336'] for m in detail_metrics_demo1]

x = np.arange(len(labels_d1))
width = 0.35

bars1 = axes[0].bar(x - width/2, ssim_224_d1, width, label='Global 224→Crop', color='#FF5722', alpha=0.8)
bars2 = axes[0].bar(x + width/2, ssim_336_d1, width, label='Global 336→Crop', color='#FF9800', alpha=0.8)
axes[0].axhline(y=1.0, color='#4CAF50', linestyle='--', linewidth=2, label='Task-Guided Crop (Reference)')
axes[0].set_ylabel('SSIM vs Task-Guided Crop', fontsize=12)
axes[0].set_title('Demo1: Detail Preservation Comparison', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_d1)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, 1.15)

# Demo2
labels_d2 = [f'Region {i+1}' for i in range(len(detail_metrics_demo2))]
ssim_224_d2 = [m['ssim_224'] for m in detail_metrics_demo2]
ssim_336_d2 = [m['ssim_336'] for m in detail_metrics_demo2]

x2 = np.arange(len(labels_d2))
bars3 = axes[1].bar(x2 - width/2, ssim_224_d2, width, label='Global 224→Crop', color='#FF5722', alpha=0.8)
bars4 = axes[1].bar(x2 + width/2, ssim_336_d2, width, label='Global 336→Crop', color='#FF9800', alpha=0.8)
axes[1].axhline(y=1.0, color='#4CAF50', linestyle='--', linewidth=2, label='Task-Guided Crop (Reference)')
axes[1].set_ylabel('SSIM vs Task-Guided Crop', fontsize=12)
axes[1].set_title('Demo2: Detail Preservation Comparison', fontsize=13, fontweight='bold')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(labels_d2)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "detail_preservation_barchart.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: detail_preservation_barchart.png")

# ============================================================
# 8. Figure: Pixel Budget Analysis
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pixel allocation comparison
methods = ['CLIP 224²\n(Global Only)', 'CLIP 336²\n(Global Only)', 'Monkey\n(448² × 4 patches)', 
           'Task-Guided\nCrop (448² × 3)']
total_pixels = [224*224, 336*336, 448*448*4, 448*448*3 + 224*224]  # last includes global context
effective_pixels_roi = [224*224*0.05, 336*336*0.05, 448*448, 448*448]  # approximate ROI pixels

colors = ['#FF5722', '#FF9800', '#2196F3', '#4CAF50']

axes[0].bar(methods, total_pixels, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Total Pixels Processed', fontsize=12)
axes[0].set_title('Pixel Budget: Total Computation', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(total_pixels):
    axes[0].text(i, v + 50000, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

axes[1].bar(methods, effective_pixels_roi, color=colors, alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Effective Pixels in ROI', fontsize=12)
axes[1].set_title('Pixel Budget: Effective Detail in ROI', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(effective_pixels_roi):
    axes[1].text(i, v + 5000, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "pixel_budget_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pixel_budget_analysis.png")

# ============================================================
# 9. Figure: Framework Architecture Diagram (Conceptual)
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'Training-Free Task-Guided Cropping Framework', fontsize=16, fontweight='bold',
        ha='center', va='top')

# Step 1: Input
rect1 = Rectangle((0.5, 5.5), 2.5, 1.5, linewidth=2, edgecolor='#2196F3', facecolor='#E3F2FD')
ax.add_patch(rect1)
ax.text(1.75, 6.25, 'High-Res\nInput Image', fontsize=11, ha='center', va='center', fontweight='bold')

# Step 2: Global view through encoder
rect2 = Rectangle((4, 5.5), 2.5, 1.5, linewidth=2, edgecolor='#FF9800', facecolor='#FFF3E0')
ax.add_patch(rect2)
ax.text(5.25, 6.25, 'Global View\n@ 224²/336²', fontsize=11, ha='center', va='center', fontweight='bold')

# Step 3: Task-guided search
rect3 = Rectangle((7.5, 5.5), 2.5, 1.5, linewidth=2, edgecolor='#4CAF50', facecolor='#E8F5E9')
ax.add_patch(rect3)
ax.text(8.75, 6.25, 'Task-Guided\nROI Search', fontsize=11, ha='center', va='center', fontweight='bold')

# Step 4: Local crops
rect4 = Rectangle((11, 5.5), 2.5, 1.5, linewidth=2, edgecolor='#9C27B0', facecolor='#F3E5F5')
ax.add_patch(rect4)
ax.text(12.25, 6.25, 'Local Crops\n@ 448²', fontsize=11, ha='center', va='center', fontweight='bold')

# Step 5: VWM integration
rect5 = Rectangle((4, 2.5), 6, 1.5, linewidth=2, edgecolor='#607D8B', facecolor='#ECEFF1')
ax.add_patch(rect5)
ax.text(7, 3.25, 'Visual Working Memory (VWM)\nGlobal Context + Local Details + Coordinates', 
        fontsize=11, ha='center', va='center', fontweight='bold')

# Step 6: MLLM output
rect6 = Rectangle((4, 0.5), 6, 1.5, linewidth=2, edgecolor='#795548', facecolor='#EFEBE9')
ax.add_patch(rect6)
ax.text(7, 1.25, 'MLLM Reasoning & Response\n(Accurate Fine-Grained Perception)', 
        fontsize=11, ha='center', va='center', fontweight='bold')

# Arrows
arrow_style = dict(arrowstyle='->', lw=2, color='#333333')
ax.annotate('', xy=(4, 6.25), xytext=(3, 6.25), arrowprops=arrow_style)
ax.annotate('', xy=(7.5, 6.25), xytext=(6.5, 6.25), arrowprops=arrow_style)
ax.annotate('', xy=(11, 6.25), xytext=(10, 6.25), arrowprops=arrow_style)
ax.annotate('', xy=(7, 4), xytext=(5.25, 5.5), arrowprops=dict(arrowstyle='->', lw=2, color='#FF9800'))
ax.annotate('', xy=(7, 4), xytext=(12.25, 5.5), arrowprops=dict(arrowstyle='->', lw=2, color='#9C27B0'))
ax.annotate('', xy=(7, 2), xytext=(7, 2.5), arrowprops=arrow_style)

# Key insight annotation
ax.text(1, 1.5, 'Key Insight:\nFixed-resolution encoders\nlose small-object details.\nTask-guided cropping\nrecovers this information\nwithout additional training.', 
        fontsize=10, ha='left', va='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='#FBC02D'))

plt.savefig(os.path.join(FIGURE_DIR, "framework_architecture.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: framework_architecture.png")

# ============================================================
# 10. Figure: V* Bench Performance Comparison (from paper data)
# ============================================================

# Data from V* paper Table 1
models = ['BLIP2', 'MiniGPT-4', 'LLaVA', 'InstructBLIP', 'Otter', 'LLaVA-1.5', 
          'Gemini Pro', 'GPT-4V', 'SEAL (Ours)']
attribute_scores = [26.95, 30.43, 23.47, 25.21, 26.95, 43.47, 40.86, 51.30, 74.78]
spatial_scores = [53.94, 50.00, 53.94, 47.36, 56.57, 56.57, 59.21, 60.52, 76.31]
overall_scores = [37.69, 38.22, 35.59, 34.02, 38.74, 48.68, 48.16, 54.97, 75.39]

categories = ['Open-source\nEnd-to-end', 'Open-source\nEnd-to-end', 'Open-source\nEnd-to-end',
              'Open-source\nEnd-to-end', 'Open-source\nEnd-to-end', 'Open-source\nEnd-to-end',
              'Commercial', 'Commercial', 'With Visual\nSearch']

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(models))
width = 0.25

bars_attr = ax.bar(x - width, attribute_scores, width, label='Attribute Recognition', 
                   color='#2196F3', alpha=0.85, edgecolor='black', linewidth=0.5)
bars_spatial = ax.bar(x, spatial_scores, width, label='Spatial Relationship', 
                     color='#FF5722', alpha=0.85, edgecolor='black', linewidth=0.5)
bars_overall = ax.bar(x + width, overall_scores, width, label='Overall', 
                     color='#4CAF50', alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('V* Bench Performance: Visual Search Enables Fine-Grained Perception', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9, rotation=30, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add category separators
ax.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=7.5, color='gray', linestyle=':', alpha=0.5)

# Annotate categories
ax.text(2.5, 82, 'Open-source End-to-end', fontsize=10, ha='center', color='gray', style='italic')
ax.text(6.5, 82, 'Commercial', fontsize=10, ha='center', color='gray', style='italic')
ax.text(8, 82, 'With Visual Search', fontsize=10, ha='center', color='#4CAF50', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "vstar_bench_performance.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: vstar_bench_performance.png")

# ============================================================
# 11. Figure: Search Efficiency Comparison (from V* paper Table 3)
# ============================================================

search_methods = ['Random-DFS', 'Random-BFS', 'Sequential-DFS', 'Sequential-BFS', 
                  'V* (LLM-guided)', 'V* w/o target cue', 'V* w/o contextual cue']
search_lengths = [8.94, 7.18, 11.39, 6.62, 4.65, 5.22, 5.36]

colors_search = ['#FF5722', '#FF9800', '#F44336', '#E91E63', '#4CAF50', '#8BC34A', '#CDDC39']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(search_methods, search_lengths, color=colors_search, alpha=0.85, 
               edgecolor='black', linewidth=0.5)

# Highlight V*
bars[4].set_edgecolor('#2E7D32')
bars[4].set_linewidth(3)

ax.set_xlabel('Average Search Length (Steps)', fontsize=12)
ax.set_title('Visual Search Efficiency: LLM-Guided Search Reduces Steps by ~50%', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Add value labels
for i, v in enumerate(search_lengths):
    ax.text(v + 0.3, i, f'{v:.2f}', fontsize=10, va='center')

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "search_efficiency.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: search_efficiency.png")

# ============================================================
# 12. Figure: Monkey Resolution Ablation (from paper data)
# ============================================================

# Data from Monkey paper Table 5
res_labels = ['r1: 896²\n(interp, no LoRA)', 'r2: 896²\n(interp, 1 LoRA)', 
              'r3: 672²\n(4 LoRA)', 'r4: 784²\n(4 LoRA)',
              'r5: 896×1344\n(6 LoRA)', 'r6: 1344×896\n(6 LoRA)',
              'r7: 896²\n(no LoRA)', 'r8: 896²\n(1 LoRA)', 'r9: 896²\n(4 LoRA)']
vqav2_scores = [74.1, 71.4, 80.0, 79.9, 80.1, 80.2, 80.1, 80.0, 80.3]
textvqa_scores = [44.7, 41.7, 67.3, 67.5, 67.3, 67.7, 67.5, 67.6, 67.6]
docvqa_scores = [53.9, 47.5, 66.4, 66.5, 66.3, 64.5, 66.1, 66.7, 66.5]

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(res_labels))
width = 0.25

ax.bar(x - width, vqav2_scores, width, label='VQAv2', color='#2196F3', alpha=0.85)
ax.bar(x, textvqa_scores, width, label='TextVQA', color='#FF5722', alpha=0.85)
ax.bar(x + width, docvqa_scores, width, label='DocVQA', color='#4CAF50', alpha=0.85)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Resolution Enhancement Impact: Patch-Based vs Position Interpolation (Monkey)', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(res_labels, fontsize=8, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "resolution_ablation_monkey.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: resolution_ablation_monkey.png")

# ============================================================
# 13. Summary Statistics
# ============================================================

summary = {
    'resolution_metrics': all_metrics,
    'detail_preservation': {
        'demo1': detail_metrics_demo1,
        'demo2': detail_metrics_demo2
    },
    'key_findings': {
        'ssim_loss_at_224': f"{metrics_demo1[0]['ssim']:.3f} (Demo1), {metrics_demo2[0]['ssim']:.3f} (Demo2)",
        'ssim_gain_from_224_to_448': f"+{metrics_demo1[2]['ssim'] - metrics_demo1[0]['ssim']:.3f} (Demo1), +{metrics_demo2[2]['ssim'] - metrics_demo2[0]['ssim']:.3f} (Demo2)",
        'avg_detail_preservation_loss_224': f"{np.mean([m['ssim_224'] for m in detail_metrics_demo1 + detail_metrics_demo2]):.3f}",
        'avg_detail_preservation_loss_336': f"{np.mean([m['ssim_336'] for m in detail_metrics_demo1 + detail_metrics_demo2]):.3f}",
        'vstar_seal_overall_accuracy': 75.39,
        'vstar_gpt4v_overall_accuracy': 54.97,
        'vstar_improvement': '+20.42%',
        'search_efficiency_vstar': 4.65,
        'search_efficiency_random_dfs': 8.94,
        'search_reduction': '~48%'
    }
}

with open(os.path.join(OUTPUT_DIR, 'analysis_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n=== Analysis Complete ===")
print(f"All figures saved to: {FIGURE_DIR}")
print(f"All outputs saved to: {OUTPUT_DIR}")