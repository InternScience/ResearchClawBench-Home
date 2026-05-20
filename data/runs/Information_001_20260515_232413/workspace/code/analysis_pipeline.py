#!/usr/bin/env python3
"""
V* (SEAL) Framework Analysis Pipeline
======================================
Demonstrates the information loss caused by fixed-resolution vision encoders
and validates the task-guided cropping strategy proposed in the SEAL framework.

Reference: "V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs" (CVPR)
"""

import numpy as np
from PIL import Image, ImageFilter, ImageStat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, sys, json
from pathlib import Path

# Workspace paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_001_20260515_232413')
DATA_DIR = WORKSPACE / 'data' / 'demo_imgs'
OUTPUT_DIR = WORKSPACE / 'outputs'
REPORT_IMG_DIR = WORKSPACE / 'report' / 'images'
CODE_DIR = WORKSPACE / 'code'

for d in [OUTPUT_DIR, REPORT_IMG_DIR, CODE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Standard resolutions used by CLIP-based vision encoders
CLIP_RESOLUTIONS = {
    'CLIP ViT-B/32': 224,
    'CLIP ViT-L/14': 224,
    'CLIP ViT-L/14@336': 336,
    'EVA-CLIP ViT-g/14': 224,
}

# The V* Bench average resolution
VSTAR_AVG_RES = (2246, 1582)

def load_image(path):
    """Load image as RGB numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img), img

def compute_image_statistics(img_array):
    """Compute comprehensive image statistics."""
    h, w, c = img_array.shape
    gray = np.mean(img_array, axis=2)

    stats = {
        'width': w,
        'height': h,
        'total_pixels': w * h,
        'mean_rgb': [float(img_array[:,:,i].mean()) for i in range(3)],
        'std_rgb': [float(img_array[:,:,i].std()) for i in range(3)],
        'mean_gray': float(gray.mean()),
        'std_gray': float(gray.std()),
        'min_gray': float(gray.min()),
        'max_gray': float(gray.max()),
    }

    # Entropy estimation (per channel, binned)
    for i, ch_name in enumerate(['R', 'G', 'B']):
        hist, _ = np.histogram(img_array[:,:,i].ravel(), bins=256, range=(0,256), density=True)
        hist = hist[hist > 0]
        stats[f'entropy_{ch_name}'] = float(-np.sum(hist * np.log2(hist)))

    # Gradient magnitude (edge energy)
    gy, gx = np.gradient(gray.astype(float))
    gmag = np.sqrt(gx**2 + gy**2)
    stats['gradient_mean'] = float(gmag.mean())
    stats['gradient_std'] = float(gmag.std())

    # Local variance (texture measure) using 8x8 blocks
    block_size = 8
    local_vars = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            local_vars.append(block.var())
    local_vars = np.array(local_vars)
    stats['local_variance_mean'] = float(local_vars.mean())
    stats['local_variance_std'] = float(local_vars.std())

    return stats

def compute_multi_scale_stats(img_array, resolutions=[64, 128, 224, 336, 448, 672, 896]):
    """Compute statistics at multiple resolutions to show information loss."""
    h, w = img_array.shape[:2]
    pil_img = Image.fromarray(img_array)

    multi_stats = {}
    for res in resolutions:
        # Maintain aspect ratio, scale so that the smaller dimension equals res
        scale = res / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = np.array(pil_img.resize((new_w, new_h), Image.LANCZOS))
        multi_stats[res] = compute_image_statistics(resized)

    # Also compute at original resolution
    multi_stats['original'] = compute_image_statistics(img_array)

    return multi_stats

def detect_regions_of_interest(img_array, n_regions=5, method='entropy'):
    """
    Detect regions of interest using sliding-window saliency estimation.
    This simulates the V* framework's task-guided region identification.

    Methods:
      - 'entropy': local entropy (information density)
      - 'gradient': local edge density
      - 'variance': local intensity variance
    """
    h, w = img_array.shape[:2]
    gray = np.mean(img_array, axis=2)

    # Use adaptive window size: roughly 20% of image dimensions
    win_h, win_w = max(32, h // 5), max(32, w // 5)
    stride_h, stride_w = max(8, win_h // 4), max(8, win_w // 4)

    scores = []
    positions = []

    for y in range(0, h - win_h, stride_h):
        for x in range(0, w - win_w, stride_w):
            patch = gray[y:y+win_h, x:x+win_w]
            if method == 'entropy':
                hist, _ = np.histogram(patch.ravel(), bins=64, density=True)
                hist = hist[hist > 0]
                score = -np.sum(hist * np.log2(hist + 1e-10))
            elif method == 'gradient':
                gy, gx = np.gradient(patch.astype(float))
                score = np.sqrt(gx**2 + gy**2).mean()
            elif method == 'variance':
                score = patch.var()
            else:
                score = patch.std()

            scores.append(score)
            positions.append((x, y, win_w, win_h))

    # Select top-N non-overlapping regions
    scores = np.array(scores)
    sorted_idx = np.argsort(scores)[::-1]

    selected = []
    selected_boxes = []
    for idx in sorted_idx:
        if len(selected) >= n_regions:
            break
        x, y, pw, ph = positions[idx]
        # Check overlap with existing selections (IoU threshold)
        overlap = False
        for bx, by, bw, bh in selected_boxes:
            iou = compute_iou(x, y, pw, ph, bx, by, bw, bh)
            if iou > 0.3:
                overlap = True
                break
        if not overlap:
            selected.append((x, y, pw, ph, scores[idx], positions[idx]))
            selected_boxes.append((x, y, pw, ph))

    return selected

def compute_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    """Compute IoU between two boxes."""
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    return inter / (area1 + area2 - inter + 1e-10)

def simulate_information_loss(img_array, target_resolutions=[224, 336]):
    """
    Simulate CLIP-style fixed-resolution encoding:
    1. Resize full image to target resolution (simulating CLIP preprocess)
    2. Upsample back to original size
    3. Compute per-pixel error
    """
    h, w = img_array.shape[:2]
    pil_img = Image.fromarray(img_array)
    results = {}

    for target_res in target_resolutions:
        # CLIP-style resize: resize shorter side, center crop
        scale = target_res / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to target_res x target_res
        left = (new_w - target_res) // 2
        top = (new_h - target_res) // 2
        cropped = resized.crop((left, top, left + target_res, top + target_res))

        # Upsample back to original for comparison
        upsampled = np.array(cropped.resize((w, h), Image.LANCZOS))
        error = np.abs(img_array.astype(float) - upsampled.astype(float))

        results[target_res] = {
            'mse': float((error ** 2).mean()),
            'mae': float(error.mean()),
            'psnr': float(20 * np.log10(255.0 / np.sqrt((error ** 2).mean() + 1e-10))),
            'ssim_approx': float(1.0 - (error ** 2).mean() / (img_array.var() + 1e-10)),
            'error_map': error,
            'resized_image': upsampled,
        }

    return results

def simulate_zoom_recovery(img_array, regions, target_res=224):
    """
    Simulate the V* zoom-and-recover strategy:
    1. For each ROI, crop and resize to target_res (simulating focused encoding)
    2. Compare local-detail extraction vs global encoding
    """
    h, w = img_array.shape[:2]
    pil_img = Image.fromarray(img_array)

    recovery_results = []
    for i, (rx, ry, rw, rh, score, _) in enumerate(regions):
        # Crop the region
        crop = img_array[ry:ry+rh, rx:rx+rw]
        crop_pil = Image.fromarray(crop)

        # Simulate CLIP encoding: resize to target_res
        scale = target_res / min(rh, rw)
        new_w, new_h = int(rw * scale), int(h * scale)
        resized_crop = crop_pil.resize((new_w, new_h), Image.LANCZOS)

        # Center crop
        left = (new_w - target_res) // 2
        top = (new_h - target_res) // 2
        encoded_crop = resized_crop.crop((left, top, left + target_res, top + target_res))

        # Upsample crop back for comparison
        upsampled_crop = np.array(encoded_crop.resize((rw, rh), Image.LANCZOS))
        crop_error = np.abs(crop.astype(float) - upsampled_crop.astype(float))

        # For global encoding, simulate what portion of the image this region occupies
        # after global downsampling
        global_scale = target_res / min(h, w)
        global_crop_w = int(rw * global_scale)
        global_crop_h = int(rh * global_scale)
        effective_pixels_global = global_crop_w * global_crop_h
        effective_pixels_local = target_res * target_res

        recovery_results.append({
            'region_id': i,
            'box': [rx, ry, rw, rh],
            'score': float(score),
            'crop_mse': float((crop_error ** 2).mean()),
            'crop_psnr': float(20 * np.log10(255.0 / np.sqrt((crop_error ** 2).mean() + 1e-10))),
            'effective_pixels_global': effective_pixels_global,
            'effective_pixels_local': effective_pixels_local,
            'resolution_gain': effective_pixels_local / max(effective_pixels_global, 1),
        })

    return recovery_results

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    print("=" * 70)
    print("V* (SEAL) Framework Analysis Pipeline")
    print("=" * 70)

    # Load demo images
    image_files = {
        'demo1': DATA_DIR / 'demo1.png',
        'demo2': DATA_DIR / 'demo2.png',
        'method_case': DATA_DIR / 'method_case.png',
    }

    all_results = {}

    for name, path in image_files.items():
        print(f"\n{'='*50}")
        print(f"Processing: {name} ({path})")
        img_array, pil_img = load_image(path)
        h, w = img_array.shape[:2]
        print(f"  Resolution: {w}x{h}, Total pixels: {w*h:,}")

        # 1. Full image statistics
        print("  Computing full image statistics...")
        full_stats = compute_image_statistics(img_array)
        all_results[f'{name}_full_stats'] = full_stats

        # 2. Multi-scale analysis
        print("  Computing multi-scale statistics...")
        multi_stats = compute_multi_scale_stats(img_array)
        all_results[f'{name}_multi_scale'] = multi_stats

        # 3. Information loss simulation at CLIP resolutions
        print("  Simulating CLIP fixed-resolution information loss...")
        info_loss = simulate_information_loss(img_array)
        all_results[f'{name}_info_loss'] = {k: {
            'mse': v['mse'], 'mae': v['mae'], 'psnr': v['psnr'],
            'ssim_approx': v['ssim_approx']
        } for k, v in info_loss.items()}

        # 4. Region of Interest detection
        print("  Detecting regions of interest...")
        for method in ['entropy', 'gradient', 'variance']:
            rois = detect_regions_of_interest(img_array, n_regions=5, method=method)
            all_results[f'{name}_rois_{method}'] = [{
                'box': r[:4], 'score': r[4]
            } for r in rois]

        # 5. Zoom recovery simulation
        print("  Simulating zoom recovery...")
        rois_entropy = detect_regions_of_interest(img_array, n_regions=5, method='entropy')
        recovery = simulate_zoom_recovery(img_array, rois_entropy)
        all_results[f'{name}_zoom_recovery'] = recovery

        # 6. Patch-level analysis at different zoom levels
        print("  Analyzing patch-level detail preservation...")
        # Divide image into grid, compute detail metrics at different scales
        grid_sizes = [(2,2), (4,4), (8,8)]
        patch_analysis = {}
        for gh, gw in grid_sizes:
            patch_h, patch_w = h // gh, w // gw
            patch_stats = []
            for py in range(gh):
                for px in range(gw):
                    y1, y2 = py * patch_h, (py + 1) * patch_h
                    x1, x2 = px * patch_w, (px + 1) * patch_w
                    patch = img_array[y1:y2, x1:x2]
                    patch_stats.append({
                        'grid_pos': [px, py],
                        'mean': float(patch.mean()),
                        'std': float(patch.std()),
                        'gradient_mean': float(np.mean(np.abs(np.diff(patch.mean(axis=2), axis=1))) +
                                             np.mean(np.abs(np.diff(patch.mean(axis=2), axis=0)))),
                    })
            patch_analysis[f'{gh}x{gw}'] = patch_stats
        all_results[f'{name}_patch_analysis'] = patch_analysis

    # Save all results
    results_path = OUTPUT_DIR / 'analysis_results.json'
    # Convert non-serializable items
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            serializable[k] = {str(sk): sv for sk, sv in v.items()}
        else:
            serializable[k] = v

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ==========================================================================
    # FIGURE GENERATION
    # ==========================================================================

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'figure.dpi': 150,
    })

    # ---- Figure 1: Multi-Resolution Information Loss ----
    print("\nGenerating Figure 1: Multi-Resolution Information Loss...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    resolutions = [64, 128, 224, 336, 448, 672, 896]
    colors = {'demo1': '#2196F3', 'demo2': '#FF5722', 'method_case': '#4CAF50'}

    for idx, (name, path) in enumerate(image_files.items()):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        ms = all_results[f'{name}_multi_scale']

        grad_vals = [ms[r]['gradient_mean'] for r in resolutions]
        ax.plot(resolutions, grad_vals, 'o-', color=colors[name], linewidth=2, markersize=6)
        ax.axhline(y=ms['original']['gradient_mean'], color=colors[name], linestyle='--',
                   alpha=0.5, label=f'Original ({ms["original"]["width"]}x{ms["original"]["height"]})')
        ax.set_xlabel('Resolution (short side, px)')
        ax.set_ylabel('Mean Gradient Magnitude')
        ax.set_title(f'{name} — Edge Detail vs Resolution')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Combined comparison of entropy
    ax = axes[1, 2]
    for name in image_files:
        ms = all_results[f'{name}_multi_scale']
        ent_vals = [ms[r]['entropy_R'] for r in resolutions]
        ax.plot(resolutions, ent_vals, 'o-', color=colors[name], linewidth=2, markersize=6, label=name)
    ax.set_xlabel('Resolution (short side, px)')
    ax.set_ylabel('Entropy (bits, R channel)')
    ax.set_title('Information Content vs Resolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark CLIP resolutions
    for ax in axes.flat:
        for clip_res in [224, 336]:
            ax.axvline(x=clip_res, color='red', linestyle=':', alpha=0.4, linewidth=1)
        if ax == axes[1, 2]:
            ax.text(224, ax.get_ylim()[0], 'CLIP-224', rotation=90, va='bottom', fontsize=8, color='red', alpha=0.6)
            ax.text(336, ax.get_ylim()[0], 'CLIP-336', rotation=90, va='bottom', fontsize=8, color='red', alpha=0.6)

    axes[0, 0].annotate('', xy=(0, 0), fontsize=10)
    plt.suptitle('Figure 1: Information Loss from Fixed-Resolution Encoding\n(CLIP standard resolutions marked in red)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'figure1_multi_resolution_loss.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  -> figure1_multi_resolution_loss.png saved")

    # ---- Figure 2: CLIP Encoding Reconstruction Error ----
    print("\nGenerating Figure 2: CLIP Encoding Reconstruction Error...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for idx, (name, path) in enumerate(image_files.items()):
        img_array, _ = load_image(path)
        info_loss = simulate_information_loss(img_array)

        # Original thumbnail
        axes[idx, 0].imshow(img_array)
        axes[idx, 0].set_title(f'{name} — Original\n({img_array.shape[1]}×{img_array.shape[0]})', fontsize=10)
        axes[idx, 0].axis('off')

        # 224px reconstruction error
        err_224 = info_loss[224]['error_map'].mean(axis=2)
        im = axes[idx, 1].imshow(err_224, cmap='hot')
        axes[idx, 1].set_title(f'Error Map @224px\nPSNR={info_loss[224]["psnr"]:.1f}dB', fontsize=10)
        axes[idx, 1].axis('off')
        plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)

        # 336px reconstruction error
        err_336 = info_loss[336]['error_map'].mean(axis=2)
        im = axes[idx, 2].imshow(err_336, cmap='hot')
        axes[idx, 2].set_title(f'Error Map @336px\nPSNR={info_loss[336]["psnr"]:.1f}dB', fontsize=10)
        axes[idx, 2].axis('off')
        plt.colorbar(im, ax=axes[idx, 2], fraction=0.046, pad=0.04)

    plt.suptitle('Figure 2: Reconstruction Error from Fixed-Resolution CLIP Encoding\n(Brighter = more information lost)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'figure2_clip_reconstruction_error.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  -> figure2_clip_reconstruction_error.png saved")

    # ---- Figure 3: Region of Interest Detection & Zoom Recovery ----
    print("\nGenerating Figure 3: ROI Detection and Zoom Recovery...")
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25)

    for idx, (name, path) in enumerate(image_files.items()):
        img_array, pil_img = load_image(path)
        h, w = img_array.shape[:2]

        # Original with ROI boxes
        ax_orig = fig.add_subplot(gs[idx, 0])
        ax_orig.imshow(img_array)
        rois = detect_regions_of_interest(img_array, n_regions=5, method='entropy')
        colors_roi = plt.cm.tab10(np.linspace(0, 1, len(rois)))
        for i, (rx, ry, rw, rh, score, _) in enumerate(rois):
            rect = plt.Rectangle((rx, ry), rw, rh, fill=False, edgecolor=colors_roi[i],
                                  linewidth=2, label=f'ROI {i+1}')
            ax_orig.add_patch(rect)
            ax_orig.text(rx + 2, ry + 15, f'{i+1}', color=colors_roi[i],
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        ax_orig.set_title(f'{name} — Detected ROIs', fontsize=11)
        ax_orig.axis('off')

        # ROI zoom crops
        for i, (rx, ry, rw, rh, score, _) in enumerate(rois[:3]):
            ax_crop = fig.add_subplot(gs[idx, i + 1])
            crop = img_array[ry:ry+rh, rx:rx+rw]
            ax_crop.imshow(crop)
            ax_crop.set_title(f'ROI {i+1} ({rw}×{rh})\nScore={score:.2f}', fontsize=9)
            ax_crop.axis('off')

    plt.suptitle('Figure 3: Task-Guided Region of Interest Detection\n(Simulating V* visual search attention mechanism)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'figure3_roi_detection.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  -> figure3_roi_detection.png saved")

    # ---- Figure 4: Zoom Recovery Resolution Gain ----
    print("\nGenerating Figure 4: Zoom Recovery Resolution Gain...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (name, path) in enumerate(image_files.items()):
        recovery = all_results[f'{name}_zoom_recovery']
        region_ids = [r['region_id'] + 1 for r in recovery]
        gains = [r['resolution_gain'] for r in recovery]
        psnrs = [r['crop_psnr'] for r in recovery]

        ax1 = axes[idx]
        bar_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(gains)))
        bars = ax1.bar(region_ids, gains, color=bar_colors)
        ax1.set_xlabel('ROI Index')
        ax1.set_ylabel('Resolution Gain (×)')
        ax1.set_title(f'{name} — Zoom Resolution Gain')
        ax1.axhline(y=1.0, color='red', linestyle='--', label='Global baseline', linewidth=1)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add PSNR labels
        for bar, psnr in zip(bars, psnrs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'PSNR={psnr:.0f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Figure 4: Zoom Recovery — Effective Resolution Gain over Global Encoding', fontsize=14, y=1.03)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'figure4_zoom_recovery_gain.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  -> figure4_zoom_recovery_gain.png saved")

    # ---- Figure 5: Patch-Level Detail Preservation ----
    print("\nGenerating Figure 5: Patch-Level Detail Preservation...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, (name, path) in enumerate(image_files.items()):
        img_array, _ = load_image(path)
        h, w = img_array.shape[:2]

        # Compute per-patch gradient (detail density) at original res
        patch_h, patch_w = h // 8, w // 8
        detail_map = np.zeros((8, 8))
        for py in range(8):
            for px in range(8):
                y1, y2 = py * patch_h, (py + 1) * patch_h
                x1, x2 = px * patch_w, (px + 1) * patch_w
                patch = img_array[y1:y2, x1:x2]
                gray = patch.mean(axis=2)
                gy, gx = np.gradient(gray.astype(float))
                detail_map[py, px] = np.sqrt(gx**2 + gy**2).mean()

        # Row 0: Detail map at original resolution
        im = axes[0, idx].imshow(detail_map, cmap='viridis', aspect='auto')
        axes[0, idx].set_title(f'{name} — Detail Density Map\n(Original: {w}×{h})', fontsize=10)
        axes[0, idx].set_xlabel('Patch X (8×8 grid)')
        axes[0, idx].set_ylabel('Patch Y (8×8 grid)')
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046, pad=0.04)

        # Row 1: Simulate detail map after CLIP-224 encoding
        # Resize image to 224, then upsample, then compute detail map
        pil_img = Image.fromarray(img_array)
        scale = 224 / min(h, w)
        cw, ch = int(w * scale), int(h * scale)
        resized = pil_img.resize((cw, ch), Image.LANCZOS)
        left = (cw - 224) // 2
        top = (ch - 224) // 2
        cropped = resized.crop((left, top, left + 224, top + 224))
        upsampled = np.array(cropped.resize((w, h), Image.LANCZOS))

        sim_detail = np.zeros((8, 8))
        for py in range(8):
            for px in range(8):
                y1, y2 = py * patch_h, (py + 1) * patch_h
                x1, x2 = px * patch_w, (px + 1) * patch_w
                patch = upsampled[y1:y2, x1:x2]
                gray_sim = patch.mean(axis=2)
                gy_s, gx_s = np.gradient(gray_sim.astype(float))
                sim_detail[py, px] = np.sqrt(gx_s**2 + gy_s**2).mean()

        im = axes[1, idx].imshow(sim_detail, cmap='viridis', aspect='auto')
        axes[1, idx].set_title(f'{name} — Detail after CLIP-224\n(Loss: {(1 - sim_detail.sum()/max(detail_map.sum(), 1e-10))*100:.1f}%)', fontsize=10)
        axes[1, idx].set_xlabel('Patch X (8×8 grid)')
        axes[1, idx].set_ylabel('Patch Y (8×8 grid)')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)

    plt.suptitle('Figure 5: Spatial Detail Preservation Before vs After CLIP-224 Encoding', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'figure5_detail_preservation.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  -> figure5_detail_preservation.png saved")

    # ---- Figure 6: Quantitative Information Loss Summary ----
    print("\nGenerating Figure 6: Quantitative Information Loss Summary...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PSNR comparison
    ax = axes[0, 0]
    x_labels = []
    psnr_224 = []
    psnr_336 = []
    for name in image_files:
        x_labels.append(name)
        il = all_results[f'{name}_info_loss']
        psnr_224.append(il[224]['psnr'])
        psnr_336.append(il[336]['psnr'])

    x = np.arange(len(x_labels))
    w = 0.35
    ax.bar(x - w/2, psnr_224, w, label='CLIP-224', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, psnr_336, w, label='CLIP-336', color='#FF5722', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Reconstruction Quality (PSNR)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Entropy loss
    ax = axes[0, 1]
    for name in image_files:
        ms = all_results[f'{name}_multi_scale']
        orig_ent = ms['original']['entropy_R'] 
        ent_loss = [(orig_ent - ms[r]['entropy_R']) / orig_ent * 100 for r in [224, 336]]
        bars = ax.bar([f'{name}-224', f'{name}-336'], ent_loss, color=[colors[name], colors[name]])
        bars[0].set_alpha(0.8)
        bars[1].set_alpha(0.5)
    ax.set_ylabel('Relative Entropy Loss (%)')
    ax.set_title('Information Content Loss')
    ax.grid(True, alpha=0.3, axis='y')

    # Zoom recovery gain
    ax = axes[1, 0]
    all_gains = []
    all_names = []
    for name in image_files:
        recovery = all_results[f'{name}_zoom_recovery']
        for r in recovery:
            all_gains.append(r['resolution_gain'])
            all_names.append(f'{name}-ROI{r["region_id"]+1}')
    colors_bar = [colors[n.split('-')[0]] for n in all_names]
    ax.barh(range(len(all_gains)), all_gains, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel('Effective Resolution Gain (×)')
    ax.set_title('Zoom Recovery — Resolution Multiplier')
    ax.axvline(x=1.0, color='red', linestyle='--')
    ax.grid(True, alpha=0.3, axis='x')

    # Information retention curve (combined)
    ax = axes[1, 1]
    for name in image_files:
        ms = all_results[f'{name}_multi_scale']
        orig_grad = ms['original']['gradient_mean'] 
        retention = [ms[r]['gradient_mean'] / orig_grad * 100 for r in resolutions]
        ax.plot(resolutions, retention, 'o-', color=colors[name], linewidth=2, label=name)
    ax.set_xlabel('Resolution (short side, px)')
    ax.set_ylabel('Edge Detail Retention (%)')
    ax.set_title('Detail Retention vs Resolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=224, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=336, color='red', linestyle=':', alpha=0.5)

    plt.suptitle('Figure 6: Quantitative Summary of Information Loss and Zoom Recovery', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'figure6_quantitative_summary.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  -> figure6_quantitative_summary.png saved")

    # ---- Figure 7: Zoom Recovery Demonstration ----
    print("\nGenerating Figure 7: Zoom Recovery Demonstration...")
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))

    for idx, (name, path) in enumerate(image_files.items()):
        img_array, pil_img = load_image(path)
        h, w = img_array.shape[:2]

        # Full image
        axes[idx, 0].imshow(img_array)
        axes[idx, 0].set_title(f'{name}\n({w}×{h})', fontsize=9)
        axes[idx, 0].axis('off')

        # CLIP-224 downsampled version
        scale_224 = 224 / min(h, w)
        cw, ch = int(w * scale_224), int(h * scale_224)
        clip224 = np.array(pil_img.resize((cw, ch), Image.LANCZOS))
        left = (cw - 224) // 2
        top = (ch - 224) // 2
        clip224_cropped = clip224[top:top+224, left:left+224]
        axes[idx, 1].imshow(clip224_cropped)
        axes[idx, 1].set_title(f'CLIP-224 Encoding\n(What the encoder sees)', fontsize=9)
        axes[idx, 1].axis('off')

        # Best ROI (zoomed)
        rois = detect_regions_of_interest(img_array, n_regions=1, method='entropy')
        if rois:
            rx, ry, rw, rh = rois[0][:4]
            crop = img_array[ry:ry+rh, rx:rx+rw]
            axes[idx, 2].imshow(crop)
            axes[idx, 2].set_title(f'Top ROI (Zoomed)\n({rw}×{rh})', fontsize=9)
            axes[idx, 2].axis('off')

        # ROI re-encoded at CLIP-224
        if rois:
            crop_pil = Image.fromarray(crop)
            scale_c = 224 / min(rh, rw)
            ccw, cch = int(rw * scale_c), int(rh * scale_c)
            crop_resized = crop_pil.resize((ccw, cch), Image.LANCZOS)
            cleft = (ccw - 224) // 2
            ctop = (cch - 224) // 2
            crop_224 = np.array(crop_resized.crop((cleft, ctop, cleft + 224, ctop + 224)))
            axes[idx, 3].imshow(crop_224)
            axes[idx, 3].set_title(f'ROI @CLIP-224\n(Detail-preserved zoom)', fontsize=9)
            axes[idx, 3].axis('off')

    plt.suptitle('Figure 7: V* Zoom Recovery Strategy — Global vs Local Encoding', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'figure7_zoom_recovery_demo.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  -> figure7_zoom_recovery_demo.png saved")

    print("\n" + "=" * 70)
    print("Analysis complete. All figures saved to report/images/")
    print("=" * 70)

    return all_results

if __name__ == '__main__':
    results = main()
