#!/usr/bin/env python3
"""
Visualization Script: Generate all figures for the research report.
Generates:
  - Figure 1: Framework overview diagram
  - Figure 2: Saliency maps and detected regions
  - Figure 3: Crop comparison (baseline vs task-guided)
  - Figure 4: Information preservation analysis
  - Figure 5: Resolution vs detail trade-off analysis
"""

import os, sys, json, numpy as np
from pathlib import Path

WORKSPACE = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_001_20260415_144442")
DATA_DIR = WORKSPACE / "data" / "demo_imgs"
OUTPUT_DIR = WORKSPACE / "outputs"
REPORT_IMG_DIR = WORKSPACE / "report" / "images"

for d in [OUTPUT_DIR, REPORT_IMG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy import ndimage

# Re-run analysis to get results
exec(open(WORKSPACE / "code" / "analysis_pipeline.py").read())

print("\n" + "="*60)
print("Generating Figures")
print("="*60)

# ============================================================
# Figure 1: Framework Overview Diagram
# ============================================================
def generate_framework_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Training-Free Fine-Grained Perception Framework', 
            fontsize=16, ha='center', fontweight='bold')
    
    # Input image box
    rect_img = patches.FancyBboxPatch((0.5, 5.5), 2, 1.5, boxstyle="round,pad=0.1",
                                       facecolor='#e8f4f8', edgecolor='#2980b9', linewidth=2)
    ax.add_patch(rect_img)
    ax.text(1.5, 6.25, 'Input\nImage\n(H×W)', fontsize=11, ha='center', va='center')
    
    # Arrow to encoder
    ax.annotate('', xy=(3.5, 6.25), xytext=(2.5, 6.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    
    # Fixed-resolution encoder box
    rect_enc = patches.FancyBboxPatch((3.5, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                                       facecolor='#fde8e8', edgecolor='#c0392b', linewidth=2)
    ax.add_patch(rect_enc)
    ax.text(4.75, 6.25, 'Vision\nEncoder\n(224×224)', fontsize=11, ha='center', va='center')
    
    # Loss annotation
    ax.text(4.75, 4.8, '⚠ Information Loss\nfor small objects', fontsize=10, 
            ha='center', color='#c0392b', style='italic')
    
    # Arrow down to LLM
    ax.annotate('', xy=(4.75, 4.3), xytext=(4.75, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    
    # Task-guided cropping path
    # Arrow from input to saliency
    ax.annotate('', xy=(3.5, 3.5), xytext=(2.5, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60', linestyle='--'))
    
    # Saliency computation box
    rect_sal = patches.FancyBboxPatch((3.5, 2.8), 2.5, 1.5, boxstyle="round,pad=0.1",
                                       facecolor='#e8f8e8', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(rect_sal)
    ax.text(4.75, 3.55, 'Saliency\nMap\nGeneration', fontsize=11, ha='center', va='center')
    
    # Arrow to region detection
    ax.annotate('', xy=(7.5, 3.55), xytext=(6, 3.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60'))
    
    # Region detection box
    rect_reg = patches.FancyBboxPatch((7.5, 2.8), 2.5, 1.5, boxstyle="round,pad=0.1",
                                       facecolor='#e8f8e8', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(rect_reg)
    ax.text(8.75, 3.55, 'Top-K\nRegion\nDetection', fontsize=11, ha='center', va='center')
    
    # Arrow to crop extraction
    ax.annotate('', xy=(11.5, 3.55), xytext=(10, 3.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60'))
    
    # Crop extraction box
    rect_crop = patches.FancyBboxPatch((11.5, 2.8), 2.5, 1.5, boxstyle="round,pad=0.1",
                                        facecolor='#e8f8e8', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(rect_crop)
    ax.text(12.75, 3.55, 'Adaptive\nCrop\nExtraction', fontsize=11, ha='center', va='center')
    
    # Arrows up to integration
    ax.annotate('', xy=(12.75, 4.5), xytext=(12.75, 4.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60'))
    
    # Integration box
    rect_int = patches.FancyBboxPatch((11, 4.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                                       facecolor='#fff8e8', edgecolor='#f39c12', linewidth=2)
    ax.add_patch(rect_int)
    ax.text(12.75, 5.25, 'Multi-Scale\nFeature\nIntegration', fontsize=11, ha='center', va='center')
    
    # Arrow to LLM
    ax.annotate('', xy=(12.75, 4.5), xytext=(12.75, 6.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='#f39c12'))
    
    # LLM box
    rect_llm = patches.FancyBboxPatch((10, 6.2), 5.5, 1.2, boxstyle="round,pad=0.1",
                                       facecolor='#f0e8ff', edgecolor='#8e44ad', linewidth=2)
    ax.add_patch(rect_llm)
    ax.text(12.75, 6.8, 'MLLM (LLaVA / GPT-4V)', fontsize=11, ha='center', va='center')
    
    # Output
    ax.annotate('', xy=(12.75, 7.2), xytext=(12.75, 7.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#8e44ad'))
    ax.text(12.75, 7.5, 'Accurate Answer', fontsize=11, ha='center', fontweight='bold')
    
    # Legend
    legend_y = 1.5
    ax.text(0.5, legend_y, 'Legend:', fontsize=10, fontweight='bold')
    ax.text(0.5, legend_y - 0.4, '🔴 Fixed encoding: loses fine details', fontsize=9)
    ax.text(0.5, legend_y - 0.7, '🟢 Task-guided cropping: preserves local detail', fontsize=9)
    ax.text(0.5, legend_y - 1.0, '🟡 Multi-scale integration: combines global + local', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'figure1_framework.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: figure1_framework.png")

generate_framework_diagram()

# ============================================================
# Figure 2: Saliency Maps and Detected Regions
# ============================================================
def generate_saliency_figure():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    case_keys = list(results.keys())
    
    for idx, case_key in enumerate(case_keys):
        r = results[case_key]
        image = r['image']
        saliency = r['saliency']
        regions = r['regions']
        h, w = image.shape[:2]
        
        # Original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Original Image ({w}×{h})', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Saliency map
        im = axes[idx, 1].imshow(saliency, cmap='hot')
        axes[idx, 1].set_title('Computed Saliency Map', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)
        
        # Image with region overlays
        axes[idx, 2].imshow(image)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, (y1, x1, y2, x2) in enumerate(regions):
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=3, edgecolor=colors[i % len(colors)],
                                     facecolor='none', linestyle='-')
            axes[idx, 2].add_patch(rect)
            axes[idx, 2].text(x1, y1-5, f'ROI-{i+1}', color=colors[i % len(colors)],
                             fontsize=10, fontweight='bold')
        axes[idx, 2].set_title(f'Detected Regions of Interest (n={len(regions)})', 
                               fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
    
    plt.suptitle('Saliency-Guided Region Detection for Task-Driven Cropping', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'figure2_saliency.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: figure2_saliency.png")

generate_saliency_figure()

# ============================================================
# Figure 3: Baseline vs Crop Comparison
# ============================================================
def generate_comparison_figure():
    case_keys = list(results.keys())
    
    for idx, case_key in enumerate(case_keys):
        r = results[case_key]
        image = r['image']
        baseline = r['baseline']['CLIP ViT-B']['encoded_image']
        crops = r['crops']
        regions = r['regions']
        h, w = image.shape[:2]
        
        n_crops = len(crops)
        ncols = 2 + n_crops
        fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 5))
        
        # Original
        axes[0].imshow(image)
        axes[0].set_title(f'Original\n({w}×{h}, {w*h:,} px)', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # Baseline encoded
        axes[1].imshow(baseline)
        axes[1].set_title(f'Fixed Encoding\n(224×224, {224*224:,} px)\nLoss: {r["info_metrics"]["resolution_ratio"]*100:.1f}% info',
                          fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        # Crops
        for i, crop in enumerate(crops):
            y1, x1, y2, x2 = regions[i]
            orig_region = image[y1:y2, x1:x2]
            
            # Show original region patch
            if i < n_crops // 2 or n_crops <= 2:
                axes[2+i].imshow(crop)
                axes[2+i].set_title(f'Crop {i+1}\nRegion: ({x1},{y1})-({x2},{y2})\n→ 224×224',
                                    fontsize=10, fontweight='bold')
                axes[2+i].axis('off')
        
        plt.suptitle(f'Comparison: Fixed Encoding vs Task-Guided Cropping — {case_key}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(REPORT_IMG_DIR / f'figure3_comparison_{idx+1}.png', dpi=150, 
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: figure3_comparison_{idx+1}.png")

generate_comparison_figure()

# ============================================================
# Figure 4: Information Preservation Analysis (Bar chart)
# ============================================================
def generate_metrics_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    case_keys = list(results.keys())
    encoders = list(ENCODER_RESOLUTIONS.keys())
    
    # Panel A: Pixel loss comparison
    x = np.arange(len(case_keys))
    width = 0.25
    
    for i, ename in enumerate(encoders):
        losses = []
        for ck in case_keys:
            r = results[ck]
            ew, eh = ENCODER_RESOLUTIONS[ename]
            ow, oh = r['image_size']
            loss = 1 - (ew * eh) / (ow * oh)
            losses.append(loss * 100)
        axes[0].bar(x + i*width, losses, width, label=ename)
    
    axes[0].set_xlabel('Demo Image', fontsize=11)
    axes[0].set_ylabel('Pixel Information Loss (%)', fontsize=11)
    axes[0].set_title('(A) Fixed-Resolution Encoding Loss', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([ck.replace('.png','') for ck in case_keys])
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Panel B: Effective pixel density gain
    densities = []
    labels = []
    for ck in case_keys:
        r = results[ck]
        m = r['info_metrics']
        densities.append(m['effective_pixel_density'])
        labels.append(ck.replace('.png',''))
    
    colors = ['#3498db', '#2ecc71']
    bars = axes[1].bar(labels, densities, color=colors, width=0.5)
    for bar, val in zip(bars, densities):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.2f}x', ha='center', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Effective Pixel Density Gain', fontsize=11)
    axes[1].set_title('(B) Crop-Based Detail Enhancement', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Panel C: Resolution ratio comparison
    res_ratios_fixed = []
    res_ratios_crop = []
    for ck in case_keys:
        r = results[ck]
        m = r['info_metrics']
        res_ratios_fixed.append(m['resolution_ratio'] * 100)
        # With cropping: each crop gets full encoder resolution for its region
        crop_coverage = m['region_coverage']
        crop_effective = crop_coverage * 100  # effective coverage with full-res crops
        res_ratios_crop.append(crop_effective)
    
    x2 = np.arange(len(case_keys))
    axes[2].bar(x2 - 0.2, res_ratios_fixed, 0.35, label='Fixed Encoding', color='#e74c3c')
    axes[2].bar(x2 + 0.2, res_ratios_crop, 0.35, label='Task-Guided Crops', color='#2ecc71')
    axes[2].set_xlabel('Demo Image', fontsize=11)
    axes[2].set_ylabel('Effective Resolution Coverage (%)', fontsize=11)
    axes[2].set_title('(C) Resolution Coverage Comparison', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels([ck.replace('.png','') for ck in case_keys])
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Information Preservation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'figure4_metrics.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: figure4_metrics.png")

generate_metrics_figure()

# ============================================================
# Figure 5: Resolution vs Detail Trade-off Curve
# ============================================================
def generate_tradeoff_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Resolution scaling curve
    test_resolutions = [(112,112), (224,224), (336,336), (448,448), (560,560), (672,672)]
    
    for idx, case_key in enumerate(list(results.keys())):
        r = results[case_key]
        ow, oh = r['image_size']
        total_px = ow * oh
        
        fixed_losses = []
        crop_gains = []
        
        for (ew, eh) in test_resolutions:
            enc_px = ew * eh
            loss = 1 - enc_px / total_px
            fixed_losses.append(loss * 100)
            
            # With 4 crops at this resolution covering ~25% area each
            n_crops = 4
            crop_region_px = total_px * 0.25
            effective_crop_px = n_crops * enc_px
            gain = effective_crop_px / crop_region_px
            crop_gains.append(gain)
        
        label = case_key.replace('.png','')
        axes[0].plot([f'{ew}²' for ew,eh in test_resolutions], fixed_losses, 
                     'o-', label=f'{label} (fixed)', linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Encoder Resolution', fontsize=11)
    axes[0].set_ylabel('Information Loss (%)', fontsize=11)
    axes[0].set_title('(A) Information Loss vs Encoder Resolution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis='x', rotation=30)
    
    # Panel B: Per-region detail preservation
    case_keys = list(results.keys())
    x = np.arange(len(case_keys))
    width = 0.35
    
    baseline_densities = []
    crop_densities = []
    
    for ck in case_keys:
        r = results[ck]
        m = r['info_metrics']
        # Baseline: 1x encoder resolution spread over whole image
        baseline_densities.append(m['resolution_ratio'])
        # Crop: N crops each at encoder resolution focused on salient regions
        crop_densities.append(m['effective_pixel_density'] * m['resolution_ratio'])
    
    axes[1].bar(x - width/2, baseline_densities, width, label='Baseline (Fixed)', color='#e74c3c')
    axes[1].bar(x + width/2, crop_densities, width, label='Task-Guided Crops', color='#2ecc71')
    axes[1].set_xlabel('Demo Image', fontsize=11)
    axes[1].set_ylabel('Relative Detail Preservation', fontsize=11)
    axes[1].set_title('(B) Detail Preservation per Salient Region', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([ck.replace('.png','') for ck in case_keys])
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add annotation
    axes[1].text(0.5, 0.95, 'Higher = better detail\npreservation in ROIs', 
                 transform=axes[1].transAxes, fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Resolution-Detail Trade-off Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'figure5_tradeoff.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: figure5_tradeoff.png")

generate_tradeoff_figure()

# ============================================================
# Figure 6: Qualitative question-answer simulation
# ============================================================
def generate_qa_figure():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    case_keys = list(results.keys())
    
    for idx, case_key in enumerate(case_keys):
        r = results[case_key]
        image = r['image']
        regions = r['regions']
        questions = r['questions']
        h, w = image.shape[:2]
        
        # Top row: image with question
        axes[idx, 0].imshow(image)
        q_text = '\n'.join([f'Q{i+1}: {q}' for i, q in enumerate(questions)])
        axes[idx, 0].text(0.02, 0.98, q_text, transform=axes[idx, 0].transAxes,
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        axes[idx, 0].set_title(f'{case_key}: Visual Questions', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Bottom row: cropped regions that would help answer
        n_show = min(2, len(regions))
        for j in range(n_show):
            if j < len(regions):
                y1, x1, y2, x2 = regions[j]
                crop = image[y1:y2, x1:x2]
                col = 1 if j == 0 else 1
                row_offset = 0
                # Use second column for first crop, show both crops in right panel
                if j == 0:
                    axes[idx, 1].imshow(crop)
                    axes[idx, 1].set_title(f'Salient Region {j+1}\n({x1},{y1})-({x2},{y2})\nSize: {x2-x1}×{y2-y1}',
                                           fontsize=10, fontweight='bold')
                    axes[idx, 1].axis('off')
                else:
                    # Create a combined view
                    pass
        
        # If more than 1 region, show a grid
        if len(regions) > 1:
            axes[idx, 1].clear()
            n_r = len(regions)
            cols = 2
            rows = (n_r + 1) // 2
            for j in range(n_r):
                y1, x1, y2, x2 = regions[j]
                crop = image[y1:y2, x1:x2]
                ax_sub = axes[idx, 1] if n_r <= 2 else None
                if n_r <= 2:
                    if j == 0:
                        axes[idx, 1].imshow(crop)
                        axes[idx, 1].set_title(f'ROI-{j+1}: ({x1},{y1})-({x2},{y2})',
                                               fontsize=10, fontweight='bold')
                        axes[idx, 1].axis('off')
                    else:
                        # For 2 regions, show side by side using subplots
                        pass
        
        # Simplified: just show the first crop zoomed
        if len(regions) > 0:
            y1, x1, y2, x2 = regions[0]
            crop = image[y1:y2, x1:x2]
            axes[idx, 1].imshow(crop)
            axes[idx, 1].set_title(f'Primary ROI (zoomed)\nRegion: ({x1},{y1})-({x2},{y2})\nOriginal: {x2-x1}×{y2-y1} → Encoded: 224×224',
                                   fontsize=10, fontweight='bold')
            axes[idx, 1].axis('off')
    
    plt.suptitle('Task-Guided Cropping: Enabling Fine-Grained Visual Question Answering',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'figure6_qa.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("  Saved: figure6_qa.png")

generate_qa_figure()

# ============================================================
# Save summary table as JSON
# ============================================================
summary = {
    'framework': 'Training-Free Task-Guided Cropping for MLLM Fine-Grained Perception',
    'baseline_encoders': {k: list(v) for k, v in ENCODER_RESOLUTIONS.items()},
    'crop_config': CROP_CONFIG,
    'results_per_image': {}
}

for ck, r in results.items():
    m = r['info_metrics']
    summary['results_per_image'][ck] = {
        'original_resolution': f"{r['image_size'][0]}×{r['image_size'][1]}",
        'original_pixels': m['original_pixels'],
        'baseline_encoded_pixels': m['encoded_pixels'],
        'pixel_loss_pct': round((1 - m['resolution_ratio']) * 100, 2),
        'num_detected_regions': len(r['regions']),
        'region_coverage_pct': round(m['region_coverage'] * 100, 2),
        'effective_density_gain': round(m['effective_pixel_density'], 2),
        'total_crop_pixels': m['total_crop_pixels'],
    }

with open(OUTPUT_DIR / 'summary_table.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR / 'summary_table.json'}")

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
