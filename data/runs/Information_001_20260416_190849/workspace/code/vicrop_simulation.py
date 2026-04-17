"""
Simulate the ViCrop-like task-guided cropping approach on demo images.
Demonstrates how the framework would work for specific visual questions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
import json
import os

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Information_001_20260416_190849"
DATA_DIR = os.path.join(WORKSPACE, "data", "demo_imgs")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMG_DIR = os.path.join(WORKSPACE, "report", "images")

demo1 = Image.open(os.path.join(DATA_DIR, "demo1.png")).convert("RGB")
demo2 = Image.open(os.path.join(DATA_DIR, "demo2.png")).convert("RGB")

# ============================================================
# Simulate task-specific questions and cropping
# ============================================================

# For demo1 (street scene with taxis):
# Question: "What is the license plate number of the silver car?"
# The silver car is in the center - license plate is a small detail

# For demo2 (flower exhibition):
# Question: "What color are the flowers in the bottom-right corner?"
# Small region with specific flower details

def simulate_vicrop_pipeline(img, question, roi_bbox, name="demo"):
    """
    Simulate the complete ViCrop pipeline:
    1. Process full image at CLIP resolution (global context)
    2. Identify region of interest based on question
    3. Crop and zoom into ROI
    4. Compare information available at each stage
    """
    x1, y1, x2, y2 = roi_bbox
    
    # Stage 1: Global encoding at CLIP-224
    global_224 = img.resize((224, 224), Image.BILINEAR)
    
    # Stage 2: The ROI as it appears in the global encoding
    # Calculate where the ROI maps to in the 224x224 image
    w, h = img.size
    scale_x = 224 / w
    scale_y = 224 / h
    roi_in_global = (int(x1 * scale_x), int(y1 * scale_y), 
                     int(x2 * scale_x), int(y2 * scale_y))
    roi_from_global = global_224.crop(roi_in_global)
    
    # Stage 3: Direct crop from original at CLIP resolution
    crop_original = img.crop((x1, y1, x2, y2))
    crop_at_224 = crop_original.resize((224, 224), Image.LANCZOS)
    crop_at_448 = crop_original.resize((448, 448), Image.LANCZOS)
    
    # Compute quality metrics for each stage
    def compute_detail_score(pil_img):
        arr = np.array(pil_img.convert("L"))
        # Edge density
        edges = cv2.Canny(arr, 50, 150)
        edge_density = np.mean(edges > 0)
        # Local contrast
        local_std = ndimage.generic_filter(arr.astype(np.float64), np.std, size=5)
        contrast = np.mean(local_std)
        # Entropy
        hist, _ = np.histogram(arr, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return {"edge_density": float(edge_density), "contrast": float(contrast), "entropy": float(entropy)}
    
    results = {
        "question": question,
        "roi_bbox": roi_bbox,
        "global_224_metrics": compute_detail_score(global_224),
        "roi_from_global_metrics": compute_detail_score(roi_from_global) if roi_from_global.size[0] > 2 and roi_from_global.size[1] > 2 else {"edge_density": 0, "contrast": 0, "entropy": 0},
        "crop_at_224_metrics": compute_detail_score(crop_at_224),
        "crop_at_448_metrics": compute_detail_score(crop_at_448),
    }
    
    # Calculate improvement
    global_ent = results["global_224_metrics"]["entropy"]
    crop448_ent = results["crop_at_448_metrics"]["entropy"]
    results["entropy_improvement"] = (crop448_ent - global_ent) / global_ent * 100
    
    global_edge = results["global_224_metrics"]["edge_density"]
    crop448_edge = results["crop_at_448_metrics"]["edge_density"]
    results["edge_improvement"] = (crop448_edge - global_edge) / global_edge * 100 if global_edge > 0 else 0
    
    return results, global_224, roi_from_global, crop_at_224, crop_at_448, crop_original

# Define task-specific scenarios
scenarios = [
    {
        "name": "demo1",
        "img": demo1,
        "question": "What is the license plate number of the silver car in the center?",
        "roi_bbox": (390, 400, 620, 550),  # Silver car area
        "description": "License plate detail on silver car"
    },
    {
        "name": "demo1",
        "img": demo1,
        "question": "What text is on the building sign on the left?",
        "roi_bbox": (0, 50, 350, 200),  # Building sign area
        "description": "Building signage text"
    },
    {
        "name": "demo2",
        "img": demo2,
        "question": "What color are the flowers in the bottom-right corner?",
        "roi_bbox": (1700, 1100, 2250, 1500),  # Bottom-right flowers
        "description": "Specific flower colors in corner"
    },
    {
        "name": "demo2",
        "img": demo2,
        "question": "How many people are visible in the background?",
        "roi_bbox": (0, 0, 2250, 500),  # Background area
        "description": "People in background of exhibition"
    }
]

# ============================================================
# Generate ViCrop Simulation Figure
# ============================================================
print("Generating ViCrop Simulation Figure...")

fig = plt.figure(figsize=(22, 20))

all_results = []

for s_idx, scenario in enumerate(scenarios):
    results, global_224, roi_from_global, crop_224, crop_448, crop_original = \
        simulate_vicrop_pipeline(scenario["img"], scenario["question"], scenario["roi_bbox"], scenario["name"])
    all_results.append(results)
    
    row_base = s_idx * 5
    
    # Row for each scenario: Original with ROI, Global@224, ROI from Global, Crop@224, Crop@448
    ax1 = fig.add_subplot(4, 5, s_idx * 5 + 1)
    ax1.imshow(np.array(scenario["img"]))
    x1, y1, x2, y2 = scenario["roi_bbox"]
    rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.set_title(f'Original Image\n"{scenario["question"][:40]}..."', fontsize=8)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(4, 5, s_idx * 5 + 2)
    ax2.imshow(np.array(global_224))
    ax2.set_title(f'Global@224\nEntropy: {results["global_224_metrics"]["entropy"]:.2f}', fontsize=8)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(4, 5, s_idx * 5 + 3)
    if roi_from_global.size[0] > 2 and roi_from_global.size[1] > 2:
        roi_display = roi_from_global.resize((112, 112), Image.NEAREST)
        ax3.imshow(np.array(roi_display))
        ax3.set_title(f'ROI in Global\n(heavily degraded)', fontsize=8, color='red')
    else:
        ax3.text(0.5, 0.5, 'ROI too small\nin global view', ha='center', va='center', fontsize=8)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(4, 5, s_idx * 5 + 4)
    ax4.imshow(np.array(crop_224))
    ax4.set_title(f'Crop@224\nEntropy: {results["crop_at_224_metrics"]["entropy"]:.2f}', fontsize=8)
    ax4.axis('off')
    
    ax5 = fig.add_subplot(4, 5, s_idx * 5 + 5)
    ax5.imshow(np.array(crop_448))
    ax5.set_title(f'Crop@448 (ViCrop)\nEntropy: {results["crop_at_448_metrics"]["entropy"]:.2f}', fontsize=8, color='green')
    ax5.axis('off')

plt.suptitle('ViCrop-Style Task-Guided Cropping: Question-Driven ROI Processing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "vicrop_simulation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved vicrop_simulation.png")

# Save simulation results
with open(os.path.join(OUTPUT_DIR, "vicrop_simulation_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# ============================================================
# Generate Summary Bar Chart
# ============================================================
print("Generating Summary Bar Chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scenario_labels = [f"S{i+1}: {s['description'][:25]}" for i, s in enumerate(scenarios)]
entropy_improvements = [r["entropy_improvement"] for r in all_results]
edge_improvements = [r["edge_improvement"] for r in all_results]

colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in entropy_improvements]
axes[0].barh(range(len(scenario_labels)), entropy_improvements, color=colors, alpha=0.8)
axes[0].set_yticks(range(len(scenario_labels)))
axes[0].set_yticklabels(scenario_labels, fontsize=9)
axes[0].set_xlabel('Entropy Improvement (%)')
axes[0].set_title('Information Entropy Gain\n(Crop@448 vs Global@224)')
axes[0].axvline(x=0, color='black', linewidth=0.5)
for i, v in enumerate(entropy_improvements):
    axes[0].text(v + 0.1, i, f'{v:+.1f}%', va='center', fontsize=9)

colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in edge_improvements]
axes[1].barh(range(len(scenario_labels)), edge_improvements, color=colors, alpha=0.8)
axes[1].set_yticks(range(len(scenario_labels)))
axes[1].set_yticklabels(scenario_labels, fontsize=9)
axes[1].set_xlabel('Edge Density Improvement (%)')
axes[1].set_title('Edge Detail Gain\n(Crop@448 vs Global@224)')
axes[1].axvline(x=0, color='black', linewidth=0.5)
for i, v in enumerate(edge_improvements):
    axes[1].text(v + 0.5, i, f'{v:+.1f}%', va='center', fontsize=9)

plt.suptitle('Task-Guided Cropping: Quantitative Improvement Summary', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "improvement_summary.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved improvement_summary.png")

# Print summary
print("\n=== ViCrop Simulation Summary ===")
for i, (s, r) in enumerate(zip(scenarios, all_results)):
    print(f"\nScenario {i+1}: {s['description']}")
    print(f"  Question: {s['question']}")
    print(f"  Global@224 entropy: {r['global_224_metrics']['entropy']:.3f}")
    print(f"  Crop@448 entropy:   {r['crop_at_448_metrics']['entropy']:.3f}")
    print(f"  Entropy improvement: {r['entropy_improvement']:+.1f}%")
    print(f"  Edge improvement:    {r['edge_improvement']:+.1f}%")

print("\nDone!")
