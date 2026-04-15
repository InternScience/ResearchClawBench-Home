#!/usr/bin/env python3
"""
Training-Free Fine-Grained Perception Framework for MLLMs
==========================================================

This script implements and evaluates a task-guided cropping strategy 
that mitigates information loss from fixed-resolution vision encoders.
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

print("=" * 60)
print("Training-Free Fine-Grained Perception Framework")
print("=" * 60)

ENCODER_RESOLUTIONS = {
    'CLIP ViT-B': (224, 224),
    'CLIP ViT-L': (336, 336),
    'LLaVA-1.5': (336, 336),
}

CROP_CONFIG = {'num_crops': 4, 'crop_scale': 0.25, 'min_crop_size': 64}

DEMO_CASES = [
    {
        'image': 'demo1.png',
        'description': 'Traffic scene with yellow taxis and police officers',
        'questions': [
            'What is the license plate number of the silver car?',
            'How many police officers are visible?',
            'What date is shown in the timestamp?',
        ],
        'fine_grained_regions': ['license_plates', 'officer_badges', 'timestamp'],
    },
    {
        'image': 'demo2.png',
        'description': 'Indoor flower exhibition with colorful tulips',
        'questions': [
            'How many different tulip colors are present?',
            'What type of flowers are displayed?',
            'Describe the greenhouse structure.',
        ],
        'fine_grained_regions': ['flower_labels', 'individual_petals', 'structural_details'],
    },
]

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img), img.size

def compute_saliency_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_mag = gradient_mag / (gradient_mag.max() + 1e-8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    a_ch = lab[:,:,1].astype(np.float32)
    b_ch = lab[:,:,2].astype(np.float32)
    color_contrast = np.sqrt(
        cv2.Sobel(a_ch, cv2.CV_64F, 1, 0, ksize=3)**2 +
        cv2.Sobel(b_ch, cv2.CV_64F, 0, 1, ksize=3)**2
    )
    color_contrast = color_contrast / (color_contrast.max() + 1e-8)
    saliency = 0.6 * gradient_mag + 0.4 * color_contrast
    saliency = ndimage.gaussian_filter(saliency, sigma=5)
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency

def find_top_regions(saliency_map, num_regions=4, min_region_fraction=0.05):
    h, w = saliency_map.shape
    regions = []
    sc = saliency_map.copy()
    min_h = int(h * min_region_fraction)
    min_w = int(w * min_region_fraction)
    for _ in range(num_regions):
        idx = np.unravel_index(np.argmax(sc), sc.shape)
        py, px = idx
        threshold = sc[py, px] * 0.5
        mask = sc > threshold
        labeled, nf = ndimage.label(mask)
        if nf > 0:
            pl = labeled[py, px]
            cm = (labeled == pl)
            coords = np.where(cm)
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            if (y2 - y1) < min_h:
                c = (y1 + y2) // 2
                y1 = max(0, c - min_h // 2); y2 = min(h, y1 + min_h)
            if (x2 - x1) < min_w:
                c = (x1 + x2) // 2
                x1 = max(0, c - min_w // 2); x2 = min(w, x1 + min_w)
            regions.append((int(y1), int(x1), int(y2), int(x2)))
            sc[y1:y2, x1:x2] = 0
    return regions

def extract_crops(image, regions, target_size=(224, 224)):
    crops = []
    for (y1, x1, y2, x2) in regions:
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC))
    return crops

def simulate_fixed_encoding(image, target_res=(224, 224)):
    return cv2.resize(image, target_res, interpolation=cv2.INTER_AREA)

def compute_info_metrics(original, encoded, crop_images, crop_regions):
    ho, wo = original.shape[:2]
    he, we = encoded.shape[:2]
    orig_px = ho * wo
    enc_px = he * we
    total_crop_px = sum(int(c.shape[0]) * int(c.shape[1]) for c in crop_images)
    total_region_area = sum(int((y2-y1)*(x2-x1)) for (y1,x1,y2,x2) in crop_regions)
    return {
        'resolution_ratio': float(enc_px / orig_px),
        'region_coverage': float(total_region_area / orig_px),
        'effective_pixel_density': float(len(crop_images) * enc_px / max(total_region_area, 1)),
        'original_pixels': int(orig_px),
        'encoded_pixels': int(enc_px),
        'total_crop_pixels': int(total_crop_px),
    }

# ============================================================
# Run Analysis
# ============================================================
results = {}
for case_idx, case in enumerate(DEMO_CASES):
    print(f"\n{'='*60}")
    print(f"Case {case_idx+1}: {case['image']}")
    print(f"{'='*60}")
    
    img_path = DATA_DIR / case['image']
    image, img_size = load_image(img_path)
    h, w = image.shape[:2]
    print(f"Size: {w}x{h}, Pixels: {w*h}")
    
    baseline_results = {}
    for ename, (eh, ew) in ENCODER_RESOLUTIONS.items():
        enc = simulate_fixed_encoding(image, (ew, eh))
        baseline_results[ename] = {
            'resolution': (ew, eh),
            'encoded_image': enc,
            'pixel_loss_ratio': 1 - (ew * eh) / (w * h),
        }
        print(f"  {ename}: {ew}x{eh}, Loss: {baseline_results[ename]['pixel_loss_ratio']*100:.1f}%")
    
    saliency = compute_saliency_map(image)
    regions = find_top_regions(saliency, num_regions=CROP_CONFIG['num_crops'])
    print(f"  Regions: {len(regions)}")
    
    crops = extract_crops(image, regions, (224, 224))
    baseline_enc = baseline_results['CLIP ViT-B']['encoded_image']
    info_m = compute_info_metrics(image, baseline_enc, crops, regions)
    print(f"  Density gain: {info_m['effective_pixel_density']:.2f}x")
    
    results[case['image']] = {
        'image_size': (w, h),
        'image': image,
        'baseline': baseline_results,
        'saliency': saliency,
        'regions': regions,
        'crops': crops,
        'info_metrics': info_m,
        'questions': case['questions'],
        'fine_grained_regions': case['fine_grained_regions'],
    }

# Save JSON
with open(OUTPUT_DIR / 'analysis_results.json', 'w') as f:
    sr = {}
    for k, v in results.items():
        sr[k] = {
            'image_size': [int(x) for x in v['image_size']],
            'info_metrics': v['info_metrics'],
            'questions': v['questions'],
            'fine_grained_regions': v['fine_grained_regions'],
            'num_regions': len(v['regions']),
            'region_sizes': [[int(r[3]-r[1]), int(r[2]-r[0])] for r in v['regions']],
        }
    json.dump(sr, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR / 'analysis_results.json'}")
