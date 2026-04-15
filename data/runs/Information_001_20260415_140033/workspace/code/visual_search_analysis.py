#!/usr/bin/env python3
"""
Visual Search Analysis for Fine-Grained MLLM Perception
Implements task-guided cropping strategy to mitigate information loss
from fixed-resolution vision encoders.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import os
from typing import Tuple, List, Dict, Optional
import cv2

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


def load_image(image_path: str) -> np.ndarray:
    """Load an image and convert to numpy array."""
    img = Image.open(image_path)
    return np.array(img)


def simulate_clip_encoding(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Simulate CLIP encoder processing by resizing to fixed resolution.
    This represents the information bottleneck from fixed-resolution encoders.
    """
    img_pil = Image.fromarray(image)
    resized = img_pil.resize(target_size, Image.LANCZOS)
    return np.array(resized)


def calculate_saliency_map(image: np.ndarray, question: str = None) -> np.ndarray:
    """
    Simulate a task-guided saliency map based on image features.
    In practice, this would use attention mechanisms or learned models.
    """
    # Convert to grayscale for basic saliency
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Simple saliency using Gaussian difference (simulating visual attention)
    gaussian1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gaussian2 = cv2.GaussianBlur(gray, (5, 5), 2.0)
    saliency = np.abs(gaussian1 - gaussian2)
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def identify_regions_of_interest(
    saliency_map: np.ndarray, 
    num_regions: int = 3,
    min_region_size: Tuple[int, int] = (50, 50)
) -> List[Tuple[int, int, int, int]]:
    """
    Identify top regions of interest based on saliency.
    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    regions = []
    temp_saliency = saliency_map.copy()
    
    h, w = saliency_map.shape
    
    for _ in range(num_regions):
        # Find maximum saliency point
        max_idx = np.unravel_index(np.argmax(temp_saliency), temp_saliency.shape)
        cy, cx = max_idx
        
        # Define region around maximum (adaptive size based on image dimensions)
        region_h = min(h // 3, 200)
        region_w = min(w // 3, 200)
        
        x1 = max(0, cx - region_w // 2)
        y1 = max(0, cy - region_h // 2)
        x2 = min(w, cx + region_w // 2)
        y2 = min(h, cy + region_h // 2)
        
        regions.append((x1, y1, x2, y2))
        
        # Suppress this region to find next
        temp_saliency[y1:y2, x1:x2] = 0
    
    return regions


def extract_and_enhance_region(
    image: np.ndarray, 
    bbox: Tuple[int, int, int, int],
    enhance_scale: int = 2
) -> np.ndarray:
    """
    Extract a region and enhance (zoom) it for better detail.
    """
    x1, y1, x2, y2 = bbox
    region = image[y1:y2, x1:x2]
    
    # Upsample for enhanced detail
    h, w = region.shape[:2]
    new_h, new_w = h * enhance_scale, w * enhance_scale
    
    region_pil = Image.fromarray(region)
    enhanced = region_pil.resize((new_w, new_h), Image.LANCZOS)
    
    return np.array(enhanced)


def compute_detail_preservation_score(
    original: np.ndarray, 
    encoded: np.ndarray
) -> float:
    """
    Compute a metric for detail preservation.
    Higher values indicate better preservation of fine details.
    """
    # Use gradient magnitude as proxy for detail
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enc_gray = cv2.cvtColor(encoded, cv2.COLOR_RGB2GRAY)
    else:
        orig_gray = original
        enc_gray = encoded
    
    # Resize encoded back to original for comparison
    enc_resized = cv2.resize(enc_gray, (orig_gray.shape[1], orig_gray.shape[0]))
    
    # Compute gradients
    orig_grad_x = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
    orig_grad_y = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
    orig_gradient = np.sqrt(orig_grad_x**2 + orig_grad_y**2)
    
    enc_grad_x = cv2.Sobel(enc_resized, cv2.CV_64F, 1, 0, ksize=3)
    enc_grad_y = cv2.Sobel(enc_resized, cv2.CV_64F, 0, 1, ksize=3)
    enc_gradient = np.sqrt(enc_grad_x**2 + enc_grad_y**2)
    
    # Preservation ratio (how much gradient/detail is preserved)
    preservation = np.mean(enc_gradient) / (np.mean(orig_gradient) + 1e-8)
    return min(preservation, 1.0)


def visualize_visual_search_process(
    image_path: str,
    question: str,
    output_path: str,
    num_regions: int = 3
):
    """
    Create a comprehensive visualization of the visual search process.
    """
    image = load_image(image_path)
    h, w = image.shape[:2]
    
    # Simulate CLIP encoding (information loss)
    clip_encoded = simulate_clip_encoding(image, target_size=(224, 224))
    
    # Generate saliency map
    saliency = calculate_saliency_map(image, question)
    
    # Identify regions of interest
    regions = identify_regions_of_interest(saliency, num_regions=num_regions)
    
    # Extract and enhance regions
    enhanced_regions = [extract_and_enhance_region(image, r) for r in regions]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Original image with regions highlighted
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(image)
    ax1.set_title('Original High-Resolution Image\n(With Regions of Interest)', fontsize=12, fontweight='bold')
    colors = ['cyan', 'yellow', 'magenta']
    for i, (bbox, color) in enumerate(zip(regions, colors)):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                   edgecolor=color, facecolor='none', linestyle='--')
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f'ROI {i+1}', color=color, fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # CLIP encoded (information loss demonstration)
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.imshow(clip_encoded)
    ax2.set_title('CLIP Fixed-Resolution Encoding (224×224)\n(Information Loss)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Saliency map
    ax3 = fig.add_subplot(gs[1, :2])
    im = ax3.imshow(saliency, cmap='viridis')
    ax3.set_title('Task-Guided Saliency Map\n(Identifies Regions Needing Detail)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    ax3.axis('off')
    
    # Detail preservation comparison
    ax4 = fig.add_subplot(gs[1, 2:])
    global_score = compute_detail_preservation_score(image, clip_encoded)
    local_scores = []
    for bbox in regions:
        x1, y1, x2, y2 = bbox
        local_region = image[y1:y2, x1:x2]
        local_clip = simulate_clip_encoding(local_region, target_size=(224, 224))
        score = compute_detail_preservation_score(local_region, local_clip)
        local_scores.append(score)
    
    methods = ['Global (CLIP)'] + [f'ROI {i+1}' for i in range(len(regions))]
    scores = [global_score] + local_scores
    colors_bar = ['red'] + colors[:len(regions)]
    
    bars = ax4.bar(methods, scores, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Detail Preservation Score', fontsize=11)
    ax4.set_title('Detail Preservation: Global vs Local Processing', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.axhline(y=global_score, color='red', linestyle='--', alpha=0.5, label='Global baseline')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Enhanced regions
    for i, (region, bbox) in enumerate(zip(enhanced_regions, regions)):
        ax = fig.add_subplot(gs[2 + i//2, (i%2)*2:(i%2)*2+2])
        ax.imshow(region)
        x1, y1, x2, y2 = bbox
        ax.set_title(f'Enhanced ROI {i+1} ({x2-x1}×{y2-y1} → {region.shape[1]}×{region.shape[0]})\n'
                    f'Detail Preservation: {local_scores[i]:.2f}', 
                    fontsize=11, fontweight='bold', color=colors[i])
        ax.axis('off')
    
    # Question and summary
    fig.text(0.5, 0.02, f'Question: "{question}"', ha='center', fontsize=14, 
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'global_score': global_score,
        'local_scores': local_scores,
        'regions': regions,
        'image_shape': image.shape
    }


def create_information_loss_analysis(image_paths: List[str], output_dir: str):
    """
    Create analysis comparing information loss across different image types.
    """
    results = []
    
    for img_path in image_paths:
        image = load_image(img_path)
        filename = os.path.basename(img_path)
        
        # Test different CLIP resolutions
        resolutions = [(224, 224), (336, 336), (448, 448)]
        resolution_scores = []
        
        for res in resolutions:
            encoded = simulate_clip_encoding(image, target_size=res)
            score = compute_detail_preservation_score(image, encoded)
            resolution_scores.append(score)
        
        results.append({
            'filename': filename,
            'image_shape': image.shape,
            'resolutions': resolutions,
            'scores': resolution_scores
        })
    
    # Create visualization
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        x_labels = [f'{r[0]}×{r[1]}' for r in result['resolutions']]
        bars = ax.bar(x_labels, result['scores'], color=['#e74c3c', '#f39c12', '#27ae60'], 
                     alpha=0.8, edgecolor='black')
        ax.set_ylabel('Detail Preservation Score', fontsize=11)
        ax.set_title(f'{result["filename"]}\n{result["image_shape"][0]}×{result["image_shape"][1]}', 
                    fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, result['scores']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'information_loss_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def create_method_comparison(output_dir: str):
    """
    Create a comparison table/chart of different approaches to fine-grained perception.
    """
    methods = [
        'Standard CLIP',
        'Higher Resolution CLIP',
        'Multi-Scale Features',
        'Task-Guided Cropping (ViCrop)',
        'Visual Search (SEAL/V*)'
    ]
    
    # Attributes comparison
    attributes = {
        'Training Required': [False, True, True, False, True],
        'Handles Small Objects': [False, True, True, True, True],
        'Task-Adaptive': [False, False, False, True, True],
        'Computational Cost': ['Low', 'High', 'Medium', 'Medium', 'High'],
        'Integration Ease': ['Baseline', 'Hard', 'Medium', 'Easy', 'Medium']
    }
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Categories for radar chart
    categories = ['Small Object Detection', 'Task Adaptivity', 'Efficiency', 
                 'Ease of Integration', 'Detail Preservation']
    N = len(categories)
    
    # Scores for each method (subjective but based on paper analysis)
    scores = {
        'Standard CLIP': [1, 1, 5, 5, 1],
        'Task-Guided Cropping': [4, 5, 3, 4, 4],
        'Visual Search (SEAL)': [5, 5, 2, 3, 5]
    }
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for (method, score), color in zip(scores.items(), colors):
        values = score + score[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Method Comparison: Fine-Grained Perception\n(1=Poor, 5=Excellent)', 
                fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_framework_architecture_diagram(output_dir: str):
    """
    Create a visual diagram of the task-guided cropping framework.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define boxes
    boxes = [
        # (x, y, w, h, text, color)
        (0.5, 8, 2, 1, 'Input Image\n(High Resolution)', '#3498db'),
        (3.5, 8, 2.5, 1, 'Global CLIP Encoding\n(224×224)', '#e74c3c'),
        (7, 8, 2.5, 1, 'VQA LLM\nInitial Assessment', '#9b59b6'),
        
        (0.5, 5.5, 2.5, 1, 'Task Analysis\n(What is needed?)', '#f39c12'),
        (3.5, 5.5, 2.5, 1, 'Visual Search /\nSaliency Detection', '#f39c12'),
        (7, 5.5, 2.5, 1, 'Region Proposal\n(ROIs)', '#f39c12'),
        
        (0.5, 3, 2.5, 1, 'Crop & Enhance\n(Zoom)', '#27ae60'),
        (3.5, 3, 2.5, 1, 'Local CLIP Encoding\n(High Detail)', '#27ae60'),
        (7, 3, 2.5, 1, 'Visual Working\nMemory (VWM)', '#27ae60'),
        
        (3.5, 0.5, 3, 1, 'Final VQA with\nGlobal + Local Context', '#9b59b6'),
    ]
    
    # Draw boxes
    for x, y, w, h, text, color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                                       linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
    
    # Draw arrows
    arrows = [
        ((2.5, 8.5), (3.5, 8.5)),
        ((6, 8.5), (7, 8.5)),
        ((8.25, 8), (8.25, 6.5)),
        ((8.25, 5.5), (7, 5.5)),
        ((5.75, 5.5), (5.5, 6)),  # feedback
        ((3.5, 6), (1.75, 6)),
        ((3, 5.5), (3, 4)),
        ((4.75, 5.5), (4.75, 4)),
        ((7, 5.5), (7, 4)),
        ((1.75, 3), (2.5, 3)),
        ((6, 3), (7, 3)),
        ((8.25, 3), (8.25, 1.5), (6.5, 1)),
        ((2.5, 3), (2.5, 1.5), (3.5, 1)),
        ((5, 3), (5, 1.5)),
    ]
    
    for arrow in arrows:
        if len(arrow) == 2:
            ax.annotate('', xy=arrow[1], xytext=arrow[0],
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        else:
            # Multi-segment arrow
            for i in range(len(arrow)-1):
                ax.annotate('', xy=arrow[i+1], xytext=arrow[i],
                           arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add labels
    ax.text(5, 9.5, 'Task-Guided Visual Search Framework (SEAL/ViCrop)', 
           ha='center', fontsize=14, fontweight='bold')
    ax.text(5, 9.1, 'Training-Free Fine-Grained Perception for MLLMs', 
           ha='center', fontsize=11, style='italic')
    
    # Legend
    legend_items = [
        ('#3498db', 'Input'),
        ('#e74c3c', 'Encoding (Information Loss)'),
        ('#f39c12', 'Visual Search'),
        ('#27ae60', 'Enhancement'),
        ('#9b59b6', 'Reasoning'),
    ]
    
    for i, (color, label) in enumerate(legend_items):
        rect = patches.Rectangle((0.5 + i*1.8, -0.8), 0.3, 0.3, facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(0.9 + i*1.8, -0.65, label, fontsize=9, va='center')
    
    plt.savefig(os.path.join(output_dir, 'framework_architecture.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    # Setup paths
    data_dir = 'data/demo_imgs'
    output_dir = 'outputs'
    report_img_dir = 'report/images'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_img_dir, exist_ok=True)
    
    # Process demo images
    demo_images = [
        os.path.join(data_dir, 'demo1.png'),
        os.path.join(data_dir, 'demo2.png')
    ]
    
    questions = [
        "What is the license plate number of the silver car?",
        "How many yellow tulips are in the flower bed?"
    ]
    
    print("Processing demo images with visual search analysis...")
    all_results = []
    
    for img_path, question in zip(demo_images, questions):
        if os.path.exists(img_path):
            print(f"  Processing {os.path.basename(img_path)}...")
            result = visualize_visual_search_process(
                img_path, 
                question, 
                os.path.join(report_img_dir, f'visual_search_{os.path.basename(img_path).replace(".png", "")}.png')
            )
            result['filename'] = os.path.basename(img_path)
            all_results.append(result)
    
    # Information loss analysis
    print("Creating information loss analysis...")
    loss_results = create_information_loss_analysis(demo_images, report_img_dir)
    
    # Method comparison
    print("Creating method comparison visualization...")
    create_method_comparison(report_img_dir)
    
    # Framework architecture
    print("Creating framework architecture diagram...")
    create_framework_architecture_diagram(report_img_dir)
    
    # Save results to JSON
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump({
            'visual_search_results': all_results,
            'information_loss_results': loss_results
        }, f, indent=2, default=convert_to_native)
    
    print("Analysis complete! Results saved to outputs/ and report/images/")
