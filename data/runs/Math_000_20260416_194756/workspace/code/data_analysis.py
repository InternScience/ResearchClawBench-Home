#!/usr/bin/env python3
"""
Data analysis module for multi-object tracking evaluation.
Analyzes the simulated sequence data and generates overview statistics.
"""

import json
import numpy as np
from pathlib import Path

def load_data(data_path):
    """Load the simulated sequence JSON data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_data(data):
    """Analyze the simulated sequence data and return statistics."""
    num_frames = len(data)
    
    # Collect statistics per frame
    frame_stats = []
    all_detection_scores = []
    all_gt_ids = set()
    occlusion_counts = []
    
    for frame_data in data:
        frame_num = frame_data['frame']
        gt_bboxes = frame_data.get('gt_bboxes', [])
        gt_ids = frame_data.get('gt_ids', [])
        detections = frame_data.get('detections', [])
        
        num_gt = len(gt_bboxes)
        num_detections = len(detections)
        detection_rate = num_detections / num_gt if num_gt > 0 else 0
        
        scores = [d['score'] for d in detections]
        all_detection_scores.extend(scores)
        
        # Count occluded detections (those with lower scores typically indicate occlusion)
        occluded_count = sum(1 for d in detections if d['score'] < 0.3)
        
        all_gt_ids.update(gt_ids)
        
        frame_stats.append({
            'frame': frame_num,
            'num_gt': num_gt,
            'num_detections': num_detections,
            'detection_rate': detection_rate,
            'avg_score': np.mean(scores) if scores else 0,
            'occluded_count': occluded_count
        })
        occlusion_counts.append(occluded_count)
    
    summary = {
        'num_frames': num_frames,
        'total_objects': len(all_gt_ids),
        'avg_detections_per_frame': np.mean([s['num_detections'] for s in frame_stats]),
        'avg_detection_rate': np.mean([s['detection_rate'] for s in frame_stats]),
        'avg_detection_score': np.mean(all_detection_scores) if all_detection_scores else 0,
        'detection_score_std': np.std(all_detection_scores) if all_detection_scores else 0,
        'min_detection_score': min(all_detection_scores) if all_detection_scores else 0,
        'max_detection_score': max(all_detection_scores) if all_detection_scores else 0,
        'frame_stats': frame_stats,
        'occlusion_stats': {
            'avg_occluded_per_frame': np.mean(occlusion_counts),
            'total_occluded_detections': sum(occlusion_counts)
        }
    }
    
    return summary

def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union between two bounding boxes.
    Bboxes are in format [x1, y1, x2, y2].
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    
    # Compute areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def estimate_pseudo_depth(bbox, frame_height=640):
    """
    Estimate pseudo-depth from bounding box.
    Smaller boxes (higher in image or smaller area) are assumed to be farther.
    Uses a combination of vertical position and box area.
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    center_y = (y1 + y2) / 2
    
    # Normalize: higher y (lower in image) = closer, larger area = closer
    # Pseudo-depth: lower value = closer, higher value = farther
    depth_from_y = center_y / frame_height  # 0 = top (far), 1 = bottom (close)
    depth_from_area = 1.0 / (1.0 + area / 10000)  # Larger area = smaller depth
    
    # Combine: weight vertical position more heavily
    pseudo_depth = 0.7 * (1 - depth_from_y) + 0.3 * depth_from_area
    return pseudo_depth

if __name__ == '__main__':
    data_path = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_000_20260416_194756/data/simulated_sequence.json')
    output_path = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_000_20260416_194756/outputs/data_analysis.json')
    
    print("Loading data...")
    data = load_data(data_path)
    
    print("Analyzing data...")
    summary = analyze_data(data)
    
    # Save summary
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Data Analysis Summary ===")
    print(f"Number of frames: {summary['num_frames']}")
    print(f"Total unique objects: {summary['total_objects']}")
    print(f"Average detections per frame: {summary['avg_detections_per_frame']:.2f}")
    print(f"Average detection rate: {summary['avg_detection_rate']:.2%}")
    print(f"Average detection score: {summary['avg_detection_score']:.3f}")
    print(f"Detection score range: [{summary['min_detection_score']:.3f}, {summary['max_detection_score']:.3f}]")
    print(f"Average occluded detections per frame: {summary['occlusion_stats']['avg_occluded_per_frame']:.2f}")
    print(f"\nResults saved to: {output_path}")
