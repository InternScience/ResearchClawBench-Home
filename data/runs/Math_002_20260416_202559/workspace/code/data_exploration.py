#!/usr/bin/env python3
"""
Data exploration script for MAPF datasets.
Analyzes map structures, obstacle densities, and agent configurations.
"""

import numpy as np
import os
import json
from pathlib import Path

WORKSPACE_ROOT = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Math_002_20260416_202559"
DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")

def analyze_map_file(filepath):
    """Analyze a single map file."""
    try:
        data = np.load(filepath, allow_pickle=True)
        if data.ndim == 2:
            height, width = data.shape
            obstacles = np.sum(data == -1)
            free_cells = np.sum(data == 0)
            obstacle_density = obstacles / (height * width)
            return {
                'type': 'map',
                'width': width,
                'height': height,
                'total_cells': height * width,
                'obstacles': int(obstacles),
                'free_cells': int(free_cells),
                'obstacle_density': float(obstacle_density)
            }
        else:
            return {'type': 'other', 'shape': list(data.shape)}
    except Exception as e:
        return {'error': str(e)}

def scan_dataset_folder(folder_path):
    """Scan a dataset folder and collect statistics."""
    results = {
        'folder': os.path.basename(folder_path),
        'files': [],
        'subfolders': []
    }
    
    for item in sorted(os.listdir(folder_path)):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) and item.endswith('.npy'):
            stats = analyze_map_file(item_path)
            stats['filename'] = item
            results['files'].append(stats)
        elif os.path.isdir(item_path):
            subfolder_stats = scan_dataset_folder(item_path)
            results['subfolders'].append(subfolder_stats)
    
    return results

def compute_dataset_summary(dataset_results):
    """Compute summary statistics for a dataset."""
    maps = []
    
    def collect_maps(results):
        for f in results.get('files', []):
            if f.get('type') == 'map':
                maps.append(f)
        for sub in results.get('subfolders', []):
            collect_maps(sub)
    
    collect_maps(dataset_results)
    
    if not maps:
        return None
    
    widths = [m['width'] for m in maps]
    heights = [m['height'] for m in maps]
    densities = [m['obstacle_density'] for m in maps]
    obstacles = [m['obstacles'] for m in maps]
    
    return {
        'num_maps': len(maps),
        'width_range': [min(widths), max(widths)],
        'height_range': [min(heights), max(widths)],
        'avg_width': np.mean(widths),
        'avg_height': np.mean(heights),
        'avg_obstacle_density': np.mean(densities),
        'std_obstacle_density': np.std(densities),
        'avg_obstacles': np.mean(obstacles),
        'std_obstacles': np.std(obstacles)
    }

def main():
    print("=" * 60)
    print("MAPF Dataset Exploration")
    print("=" * 60)
    
    all_summaries = {}
    
    # Scan each dataset type
    dataset_types = ['empty', 'maze', 'random_large', 'random_medium', 
                     'random_small', 'room', 'warehouse', 'maps_60_10_10_0.175']
    
    for dataset_type in dataset_types:
        dataset_path = os.path.join(DATA_ROOT, dataset_type)
        if os.path.exists(dataset_path):
            print(f"\n--- Analyzing: {dataset_type} ---")
            results = scan_dataset_folder(dataset_path)
            summary = compute_dataset_summary(results)
            if summary:
                all_summaries[dataset_type] = summary
                print(f"  Number of maps: {summary['num_maps']}")
                print(f"  Map size: {summary['width_range'][0]}x{summary['height_range'][0]} to {summary['width_range'][1]}x{summary['height_range'][1]}")
                print(f"  Avg obstacle density: {summary['avg_obstacle_density']:.3f} (+/- {summary['std_obstacle_density']:.3f})")
                print(f"  Avg obstacles: {summary['avg_obstacles']:.1f} (+/- {summary['std_obstacles']:.1f})")
    
    # Save summary
    output_path = os.path.join(WORKSPACE_ROOT, "outputs", "data_summary.json")
    with open(output_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to: {output_path}")
    
    return all_summaries

if __name__ == "__main__":
    main()
