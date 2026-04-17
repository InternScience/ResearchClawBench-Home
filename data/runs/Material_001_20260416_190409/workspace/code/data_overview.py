#!/usr/bin/env python3
"""
Data Overview Module for Materials AI

This module generates comprehensive data overview visualizations and
summary statistics for the multimodal materials dataset.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns


def parse_full_dataset(filepath):
    """
    Parse all sections of the M-AI-Synth dataset.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    sections = {}
    current_section = None
    arrays_in_section = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for section headers
        if 'property_prediction' in line.lower():
            if current_section and arrays_in_section:
                sections[current_section] = arrays_in_section
            current_section = 'property_prediction'
            arrays_in_section = []
        elif 'structure_generation' in line.lower():
            if current_section and arrays_in_section:
                sections[current_section] = arrays_in_section
            current_section = 'structure_generation'
            arrays_in_section = []
        elif 'autonomous_optimization' in line.lower():
            if current_section and arrays_in_section:
                sections[current_section] = arrays_in_section
            current_section = 'autonomous_optimization'
            arrays_in_section = []
        elif line_stripped.startswith('['):
            # Parse array
            array_str = line_stripped.strip('[]')
            values = [float(x) for x in array_str.split(', ') if x.strip()]
            arrays_in_section.append(values)
    
    # Save last section
    if current_section and arrays_in_section:
        sections[current_section] = arrays_in_section
    
    return sections


def generate_data_overview_plots(sections, output_dir):
    """
    Generate comprehensive data overview visualizations.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # === Row 1: Property Prediction Data ===
    
    # Plot 1: Feature distribution (constant features)
    ax = axes[0, 0]
    prop_features = sections['property_prediction'][0]
    ax.hist(prop_features, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Property Prediction: Input Features')
    ax.axvline(np.mean(prop_features), color='red', linestyle='--', 
               label=f'Mean: {np.mean(prop_features):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Target property distribution
    ax = axes[0, 1]
    prop_targets = sections['property_prediction'][1]
    ax.hist(prop_targets, bins=20, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Property Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Property Prediction: Target Values')
    ax.axvline(np.mean(prop_targets), color='blue', linestyle='--',
               label=f'Mean: {np.mean(prop_targets):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Class distribution
    ax = axes[0, 2]
    prop_classes = sections['property_prediction'][2]
    unique_classes, counts = np.unique(prop_classes, return_counts=True)
    bars = ax.bar(unique_classes.astype(int), counts, color='#9b59b6', edgecolor='black')
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Count')
    ax.set_title('Property Prediction: Class Distribution')
    ax.set_xticks(unique_classes.astype(int))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    # === Row 2: Structure Generation Data ===
    
    # Plot 4: Lattice parameter a distribution
    ax = axes[1, 0]
    struct_a = sections['structure_generation'][0]
    ax.hist(struct_a, bins=15, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax.set_xlabel('Lattice Parameter a (Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('Structure Generation: Lattice a')
    ax.axvline(np.mean(struct_a), color='red', linestyle='--',
               label=f'Mean: {np.mean(struct_a):.2f} Å')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Lattice parameter b distribution
    ax = axes[1, 1]
    struct_b = sections['structure_generation'][1]
    ax.hist(struct_b, bins=15, alpha=0.7, color='#f39c12', edgecolor='black')
    ax.set_xlabel('Lattice Parameter b (Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('Structure Generation: Lattice b')
    ax.axvline(np.mean(struct_b), color='red', linestyle='--',
               label=f'Mean: {np.mean(struct_b):.2f} Å')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Lattice parameters scatter
    ax = axes[1, 2]
    ax.scatter(struct_a, struct_b, alpha=0.6, color='#1abc9c', s=50, edgecolors='black')
    ax.set_xlabel('Lattice a (Å)')
    ax.set_ylabel('Lattice b (Å)')
    ax.set_title('Structure Generation: Lattice Parameter Space')
    corr = np.corrcoef(struct_a, struct_b)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)
    
    # === Row 3: Optimization Data ===
    
    # Plot 7: Temperature range
    ax = axes[2, 0]
    opt_temp = sections['autonomous_optimization'][0]
    ax.bar(['Min', 'Max'], opt_temp, color='#e74c3c', edgecolor='black')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Optimization: Temperature Range')
    for i, v in enumerate(opt_temp):
        ax.text(i, v + 10, f'{v:.1f}°C', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Pressure range
    ax = axes[2, 1]
    opt_pres = sections['autonomous_optimization'][1]
    ax.bar(['Min', 'Max'], opt_pres, color='#3498db', edgecolor='black')
    ax.set_ylabel('Pressure (bar)')
    ax.set_title('Optimization: Pressure Range')
    for i, v in enumerate(opt_pres):
        ax.text(i, v + 2, f'{v:.1f} bar', ha='center', va='bottom')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: All optimization parameters
    ax = axes[2, 2]
    opt_params = sections['autonomous_optimization']
    param_labels = ['Temp', 'Pressure', 'Conc', 'Time', 'pH', 'Stirring']
    param_values = [p[0] if len(p) == 1 else (p[0] + p[1])/2 for p in opt_params]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax.bar(param_labels[:len(param_values)], param_values, color=colors, edgecolor='black')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Optimization: All Parameters (Central Values)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_dataset_summary(sections, output_dir):
    """
    Create comprehensive dataset summary JSON.
    """
    summary = {
        'dataset_name': 'M-AI-Synth Materials AI Dataset',
        'description': 'Dataset for rapid validation of three core AI application workflows in materials science',
        'sections': {}
    }
    
    # Property prediction section
    prop = sections['property_prediction']
    summary['sections']['property_prediction'] = {
        'n_arrays': len(prop),
        'features': {
            'length': len(prop[0]),
            'min': float(np.min(prop[0])),
            'max': float(np.max(prop[0])),
            'mean': float(np.mean(prop[0])),
            'std': float(np.std(prop[0]))
        },
        'targets': {
            'length': len(prop[1]),
            'min': float(np.min(prop[1])),
            'max': float(np.max(prop[1])),
            'mean': float(np.mean(prop[1])),
            'std': float(np.std(prop[1]))
        },
        'classes': {
            'length': len(prop[2]),
            'unique_values': [int(x) for x in np.unique(prop[2])],
            'class_counts': {int(k): int(v) for k, v in zip(*np.unique(prop[2], return_counts=True))}
        },
        'descriptors': {
            'length': len(prop[3]),
            'min': float(np.min(prop[3])),
            'max': float(np.max(prop[3])),
            'mean': float(np.mean(prop[3])),
            'std': float(np.std(prop[3]))
        }
    }
    
    # Structure generation section
    struct = sections['structure_generation']
    summary['sections']['structure_generation'] = {
        'n_arrays': len(struct),
        'lattice_a': {
            'length': len(struct[0]),
            'min': float(np.min(struct[0])),
            'max': float(np.max(struct[0])),
            'mean': float(np.mean(struct[0])),
            'std': float(np.std(struct[0]))
        },
        'lattice_b': {
            'length': len(struct[1]),
            'min': float(np.min(struct[1])),
            'max': float(np.max(struct[1])),
            'mean': float(np.mean(struct[1])),
            'std': float(np.std(struct[1]))
        },
        'correlation_ab': float(np.corrcoef(struct[0], struct[1])[0, 1])
    }
    
    # Autonomous optimization section
    opt = sections['autonomous_optimization']
    param_names = ['temperature', 'pressure', 'concentration', 'time', 'ph', 'stirring_rate']
    summary['sections']['autonomous_optimization'] = {
        'n_parameters': len(opt),
        'parameters': {}
    }
    for i, name in enumerate(param_names[:len(opt)]):
        values = opt[i]
        if len(values) == 1:
            summary['sections']['autonomous_optimization']['parameters'][name] = {
                'type': 'single_value',
                'value': float(values[0])
            }
        else:
            summary['sections']['autonomous_optimization']['parameters'][name] = {
                'type': 'range',
                'min': float(values[0]),
                'max': float(values[1]),
                'midpoint': float((values[0] + values[1]) / 2)
            }
    
    
    # Overall statistics
    summary['overall_statistics'] = {
        'total_data_points': sum(len(s) for s in sections.values()),
        'total_arrays': sum(len(sections[s]) for s in sections),
        'workflows_supported': ['property_prediction', 'structure_generation', 'autonomous_optimization']
    }
    
    with open(f'{output_dir}/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == '__main__':
    import os
    
    # Paths
    data_path = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_001_20260416_190409/data/M-AI-Synth__Materials_AI_Dataset_.txt'
    output_dir = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_001_20260416_190409/outputs'
    
    print("=" * 60)
    print("DATA OVERVIEW ANALYSIS")
    print("=" * 60)
    
    # Parse dataset
    print("\n[1] Parsing dataset...")
    sections = parse_full_dataset(data_path)
    for section, arrays in sections.items():
        print(f"    {section}: {len(arrays)} arrays")
    
    # Generate overview plots
    print("\n[2] Generating data overview plots...")
    generate_data_overview_plots(sections, output_dir)
    print(f"    Saved: {output_dir}/data_overview.png")
    
    # Create summary
    print("\n[3] Creating dataset summary...")
    summary = create_dataset_summary(sections, output_dir)
    print(f"    Saved: {output_dir}/dataset_summary.json")
    
    print("\n" + "=" * 60)
    print("DATA OVERVIEW COMPLETE")
    print("=" * 60)
