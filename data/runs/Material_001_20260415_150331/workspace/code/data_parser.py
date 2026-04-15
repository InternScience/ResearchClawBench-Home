"""
Data parser for M-AI-Synth Materials AI Dataset
Parses the multimodal materials data for three AI workflows:
1. Property Prediction
2. Structure Generation
3. Autonomous Optimization
"""

import json
import numpy as np


def parse_dataset(filepath):
    """
    Parse the materials dataset file containing three workflow data sections.
    
    Returns:
        dict: Contains 'property_prediction', 'structure_generation', 'autonomous_optimization'
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse property prediction data
    # Lines after "property_prediction.py"
    lines = content.split('\n')
    
    data = {
        'property_prediction': {},
        'structure_generation': {},
        'autonomous_optimization': {}
    }
    
    # Find sections
    section_idx = {}
    for i, line in enumerate(lines):
        if 'property_prediction' in line:
            section_idx['property'] = i
        elif 'structure_generation' in line:
            section_idx['structure'] = i
        elif 'autonomous_optimization' in line:
            section_idx['optimization'] = i
    
    # Parse property prediction section
    prop_lines = []
    for i in range(section_idx['property'] + 1, section_idx['structure']):
        line = lines[i].strip()
        if line and line.startswith('[') and line.endswith(']'):
            try:
                prop_lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    
    # Property prediction data structure:
    # Line 0: Atomic numbers (lattice size = 5 for all)
    # Line 1: Feature values (energy or position-related)
    # Line 2: Edge indices (atom connections)
    # Line 3: Target properties (various material properties)
    if len(prop_lines) < 4:
        # Fallback: create synthetic data based on what's available
        pass
    
    data['property_prediction']['atomic_numbers'] = prop_lines[0] if len(prop_lines) > 0 else []
    data['property_prediction']['features'] = prop_lines[1] if len(prop_lines) > 1 else []
    data['property_prediction']['edge_indices'] = prop_lines[2] if len(prop_lines) > 2 else []
    data['property_prediction']['targets'] = prop_lines[3] if len(prop_lines) > 3 else []
    
    # Parse structure generation section
    struct_lines = []
    for i in range(section_idx['structure'] + 1, section_idx['optimization']):
        line = lines[i].strip()
        if line and line.startswith('[') and line.endswith(']'):
            try:
                struct_lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    
    # Structure generation data: generated lattice constants and target lattice constants
    data['structure_generation']['generated_lattice'] = struct_lines[0] if len(struct_lines) > 0 else []
    data['structure_generation']['target_lattice'] = struct_lines[1] if len(struct_lines) > 1 else []
    
    # Parse autonomous optimization section
    opt_lines = []
    for i in range(section_idx['optimization'] + 1, len(lines)):
        line = lines[i].strip()
        if line and line.startswith('[') and line.endswith(']'):
            try:
                opt_lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    
    # Optimization data: parameter bounds and optimal values
    # [Temperature range], [Time range], [Optimal Temp], [Optimal Time], [Optimal Yield], [Confidence]
    data['autonomous_optimization']['temperature_range'] = opt_lines[0] if len(opt_lines) > 0 else []
    data['autonomous_optimization']['time_range'] = opt_lines[1] if len(opt_lines) > 1 else []
    data['autonomous_optimization']['optimal_temperature'] = opt_lines[2] if len(opt_lines) > 2 else []
    data['autonomous_optimization']['optimal_time'] = opt_lines[3] if len(opt_lines) > 3 else []
    data['autonomous_optimization']['optimal_yield'] = opt_lines[4] if len(opt_lines) > 4 else []
    data['autonomous_optimization']['confidence'] = opt_lines[5] if len(opt_lines) > 5 else []
    
    return data


def print_data_summary(data):
    """Print summary statistics for each workflow."""
    print("=" * 60)
    print("M-AI-Synth Dataset Summary")
    print("=" * 60)
    
    # Property Prediction Summary
    print("\n1. PROPERTY PREDICTION WORKFLOW")
    print("-" * 40)
    pp = data['property_prediction']
    print(f"  Number of samples: {len(pp.get('atomic_numbers', []))}")
    print(f"  Feature vector size: {len(pp.get('features', []))}")
    print(f"  Edge connections: {len(pp.get('edge_indices', []))}")
    print(f"  Target properties: {len(pp.get('targets', []))}")
    if pp.get('targets'):
        targets = np.array(pp['targets'])
        print(f"  Target range: [{np.min(targets):.4f}, {np.max(targets):.4f}]")
        print(f"  Target mean ± std: {np.mean(targets):.4f} ± {np.std(targets):.4f}")
    
    # Structure Generation Summary
    print("\n2. STRUCTURE GENERATION WORKFLOW")
    print("-" * 40)
    sg = data['structure_generation']
    print(f"  Generated structures: {len(sg.get('generated_lattice', []))}")
    print(f"  Target structures: {len(sg.get('target_lattice', []))}")
    if sg.get('generated_lattice') and sg.get('target_lattice'):
        gen = np.array(sg['generated_lattice'])
        tgt = np.array(sg['target_lattice'])
        print(f"  Generated lattice mean: {np.mean(gen):.4f} ± {np.std(gen):.4f}")
        print(f"  Target lattice mean: {np.mean(tgt):.4f} ± {np.std(tgt):.4f}")
        print(f"  Mean absolute error: {np.mean(np.abs(gen - tgt)):.4f}")
    
    # Autonomous Optimization Summary
    print("\n3. AUTONOMOUS OPTIMIZATION WORKFLOW")
    print("-" * 40)
    ao = data['autonomous_optimization']
    print(f"  Temperature range: {ao.get('temperature_range', [])}")
    print(f"  Time range: {ao.get('time_range', [])}")
    print(f"  Optimal temperature: {ao.get('optimal_temperature', [])}")
    print(f"  Optimal time: {ao.get('optimal_time', [])}")
    print(f"  Optimal yield: {ao.get('optimal_yield', [])}")
    print(f"  Model confidence: {ao.get('confidence', [])}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    import sys
    filepath = '../data/M-AI-Synth__Materials_AI_Dataset_.txt'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    data = parse_dataset(filepath)
    print_data_summary(data)
    
    # Save parsed data as JSON for other scripts
    with open('../outputs/parsed_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("\nParsed data saved to outputs/parsed_data.json")
