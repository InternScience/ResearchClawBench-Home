#!/usr/bin/env python3
"""
M-AI-Synth Materials AI Analysis Pipeline
==========================================
Three core AI workflows for materials science:
1. Property Prediction using Crystal Graph features
2. Structure Generation using Variational Autoencoder
3. Experimental Optimization using Bayesian Optimization

Author: AI Research Agent
Date: 2026-04-16
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set working directory
WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_001_20260416_221126"
DATA_FILE = os.path.join(WORKSPACE, "data", "M-AI-Synth__Materials_AI_Dataset_.txt")
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report", "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# STEP 1: Parse the dataset
# ============================================================
print("=" * 60)
print("STEP 1: Parsing M-AI-Synth Dataset")
print("=" * 60)

with open(DATA_FILE, 'r') as f:
    lines = f.readlines()

# Parse property prediction data
# Line 0: comment
# Line 1: atomic numbers (node features)
# Line 2: feature values (node positions/energies)
# Line 3: edge indices
# Line 4: edge attributes

atomic_numbers = json.loads(lines[1].strip())
feature_values = json.loads(lines[2].strip())
edge_indices = json.loads(lines[3].strip())
edge_attributes = json.loads(lines[4].strip())

# Parse structure generation data
# Line 6: lattice parameter a
# Line 7: lattice parameter b
lattice_a = json.loads(lines[6].strip())
lattice_b = json.loads(lines[7].strip())

# Parse optimization data
# Lines 9-14: bounds and initial conditions
temp_bounds = json.loads(lines[9].strip())
time_bounds = json.loads(lines[10].strip())
initial_temp = json.loads(lines[11].strip())
initial_time = json.loads(lines[12].strip())
learning_rate = json.loads(lines[13].strip())
n_iterations = json.loads(lines[14].strip())

print(f"Property Prediction Data:")
print(f"  Atomic numbers: {len(atomic_numbers)} nodes (element Z={atomic_numbers[0]})")
print(f"  Feature values: {len(feature_values)} values")
print(f"  Edge indices: {len(edge_indices)} values ({len(edge_indices)//2} edges)")
print(f"  Edge attributes: {len(edge_attributes)} values")
print(f"\nStructure Generation Data:")
print(f"  Lattice a: {len(lattice_a)} samples")
print(f"  Lattice b: {len(lattice_b)} samples")
print(f"\nOptimization Data:")
print(f"  Temperature bounds: {temp_bounds}")
print(f"  Time bounds: {time_bounds}")
print(f"  Initial temp: {initial_temp}, Initial time: {initial_time}")
print(f"  Learning rate: {learning_rate}, Iterations: {n_iterations}")

# Save data overview
data_overview = {
    "property_prediction": {
        "n_nodes": len(atomic_numbers),
        "element": "Boron (Z=5)",
        "n_features": len(feature_values),
        "n_edges": len(edge_indices) // 2,
        "n_edge_attributes": len(edge_attributes),
        "feature_range": [min(feature_values), max(feature_values)],
        "edge_attr_range": [min(edge_attributes), max(edge_attributes)]
    },
    "structure_generation": {
        "n_samples": len(lattice_a),
        "lattice_a_range": [min(lattice_a), max(lattice_a)],
        "lattice_b_range": [min(lattice_b), max(lattice_b)],
        "lattice_a_mean": np.mean(lattice_a),
        "lattice_b_mean": np.mean(lattice_b)
    },
    "optimization": {
        "temperature_bounds": temp_bounds,
        "time_bounds": time_bounds,
        "initial_temperature": initial_temp[0],
        "initial_time": initial_time[0],
        "learning_rate": learning_rate[0],
        "n_iterations": int(n_iterations[0])
    }
}

with open(os.path.join(OUTPUT_DIR, "data_overview.json"), 'w') as f:
    json.dump(data_overview, f, indent=2)

print("\nData overview saved to outputs/data_overview.json")
print("STEP 1 COMPLETE")
