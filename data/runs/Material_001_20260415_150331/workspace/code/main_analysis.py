"""
Main Analysis Script for M-AI-Synth Materials AI Dataset
Runs all three AI workflow analyses and generates comprehensive visualizations.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Ensure output directories exist
os.makedirs('../outputs', exist_ok=True)
os.makedirs('../report/images', exist_ok=True)

print("=" * 70)
print("M-AI-SYNTH: MATERIALS AI DATASET COMPREHENSIVE ANALYSIS")
print("=" * 70)

# Step 1: Parse the data
print("\n[Step 1] Parsing dataset...")
exec(open('data_parser.py').read())

# Step 2: Property Prediction Analysis
print("\n[Step 2] Running Property Prediction Analysis...")
exec(open('property_prediction.py').read())

# Step 3: Structure Generation Analysis  
print("\n[Step 3] Running Structure Generation Analysis...")
exec(open('structure_generation.py').read())

# Step 4: Autonomous Optimization Analysis
print("\n[Step 4] Running Autonomous Optimization Analysis...")
exec(open('autonomous_optimization.py').read())

print("\n" + "=" * 70)
print("All individual analyses completed successfully!")
print("=" * 70)
