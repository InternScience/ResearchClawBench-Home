import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()
    
# Extract lines 2, 3, 4, 5
feature_1 = eval(lines[1].strip())
feature_2 = eval(lines[2].strip())
target_class = eval(lines[3].strip())
target_prop = eval(lines[4].strip())

print(f"feature_1 shape: {len(feature_1)}")
print(f"feature_2 shape: {len(feature_2)}")
print(f"target_class shape: {len(target_class)}")
print(f"target_prop shape: {len(target_prop)}")
