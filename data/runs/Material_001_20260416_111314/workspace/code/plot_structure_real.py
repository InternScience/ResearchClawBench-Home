import numpy as np
import matplotlib.pyplot as plt

with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()

arr5 = eval(lines[7].strip())
arr6 = eval(lines[8].strip())

# Plot a distribution of generated vs target structure parameters (e.g., lattice constants)
plt.figure(figsize=(6, 5))
plt.hist(arr5, bins=10, alpha=0.5, label='Generated Structure Parameter', color='blue')
plt.hist(arr6, bins=10, alpha=0.5, label='Target Structure Parameter', color='orange')
plt.xlabel('Structure Parameter Value')
plt.ylabel('Frequency')
plt.title('Structure Generation: Generated vs Target')
plt.legend()
plt.grid(True)
plt.savefig('report/images/structure_generation.png')

