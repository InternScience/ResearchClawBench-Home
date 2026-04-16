import numpy as np
import matplotlib.pyplot as plt

with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()

arr1 = eval(lines[1].strip())
arr2 = eval(lines[2].strip())
arr3 = eval(lines[3].strip())
arr4 = eval(lines[4].strip())

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(arr1)
axs[0, 0].set_title('Array 1')

axs[0, 1].plot(arr2)
axs[0, 1].set_title('Array 2')

axs[1, 0].plot(arr3)
axs[1, 0].set_title('Array 3')

axs[1, 1].plot(arr4)
axs[1, 1].set_title('Array 4')

plt.tight_layout()
plt.savefig('report/images/property_data.png')
