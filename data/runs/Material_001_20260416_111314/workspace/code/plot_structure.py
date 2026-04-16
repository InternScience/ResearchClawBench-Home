import numpy as np
import matplotlib.pyplot as plt

with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()

arr5 = eval(lines[7].strip())
arr6 = eval(lines[8].strip())

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(arr5)
axs[0].set_title('Array 5')

axs[1].plot(arr6)
axs[1].set_title('Array 6')

plt.tight_layout()
plt.savefig('report/images/structure_data.png')
