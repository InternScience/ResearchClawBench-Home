import matplotlib.pyplot as plt
import numpy as np

# Let's plot the rotation matrix as a heatmap
matrix_data = []
with open('outputs/matrix.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('0') or line.startswith('1') or line.startswith('2'):
            parts = line.split()
            matrix_data.append([float(parts[2]), float(parts[3]), float(parts[4])])

matrix_data = np.array(matrix_data)

fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(matrix_data, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

for (i, j), z in np.ndenumerate(matrix_data):
    ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

plt.title('Rotation Matrix (Structure 1 -> Structure 2)')
plt.xticks([0, 1, 2], ['x', 'y', 'z'])
plt.yticks([0, 1, 2], ['X', 'Y', 'Z'])
plt.savefig('report/images/rotation_matrix.png')

