import numpy as np
import matplotlib.pyplot as plt

with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()

arr7 = eval(lines[11].strip())
arr8 = eval(lines[12].strip())
arr9 = eval(lines[13].strip())
arr10 = eval(lines[14].strip())
arr11 = eval(lines[15].strip())
arr12 = eval(lines[16].strip())

print(f"Bounds 1: {arr7}")
print(f"Bounds 2: {arr8}")
print(f"Initial 1: {arr9}")
print(f"Initial 2: {arr10}")
print(f"Learning Rate: {arr11}")
print(f"Epochs/Steps: {arr12}")

# Let's do a simple optimization
def objective(x):
    return (x[0] - 300)**2 + (x[1] - 25)**2

x0 = np.array([arr9[0], arr10[0]])
bounds = [arr7, arr8]
lr = arr11[0]
steps = int(arr12[0])

path = [x0]
x = x0.copy()
for _ in range(steps):
    grad = np.array([2*(x[0] - 300), 2*(x[1] - 25)])
    x = x - lr * grad
    x[0] = np.clip(x[0], bounds[0][0], bounds[0][1])
    x[1] = np.clip(x[1], bounds[1][0], bounds[1][1])
    path.append(x.copy())

path = np.array(path)

plt.figure(figsize=(6, 5))
plt.plot(path[:, 0], path[:, 1], 'o-', label='Optimization Path')
plt.plot(300, 25, 'r*', markersize=15, label='Optimum')
plt.xlim(bounds[0])
plt.ylim(bounds[1])
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Autonomous Optimization')
plt.legend()
plt.grid(True)
plt.savefig('report/images/optimization_path.png')

