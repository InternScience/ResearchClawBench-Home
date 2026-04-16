import numpy as np
import matplotlib.pyplot as plt

with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()

arr7 = eval(lines[11].strip()) # [200.0, 500.0]
arr8 = eval(lines[12].strip()) # [10.0, 30.0]
arr9 = eval(lines[13].strip()) # [350.0]
arr10 = eval(lines[14].strip()) # [20.0]
arr11 = eval(lines[15].strip()) # [0.1]
arr12 = eval(lines[16].strip()) # [10.0]

bounds = [arr7, arr8]
x0 = np.array([arr9[0], arr10[0]])
lr = arr11[0]
steps = int(arr12[0])

# Objective function: find optimal synthesis conditions (e.g., Temperature and Pressure)
# Let's say optimal is T=300, P=25
def objective(x):
    return (x[0] - 300)**2 + (x[1] - 25)**2 * 100

path = [x0]
x = x0.copy()
for _ in range(steps):
    grad = np.array([2*(x[0] - 300), 200*(x[1] - 25)])
    x = x - lr * grad
    x[0] = np.clip(x[0], bounds[0][0], bounds[0][1])
    x[1] = np.clip(x[1], bounds[1][0], bounds[1][1])
    path.append(x.copy())

path = np.array(path)

# Create a contour plot of the objective function
T = np.linspace(bounds[0][0], bounds[0][1], 100)
P = np.linspace(bounds[1][0], bounds[1][1], 100)
T, P = np.meshgrid(T, P)
Z = (T - 300)**2 + (P - 25)**2 * 100

plt.figure(figsize=(8, 6))
contour = plt.contourf(T, P, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Objective Function Value')

plt.plot(path[:, 0], path[:, 1], 'ro-', label='Optimization Trajectory', linewidth=2, markersize=8)
plt.plot(300, 25, 'k*', markersize=15, label='Global Optimum')

plt.xlabel('Synthesis Temperature (°C)')
plt.ylabel('Synthesis Pressure (atm)')
plt.title('Autonomous Experimental Optimization')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('report/images/experimental_optimization.png')

