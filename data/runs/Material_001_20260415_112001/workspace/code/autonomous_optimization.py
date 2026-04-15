import json
import numpy as np
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# Load data
with open('outputs/dataset.json', 'r') as f:
    data = json.load(f)

# Mock objective: minimize distance to (350,20)
def objective(temp, time):
    target_temp, target_time = 350.0, 20.0
    return -np.sqrt((temp - target_temp)**2 + (time - target_time)**2)  # negative for max

# Bounds from data
pbounds = {'temp': (200, 500), 'time': (10, 30)}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
    verbose=0
)

# Run n_iter
n_iter = int(data['autonomous_optimization'][-1])
optimizer.maximize(init_points=2, n_iter=n_iter)

# Plot trajectory
history = np.array([[t['target']['temp'], t['target']['time'], t['target']['func']] for t in optimizer.space.params])
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(history[:,2], 'o-')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective (neg distance)')
ax.set_title('BO Trajectory')
plt.savefig('report/images/optimization_trajectory.png', dpi=300, bbox_inches='tight')
plt.close()

# Optimal
optimal = optimizer.max['target']
print(f'Optimal: temp={optimal[\"temp\"]:.1f}, time={optimal[\"time\"]:.1f}')

# Save
results = {'optimal_params': optimal, 'history': history.tolist()}
with open('outputs/optimization_results.json', 'w') as f:
    json.dump(results, f)
