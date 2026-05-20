#!/usr/bin/env python3
"""
Experimental Optimization Workflow
Optimize synthesis parameters using Bayesian optimization and grid search.
Uses data from File 3 of the M-AI-Synth dataset.
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Parse data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'M-AI-Synth__Materials_AI_Dataset_.txt')
with open(data_path, 'r') as f:
    content = f.read()

sections = content.split('# 文件')
file3_content = sections[3]

lines = [l.strip() for l in file3_content.split('\n') if l.strip().startswith('[')]
temp_range = np.array(json.loads(lines[0]))   # [200, 500]
time_range = np.array(json.loads(lines[1]))   # [10, 30]
opt_temp = json.loads(lines[2])[0]            # 350
opt_time = json.loads(lines[3])[0]            # 20
step_size = json.loads(lines[4])[0]           # 0.1
n_iter = int(json.loads(lines[5])[0])         # 10

print(f"Temperature range: {temp_range}")
print(f"Time range: {time_range}")
print(f"Optimal temperature: {opt_temp}")
print(f"Optimal time: {opt_time}")
print(f"Step size: {step_size}")
print(f"Iterations: {n_iter}")

# Define the objective function (simulated yield)
# The true optimum is at (350, 20)
def synthesis_yield(temp, time):
    """Simulated synthesis yield function with optimum at (350, 20)"""
    # Gaussian-like surface centered at optimum
    temp_term = -((temp - opt_temp) / 100) ** 2
    time_term = -((time - opt_time) / 10) ** 2
    interaction = -0.3 * ((temp - opt_temp) / 100) * ((time - opt_time) / 10)
    
    # Base yield with noise
    noise = np.random.normal(0, 0.02)
    yield_val = 0.9 * np.exp(temp_term + time_term + interaction) + 0.1 * np.random.random() + noise
    return np.clip(yield_val, 0, 1)

# 1. Grid Search
grid_temp = np.linspace(temp_range[0], temp_range[1], 20)
grid_time = np.linspace(time_range[0], time_range[1], 20)
grid_T, grid_t = np.meshgrid(grid_temp, grid_time)

np.random.seed(42)
grid_yields = np.zeros_like(grid_T)
for i in range(grid_T.shape[0]):
    for j in range(grid_T.shape[1]):
        grid_yields[i, j] = synthesis_yield(grid_T[i, j], grid_t[i, j])

best_idx = np.unravel_index(np.argmax(grid_yields), grid_yields.shape)
grid_best_temp = grid_T[best_idx]
grid_best_time = grid_t[best_idx]
grid_best_yield = grid_yields[best_idx]

print(f"\nGrid Search Best: T={grid_best_temp:.1f}, t={grid_best_time:.1f}, yield={grid_best_yield:.4f}")

# 2. Random Search
n_random = 50
np.random.seed(123)
random_temps = np.random.uniform(temp_range[0], temp_range[1], n_random)
random_times = np.random.uniform(time_range[0], time_range[1], n_random)
random_yields = np.array([synthesis_yield(t, tm) for t, tm in zip(random_temps, random_times)])

rand_best_idx = np.argmax(random_yields)
rand_best_temp = random_temps[rand_best_idx]
rand_best_time = random_times[rand_best_idx]
rand_best_yield = random_yields[rand_best_idx]

print(f"Random Search Best: T={rand_best_temp:.1f}, t={rand_best_time:.1f}, yield={rand_best_yield:.4f}")

# 3. Bayesian Optimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self, bounds, objective_func, xi=0.01):
        self.bounds = np.array(bounds)
        self.obj_func = objective_func
        self.xi = xi
        
        # GP kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.01)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        
        self.X_observed = []
        self.y_observed = []
        
    def _expected_improvement(self, X, best_y):
        """Compute expected improvement acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        
        with np.errstate(divide='ignore'):
            Z = (mu - best_y - self.xi) / sigma
            ei = (mu - best_y - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def suggest(self, n_candidates=1000):
        """Suggest next point to evaluate"""
        if len(self.X_observed) < 3:
            # Random initialization
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            return x
        
        # Generate candidates
        candidates = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], 
            size=(n_candidates, self.bounds.shape[0])
        )
        
        best_y = np.max(self.y_observed)
        ei = self._expected_improvement(candidates, best_y)
        best_idx = np.argmax(ei)
        
        return candidates[best_idx]
    
    def update(self, x, y):
        """Update GP with new observation"""
        self.X_observed.append(x)
        self.y_observed.append(y)
        
        if len(self.X_observed) > 2:
            X_arr = np.array(self.X_observed)
            y_arr = np.array(self.y_observed)
            self.gp.fit(X_arr, y_arr)

# Run Bayesian Optimization
np.random.seed(42)
bounds = [[temp_range[0], temp_range[1]], [time_range[0], time_range[1]]]
bo = BayesianOptimizer(bounds, lambda x: synthesis_yield(x[0], x[1]))

bo_history = []
for i in range(n_iter):
    x_next = bo.suggest()
    y_next = synthesis_yield(x_next[0], x_next[1])
    bo.update(x_next, y_next)
    bo_history.append({
        'iteration': i + 1,
        'temperature': float(x_next[0]),
        'time': float(x_next[1]),
        'yield': float(y_next)
    })
    print(f"BO Iter {i+1}: T={x_next[0]:.1f}, t={x_next[1]:.1f}, yield={y_next:.4f}")

bo_best_idx = np.argmax([h['yield'] for h in bo_history])
bo_best = bo_history[bo_best_idx]
print(f"\nBO Best: T={bo_best['temperature']:.1f}, t={bo_best['time']:.1f}, yield={bo_best['yield']:.4f}")

# 4. Gradient-based optimization (simulated)
# Start from initial guess, follow gradient
def gradient_step(temp, time, lr=0.1):
    """Simulated gradient ascent"""
    h = 1.0
    y_center = synthesis_yield(temp, time)
    y_t_plus = synthesis_yield(temp + h, time)
    y_t_minus = synthesis_yield(temp - h, time)
    y_tm_plus = synthesis_yield(temp, time + h)
    y_tm_minus = synthesis_yield(temp, time - h)
    
    grad_t = (y_t_plus - y_t_minus) / (2 * h)
    grad_tm = (y_tm_plus - y_tm_minus) / (2 * h)
    
    new_temp = temp + lr * grad_t * 100  # Scale to parameter space
    new_time = time + lr * grad_tm * 10
    new_temp = np.clip(new_temp, temp_range[0], temp_range[1])
    new_time = np.clip(new_time, time_range[0], time_range[1])
    
    return new_temp, new_time, synthesis_yield(new_temp, new_time)

# Run gradient ascent
grad_history = []
curr_temp = 300.0
curr_time = 15.0
np.random.seed(99)

for i in range(n_iter):
    curr_temp, curr_time, curr_yield = gradient_step(curr_temp, curr_time, lr=step_size)
    grad_history.append({
        'iteration': i + 1,
        'temperature': float(curr_temp),
        'time': float(curr_time),
        'yield': float(curr_yield)
    })

grad_best_idx = np.argmax([h['yield'] for h in grad_history])
grad_best = grad_history[grad_best_idx]
print(f"\nGradient Ascent Best: T={grad_best['temperature']:.1f}, t={grad_best['time']:.1f}, yield={grad_best['yield']:.4f}")

# 5. Compute full response surface
n_surface = 50
surf_T = np.linspace(temp_range[0], temp_range[1], n_surface)
surf_t = np.linspace(time_range[0], time_range[1], n_surface)
surf_TT, surf_tt = np.meshgrid(surf_T, surf_t)
surf_yields = np.zeros_like(surf_TT)
np.random.seed(0)
for i in range(n_surface):
    for j in range(n_surface):
        surf_yields[i, j] = synthesis_yield(surf_TT[i, j], surf_tt[i, j])

# Save results
results = {
    'optimization_setup': {
        'temp_range': temp_range.tolist(),
        'time_range': time_range.tolist(),
        'true_optimum': {'temperature': float(opt_temp), 'time': float(opt_time)},
        'n_iterations': n_iter
    },
    'grid_search': {
        'best_temperature': float(grid_best_temp),
        'best_time': float(grid_best_time),
        'best_yield': float(grid_best_yield),
        'temp_error': float(abs(grid_best_temp - opt_temp)),
        'time_error': float(abs(grid_best_time - opt_time))
    },
    'random_search': {
        'best_temperature': float(rand_best_temp),
        'best_time': float(rand_best_time),
        'best_yield': float(rand_best_yield),
        'temp_error': float(abs(rand_best_temp - opt_temp)),
        'time_error': float(abs(rand_best_time - opt_time))
    },
    'bayesian_optimization': {
        'history': bo_history,
        'best_temperature': float(bo_best['temperature']),
        'best_time': float(bo_best['time']),
        'best_yield': float(bo_best['yield']),
        'temp_error': float(abs(bo_best['temperature'] - opt_temp)),
        'time_error': float(abs(bo_best['time'] - opt_time))
    },
    'gradient_ascent': {
        'history': grad_history,
        'best_temperature': float(grad_best['temperature']),
        'best_time': float(grad_best['time']),
        'best_yield': float(grad_best['yield']),
        'temp_error': float(abs(grad_best['temperature'] - opt_temp)),
        'time_error': float(abs(grad_best['time'] - opt_time))
    },
    'response_surface': {
        'temperature': surf_T.tolist(),
        'time': surf_t.tolist(),
        'yields': surf_yields.tolist()
    }
}

output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'optimization_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Summary comparison
print("\n=== Optimization Comparison ===")
methods = ['grid_search', 'random_search', 'bayesian_optimization', 'gradient_ascent']
for m in methods:
    r = results[m]
    print(f"\n{m}:")
    print(f"  Best T={r['best_temperature']:.1f} (error={r['temp_error']:.1f})")
    print(f"  Best t={r['best_time']:.1f} (error={r['time_error']:.1f})")
    print(f"  Yield={r['best_yield']:.4f}")

print(f"\nResults saved to outputs/optimization_results.json")
