"""
Workflow 3: Autonomous Experimental Optimization
- Implements Bayesian optimization for synthesis parameter optimization
- Uses Gaussian Process surrogate model
- Tracks optimization trajectory and convergence
- Visualizes parameter landscape and acquisition function
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = 'report/images'

# Load optimization parameters from dataset
with open('outputs/parsed_dataset.json', 'r') as f:
    parsed = json.load(f)

opt_params = parsed['optimization']
temp_range = opt_params['temperature_range']  # [200, 500]
time_range = opt_params['time_range']  # [10, 30]
target_temp = opt_params['target_temp']  # 350
target_time = opt_params['target_time']  # 20
tolerance = opt_params['tolerance']  # 0.1
max_iterations = opt_params['max_iterations']  # 10

print(f"Optimization Parameters:")
print(f"  Temperature range: {temp_range}")
print(f"  Time range: {time_range}")
print(f"  Target temperature: {target_temp}")
print(f"  Target time: {target_time}")
print(f"  Tolerance: {tolerance}")
print(f"  Max iterations: {max_iterations}")


# ============================================================
# Define objective function (simulated synthesis quality)
# ============================================================
def synthesis_objective(temp, time_h, noise_std=0.05):
    """
    Simulated synthesis quality function.
    Models the quality of a material synthesis as a function of temperature and time.
    Higher values = better quality. Target: maximize quality near target conditions.
    
    The function has a global optimum near the target conditions with
    additional local optima to make optimization challenging.
    """
    # Main peak near target conditions
    target_quality = np.exp(-0.5 * ((temp - target_temp) / 30) ** 2) * \
                     np.exp(-0.5 * ((time_h - target_time) / 3) ** 2)
    
    # Secondary peak (local optimum - wrong conditions)
    secondary = 0.4 * np.exp(-0.5 * ((temp - 420) / 40) ** 2) * \
                np.exp(-0.5 * ((time_h - 25) / 4) ** 2)
    
    # Add some asymmetry
    asymmetry = 0.1 * np.sin(temp / 50) * np.cos(time_h / 5)
    
    quality = target_quality + secondary + asymmetry
    
    if noise_std > 0:
        quality += np.random.normal(0, noise_std)
    
    return quality


def neg_objective(x, noise_std=0.05):
    """Negative objective for minimization."""
    return -synthesis_objective(x[0], x[1], noise_std)


# ============================================================
# Bayesian Optimization Implementation
# ============================================================
class BayesianOptimizer:
    """Bayesian optimization with Expected Improvement acquisition."""
    
    def __init__(self, bounds, objective, n_initial=5):
        self.bounds = np.array(bounds)
        self.objective = objective
        self.n_initial = n_initial
        self.X_observed = []
        self.y_observed = []
        self.gp = None
        self.rng = np.random.RandomState(42)
        
    def initialize(self):
        """Generate initial observations using Latin Hypercube Sampling."""
        n_dims = self.bounds.shape[0]
        # Simple random sampling
        for _ in range(self.n_initial):
            x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = self.objective(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
    
    def fit_gp(self):
        """Fit Gaussian Process to observed data."""
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        kernel = ConstantKernel(1.0) * Matern(length_scale=[30, 3], nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10,
            alpha=0.01, random_state=42
        )
        self.gp.fit(X, y)
    
    def expected_improvement(self, x):
        """Compute Expected Improvement at point x."""
        x = x.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if sigma == 0:
            return 0
        
        y_best = np.max(self.y_observed)
        xi = 0.01  # Exploration-exploitation trade-off
        
        z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
    
    def propose_next(self):
        """Propose next experiment location."""
        # Multi-start optimization of acquisition function
        best_ei = -np.inf
        best_x = None
        
        for _ in range(100):
            x0 = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            try:
                result = minimize(
                    lambda x: -self.expected_improvement(x),
                    x0, bounds=self.bounds, method='L-BFGS-B'
                )
                if -result.fun > best_ei:
                    best_ei = -result.fun
                    best_x = result.x
            except:
                continue
        
        if best_x is None:
            best_x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
        
        return best_x
    
    def step(self):
        """Perform one optimization step."""
        self.fit_gp()
        x_next = self.propose_next()
        y_next = self.objective(x_next)
        self.X_observed.append(x_next)
        self.y_observed.append(y_next)
        return x_next, y_next
    
    def optimize(self, n_iterations):
        """Run full optimization."""
        self.initialize()
        trajectory = {
            'X': [list(x) for x in self.X_observed],
            'y': [float(y) for y in self.y_observed]
        }
        
        for i in range(n_iterations):
            x_next, y_next = self.step()
            trajectory['X'].append(list(x_next))
            trajectory['y'].append(float(y_next))
            
            best_y = max(self.y_observed)
            print(f"  Iteration {i+1}/{n_iterations}: "
                  f"T={x_next[0]:.1f}°C, t={x_next[1]:.1f}h, "
                  f"Quality={y_next:.4f}, Best={best_y:.4f}")
        
        return trajectory


# ============================================================
# Run Bayesian Optimization
# ============================================================
print("\n=== Bayesian Optimization ===")
bounds = np.array([
    [temp_range[0], temp_range[1]],  # Temperature
    [time_range[0], time_range[1]]   # Time
])

optimizer = BayesianOptimizer(
    bounds=bounds,
    objective=lambda x: synthesis_objective(x[0], x[1], noise_std=0.02),
    n_initial=5
)

trajectory = optimizer.optimize(n_iterations=int(max_iterations))

# Find best result
best_idx = np.argmax(trajectory['y'])
best_x = trajectory['X'][best_idx]
best_y = trajectory['y'][best_idx]

print(f"\nBest synthesis conditions:")
print(f"  Temperature: {best_x[0]:.1f}°C (target: {target_temp}°C)")
print(f"  Time: {best_x[1]:.1f}h (target: {target_time}h)")
print(f"  Quality: {best_y:.4f}")

# Check if within tolerance
temp_error = abs(best_x[0] - target_temp) / target_temp
time_error = abs(best_x[1] - target_time) / target_time
converged = temp_error < tolerance and time_error < tolerance
print(f"  Temperature error: {temp_error:.4f} (tolerance: {tolerance})")
print(f"  Time error: {time_error:.4f} (tolerance: {tolerance})")
print(f"  Converged: {converged}")

# Save optimization results
opt_results = {
    'best_temperature': float(best_x[0]),
    'best_time': float(best_x[1]),
    'best_quality': float(best_y),
    'target_temperature': float(target_temp),
    'target_time': float(target_time),
    'temperature_error': float(temp_error),
    'time_error': float(time_error),
    'converged': bool(converged),
    'n_iterations': max_iterations,
    'trajectory': trajectory
}

with open('outputs/optimization_results.json', 'w') as f:
    json.dump(opt_results, f, indent=2)

# ============================================================
# Run comparison: Random Search baseline
# ============================================================
print("\n=== Random Search Baseline ===")
rng = np.random.RandomState(42)
random_trajectory = {'X': [], 'y': []}
random_best_y = -np.inf

for i in range(int(max_iterations) + 5):
    x_rand = rng.uniform(bounds[:, 0], bounds[:, 1])
    y_rand = synthesis_objective(x_rand[0], x_rand[1], noise_std=0.02)
    random_trajectory['X'].append(list(x_rand))
    random_trajectory['y'].append(float(y_rand))
    if y_rand > random_best_y:
        random_best_y = y_rand

print(f"Random search best quality: {random_best_y:.4f}")

# ============================================================
# Run comparison: Grid Search baseline
# ============================================================
print("\n=== Grid Search Baseline ===")
temp_grid = np.linspace(temp_range[0], temp_range[1], 15)
time_grid = np.linspace(time_range[0], time_range[1], 15)
grid_best_y = -np.inf
grid_best_x = None

for t in temp_grid:
    for h in time_grid:
        y_grid = synthesis_objective(t, h, noise_std=0.0)
        if y_grid > grid_best_y:
            grid_best_y = y_grid
            grid_best_x = [t, h]

print(f"Grid search best: T={grid_best_x[0]:.1f}°C, t={grid_best_x[1]:.1f}h, Quality={grid_best_y:.4f}")

# Save comparison
comparison = {
    'bayesian_opt': {'best_quality': float(best_y), 'best_temp': float(best_x[0]), 'best_time': float(best_x[1])},
    'random_search': {'best_quality': float(random_best_y)},
    'grid_search': {'best_quality': float(grid_best_y), 'best_temp': float(grid_best_x[0]), 'best_time': float(grid_best_x[1])}
}
with open('outputs/optimization_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

# ============================================================
# FIGURE 11: Optimization trajectory
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Bayesian Optimization Trajectory for Synthesis Conditions', fontsize=14, fontweight='bold')

# Plot 1: Quality over iterations
ax = axes[0]
bo_y = trajectory['y']
rand_y = random_trajectory['y']
bo_running_max = np.maximum.accumulate(bo_y)
rand_running_max = np.maximum.accumulate(rand_y)

ax.plot(range(1, len(bo_y)+1), bo_running_max, 'o-', color='steelblue', label='Bayesian Opt.', linewidth=2)
ax.plot(range(1, len(rand_y)+1), rand_running_max, 's--', color='coral', label='Random Search', linewidth=2)
ax.axhline(y=grid_best_y, color='green', linestyle=':', label='Grid Search Best', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Quality Found')
ax.set_title('Convergence Comparison')
ax.legend()

# Plot 2: Temperature trajectory
ax = axes[1]
bo_temps = [x[0] for x in trajectory['X']]
ax.plot(range(1, len(bo_temps)+1), bo_temps, 'o-', color='steelblue', linewidth=2)
ax.axhline(y=target_temp, color='red', linestyle='--', label=f'Target: {target_temp}°C', linewidth=2)
ax.fill_between(range(1, len(bo_temps)+1), temp_range[0], temp_range[1], alpha=0.1, color='gray')
ax.set_xlabel('Iteration')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature Exploration')
ax.legend()

# Plot 3: Time trajectory
ax = axes[2]
bo_times = [x[1] for x in trajectory['X']]
ax.plot(range(1, len(bo_times)+1), bo_times, 'o-', color='steelblue', linewidth=2)
ax.axhline(y=target_time, color='red', linestyle='--', label=f'Target: {time_range[0]}h', linewidth=2)
ax.fill_between(range(1, len(bo_times)+1), time_range[0], time_range[1], alpha=0.1, color='gray')
ax.set_xlabel('Iteration')
ax.set_ylabel('Time (hours)')
ax.set_title('Time Exploration')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig11_optimization_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_optimization_trajectory.png")

# ============================================================
# FIGURE 12: Parameter landscape with observations
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Synthesis Parameter Landscape', fontsize=14, fontweight='bold')

# Create landscape
temp_grid_fine = np.linspace(temp_range[0], temp_range[1], 100)
time_grid_fine = np.linspace(time_range[0], time_range[1], 100)
T, H = np.meshgrid(temp_grid_fine, time_grid_fine)
Z = np.zeros_like(T)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i, j] = synthesis_objective(T[i, j], H[i, j], noise_std=0)

# Plot 1: True landscape
ax = axes[0]
contour = ax.contourf(T, H, Z, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax, label='Synthesis Quality')
ax.scatter([x[0] for x in trajectory['X'][:5]], [x[1] for x in trajectory['X'][:5]], 
           c='white', marker='o', s=80, label='Initial', edgecolors='black', zorder=5)
ax.scatter([x[0] for x in trajectory['X'][5:]], [x[1] for x in trajectory['X'][5:]], 
           c='red', marker='^', s=80, label='BO Suggested', edgecolors='black', zorder=5)
ax.scatter([best_x[0]], [best_x[1]], c='yellow', marker='*', s=200, label='Best Found', 
           edgecolors='black', zorder=6)
ax.scatter([target_temp], [target_time], c='magenta', marker='P', s=150, label='Target', 
           edgecolors='black', zorder=6)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Time (hours)')
ax.set_title('True Objective Landscape')
ax.legend(fontsize=8, loc='upper left')

# Plot 2: GP surrogate model
ax = axes[1]
X_obs = np.array(trajectory['X'])
y_obs = np.array(trajectory['y'])

kernel = ConstantKernel(1.0) * Matern(length_scale=[30, 3], nu=2.5)
gp_final = GaussianProcessRegressor(kernel=kernel, alpha=0.01, random_state=42)
gp_final.fit(X_obs, y_obs)

Z_pred = np.zeros_like(T)
Z_std = np.zeros_like(T)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        x_test = np.array([[T[i, j], H[i, j]]])
        mu, sigma = gp_final.predict(x_test, return_std=True)
        Z_pred[i, j] = mu[0]
        Z_std[i, j] = sigma[0]

contour = ax.contourf(T, H, Z_pred, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax, label='Predicted Quality')
ax.scatter(X_obs[:, 0], X_obs[:, 1], c='white', marker='o', s=60, 
           edgecolors='black', zorder=5)
ax.scatter([best_x[0]], [best_x[1]], c='yellow', marker='*', s=200, label='Best Found', 
           edgecolors='black', zorder=6)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Time (hours)')
ax.set_title('GP Surrogate Model')
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig12_parameter_landscape.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig12_parameter_landscape.png")

# ============================================================
# FIGURE 13: GP uncertainty map
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Gaussian Process Uncertainty and Acquisition Function', fontsize=14, fontweight='bold')

# Uncertainty map
ax = axes[0]
contour = ax.contourf(T, H, Z_std, levels=20, cmap='hot_r')
plt.colorbar(contour, ax=ax, label='Predictive Uncertainty (σ)')
ax.scatter(X_obs[:, 0], X_obs[:, 1], c='cyan', marker='o', s=60, 
           edgecolors='black', zorder=5)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Time (hours)')
ax.set_title('GP Predictive Uncertainty')

# Expected Improvement map
ax = axes[1]
EI_map = np.zeros_like(T)
y_best = max(trajectory['y'])
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        mu = Z_pred[i, j]
        sigma = Z_std[i, j]
        if sigma > 0:
            xi = 0.01
            z_val = (mu - y_best - xi) / sigma
            EI_map[i, j] = (mu - y_best - xi) * norm.cdf(z_val) + sigma * norm.pdf(z_val)

contour = ax.contourf(T, H, EI_map, levels=20, cmap='YlOrRd')
plt.colorbar(contour, ax=ax, label='Expected Improvement')
ax.scatter(X_obs[:, 0], X_obs[:, 1], c='blue', marker='o', s=60, 
           edgecolors='white', zorder=5)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Time (hours)')
ax.set_title('Expected Improvement Acquisition Function')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig13_gp_uncertainty_ei.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig13_gp_uncertainty_ei.png")

# ============================================================
# FIGURE 14: Optimization efficiency comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Optimization Strategy Comparison', fontsize=14, fontweight='bold')

# Quality improvement over iterations
ax = axes[0]
n_evals = list(range(1, len(bo_y) + 1))
ax.plot(n_evals, bo_running_max, 'o-', color='steelblue', label='Bayesian Optimization', linewidth=2, markersize=8)
ax.plot(range(1, len(rand_y)+1), rand_running_max, 's--', color='coral', label='Random Search', linewidth=2, markersize=8)
ax.axhline(y=grid_best_y, color='green', linestyle=':', label=f'Grid Search Best ({grid_best_y:.3f})', linewidth=2)
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Best Quality Found')
ax.set_title('Sample Efficiency')
ax.legend()

# Error to target over iterations
ax = axes[1]
bo_errors = []
running_best_x = None
running_best_y = -np.inf
for i, (x, y) in enumerate(zip(trajectory['X'], trajectory['y'])):
    if y > running_best_y:
        running_best_y = y
        running_best_x = x
    error = np.sqrt((running_best_x[0] - target_temp)**2 + (running_best_x[1] - target_time)**2)
    bo_errors.append(error)

ax.plot(range(1, len(bo_errors)+1), bo_errors, 'o-', color='steelblue', linewidth=2, markersize=8)
ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Distance to Target Conditions')
ax.set_title('Convergence to Target Conditions')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig14_optimization_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig14_optimization_efficiency.png")

print("\nAutonomous optimization workflow complete!")
