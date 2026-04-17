"""Part 4: Experimental Optimization using Bayesian Optimization"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Material_001_20260416_221126"
OUTPUT_DIR = os.path.join(WORKSPACE, "outputs")
IMAGE_DIR = os.path.join(WORKSPACE, "report", "images")

# Optimization parameters from dataset
temp_bounds = [200.0, 500.0]
time_bounds = [10.0, 30.0]
initial_temp = 350.0
initial_time = 20.0
lr = 0.1
n_iter = int(10.0)

print("=" * 60)
print("Bayesian Optimization for Synthesis Parameters")
print("=" * 60)
print(f"Temperature bounds: {temp_bounds} K")
print(f"Time bounds: {time_bounds} min")
print(f"Initial: T={initial_temp} K, t={initial_time} min")

# Define objective function (simulated material property)
# This represents a realistic synthesis optimization landscape
def objective(temp, time):
    """Simulated material quality as function of synthesis parameters.
    Models a realistic synthesis landscape with:
    - Optimal temperature around 380K
    - Optimal time around 22 min
    - Non-trivial interaction effects
    """
    # Normalize
    t_norm = (temp - 350) / 100
    time_norm = (time - 20) / 10
    
    # Multi-modal landscape
    quality = (5.0 
               - 2.0 * (t_norm - 0.3)**2 
               - 1.5 * (time_norm - 0.2)**2
               + 0.5 * t_norm * time_norm
               - 0.3 * np.sin(3 * t_norm) * np.cos(2 * time_norm)
               + 0.1 * np.random.randn())  # noise
    return quality

# ============================================================
# Gaussian Process Regression (from scratch)
# ============================================================
class GaussianProcess:
    """Simple GP with RBF kernel for Bayesian Optimization."""
    
    def __init__(self, length_scale=1.0, noise=0.1):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
    
    def rbf_kernel(self, X1, X2):
        """RBF (squared exponential) kernel."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return np.exp(-0.5 * sqdist / self.length_scale**2)
    
    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        K = self.rbf_kernel(X, X) + self.noise**2 * np.eye(len(X))
        self.K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
    
    def predict(self, X_test):
        K_star = self.rbf_kernel(X_test, self.X_train)
        K_star_star = self.rbf_kernel(X_test, X_test)
        
        mu = K_star @ self.K_inv @ self.y_train
        cov = K_star_star - K_star @ self.K_inv @ K_star.T
        std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        return mu, std

def expected_improvement(X, gp, y_best, xi=0.01):
    """Expected Improvement acquisition function."""
    mu, std = gp.predict(X)
    with np.errstate(divide='warn'):
        imp = mu - y_best - xi
        Z = imp / (std + 1e-10)
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei

# ============================================================
# Run Bayesian Optimization
# ============================================================
np.random.seed(42)

# Normalize bounds to [0, 1]
def normalize(temp, time):
    return np.array([(temp - temp_bounds[0]) / (temp_bounds[1] - temp_bounds[0]),
                     (time - time_bounds[0]) / (time_bounds[1] - time_bounds[0])])

def denormalize(x):
    temp = x[0] * (temp_bounds[1] - temp_bounds[0]) + temp_bounds[0]
    time = x[1] * (time_bounds[1] - time_bounds[0]) + time_bounds[0]
    return temp, time

# Initial samples (Latin Hypercube-like)
n_initial = 5
X_init = np.random.rand(n_initial, 2)
y_init = np.array([objective(*denormalize(x)) for x in X_init])

# BO loop
X_observed = X_init.copy()
y_observed = y_init.copy()
n_bo_iterations = 30

best_values = [np.max(y_observed)]
best_params = [denormalize(X_observed[np.argmax(y_observed)])]

print(f"\nInitial best: quality={best_values[0]:.4f}")

gp = GaussianProcess(length_scale=0.3, noise=0.1)

for i in range(n_bo_iterations):
    # Fit GP
    gp.fit(X_observed, y_observed)
    
    # Find next point by maximizing EI
    best_ei = -1
    best_x = None
    
    # Random search for acquisition function maximum
    candidates = np.random.rand(1000, 2)
    ei_values = expected_improvement(candidates, gp, np.max(y_observed))
    best_idx = np.argmax(ei_values)
    next_x = candidates[best_idx]
    
    # Evaluate
    temp, time = denormalize(next_x)
    y_new = objective(temp, time)
    
    # Update
    X_observed = np.vstack([X_observed, next_x])
    y_observed = np.append(y_observed, y_new)
    
    current_best = np.max(y_observed)
    best_values.append(current_best)
    best_params.append(denormalize(X_observed[np.argmax(y_observed)]))
    
    if (i + 1) % 5 == 0:
        print(f"Iteration {i+1}/{n_bo_iterations}: best={current_best:.4f}, "
              f"T={best_params[-1][0]:.1f}K, t={best_params[-1][1]:.1f}min")

# Also run random search for comparison
np.random.seed(42)
X_random = np.random.rand(n_initial + n_bo_iterations, 2)
y_random = np.array([objective(*denormalize(x)) for x in X_random])
random_best = [np.max(y_random[:i+1]) for i in range(len(y_random))]

# Also run grid search for comparison
n_grid = 6
temp_grid = np.linspace(0, 1, n_grid)
time_grid = np.linspace(0, 1, n_grid)
grid_results = np.zeros((n_grid, n_grid))
for i, t in enumerate(temp_grid):
    for j, tm in enumerate(time_grid):
        temp, time = denormalize(np.array([t, tm]))
        grid_results[i, j] = objective(temp, time)

final_best_idx = np.argmax(y_observed)
final_temp, final_time = denormalize(X_observed[final_best_idx])
print(f"\n{'='*60}")
print(f"Optimization Complete!")
print(f"Best quality: {y_observed[final_best_idx]:.4f}")
print(f"Optimal T: {final_temp:.1f} K")
print(f"Optimal time: {final_time:.1f} min")
print(f"{'='*60}")

# ============================================================
# FIGURE 8: Bayesian Optimization Results
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Bayesian Optimization for Synthesis Parameters', fontsize=14, fontweight='bold')

# 8a: Convergence plot
axes[0, 0].plot(range(len(best_values)), best_values, 'b-o', markersize=3, label='Bayesian Opt.', linewidth=2)
axes[0, 0].plot(range(len(random_best)), random_best, 'r--s', markersize=3, label='Random Search', linewidth=1.5, alpha=0.7)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Best Quality Score')
axes[0, 0].set_title('(a) Optimization Convergence')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 8b: Sampled points
temps = [denormalize(x)[0] for x in X_observed]
times = [denormalize(x)[1] for x in X_observed]
sc = axes[0, 1].scatter(temps, times, c=y_observed, cmap='RdYlGn', s=50, edgecolors='black', linewidth=0.5)
axes[0, 1].scatter([final_temp], [final_time], c='red', s=200, marker='*', zorder=5, label=f'Best: T={final_temp:.0f}K, t={final_time:.0f}min')
plt.colorbar(sc, ax=axes[0, 1], label='Quality')
axes[0, 1].set_xlabel('Temperature (K)')
axes[0, 1].set_ylabel('Time (min)')
axes[0, 1].set_title('(b) Sampled Points')
axes[0, 1].legend(fontsize=8)

# 8c: GP Surrogate Surface
temp_test = np.linspace(0, 1, 50)
time_test = np.linspace(0, 1, 50)
TT, TM = np.meshgrid(temp_test, time_test)
X_test = np.column_stack([TT.ravel(), TM.ravel()])
mu_pred, std_pred = gp.predict(X_test)
mu_grid = mu_pred.reshape(50, 50)

temp_labels = np.linspace(temp_bounds[0], temp_bounds[1], 50)
time_labels = np.linspace(time_bounds[0], time_bounds[1], 50)

im = axes[0, 2].contourf(temp_labels, time_labels, mu_grid, levels=20, cmap='RdYlGn')
plt.colorbar(im, ax=axes[0, 2], label='Predicted Quality')
axes[0, 2].scatter(temps, times, c='black', s=20, zorder=5, alpha=0.7)
axes[0, 2].scatter([final_temp], [final_time], c='red', s=200, marker='*', zorder=6)
axes[0, 2].set_xlabel('Temperature (K)')
axes[0, 2].set_ylabel('Time (min)')
axes[0, 2].set_title('(c) GP Surrogate Surface')

# 8d: Uncertainty Surface
std_grid = std_pred.reshape(50, 50)
im = axes[1, 0].contourf(temp_labels, time_labels, std_grid, levels=20, cmap='YlOrRd')
plt.colorbar(im, ax=axes[1, 0], label='Uncertainty (std)')
axes[1, 0].scatter(temps, times, c='black', s=20, zorder=5, alpha=0.7)
axes[1, 0].set_xlabel('Temperature (K)')
axes[1, 0].set_ylabel('Time (min)')
axes[1, 0].set_title('(d) GP Uncertainty')

# 8e: EI Surface
ei_values_grid = expected_improvement(X_test, gp, np.max(y_observed))
ei_grid = ei_values_grid.reshape(50, 50)
im = axes[1, 1].contourf(temp_labels, time_labels, ei_grid, levels=20, cmap='hot_r')
plt.colorbar(im, ax=axes[1, 1], label='Expected Improvement')
axes[1, 1].set_xlabel('Temperature (K)')
axes[1, 1].set_ylabel('Time (min)')
axes[1, 1].set_title('(e) Acquisition Function (EI)')

# 8f: Quality vs iteration (all points)
axes[1, 2].scatter(range(len(y_observed)), y_observed, c=range(len(y_observed)), cmap='viridis', s=40, edgecolors='black', linewidth=0.5)
axes[1, 2].axhline(y=y_observed[final_best_idx], color='red', linestyle='--', label=f'Best={y_observed[final_best_idx]:.3f}')
axes[1, 2].set_xlabel('Sample Index')
axes[1, 2].set_ylabel('Quality Score')
axes[1, 2].set_title('(f) All Evaluated Samples')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "bayesian_optimization.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 8: Bayesian optimization saved")

# ============================================================
# FIGURE 9: Optimization Landscape (Ground Truth)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Synthesis Optimization Landscape', fontsize=14, fontweight='bold')

# True landscape (dense evaluation)
np.random.seed(0)  # Fixed seed for reproducible landscape
temp_dense = np.linspace(temp_bounds[0], temp_bounds[1], 100)
time_dense = np.linspace(time_bounds[0], time_bounds[1], 100)
TD, TMD = np.meshgrid(temp_dense, time_dense)

# Evaluate without noise for clean landscape
quality_landscape = np.zeros_like(TD)
for i in range(100):
    for j in range(100):
        t_norm = (TD[i, j] - 350) / 100
        time_norm = (TMD[i, j] - 20) / 10
        quality_landscape[i, j] = (5.0 
                                    - 2.0 * (t_norm - 0.3)**2 
                                    - 1.5 * (time_norm - 0.2)**2
                                    + 0.5 * t_norm * time_norm
                                    - 0.3 * np.sin(3 * t_norm) * np.cos(2 * time_norm))

im = axes[0].contourf(temp_dense, time_dense, quality_landscape, levels=30, cmap='RdYlGn')
plt.colorbar(im, ax=axes[0], label='Quality Score')
axes[0].scatter(temps, times, c='black', s=15, zorder=5, alpha=0.5, label='BO samples')
axes[0].scatter([final_temp], [final_time], c='red', s=200, marker='*', zorder=6, label='BO optimum')
axes[0].set_xlabel('Temperature (K)')
axes[0].set_ylabel('Time (min)')
axes[0].set_title('(a) True Objective Landscape')
axes[0].legend(fontsize=8)

# 3D surface
from mpl_toolkits.mplot3d import Axes3D
ax3d = fig.add_subplot(122, projection='3d')
ax3d.plot_surface(TD, TMD, quality_landscape, cmap='RdYlGn', alpha=0.7, edgecolor='none')
ax3d.scatter(temps, times, y_observed, c='black', s=20, zorder=5)
ax3d.set_xlabel('Temperature (K)')
ax3d.set_ylabel('Time (min)')
ax3d.set_zlabel('Quality')
ax3d.set_title('(b) 3D Objective Surface')

# Remove the flat axes[1] since we replaced it
axes[1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "optimization_landscape.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 9: Optimization landscape saved")

# Save optimization results
opt_results = {
    "best_quality": float(y_observed[final_best_idx]),
    "optimal_temperature_K": float(final_temp),
    "optimal_time_min": float(final_time),
    "n_iterations": n_bo_iterations,
    "n_initial_samples": n_initial,
    "convergence_history": [float(v) for v in best_values],
    "random_search_best": [float(v) for v in random_best],
    "bo_improvement_over_random": float(best_values[-1] - random_best[-1]),
    "all_temperatures": [float(t) for t in temps],
    "all_times": [float(t) for t in times],
    "all_qualities": [float(q) for q in y_observed]
}

with open(os.path.join(OUTPUT_DIR, "optimization_results.json"), 'w') as f:
    json.dump(opt_results, f, indent=2)

print("\nPart 4 complete!")
