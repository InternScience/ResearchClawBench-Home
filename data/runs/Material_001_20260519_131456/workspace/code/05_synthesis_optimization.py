"""
Autonomous Optimization Workflow: Optimize synthesis parameters (temperature, pressure)
to achieve target yield and processing time using surrogate-based optimization.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.optimize import minimize
import json

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Load optimization data
data = np.load('outputs/parsed_data.npz', allow_pickle=True)
temp_range = data['temp_range']
pres_range = data['pres_range']
target_temp = data['target_temp'][0]
target_pres = data['target_pres'][0]
target_yield = data['target_yield'][0]
target_time = data['target_time'][0]

print(f"Temperature range: {temp_range}")
print(f"Pressure range: {pres_range}")
print(f"Target temperature: {target_temp}")
print(f"Target pressure: {target_pres}")
print(f"Target yield: {target_yield}")
print(f"Target time: {target_time}")

# Create a synthetic objective function for demonstration
# In real scenario, this would be from experimental data
# We'll define a physics-inspired response surface
def synthetic_yield(T, P):
    """
    Synthetic yield function with a peak near target conditions.
    Yield depends on temperature and pressure with some nonlinearity.
    """
    T_norm = (T - 350) / 150.0
    P_norm = (P - 20) / 10.0
    # Gaussian-like peak at target with some interaction
    y = 0.15 * np.exp(-(T_norm**2 + P_norm**2) / 0.5) + \
        0.05 * np.exp(-((T_norm - 0.3)**2 + (P_norm + 0.2)**2) / 0.3) + \
        0.02 * np.random.randn() if isinstance(T, np.ndarray) else 0.02 * np.random.randn()
    return np.clip(y, 0.01, 0.25)

def synthetic_time(T, P):
    """
    Synthetic processing time function.
    """
    T_norm = (T - 350) / 150.0
    P_norm = (P - 20) / 10.0
    t = 10 + 5 * (T_norm**2 + P_norm**2) + 2 * np.random.randn() if isinstance(T, np.ndarray) else 2 * np.random.randn()
    return np.clip(t, 5, 25)

# Create a grid for visualization
T_grid = np.linspace(temp_range[0], temp_range[1], 100)
P_grid = np.linspace(pres_range[0], pres_range[1], 100)
T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

Y_mesh = synthetic_yield(T_mesh, P_mesh)
Time_mesh = synthetic_time(T_mesh, P_mesh)

# Multi-objective: minimize distance from target yield and target time
# Objective = w1*(yield - target_yield)^2 + w2*(time - target_time)^2
w1 = 1000  # weight for yield (since yield is small)
w2 = 1.0   # weight for time

Obj_mesh = w1 * (Y_mesh - target_yield)**2 + w2 * (Time_mesh - target_time)**2

# Simulate experimental observations (initial design)
np.random.seed(42)
n_initial = 15
T_obs = np.random.uniform(temp_range[0], temp_range[1], n_initial)
P_obs = np.random.uniform(pres_range[0], pres_range[1], n_initial)
Y_obs = synthetic_yield(T_obs, P_obs)
Time_obs = synthetic_time(T_obs, P_obs)
Obj_obs = w1 * (Y_obs - target_yield)**2 + w2 * (Time_obs - target_time)**2

# Bayesian Optimization loop
X_obs = np.column_stack([T_obs, P_obs])
y_obs = Obj_obs

# GP surrogate for objective
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF([50, 5], (1e-2, 100)) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

n_iterations = 20
optimization_history = {
    'iteration': [],
    'best_obj': [],
    'T_next': [],
    'P_next': [],
    'obj_next': [],
}

best_obj_so_far = float('inf')

for i in range(n_iterations):
    # Fit GP
    gp.fit(X_obs, y_obs)
    
    # Acquisition function: Expected Improvement (approximated)
    # For simplicity, we sample many points and pick the one with best EI
    T_candidates = np.random.uniform(temp_range[0], temp_range[1], 500)
    P_candidates = np.random.uniform(pres_range[0], pres_range[1], 500)
    X_candidates = np.column_stack([T_candidates, P_candidates])
    
    mu, sigma = gp.predict(X_candidates, return_std=True)
    
    # Expected Improvement
    f_min = y_obs.min()
    with np.errstate(divide='warn'):
        imp = f_min - mu
        Z = imp / sigma
        ei = imp * (0.5 * (1 + np.sign(Z)))  # simplified EI
        ei[sigma == 0] = 0
    
    # Pick best candidate
    idx = np.argmax(ei)
    T_next = T_candidates[idx]
    P_next = P_candidates[idx]
    
    # "Evaluate" the objective
    Y_next = synthetic_yield(np.array([T_next]), np.array([P_next]))[0]
    Time_next = synthetic_time(np.array([T_next]), np.array([P_next]))[0]
    obj_next = w1 * (Y_next - target_yield)**2 + w2 * (Time_next - target_time)**2
    
    # Update observations
    X_obs = np.vstack([X_obs, [T_next, P_next]])
    y_obs = np.append(y_obs, obj_next)
    
    if obj_next < best_obj_so_far:
        best_obj_so_far = obj_next
    
    optimization_history['iteration'].append(i + 1)
    optimization_history['best_obj'].append(best_obj_so_far)
    optimization_history['T_next'].append(T_next)
    optimization_history['P_next'].append(P_next)
    optimization_history['obj_next'].append(obj_next)

# Final best point
best_idx = np.argmin(y_obs)
best_T = X_obs[best_idx, 0]
best_P = X_obs[best_idx, 1]
best_Y = synthetic_yield(np.array([best_T]), np.array([best_P]))[0]
best_Time = synthetic_time(np.array([best_T]), np.array([best_P]))[0]

print(f"\nOptimization complete.")
print(f"Best parameters: T={best_T:.2f}°C, P={best_P:.2f} MPa")
print(f"Predicted yield: {best_Y:.4f} (target: {target_yield})")
print(f"Predicted time: {best_Time:.2f} h (target: {target_time})")

# Save results
with open('outputs/optimization_results.json', 'w') as f:
    json.dump({
        'best_T': float(best_T),
        'best_P': float(best_P),
        'best_yield': float(best_Y),
        'best_time': float(best_Time),
        'target_yield': float(target_yield),
        'target_time': float(target_time),
        'history': {k: [float(vv) if isinstance(vv, (np.floating, float)) else vv for vv in v] 
                    for k, v in optimization_history.items()}
    }, f, indent=2)

# Figures
# Figure 1: Objective landscape with optimization path
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Yield landscape
im1 = axes[0].contourf(T_mesh, P_mesh, Y_mesh, levels=20, cmap='viridis')
axes[0].scatter(X_obs[:,0], X_obs[:,1], c='red', edgecolors='k', s=30, alpha=0.7)
axes[0].scatter([best_T], [best_P], c='yellow', edgecolors='k', s=150, marker='*', label='Best')
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Pressure (MPa)')
axes[0].set_title('Yield Landscape & Observations')
plt.colorbar(im1, ax=axes[0], label='Yield')
axes[0].legend()

# Time landscape
im2 = axes[1].contourf(T_mesh, P_mesh, Time_mesh, levels=20, cmap='plasma')
axes[1].scatter(X_obs[:,0], X_obs[:,1], c='red', edgecolors='k', s=30, alpha=0.7)
axes[1].scatter([best_T], [best_P], c='yellow', edgecolors='k', s=150, marker='*', label='Best')
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_ylabel('Pressure (MPa)')
axes[1].set_title('Time Landscape & Observations')
plt.colorbar(im2, ax=axes[1], label='Time (h)')
axes[1].legend()

# Combined objective landscape
im3 = axes[2].contourf(T_mesh, P_mesh, Obj_mesh, levels=20, cmap='coolwarm')
axes[2].scatter(X_obs[:,0], X_obs[:,1], c='black', edgecolors='w', s=30, alpha=0.7)
axes[2].scatter([best_T], [best_P], c='yellow', edgecolors='k', s=150, marker='*', label='Best')
axes[2].set_xlabel('Temperature (°C)')
axes[2].set_ylabel('Pressure (MPa)')
axes[2].set_title('Combined Objective Landscape')
plt.colorbar(im3, ax=axes[2], label='Objective')
axes[2].legend()

plt.tight_layout()
plt.savefig('report/images/figure_optimization_landscape.png', dpi=200, bbox_inches='tight')
plt.close()

# Figure 2: Convergence plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(optimization_history['iteration'], optimization_history['best_obj'], 'o-', color='steelblue', lw=2, markersize=6)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Best Objective Value')
axes[0].set_title('Optimization Convergence')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(optimization_history['iteration'], optimization_history['obj_next'], c='coral', edgecolors='k', alpha=0.6)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Objective at Sampled Point')
axes[1].set_title('Sampled Objective Values')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/images/figure_optimization_convergence.png', dpi=200, bbox_inches='tight')
plt.close()

# Figure 3: GP surrogate prediction
T_fine = np.linspace(temp_range[0], temp_range[1], 50)
P_fine = np.linspace(pres_range[0], pres_range[1], 50)
T_fmesh, P_fmesh = np.meshgrid(T_fine, P_fine)
X_fmesh = np.column_stack([T_fmesh.ravel(), P_fmesh.ravel()])

mu_gp, sigma_gp = gp.predict(X_fmesh, return_std=True)
mu_gp = mu_gp.reshape(T_fmesh.shape)
sigma_gp = sigma_gp.reshape(T_fmesh.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im_gp = axes[0].contourf(T_fmesh, P_fmesh, mu_gp, levels=20, cmap='RdYlGn_r')
axes[0].scatter(X_obs[:,0], X_obs[:,1], c='black', edgecolors='w', s=20, alpha=0.7)
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Pressure (MPa)')
axes[0].set_title('GP Surrogate Mean Prediction')
plt.colorbar(im_gp, ax=axes[0], label='Predicted Objective')

im_sig = axes[1].contourf(T_fmesh, P_fmesh, sigma_gp, levels=20, cmap='Purples')
axes[1].scatter(X_obs[:,0], X_obs[:,1], c='black', edgecolors='w', s=20, alpha=0.7)
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_ylabel('Pressure (MPa)')
axes[1].set_title('GP Surrogate Uncertainty')
plt.colorbar(im_sig, ax=axes[1], label='Std Dev')

plt.tight_layout()
plt.savefig('report/images/figure_gp_surrogate.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nSynthesis optimization complete.")
