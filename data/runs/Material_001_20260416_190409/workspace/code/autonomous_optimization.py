#!/usr/bin/env python3
"""
Autonomous Optimization Module for Materials AI

This module implements optimization workflows for synthesis and processing
parameters. It demonstrates the core AI workflow for experimental optimization
using Bayesian optimization and response surface methodology.

Based on related work:
- Machine-learning-assisted materials discovery using failed experiments
- Physics-informed machine learning for inverse problems
- Autonomous experimentation frameworks
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler


def load_optimization_data(filepath):
    """
    Load autonomous optimization data from the M-AI-Synth dataset.
    
    The dataset contains synthesis conditions and experimental parameters:
    - Temperature ranges
    - Pressure ranges  
    - Concentration values
    - Time durations
    - pH values
    - Other experimental parameters
    
    Returns:
        dict: Parsed optimization data with synthesis parameters
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    in_optimization_section = False
    params = {}
    param_names = ['temperature', 'pressure', 'concentration', 'time', 'ph', 'stirring_rate']
    param_idx = 0
    
    for line in lines:
        line = line.strip()
        if 'autonomous_optimization' in line.lower():
            in_optimization_section = True
            continue
        if in_optimization_section and line.startswith('['):
            array_str = line.strip('[]')
            values = [float(x) for x in array_str.split(', ') if x.strip()]
            if param_idx < len(param_names):
                params[param_names[param_idx]] = np.array(values)
                param_idx += 1
    
    return params


def define_synthesis_objective(params):
    """
    Define a synthetic objective function for synthesis optimization.
    
    This simulates a complex synthesis landscape where:
    - Optimal temperature is around 350°C
    - Optimal pressure is around 20 bar
    - There are interactions between parameters
    
    Returns:
        callable: Objective function to minimize (negative yield)
    """
    # Extract reference values
    temp_range = params.get('temperature', [200, 500])
    pressure_range = params.get('pressure', [10, 30])
    target_temp = params.get('concentration', [350])[0] if 'concentration' in params else 350
    target_pressure = params.get('time', [20])[0] if 'time' in params else 20
    
    def objective(x):
        """
        Synthetic yield function.
        x[0] = temperature, x[1] = pressure
        Returns negative yield (to minimize)
        """
        temp, pressure = x
        
        # Base yield centered around optimal conditions
        temp_term = -((temp - target_temp) ** 2) / (2 * 50 ** 2)
        pressure_term = -((pressure - target_pressure) ** 2) / (2 * 5 ** 2)
        
        # Interaction term
        interaction = 0.3 * (temp - target_temp) * (pressure - target_pressure) / 250
        
        # Noise/complexity
        noise = 0.1 * np.sin(temp * 0.05) * np.cos(pressure * 0.1)
        
        yield_value = 100 + temp_term + pressure_term + interaction + noise
        
        return -yield_value  # Minimize negative yield
    
    return objective


def bayesian_optimization_step(objective_func, bounds, n_initial=5, n_iterations=20, seed=42):
    """
    Perform Bayesian optimization using Gaussian Process surrogate model.
    
    Args:
        objective_func: Function to optimize
        bounds: Parameter bounds [(temp_min, temp_max), (pres_min, pres_max)]
        n_initial: Number of initial random samples
        n_iterations: Number of optimization iterations
        seed: Random seed
    
    Returns:
        dict: Optimization results with trajectory
    """
    np.random.seed(seed)
    
    # Initial random samples
    X_initial = np.zeros((n_initial, 2))
    X_initial[:, 0] = np.random.uniform(bounds[0][0], bounds[0][1], n_initial)
    X_initial[:, 1] = np.random.uniform(bounds[1][0], bounds[1][1], n_initial)
    
    y_initial = np.array([objective_func(x) for x in X_initial])
    
    # Store all evaluations
    X_history = [X_initial.copy()]
    y_history = [y_initial.copy()]
    
    # Fit Gaussian Process
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[10, 5], length_scale_bounds=(1e-1, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    
    best_x = X_initial[np.argmin(y_initial)]
    best_y = y_initial.min()
    
    acquisition_history = []
    
    for i in range(n_iterations):
        # Fit GP on current data
        X_all = np.vstack(X_history)
        y_all = np.concatenate(y_history)
        
        gp.fit(X_all, y_all)
        
        # Optimize acquisition function (Expected Improvement approximation)
        def acquisition(x):
            x = np.atleast_2d(x)
            mean, std = gp.predict(x, return_std=True)
            # Simple upper confidence bound
            return -(mean - 2 * std)
        
        # Find next point
        result = differential_evolution(acquisition, bounds, seed=seed+i, maxiter=50)
        x_next = result.x
        
        # Evaluate
        y_next = objective_func(x_next)
        
        # Update history
        X_history.append(x_next.reshape(1, -1))
        y_history.append(np.array([y_next]))
        
        # Track best
        if y_next < best_y:
            best_y = y_next
            best_x = x_next
        
        acquisition_history.append({
            'iteration': i + 1,
            'x': x_next.tolist(),
            'y': float(y_next),
            'best_so_far': float(best_y)
        })
        
        print(f"    Iteration {i+1}: x={x_next}, y={y_next:.4f}, best={best_y:.4f}")
    
    X_final = np.vstack(X_history)
    y_final = np.concatenate(y_history)
    
    return {
        'best_x': best_x.tolist(),
        'best_y': float(best_y),
        'best_temperature': float(best_x[0]),
        'best_pressure': float(best_x[1]),
        'estimated_yield': float(-best_y),
        'X_history': X_final.tolist(),
        'y_history': y_final.tolist(),
        'acquisition_trajectory': acquisition_history,
        'n_evaluations': len(y_final)
    }


def response_surface_analysis(objective_func, bounds, resolution=50):
    """
    Generate response surface for visualization.
    """
    temp_range = np.linspace(bounds[0][0], bounds[0][1], resolution)
    pressure_range = np.linspace(bounds[1][0], bounds[1][1], resolution)
    
    T, P = np.meshgrid(temp_range, pressure_range)
    Z = np.zeros_like(T)
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = -objective_func([T[i, j], P[i, j]])  # Convert back to yield
    
    return {
        'temperature_grid': T,
        'pressure_grid': P,
        'yield_surface': Z,
        'max_yield': float(Z.max()),
        'optimal_temp': float(T[np.unravel_index(Z.argmax(), Z.shape)]),
        'optimal_pressure': float(P[np.unravel_index(Z.argmax(), Z.shape)])
    }


def generate_optimization_plots(opt_results, response_surface, bounds, output_dir):
    """
    Generate visualization plots for optimization results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Response surface contour
    ax = axes[0, 0]
    T = response_surface['temperature_grid']
    P = response_surface['pressure_grid']
    Z = response_surface['yield_surface']
    
    contour = ax.contourf(T, P, Z, levels=20, cmap='viridis', alpha=0.8)
    ax.plot(opt_results['best_x'][0], opt_results['best_x'][1], 'r*', markersize=15, 
            label=f'Optimum: {opt_results["estimated_yield"]:.1f}%')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_title('Synthesis Yield Response Surface')
    ax.legend()
    plt.colorbar(contour, ax=ax, label='Yield (%)')
    
    # Plot 2: Optimization convergence
    ax = axes[0, 1]
    y_history = opt_results['y_history']
    best_so_far = np.minimum.accumulate(y_history)
    ax.plot(range(len(y_history)), y_history, 'b-', alpha=0.5, label='All evaluations')
    ax.plot(range(len(best_so_far)), best_so_far, 'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Negative Yield')
    ax.set_title('Optimization Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Acquisition trajectory
    ax = axes[0, 2]
    X_hist = np.array(opt_results['X_history'])
    ax.scatter(X_hist[:, 0], X_hist[:, 1], c=range(len(X_hist)), 
               cmap='viridis', s=50, edgecolors='black')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (bar)')
    ax.set_title('Sampling Trajectory')
    plt.colorbar(ax.collections[0], ax=ax, label='Iteration')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Yield distribution
    ax = axes[1, 0]
    yields = [-y for y in y_history]  # Convert to positive yields
    ax.hist(yields, bins=15, alpha=0.7, color='#3498db', edgecolor='black')
    ax.axvline(opt_results['estimated_yield'], color='red', linestyle='--', 
               label=f'Optimal: {opt_results["estimated_yield"]:.1f}%')
    ax.set_xlabel('Yield (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Yield Distribution Across Evaluations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Temperature vs Yield
    ax = axes[1, 1]
    ax.scatter(X_hist[:, 0], yields, alpha=0.6, c=yields, cmap='viridis')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Yield (%)')
    ax.set_title('Temperature Effect on Yield')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Pressure vs Yield
    ax = axes[1, 2]
    ax.scatter(X_hist[:, 1], yields, alpha=0.6, c=yields, cmap='plasma')
    ax.set_xlabel('Pressure (bar)')
    ax.set_ylabel('Yield (%)')
    ax.set_title('Pressure Effect on Yield')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimization.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_optimization_results(opt_results, response_surface, params, output_dir):
    """Save detailed results to JSON."""
    output = {
        'input_parameters': {
            'temperature_range': params.get('temperature', [200, 500]).tolist() if hasattr(params.get('temperature', []), 'tolist') else list(params.get('temperature', [200, 500])),
            'pressure_range': params.get('pressure', [10, 30]).tolist() if hasattr(params.get('pressure', []), 'tolist') else list(params.get('pressure', [10, 30])),
            'target_conditions': {
                'temperature': float(params.get('concentration', [350])[0]) if 'concentration' in params else 350,
                'pressure': float(params.get('time', [20])[0]) if 'time' in params else 20
            }
        },
        'optimization_results': {
            'optimal_temperature': opt_results['best_temperature'],
            'optimal_pressure': opt_results['best_pressure'],
            'estimated_max_yield': opt_results['estimated_yield'],
            'n_evaluations': opt_results['n_evaluations'],
            'convergence_history': opt_results['acquisition_trajectory'][-5:]  # Last 5 iterations
        },
        'response_surface_summary': {
            'max_yield': response_surface['max_yield'],
            'optimal_temp_from_surface': response_surface['optimal_temp'],
            'optimal_pressure_from_surface': response_surface['optimal_pressure']
        }
    }
    
    with open(f'{output_dir}/optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


if __name__ == '__main__':
    import os
    
    # Paths
    data_path = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_001_20260416_190409/data/M-AI-Synth__Materials_AI_Dataset_.txt'
    output_dir = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Material_001_20260416_190409/outputs'
    
    print("=" * 60)
    print("AUTONOMOUS OPTIMIZATION WORKFLOW")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading optimization data...")
    params = load_optimization_data(data_path)
    for key, value in params.items():
        print(f"    {key}: {value}")
    
    # Define objective
    print("\n[2] Defining synthesis objective function...")
    objective = define_synthesis_objective(params)
    
    # Set bounds
    bounds = [
        (params['temperature'][0], params['temperature'][1]),
        (params['pressure'][0], params['pressure'][1])
    ]
    print(f"    Search bounds: T={bounds[0]}, P={bounds[1]}")
    
    # Run Bayesian optimization
    print("\n[3] Running Bayesian optimization...")
    opt_results = bayesian_optimization_step(objective, bounds, n_initial=5, n_iterations=20)
    print(f"\n    Best conditions found:")
    print(f"      Temperature: {opt_results['best_temperature']:.2f} °C")
    print(f"      Pressure: {opt_results['best_pressure']:.2f} bar")
    print(f"      Estimated yield: {opt_results['estimated_yield']:.2f}%")
    
    # Response surface analysis
    print("\n[4] Generating response surface...")
    response_surface = response_surface_analysis(objective, bounds)
    print(f"    Surface maximum yield: {response_surface['max_yield']:.2f}%")
    print(f"    Optimal T from surface: {response_surface['optimal_temp']:.2f} °C")
    print(f"    Optimal P from surface: {response_surface['optimal_pressure']:.2f} bar")
    
    # Generate plots
    print("\n[5] Generating visualization plots...")
    generate_optimization_plots(opt_results, response_surface, bounds, output_dir)
    print(f"    Saved: {output_dir}/optimization.png")
    
    # Save results
    print("\n[6] Saving results...")
    summary = save_optimization_results(opt_results, response_surface, params, output_dir)
    print(f"    Saved: {output_dir}/optimization_results.json")
    
    print("\n" + "=" * 60)
    print("AUTONOMOUS OPTIMIZATION COMPLETE")
    print("=" * 60)
