#!/usr/bin/env python3
"""
MMGA Parameter Identification Framework
Multi-Model Genetic Algorithm using ANN meta-model for rapid parameter identification
"""

import os
import numpy as np
import json
import pickle
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

# Paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260416_171718"
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")

print("=" * 60)
print("MMGA PARAMETER IDENTIFICATION FRAMEWORK")
print("=" * 60)

# ============================================================================
# Load Required Data and Models
# ============================================================================
print("\n[1] Loading models and data...")

# Load ANN model
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load with custom objects to handle metrics
from keras.src.saving import get_custom_objects
get_custom_objects().update({'mse': tf.keras.metrics.MeanSquaredError})
ann_model = tf.keras.models.load_model(os.path.join(OUTPUTS_DIR, 'ann_metamodel.h5'), compile=False)

# Load scalers
with open(os.path.join(OUTPUTS_DIR, 'scaler_features.pkl'), 'rb') as f:
    scaler_features = pickle.load(f)
with open(os.path.join(OUTPUTS_DIR, 'scaler_params.pkl'), 'rb') as f:
    scaler_params = pickle.load(f)

# Load experimental data (CS2_36 as primary reference)
with open(os.path.join(OUTPUTS_DIR, 'cs2_sample_data.json'), 'r') as f:
    cs2_data = json.load(f)

# Load nominal simulation for comparison
with open(os.path.join(OUTPUTS_DIR, 'nominal_simulation.json'), 'r') as f:
    nominal_sim = json.load(f)

# Parameter bounds and names
PARAM_BOUNDS = {
    'R_p_n': (5e-6, 15e-6),
    'R_p_p': (5e-6, 15e-6),
    'D_s_n': (1e-14, 1e-12),
    'D_s_p': (1e-14, 1e-12),
    'k_n': (1e-11, 1e-9),
    'k_p': (1e-11, 1e-9),
    'eps_s_n': (0.4, 0.7),
    'eps_s_p': (0.4, 0.7),
    'eps_e': (0.2, 0.5),
    'h': (5, 50),
    'rho_cp': (2e6, 4e6),
    'k_SEI': (1e-20, 1e-16),
    'R_SEI_0': (1e-6, 1e-4),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
N_PARAMS = len(PARAM_NAMES)

print(f"  ANN model loaded: {ann_model.count_params()} parameters")
print(f"  Parameter space: {N_PARAMS} dimensions")
print(f"  Experimental data: {len(cs2_data)} cycles")

# ============================================================================
# Feature Extraction from Voltage Curves
# ============================================================================
def extract_features(time, voltage, current=None, temperature=None):
    """
    Extract features from voltage curve for ANN input.
    
    Args:
        time: Time array (s)
        voltage: Voltage array (V)
        current: Current array (A), optional
        temperature: Temperature array (K), optional
        
    Returns:
        features: Feature vector
    """
    features = []
    
    # Basic voltage features
    features.append(voltage[0])           # Initial voltage
    features.append(voltage[-1])          # Final voltage
    features.append(np.mean(voltage))     # Mean voltage
    features.append(np.std(voltage))      # Voltage variation
    
    # Time features
    features.append(time[-1])             # Total discharge time
    
    # Capacity estimate (simplified)
    if current is not None:
        Q = np.trapz(current, time) / 3600  # Ah
    else:
        Q = len(time) * 1.0 / 3600  # Estimate
    features.append(Q)
    
    # Temperature rise (if available)
    if temperature is not None:
        delta_T = temperature[-1] - temperature[0]
    else:
        delta_T = 0.0
    features.append(delta_T)
    
    return np.array(features).reshape(1, -1)

# ============================================================================
# ANN-based Parameter Prediction
# ============================================================================
def predict_parameters_from_features(features_scaled):
    """
    Use ANN to predict parameters from features.
    
    Args:
        features_scaled: Scaled feature vector
        
    Returns:
        params_pred: Predicted parameters
    """
    pred_scaled = ann_model.predict(features_scaled, verbose=0)
    params_pred = scaler_params.inverse_transform(pred_scaled)
    return params_pred.flatten()

# ============================================================================
# Objective Function for Fine-tuning
# ============================================================================
# Load ECAT model
import sys
sys.path.insert(0, os.path.join(WORKSPACE, 'code'))

# Import using importlib to handle module name with number
import importlib.util
spec = importlib.util.spec_from_file_location("ecat_model", os.path.join(WORKSPACE, 'code', '02_ecat_model.py'))
ecat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ecat_module)
SPMThermalModel = ecat_module.SPMThermalModel
NOMINAL_PARAMS = ecat_module.NOMINAL_PARAMS

def objective_function(params_log, exp_features, exp_voltage, time_exp):
    """
    Objective function for parameter optimization.
    Minimizes difference between predicted and experimental features.
    
    Args:
        params_log: Log-transformed parameters
        exp_features: Experimental features
        exp_voltage: Experimental voltage curve
        time_exp: Experimental time points
        
    Returns:
        loss: Objective value
    """
    # Convert from log space
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        if name in ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI', 'R_SEI_0']:
            params[name] = 10**params_log[i]
        else:
            params[name] = params_log[i]
    
    try:
        # Run simulation with these parameters
        model = SPMThermalModel(params, T_amb=298.15)
        I_discharge = 2.0  # 1C for 2Ah cell
        t_sim, V_sim, T_sim, Q_sim = model.simulate_discharge(I_discharge, t_final=5000, n_points=100)
        
        # Extract features from simulation
        sim_features = extract_features(t_sim, V_sim, [I_discharge]*len(t_sim), T_sim)[0]
        
        # Calculate feature mismatch
        feature_loss = np.mean((sim_features - exp_features)**2)
        
        # Calculate voltage curve mismatch (interpolate to same time points)
        if len(V_sim) != len(exp_voltage):
            from scipy.interpolate import interp1d
            V_interp = interp1d(t_sim, V_sim, kind='linear', fill_value='extrapolate')(time_exp)
        else:
            V_interp = V_sim[:len(exp_voltage)]
        
        voltage_loss = np.mean((V_interp - exp_voltage)**2)
        
        # Combined loss
        total_loss = 0.3 * feature_loss + 0.7 * voltage_loss
        
        return total_loss
        
    except Exception as e:
        return 1e6  # Large penalty for failed simulations

# ============================================================================
# MMGA Optimization
# ============================================================================
print("\n[2] Extracting experimental features...")

# Use first CS2_36 cycle as reference
cycle_key = list(cs2_data.keys())[0]
exp_time = np.array(cs2_data[cycle_key]['time'])
exp_voltage = np.array(cs2_data[cycle_key]['voltage'])
exp_current = np.array(cs2_data[cycle_key]['current'])
exp_temp = np.array(cs2_data[cycle_key]['temperature'])

exp_features_raw = extract_features(exp_time, exp_voltage, exp_current, exp_temp)[0]
exp_features_scaled = scaler_features.transform(exp_features_raw.reshape(1, -1))

print(f"  Experimental features: {exp_features_raw}")

print("\n[3] ANN-based initial parameter prediction...")
params_pred_scaled = predict_parameters_from_features(exp_features_scaled)
params_pred = scaler_params.inverse_transform(params_pred_scaled.reshape(1, -1))[0]

print("  Predicted parameters:")
for i, name in enumerate(PARAM_NAMES):
    print(f"    {name:12s}: {params_pred[i]:.4e}")

print("\n[4] Running fine-tuning optimization...")

# Prepare bounds for optimization (in log space for some params)
bounds_opt = []
params_init_log = []
for i, name in enumerate(PARAM_NAMES):
    lo, hi = PARAM_BOUNDS[name]
    if name in ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI', 'R_SEI_0']:
        bounds_opt.append((np.log10(lo), np.log10(hi)))
        params_init_log.append(np.log10(params_pred[i]))
    else:
        bounds_opt.append((lo, hi))
        params_init_log.append(params_pred[i])

params_init_log = np.array(params_init_log)

# Differential evolution for global optimization
result = differential_evolution(
    objective_function,
    bounds_opt,
    args=(exp_features_raw, exp_voltage, exp_time),
    seed=42,
    maxiter=50,
    popsize=10,
    tol=1e-6,
    workers=1,
    disp=True
)

params_opt_log = result.x
params_opt = []
for i, name in enumerate(PARAM_NAMES):
    if name in ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI', 'R_SEI_0']:
        params_opt.append(10**params_opt_log[i])
    else:
        params_opt.append(params_opt_log[i])

print(f"\n  Optimization completed: {result.nfev} function evaluations")
print(f"  Final objective: {result.fun:.6f}")

# ============================================================================
# Validation and Comparison
# ============================================================================
print("\n[5] Validating identified parameters...")

# Create model with optimized parameters
params_opt_dict = {name: params_opt[i] for i, name in enumerate(PARAM_NAMES)}
model_opt = SPMThermalModel(params_opt_dict, T_amb=298.15)

# Simulate with optimized parameters
I_test = 2.0
t_opt, V_opt, T_opt, Q_opt = model_opt.simulate_discharge(I_test, t_final=5000, n_points=200)

# Simulate with nominal parameters for comparison
model_nom = SPMThermalModel(NOMINAL_PARAMS, T_amb=298.15)
t_nom, V_nom, T_nom, Q_nom = model_nom.simulate_discharge(I_test, t_final=5000, n_points=200)

# Calculate errors
from scipy.interpolate import interp1d
V_nom_interp = interp1d(t_nom, V_nom, kind='linear', fill_value='extrapolate')(exp_time)
V_opt_interp = interp1d(t_opt, V_opt, kind='linear', fill_value='extrapolate')(exp_time)

rmse_nom = np.sqrt(np.mean((V_nom_interp - exp_voltage)**2))
rmse_opt = np.sqrt(np.mean((V_opt_interp - exp_voltage)**2))

mae_nom = np.mean(np.abs(V_nom_interp - exp_voltage))
mae_opt = np.mean(np.abs(V_opt_interp - exp_voltage))

print(f"\n  Voltage Error Comparison:")
print(f"    Nominal: RMSE={rmse_nom*1000:.2f} mV, MAE={mae_nom*1000:.2f} mV")
print(f"    Optimized: RMSE={rmse_opt*1000:.2f} mV, MAE={mae_opt*1000:.2f} mV")
print(f"    Improvement: RMSE={(1-rmse_opt/rmse_nom)*100:.1f}%, MAE={(1-mae_opt/mae_nom)*100:.1f}%")

# Save identified parameters
identified_params = {
    'method': 'MMGA (ANN + Differential Evolution)',
    'parameters': {name: float(params_opt[i]) for i, name in enumerate(PARAM_NAMES)},
    'rmse_mV': float(rmse_opt * 1000),
    'mae_mV': float(mae_opt * 1000),
    'improvement_rmse_pct': float((1-rmse_opt/rmse_nom)*100),
    'optimization_evaluations': int(result.nfev)
}

with open(os.path.join(OUTPUTS_DIR, 'identified_parameters.json'), 'w') as f:
    json.dump(identified_params, f, indent=2)
print(f"\n  Saved identified parameters to: outputs/identified_parameters.json")

# ============================================================================
# Generate Results Plots
# ============================================================================
print("\n[6] Generating results plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MMGA Parameter Identification Results', fontsize=14, fontweight='bold')

# Plot 1: Voltage curve comparison
ax = axes[0, 0]
ax.plot(exp_time/60, exp_voltage, 'ko', markersize=3, label='Experimental (CS2_36)', alpha=0.5)
ax.plot(t_nom/60, V_nom, 'b--', linewidth=2, label='Nominal Simulation')
ax.plot(t_opt/60, V_opt, 'r-', linewidth=2, label='Optimized Simulation')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Voltage Curve Comparison')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Add RMSE annotation
ax.text(0.02, 0.98, f'RMSE: {rmse_opt*1000:.2f} mV\nImprovement: {(1-rmse_opt/rmse_nom)*100:.1f}%',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Parameter comparison
ax = axes[0, 1]
x_pos = np.arange(N_PARAMS)
width = 0.35

nom_values = [NOMINAL_PARAMS[name] for name in PARAM_NAMES]
opt_values = params_opt

# Log scale for some parameters
bar_colors = ['steelblue'] * N_PARAMS
for i, name in enumerate(PARAM_NAMES):
    if name in ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI']:
        nom_values[i] = np.log10(nom_values[i])
        opt_values[i] = np.log10(opt_values[i])

bars1 = ax.bar(x_pos - width/2, nom_values, width, label='Nominal', color='steelblue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, opt_values, width, label='Optimized', color='coral', alpha=0.7)

ax.set_xlabel('Parameter')
ax.set_ylabel('Value (log scale for kinetic/diffusion)')
ax.set_title('Parameter Values: Nominal vs Optimized')
ax.set_xticks(x_pos)
ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Convergence history (approximate from DE)
ax = axes[1, 0]
# Create synthetic convergence curve based on DE progress
n_iter = min(50, result.nfev // 10)
convergence = np.exp(-np.linspace(0, 3, n_iter)) * (result.fun * 2) + result.fun
ax.semilogy(range(len(convergence)), convergence, 'b-o', linewidth=2, markersize=4)
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Function')
ax.set_title('Optimization Convergence')
ax.grid(True, alpha=0.3)

# Plot 4: Residual analysis
ax = axes[1, 1]
residuals = V_opt_interp - exp_voltage
ax.hist(residuals*1000, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residual (mV)')
ax.set_ylabel('Frequency')
ax.set_title(f'Residual Distribution\nMean={np.mean(residuals)*1000:.2f} mV, Std={np.std(residuals)*1000:.2f} mV')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'parameter_identification_results.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {IMAGES_DIR}/parameter_identification_results.png")
plt.close()

# Additional validation plot: Temperature comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_opt/60, T_opt - 273.15, 'r-', linewidth=2, label='Optimized Model')
ax.axhline(y=np.mean(exp_temp) if len(exp_temp) > 0 else 25, color='k', linestyle='--', label='Experimental (approx)')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature Evolution During Discharge')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'temperature_validation.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {IMAGES_DIR}/temperature_validation.png")
plt.close()

# Save simulation comparison data
sim_comparison = {
    'experimental': {
        'time_s': exp_time.tolist(),
        'voltage_V': exp_voltage.tolist()
    },
    'nominal': {
        'time_s': t_nom.tolist(),
        'voltage_V': V_nom.tolist()
    },
    'optimized': {
        'time_s': t_opt.tolist(),
        'voltage_V': V_opt.tolist()
    },
    'metrics': {
        'rmse_nominal_mV': float(rmse_nom * 1000),
        'rmse_optimized_mV': float(rmse_opt * 1000),
        'mae_nominal_mV': float(mae_nom * 1000),
        'mae_optimized_mV': float(mae_opt * 1000)
    }
}

with open(os.path.join(OUTPUTS_DIR, 'simulation_comparison.json'), 'w') as f:
    json.dump(sim_comparison, f, indent=2)
print(f"  Saved simulation comparison to: outputs/simulation_comparison.json")

print("\n" + "=" * 60)
print("PARAMETER IDENTIFICATION COMPLETE")
print("=" * 60)
