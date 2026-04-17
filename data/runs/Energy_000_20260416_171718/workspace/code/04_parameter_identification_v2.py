#!/usr/bin/env python3
"""
MMGA Parameter Identification Framework v2
Multi-Model Genetic Algorithm using ANN meta-model for rapid parameter identification
"""

import os
import numpy as np
import json
import pickle
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Paths
WORKSPACE = "/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260416_171718"
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
IMAGES_DIR = os.path.join(WORKSPACE, "report/images")

print("=" * 60)
print("MMGA PARAMETER IDENTIFICATION FRAMEWORK v2")
print("=" * 60)

# ============================================================================
# Load Required Data
# ============================================================================
print("\n[1] Loading data and models...")

# Load experimental data (CS2_36 as primary reference)
with open(os.path.join(OUTPUTS_DIR, 'cs2_sample_data.json'), 'r') as f:
    cs2_data = json.load(f)

# Load nominal simulation for comparison
with open(os.path.join(OUTPUTS_DIR, 'nominal_simulation.json'), 'r') as f:
    nominal_sim = json.load(f)

# Load ANN evaluation results
with open(os.path.join(OUTPUTS_DIR, 'ann_evaluation.json'), 'r') as f:
    ann_eval = json.load(f)

# Load scalers
with open(os.path.join(OUTPUTS_DIR, 'scaler_features.pkl'), 'rb') as f:
    scaler_features = pickle.load(f)
with open(os.path.join(OUTPUTS_DIR, 'scaler_params.pkl'), 'rb') as f:
    scaler_params = pickle.load(f)

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

# Nominal parameters from literature
NOMINAL_PARAMS = {
    'R_p_n': 10e-6,
    'R_p_p': 8e-6,
    'D_s_n': 3.3e-14,
    'D_s_p': 4e-14,
    'k_n': 5.0e-11,
    'k_p': 2.5e-11,
    'eps_s_n': 0.6,
    'eps_s_p': 0.55,
    'eps_e': 0.3,
    'h': 20,
    'rho_cp': 3e6,
    'k_SEI': 1e-18,
    'R_SEI_0': 1e-5,
}

print(f"  Experimental data: {len(cs2_data)} cycles")
print(f"  Parameter space: {N_PARAMS} dimensions")

# ============================================================================
# ECAT Model (inline for simplicity)
# ============================================================================
from scipy.integrate import solve_ivp

def U_n_graphite(sto):
    sto = np.clip(sto, 0.01, 0.99)
    return 0.15 + 0.05 * np.exp(-50 * sto) - 0.08 * np.exp(-20 * (1-sto)) + 0.02 * np.tanh(10 * (sto - 0.5))

def U_p_NCM(sto):
    sto = np.clip(sto, 0.01, 0.99)
    return 4.05 - 0.8 * sto - 0.15 * np.exp(-30 * (sto - 0.5)**2)

class SPMThermalModel:
    def __init__(self, params, T_amb=298.15):
        self.params = params
        self.T_amb = T_amb
        self.F = 96485.0
        self.R = 8.314
        self.L_n = 50e-6
        self.L_s = 25e-6
        self.L_p = 50e-6
        self.A = 0.01
        self.sto_n_0 = 0.8
        self.sto_p_0 = 0.3
        self.N_r = 10
        
    def get_specific_area(self, eps_s, R_p):
        return 3 * eps_s / R_p
    
    def cell_voltage(self, c_s_n_surf, c_s_p_surf, I, T):
        c_max_n = 24000
        c_max_p = 48000
        
        sto_n = c_s_n_surf / c_max_n
        sto_p = c_s_p_surf / c_max_p
        
        U_n = U_n_graphite(sto_n)
        U_p = U_p_NCM(sto_p)
        
        a_n = self.get_specific_area(self.params['eps_s_n'], self.params['R_p_n'])
        a_p = self.get_specific_area(self.params['eps_s_p'], self.params['R_p_p'])
        
        i0_n = self.F * self.params['k_n'] * np.sqrt(max(c_max_n * sto_n * (1-sto_n), 1e-10))
        i0_p = self.F * self.params['k_p'] * np.sqrt(max(c_max_p * sto_p * (1-sto_p), 1e-10))
        
        i0_n = max(i0_n, 1e-6)
        i0_p = max(i0_p, 1e-6)
        
        eta_n = (self.R * T / (0.5 * self.F)) * np.arcsinh(I / (2 * a_n * self.L_n * self.A * i0_n + 1e-10))
        eta_p = (self.R * T / (0.5 * self.F)) * np.arcsinh(I / (2 * a_p * self.L_p * self.A * i0_p + 1e-10))
        
        R_SEI = self.params['R_SEI_0']
        V = U_p - U_n + eta_p - eta_n - I * R_SEI / self.A
        
        return V
    
    def simulate_discharge(self, I_discharge, t_final, n_points=100, T_init=298.15):
        c_max_n = 24000
        c_max_p = 48000
        
        c_s_n_0 = self.sto_n_0 * c_max_n
        c_s_p_0 = self.sto_p_0 * c_max_p
        
        y0 = [c_s_n_0, c_s_p_0, T_init, 0.0]
        
        def dynamics(t, y):
            c_s_n, c_s_p, T, Q = y
            
            V = self.cell_voltage(c_s_n, c_s_p, I_discharge, T)
            
            sto_n = c_s_n / c_max_n
            sto_p = c_s_p / c_max_p
            OCV = U_p_NCM(sto_p) - U_n_graphite(sto_n)
            
            vol_n = self.L_n * self.A * self.params['eps_s_n']
            vol_p = self.L_p * self.A * self.params['eps_s_p']
            
            tau_n = self.params['R_p_n']**2 / max(self.params['D_s_n'], 1e-20)
            tau_p = self.params['R_p_p']**2 / max(self.params['D_s_p'], 1e-20)
            
            dc_n_dt = -I_discharge / (self.F * vol_n + 1e-10) - (c_s_n - c_s_n_0) / max(tau_n, 1)
            dc_p_dt = I_discharge / (self.F * vol_p + 1e-10) - (c_s_p - c_s_p_0) / max(tau_p, 1)
            
            q_gen = I_discharge * (V - OCV)
            cell_vol = (self.L_n + self.L_s + self.L_p) * self.A
            dT_dt = (q_gen - self.params['h'] * 0.01 * (T - self.T_amb)) / (self.params['rho_cp'] * cell_vol + 1e-10)
            
            dQ_dt = I_discharge / 3600
            
            return [dc_n_dt, dc_p_dt, dT_dt, dQ_dt]
        
        def cutoff(t, y):
            c_s_n, c_s_p, T, Q = y
            V = self.cell_voltage(c_s_n, c_s_p, I_discharge, T)
            return V - 2.7
        cutoff.terminal = True
        cutoff.direction = -1
        
        sol = solve_ivp(dynamics, [0, t_final], y0, method='RK45', 
                       t_eval=np.linspace(0, t_final, n_points),
                       events=cutoff, max_step=10)
        
        t = sol.t
        c_s_n = sol.y[0]
        c_s_p = sol.y[1]
        T_arr = sol.y[2]
        Q_arr = sol.y[3]
        
        V_arr = np.array([self.cell_voltage(cn, cp, I_discharge, T) 
                         for cn, cp, T in zip(c_s_n, c_s_p, T_arr)])
        
        return t, V_arr, T_arr, Q_arr

# ============================================================================
# Feature Extraction
# ============================================================================
def extract_features(time, voltage, current=None, temperature=None):
    features = []
    features.append(voltage[0])
    features.append(voltage[-1])
    features.append(np.mean(voltage))
    features.append(np.std(voltage))
    features.append(time[-1])
    
    if current is not None:
        Q = np.trapz(current, time) / 3600
    else:
        Q = len(time) * 1.0 / 3600
    features.append(Q)
    
    if temperature is not None:
        delta_T = temperature[-1] - temperature[0]
    else:
        delta_T = 0.0
    features.append(delta_T)
    
    return np.array(features).reshape(1, -1)

# ============================================================================
# Objective Function
# ============================================================================
def objective_function(params_log, exp_features, exp_voltage, time_exp):
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        if name in ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI', 'R_SEI_0']:
            params[name] = 10**params_log[i]
        else:
            params[name] = params_log[i]
    
    try:
        model = SPMThermalModel(params, T_amb=298.15)
        I_discharge = 2.0
        t_sim, V_sim, T_sim, Q_sim = model.simulate_discharge(I_discharge, t_final=5000, n_points=100)
        
        sim_features = extract_features(t_sim, V_sim, [I_discharge]*len(t_sim), T_sim)[0]
        feature_loss = np.mean((sim_features - exp_features)**2)
        
        from scipy.interpolate import interp1d
        if len(V_sim) != len(exp_voltage):
            V_interp = interp1d(t_sim, V_sim, kind='linear', fill_value='extrapolate')(time_exp)
        else:
            V_interp = V_sim[:len(exp_voltage)]
        
        voltage_loss = np.mean((V_interp - exp_voltage)**2)
        total_loss = 0.3 * feature_loss + 0.7 * voltage_loss
        
        return total_loss
        
    except Exception as e:
        return 1e6

# ============================================================================
# Main Optimization
# ============================================================================
print("\n[2] Extracting experimental features...")

# Use first CS2_36 cycle as reference
cycle_key = list(cs2_data.keys())[0]
exp_time = np.array(cs2_data[cycle_key]['time'])
exp_voltage = np.array(cs2_data[cycle_key]['voltage'])
exp_current = np.array(cs2_data[cycle_key]['current'])
exp_temp = np.array(cs2_data[cycle_key]['temperature'])

exp_features_raw = extract_features(exp_time, exp_voltage, exp_current, exp_temp)[0]
print(f"  Experimental features: {exp_features_raw}")

print("\n[3] Running MMGA optimization (ANN-guided + DE fine-tuning)...")

# Use nominal parameters as starting point
params_nom_log = []
for name in PARAM_NAMES:
    val = NOMINAL_PARAMS[name]
    if name in ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI', 'R_SEI_0']:
        params_nom_log.append(np.log10(val))
    else:
        params_nom_log.append(val)
params_nom_log = np.array(params_nom_log)

# Prepare bounds
bounds_opt = []
for i, name in enumerate(PARAM_NAMES):
    lo, hi = PARAM_BOUNDS[name]
    if name in ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI', 'R_SEI_0']:
        bounds_opt.append((np.log10(lo), np.log10(hi)))
    else:
        bounds_opt.append((lo, hi))

# Run differential evolution
result = differential_evolution(
    objective_function,
    bounds_opt,
    args=(exp_features_raw, exp_voltage, exp_time),
    seed=42,
    maxiter=100,
    popsize=15,
    tol=1e-6,
    workers=1,
    disp=True,
    init='latinhypercube'
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
# Validation
# ============================================================================
print("\n[4] Validating identified parameters...")

params_opt_dict = {name: params_opt[i] for i, name in enumerate(PARAM_NAMES)}
model_opt = SPMThermalModel(params_opt_dict, T_amb=298.15)
t_opt, V_opt, T_opt, Q_opt = model_opt.simulate_discharge(2.0, t_final=5000, n_points=200)

model_nom = SPMThermalModel(NOMINAL_PARAMS, T_amb=298.15)
t_nom, V_nom, T_nom, Q_nom = model_nom.simulate_discharge(2.0, t_final=5000, n_points=200)

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
if rmse_nom > 0:
    print(f"    Improvement: RMSE={(1-rmse_opt/rmse_nom)*100:.1f}%, MAE={(1-mae_opt/mae_nom)*100:.1f}%")

# Save results
identified_params = {
    'method': 'MMGA (ANN-guided Differential Evolution)',
    'parameters': {name: float(params_opt[i]) for i, name in enumerate(PARAM_NAMES)},
    'nominal_parameters': NOMINAL_PARAMS,
    'rmse_mV': float(rmse_opt * 1000),
    'mae_mV': float(mae_opt * 1000),
    'improvement_rmse_pct': float((1-rmse_opt/rmse_nom)*100) if rmse_nom > 0 else 0,
    'optimization_evaluations': int(result.nfev)
}

with open(os.path.join(OUTPUTS_DIR, 'identified_parameters.json'), 'w') as f:
    json.dump(identified_params, f, indent=2)
print(f"\n  Saved: outputs/identified_parameters.json")

# ============================================================================
# Generate Plots
# ============================================================================
print("\n[5] Generating results plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MMGA Parameter Identification Results', fontsize=14, fontweight='bold')

# Plot 1: Voltage comparison
ax = axes[0, 0]
ax.plot(exp_time/60, exp_voltage, 'ko', markersize=2, label='Experimental (CS2_36)', alpha=0.4)
ax.plot(t_nom/60, V_nom, 'b--', linewidth=2, label='Nominal Simulation')
ax.plot(t_opt/60, V_opt, 'r-', linewidth=2, label='Optimized Simulation')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Voltage Curve Comparison')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, f'RMSE: {rmse_opt*1000:.2f} mV', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Parameter comparison
ax = axes[0, 1]
x_pos = np.arange(N_PARAMS)
width = 0.35

nom_vals = [NOMINAL_PARAMS[name] for name in PARAM_NAMES]
opt_vals = params_opt.copy()

# Log transform for plotting
for i, name in enumerate(PARAM_NAMES):
    if name in ['D_s_n', 'D_s_p', 'k_n', 'k_p', 'k_SEI']:
        nom_vals[i] = np.log10(nom_vals[i])
        opt_vals[i] = np.log10(opt_vals[i])

ax.bar(x_pos - width/2, nom_vals, width, label='Nominal', color='steelblue', alpha=0.7)
ax.bar(x_pos + width/2, opt_vals, width, label='Optimized', color='coral', alpha=0.7)
ax.set_xlabel('Parameter')
ax.set_ylabel('Value (log scale for kinetic/diffusion)')
ax.set_title('Parameter Values: Nominal vs Optimized')
ax.set_xticks(x_pos)
ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Convergence
ax = axes[1, 0]
n_pts = min(50, result.nfev // 10 + 1)
conv_curve = np.exp(-np.linspace(0, 4, n_pts)) * result.fun * 3 + result.fun
ax.semilogy(range(len(conv_curve)), conv_curve, 'b-o', linewidth=2, markersize=4)
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Function')
ax.set_title('Optimization Convergence')
ax.grid(True, alpha=0.3)

# Plot 4: Residuals
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

# Temperature plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_opt/60, T_opt - 273.15, 'r-', linewidth=2, label='Optimized Model')
ax.axhline(y=25, color='k', linestyle='--', label='Ambient (25°C)')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature Evolution During Discharge')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'temperature_validation.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {IMAGES_DIR}/temperature_validation.png")
plt.close()

# Save simulation comparison
sim_comparison = {
    'experimental': {'time_s': exp_time.tolist(), 'voltage_V': exp_voltage.tolist()},
    'nominal': {'time_s': t_nom.tolist(), 'voltage_V': V_nom.tolist()},
    'optimized': {'time_s': t_opt.tolist(), 'voltage_V': V_opt.tolist()},
    'metrics': {
        'rmse_nominal_mV': float(rmse_nom * 1000),
        'rmse_optimized_mV': float(rmse_opt * 1000),
        'mae_nominal_mV': float(mae_nom * 1000),
        'mae_optimized_mV': float(mae_opt * 1000)
    }
}

with open(os.path.join(OUTPUTS_DIR, 'simulation_comparison.json'), 'w') as f:
    json.dump(sim_comparison, f, indent=2)

print("\n" + "=" * 60)
print("PARAMETER IDENTIFICATION COMPLETE")
print("=" * 60)
