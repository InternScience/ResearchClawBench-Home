#!/usr/bin/env python3
"""
Main analysis pipeline for MMGA parameter identification.
Runs: data loading -> LHS sampling -> ANN training -> GA optimization -> validation
"""
import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_000_20260416_182016"
sys.path.insert(0, os.path.join(WORKSPACE, 'code'))

from data_loading import load_nasa_data, load_cs2_36_data, load_oxford_data
from ecat_model import ECATModel, get_parameter_bounds, get_identifiable_params
from mmga_framework import (generate_training_data, ANNMetaModel, 
                            GeneticAlgorithm, DirectGA, sensitivity_analysis)

os.makedirs(os.path.join(WORKSPACE, 'outputs'), exist_ok=True)
os.makedirs(os.path.join(WORKSPACE, 'report/images'), exist_ok=True)

# ============================================================
# Step 1: Load experimental data
# ============================================================
print("=" * 60)
print("STEP 1: Loading Experimental Data")
print("=" * 60)

# NASA data
nasa_data = {}
for bid in ['B0005', 'B0006', 'B0007', 'B0018']:
    nasa_data[bid] = load_nasa_data(bid)
    print(f"  {bid}: {len(nasa_data[bid])} discharge cycles")

# CS2_36 data
cs2_data = []
for i in range(4):
    d = load_cs2_36_data(i)
    if d is not None:
        cs2_data.append(d)
        print(f"  CS2_36 file {i}: {len(d['time'])} points")

# Oxford data
oxford_data = load_oxford_data()
print(f"  Oxford: {len(oxford_data['discharge']['voltage'])} discharge points")

# Extract primary reference: CS2_36 first discharge cycle
cs2 = cs2_data[0]
# Find discharge segments (negative current)
discharge_mask = cs2['current'] < -0.01
# Get first complete discharge cycle
cycles = np.unique(cs2['cycle_index'])
print(f"  CS2_36 cycles: {cycles[:5]}...{cycles[-5:]}")

# Extract a clean discharge curve from CS2_36
# Find step with negative current (discharge)
ref_voltage = []
ref_time = []
ref_current = []
for cyc in cycles[1:3]:  # Use cycle 2 as reference
    mask = (cs2['cycle_index'] == cyc) & (cs2['current'] < -0.01)
    if mask.sum() > 10:
        t = cs2['time'][mask]
        v = cs2['voltage'][mask]
        c = cs2['current'][mask]
        ref_time = t - t[0]
        ref_voltage = v
        ref_current = c
        print(f"  Reference cycle {cyc}: {len(v)} points, V=[{v.min():.3f}, {v.max():.3f}], I={c.mean():.3f}A")
        break

if len(ref_voltage) == 0:
    # Fallback: use NASA data
    print("  Using NASA B0005 cycle 0 as reference")
    ref_cycle = nasa_data['B0005'][0]
    ref_voltage = ref_cycle['voltage']
    ref_time = ref_cycle['time']
    ref_current = ref_cycle['current']

ref_voltage = np.array(ref_voltage)
ref_time = np.array(ref_time)
ref_current = np.array(ref_current)

# Save reference data
np.savez(os.path.join(WORKSPACE, 'outputs/reference_data.npz'),
         voltage=ref_voltage, time=ref_time, current=ref_current)

# Extract target features from reference data
I_ref = abs(ref_current.mean())
print(f"\nReference: I={I_ref:.3f}A, duration={ref_time[-1]:.0f}s, V=[{ref_voltage.min():.3f}, {ref_voltage.max():.3f}]")

# Create target feature vector
n_features = 50
cap_ref = I_ref * ref_time / 3600  # Ah
cap_points = np.linspace(0, cap_ref[-1]*0.98, n_features)
v_interp = np.interp(cap_points, cap_ref, ref_voltage)
# Estimate temperature rise (not available in CS2_36, use ~10C as typical)
temp_rise = 10.0
target_features = np.concatenate([v_interp, [cap_ref[-1], temp_rise]])

print(f"Target features: {len(target_features)} values")
print(f"  V range: [{v_interp.min():.3f}, {v_interp.max():.3f}]")
print(f"  Capacity: {cap_ref[-1]:.3f} Ah")

# ============================================================
# Step 2: Generate LHS Training Data
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Generating LHS Training Data")
print("=" * 60)

t_start = time.time()
X_train, Y_train, param_names = generate_training_data(n_samples=800, I_app=I_ref, seed=42)
t_lhs = time.time() - t_start
print(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
print(f"LHS generation time: {t_lhs:.1f}s")

np.savez(os.path.join(WORKSPACE, 'outputs/training_data.npz'),
         X=X_train, Y=Y_train, param_names=param_names)

# ============================================================
# Step 3: Train ANN Meta-Model
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Training ANN Meta-Model")
print("=" * 60)

input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
print(f"ANN architecture: {input_dim} -> [128, 256, 128] -> {output_dim}")

ann = ANNMetaModel(input_dim, output_dim, hidden_layers=[128, 256, 128])
t_start = time.time()
train_losses, val_losses = ann.train(X_train, Y_train, epochs=300, lr=0.001, batch_size=32)
t_ann = time.time() - t_start
print(f"ANN training time: {t_ann:.1f}s")

ann.save(os.path.join(WORKSPACE, 'outputs/ann_model.pt'))

# Save training history
np.savez(os.path.join(WORKSPACE, 'outputs/ann_training_history.npz'),
         train_losses=train_losses, val_losses=val_losses)

# Evaluate ANN accuracy
Y_pred = ann.predict(X_train)
v_rmse = np.sqrt(np.mean((Y_pred[:, :n_features] - Y_train[:, :n_features])**2))
cap_mae = np.mean(np.abs(Y_pred[:, -2] - Y_train[:, -2]))
print(f"ANN training accuracy: V_RMSE={v_rmse*1000:.1f}mV, Cap_MAE={cap_mae*1000:.1f}mAh")

# ============================================================
# Step 4: MMGA Parameter Identification
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: MMGA Parameter Identification")
print("=" * 60)

bounds = get_parameter_bounds()

t_start = time.time()
ga = GeneticAlgorithm(
    ann_model=ann,
    target_features=target_features,
    param_names=param_names,
    bounds=bounds,
    pop_size=150,
    n_generations=300,
    mutation_rate=0.15,
    crossover_rate=0.8,
    elite_frac=0.1
)

best_params_arr, best_fitness, fitness_history, avg_history = ga.run()
t_mmga = time.time() - t_start
print(f"\nMMGA optimization time: {t_mmga:.1f}s")
print(f"Best fitness: {best_fitness:.6f}")

# Convert to parameter dict
identified_params = {}
for j, pname in enumerate(param_names):
    identified_params[pname] = float(best_params_arr[j])
    print(f"  {pname:15s}: {best_params_arr[j]:.6e}")

# Save results
with open(os.path.join(WORKSPACE, 'outputs/identified_params.json'), 'w') as f:
    json.dump(identified_params, f, indent=2)

np.savez(os.path.join(WORKSPACE, 'outputs/ga_history.npz'),
         fitness_history=fitness_history, avg_history=avg_history)

# ============================================================
# Step 5: Direct GA Comparison (smaller scale for timing)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Direct GA Comparison")
print("=" * 60)

# Use NASA B0005 first discharge for comparison
nasa_ref = nasa_data['B0005'][0]
t_start = time.time()
direct_ga = DirectGA(
    target_voltage=nasa_ref['voltage'],
    target_time=nasa_ref['time'],
    I_app=abs(nasa_ref['current'].mean()),
    param_names=param_names,
    bounds=bounds,
    pop_size=20,
    n_generations=30,
    V_cutoff=2.7
)
direct_best, direct_fitness, direct_history = direct_ga.run()
t_direct = time.time() - t_start
print(f"Direct GA time: {t_direct:.1f}s")
print(f"Direct GA best fitness: {direct_fitness:.6f}")

# ============================================================
# Step 6: Validation
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Validation on All Datasets")
print("=" * 60)

# 6a. Validate on CS2_36 reference
model_id = ECATModel(identified_params)
sim_ref = model_id.simulate_cc_discharge(I_app=I_ref, t_end=5000, dt=1.0, V_cutoff=2.5)

# Compute RMSE on reference
from scipy.interpolate import interp1d
common_t = np.linspace(0, min(ref_time[-1], sim_ref['time'][-1]), 200)
v_exp_interp = np.interp(common_t, ref_time, ref_voltage)
v_sim_interp = np.interp(common_t, sim_ref['time'], sim_ref['voltage'])
rmse_cs2 = np.sqrt(np.mean((v_exp_interp - v_sim_interp)**2))
mae_cs2 = np.mean(np.abs(v_exp_interp - v_sim_interp))
print(f"CS2_36 validation: RMSE={rmse_cs2*1000:.1f}mV, MAE={mae_cs2*1000:.1f}mV")

# 6b. Validate on NASA B0005
nasa_ref_cycle = nasa_data['B0005'][0]
I_nasa = abs(nasa_ref_cycle['current'].mean())
model_nasa = ECATModel(identified_params)
model_nasa.params['Q_nom'] = nasa_ref_cycle['capacity']
model_nasa.params['T_amb'] = nasa_ref_cycle['temperature'][0] + 273.15
sim_nasa = model_nasa.simulate_cc_discharge(I_app=I_nasa, t_end=5000, dt=1.0, V_cutoff=2.5)

common_t_nasa = np.linspace(0, min(nasa_ref_cycle['time'][-1], sim_nasa['time'][-1]), 200)
v_exp_nasa = np.interp(common_t_nasa, nasa_ref_cycle['time'], nasa_ref_cycle['voltage'])
v_sim_nasa = np.interp(common_t_nasa, sim_nasa['time'], sim_nasa['voltage'])
rmse_nasa = np.sqrt(np.mean((v_exp_nasa - v_sim_nasa)**2))
mae_nasa = np.mean(np.abs(v_exp_nasa - v_sim_nasa))
print(f"NASA B0005 validation: RMSE={rmse_nasa*1000:.1f}mV, MAE={mae_nasa*1000:.1f}mV")

# 6c. Validate on Oxford dynamic profile
ox_dc = oxford_data['discharge']
model_ox = ECATModel(identified_params)
model_ox.params['Q_nom'] = 0.74  # 740mAh
model_ox.params['T_amb'] = 273.15 + 40  # 40C chamber
model_ox.params['SOC_init'] = 1.0
ox_time = ox_dc['time'] - ox_dc['time'][0]
sim_ox = model_ox.simulate_dynamic(
    current_profile=ox_dc['current'],  # already in mA
    time_profile=ox_time,
    T_init=273.15 + 40,
    Q_nom=0.74
)

rmse_ox = np.sqrt(np.mean((sim_ox['voltage'] - ox_dc['voltage'])**2))
mae_ox = np.mean(np.abs(sim_ox['voltage'] - ox_dc['voltage']))
print(f"Oxford dynamic validation: RMSE={rmse_ox*1000:.1f}mV, MAE={mae_ox*1000:.1f}mV")

# 6d. NASA aging tracking
print("\nNASA B0005 capacity degradation tracking:")
cap_exp = []
cap_sim = []
cycle_nums = []
for idx in range(0, len(nasa_data['B0005']), 10):
    c = nasa_data['B0005'][idx]
    if c['capacity'] is not None and c['capacity'] > 0.5:
        cap_exp.append(c['capacity'])
        cycle_nums.append(idx)
        
        # Simulate with aging
        model_aging = ECATModel(identified_params)
        model_aging.params['Q_nom'] = c['capacity']
        model_aging.params['R_SEI_0'] = identified_params.get('R_SEI_0', 0.005) + idx * 1e-5
        sim_aging = model_aging.simulate_cc_discharge(
            I_app=abs(c['current'].mean()), t_end=5000, dt=5.0, V_cutoff=2.5)
        cap_sim.append(sim_aging['capacity'][-1] if len(sim_aging['capacity']) > 0 else 0)

cap_exp = np.array(cap_exp)
cap_sim = np.array(cap_sim)
cycle_nums = np.array(cycle_nums)

# ============================================================
# Step 7: Sensitivity Analysis
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Parameter Sensitivity Analysis")
print("=" * 60)

sens = sensitivity_analysis(identified_params, I_app=I_ref)
sens_data = {}
for pname, s in sorted(sens.items(), key=lambda x: -x[1]['overall']):
    print(f"  {pname:15s}: V={s['voltage']:.4f}, cap={s['capacity']:.4f}, T={s['temperature']:.4f}")
    sens_data[pname] = s

with open(os.path.join(WORKSPACE, 'outputs/sensitivity_analysis.json'), 'w') as f:
    json.dump(sens_data, f, indent=2, default=float)

# ============================================================
# Step 8: Save all results
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Saving Results")
print("=" * 60)

results = {
    'identified_parameters': identified_params,
    'validation_metrics': {
        'CS2_36': {'RMSE_mV': float(rmse_cs2*1000), 'MAE_mV': float(mae_cs2*1000)},
        'NASA_B0005': {'RMSE_mV': float(rmse_nasa*1000), 'MAE_mV': float(mae_nasa*1000)},
        'Oxford_dynamic': {'RMSE_mV': float(rmse_ox*1000), 'MAE_mV': float(mae_ox*1000)},
    },
    'computation_time': {
        'LHS_generation_s': float(t_lhs),
        'ANN_training_s': float(t_ann),
        'MMGA_optimization_s': float(t_mmga),
        'MMGA_total_s': float(t_lhs + t_ann + t_mmga),
        'DirectGA_s': float(t_direct),
        'speedup_factor': float(t_direct / (t_mmga + 1e-10)),
    },
    'ga_convergence': {
        'final_fitness': float(best_fitness),
        'n_generations': 300,
        'population_size': 150,
    },
    'ann_performance': {
        'voltage_RMSE_mV': float(v_rmse*1000),
        'capacity_MAE_mAh': float(cap_mae*1000),
        'n_training_samples': int(X_train.shape[0]),
    }
}

with open(os.path.join(WORKSPACE, 'outputs/results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save validation data for plotting
np.savez(os.path.join(WORKSPACE, 'outputs/validation_data.npz'),
         # CS2_36
         cs2_time_exp=ref_time, cs2_voltage_exp=ref_voltage,
         cs2_time_sim=sim_ref['time'], cs2_voltage_sim=sim_ref['voltage'],
         cs2_temp_sim=sim_ref['temperature'],
         # NASA
         nasa_time_exp=nasa_ref_cycle['time'], nasa_voltage_exp=nasa_ref_cycle['voltage'],
         nasa_temp_exp=nasa_ref_cycle['temperature'],
         nasa_time_sim=sim_nasa['time'], nasa_voltage_sim=sim_nasa['voltage'],
         nasa_temp_sim=sim_nasa['temperature'],
         # Oxford
         ox_time=ox_time, ox_voltage_exp=ox_dc['voltage'],
         ox_voltage_sim=sim_ox['voltage'], ox_temp_exp=ox_dc['temperature'],
         ox_temp_sim=sim_ox['temperature'],
         # Aging
         aging_cycles=cycle_nums, aging_cap_exp=cap_exp, aging_cap_sim=cap_sim,
         # GA history
         fitness_history=np.array(fitness_history), avg_history=np.array(avg_history),
         direct_history=np.array(direct_history),
         # ANN history
         train_losses=np.array(train_losses), val_losses=np.array(val_losses))

print("\nAll results saved to outputs/")
print(f"\nTotal pipeline time: {t_lhs + t_ann + t_mmga:.1f}s")
print("DONE!")
