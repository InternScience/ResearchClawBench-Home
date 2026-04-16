"""
Full analysis pipeline for MMGA framework.
"""

import numpy as np
import json
import os
import sys
import time as time_mod
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_pipeline import (load_cs2_36_data, load_nasa_data, load_oxford_data,
                            preprocess_experimental_data, ANNMetaModel, MMGA, BaselineGA)
from ecat_model import (ECATModel, get_default_params, get_parameter_bounds,
                         get_identifiable_params, generate_lhs_samples,
                         extract_features, run_lhs_simulations)

OUTPUTS = '../outputs'
IMAGES = '../report/images'
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(IMAGES, exist_ok=True)

def save_json(data, filename):
    with open(os.path.join(OUTPUTS, filename), 'w') as f:
        json.dump(data, f, indent=2, default=str)

def extract_exp_features(proc_data, n_features=20):
    V = proc_data['voltage']
    T = proc_data['temperature']
    n = len(V)
    t_norm = np.linspace(0, 1, n)
    t_interp = np.linspace(0, 1, n_features)
    V_interp = np.interp(t_interp, t_norm, V)
    T_interp = np.interp(t_interp, t_norm, T)
    features = np.concatenate([V_interp, T_interp,
        [np.mean(V), np.std(V), np.max(T), np.mean(T), proc_data['capacity_final']]])
    return features

def compute_metrics(result, exp_data):
    if not result['success'] or len(result['voltage']) < 5:
        return {'rmse_V': 999, 'mae_V': 999, 'rmse_T': 999, 'mae_T': 999}
    exp_V = exp_data['voltage']
    exp_T = exp_data['temperature']
    n_exp = len(exp_V)
    t_norm = result['time'] / result['time'][-1] if result['time'][-1] > 0 else result['time']
    t_interp = np.linspace(0, 1, n_exp)
    model_V = np.interp(t_interp, t_norm, result['voltage'])
    model_T = np.interp(t_interp, t_norm, result['temperature'])
    rmse_V = np.sqrt(np.mean((model_V - exp_V)**2))
    mae_V = np.mean(np.abs(model_V - exp_V))
    rmse_T = np.sqrt(np.mean((model_T - exp_T)**2))
    mae_T = np.mean(np.abs(model_T - exp_T))
    return {'rmse_V': float(rmse_V), 'mae_V': float(mae_V),
            'rmse_T': float(rmse_T), 'mae_T': float(mae_T)}

# ============================================================
# Step 1: Load data
# ============================================================
print("=" * 60)
print("MMGA Framework - Full Analysis")
print("=" * 60)

print("\n[Step 1] Loading experimental data...")
cs2_data = load_cs2_36_data('../data/CS2_36/CS2_36_1_10_11.xlsx', cycle_idx=1)
cs2_proc = preprocess_experimental_data(cs2_data, n_points=100, cutoff_voltage=2.7)

nasa_data = load_nasa_data('../data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/B0005.mat', 'B0005', 0)
nasa_proc = preprocess_experimental_data(nasa_data, n_points=100, cutoff_voltage=2.7)

oxford_data = load_oxford_data('../data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat')
oxford_proc = preprocess_experimental_data(oxford_data, n_points=100, cutoff_voltage=2.7)

print(f"  CS2_36: V={cs2_proc['voltage'].min():.3f}-{cs2_proc['voltage'].max():.3f}, Cap={cs2_proc['capacity_final']:.4f} Ah")
print(f"  NASA: V={nasa_proc['voltage'].min():.3f}-{nasa_proc['voltage'].max():.3f}, Cap={nasa_proc['capacity_final']:.4f} Ah")
print(f"  Oxford: V={oxford_proc['voltage'].min():.3f}-{oxford_proc['voltage'].max():.3f}, Cap={oxford_proc['capacity_final']:.4f} Ah")

# ============================================================
# Step 2: LHS sampling and simulations
# ============================================================
print("\n[Step 2] Generating LHS samples and running ECAT simulations...")
param_bounds = get_parameter_bounds()
param_names = get_identifiable_params()
n_lhs = 300

# Use CS2_36 conditions for LHS
I_cs2 = 1.1
t_end_cs2 = 3600
T_amb_cs2 = 298.15
A_cell_cs2 = 0.030  # Match CS2_36 capacity

samples, pnames = generate_lhs_samples(n_lhs, param_bounds, seed=42)

# Run LHS simulations with adjusted A_cell
default = get_default_params()
default['A_cell'] = A_cell_cs2

X_lhs = np.zeros((n_lhs, len(pnames)))
Y_list = []
valid_mask = np.ones(n_lhs, dtype=bool)

for i in range(n_lhs):
    params = default.copy()
    for j, name in enumerate(pnames):
        params[name] = samples[name][i]
        X_lhs[i, j] = samples[name][i]
    model = ECATModel(params)
    result = model.simulate_discharge(I_cs2, t_end_cs2, T_amb_cs2, 100, V_cutoff=2.5)
    features = extract_features(result)
    if features is not None:
        Y_list.append(features)
    else:
        Y_list.append(np.zeros(45))
        valid_mask[i] = False

X_valid = X_lhs[valid_mask]
Y_valid = np.array(Y_list)
n_valid = np.sum(valid_mask)
print(f"  Valid simulations: {n_valid}/{n_lhs}")

np.savez(os.path.join(OUTPUTS, 'lhs_data.npz'), X=X_valid, Y=Y_valid,
         param_names=pnames, allow_pickle=True)

# ============================================================
# Step 3: Train ANN
# ============================================================
print("\n[Step 3] Training ANN meta-model...")
n_inputs = X_valid.shape[1]
n_outputs = Y_valid.shape[1]

ann = ANNMetaModel(n_inputs, n_outputs, hidden_layers=[128, 64, 32])
train_score = ann.train(X_valid, Y_valid)
print(f"  ANN training R^2: {train_score:.6f}")

# Test set
n_test = 50
test_samples, _ = generate_lhs_samples(n_test, param_bounds, seed=123)
X_test = np.zeros((n_test, len(pnames)))
Y_test_list = []
test_valid = np.ones(n_test, dtype=bool)

for i in range(n_test):
    params = default.copy()
    for j, name in enumerate(pnames):
        params[name] = test_samples[name][i]
        X_test[i, j] = test_samples[name][i]
    model = ECATModel(params)
    result = model.simulate_discharge(I_cs2, t_end_cs2, T_amb_cs2, 100, V_cutoff=2.5)
    features = extract_features(result)
    if features is not None:
        Y_test_list.append(features)
    else:
        Y_test_list.append(np.zeros(45))
        test_valid[i] = False

X_test_v = X_test[test_valid]
Y_test_v = np.array(Y_test_list)[test_valid]

Y_pred = ann.predict(X_test_v)
ann_rmse = np.sqrt(np.mean((Y_pred - Y_test_v)**2))
print(f"  ANN test RMSE: {ann_rmse:.6f}")

save_json({
    'train_r2': float(train_score),
    'test_rmse': float(ann_rmse),
    'n_training_samples': int(n_valid),
    'n_test_samples': int(np.sum(test_valid)),
    'architecture': [128, 64, 32],
}, 'ann_metrics.json')

# ============================================================
# Step 4: MMGA optimization
# ============================================================
print("\n[Step 4] Running MMGA optimization on CS2_36 data...")
cs2_features = extract_exp_features(cs2_proc, n_features=20)

t_start = time_mod.time()
mmga = MMGA(ann, param_bounds, cs2_features,
            population_size=100, n_generations=200,
            crossover_rate=0.8, mutation_rate=0.1,
            elite_fraction=0.1, n_refine=20)

best_params_ann, best_fitness_ann = mmga.run(verbose=True)
mmga_time = time_mod.time() - t_start
print(f"  MMGA completed in {mmga_time:.2f}s, best fitness: {best_fitness_ann:.6f}")

# Refine with full model
print("  Refining with full ECAT model...")
best_params_mmga, best_fitness_mmga = mmga.refine_with_model(
    best_params_ann, I_cs2, t_end_cs2, T_amb_cs2)
print(f"  After refinement: fitness={best_fitness_mmga:.6f}")

# ============================================================
# Step 5: Baseline GA
# ============================================================
print("\n[Step 5] Running Baseline GA for comparison...")
t_start = time_mod.time()
baseline = BaselineGA(param_bounds, cs2_proc, I_cs2, t_end_cs2, T_amb_cs2,
                      population_size=50, n_generations=50)
best_params_baseline, best_fitness_baseline = baseline.run(verbose=True)
baseline_time = time_mod.time() - t_start
print(f"  Baseline GA completed in {baseline_time:.2f}s, best fitness: {best_fitness_baseline:.6f}")
print(f"  Baseline model evaluations: {baseline.n_evals}")

# ============================================================
# Step 6: Validate
# ============================================================
print("\n[Step 6] Validating identified parameters...")

# Build parameter dicts with A_cell for CS2_36
mmga_params = default.copy()
for i, name in enumerate(pnames):
    mmga_params[name] = best_params_mmga[i]

model_mmga = ECATModel(mmga_params)
result_mmga = model_mmga.simulate_discharge(I_cs2, t_end_cs2, T_amb_cs2, 200, V_cutoff=2.5)

baseline_params = default.copy()
for i, name in enumerate(pnames):
    baseline_params[name] = best_params_baseline[i]

model_baseline = ECATModel(baseline_params)
result_baseline = model_baseline.simulate_discharge(I_cs2, t_end_cs2, T_amb_cs2, 200, V_cutoff=2.5)

mmga_metrics_cs2 = compute_metrics(result_mmga, cs2_proc)
baseline_metrics_cs2 = compute_metrics(result_baseline, cs2_proc)

print(f"  MMGA on CS2_36: RMSE_V={mmga_metrics_cs2['rmse_V']*1000:.2f} mV, RMSE_T={mmga_metrics_cs2['rmse_T']:.2f} K")
print(f"  Baseline on CS2_36: RMSE_V={baseline_metrics_cs2['rmse_V']*1000:.2f} mV, RMSE_T={baseline_metrics_cs2['rmse_T']:.2f} K")

# NASA cross-validation
print("\n  Cross-validating on NASA data...")
I_nasa = 2.0
t_end_nasa = 3600
T_amb_nasa = 297.15

mmga_params_nasa = mmga_params.copy()
mmga_params_nasa['A_cell'] = 0.065
result_mmga_nasa = ECATModel(mmga_params_nasa).simulate_discharge(I_nasa, t_end_nasa, T_amb_nasa, 200, V_cutoff=2.5)
mmga_metrics_nasa = compute_metrics(result_mmga_nasa, nasa_proc)

baseline_params_nasa = baseline_params.copy()
baseline_params_nasa['A_cell'] = 0.065
result_baseline_nasa = ECATModel(baseline_params_nasa).simulate_discharge(I_nasa, t_end_nasa, T_amb_nasa, 200, V_cutoff=2.5)
baseline_metrics_nasa = compute_metrics(result_baseline_nasa, nasa_proc)

print(f"  MMGA on NASA: RMSE_V={mmga_metrics_nasa['rmse_V']*1000:.2f} mV")
print(f"  Baseline on NASA: RMSE_V={baseline_metrics_nasa['rmse_V']*1000:.2f} mV")

# ============================================================
# Step 7: Save results
# ============================================================
print("\n[Step 7] Saving results...")

mmga_param_dict = {name: float(best_params_mmga[i]) for i, name in enumerate(pnames)}
baseline_param_dict = {name: float(best_params_baseline[i]) for i, name in enumerate(pnames)}

save_json({
    'mmga_parameters': mmga_param_dict,
    'baseline_parameters': baseline_param_dict,
    'mmga_metrics_cs2': mmga_metrics_cs2,
    'baseline_metrics_cs2': baseline_metrics_cs2,
    'mmga_metrics_nasa': mmga_metrics_nasa,
    'baseline_metrics_nasa': baseline_metrics_nasa,
    'mmga_time_s': float(mmga_time),
    'baseline_time_s': float(baseline_time),
    'baseline_n_evals': int(baseline.n_evals),
    'speedup_factor': float(baseline_time / mmga_time) if mmga_time > 0 else 0,
}, 'identification_results.json')

np.savez(os.path.join(OUTPUTS, 'convergence.npz'),
         mmga_gen=mmga.history['generation'],
         mmga_best=mmga.history['best_fitness'],
         mmga_mean=mmga.history['mean_fitness'],
         baseline_gen=baseline.history['generation'],
         baseline_best=baseline.history['best_fitness'],
         baseline_mean=baseline.history['mean_fitness'])

print("[Step 7] Results saved.")

# ============================================================
# Step 8: Figures
# ============================================================
print("\n[Step 8] Generating figures...")

# Figure 1: Data Overview
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0,0].plot(cs2_proc['time_norm'], cs2_proc['voltage'], 'b-', lw=1.5)
axes[0,0].set_ylabel('Voltage (V)'); axes[0,0].set_title('CS2_36 (CALCE NCM 18650)')
axes[0,0].set_ylim([2.5, 4.3])

axes[0,1].plot(nasa_proc['time_norm'], nasa_proc['voltage'], 'r-', lw=1.5)
axes[0,1].set_ylabel('Voltage (V)'); axes[0,1].set_title('NASA PCoE B0005')
axes[0,1].set_ylim([2.5, 4.3])

axes[0,2].plot(oxford_proc['time_norm'], oxford_proc['voltage'], 'g-', lw=1.5)
axes[0,2].set_ylabel('Voltage (V)'); axes[0,2].set_title('Oxford Battery (Dynamic)')
axes[0,2].set_ylim([2.5, 4.3])

axes[1,0].plot(cs2_proc['time_norm'], cs2_proc['temperature'], 'b-', lw=1.5)
axes[1,0].set_xlabel('Normalized Time'); axes[1,0].set_ylabel('Temperature (K)')

axes[1,1].plot(nasa_proc['time_norm'], nasa_proc['temperature'], 'r-', lw=1.5)
axes[1,1].set_xlabel('Normalized Time'); axes[1,1].set_ylabel('Temperature (K)')

axes[1,2].plot(oxford_proc['time_norm'], oxford_proc['temperature'], 'g-', lw=1.5)
axes[1,2].set_xlabel('Normalized Time'); axes[1,2].set_ylabel('Temperature (K)')

plt.suptitle('Experimental Data Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig1_data_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 1 saved")

# Figure 2: LHS distributions
fig, axes = plt.subplots(3, 4, figsize=(18, 10))
axes = axes.flatten()
for i, name in enumerate(pnames):
    axes[i].hist(X_valid[:, i], bins=30, alpha=0.7, color='steelblue', edgecolor='white')
    axes[i].set_xlabel(name, fontsize=9); axes[i].set_ylabel('Count', fontsize=9)
    axes[i].tick_params(labelsize=8)
for j in range(len(pnames), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('LHS Parameter Distribution (300 samples)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig2_lhs_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 2 saved")

# Figure 3: ANN Performance
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for i in range(min(5, Y_test_v.shape[1])):
    ax.scatter(Y_test_v[:, i], Y_pred[:, i], alpha=0.6, s=20, label=f'Feature {i}')
lims = [min(Y_test_v[:, :5].min(), Y_pred[:, :5].min()),
        max(Y_test_v[:, :5].max(), Y_pred[:, :5].max())]
ax.plot(lims, lims, 'k--', lw=1)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
ax.set_title(f'ANN Prediction (Train R²={train_score:.4f})')
ax.legend(fontsize=7)

residuals = (Y_pred - Y_test_v).flatten()
axes[1].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Count')
axes[1].set_title(f'ANN Residual Distribution (Test RMSE={ann_rmse:.4f})')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig3_ann_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 3 saved")

# Figure 4: Convergence
fig, ax = plt.subplots(figsize=(10, 6))
mmga_best = np.array(mmga.history['best_fitness'])
baseline_best = np.array(baseline.history['best_fitness'])
ax.plot(mmga.history['generation'], -mmga_best, 'b-', lw=2, label='MMGA Best')
ax.plot(baseline.history['generation'], -baseline_best, 'r-', lw=2, label='Baseline GA Best')
ax.set_xlabel('Generation'); ax.set_ylabel('RMSE')
ax.set_title('Convergence Comparison: MMGA vs Baseline GA')
ax.legend(); ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig4_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 4 saved")

# Figure 5: CS2_36 Fitting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(cs2_proc['time_norm'], cs2_proc['voltage'], 'ko', ms=4, label='Experimental')
if result_mmga['success']:
    t_m = result_mmga['time'] / result_mmga['time'][-1] if result_mmga['time'][-1] > 0 else result_mmga['time']
    ax.plot(t_m, result_mmga['voltage'], 'b-', lw=2, label=f'MMGA (RMSE={mmga_metrics_cs2["rmse_V"]*1000:.1f} mV)')
if result_baseline['success']:
    t_b = result_baseline['time'] / result_baseline['time'][-1] if result_baseline['time'][-1] > 0 else result_baseline['time']
    ax.plot(t_b, result_baseline['voltage'], 'r--', lw=2, label=f'Baseline (RMSE={baseline_metrics_cs2["rmse_V"]*1000:.1f} mV)')
ax.set_xlabel('Normalized Time'); ax.set_ylabel('Voltage (V)')
ax.set_title('CS2_36 Voltage Fitting'); ax.legend()

ax = axes[1]
ax.plot(cs2_proc['time_norm'], cs2_proc['temperature'], 'ko', ms=4, label='Experimental')
if result_mmga['success']:
    ax.plot(t_m, result_mmga['temperature'], 'b-', lw=2, label='MMGA')
if result_baseline['success']:
    ax.plot(t_b, result_baseline['temperature'], 'r--', lw=2, label='Baseline')
ax.set_xlabel('Normalized Time'); ax.set_ylabel('Temperature (K)')
ax.set_title('CS2_36 Temperature Profile'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig5_cs2_fitting.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 5 saved")

# Figure 6: NASA Cross-validation
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(nasa_proc['time_norm'], nasa_proc['voltage'], 'ko', ms=4, label='Experimental (NASA B0005)')
if result_mmga_nasa['success']:
    t_n = result_mmga_nasa['time'] / result_mmga_nasa['time'][-1] if result_mmga_nasa['time'][-1] > 0 else result_mmga_nasa['time']
    ax.plot(t_n, result_mmga_nasa['voltage'], 'b-', lw=2, label=f'MMGA (RMSE={mmga_metrics_nasa["rmse_V"]*1000:.1f} mV)')
if result_baseline_nasa['success']:
    t_nb = result_baseline_nasa['time'] / result_baseline_nasa['time'][-1] if result_baseline_nasa['time'][-1] > 0 else result_baseline_nasa['time']
    ax.plot(t_nb, result_baseline_nasa['voltage'], 'r--', lw=2, label=f'Baseline (RMSE={baseline_metrics_nasa["rmse_V"]*1000:.1f} mV)')
ax.set_xlabel('Normalized Time'); ax.set_ylabel('Voltage (V)')
ax.set_title('Cross-Validation: NASA B0005'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig6_nasa_validation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 6 saved")

# Figure 7: Parameters Comparison
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(pnames))
bw = 0.35
mmga_norm = np.zeros(len(pnames))
baseline_norm = np.zeros(len(pnames))
for i, name in enumerate(pnames):
    low, high = param_bounds[name]
    mmga_norm[i] = (best_params_mmga[i] - low) / (high - low) if high > low else 0
    baseline_norm[i] = (best_params_baseline[i] - low) / (high - low) if high > low else 0
ax.bar(x_pos - bw/2, mmga_norm, bw, label='MMGA', color='steelblue', alpha=0.8)
ax.bar(x_pos + bw/2, baseline_norm, bw, label='Baseline GA', color='coral', alpha=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(pnames, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Normalized Value'); ax.set_title('Identified Parameters Comparison')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig7_params_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 7 saved")

# Figure 8: Sensitivity
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_valid, Y_valid[:, 0])
importances = rf.feature_importances_

fig, ax = plt.subplots(figsize=(10, 5))
sorted_idx = np.argsort(importances)
ax.barh(np.array(pnames)[sorted_idx], importances[sorted_idx], color='steelblue', alpha=0.8)
ax.set_xlabel('Feature Importance'); ax.set_title('Parameter Sensitivity (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig8_sensitivity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 8 saved")

save_json({'param_names': pnames, 'importances': [float(x) for x in importances]},
          'sensitivity_results.json')

# Figure 9: Framework diagram
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis('off')
boxes = [
    (1.5, 8.5, 'LHS Sampling\n(300 samples)', '#AEC6CF'),
    (5, 8.5, 'ECAT Model\nSimulations', '#FFDAB9'),
    (8.5, 8.5, 'Feature\nExtraction', '#B0E0B0'),
    (1.5, 6, 'ANN Meta-Model\nTraining', '#AEC6CF'),
    (5, 6, 'GA Optimization\n(ANN surrogate)', '#FFDAB9'),
    (8.5, 6, 'Candidate\nParameters', '#B0E0B0'),
    (1.5, 3.5, 'Full ECAT Model\nRefinement', '#AEC6CF'),
    (5, 3.5, 'Identified\nParameters', '#FFDAB9'),
    (8.5, 3.5, 'Cross-Dataset\nValidation', '#B0E0B0'),
]
for x, y, text, color in boxes:
    rect = plt.Rectangle((x-1.1, y-0.6), 2.6, 1.2, facecolor=color, edgecolor='black', lw=1.5, alpha=0.9, zorder=2)
    ax.add_patch(rect)
    ax.text(x+0.2, y, text, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)
arrows = [(3.7, 8.5, 3.9, 8.5), (6.3, 8.5, 7.4, 8.5),
          (8.7, 7.9, 2.7, 6.6), (3.7, 6, 3.9, 6),
          (6.3, 6, 7.4, 6), (8.7, 5.4, 2.7, 4.1),
          (3.7, 3.5, 3.9, 3.5), (6.3, 3.5, 7.4, 3.5)]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='black', lw=2), zorder=1)
ax.text(0.3, 9.5, 'Phase 1: Offline Training', fontsize=12, fontweight='bold', color='navy')
ax.text(0.3, 7.0, 'Phase 2: Online Optimization', fontsize=12, fontweight='bold', color='darkred')
ax.text(0.3, 4.5, 'Phase 3: Refinement & Validation', fontsize=12, fontweight='bold', color='darkgreen')
ax.set_title('MMGA Framework: ANN Meta-Model Guided Genetic Algorithm', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig9_framework.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 9 saved")

# Figure 10: Oxford validation
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(oxford_proc['time_norm'], oxford_proc['voltage'], 'ko', ms=3, label='Experimental (Oxford)')
oxford_p = mmga_params.copy()
oxford_p['A_cell'] = 0.025
I_ox = oxford_proc['current_mean']
result_ox = ECATModel(oxford_p).simulate_discharge(I_ox, 3600, 313.15, 200, V_cutoff=2.5)
if result_ox['success']:
    t_o = result_ox['time'] / result_ox['time'][-1] if result_ox['time'][-1] > 0 else result_ox['time']
    ax.plot(t_o, result_ox['voltage'], 'b-', lw=2, label='MMGA Model')
ax.set_xlabel('Normalized Time'); ax.set_ylabel('Voltage (V)')
ax.set_title('Generalization Test: Oxford Dynamic Profile'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES, 'fig10_oxford_validation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Fig 10 saved")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
