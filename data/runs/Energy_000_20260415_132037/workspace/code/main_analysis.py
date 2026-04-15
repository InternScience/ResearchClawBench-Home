"""
Main Analysis Script for MMGA Parameter Identification Framework
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import BatteryDataLoader
from battery_model import SingleParticleModel, ECATModel
from ann_metamodel import ANNMetaModel, ParameterToCurveMapper
from mmga_optimizer import MMGAOptimizer

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
OUTPUT_DIR = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260415_132037/outputs')
REPORT_IMAGES = Path('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260415_132037/report/images')
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_IMAGES.mkdir(parents=True, exist_ok=True)


def plot_data_overview(data_loader):
    """Generate data overview plots"""
    print("Generating data overview plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # NASA B0005 data
    nasa_data = data_loader.load_nasa_data('B0005')
    
    # Plot discharge cycles
    for i, cycle in enumerate(nasa_data[:5]):
        axes[0, 0].plot(cycle['time'], cycle['voltage'], 
                       label=f'Cycle {i+1}', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].set_title('NASA B0005: Discharge Voltage Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Capacity fade
    capacities = [c['capacity'] for c in nasa_data]
    cycle_numbers = range(len(capacities))
    axes[0, 1].plot(cycle_numbers, capacities, 'b-o', markersize=3)
    axes[0, 1].set_xlabel('Cycle Number')
    axes[0, 1].set_ylabel('Capacity (Ah)')
    axes[0, 1].set_title('NASA B0005: Capacity Fade')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Oxford data
    oxford_data = data_loader.load_oxford_data()
    
    # Charge profile
    ch = oxford_data['charge']
    axes[1, 0].plot(ch['time'], ch['voltage'], 'g-', label='Voltage')
    ax2 = axes[1, 0].twinx()
    ax2.plot(ch['time'], ch['current'], 'r--', label='Current')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Voltage (V)', color='g')
    ax2.set_ylabel('Current (A)', color='r')
    axes[1, 0].set_title('Oxford: CC-CV Charge Profile')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Discharge profile (dynamic)
    dc = oxford_data['discharge']
    axes[1, 1].plot(dc['time'], dc['voltage'], 'b-', label='Voltage')
    ax3 = axes[1, 1].twinx()
    ax3.plot(dc['time'], dc['current'], 'r--', label='Current', alpha=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Voltage (V)', color='b')
    ax3.set_ylabel('Current (A)', color='r')
    axes[1, 1].set_title('Oxford: Dynamic Discharge (Artemis Urban)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Temperature profiles comparison
    axes[2, 0].plot(ch['time'], ch['temperature'], 'g-', label='Charge')
    axes[2, 0].plot(dc['time'], dc['temperature'], 'b-', label='Discharge')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Temperature (°C)')
    axes[2, 0].set_title('Oxford: Temperature Profiles')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Multi-battery comparison
    for bid in ['B0005', 'B0006', 'B0007', 'B0018']:
        try:
            data = data_loader.load_nasa_data(bid)
            caps = [c['capacity'] for c in data]
            axes[2, 1].plot(range(len(caps)), caps, '-o', markersize=2, 
                          label=f'Battery {bid}', alpha=0.7)
        except:
            pass
    axes[2, 1].set_xlabel('Cycle Number')
    axes[2, 1].set_ylabel('Capacity (Ah)')
    axes[2, 1].set_title('NASA: Multi-Battery Capacity Fade')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / 'data_overview.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: data_overview.png")


def plot_lhs_sampling(param_bounds, n_samples=500):
    """Visualize LHS sampling distribution"""
    print("Generating LHS sampling visualization...")
    
    from ann_metamodel import ANNMetaModel
    
    ann = ANNMetaModel()
    samples, names = ann.generate_lhs_samples(param_bounds, n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    # Plot 2D projections of parameter space
    param_pairs = [
        (0, 1, f'{names[0]} vs {names[1]}'),
        (0, 2, f'{names[0]} vs {names[2]}'),
        (1, 2, f'{names[1]} vs {names[2]}'),
        (3, 4, f'{names[3]} vs {names[4]}'),
        (3, 5, f'{names[3]} vs {names[5]}'),
        (4, 5, f'{names[4]} vs {names[5]}'),
    ]
    
    for ax, (i, j, title) in zip(axes, param_pairs):
        ax.scatter(samples[:, i], samples[:, j], c='blue', alpha=0.5, s=10)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latin Hypercube Sampling (LHS) Parameter Space Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / 'lhs_sampling.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'lhs_sampling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: lhs_sampling.png")
    
    return samples, names


def train_ann_metamodel(param_bounds, n_samples=1000):
    """Train ANN meta-model"""
    print("Training ANN meta-model...")
    
    # Initialize components
    spm = SingleParticleModel()
    mapper = ParameterToCurveMapper(spm)
    ann = ANNMetaModel(hidden_layers=(128, 64, 32), max_iter=2000)
    
    # Generate LHS samples
    samples, names = ann.generate_lhs_samples(param_bounds, n_samples)
    
    # Generate training data
    print(f"  Generating {n_samples} simulation samples...")
    X_train, y_train = mapper.generate_training_data(samples, names, 
                                                      current=2.0, T_sim=3600)
    
    # Split train/validation
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, 
                                                 random_state=42)
    
    # Train model
    print("  Training ANN...")
    ann.train(X_tr, y_tr, X_val, y_val)
    
    # Evaluate
    metrics = ann.evaluate(X_val, y_val)
    print(f"  Validation R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Save model
    ann.save(OUTPUT_DIR / 'ann_metamodel.pkl')
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(ann.training_history['loss'])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('ANN Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Prediction vs actual
    y_pred = ann.predict(X_val[:100])
    axes[1].scatter(y_val[:100, 0], y_pred[:, 0], alpha=0.5)
    axes[1].plot([y_val[:, 0].min(), y_val[:, 0].max()], 
                [y_val[:, 0].min(), y_val[:, 0].max()], 'r--')
    axes[1].set_xlabel('Actual Voltage at 100% SOC')
    axes[1].set_ylabel('Predicted Voltage at 100% SOC')
    axes[1].set_title('ANN Prediction Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / 'ann_training.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'ann_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ann_training.png")
    
    return ann, names


def run_mmga_identification(ann_model, param_bounds, param_names, target_features):
    """Run MMGA parameter identification"""
    print("Running MMGA parameter identification...")
    
    mmga = MMGAOptimizer(
        ann_model, 
        param_bounds, 
        pop_size=80,
        n_generations=150,
        crossover_rate=0.8,
        mutation_rate=0.15
    )
    
    best_params, pareto = mmga.optimize(target_features, verbose=True)
    
    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    best_fit = np.array(mmga.history['best_fitness'])
    axes[0].plot(best_fit[:, 0], 'b-', label='Voltage Error')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Voltage RMSE (V)')
    axes[0].set_title('MMGA Convergence: Voltage Error')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(best_fit[:, 1], 'r-', label='Capacity Error')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Capacity Error')
    axes[1].set_title('MMGA Convergence: Capacity Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / 'mmga_convergence.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'mmga_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: mmga_convergence.png")
    
    # Plot Pareto front
    fig, ax = plt.subplots(figsize=(8, 6))
    pareto_obj = pareto['pareto_objectives']
    ax.scatter(pareto_obj[:, 0], pareto_obj[:, 1], c='blue', alpha=0.6, s=50)
    ax.set_xlabel('Voltage RMSE (V)')
    ax.set_ylabel('Capacity Error')
    ax.set_title('MMGA Pareto Front: Multi-Objective Optimization')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / 'pareto_front.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'pareto_front.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: pareto_front.png")
    
    return best_params, mmga


def validate_results(best_params, data_loader):
    """Validate identified parameters against experimental data"""
    print("Validating identified parameters...")
    
    # Load experimental data
    nasa_data = data_loader.load_nasa_data('B0005')
    target_cycle = nasa_data[0]
    
    # Create model with identified parameters
    model = SingleParticleModel(best_params)
    
    # Simulate with similar conditions
    sim_result = model.simulate_discharge(
        current=2.0, 
        T_sim=len(target_cycle['time']),
        dt=1.0
    )
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Voltage comparison
    axes[0, 0].plot(target_cycle['time'], target_cycle['voltage'], 
                   'b-', label='Experimental', linewidth=2)
    axes[0, 0].plot(sim_result['time'], sim_result['voltage'], 
                   'r--', label='Simulated', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].set_title('Voltage Profile Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error analysis
    # Interpolate simulation to match experimental time points
    v_sim_interp = np.interp(target_cycle['time'], 
                             sim_result['time'], 
                             sim_result['voltage'])
    error = target_cycle['voltage'] - v_sim_interp
    
    axes[0, 1].plot(target_cycle['time'], error, 'g-', linewidth=1)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Voltage Error (V)')
    axes[0, 1].set_title('Prediction Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1, 0].hist(error, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Voltage Error (V)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics text
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    
    stats_text = f"""Error Statistics:
    RMSE: {rmse:.4f} V
    MAE: {mae:.4f} V
    Max Error: {max_error:.4f} V
    
    Identified Parameters:
    Rs_p: {best_params.get('Rs_p', 'N/A'):.2e} m
    Rs_n: {best_params.get('Rs_n', 'N/A'):.2e} m
    D_s_p: {best_params.get('D_s_p', 'N/A'):.2e} m²/s
    D_s_n: {best_params.get('D_s_n', 'N/A'):.2e} m²/s
    k_p: {best_params.get('k_p', 'N/A'):.2e}
    k_n: {best_params.get('k_n', 'N/A'):.2e}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                   verticalalignment='center', fontfamily='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Results Summary')
    
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / 'validation_results.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'validation_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: validation_results.png")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error
    }


def plot_sensitivity_analysis(param_bounds, ann_model, param_names):
    """Perform and plot parameter sensitivity analysis"""
    print("Performing sensitivity analysis...")
    
    # Create base parameter set (mean of bounds)
    base_params = []
    for name in param_names:
        low, high = param_bounds[name]
        base_params.append((low + high) / 2)
    base_params = np.array(base_params)
    
    # Predict base output
    base_output = ann_model.predict(base_params.reshape(1, -1))[0]
    base_voltage = base_output[0]  # Voltage at first SOC point
    
    sensitivities = []
    perturbation = 0.1  # 10% perturbation
    
    for i, name in enumerate(param_names):
        # Perturb parameter up and down
        params_up = base_params.copy()
        params_down = base_params.copy()
        
        delta = base_params[i] * perturbation
        params_up[i] += delta
        params_down[i] -= delta
        
        # Predict
        out_up = ann_model.predict(params_up.reshape(1, -1))[0]
        out_down = ann_model.predict(params_down.reshape(1, -1))[0]
        
        # Calculate sensitivity
        voltage_up = out_up[0]
        voltage_down = out_down[0]
        sensitivity = abs(voltage_up - voltage_down) / (2 * delta) * base_params[i]
        sensitivities.append(sensitivity)
    
    # Sort by sensitivity
    sorted_idx = np.argsort(sensitivities)[::-1]
    sorted_names = [param_names[i] for i in sorted_idx]
    sorted_sens = [sensitivities[i] for i in sorted_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_names)))
    bars = ax.barh(sorted_names, sorted_sens, color=colors)
    ax.set_xlabel('Sensitivity (normalized)')
    ax.set_title('Parameter Sensitivity Analysis')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(REPORT_IMAGES / 'sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: sensitivity_analysis.png")
    
    return sorted_names, sorted_sens


def main():
    """Main analysis pipeline"""
    print("=" * 60)
    print("MMGA Parameter Identification Framework for Li-ion Batteries")
    print("=" * 60)
    
    # Initialize data loader
    data_path = '/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260415_132037/data'
    data_loader = BatteryDataLoader(data_path)
    
    # Print data summary
    summary = data_loader.get_experimental_data_summary()
    print("\nDataset Summary:")
    for name, info in summary.items():
        print(f"  {name}: {info}")
    
    # Step 1: Data overview plots
    plot_data_overview(data_loader)
    
    # Define parameter bounds for NMC/Graphite cell
    param_bounds = {
        'Rs_p': (1e-6, 15e-6),      # Cathode particle radius (m)
        'Rs_n': (1e-6, 15e-6),      # Anode particle radius (m)
        'D_s_p': (1e-15, 1e-13),    # Solid diffusion cathode (m^2/s)
        'D_s_n': (1e-15, 1e-13),    # Solid diffusion anode (m^2/s)
        'k_p': (1e-12, 1e-10),      # Reaction rate cathode
        'k_n': (1e-12, 1e-10),      # Reaction rate anode
    }
    
    # Step 2: LHS Sampling visualization
    samples, names = plot_lhs_sampling(param_bounds, n_samples=500)
    
    # Step 3: Train ANN meta-model
    ann_model, param_names = train_ann_metamodel(param_bounds, n_samples=800)
    
    # Step 4: Generate target features from experimental data
    print("\nExtracting target features from experimental data...")
    nasa_data = data_loader.load_nasa_data('B0005')
    features, capacities, soc_points = data_loader.extract_discharge_features(
        nasa_data[:10], soc_points=np.linspace(0, 1, 21)
    )
    target_features = np.concatenate([
        features[0],  # Voltage features
        [capacities[0]],  # Capacity
        [5.0]  # Temperature rise (simplified)
    ])
    print(f"  Target features shape: {target_features.shape}")
    
    # Step 5: Run MMGA optimization
    best_params, mmga = run_mmga_identification(
        ann_model, param_bounds, param_names, target_features
    )
    
    # Step 6: Validate results
    metrics = validate_results(best_params, data_loader)
    
    # Step 7: Sensitivity analysis
    sens_names, sens_values = plot_sensitivity_analysis(
        param_bounds, ann_model, param_names
    )
    
    # Save results
    results = {
        'identified_parameters': best_params,
        'validation_metrics': metrics,
        'parameter_bounds': {k: list(v) for k, v in param_bounds.items()},
        'sensitivity_ranking': {name: float(val) for name, val in zip(sens_names, sens_values)}
    }
    
    with open(OUTPUT_DIR / 'identification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nIdentified Parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value:.4e}")
    print(f"\nValidation Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f} V")
    print(f"  MAE: {metrics['mae']:.4f} V")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {REPORT_IMAGES}")
    
    return results


if __name__ == "__main__":
    main()
