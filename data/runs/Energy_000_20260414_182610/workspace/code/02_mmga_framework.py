"""
MMGA Framework: Meta-Model Genetic Algorithm for Li-ion Battery Parameter Identification
Implements:
  1. Simplified Single Particle Model (SPM) with SEI aging and thermal coupling
  2. Latin Hypercube Sampling for parameter space exploration
  3. ANN meta-model training
  4. Genetic Algorithm optimization with ANN surrogate
  5. Validation against experimental data
"""
import numpy as np
import pandas as pd
import scipy.io
from scipy.interpolate import interp1d
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, time, warnings
warnings.filterwarnings('ignore')

os.makedirs('report/images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('code', exist_ok=True)

# ============================================================
# Physical Constants
# ============================================================
F = 96485.0      # Faraday constant, C/mol
R_GAS = 8.314    # Gas constant, J/(mol·K)

# ============================================================
# 1. Simplified Single Particle Model (SPM) with SEI & Thermal
# ============================================================
class SPMModel:
    """
    Simplified Single Particle Model for Li-ion battery.
    Parameters to identify:
      - R_p_pos: cathode particle radius (m)
      - R_p_neg: anode particle radius (m)
      - D_s_pos: cathode solid diffusivity (m^2/s)
      - D_s_neg: anode solid diffusivity (m^2/s)
      - k_0_pos: cathode reaction rate constant (m^2.5 mol^-0.5 s^-1)
      - k_0_neg: anode reaction rate constant (m^2.5 mol^-0.5 s^-1)
      - eps_s_pos: cathode active material volume fraction
      - eps_s_neg: anode active material volume fraction
      - c_s_max_pos: max solid concentration cathode (mol/m^3)
      - c_s_max_neg: max solid concentration anode (mol/m^3)
      - k_SEI: SEI growth rate constant (m/s)
      - D_EC: EC diffusivity in SEI (m^2/s)
      - h_thermal: heat transfer coefficient (W/(m^2·K))
      - c_e0: initial electrolyte concentration (mol/m^3)
    """
    
    # Default parameter values (literature-based for NCM/graphite 18650)
    DEFAULT_PARAMS = {
        'R_p_pos': 5e-6,       # 5 um
        'R_p_neg': 5e-6,       # 5 um
        'D_s_pos': 1e-14,      # 1e-14 m^2/s
        'D_s_neg': 3.9e-14,    # 3.9e-14 m^2/s
        'k_0_pos': 2.334e-11,  # reaction rate
        'k_0_neg': 6.667e-11,  # reaction rate
        'eps_s_pos': 0.5,      # volume fraction
        'eps_s_neg': 0.47,     # volume fraction
        'c_s_max_pos': 51554,  # mol/m^3
        'c_s_max_neg': 30555,  # mol/m^3
        'k_SEI': 1e-14,        # SEI growth rate
        'D_EC': 2e-18,         # EC diffusivity in SEI
        'h_thermal': 10.0,     # heat transfer coeff
        'c_e0': 1200.0,        # initial electrolyte conc
    }
    
    # Parameter bounds for LHS sampling (from literature)
    PARAM_BOUNDS = {
        'R_p_pos': (1e-6, 11e-6),
        'R_p_neg': (1e-6, 12e-6),
        'D_s_pos': (1e-15, 1e-13),
        'D_s_neg': (1e-15, 1e-13),
        'k_0_pos': (1e-12, 1e-10),
        'k_0_neg': (1e-12, 1e-10),
        'eps_s_pos': (0.35, 0.6),
        'eps_s_neg': (0.4, 0.6),
        'c_s_max_pos': (45000, 55000),
        'c_s_max_neg': (28000, 33000),
        'k_SEI': (1e-16, 1e-12),
        'D_EC': (1e-19, 1e-16),
        'h_thermal': (5.0, 50.0),
        'c_e0': (1000.0, 1500.0),
    }
    
    PARAM_NAMES = list(DEFAULT_PARAMS.keys())
    
    def __init__(self, params=None):
        self.p = self.DEFAULT_PARAMS.copy()
        if params:
            self.p.update(params)
    
    def equilibrium_potential_pos(self, sto):
        """Cathode OCP (NCM) - polynomial fit"""
        x = np.clip(sto, 0.01, 0.99)
        U = (3.4 + 0.5*np.tanh(20*(0.5-x)) + 0.3*np.tanh(15*(x-0.15)) 
             + 0.2*np.tanh(10*(0.85-x)) - 0.05*np.tanh(5*(x-0.5)))
        return U
    
    def equilibrium_potential_neg(self, sto):
        """Anode OCP (Graphite) - polynomial fit"""
        y = np.clip(sto, 0.01, 0.99)
        U = (0.1 + 0.8*np.exp(-30*y) + 0.3*np.exp(-5*(1-y)) 
             - 0.02*np.tanh(10*(y-0.5)) + 0.01*np.tanh(20*(y-0.8)))
        return U
    
    def simulate_discharge(self, current_density, dt, n_steps, T_ambient=298.15, 
                           soc_init_pos=0.5, soc_init_neg=0.8, cycle_number=1):
        """
        Simulate constant-current discharge.
        Returns: time, voltage, temperature, capacity
        """
        p = self.p
        A_particle_pos = 3 * p['eps_s_pos'] / p['R_p_pos']
        A_particle_neg = 3 * p['eps_s_neg'] / p['R_p_neg']
        
        # Initial concentrations
        c_s_avg_pos = soc_init_pos * p['c_s_max_pos']
        c_s_avg_neg = soc_init_neg * p['c_s_max_neg']
        c_s_surf_pos = c_s_avg_pos
        c_s_surf_neg = c_s_avg_neg
        
        # SEI resistance (grows with cycle)
        delta_SEI = p['k_SEI'] * np.sqrt(cycle_number) * 1e6  # simplified growth
        R_SEI = delta_SEI / 1e-6  # simplified SEI resistance
        
        T = T_ambient
        time_arr = np.zeros(n_steps)
        voltage_arr = np.zeros(n_steps)
        temp_arr = np.zeros(n_steps)
        capacity_arr = np.zeros(n_steps)
        
        total_capacity = 0.0
        
        for i in range(n_steps):
            # Stoichiometries
            sto_pos = c_s_surf_pos / p['c_s_max_pos']
            sto_neg = c_s_surf_neg / p['c_s_max_neg']
            
            # OCP
            U_pos = self.equilibrium_potential_pos(sto_pos)
            U_neg = self.equilibrium_potential_neg(sto_neg)
            
            # Exchange current densities
            i_0_pos = p['k_0_pos'] * F * np.sqrt(
                c_s_surf_pos * (p['c_s_max_pos'] - c_s_surf_pos) * p['c_e0'])
            i_0_neg = p['k_0_neg'] * F * np.sqrt(
                c_s_surf_neg * (p['c_s_max_neg'] - c_s_surf_neg) * p['c_e0'])
            
            # Overpotentials (simplified Butler-Volmer)
            eta_pos = (R_GAS * T / (0.5 * F)) * np.arcsinh(
                current_density / (2 * i_0_pos + 1e-20))
            eta_neg = (R_GAS * T / (0.5 * F)) * np.arcsinh(
                -current_density / (2 * i_0_neg + 1e-20))
            
            # Terminal voltage
            V = U_pos - U_neg + eta_pos + eta_neg - current_density * R_SEI
            V = max(V, 2.0)  # cutoff voltage
            
            # Update solid concentrations
            j_pos = current_density / (F * A_particle_pos + 1e-20)
            j_neg = -current_density / (F * A_particle_neg + 1e-20)
            
            # Simplified diffusion (lumped model)
            tau_diff_pos = p['R_p_pos']**2 / (15 * p['D_s_pos'] + 1e-30)
            tau_diff_neg = p['R_p_neg']**2 / (15 * p['D_s_neg'] + 1e-30)
            
            c_s_avg_pos += -j_pos * A_particle_pos * dt / (F + 1e-20)
            c_s_avg_neg += -j_neg * A_particle_neg * dt / (F + 1e-20)
            
            # Surface concentration from diffusion limitation
            c_s_surf_pos = c_s_avg_pos - j_pos * p['R_p_pos'] / (3 * p['D_s_pos'] * p['c_s_max_pos'] + 1e-30) * (p['c_s_max_pos'] * 0.01)
            c_s_surf_neg = c_s_avg_neg - j_neg * p['R_p_neg'] / (3 * p['D_s_neg'] * p['c_s_max_neg'] + 1e-30) * (p['c_s_max_neg'] * 0.01)
            
            c_s_surf_pos = np.clip(c_s_surf_pos, 0.01*p['c_s_max_pos'], 0.99*p['c_s_max_pos'])
            c_s_surf_neg = np.clip(c_s_surf_neg, 0.01*p['c_s_max_neg'], 0.99*p['c_s_max_neg'])
            
            # Thermal model
            Q_rxn = abs(current_density) * (eta_pos - eta_neg)  # reaction heat
            Q_ohmic = current_density**2 * R_SEI  # ohmic heat
            Q_total = Q_rxn + Q_ohmic
            dT = (Q_total - p['h_thermal'] * (T - T_ambient)) * dt / 1000.0  # simplified thermal mass
            T = T + dT
            T = np.clip(T, T_ambient, T_ambient + 50)
            
            # Capacity
            total_capacity += abs(current_density) * dt / 3600.0  # Ah/m^2
            
            time_arr[i] = i * dt
            voltage_arr[i] = V
            temp_arr[i] = T - 273.15 if T > 100 else T  # convert to Celsius if in Kelvin
            capacity_arr[i] = total_capacity
            
            # Stop at cutoff voltage
            if V <= 2.5:
                time_arr = time_arr[:i+1]
                voltage_arr = voltage_arr[:i+1]
                temp_arr = temp_arr[:i+1]
                capacity_arr = capacity_arr[:i+1]
                break
        
        return time_arr, voltage_arr, temp_arr, capacity_arr


# ============================================================
# 2. Latin Hypercube Sampling
# ============================================================
def generate_lhs_samples(n_samples, param_names, param_bounds, seed=42):
    """Generate LHS samples in parameter space."""
    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)
    unit_samples = sampler.random(n=n_samples)
    
    samples = np.zeros_like(unit_samples)
    for i, name in enumerate(param_names):
        lo, hi = param_bounds[name]
        samples[:, i] = lo + unit_samples[:, i] * (hi - lo)
    
    return samples


# ============================================================
# 3. Generate Training Data for ANN
# ============================================================
def generate_training_data(n_samples=200, seed=42):
    """Generate training data by running SPM with LHS samples."""
    print(f"Generating {n_samples} training samples via LHS...")
    
    param_names = SPMModel.PARAM_NAMES
    param_bounds = SPMModel.PARAM_BOUNDS
    
    samples = generate_lhs_samples(n_samples, param_names, param_bounds, seed)
    
    # Reference discharge conditions (1C for 18650 NCM cell ~2Ah)
    current_density = 2.0  # A/m^2 (simplified)
    dt = 10.0  # seconds
    n_steps = 360  # 1 hour max
    
    # Fixed reference points for feature extraction
    n_features = 50  # voltage curve sampled at 50 points
    
    X_params = []  # parameter vectors
    Y_features = []  # voltage curve features
    
    for i in range(n_samples):
        if (i+1) % 50 == 0:
            print(f"  Sample {i+1}/{n_samples}")
        
        params_dict = {name: samples[i, j] for j, name in enumerate(param_names)}
        model = SPMModel(params_dict)
        
        try:
            t, V, T_cap, cap = model.simulate_discharge(
                current_density, dt, n_steps, T_ambient=298.15)
            
            if len(V) < 10:
                continue
            
            # Resample voltage curve to fixed number of points
            cap_normalized = cap / (cap[-1] + 1e-10)
            V_resampled = np.interp(np.linspace(0, 1, n_features), cap_normalized, V)
            
            # Temperature features (start, mid, end)
            T_resampled = np.interp(np.linspace(0, 1, n_features), 
                                     np.linspace(0, 1, len(T_cap)), T_cap)
            
            # Combined features: voltage curve + temperature curve + capacity
            features = np.concatenate([V_resampled, T_resampled, [cap[-1]]])
            
            X_params.append(samples[i, :])
            Y_features.append(features)
        except Exception as e:
            continue
    
    X_params = np.array(X_params)
    Y_features = np.array(Y_features)
    
    print(f"  Generated {len(X_params)} valid samples")
    return X_params, Y_features, param_names


# ============================================================
# 4. ANN Meta-Model
# ============================================================
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class ANNMetaModel(nn.Module):
    """ANN that maps parameters -> voltage/temperature/capacity features."""
    
    def __init__(self, n_input, n_output, hidden_sizes=[128, 256, 128]):
        super().__init__()
        layers = []
        prev = n_input
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, n_output))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def train_ann(X, Y, epochs=500, lr=1e-3, batch_size=32):
    """Train ANN meta-model."""
    print("Training ANN meta-model...")
    
    # Normalize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    Y_tensor = torch.FloatTensor(Y_scaled)
    
    # Split train/val
    n_train = int(0.8 * len(X))
    indices = torch.randperm(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, Y_train = X_tensor[train_idx], Y_tensor[train_idx]
    X_val, Y_val = X_tensor[val_idx], Y_tensor[val_idx]
    
    # Model
    model = ANNMetaModel(X.shape[1], Y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            pred = model(X_train[idx])
            loss = criterion(pred, Y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()
        
        train_losses.append(epoch_loss / n_batches)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_losses[-1]:.6f}, val_loss={val_loss:.6f}")
    
    return model, scaler_X, scaler_Y, train_losses, val_losses


# ============================================================
# 5. Genetic Algorithm with ANN Surrogate
# ============================================================
class GeneticAlgorithm:
    """GA for parameter identification using ANN surrogate."""
    
    def __init__(self, ann_model, scaler_X, scaler_Y, param_names, param_bounds,
                 target_features, pop_size=100, n_gen=200, crossover_rate=0.8, 
                 mutation_rate=0.1, elite_frac=0.1):
        self.ann = ann_model
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.target_features = target_features
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_elite = max(2, int(elite_frac * pop_size))
        self.n_params = len(param_names)
    
    def fitness(self, params_batch):
        """Evaluate fitness using ANN surrogate."""
        params_scaled = self.scaler_X.transform(params_batch)
        with torch.no_grad():
            pred_scaled = self.ann(torch.FloatTensor(params_scaled)).numpy()
        pred = self.scaler_Y.inverse_transform(pred_scaled)
        
        # MSE between predicted features and target
        errors = np.mean((pred - self.target_features)**2, axis=1)
        return 1.0 / (1.0 + errors)  # higher is better
    
    def run(self):
        """Run GA optimization."""
        print(f"Running GA: pop={self.pop_size}, gens={self.n_gen}")
        
        # Initialize population with LHS
        pop = generate_lhs_samples(self.pop_size, self.param_names, self.param_bounds, seed=123)
        
        best_fitness_history = []
        best_params_history = []
        
        for gen in range(self.n_gen):
            # Evaluate fitness
            fit = self.fitness(pop)
            best_idx = np.argmax(fit)
            
            best_fitness_history.append(fit[best_idx])
            best_params_history.append(pop[best_idx].copy())
            
            if (gen + 1) % 50 == 0:
                print(f"  Gen {gen+1}: best_fitness={fit[best_idx]:.6f}, "
                      f"mean_fitness={np.mean(fit):.6f}")
            
            # Selection (tournament)
            new_pop = []
            
            # Elitism
            elite_idx = np.argsort(fit)[-self.n_elite:]
            for idx in elite_idx:
                new_pop.append(pop[idx].copy())
            
            # Crossover and mutation
            while len(new_pop) < self.pop_size:
                # Tournament selection
                i1, i2 = np.random.randint(0, self.pop_size, 2)
                parent1 = pop[i1] if fit[i1] > fit[i2] else pop[i2]
                i1, i2 = np.random.randint(0, self.pop_size, 2)
                parent2 = pop[i1] if fit[i1] > fit[i2] else pop[i2]
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    alpha = np.random.random(self.n_params)
                    child = alpha * parent1 + (1 - alpha) * parent2
                else:
                    child = parent1.copy()
                
                # Mutation
                for j in range(self.n_params):
                    if np.random.random() < self.mutation_rate:
                        lo, hi = self.param_bounds[self.param_names[j]]
                        child[j] += np.random.normal(0, (hi - lo) * 0.1)
                        child[j] = np.clip(child[j], lo, hi)
                
                new_pop.append(child)
            
            pop = np.array(new_pop[:self.pop_size])
        
        # Final evaluation
        fit = self.fitness(pop)
        best_idx = np.argmax(fit)
        
        return pop[best_idx], fit[best_idx], best_fitness_history, best_params_history


# ============================================================
# 6. Main Execution
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("MMGA Framework for Li-ion Battery Parameter Identification")
    print("="*60)
    
    # Step 1: Generate training data
    t0 = time.time()
    X_params, Y_features, param_names = generate_training_data(n_samples=300, seed=42)
    t1 = time.time()
    print(f"Training data generation took {t1-t0:.1f}s")
    
    # Save training data
    np.save('outputs/X_params.npy', X_params)
    np.save('outputs/Y_features.npy', Y_features)
    
    # Step 2: Train ANN meta-model
    t2 = time.time()
    ann_model, scaler_X, scaler_Y, train_losses, val_losses = train_ann(
        X_params, Y_features, epochs=600, lr=1e-3)
    t3 = time.time()
    print(f"ANN training took {t3-t2:.1f}s")
    
    # Save model
    torch.save(ann_model.state_dict(), 'outputs/ann_model.pth')
    import pickle
    with open('outputs/scalers.pkl', 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_Y': scaler_Y}, f)
    
    # Step 3: Extract target features from CS2_36 experimental data
    print("\nExtracting target features from CS2_36 experimental data...")
    cs2_file = 'data/CS2_36/CS2_36_1_10_11.xlsx'
    df = pd.read_excel(cs2_file, sheet_name='Channel_1-009')
    
    # Get first discharge cycle
    discharge_data = df[(df['Current(A)'] < -0.1) & (df['Cycle_Index'] == df['Cycle_Index'].min())]
    if len(discharge_data) == 0:
        # Try finding any discharge
        for cyc in df['Cycle_Index'].unique():
            cyc_data = df[(df['Current(A)'] < -0.1) & (df['Cycle_Index'] == cyc)]
            if len(cyc_data) > 10:
                discharge_data = cyc_data
                break
    
    V_exp = discharge_data['Voltage(V)'].values
    cap_exp = discharge_data['Discharge_Capacity(Ah)'].values
    
    # Normalize and resample
    n_features = 50
    cap_norm = cap_exp / (cap_exp[-1] + 1e-10)
    V_exp_resampled = np.interp(np.linspace(0, 1, n_features), cap_norm, V_exp)
    
    # Estimate temperature (not available in CS2_36, use flat 25C)
    T_exp_resampled = np.full(n_features, 25.0)
    
    target_features = np.concatenate([V_exp_resampled, T_exp_resampled, [cap_exp[-1]]])
    print(f"  Target capacity: {cap_exp[-1]:.3f} Ah")
    
    # Step 4: Run GA optimization
    t4 = time.time()
    ga = GeneticAlgorithm(
        ann_model, scaler_X, scaler_Y, param_names, SPMModel.PARAM_BOUNDS,
        target_features, pop_size=150, n_gen=300, crossover_rate=0.85, mutation_rate=0.15
    )
    best_params, best_fit, fitness_history, params_history = ga.run()
    t5 = time.time()
    print(f"\nGA optimization took {t5-t4:.1f}s")
    
    # Step 5: Extract identified parameters
    identified_params = {name: best_params[i] for i, name in enumerate(param_names)}
    print("\nIdentified Parameters:")
    for name, val in identified_params.items():
        default = SPMModel.DEFAULT_PARAMS[name]
        ratio = val / default if default != 0 else float('inf')
        print(f"  {name}: {val:.6e} (default: {default:.6e}, ratio: {ratio:.3f})")
    
    # Step 6: Validate with identified parameters
    print("\nValidating with identified parameters...")
    model_identified = SPMModel(identified_params)
    t_sim, V_sim, T_sim, cap_sim = model_identified.simulate_discharge(
        2.0, 10.0, 360, T_ambient=298.15)
    
    # Also simulate with default parameters for comparison
    model_default = SPMModel()
    t_def, V_def, T_def, cap_def = model_default.simulate_discharge(
        2.0, 10.0, 360, T_ambient=298.15)
    
    # Step 7: Generate all figures
    print("\nGenerating figures...")
    
    # Figure 3: ANN Training Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Training Loss', linewidth=1.5)
    ax.plot(val_losses, label='Validation Loss', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('ANN Meta-Model Training Convergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig3_ann_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_ann_training.png")
    
    # Figure 4: GA Convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fitness_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.set_title('Genetic Algorithm Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig4_ga_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_ga_convergence.png")
    
    # Figure 5: Voltage Comparison (Main Result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Voltage vs capacity
    axes[0].plot(cap_exp, V_exp, 'ko', markersize=3, label='Experimental (CS2_36)', alpha=0.7)
    axes[0].plot(cap_sim, V_sim, 'r-', linewidth=2, label='MMGA Identified')
    axes[0].plot(cap_def, V_def, 'b--', linewidth=1.5, label='Default Parameters', alpha=0.7)
    axes[0].set_xlabel('Discharge Capacity (Ah)', fontsize=12)
    axes[0].set_ylabel('Voltage (V)', fontsize=12)
    axes[0].set_title('Discharge Curve Comparison', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Voltage error
    # Interpolate to common capacity axis
    cap_common = np.linspace(0, min(cap_exp[-1], cap_sim[-1]), 100)
    V_exp_interp = np.interp(cap_common, cap_exp, V_exp)
    V_sim_interp = np.interp(cap_common, cap_sim, V_sim)
    V_def_interp = np.interp(cap_common, cap_def, V_def)
    
    error_mmga = np.abs(V_exp_interp - V_sim_interp) * 1000  # mV
    error_default = np.abs(V_exp_interp - V_def_interp) * 1000  # mV
    
    axes[1].plot(cap_common, error_mmga, 'r-', linewidth=1.5, label='MMGA Error')
    axes[1].plot(cap_common, error_default, 'b--', linewidth=1.5, label='Default Error')
    axes[1].set_xlabel('Discharge Capacity (Ah)', fontsize=12)
    axes[1].set_ylabel('Voltage Error (mV)', fontsize=12)
    axes[1].set_title('Voltage Prediction Error', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig5_voltage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig5_voltage_comparison.png")
    
    # Figure 6: Parameter Identification Results (Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    param_labels = list(identified_params.keys())
    identified_vals = [identified_params[k] / SPMModel.DEFAULT_PARAMS[k] for k in param_labels]
    x = np.arange(len(param_labels))
    
    bars = ax.bar(x, identified_vals, color=['#2ecc71' if 0.8 < v < 1.2 else '#e74c3c' for v in identified_vals],
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Default (ratio=1)')
    ax.axhline(y=1.1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Identified / Default Ratio', fontsize=12)
    ax.set_title('Parameter Identification Results (MMGA)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('report/images/fig6_parameter_ratios.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig6_parameter_ratios.png")
    
    # Figure 7: Temperature Profile Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cap_sim, T_sim, 'r-', linewidth=2, label='MMGA (Simulated)')
    ax.plot(cap_def, T_def, 'b--', linewidth=1.5, label='Default (Simulated)')
    ax.axhline(y=25, color='gray', linestyle=':', label='Ambient (25°C)')
    ax.set_xlabel('Discharge Capacity (Ah)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Rise During Discharge', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig7_temperature.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig7_temperature.png")
    
    # Figure 8: Multi-battery validation (NASA)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, bat_name in enumerate(['B0005', 'B0006', 'B0007', 'B0018']):
        ax = axes[idx // 2, idx % 2]
        mat = scipy.io.loadmat(
            f'data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/{bat_name}.mat',
            squeeze_me=True, struct_as_record=False)
        bat = mat[bat_name]
        
        # Plot first few discharge cycles
        dc_count = 0
        for c in bat.cycle:
            if c.type == 'discharge' and dc_count < 5:
                d = c.data
                V = np.array(d.Voltage_measured).flatten()
                I = np.array(d.Current_measured).flatten()
                # Compute capacity from current integration
                dt_est = 1.0  # assume 1s sampling
                cap_cum = np.cumsum(np.abs(I)) * dt_est / 3600.0
                ax.plot(cap_cum, V, linewidth=1, label=f"Cycle {c.index}" if hasattr(c, 'index') else f"#{dc_count}")
                dc_count += 1
        
        ax.set_xlabel('Capacity (Ah)', fontsize=10)
        ax.set_ylabel('Voltage (V)', fontsize=10)
        ax.set_title(f'{bat_name} Discharge Curves', fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/images/fig8_nasa_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig8_nasa_validation.png")
    
    # Load Oxford data
    mat = scipy.io.loadmat('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat', squeeze_me=True, struct_as_record=False)
    oxford = mat['ExampleDC_C1']
    oxford_charge = {
        'time': np.array(oxford.ch.t).flatten(),
        'voltage': np.array(oxford.ch.v).flatten(),
        'current': np.array(oxford.ch.i).flatten(),
        'capacity': np.array(oxford.ch.q).flatten(),
    }
    oxford_discharge = {
        'time': np.array(oxford.dc.t).flatten(),
        'voltage': np.array(oxford.dc.v).flatten(),
        'current': np.array(oxford.dc.i).flatten(),
        'capacity': np.array(oxford.dc.q).flatten(),
    }
    
    # Figure 9: Oxford dynamic profile validation
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(oxford_discharge['time'], oxford_discharge['voltage'], 'b-', linewidth=1, label='Voltage')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12, color='b')
    ax2 = ax1.twinx()
    ax2.plot(oxford_discharge['time'], oxford_discharge['current'], 'r-', linewidth=0.8, alpha=0.7, label='Current')
    ax2.set_ylabel('Current (A)', fontsize=12, color='r')
    ax1.set_title('Oxford Battery: Dynamic Discharge Profile', fontsize=14)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    plt.tight_layout()
    plt.savefig('report/images/fig9_oxford_dynamic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig9_oxford_dynamic.png")
    
    # Figure 10: MMGA Framework Schematic
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    boxes = [
        (0.05, 0.5, 'Experimental\nData\n(CS2_36, NASA,\nOxford)'),
        (0.25, 0.5, 'LHS Parameter\nSampling\n(300 samples)'),
        (0.45, 0.5, 'SPM/ECAT\nSimulation\n(Training Data)'),
        (0.65, 0.5, 'ANN\nMeta-Model\nTraining'),
        (0.85, 0.5, 'GA\nOptimization\n→ Identified\nParameters'),
    ]
    
    colors_box = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
    
    for (x, y, text), color in zip(boxes, colors_box):
        bbox_props = dict(boxstyle="round,pad=0.3", fc=color, ec="black", alpha=0.8)
        ax.text(x, y, text, transform=ax.transAxes, fontsize=11, 
                ha='center', va='center', bbox=bbox_props, fontweight='bold', color='white')
    
    # Arrows
    for i in range(len(boxes)-1):
        ax.annotate('', xy=(boxes[i+1][0]-0.06, 0.5), xytext=(boxes[i][0]+0.06, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Feedback arrow
    ax.annotate('', xy=(0.65, 0.25), xytext=(0.85, 0.25),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0.3'))
    ax.text(0.75, 0.15, 'Fitness Evaluation\n(ANN Surrogate)', transform=ax.transAxes,
            fontsize=9, ha='center', va='center', style='italic', color='gray')
    
    ax.set_title('MMGA Framework: Meta-Model Genetic Algorithm for Battery Parameter Identification',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('report/images/fig10_framework_schematic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig10_framework_schematic.png")
    
    # Save results
    results = {
        'identified_parameters': {k: float(v) for k, v in identified_params.items()},
        'default_parameters': {k: float(v) for k, v in SPMModel.DEFAULT_PARAMS.items()},
        'parameter_ratios': {k: float(identified_params[k] / SPMModel.DEFAULT_PARAMS[k]) for k in param_names},
        'ga_best_fitness': float(best_fit),
        'ann_final_train_loss': float(train_losses[-1]),
        'ann_final_val_loss': float(val_losses[-1]),
        'training_samples': len(X_params),
        'voltage_rmse_mV': float(np.sqrt(np.mean(error_mmga**2))),
        'voltage_max_error_mV': float(np.max(error_mmga)),
        'computation_times': {
            'data_generation_s': t1-t0,
            'ann_training_s': t3-t2,
            'ga_optimization_s': t5-t4,
            'total_s': t5-t0,
        }
    }
    
    with open('outputs/identification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nVoltage RMSE: {results['voltage_rmse_mV']:.1f} mV")
    print(f"Voltage Max Error: {results['voltage_max_error_mV']:.1f} mV")
    print(f"Total computation time: {results['computation_times']['total_s']:.1f}s")
    print("\nResults saved to outputs/identification_results.json")
    print("All figures saved to report/images/")
    print("\nMMGA Framework execution complete!")
