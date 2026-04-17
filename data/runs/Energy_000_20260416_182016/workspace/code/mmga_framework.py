"""
MMGA Framework: Meta-Model based Genetic Algorithm
for rapid parameter identification of ECAT model
"""
import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecat_model import ECATModel, get_parameter_bounds, get_identifiable_params

WORKSPACE = "/mnt/shared-storage-user/chenyixin/ResearchClawBench/workspaces/Energy_000_20260416_182016"


# ============================================================
# 1. Latin Hypercube Sampling (LHS)
# ============================================================
def generate_lhs_samples(n_samples=500, seed=42):
    """Generate parameter samples using Latin Hypercube Sampling"""
    from pyDOE2 import lhs
    
    param_names = get_identifiable_params()
    bounds = get_parameter_bounds()
    n_params = len(param_names)
    
    np.random.seed(seed)
    lhs_samples = lhs(n_params, samples=n_samples, criterion='maximin')
    
    # Scale to parameter bounds (log-scale for parameters spanning orders of magnitude)
    log_params = ['D_s_neg', 'D_s_pos', 'k_neg', 'k_pos', 'k_SEI', 'R_p_neg', 'R_p_pos']
    
    param_samples = np.zeros((n_samples, n_params))
    for j, pname in enumerate(param_names):
        lb, ub = bounds[pname]
        if pname in log_params:
            param_samples[:, j] = 10**(np.log10(lb) + lhs_samples[:, j] * (np.log10(ub) - np.log10(lb)))
        else:
            param_samples[:, j] = lb + lhs_samples[:, j] * (ub - lb)
    
    return param_samples, param_names


def generate_training_data(n_samples=500, I_app=2.0, seed=42):
    """Generate training data by running ECAT model with LHS samples"""
    param_samples, param_names = generate_lhs_samples(n_samples, seed)
    
    n_features = 50  # voltage features
    X = param_samples  # input: parameters
    Y = []  # output: voltage features + capacity + temp rise
    
    valid_indices = []
    
    for i in range(n_samples):
        params = {}
        for j, pname in enumerate(param_names):
            params[pname] = param_samples[i, j]
        
        model = ECATModel(params)
        try:
            features = model.compute_features(I_app=I_app, n_features=n_features)
            if features[-2] > 0.5 and features[:n_features].max() > 3.0:
                Y.append(features)
                valid_indices.append(i)
        except Exception as e:
            continue
        
        if (i+1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples} samples ({len(valid_indices)} valid)")
    
    X_valid = X[valid_indices]
    Y_valid = np.array(Y)
    
    print(f"Total valid samples: {len(valid_indices)}/{n_samples}")
    return X_valid, Y_valid, param_names


# ============================================================
# 2. ANN Meta-Model
# ============================================================
class ANNMetaModel:
    """
    Artificial Neural Network meta-model to replace ECAT simulation.
    Maps: parameters -> voltage/temperature features
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=[128, 256, 128]):
        import torch
        import torch.nn as nn
        
        self.device = torch.device('cpu')
        
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.input_scaler_mean = None
        self.input_scaler_std = None
        self.output_scaler_mean = None
        self.output_scaler_std = None
    
    def normalize_inputs(self, X):
        """Log-transform and standardize inputs"""
        param_names = get_identifiable_params()
        log_params = ['D_s_neg', 'D_s_pos', 'k_neg', 'k_pos', 'k_SEI', 'R_p_neg', 'R_p_pos']
        
        X_proc = X.copy()
        for j, pname in enumerate(param_names):
            if pname in log_params:
                X_proc[:, j] = np.log10(X_proc[:, j] + 1e-20)
        
        if self.input_scaler_mean is None:
            self.input_scaler_mean = X_proc.mean(axis=0)
            self.input_scaler_std = X_proc.std(axis=0) + 1e-10
        
        return (X_proc - self.input_scaler_mean) / self.input_scaler_std
    
    def normalize_outputs(self, Y):
        """Standardize outputs"""
        if self.output_scaler_mean is None:
            self.output_scaler_mean = Y.mean(axis=0)
            self.output_scaler_std = Y.std(axis=0) + 1e-10
        return (Y - self.output_scaler_mean) / self.output_scaler_std
    
    def denormalize_outputs(self, Y_norm):
        """Reverse standardization"""
        return Y_norm * self.output_scaler_std + self.output_scaler_mean
    
    def train(self, X, Y, epochs=300, lr=0.001, batch_size=32, val_split=0.15):
        """Train the ANN meta-model"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Normalize
        X_norm = self.normalize_inputs(X)
        Y_norm = self.normalize_outputs(Y)
        
        # Split
        n = len(X)
        n_val = int(n * val_split)
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        X_train = torch.FloatTensor(X_norm[train_idx]).to(self.device)
        Y_train = torch.FloatTensor(Y_norm[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X_norm[val_idx]).to(self.device)
        Y_val = torch.FloatTensor(Y_norm[val_idx]).to(self.device)
        
        dataset = TensorDataset(X_train, Y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            
            epoch_loss /= len(X_train)
            
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, Y_val).item()
            
            scheduler.step(val_loss)
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            if (epoch+1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        return train_losses, val_losses
    
    def predict(self, X):
        """Predict output features for given parameters"""
        import torch
        
        X_norm = self.normalize_inputs(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_norm).to(self.device)
            Y_pred_norm = self.model(X_t).numpy()
        
        return self.denormalize_outputs(Y_pred_norm)
    
    def save(self, path):
        """Save model"""
        import torch
        torch.save({
            'model_state': self.model.state_dict(),
            'input_mean': self.input_scaler_mean,
            'input_std': self.input_scaler_std,
            'output_mean': self.output_scaler_mean,
            'output_std': self.output_scaler_std,
        }, path)
    
    def load(self, path):
        """Load model"""
        import torch
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.input_scaler_mean = checkpoint['input_mean']
        self.input_scaler_std = checkpoint['input_std']
        self.output_scaler_mean = checkpoint['output_mean']
        self.output_scaler_std = checkpoint['output_std']


# ============================================================
# 3. Genetic Algorithm (GA)
# ============================================================
class GeneticAlgorithm:
    """
    Genetic Algorithm for parameter optimization using ANN meta-model.
    """
    
    def __init__(self, ann_model, target_features, param_names, bounds,
                 pop_size=100, n_generations=200, mutation_rate=0.15,
                 crossover_rate=0.8, elite_frac=0.1):
        self.ann = ann_model
        self.target = target_features
        self.param_names = param_names
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_gen = n_generations
        self.mut_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.elite_frac = elite_frac
        
        self.log_params = ['D_s_neg', 'D_s_pos', 'k_neg', 'k_pos', 'k_SEI', 'R_p_neg', 'R_p_pos']
        
        # Build bounds arrays
        self.lb = np.array([bounds[p][0] for p in param_names])
        self.ub = np.array([bounds[p][1] for p in param_names])
    
    def initialize_population(self):
        """Initialize population with LHS"""
        from pyDOE2 import lhs
        n_params = len(self.param_names)
        samples = lhs(n_params, samples=self.pop_size, criterion='maximin')
        
        pop = np.zeros((self.pop_size, n_params))
        for j, pname in enumerate(self.param_names):
            lb, ub = self.bounds[pname]
            if pname in self.log_params:
                pop[:, j] = 10**(np.log10(lb) + samples[:, j] * (np.log10(ub) - np.log10(lb)))
            else:
                pop[:, j] = lb + samples[:, j] * (ub - lb)
        
        return pop
    
    def fitness(self, population):
        """Evaluate fitness using ANN meta-model"""
        predictions = self.ann.predict(population)
        
        # Multi-objective: voltage RMSE + capacity error + temperature error
        n_v = len(self.target) - 2
        v_pred = predictions[:, :n_v]
        v_target = self.target[:n_v]
        
        cap_pred = predictions[:, -2]
        cap_target = self.target[-2]
        
        temp_pred = predictions[:, -1]
        temp_target = self.target[-1]
        
        # Voltage RMSE
        v_rmse = np.sqrt(np.mean((v_pred - v_target)**2, axis=1))
        
        # Capacity relative error
        cap_err = np.abs(cap_pred - cap_target) / (cap_target + 1e-10)
        
        # Temperature error
        temp_err = np.abs(temp_pred - temp_target) / (temp_target + 1e-10 + 5)
        
        # Combined fitness (lower is better)
        fitness_vals = v_rmse + 0.5 * cap_err + 0.2 * temp_err
        
        return fitness_vals
    
    def selection(self, population, fitness_vals):
        """Tournament selection"""
        n = len(population)
        selected = np.zeros_like(population)
        for i in range(n):
            idx1, idx2 = np.random.randint(0, n, 2)
            if fitness_vals[idx1] < fitness_vals[idx2]:
                selected[i] = population[idx1]
            else:
                selected[i] = population[idx2]
        return selected
    
    def crossover(self, parent1, parent2):
        """BLX-alpha crossover"""
        alpha = 0.5
        n = len(parent1)
        child1 = np.zeros(n)
        child2 = np.zeros(n)
        
        for j in range(n):
            d = abs(parent1[j] - parent2[j])
            low = min(parent1[j], parent2[j]) - alpha * d
            high = max(parent1[j], parent2[j]) + alpha * d
            low = max(low, self.lb[j])
            high = min(high, self.ub[j])
            child1[j] = np.random.uniform(low, high)
            child2[j] = np.random.uniform(low, high)
        
        return child1, child2
    
    def mutate(self, individual, generation, max_gen):
        """Adaptive mutation with decreasing magnitude"""
        n = len(individual)
        mutant = individual.copy()
        
        decay = 1.0 - generation / max_gen
        
        for j in range(n):
            if np.random.random() < self.mut_rate:
                pname = self.param_names[j]
                if pname in self.log_params:
                    log_val = np.log10(mutant[j] + 1e-20)
                    log_lb = np.log10(self.lb[j])
                    log_ub = np.log10(self.ub[j])
                    sigma = (log_ub - log_lb) * 0.1 * decay
                    log_val += np.random.normal(0, sigma)
                    log_val = np.clip(log_val, log_lb, log_ub)
                    mutant[j] = 10**log_val
                else:
                    sigma = (self.ub[j] - self.lb[j]) * 0.1 * decay
                    mutant[j] += np.random.normal(0, sigma)
                    mutant[j] = np.clip(mutant[j], self.lb[j], self.ub[j])
        
        return mutant
    
    def run(self):
        """Run the genetic algorithm"""
        population = self.initialize_population()
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        
        best_fitness_history = []
        avg_fitness_history = []
        best_individual = None
        best_fitness = float('inf')
        
        for gen in range(self.n_gen):
            fitness_vals = self.fitness(population)
            
            # Track best
            gen_best_idx = np.argmin(fitness_vals)
            gen_best_fit = fitness_vals[gen_best_idx]
            
            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_individual = population[gen_best_idx].copy()
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitness_vals))
            
            if (gen+1) % 50 == 0:
                print(f"  Gen {gen+1}/{self.n_gen}: best_fit={best_fitness:.6f}, avg_fit={np.mean(fitness_vals):.6f}")
            
            # Elitism
            elite_idx = np.argsort(fitness_vals)[:n_elite]
            elites = population[elite_idx].copy()
            
            # Selection
            selected = self.selection(population, fitness_vals)
            
            # Crossover and mutation
            new_pop = list(elites)
            while len(new_pop) < self.pop_size:
                i1, i2 = np.random.randint(0, len(selected), 2)
                if np.random.random() < self.cross_rate:
                    c1, c2 = self.crossover(selected[i1], selected[i2])
                else:
                    c1, c2 = selected[i1].copy(), selected[i2].copy()
                
                c1 = self.mutate(c1, gen, self.n_gen)
                c2 = self.mutate(c2, gen, self.n_gen)
                
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)
            
            population = np.array(new_pop[:self.pop_size])
        
        return best_individual, best_fitness, best_fitness_history, avg_fitness_history


# ============================================================
# 4. Direct GA (without ANN, for comparison)
# ============================================================
class DirectGA:
    """GA that directly evaluates the ECAT model (for timing comparison)"""
    
    def __init__(self, target_voltage, target_time, I_app, param_names, bounds,
                 pop_size=30, n_generations=50, V_cutoff=2.7):
        self.target_v = target_voltage
        self.target_t = target_time
        self.I_app = I_app
        self.param_names = param_names
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_gen = n_generations
        self.V_cutoff = V_cutoff
        self.log_params = ['D_s_neg', 'D_s_pos', 'k_neg', 'k_pos', 'k_SEI', 'R_p_neg', 'R_p_pos']
        self.lb = np.array([bounds[p][0] for p in param_names])
        self.ub = np.array([bounds[p][1] for p in param_names])
    
    def fitness_single(self, individual):
        """Evaluate single individual using ECAT model"""
        params = {}
        for j, pname in enumerate(self.param_names):
            params[pname] = individual[j]
        
        model = ECATModel(params)
        try:
            result = model.simulate_cc_discharge(
                I_app=self.I_app, t_end=5000, dt=5.0, V_cutoff=self.V_cutoff)
            
            if len(result['time']) < 10:
                return 100.0
            
            from scipy.interpolate import interp1d
            f_sim = interp1d(result['time'], result['voltage'], 
                           kind='linear', fill_value='extrapolate')
            
            # Evaluate at target time points
            common_t = self.target_t[self.target_t <= result['time'][-1]]
            if len(common_t) < 10:
                return 50.0
            
            v_sim = f_sim(common_t)
            v_exp = np.interp(common_t, self.target_t, self.target_v)
            
            rmse = np.sqrt(np.mean((v_sim - v_exp)**2))
            return rmse
        except:
            return 100.0
    
    def run(self):
        """Run direct GA"""
        from pyDOE2 import lhs
        n_params = len(self.param_names)
        
        # Initialize
        samples = lhs(n_params, samples=self.pop_size, criterion='maximin')
        population = np.zeros((self.pop_size, n_params))
        for j, pname in enumerate(self.param_names):
            lb, ub = self.bounds[pname]
            if pname in self.log_params:
                population[:, j] = 10**(np.log10(lb) + samples[:, j] * (np.log10(ub) - np.log10(lb)))
            else:
                population[:, j] = lb + samples[:, j] * (ub - lb)
        
        best_fitness = float('inf')
        best_individual = None
        history = []
        
        for gen in range(self.n_gen):
            fitness_vals = np.array([self.fitness_single(ind) for ind in population])
            
            idx = np.argmin(fitness_vals)
            if fitness_vals[idx] < best_fitness:
                best_fitness = fitness_vals[idx]
                best_individual = population[idx].copy()
            
            history.append(best_fitness)
            
            if (gen+1) % 10 == 0:
                print(f"  DirectGA Gen {gen+1}/{self.n_gen}: best_fit={best_fitness:.6f}")
            
            # Simple evolution
            elite_idx = np.argsort(fitness_vals)[:max(1, self.pop_size//5)]
            new_pop = list(population[elite_idx])
            
            while len(new_pop) < self.pop_size:
                i1 = np.random.choice(elite_idx)
                parent = population[i1].copy()
                for j in range(n_params):
                    if np.random.random() < 0.2:
                        sigma = (self.ub[j] - self.lb[j]) * 0.1
                        parent[j] += np.random.normal(0, sigma)
                        parent[j] = np.clip(parent[j], self.lb[j], self.ub[j])
                new_pop.append(parent)
            
            population = np.array(new_pop[:self.pop_size])
        
        return best_individual, best_fitness, history


# ============================================================
# 5. Sensitivity Analysis
# ============================================================
def sensitivity_analysis(model_params=None, I_app=2.0, n_features=50):
    """
    One-at-a-time sensitivity analysis for all identifiable parameters.
    """
    param_names = get_identifiable_params()
    bounds = get_parameter_bounds()
    
    if model_params is None:
        model_params = ECATModel.DEFAULT_PARAMS.copy()
    
    # Baseline
    base_model = ECATModel(model_params)
    base_features = base_model.compute_features(I_app=I_app, n_features=n_features)
    
    sensitivities = {}
    perturbation = 0.1  # 10% perturbation
    
    for pname in param_names:
        base_val = model_params.get(pname, ECATModel.DEFAULT_PARAMS[pname])
        
        # Perturb up
        params_up = dict(model_params)
        params_up[pname] = base_val * (1 + perturbation)
        params_up[pname] = min(params_up[pname], bounds[pname][1])
        
        model_up = ECATModel(params_up)
        feat_up = model_up.compute_features(I_app=I_app, n_features=n_features)
        
        # Perturb down
        params_dn = dict(model_params)
        params_dn[pname] = base_val * (1 - perturbation)
        params_dn[pname] = max(params_dn[pname], bounds[pname][0])
        
        model_dn = ECATModel(params_dn)
        feat_dn = model_dn.compute_features(I_app=I_app, n_features=n_features)
        
        # Sensitivity: normalized change in output / normalized change in input
        delta_out = np.abs(feat_up - feat_dn)
        delta_in = abs(params_up[pname] - params_dn[pname]) / base_val
        
        if delta_in > 0:
            sens = np.mean(delta_out) / (delta_in + 1e-10)
        else:
            sens = 0
        
        # Voltage sensitivity
        v_sens = np.mean(np.abs(feat_up[:n_features] - feat_dn[:n_features])) / (delta_in + 1e-10)
        # Capacity sensitivity
        cap_sens = abs(feat_up[-2] - feat_dn[-2]) / (delta_in + 1e-10)
        # Temperature sensitivity
        temp_sens = abs(feat_up[-1] - feat_dn[-1]) / (delta_in + 1e-10)
        
        sensitivities[pname] = {
            'overall': sens,
            'voltage': v_sens,
            'capacity': cap_sens,
            'temperature': temp_sens
        }
    
    return sensitivities


if __name__ == '__main__':
    print("Testing MMGA components...")
    
    # Test LHS
    print("\n1. LHS Sampling...")
    samples, names = generate_lhs_samples(n_samples=10)
    print(f"  Generated {len(samples)} samples with {len(names)} parameters")
    
    # Test sensitivity
    print("\n2. Sensitivity Analysis...")
    sens = sensitivity_analysis()
    for pname, s in sorted(sens.items(), key=lambda x: -x[1]['overall']):
        print(f"  {pname:15s}: overall={s['overall']:.4f}, V={s['voltage']:.4f}, cap={s['capacity']:.4f}, T={s['temperature']:.4f}")
