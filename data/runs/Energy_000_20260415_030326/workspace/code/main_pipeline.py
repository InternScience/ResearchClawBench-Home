"""
Main pipeline for MMGA framework.
"""

import numpy as np
import json
import os
import sys
import time as time_mod
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecat_model import (ECATModel, get_default_params, get_parameter_bounds,
                         get_identifiable_params, generate_lhs_samples,
                         extract_features, run_lhs_simulations)

# ============================================================
# 1. Data Loading Functions
# ============================================================

def load_cs2_36_data(filepath, cycle_idx=1):
    import openpyxl
    wb = openpyxl.load_workbook(filepath, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    
    time_data, volt_data, curr_data, cap_data = [], [], [], []
    for row in rows[1:]:
        if row[5] is not None and int(row[5]) == cycle_idx:
            if row[6] is not None and row[6] < -0.01:
                time_data.append(row[3])
                volt_data.append(row[7])
                curr_data.append(row[6])
                cap_data.append(abs(row[9]) if row[9] is not None else 0)
    wb.close()
    
    if len(time_data) == 0:
        return None
    return {
        'time': np.array(time_data, dtype=float),
        'voltage': np.array(volt_data, dtype=float),
        'current': np.array(curr_data, dtype=float),
        'capacity': np.array(cap_data, dtype=float),
        'temperature': np.full(len(time_data), 298.15),
    }


def load_nasa_data(filepath, battery_id='B0005', cycle_idx=0):
    import scipy.io as sio
    data = sio.loadmat(filepath)
    b = data[battery_id]
    cycle = b['cycle'][0,0]
    
    discharge_indices = []
    for i in range(cycle.shape[1]):
        c = cycle[0,i]
        if c['type'][0] == 'discharge':
            discharge_indices.append(i)
    
    if cycle_idx >= len(discharge_indices):
        return None
    
    idx = discharge_indices[cycle_idx]
    d = cycle[0,idx]['data'][0,0]
    
    t = d['Time'].flatten().astype(float)
    I = d['Current_measured'].flatten().astype(float)
    cap = np.cumsum(np.abs(I) * np.diff(np.concatenate([[0], t])) / 3600.0)
    
    return {
        'time': t,
        'voltage': d['Voltage_measured'].flatten().astype(float),
        'current': I,
        'temperature': d['Temperature_measured'].flatten().astype(float) + 273.15,
        'capacity': cap,
    }


def load_oxford_data(filepath):
    import scipy.io as sio
    data = sio.loadmat(filepath)
    ex = data['ExampleDC_C1'][0,0]
    dc = ex['dc']
    
    t = dc['t'][0,0].flatten().astype(float)
    v = dc['v'][0,0].flatten().astype(float)
    i_dc = dc['i'][0,0].flatten().astype(float) / 1000.0
    T = dc['T'][0,0].flatten().astype(float) + 273.15
    q = dc['q'][0,0].flatten().astype(float) / 1000.0
    
    t = t - t[0]
    return {
        'time': t, 'voltage': v, 'current': i_dc,
        'temperature': T, 'capacity': np.abs(q),
    }


def preprocess_experimental_data(data, n_points=100, cutoff_voltage=2.7):
    V = data['voltage']
    t = data['time']
    T = data['temperature']
    cap = data['capacity']
    I = data['current']
    
    cutoff_idx = len(V)
    below_cutoff = np.where(V < cutoff_voltage)[0]
    if len(below_cutoff) > 0:
        cutoff_idx = below_cutoff[0] + 1
    
    V = V[:cutoff_idx]
    t = t[:cutoff_idx]
    T = T[:cutoff_idx]
    cap = cap[:cutoff_idx]
    I = I[:cutoff_idx]
    
    if t[-1] > 0:
        t_norm = t / t[-1]
    else:
        t_norm = t
    
    t_interp = np.linspace(0, 1, n_points)
    V_interp = np.interp(t_interp, t_norm, V)
    T_interp = np.interp(t_interp, t_norm, T)
    cap_interp = np.interp(t_interp, t_norm, cap)
    
    return {
        'time_norm': t_interp,
        'voltage': V_interp,
        'temperature': T_interp,
        'capacity': cap_interp,
        'voltage_raw': V,
        'temperature_raw': T,
        'time_raw': t,
        'capacity_final': cap[-1] if len(cap) > 0 else 0,
        'current_mean': np.mean(np.abs(I)),
    }


# ============================================================
# 2. ANN Meta-Model
# ============================================================

class ANNMetaModel:
    def __init__(self, n_inputs, n_outputs, hidden_layers=[64, 32, 16]):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.model = None
        self.X_scaler = None
        self.Y_scaler = None
        self.is_trained = False
    
    def build_model(self):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layers),
            activation='relu', solver='adam', alpha=0.001,
            batch_size='auto', learning_rate='adaptive',
            learning_rate_init=0.001, max_iter=2000, tol=1e-6,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42,
        )
    
    def train(self, X, Y):
        self.build_model()
        X_scaled = self.X_scaler.fit_transform(X)
        Y_scaled = self.Y_scaler.fit_transform(Y)
        self.model.fit(X_scaled, Y_scaled)
        self.is_trained = True
        return self.model.score(X_scaled, Y_scaled)
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.model.predict(X_scaled)
        return self.Y_scaler.inverse_transform(Y_scaled)
    
    def predict_single(self, x):
        return self.predict(x.reshape(1, -1)).flatten()


# ============================================================
# 3. MMGA Optimizer
# ============================================================

class MMGA:
    def __init__(self, ann_model, param_bounds, experimental_features,
                 population_size=100, n_generations=200,
                 crossover_rate=0.8, mutation_rate=0.1,
                 elite_fraction=0.1, n_refine=10):
        self.ann_model = ann_model
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.exp_features = experimental_features
        self.pop_size = population_size
        self.n_gen = n_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.elite_frac = elite_fraction
        self.n_refine = n_refine
        self.history = {'best_fitness': [], 'mean_fitness': [], 'generation': []}
    
    def fitness(self, individual):
        pred = self.ann_model.predict_single(individual)
        rmse = np.sqrt(np.mean((pred - self.exp_features)**2))
        return -rmse
    
    def initialize_population(self):
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.n_params, seed=42)
        unit = sampler.random(n=self.pop_size)
        pop = np.zeros((self.pop_size, self.n_params))
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            pop[:, i] = qmc.scale(unit[:, i:i+1], low, high).flatten()
        return pop
    
    def crossover(self, p1, p2):
        alpha = np.random.random(self.n_params)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return c1, c2
    
    def mutate(self, ind):
        m = ind.copy()
        for i, name in enumerate(self.param_names):
            if np.random.random() < self.mut_rate:
                low, high = self.param_bounds[name]
                m[i] = np.clip(m[i] + (high - low) * 0.1 * np.random.randn(), low, high)
        return m
    
    def run(self, verbose=True):
        pop = self.initialize_population()
        for gen in range(self.n_gen):
            fitnesses = np.array([self.fitness(ind) for ind in pop])
            self.history['best_fitness'].append(np.max(fitnesses))
            self.history['mean_fitness'].append(np.mean(fitnesses))
            self.history['generation'].append(gen)
            
            if verbose and gen % 50 == 0:
                print(f"  MMGA Gen {gen}: best={np.max(fitnesses):.6f}")
            
            n_elite = max(2, int(self.elite_frac * self.pop_size))
            elite_idx = np.argsort(fitnesses)[-n_elite:]
            new_pop = [pop[i].copy() for i in elite_idx]
            
            while len(new_pop) < self.pop_size:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = pop[i1] if fitnesses[i1] > fitnesses[i2] else pop[i2]
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = pop[i3] if fitnesses[i3] > fitnesses[i4] else pop[i4]
                
                if np.random.random() < self.cx_rate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))
            
            pop = np.array(new_pop[:self.pop_size])
        
        fitnesses = np.array([self.fitness(ind) for ind in pop])
        best_idx = np.argmax(fitnesses)
        return pop[best_idx], fitnesses[best_idx]
    
    def refine_with_model(self, best_individual, I_app, t_end, T_amb=298.15):
        default = get_default_params()
        best_fitness = -np.inf
        best_params = best_individual.copy()
        
        for _ in range(self.n_refine):
            candidate = best_params.copy()
            for i, name in enumerate(self.param_names):
                low, high = self.param_bounds[name]
                candidate[i] = np.clip(candidate[i] + (high - low) * 0.02 * np.random.randn(), low, high)
            
            params = default.copy()
            for i, name in enumerate(self.param_names):
                params[name] = candidate[i]
            
            model = ECATModel(params)
            result = model.simulate_discharge(I_app, t_end, T_amb, 100)
            features = extract_features(result)
            
            if features is not None:
                rmse = np.sqrt(np.mean((features - self.exp_features)**2))
                fitness = -rmse
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = candidate.copy()
        
        return best_params, best_fitness


# ============================================================
# 4. Baseline GA
# ============================================================

class BaselineGA:
    def __init__(self, param_bounds, exp_data, I_app, t_end, T_amb=298.15,
                 population_size=50, n_generations=50):
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.exp_data = exp_data
        self.I_app = I_app
        self.t_end = t_end
        self.T_amb = T_amb
        self.pop_size = population_size
        self.n_gen = n_generations
        self.history = {'best_fitness': [], 'mean_fitness': [], 'generation': []}
        self.n_evals = 0
    
    def fitness(self, individual):
        default = get_default_params()
        params = default.copy()
        for i, name in enumerate(self.param_names):
            params[name] = individual[i]
        
        model = ECATModel(params)
        result = model.simulate_discharge(self.I_app, self.t_end, self.T_amb, 100)
        self.n_evals += 1
        
        if not result['success'] or len(result['voltage']) < 5:
            return -np.inf
        
        exp_V = self.exp_data['voltage']
        exp_T = self.exp_data['temperature']
        n_exp = len(exp_V)
        
        t_norm_model = result['time'] / result['time'][-1] if result['time'][-1] > 0 else result['time']
        t_norm_exp = np.linspace(0, 1, n_exp)
        
        model_V = np.interp(t_norm_exp, t_norm_model, result['voltage'])
        model_T = np.interp(t_norm_exp, t_norm_model, result['temperature'])
        
        rmse_V = np.sqrt(np.mean((model_V - exp_V)**2))
        rmse_T = np.sqrt(np.mean((model_T - exp_T)**2))
        return -(rmse_V + 0.1 * rmse_T)
    
    def run(self, verbose=True):
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.n_params, seed=42)
        unit = sampler.random(n=self.pop_size)
        pop = np.zeros((self.pop_size, self.n_params))
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            pop[:, i] = qmc.scale(unit[:, i:i+1], low, high).flatten()
        
        for gen in range(self.n_gen):
            fitnesses = np.array([self.fitness(ind) for ind in pop])
            valid = fitnesses > -np.inf
            best_val = np.max(fitnesses[valid]) if np.any(valid) else -10
            mean_val = np.mean(fitnesses[valid]) if np.any(valid) else -10
            self.history['best_fitness'].append(best_val)
            self.history['mean_fitness'].append(mean_val)
            self.history['generation'].append(gen)
            
            if verbose and gen % 10 == 0:
                print(f"  Baseline Gen {gen}: best={best_val:.6f}")
            
            n_elite = max(2, int(0.1 * self.pop_size))
            if np.any(valid):
                elite_idx = np.argsort(fitnesses)[-n_elite:]
                new_pop = [pop[i].copy() for i in elite_idx]
            else:
                new_pop = [pop[0].copy()]
            
            while len(new_pop) < self.pop_size:
                idx = np.random.choice(self.pop_size, 2, replace=False)
                parent = pop[idx[0]] if fitnesses[idx[0]] > fitnesses[idx[1]] else pop[idx[1]]
                child = parent.copy()
                for i, name in enumerate(self.param_names):
                    if np.random.random() < 0.2:
                        low, high = self.param_bounds[name]
                        child[i] = np.clip(child[i] + (high - low) * 0.1 * np.random.randn(), low, high)
                new_pop.append(child)
            
            pop = np.array(new_pop[:self.pop_size])
        
        fitnesses = np.array([self.fitness(ind) for ind in pop])
        best_idx = np.argmax(fitnesses)
        return pop[best_idx], fitnesses[best_idx]


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MMGA Framework for ECAT Parameter Identification")
    print("=" * 60)
    
    # Step 1: Load experimental data
    print("\n[Step 1] Loading experimental data...")
    cs2_data = load_cs2_36_data('data/CS2_36/CS2_36_1_10_11.xlsx', cycle_idx=1)
    cs2_proc = preprocess_experimental_data(cs2_data, n_points=100)
    print(f"  CS2_36: V={cs2_proc['voltage'].min():.3f}-{cs2_proc['voltage'].max():.3f}")
    
    nasa_data = load_nasa_data('data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/B0005.mat', 'B0005', 0)
    nasa_proc = preprocess_experimental_data(nasa_data, n_points=100)
    print(f"  NASA B0005: V={nasa_proc['voltage'].min():.3f}-{nasa_proc['voltage'].max():.3f}")
    
    oxford_data = load_oxford_data('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat')
    oxford_proc = preprocess_experimental_data(oxford_data, n_points=100)
    print(f"  Oxford: V={oxford_proc['voltage'].min():.3f}-{oxford_proc['voltage'].max():.3f}")
    
    print("\n[Step 1] Complete.")
