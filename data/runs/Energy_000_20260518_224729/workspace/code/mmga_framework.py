"""
MMGA Framework: Multi-objective Multi-modal Genetic Algorithm with ANN surrogate
for rapid parameter identification of electrochemical-aging-thermal battery models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from battery_model import BatteryDischargeModel, get_param_bounds, get_default_params

np.random.seed(42)

os.makedirs('outputs', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

# ==================== LHS SAMPLING ====================
def latin_hypercube_sampling(bounds, n_samples):
    """Generate Latin Hypercube Samples within given bounds."""
    n_params = len(bounds)
    samples = np.zeros((n_samples, n_params))
    for i in range(n_params):
        perm = np.random.permutation(n_samples)
        samples[:, i] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples
    
    param_names = list(bounds.keys())
    scaled = {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[name]
        scaled[name] = samples[:, i] * (hi - lo) + lo
    
    return scaled, param_names

# ==================== FEATURE EXTRACTION ====================
def extract_curve_features(time, voltage, temperature=None):
    """Extract features from discharge curve for ANN training."""
    features = {}
    
    features['V_mean'] = float(np.mean(voltage))
    features['V_min'] = float(np.min(voltage))
    features['V_max'] = float(np.max(voltage))
    features['V_std'] = float(np.std(voltage))
    features['V_range'] = features['V_max'] - features['V_min']
    
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        idx = int(frac * (len(voltage) - 1))
        features[f'V_at_{int(frac*100)}pct'] = float(voltage[idx])
    
    dv_dt = np.gradient(voltage, time)
    features['dVdt_mean'] = float(np.mean(dv_dt))
    features['dVdt_min'] = float(np.min(dv_dt))
    features['dVdt_max'] = float(np.max(dv_dt))
    
    mid_v = (features['V_max'] + features['V_min']) / 2
    features['time_to_midV'] = float(time[np.argmin(np.abs(voltage - mid_v))]) if len(time) > 0 else 0.0
    features['capacity_approx'] = float(time[-1] * 1.1 / 3600)
    
    if temperature is not None and len(temperature) > 0:
        features['T_max'] = float(np.max(temperature))
        features['T_rise'] = float(temperature[-1] - temperature[0])
        features['T_mean'] = float(np.mean(temperature))
    else:
        features['T_max'] = 25.0
        features['T_rise'] = 0.0
        features['T_mean'] = 25.0
    
    return features

# ==================== DATA GENERATION ====================
def generate_training_data(n_samples=2000):
    """Generate synthetic training data using LHS + physics model."""
    print(f"Generating {n_samples} LHS samples...")
    bounds = get_param_bounds()
    lhs_samples, param_names = latin_hypercube_sampling(bounds, n_samples)
    
    default_params = get_default_params()
    
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        if i % 200 == 0:
            print(f"  Sample {i}/{n_samples}")
        
        params = default_params.copy()
        for name in param_names:
            params[name] = lhs_samples[name][i]
        
        model = BatteryDischargeModel(params)
        result = model.solve(t_end=3600, dt=30)
        
        features = extract_curve_features(result['time'], result['voltage'], result['temperature'])
        
        param_values = [params[name] for name in param_names]
        X_list.append(param_values)
        y_list.append(list(features.values()))
    
    X = np.array(X_list)
    y = np.array(y_list)
    feature_names = list(features.keys())
    
    return X, y, param_names, feature_names

# ==================== ANN SURROGATE ====================
def train_ann_surrogate(X, y, param_names, feature_names):
    """Train ANN meta-model."""
    print("\nTraining ANN surrogate model...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_s = scaler_X.fit_transform(X_train)
    y_train_s = scaler_y.fit_transform(y_train)
    X_test_s = scaler_X.transform(X_test)
    y_test_s = scaler_y.transform(y_test)
    
    ann = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )
    
    ann.fit(X_train_s, y_train_s)
    
    y_pred_s = ann.predict(X_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s)
    
    mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    
    print(f"ANN Test MSE: {mse:.6f}")
    print(f"ANN Test R2: {r2:.4f}")
    
    with open('outputs/ann_surrogate.pkl', 'wb') as f:
        pickle.dump({'ann': ann, 'scaler_X': scaler_X, 'scaler_y': scaler_y,
                     'param_names': param_names, 'feature_names': feature_names}, f)
    
    return ann, scaler_X, scaler_y, param_names, feature_names

# ==================== GENETIC ALGORITHM ====================
class MultiObjectiveGA:
    def __init__(self, ann, scaler_X, scaler_y, feature_names, param_names, bounds,
                 pop_size=100, n_generations=200,
                 crossover_rate=0.8, mutation_rate=0.1,
                 tournament_size=5):
        self.ann = ann
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_names = feature_names
        self.param_names = param_names
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.n_params = len(param_names)
        
        self.voltage_feature_idx = [i for i, name in enumerate(self.feature_names)
                                    if 'V_' in name or 'dVdt' in name or 'capacity' in name or 'time_to' in name]
        self.temp_feature_idx = [i for i, name in enumerate(self.feature_names)
                                 if 'T_' in name]
    
    def predict_features(self, params_array):
        params_s = self.scaler_X.transform(params_array)
        features_s = self.ann.predict(params_s)
        features = self.scaler_y.inverse_transform(features_s)
        return features
    
    def evaluate_fitness(self, population, target_features):
        predicted = self.predict_features(population)
        
        f1 = np.mean((predicted[:, self.voltage_feature_idx] - target_features[self.voltage_feature_idx])**2, axis=1)
        f2 = np.mean((predicted[:, self.temp_feature_idx] - target_features[self.temp_feature_idx])**2, axis=1)
        
        return f1, f2
    
    def non_dominated_sort(self, f1, f2):
        n = len(f1)
        ranks = np.zeros(n, dtype=int)
        domination_counts = np.zeros(n, dtype=int)
        dominated_sets = [set() for _ in range(n)]
        
        fronts = [[]]
        for i in range(n):
            for j in range(i+1, n):
                if f1[i] <= f1[j] and f2[i] <= f2[j] and (f1[i] < f1[j] or f2[i] < f2[j]):
                    dominated_sets[i].add(j)
                    domination_counts[j] += 1
                elif f1[j] <= f1[i] and f2[j] <= f2[i] and (f1[j] < f1[i] or f2[j] < f2[i]):
                    dominated_sets[j].add(i)
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                ranks[i] = 0
                fronts[0].append(i)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        fronts = fronts[:-1]
        return ranks, fronts
    
    def crowding_distance(self, f1, f2, front):
        if len(front) <= 2:
            return {i: float('inf') for i in front}
        
        distances = {i: 0.0 for i in front}
        
        for objectives in [f1, f2]:
            sorted_front = sorted(front, key=lambda i: objectives[i])
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')
            
            f_max = objectives[sorted_front[-1]]
            f_min = objectives[sorted_front[0]]
            if f_max - f_min > 1e-10:
                for j in range(1, len(sorted_front) - 1):
                    distances[sorted_front[j]] += (objectives[sorted_front[j+1]] - objectives[sorted_front[j-1]]) / (f_max - f_min)
        
        return distances
    
    def tournament_selection(self, ranks, distances):
        selected = []
        for _ in range(self.pop_size):
            contestants = np.random.choice(len(ranks), self.tournament_size, replace=False)
            best = contestants[0]
            for c in contestants[1:]:
                if ranks[c] < ranks[best] or (ranks[c] == ranks[best] and distances.get(c, 0) > distances.get(best, 0)):
                    best = c
            selected.append(best)
        return selected
    
    def crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        eta = 20.0
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(self.n_params):
            if np.random.rand() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    beta = 1.0 + (2.0 * (y1 - self.bounds[self.param_names[i]][0]) / (y2 - y1))
                    alpha = 2.0 - beta**(-(eta + 1))
                    rand = np.random.rand()
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                    
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (self.bounds[self.param_names[i]][1] - y2) / (y2 - y1))
                    alpha = 2.0 - beta**(-(eta + 1))
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha)**(1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                    
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    child1[i] = c1
                    child2[i] = c2
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        return child1, child2
    
    def mutate(self, individual):
        eta_m = 20.0
        for i in range(self.n_params):
            if np.random.rand() < self.mutation_rate:
                lo, hi = self.bounds[self.param_names[i]]
                delta1 = (individual[i] - lo) / (hi - lo)
                delta2 = (hi - individual[i]) / (hi - lo)
                
                rand = np.random.rand()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy**(eta_m + 1))
                    delta_q = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy**(eta_m + 1))
                    delta_q = 1.0 - val**mut_pow
                
                individual[i] += delta_q * (hi - lo)
                individual[i] = np.clip(individual[i], lo, hi)
        
        return individual
    
    def optimize(self, target_features):
        print(f"\nStarting MMGA optimization...")
        print(f"Population: {self.pop_size}, Generations: {self.n_generations}")
        
        population = np.zeros((self.pop_size, self.n_params))
        for i, name in enumerate(self.param_names):
            lo, hi = self.bounds[name]
            population[:, i] = np.random.uniform(lo, hi, self.pop_size)
        
        best_f1_history = []
        best_f2_history = []
        
        for gen in range(self.n_generations):
            f1, f2 = self.evaluate_fitness(population, target_features)
            ranks, fronts = self.non_dominated_sort(f1, f2)
            distances = {}
            for front in fronts:
                cd = self.crowding_distance(f1, f2, front)
                distances.update(cd)
            
            best_f1_history.append(float(np.min(f1)))
            best_f2_history.append(float(np.min(f2)))
            
            if gen % 20 == 0:
                print(f"  Gen {gen}: best f1={np.min(f1):.6f}, best f2={np.min(f2):.6f}, front0 size={len(fronts[0])}")
            
            selected_idx = self.tournament_selection(ranks, distances)
            parents = population[selected_idx]
            
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1 = parents[i % len(parents)]
                p2 = parents[(i + 1) % len(parents)]
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                offspring.append(c1)
                offspring.append(c2)
            
            offspring = np.array(offspring[:self.pop_size])
            
            combined = np.vstack([population, offspring])
            f1_c, f2_c = self.evaluate_fitness(combined, target_features)
            ranks_c, fronts_c = self.non_dominated_sort(f1_c, f2_c)
            
            new_pop_idx = []
            for front in fronts_c:
                if len(new_pop_idx) + len(front) <= self.pop_size:
                    new_pop_idx.extend(front)
                else:
                    cd = self.crowding_distance(f1_c, f2_c, front)
                    sorted_front = sorted(front, key=lambda i: cd.get(i, 0), reverse=True)
                    remaining = self.pop_size - len(new_pop_idx)
                    new_pop_idx.extend(sorted_front[:remaining])
                    break
            
            population = combined[new_pop_idx]
        
        f1, f2 = self.evaluate_fitness(population, target_features)
        ranks, fronts = self.non_dominated_sort(f1, f2)
        
        pareto_front = population[fronts[0]]
        pareto_f1 = f1[fronts[0]]
        pareto_f2 = f2[fronts[0]]
        
        distances_to_origin = np.sqrt(pareto_f1**2 + pareto_f2**2)
        best_idx = np.argmin(distances_to_origin)
        best_params = pareto_front[best_idx]
        
        return best_params, pareto_front, pareto_f1, pareto_f2, best_f1_history, best_f2_history


# ==================== EXPERIMENTAL DATA LOADING ====================
def load_experimental_data():
    datasets = {}
    
    print("Loading NASA PCoE data...")
    nasa_data = {}
    for f in ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']:
        mat = loadmat(f'data/NASA PCoE Dataset Repository/1. BatteryAgingARC-FY08Q4/{f}')
        bname = f.split('.')[0]
        b = mat[bname]
        cycle = b[0,0]['cycle']
        
        for i in range(cycle.shape[1]):
            c = cycle[0,i]
            if c['type'][0] == 'discharge':
                data = c['data'][0,0]
                voltage = data['Voltage_measured'].flatten()
                current = data['Current_measured'].flatten()
                temp = data['Temperature_measured'].flatten()
                time = data['Time'].flatten()
                if len(voltage) > 50:
                    nasa_data[bname] = {
                        'time': time,
                        'voltage': voltage,
                        'current': current,
                        'temperature': temp
                    }
                    break
    datasets['NASA'] = nasa_data
    
    print("Loading CS2_36 data...")
    cs2_data = {}
    for f in ['CS2_36_1_10_11.xlsx', 'CS2_36_1_18_11.xlsx', 'CS2_36_1_24_11.xlsx', 'CS2_36_1_28_11.xlsx']:
        df = pd.read_excel(f'data/CS2_36/{f}', sheet_name='Channel_1-009')
        discharge = df[df['Step_Index'] == 7].copy()
        discharge = discharge.sort_values('Step_Time(s)')
        name = f.split('.')[0]
        cs2_data[name] = {
            'time': discharge['Step_Time(s)'].values,
            'voltage': discharge['Voltage(V)'].values,
            'current': discharge['Current(A)'].values,
            'temperature': np.full(len(discharge), 25.0)
        }
    datasets['CS2_36'] = cs2_data
    
    print("Loading Oxford data...")
    mat = loadmat('data/Oxford Battery Degradation Dataset/ExampleDC_C1.mat')
    dc = mat['ExampleDC_C1'][0,0]['dc'][0,0]
    oxford_data = {
        'Oxford_DC': {
            'time': dc['t'].flatten(),
            'voltage': dc['v'].flatten(),
            'current': dc['i'].flatten() / 1000,
            'temperature': dc['T'].flatten()
        }
    }
    datasets['Oxford'] = oxford_data
    
    return datasets


# ==================== MAIN PIPELINE ====================
def main():
    print("="*60)
    print("MMGA Parameter Identification Framework")
    print("="*60)
    
    X, y, param_names, feature_names = generate_training_data(n_samples=2000)
    np.savez('outputs/training_data.npz', X=X, y=y, param_names=param_names, feature_names=feature_names)
    
    ann, scaler_X, scaler_y, param_names, feature_names = train_ann_surrogate(X, y, param_names, feature_names)
    
    datasets = load_experimental_data()
    
    bounds = get_param_bounds()
    results = {}
    
    for dataset_name, data_dict in datasets.items():
        print(f"\n{'='*40}")
        print(f"Processing {dataset_name} dataset...")
        print(f"{'='*40}")
        
        dataset_results = {}
        for cell_name, exp_data in data_dict.items():
            print(f"\n  Cell: {cell_name}")
            
            target_features = extract_curve_features(
                exp_data['time'],
                exp_data['voltage'],
                exp_data.get('temperature', None)
            )
            target_vector = np.array([target_features[name] for name in feature_names])
            
            moga = MultiObjectiveGA(
                ann, scaler_X, scaler_y, feature_names, param_names, bounds,
                pop_size=100, n_generations=150,
                crossover_rate=0.8, mutation_rate=0.15
            )
            
            best_params, pareto_front, pareto_f1, pareto_f2, f1_hist, f2_hist = moga.optimize(target_vector)
            
            default = get_default_params()
            identified = default.copy()
            for i, name in enumerate(param_names):
                identified[name] = best_params[i]
            
            model = BatteryDischargeModel(identified)
            simulated = model.solve(t_end=exp_data['time'][-1], dt=10)
            
            dataset_results[cell_name] = {
                'identified_params': identified,
                'best_params_array': best_params.tolist(),
                'pareto_front': pareto_front.tolist(),
                'pareto_f1': pareto_f1.tolist(),
                'pareto_f2': pareto_f2.tolist(),
                'f1_history': f1_hist,
                'f2_history': f2_hist,
                'experimental': exp_data,
                'simulated': simulated,
                'target_features': target_features
            }
            
            print(f"  Best params: {dict(zip(param_names, best_params.tolist()))}")
        
        results[dataset_name] = dataset_results
    
    with open('outputs/identification_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    generate_figures(results, param_names, feature_names, X, y, ann, scaler_X, scaler_y)
    
    print("\n" + "="*60)
    print("MMGA Framework complete!")
    print("="*60)


def generate_figures(results, param_names, feature_names, X, y, ann, scaler_X, scaler_y):
    print("\nGenerating figures...")
    
    # Figure 1: Pareto fronts
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    idx = 0
    for dataset_name, dataset_results in results.items():
        if idx >= 4:
            break
        ax = axes[idx]
        for cell_name, res in dataset_results.items():
            ax.scatter(res['pareto_f1'], res['pareto_f2'], s=30, alpha=0.6, label=cell_name)
        ax.set_xlabel('Voltage Error (f1)')
        ax.set_ylabel('Temperature Error (f2)')
        ax.set_title(f'{dataset_name}: Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1
    plt.tight_layout()
    plt.savefig('report/images/fig1_pareto_fronts.png', dpi=150)
    plt.close()
    
    # Figure 2: Convergence curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    idx = 0
    for dataset_name, dataset_results in results.items():
        if idx >= 4:
            break
        ax = axes[idx]
        for cell_name, res in dataset_results.items():
            ax.plot(res['f1_history'], label=f'{cell_name} (f1)', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Objective Value')
        ax.set_title(f'{dataset_name}: Convergence')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1
    plt.tight_layout()
    plt.savefig('report/images/fig2_convergence.png', dpi=150)
    plt.close()
    
    # Figure 3: Experimental vs Simulated voltage
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    idx = 0
    for dataset_name, dataset_results in results.items():
        if idx >= 4:
            break
        ax = axes[idx]
        for cell_name, res in dataset_results.items():
            exp = res['experimental']
            sim = res['simulated']
            ax.plot(exp['time'], exp['voltage'], 'b-', alpha=0.7, label=f'{cell_name} (Exp)')
            ax.plot(sim['time'], sim['voltage'], 'r--', alpha=0.7, label=f'{cell_name} (Sim)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'{dataset_name}: Voltage Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        idx += 1
    plt.tight_layout()
    plt.savefig('report/images/fig3_voltage_comparison.png', dpi=150)
    plt.close()
    
    # Figure 4: Temperature comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    idx = 0
    for dataset_name, dataset_results in results.items():
        if idx >= 4:
            break
        ax = axes[idx]
        for cell_name, res in dataset_results.items():
            exp = res['experimental']
            sim = res['simulated']
            if 'temperature' in exp and np.max(exp['temperature']) > 20:
                ax.plot(exp['time'], exp['temperature'], 'b-', alpha=0.7, label=f'{cell_name} (Exp)')
            ax.plot(sim['time'], sim['temperature'], 'r--', alpha=0.7, label=f'{cell_name} (Sim)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{dataset_name}: Temperature Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        idx += 1
    plt.tight_layout()
    plt.savefig('report/images/fig4_temperature_comparison.png', dpi=150)
    plt.close()
    
    # Figure 5: Parameter comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    param_data = []
    labels = []
    for dataset_name, dataset_results in results.items():
        for cell_name, res in dataset_results.items():
            params = [res['identified_params'][name] for name in param_names]
            bounds = get_param_bounds()
            normalized = [(params[i] - bounds[param_names[i]][0]) / 
                         (bounds[param_names[i]][1] - bounds[param_names[i]][0]) 
                         for i in range(len(param_names))]
            param_data.append(normalized)
            labels.append(f'{dataset_name}_{cell_name}')
    
    x = np.arange(len(param_names))
    width = 0.8 / len(param_data)
    for i, (pdata, label) in enumerate(zip(param_data, labels)):
        ax.bar(x + i * width, pdata, width, label=label, alpha=0.7)
    ax.set_xticks(x + width * (len(param_data) - 1) / 2)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_ylabel('Normalized Parameter Value')
    ax.set_title('Identified Parameters Across Datasets')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('report/images/fig5_parameter_comparison.png', dpi=150)
    plt.close()
    
    # Figure 6: ANN prediction accuracy
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    X_s = scaler_X.transform(X)
    y_pred_s = ann.predict(X_s)
    y_pred = scaler_y.inverse_transform(y_pred_s)
    
    for i in range(min(6, len(feature_names))):
        ax = axes[i]
        ax.scatter(y[:, i], y_pred[:, i], s=10, alpha=0.3)
        ax.plot([y[:, i].min(), y[:, i].max()], [y[:, i].min(), y[:, i].max()], 'r--')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{feature_names[i]}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig6_ann_accuracy.png', dpi=150)
    plt.close()
    
    # Figure 7: LHS sampling visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title('LHS Sampling Distribution (First 2 Parameters)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig7_lhs_distribution.png', dpi=150)
    plt.close()
    
    print("Figures generated successfully.")


if __name__ == '__main__':
    main()
