"""
MMGA Framework v2: Direct curve-matching approach.
ANN predicts full voltage curves at fixed time points.
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

# Fixed time points for curve representation
TIME_POINTS = np.linspace(0, 3600, 121)  # 30s intervals

def latin_hypercube_sampling(bounds, n_samples):
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

def interpolate_to_fixed_time(time, voltage, t_target=TIME_POINTS):
    """Interpolate voltage to fixed time points."""
    f = np.interp
    return f(t_target, time, voltage)

def generate_training_data(n_samples=3000):
    print(f"Generating {n_samples} LHS samples...")
    bounds = get_param_bounds()
    lhs_samples, param_names = latin_hypercube_sampling(bounds, n_samples)
    default_params = get_default_params()
    
    X_list = []
    y_voltage_list = []
    y_temp_list = []
    
    for i in range(n_samples):
        if i % 300 == 0:
            print(f"  Sample {i}/{n_samples}")
        
        params = default_params.copy()
        for name in param_names:
            params[name] = lhs_samples[name][i]
        
        model = BatteryDischargeModel(params)
        result = model.solve(t_end=3600, dt=30)
        
        V_interp = interpolate_to_fixed_time(result['time'], result['voltage'])
        T_interp = interpolate_to_fixed_time(result['time'], result['temperature'])
        
        param_values = [params[name] for name in param_names]
        X_list.append(param_values)
        y_voltage_list.append(V_interp)
        y_temp_list.append(T_interp)
    
    X = np.array(X_list)
    y_voltage = np.array(y_voltage_list)
    y_temp = np.array(y_temp_list)
    
    return X, y_voltage, y_temp, param_names

def train_ann_surrogate(X, y_voltage, y_temp, param_names):
    print("\nTraining ANN surrogate models...")
    
    X_train, X_test, yv_train, yv_test, yt_train, yt_test = train_test_split(
        X, y_voltage, y_temp, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    
    # Voltage ANN
    print("  Training voltage ANN...")
    scaler_yv = StandardScaler()
    yv_train_s = scaler_yv.fit_transform(yv_train)
    yv_test_s = scaler_yv.transform(yv_test)
    
    ann_v = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    )
    ann_v.fit(X_train_s, yv_train_s)
    
    yv_pred_s = ann_v.predict(X_test_s)
    yv_pred = scaler_yv.inverse_transform(yv_pred_s)
    mse_v = mean_squared_error(yv_test, yv_pred)
    r2_v = r2_score(yv_test, yv_pred, multioutput='uniform_average')
    print(f"  Voltage ANN - MSE: {mse_v:.4f}, R2: {r2_v:.4f}")
    
    # Temperature ANN
    print("  Training temperature ANN...")
    scaler_yt = StandardScaler()
    yt_train_s = scaler_yt.fit_transform(yt_train)
    yt_test_s = scaler_yt.transform(yt_test)
    
    ann_t = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    )
    ann_t.fit(X_train_s, yt_train_s)
    
    yt_pred_s = ann_t.predict(X_test_s)
    yt_pred = scaler_yt.inverse_transform(yt_pred_s)
    mse_t = mean_squared_error(yt_test, yt_pred)
    r2_t = r2_score(yt_test, yt_pred, multioutput='uniform_average')
    print(f"  Temperature ANN - MSE: {mse_t:.4f}, R2: {r2_t:.4f}")
    
    with open('outputs/ann_surrogate_v2.pkl', 'wb') as f:
        pickle.dump({
            'ann_v': ann_v, 'ann_t': ann_t,
            'scaler_X': scaler_X,
            'scaler_yv': scaler_yv, 'scaler_yt': scaler_yt,
            'param_names': param_names
        }, f)
    
    return ann_v, ann_t, scaler_X, scaler_yv, scaler_yt, param_names

class MultiObjectiveGA:
    def __init__(self, ann_v, ann_t, scaler_X, scaler_yv, scaler_yt, param_names, bounds,
                 pop_size=100, n_generations=150,
                 crossover_rate=0.8, mutation_rate=0.15):
        self.ann_v = ann_v
        self.ann_t = ann_t
        self.scaler_X = scaler_X
        self.scaler_yv = scaler_yv
        self.scaler_yt = scaler_yt
        self.param_names = param_names
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_params = len(param_names)
    
    def predict_curves(self, params_array):
        params_s = self.scaler_X.transform(params_array)
        v_s = self.ann_v.predict(params_s)
        t_s = self.ann_t.predict(params_s)
        voltage = self.scaler_yv.inverse_transform(v_s)
        temperature = self.scaler_yt.inverse_transform(t_s)
        return voltage, temperature
    
    def evaluate_fitness(self, population, target_voltage, target_temp):
        pred_v, pred_t = self.predict_curves(population)
        
        # Voltage RMSE
        f1 = np.sqrt(np.mean((pred_v - target_voltage)**2, axis=1))
        # Temperature RMSE
        f2 = np.sqrt(np.mean((pred_t - target_temp)**2, axis=1))
        
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
            contestants = np.random.choice(len(ranks), 5, replace=False)
            best = contestants[0]
            for c in contestants[1:]:
                if ranks[c] < ranks[best] or (ranks[c] == ranks[best] and distances.get(c, 0) > distances.get(best, 0)):
                    best = c
            selected.append(best)
        return selected
    
    def sbx_crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        eta = 20.0
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        for i in range(self.n_params):
            if np.random.rand() <= 0.5 and abs(parent1[i] - parent2[i]) > 1e-14:
                if parent1[i] < parent2[i]:
                    y1, y2 = parent1[i], parent2[i]
                else:
                    y1, y2 = parent2[i], parent1[i]
                lo, hi = self.bounds[self.param_names[i]]
                beta = 1.0 + (2.0 * (y1 - lo) / (y2 - y1))
                alpha = 2.0 - beta**(-(eta + 1))
                rand = np.random.rand()
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha)**(1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                beta = 1.0 + (2.0 * (hi - y2) / (y2 - y1))
                alpha = 2.0 - beta**(-(eta + 1))
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha)**(1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                child1[i] = np.clip(c1, lo, hi)
                child2[i] = np.clip(c2, lo, hi)
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        return child1, child2
    
    def polynomial_mutation(self, individual):
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
    
    def optimize(self, target_voltage, target_temp):
        print(f"\nStarting MMGA optimization...")
        print(f"Population: {self.pop_size}, Generations: {self.n_generations}")
        
        population = np.zeros((self.pop_size, self.n_params))
        for i, name in enumerate(self.param_names):
            lo, hi = self.bounds[name]
            population[:, i] = np.random.uniform(lo, hi, self.pop_size)
        
        best_f1_history = []
        best_f2_history = []
        
        for gen in range(self.n_generations):
            f1, f2 = self.evaluate_fitness(population, target_voltage, target_temp)
            ranks, fronts = self.non_dominated_sort(f1, f2)
            distances = {}
            for front in fronts:
                cd = self.crowding_distance(f1, f2, front)
                distances.update(cd)
            
            best_f1_history.append(float(np.min(f1)))
            best_f2_history.append(float(np.min(f2)))
            
            if gen % 20 == 0:
                print(f"  Gen {gen}: best f1={np.min(f1):.4f}, best f2={np.min(f2):.4f}, front0={len(fronts[0])}")
            
            selected_idx = self.tournament_selection(ranks, distances)
            parents = population[selected_idx]
            
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1 = parents[i % len(parents)]
                p2 = parents[(i + 1) % len(parents)]
                c1, c2 = self.sbx_crossover(p1, p2)
                c1 = self.polynomial_mutation(c1)
                c2 = self.polynomial_mutation(c2)
                offspring.append(c1)
                offspring.append(c2)
            offspring = np.array(offspring[:self.pop_size])
            
            combined = np.vstack([population, offspring])
            f1_c, f2_c = self.evaluate_fitness(combined, target_voltage, target_temp)
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
        
        f1, f2 = self.evaluate_fitness(population, target_voltage, target_temp)
        ranks, fronts = self.non_dominated_sort(f1, f2)
        pareto_front = population[fronts[0]]
        pareto_f1 = f1[fronts[0]]
        pareto_f2 = f2[fronts[0]]
        
        distances_to_origin = np.sqrt(pareto_f1**2 + pareto_f2**2)
        best_idx = np.argmin(distances_to_origin)
        best_params = pareto_front[best_idx]
        
        return best_params, pareto_front, pareto_f1, pareto_f2, best_f1_history, best_f2_history


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
                    nasa_data[bname] = {'time': time, 'voltage': voltage, 'current': current, 'temperature': temp}
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


def main():
    print("="*60)
    print("MMGA Parameter Identification Framework v2")
    print("="*60)
    
    # Step 1: Generate training data
    X, y_voltage, y_temp, param_names = generate_training_data(n_samples=3000)
    np.savez('outputs/training_data_v2.npz', X=X, y_voltage=y_voltage, y_temp=y_temp, param_names=param_names)
    
    # Step 2: Train ANN surrogate
    ann_v, ann_t, scaler_X, scaler_yv, scaler_yt, param_names = train_ann_surrogate(X, y_voltage, y_temp, param_names)
    
    # Step 3: Load experimental data
    datasets = load_experimental_data()
    
    # Step 4: Parameter identification
    bounds = get_param_bounds()
    results = {}
    
    for dataset_name, data_dict in datasets.items():
        print(f"\n{'='*40}")
        print(f"Processing {dataset_name} dataset...")
        print(f"{'='*40}")
        
        dataset_results = {}
        for cell_name, exp_data in data_dict.items():
            print(f"\n  Cell: {cell_name}")
            
            # Interpolate experimental data to fixed time points
            target_voltage = interpolate_to_fixed_time(exp_data['time'], exp_data['voltage'])
            target_temp = interpolate_to_fixed_time(exp_data['time'], exp_data['temperature'])
            
            moga = MultiObjectiveGA(
                ann_v, ann_t, scaler_X, scaler_yv, scaler_yt, param_names, bounds,
                pop_size=100, n_generations=150,
                crossover_rate=0.8, mutation_rate=0.15
            )
            
            best_params, pareto_front, pareto_f1, pareto_f2, f1_hist, f2_hist = moga.optimize(target_voltage, target_temp)
            
            default = get_default_params()
            identified = default.copy()
            for i, name in enumerate(param_names):
                identified[name] = best_params[i]
            
            # Evaluate with actual model
            model = BatteryDischargeModel(identified)
            simulated = model.solve(t_end=3600, dt=30)
            
            # Compute RMSE
            sim_v_interp = interpolate_to_fixed_time(simulated['time'], simulated['voltage'])
            sim_t_interp = interpolate_to_fixed_time(simulated['time'], simulated['temperature'])
            rmse_v = np.sqrt(np.mean((sim_v_interp - target_voltage)**2))
            rmse_t = np.sqrt(np.mean((sim_t_interp - target_temp)**2))
            
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
                'rmse_voltage': float(rmse_v),
                'rmse_temperature': float(rmse_t),
            }
            
            print(f"  RMSE V={rmse_v:.4f}V, T={rmse_t:.2f}°C")
            print(f"  Params: {dict(zip(param_names, [float(v) for v in best_params]))}")
        
        results[dataset_name] = dataset_results
    
    with open('outputs/identification_results_v2.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    generate_figures(results, param_names, X, y_voltage, ann_v, scaler_X, scaler_yv)
    
    print("\n" + "="*60)
    print("MMGA Framework v2 complete!")
    print("="*60)


def generate_figures(results, param_names, X, y_voltage, ann_v, scaler_X, scaler_yv):
    print("\nGenerating figures...")
    
    # Figure 1: Pareto fronts
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    idx = 0
    for dataset_name, dataset_results in results.items():
        if idx >= 4: break
        ax = axes[idx]
        for cell_name, res in dataset_results.items():
            ax.scatter(res['pareto_f1'], res['pareto_f2'], s=30, alpha=0.6, label=cell_name)
        ax.set_xlabel('Voltage RMSE (V)')
        ax.set_ylabel('Temperature RMSE (°C)')
        ax.set_title(f'{dataset_name}: Pareto Front')
        ax.legend(fontsize=8)
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
        if idx >= 4: break
        ax = axes[idx]
        for cell_name, res in dataset_results.items():
            ax.plot(res['f1_history'], label=f'{cell_name}', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Voltage RMSE (V)')
        ax.set_title(f'{dataset_name}: Convergence')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
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
        if idx >= 4: break
        ax = axes[idx]
        for cell_name, res in dataset_results.items():
            exp = res['experimental']
            sim = res['simulated']
            ax.plot(exp['time'], exp['voltage'], 'b-', alpha=0.7, linewidth=1.5, label=f'{cell_name} (Exp)')
            ax.plot(sim['time'], sim['voltage'], 'r--', alpha=0.7, linewidth=1.5, label=f'{cell_name} (Sim)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'{dataset_name}: Voltage Comparison')
        ax.legend(fontsize=7)
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
        if idx >= 4: break
        ax = axes[idx]
        for cell_name, res in dataset_results.items():
            exp = res['experimental']
            sim = res['simulated']
            if 'temperature' in exp and np.max(exp['temperature']) > 20:
                ax.plot(exp['time'], exp['temperature'], 'b-', alpha=0.7, linewidth=1.5, label=f'{cell_name} (Exp)')
            ax.plot(sim['time'], sim['temperature'], 'r--', alpha=0.7, linewidth=1.5, label=f'{cell_name} (Sim)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'{dataset_name}: Temperature Comparison')
        ax.legend(fontsize=7)
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
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('report/images/fig5_parameter_comparison.png', dpi=150)
    plt.close()
    
    # Figure 6: ANN accuracy - sample predictions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    idx_test = np.random.choice(len(X), 6, replace=False)
    for i, idx in enumerate(idx_test):
        ax = axes[i]
        true_curve = y_voltage[idx]
        pred_curve = scaler_yv.inverse_transform(ann_v.predict(scaler_X.transform(X[idx:idx+1])))[0]
        ax.plot(TIME_POINTS, true_curve, 'b-', label='True')
        ax.plot(TIME_POINTS, pred_curve, 'r--', label='ANN Pred')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Sample {idx}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig6_ann_accuracy.png', dpi=150)
    plt.close()
    
    # Figure 7: LHS distribution
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.log10(X[:, 0]), np.log10(X[:, 2]), s=5, alpha=0.5)
    ax.set_xlabel(f'log10({param_names[0]})')
    ax.set_ylabel(f'log10({param_names[2]})')
    ax.set_title('LHS Sampling Distribution')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/fig7_lhs_distribution.png', dpi=150)
    plt.close()
    
    # Figure 8: RMSE comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    rmse_data = []
    labels = []
    for dataset_name, dataset_results in results.items():
        for cell_name, res in dataset_results.items():
            rmse_data.append([res['rmse_voltage'], res['rmse_temperature']])
            labels.append(f'{dataset_name}\n{cell_name}')
    rmse_data = np.array(rmse_data)
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, rmse_data[:, 0], width, label='Voltage RMSE (V)', alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, rmse_data[:, 1], width, label='Temp RMSE (°C)', alpha=0.7, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Voltage RMSE (V)', color='blue')
    ax2.set_ylabel('Temperature RMSE (°C)', color='orange')
    ax.set_title('Identification Accuracy Across Datasets')
    ax.grid(True, alpha=0.3, axis='y')
    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('report/images/fig8_rmse_comparison.png', dpi=150)
    plt.close()
    
    print("Figures generated successfully.")


if __name__ == '__main__':
    main()
