"""
MMGA Parameter Identification Framework for Li-ion Battery Digital Twins
- Loads CS2_36 experimental discharge data
- Performs LHS sampling of ECAT parameters
- Trains ANN surrogate model
- Runs GA optimization to identify high-fidelity parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ECAT model parameter search space (example bounds from literature)
PARAM_BOUNDS = {
    'particle_radius': (5e-6, 15e-6),      # m
    'reaction_rate': (1e-11, 1e-9),        # m/s
    'thermal_coeff': (0.5, 2.0),           # W/mK
    'diffusion_coeff': (1e-14, 1e-12),     # m^2/s
    'conductivity': (0.1, 10.0)            # S/m
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

def load_cs2_data():
    """Load and preprocess CS2_36 1C discharge data"""
    df = pd.read_excel('data/CS2_36/CS2_36_1_10_11.xlsx', sheet_name='Channel_1-009')
    # Filter for discharge segments (Step_Index == 2 for typical CC discharge)
    discharge = df[df['Step_Index'] == 2].copy()
    # Normalize time and extract key curves
    discharge['t_norm'] = (discharge['Test_Time(s)'] - discharge['Test_Time(s)'].min()) / \
                          (discharge['Test_Time(s)'].max() - discharge['Test_Time(s)'].min())
    return discharge[['t_norm', 'Voltage(V)', 'Current(A)', 'Discharge_Capacity(Ah)']].dropna()

def latin_hypercube_sampling(n_samples=500):
    """Generate LHS samples in normalized [0,1] space and scale to bounds"""
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES))
    samples_norm = sampler.random(n=n_samples)
    samples = {}
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        samples[name] = low + samples_norm[:, i] * (high - low)
    return pd.DataFrame(samples)

def simulate_ecat(params, t):
    """
    Simplified ECAT surrogate simulation (voltage curve).
    In real implementation this would call the full physics solver.
    Here we use a physics-inspired empirical model for demonstration.
    """
    R = params['particle_radius']
    k = params['reaction_rate']
    alpha = params['thermal_coeff']
    D = params['diffusion_coeff']
    sigma = params['conductivity']

    # Empirical voltage model (OCV + overpotentials)
    ocv = 4.2 - 1.2 * t**0.6
    eta_ohm = (1.0 / sigma) * 0.5
    eta_act = (8.314 * 298 / (0.5 * 96485)) * np.arcsinh(0.5 / (2 * k * np.sqrt(R)))
    eta_diff = (0.1 / D) * t**1.2
    thermal = alpha * 0.02 * t

    v = ocv - eta_ohm - eta_act - eta_diff + thermal
    v = np.clip(v, 2.5, 4.2)
    return v

def generate_training_data(n_samples=500):
    """Generate synthetic training data using the surrogate simulator"""
    params_df = latin_hypercube_sampling(n_samples)
    t = np.linspace(0, 1, 100)
    X, y = [], []
    for _, p in params_df.iterrows():
        v_curve = simulate_ecat(p.to_dict(), t)
        X.append(p.values)
        y.append(v_curve)
    return np.array(X), np.array(y), t

def train_ann(X, y):
    """Train ANN meta-model"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = MLPRegressor(hidden_layer_sizes=(64, 64, 32),
                         activation='relu',
                         solver='adam',
                         max_iter=500,
                         random_state=42,
                         early_stopping=True)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"ANN validation R^2: {val_score:.4f}")
    return model, scaler

def objective(params, model, scaler, t, target_v):
    """Objective: RMSE between ANN surrogate voltage and target"""
    X_in_scaled = scaler.transform(params.reshape(1, -1))
    pred_v = model.predict(X_in_scaled).ravel()
    return np.sqrt(np.mean((pred_v - target_v)**2))

def genetic_algorithm(model, scaler, t, target_v, pop_size=50, generations=30):
    """Simple GA implementation"""
    n_params = len(PARAM_NAMES)
    pop = np.random.rand(pop_size, n_params)
    for gen in range(generations):
        fitness = np.array([objective(p, model, scaler, t, target_v) for p in pop])
        idx = np.argsort(fitness)
        pop = pop[idx]
        elites = pop[:10]
        # Crossover & mutation
        offspring = []
        for _ in range(pop_size - 10):
            p1, p2 = elites[np.random.choice(10, 2, replace=False)]
            child = (p1 + p2) / 2
            child += np.random.normal(0, 0.05, n_params)
            child = np.clip(child, 0, 1)
            offspring.append(child)
        pop = np.vstack([elites, offspring])
    best_idx = np.argmin([objective(p, model, scaler, t, target_v) for p in pop])
    return pop[best_idx]

def main():
    print("Loading CS2_36 experimental data...")
    data = load_cs2_data()
    t_orig = data['t_norm'].values
    v_orig = data['Voltage(V)'].values

    # Resample experimental data to training grid (100 points) for consistency with ANN surrogate
    t_sim = np.linspace(0, 1, 100)
    target_voltage = np.interp(t_sim, t_orig, v_orig)

    print("Generating LHS samples and training data...")
    X, y, t_sim = generate_training_data(600)

    print("Training ANN surrogate model...")
    ann_model, scaler = train_ann(X, y)

    print("Running Genetic Algorithm for parameter identification...")
    best_params_norm = genetic_algorithm(ann_model, scaler, t_sim, target_voltage)
    best_params = {}
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        best_params[name] = low + best_params_norm[i] * (high - low)

    print("\nIdentified high-fidelity parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6e}")

    # Generate comparison figure
    plt.figure(figsize=(8, 5))
    pred_v = ann_model.predict(scaler.transform([best_params_norm]))[0].ravel()
    plt.plot(t_sim, target_voltage, 'b-', label='Experimental (CS2_36)')
    plt.plot(t_sim, pred_v, 'r--', label='MMGA + ANN prediction')
    plt.xlabel('Normalized Time')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.title('MMGA Parameter Identification Validation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/figure1_identification.png', dpi=150)
    print("\nFigure saved to report/images/figure1_identification.png")

    # Save parameters
    pd.DataFrame([best_params]).to_csv('outputs/identified_parameters.csv', index=False)
    print("Parameters saved to outputs/identified_parameters.csv")

if __name__ == "__main__":
    main()
