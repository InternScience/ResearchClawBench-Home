import numpy as np
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import qmc

def load_cs2_36_data():
    df = pd.read_excel('data/CS2_36/CS2_36_1_10_11.xlsx', sheet_name='Channel_1-009')
    discharge = df[df['Step_Index'] == 7].copy()
    discharge['Time_s'] = discharge['Test_Time(s)'] - discharge['Test_Time(s)'].min()
    return discharge['Time_s'].values, discharge['Current(A)'].values, discharge['Voltage(V)'].values

time, current, voltage = load_cs2_36_data()

idx = np.linspace(0, len(time)-1, 200).astype(int)
time_ds = time[idx]
current_ds = current[idx]
voltage_ds = voltage[idx]

def simulate_ecat(params, t, i):
    R_int, C_dl, R_ct, E0_shift, k_aging, C_th, R_th = params
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    
    V_dl = 0.0
    SOC = 1.0
    Capacity = 3600
    
    v_sim = np.zeros_like(t)
    
    for j in range(len(t)):
        I = i[j]
        SOC += I * dt[j] / Capacity
        SOC = max(0.0, min(1.0, SOC))
        
        OCV = 3.6 + 0.5 * SOC + E0_shift - 0.1 * np.exp(-20 * SOC)
        
        # Prevent V_dl overflow
        dV_dl = (I - V_dl / R_ct) / C_dl
        V_dl += dV_dl * dt[j]
        # Clip V_dl to realistic bounds
        V_dl = np.clip(V_dl, -1.0, 1.0)
        
        V = OCV + I * R_int + V_dl
        v_sim[j] = V
        
    return v_sim

num_samples = 2000
num_params = 7
sampler = qmc.LatinHypercube(d=num_params)
sample = sampler.random(n=num_samples)

bounds = np.array([
    [0.01, 0.2],     # R_int
    [100, 2000],     # C_dl
    [0.01, 0.5],     # R_ct
    [-0.5, 0.5],     # E0_shift
    [1e-6, 1e-4],    # k_aging
    [50, 200],       # C_th
    [1, 10]          # R_th
])

X_train = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
Y_train = np.zeros((num_samples, len(time_ds)))

for i in range(num_samples):
    Y_train[i] = simulate_ecat(X_train[i], time_ds, current_ds)

ann = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
ann.fit(X_train, Y_train)

def objective(params):
    v_pred = ann.predict(params.reshape(1, -1))[0]
    rmse = np.sqrt(np.mean((v_pred - voltage_ds)**2))
    return rmse

result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=100, popsize=15, tol=1e-4, mutation=(0.5, 1), recombination=0.7, seed=42)

best_params = result.x
print("Identified Parameters:", best_params)

v_sim_best = simulate_ecat(best_params, time_ds, current_ds)

plt.figure(figsize=(8, 5))
plt.plot(time_ds, voltage_ds, label='Experimental (CS2_36)', color='black', linewidth=2)
plt.plot(time_ds, v_sim_best, label='Simulated (ECAT Identified)', color='red', linestyle='dashed')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Parameter Identification Result on CS2_36 (1C Discharge)')
plt.legend()
plt.grid()
plt.savefig('report/images/cs2_36_identification.png')

np.savetxt('outputs/identified_parameters.txt', best_params)
