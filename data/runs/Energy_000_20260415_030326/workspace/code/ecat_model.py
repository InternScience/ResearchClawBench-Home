"""
MMGA Framework: ANN Meta-Model Guided Genetic Algorithm for ECAT Parameter Identification
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# OCP Functions
_NMC_STO = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
_NMC_VOLT = np.array([4.35, 4.25, 4.22, 4.15, 4.05, 3.95, 3.85, 3.75, 3.65, 3.55, 3.40, 3.20, 3.05, 2.80, 2.50])
_GRAPH_STO = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
_GRAPH_VOLT = np.array([1.50, 0.85, 0.35, 0.22, 0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.02])
_OCP_NMC_FUNC = interp1d(_NMC_STO, _NMC_VOLT, kind='linear', fill_value='extrapolate')
_OCP_GRAPH_FUNC = interp1d(_GRAPH_STO, _GRAPH_VOLT, kind='linear', fill_value='extrapolate')

def OCP_negative(sto):
    return float(_OCP_GRAPH_FUNC(np.clip(sto, 0.001, 0.999)))

def OCP_positive(sto):
    return float(_OCP_NMC_FUNC(np.clip(sto, 0.001, 0.999)))


class ECATModel:
    """Simplified Electrochemical-Aging-Thermal coupled model (SPM + thermal)."""
    
    def __init__(self, params):
        self.p = params
        self.F = 96485.0
        self.R_gas = 8.314
        
    def simulate_discharge(self, I_app, t_end, T_amb=298.15, n_points=200,
                           V_cutoff=2.5, x_n_0=None, x_p_0=None):
        """Simulate CC discharge using SPM + thermal model."""
        p = self.p
        a_n = 3 * p['eps_s_n'] / p['R_p_n']
        a_p = 3 * p['eps_s_p'] / p['R_p_p']
        j_n = I_app / (self.F * p['A_cell'] * p['L_n'] * a_n) if a_n * p['L_n'] > 0 else 0
        
        # Initial conditions (adjustable)
        if x_n_0 is None:
            x_n_0 = p.get('x_n_0', 0.80)
        if x_p_0 is None:
            x_p_0 = p.get('x_p_0', 0.05)
        T_0 = T_amb
        
        y0 = [x_n_0, x_p_0, T_0]
        t_eval = np.linspace(0, t_end, n_points)
        
        def rhs(t, y):
            x_n, x_p, T = y
            x_n = np.clip(x_n, 0.002, 0.998)
            x_p = np.clip(x_p, 0.002, 0.998)
            
            U_n = OCP_negative(x_n)
            U_p = OCP_positive(x_p)
            
            c_e_ref = 1000.0
            i_0_n = self.F * p['k_n'] * c_e_ref**0.5 * p['c_s_max_n'] * (x_n**0.5) * ((1-x_n)**0.5)
            i_0_p = self.F * p['k_p'] * c_e_ref**0.5 * p['c_s_max_p'] * (x_p**0.5) * ((1-x_p)**0.5)
            
            i_app_n = I_app / (p['A_cell'] * p['L_n'] * a_n) if a_n * p['L_n'] > 0 else 0
            i_app_p = I_app / (p['A_cell'] * p['L_p'] * a_p) if a_p * p['L_p'] > 0 else 0
            
            alpha = 0.5
            eta_n = (self.R_gas * T / (alpha * self.F)) * np.arcsinh(i_app_n / (2 * i_0_n + 1e-20))
            eta_p = (self.R_gas * T / (alpha * self.F)) * np.arcsinh(i_app_p / (2 * i_0_p + 1e-20))
            eta_SEI = i_app_n * p['R_SEI_0']
            L_total = p['L_n'] + p['L_sep'] + p['L_p']
            phi_e_drop = (I_app / p['A_cell']) * L_total / (p['kappa_e'] + 1e-10)
            
            V = U_p - U_n + eta_p - eta_n - eta_SEI - phi_e_drop - I_app * p['R_internal']
            
            dx_n_dt = -j_n / (p['c_s_max_n'] * p['R_p_n'] / 3)
            dx_p_dt = j_n / (p['c_s_max_p'] * p['R_p_p'] / 3)
            
            Q_gen = I_app * (U_p - U_n - V)
            Q_conv = p['h_conv'] * p['A_surf'] * (T - T_amb)
            dT_dt = (Q_gen - Q_conv) / (p['m_cell'] * p['C_p'])
            
            return [dx_n_dt, dx_p_dt, dT_dt]
        
        def voltage_cutoff(t, y):
            x_n = np.clip(y[0], 0.002, 0.998)
            x_p = np.clip(y[1], 0.002, 0.998)
            V_est = OCP_positive(x_p) - OCP_negative(x_n) - 0.15
            return V_est - V_cutoff
        voltage_cutoff.terminal = True
        voltage_cutoff.direction = -1
        
        def x_n_min(t, y):
            return y[0] - 0.005
        x_n_min.terminal = True
        x_n_min.direction = -1
        
        try:
            sol = solve_ivp(rhs, [0, t_end], y0, t_eval=t_eval,
                          method='RK45', rtol=1e-6, atol=1e-9,
                          events=[voltage_cutoff, x_n_min])
            
            t = sol.t
            x_n = np.clip(sol.y[0], 0.002, 0.998)
            x_p = np.clip(sol.y[1], 0.002, 0.998)
            T = sol.y[2]
            
            V = np.zeros_like(t)
            for i in range(len(t)):
                U_n = OCP_negative(x_n[i])
                U_p = OCP_positive(x_p[i])
                i_0_n = self.F * p['k_n'] * 1000.0**0.5 * p['c_s_max_n'] * (x_n[i]**0.5) * ((1-x_n[i])**0.5)
                i_0_p = self.F * p['k_p'] * 1000.0**0.5 * p['c_s_max_p'] * (x_p[i]**0.5) * ((1-x_p[i])**0.5)
                i_app_n = I_app / (p['A_cell'] * p['L_n'] * a_n) if a_n * p['L_n'] > 0 else 0
                i_app_p = I_app / (p['A_cell'] * p['L_p'] * a_p) if a_p * p['L_p'] > 0 else 0
                alpha = 0.5
                eta_n = (self.R_gas * T[i] / (alpha * self.F)) * np.arcsinh(i_app_n / (2 * i_0_n + 1e-20))
                eta_p = (self.R_gas * T[i] / (alpha * self.F)) * np.arcsinh(i_app_p / (2 * i_0_p + 1e-20))
                eta_SEI = i_app_n * p['R_SEI_0']
                L_total = p['L_n'] + p['L_sep'] + p['L_p']
                phi_e_drop = (I_app / p['A_cell']) * L_total / (p['kappa_e'] + 1e-10)
                V[i] = U_p - U_n + eta_p - eta_n - eta_SEI - phi_e_drop - I_app * p['R_internal']
            
            capacity = I_app * t / 3600.0
            return {'time': t, 'voltage': V, 'temperature': T, 'capacity': capacity,
                    'x_n': x_n, 'x_p': x_p, 'success': True}
        except Exception as e:
            return {'time': np.array([0]), 'voltage': np.array([4.2]),
                    'temperature': np.array([298.15]), 'capacity': np.array([0]),
                    'x_n': np.array([0.8]), 'x_p': np.array([0.05]),
                    'success': False, 'error': str(e)}


def get_default_params():
    """Default parameters for NMC/Graphite 18650 cell."""
    return {
        'R_p_n': 5.0e-6, 'R_p_p': 4.0e-6,
        'D_s_n': 3.9e-14, 'D_s_p': 1.0e-13,
        'k_n': 2.0e-11, 'k_p': 5.0e-11,
        'c_s_max_n': 30555, 'c_s_max_p': 51555,
        'L_n': 70e-6, 'L_p': 60e-6, 'L_sep': 25e-6,
        'eps_s_n': 0.58, 'eps_s_p': 0.50,
        'kappa_e': 1.0, 'R_SEI_0': 0.01,
        'R_internal': 0.03,
        'h_conv': 12.0, 'C_p': 900.0,
        'm_cell': 0.045, 'A_cell': 0.04, 'A_surf': 0.004,
        'x_n_0': 0.80, 'x_p_0': 0.05,
    }

def get_parameter_bounds():
    """Define parameter bounds for identification."""
    return {
        'R_p_n': (2.0e-6, 10.0e-6),
        'R_p_p': (1.5e-6, 8.0e-6),
        'D_s_n': (1.0e-14, 1.0e-13),
        'D_s_p': (1.0e-14, 5.0e-13),
        'k_n': (1.0e-12, 1.0e-10),
        'k_p': (1.0e-12, 5.0e-10),
        'kappa_e': (0.3, 2.0),
        'R_SEI_0': (0.001, 0.05),
        'R_internal': (0.01, 0.08),
        'h_conv': (5.0, 30.0),
        'C_p': (500.0, 1500.0),
    }

def get_identifiable_params():
    return list(get_parameter_bounds().keys())

def generate_lhs_samples(n_samples, param_bounds=None, seed=42):
    if param_bounds is None:
        param_bounds = get_parameter_bounds()
    param_names = list(param_bounds.keys())
    n_params = len(param_names)
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    samples = {}
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        samples[name] = qmc.scale(unit_samples[:, i:i+1], low, high).flatten()
    return samples, param_names

def extract_features(result, n_features=20):
    if not result['success']:
        return None
    V = result['voltage']
    T = result['temperature']
    cap = result['capacity']
    t = result['time']
    if len(V) < 5:
        return None
    t_norm = t / t[-1] if t[-1] > 0 else t
    t_interp = np.linspace(0, 1, n_features)
    V_interp = np.interp(t_interp, t_norm, V)
    T_interp = np.interp(t_interp, t_norm, T)
    features = np.concatenate([V_interp, T_interp,
        [np.mean(V), np.std(V), np.max(T), np.mean(T), cap[-1]]])
    return features

def run_lhs_simulations(samples, param_names, I_app, t_end, T_amb=298.15, n_points=100):
    n_samples = len(samples[param_names[0]])
    n_params = len(param_names)
    X = np.zeros((n_samples, n_params))
    feature_list = []
    valid_mask = np.ones(n_samples, dtype=bool)
    default = get_default_params()
    for i in range(n_samples):
        params = default.copy()
        for j, name in enumerate(param_names):
            params[name] = samples[name][i]
            X[i, j] = samples[name][i]
        model = ECATModel(params)
        result = model.simulate_discharge(I_app, t_end, T_amb, n_points, V_cutoff=2.5)
        features = extract_features(result)
        if features is not None:
            feature_list.append(features)
        else:
            feature_list.append(np.zeros(45))
            valid_mask[i] = False
    Y = np.array(feature_list)
    return X, Y, valid_mask

if __name__ == '__main__':
    params = get_default_params()
    model = ECATModel(params)
    result = model.simulate_discharge(1.1, 3600, 298.15, 200, V_cutoff=2.5)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"V: {result['voltage'].min():.4f} - {result['voltage'].max():.4f}")
        print(f"T: {result['temperature'].min():.2f} - {result['temperature'].max():.2f}")
        print(f"Cap: {result['capacity'][-1]:.4f} Ah")
