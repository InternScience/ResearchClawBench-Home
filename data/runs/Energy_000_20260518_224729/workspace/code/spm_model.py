"""
Simplified Single Particle Model (SPM) with thermal coupling for Li-ion batteries.
Generates synthetic voltage curves for parameter identification framework.
"""
import numpy as np
from scipy.integrate import solve_ivp

F = 96485.33212
R_gas = 8.314

class SingleParticleModel:
    def __init__(self, params):
        self.params = params
        
    def temperature_dependent_k(self, T, electrode='neg'):
        Ea = self.params['Ea_k']
        T_ref = self.params['T_ref']
        key = f'k_{electrode}_ref'
        k_ref = self.params[key]
        return k_ref * np.exp(-Ea/R_gas * (1/T - 1/T_ref))
    
    def solve_discharge(self, t_end, dt=1.0, method='BDF'):
        p = self.params
        
        sto_0_neg = p['stoichiometry_neg_0']
        sto_1_neg = p['stoichiometry_neg_1']
        sto_0_pos = p['stoichiometry_pos_0']
        sto_1_pos = p['stoichiometry_pos_1']
        
        cs_max_neg = p['cs_max_neg']
        cs_max_pos = p['cs_max_pos']
        
        x_neg_0 = sto_1_neg
        x_pos_0 = sto_0_pos
        T0 = p['T_amb']
        
        C_nom = 1.1 * 3600
        i_app = abs(p['i0_app'])
        
        # Improved OCV functions
        def ocv_neg(sto):
            sto = np.clip(sto, 0.001, 0.999)
            return (0.7222 + 0.1387*sto + 0.029*np.sqrt(sto) - 0.0172/sto + 
                    0.0019/(sto**1.5) + 0.2808*np.exp(0.9 - 15*sto) - 
                    0.7984*np.exp(0.4465*sto - 0.4108))
        
        def ocv_pos(sto):
            sto = np.clip(sto, 0.001, 0.999)
            return (4.345 - 1.6518*sto + 1.6225*(sto**2) - 2.084*(sto**3) + 
                    3.5146*(sto**4) - 2.2166*(sto**5) + 0.2813*(sto**6) - 
                    0.1085*(sto**7) + 0.1414*(sto**8))
        
        def dynamics(t, y):
            x_neg, x_pos, T = y
            x_neg = np.clip(x_neg, sto_0_neg, sto_1_neg)
            x_pos = np.clip(x_pos, sto_0_pos, sto_1_pos)
            
            dx_neg_dt = -i_app / C_nom * (sto_1_neg - sto_0_neg)
            dx_pos_dt = i_app / C_nom * (sto_1_pos - sto_0_pos)
            
            k_neg = self.temperature_dependent_k(T, 'neg')
            k_pos = self.temperature_dependent_k(T, 'pos')
            
            i0_neg = F * k_neg * np.sqrt(p['ce0'] * max(x_neg, 0.001) * cs_max_neg * (cs_max_neg - max(x_neg, 0.001) * cs_max_neg))
            i0_pos = F * k_pos * np.sqrt(p['ce0'] * max(x_pos, 0.001) * cs_max_pos * (cs_max_pos - max(x_pos, 0.001) * cs_max_pos))
            
            eta_neg = -R_gas * T / (0.5 * F) * np.arcsinh(i_app / (2 * max(i0_neg, 1e-10)))
            eta_pos = R_gas * T / (0.5 * F) * np.arcsinh(i_app / (2 * max(i0_pos, 1e-10)))
            
            R_ohm = p.get('R_ohm', 0.015)
            q_gen = i_app * (abs(eta_neg) + abs(eta_pos) + i_app * R_ohm)
            q_conv = p['h_conv'] * p['A_surf'] * (T - p['T_amb'])
            dT_dt = (q_gen - q_conv) / (p['rho'] * p['Cp'] * p['V_cell'])
            
            return [dx_neg_dt, dx_pos_dt, dT_dt]
        
        t_span = (0, t_end)
        t_eval = np.arange(0, t_end + dt, dt)
        
        sol = solve_ivp(dynamics, t_span, [x_neg_0, x_pos_0, T0], 
                       t_eval=t_eval, method=method, max_step=dt*2)
        
        times = sol.t
        voltages = []
        temps = []
        for i in range(len(times)):
            x_neg = np.clip(sol.y[0, i], sto_0_neg, sto_1_neg)
            x_pos = np.clip(sol.y[1, i], sto_0_pos, sto_1_pos)
            T = sol.y[2, i]
            
            U_neg = (0.7222 + 0.1387*x_neg + 0.029*np.sqrt(x_neg) - 0.0172/x_neg + 
                    0.0019/(x_neg**1.5) + 0.2808*np.exp(0.9 - 15*x_neg) - 
                    0.7984*np.exp(0.4465*x_neg - 0.4108))
            U_pos = (4.345 - 1.6518*x_pos + 1.6225*(x_pos**2) - 2.084*(x_pos**3) + 
                    3.5146*(x_pos**4) - 2.2166*(x_pos**5) + 0.2813*(x_pos**6) - 
                    0.1085*(x_pos**7) + 0.1414*(x_pos**8))
            
            k_neg = self.temperature_dependent_k(T, 'neg')
            k_pos = self.temperature_dependent_k(T, 'pos')
            i0_neg = F * k_neg * np.sqrt(p['ce0'] * max(x_neg, 0.001) * cs_max_neg * (cs_max_neg - max(x_neg, 0.001) * cs_max_neg))
            i0_pos = F * k_pos * np.sqrt(p['ce0'] * max(x_pos, 0.001) * cs_max_pos * (cs_max_pos - max(x_pos, 0.001) * cs_max_pos))
            
            eta_neg = -R_gas * T / (0.5 * F) * np.arcsinh(i_app / (2 * max(i0_neg, 1e-10)))
            eta_pos = R_gas * T / (0.5 * F) * np.arcsinh(i_app / (2 * max(i0_pos, 1e-10)))
            
            R_ohm = p.get('R_ohm', 0.015)
            V = U_pos - U_neg - eta_pos + eta_neg - i_app * R_ohm
            voltages.append(V)
            temps.append(T)
        
        return {
            'time': times,
            'voltage': np.array(voltages),
            'temperature': np.array(temps),
            'soc_neg': sol.y[0, :],
            'soc_pos': sol.y[1, :]
        }


def get_default_params():
    return {
        'Rs_neg': 2e-6,
        'Rs_pos': 2e-6,
        'Ds_neg_ref': 3.9e-14,
        'Ds_pos_ref': 1.0e-13,
        'k_neg_ref': 5.03e-11,
        'k_pos_ref': 2.33e-11,
        'cs_max_neg': 30555,
        'cs_max_pos': 51554,
        'ce0': 1000,
        'T_ref': 298.15,
        'Ea_Ds': 5000,
        'Ea_k': 5000,
        'h_conv': 10,
        'rho': 1626,
        'Cp': 750,
        'A_surf': 0.0045,
        'V_cell': 1.7e-5,
        'L_neg': 50e-6,
        'L_sep': 25e-6,
        'L_pos': 50e-6,
        'eps_neg': 0.3,
        'eps_pos': 0.3,
        'stoichiometry_neg_0': 0.01,
        'stoichiometry_neg_1': 0.8,
        'stoichiometry_pos_0': 0.4,
        'stoichiometry_pos_1': 0.95,
        'i0_app': 1.1,
        'T_amb': 298.15,
        'R_ohm': 0.015,
    }


if __name__ == '__main__':
    params = get_default_params()
    spm = SingleParticleModel(params)
    result = spm.solve_discharge(t_end=3600, dt=10)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(result['time'], result['voltage'])
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_ylim([2.5, 4.5])
    axes[1].plot(result['time'], result['temperature'] - 273.15)
    axes[1].set_ylabel('Temperature (°C)')
    axes[2].plot(result['time'], result['soc_neg'], label='Anode SOC')
    axes[2].plot(result['time'], result['soc_pos'], label='Cathode SOC')
    axes[2].set_ylabel('SOC')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig('report/images/spm_test.png', dpi=150)
    plt.close()
    print("SPM test complete. V range:", result['voltage'].min(), "to", result['voltage'].max())
