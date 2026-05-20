"""
Semi-empirical battery discharge model with physics-inspired parameters.
Generates synthetic voltage and temperature curves for MMGA framework training.
"""
import numpy as np

F = 96485.33212
R_gas = 8.314

class BatteryDischargeModel:
    """
    Semi-empirical discharge model that produces realistic Li-ion battery
    voltage and temperature curves based on physics-inspired parameters.
    """
    def __init__(self, params):
        self.params = params
    
    def solve(self, t_end=3600, dt=10):
        p = self.params
        
        Rs_neg = p['Rs_neg']
        Rs_pos = p['Rs_pos']
        Ds_neg = p['Ds_neg_ref']
        Ds_pos = p['Ds_pos_ref']
        k_neg = p['k_neg_ref']
        k_pos = p['k_pos_ref']
        R_ohm = p['R_ohm']
        h_conv = p['h_conv']
        T_amb = p['T_amb']
        i_app = abs(p['i0_app'])
        
        t = np.arange(0, t_end + dt, dt)
        tau = np.clip(t / t_end, 0, 1)
        
        # Kinetic factor: faster kinetics -> higher voltage
        k_factor = 0.85 + 0.15 * np.tanh(np.log10(k_neg / 1e-11) / 2) * np.tanh(np.log10(k_pos / 1e-11) / 2)
        
        # Diffusion limitation factor
        diff_factor = 1.0 + 0.4 * (Rs_neg / 2e-6) * np.sqrt(3.9e-14 / max(Ds_neg, 1e-16))
        
        # Base curve parameters tuned to match typical NMC/graphite discharge
        a = 4.18 * k_factor
        b = 0.12 * diff_factor
        c = 0.25
        d = 0.35 * diff_factor
        e = 8.0
        f = 0.8
        g = 0.15
        
        # Voltage curve with plateau and knee
        V_base = a - b*tau - c*tau**2 - d * np.exp(-e*(1-tau)) - f*tau**3 + g*tau**2*(1-tau)
        
        # Ohmic drop
        ohmic_drop = i_app * R_ohm * (1.0 + 0.5*tau)
        
        # Polarization
        pol = 0.08 * (1.0 + 0.5*tau**2) * (1.0 + 0.3 * np.exp(-k_neg / 5e-11))
        
        # Temperature-dependent voltage correction
        T_rise_est = i_app**2 * R_ohm / (h_conv * 0.0045)
        T_factor = 1.0 + 0.002 * T_rise_est
        
        V = V_base * T_factor - ohmic_drop - pol
        
        # Thermal model (lumped)
        mCp = p['rho'] * p['Cp'] * p['V_cell']
        q_gen = i_app**2 * R_ohm + i_app * pol
        tau_th = mCp / (h_conv * p['A_surf'])
        T = T_amb + q_gen / (h_conv * p['A_surf']) * (1.0 - np.exp(-t / tau_th))
        
        # Apply cutoff
        cutoff = p.get('cutoff_voltage', 2.7)
        V = np.clip(V, cutoff, 5.0)
        
        return {
            'time': t,
            'voltage': V,
            'temperature': T - 273.15,
            'tau': tau
        }


def get_param_bounds():
    """Parameter search space bounds for LHS sampling."""
    return {
        'Rs_neg': (1e-7, 1e-5),
        'Rs_pos': (1e-7, 1e-5),
        'Ds_neg_ref': (1e-15, 1e-12),
        'Ds_pos_ref': (1e-15, 1e-12),
        'k_neg_ref': (1e-12, 1e-9),
        'k_pos_ref': (1e-12, 1e-9),
        'R_ohm': (0.001, 0.1),
        'h_conv': (1, 50),
        'Ea_k': (1000, 20000),
    }


def get_default_params():
    return {
        'Rs_neg': 2e-6,
        'Rs_pos': 2e-6,
        'Ds_neg_ref': 3.9e-14,
        'Ds_pos_ref': 1.0e-13,
        'k_neg_ref': 5.03e-11,
        'k_pos_ref': 2.33e-11,
        'R_ohm': 0.015,
        'h_conv': 10,
        'Ea_k': 5000,
        'T_amb': 298.15,
        'i0_app': 1.1,
        'rho': 1626,
        'Cp': 750,
        'A_surf': 0.0045,
        'V_cell': 1.7e-5,
        'cutoff_voltage': 2.7,
    }


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    params = get_default_params()
    model = BatteryDischargeModel(params)
    result = model.solve(t_end=3600, dt=10)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(result['time'], result['voltage'])
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_ylim([2.5, 4.5])
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(result['time'], result['temperature'])
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/images/model_test.png', dpi=150)
    plt.close()
    
    print(f"Model test: V={result['voltage'].min():.3f}-{result['voltage'].max():.3f}, "
          f"T={result['temperature'].min():.1f}-{result['temperature'].max():.1f}°C")
