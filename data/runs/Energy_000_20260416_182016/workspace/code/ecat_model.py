"""
Simplified Electrochemical-Aging-Thermal (ECAT) Model
Based on Single Particle Model (SPM) with thermal coupling
Calibrated for 18650 NMC/graphite cells
"""
import numpy as np

class ECATModel:
    """
    Simplified ECAT coupled model for Li-ion battery.
    Combines:
    - Electrochemical: Single Particle Model (SPM) 
    - Aging: SEI growth model
    - Thermal: Lumped thermal model
    """
    
    DEFAULT_PARAMS = {
        # Geometric
        'R_p_neg': 10e-6,
        'R_p_pos': 5e-6,
        'L_neg': 70e-6,
        'L_sep': 25e-6,
        'L_pos': 60e-6,
        'A_cell': 0.06,
        
        # Electrochemical
        'D_s_neg': 3.9e-14,
        'D_s_pos': 1.0e-13,
        'k_neg': 2.0e-11,
        'k_pos': 2.0e-11,
        'c_s_max_neg': 30555,
        'c_s_max_pos': 51555,
        'c_e': 1000,
        'eps_neg': 0.485,
        'eps_pos': 0.385,
        'eps_sep': 0.724,
        
        # Kinetic
        'alpha_a': 0.5,
        'alpha_c': 0.5,
        'Ea_neg': 20000,
        'Ea_pos': 20000,
        
        # Thermal
        'rho_cell': 2500,
        'Cp_cell': 1000,
        'h_conv': 5.0,
        'k_therm': 1.0,
        'T_amb': 298.15,
        'A_surf': 0.004,
        'm_cell': 0.045,
        
        # Aging
        'k_SEI': 1.0e-12,
        'E_a_SEI': 37000,
        'R_SEI_0': 0.005,
        
        # Cell
        'R_cc': 0.02,
        'SOC_init': 1.0,
        'Q_nom': 2.0,
    }
    
    F = 96485.0
    R_gas = 8.314
    
    # Stoichiometry windows
    x_100 = 0.9    # neg at full charge (lithiated)
    x_0 = 0.005    # neg at full discharge
    y_100 = 0.36   # pos at full charge (delithiated)
    y_0 = 0.93     # pos at full discharge (lithiated)
    
    def __init__(self, params=None):
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)
    
    def OCV_neg(self, x):
        """Graphite anode OCV (Doyle-Fuller-Newman)"""
        x = np.clip(x, 0.005, 0.995)
        U = (0.7222 + 0.1387*x + 0.029*np.sqrt(x) - 0.0172/x 
             + 0.0019/x**1.5 + 0.2808*np.exp(0.9 - 15*x) 
             - 0.7984*np.exp(0.4465*x - 0.4108))
        return U
    
    def OCV_pos(self, y):
        """NMC cathode OCV (polynomial fit)"""
        y = np.clip(y, 0.005, 0.995)
        U = (-10.72*y**4 + 23.88*y**3 - 16.77*y**2 + 2.595*y + 4.563)
        return U
    
    def exchange_current_density(self, c_surf, c_max, c_e, k, T, Ea):
        """Butler-Volmer exchange current density"""
        soc = np.clip(c_surf / c_max, 0.005, 0.995)
        c_surf_c = soc * c_max
        i0 = k * self.F * (c_e**0.5) * ((c_max - c_surf_c)**0.5) * (c_surf_c**0.5)
        i0 *= np.exp(-Ea/self.R_gas * (1/T - 1/298.15))
        return max(i0, 1e-10)
    
    def get_stoichiometry(self, soc):
        """Convert SOC to electrode stoichiometries"""
        x_neg = self.x_0 + soc * (self.x_100 - self.x_0)
        y_pos = self.y_0 + soc * (self.y_100 - self.y_0)
        return x_neg, y_pos
    
    def compute_voltage(self, soc, I_app, T, R_SEI=0.005):
        """Compute terminal voltage for given state"""
        p = self.params
        x_neg, y_pos = self.get_stoichiometry(soc)
        
        U_neg = self.OCV_neg(x_neg)
        U_pos = self.OCV_pos(y_pos)
        OCV = U_pos - U_neg
        
        c_surf_neg = x_neg * p['c_s_max_neg']
        c_surf_pos = y_pos * p['c_s_max_pos']
        
        i0_neg = self.exchange_current_density(
            c_surf_neg, p['c_s_max_neg'], p['c_e'], p['k_neg'], T, p['Ea_neg'])
        i0_pos = self.exchange_current_density(
            c_surf_pos, p['c_s_max_pos'], p['c_e'], p['k_pos'], T, p['Ea_pos'])
        
        a_s_neg = 3 * (1 - p['eps_neg']) / p['R_p_neg']
        a_s_pos = 3 * (1 - p['eps_pos']) / p['R_p_pos']
        
        j_loc_neg = I_app / (a_s_neg * p['A_cell'] * p['L_neg'] + 1e-20)
        j_loc_pos = I_app / (a_s_pos * p['A_cell'] * p['L_pos'] + 1e-20)
        
        eta_neg = 2*self.R_gas*T/self.F * np.arcsinh(j_loc_neg / (2*i0_neg + 1e-20))
        eta_pos = 2*self.R_gas*T/self.F * np.arcsinh(-j_loc_pos / (2*i0_pos + 1e-20))
        
        V = OCV - eta_neg + eta_pos - I_app * p['R_cc'] - I_app * R_SEI
        return V, OCV, eta_neg, eta_pos
    
    def simulate_cc_discharge(self, I_app, t_end, dt=1.0, V_cutoff=2.7):
        """
        Simulate constant current discharge.
        I_app: discharge current [A] (positive value)
        """
        p = self.params
        soc = p['SOC_init']
        T = p['T_amb']
        R_SEI = p['R_SEI_0']
        
        n_max = int(t_end / dt) + 1
        time_arr = np.zeros(n_max)
        voltage_arr = np.zeros(n_max)
        temp_arr = np.zeros(n_max)
        cap_arr = np.zeros(n_max)
        soc_arr = np.zeros(n_max)
        
        actual_steps = 0
        
        for i in range(n_max):
            t = i * dt
            
            V, OCV, eta_neg, eta_pos = self.compute_voltage(soc, I_app, T, R_SEI)
            
            time_arr[i] = t
            voltage_arr[i] = V
            temp_arr[i] = T - 273.15
            cap_arr[i] = I_app * t / 3600
            soc_arr[i] = soc
            actual_steps = i + 1
            
            if V < V_cutoff or soc < 0.005:
                break
            
            # Update SOC
            dQ = I_app * dt / 3600
            soc -= dQ / p['Q_nom']
            soc = np.clip(soc, 0.0, 1.0)
            
            # Thermal model
            V_ohm = I_app * p['R_cc']
            Q_gen = abs(I_app) * (abs(eta_neg) + abs(eta_pos) + V_ohm)
            Q_cool = p['h_conv'] * p['A_surf'] * (T - p['T_amb'])
            dT = (Q_gen - Q_cool) * dt / (p['m_cell'] * p['Cp_cell'])
            T += dT
            
            # SEI growth
            dR_SEI = p['k_SEI'] * np.exp(-p['E_a_SEI']/(self.R_gas*T)) * dt
            R_SEI += dR_SEI
        
        return {
            'time': time_arr[:actual_steps],
            'voltage': voltage_arr[:actual_steps],
            'temperature': temp_arr[:actual_steps],
            'capacity': cap_arr[:actual_steps],
            'soc': soc_arr[:actual_steps]
        }
    
    def simulate_dynamic(self, current_profile, time_profile, T_init=None, Q_nom=None):
        """
        Simulate with dynamic current profile.
        current_profile: array of current [mA] (positive = discharge, negative = charge)
        time_profile: array of time [s]
        """
        p = self.params
        if T_init is None:
            T_init = p['T_amb']
        if Q_nom is None:
            Q_nom = p['Q_nom']
        
        soc = p['SOC_init']
        T = T_init
        
        n = len(time_profile)
        voltage = np.zeros(n)
        temperature = np.zeros(n)
        
        for i in range(n):
            I_app = abs(current_profile[i]) / 1000.0  # mA to A
            sign = 1 if current_profile[i] > 0 else -1
            
            V, OCV, eta_neg, eta_pos = self.compute_voltage(soc, I_app * sign, T)
            
            # For discharge (sign>0): V = OCV - losses
            # For charge (sign<0): V = OCV + losses
            if sign > 0:
                V = OCV - (abs(eta_neg) + abs(eta_pos) + I_app * p['R_cc'])
            else:
                V = OCV + (abs(eta_neg) + abs(eta_pos) + I_app * p['R_cc'])
            
            voltage[i] = V
            temperature[i] = T - 273.15
            
            if i < n - 1:
                dt = time_profile[i+1] - time_profile[i]
                if dt <= 0:
                    dt = 1.0
                dQ = I_app * dt / 3600
                if sign > 0:
                    soc -= dQ / Q_nom
                else:
                    soc += dQ / Q_nom
                soc = np.clip(soc, 0.0, 1.0)
                
                Q_gen = abs(I_app) * (abs(eta_neg) + abs(eta_pos) + I_app * p['R_cc'])
                Q_cool = p['h_conv'] * p['A_surf'] * (T - p['T_amb'])
                dT = (Q_gen - Q_cool) * dt / (p['m_cell'] * p['Cp_cell'])
                T += dT
        
        return {
            'time': time_profile,
            'voltage': voltage,
            'temperature': temperature
        }
    
    def compute_features(self, I_app, t_end=4000, dt=5.0, V_cutoff=2.7, n_features=50):
        """
        Run simulation and extract feature vector for ANN training.
        Returns voltage at n_features equally spaced SOC points + final capacity + max temp.
        """
        result = self.simulate_cc_discharge(I_app, t_end, dt, V_cutoff)
        
        if len(result['time']) < 10:
            return np.zeros(n_features + 2)
        
        # Interpolate voltage at equally spaced capacity points
        cap = result['capacity']
        vol = result['voltage']
        temp = result['temperature']
        
        cap_points = np.linspace(0, cap[-1]*0.98, n_features)
        try:
            from scipy.interpolate import interp1d
            f_v = interp1d(cap, vol, kind='linear', fill_value='extrapolate')
            v_features = f_v(cap_points)
        except:
            v_features = np.interp(cap_points, cap, vol)
        
        features = np.concatenate([v_features, [cap[-1], temp[-1] - temp[0]]])
        return features


def get_parameter_bounds():
    """Define parameter search space for LHS"""
    bounds = {
        'R_p_neg':      (2e-6,   25e-6),
        'R_p_pos':      (1e-6,   15e-6),
        'D_s_neg':      (1e-15,  1e-12),
        'D_s_pos':      (1e-14,  1e-11),
        'k_neg':        (1e-12,  1e-9),
        'k_pos':        (1e-12,  1e-9),
        'c_s_max_neg':  (20000,  40000),
        'c_s_max_pos':  (40000,  60000),
        'eps_neg':      (0.3,    0.6),
        'eps_pos':      (0.25,   0.5),
        'R_cc':         (0.005,  0.08),
        'Cp_cell':      (500,    1500),
        'h_conv':       (2.0,    20.0),
        'k_SEI':        (1e-13,  1e-11),
        'Q_nom':        (1.5,    2.5),
    }
    return bounds


def get_identifiable_params():
    """Return list of parameter names to identify"""
    return ['R_p_neg', 'R_p_pos', 'D_s_neg', 'D_s_pos', 
            'k_neg', 'k_pos', 'c_s_max_neg', 'c_s_max_pos',
            'eps_neg', 'eps_pos', 'R_cc', 'Cp_cell', 'h_conv',
            'k_SEI', 'Q_nom']


if __name__ == '__main__':
    model = ECATModel()
    result = model.simulate_cc_discharge(I_app=2.0, t_end=5000, dt=1.0, V_cutoff=2.7)
    print(f"Discharge simulation: {len(result['time'])} steps")
    print(f"  V range: [{result['voltage'].min():.3f}, {result['voltage'].max():.3f}]")
    print(f"  T range: [{result['temperature'].min():.2f}, {result['temperature'].max():.2f}] °C")
    print(f"  Capacity: {result['capacity'][-1]:.3f} Ah")
    print(f"  Duration: {result['time'][-1]:.0f} s")
    
    # Test feature extraction
    features = model.compute_features(I_app=2.0)
    print(f"\nFeature vector: length={len(features)}")
    print(f"  V features range: [{features[:50].min():.3f}, {features[:50].max():.3f}]")
    print(f"  Final capacity: {features[-2]:.3f} Ah")
    print(f"  Temp rise: {features[-1]:.2f} °C")
