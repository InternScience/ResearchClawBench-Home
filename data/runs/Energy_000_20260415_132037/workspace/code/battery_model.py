"""
Simplified Single Particle Model (SPM) for Li-ion Battery
ECAT (Electrochemical-Aging-Thermal) Coupled Model
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

class SingleParticleModel:
    """
    Simplified Single Particle Model for Li-ion battery
    Based on pseudo-two-dimensional (P2D) model simplification
    """
    
    def __init__(self, params=None):
        # Default parameters for NMC/Graphite cell
        self.params = {
            # Geometric parameters
            'Rs_p': 5e-6,      # Cathode particle radius (m)
            'Rs_n': 5e-6,      # Anode particle radius (m)
            'L_p': 50e-6,      # Cathode thickness (m)
            'L_n': 50e-6,      # Anode thickness (m)
            'A': 0.1,          # Electrode area (m^2)
            
            # Volume fractions
            'eps_s_p': 0.5,    # Cathode active material volume fraction
            'eps_s_n': 0.5,    # Anode active material volume fraction
            
            # Transport parameters
            'D_s_p': 1e-14,    # Solid diffusion coefficient cathode (m^2/s)
            'D_s_n': 1e-14,    # Solid diffusion coefficient anode (m^2/s)
            
            # Kinetic parameters
            'k_p': 1e-11,      # Reaction rate constant cathode (m^2.5/mol^0.5/s)
            'k_n': 1e-11,      # Reaction rate constant anode (m^2.5/mol^0.5/s)
            
            # Concentration parameters
            'c_max_p': 50000,  # Max solid concentration cathode (mol/m^3)
            'c_max_n': 30000,  # Max solid concentration anode (mol/m^3)
            'c_e': 1000,       # Electrolyte concentration (mol/m^3)
            
            # Thermal parameters
            'h': 10,           # Heat transfer coefficient (W/m^2/K)
            'T_amb': 298,      # Ambient temperature (K)
        }
        
        if params:
            self.params.update(params)
        
        # Constants
        self.F = 96487       # Faraday constant (C/mol)
        self.R = 8.314       # Gas constant (J/mol/K)
        
    def set_params(self, params):
        """Update model parameters"""
        self.params.update(params)
    
    def get_solid_concentration_profile(self, r, t, D_s, Rs, j_n):
        """Solve solid diffusion equation using eigenfunction expansion"""
        # Simplified polynomial approximation
        c_surface = j_n * Rs / (5 * D_s)  # First-order approximation
        return c_surface
    
    def ocp_cathode(self, soc):
        """Open circuit potential for NMC cathode (V vs Li/Li+)"""
        # Empirical fit for NMC
        soc = np.clip(soc, 0.01, 0.99)
        return (4.04596 + 0.16339 * np.exp(-74.0038 * soc) 
                - 0.06339 * np.exp(-10.3585 * (soc - 0.5))
                - 0.00289 * np.exp(-25.0000 * (soc - 0.5)))
    
    def ocp_anode(self, soc):
        """Open circuit potential for graphite anode (V vs Li/Li+)"""
        # Empirical fit for graphite
        soc = np.clip(soc, 0.01, 0.99)
        return (0.1243 + 1.5 * np.exp(-160.0 * soc) 
                + 0.0351 * np.tanh(20.0 * (soc - 0.286))
                - 0.0045 * np.tanh(40.0 * (soc - 0.95)))
    
    def butler_volmer(self, i_0, eta, T):
        """Butler-Volmer equation for electrochemical kinetics"""
        alpha = 0.5
        F_RT = self.F / (self.R * T)
        return i_0 * (np.exp(alpha * F_RT * eta) - np.exp(-alpha * F_RT * eta))
    
    def simulate_discharge(self, current, T_sim, dt=1.0, initial_soc=1.0):
        """
        Simulate constant current discharge
        
        Parameters:
        -----------
        current : float
            Applied current (A), positive for discharge
        T_sim : float
            Simulation time (s)
        dt : float
            Time step (s)
        T_amb : float
            Ambient temperature (K)
            
        Returns:
        --------
        results : dict
            Contains time, voltage, soc_p, soc_n, temperature arrays
        """
        p = self.params
        num_steps = int(T_sim / dt) + 1
        t_array = np.linspace(0, T_sim, num_steps)
        
        # Initialize state variables
        soc_p = initial_soc * np.ones(num_steps)  # Cathode SOC
        soc_n = initial_soc * np.ones(num_steps)  # Anode SOC
        T = p['T_amb'] * np.ones(num_steps)       # Temperature
        voltage = np.zeros(num_steps)
        
        # Initial capacity calculation
        Q_p = (p['eps_s_p'] * p['L_p'] * p['A'] * p['c_max_p'] * self.F)
        Q_n = (p['eps_s_n'] * p['L_n'] * p['A'] * p['c_max_n'] * self.F)
        
        for i in range(num_steps):
            # Calculate open circuit potentials
            ocp_p = self.ocp_cathode(soc_p[i])
            ocp_n = self.ocp_anode(soc_n[i])
            
            # Overpotential estimation (simplified)
            eta_p = current * 0.01  # Simplified
            eta_n = -current * 0.01
            
            # Terminal voltage
            voltage[i] = ocp_p - ocp_n - 0.1  # Simplified overpotential drop
            
            # Update SOC
            if i < num_steps - 1:
                dsoc_p = -current * dt / Q_p
                dsoc_n = current * dt / Q_n
                soc_p[i+1] = np.clip(soc_p[i] + dsoc_p, 0.01, 0.99)
                soc_n[i+1] = np.clip(soc_n[i] + dsoc_n, 0.01, 0.99)
                
                # Simple thermal model
                q_gen = current**2 * 0.01  # Simplified heat generation
                q_loss = p['h'] * p['A'] * (T[i] - p['T_amb'])
                dT = (q_gen - q_loss) * dt / (1000 * 1000)  # Simplified thermal mass
                T[i+1] = T[i] + dT
        
        return {
            'time': t_array,
            'voltage': voltage,
            'soc_p': soc_p,
            'soc_n': soc_n,
            'temperature': T
        }


class ECATModel:
    """
    Electrochemical-Aging-Thermal coupled model
    Extends SPM with aging mechanisms
    """
    
    def __init__(self, params=None):
        self.spm = SingleParticleModel(params)
        
        # Aging parameters
        self.aging_params = {
            'SEI_growth_rate': 1e-15,    # SEI growth rate constant
            'R_SEI_initial': 0.001,       # Initial SEI resistance (Ohm)
            'k_cal': 1e-5,                # Calendar aging rate
        }
        
        if params and 'aging' in params:
            self.aging_params.update(params['aging'])
    
    def simulate_with_aging(self, current_profile, T_amb=298):
        """
        Simulate with aging effects
        
        Parameters:
        -----------
        current_profile : callable or array
            Current as function of time or time series
        T_amb : float
            Ambient temperature
            
        Returns:
        --------
        results : dict
            Simulation results with aging indicators
        """
        # For simplified implementation, use SPM simulation
        # In full implementation, this would include:
        # - SEI layer growth
        # - Capacity fade tracking
        # - Impedance rise
        
        if callable(current_profile):
            t = np.linspace(0, 3600, 1000)
            I = current_profile(t)
        else:
            t = np.linspace(0, len(current_profile), len(current_profile))
            I = current_profile
        
        # Simplified constant current simulation
        avg_current = np.mean(np.abs(I))
        results = self.spm.simulate_discharge(avg_current, t[-1], dt=t[1]-t[0])
        
        return results
