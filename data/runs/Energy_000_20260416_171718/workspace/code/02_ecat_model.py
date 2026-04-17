#!/usr/bin/env python3
"""
ECAT (Electrochemical-Aging-Thermal) Coupled Model Implementation
Simplified Single Particle Model (SPM) with thermal coupling for parameter identification
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import json

# ============================================================================
# ECAT Model Parameters - Search Space Definition
# ============================================================================

# Parameter bounds based on literature (paper_001, paper_003, paper_004)
PARAM_BOUNDS = {
    # Electrochemical parameters
    'R_p_n': (5e-6, 15e-6),       # Negative electrode particle radius (m)
    'R_p_p': (5e-6, 15e-6),       # Positive electrode particle radius (m)
    'D_s_n': (1e-14, 1e-12),      # Solid diffusion coefficient negative (m²/s)
    'D_s_p': (1e-14, 1e-12),      # Solid diffusion coefficient positive (m²/s)
    'k_n': (1e-11, 1e-9),         # Reaction rate constant negative (m²·⁵mol⁻⁰·⁵s⁻¹)
    'k_p': (1e-11, 1e-9),         # Reaction rate constant positive (m²·⁵mol⁻⁰·⁵s⁻¹)
    'eps_s_n': (0.4, 0.7),        # Volume fraction solid negative
    'eps_s_p': (0.4, 0.7),        # Volume fraction solid positive
    'eps_e': (0.2, 0.5),          # Electrolyte volume fraction
    
    # Thermal parameters
    'h': (5, 50),                 # Heat transfer coefficient (W/m²K)
    'rho_cp': (2e6, 4e6),         # Volumetric heat capacity (J/m³K)
    
    # Aging parameters (SEI growth)
    'k_SEI': (1e-20, 1e-16),      # SEI growth rate constant
    'R_SEI_0': (1e-6, 1e-4),      # Initial SEI resistance (Ohm·m²)
}

# Nominal values for NCM/graphite cells (from literature)
NOMINAL_PARAMS = {
    'R_p_n': 10e-6,
    'R_p_p': 8e-6,
    'D_s_n': 3.3e-14,
    'D_s_p': 4e-14,
    'k_n': 5.0e-11,
    'k_p': 2.5e-11,
    'eps_s_n': 0.6,
    'eps_s_p': 0.55,
    'eps_e': 0.3,
    'h': 20,
    'rho_cp': 3e6,
    'k_SEI': 1e-18,
    'R_SEI_0': 1e-5,
}

# OCP curves (simplified polynomial fits for graphite and NCM)
def U_n_graphite(sto):
    """Negative electrode OCP (graphite) vs Li/Li+"""
    sto = np.clip(sto, 0.01, 0.99)
    # Simplified fit based on typical graphite OCP
    return 0.15 + 0.05 * np.exp(-50 * sto) - 0.08 * np.exp(-20 * (1-sto)) + 0.02 * np.tanh(10 * (sto - 0.5))

def U_p_NCM(sto):
    """Positive electrode OCP (NCM) vs Li/Li+"""
    sto = np.clip(sto, 0.01, 0.99)
    # Simplified fit based on typical NCM OCP
    return 4.05 - 0.8 * sto - 0.15 * np.exp(-30 * (sto - 0.5)**2)

class SPMThermalModel:
    """
    Single Particle Model with thermal coupling for Li-ion battery simulation.
    Simplified version suitable for parameter identification.
    """
    
    def __init__(self, params, T_amb=298.15):
        """
        Initialize the model with parameters.
        
        Args:
            params: Dictionary of model parameters
            T_amb: Ambient temperature (K)
        """
        self.params = params
        self.T_amb = T_amb
        
        # Physical constants
        self.F = 96485.0  # Faraday constant (C/mol)
        self.R = 8.314    # Gas constant (J/mol·K)
        
        # Cell geometry (fixed)
        self.L_n = 50e-6   # Negative electrode thickness (m)
        self.L_s = 25e-6   # Separator thickness (m)
        self.L_p = 50e-6   # Positive electrode thickness (m)
        self.A = 0.01      # Electrode area (m²)
        
        # Initial conditions
        self.sto_n_0 = 0.8  # Initial stoichiometry negative
        self.sto_p_0 = 0.3  # Initial stoichiometry positive
        
        # Discretization for solid diffusion
        self.N_r = 10  # Number of radial nodes
        
    def get_specific_area(self, eps_s, R_p):
        """Calculate specific interfacial area"""
        return 3 * eps_s / R_p
    
    def ocp_negative(self, sto):
        """OCP of negative electrode"""
        return U_n_graphite(sto)
    
    def ocp_positive(self, sto):
        """OCP of positive electrode"""
        return U_p_NCM(sto)
    
    def cell_voltage(self, c_s_n_surf, c_s_p_surf, I, T):
        """
        Calculate cell terminal voltage.
        
        Args:
            c_s_n_surf: Surface concentration negative (mol/m³)
            c_s_p_surf: Surface concentration positive (mol/m³)
            I: Current (A)
            T: Temperature (K)
            
        Returns:
            V: Terminal voltage (V)
        """
        # Maximum concentrations
        c_max_n = 24000  # Graphite (mol/m³)
        c_max_p = 48000  # NCM (mol/m³)
        
        # Stoichiometries
        sto_n = c_s_n_surf / c_max_n
        sto_p = c_s_p_surf / c_max_p
        
        # OCPs
        U_n = self.ocp_negative(sto_n)
        U_p = self.ocp_positive(sto_p)
        
        # Overpotentials (simplified Butler-Volmer)
        a_n = self.get_specific_area(self.params['eps_s_n'], self.params['R_p_n'])
        a_p = self.get_specific_area(self.params['eps_s_p'], self.params['R_p_p'])
        
        # Exchange current densities (simplified)
        i0_n = self.F * self.params['k_n'] * np.sqrt(c_max_n * sto_n * (1-sto_n))
        i0_p = self.F * self.params['k_p'] * np.sqrt(c_max_p * sto_p * (1-sto_p))
        
        # Avoid division by zero
        i0_n = max(i0_n, 1e-6)
        i0_p = max(i0_p, 1e-6)
        
        # Charge transfer overpotentials (simplified)
        eta_n = (self.R * T / (0.5 * self.F)) * np.arcsinh(I / (2 * a_n * self.L_n * self.A * i0_n))
        eta_p = (self.R * T / (0.5 * self.F)) * np.arcsinh(I / (2 * a_p * self.L_p * self.A * i0_p))
        
        # SEI resistance (aging)
        R_SEI = self.params['R_SEI_0']
        
        # Terminal voltage
        V = U_p - U_n + eta_p - eta_n - I * R_SEI / self.A
        
        return V
    
    def heat_generation(self, I, V, OCV, T):
        """
        Calculate heat generation rate.
        
        Args:
            I: Current (A)
            V: Terminal voltage (V)
            OCV: Open circuit voltage (V)
            T: Temperature (K)
            
        Returns:
            q: Heat generation rate (W)
        """
        # Irreversible heat (Joule heating)
        q_irrev = I * (V - OCV)
        
        # Reversible heat (entropic)
        # Simplified: assume small entropic contribution
        q_rev = 0.0
        
        return q_irrev + q_rev
    
    def simulate_discharge(self, I_discharge, t_final, n_points=100, T_init=298.15):
        """
        Simulate constant current discharge.
        
        Args:
            I_discharge: Discharge current (A, positive for discharge)
            t_final: Final time (s)
            n_points: Number of output points
            T_init: Initial temperature (K)
            
        Returns:
            t: Time array (s)
            V: Voltage array (V)
            T: Temperature array (K)
            Q: Capacity throughput (Ah)
        """
        # Maximum concentrations
        c_max_n = 24000  # mol/m³
        c_max_p = 48000  # mol/m³
        
        # Initial surface concentrations
        c_s_n_0 = self.sto_n_0 * c_max_n
        c_s_p_0 = self.sto_p_0 * c_max_p
        
        # State: [c_s_n_surf, c_s_p_surf, T_cell, Q_throughput]
        y0 = [c_s_n_0, c_s_p_0, T_init, 0.0]
        
        def dynamics(t, y):
            c_s_n, c_s_p, T, Q = y
            
            # Voltage
            V = self.cell_voltage(c_s_n, c_s_p, I_discharge, T)
            
            # OCV
            sto_n = c_s_n / c_max_n
            sto_p = c_s_p / c_max_p
            OCV = self.ocp_positive(sto_p) - self.ocp_negative(sto_n)
            
            # Concentration change (simplified diffusion)
            # dc_s/dt = -I / (F * volume_factor)
            vol_n = (4/3) * np.pi * self.params['R_p_n']**3 * (self.L_n * self.A / ((4/3)*np.pi*self.params['R_p_n']**3))
            vol_p = (4/3) * np.pi * self.params['R_p_p']**3 * (self.L_p * self.A / ((4/3)*np.pi*self.params['R_p_p']**3))
            
            # Simplified: lumped concentration dynamics
            tau_n = self.params['R_p_n']**2 / self.params['D_s_n']
            tau_p = self.params['R_p_p']**2 / self.params['D_s_p']
            
            dc_n_dt = -I_discharge / (self.F * vol_n) - (c_s_n - c_s_n_0) / tau_n
            dc_p_dt = I_discharge / (self.F * vol_p) - (c_s_p - c_s_p_0) / tau_p
            
            # Heat generation
            q_gen = self.heat_generation(I_discharge, V, OCV, T)
            
            # Thermal dynamics
            # dT/dt = (q_gen - h*A*(T-T_amb)) / (rho_cp * volume)
            cell_vol = (self.L_n + self.L_s + self.L_p) * self.A
            dT_dt = (q_gen - self.params['h'] * 6 * np.sqrt(cell_vol**(2/3)) * (T - self.T_amb)) / (self.params['rho_cp'] * cell_vol)
            
            # Capacity throughput
            dQ_dt = I_discharge / 3600  # Ah/s
            
            # Cut-off conditions handled externally
            return [dc_n_dt, dc_p_dt, dT_dt, dQ_dt]
        
        def cutoff(t, y):
            """Stop when voltage reaches cutoff or time expires"""
            c_s_n, c_s_p, T, Q = y
            V = self.cell_voltage(c_s_n, c_s_p, I_discharge, T)
            return V - 2.7  # Stop at 2.7V
        cutoff.terminal = True
        cutoff.direction = -1
        
        # Solve ODE
        sol = solve_ivp(dynamics, [0, t_final], y0, method='RK45', 
                       t_eval=np.linspace(0, t_final, n_points),
                       events=cutoff, max_step=10)
        
        t = sol.t
        c_s_n = sol.y[0]
        c_s_p = sol.y[1]
        T_arr = sol.y[2]
        Q_arr = sol.y[3]
        
        # Calculate voltage
        V_arr = np.array([self.cell_voltage(cn, cp, I_discharge, T) 
                         for cn, cp, T in zip(c_s_n, c_s_p, T_arr)])
        
        return t, V_arr, T_arr, Q_arr


def generate_lhs_samples(n_samples=100, seed=42):
    """
    Generate Latin Hypercube Sampling of parameter space.
    
    Args:
        n_samples: Number of samples
        seed: Random seed
        
    Returns:
        samples: Array of shape (n_samples, n_params)
        param_names: List of parameter names
    """
    from sklearn.gaussian_process.kernels import RBF
    import scipy.stats as stats
    
    np.random.seed(seed)
    param_names = list(PARAM_BOUNDS.keys())
    n_params = len(param_names)
    
    # Generate LHS
    samples = np.zeros((n_samples, n_params))
    
    for i, (param, (lo, hi)) in enumerate(PARAM_BOUNDS.items()):
        # Log-uniform sampling for parameters spanning orders of magnitude
        log_lo = np.log10(lo)
        log_hi = np.log10(hi)
        
        # LHS in log space
        lhs = stats.qmc.LatinHypercube(d=1, seed=seed+i)
        u = lhs.random(n_samples)
        samples[:, i] = 10**(log_lo + u.flatten() * (log_hi - log_lo))
    
    return samples, param_names


if __name__ == "__main__":
    print("=" * 60)
    print("ECAT MODEL TEST AND LHS SAMPLE GENERATION")
    print("=" * 60)
    
    # Test with nominal parameters
    print("\n[1] Testing SPM-Thermal Model with nominal parameters...")
    model = SPMThermalModel(NOMINAL_PARAMS, T_amb=298.15)
    
    # Simulate 1C discharge
    I_1C = 2.0  # Assume 2Ah cell
    t, V, T, Q = model.simulate_discharge(I_1C, t_final=7200, n_points=200)
    
    print(f"  Simulation completed: {len(t)} time points")
    print(f"  Initial voltage: {V[0]:.3f} V")
    print(f"  Final voltage: {V[-1]:.3f} V")
    print(f"  Discharge time: {t[-1]:.1f} s ({t[-1]/60:.1f} min)")
    print(f"  Capacity delivered: {Q[-1]:.3f} Ah")
    print(f"  Temperature rise: {T[-1] - T[0]:.2f} K")
    
    # Save simulation results
    sim_result = {
        'time_s': t.tolist(),
        'voltage_V': V.tolist(),
        'temperature_K': T.tolist(),
        'capacity_Ah': Q.tolist(),
        'current_A': I_1C,
        'params': NOMINAL_PARAMS
    }
    
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260416_171718/outputs/nominal_simulation.json', 'w') as f:
        json.dump(sim_result, f, indent=2)
    
    # Generate LHS samples
    print("\n[2] Generating Latin Hypercube Samples...")
    n_lhs = 200
    lhs_samples, param_names = generate_lhs_samples(n_lhs)
    
    print(f"  Generated {n_lhs} samples for {len(param_names)} parameters")
    
    # Save LHS samples
    lhs_df_dict = {name: samples for name, samples in zip(param_names, lhs_samples.T)}
    import pandas as pd
    lhs_df = pd.DataFrame(lhs_df_dict)
    lhs_df.to_csv('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260416_171718/outputs/lhs_samples.csv', index=False)
    print(f"  Saved LHS samples to: outputs/lhs_samples.csv")
    
    # Run simulations for a subset of LHS samples (for ANN training)
    print("\n[3] Running simulations for ANN training data...")
    n_train = 50  # Subset for demo
    
    train_data = {
        'params': [],
        'features': [],
        'voltage_curves': []
    }
    
    for i in range(n_train):
        params_i = {name: float(lhs_samples[i, j]) for j, name in enumerate(param_names)}
        model_i = SPMThermalModel(params_i, T_amb=298.15)
        
        try:
            t_i, V_i, T_i, Q_i = model_i.simulate_discharge(I_1C, t_final=5000, n_points=100)
            
            # Features: key characteristics of voltage curve
            features = [
                V_i[0],           # Initial voltage
                V_i[-1],          # Final voltage
                np.mean(V_i),     # Mean voltage
                np.std(V_i),      # Voltage variation
                t_i[-1],          # Discharge time
                T_i[-1] - T_i[0], # Temperature rise
                Q_i[-1],          # Capacity
            ]
            
            train_data['params'].append([params_i[name] for name in param_names])
            train_data['features'].append(features)
            train_data['voltage_curves'].append(V_i.tolist())
            
            if (i+1) % 10 == 0:
                print(f"    Completed {i+1}/{n_train} simulations")
                
        except Exception as e:
            print(f"    Simulation {i} failed: {e}")
    
    # Save training data
    with open('/mnt/shared-storage-gpfs2/sciprismax2/gaohengjian/ResearchClawBench/workspaces/Energy_000_20260416_171718/outputs/ann_training_data.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    print(f"  Saved training data for {len(train_data['params'])} samples")
    
    print("\n" + "=" * 60)
    print("ECAT MODEL SETUP COMPLETE")
    print("=" * 60)
